
import carla
import numpy as np
from laneinfo import LaneInfo, LaneType
from lanetrace import LaneTrace
from algorithm.driver import Driver
import weakref
import random


class Driver_Agent(Driver):
    def __init__(self, actor, laneinfo, neighbors = None):
        super().__init__(actor, laneinfo)
        self.laneinfo = laneinfo
        self.neighbors = neighbors
        self.collision_sensor = CollisionSensor(actor)
        self.laneinvasion_sensor = LaneInvasionSensor(actor)

        self.fail_message = ""
        self.tlight_state = carla.TrafficLightState.Unknown
        self.reward = 0
        self.lane_bias = 0
        self.light_checked = True
        self.invasion_timer = 0

        self.des_direction = LaneType.Follow
        self.cur_direction = LaneType.Follow

        self.success_entering = 0
        self.failed_entering = 0

    def tick(self):
        super().tick()
        self.vel = np.sqrt(self.v.x * self.v.x + self.v.y * self.v.y)
    
    def perception(self):
        if self.survive:
            self.get_reward()
            self.get_state()

    def apply(self, action):

        if self.survive:
            control = carla.VehicleControl()
            if action[0] > 0:
                if action[0] < 1:
                    control.throttle = float(action[0])
                else:
                    control.throttle = 1.
            else:
                if action[0] > -1:
                    control.brake = float(-action[0])
                else:
                    control.brake = 1.
            
            if action[1] > 1:
                control.steer = 1.
            elif action[1] < -1:
                control.steer = -1.
            else:
                control.steer = float(action[1])
                    

            self.actor.apply_control(control)

    def get_reward(self):
        self.reward = 0


        tlight = self.actor.get_traffic_light()
        if tlight != None:
            tlight_state = tlight.get_state()
            if tlight_state != None and tlight_state != carla.TrafficLightState.Unknown:
                self.tlight_state = tlight_state
        

        if self.lanetrace.laneid == None:
            self.fail_message = "OUT_OF_MAP"
            self.survive = False
            self.destroy()
            return

        lanetype = self.laneinfo.lanes[self.lanetrace.laneid]["type"]


        if self.collision_sensor.collision:
            self.fail_message = "COLLISION"
            self.survive = False
            self.destroy()
            return

        if self.vel > 11.1111:
            self.fail_message = "SPEEDING"
            self.survive = False
            self.destroy()
            return
        else:
            self.reward += self.vel * 0.1
        
        if self.laneinvasion_sensor.collision and self.laneinvasion_sensor.lane_change == False:
            self.fail_message = "CROSS_BORDER"
            self.survive = False
            self.destroy()
            return
        
        if lanetype == LaneType.Follow:
            self.light_checked = False
            if self.des_direction == LaneType.Follow:
                self.set_desdirection()

            if self.lanetrace_result[1][0] == True:
                dx1, dy1 = self.lanetrace_result[0][0][0][0], self.lanetrace_result[0][0][0][1]
                dx2, dy2 = self.lanetrace_result[0][0][1][0], self.lanetrace_result[0][0][1][1]
                lat = abs(((dx2 - dx1) * (dy1 - self.tr.location.y) - (dx1 - self.tr.location.x) * (dy2 - dy1)) / 2.)
                if lat > 1.5:
                    self.invasion_timer += 1
                else :
                    self.invasion_timer = 0
                
                if self.invasion_timer > 100:
                    self.fail_message = "CROSS_LANE"
                    self.survive = False
                    self.destroy()
                    return

                self.reward -= lat * 0.1
                self.lane_bias = lat

            self.get_curdirection()
            if self.cur_direction == self.des_direction:
                self.reward += 0.01
        else:
            self.invasion_timer = 0
            if self.light_checked == False:
                if self.tlight_state != carla.TrafficLightState.Green:
                    self.fail_message = "RED_LIGHT"
                    self.survive = False
                    self.destroy()
                    return
                self.light_checked = True

            if self.des_direction != LaneType.Follow:
                if self.cur_direction == self.des_direction:
                    self.reward += 1.
                    self.success_entering += 1
                else:
                    self.reward -= 0.1
                    self.failed_entering = 0
                self.des_direction = LaneType.Follow

    def get_state(self):
        if self.lanetrace_result[0] != None:
            x = self.tr.location.x
            y = self.tr.location.y
            yawsin = np.sin(self.tr.rotation.yaw  * -0.017453293)
            yawcos = np.cos(self.tr.rotation.yaw  * -0.017453293)

            distance_array = [(j.tr.location.x - x) ** 2 + (j.tr.location.y - y) ** 2 for j in self.neighbors]
            distance_indicies = np.array(distance_array).argsort()

            other_vcs = []
            for j in distance_indicies[:8]:
                relposx = self.neighbors[j].tr.location.x - x
                relposy = self.neighbors[j].tr.location.y - y
                px, py = rotate(relposx, relposy, yawsin, yawcos)
                vx, vy = rotate(self.neighbors[j].v.x, self.neighbors[j].v.y, yawsin, yawcos)
                relyaw = (self.neighbors[j].tr.rotation.yaw - self.tr.rotation.yaw)   * 0.017453293
                other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy])

            
            route = []
            for trace in self.lanetrace_result[0]:
                waypoints = []
                for j in trace:
                    px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                    waypoints.extend([px, py])
                route.append(waypoints)

            self.state = np.concatenate([[self.vel, self.tlight_state, self.des_direction], 
                                                                np.array(other_vcs).flatten(), np.array(route).flatten()])
        else:
            self.state = None


    def get_curdirection(self):
        self.cur_direction = LaneType.Follow
        laneid = self.lanetrace.laneid
        while len(self.laneinfo.lanes[laneid]["next"]) > 0:
            laneid = self.laneinfo.lanes[laneid]["next"][0]
            lanetype = self.laneinfo.lanes[laneid]["type"]
            if lanetype != LaneType.Follow:
                self.cur_direction = lanetype
                break

    def set_desdirection(self):
        candidates = []
        candlanes = [self.lanetrace.laneid]
        laneid = self.lanetrace.laneid
        while self.laneinfo.lanes[laneid]["left"] != None:
            laneid = self.laneinfo.lanes[laneid]["left"]
            candlanes.append(laneid)
        laneid = self.lanetrace.laneid
        while self.laneinfo.lanes[laneid]["right"] != None:
            laneid = self.laneinfo.lanes[laneid]["right"]
            candlanes.append(laneid)
        for candlane in candlanes:
            laneid = candlane
            while len(self.laneinfo.lanes[laneid]["next"]) > 0:
                laneid = self.laneinfo.lanes[laneid]["next"][0]
                lanetype = self.laneinfo.lanes[laneid]["type"]
                if lanetype != LaneType.Follow:
                    candidates.append(lanetype)
                    break

        if len(candidates) > 0:
            self.des_direction = random.choice(candidates)
    

    def destroy(self):
        try:
            self.collision_sensor.destroy()
            self.laneinvasion_sensor.destroy()
            self.actor.destroy()
        except:
            pass

    def assign_neighbors(self, neighbors):
        self.neighbors = neighbors

    @staticmethod
    def state_len():
        return 81

    @staticmethod
    def action_len():
        return 2



class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.collision = False
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))


    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision = True

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()

class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.collision = False
        self.lane_change = False
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.collision = True
        self.lane_change = True
        if len(event.crossed_lane_markings) > 0:
            if event.crossed_lane_markings[0].lane_change == carla.LaneChange.NONE:
                self.lane_change = False
        

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos