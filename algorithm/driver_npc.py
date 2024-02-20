
import carla
import numpy as np
from algorithm.driver import Driver

class Driver_NPC(Driver):
    def __init__(self, actor, laneinfo, traffic_manager):
        super().__init__(actor, laneinfo)
        self.traffic_manager = traffic_manager
        actor.set_autopilot(True, traffic_manager.get_port())

        self.distance_to_leading_vehicle = np.random.uniform(5.0, 15.0)
        self.vehicle_lane_offset = np.random.uniform(-0.3, 0.3)
        self.vehicle_speed = np.random.uniform(-50.0, 50.0)
        self.impatient_lane_change = np.abs(np.random.normal(0.0, 100.0) + 20.)
        self.ignore_lights = np.clip(np.random.normal(0.0, 1.0), 0.0, 6.0)

        traffic_manager.distance_to_leading_vehicle(actor, self.distance_to_leading_vehicle)
        traffic_manager.vehicle_lane_offset(actor, self.vehicle_lane_offset)
        traffic_manager.vehicle_percentage_speed_difference(actor, self.vehicle_speed)
        traffic_manager.ignore_lights_percentage(actor, self.ignore_lights)

        self.desired_velocity = 11.1111 * (1.0 - self.vehicle_speed / 100.0)
        self.impatiece = 0
 
    def tick(self):
        super().tick()
        if self.survive:
            vel = np.sqrt(self.v.x * self.v.x + self.v.y * self.v.y)

            if vel > 0.1:
                if self.desired_velocity > vel + 3.0:
                    self.impatiece += (self.desired_velocity * 1.5 - vel) * 0.1
            else:
                self.impatiece = 0.
            if self.impatiece < 0.:
                self.impatiece = 0.
            
            if self.impatient_lane_change < self.impatiece:
                self.traffic_manager.random_left_lanechange_percentage(self.actor, 100)
                self.traffic_manager.random_right_lanechange_percentage(self.actor, 100)
            else:
                self.traffic_manager.random_left_lanechange_percentage(self.actor, 0)
                self.traffic_manager.random_right_lanechange_percentage(self.actor, 0)


    def destroy(self):
        try:
            self.actor.destroy()
        except:
            pass