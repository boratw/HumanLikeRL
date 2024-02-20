
import numpy as np
import carla
import tensorflow.compat.v1 as tf
import random
import time
import datetime


from algorithm.driver_npc import Driver_NPC 
from algorithm.driver_agent import Driver_Agent 
from network.sac import SAC 
from laneinfo import LaneInfo

LOG_DIR = "train_log/RL_Default/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')
log_file = open(LOG_DIR + "log.txt", "wt")
log_fail_file = open(LOG_DIR + "log_fail.txt", "wt")

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10.0)

npc_agent_num = 100
npc_vehicle_list = []
npc_driver_list = []
ego_agent_num = 8
ego_vehicle_list = []
ego_driver_list = []

try:
    world = client.get_world()

    settings = world.get_settings()
    settings.substepping = False
    #settings.max_substep_delta_time = 0.01
    #settings.max_substeps = 5
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.actor_active_distance = 100000
    #settings.no_rendering_mode = True
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    traffic_manager.set_respawn_dormant_vehicles(False)
    traffic_manager.set_hybrid_physics_mode(False) 
    traffic_manager.set_synchronous_mode(True)

    blueprints = world.get_blueprint_library().filter('vehicle.*')
    blueprints = [x for x in blueprints if 
                    x.id.endswith('a2') or
                    x.id.endswith('etron') or
                    x.id.endswith('tt') or
                    x.id.endswith('grandtourer') or
                    x.id.endswith('impala') or
                    x.id.endswith('c3') or
                    x.id.endswith('charger_2020') or
                    x.id.endswith('crown') or
                    x.id.endswith('mkz_2017') or
                    x.id.endswith('mkz_2020') or
                    x.id.endswith('coupe') or
                    x.id.endswith('coupe_2020') or
                    x.id.endswith('cooper_s') or
                    x.id.endswith('cooper_s_2021') or
                    x.id.endswith('mustang') or
                    x.id.endswith('micra') or
                    x.id.endswith('leon') or
                    x.id.endswith('model3') or
                    x.id.endswith('prius')]

    world_map = world.get_map()
    spawn_points = world_map.get_spawn_points()
    random.shuffle(spawn_points)

    tf.disable_eager_execution()
    sess = tf.Session()
    with sess.as_default():

        learner = SAC(state_len=Driver_Agent.state_len(), action_len=Driver_Agent.action_len())
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)

        init = tf.global_variables_initializer()
        sess.run(init)
        learner.network_initialize()
        history = []

        for exp in range(1, 10001):
            print("Exp " + str(exp))
            for tr in spawn_points:
                if len(npc_vehicle_list) >= npc_agent_num:
                    break
                
                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                    blueprint.set_attribute('role_name', 'autopilot')

                actor = world.try_spawn_actor(blueprint, tr)
                if actor != None:
                    npc_vehicle_list.append(actor)
            
            for tr in spawn_points[::-1]:
                if len(ego_vehicle_list) >= ego_agent_num:
                    break

                blueprint = random.choice(blueprints)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                    blueprint.set_attribute('role_name', 'hero')

                actor = world.try_spawn_actor(blueprint, tr)
                if actor != None:
                    ego_vehicle_list.append(actor)
            
            for actor in npc_vehicle_list:
                driver = Driver_NPC(actor, laneinfo, traffic_manager)
                npc_driver_list.append(driver)

            for actor in ego_vehicle_list:
                driver = Driver_Agent(actor, laneinfo)
                ego_driver_list.append(driver)


            for actor in ego_driver_list:
                neighbors = npc_driver_list.copy()
                for a in ego_driver_list:
                    if actor != a:
                        neighbors.append(a)
                actor.assign_neighbors(neighbors)


            world.tick()
            prev_state_list = [None] * ego_agent_num
            survive_vec = [True] * ego_agent_num
            rnd = np.array([np.random.normal(0.25, 0.125), np.random.normal(0.0, 0.25)])
            avg_reward = 0.
            avg_step = 0.
            avg_vel = 0.

            for step in range(5000):

                for driver in npc_driver_list:
                    driver.tick()
                
                for i, driver in enumerate(ego_driver_list):
                    driver.tick()

                state_list = []
                for i, driver in enumerate(ego_driver_list):
                    driver.perception()
                    state_list.append(driver.state)
                    if isinstance(prev_state_list[i], np.ndarray) and isinstance(driver.state, np.ndarray):
                        if survive_vec[i] == True:
                            history.append([prev_state_list[i], driver.state, action_list[i], [driver.reward - (0. if driver.survive else 1.)],
                                 [1. if driver.survive else 1.]])
                            avg_reward += driver.reward
                            avg_step += 1.
                            avg_vel += driver.vel
                            if driver.survive == False:
                                survive_vec[i] = False
                                print("Agent " + str(i) + " Failed : " + driver.fail_message)
                                log_fail_file.write(driver.fail_message)

                action_list = learner.get_action(state_list)
                rnd_action_actor = (step // 100) % ego_agent_num
                if action_list[rnd_action_actor][0] < -0.1:
                    action_list[rnd_action_actor][0] = -0.1
                action_list[rnd_action_actor] = action_list[rnd_action_actor] * 0.5 + rnd
                rnd[0] = (rnd[0] - 0.25) * 0.9 + 0.25 + np.random.normal(0., 0.0125)
                rnd[1] = rnd[1] * 0.9 + np.random.normal(0., 0.025)

                for i, driver in enumerate(ego_driver_list):
                    if survive_vec[i] == True:
                        driver.apply(action_list[i])
                    else:
                        driver.apply([-1., 0.])

                world.tick()
                if np.any(survive_vec) == False:
                    break
                
                prev_state_list = state_list

            for driver in npc_driver_list:
                driver.destroy()
            for i, driver in enumerate(ego_driver_list):
                driver.destroy()
            world.tick()
            npc_vehicle_list = []
            ego_vehicle_list = []
            npc_driver_list = []
            ego_driver_list = []

            avg_reward /= ego_agent_num
            avg_step /= ego_agent_num
            avg_vel /= ego_agent_num
            print("Reward : " + str(avg_reward))
            print("Step : " + str(avg_step))
            print("Vel : " + str(avg_vel))

            for iter in range(32):
                print("Train " + str(iter))
                for iter2 in range(128):
                    dic = random.sample(range(len(history)), 32)

                    state_dic = [history[x][0] for x in dic]
                    nextstate_dic = [history[x][1] for x in dic]
                    action_dic = [history[x][2] for x in dic]
                    reward_dic = [history[x][3] for x in dic]
                    survive_dic = [history[x][4] for x in dic]

                    learner.optimize(state_dic, nextstate_dic, action_dic, reward_dic, survive_dic, exp)
                learner.network_intermediate_update()
            

            learner.log_print()
            log_file.write(str(exp) + "\t" + str(avg_reward) + "\t" + str(avg_step) + "\t" + str(avg_vel) + "\t" + learner.current_log() + "\n")

            learner.network_update()

            history = history[(len(history) // 20) :]

            if exp % 100 == 0:
                learner_saver.save(sess, LOG_DIR + "log1_" + str(exp) + ".ckpt")

finally:
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    for driver in npc_driver_list:
        driver.destroy()
    for driver in ego_driver_list:
        driver.destroy()

    time.sleep(1.)