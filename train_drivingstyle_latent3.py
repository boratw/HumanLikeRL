
import glob
import os
import sys

try:
    sys.path.append(glob.glob('/home/user/carla-0.9.14/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import tensorflow.compat.v1 as tf
from laneinfo import LaneInfo
from lanetrace import LaneTrace
from network.DrivingStyle5_bayesian_latent import DrivingStyleLearner
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import multiprocessing

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_Batjeon.pkl")

state_len = 53
nextstate_len = 6
route_len = 20
action_len = 3
global_latent_len = 4


log_name = "train_log/DrivingStyle5_Bayesian_Latent/log_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_file = open(log_name + ".txt", "wt")

def rotate(posx, posy, yawsin, yawcos):
    return posx * yawcos - posy * yawsin, posx * yawsin + posy * yawcos

ReadOption = { "LaneFollow" : [1., 0., 0.],
              "Left" : [0., 0., 1.],
              "Right" : [0., 0., -1.],
              "ChangeLaneLeft" : [0., 1., 0.],
              "ChangeLaneRight" : [0., -1, 0.],
              "Straight" : [1., 0., 0.]
              }

def parallel_task(item):

    state_vectors = item["state_vectors"]
    control_vectors = item["control_vectors"]
    agent_count = len(item["state_vectors"][0])

    history_exp = [[] for _ in range(agent_count)]
    torque_added = [0 for _ in range(agent_count)]
    stepstart = random.randrange(50, 60)
    lane_tracers = [LaneTrace(laneinfo, 10) for _ in range(agent_count)]
    for step in range(stepstart, len(state_vectors)-100, 4):
        for i in range(agent_count):
            added = False
            if torque_added[i] == 0:
                if control_vectors[step+20][i][0] != 0 or control_vectors[step+21][i][0] != 0 or control_vectors[step+22][i][0] != 0 or control_vectors[step+23][i][0] != 0:
                    torque_added[i] = 5
                else:
                    other_vcs = []
                    x = state_vectors[step][i][0]
                    y = state_vectors[step][i][1]
                    relposx = state_vectors[step+60][i][0] - x
                    relposy = state_vectors[step+60][i][1] - y
                    distance = np.sqrt(relposx * relposx + relposy * relposy)
                    if distance < 200 and distance > 0.01:
                            

                        yawsin = np.sin(state_vectors[step][i][2]  * -0.017453293)
                        yawcos = np.cos(state_vectors[step][i][2]  * -0.017453293)
                        nextstate = []
                        for j in range(0, 45, 15) :
                            relposx = state_vectors[step + j + 15][i][0] - state_vectors[step + j][i][0]
                            relposy = state_vectors[step + j + 15][i][1] - state_vectors[step + j][i][1]
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            nextstate.extend([px, py])

                        if len(nextstate) == 6 and random.random() < (abs(nextstate[1] + nextstate[3] + nextstate[5]) ** 0.5 + 0.1):
                            traced, tracec = lane_tracers[i].Trace(x, y)
                            if traced != None:

                                distance_array = [(state_vectors[step][j][0] - x) ** 2 + (state_vectors[step][j][1] - y) ** 2 for j in range(agent_count)]
                                distance_indicies = np.array(distance_array).argsort()

                                other_vcs = []
                                for j in distance_indicies[1:9]:
                                    relposx = state_vectors[step][j][0] - x
                                    relposy = state_vectors[step][j][1] - y
                                    px, py = rotate(relposx, relposy, yawsin, yawcos)
                                    vx, vy = rotate(state_vectors[step][j][3], state_vectors[step][j][4], yawsin, yawcos)
                                    relyaw = (state_vectors[step][j][2] - state_vectors[step][i][2])   * 0.017453293
                                    other_vcs.append([px, py, np.cos(relyaw), np.sin(relyaw), vx, vy])

                                velocity = np.sqrt(state_vectors[step][i][3] ** 2 + state_vectors[step][i][4] ** 2)

                                route = []
                                for trace in traced:
                                    waypoints = []
                                    for j in trace:
                                        px, py = rotate(j[0] - x, j[1] - y, yawsin, yawcos)
                                        waypoints.extend([px, py])
                                    route.append(waypoints)

                                if (nextstate[1] + nextstate[3] + nextstate[5]) < -1.5:
                                    action = 0
                                elif (nextstate[1] + nextstate[3] + nextstate[5]) > 1.5:
                                    action = 2
                                else:
                                    action = 1
                                    
                                px, py = 50., 0.
                                for t in state_vectors[step][i][6]:
                                    if (px * px + py * py) >  ((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                                        px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
                                history_exp[i].append( [True, np.concatenate([[velocity, (1. if state_vectors[step][i][5] == 0. else 0.), px, py, control_vectors[step][i][1]], np.array(other_vcs).flatten()]), nextstate, route, action])
                                added = True
            else:
                torque_added[i] -= 1
            if added == False:
                history_exp[i].append( [False])
    history = []
    for exp in history_exp:
        if len(exp) > 200:
            history.append(exp)
    return history

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    with multiprocessing.Pool(20) as pool:
        learner = DrivingStyleLearner(state_len=state_len, nextstate_len=nextstate_len, route_len=route_len, action_len= action_len)
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        sess.run(tf.global_variables_initializer())
        learner.network_initialize()
        log_file.write("Epoch" + learner.log_caption() + "\n")

        history = []

        for epoch in range(1, 10000):
            pkl_index = random.randrange(26)
            with open("data/gathered_from_npc_batjeon6/data_" + str(pkl_index) + ".pkl","rb") as fr:
                data = pickle.load(fr)
            print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

            history_data = []
            for result in pool.imap_unordered(parallel_task, data):
                history_data.append(result)
            history.append(history_data)

            print("Current History Length : " + str(len(history)))

            for iter in range(len(history) * 8):

                data_index = random.randrange(len(history))
                exp_index = random.randrange(len(history[data_index]))
                print("Train Step #" + str(iter) + "Read data " + str(data_index) + " exp " + str(exp_index))

                cur_history = history[data_index][exp_index]
                agent_num = len(cur_history)
                
                #agent_dic = random.choices(list(range(agent_num)), k=16)
                agent_dic = np.random.randint(0, agent_num, (64,))


                for x in agent_dic:
                    state_dic = []
                    nextstate_dic = []
                    route_dic = []
                    action_dic = []

                    c = cur_history[x]
                    step_dic = list(range(len(c)))
                    prev_len = len(state_dic)
                    random.shuffle(step_dic)
                    state_dic.extend([c[step][1] for step in step_dic if c[step][0]])
                    nextstate_dic.extend([c[step][2] for step in step_dic if c[step][0]])
                    route_dic.extend([c[step][3] for step in step_dic if c[step][0]])
                    action_dic.extend([c[step][4] for step in step_dic if c[step][0]])


                    if len(state_dic) >= 32:
                        truncate = int((len(state_dic) // 8) * 8)
                        learner.optimize(state_dic[:truncate], nextstate_dic[:truncate], route_dic[:truncate], action_dic[:truncate])
                learner.optimize_update()
                
            if len(history) > 32:
                history = history[1:]

            learner.log_print()
            log_file.write(str(epoch) + "\t" + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()


            if epoch % 20 == 0:
                learner_saver.save(sess, log_name + "_" + str(epoch) + ".ckpt")
