
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
from laneinfo import LaneInfo, RouteTracer
from network.DrivingStyle import DrivingStyleLearner
from datetime import datetime
import numpy as np
import pickle
import time
import random
import carla
import multiprocessing

laneinfo = LaneInfo()
laneinfo.Load_from_File("laneinfo_World10Opt.pkl")

state_len = 59
traj_len = 100
global_input_len = 10
global_latent_len = 4


log_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
log_file = open("train_log/DrivingStyle/log_" + log_name + ".txt", "wt")

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
    history_exp = [[] for _ in range(100)]

    state_vectors = item["state_vectors"]
    agent_count = len(item["state_vectors"][0])

    stepstart = random.randrange(50, 60)
    for step, state_vector in enumerate(state_vectors[stepstart:-60:10]):
        for i in range(agent_count):
            other_vcs = []
            x = state_vector[i][0]
            y = state_vector[i][1]
            yawsin = np.sin(state_vector[i][2]  * -0.017453293)
            yawcos = np.cos(state_vector[i][2]  * -0.017453293)
            for j in range(agent_count):
                if i != j:
                    relposx = state_vector[j][0] - x
                    relposy = state_vector[j][1] - y
                    px, py = rotate(relposx, relposy, yawsin, yawcos)
                    vx, vy = rotate(state_vector[j][3], state_vector[j][4], yawsin, yawcos)
                    relyaw = (state_vector[j][2] - state_vector[i][2])   * 0.017453293
                    if relyaw < -np.pi:
                        relyaw += 2 * np.pi
                    elif relyaw > np.pi:
                        relyaw -= 2 * np.pi
                    other_vcs.append([relposx, relposy, relyaw, vx, vy, np.sqrt(relposx * relposx + relposy * relposy)])
            other_vcs = np.array(sorted(other_vcs, key=lambda s: s[5]))
            velocity = np.sqrt(state_vector[i][3] ** 2 + state_vector[i][4] ** 2)

            relposx = state_vectors[step+20][i][0] - x
            relposy = state_vectors[step+20][i][1] - y
            px, py = rotate(relposx, relposy, yawsin, yawcos)
            route = [px, py]
            
            waypoints = []
            option = [0., 0., 0.]
            px, py = 0., 0.
            prevx = 0.
            prevy = 0.
            k = step
            for j in range(3):
                while k < len(state_vectors):
                    if len(state_vectors[k][i][8]) > 0 :
                        if state_vectors[k][i][8][0][1] != prevx or state_vectors[k][i][8][0][2] != prevy:
                            relposx = state_vectors[k][i][8][0][1] - x
                            relposy = state_vectors[k][i][8][0][2] - y
                            px, py = rotate(relposx, relposy, yawsin, yawcos)
                            if state_vectors[k][i][8][0][0] in ReadOption:
                                option = ReadOption[state_vectors[k][i][8][0][0]]
                            else:
                                print("Unknown RoadOption " + state_vectors[k][i][8][0][0])
                            prevx = state_vectors[k][i][8][0][1]
                            prevy = state_vectors[k][i][8][0][2]
                            break
                    k += 1
                waypoints.extend([option[0], option[1], option[2], px, py])
                
            px, py = 9999., 9999.
            for t in state_vector[i][6]:
                if np.sqrt(px * px + py * py) >  np.sqrt((t[0] - x) * (t[0] - x) + (t[1] - y) * (t[1] - y)):
                    px, py = rotate(t[0] - x, t[1] - y, yawsin, yawcos)
            if px == 9999.:
                px = 0.
                py = 0.
            history_exp[i].append( [np.concatenate([[velocity, state_vector[i][5], px, py], waypoints, other_vcs[:8,:5].flatten()]), route])
    return history_exp

tf.disable_eager_execution()
sess = tf.Session()
with sess.as_default():
    with multiprocessing.Pool(processes=50) as pool:
        learner = DrivingStyleLearner(state_len=state_len, traj_len=traj_len, global_input_len=global_input_len, global_latent_len=global_latent_len)
        learner_saver = tf.train.Saver(var_list=learner.trainable_dict, max_to_keep=0)
        sess.run(tf.global_variables_initializer())
        learner.network_initialize()
        log_file.write("Epoch" + learner.log_caption() + "\n")

        history = []

        for epoch in range(1, 10000):
            pkl_index = random.randrange(7)
            with open("data/gathered_from_npc_batjeon/data_" + str(pkl_index) + ".pkl","rb") as fr:
                data = pickle.load(fr)
            print("Epoch " + str(epoch) + " Start with data " + str(pkl_index))

            history_data = []
            for result in pool.imap(parallel_task, data):
                history_data.append(result)
            history.append(history_data)

            print("Current History Length : " + str(len(history)))
            for iter in range(len(history) * 256):

                data_index = random.randrange(len(history))
                exp_index = random.randrange(len(history[data_index]))
                print("Train Step #" + str(iter) + "Read data " + str(data_index) + " exp " + str(exp_index))

                cur_history = history[data_index][exp_index]
                agent_num = len(cur_history)
                
                agent_dic = random.choices(list(range(agent_num)), k=16)
                step_dic = [ random.randrange(len(cur_history[x]) - traj_len) for x in agent_dic ]

                state_dic = []
                nextstate_dic = []
                for x in range(16):
                    state_dic.append([cur_history[agent_dic[x]][step][0] for step in range(step_dic[x], step_dic[x] + traj_len)])
                    nextstate_dic.append([cur_history[agent_dic[x]][step][1] for step in range(step_dic[x], step_dic[x] + traj_len)])
                learner.optimize(state_dic, nextstate_dic)
        
                
            if len(history) > 8:
                history = history[1:]

            learner.log_print()
            log_file.write(str(epoch) + "\t" + learner.current_log() + "\n")
            log_file.flush()
            learner.network_update()


            if epoch % 50 == 0:
                learner_saver.save(sess, "train_log/DrivingStyle/log_" + log_name + "_" + str(epoch) + ".ckpt")
