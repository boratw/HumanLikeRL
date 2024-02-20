# Copyright (c) 2023 Taewoo Kim

import numpy as np
import tensorflow.compat.v1 as tf
import math
from network.gaussian_policy import GaussianPolicy
from network.mlp import MLP


EPS = 1e-5

class SAC:
    def __init__(self, state_len, action_len,  name="",
        value_hidden_len=[256, 256], value_hidden_nonlinearity=tf.nn.leaky_relu, policy_hidden_len=[256, 256], policy_hidden_nonlinearity=tf.nn.leaky_relu,
        value_lr=0.0001, policy_lr=0.0001, alpha_lr = 0.0001, policy_gamma=0.98, kl_reg=0.001,  l2_reg=0.0001, policy_update_ratio=0.05, learning_rate_decay=None) :

        self.name = "SAC" + name
        self.target_entropy=-action_len
        with tf.variable_scope(self.name): 
            self.input_state = tf.placeholder(tf.float32, [None, state_len], name="input_state")
            self.input_next_state = tf.placeholder(tf.float32, [None, state_len], name="input_next_state")
            self.input_action = tf.placeholder(tf.float32, [None, action_len], name="input_action")
            self.input_reward = tf.placeholder(tf.float32, [None, 1], name="input_reward")
            self.input_survive = tf.placeholder(tf.float32, [None, 1], name="input_survive")
            self.input_iter = tf.placeholder(tf.int32, [], name="input_iter")
            self.input_dropout = tf.placeholder(tf.float32, None, name="input_dropout")

            if learning_rate_decay is not None:
                value_lr = tf.train.exponential_decay(value_lr, self.input_iter, 100, learning_rate_decay)
                policy_lr = tf.train.exponential_decay(policy_lr, self.input_iter, 100, learning_rate_decay)
                alpha_lr = tf.train.exponential_decay(alpha_lr, self.input_iter, 100, learning_rate_decay)
                
            

            self.log_alpha = tf.Variable(0., trainable=True)
            self.alpha = tf.exp(self.log_alpha)
            self.policy = GaussianPolicy("policy", state_len, action_len, policy_hidden_len, hidden_nonlinearity=policy_hidden_nonlinearity,
                input_tensor=self.input_state, output_tanh=True, input_dropout=self.input_dropout)
            self.next_policy = GaussianPolicy("policy", state_len , action_len, policy_hidden_len, hidden_nonlinearity=policy_hidden_nonlinearity,
                input_tensor=self.input_next_state, output_tanh=True, input_dropout=self.input_dropout, reuse=True)

            self.qvalue1 = MLP("qvalue1", state_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action, input_dropout=self.input_dropout)
            self.qvalue2 = MLP("qvalue2", state_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.input_action, input_dropout=self.input_dropout)
            self.qvalue1_policy = MLP("qvalue1", state_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.policy.reparameterized, 
                input_dropout=self.input_dropout, reuse=True)
            self.qvalue2_policy = MLP("qvalue2", state_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                input_tensor=self.input_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.policy.reparameterized, 
                input_dropout=self.input_dropout, reuse=True)

            self.qvalue1_target = MLP("qvalue1_target", state_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                input_tensor=self.input_next_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.next_policy.reparameterized, 
                input_dropout=self.input_dropout)
            self.qvalue2_target = MLP("qvalue2_target", state_len, 1, value_hidden_len, hidden_nonlinearity=value_hidden_nonlinearity,
                input_tensor=self.input_next_state, additional_input=True, additional_input_dim=action_len, additional_input_tensor=self.next_policy.reparameterized, 
                input_dropout=self.input_dropout)

            self.qvalue1_assign = self.qvalue1_target.build_add_weighted(self.qvalue1, 1.0)
            self.qvalue1_update = self.qvalue1_target.build_add_weighted(self.qvalue1, policy_update_ratio)
            self.qvalue2_assign = self.qvalue2_target.build_add_weighted(self.qvalue2, 1.0)
            self.qvalue2_update = self.qvalue2_target.build_add_weighted(self.qvalue2, policy_update_ratio)
            
            min_next_Q =  tf.minimum(self.qvalue1_target.layer_output, self.qvalue2_target.layer_output)
            Q_target = tf.stop_gradient(self.input_reward + (min_next_Q - self.next_policy.log_pi * self.alpha) * policy_gamma * self.input_survive )

            self.qvalue1_optimizer = tf.train.AdamOptimizer(value_lr)
            self.qvalue1_loss = tf.reduce_mean((self.qvalue1.layer_output - Q_target) ** 2) + self.qvalue1.l2_loss * l2_reg
            self.qvalue1_train = self.qvalue1_optimizer.minimize(self.qvalue1_loss,
                var_list=self.qvalue1.trainable_params)
            self.qvalue2_optimizer = tf.train.AdamOptimizer(value_lr)
            self.qvalue2_loss = tf.reduce_mean((self.qvalue2.layer_output - Q_target) ** 2) + self.qvalue2.l2_loss * l2_reg
            self.qvalue2_train = self.qvalue2_optimizer.minimize(self.qvalue2_loss,
                var_list=self.qvalue2.trainable_params)

            mean_Q = tf.reduce_mean([self.qvalue1_policy.layer_output, self.qvalue2_policy.layer_output], axis=0)
            self.policy_loss = tf.reduce_mean(self.policy.log_pi * self.alpha - mean_Q) \
                + self.policy.regularization_loss * kl_reg + self.policy.l2_loss * l2_reg
            self.policy_optimizer = tf.train.AdamOptimizer(policy_lr)
            self.policy_train = self.policy_optimizer.minimize(self.policy_loss,
                var_list=self.policy.trainable_params)


            self.alpha_loss = tf.reduce_mean(-1. * (self.alpha * tf.stop_gradient(self.policy.log_pi + self.target_entropy)))
            self.alpha_optimizer = tf.train.AdamOptimizer(alpha_lr)
            self.alpha_train = self.alpha_optimizer.minimize(self.alpha_loss,
                var_list=[self.log_alpha])

            self.qvalue1_average = tf.reduce_mean(self.qvalue1.layer_output)
            self.qvalue2_average = tf.reduce_mean(self.qvalue2.layer_output)
            self.policy_average = tf.reduce_mean(self.policy.log_pi)
                
                    


            self.trainable_params = tf.trainable_variables(scope=tf.get_variable_scope().name)
            def nameremover(x, n):
                index = x.rfind(n)
                x = x[index:]
                x = x[x.find("/") + 1:]
                return x
            self.trainable_dict = {nameremover(var.name, self.name) : var for var in self.trainable_params}


    def get_action(self, input_state, discrete=False):
        input_list = {self.input_state : input_state,  self.input_dropout : 0.0}
        sess = tf.get_default_session()
        if discrete:
            output = sess.run(self.policy.output_discrete, input_list)
        else:
            output = sess.run(self.policy.reparameterized, input_list)
        return output

    def optimize(self, input_state, input_next_state, input_action, input_reward, input_survive, input_iter=None):
        input_list = {self.input_state : input_state, self.input_next_state : input_next_state, 
            self.input_action : input_action, self.input_reward : input_reward,
            self.input_survive : input_survive, self.input_iter : input_iter, self.input_dropout : 0.1}
        sess = tf.get_default_session()

        _, _, l1, l2 = sess.run([self.qvalue1_train, self.qvalue2_train,
            self.qvalue1_average, self.qvalue2_average], input_list)
        _, _, l3, l4 = sess.run([self.policy_train, self.alpha_train,
            self.policy_average, self.alpha], input_list)

        self.log_policy_q1 += l1
        self.log_policy_q2 += l2
        self.log_policy_p += l3
        self.log_policy_a += l4
        self.log_num_explorer += 1


    def network_initialize(self):
        sess = tf.get_default_session()
        sess.run([self.qvalue1_assign, self.qvalue2_assign])
        self.log_policy_q1 = 0
        self.log_policy_q2 = 0
        self.log_policy_p = 0
        self.log_policy_a = 0
        self.log_num_explorer = 0

    def network_update(self):
        self.log_policy_q1 = 0
        self.log_policy_q2 = 0
        self.log_policy_p = 0
        self.log_policy_a = 0
        self.log_num_explorer = 0

    def network_intermediate_update(self):
        sess = tf.get_default_session()
        sess.run([self.qvalue1_update, self.qvalue2_update])

    def log_caption(self):
        return "\t" + self.name + "_Avg_Qvalue1\t" + self.name + "_Avg_Qvalue2\t" + self.name + "_Avg_Policy\t"  + self.name + "_Avg_Alpha\t"
            
    
    def current_log(self):
        log_num_explorer = self.log_num_explorer if self.log_num_explorer > 0 else 1
        return "\t" + str(self.log_policy_q1 / log_num_explorer) + "\t" + str(self.log_policy_q2 / log_num_explorer) \
            + "\t" + str(self.log_policy_p / log_num_explorer)  + "\t" + str(self.log_policy_a / log_num_explorer) 

    def log_print(self):
        log_num_explorer = self.log_num_explorer if self.log_num_explorer > 0 else 1
        print ( self.name + "\n" \
            + "\tAvg_Qvalue1                      : " + str(self.log_policy_q1 / log_num_explorer) + "\n" \
            + "\tAvg_Qvalue2                      : " + str(self.log_policy_q2 / log_num_explorer) + "\n" \
            + "\tAvg_Policy                       : " + str(self.log_policy_p / log_num_explorer) + "\n" \
            + "\tAvg_Alpha                        : " + str(self.log_policy_a / log_num_explorer) )