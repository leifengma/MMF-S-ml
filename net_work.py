import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque
from SST_env import Test_Env_PI,Test_Env_PID
from typing import Any, List, Sequence, Tuple

RANDOMSEED = 42             # random seed

LR_A = 0.001                # learning rate for actor
LR_C = 0.002                # learning rate for critic
GAMMA = 0.9                 # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 10000     # size of replay buffer
BATCH_SIZE = 32             # update batchsize

MAX_EPISODES = 50          # total number of episodes for training
MAX_EP_STEPS = 1000         # total number of steps for each episode
TEST_PER_EPISODES = 10      # test the model per episodes
VAR = 3                     # control exploration



###################################################################################




class DQN_agent():
    # DQN Agent
    def __init__(self): 
        # init experience replay
        self.replay_memory = deque(maxlen=100000)
        # init some parameters
        self.time_step = 0
        self.epsilon = 0
        #self.state_dim = state_dim
        #self.action_dim = action_dim
        self.input_shape = [5] # == env.observation_space.shape
        self.n_outputs = 4 # == env.action_space.n

        self.batch_size = 32
        self.discount_rate = 0.80
        self.optimizer = keras.optimizers.Adam(lr=1e-2)
        self.loss_fn = keras.losses.mean_squared_error


        ## Init session
        #self.session = tf.InteractiveSession()
        #self.session.run(tf.initialize_all_variables())

        #self.model = keras.models.Sequential([
        #    keras.layers.Dense(32, activation="elu", input_shape=self.input_shape),
        #    keras.layers.Dense(32, activation="elu"),
        #    keras.layers.Dense(self.n_outputs)
        #])

        
#
    def load(self,path):
        self.saver.restore(self.session, path)
    def save(self,path):
        self.saver.save(self.session, save_path=path)

    #def create_Q_network(self): #??????Q??????
         # network weights
        #W1 = self.weight_variable([self.state_dim, 20])
        #b1 = self.bias_variable([20])
        #W2 = self.weight_variable([20, self.action_dim])
        #b2 = self.bias_variable([self.action_dim])


        ## input layer
        ## self.state_in= tf.placeholder(shape=[None,self.state_dim],dtype=tf.float32)
        #self.state_input = tf.placeholder(shape=[None,self.state_dim],dtype=tf.float32)
        ## hidden layers
        ## hidden = slim.fully_connected(self.state_in,state_dim,biases_initializer=None,activation_fn=tf.nn.relu)
        #h_layer = keras.layers.Dense(state_dim,activation="relu")(self.state_input)
        ## Q Value layer
        #self.Q_value = keras.layers.Dense(action_dim)(h_layer)
        #model=keras.Model(inputs=[self.state_input],outputs=self.Q_value)
 

    


    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            Q_values = self.model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])

    def sample_experiences(self,batch_size):
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def run_one_step(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)
        self.replay_memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, info


    def training_step(self,batch_size):
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + self.discount_rate * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.n_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)#axis=1?????????
            #print(all_Q_values * mask)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class DDPG():
    def __init__(self, a_dim, s_dim, a_bound, MEMORY_CAPACITY):
        # memory????????????????????????????????????
        # ????????????MEMORY_CAPACITY???s_dim * 2 + a_dim + 1??????????????????state?????????action????????????reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound, self.MEMORY_CAPACITY = a_dim, s_dim, a_bound, MEMORY_CAPACITY
        



        # ??????actor???????????????s?????????a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = keras.layers.Input(shape=[input_state_shape], name='A_input')
            x = keras.layers.Dense(30, activation='relu', name='A_l1')(inputs)
            x = keras.layers.Dense(a_dim, activation='sigmoid', name='A_a')(x)
            x = keras.layers.Lambda(lambda x: np.array(350) * x)(x)            #?????????????????????tanh??????????????????[-1,1]????????????????????????
            x = keras.layers.Lambda(lambda x: np.array(-20) + x)(x) 
            model=keras.Model(inputs=[inputs], outputs=[x], name='Actor' + name)
            return model

        #??????Critic???????????????s???a?????????Q???
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = keras.layers.Input(shape=[input_state_shape], name='C_s_input')
            a = keras.layers.Input(shape=[input_action_shape], name='C_a_input')
            x = keras.layers.Concatenate()([s, a])
            x = keras.layers.Dense(60, activation='relu', name='C_l1')(x)
            x = keras.layers.Dense(1, name='C_out')(x)
            return keras.Model(inputs=[s, a], outputs=[x], name='Critic' + name)

        self.actor = get_actor(s_dim)
        self.critic = get_critic(s_dim, a_dim)

         #?????????????????????????????????????????????????????????
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)


        #??????actor_target???????????????actor???????????????????????????
        self.actor_target = keras.models.clone_model(self.actor)
        self.actor_target.set_weights(self.actor.get_weights())#????????? BatchNormalization ??? Dropout?????????BN???dropout???????????????

        #??????critic_target???????????????actor???????????????????????????
        self.critic_target = keras.models.clone_model(self.critic)
        self.critic_target.set_weights(self.critic.get_weights())

        #self.R = keras.layers.Input([None, 1], tf.float32, 'r')

        #??????ema??????????????????
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        ??????????????????
        """
        # ???????????????????????????????????????????????????????????????????????????ema.average???
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #??????????????????????????????actor???critic???
        self.ema.apply(paras)                                                   #???????????????????????????
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j)) 
    
    # ??????????????????s??????????????????a
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act
        """
        #a=[[0] * 1 for _ in range(2)]
        #a[0][0]=s[0]
        #a[1][0]=s[1]
        #print(s)
#print( '\rState: {}  '.format(s))
        return self.actor.predict(np.array([s], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #??????BATCH_SIZE????????????
        bt = self.memory[indices, :]                    #??????indices???????????????bt??????????????????
        bs = bt[:, :self.s_dim]                         #???bt????????????s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #???bt????????????a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #???bt????????????r
        bs_ = bt[:, -self.s_dim:]                       #???bt????????????s'

        # Critic???
        # Critic?????????DQN???????????????target??????argmax????????????critic_target??????????????????
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # Actor???
        # Actor???????????????????????????Q?????????
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.ema_update()

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # ??????s???s_,??????????????????????????????
        s = np.array(s,dtype=np.float32)
        s_ =np.array(s_,dtype=np.float32)

        #???s, a, [r], s_????????????
        s_=np.hstack((s_))
        transition = np.hstack((s, a, r, s_))

        #pointer??????????????????????????????????????????
        #index?????????????????????????????????????????????
        #???????????????????????????MEMORY_CAPACITY???????????????index???????????????????????????
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #???transition????????????s, a, [r], s_????????????
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        self.actor.save('model/ddpg_actor.hdf5')
        self.actor_target.save('model/ddpg_actor_target.hdf5')
        self.critic.save('model/ddpg_critic.hdf5')
        self.critic_target.save('model/ddpg_critic_target.hdf5')

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        self.actor=keras.models.load_model('model/ddpg_actor.hdf5')
        self.actor_target=keras.models.load_model('model/ddpg_actor_target.hdf5')
        self.critic=keras.models.load_model('model/ddpg_critic.hdf5')
        self.critic_target=keras.models.load_model('model/ddpg_critic_target.hdf5')





    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    
    def compute_loss(self,
        action_probs: tf.Tensor,  
        values: tf.Tensor,  
        returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - values

        action_log_probs = tf.math.log(action_probs)#????????????????????????pai???
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)

        return actor_loss + critic_loss

    @tf.function
    def train_step(self,
        initial_state: tf.Tensor, 
        model: tf.keras.Model,         
        gamma: float, 
        max_steps_per_episode: int) -> tf.Tensor:
        
        with tf.GradientTape() as tape:
             action_probs, values, rewards = Test_Env_PI.run_episode(
                                        initial_state, model, max_steps_per_episode)
        
             # Calculate expected returns
             returns = Test_Env_PI.get_expected_return(rewards, gamma)

             # Convert training data to appropriate TF tensor shapes
             action_probs, values, returns = [
                tf.expand_dims(x, 1) for x in [action_probs, values, returns]] 

             # Calculating loss values to update our network
             loss = compute_loss(action_probs, values, returns)

        # Compute the gradients from the loss
        grads = tape.gradient(loss, model.trainable_variables)

        # Apply the gradients to the model's parameters
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        return episode_reward