from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import math
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from typing import Any, List, Sequence, Tuple
import numpy as np

tf.compat.v1.enable_v2_behavior()

def constrain_amp(input):
    if input>10000:
        return 10000
    if input<-10000:
        return -10000

class SST_Env(gym.Env):



    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(4,), dtype=np.float64, minimum=[-20.0, -10.0,-20.0,-10.0], maximum=[20.0, 10.0,20.0,10.0],name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.float64, minimum=600,maximum=650, name='observation')
        #self._reward_spec = array_spec.BoundedArraySpec(
        #shape=(1,), dtype=np.int32, minimum=-30,maximum=30, name='reward')
        self.viewer = None
        self.state = 22
        self.P_1=[-1.122]
        self.I_1=[-0.0299]
        self.D_1=[0.6151]
        #self.P_2=[0.5848]
        #self.I_2=[0.005694]
        self.u_1=[0]*3
        #self.u_2=[0]*4

        self._current_time_step = None
        self._state = 0
        self._episode_ended = False
        self.max_step_per_episode = 200

        self.Error_1=[0]*2
        self.Error_2=[0]*2
        self.TEMP_set=650
        self.TEMP_current=640
        self.Temp_pre_current=645
        self.y_1=[self.Temp_pre_current]*5
        self.y_2=[self.TEMP_current]*5
        
        

        self.Error_2_rec=[]
        self.u_1_rec=[]
        self.u_2_rec=[]
        self.y_2_rec=[]
        #self._reset()

    def seed(self, seed=None):####????????
        self.np_random, seed = seeding.np_random(seed) 
        return [seed]

    def states(self):
        return dict(type="float", shape=(5,))

    def actions(self):
        return {
            "P_1": dict(type="float", num_values=range(-20,20)),
            "P_2": dict(type="float", num_values=range(-20,20)),
            "I_1": dict(type="float", num_values=range(-10,10)),
            "I_2": dict(type="float", num_values=range(-10,10)),
                }

    def step(self, action):
        #err_msg = "%r (%s) invalid" % (action, type(action))
        #assert self._action_spec.check_array(action), err_msg
        
        if action==0:
            self.P_1[0]+=1
        if action==1:
            self.I_1[0]-=1
        if action==2:
            self.D_2[0]+=1
        

        #传递函数
        #外环计算
        self.Error_2[1]=self.TEMP_set-self.y_2[3]
        self.Error_2_rec.append(self.Error_2[1])
        Delta_u_2=self.P_2[0]*(self.Error_2[1]-self.Error_2[0])+self.I_2[0]*self.Error_2[1]
        self.u_2[3]=Delta_u_2+self.u_2[3]
        if self.u_2[3]>10000:
            self.u_2[3] = 10000        
        if self.u_2[3]<-10000:
            self.u_2[3] = -10000
        self.u_2_rec.append(self.u_2[3])

        #内环计算
        self.Error_1[1]=self.u_2[3]-self.y_1[4]
        Delta_u_1=self.P_1[0]*(self.Error_1[1]-self.Error_1[0])+self.I_1[0]*self.Error_1[1]
        self.u_1[2]=Delta_u_1+self.u_1[2] 
        if self.u_1[2]>10000:
            self.u_1[2] = 10000
        if self.u_1[2]<-10000:
            self.u_1[2] = -10000
        self.u_1_rec.append(self.u_1[2])
        self.y_1[4]=2*self.y_1[3]-self.y_1[2]+2e-7*self.u_1[1]-2e-7*self.u_1[0]

        #内环转外环
        self.y_2[4]=(4*self.y_2[3]-6*self.y_2[2]+4*self.y_2[1]-self.y_2[0]+
                     8e-16*self.y_1[3]+1e-15*self.y_1[2]+1e-15*self.y_1[1]+9e-16*self.y_1[0])
        self.y_2_rec.append(self.y_2[4])
        #赋值
        self.Error_2.pop(0)
        self.Error_2.append(0)
        self.u_2.pop(0)
        self.u_2.append(0)
        self.Error_1.pop(0)
        self.Error_1.append(0)
        self.u_1.pop(0)
        self.u_1.append(0)
        self.y_1.pop(0)
        self.y_1.append(0)
        self.y_2.pop(0)
        self.y_2.append(0)
        
        reward=self.Error_2
    
        return self.Error_2_rec
    

    #def _reset(self):
    #    self.y_1=[0]*3
    #    self.y_2=[0]*4
    #    self.TEMP_set=1000
  
    #    self._current_time_step = self._reset()
    #    return self._current_time_step


    #def step(self, action):
    #    #err_msg = "%r (%s) invalid" % (action, type(action))
    #    #assert self._action_spec.check_array(action), err_msg
        
    #    if action==0:
    #        self.P_1[0]+=1
    #    if action==1:
    #        self.P_1[0]-=1
    #    if action==2:
    #        self.P_2[0]+=1
    #    if action==3:
    #        self.P_2[0]-=1
    #    if action==4:
    #        self.I_1[0]+=1
    #    if action==5:
    #        self.I_1[0]-=1
    #    if action==6:
    #        self.I_2[0]+=1
    #    if action==7:
    #        self.I_2[0]-=1

    #    #传递函数
    #    #外环计算
    #    self.Error_2[1]=self.TEMP_set-self.y_2[3]
    #    self.Error_2_rec.append(self.Error_2[1])
    #    Delta_u_2=self.P_2[0]*(self.Error_2[1]-self.Error_2[0])+self.I_2[0]*self.Error_2[1]
    #    self.u_2[3]=Delta_u_2+self.u_2[3]
    #    if self.u_2[3]>10000:
    #        self.u_2[3] = 10000        
    #    if self.u_2[3]<-10000:
    #        self.u_2[3] = -10000
    #    self.u_2_rec.append(self.u_2[3])

    #    #内环计算
    #    self.Error_1[1]=self.u_2[3]-self.y_1[4]
    #    Delta_u_1=self.P_1[0]*(self.Error_1[1]-self.Error_1[0])+self.I_1[0]*self.Error_1[1]
    #    self.u_1[2]=Delta_u_1+self.u_1[2] 
    #    if self.u_1[2]>10000:
    #        self.u_1[2] = 10000
    #    if self.u_1[2]<-10000:
    #        self.u_1[2] = -10000
    #    self.u_1_rec.append(self.u_1[2])
    #    self.y_1[4]=2*self.y_1[3]-self.y_1[2]+2e-7*self.u_1[1]-2e-7*self.u_1[0]

    #    #内环转外环
    #    self.y_2[4]=(4*self.y_2[3]-6*self.y_2[2]+4*self.y_2[1]-self.y_2[0]+
    #                 8e-16*self.y_1[3]+1e-15*self.y_1[2]+1e-15*self.y_1[1]+9e-16*self.y_1[0])
    #    self.y_2_rec.append(self.y_2[4])
    #    #赋值
    #    self.Error_2.pop(0)
    #    self.Error_2.append(0)
    #    self.u_2.pop(0)
    #    self.u_2.append(0)
    #    self.Error_1.pop(0)
    #    self.Error_1.append(0)
    #    self.u_1.pop(0)
    #    self.u_1.append(0)
    #    self.y_1.pop(0)
    #    self.y_1.append(0)
    #    self.y_2.pop(0)
    #    self.y_2.append(0)
        
    #    reward=self.Error_2
    
    #    return self.Error_2_rec
    #def step(self):
    #    #err_msg = "%r (%s) invalid" % (action, type(action))
    #    #assert self._action_spec.check_array(action), err_msg
        
        

    #    #传递函数
    #    #外环计算
    #    self.Error_2[1]=self.TEMP_set-self.y_2[4]
    #    self.Error_2_rec.append(self.Error_2[1])
    #    Delta_u_2=self.P_2[0]*(self.Error_2[1]-self.Error_2[0])+self.I_2[0]*self.Error_2[1]
    #    self.u_2[3]=Delta_u_2+self.u_2[3]
    #    if self.u_2[3]>10:
    #        self.u_2[3] =10        
    #    if self.u_2[3]<-10:
    #        self.u_2[3] = -10
    #    self.u_2_rec.append(self.u_2[3])

    #    #内环计算
    #    self.Error_1[1]=self.u_2[3]-self.y_1[4]
    #    self.y_2.pop(0)
    #    self.y_2.append(0)
    #    Delta_u_1=self.P_1[0]*(self.Error_1[1]-self.Error_1[0])+self.I_1[0]*self.Error_1[1]
    #    self.u_1[2]=Delta_u_1+self.u_1[2] 
    #    if self.u_1[2]>10:
    #        self.u_1[2] = 10
    #    if self.u_1[2]<-10:
    #        self.u_1[2] = -10
    #    self.u_1_rec.append(self.u_1[2])
    #    self.y_1[4]=2*self.y_1[3]-self.y_1[2]-2e-7*self.u_1[1]-2e-7*self.u_1[0]

    #    #内环转外环
    #    self.y_2[4]=(4*self.y_2[3]-6*self.y_2[2]+4*self.y_2[1]-self.y_2[0]+
    #                 9e-16*self.y_1[3]+1e-14*self.y_1[2]+1e-14*self.y_1[1]+9e-16*self.y_1[0])
    #    if self.y_2[4]>700:
    #        self.y_2[4] = 700
    #    if self.y_2[4]<600:
    #        self.y_2[4] = 600
    #    self.y_2_rec.append(self.y_2[4])


    #    #赋值
    #    self.Error_2.pop(0)
    #    self.Error_2.append(0)
    #    self.u_2.pop(0)
    #    self.u_2.append(0)
    #    self.Error_1.pop(0)
    #    self.Error_1.append(0)
    #    self.u_1.pop(0)
    #    self.u_1.append(0)
    #    self.y_1.pop(0)
    #    self.y_1.append(0)

        
    #    reward=self.Error_2
    
    #    return self.Error_2_rec

        

        #return ts.transition(
        #     np.array([self.state], dtype=np.int32), reward=reward, discount=0.8)

class Test_Env_PID(gym.Env):



    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(4,), dtype=np.float64, minimum=[-20.0, -10.0,-20.0,-10.0], maximum=[20.0, 10.0,20.0,10.0],name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.float64, minimum=600,maximum=650, name='observation')
        #self._reward_spec = array_spec.BoundedArraySpec(
        #shape=(1,), dtype=np.int32, minimum=-30,maximum=30, name='reward')
        self.viewer = None
        self.state = 22
        self.P_1=[4.2]
        self.I_1=[0.01*6.339]
        self.D_1=[0.6151/0.01]
        self.u_1=[0]*3
 

        self._current_time_step = None
        self._state = 0
        self._episode_ended = False
        self.max_step_per_episode = 200

        
        self.TEMP_set=1
        self.TEMP_current=0
        self.Error_1=[self.TEMP_set-self.TEMP_current]*3
        self.y_1=[self.TEMP_current]*3


        self.Error_1_rec=[]
        self.y_1_rec=[]
        
    def step(self):
                
        #if action==0:
        #    self.P_1[0]+=1
        #if action==1:
        #    self.P_1[0]-=1
        #if action==2:
        #    self.I_1[0]+=1
        #if action==3:
        #    self.I_1[0]+=1
        #if action==4:
        #    self.D_1[0]-=1
        #if action==5:
        #    self.D_1[0]+=1


        #传递函数
        self.Error_1[2]=self.TEMP_set-self.y_1[2]
        self.Error_1_rec.append(self.Error_1[2])
        self.y_1.pop(0)
        self.y_1.append(0)
        Delta_u_1=self.P_1[0]*(self.Error_1[2]-self.Error_1[1])+self.I_1[0]*self.Error_1[2]+self.D_1[0]*(self.Error_1[2]-2*self.Error_1[1]+self.Error_1[0])
        self.u_1[2]=Delta_u_1+self.u_1[1]
        #if self.u_2[3]>10000:
        #    self.u_2[3] = 10000        
        #if self.u_2[3]<-10000:
        #    self.u_2[3] = -10000
        #self.u_2_rec.append(self.u_2[3])
        self.y_1[2]=1.979*self.y_1[1]-0.9802*self.y_1[0]+0.0004966*self.u_1[1]+0.0004933*self.u_1[0]
        self.y_1_rec.append(self.y_1[2])

        
        #赋值
        self.Error_1.pop(0)
        self.Error_1.append(0)
        self.u_1.pop(0)
        self.u_1.append(0)
        
  
  

        
        reward=self.Error_1
    
        return self.Error_1_rec


class Test_Env_PI(gym.Env):

    def __init__(self):
   
        self.reward=0
        self.u_1=[0]*3
        self.current_step = 0       
        self.max_step_per_episode = 20000
        
        self.TEMP_set=10
        self.TEMP_current=0
        self.Error_1=[self.TEMP_set-self.TEMP_current]*2
        self.Error_speed=(self.Error_1[1]-self.Error_1[0])/0.01
        self.y_1=[self.TEMP_current]*3
        self.step_flag=False
        

        self.Error_1_rec=[]
        self.y_1_rec=[]

        self.np_random=42
        self.seed()
        self.reset()
        self.state=(self.Error_1[1],self.Error_speed)



    def judge(self,done_flag):
        if done_flag:
            temp=bool(abs(self.Error_1[0])>1.5)
        else:
            temp=False
        return temp     
    

    def step(self,action):
        self.current_step+=1
###########################################################################################################################
        #传递函数
        error_tmp=self.Error_1[0]
        self.Error_1[1]=self.TEMP_set-self.y_1[2]
        self.Error_1_rec.append(self.Error_1[1])
        self.y_1.pop(0)
        self.y_1.append(0)
        #Delta_u_1=self.P_1[0]*(self.Error_1[1]-self.Error_1[0])+self.I_1[0]*self.Error_1[1]#+self.D_1[0]*(self.Error_1[2]-2*self.Error_1[1]+self.Error_1[0])
        self.u_1[2]=action       
        #self.u_2_rec.append(self.u_2[3])
        self.y_1[2]=1.979*self.y_1[1]-0.9802*self.y_1[0]+0.00059592*self.u_1[1]+0.00059196*self.u_1[0]
        #self.y_1[2]=1.979*self.y_1[1]-0.9802*self.y_1[0]+0.0005966*self.u_1[1]+0.0004933*self.u_1[0]
        self.y_1_rec.append(self.y_1[2])
        self.TEMP_current=self.y_1[2]
        #赋值
        self.Error_1.pop(0)
        self.Error_1.append(0)
        self.u_1.pop(0)
        self.u_1.append(0)
################################################################################################################################
        self.Error_speed=(self.Error_1[0]-error_tmp)/0.01
        self.state=(self.Error_1[0],self.Error_speed)
        #self.state.append(self.Error_1[0])
        #self.state.append(self.Error_speed)
        done=False
        if self.current_step>0 and self.current_step<30:
            if abs(self.Error_1[0])>10:
                done=True
        if self.current_step>30 and self.current_step<100:
            if abs(self.Error_1[0])>5:
                done=True
        if self.current_step>100 and self.current_step<300:
            if abs(self.Error_1[0])>4:
                done=True
        if self.current_step>300 and self.current_step<600:
            if abs(self.Error_1[0])>2:
                done=True
        if self.current_step>600 and self.current_step<800:
            if abs(self.Error_1[0])>1.5:
                done=True
        if self.current_step>800:
            if abs(self.Error_1[0])>1:
                done=True

        
        reward_1=-abs(self.Error_1[0])

        if abs(self.Error_1[0])<=0.3:
            reward_2=1/(abs(self.Error_1[0])+0.01)
        else:
            reward_2=0

        if abs(self.Error_1[0])>=10:
            reward_3=-100
        else:
            reward_3=0
        
        reward=reward_1+reward_2+reward_3
        
        #if self.Error_1[0]==0:
        #    reward=5
        #else:
        #    reward=0
    
        return self.state, np.array(reward, np.float32), np.array(done, np.int32), {}
               
              
              


    def tf_env_step(self,action: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(step, [action], 
                                [tf.float32, tf.int32, tf.int32])


    def run_episode(self,
        initial_state: tf.Tensor,  
        model: tf.keras.Model, 
        max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs a single episode to collect training data."""

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor (batch size = 1)
            state = tf.expand_dims(state, 0)
  
            # Run the model and to get action probabilities and critic value
            action_logits_t, value = model(state)
  
            # Sample next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]#从分类分布中抽取样本
            action_probs_t = tf.nn.softmax(action_logits_t)#计算softmax激活

            # Store critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chosen
            action_probs = action_probs.write(t, action_probs_t[0, action])
  
            # Apply action to the environment to get next state and reward
            state, reward, done = self.tf_env_step(action)
            state.set_shape(initial_state_shape)
  
            # Store reward
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):#转换格式
                break

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
  
        return action_probs, values, rewards


    def get_expected_return(self,
        rewards: tf.Tensor, 
        gamma: float, 
        standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""

        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)#倒序复制
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        ############################################################################
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / 
                        (tf.math.reduce_std(returns) + eps))

        return returns

    def reset(self):
        self.reward=0
        self.u_1=[0]*3
        self.current_step = 0       
        self.max_step_per_episode = 20000
        
        self.TEMP_set=10
        self.TEMP_current=0
        self.Error_1=[self.TEMP_set-self.TEMP_current]*2
        self.Error_speed=(self.Error_1[1]-self.Error_1[0])/0.01
        self.y_1=[self.TEMP_current]*3
        self.step_flag=False
        

        self.Error_1_rec=[]
        self.y_1_rec=[]

        self.np_random=42
        self.state=(self.Error_1[1],self.Error_speed)
        return np.array(self.state)

    def y_1(self):
        return self.y_1[2]










