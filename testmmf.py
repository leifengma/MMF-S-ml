#import SST_env
import net_work
from net_work import DDPG
from SST_env import Test_Env_PI
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import IPython
import tqdm
import time


PROJECT_ROOT_DIR = "."
CHAPTER_ID = "search_test"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

WEIGHT_PATH = os.path.join(PROJECT_ROOT_DIR, "WEIGHT")
os.makedirs(WEIGHT_PATH, exist_ok=True)



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




def train_model():
    
    #初始化环境
    env=Test_Env_PI()

    # reproducible，设置随机种子，为了能够重现
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    #定义状态空间，动作空间，动作幅度范围
    s_dim = 2
    a_dim = 1
    a_bound = 350


    #用DDPG算法
    ddpg = DDPG(a_dim, s_dim, a_bound, MEMORY_CAPACITY)

    #训练部分：
    #if args.train:  # train
        
    reward_buffer = []      #用于记录每个EP的reward，统计变化
    t0 = time.time()        #统计时间
    best_score=-np.inf
    current_score=0
    for i in range(MAX_EPISODES):
        t1 = time.time()
        s = env.reset()
        ep_reward = 0       #记录当前EP的reward
        for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = ddpg.choose_action(s)       #这里很简单，直接用actor估算出a动作

            # 为了能保持开发，这里用了另外一种方式增加探索。
            # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
            # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
            # 然后进行裁剪
            a = np.clip(np.random.normal(a, VAR), -20, 650)[0]  
            # 与环境进行互动
            s_, r, done, info = env.step(a)

            # 保存s，a，r，s_
            ddpg.store_transition(s, a, r / 10, s_)

            # 第一次数据满了，就可以开始学习
            if ddpg.pointer > MEMORY_CAPACITY:
                ddpg.learn()

            #输出数据记录
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            #if j == MAX_EP_STEPS - 1:
            print(
                '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}| State: {}'.format(
                    i, MAX_EPISODES, ep_reward,
                    time.time() - t1,s
                ), end=''
            )
            #plt.show()
        # test
        if i and not i % TEST_PER_EPISODES:
            #t1 = time.time()
            s = env.reset()
            ep_reward = 0
            for j in range(MAX_EP_STEPS):

                a = ddpg.choose_action(s)  
                s_, r, done, info = env.step(a)

                s = s_
                ep_reward += r
                if j == MAX_EP_STEPS - 1:
                    print("Episode Reward:"+str(ep_reward))
                    #    '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    #        i, MAX_EPISODES, ep_reward,
                    #        time.time() - t1
                    #    )
                    #)
                    print('\n')
                    current_score=ep_reward
                    reward_buffer.append(ep_reward)
                if current_score>best_score:
                    ddpg.save_ckpt()

        if reward_buffer:
            plt.ion()
            plt.cla()
            plt.title('DDPG')
            plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
            plt.xlabel('episode steps')
            plt.ylabel('normalized state-action value')
            plt.ylim(-2000, 0)
            plt.show()
    plt.ioff()
    plt.show()
    print('\nRunning time: ', time.time() - t0)
    ddpg.save_ckpt()







def use_Agent():
    #初始化环境
    env=Test_Env_PI()

    # reproducible，设置随机种子，为了能够重现
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    #定义状态空间，动作空间，动作幅度范围
    s_dim = 2
    a_dim = 1
    a_bound = 350


    #用DDPG算法
    ddpg = DDPG(a_dim, s_dim, a_bound, MEMORY_CAPACITY)


    ddpg.load_ckpt()
    state = env.reset()
    action=ddpg.choose_action(state)
    for episode in range(1000):
        env.step(action)
        state=env.state
        action=ddpg.choose_action(state)
    
    y_print=env.y_1_rec


    plt.plot(y_print)
    plt.show()
    #plt.plot(Y)
    #plt.show()
    #plt.plot(Z)
    #plt.show()




if __name__ == '__main__':
    train_model()
    use_Agent()







