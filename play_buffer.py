import os
import time
from collections import deque
import tensorflow as tf
import numpy as np


#batch=np.array([[[1,2,3],[123],[200],[4,5,6],[True]],[[1,2,3],[123],[200],[4,5,6],[True]]],dtype=object)      
#states, actions, rewards, next_states, dones = [
#                np.array([experience[field_index] for experience in batch])
#                for field_index in range(5)]

#print(states)
s=[1,2,3,4,5,5]
x=np.array(s)
a=np.array(2) * x

a = np.clip(np.random.normal(350, 3), -20, 650)
print(a)