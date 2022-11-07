import numpy as np
import os

data_dir = './data/ED/'
os.chdir(data_dir)

np.set_printoptions(edgeitems=50)

# print(np.load('sys_dialog_texts.train.npy', allow_pickle=True))   # dialogue
print(np.load('sys_target_texts.train.npy', allow_pickle=True))     # emotion