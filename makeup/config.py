import os.path as osp

from easydict import EasyDict as edict

config = edict()
config.pwd = osp.split(osp.realpath(__file__))[0]

# add by VBT
config.w_visual = 0.01
config.w_weight = 200
config.bottlenect_stage = [2]
config.tnet_whiten = True
config.num_channel = 1
config.diff_size = 64
config.k_size = 1

