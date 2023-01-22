#!/usr/bin/env python

import logging
import argparse
import os
import sys
sys.path.insert(1, "/home/avalocal/catkin_ws/src/laneatt_ros")
sys.path.append('/home/avalocal/anaconda3/envs/laneatt/lib/python3.8/site-packages')
#print(sys.path)
import torch
import rospy
import laneatt_ros
from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment


exp_name="/home/avalocal/catkin_ws/src/laneatt_ros/src/experiments/laneatt_r34_culane_final_annotations"
mode="Test"
#cfg=None
args="None"
resume="True"
view='all'
img_topic="/kitti/camera_color_right/image_raw"
def main():

    exp = Experiment(exp_name, args, mode=mode)

    cfg_path = exp.cfg_path
 
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    runner = Runner(cfg, exp, device,img_topic, resume=resume, view=view, deterministic=True)
    
    runner.eval(exp.get_last_checkpoint_epoch(), save_predictions=False)


if __name__ == '__main__':
    
    rospy.init_node('laneattNode')

    main()
    rospy.spin()

