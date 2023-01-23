# Kitti_LaneDetection
This repo contains LaneAtt merged into ROS along with a script that converts the 2D image lanes to 3D points by fusing the detections with Lidar data. 
This implementation is currently based on the Kitti-Dataset converted to rosbags. The 2D to 3D fusion/conversion relies on the intrinsic and extrinsic matrices provided by the Kitti-Dataset for the right camera. Please ask Pardis for more information on this. 

# Installation: 
- For LaneAtt pre-requisites and installation follow instructions found in laneatt_ros/src. 
- then in the root of the folder (your_path_to/Test) run catkin_make (or catkin build) 
- Finally, in your terminal run: source devel/setup.sh (or add this command to your .bashrc to be persistent) 

In order to run the LaneDetection you will need a trained model, you can use any model available, train your own or use the one provided by us. 
This will require some minor modifications of the /laneatt_ros/src/main.py file. 

If you use the model below, nothing needs to be changed.

Download the "experiments" folder from the link below and save it into the laneatt_ros/src folder. 
https://drive.google.com/drive/folders/15FslYMfT4efufdZNE7CmZ3-H1I7T3qaJ?usp=share_link

# Instructions
1. conda activate *your_conda_environment* 
2. roslaunch laneatt_ros laneattLaunch.launch
3. roslaunch line3d line3dLaunch.launch
