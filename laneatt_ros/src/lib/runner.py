import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange

from PIL import Image as IM
import torchvision.transforms as transforms
import rospy
import ros_numpy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Point32
from laneatt_ros.msg import DetectedLane

from cv_bridge import CvBridge, CvBridgeError

rect=np.array([[3407.91772, 0.0000000000, 1066.72048, 0.0000000000],
                          [0.00000000, 3451.94116, 825.36976, 0.00000000000],
                          [0.0000000000, 0.0000000000, 1.0000000, 0.0000000000]])

D= np.array([-0.245253, 0.149647, 0.003117, 0.000761, 0.0])
width, height=2064, 1544
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(rect[0:3, 0:3], D, (width, height), 1, (width, height))


class Runner:
    def __init__(self, cfg, exp, device, image_topic, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.image_topic=image_topic
        self.logger = logging.getLogger(__name__)
        self.output_lanepoints = []
        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


    def eval(self, epoch, on_val=False, save_predictions=False):

        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            self.bridge = CvBridge()
            #dataloader = self.get_test_dataloader()
            global model 
            model = self.cfg.get_model() #these lines load the model parameters - I had them in the callback so it reloaded them everytime there was a new image which makes it slow down
            model_path = self.exp.get_checkpoint_path(15)
            self.logger.info('Loading model %s', model_path)
            model.load_state_dict(self.exp.get_epoch_model(15))
            model = model.to(self.device)
            model.eval()
            test_parameters = self.cfg.get_test_parameters()
            predictions = []
            conf = 0
            self.exp.eval_start_callback(self.cfg)



            images= rospy.Subscriber(self.image_topic, Image, self.img_callback,queue_size=1, buff_size=2**12) #subscribes to the image topic from ros (realtime or rosbag) /camera_fl/image_color  /image_proc_resize/image
            #waymo
            #images = rospy.Subscriber('/kitti/camera_color_right/image_raw', Image, self.img_callback,queue_size=1, buff_size=2**12)
            
            
            self.line_pub=rospy.Publisher("~published_line", Image, queue_size=1)
            self.lane_pub=rospy.Publisher("~LanesArrays", DetectedLane, queue_size=1)
               
    def img_callback(self, data):
        #rate = rospy.Rate(10)
        try:
            #time1 = rospy.Time.now()

            transform = transforms.ToTensor() #sets up the transformation function
            images = ros_numpy.numpify(data) #extract image data from ROS image message
            #camera_id = data.header.frame_id ########################################################################################unditortion
            undis=cv2.undistort(images, rect[0:3,0:3], D, None , newcameramtx)
            x, y, w, h = roi
            images = undis[y:y+h, x:x+w]

            #####################################################################################
            img = images
            images = cv2.resize(images, (640,360))#(1640, 590)) # resize to correct model input size 

            #img = images #save as resized image

            images = transform(images.astype(np.uint8)) #transform images to Tensor (the uint8 may need to be changed depending on camera or image format?)
            
            #print(images)
        except CvBridgeError as e:
            print('e')

        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        conf = 0
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
                #images = images[:3, :, :] #only needed if using carla images

                images = images.unsqueeze(0) #this adds a line to the beginning of the images 
		
                #print(images.shape) #for debugging ,needs to be 1, 3, height, width
                images = images.to(self.device) #moves tensor to GPU format (?!)
                #print("1")
                output, conf = model(images, **test_parameters) #calls the model and inputs images + parameters
                avg_conf =(np.average(conf.cpu())) #convert tensor to cpu and average
                avg_conf = avg_conf.item() #convert numpy float32 to native python float32

                prediction, midlane = model.decode(output, as_lanes=True) #outputs the predicted lanes
                predictions.extend(prediction)
                
                #print("Confidence Scores for points:")
                #print(conf)

                #print("2")
                #print(prediction)
                #print(len(prediction[0])) #prints numbers of predictions (for debugging
                #print(images.shape)
                i = 0
                lanemsg = DetectedLane()

                while i < len(prediction[0]): #iterate through the predicted lanes, extract and draw the lines, overlay on the image
                        points = []

                	#print(len(prediction[0][i].points))
                        points = prediction[0][i].points
                        #print(points)
                        
                        points[:, 0] *= img.shape[1] #scales the detected lanes to the image width and height
                        points[:, 1] *= img.shape[0]  #scales the detected lanes to the image width and height
                        points = points.round().astype(int)  #rounds the points to int because cv2.line can't handle floats
                        if len(midlane)>0: 
                            midlane[:, 0] *= img.shape[1]
                            midlane[:, 1] *= img.shape[0]
                            midlane = midlane.round().astype(int)

                        for curr_p, next_p in zip(points[:-1], points[1:]):
                            img = cv2.line(img, tuple(curr_p), tuple(next_p), color=(255, 0, 0), thickness=3 )
                            if i == 0:
                                lanemsg.line1.append(Point32(curr_p[0], curr_p[1], 0)) 
                            elif i == 1:
                                lanemsg.line2.append(Point32(curr_p[0], curr_p[1], 0)) 
                            elif i == 2: 
                                lanemsg.line3.append(Point32(curr_p[0], curr_p[1], 0))
 
                        #for curr_p1, next_p1 in zip(midlane[:-1], midlane[1:]):
                        #        img = cv2.line(img, tuple(curr_p1), tuple(next_p1), color=(0, 255, 0), thickness=2)
                        i+=1
                	#print(i)

                #shows the images with overlay - this can just be replaced by publishing a rosimage but this was easier for now
                #cv2.imshow('predictions', img)
                #cv2.waitKey(1)

                img_out0=img[...,::-1]
                img_out=IM.fromarray(img_out0,'RGB') #RGBA with img for CARLA, RGB with img_out0 for actual images
                msg=Image()
                msg.header.stamp=data.header.stamp#rospy.Time.now()
                #lanemsg.camera_id = camera_id
                lanemsg.header.stamp=data.header.stamp#rospy.Time.now()
                msg.height=img_out.height
                msg.width=img_out.width
                msg.encoding="rgb8" #needs to be changed to bgra8 when using carla, rgb8 for actual images, bgr8 for waymo
                msg.is_bigendian=False
                msg.step=3*img_out.width
                msg.data=np.array(img_out).tobytes()
                lanemsg.confidence.data = avg_conf
                self.line_pub.publish(msg)
                self.lane_pub.publish(lanemsg)
                #time2 = rospy.Time.now()
                #rate.sleep()

                #time2 = rospy.Time.now()
                #print("Runtime:") #can be used to find processing time for each callback
                #print(str(time2.to_sec()-time1.to_sec()))#1 = rospy.time.now()
                

  


    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
