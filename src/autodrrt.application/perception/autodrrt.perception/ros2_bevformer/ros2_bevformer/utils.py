from autoware_auto_perception_msgs.msg import DetectedObject
from autoware_auto_perception_msgs.msg import ObjectClassification
from autoware_auto_perception_msgs.msg import DetectedObjectKinematics
from autoware_auto_perception_msgs.msg import Shape

from geometry_msgs.msg import Point
from geometry_msgs.msg import Vector3
from trimesh.transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion as GeoQuaternion
from geometry_msgs.msg import Twist
import numpy as np 
import math

class CustomBox3D(object):
  def __init__(self,nid,score,x,y,z,w,l,h,rt,vel_x,vel_y):
      self.id = nid
      self.score = score
      self.x = x
      self.y = y
      self.z = z
      self.w = w
      self.l = l
      self.h = h
      self.rt = rt
      self.vel_x = vel_x
      self.vel_y = vel_y

def isCarLikeVehicle(label):
   return label == ObjectClassification.BICYCLE or label == ObjectClassification.BUS or \
         label == ObjectClassification.CAR or label == ObjectClassification.MOTORCYCLE or \
         label == ObjectClassification.TRAILER or label == ObjectClassification.TRUCK 

def getSemanticType(class_name):
    if (class_name == "CAR" or class_name == "Car"):
       return ObjectClassification.CAR
    elif (class_name == "TRUCK" or class_name == "Medium_Truck" or class_name =="Big_Truck"):
       return ObjectClassification.TRUCK
    elif (class_name == "BUS"):
       return ObjectClassification.BUS
    elif (class_name == "TRAILER"):
       return ObjectClassification.TRAILER
    elif (class_name == "BICYCLE"):
       return ObjectClassification.BICYCLE
    elif (class_name == "MOTORBIKE"):
       return ObjectClassification.MOTORCYCLE
    elif (class_name == "PEDESTRIAN" or class_name == "Pedestrian"):
       return ObjectClassification.PEDESTRIAN
    else: 
       return ObjectClassification.UNKNOWN

def createQuaternionFromYaw(yaw):
  # tf2.Quaternion
  # q.setRPY(0, 0, yaw)
  q = quaternion_from_euler(0, 0, yaw)
  # geometry_msgs.msg.Quaternion
  #return tf2.toMsg(q)
  #return GeoQuaternion(*q)
  return GeoQuaternion(x=q[0],y=q[1],z=q[2],w=q[3])


def createPoint(x, y, z):
  p = Point()
  p.x = float(x)
  p.y = float(y)
  p.z = float(z)
  return p

def box3DToDetectedObject(box3d, class_names, has_twist, is_sign):
  obj = DetectedObject()
  obj.existence_probability = float(box3d.score)
 
  classification = ObjectClassification()
  classification.probability = 1.0
  if (box3d.id >= 0 and box3d.id < len(class_names)):
    classification.label = getSemanticType(class_names[box3d.id])
  else: 
    if is_sign:
      sign_label = 255
      classification.label = sign_label
    else:
      classification.label = ObjectClassification.UNKNOWN
      print("Unexpected label: UNKNOWN is set.")
  
  if (isCarLikeVehicle(classification.label)):
    obj.kinematics.orientation_availability = DetectedObjectKinematics.SIGN_UNKNOWN
 
  obj.classification.append(classification)
 
  # pose and shape
  # mmdet3d yaw format to ros yaw format
  yaw = -box3d.rt - np.pi / 2
  obj.kinematics.pose_with_covariance.pose.position = createPoint(box3d.x, box3d.y, box3d.z)
  obj.kinematics.pose_with_covariance.pose.orientation = createQuaternionFromYaw(yaw)
  obj.shape.type = Shape.BOUNDING_BOX
  obj.shape.dimensions = createTranslation(box3d.l, box3d.w, box3d.h)
  # twist
  if (has_twist):
    vel_x = float(box3d.vel_x)
    vel_y = float(box3d.vel_y)
    twist = Twist()
    twist.linear.x = math.sqrt(pow(vel_x, 2) + pow(vel_y, 2))
    twist.angular.z = 2 * (math.atan2(vel_y, vel_x) - yaw)
    obj.kinematics.twist_with_covariance.twist = twist
    obj.kinematics.has_twist = has_twist
  return obj  


def createTranslation(x, y, z):
  v = Vector3()
  v.x = float(x)
  v.y = float(y)
  v.z = float(z)
  return v