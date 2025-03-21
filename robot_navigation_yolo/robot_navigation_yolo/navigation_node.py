import rclpy
import random
import tf_transformations
import mediapipe as mp
import cv2
import math
import json
import threading
from sensor_msgs.msg import Image
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class Navigation_robot(Node):
    
    def __init__(self):
        super().__init__('robot_nav')
        
        self.goal = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.person_detector = self.create_subscription(String, 'person_location', self.detector_callback, 10)
        self.image_subscriber = self.create_subscription(Image, 'robot_vision_yolo', self.orientation_callback, 10)
        self.bridge = CvBridge()
        self.step_exploration = 1.0
        self.step_exploration_max = 5.0
        self.robo_pose_x = 0.0
        self.robo_pose_y = 0.0
        self.result_detector = False
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.exploration_thread = threading.Thread(target=self.exploration, daemon=True)
        self.exploration_thread.start()
        
    def exploration(self):

        while rclpy.ok():
            if self.result_detector:
                
                self.get_logger().info("Pessoa detectada! Parando exploração.")
                break  

            
            move_x = random.uniform(-self.step_exploration, self.step_exploration)
            move_y = random.uniform(-self.step_exploration, self.step_exploration)

            self.robo_pose_x += move_x
            self.robo_pose_y += move_y

            goal_msg = PoseStamped()
            goal_msg.header.frame_id = 'map'
            goal_msg.header.stamp = self.get_clock().now().to_msg()

            goal_msg.pose.position.x = self.robo_pose_x
            goal_msg.pose.position.y = self.robo_pose_y
            goal_msg.pose.position.z = 0.0

            quat = tf_transformations.quaternion_from_euler(0, 0, random.uniform(-3.14, 3.14))
            goal_msg.pose.orientation.x = quat[0]
            goal_msg.pose.orientation.y = quat[1]
            goal_msg.pose.orientation.z = quat[2]
            goal_msg.pose.orientation.w = quat[3]

            self.goal.publish(goal_msg)  

            
            if self.step_exploration < self.step_exploration_max:
                self.step_exploration += 0.1

            
            rclpy.spin_once(self, timeout_sec=0.5)

        self.get_logger().info("Exploração finalizada, pessoa encontrada inciando a aproximação...")
            
    
    def detector_callback(self, msg):
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()

        if not self.result_detector:
            self.result_detector = True
            self.get_logger().info(f"Uma pessoa foi encontrada em: {msg.data}")
            
            _, offset, theta = self.orientation_callback(msg)
            
            try:
                
                pose = json.loads(msg.data)
                x1, y1, x2, y2 = pose[0]
                
                pos_personX = (x1 + x2) / 2
                pos_personY = (y1 + y2) / 2
                
                map_x = pos_personX * 0.01
                map_y = pos_personY * 0.01
                
                goal_msg.pose.position.x = map_x + offset
                goal_msg.pose.position.y = map_y
                goal_msg.pose.position.z = 0.0
                quat = tf_transformations.quaternion_from_euler(0, 0, theta)  
                goal_msg.pose.orientation.x = quat[0]
                goal_msg.pose.orientation.y = quat[1]
                goal_msg.pose.orientation.z = quat[2]
                goal_msg.pose.orientation.w = quat[3]
                
                self.goal.publish(goal_msg)
                
            except Exception as e:
                self.get_logger().error(f"Erro ao processar posição da pessoa: {e}")
                
    def orientation_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_img)
        
        theta = 0.0  # Valor padrão caso a detecção falhe
        
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            omb_esq = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            omb_dir = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            quad_esq = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            quad_dir = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

            larg_omb = abs(omb_esq.x - omb_dir.x)
            larg_quad = abs(quad_esq.x - quad_dir.x)
            
            x_left, y_left = omb_esq.x, omb_esq.y
            x_right, y_right = omb_dir.x, omb_dir.y
            
            theta = math.atan2(y_right - y_left, x_right - x_left)
            
            if larg_omb > larg_quad * 1.2:
                return "frente ou costas", 0.5, theta 
            elif larg_omb < larg_quad * 0.8:
                return "de lado", 0.0, theta
        
        return "Falha na deteccao", 0.0, theta
        

def main(args=None):
    rclpy.init(args=args)
    node = Navigation_robot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
