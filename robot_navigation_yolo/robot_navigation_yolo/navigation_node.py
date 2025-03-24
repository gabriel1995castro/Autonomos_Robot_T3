import rclpy
import json
import math
import tf_transformations
import mediapipe as mp
import cv2
from sensor_msgs.msg import Image
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose  
from rclpy.action import ActionClient
from cv_bridge import CvBridge

class NavigationRobot(Node):
    
    def __init__(self):
        super().__init__('robot_nav')
        
        # ActionClient para o Nav2
        self._client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.stop_exploration_publisher = self.create_publisher(Bool, 'stop_exploration', 10)
        self.person_detector = self.create_subscription(String, 'person_location', self.detector_callback, 10)
        self.image_subscriber = self.create_subscription(Image, '/a200_0000/sensors/camera_0/color/image', self.image_callback, 10)
        
        self.bridge = CvBridge()
        self.result_detector = False
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.current_image = None  

    def detector_callback(self, msg):

        if not self.result_detector:
            self.result_detector = True
            self.get_logger().info(f"Pessoa encontrada em: {msg.data}")
            
    
            stop_msg = Bool()
            stop_msg.data = True
            self.stop_exploration_publisher.publish(stop_msg)
            
            try:
               
                pose = json.loads(msg.data)
                self.get_logger().info(f"msg.data: {msg.data}")
                if len(pose) == 1 and len(pose[0]) == 3:
                    x, y, z = pose[0]
                
                
                map_x = x   
                map_y = y  
                
                goal_msg = NavigateToPose.Goal()
                
    
                if self.current_image is not None:
                    person_status, confidence, theta = self.orientation_callback(self.current_image)
                    self.get_logger().info(f"Pessoa está: {person_status}, com confiança: {confidence:.2f}, Ângulo: {theta:.2f}")
                    
                    
                    if person_status == "frente ou costas":
                        goal_msg.pose.pose.position.x = map_x + 0.5 * math.cos(theta)  
                        goal_msg.pose.pose.position.y = map_y + 0.5 * math.sin(theta)  
                    elif person_status == "de lado":

                        goal_msg.pose.pose.position.x = map_x + 0.5
                        goal_msg.pose.pose.position.y = map_y

                    goal_msg.pose.pose.position.z = z  
                    
                    
                    quat = tf_transformations.quaternion_from_euler(0, 0, theta)
                    goal_msg.pose.pose.orientation.x = quat[0]
                    goal_msg.pose.pose.orientation.y = quat[1]
                    goal_msg.pose.pose.orientation.z = quat[2]
                    goal_msg.pose.pose.orientation.w = quat[3]
                    
            
                    self._client.wait_for_server()
                    self._client.send_goal_async(goal_msg)
                else:
                    self.get_logger().warning("Imagem não recebida ainda. Não pode processar orientação.")
                
            except Exception as e:
                self.get_logger().error(f"Erro ao processar posição da pessoa: {e}")

    def orientation_callback(self, msg):
      
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        rgb_img  =cv2.rotate(rgb_img , cv2.ROTATE_180)
        result = self.pose.process(rgb_img)
        
        theta = 0.0  
        
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

    def image_callback(self, msg):
        
        self.current_image = msg
        self.get_logger().info("Imagem recebida e atualizada.")

def main(args=None):
    rclpy.init(args=args)
    node = NavigationRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
