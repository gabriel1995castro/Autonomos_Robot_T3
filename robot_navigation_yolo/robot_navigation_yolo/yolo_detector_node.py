import rclpy
import torch
import cv2
import json
import time
import numpy as np
from ultralytics import YOLO
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
import math
from geometry_msgs.msg import Pose, Quaternion,PoseWithCovarianceStamped


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_navigation_node')
        
        # Parameters
        self.declare_parameter('confidence_threshold', 0.25)
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
        # Subscribers and publishers
        self.subscription = self.create_subscription( Image, '/a200_0000/sensors/camera_0/color/image', self.listener_callback, 10)
        self.subscription2 = self.create_subscription(Image,'/a200_0000/sensors/camera_0/depth/image',self.depth_callback,10)
        #self.subscription_odom = self.create_subscription(Odometry,'/a200_0000/platform/odom/filtered',self.odom_callback,10)
        self.amcl_subscriber = self.create_subscription(PoseWithCovarianceStamped,'/a200_0000/amcl_pose',self.amcl_callback,10)
        self.publisher = self.create_publisher(Image, 'robot_vision_yolo', 10)
        self.publish_person_localization = self.create_publisher(String, 'person_location', 10)
        self.debug_publisher = self.create_publisher(Image, 'debug_image', 10)
        self.object_center = None

        
        # Parâmetros da câmera Intel RealSense D435
        self.fx = 277.0  
        self.fy = 277.0  
        self.cx = 160.0  
        self.cy = 120.0
        self.person_detected = False  
        

        # CV Bridge
        self.bridge = CvBridge()

        # Load YOLO model
        self.get_logger().info("Loading YOLOv5 model...")
        try:
            self.model = YOLO("yolov8n.pt")
            self.model.conf = self.confidence_threshold
            self.model.classes = [0]
            self.get_logger().info("YOLO model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            
        # Performance tracking
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
   
    """def odom_callback(self, msg: Odometry):
        
        self.robot_pose = msg.pose.pose
        self.get_logger().info(f"Received robot pose: {self.robot_pose}")"""
    robot_pose_received = None
    def amcl_callback(self, msg):
        """ Callback que recebe a pose do robô no mundo """
        self.get_logger().info("AMCL callback triggered.")
        self.robot_pose = msg.pose.pose  # Armazena a posição e orientação do robô
        self.get_logger().info(f"Posição do robô: x={self.robot_pose.position.x:.2f}, y={self.robot_pose.position.y:.2f}")
        self.robot_pose_received = True

    def listener_callback(self, msg):
        if not self.robot_pose_received:
            self.get_logger().info("AMCL Pose not received, skipping camera processing.")
            return
        try:
            # Convert ROS Image to OpenCV format
            self.get_logger().debug("Converting image message to OpenCV format")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)

            height, width = cv_image.shape[:2]
            self.get_logger().debug(f"Image dimensions: {width}x{height}")
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 3 == 0:
                current_time = time.time()
                self.fps = 10 / (current_time - self.last_time)
                self.last_time = current_time
                self.get_logger().info(f"Processing at {self.fps:.2f} FPS")
            
            # Perform inference using YOLOv8
            self.get_logger().debug("Running YOLO inference")
            results = self.model.predict(cv_image, conf=self.confidence_threshold)

            try:
                # Process person detections
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
                        confidence = float(box.conf[0])  # Confiança da detecção
                        class_id = int(box.cls[0])  # Classe detectada

                        if class_id == 0 and confidence > self.confidence_threshold:  # Classe 0 = "person"
                            self.get_logger().info(f"Person detected at {x1}, {y1}, {x2}, {y2} with confidence {confidence:.2f}")

                            # Desenhar bounding box na imagem
                            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Adicionar texto com a confiança
                            label = f"Person: {confidence:.2f}"
                            cv2.putText(cv_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Atualizar a posição central da pessoa detectada
                            self.object_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            except Exception as e:
                self.get_logger().error(f"Error processing detections: {e}")
            
            # Publish processed image
            processed_img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publisher.publish(processed_img_msg)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")


    def depth_callback(self, msg):
        
        if not self.robot_pose_received:
            self.get_logger().info("AMCL Pose not received, skipping camera processing.")
            return
        person_locations = []
        self.get_logger().info("Depth callback chamado!")
        
        if self.person_detected:
             return 

        # Verifique se o objeto foi detectado
        if self.object_center is None:
            self.get_logger().info("No object center available.")
            return

        try:
            # Converter imagem de profundidade para OpenCV
            depth_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            x, y = self.object_center

            # Obter a profundidade do objeto detectado (em mm, convertendo para metros)
            depth = depth_image[y, x] / 1000.0
            self.get_logger().info(f"Depth at object center: {depth} meters")

            # Posição do objeto no sistema de coordenadas da câmera
            x_camera = (x - self.cx) * depth / self.fx
            y_camera = (y - self.cy) * depth / self.fy
            z_camera = depth  # A profundidade é a coordenada Z da câmera

            # Supondo que você tenha a pose do robô (posição e orientação)
            # Exemplo fictício de pose do robô (em um cenário real, você obteria isso de um tópico)
            robot_position = self.robot_pose.position
            robot_orientation = self.robot_pose.orientation


            # Converta a orientação do robô (quaternion) para ângulos de Euler
            qw, qx, qy, qz = robot_orientation.w, robot_orientation.x, robot_orientation.y, robot_orientation.z
            roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx ** 2 + qy ** 2))
            pitch = math.asin(2 * (qw * qy - qz * qx))
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))

            # Aplicar rotação à posição do objeto (transformar do sistema da câmera para o sistema do robô)
            x_robot = robot_position.x + x_camera * math.cos(yaw) - y_camera * math.sin(yaw)
            y_robot = robot_position.y + x_camera * math.sin(yaw) + y_camera * math.cos(yaw)
            z_robot = robot_position.z + z_camera  # A profundidade não precisa de rotação, apenas translação

            # Adicionar as posições transformadas à lista de localizações
            person_locations.append((x_robot, y_robot, z_robot))

            self.get_logger().info(f"Person detected at positions: {person_locations}")

            dx = x_robot - robot_position.x
            dy = x_robot - robot_position.y
            yaw = math.atan2(dy, dx)

            person_locations.append((dx, dy, yaw))

            # Publicar a localização da pessoa detectada
            if person_locations:
                coordinates_msg = String()
                coordinates_msg.data = json.dumps(person_locations)
                self.publish_person_localization.publish(coordinates_msg)
                self.get_logger().info(f"Person detected at positions: {person_locations}")
                self.person_detected = True
            else:
                self.get_logger().debug("No person detected in this frame")

        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")



def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()