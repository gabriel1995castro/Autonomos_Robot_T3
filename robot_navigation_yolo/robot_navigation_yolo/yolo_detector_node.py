import rclpy
import torch
import cv2
import json
import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_navigation_node')
        
        # Parameters
        self.declare_parameter('confidence_threshold', 0.25)
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        
        # Subscribers and publishers
        self.subscription = self.create_subscription(
            Image, '/a200_0000/sensors/camera_0/color/image', self.listener_callback, 10
        )
        self.publisher = self.create_publisher(Image, 'robot_vision_yolo', 10)
        self.publish_person_localization = self.create_publisher(String, 'person_location', 10)
        self.debug_publisher = self.create_publisher(Image, 'debug_image', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLO model
        self.get_logger().info("Loading YOLOv5 model...")
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
            # Set confidence threshold
            self.model.conf = self.confidence_threshold
            # Set to only detect persons (class 0)
            self.model.classes = [0]
            self.get_logger().info("YOLO model loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            
        # Performance tracking
        self.frame_count = 0
        self.last_time = time.time()
        self.fps = 0
        
    def listener_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            self.get_logger().debug("Converting image message to OpenCV format")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
            # Create a debug copy
            debug_image = cv_image.copy()
            
            # Resize if needed (can help with performance)
            height, width = cv_image.shape[:2]
            self.get_logger().debug(f"Image dimensions: {width}x{height}")
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                current_time = time.time()
                self.fps = 10 / (current_time - self.last_time)
                self.last_time = current_time
                self.get_logger().info(f"Processing at {self.fps:.2f} FPS")
            
            # Perform inference
            self.get_logger().debug("Running YOLO inference")
            results = self.model(cv_image)
            
            # Access results
            person_locations = []
            try:
                # Get detections
                detections = results.pandas().xyxy[0]
                self.get_logger().debug(f"Detection results: {len(detections)} objects found")
                
                # Display classes found for debugging
                classes_found = detections['class'].unique().tolist()
                if classes_found:
                    self.get_logger().info(f"Classes detected: {classes_found}")
                
                # Process person detections
                person_detections = detections[detections['class'] == 0]
                self.get_logger().info(f"Persons detected: {len(person_detections)}")
                
                for _, row in person_detections.iterrows():
                    confidence = float(row['confidence'])
                    x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                    person_locations.append((x1, y1, x2, y2, confidence))
                    
                    # Draw bounding box
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add confidence text
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(cv_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add FPS info to image
                #cv2.putText(cv_image, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            except Exception as e:
                self.get_logger().error(f"Error processing detections: {e}")
            
            # Publish processed image
            processed_img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.publisher.publish(processed_img_msg)
            
            # Publish person locations if detected
            if person_locations:
                coordinates_msg = String()
                coordinates_msg.data = json.dumps(person_locations)
                self.publish_person_localization.publish(coordinates_msg)
                self.get_logger().info(f"Person detected at positions: {person_locations}")
            else:
                self.get_logger().debug("No persons detected in this frame")
                
            # Create and publish debug image
            debug_image = results.render()[0]  # Get YOLO's visualization
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
            self.debug_publisher.publish(debug_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()