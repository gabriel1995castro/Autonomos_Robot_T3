import rclpy
import torch
import  cv2
import json 
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

class YoloDetector (Node):
    def __init__(self):
        super().__init__('yolo_navigation_node')
        self.subscription = self.create_subscription (Image, 'camera/image_raw', self.listener_callback, 10)
        self.publisher = self.create_publisher (Image, 'robot_vision_yolo', 10)
        self.publish_person_localization = self.create_publisher(String, 'person_location', 10)
        self.bridge = CvBridge()
        self.model = torch.hub.load ('ultralytics/yolov5','yolov5n')
        
    def listener_callback (self,msg):
        cv_image = self.bridge.imgmsg_to_cv2 (msg, desired_encoding= 'bgr8')
        results =self. model (cv_image)
        person_locations = []
        
        for result in results:
            for box, classe in zip (result.boxes.xyxy, result.boxes.classe):
                if int (classe == 0):
                    x1, y1, x2, y2 = map(int, box)
                    person_locations.append((x1, y1, x2, y2))
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
            
        processed_img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.publisher.publish(processed_img_msg)
        
        if person_locations:
            coordenates_msg = String()
            coordenates_msg.data = json.dumps(person_locations) 
            self.publish_person_localization.publish(coordenates_msg)
            self.get_logger().info(f"Pessoa detectada nas posições: {person_locations}")

def main(args=None):
    
    rclpy.init(args=args)
    node = YoloDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()