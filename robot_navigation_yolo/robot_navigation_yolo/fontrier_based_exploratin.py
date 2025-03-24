#!/usr/bin/env python3

import rclpy
import numpy as np
import time
import math
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String, Bool
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from scipy.ndimage import label
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")
        
        map_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        
        # Subscribe nos tópicos do robo.
        self.map_sub = self.create_subscription(OccupancyGrid, "/a200_0000/map", self.map_callback, qos_profile=map_qos)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped,  "/a200_0000/pose",self.pose_callback,10)
        self.person_sub = self.create_subscription(Bool, "stop_exploration",self.person_callback,10)
        self.move_base_client = ActionClient(self, NavigateToPose, '/a200_0000/navigate_to_pose')
        
        #Dados do mapa
        self.map_data = None
        self.resolution = None
        self.origin = None
        self.width = 0
        self.height = 0

        #Flag de critério de parada para detecção de pessoas
        self.person_detected = False
        
        #Variáveis para armazenamento da pose do robô
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_position_initialized = False
        
        #Parâmetros utilizados no procedimento de exploração
        self.max_open_area = 60
        self.min_frontier_size = 10
        self.buffer_distance = 1 
        self.lidar_max_range = 1.0
        self.frontier_distance_threshold = 2.0
        self.goal_timeout = 60.0
        
        #Estado da navegação.
        self.visited_frontiers = set()
        self.discarded_frontiers = set()
        self.exploring = False
        self.current_frontier = None
        self.goal_start_time = None
        self._send_goal_future = None
        self._get_result_future = None
        self.last_position = None  
        
        #Timers utilizados.
        self.exploration_timer = self.create_timer(2.0, self.exploration_cycle)
        self.timeout_timer = None
        
        self.get_logger().info("Frontier Explorer initialized")

   #Avalia o critério de parada para encerrar a exploração. 
    def person_callback(self, msg):
        self.person_detected = msg.data
        if self.person_detected:
            self.get_logger().info("Pessoa detectada! Pausando exploração.")
            self.cancel_navigation_goal()
   
   #Obtem os dados do mapa durante o processo de exploração.
    def map_callback(self, msg):
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.resolution = msg.info.resolution
        self.origin = msg.info.origin.position
        self.width = msg.info.width
        self.height = msg.info.height
        
        self.get_logger().debug(f"Map updated: {self.width}x{self.height}, resolution: {self.resolution}")

    #Obtenção da pose do robô.
    def pose_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.robot_position_initialized = True
 

    def exploration_cycle(self):
        if self.map_data is None:
            self.get_logger().warn("No map data received yet")
            return
            
        if not self.robot_position_initialized:
            self.get_logger().warn("No pose data received yet")
            return
            
        if self.exploring:
            self.get_logger().debug("Already exploring, waiting for goal completion")
            return
            
        self.exploring = True
        self.explore_once()

   #Cancela imediatamente o objeto atual, se uma pessoa for encontrada.
    def cancel_navigation_goal(self):
        if self._send_goal_future and not self._send_goal_future.done():
            self.get_logger().info("Cancelando objetivo atual de navegação...")
            self.move_base_client.cancel_goal_async()
   
    def explore_once(self):
        
        if self.person_detected:
            self.get_logger().warn("Exploração pausada devido à detecção de pessoa.")
            self.exploring = False
            return
        
        self.get_logger().info("Starting exploration cycle...")
        
        robot_pos = self.get_robot_position()
        if robot_pos is None:
            self.get_logger().warn("Cannot determine robot position. Waiting...")
            self.exploring = False
            return
            
        frontiers = self.find_frontiers()
        self.get_logger().info(f"Found {len(frontiers)} frontier points")
        
        if not frontiers:
            self.get_logger().info("No frontiers found. Exploration may be complete.")
            self.exploring = False
            return
            
        clusters = self.group_frontiers(frontiers)
        self.get_logger().info(f"Grouped into {len(clusters)} clusters")
        
        best_frontier = self.select_best_frontier(clusters, robot_pos)
        
        if best_frontier is None:
            self.get_logger().info("No valid frontiers found. Waiting for map updates.")
            self.exploring = False
            return
        
        self.send_goal(best_frontier)

    #Encontra as fronteiras durante o processo de exploração.
    def find_frontiers(self):
        frontiers = []
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.map_data[y, x] == 0:
                    neighbors = self.map_data[y - 1:y + 2, x - 1:x + 2].flatten()
                    if -1 in neighbors:
                        world_x = self.origin.x + x * self.resolution
                        world_y = self.origin.y + y * self.resolution
                        frontiers.append((x, y, world_x, world_y))
        return frontiers

    def group_frontiers(self, frontiers):
        frontier_map = np.zeros_like(self.map_data)
        for x, y, _, _ in frontiers:
            frontier_map[y, x] = 1
        labeled_array, num_features = label(frontier_map)
        clusters = {}
        for x, y, wx, wy in frontiers:
            label_id = labeled_array[y, x]
            if label_id in clusters:
                clusters[label_id].append((x, y, wx, wy))
            else:
                clusters[label_id] = [(x, y, wx, wy)]

        return {k: v for k, v in clusters.items() if len(v) >= self.min_frontier_size}

    #Busca a melhor fronteira no momento atual.
    def select_best_frontier(self, clusters, robot_pos):
        if not clusters:
            return None

        robot_x, robot_y = robot_pos
        
        best_frontier = None
        min_cost = float("inf")
        
        for cluster_id, frontier_cells in clusters.items():
            if len(frontier_cells) > self.max_open_area:
                self.get_logger().debug(f"Discarding cluster {cluster_id} (too large: {len(frontier_cells)} cells)")
                continue

        
            centroid_x = sum(wx for _, _, wx, _ in frontier_cells) / len(frontier_cells)
            centroid_y = sum(wy for _, _, _, wy in frontier_cells) / len(frontier_cells)
        
            distance = math.sqrt((centroid_x - robot_x)**2 + (centroid_y - robot_y)**2)
            
 
            proximity_cost = 1 / (distance + 1e-3)  
            
           
            size_factor = 1.0 / (1.0 + len(frontier_cells) / 50.0)
            
           
            obstacle_penalty = self.calculate_obstacle_proximity(centroid_x, centroid_y)
            
            
            cost = distance * size_factor * obstacle_penalty * proximity_cost  
            
            if cost < min_cost and self.is_safe_to_move(centroid_x, centroid_y):
                min_cost = cost
                best_frontier = (centroid_x, centroid_y)
        
        if best_frontier:
            self.get_logger().info(f"Selected frontier: ({best_frontier[0]:.2f}, {best_frontier[1]:.2f}), cost: {min_cost:.2f}")
        
        return best_frontier


    def calculate_obstacle_proximity(self, world_x, world_y):
       
        grid_x = int((world_x - self.origin.x) / self.resolution)
        grid_y = int((world_y - self.origin.y) / self.resolution)
        check_distance = self.buffer_distance * 2
        
        obstacle_count = 0
        total_cells = 0
        
        for dy in range(-check_distance, check_distance + 1):
            for dx in range(-check_distance, check_distance + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    total_cells += 1
                    if self.map_data[check_y, check_x] == 100: 
                        dist = math.sqrt(dx**2 + dy**2)
                        obstacle_count += (check_distance - dist) / check_distance
        
 
        if total_cells > 0:
            proximity_factor = 1.0 + (obstacle_count / total_cells) * 3.0
            return proximity_factor
        return 1.0

    def is_near_visited_frontier(self, x, y, threshold=None):
        if threshold is None:
            threshold = self.frontier_distance_threshold
            
        for vx, vy in self.visited_frontiers:
            if math.sqrt((x - vx)**2 + (y - vy)**2) < threshold:
                return True
                
        for vx, vy in self.discarded_frontiers:
            if math.sqrt((x - vx)**2 + (y - vy)**2) < threshold:
                return True
                
        return False

    def is_safe_to_move(self, target_x, target_y):
       
        grid_x = int((target_x - self.origin.x) / self.resolution)
        grid_y = int((target_y - self.origin.y) / self.resolution)
        
        if not (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            self.get_logger().warn(f"Target point ({target_x:.2f}, {target_y:.2f}) is outside map bounds")
            return False
  
        if self.map_data[grid_y, grid_x] != 0:  # Must be free space
            self.get_logger().debug(f"Target point is not free space: value={self.map_data[grid_y, grid_x]}")
            return False
        

        for dy in range(-self.buffer_distance, self.buffer_distance + 1):
            for dx in range(-self.buffer_distance, self.buffer_distance + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                if 0 <= check_x < self.width and 0 <= check_y < self.height:
                    if self.map_data[check_y, check_x] == 100:  # Obstacle
                        self.get_logger().debug(f"Obstacle near target at ({check_x}, {check_y})")
                        return False
        
        return True

    def send_goal(self, frontier):
        if not frontier:
            self.exploring = False
            return
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = frontier[0]
        goal_msg.pose.pose.position.y = frontier[1]
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f"Sending goal to ({frontier[0]:.2f}, {frontier[1]:.2f})...")
        
   
        if not self.move_base_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Navigation action server not available")
            self.exploring = False
            return
        
      
        self.current_frontier = frontier
        self.goal_start_time = time.time()
        self.last_position = self.get_robot_position() 
        self._send_goal_future = self.move_base_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)
        self.timeout_timer = self.create_timer(1.0, self.check_goal_timeout)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal rejected by navigation server')
            if self.current_frontier:
                self.add_to_discarded_frontiers(self.current_frontier)
            self.reset_exploration()
            return

        self.get_logger().info('Goal accepted')
        
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        status = future.result().status
        if status == 4:  
            self.get_logger().info('Goal succeeded!')
            if self.current_frontier:
                self.visited_frontiers.add(self.current_frontier)
        else:
            self.get_logger().info(f'Goal failed with status: {status}')
            if self.current_frontier:
                self.add_to_discarded_frontiers(self.current_frontier)
        
        self.reset_exploration()

    def add_to_discarded_frontiers(self, frontier):

        x, y = frontier
        self.discarded_frontiers.add((x, y))

        for dx in [-0.5, 0, 0.5]:
            for dy in [-0.5, 0, 0.5]:
                self.discarded_frontiers.add((x+dx, y+dy))

    def check_goal_timeout(self):
        if not self.goal_start_time:
            return
            
        elapsed = time.time() - self.goal_start_time
        
    
        if elapsed > self.goal_timeout * 0.5:  
            current_pos = self.get_robot_position()
            if self.last_position and current_pos:
              
                dx = current_pos[0] - self.last_position[0]
                dy = current_pos[1] - self.last_position[1]
                dist_moved = math.sqrt(dx*dx + dy*dy)
                
               
                if dist_moved < 0.05:  
                    self.get_logger().warn("Robot appears stuck, canceling goal early")
                    if self._send_goal_future and self._send_goal_future.done():
                        goal_handle = self._send_goal_future.result()
                        if goal_handle.accepted:
                            self.get_logger().info("Canceling current goal due to lack of progress")
                            goal_handle.cancel_goal_async()
                    
                    if self.current_frontier:
                        self.add_to_discarded_frontiers(self.current_frontier)
                    
                    self.reset_exploration()
                    return
        

        self.last_position = self.get_robot_position()
        
 
        if elapsed > self.goal_timeout:
            self.get_logger().warn(f"Goal timed out after {elapsed:.1f} seconds")
            
    
            if self._send_goal_future and self._send_goal_future.done():
                goal_handle = self._send_goal_future.result()
                if goal_handle.accepted:
                    self.get_logger().info("Canceling current goal")
                    goal_handle.cancel_goal_async()
            
    
            if self.current_frontier:
                self.add_to_discarded_frontiers(self.current_frontier)
            
            self.reset_exploration()

    def reset_exploration(self):
       
        if self.timeout_timer:
            self.destroy_timer(self.timeout_timer)
            self.timeout_timer = None
        
       
        self.goal_start_time = None
        self.current_frontier = None
        self.exploring = False
        self.last_position = None
        
      
        self.get_logger().info("Exploration cycle complete. Waiting for next cycle.")

    def get_robot_position(self):

        if not self.robot_position_initialized:
            self.get_logger().warn("Robot position not yet available from pose topic")
            return None
            
        return (self.robot_x, self.robot_y)

def main(args=None):
    rclpy.init(args=args)
    explorer = FrontierExplorer()
    
    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        pass
    finally:
        explorer.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()