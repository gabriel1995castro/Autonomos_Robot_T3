# Implementation of a Mobile Robot for Human Search.

The objective of this project is to develop a navigation system for an autonomous robot capable of performing the following tasks:

- **Exploring** a partially unknown environment in a fully autonomous manner.
- **Locating** a person randomly positioned in the environment.
- **Positioning itself** next to the person found.
- Upon completing the task, generating a file containing the **approximate location** of the person and other relevant mission data..
  
## Tools

- **Programming Language:** Python (ROS 2
- **Simulation:** Ignition Gazebo
- **Robot Model:** Clearpath Husky A200
- **YOLO Version:**  Yolov8n

## System Requirements

- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Gazebo 11.10.2
- Python 3.10.12

## Navigation and exploration strategies.

During the development of the work, the robot's movement procedure in the environment was divided into two distinct stages: exploration and navigation.

1.**Exploration:** phase in which the robot moves through the environment autonomously in search of a person.

2.**Navigation:** begins as soon as the person is found, at which point the robot defines a final objective and adjusts its path to complete the task.

### Exploration strategy:

The exploration algorithm is designed to perform **autonomous exploration** in unknown environments. It uses an **occupancy map** to identify unexplored boundaries and determine the best navigation points for map expansion.

**Map Reception:** The node listens to map data and internally updates information about the occupancy of the environment.

**Robot Position Reception:** The robot's pose is recorded to determine the current location and calculate the destination points.

**Boundary Identification:** The map is analyzed to detect cells that border unknown regions.

**Boundary Clustering:** A clustering algorithm separates the boundaries into coherent sets.

**Best Boundary Selection:** Criteria such as distance, proximity to obstacles, and boundary size are used to select the best destination.

**Navigation Goal Sending:** The chosen point is sent as a navigation goal to the robot.

**Monitoring and Cancellation:** The execution is monitored and, if a person is detected, the exploration is stopped.

| Exploration process.|
|--------------------------|
| ![](https://github.com/gabriel1995castro/Autonomos_Robot_T3/blob/441654e2a2e7af0622b6991d619cc46ba892d300/Images/Screenshot%20from%202025-03-24%2012-36-21.png) |


### Navigation strategy:

The navigation method begins with the detection of a person by the node responsible for image processing.

The algorithm receives, through the **person_location** topic, the coordinates of the person in the scene and the image captured by the robot's camera, accessible through the **/a200_0000/sensors/camera_0/color/image topic**.

The code uses **OpenCV** and **MediaPipe** to analyze the image and determine the person's orientation (front, back or side) at the time of detection.

The robot's **orientation** is adjusted based on the positions of the person's shoulders and hips, allowing the ideal angle for positioning to be calculated.

1. If the person is facing the front or back, the robot positions itself sideways to them.
2. If the person is facing the side, the robot slightly adjusts its trajectory to ensure proper alignment.

Navigation to the final position is performed with **Nav2**, taking into account the orientation and displacement calculations within the detection node.

### People detection process

For the people detection process, YOLOv8n was used, a model developed by Ultralytics, optimized to offer high speed and low computational demand. It is widely used in mobile devices and systems with limited resources, ensuring good accuracy and efficiency in real-time object detection.

| Yolov8n image test 1 | Yolov8n image test 2|
|--------------------|--------------------------------|
| ![](https://github.com/gabriel1995castro/Autonomos_Robot_T3/blob/441654e2a2e7af0622b6991d619cc46ba892d300/Images/output_image2.jpg) | ![](https://github.com/gabriel1995castro/Autonomos_Robot_T3/blob/91a828dee3f655615f1e71e98ca7f5be80c2f922/Images/output_image.jpg)|

The node responsible for person detection receives data from the robot's RGB camera and depth camera. To avoid false positive detections, the code uses a **confidence threshold** (confidence_threshold), which eliminates all detections with less than 60% certainty.

The **Intel RealSense D435** camera parameters are initialized with the values ​​fx, fy, cx, and cy, which are used in the depth calculation. The YOLO model is loaded with detection focused only on the "person" (class 0).

When a person is detected, the algorithm **draws a bounding box** around the person and stores the **center** of the object to be used in the **depth calculation**.

| Person detection in simulated environment|
|--------------------------|
| ![](https://github.com/gabriel1995castro/Autonomos_Robot_T3/blob/441654e2a2e7af0622b6991d619cc46ba892d300/Images/image.png) |


The 3D position of the person in the environment is calculated in the camera coordinate system and **transformed** to the robot coordinate system using the robot's pose at the time of detection.

# Calculating coordinates in the camera coordinate system:

The first conversion process involves using the depth image and the pixel coordinates of the detected object to calculate the object coordinates. The formulas for the **X_camera​** and **Y_camera**​ coordinates are as follows:

$$
X_{camera} = \frac{(x_{obj} - c_x) \cdot d}{f_x}
$$

$$
Y_{camera} = \frac{(y_{obj} - c_y) \cdot d}{f_y}
$$

where:

- $x_{obj​}/y_{obj}$ are the X and Y coordinates of the pixel where the object was detected.
- $c_x/c_y$ are the central points of the image in relation to the X (horizontal focal point) and Y (vertical focal point) axes.
- $d$ is the depth of the object (obtained from the depth image, in meters).
- $f_x/f_y$ are the focal distances of the camera (horizontal focal point and vertical focal point).

The Z coordinate is simply the depth:

$$
Z_{camera} = d
$$

# Transformation to the robot coordinate system:

The transformation of the camera coordinates to the robot coordinate system is done using the robot pose, which includes its position (x,y,z) and its orientation (quaternion). The calculation for the coordinates in the robot system is given by:

$$
x_{robot} = x_{robot position} + x_{camera} \cdot \cos(yaw) - y_{camera} \cdot \sin(yaw)
$$

$$
y_{robot} = y_{robot position} + x_{camera} \cdot \sin(yaw) + y_{camera} \cdot \cos(yaw)
$$

$$
z_{robot} = z_{robot position} + z_{camera}
$$

Where:

- $x_{robot position},y_{robot position},z_{robot position}$: are the coordinates of the robot in global space (obtained from the robot's location).
- $x_{camera},y_{camera},z_{camera​}$: are the calculated coordinates for the object in the camera coordinate system.
- $yaw$: is the rotation of the robot around the vertical axis, obtained from the orientation quaternion.

# Calculating the direction of the robot to the object:

The direction from the robot to the object is defined by the difference between the x and y coordinates of the robot and the object. 
The calculation is given by:

$$
dx=x_{person} − x_{robot position}
​$$

$$
dy=y_{person} − y_{robot position}
$$

$$
yaw=atan2(dy,dx)
$$

These formulas determine the **angle (yaw)** the robot needs to follow to reach the object, based on the differences in position in the **XY** plane.
The person's position is published in the **person_locate** topic, in addition to being saved in a JSON file for operation reporting, along with the task execution time.

The **final adjustment** to the robot's positioning is made using MediaPipe to track the person's body pose in the image. To do this, the distances between the person's shoulders and hips are analyzed. The larg_omb variable calculates the width between the shoulders, which is the difference between the X coordinates of the left and right shoulders, while larg_quad calculates the width between the hips, which is the difference between the X coordinates of the left and right hips.

| Person pose estimation using MediaPipe.|
|--------------------------|
| ![](https://github.com/gabriel1995castro/Autonomos_Robot_T3/blob/ac7243f505398ff5935a933391e9f63407c3716f/Images/imagem_com_pose.jpg)|


**If larg_omb > larg_quad * 1.2** - this indicates that the person is in a more forward or back position, as the width of the shoulders is significantly greater than that of the hips.

**If shoulder_width < hip_width * 0.8** - this suggests that the person is lying sideways, as the width of the shoulders is much smaller than the width of the hips.

These calculations help determine the person's orientation, precisely adjusting the robot's behavior relative to its position in the environment.

### Cloning repository 

```bash 
https://github.com/gabriel1995castro/Autonomos_Robot_T3.git
cd Autonomous_Robots_T3
```

### Building the package

```bash 
colcon build --packages-select robot_navigation_yolo
source install/setup.bash
```
To use the developed solution, the Clearpath simulator must have been correctly installed, as well as the sensors package must have been obtained for possible troubleshooting.

```bash
https://docs.clearpathrobotics.com/docs/ros/tutorials/simulator/install
```
# Executing the proposed solution:
The robot can be positioned in a random position using the command:

 ```bash
ros2 launch clearpath_gz simulation.launch.py x:=1.5 y:=2.7 yaw:=1.5707
```

Replace the values ​​of x,y,yaw to the desired values.

To visualize the development of the exploration process in rviz:

 ```bash
ros2 launch clearpath_viz view_navigation.launch.py namespace:=a200_0000
```

Launch nav2 demos to run the navigation process:

 ```bash
ros2 launch clearpath_nav2_demos nav2.launch.py setup_path:=$HOME/clearpath/ use_sim_time:=true
```

Initialize the exploration node:

 ```bash
ros2 run robot_navigation_yolo fontrier_based_exploratin 
```
Initialize the navigation node:

 ```bash
ros2 run robot_navigation_yolo navigation_node
```
Initialize the node for people detection using:

 ```bash
ros2 run robot_navigation_yolo yolo_detector_node
```
