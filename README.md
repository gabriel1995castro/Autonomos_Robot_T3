# T3: Implementation of a Mobile Robot for Human Search.

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

2. **Navigation:** begins as soon as the person is found, at which point the robot defines a final objective and adjusts its path to complete the task.

### Exploration strategy:

Inserir o texto do algoritmo de exploração.

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


The node responsible for person detection receives data from the robot's RGB camera and depth camera. To avoid false positive detections, the code uses a **confidence threshold** (confidence_threshold), which eliminates all detections with less than 60% certainty.

The **Intel RealSense D435** camera parameters are initialized with the values ​​fx, fy, cx, and cy, which are used in the depth calculation. The YOLO model is loaded with detection focused only on the "person" (class 0).

When a person is detected, the algorithm **draws a bounding box** around the person and stores the **center** of the object to be used in the **depth calculation**.

The 3D position of the person in the environment is calculated in the camera coordinate system and **transformed** to the robot coordinate system using the robot's pose at the time of detection.

# Calculating coordinates in the camera coordinate system:

The first conversion process involves using the depth image and the pixel coordinates of the detected object to calculate the object coordinates. The formulas for the **X_camera​** and **Y_camera**​ coordinates are as follows:

$$
X_camera = ((x_obj_ - c_x) * d) / f_x
$$

$$
Y_camera = ((y_obj_ - c_y) * d) / f_y
$$

where:

- $x_obj​/y_obj$ are the X and Y coordinates of the pixel where the object was detected.
- $c_x/c_y$ are the central points of the image in relation to the X (horizontal focal point) and Y (vertical focal point) axes.
- $d$ is the depth of the object (obtained from the depth image, in meters).
- $f_x/f_y$ are the focal distances of the camera (horizontal focal point and vertical focal point).

The Z coordinate is simply the depth:

$Z_camera$= d

# Transformation to the robot coordinate system:

The transformation of the camera coordinates to the robot coordinate system is done using the robot pose, which includes its position (x,y,z) and its orientation (quaternion). The calculation for the coordinates in the robot system is given by:

$$
x_robot​= x_robot position​ + x_camera​⋅cos(yaw) − y_camera​⋅sin(yaw)
y_robot= y_robot position + x_camera⋅sin⁡(yaw) + y_camera⋅cos⁡(yaw)
y_robot​= y_robot position​ + x_camera​⋅sin(yaw) + y_camera​⋅cos(yaw)
z_robot= z_robot position + z_camera
z_robot​= z_robot position​ + z_camera​
$$

Where:

- $x_robot position,y_robot position,z_robot position​$: are the coordinates of the robot in global space (obtained from the robot's location).
- $x_camera,y_camera,z_camera​$: are the calculated coordinates for the object in the camera coordinate system.
- $yaw$: is the rotation of the robot around the vertical axis, obtained from the orientation quaternion.

# Calculating the direction of the robot to the object:

The direction from the robot to the object is defined by the difference between the x and y coordinates of the robot and the object. 
The calculation is given by:
$$
dx=x_person − x_robot position​
dy=y_person − y_robot position
yaw=atan2(dy,dx)
$$

These formulas determine the **angle (yaw)** the robot needs to follow to reach the object, based on the differences in position in the **XY** plane.
The person's position is published in the **person_locate** topic, in addition to being saved in a JSON file for operation reporting, along with the task execution time.

The **final adjustment** to the robot's positioning is made using MediaPipe to track the person's body pose in the image. To do this, the distances between the person's shoulders and hips are analyzed. The larg_omb variable calculates the width between the shoulders, which is the difference between the X coordinates of the left and right shoulders, while larg_quad calculates the width between the hips, which is the difference between the X coordinates of the left and right hips.

$$
**If larg_omb > larg_quad * 1.2** - this indicates that the person is in a more forward or back position, as the width of the shoulders is significantly greater than that of the hips.

**If shoulder_width < hip_width * 0.8** - this suggests that the person is lying sideways, as the width of the shoulders is much smaller than the width of the hips.
$$

These calculations help determine the person's orientation, precisely adjusting the robot's behavior relative to its position in the environment.

### Cloning repository 

```bash 
git clone https://github.com/gabriel1995castro/autonomous_robots.git
cd autonomous_robots
```

### Building the package

```bash 
colcon build --packages-select robot_controller
source install/setup.bash
```

To start the navigation and obstacle detection system, both programs must be called via a launch file. This file ensures that the two modules run together and are correctly synchronized.

Loading the world A:

```bash 
ros2 launch robot_controller world_A_launch.py
```
Loading the world B:

```bash 
ros2 launch robot_controller world_B_launch.py
```

Execute the navigation and detectetion:

```bash 
ros2 launch robot_controller robot_controller_launch.py
```
