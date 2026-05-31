# ROS 2 CV Perception Node

This repository contains the core **ROS 2 Package** responsible for integrating Computer Vision pipelines into the autonomous security patrolling robot. It handles image processing and nodes lifecycle management.

---

## 🧠 AI Module Integration
The deep learning models, YOLOv8 training pipelines, and core computer vision scripts used by this node are maintained in a dedicated repository:
👉 **[Yaqz AI Module (ComputerVision_FP)](https://github.com/neiorz/ComputerVision_FP)**

---

##  Installation & Setup

### 1. Clone the Workspace
```bash
mkdir -p ~/yaqz_ws/src
cd ~/yaqz_ws/src
git clone [https://github.com/neiorz/ROS2_CV_perception.git](https://github.com/neiorz/ROS2_CV_perception.git) .
```
### 2. Install Dependencies

Make sure you have ROS 2 and OpenCV dependencies installed:
Bash
```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-cv-bridge ros-$ROS_DISTRO-vision-opencv
```
### 3. Build the Workspace

Navigate back to the workspace root and build the package:
Bash
```bash
cd ~/yaqz_ws
colcon build --packages-select yaqz_vision_integration
```
### Run & Execution

Always source the workspace environment before running the node:
Bash
1. Run the Vision Lifecycle Node
Bash
```bash
# Source the setup file
source install/setup.bash

# Run the node directly using Python
python3 src/yaqz_vision_integration/yaqz_vision_integration/vision_node.py
```
2. Manage Node Lifecycle States

Since this is a managed (Lifecycle) node, you need to transition its state from a new terminal (don't forget to source in the new terminal too):

**Configure the Node:**

    
```bash
ros2 lifecycle set /vision_node configure
```

  **Activate the Node (Start Processing/Camera):**
```bash
ros2 lifecycle set /vision_node activate
```

  **Deactivate the Node (Pause):**
```bash

ros2 lifecycle set /vision_node deactivate
```

  **Check Current State:**

```bash
ros2 lifecycle get /vision_node
```
