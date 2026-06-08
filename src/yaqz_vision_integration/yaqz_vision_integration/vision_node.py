#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.qos import qos_profile_sensor_data

sys.path.append(os.path.expanduser('~/yaqz_ws/src/ComputerVision_FP'))
try:
    from pose_analysis import BehaviorAnalyzer
except ImportError:
    class BehaviorAnalyzer:
        def get_behavior(self, track_id, kpts, box): 
            return "UNKNOWN", (255, 255, 255)


class YaqzVisionPipeline:
    def __init__(self):
        """
        Lightweight initialization. Models are NOT loaded here to respect 
        the ROS 2 Lifecycle architecture constraints (lazy resource allocation).
        """
        self.pose_model = None
        self.id_seg_model = None
        self.analyzer = None
        self.track_history = {}

    def load_resources(self, pose_path, seg_path):
        """Dynamically allocates heavy AI models into RAM and shifts them to GPU."""
        self.pose_model = YOLO(pose_path)
        self.id_seg_model = YOLO(seg_path)
        
        # Enforce GPU (CUDA) execution for maximum acceleration
        self.pose_model.to('cuda')
        self.id_seg_model.to('cuda')
        
        self.analyzer = BehaviorAnalyzer()

    def unload_resources(self):
        """Purges models from memory to optimize system resources when inactive."""
        self.pose_model = None
        self.id_seg_model = None
        self.analyzer = None
        self.track_history.clear()

    def process_frame(self, frame):
        """Core AI pipeline execution loop operating entirely on GPU."""
        if self.pose_model is None or self.id_seg_model is None:
            return frame

        # Multi-Person Pose Tracking enforced on CUDA
        results = self.pose_model.track(
            frame, conf=0.5, iou=0.7, device="cuda", imgsz=640,
            tracker="bytetrack.yaml", persist=True, retina_masks=True, augment=False
        )

        img_annotated = results[0].plot(boxes=True)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints_data = results[0].keypoints.data

            for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
                x1, y1, x2, y2 = map(int, box.tolist())
                
                # --- A. Chest ROI Segmentation (Security Badge Verification) ---
                roi_h = int((y2 - y1) * 0.4)
                chest_roi = frame[max(0, y1):y1+roi_h, max(0, x1):x2]
                
                is_staff = False
                if chest_roi.size > 0:
                    # Enforce CUDA evaluation for internal segmentations
                    id_results = self.id_seg_model(chest_roi, conf=0.5, device="cuda", verbose=False)
                    if id_results[0].masks is not None:
                        is_staff = True
                
                status_txt = "STAFF" if is_staff else "VISITOR"
                status_color = (0, 255, 0) if is_staff else (0, 0, 255)
                
                # --- B. Predictive Behavior Assessment ---
                current_kpts = keypoints_data[i].cpu().numpy()
                current_box = box.tolist()
                behavior, b_color = self.analyzer.get_behavior(track_id, current_kpts, current_box)
                
                # Overlay specialized metrics onto frame
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), status_color, 2)
                cv2.putText(img_annotated, f"ID:{track_id} {status_txt}", 
                            (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(img_annotated, f"Act: {behavior}", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, b_color, 1)

                # --- C. Historical Motion Trajectory Trails ---
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                track = self.track_history.get(track_id, [])
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 10: track.pop(0)
                self.track_history[track_id] = track
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_annotated, [points], isClosed=False, color=(0, 0, 255), thickness=2)

        return img_annotated


class YaqzVisionLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('yaqz_vision_node')
        self.subscription_ = None
        self.bridge = CvBridge()
        self.ai_pipeline = YaqzVisionPipeline()
        
        # Switch to a regular publisher — no lifecycle gating needed for the output channel
        self.publisher_ = self.create_publisher(Image, '/yaqz/processed_image', 10)
        
        self.get_logger().info('Yaqz Vision Lifecycle Node instantiated. Status: UNCONFIGURED')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Config State: Allocates weights into VRAM."""
        self.get_logger().info('Configuring Yaqz Vision Node... Initializing AI Pipeline models.')
        
        pose_path = os.path.expanduser('~/yaqz_ws/src/ComputerVision_FP/yolov8n-pose.pt')
        seg_path = os.path.expanduser('~/yaqz_ws/src/ComputerVision_FP/id_seg_model.pt')
        
        try:
            self.ai_pipeline.load_resources(pose_path, seg_path)
            self.get_logger().info('Yaqz AI Pipeline successfully mapped onto GPU (CUDA).')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize AI Pipeline: {str(e)}')
            return TransitionCallbackReturn.FAILURE

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activation State: Subscribes to the external camera sensor using sensor QoS."""
        self.get_logger().info('Activating Yaqz Vision Node... Connecting to distributed image stream.')
        
        # Enforce Best-Effort Sensor Data QoS profile to match v4l2_camera constraint perfectly
        self.subscription_ = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile=qos_profile_sensor_data
        )
        
        self.get_logger().info('Yaqz Vision Node transition verified. Status: ACTIVE')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivation State: Safely disarms incoming data subscription flows."""
        self.get_logger().info('Deactivating Yaqz Vision Node... Freezing subscription hooks.')
        if self.subscription_:
            self.destroy_subscription(self.subscription_)
            self.subscription_ = None
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup State: Releases deep learning weights blocks from memory allocations."""
        self.get_logger().info('Cleaning up resources... Freeing AI Pipeline GPU/RAM configurations.')
        self.ai_pipeline.unload_resources()
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg: Image):
        """Pipeline callback synchronized with external camera frames stream."""
        try:
            # Step 1: Decode ROS Image message back into OpenCV Matrix
            cv_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Step 2: Feed frame to the GPU-accelerated execution loop
            annotated_frame = self.ai_pipeline.process_frame(cv_frame)
            
            # Step 3: Serialize Matrix array back into standard ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            ros_image.header = msg.header
            
            # Step 4: Publish inference metrics out to the ecosystem network
            self.publisher_.publish(ros_image)
            
            cv2.imshow("Yaqz Security Robot - AI Full Analysis", annotated_frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'AI Pipeline Execution Error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = YaqzVisionLifecycleNode()
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt detected.')
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
