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

# Inject the ComputerVision_FP path so the node can locate 'pose_analysis.py' autonomously
sys.path.append(os.path.expanduser('~/yaqz_ws/src/ComputerVision_FP'))
try:
    from pose_analysis import BehaviorAnalyzer
except ImportError:
    # Fallback placeholder if behavior analyzer is missing during pure build tests
    class BehaviorAnalyzer:
        def get_behavior(self, track_id, kpts, box): return "UNKNOWN", (255, 255, 255)

class YaqzAIPipeline:
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
        """Dynamically allocates heavy AI models into RAM only when triggered."""
        self.pose_model = YOLO(pose_path)
        self.id_seg_model = YOLO(seg_path)
        self.analyzer = BehaviorAnalyzer()

    def unload_resources(self):
        """Purges models from memory to optimize system resources when inactive."""
        self.pose_model = None
        self.id_seg_model = None
        self.analyzer = None
        self.track_history.clear()

    def process_frame(self, frame):
        """Core AI pipeline execution loop."""
        if self.pose_model is None or self.id_seg_model is None:
            return frame

        # 1. Multi-Person Pose Tracking
        results = self.pose_model.track(
            frame, conf=0.5, iou=0.7, device="cpu", imgsz=640,
            tracker="bytetrack.yaml", persist=True, retina_masks=True, augment=False
        )

        img_annotated = results[0].plot(boxes=True)

        # 2. Extract bounding boxes and localized tracking analytics
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
                    id_results = self.id_seg_model(chest_roi, conf=0.5, verbose=False)
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
        self.publisher_ = None
        self.timer = None
        self.cap = None
        self.bridge = CvBridge()
        self.ai_pipeline = YaqzAIPipeline()  # Instantiated cleanly without loading models yet
        
        self.get_logger().info('Yaqz Vision Lifecycle Node instantiated. Status: UNCONFIGURED 💤')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Config State: Triggered manually. Loads weights into RAM."""
        self.get_logger().info('Configuring Yaqz Vision Node... Initializing AI Pipeline models.')
        
        pose_path = os.path.expanduser('~/yaqz_ws/src/ComputerVision_FP/yolov8n-pose.pt')
        seg_path = os.path.expanduser('~/yaqz_ws/src/ComputerVision_FP/id_seg_model.pt')
        
        try:
            # Models are ONLY injected into runtime memory during configuration sequence
            self.ai_pipeline.load_resources(pose_path, seg_path)
            self.get_logger().info('Yaqz AI Pipeline (Pose + Staff Seg + Behavior) loaded successfully! 🧠🚀')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize AI Pipeline: {str(e)}')
            return TransitionCallbackReturn.FAILURE

        self.publisher_ = self.create_lifecycle_publisher(Image, '/yaqz/processed_image', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activation State: Connects to hardware streaming devices and triggers callback timers."""
        self.get_logger().info('Activating Yaqz Vision Node... Spinning up hardware webcam channel. 🎥')
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.get_logger().error('CRITICAL: Webcam stream could not be established!')
            return TransitionCallbackReturn.FAILURE
            
        self.get_logger().info('Webcam stream securely opened! 🎉')
        self.timer = self.create_timer(0.05, self.process_frame)
        
        self.get_logger().info('Yaqz Vision Node transition verified. Status: ACTIVE 🔥')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivation State: Safely disarms timers and cameras to preserve battery energy."""
        self.get_logger().info('Deactivating Yaqz Vision Node... Freezing hardware loops.')
        if self.timer:
            self.destroy_timer(self.timer)
            self.timer = None
        if self.cap:
            self.cap.release()
            self.cap = None
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup State: Completely drops models from memory block."""
        self.get_logger().info('Cleaning up resources... Freeing AI Pipeline memory allocations.')
        self.ai_pipeline.unload_resources()
        return TransitionCallbackReturn.SUCCESS

    def process_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            annotated_frame = self.ai_pipeline.process_frame(frame)
            ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
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