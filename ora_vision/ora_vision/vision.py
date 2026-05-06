# Ros 2 imports
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
# Edge detection imports
import cv2
import numpy as np

# Ros 2 topic names
RAW_IMAGE_TOPIC = "/depth_camera/image_raw"
EDGE_IMAGE_TOPIC = "/vision/detected_edges"

# Default ROI points
X1 = 100
Y1 = 100
X2 = 100
Y2 = 100

# Gaussian blur parameters
BLUR_KERNEL_SIZE = 5
SIGMA_BLUR_CONTROL = 0

# Morphological parameters
MORPH_KERNEL_SIZE = 3

# Canny edge detection thresholds
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 100

# Hough transform parameters
HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LEN = 40
HOUGH_MAX_LINE_GAP = 5

class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.image_subscription = self.create_subscription(
            Image,
            RAW_IMAGE_TOPIC,
            self.process_image,
            10
        )

        self.edge_publisher = self.create_publisher(
            Image,
            EDGE_IMAGE_TOPIC,
            10
        )

    def draw_lines(self, frame):
        # Blur to reduce noise
        blur = cv2.GaussianBlur(
            src=frame, 
            ksize=(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 
            sigmaX=0
        )

        # Convert to HLS
        hls = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)

        # Define range for white lane lines
        white_lower = np.array([0, 200, 0])
        white_upper = np.array([255, 255, 255])
        white_mask = cv2.inRange(hls, white_lower, white_upper)

        # Define range for yellow lane lines
        yellow_lower = np.array([10, 0, 100])
        yellow_upper = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

        # Combine masks
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        filtered_image = cv2.bitwise_and(blur, blur, mask=mask)

        # Canny edge detection
        # 1st and 2nd parameters specify the threshold of brightness change to classify as a line
        edges = cv2.Canny(
            image=filtered_image, 
            threshold1=CANNY_LOW_THRESHOLD, 
            threshold2=CANNY_HIGH_THRESHOLD
        )

        # Hough Line Transform
        # 3rd parameter specifies the # of pixels a line must extend in order to be classified as a line
        lines = cv2.HoughLinesP(
            image=edges, 
            rho=HOUGH_RHO, 
            theta=HOUGH_THETA, 
            threshold=HOUGH_THRESHOLD, 
            minLineLength=HOUGH_MIN_LINE_LEN, 
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        # Draw Lines on original image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 3)

        return frame

    def process_image(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        processed_image = self.draw_lines(image)
        self.edge_publisher.publish(processed_image)

def main(args=None):
    rclpy.init(args=args)

    vision_node = VisionNode()

    rclpy.spin(vision_node)

    # Destroy the node explicitly
    # This step is optional, if not used the node will be destroyed by garbage collector
    vision_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
