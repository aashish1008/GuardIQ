import supervision as sv
from ultralytics import YOLO
import cv2
from datetime import datetime
import logging
from bot import BankGuardBot

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def resize_frame(frame, scale=0.5):
    # Images, Video and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


class GuardIQ:
    def __init__(self, model_name):
        self.model = YOLO(model_name)
        self.tracker = sv.ByteTrack()
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

        self.class_names = {
            0: 'Gun',
            1: 'Hands-up',
            2: 'Knife',
            3: 'Mask',
            4: 'Normal-Person',
            5: 'Security-Guard'
        }

        self.bot = BankGuardBot()

    def add_timestamp(self, img):
        """Add timestamp to the image"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        height, width, _ = img.shape
        cv2.putText(img, f"Time: {current_time}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def check_threats(self, detections):
        """Check for specific threat conditions"""
        if len(detections.class_id) == 0:
            return False, ""

        class_ids = set(detections.class_id)

        # Check for gun and mask combination
        if 0 in class_ids and 3 in class_ids:
            return True, "Detected armed person with mask!"

        # Check for knife
        if 2 in class_ids:
            return True, "Detected person with knife!"

        # Check for hands up (potential robbery)
        if 1 in class_ids:
            return True, "Detected person with hands up - possible robbery!"

        return False, ""

    def process_frame(self, frame):
        """Process a single frame and return annotated frame"""
        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = self.tracker.update_with_detections(detections)

        # Create labels with tracker IDs
        labels = [f"#{tracker_id}-{self.class_names[class_id]}"
                  for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)]

        # Check for threats
        threat_detected, alert_message = self.check_threats(detections)
        if threat_detected:
            self.bot.send_alert(frame, alert_message)

        # Annotate frame
        annotated_frame = self.bounding_box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        # Add timestamp
        annotated_frame = self.add_timestamp(annotated_frame)

        return annotated_frame

    def run(self, video_source):
        """Main run loop"""
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                logging.error(f"Failed to open video source: {video_source}")
                return

            w, h, fps = (int(cap.get(x)) for x in
                         (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

            out = cv2.VideoWriter("video/bank_robbery_detection.mp4", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = resize_frame(frame)

                annotated_frame = self.process_frame(frame)
                out.write(annotated_frame)
                cv2.imshow("GuardIQ: AI Powered Surveillance System for Banks", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()

        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")


if __name__ == "__main__":
    model = "model/best.pt"
    video_source = "video/robbery.mp4"
    inference = GuardIQ(model)
    inference.run(video_source=video_source)
