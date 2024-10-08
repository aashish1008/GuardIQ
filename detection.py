import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")
tracker = sv.ByteTrack()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

class_names = {
    0: 'Gun',
    1: 'Hands-up',
    2: 'Knife',
    3: 'Mask',
    4: 'Normal-Person',
    5: 'Security-Guard'
}

cap = cv2.VideoCapture("video/robbery.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("threat_detection.mp4", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
while True:
    _, frame = cap.read()
    results = model(frame)[0]

    # print(results)
    detections = sv.Detections.from_ultralytics(results)

    detections = tracker.update_with_detections(detections)

    labels = [f"#{tracker_id}-{results.names[class_id]}" for tracker_id, class_id in
              zip(detections.tracker_id, detections.class_id)]

    if detections.class_id == 0 & detections.class_id == 3:
        # send alert if this condition matched
        pass

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(),
        detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    cv2.putText(annotated_frame, f"FPS: {fps}", (25, 62),
                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
                color=(0, 0, 0), thickness=2)

    out.write(annotated_frame)

    cv2.imshow("image", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
