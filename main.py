
from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture("dataset/videos/traffic.mp4")

class_mapping = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}
vehicle_history = {}
frame_index = 0
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

ttl_seconds = 3
ttl_frames = int(fps * ttl_seconds)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        conf=0.4,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )
    vehicle_history = {}
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30

    ttl_seconds = 3
    ttl_frames = int(fps * ttl_seconds)
    
    for track_id in list(vehicle_history.keys()):
        last_seen = vehicle_history[track_id]["last_seen"]
        if frame_index - last_seen > ttl_frames:
            del vehicle_history[track_id]

    vehicle_counts = {
        "bicycle": 0,
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    for box in results[0].boxes:
        class_id = int(box.cls[0])

        if class_id in class_mapping:
            vehicle = class_mapping[class_id]
            vehicle_counts[vehicle] += 1
    for result in results:
        boxes = result.boxes

        if boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            for track_id in track_ids:
                print(track_id)
                if class_id in class_mapping:
                    vehicle = class_mapping[class_id]
                    print(f"ID: {track_id} | Vehicle{vehicle} ")

    print(vehicle_counts)

    annotated = results[0].plot()

    cv2.imshow("Traffic Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
