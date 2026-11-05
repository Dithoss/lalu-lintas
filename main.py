
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

# Vehicle History
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
    frame_index += 1

    # YOLO + ByteTrack
    results = model.track(
        frame,
        conf=0.4,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    boxes = results[0].boxes

    # Vehicle Counting
    vehicle_counts = {
        "bicycle": 0,
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    # Vehicle Tracking + History
    if boxes.id is not None:

        track_ids = boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            class_id = int(box.cls[0])
            if class_id not in class_mapping:
                continue
            vehicle = class_mapping[class_id]
            vehicle_counts[vehicle] += 1

            # Mengambil posisi kendaraan
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            center_x = int((x1 + x2) / 2)
            bottom_y = int(y2)

            # Membuat histori kendaraan
            if track_id not in vehicle_history:
                vehicle_history[track_id] = {
                    "vehicle": vehicle,
                    "first_seen": frame_index,
                    "last_seen": frame_index,
                    "position": (center_x, bottom_y)
                }

            else:
                vehicle_history[track_id]["last_seen"] = frame_index
                vehicle_history[track_id]["position"] = (
                    center_x,
                    bottom_y
                )

            print(f"ID: {track_id} | Vehicle: {vehicle}")

    # Menghapus histori yang sudah kedaluwarsa
    for track_id in list(vehicle_history.keys()):
        last_seen = vehicle_history[track_id]["last_seen"]
        if frame_index - last_seen > ttl_frames:
            del vehicle_history[track_id]

    print("Vehicle Counts:", vehicle_counts)

    # Visualisasi
    annotated = results[0].plot()
    for track_id, data in vehicle_history.items():
        x, y = data["position"]
        cv2.circle(
            annotated,
            (x, y),
            5,
            (0, 255, 0),
            -1
        )

    cv2.imshow("Traffic Tracking", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
