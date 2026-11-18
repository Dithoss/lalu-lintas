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

# Configuration
vehicle_history = {}

frame_index = 0

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

ttl_seconds = 3
ttl_frames = int(fps * ttl_seconds)

movement_threshold = 3
stop_duration = 1.5
stop_frames = int(fps * stop_duration)


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

            # Vehicle Position
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            center_x = int((x1 + x2) / 2)
            bottom_y = int(y2)

            # New Vehicle
            if track_id not in vehicle_history:

                vehicle_history[track_id] = {
                    "vehicle": vehicle,
                    "first_seen": frame_index,
                    "last_seen": frame_index,
                    "position": (center_x, bottom_y),
                    "stationary_frames": 0,
                    "is_stopped": False
                }

            # Existing Vehicle
            else:

                data = vehicle_history[track_id]

                previous_x, previous_y = data["position"]

                movement = (
                    (center_x - previous_x) ** 2 +
                    (bottom_y - previous_y) ** 2
                ) ** 0.5

                if movement <= movement_threshold:

                    data["stationary_frames"] += 1

                else:

                    data["stationary_frames"] = 0

                data["is_stopped"] = (
                    data["stationary_frames"] >= stop_frames
                )

                data["last_seen"] = frame_index

                data["position"] = (
                    center_x,
                    bottom_y
                )

            print(
                f"ID: {track_id} | "
                f"Vehicle: {vehicle} | "
                f"Stopped: {vehicle_history[track_id]['is_stopped']}"
            )

    # Remove Expired History
    for track_id in list(vehicle_history.keys()):
        last_seen = vehicle_history[track_id]["last_seen"]
        if frame_index - last_seen > ttl_frames:
            del vehicle_history[track_id]

    print("Vehicle Counts:", vehicle_counts)

    # Visualization
    annotated = results[0].plot()

    for track_id, data in vehicle_history.items():

        x, y = data["position"]

        if data["is_stopped"]:
            color = (0, 0, 255)
            status = "STOPPED"
        else:
            color = (0, 255, 0)
            status = "MOVING"

        cv2.circle(
            annotated,
            (x, y),
            5,
            color,
            -1
        )

        cv2.putText(
            annotated,
            f"ID {track_id}: {status}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.imshow("Traffic Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
