import cv2
import numpy as np
import time
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from collections import defaultdict

video_path = "D:/sarvin/ترم چهار/signal/project/car1.mp4"

model = YOLO("yolov8m.pt")
allowed_classes = [0, 2, 3]

def select_tracker_for_box(cls, box):
    _, _, w, h = box
    if cls == 0:
        return "CSRT"
    elif max(w, h) > 150:
        return "MOSSE"
    else:
        return "KCF"

def create_tracker(name):
    if name == "CSRT": return cv2.TrackerCSRT_create()
    elif name == "KCF": return cv2.TrackerKCF_create()
    elif name == "MOSSE": return cv2.TrackerMOSSE_create()
    else: raise ValueError("Unsupported tracker: " + name)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_mixed_trackers.mp4", fourcc, fps, (width, height))

trackers = []
tracker_ids = []
tracker_types_assigned = []
kalman_filters = {}
next_id = 0
frame_id = 0
prev_time = time.time()
MAX_DETECT_FRAME = 50
fps_log = []
tracker_count = defaultdict(int)

def create_kalman_filter(x, y):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([x, y, 0, 0])
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000.
    kf.R *= 5.
    kf.Q *= 0.01
    return kf

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    updated_boxes = []
    updated_ok = []
    for t in trackers:
        ok, box = t.update(frame)
        updated_ok.append(ok)
        updated_boxes.append(box if ok else None)

    if frame_id <= MAX_DETECT_FRAME:
        detections = model(frame)[0]
        boxes = []

        for det in detections.boxes:
            cls = int(det.cls.item())
            if cls not in allowed_classes:
                continue
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            boxes.append((x1, y1, w, h, cls))

        for x, y, w, h, cls in boxes:
            box = (x, y, w, h)
            matched = False
            for i, trk_box in enumerate(updated_boxes):
                if not updated_ok[i]:
                    continue
                if iou(box, trk_box) > 0.3:
                    matched = True
                    break
            if not matched:
                chosen_type = select_tracker_for_box(cls, box)
                tracker = create_tracker(chosen_type)
                tracker.init(frame, box)
                trackers.append(tracker)
                tracker_ids.append(next_id)
                tracker_types_assigned.append(chosen_type)
                tracker_count[chosen_type] += 1
                cx = x + w / 2
                cy = y + h / 2
                kalman_filters[next_id] = create_kalman_filter(cx, cy)
                next_id += 1

    temp_trackers = []
    temp_ids = []
    temp_types = []
    temp_kfs = {}

    for i, ok in enumerate(updated_ok):
        id_ = tracker_ids[i]
        if ok:
            box = updated_boxes[i]
            cx = box[0] + box[2] / 2
            cy = box[1] + box[3] / 2
            kalman_filters[id_].predict()
            kalman_filters[id_].update([cx, cy])
            temp_trackers.append(trackers[i])
            temp_ids.append(id_)
            temp_types.append(tracker_types_assigned[i])
            temp_kfs[id_] = kalman_filters[id_]

            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            label = f"ID {id_}-{tracker_types_assigned[i]}"
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, label, (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            kalman_filters[id_].predict()
            pred_x, pred_y = kalman_filters[id_].x[:2]
            pred_w, pred_h = 100, 100
            p1 = (int(pred_x - pred_w/2), int(pred_y - pred_h/2))
            p2 = (int(pred_x + pred_w/2), int(pred_y + pred_h/2))
            cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
            cv2.putText(frame, f"ID {id_} (pred)", (p1[0], p1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            temp_ids.append(id_)
            temp_types.append(tracker_types_assigned[i])
            temp_kfs[id_] = kalman_filters[id_]

    trackers = temp_trackers
    tracker_ids = temp_ids
    tracker_types_assigned = temp_types
    kalman_filters = temp_kfs

    curr_time = time.time()
    fps_text = 1 / (curr_time - prev_time)
    fps_log.append(fps_text)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps_text:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    out.write(frame)

cap.release()
out.release()

print("✅ Done: Output saved as 'output_mixed_trackers.mp4'")
print("📊 Tracker distribution:")
for k, v in tracker_count.items():
    print(f"{k}: {v} instances")

print(f"⚡ Average FPS: {sum(fps_log)/len(fps_log):.2f}")
