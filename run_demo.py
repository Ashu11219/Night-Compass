import cv2
from ultralytics import YOLO
from alert_logic import should_alert

THERMAL_MODEL_PATH = 'runs/detect/thermal_full/weights/best.pt'
RGB_MODEL_PATH = 'runs/detect/rgb_full/weights/best.pt'

LABEL_CONFIDENCE_THRESHOLD = 0.3  # below this, box shows but no class/confidence text

def get_detections(model, frame):
    results = model.predict(frame, verbose=False)[0]
    dets = [
        (results.names[int(box.cls)], float(box.conf), box.xywhn[0].tolist())
        for box in results.boxes
    ]
    avg_conf = sum(d[1] for d in dets) / len(dets) if dets else 0.0
    return dets, avg_conf

def draw_detections(frame, detections):
    h, w = frame.shape[:2]
    EMERALD_GREEN = (174, 193, 107)
    YELLOW = (0, 255, 255)

    for class_name, confidence, (x_center, y_center, box_w, box_h) in detections:
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)  # border color changed

        if confidence >= LABEL_CONFIDENCE_THRESHOLD:
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, EMERALD_GREEN, 2)
    return frame

def run_on_video(video_path):
    thermal_model = YOLO(THERMAL_MODEL_PATH)
    rgb_model = YOLO(RGB_MODEL_PATH)

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        thermal_dets, thermal_conf = get_detections(thermal_model, frame)
        rgb_dets, rgb_conf = get_detections(rgb_model, frame)

        if thermal_conf >= rgb_conf:
            chosen_dets, source = thermal_dets, 'THERMAL'
        else:
            chosen_dets, source = rgb_dets, 'RGB'

        annotated_frame = draw_detections(frame.copy(), chosen_dets)

        # only show source label when something was actually detected
        if chosen_dets:
            cv2.putText(annotated_frame, f"Source: {source}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        alerts = should_alert(chosen_dets)
        for alert in alerts:
            cv2.putText(annotated_frame, f"ALERT: {alert['class']} {alert['direction']}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('NightCompass Demo', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_on_video('data/video_thermal_test/sample.mp4')