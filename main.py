import cv2
from ultralytics import YOLO
# pip install opencv-python ultralytics


def main():
    # Load the YOLOv8 model
    print("Loading model...")
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # YOLO assigns random IDs
    id_map = {} 
    next_player_label = 1

    print("Starting video stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=0, verbose=False) # classes=0 means only detect 'person' class
        boxes = results[0].boxes

        if boxes is not None:
            for box in boxes:
                if box.id is not None:
                    track_id = int(box.id.item())
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # If we haven't seen this ID before, assign it a Player Number
                    if track_id not in id_map:
                        if next_player_label <= 10: # only allow 10 players
                            id_map[track_id] = f"Player {next_player_label}"
                            next_player_label += 1
                        else:
                            id_map[track_id] = "Spectator"
                    
                    display_label = id_map[track_id]

                    color = (255, 0, 0) # bounding box color, blue

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(frame, (x1, y1 - 30), (x1 + 150, y1), color, -1)
                    cv2.putText(frame, display_label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Player Tracking System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
