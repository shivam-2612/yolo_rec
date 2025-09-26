import cv2
import numpy as np
from ultralytics import YOLO
from database.mongo_setup import get_db
from utils.face_utils import get_embedding, compare_embeddings
from utils.attendance_utils import mark_attendance

model = YOLO("models/yolov8n-face.pt")

def recognize_faces():
    db = get_db()
    users_col = db.users

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting attendance recognition. Press 'q' to quit.")

    recognized_track_ids = {}  # track_id: user_name
    unknown_faces = set()      # track_ids of unknown faces

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_resized = cv2.resize(frame, (640, 480))
        scale_x = frame.shape[1] / frame_resized.shape[1]
        scale_y = frame.shape[0] / frame_resized.shape[1]

        results = model.track(frame_resized, tracker="bytetrack.yaml", verbose=False)

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box, track_id in zip(r.boxes, r.boxes.id):
                    track_id = int(track_id)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)

                    face_img = frame[y1:y2, x1:x2]
                    if face_img.size == 0:
                        continue

                    name = "Unknown"

                    # Check if already recognized
                    if track_id in recognized_track_ids:
                        name = recognized_track_ids[track_id]
                    else:
                        embedding = get_embedding(face_img)
                        for user in users_col.find():
                            stored_emb = np.array(user["embedding"])
                            if compare_embeddings(embedding, stored_emb):
                                name = user["name"]
                                mark_attendance(db, user["user_id"], name)
                                recognized_track_ids[track_id] = name
                                break
                        # If still unknown, store in unknown_faces for reference
                        if name == "Unknown":
                            unknown_faces.add(track_id)

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_faces()
