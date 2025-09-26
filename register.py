import cv2
import numpy as np
from ultralytics import YOLO
from database.mongo_setup import get_db
from utils.face_utils import get_embedding, compare_embeddings

# Load YOLO face detection model
model = YOLO("models/yolov8n-face.pt")  # CPU-friendly small face detector

def register_user():
    db = get_db()
    users_col = db.users

    name = input("Enter your name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[INFO] Looking for your face. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Detect faces in the frame
        results = model(frame, verbose=False)

        # Loop through detected faces
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    face_img = frame[y1:y2, x1:x2]

                    if face_img.size == 0:
                        continue

                    # Get face embedding using ArcFace
                    embedding = get_embedding(face_img)

                    # Check if this face is already registered
                    already_registered = False
                    for user in users_col.find():
                        stored_emb = np.array(user["embedding"])
                        if compare_embeddings(embedding, stored_emb):
                            print(f"[INFO] Face already registered as {user['name']}")
                            already_registered = True
                            break

                    if already_registered:
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    # Assign unique user_id
                    user_id = users_col.count_documents({}) + 1

                    # Save new user in MongoDB
                    users_col.insert_one({
                        "user_id": user_id,
                        "name": name,
                        "embedding": embedding.tolist()
                    })

                    print(f"[INFO] Registered {name} with ID {user_id}")

                    cap.release()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    register_user()
