import cv2
import numpy as np
import onnxruntime as ort

# Load ArcFace model
arcface_model = ort.InferenceSession("models/arc.onnx", providers=["CPUExecutionProvider"])
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (112, 112))  # resize to 112x112
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    face_img = face_img.astype(np.float32) / 255.0  # normalize to 0-1
    # face_img = np.transpose(face_img, (2, 0, 1))  # HWC to CHW
    face_img = np.expand_dims(face_img, 0)  # add batch dimension
    return face_img

def get_embedding(face_img):
    face_input = preprocess_face(face_img)
    input_name = arcface_model.get_inputs()[0].name
    output_name = arcface_model.get_outputs()[0].name
    emb = arcface_model.run([output_name], {input_name: face_input})[0]
    emb = emb / np.linalg.norm(emb)
    return emb.flatten()

def compare_embeddings(emb1, emb2, threshold=0.6):
    dist = np.linalg.norm(emb1 - emb2)
    return dist < threshold
