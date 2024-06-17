from fastapi import FastAPI, File, UploadFile
import cv2
from facenet_pytorch import MTCNN
import numpy as np
import torch
from io import BytesIO
from deepface import DeepFace

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

app = FastAPI()

def mtcnn_detect(img: np.ndarray) -> list:
    cropped_faces = []
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            x_left = int(max(0, box[0] - 0.3 * (box[2] - box[0])))
            x_right = int(min(img.shape[1], box[2] + 0.3 * (box[2] - box[0])))
            y_top = int(max(0, box[1] - 0.3 * (box[3] - box[1])))
            y_bottom = int(min(img.shape[0], box[3] + 0.3 * (box[3] - box[1])))

            cropped_face = img[y_top:y_bottom, x_left:x_right]
            cropped_faces.append(cropped_face)
    return cropped_faces

@app.post('/')
async def extract_faces(files: list[UploadFile] = File(...)):
    all_cropped_faces = []

    for file in files:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        cropped_faces = mtcnn_detect(img)
        for face in cropped_faces:
            success, encoded_img = cv2.imencode('.jpg', face)
            if success:
                all_cropped_faces.append(BytesIO(encoded_img.tobytes()))

    if len(all_cropped_faces) == 2:
        img1 = cv2.imdecode(np.frombuffer(all_cropped_faces[0].getvalue(), np.uint8), cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(np.frombuffer(all_cropped_faces[1].getvalue(), np.uint8), cv2.IMREAD_COLOR)

        try:
            result = DeepFace.verify(img1, img2, model_name='Facenet')
            verification_status = result["verified"]
            return {"message": "Face verification completed.", "verified": verification_status}
        except Exception as e:
            return {"message": "Face verification failed.", "error": str(e)}
    else:
        return {"message": "Please upload exactly two images for verification."}