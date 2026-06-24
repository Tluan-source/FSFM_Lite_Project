import sys
from pathlib import Path

import cv2
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(str(PROJECT_ROOT))

from src.models.fsfm_lite import FSFMLite

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_fsfm_lite.pth"
)

print(f"Model Path: {MODEL_PATH}")
print(f"Exists: {MODEL_PATH.exists()}")

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = FSFMLite()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Model Loaded Successfully")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

def predict_face(face_rgb):

    image = face_transform(face_rgb)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():

        logits = model(image)

        probs = torch.softmax(
            logits,
            dim=1
        )

        pred = logits.argmax(
            dim=1
        ).item()

        conf = probs.max().item()

    return pred, conf


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("Press ESC to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        pad = int(
            0.2 * max(w, h)
        )

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)

        x2 = min(
            frame.shape[1],
            x + w + pad
        )

        y2 = min(
            frame.shape[0],
            y + h + pad
        )

        face = frame[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        pred, conf = predict_face(
            face_rgb
        )

        if pred == 0:

            label = f"LIVE {conf:.2%}"
            color = (0, 255, 0)

        else:

            label = f"SPOOF {conf:.2%}"
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow(
        "FSFM-Lite Webcam",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()