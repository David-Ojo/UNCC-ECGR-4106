import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
from collections import deque, Counter


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "chemical hazard",
    "fire",
    "no helmet",
    "smoke",
    "water leak"
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224




class CustomHazardCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



classifier_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




def load_cnn_model(checkpoint_path):
    model = CustomHazardCNN(num_classes=NUM_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


def load_resnet18_model(checkpoint_path):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    return model


def load_yolo_model(yolo_path):
    model = YOLO(yolo_path)
    return model



def classify_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = classifier_transform(image).unsqueeze(0).to(DEVICE)

    start_time = time.time()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    inference_time = time.time() - start_time

    predicted_idx = predicted_idx.item()
    confidence = confidence.item()
    predicted_class = CLASS_NAMES[predicted_idx]

    return predicted_class, confidence, inference_time


def run_yolo_inference(yolo_model, image_path, output_dir, conf_threshold=0.25):
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()

    results = yolo_model.predict(
        source=image_path,
        conf=conf_threshold,
        save=True,
        project=output_dir,
        name="predictions",
        exist_ok=True,
        verbose=False
    )

    inference_time = time.time() - start_time

    result = results[0]
    boxes = result.boxes

    detections = []

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            xyxy = box.xyxy[0].cpu().numpy().tolist()

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": confidence,
                "box_xyxy": xyxy
            })

    return detections, inference_time


def print_yolo_detections(detections):
    if len(detections) == 0:
        print("YOLO detections: none")
        return

    print("YOLO detections:")
    for det in detections:
        print(
            f"  {det['class_name']} | "
            f"conf={det['confidence']:.3f} | "
            f"box={det['box_xyxy']}"
        )

        


CLASS_TO_ID = {
    "chemical hazard": 0,
    "fire": 1,
    "no helmet": 2,
    "smoke": 3,
    "water leak": 4
}


class HazardConvergenceLayer:
    def __init__(
        self,
        history_size=3,
        min_votes=2,
        classifier_weight=0.40,
        yolo_weight=0.60,
        final_score_threshold=0.50,
        weak_yolo_threshold=0.03
    ):
        """
        history_size:
            Number of sampled frames to remember.

        min_votes:
            Number of recent frames that must agree before confirming.

        classifier_weight/yolo_weight:
            Weighting used to combine classifier confidence and YOLO confidence.

        final_score_threshold:
            Minimum fused score needed for final confirmation.

        weak_yolo_threshold:
            Minimum YOLO confidence to consider a weak detection useful.
        """

        self.history_size = history_size
        self.min_votes = min_votes
        self.classifier_weight = classifier_weight
        self.yolo_weight = yolo_weight
        self.final_score_threshold = final_score_threshold
        self.weak_yolo_threshold = weak_yolo_threshold

        self.history = deque(maxlen=history_size)

    def update(self, classifier_class, classifier_conf, yolo_detections):
        """
        Inputs:
            classifier_class: predicted class from CNN or ResNet
            classifier_conf: classifier softmax confidence
            yolo_detections: list of YOLO detection dictionaries

        Returns:
            decision dictionary
        """

        best_yolo_class = None
        best_yolo_conf = 0.0
        best_yolo_box = None

        for det in yolo_detections:
            if det["confidence"] > best_yolo_conf:
                best_yolo_class = det["class_name"]
                best_yolo_conf = det["confidence"]
                best_yolo_box = det["box_xyxy"]

        # Case 1: YOLO found something and agrees with classifier
        if best_yolo_class == classifier_class and best_yolo_conf >= self.weak_yolo_threshold:
            fused_score = (
                self.classifier_weight * classifier_conf
                + self.yolo_weight * best_yolo_conf
            )

            candidate_class = classifier_class
            candidate_box = best_yolo_box

        # Case 2: YOLO found something, but it disagrees with classifier
        elif best_yolo_class is not None and best_yolo_conf >= 0.25:
            fused_score = best_yolo_conf
            candidate_class = best_yolo_class
            candidate_box = best_yolo_box

        # Case 3: Classifier says smoke strongly, but YOLO has no box
        elif classifier_class == "smoke" and classifier_conf >= 0.85:
            fused_score = classifier_conf * 0.60
            candidate_class = classifier_class
            candidate_box = None

        # Case 4: Not enough evidence
        else:
            fused_score = 0.0
            candidate_class = None
            candidate_box = None

        if candidate_class is not None and fused_score >= self.final_score_threshold:
            self.history.append(candidate_class)
        else:
            self.history.append(None)

        vote_counts = Counter([x for x in self.history if x is not None])

        if len(vote_counts) > 0:
            top_class, top_votes = vote_counts.most_common(1)[0]
        else:
            top_class, top_votes = None, 0

        confirmed = top_votes >= self.min_votes

        return {
            "confirmed": confirmed,
            "candidate_class": candidate_class,
            "final_class": top_class if confirmed else None,
            "fused_score": fused_score,
            "yolo_class": best_yolo_class,
            "yolo_conf": best_yolo_conf,
            "classifier_class": classifier_class,
            "classifier_conf": classifier_conf,
            "box_xyxy": candidate_box,
            "history": list(self.history)
        }