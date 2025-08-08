from ultralytics import YOLO
import os
from dotenv import load_dotenv

load_dotenv()

model_path = os.environ['MODEL_PATH2']
model = YOLO(model_path)

# results = model.predict("input/example2.mp4", device='mps', verbose=False, save=True)
# results = model.predict("input/gk/IMG_3388.MOV", device='CPU', verbose=False, save=True,conf=0.01)
# results = model_goal.predict("input/gk/Video-2024-06-26-13-03-18_3355.MOV", device='CPU', verbose=False, save=True)

# results = model.track("input/gk/IMG_3379.mov", tracker="models/custom_tracker.yaml", save=True)
# model.predict("input/penalties_dataset/messi.mp4", device='mps', verbose=False, save=True)
# model.track("input/penalties_dataset/messi.mp4", device='mps', verbose=False, save=True)
# model_goal.detect("input/penalties_dataset/messi.mp4", device='CPU', verbose=False, save=True)

# tracker="models/custom_tracker.yaml",

# model.predict(os.environ['VIDEO_FILE'], save=True, conf=0.3, iou=0.3, device='MPS', verbose=False)

# model_emma.predict("input/gk/IMG_3352.MOV", device='CPU', verbose=False, save=True)

results = model.predict (
    source=os.environ['VIDEO_FILE'],
    conf=0.3,
    iou=0.45,
    # device='mps',  # Use 'cpu', 'cuda', or 'mps' based on your hardware
    classes=[0],   # Only track class 0 (typically person)
    save=True,     # Save results to disk
    verbose=False  # Suppress verbose output
)
print(results)