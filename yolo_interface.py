#!/media/uzair/DC3299C63299A654/AI/Projects/Tennis_Analysis/.venv/bin/python

from ultralytics import YOLO

# model = YOLO("yolo11x.pt")
model = YOLO("models/best.pt")
resultImage = model.predict("input_videos/image.png", save=False, project="runs/detect")
# resultVideo = model.predict(
#     "input_videos/input_video.mp4", conf=0.4, save=True, project="runs/detect"
# )
trackVideo = model.track(
    "input_videos/input_video.mp4", conf=0.4, save=False, project="runs/detect"
)
# for box in resultImage[0].boxes:
#     print('box :' , box)
