import pickle

import cv2
import pandas as pd
from ultralytics import YOLO


class BallTrack:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    # Interpolate Ball Positions to show the ball position if it is not detected in a frame
    def interpolate_ball(self, balll_postions):
        ball_positions = (balll_postion.get(1, []) for balll_postion in balll_postions)
        df_ball_positions = pd.DataFrame(
            ball_positions, columns=["x1", "y1", "x2", "y2"]
        )
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames, read_from_stubs=False, stubs_path=None):
        ball_detections = []
        if read_from_stubs:
            with open(stubs_path, "rb") as f:
                ball_detections = pickle.load(f)
                return ball_detections
        for frame in frames:
            bal_dict = self.detect_frame(frame)
            ball_detections.append(bal_dict)
        if stubs_path is not None:
            with open(stubs_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame)[0]
        bal_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            bal_dict[1] = result
        return bal_dict

    # Draw Bounding Box
    def draw_bbox(self, frames, ball_detections):
        output_video_frames = []
        for frame, bal_dict in zip(frames, ball_detections):  # Process frame-by-frame
            for track_id, bbox in bal_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame, f"Ball ID: {track_id}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)  # Append modified frame
        return output_video_frames
