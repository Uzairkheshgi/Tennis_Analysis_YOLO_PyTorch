import cv2
from ultralytics import YOLO


class PlayerTrack:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        player_detections = []
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)
        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        player_dict = {}

        for box in results.boxes:
            track_id = box.id.tolist()[0]
            result = box.xyxy.tolist()[0]
            obj_class_id = box.cls.tolist()[0]
            obj_class = id_name_dict[obj_class_id]

            if obj_class == "person":
                player_dict[track_id] = result
        return player_dict

# Draw Bounding Box
    def draw_bbox(self, frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(frames, player_detections):  # Process frame-by-frame
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)  # Append modified frame
        return output_video_frames


