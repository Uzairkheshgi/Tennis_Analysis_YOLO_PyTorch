import pickle
import sys

import cv2
from ultralytics import YOLO

sys.path.append("/media/uzair/DC3299C63299A654/AI/Projects/Tennis_Analysis/utils")
from utils import measure_distance, get_center_of_bbox


class PlayerTrack:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)

        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
            
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
                    
            distances.append((track_id, min_distance))
        
        # sorrt the distances in ascending orderplayer_center
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracksplayer_center
        chosen_players = [distances[0][0], distances[1][0]]
        
        return chosen_players

    def detect_frames(self, frames, read_from_stubs=False, stubs_path=None):
        player_detections = []

        if read_from_stubs:

            with open(stubs_path, "rb") as f:
                player_detections = pickle.load(f)

                return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stubs_path is not None:

            with open(stubs_path, "wb") as f:
                pickle.dump(player_detections, f)

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

        for frame, player_dict in zip(
            frames, player_detections
        ):  # Process frame-by-frame
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Player ID: {track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2
                )

            output_video_frames.append(frame)  # Append modified frame

        return output_video_frames
