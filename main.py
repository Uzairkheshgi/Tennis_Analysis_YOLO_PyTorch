import cv2

from court_lines_detection import CourtLineDetector
from trackers import BallTrack, PlayerTrack
from utils import read_video, save_video


def main():
    # Read Videoimage
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Initialize PlayerTracking
    player_tracker = PlayerTrack(model_path="yolo11x.pt")
    player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stubs=True,
        stubs_path="trackers_stubs/players_detection.pkl",
    )
    
    # Initialize BallTracking
    ball_tracker = BallTrack(model_path="models/last.pt")
    ball_detections = ball_tracker.detect_frames(
        video_frames,
        read_from_stubs=True,
        stubs_path="trackers_stubs/ball_detection.pkl",
    )
    
    # Interpolate Ball Positions to show the ball position if it is not detected in a frame
    ball_detections = ball_tracker.interpolate_ball(ball_detections)
    
    # Court Line Detection
    court_line_detector = CourtLineDetector(
        model_path="models/keypoints_model.pth"
    )
    court_keypoints = court_line_detector.predict(
        video_frames[0],
    )

    # Choose and Filter closest two Players, ignore other people/players
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections
    )
    
    # Draw Bounding Box
    output_video_frames = player_tracker.draw_bbox(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bbox(output_video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_video(
        output_video_frames, court_keypoints
    )
    
    #write video frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 220, 115), 2)
        
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
