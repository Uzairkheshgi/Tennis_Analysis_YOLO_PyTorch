from trackers import BallTrack, PlayerTrack
from utils import read_video, save_video
from court_lines_detection import CourtLineDetector


def main():
    # Read Videoimage
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Initialize PlayerTracking
    players_tracker = PlayerTrack(model_path="yolo11x.pt")
    player_detection = players_tracker.detect_frames(
        video_frames,
        read_from_stubs=True,
        stubs_path="trackers_stubs/players_detection.pkl",
    )

    # Initialize BallTracking
    ball_tracker = BallTrack(model_path="models/last.pt")
    ball_detection = ball_tracker.detect_frames(
        video_frames,
        read_from_stubs=True,
        stubs_path="trackers_stubs/ball_detection.pkl",
    )

    # Court Line Detection
    court_line_detector = CourtLineDetector(
        model_path="models/model_tennis_court_det.pt"
    )
    court_keypoits_detection = court_line_detector.predict(
        video_frames[0],
    )

    # Draw Bounding Box
    output_video_frames = players_tracker.draw_bbox(video_frames, player_detection)
    output_video_frames = ball_tracker.draw_bbox(output_video_frames, ball_detection)
    output_video_frames = court_keypoits_detection.draw_keypoints_video(
        output_video_frames, court_keypoits_detection
    )
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
