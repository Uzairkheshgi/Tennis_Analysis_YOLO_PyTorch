from utils import read_video, save_video

from trackers import PlayerTrack


def main():
    # Read Videoimage
    input_video_path = "input_videos/input_video.mp4"
    # input_video_path = "input_videos/v1.mp4"
    # input_video_path = "input_videos/image.png"
    
    video_frames = read_video(input_video_path)
    # print("video_frames :", video_frames)
    
    # Initialize PlayerTrack
    players_tracker = PlayerTrack(model_path="yolo11x.pt")
    player_detection = players_tracker.detect_frames(video_frames)
    # print('player_detection :', player_detection)
    
    #outPut
    #Draw Bounding Box
    output_video_frames = players_tracker.draw_bbox(video_frames, player_detection)
    

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
