import os
import cv2

frame_folder = "./frames_from_videos"

def get_frames(video_path: str):
    if os.path.exists(frame_folder) and not os.path.isdir(frame_folder): os.remove(frame_folder)
    if not os.path.exists(frame_folder): os.mkdir(frame_folder)
    
    video_name = video_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
    os.mkdir(frame_folder + '/' + video_name)
    
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(frame_folder + "/" + video_name + "//%s_frame%d.jpg" % (video_name, count), image)     # save frame as JPEG file
      count += 1
      success,image = vidcap.read()


if __name__ == "__main__":
    # test
    get_frames('/home/owner/Documents/DEV/Python/SuperResolution/0_144.mp4')