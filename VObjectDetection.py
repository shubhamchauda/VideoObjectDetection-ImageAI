from imageai.Detection import VideoObjectDetection
import os


exe_path = os.getcwd()
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(exe_path,"yolo.h5"))
detector.loadModel()
vid_path = detector.detectObjectsFromVideo(input_file_path = os.path.join(exe_path,"traffic.mp4"), output_file_path=os.path.join(exe_path, "traffic_mini"),frames_per_second=30, log_progress=True)
print(vid_path)