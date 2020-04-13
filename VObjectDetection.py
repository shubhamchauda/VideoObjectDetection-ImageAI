from imageai.Detection import VideoObjectDetection
import os


exe_path = os.getcwd()
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(exe_path,"yolo.h5"))
detector.loadModel()
vid_path = detector.detectObjectsFromVideo(input_file_path = os.path.join(exe_path,"Video_name_with_extension"), output_file_path=os.path.join(exe_path, "Video_output_file_name"),frames_per_second=30, log_progress=True)
print(vid_path)
