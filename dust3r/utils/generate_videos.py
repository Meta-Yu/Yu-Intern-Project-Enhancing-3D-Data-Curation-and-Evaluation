import numpy as np
import cv2
import glob
import os


def convert_img_video(image_paths_or_tensors, output_video_file, fps=5):


   # Get the dimensions of the first image or tensor (assuming all images have the same dimensions)
   first_image_or_tensor = cv2.imread(image_paths_or_tensors[0]) if isinstance(image_paths_or_tensors[0], (str, np.ndarray)) else image_paths_or_tensors[0].cpu().numpy()
   height, width, _ = first_image_or_tensor.shape if isinstance(first_image_or_tensor, np.ndarray) else first_image_or_tensor.shape[-2:]


   # Define the codec and create VideoWriter object
   fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec to use for the video (H.264 codec)
   video_writer = cv2.VideoWriter('/private/home/wangyu1369/dust3r/ego_exo_4d/videos/{}.mp4'.format(output_video_file), fourcc, fps, (width, height))


   # Iterate over each image or tensor and write it to the video
   for image_or_tensor in image_paths_or_tensors:
       if isinstance(image_or_tensor, str):
           # If image path, read the image using OpenCV
           frame = cv2.imread(image_or_tensor)
           frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
       elif isinstance(image_or_tensor, np.ndarray):
           # If numpy array, convert to uint8 and BGR format if necessary
           frame = image_or_tensor.astype(np.uint8)
           if len(frame.shape) == 3 and frame.shape[2] == 1:
               frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
       elif isinstance(image_or_tensor, torch.Tensor):
           # If torch tensor, convert to numpy array and then to uint8 and BGR format if necessary
           frame = image_or_tensor.cpu().numpy().astype(np.uint8)
           if len(frame.shape) == 3 and frame.shape[0] == 1:
               frame = frame[0]  # Remove the batch dimension if present
           if len(frame.shape) == 3 and frame.shape[0] == 3:
               frame = np.transpose(frame, (1, 2, 0))  # Change from CHW to HWC format
           if len(frame.shape) == 2:
               frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
           else:
               frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
       else:
           raise ValueError("Unsupported image type")


       # Write the frame to the video
       video_writer.write(frame)


   # Release the VideoWriter object and close the video file
   video_writer.release()

def get_img_selected(scene_name, stride=1):

    frame_dir = "/large_experiments/eht/egopose/user/tmp_0_80000_1/cache/{}/"
    image_list = sorted(glob.glob(os.path.join(frame_dir.format(scene_name), "hand/halo/images/aria01_rgb_*.jpg")))
    
    image_selected_new = []
    for frame_path in image_list:

        index = str(int((frame_path.split('/')[-1]).split('.')[-2][-6:]))

        if 670 <= int(index) <= 880:

            image_selected_new.append(frame_path)

    return image_selected_new

# selected_frames = get_img_selected(scene_name =  "sfu_cooking015_1", stride=1)

# # print(selected_frames[0])
# convert_img_video(image_paths_or_tensors = selected_frames , output_video_file = "sfu_cooking015_1", fps=30)

selected_frames = get_img_selected(scene_name =  "sfu_cooking_003_5", stride=1)

# # print(selected_frames[0])
convert_img_video(image_paths_or_tensors = selected_frames , output_video_file = "sfu_cooking_003_5", fps=20)


# selected_frames = get_img_selected(scene_name = "fair_cooking_06_6", stride=100)

# # # print(selected_frames[0])
# convert_img_video(image_paths_or_tensors = selected_frames , output_video_file = "fair_cooking_06_6", fps=1)