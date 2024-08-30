import os
import shutil

def select_object(category_name, number_of_frames=10):

    scene_list = os.listdir('/datasets01/co3dv2/080422/{}'.format(category_name))[:-5]

    selected = []
    for scene in scene_list:
        cur_path = os.listdir('/datasets01/co3dv2/080422/{}/{}/images/'.format(category_name, scene))
        if len(cur_path)>=202:
            selected.append(scene)
        if len(selected)==number_of_frames:
            break
    
    return selected

def get_img_selected(category_name, scene_name, stride=1):
    co3d_file_path = '/datasets01/co3dv2/080422/{}/{}/images/'.format(category_name, scene_name)
    image_list = os.listdir(co3d_file_path)
    
    image_selected = []
    for t in range(1, len(image_list)+1, stride):
        image_file = os.path.join(co3d_file_path, image_list[t-1])
        image_selected.append(image_file)
    return image_selected


if __name__ == '__main__':
    stride = 100
    number_of_scenes = 10

    cur_dir = '/datasets01/co3dv2/080422/{}/{}/images/{}'
    os.makedirs('/private/home/wangyu1369/COLMAP/selected_frames_co3d/object_{}'.format(number_of_scenes), exist_ok=True)
    os.makedirs('/private/home/wangyu1369/COLMAP/selected_frames_co3d_text/object_{}'.format(number_of_scenes), exist_ok=True)
    os.makedirs('/private/home/wangyu1369/COLMAP/co3d_colmap_result/object_{}'.format(number_of_scenes), exist_ok=True)

    
    parent_dir = '/private/home/wangyu1369/COLMAP/selected_frames_co3d/object_{}'.format(number_of_scenes)
    text_dir = '/private/home/wangyu1369/COLMAP/selected_frames_co3d_text/object_{}'.format(number_of_scenes)
    result_dir = '/private/home/wangyu1369/COLMAP/co3d_colmap_result/object_{}'.format(number_of_scenes)


    os.makedirs(os.path.join(parent_dir, 'stride_{}'.format(stride)), exist_ok=True)
    parent_dir = os.path.join(parent_dir, 'stride_{}'.format(stride))

    os.makedirs(os.path.join(text_dir, 'stride_{}'.format(stride)), exist_ok=True)
    text_dir = os.path.join(text_dir, 'stride_{}'.format(stride))

    os.makedirs(os.path.join(result_dir, 'stride_{}'.format(stride)), exist_ok=True)
    result_dir = os.path.join(result_dir, 'stride_{}'.format(stride))

    for category_name in [
                          "ball", 
                          "book", 
                          "couch", 
                          "kite", 
                          "sandwich",
                          "frisbee", 
                          "hotdog", 
                          "skateboard", 
                          "suitcase"
                          ]:
        objects = select_object(category_name, number_of_frames=number_of_scenes)

        # print(objects)
        # os.makedirs(os.path.join(result_dir, '{}'.format(category_name)), exist_ok=True)
        # result_dir = os.path.join(result_dir, '{}'.format(category_name))


        with open(os.path.join(text_dir, '{}.txt'.format(category_name)), 'w') as f:
            for object in objects:
                f.write('{}_{}'.format(category_name, object) + '\n')

        for scene_name in objects:

            os.makedirs(os.path.join(result_dir, '{}_{}'.format(category_name, scene_name)), exist_ok=True)


            folder_name = '{}_{}'.format(category_name, scene_name)
            os.makedirs(os.path.join(parent_dir, folder_name), exist_ok=True)
            
            os.makedirs(os.path.join(parent_dir, folder_name), exist_ok=True)

            frames_list = get_img_selected(category_name, scene_name, stride=stride)


            for frame in frames_list:
                # print(frame)
                src_path = frame

                dst_path = os.path.join(parent_dir, folder_name, frame.split('/')[-1])

                shutil.copy(src_path, dst_path)

        
