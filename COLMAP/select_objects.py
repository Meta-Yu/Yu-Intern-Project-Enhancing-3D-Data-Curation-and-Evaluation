import os
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
