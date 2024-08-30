import sys
import os
import open3d as o3d
import plotly.graph_objects as go
import numpy as np




def est_point_clouds_visualization(category_name, scene_name):
   cur_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}.ply'
   pcd = o3d.io.read_point_cloud(cur_path.format(category_name, scene_name))
   colors = np.asarray(pcd.colors)
   points_matrix = np.asarray(pcd.points)
   fig = go.Figure(data=[go.Scatter3d(x=points_matrix[:, 0], y=points_matrix[:, 1], z=points_matrix[:, 2])])
   fig.update_traces(mode = 'markers',marker=dict(size = 3, color = colors))
   fig.write_image('/private/home/wangyu1369/DROID-SLAM/droid_slam/pcd_plots_visualization/est_pcd_{}_{}.png'.format(category_name, scene_name))

def gt_point_clouds_visualization(category_name, scene_name):
   cur_path = '/datasets01/co3dv2/080422/{}/{}/pointcloud.ply'
   pcd = o3d.io.read_point_cloud(cur_path.format(category_name, scene_name))
   colors = np.asarray(pcd.colors)
   points_matrix = np.asarray(pcd.points)
   fig = go.Figure(data=[go.Scatter3d(x=points_matrix[:, 0], y=points_matrix[:, 1], z=points_matrix[:, 2])])
   fig.update_traces(mode = 'markers',marker=dict(size = 3, color = colors))
   fig.write_image('/private/home/wangyu1369/DROID-SLAM/droid_slam/pcd_plots_visualization/gt_pcd_{}_{}.png'.format(category_name, scene_name))




if __name__ == '__main__':
    for scene in ["apple/189_20393_38136/",
                "cake/374_42274_84517/",
                "pizza/102_11950_20611",
                "kite/401_52055_102127"
                ]:
        category_name = scene.split("/")[0]
        scene_name = scene.split("/")[1]

        est_point_clouds_visualization(category_name = category_name, scene_name = scene_name)
        # gt_point_clouds_visualization(category_name = category_name, scene_name = scene_name)

        print('Finished plotting for {}'.format(category_name))