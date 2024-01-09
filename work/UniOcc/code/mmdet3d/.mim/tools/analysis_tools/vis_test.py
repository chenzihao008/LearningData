import os

import mmcv
import open3d as o3d
import numpy as np
import torch
import pickle
import math
from typing import Tuple, List, Dict, Iterable
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm

NOT_OBSERVED = -1
FREE = 0
OCCUPIED = 1
FREE_LABEL = 17
BINARY_OBSERVED = 1
BINARY_NOT_OBSERVED = 0

VOXEL_SIZE = [0.4, 0.4, 0.4]
POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]
SPTIAL_SHAPE = [200, 200, 16]
TGT_VOXEL_SIZE = [0.4, 0.4, 0.4]
TGT_POINT_CLOUD_RANGE = [-40, -40, -1, 40, 40, 5.4]


colormap_to_colors = np.array(
    [
        [0,   0,   0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 barrier  orange
        [220, 20, 60, 255],    # 2 bicycle  Blue
        [255, 127, 80, 255],   # 3 bus  Darkslategrey
        [255, 158, 0, 255],  # 4 car  Crimson
        [233, 150, 70, 255],   # 5 cons. Veh  Orangered
        [255, 61, 99, 255],  # 6 motorcycle  Darkorange
        [0, 0, 230, 255], # 7 pedestrian  Darksalmon
        [47, 79, 79, 255],  # 8 traffic cone  Red
        [255, 140, 0, 255],# 9 trailer  Slategrey
        [255, 99, 71, 255],# 10 truck Burlywood
        [0, 207, 191, 255],    # 11 drive sur  Green
        [175, 0, 75, 255],  # 12 other lat  nuTonomy green
        [75, 0, 75, 255],  # 13 sidewalk
        [112, 180, 60, 255],    # 14 terrain
        [222, 184, 135, 255],    # 15 manmade
        [0, 175, 0, 255],   # 16 vegeyation
], dtype=np.float32)



def voxel2points(voxel, occ_show, voxelSize):
    """
    Args:
        voxel: (Dx, Dy, Dz)
        occ_show: (Dx, Dy, Dz)
        voxelSize: (dx, dy, dz)

    Returns:
        points: (N, 3) 3: (x, y, z)
        voxel: (N, ) cls_id
        occIdx: (x_idx, y_idx, z_idx)
    """
    occIdx = torch.where(occ_show)
    points = torch.cat((occIdx[0][:, None] * voxelSize[0] + POINT_CLOUD_RANGE[0], \
                        occIdx[1][:, None] * voxelSize[1] + POINT_CLOUD_RANGE[1], \
                        occIdx[2][:, None] * voxelSize[2] + POINT_CLOUD_RANGE[2]),
                       dim=1)      # (N, 3) 3: (x, y, z)
    return points, voxel[occIdx], occIdx


def voxel_profile(voxel, voxel_size):
    """
    Args:
        voxel: (N, 3)  3:(x, y, z)
        voxel_size: (vx, vy, vz)

    Returns:
        box: (N, 7) (x, y, z - dz/2, vx, vy, vz, 0)
    """
    centers = torch.cat((voxel[:, :2], voxel[:, 2][:, None] - voxel_size[2] / 2), dim=1)     # (x, y, z - dz/2)
    # centers = voxel
    wlh = torch.cat((torch.tensor(voxel_size[0]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[1]).repeat(centers.shape[0])[:, None],
                     torch.tensor(voxel_size[2]).repeat(centers.shape[0])[:, None]), dim=1)
    yaw = torch.full_like(centers[:, 0:1], 0)
    return torch.cat((centers, wlh, yaw), dim=1)


def rotz(t):
    """Rotation about the z-axis."""
    c = torch.cos(t)
    s = torch.sin(t)
    return torch.tensor([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def my_compute_box_3d(center, size, heading_angle):
    """
    Args:
        center: (N, 3)  3: (x, y, z - dz/2)
        size: (N, 3)    3: (vx, vy, vz)
        heading_angle: (N, 1)
    Returns:
        corners_3d: (N, 8, 3)
    """
    h, w, l = size[:, 2], size[:, 0], size[:, 1]
    center[:, 2] = center[:, 2] + h / 2
    l, w, h = (l / 2).unsqueeze(1), (w / 2).unsqueeze(1), (h / 2).unsqueeze(1)
    x_corners = torch.cat([-l, l, l, -l, -l, l, l, -l], dim=1)[..., None]
    y_corners = torch.cat([w, w, -w, -w, w, w, -w, -w], dim=1)[..., None]
    z_corners = torch.cat([h, h, h, h, -h, -h, -h, -h], dim=1)[..., None]
    corners_3d = torch.cat([x_corners, y_corners, z_corners], dim=2)
    corners_3d[..., 0] += center[:, 0:1]
    corners_3d[..., 1] += center[:, 1:2]
    corners_3d[..., 2] += center[:, 2:3]
    return corners_3d


def show_point_cloud(points: np.ndarray, colors=True, points_colors=None, bbox3d=None, voxelize=False,
                     bbox_corners=None, linesets=None, vis=None, offset=[0,0,0], large_voxel=True, voxel_size=0.4):
    """
    :param points: (N, 3)  3:(x, y, z)
    :param colors: false 不显示点云颜色
    :param points_colors: (N, 4）
    :param bbox3d: voxel grid (N, 7) 7: (center, wlh, yaw=0)
    :param voxelize: false 不显示voxel边界
    :param bbox_corners: (N, 8, 3)  voxel grid 角点坐标, 用于绘制voxel grid 边界.
    :param linesets: 用于绘制voxel grid 边界.
    :return:
    """
    if vis is None:
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
    if isinstance(offset, list) or isinstance(offset, tuple):
        offset = np.array(offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points+offset)
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(points_colors[:, :3])
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1, origin=[0, 0, 0])

    voxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    if large_voxel:
        vis.add_geometry(voxelGrid)
    else:
        vis.add_geometry(pcd)

    if voxelize:
        line_sets = o3d.geometry.LineSet()
        line_sets.points = o3d.open3d.utility.Vector3dVector(bbox_corners.reshape((-1, 3))+offset)
        line_sets.lines = o3d.open3d.utility.Vector2iVector(linesets.reshape((-1, 2)))
        line_sets.paint_uniform_color((0, 0, 0))
        vis.add_geometry(line_sets)

    vis.add_geometry(mesh_frame)

    # ego_pcd = o3d.geometry.PointCloud()
    # ego_points = generate_the_ego_car()
    # ego_pcd.points = o3d.utility.Vector3dVector(ego_points)
    # vis.add_geometry(ego_pcd)

    return vis


def show_occ(occ_state, occ_show, voxel_size, vis=None, offset=[0, 0, 0]):
    """
    Args:
        occ_state: (Dx, Dy, Dz), cls_id
        occ_show: (Dx, Dy, Dz), bool
        voxel_size: [0.4, 0.4, 0.4]
        vis: Visualizer
        offset:

    Returns:

    """
    colors = colormap_to_colors / 255
    pcd, labels, occIdx = voxel2points(occ_state, occ_show, voxel_size)
    # pcd: (N, 3)  3: (x, y, z)
    # labels: (N, )  cls_id
    _labels = labels % len(colors)
    pcds_colors = colors[_labels]   # (N, 4)

    bboxes = voxel_profile(pcd, voxel_size)    # (N, 7)   7: (x, y, z - dz/2, dx, dy, dz, 0)
    bboxes_corners = my_compute_box_3d(bboxes[:, 0:3], bboxes[:, 3:6], bboxes[:, 6:7])      # (N, 8, 3)

    bases_ = torch.arange(0, bboxes_corners.shape[0] * 8, 8)
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])  # lines along y-axis
    edges = edges.reshape((1, 12, 2)).repeat(bboxes_corners.shape[0], 1, 1)     # (N, 12, 2)
    # (N, 12, 2) + (N, 1, 1) --> (N, 12, 2)   此时edges中记录的是bboxes_corners的整体id: (0, N*8).
    edges = edges + bases_[:, None, None]

    vis = show_point_cloud(
        points=pcd.numpy(),
        colors=True,
        points_colors=pcds_colors,
        voxelize=True,
        bbox3d=bboxes.numpy(),
        bbox_corners=bboxes_corners.numpy(),
        linesets=edges.numpy(),
        vis=vis,
        offset=offset,
        large_voxel=True,
        voxel_size=0.4
    )
    return vis


def generate_the_ego_car():
    ego_range = [-2, -1, 0, 2, 1, 1.5]
    ego_voxel_size=[0.1, 0.1, 0.1]
    ego_xdim = int((ego_range[3] - ego_range[0]) / ego_voxel_size[0])
    ego_ydim = int((ego_range[4] - ego_range[1]) / ego_voxel_size[1])
    ego_zdim = int((ego_range[5] - ego_range[2]) / ego_voxel_size[2])
    temp_x = np.arange(ego_xdim)
    temp_y = np.arange(ego_ydim)
    temp_z = np.arange(ego_zdim)
    ego_xyz = np.stack(np.meshgrid(temp_y, temp_x, temp_z), axis=-1).reshape(-1, 3)
    ego_point_x = (ego_xyz[:, 0:1] + 0.5) / ego_xdim * (ego_range[3] - ego_range[0]) + ego_range[0]
    ego_point_y = (ego_xyz[:, 1:2] + 0.5) / ego_ydim * (ego_range[4] - ego_range[1]) + ego_range[1]
    ego_point_z = (ego_xyz[:, 2:3] + 0.5) / ego_zdim * (ego_range[5] - ego_range[2]) + ego_range[2]
    ego_point_xyz = np.concatenate((ego_point_y, ego_point_x, ego_point_z), axis=-1)
    ego_points_label =  (np.ones((ego_point_xyz.shape[0]))*16).astype(np.uint8)
    ego_dict = {}
    ego_dict['point'] = ego_point_xyz
    ego_dict['label'] = ego_points_label
    return ego_point_xyz


# def parse_args():
#     parser = argparse.ArgumentParser(description='Visualize the predicted '
#                                      'result of nuScenes')
#     parser.add_argument(
#         'res', help='Path to the predicted result')
#     parser.add_argument(
#         '--canva-size', type=int, default=1000, help='Size of canva in pixel')
#     parser.add_argument(
#         '--vis-frames',
#         type=int,
#         default=500,
#         help='Number of frames for visualization')
#     parser.add_argument(
#         '--scale-factor',
#         type=int,
#         default=4,
#         help='Trade-off between image-view and bev in size of '
#         'the visualized canvas')
#     parser.add_argument(
#         '--version',
#         type=str,
#         default='val',
#         help='Version of nuScenes dataset')
#     parser.add_argument('--draw-gt', action='store_true')
#     parser.add_argument(
#         '--root_path',
#         type=str,
#         default='./data/nuscenes',
#         help='Path to nuScenes dataset')
#     parser.add_argument(
#         '--save_path',
#         type=str,
#         default='./vis',
#         help='Path to save visualization results')
#     parser.add_argument(
#         '--format',
#         type=str,
#         default='image',
#         choices=['video', 'image'],
#         help='The desired format of the visualization result')
#     parser.add_argument(
#         '--fps', type=int, default=10, help='Frame rate of video')
#     parser.add_argument(
#         '--video-prefix', type=str, default='vis', help='name of video')
#     args = parser.parse_args()
#     return args

if __name__=="__main__":
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    



    path_root = Path('/mnt/allfile/project/RenderOcc/result/')
    path_list = list(path_root.rglob('*.npy'))
    path_list = sorted(path_list,key=lambda x : int(x.stem))[10:11]
    for i in tqdm(path_list):
        pred_occ = np.load(i)[0]
        # pred_occ = np.load('/mnt/allfile/project/RenderOcc/result/5.npy')[0]
        # pred_occ = np.load('/mnt/allfile/project/FlashOCC/work_dirs/flashocc_r50/results/scene-0103/6d9984d09d52479e837da2fd09e192cc/pred.npz')
        # pred_occ = pred_occ['pred']
        # occ_canvas
        top = np.ones_like(pred_occ)
        top[:,:,-1]=0
        voxel_show = np.logical_and(pred_occ != FREE_LABEL, top)
        # voxel_show = pred_occ != FREE_LABEL
        voxel_size = VOXEL_SIZE
        vis = show_occ(torch.from_numpy(pred_occ), torch.from_numpy(voxel_show), voxel_size=voxel_size, vis=vis,
                    offset=[0, pred_occ.shape[0] * voxel_size[0] * 1.2 * 0, 0])
        
        look_at = np.array([-0.185, 0.513, 2.485])
        front = np.array([-0.974, -0.055, 0.221])
        up = np.array([0.221, 0.014, 0.975])
        zoom = 100

        view_control = vis.get_view_control()
        view_control.set_lookat(look_at)
        view_control.set_front(front)
        view_control.set_up(up)
        view_control.set_zoom(zoom)

        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])
        opt.line_width = 5

        vis.poll_events()
        vis.update_renderer()
        vis.run()
        vis.clear_geometries()

    vis.destroy_window()