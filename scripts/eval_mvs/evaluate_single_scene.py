import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse

import trimesh
from pathlib import Path
import subprocess
import open3d as o3d
import sys
import render_utils as rend_util
from tqdm import tqdm

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def cull_scan(scan, mesh_path, result_mesh_file, instance_dir, skip_cull=False, skip_scale=False):
    
    # load poses
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png"))) + sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    n_images = len(image_paths)
    # load scale
    cam_file = '{0}/cameras_sphere.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    # load corrected 
    cam_file = '{0}/cameras_corrected.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)
    
    # load mask
    mask_dir = '{0}/mask'.format(instance_dir)
    masks = []
    if os.path.exists(mask_dir):
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        for p in mask_paths:
            mask = cv2.imread(p)
            masks.append(mask)

    # load mesh
    mesh = trimesh.load(mesh_path)
    
    # load transformation matrix

    vertices = mesh.vertices

    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    if not skip_cull:
        sampled_masks = []
        for i in tqdm(range(n_images),  desc="Culling mesh given masks"):
            intrinsic = torch.from_numpy(camera_dict['intrinsic_{}'.format(i)]).float().cuda()
            pose = torch.from_numpy(camera_dict['c2w_{}'.format(i)]).float().cuda()
            
            w2c = torch.inverse(pose).cuda()

            with torch.no_grad():
                H, W, _ = masks[i].shape
                # transform and project
                cam_points = intrinsic @ w2c @ vertices
                pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
                pix_coords = pix_coords.permute(1, 0)
                pix_coords[..., 0] /= W - 1
                pix_coords[..., 1] /= H - 1
                pix_coords = (pix_coords - 0.5) * 2
                valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
                
                # dialate mask similar to unisurf
                maski = masks[i][:, :, 0].astype(np.float32) / 256.
                maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

                sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

                sampled_mask = sampled_mask + (1. - valid)
                sampled_masks.append(sampled_mask)

        sampled_masks = torch.stack(sampled_masks, -1)
        # filter
        
        mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
        face_mask = mask[mesh.faces].all(axis=1)

        mesh.update_vertices(mask)
        mesh.update_faces(face_mask)
    
    # transform vertices to world 
    if not skip_scale:
        scale_mat = scale_mats[0]
        mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    mesh.export(result_mesh_file)
    del mesh
    # # For scene-level
    # mesh = o3d.io.read_triangle_mesh(result_mesh_file)
    # mesh = post_process_mesh(mesh, cluster_to_keep=1)
    # o3d.io.write_triangle_mesh(result_mesh_file, mesh)
    # del mesh
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh')
    parser.add_argument('--MVS', type=str,  default='Offical_MVS_Dataset', help='path to the GT MVS point clouds')
    parser.add_argument('--skip_cull', action='store_true')
    parser.add_argument('--skip_scale', action='store_true')
    parser.add_argument('--nop', type=int, default=500000)
    parser.add_argument('--max_dist', type=float, default=1.0)
    args = parser.parse_args()

    Offical_MS_Dataset = args.MVS
    out_dir = os.path.dirname(args.input_mesh)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scan = args.scan_id
    ply_file = args.input_mesh
    print("cull mesh ....")
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
    cull_scan(scan, ply_file, result_mesh_file, instance_dir=os.path.join(Offical_MS_Dataset, f'{scan}'), skip_cull=args.skip_cull, skip_scale=args.skip_scale)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f"python {script_dir}/eval.py --data {result_mesh_file} --gt {os.path.join(Offical_MS_Dataset, scan, 'fuse.ply')} --nop {args.nop} --max_dist {args.max_dist} --visualize_threshold {args.max_dist / 2}"
    os.system(cmd)