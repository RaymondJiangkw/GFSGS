#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from arguments import FOOTPRINT_DISTRIBUTION, footprint_activations
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.general_utils import build_scaling_rotation
from utils.graphics_utils import BasicPointCloud, estimateTangentPlane

import tinycudann as tcnn
from itertools import chain
from pytorch3d.ops.knn import knn_points

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.geovalue_activation = lambda x: self.geovalue_mul * torch.sigmoid(x) # torch.log(1 + torch.clamp_min(x, 0))
        self.inverse_geovalue_activation = lambda y: inverse_sigmoid(y / self.geovalue_mul) # torch.exp(y / self.geovalue_mul) - 1
        self.rotation_activation = torch.nn.functional.normalize
    
    def setup_network(self):
        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3, 
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            }
        )
        
        self.color_net = tcnn.Network(
            n_input_dims=self.encoder_dir.n_output_dims * 2 + self.latent_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.latent_dim * 2,
                "n_hidden_layers": 2,
            }
        )

    def __init__(self, dataset):
        self.active_sh_degree = 0
        self.max_sh_degree = dataset.sh_degree
        self.geovalue_mul = dataset.geovalue_mul
        self._xyz = torch.empty(0)
        self._latents = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._geovalue = torch.empty(0)
        self.max_weight = torch.empty(0)
        self.count_accum = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_absgradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.K = dataset.K
        self.adjacent_matrix = None
        self.propagate_features = not dataset.not_propagate_features
        self.not_use_ray_dir = dataset.not_use_ray_dir
        self.not_use_reflected_ray_dir = dataset.not_use_reflected_ray_dir
        self.distance_coefficient = dataset.distance_coefficient
        
        self.color_mode = dataset.color_mode
        self.latent_dim = dataset.latent_dim
        
        self.setup_functions()
        self.setup_network()
    
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._latents, 
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._geovalue,
            self.max_weight, 
            self.count_accum, 
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_absgradient_accum, 
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.encoder_dir.state_dict(), 
            self.color_net.state_dict()
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._latents, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._geovalue,
        self.max_weight, 
        self.count_accum, 
        self.max_radii2D, 
        xyz_gradient_accum, 
        xyz_absgradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale, 
        enc_dir_dict, 
        color_net_dict) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_absgradient_accum = xyz_absgradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.encoder_dir.load_state_dict(enc_dir_dict)
        self.color_net.load_state_dict(color_net_dict)
    
    @torch.no_grad()
    def compute_adjacent_matrix(self):
        if not self.propagate_features:
            return
        xyz = self.get_xyz[None] # (1, N, 3)
        self.adjacent_matrix = knn_points(xyz, xyz, K=self.K).idx[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling) #.clamp(max=1)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    def get_colors(self, camera_center: torch.Tensor):
        assert self.color_mode == 'latent'
        latents = self._latents
        diffuse_colors = self._features_dc.squeeze().half()
        if self.propagate_features:
            assert self.adjacent_matrix is not None
            assert len(self.adjacent_matrix) == len(self.get_xyz)
            xyz = self.get_xyz[:, None, :].half()  # (N, 1, 3)
            neighbor_xyz = self.get_xyz[self.adjacent_matrix] # (N, K, 3)
            self._weights = torch.nn.functional.normalize((1 - torch.exp(-self.get_geovalue.half())) * torch.exp(-torch.norm(xyz - neighbor_xyz, dim=-1, p=2) * self.distance_coefficient), dim=-1, p=1, eps=1e-7) # (N, K)
            # self._weights = torch.nn.functional.normalize(footprint_activations[FOOTPRINT_DISTRIBUTION](self.get_geovalue.half()) * torch.exp(-torch.norm(xyz - neighbor_xyz, dim=-1, p=2) * self.distance_coefficient), dim=-1, p=1, eps=1e-7) # (N, K)
            neighbor_latents = latents[self.adjacent_matrix] # (N, K, F)
            neighbor_diffuse_colors = diffuse_colors[self.adjacent_matrix] # (N, K, 3)
            latents = (self._weights[:, :, None] * neighbor_latents).sum(dim=1) # (N, F)
            diffuse_colors = (self._weights[:, :, None] * neighbor_diffuse_colors).sum(dim=1) # (N, 3)
        # Ref-NeRF's design to input the reflection of the ray direction w.r.t. the local normal
        dir_pp = -(self.get_xyz - camera_center.repeat(len(latents), 1))
        ray_dir = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        normal = self.get_covariance()[:, :3, :3][:, -1, :]
        flip_mask = (ray_dir * normal).sum(dim=-1) < 0
        normal = torch.where(flip_mask[:, None].expand(-1, 3), -normal, normal)
        # assert ((ray_dir * normal).sum(dim=-1) > 0).all()
        dot = (ray_dir * normal).sum(dim=-1, keepdim=True)
        dir = 2 * dot * normal - ray_dir # (N, 3)

        # Convert to local frame
        # dir = (build_rotation(self.get_rotation).mT @ dir[:, :, None]).squeeze(dim=-1)
        encoded_ray_dir = self.encoder_dir((-ray_dir.half() + 1) / 2)
        encoded_reflected_ray_dir = self.encoder_dir((dir.half() + 1) / 2)
        if self.not_use_ray_dir:
            encoded_ray_dir = torch.zeros_like(encoded_ray_dir)
        if self.not_use_reflected_ray_dir:
            encoded_reflected_ray_dir = torch.zeros_like(encoded_reflected_ray_dir)
        h = torch.cat([encoded_reflected_ray_dir, encoded_ray_dir, latents], dim=-1) # dot.half(), 
        assert not h.isnan().any()
        h = self.color_net(h) + diffuse_colors
        color = torch.sigmoid(h).to(torch.float32)
        return color
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        features = torch.cat((features_dc, features_rest), dim=1) # (N, F, 3)
        if self.propagate_features:
            assert self.adjacent_matrix is not None
            assert len(self.adjacent_matrix) == len(self.get_xyz)
            xyz = self.get_xyz[:, None, :]  # (N, 1, 3)
            neighbor_xyz = self.get_xyz[self.adjacent_matrix, :] # (N, K, 3)
            self._weights = torch.nn.functional.normalize((1 - torch.exp(-self.get_geovalue)) * torch.exp(-torch.norm(xyz - neighbor_xyz, dim=-1, p=2) * self.distance_coefficient), dim=-1, p=1) # (N, K)
            # self._weights = torch.nn.functional.normalize(footprint_activations[FOOTPRINT_DISTRIBUTION](self.get_geovalue) * torch.exp(-torch.norm(xyz - neighbor_xyz, dim=-1, p=2) * self.distance_coefficient), dim=-1, p=1) # (N, K)
            neighbor_features = features[self.adjacent_matrix, :, :] # (N, K, F, 3)
            return (self._weights[:, :, None, None] * neighbor_features).sum(dim=1)
        return features
    
    @property
    def get_geovalue(self):
        return self.geovalue_activation(self._geovalue)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_geovalue : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        features = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        if self.color_mode == 'sh':
            features[:, :, 0] = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        else:
            features[:, :, 0] = torch.nan_to_num(inverse_sigmoid(torch.tensor(np.asarray(pcd.colors)).half().cuda()), 0., 1e4, -1e4)
        latents = (torch.randn((fused_point_cloud.shape[0], self.latent_dim)) * 1e-4).half().cuda()

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        # rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")
        rots = estimateTangentPlane(pcd.points)

        geovalue = self.inverse_geovalue_activation(init_geovalue * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._latents = nn.Parameter(latents.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._geovalue = nn.Parameter(geovalue.requires_grad_(True))
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.count_accum = torch.zeros((self.get_xyz.shape[0]), device="cuda").long()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.compute_adjacent_matrix()
    
    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_absgradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.encoder_dir = self.encoder_dir.cuda().requires_grad_(True)
        self.color_net = self.color_net.cuda().requires_grad_(True)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._geovalue], 'lr': training_args.geovalue_lr_init, "name": "geovalue"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}, 
            {'params': [self._latents], 'lr': training_args.latent_lr, "name": "latent"}, 
            {'params': chain(self.encoder_dir.parameters(), self.color_net.parameters()), 'lr': training_args.latent_lr, "name": "network"}
        ]

        # Important! Otherwise, using fp16 will lead to NaN with eps of 1e-15.
        # Thanks for Chengxuan Zhu for letting me know.
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15 if self.color_mode == "sh" else 1e-7)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.geovalue_scheduler_args = get_expon_lr_func(lr_init=training_args.geovalue_lr_init, 
                                                         lr_final=training_args.geovalue_lr_final, 
                                                         max_steps=training_args.geovalue_lr_max_steps)
        self.latent_scheduler_args = get_expon_lr_func(lr_init=training_args.latent_lr, 
                                                       lr_final=training_args.latent_lr * 0.1, 
                                                       max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "geovalue":
                lr = self.geovalue_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] in ["latent", "network"]:
                lr = self.latent_scheduler_args(iteration)
                param_group['lr'] = lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._latents.shape[1]):
            l.append('l_{}'.format(i))
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('geovalue')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        latent = self._latents.detach().cpu().numpy()
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        geovalue = self._geovalue.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, latent, f_dc, f_rest, geovalue, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        torch.save({'encoder_dir': self.encoder_dir.state_dict(), 'color_net': self.color_net.state_dict()}, path.replace('.ply', '.pth'))

    def reset_geovalue(self, value):
        geovalue_new = self.inverse_geovalue_activation(torch.min(self.get_geovalue, torch.ones_like(self.get_geovalue)*value))
        optimizable_tensors = self.replace_tensor_to_optimizer(geovalue_new, "geovalue")
        self._geovalue = optimizable_tensors["geovalue"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        geovalue = np.asarray(plydata.elements[0]["geovalue"])[..., np.newaxis]
        
        l_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('l_')]
        l_names = sorted(l_names, key = lambda x: int(x.split('_')[-1]))
        # assert len(l_names) == self.latent_dim
        latents = np.zeros((xyz.shape[0], len(l_names)))
        for idx, attr_name in enumerate(l_names):
            latents[:, idx] = np.asarray(plydata.elements[0][attr_name])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._latents = nn.Parameter(torch.tensor(latents, dtype=torch.half, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._geovalue = nn.Parameter(torch.tensor(geovalue, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.compute_adjacent_matrix()
        
        if len(l_names) > 0:
            self.setup_network()
            if os.path.exists(path.replace('.ply', '.pth')):
                state_dict = torch.load(path.replace('.ply', '.pth'), weights_only=True)
                self.encoder_dir.load_state_dict(state_dict['encoder_dir'])
                self.color_net.load_state_dict(state_dict['color_net'])
        self.encoder_dir = self.encoder_dir.cuda()
        self.color_net = self.color_net.cuda()

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "network":
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._latents = optimizable_tensors["latent"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._geovalue = optimizable_tensors["geovalue"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if self.xyz_gradient_accum is not None: self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        if self.xyz_absgradient_accum is not None: self.xyz_absgradient_accum = self.xyz_absgradient_accum[valid_points_mask]
        if self.denom is not None: self.denom = self.denom[valid_points_mask]
        if self.max_weight is not None: self.max_weight = self.max_weight[valid_points_mask]
        if self.count_accum is not None: self.count_accum = self.count_accum[valid_points_mask]
        if self.max_radii2D is not None: self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.adjacent_matrix = None
    
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if not group["name"] in tensors_dict:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_latent, new_features_dc, new_features_rest, new_geovalue, new_scaling, new_rotation, new_max_weight, new_count_accum):
        d = {"xyz": new_xyz,
        "latent": new_latent, 
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "geovalue": new_geovalue,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._latents = optimizable_tensors["latent"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._geovalue = optimizable_tensors["geovalue"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_absgradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_weight = torch.cat((self.max_weight, new_max_weight), dim=0)
        self.count_accum = torch.cat((self.count_accum, new_count_accum), dim=0)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.adjacent_matrix = None
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, percent_dense, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_latents = self._latents[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_geovalue = self._geovalue[selected_pts_mask].repeat(N,1)
        new_max_weight = self.max_weight[selected_pts_mask].repeat(N)
        new_count_accum = self.count_accum[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_latents, new_features_dc, new_features_rest, new_geovalue, new_scaling, new_rotation, new_max_weight, new_count_accum)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, percent_dense):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_latents = self._latents[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_geovalue = self._geovalue[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_max_weight = self.max_weight[selected_pts_mask]
        new_count_accum = self.count_accum[selected_pts_mask]

        self.densification_postfix(new_xyz, new_latents, new_features_dc, new_features_rest, new_geovalue, new_scaling, new_rotation, new_max_weight, new_count_accum)

    def prune(self, min_geovalue, extent = None, max_screen_size = None):
        prune_mask = (self.get_geovalue < min_geovalue).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            small_points_ws = self.get_scaling.max(dim=1).values < 1e-3
            prune_mask |= torch.logical_or(torch.logical_or(big_points_vs, big_points_ws), small_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    
    def prune_by_weight(self, min_weight):
        prune_mask = (self.max_weight < min_weight).squeeze()
        # print(f'Prune {torch.sum(prune_mask).detach().item()}/{len(prune_mask)} Surfels that are not Opaque Enough.')
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        self.max_weight = torch.zeros_like(self.max_weight)
    
    def prune_unused(self):
        prune_mask = (self.count_accum <= 0).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
        self.count_accum = torch.zeros_like(self.count_accum)

    def densify_and_prune(self, max_grad, max_absgrad, min_geovalue, extent, max_screen_size, percent_dense, use_absgrad=True):
        grads = self.xyz_gradient_accum / self.denom
        absgrads = self.xyz_absgradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        absgrads[absgrads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, percent_dense)
        if use_absgrad:
            self.densify_and_split(absgrads, max_absgrad, extent, percent_dense)
        else:
            self.densify_and_split(grads, max_grad, extent, percent_dense)

        self.prune(min_geovalue, extent, max_screen_size)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        assert viewspace_point_tensor.grad.shape[-1] == 4
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.xyz_absgradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1, keepdim=True)
        self.denom[update_filter] += 1