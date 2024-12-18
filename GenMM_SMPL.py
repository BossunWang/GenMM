import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F

from utils.base import logger
from utils.quaternion import ax_from_6v
from utils.rotation_conversions import axis_angle_to_quaternion
from utils.rotation_conversions import matrix_to_quaternion
from utils.rotation_conversions import quaternion_to_axis_angle
from utils import vis
from utils.vis import SMPLXSkeleton
from utils.vis import skeleton_render

class GenMM:
    def __init__(self, avatar_offset_path, mode = 'random_synthesis', noise_sigma = 1.0, coarse_ratio = 0.2, coarse_ratio_factor = 6, pyr_factor = 0.75, num_stages_limit = -1, device = 'cuda:0', silent = False):
        '''
        GenMM main constructor
        Args:
            device : str = 'cuda:0', default device.
            silent : bool = False, whether to mute the output.
        '''
        self.device = torch.device(device)
        self.silent = silent
        avatar_offset = np.load(avatar_offset_path)
        self.smplx = self.__load_SMPL_module(avatar_offset, True)

    def _get_pyramid_lengths(self, final_len, coarse_ratio, pyr_factor):
        '''
        Get a list of pyramid lengths using given target length and ratio
        '''
        lengths = [int(np.round(final_len * coarse_ratio))]
        while lengths[-1] < final_len:
            lengths.append(int(np.round(lengths[-1] / pyr_factor)))
            if lengths[-1] == lengths[-2]:
                lengths[-1] += 1
        lengths[-1] = final_len

        return lengths

    def _get_target_pyramid(self, target, coarse_ratio, pyr_factor, num_stages_limit=-1):
        '''
        Reads a target motion(s) and create a pyraimd out of it. Ordered in increatorch.sing size
        '''
        self.num_target = len(target)
        lengths = []
        min_len = 10000
        for i in range(len(target)):
            new_length = self._get_pyramid_lengths(target.shape[-1], coarse_ratio, pyr_factor)
            min_len = min(min_len, len(new_length))
            if num_stages_limit != -1:
                new_length = new_length[:num_stages_limit]
            lengths.append(new_length)
        for i in range(len(target)):
            lengths[i] = lengths[i][-min_len:]
        self.pyraimd_lengths = lengths

        target_pyramid = [[] for _ in range(len(lengths[0]))]
        for step in range(len(lengths[0])):
            for i in range(len(target)):
                length = lengths[i][step]
                sample_indices = torch.linspace(0, target[i].shape[-1] - 1, steps=length).long()
                target_pyramid[step].append(target[i, None, :, sample_indices].to(self.device))

        if not self.silent:
            print('Levels:', lengths)
            for i in range(len(target_pyramid)):
                print(f'Number of clips in target pyramid {i} is {len(target_pyramid[i])}, ranging {[[tgt.min(), tgt.max()] for tgt in target_pyramid[i]]}')

        return target_pyramid

    def _get_initial_motion(self, init_length, noise_sigma):
        '''
        Prepare the initial motion for optimization
        '''
        initial_motion = F.interpolate(torch.cat([self.target_pyramid[0][i] for i in range(self.num_target)], dim=-1),
                                       size=init_length, mode='linear', align_corners=True)
        if noise_sigma > 0:
            initial_motion_w_noise = initial_motion + torch.randn_like(initial_motion) * noise_sigma
            initial_motion_w_noise = torch.fmod(initial_motion_w_noise, 1.0)
        else:
            initial_motion_w_noise = initial_motion

        if not self.silent:
            print('Initial motion:', initial_motion.min(), initial_motion.max())
            print('Initial motion with noise:', initial_motion_w_noise.min(), initial_motion_w_noise.max())

        return initial_motion_w_noise

    def __load_SMPL_module(self, avatar_offset, mirror):
        offset = avatar_offset[:len(vis.smplx_parents)].copy()
        offset[1:] -= offset[vis.smplx_parents[1:]]  # relation of parents
        # generated model pose should be mirror, retarget pose don't need
        if mirror:
            offset_mirror = offset.copy()
            for i, key in enumerate(vis.JOINT_NAMES_HYBRIK_MIRROR[:len(vis.smplx_parents)]):
                map_index = vis.JOINT_NAMES_HYBRIK.index(key)
                offset[i, :] = offset_mirror[map_index, :]

        smplx = SMPLXSkeleton(parents=vis.smplx_parents, offsets=offset, device=self.device)
        return smplx

    def __forward_kinematic(self, model_q, model_x, smplx):
        # perform FK
        model_xp = smplx.forward(model_q, model_x)
        pred_pose = model_xp.cpu().detach().numpy()
        model_pose = pred_pose
        model_pose_tmp = model_pose.copy()
        model_pose[:, :, :, 1] = -model_pose_tmp[:, :, :, 2]
        model_pose[:, :, :, 2] = model_pose_tmp[:, :, :, 1]
        return model_pose

    def __post_process(self, x_out, smplx):
        # post-process
        model_contact, model_out = torch.split(
            x_out, (4, x_out.shape[2] - 4), dim=2)

        b, s, c = model_out.shape
        # X, Q
        model_x = model_out[:, :, :3]
        model_q = ax_from_6v(model_out[:, :, 3:].reshape(b, s, -1, 6))
        model_q_from_axis = axis_angle_to_quaternion(model_q)
        model_pose = self.__forward_kinematic(model_q_from_axis, model_x, smplx)

        return model_x, model_q, model_pose

    def run(self, target, criteria, num_frames, num_steps, noise_sigma, patch_size, coarse_ratio, pyr_factor, ext=None, debug_dir=None):
        '''
        generation function
        Args:
            mode             : - string = 'x?', generate x times longer frames results
                             : - int, specifying the number of times to generate
            noise_sigma      : float = 1.0, random noise.
            coarse_ratio     : float = 0.2, ratio at the coarse level.
            pyr_factor       : float = 0.75, pyramid factor.
            num_stages_limit : int = -1, no limit.
        '''
        if debug_dir is not None:
            from tensorboardX import SummaryWriter
            writer = SummaryWriter(log_dir=debug_dir)

        # build target pyramid
        if 'patchsize' in coarse_ratio:
            coarse_ratio = patch_size * float(coarse_ratio.split('x_')[0]) / max([t.shape[-1] for t in target])
        elif 'nframes' in coarse_ratio:
            coarse_ratio = float(coarse_ratio.split('x_')[0])
        else:
            raise ValueError('Unsupported coarse ratio specified')
        self.target_pyramid = self._get_target_pyramid(target, coarse_ratio, pyr_factor)

        # get the initial motion data
        if 'nframes' in num_frames:
            syn_length = int(sum([i[-1] for i in self.pyraimd_lengths]) * float(num_frames.split('x_')[0]))
        elif num_frames.isdigit():
            syn_length = int(num_frames)
        else:
            raise ValueError(f'Unsupported mode {self.mode}')
        self.synthesized_lengths = self._get_pyramid_lengths(syn_length, coarse_ratio, pyr_factor)
        if not self.silent:
            print('Synthesized lengths:', self.synthesized_lengths)
        self.synthesized = self._get_initial_motion(self.synthesized_lengths[0], noise_sigma)

        # perform the optimization
        self.synthesized.requires_grad_(False)
        self.pbar = logger(num_steps, len(self.target_pyramid))
        for lvl, lvl_target in enumerate(self.target_pyramid):
            self.pbar.new_lvl()
            if lvl > 0:
                with torch.no_grad():
                    self.synthesized = F.interpolate(self.synthesized.detach(), size=self.synthesized_lengths[lvl], mode='linear')

            self.synthesized, losses = GenMM.match_and_blend(self.synthesized, lvl_target, criteria, num_steps, self.pbar, ext=ext)

            criteria.clean_cache()
            if debug_dir is not None:
                for itr in range(len(losses)):
                    writer.add_scalar(f'optimize/losses_lvl{lvl}', losses[itr], itr)
        self.pbar.pbar.close()

        syn = self.synthesized.detach().permute(0, 2, 1)
        resample_indices = torch.linspace(0, syn.shape[1] - 1, steps=target.shape[-1]).long()
        syn = syn[:, resample_indices, :]
        target_process = target.permute(0, 2, 1).to(self.device)
        source_x, source_q, source_pose = self.__post_process(target_process, self.smplx)
        syn_x, syn_q, syn_pose = self.__post_process(syn, self.smplx)

        return {"syn_x": syn_x,
                "syn_q": syn_q,
                "syn_pose": syn_pose,
                "source_x": source_x,
                "source_q": source_q,
                "source_pose": source_pose
                }

    def visualize(self, model_pose, music_file_path, out_path, FPS):
        skeleton_render(
            model_pose,
            out=out_path,
            name=music_file_path,
            sound=True,
            stitch=False,
            render=True,
            FPS=FPS,
            zcenter=0.0,
            parents=vis.smplx_parents,
        )


    @staticmethod
    @torch.no_grad()
    def match_and_blend(synthesized, targets, criteria, n_steps, pbar, ext=None):
        '''
        Minimizes criteria bewteen synthesized and target
        Args:
            synthesized    : torch.Tensor, optimized motion data
            targets        : torch.Tensor, target motion data
            criteria       : optimmize target function
            n_steps        : int, number of steps to optimize
            pbar           : logger
            ext            : extra configurations or constraints (optional)
        '''
        losses = []
        for _i in range(n_steps):
            synthesized, loss = criteria(synthesized, targets, ext=ext, return_blended_results=True)

            # Update staus
            losses.append(loss.item())
            pbar.step()
            pbar.print()

        return synthesized, losses

