import os
import os.path as osp
import argparse
from GenMM_SMPL import GenMM
from nearest_neighbor.losses import PatchCoherentLoss
import pickle
from utils.base import ConfigParser, set_seed

args = argparse.ArgumentParser(
    description='Random shuffle the input motion sequence')
args.add_argument('-m', '--mode', default='run',
                  choices=['run', 'eval', 'debug'], type=str, help='current run mode.')
args.add_argument('-i', '--input', required=True,
                  type=str, help='exemplar motion path.')
args.add_argument('-o', '--output_dir', default='./output',
                  type=str, help='output folder path for saving results.')
args.add_argument('-c', '--config', default='./configs/default.yaml',
                  type=str, help='config file path.')
args.add_argument('-s', '--seed', default=None,
                  type=int, help='random seed used.')
args.add_argument('-d', '--device', default="cuda:0",
                  type=str, help='device to use.')
args.add_argument('--post_precess', action='store_true',
                  help='whether to use IK post-process to fix foot contact.')

# Use argsparser to overwrite the configuration
# for dataset
args.add_argument('--skeleton_name', type=str,
                  help='(used when joint_reduction==True or contact==True) skeleton name to load pre-defined joints configuration.')
args.add_argument('--use_velo', type=int,
                  help='whether to use velocity rather than global position of each joint.')
args.add_argument('--repr', choices=['repr6d', 'quat', 'euler'], type=str,
                  help='rotation representation, support [epr6d, quat, reuler].')
args.add_argument('--requires_contact', type=int,
                  help='whether to use contact label.')
args.add_argument('--keep_up_pos', type=int,
                  help='whether to do not use velocity and keep the y(up) position.')
args.add_argument('--up_axis', type=str, choices=['X_UP', 'Y_UP', 'Z_UP'],
                  help='up axis of the motion.')
args.add_argument('--padding_last', type=int,
                  help='whether to pad the last position channel to match the rotation dimension.')
args.add_argument('--joint_reduction', type=int,
                  help='whether to simplify the skeleton using provided skeleton config.')
args.add_argument('--skeleton_aware', type=int,
                  help='whether to enable skeleton-aware component.')
args.add_argument('--joints_group', type=str,
                  help='joints spliting group for using skeleton-aware component.')
# for synthesis
args.add_argument('--num_frames', type=str, 
                  help='number of synthesized frames, supported Nx(N times) and int input.')
args.add_argument('--alpha', type=float,
                  help='completeness/diversity trade-off alpha value.')
args.add_argument('--num_steps', type=int,
                  help='number of optimization steps at each pyramid level.')
args.add_argument('--noise_sigma', type=float,
                  help='standard deviation of the zero mean normal noise added to the initialization.')
args.add_argument('--coarse_ratio', type=float,
                  help='downscale ratio of the coarse level.')
args.add_argument('--coarse_ratio_factor', type=float,
                  help='downscale ratio of the coarse level.')
args.add_argument('--pyr_factor', type=float,
                  help='upsample ratio of each pyramid level.')
args.add_argument('--num_stages_limit', type=int,
                  help='limit of the number of stages.')
args.add_argument('--patch_size', type=int, help='patch size for generation.')
args.add_argument('--loop', type=int, help='whether to loop the sequence.')
cfg = ConfigParser(args)


def generate(cfg):
    # seet seed for reproducible
    set_seed(cfg.seed)
    base_dir = osp.join(
        cfg.output_dir, cfg.input.split('/')[-1].split('.')[0])

    motion_ref_slice_dict = pickle.load(open(cfg.input, "rb"))
    motion_data = motion_ref_slice_dict["pose"].permute(0, 2, 1)
    prefix = f"s{cfg.seed}+{cfg.num_frames}+{cfg.repr}+use_velo_{cfg.use_velo}+kypose_{cfg.keep_up_pos}+padding_{cfg.padding_last}" \
             f"+contact_{cfg.requires_contact}+jredu_{cfg.joint_reduction}+n{cfg.noise_sigma}+pyr{cfg.pyr_factor}" \
             f"+r{cfg.coarse_ratio}_{cfg.coarse_ratio_factor}+itr{cfg.num_steps}+ps_{cfg.patch_size}+alpha_{cfg.alpha}" \
             f"+loop_{cfg.loop}"

    save_dir = osp.join(base_dir, prefix)
    os.makedirs(save_dir, exist_ok=True)

    # perform the generation
    avatar_offset_path = "data/avatar_setting/erin_global_pos.npy"
    model = GenMM(avatar_offset_path, device=cfg.device, silent=True if cfg.mode == 'eval' else False)
    criteria = PatchCoherentLoss(patch_size=cfg.patch_size, alpha=cfg.alpha, loop=cfg.loop, cache=True)
    data_dict = model.run(motion_data, criteria,
                                        num_frames=cfg.num_frames,
                                        num_steps=cfg.num_steps,
                                        noise_sigma=cfg.noise_sigma,
                                        patch_size=cfg.patch_size,
                                        coarse_ratio=cfg.coarse_ratio,
                                        pyr_factor=cfg.pyr_factor,
                                        debug_dir=save_dir if cfg.mode == 'debug' else None)
    source_pose = data_dict["source_pose"]
    syn_pose = data_dict["syn_pose"]
    music_path = "data/audio/DanceCover3107_slice05.wav"
    render_source_path = os.path.join(base_dir, "render_source")
    render_syn_path = os.path.join(base_dir, "render_syn")
    model.visualize(source_pose[0], music_path, render_source_path, 30)
    model.visualize(syn_pose[0], music_path, render_syn_path, 30)
    
    # save the generated results
    # motion_data[0].write(f"{save_dir}/syn.bvh", syn)

    if cfg.post_precess:
        cmd = f"python fix_contact.py --prefix {osp.abspath(save_dir)} --name syn --skeleton_name={cfg.skeleton_name}"
        os.system(cmd)

if __name__ == '__main__':
    generate(cfg)
