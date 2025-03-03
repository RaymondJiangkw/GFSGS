import os
from argparse import ArgumentParser

mvs_scenes = ['EvaUnit', 'Excavator', 'Museum', 'Temple']
override_args_set = {
    'sh': {
        
    }, 
    'latent': {
        
    }
}
custom_args_set = {
    'sh': {
        
    }, 
    'latent': {
        
    }
}

render_custom_args_set = {
    'sh': {
        
    }, 
    'latent': {
        
    }
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/mvs_scene")
parser.add_argument('--mvs', "-mvs", required=True, type=str)
parser.add_argument("--color_mode", default="sh", type=str)
args = parser.parse_args()

all_scenes = []
all_scenes.extend(mvs_scenes)

if args.color_mode == 'sh':
    args.output_path += f'_sh'
elif args.color_mode == 'latent':
    args.output_path += f'_latent'

if not args.skip_training:
    common_args = f" --quiet --color_mode {args.color_mode}"
    
    for scene in mvs_scenes:
        source = args.mvs + "/" + scene
        override_args = " --lambda_dist 10"
        if scene in override_args_set[args.color_mode]:
            override_args = override_args_set[args.color_mode][scene]
        custom_args = ""
        if scene in custom_args_set[args.color_mode]:
            custom_args = custom_args_set[args.color_mode][scene]
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + override_args + custom_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + override_args + custom_args)


if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --skip_train --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 10. --nop 10000000" # Increase the amount of sampled points
    if args.skip_metrics:
        common_args += " --skip_metrics"
    for scene in mvs_scenes:
        source = args.mvs + "/" + scene
        custom_args = ""
        if scene in render_custom_args_set[args.color_mode]:
            custom_args = render_custom_args_set[args.color_mode][scene]
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args + custom_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args + custom_args)