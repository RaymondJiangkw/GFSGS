import os
from argparse import ArgumentParser

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
custom_args_set = {
    'sh': {
        'scan110': ' --geovalue_cull 0.5755' # inverse_footprint_activation(0.005)
    }, 
    'latent': {
        'scan24': ' --ndc_start_iteration 500'
    }
}

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/dtu")
parser.add_argument('--dtu', "-dtu", required=True, type=str)
parser.add_argument("--color_mode", default="sh", type=str)
parser.add_argument("--extra_commands", default="", type=str)
args = parser.parse_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

common_args = f" --quiet -r 2 --lambda_dist 1000 --color_mode {args.color_mode}"
common_args += args.extra_commands

if args.color_mode == 'sh':
    args.output_path += f'_sh'
elif args.color_mode == 'latent':
    args.output_path += f'_latent'

if not args.skip_training:
    
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        custom_args = ""
        if scene in custom_args_set[args.color_mode]:
            custom_args = custom_args_set[args.color_mode][scene]
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + custom_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + custom_args)


if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --skip_train --num_cluster 1 --voxel_size 0.002 --sdf_trunc 0.008 --depth_trunc 5.0"
    if args.skip_metrics:
        common_args += " --skip_metrics"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)