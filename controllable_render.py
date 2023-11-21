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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera

def normalize(x):
    return x / x.norm(dim=-1, keepdim=True)

def look_at(eye, at, up):
    z = normalize(eye - at)
    x = normalize(torch.linalg.cross(up, z))
    y = normalize(torch.linalg.cross(z, x))
    rot = torch.stack([x, y, z]).T
    trans = torch.tensor([x @ eye, y @ eye, z @ eye])
    return rot, trans


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "control_{}".format(iteration), "renders")


    makedirs(render_path, exist_ok=True)
    gt_img0 = views[0].original_image[0:3, :, :]

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        camera_extent = 5
        rand_pos = torch.tensor([0.0, 0.0, camera_extent])
        angle = 3.14159 * 2.0 * idx / len(views)
        rand_pos = camera_extent * torch.tensor([torch.tensor(angle).cos(), 0.0, torch.tensor(angle).sin()])

        rot, trans = look_at(rand_pos, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
        viewpoint_cam = Camera(
            colmap_id=0,
            R=rot.numpy(), #rand_angle.numpy(),
            T=trans.numpy(), #torch.zeros((3,)).numpy(),
            FoVx=90.0, 
            FoVy=90.0, 
            image=torch.zeros((3, gt_img0.shape[1], gt_img0.shape[2])),
            gt_alpha_mask=torch.ones((gt_img0.shape[1], gt_img0.shape[2])),
            image_name="control",
            uid=0,
        )

        rendering = render(viewpoint_cam, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)