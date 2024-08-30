import argparse
import os

from .frames_utils import save_frames
from .metadata_utils import calc_meta, load_colmap, save_meta


parser = argparse.ArgumentParser()


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--colmap_dir",
        type=str,
        help="Root directory of COLMAP project directory, which contains `sparse/0`.",
    )

    parser.add_argument(
        "--frames_dir",
        type=str,
        help="Root directory of video frames",
    )

    parser.add_argument(
        "--scale", default=1, type=int, help="Downscaling factor for images."
    )

    parser.add_argument(
        "--dir_dst",
        default="data/custom",
        type=str,
        help="Destination directory for converted dataset.",
    )

    parser.add_argument(
        "--split_nth",
        default=0,
        type=int,
        help="select every n-th frame as validation and every other n-th frame as test frame.",
    )
    parser.add_argument(
        "--c2w",
        default=1,
        type=int,
        help="True for NeuralDiff/ Rendering",
    )
    parser.add_argument(
        "--to_OpenGL",
        default=0,
        type=int,
        help="True for NeuralDiff; otherwise False-> right down front",
    )
    parser.add_argument(
        "--to_PT3d",
        default=0,
        type=int,
        help="Convert from opencv/colmap to pt3d",
    )
    parser.add_argument(
        "--save_frames",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--undistort",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    return args


def run(args):
    colmap_model_dir = args.colmap_dir
    colmap = load_colmap(colmap_model_dir)
    meta = calc_meta(
        colmap,
        image_downscale=args.scale,
        split_nth=args.split_nth,
        c2w=args.c2w,
        to_OpenGL=args.to_OpenGL,
        to_PT3d=args.to_PT3d,
    )

    # dataset_id = os.path.split(os.path.normpath(args.colmap_dir))[1]
    dataset_dir = args.dir_dst
    os.makedirs(args.dir_dst, exist_ok=True)
    save_meta(dataset_dir, meta)

    # SAVE FRAMES
    if args.save_frames:
        # frames_dir_src = os.path.join(args.frames_dir, 'images')
        print(f"saving images with undistort: {args.undistort}")
        frames_dir_src = args.frames_dir
        frames_dir_dst = os.path.join(dataset_dir, "frames")
        os.makedirs(frames_dir_dst, exist_ok=True)
        save_frames(frames_dir_src, frames_dir_dst, meta, undistort=args.undistort)


if __name__ == "__main__":
    args = parse_args()
    run(args)
