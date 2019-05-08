import os
import struct
import argparse
import numpy as np
import sys
from mayavi import mlab
from tvtk.util.ctf import ColorTransferFunction, PiecewiseFunction

def read_gvr_groundtruth(path):
    with open(path, "rb") as fid:
        version = np.fromfile(fid, count=1, dtype=np.uint8)
        assert version == 1

        is_big_endian = np.fromfile(fid, count=1, dtype=np.uint8)
        assert (is_big_endian == 1 and sys.byteorder == "big") or \
               (is_big_endian == 0 and sys.byteorder == "little")

        uint_size = np.fromfile(fid, count=1, dtype=np.uint8)
        assert uint_size == 4

        elem_size = np.fromfile(fid, count=1, dtype=np.uint32)
        if elem_size == 4:
            dtype = np.float32
        elif elem_size == 8:
            dtype = np.float64
        else:
            raise ValueError("Unsupported data type")

        num_labels = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        height = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        width = np.fromfile(fid, count=1, dtype=np.uint32)[0]
        depth = np.fromfile(fid, count=1, dtype=np.uint32)[0]

        num_elems = width * height * depth * num_labels
        assert num_elems > 0

        grid = np.fromfile(fid, count=num_elems, dtype=dtype)
        grid =  grid.reshape(num_labels, height, width, depth)

        grid = grid.transpose(1, 2, 3, 0)

        return grid

def visualize_occupancy(occupancy, label_colors,
                        show_unknown_space=False, show_free_space=False):

    for label, rgb in label_colors.items():
        if not show_unknown_space and label == len(label_colors) - 2:
            continue
        if not show_free_space and label == len(label_colors) - 1:
            continue
        if label == len(label_colors) - 1:
            rgb = (255, 255, 255)
            opacity = 0.005
        else:
            opacity = 1
        xx, yy, zz = np.where(occupancy == label)
        mesh = mlab.points3d(xx, yy, zz, mode="cube",
                             color=tuple(c / 255.0 for c in rgb),
                             opacity=opacity, scale_factor=1)

        # mesh.scene.light_manager.light_mode = "vtk"

        # mesh.actor.property.interpolation = 'phong'
        # mesh.actor.property.specular = 0.1
        # mesh.actor.property.specular_power = 5

        # break

    # if show_unknown_space:
    #     xx, yy, zz = np.where(occupancy == -1)
    #     mlab.points3d(xx, yy, zz, mode="cube", color=(0, 0, 0),
    #                   scale_factor=1, opacity=0.005)

    # if show_free_space:
    #     xx, yy, zz = np.where(occupancy == -2)
    #     mlab.points3d(xx, yy, zz, mode="cube", color=(1, 1, 1),
    #                   scale_factor=1, opacity=0.01)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--screenshot_path", required=True)
    parser.add_argument("--label_map_path", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.path.endswith('.dat'):
        probs = read_gvr_groundtruth(args.path)
    else:
        probs = np.load(args.path)["volume"]

    occupancy = np.argmax(probs, axis=-1)
    occupancy[np.max(probs, axis=-1) < 0.5] = probs.shape[-1] - 1

    label_names = {}
    label_colors = {}
    with open(args.label_map_path, "r") as fid:
        for line in fid:
            line = line.strip()
            if not line:
                continue
            label = int(line.split(":")[0].split()[0])
            name = line.split(":")[0].split()[1]
            color = tuple(map(int, line.split(":")[1].split()))
            label_names[label] = name
            label_colors[label] = color

    mlab.figure(size=(1600, 1200), bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))
    visualize_occupancy(occupancy, label_colors, False, False)
    mlab.view(elevation=20)
    if args.screenshot_path:
        mlab.savefig(args.screenshot_path)
    # mlab.show()


if __name__ == "__main__":
    main()
