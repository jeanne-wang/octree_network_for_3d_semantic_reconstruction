import os
import glob
import shutil
import argparse
import tempfile
import numpy as np
import plyfile
from skimage.measure import marching_cubes_lewiner
import tensorflow as tf
import sys
import read
def read_binary(path):
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
        if num_labels == 1:
            grid =  grid.reshape(height, width, depth)
        else:
            grid =  grid.reshape(num_labels, height, width, depth)
            grid = grid.transpose(1,2,3,0)

        return grid

def mkdir_if_not_exists(path):
    assert os.path.exists(os.path.dirname(path.rstrip("/")))
    if not os.path.exists(path):
        os.makedirs(path)


def extract_mesh_marching_cubes(path, volume, color=None, level=0.5,
                                step_size=1.0, gradient_direction="ascent"):
    if level > volume.max() or level < volume.min():
        return

    verts, faces, normals, values = marching_cubes_lewiner(
        volume, level=level, step_size=step_size,
        gradient_direction=gradient_direction)

    ply_verts = np.empty(len(verts),
                         dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    ply_verts["x"] = verts[:, 0]
    ply_verts["y"] = verts[:, 1]
    ply_verts["z"] = verts[:, 2]
    ply_verts = plyfile.PlyElement.describe(ply_verts, "vertex")

    if color is None:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,))])
    else:
        ply_faces = np.empty(
            len(faces), dtype=[("vertex_indices", "i4", (3,)),
                               ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        ply_faces["red"] = color[0]
        ply_faces["green"] = color[1]
        ply_faces["blue"] = color[2]
    ply_faces["vertex_indices"] = faces
    ply_faces = plyfile.PlyElement.describe(ply_faces, "face")

    with tempfile.NamedTemporaryFile(dir=".", delete=False) as tmpfile:
        plyfile.PlyData([ply_verts, ply_faces]).write(tmpfile.name)
        shutil.move(tmpfile.name, path)

def parse_args():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--datacost_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--crop", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=42)
    parser.add_argument("--label_map_path", type=str, default='labels.txt')
 

    return parser.parse_args()


def main():

    args = parse_args()
    mkdir_if_not_exists(args.output_path)

    if args.crop == 'true':
        datacost = read_binary(args.datacost_path)
    else:
        datacost = read.read_gvr_datacost(args.datacost_path, args.num_classes)

    datacost_label = np.argmin(datacost, axis=3)

    print datacost_label.shape
    print np.unique(datacost_label)

    x = tf.placeholder(tf.int32, shape=datacost_label.shape)
    y = tf.one_hot(x, args.num_classes)

    init = tf.initialize_all_variables()
        
    with tf.Session() as sess:
        probs = sess.run(y, feed_dict={x: datacost_label.astype('int32')})
    
    print probs.shape[3]
    
    if args.label_map_path:
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


    for label in range(probs.shape[-1]):
        if args.label_map_path:
            path = os.path.join(args.output_path,
                                "{}-{}.ply".format(label, label_names[label]))
            color = label_colors[label]
        else:
            path = os.path.join(args.output_path, "{}.ply".format(label))
            color = None

        extract_mesh_marching_cubes(path, probs[..., label], color=color)


if __name__ == "__main__":
    main()
