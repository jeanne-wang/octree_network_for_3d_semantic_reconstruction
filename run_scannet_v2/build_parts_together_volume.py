import os
import glob
import argparse
import numpy as np

import re
import struct
import sys
import fnmatch
def read_pred_probs(path):
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_scene", required=True)
    parser.add_argument("--num_classes", type=int, default=42)
    parser.add_argument("--freespace_label", type=int, default=41)
    parser.add_argument("--unknown_label", type=int, default=40)
    return parser.parse_args()


def main():

    args = parse_args()
    test_scene = args.test_scene
    num_classes = args.num_classes

    with open(os.path.join(test_scene, 'info.txt')) as f:
        height, width, depth = [int(x) for x in next(f).split()] # read first line
    volume = np.zeros((height, width, depth, num_classes), dtype=np.float32)

    part_files = fnmatch.filter(os.listdir(test_scene), 'probs*.dat')
    assert(len(part_files) <=10)
    for part_file in part_files:
        
        part_info_file = 'info'+ part_file[-5]+'.txt'
        crop = []
        with open(os.path.join(test_scene,part_info_file)) as f:
            for line in f: # read rest of lines
                crop.append([int(x) for x in line.split()])

        crop = np.asarray(crop)
        part_volume = read_pred_probs(os.path.join(test_scene, part_file))

        hstart = crop[0,0]
        wstart = crop[1,0]
        dstart = crop[2,0]

        hend = crop[0,1]
        wend = crop[1,1]
        dend = crop[2,1]

        volume[hstart:hend, wstart:wend, dstart:dend,:]=part_volume[:hend-hstart, :wend-wstart, :dend-dstart,:]

    


    if os.path.exists(os.path.join(test_scene, 'pred_probs.npz')):
        os.remove(os.path.join(test_scene, 'pred_probs.npz'))
    
    np.savez(os.path.join(test_scene, 'pred_probs.npz'),
             volume=volume)

if __name__ == "__main__":
    main()
