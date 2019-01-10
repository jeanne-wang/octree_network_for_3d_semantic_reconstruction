import os
import glob
import numpy as np

   

def main(read_filename, write_filename):

	scene_paths = [line.rstrip('\n') for line in open(read_filename,'r')]
	print(len(scene_paths))

	scenes = []
	for scene_path in scene_paths:
		folders = scene_path.split('/')
		scenes.append(folders[-2])



	with open(write_filename, 'w') as fid:
		for scene in scenes:
			fid.write(scene)
			fid.write('\n')

	











if __name__ == '__main__':
	main('./val_datacost.txt', './val.txt')
