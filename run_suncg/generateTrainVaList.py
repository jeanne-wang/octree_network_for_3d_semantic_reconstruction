import os
import glob
import numpy as np
from random import shuffle
   
def main(num_train, num_test):

	d = '/home/xiaojwan/thesis/experiment/octree_primal_dual/data/SUNCG_DAT/'
	scenes = [os.path.join(d, o) for o in os.listdir(d) 
				if os.path.isdir(os.path.join(d,o))]

	print(len(scenes))
	shuffle(scenes)
	print(len(scenes))

	train_scenes = scenes[:num_train]
	test_scenes = scenes[num_train:num_train+num_test]

	train_datacost_list = []
	train_gt_list = []
	for scene in train_scenes:
		datacost_path = os.path.join(scene, "datacost.dat")
		groundtruth_path = os.path.join(scene, "groundtruth.dat")
		
		if not os.path.exists(datacost_path) or not os.path.exists(groundtruth_path):
			print("  Warning: datacost or groundtruth does not exist")
			continue

		train_datacost_list.append(datacost_path)
		train_gt_list.append(groundtruth_path)

	with open('./train_datacost.txt','w') as fid:
		for datacost_path in train_datacost_list:
			fid.write(datacost_path)
			fid.write('\n')

	with open('./train_groundtruth.txt','w') as fid:
		for gt_path in train_gt_list:
			fid.write(gt_path)
			fid.write('\n')


	test_datacost_list = []
	test_gt_list = []
	for scene in test_scenes:
		datacost_path = os.path.join(scene, "datacost.dat")
		groundtruth_path = os.path.join(scene, "groundtruth.dat")
		
		if not os.path.exists(datacost_path) or not os.path.exists(groundtruth_path):
			print("  Warning: datacost or groundtruth does not exist")
			continue

		test_datacost_list.append(datacost_path)
		test_gt_list.append(groundtruth_path)

	with open('./test_datacost.txt','w') as fid:
		for datacost_path in test_datacost_list:
			fid.write(datacost_path)
			fid.write('\n')

	with open('./test_groundtruth.txt','w') as fid:
		for gt_path in test_gt_list:
			fid.write(gt_path)
			fid.write('\n')



if __name__ == '__main__':
	main(400, 100)