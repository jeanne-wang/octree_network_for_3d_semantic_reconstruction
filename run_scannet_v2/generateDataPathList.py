import os
import glob
import numpy as np

   

def main(read_filename, write_filename):

	d = '/media/root/data/ScanNet_DAT/'
	scenes= [line.rstrip('\n') for line in open(read_filename,'r')]
	print(len(scenes))


	valid_datacost_list = []
	valid_gt_list = []

	count = 0
	for scene in scenes:

		datacost_path = os.path.join(d, scene, "pred_datacost.dat")
		groundtruth_path = os.path.join(d, scene, "groundtruth.dat")
		
		if not os.path.exists(datacost_path) or not os.path.exists(groundtruth_path):
			print("  Warning: datacost or groundtruth does not exist")
			print(scene)
			continue

		valid_datacost_list.append(datacost_path)
		valid_gt_list.append(groundtruth_path)

		count =  count+1


	print("There are totally {} valid scenes.".format(count))

	with open(write_filename+'_datacost.txt','w') as fid:
		for datacost_path in valid_datacost_list:
			fid.write(datacost_path)
			fid.write('\n')

	with open(write_filename+'_groundtruth.txt','w') as fid:
		for gt_path in valid_gt_list:
			fid.write(gt_path)
			fid.write('\n')










if __name__ == '__main__':
	main('scannetv2_val.txt', './val')
