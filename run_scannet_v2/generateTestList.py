import os
import glob
import numpy as np
from random import shuffle
   
def main(generate_gt):

	d = '/media/root/data/scans_val/'
	scenes = sorted(glob.glob(os.path.join(d, 'scene*')))

	print(len(scenes))
	


	test_list = []
	for scene in scenes:
		datacost_paths = sorted(glob.glob(os.path.join(scene,'datacost*.ot')))
		
		if len(datacost_paths) == 1:
			test_list.append(datacost_paths[0])
		else:
			assert(len(datacost_paths) >1)
			test_list.extend(datacost_paths)
	

	with open('./val_datacost_oct.txt','w') as fid:
		for datacost_path in test_list:
			fid.write(datacost_path)
			fid.write('\n')

	if generate_gt:
		test_gt_list = []
		for scene in scenes:
			gt_paths = sorted(glob.glob(os.path.join(scene,'groundtruth*.ot')))
		
			if len(gt_paths) == 1:
				test_gt_list.append(gt_paths[0])
			else:
				assert(len(gt_paths) > 1)
				test_gt_list.extend(gt_paths)

		with open('./val_groundtruth_oct.txt','w') as fid:
			for gt_path in test_gt_list:
				fid.write(gt_path)
				fid.write('\n')




if __name__ == '__main__':
	generate_gt = True
	main(generate_gt)
