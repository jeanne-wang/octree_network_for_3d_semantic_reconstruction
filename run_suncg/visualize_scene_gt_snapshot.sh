SCANNET_EVAL_PATH=/home/xiaojwan/thesis/experiment/octree_primal_dual/data/SUNCG_DAT
SCREENSHOT_PATH=/media/root/data/suncg_val_screenshot

while IFS='' read -r line || [[ -n "$line" ]]; 
do
	echo $line

    python3 visualize_scene_gt_snapshot.py --path $SCANNET_EVAL_PATH/$line/groundtruth.dat \
        --screenshot_path $SCREENSHOT_PATH/$line"_gt.png" \
        --label_map_path ./labels.txt

done < "$1"
