SCANNET_EVAL_PATH=/home/xiaojwan/thesis/experiment/octree_primal_dual/data/SUNCG_OCT
SCREENSHOT_PATH=/media/root/data/suncg_val_screenshot

for scene_path in $(ls -d $SCANNET_EVAL_PATH/*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    python3 visualize_scene_snapshot.py --path $scene_path/pred_probs_ian.npz \
        --screenshot_path $SCREENSHOT_PATH/$scene_name"_ian.png" \
        --label_map_path ./labels.txt

done
