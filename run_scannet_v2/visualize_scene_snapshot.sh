SCANNET_EVAL_PATH=/media/root/data/scans_val
SCREENSHOT_PATH=/media/root/data/scans_val_screenshot

for scene_path in $(ls -d $SCANNET_EVAL_PATH/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    python3 visualize_scene_snapshot.py --path $scene_path/groundtruth.dat \
        --screenshot_path $SCREENSHOT_PATH/$scene_name"_gt.png" \
        --label_map_path /media/root/data/labels.txt

done
