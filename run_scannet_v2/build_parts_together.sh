SCANNET_EVAL_PATH=/media/root/data/scans_test

for scene_path in $(ls -d $SCANNET_EVAL_PATH/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    python build_parts_together.py --test_scene $scene_path \
        --num_classes 42 \
        --freespace_label 41

done
