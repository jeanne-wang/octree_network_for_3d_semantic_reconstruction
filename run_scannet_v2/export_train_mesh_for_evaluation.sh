SCANNET_EVAL_PATH=/media/root/data/scans_val
OUTPUT_PATH=/media/root/data/scans_val_gt
for scene_path in $(ls -d $SCANNET_EVAL_PATH/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    python BenchmarkScripts/export_train_mesh_for_evaluation.py \
    	--type 'label' \
    	--scan_path $scene_path \
    	--label_map_file scannetv2-labels.combined.tsv \
    	--output_file $OUTPUT_PATH/$scene_name'.txt'
  

done
