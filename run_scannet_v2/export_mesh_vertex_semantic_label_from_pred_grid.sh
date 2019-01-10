SCANNET_EVAL_PATH=/media/root/data/scans_test
OUTPUT_PATH=/media/root/data/scans_test_pred_datacost_prop_pred
for scene_path in $(ls -d $SCANNET_EVAL_PATH/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name

    python BenchmarkScripts/export_semantic_label_grid_for_evaluation.py \
    	--grid_file $scene_path/pred.npy \
    	--bbox_file $scene_path/bbox.txt \
    	--mesh_file $scene_path/$scene_name"_vh_clean_2.ply" \
    	--voxel_resolution 0.05 \
    	--output_file $OUTPUT_PATH/$scene_name'.txt'
  

done
