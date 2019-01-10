
#!/bin/bash
SCENE_PATH=/home/xiaojwan/thesis/experiment/octree_primal_dual/data/SUNCG_DAT/
OUTPUT_PATH=/media/root/data/SUNCG_OCT/

while IFS='' read -r line || [[ -n "$line" ]]; 
do
    echo "Convert the data cost of scene: $line"
    # mkdir -p $OUTPUT_PATH/$line
    ../build/tools/suncg_groundtruth_octree_converter -i $SCENE_PATH/$line/groundtruth.dat -o $OUTPUT_PATH/$line/groundtruth --min_level 5 --max_level 7 

done < "$1"





