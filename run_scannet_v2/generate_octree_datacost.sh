
#!/bin/bash
SCENE_PATH=/media/root/data/scans_val

for scene_path in $(ls -d $SCENE_PATH/scene*)
do
    scene_name=$(basename $scene_path)
    echo $scene_name
  
    ../build/tools/scannet_datacost_octree_converter -i $scene_path/datacost.dat -o $scene_path/datacost --min_level 5 --max_level 7 --num_classes 42

done





