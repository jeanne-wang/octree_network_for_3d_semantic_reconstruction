SCANNET_PATH=/home/xiaojwan/thesis/experiment/octree_primal_dual/data/SUNCG_OCT
for scene_path in $(ls -d $SCANNET_PATH/*)
do

    scene_name=$(basename $scene_path)
    echo $scene_name
    #mv $scene_path/converted/*  $scene_path/
    rm $scene_path/pred_probs_ian.npz
  
    #rm $scene_path/info*.txt
    #rm $scene_path/datacost*.ot
    #rm  $scene_path/pred_datacost.dat

done



