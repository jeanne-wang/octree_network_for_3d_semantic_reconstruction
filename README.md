# Octree Primal Dual network
This repo contains the code for my master thesis "learn semantic 3D reconstruction on octree". It includes the code for conv/deconv on octree in include/ and src/, as well as other layers for primal dual update. It also contains the code for octree/dense voxel grid conversion in tools/.

# Build
This code is built upon the caffe framework, and thus you can build it using CMake the same way as in caffe. 
1. mkdir build && cd build
2. make all
3. make install

# Training
run_scannet_v2 and run_suncg provides two examples on how to use the code to do training and testing.
1. Generate the voxelized TSDFs, i.e., datacost, and groundtruth for each scene
2. run generate_train_crop_proto.sh to obtain the prototxt file for training. 
3. run train.sh to start traing. (The solver.prototxt and generate_train_crop_proto.py contains default param configs)

# Testing
1. Generate the voxelized datacosts for each test scene
2. run genetate_octree_datacost.sh to obtain the encoded octree datacost for test input
3. run generate_test_proto.sh to obtain the test protoxt file
4. run test.sh to do inference, and this will generate the resulting 3d reconstruction model.
