# octree primal dual network
This repo contains the code for my master thesis "learn semantic 3D reconstruction on octree"

# Build
This code is build upon the caffe framework, and thus you can build it using CMake the same way as in caffe. 
mkdir build && cd build
make all
make install

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
