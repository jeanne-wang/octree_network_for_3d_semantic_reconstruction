../build/tools/caffe train --solver=solver.prototxt 2>&1| tee logs/prop_known_crop_32_lr_0_0001_batch_4_niter_50_10_10_conv_3-`date +%F_%R`.log
