# ../build/tools/caffe test --model test_prop_pred.prototxt \
# --weights=snapshots/prop_known_crop_32_lr_0_0001_batch_4_niter_50_10_10_conv_3/snapshot_iter_100000.caffemodel \
# --gpu 0 \
# --iterations 100 #2>&1| tee crop_lr_0_0001_batch_4_niter_50_10_10_conv_3_100000_iters_prop_pred_split_accu

../build/tools/caffe test --model test_dense.prototxt \
--weights=../run_scannet/snapshots/dense_cudnn_lr_0_00001_batch_5/snapshot_iter_50000.caffemodel \
--gpu 0 \
--iterations 1 #2>&1| tee crop_lr_0_0001_batch_4_niter_50_10_10_conv_3_100000_iters_prop_pred_split_accu
