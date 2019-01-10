../build/tools/caffe train --solver=solver.prototxt 2>&1| tee logs/lr_0_0001_batch_4_niter_50_10_10_conv_3_pred_datacost-`date +%F_%R`.log
