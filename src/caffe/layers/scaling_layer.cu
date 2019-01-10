#include <algorithm>
#include <vector>

#include "caffe/layers/scaling_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScalingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

  caffe_copy<Dtype>(count, bottom_data, top_data);
  caffe_gpu_scal<Dtype>(count, Dtype(scale_), top_data);
 
}

template <typename Dtype>
void ScalingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {

    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    caffe_copy<Dtype>(count, top_diff, bottom_diff);
    caffe_gpu_scal<Dtype>(count, Dtype(scale_), bottom_diff);

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScalingLayer);

}  // namespace caffe
