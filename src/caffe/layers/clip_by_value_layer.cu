#include <algorithm>
#include <vector>

#include "caffe/layers/clip_by_value_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ClipForward(const int n, const Dtype* in, Dtype* out, const float min_val, const float max_val) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] > min_val ? in[index] : min_val;
    out[index] = out[index] < max_val ? out[index]: max_val;
  }
}

template <typename Dtype>
void ClipByValueLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  for(int n = 0; n < bottom.size(); n++){
    const Dtype* bottom_data = bottom[n]->gpu_data();
    Dtype* top_data = top[n]->mutable_gpu_data();
    const int count = bottom[n]->count();
    ClipForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, min_val_, max_val_);
    CUDA_POST_KERNEL_CHECK;
  } 
}

template <typename Dtype>
__global__ void ClipBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, const float min_val, const float max_val) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > min_val) && (in_data[index] < max_val));
  }
}

template <typename Dtype>
void ClipByValueLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  for(int n = 0; n < bottom.size(); n++){
    if (propagate_down[n]) {
      const Dtype* bottom_data = bottom[n]->gpu_data();
      const Dtype* top_diff = top[n]->gpu_diff();
      Dtype* bottom_diff = bottom[n]->mutable_gpu_diff();
      const int count = bottom[n]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      ClipBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, min_val_, max_val_);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ClipByValueLayer);


}  // namespace caffe
