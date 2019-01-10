#include <algorithm>
#include <vector>

#include "caffe/layers/clip_by_value_layer.hpp"

namespace caffe {

template <typename Dtype>
void ClipByValueLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  min_val_ = this->layer_param_.clip_by_value_param().min_val();
  max_val_ = this->layer_param_.clip_by_value_param().max_val();

}

template <typename Dtype>
void ClipByValueLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  for(int n = 0; n < bottom.size(); n++){
    top[n]->ReshapeLike(*bottom[n]);
  }

}
template <typename Dtype>
void ClipByValueLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  for(int n = 0; n < bottom.size(); n++){
    const Dtype* bottom_data = bottom[n]->cpu_data();
    Dtype* top_data = top[n]->mutable_cpu_data();
    const int count = bottom[n]->count();
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::min(std::max(bottom_data[i], Dtype(min_val_)), Dtype(max_val_));
    }
  }  
}

template <typename Dtype>
void ClipByValueLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  for(int n = 0; n < bottom.size(); n++){
    if (propagate_down[n]) {
      const Dtype* bottom_data = bottom[n]->cpu_data();
      const Dtype* top_diff = top[n]->cpu_diff();
      Dtype* bottom_diff = bottom[n]->mutable_cpu_diff();
      const int count = bottom[n]->count();
      for (int i = 0; i < count; ++i) {
        bottom_diff[i] = top_diff[i] * ((bottom_data[i] > min_val_) && (bottom_data[i] < max_val_));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ClipByValueLayer);
#endif

INSTANTIATE_CLASS(ClipByValueLayer);

}  // namespace caffe
