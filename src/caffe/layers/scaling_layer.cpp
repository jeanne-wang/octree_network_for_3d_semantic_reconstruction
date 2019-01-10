#include <algorithm>
#include <vector>

#include "caffe/layers/scaling_layer.hpp"

namespace caffe {

template <typename Dtype>
void ScalingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  scale_ = this->layer_param_.scaling_param().scale();
}


template <typename Dtype>
void ScalingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  caffe_copy<Dtype>(count, bottom_data, top_data);
  caffe_scal<Dtype>(count, Dtype(scale_), top_data);

}

template <typename Dtype>
void ScalingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    caffe_copy<Dtype>(count, top_diff, bottom_diff);
    caffe_scal<Dtype>(count, Dtype(scale_), bottom_diff);

  }
}


#ifdef CPU_ONLY
STUB_GPU(ScalingLayer);
#endif

INSTANTIATE_CLASS(ScalingLayer);

REGISTER_LAYER_CLASS(Scaling);

}  // namespace caffe
