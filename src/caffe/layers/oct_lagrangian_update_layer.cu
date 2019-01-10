#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/oct_lagrangian_update_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void OctLagrangianUpdateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int num_channels = bottom[1]->shape(1);
  const int num_ = bottom[1]->shape(0);
  const int spatial_count = bottom[1]->count(2);
  
  const Dtype* primal_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  caffe_gpu_set<Dtype>(top[0]->count(), Dtype(0.), top_data);
  for(int n = 0; n < num_; n++){
    for(int c = 0; c < num_channels; c++){
      caffe_gpu_axpy<Dtype>(spatial_count, Dtype(1.),  primal_data, top_data);
      primal_data += spatial_count;
    }
    top_data += spatial_count;

  }

  const Dtype* lagrangian_data = bottom[0]->gpu_data();
  top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
  caffe_gpu_add_scalar<Dtype>(count, Dtype(-1.), top_data);
  caffe_gpu_scal<Dtype>(count, Dtype(sigma_), top_data);
  caffe_gpu_axpy<Dtype>(count, Dtype(1.), lagrangian_data, top_data);

}

template <typename Dtype>
void OctLagrangianUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  if(propagate_down[0]){
    Dtype* lagrangian_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy<Dtype>(bottom[0]->count(), top_diff, lagrangian_diff);
  }

  if(propagate_down[1]){

    Dtype* primal_diff = bottom[1]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(bottom[1]->count(), Dtype(0.), primal_diff);

    const int num_channels = bottom[1]->shape(1);
    const int num_ = bottom[1]->shape(0);
    const int spatial_count = bottom[1]->count(2);
    for(int n = 0; n < num_; n++){
      for(int c = 0; c < num_channels; c++){
        caffe_gpu_axpy<Dtype>(spatial_count, Dtype(sigma_), top_diff, primal_diff); // gradient to bottom primal need to be accumulated
        primal_diff += spatial_count;
      }
      top_diff += spatial_count;
    }

  }
  
}


INSTANTIATE_LAYER_GPU_FUNCS(OctLagrangianUpdateLayer);

}  // namespace caffe
