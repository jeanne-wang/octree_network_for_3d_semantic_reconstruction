#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/lagrangian_update_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LagrangianUpdateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // bottom[0]: l
  // bottom[1]: primal
  
  sigma_ =  this->layer_param_.lagrangian_update_param().sigma();


}

template <typename Dtype>
void LagrangianUpdateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(2));
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(3));
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(4));

}

template <typename Dtype>
void LagrangianUpdateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const int num_channels = bottom[1]->shape(1);
  const int num_ = bottom[1]->shape(0);
  const int spatial_count = bottom[1]->count(2);
  
  const Dtype* primal_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  caffe_set<Dtype>(top[0]->count(), Dtype(0.), top_data);
  for(int n = 0; n < num_; n++){
    for(int c = 0; c < num_channels; c++){
      caffe_axpy<Dtype>(spatial_count, Dtype(1.),  primal_data, top_data);
      primal_data += spatial_count;
    }
    top_data += spatial_count;

  }

  const Dtype* lagrangian_data = bottom[0]->cpu_data();
  top_data = top[0]->mutable_cpu_data();
  const int count = top[0]->count();
  caffe_add_scalar<Dtype>(count, Dtype(-1.), top_data);
  caffe_scal<Dtype>(count, Dtype(sigma_), top_data);
  caffe_axpy<Dtype>(count, Dtype(1.), lagrangian_data, top_data);

}

template <typename Dtype>
void LagrangianUpdateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  if(propagate_down[0]){
    Dtype* lagrangian_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy<Dtype>(bottom[0]->count(), top_diff, lagrangian_diff);
  }

  if(propagate_down[1]){

    Dtype* primal_diff = bottom[1]->mutable_cpu_diff();
    caffe_set<Dtype>(bottom[1]->count(), Dtype(0.), primal_diff);

    const int num_channels = bottom[1]->shape(1);
    const int num_ = bottom[1]->shape(0);
    const int spatial_count = bottom[1]->count(2);
    for(int n = 0; n < num_; n++){
      for(int c = 0; c < num_channels; c++){
        caffe_axpy<Dtype>(spatial_count, Dtype(sigma_), top_diff, primal_diff); // gradient to bottom primal need to be accumulated
        primal_diff += spatial_count;
      }
      top_diff += spatial_count;
    }

  }
  
}


#ifdef CPU_ONLY
STUB_GPU(LagrangianUpdateLayer);
#endif

INSTANTIATE_CLASS(LagrangianUpdateLayer);
REGISTER_LAYER_CLASS(LagrangianUpdate);

}  // namespace caffe
