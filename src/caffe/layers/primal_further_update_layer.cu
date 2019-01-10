#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/primal_further_update_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void PrimalFurtherUpdateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* u_data = bottom[0]->gpu_data();
  const Dtype* u_prev_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const int count = top[0]->count();
  caffe_gpu_set<Dtype>(count, Dtype(0.), top_data);
  caffe_gpu_axpy<Dtype>(count, Dtype(2.), u_data, top_data);
  caffe_gpu_axpy<Dtype>(count, Dtype(-1.), u_prev_data, top_data);

}

template <typename Dtype>
void PrimalFurtherUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->gpu_diff();
  const int count = top[0]->count();
  if(propagate_down[0]){
    Dtype* u_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(count, Dtype(0.), u_diff);
    caffe_gpu_axpy<Dtype>(count, Dtype(2.), top_diff, u_diff);
  }

  if(propagate_down[1]){

    Dtype* u_prev_diff = bottom[1]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(count, Dtype(0.), u_prev_diff);
    caffe_gpu_axpy<Dtype>(count, Dtype(-1.), top_diff, u_prev_diff);
    

  }
  
}


INSTANTIATE_LAYER_GPU_FUNCS(PrimalFurtherUpdateLayer);

}  // namespace caffe
