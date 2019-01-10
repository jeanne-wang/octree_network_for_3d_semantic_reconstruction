#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/primal_further_update_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void PrimalFurtherUpdateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // bottom[0] u_t
  // bottom[1] u_t-1
  top[0]->ReshapeLike(*bottom[0]);
  CHECK_EQ(bottom[0]->num_axes(), 5);
  CHECK_EQ(bottom[1]->num_axes(), 5);
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2));
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3));
  CHECK_EQ(bottom[0]->shape(4), bottom[1]->shape(4));
  

}

template <typename Dtype>
void PrimalFurtherUpdateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* u_data = bottom[0]->cpu_data();
  const Dtype* u_prev_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int count = top[0]->count();
  caffe_set<Dtype>(count, Dtype(0.), top_data);
  caffe_axpy<Dtype>(count, Dtype(2.), u_data, top_data);
  caffe_axpy<Dtype>(count, Dtype(-1.), u_prev_data, top_data);

}

template <typename Dtype>
void PrimalFurtherUpdateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = top[0]->count();
  if(propagate_down[0]){
    Dtype* u_diff = bottom[0]->mutable_cpu_diff();
    caffe_set<Dtype>(count, Dtype(0.), u_diff);
    caffe_axpy<Dtype>(count, Dtype(2.), top_diff, u_diff);
  }

  if(propagate_down[1]){

    Dtype* u_prev_diff = bottom[1]->mutable_cpu_diff();
    caffe_set<Dtype>(count, Dtype(0.), u_prev_diff);
    caffe_axpy<Dtype>(count, Dtype(-1.), top_diff, u_prev_diff);
    

  }
  
}


#ifdef CPU_ONLY
STUB_GPU(PrimalFurtherUpdateLayer);
#endif

INSTANTIATE_CLASS(PrimalFurtherUpdateLayer);
REGISTER_LAYER_CLASS(PrimalFurtherUpdate);

}  // namespace caffe
