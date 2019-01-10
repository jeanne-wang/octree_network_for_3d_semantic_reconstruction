#include <vector>

#include "caffe/layers/dual_update_layer.hpp"

namespace caffe {

template <typename Dtype>
void DualUpdateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* primal_data = bottom[1]->gpu_data();
  Dtype* conv_res_data = conv_res_.mutable_gpu_data();
  for(int n = 0; n < num_; n++){
    forward_gpu_gemm(primal_data + n * primal_dim_, weight, conv_res_data + n * dual_dim_);
  }

  Dtype* dual_before_proj_data = dual_before_proj_.mutable_gpu_data();
  const int count = bottom[0]->count();
  caffe_copy<Dtype>(count, bottom[0]->gpu_data(), dual_before_proj_data);
  // add conv result
  const Dtype* conv_res_data_const = conv_res_.gpu_data();
  caffe_gpu_axpy<Dtype>(count, Dtype(sigma_* 1.), conv_res_data_const, dual_before_proj_data);

  // projection to ||.|| >=1
  dual_projection_layer_->Forward(dual_projection_bottom_vecs_, top);
}

template <typename Dtype>
void DualUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  vector<bool> down;
  down.clear();
  down.push_back(true);
  dual_projection_layer_->Backward(top, down, dual_projection_bottom_vecs_);
  const Dtype* dual_before_proj_diff = dual_before_proj_.gpu_diff();

  // compute gradient with respect to input previous dual data  
  const int count = bottom[0]->count();
  if(propagate_down[0]){
    Dtype* dual_diff = bottom[0]->mutable_gpu_diff(); 
    caffe_copy<Dtype>(count, dual_before_proj_diff, dual_diff);
  }

  // compute gradient with respect to dual term and weight.
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

  if(propagate_down[1] || this->param_propagate_down_[0]){

    
    Dtype* conv_res_diff = conv_res_.mutable_gpu_diff();
    caffe_gpu_set<Dtype>(conv_res_.count(), Dtype(0.), conv_res_diff);
    caffe_gpu_axpy<Dtype>(conv_res_.count(), Dtype(sigma_), dual_before_proj_diff, 
      conv_res_diff);


    Dtype* primal_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* primal_data = bottom[1]->gpu_data();
    for(int n = 0; n < num_; n++){
      if(this->param_propagate_down_[0]){
        weight_gpu_gemm(primal_data + n * primal_dim_,
          conv_res_diff + n * dual_dim_, weight_diff);
      }

      if(propagate_down[1]){
        backward_gpu_gemm(conv_res_diff + n * dual_dim_, weight,
          primal_diff + n * primal_dim_);
      }
      
    }
  }  
}

INSTANTIATE_LAYER_GPU_FUNCS(DualUpdateLayer);

}  // namespace caffe
