#include <vector>

#include "caffe/layers/primal_update_layer.hpp"

namespace caffe {

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* dual_data = bottom[1]->gpu_data();
  Dtype* deconv_res_data = deconv_res_.mutable_gpu_data();
  for(int n = 0; n < num_; n++){
    backward_gpu_gemm(dual_data+n*dual_dim_, weight, deconv_res_data+n*primal_dim_);
  }

  
  Dtype* primal_before_proj_data = primal_before_proj_.mutable_gpu_data();
  const Dtype* deconv_res_data_const = deconv_res_.gpu_data();
  const int count = primal_before_proj_.count();
  caffe_copy<Dtype>(count, deconv_res_data_const, primal_before_proj_data);

  // add datacost term
  caffe_gpu_axpy<Dtype>(count, Dtype(1.), bottom[3]->gpu_data(), primal_before_proj_data);

  // add lagrangian
  const Dtype* lagrangian_data = bottom[2]->gpu_data();
  const int spatial_count = primal_before_proj_.count(channel_axis_+1);
  for(int n = 0; n < num_; n++){
    for(int c = 0; c < primal_before_proj_.shape(channel_axis_); c++){   
      caffe_gpu_axpy<Dtype>(spatial_count, Dtype(1.), lagrangian_data + n * spatial_count, 
        primal_before_proj_data+ n*primal_dim_ + c * spatial_count);
    }
  }


  caffe_gpu_scal<Dtype>(count, Dtype(-1. * tau_), primal_before_proj_data);
  // add previous primal data
  caffe_gpu_axpy<Dtype>(count, Dtype(1.), bottom[0]->gpu_data(), primal_before_proj_data);

  // projection to [0, 1]
  primal_projection_layer_->Forward(primal_projection_bottom_vecs_, top); 
  
}

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  vector<bool> down;
  down.clear();
  down.push_back(true);
  primal_projection_layer_->Backward(top, down, primal_projection_bottom_vecs_);
  const Dtype* primal_before_proj_diff = primal_before_proj_.gpu_diff();

  


  // compute gradient with respect to input previous primal data  
  const int count = bottom[0]->count();
  if(propagate_down[0]){
    Dtype* primal_diff = bottom[0]->mutable_gpu_diff(); 
    caffe_copy<Dtype>(count, primal_before_proj_diff, primal_diff); // gradient to primal need to be accumulated
  }
  

  // compute gradient with respect to data cost term
  if(propagate_down[3]){
    Dtype* datacost_diff = bottom[3]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(count, Dtype(0), datacost_diff);
    caffe_gpu_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, datacost_diff);
  }
  
  // compute gradient with respect to lagrangian term

  if(propagate_down[2]){
    const int spatial_count = primal_before_proj_.count(channel_axis_+1);
    Dtype* lagrangian_diff = bottom[2]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(bottom[2]->count(), Dtype(0.), lagrangian_diff);
    for(int n = 0; n < num_; n++){
      for(int c = 0; c < primal_before_proj_.shape(channel_axis_); c++){     
        caffe_gpu_axpy<Dtype>(spatial_count, Dtype(-1.*tau_), 
          primal_before_proj_diff + n * primal_dim_ + c * spatial_count,
          lagrangian_diff + n * spatial_count);
      }
    }
  } 

  // compute gradient with respect to dual term and weight.
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

  if(propagate_down[1] || this->param_propagate_down_[0]){

    
    Dtype* deconv_res_diff = deconv_res_.mutable_gpu_diff();
    caffe_gpu_set<Dtype>(count, Dtype(0.), deconv_res_diff);
    caffe_gpu_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, 
      deconv_res_diff); 


    Dtype* dual_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* dual_data = bottom[1]->gpu_data();
    for(int n = 0; n < num_; n++){
      if(this->param_propagate_down_[0]){
        weight_gpu_gemm(deconv_res_diff + n * primal_dim_,
          dual_data + n * dual_dim_, weight_diff);
      }

      if(propagate_down[1]){
        forward_gpu_gemm(deconv_res_diff + n * primal_dim_, weight,
          dual_diff + n * dual_dim_, this->param_propagate_down_[0]);
      }
      
    }
  }  
}

INSTANTIATE_LAYER_GPU_FUNCS(PrimalUpdateLayer);

}  // namespace caffe
