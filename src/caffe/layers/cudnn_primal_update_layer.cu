#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_primal_update_layer.hpp"

namespace caffe {

__global__ void sync_primal_deconv_groups() { }

template <typename Dtype>
void CuDNNPrimalUpdateLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* dual_data = bottom[1]->gpu_data();
  Dtype* deconv_res_data = this->deconv_res_.mutable_gpu_data();

  CUDNN_CHECK(cudnnConvolutionBackwardData(handle_[0],
    cudnn::dataType<Dtype>::one,
    filter_desc_, weight,
    bottom_desc_, dual_data,
    conv_desc_,
    bwd_data_algo_[0], workspace[0], workspace_bwd_data_sizes_[0],
    cudnn::dataType<Dtype>::zero,
    top_desc_, deconv_res_data));

  // Synchronize the work across groups, each of which went into its own
  // stream, by launching an empty kernel into the default (null) stream.
  // NOLINT_NEXT_LINE(whitespace/operators)
  sync_primal_deconv_groups<<<1, 1>>>();

  Dtype* primal_before_proj_data = this->primal_before_proj_.mutable_gpu_data();
  const Dtype* deconv_res_data_const = this->deconv_res_.gpu_data();
  const int count = this->primal_before_proj_.count();
  caffe_copy<Dtype>(count, deconv_res_data_const, primal_before_proj_data);

  // add datacost term
  caffe_gpu_axpy<Dtype>(count, Dtype(1.), bottom[3]->gpu_data(), primal_before_proj_data);

  // add lagrangian
  const Dtype* lagrangian_data = bottom[2]->gpu_data();
  const int spatial_count = this->primal_before_proj_.count(this->channel_axis_+1);
  for(int n = 0; n < this->num_; n++){
    for(int c = 0; c < this->primal_before_proj_.shape(this->channel_axis_); c++){   
      caffe_gpu_axpy<Dtype>(spatial_count, Dtype(1.), lagrangian_data + n * spatial_count, 
        primal_before_proj_data+ n*this->primal_dim_ + c * spatial_count);
    }
  }


  caffe_gpu_scal<Dtype>(count, Dtype(-1. * this->tau_), primal_before_proj_data);
  // add previous primal data
  caffe_gpu_axpy<Dtype>(count, Dtype(1.), bottom[0]->gpu_data(), primal_before_proj_data);

  // projection to [0, 1]
  (this->primal_projection_layer_)->Forward(this->primal_projection_bottom_vecs_, top); 
  
}

template <typename Dtype>
void CuDNNPrimalUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  vector<bool> down;
  down.clear();
  down.push_back(true);
  (this->primal_projection_layer_)->Backward(top, down, this->primal_projection_bottom_vecs_);
  const Dtype* primal_before_proj_diff = this->primal_before_proj_.gpu_diff();

  
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
    caffe_gpu_axpy<Dtype>(count, Dtype(-1.*this->tau_), primal_before_proj_diff, datacost_diff);
  }
  
  // compute gradient with respect to lagrangian term

  if(propagate_down[2]){
    const int spatial_count = this->primal_before_proj_.count(this->channel_axis_+1);
    Dtype* lagrangian_diff = bottom[2]->mutable_gpu_diff();
    caffe_gpu_set<Dtype>(bottom[2]->count(), Dtype(0.), lagrangian_diff);
    for(int n = 0; n < this->num_; n++){
      for(int c = 0; c < this->primal_before_proj_.shape(this->channel_axis_); c++){     
        caffe_gpu_axpy<Dtype>(spatial_count, Dtype(-1.* this->tau_), 
          primal_before_proj_diff + n * this->primal_dim_ + c * spatial_count,
          lagrangian_diff + n * spatial_count);
      }
    }
  } 


  Dtype* deconv_res_diff = this->deconv_res_.mutable_gpu_diff();
  caffe_gpu_set<Dtype>(count, Dtype(0.), deconv_res_diff);
  caffe_gpu_axpy<Dtype>(count, Dtype(-1.* this->tau_), primal_before_proj_diff, 
      deconv_res_diff);

  
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  
  // Backward through cuDNN in parallel over groups and gradients.

  // Gradient w.r.t. weights.
  if (this->param_propagate_down_[0]) {
    const Dtype* dual_data = bottom[1]->gpu_data();
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      handle_[1],
      cudnn::dataType<Dtype>::one,
      top_desc_, deconv_res_diff,
      bottom_desc_, dual_data,
      conv_desc_,
      bwd_filter_algo_[0], workspace[1],
      workspace_bwd_filter_sizes_[0],
      cudnn::dataType<Dtype>::one,
      filter_desc_, weight_diff));
  }

    // Gradient w.r.t. bottom data.
  if (propagate_down[1]) {
    if (weight == NULL) {
      weight = this->blobs_[0]->gpu_data();
    }
    
    Dtype* dual_diff = bottom[1]->mutable_gpu_diff();
    CUDNN_CHECK(cudnnConvolutionForward(
      handle_[2],
      cudnn::dataType<Dtype>::one,
      top_desc_, deconv_res_diff,
      filter_desc_, weight,
      conv_desc_,
      fwd_algo_[0], workspace[2],
      workspace_fwd_sizes_[0],
      cudnn::dataType<Dtype>::zero,
      bottom_desc_, dual_diff));
  }
    

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
  sync_primal_deconv_groups<<<1, 1>>>();
  
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNPrimalUpdateLayer);

}  // namespace caffe
#endif
