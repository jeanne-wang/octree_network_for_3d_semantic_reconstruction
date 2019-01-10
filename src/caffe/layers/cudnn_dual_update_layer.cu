#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_dual_update_layer.hpp"

namespace caffe {

__global__ void sync_dual_conv_groups() { }

template <typename Dtype>
void CuDNNDualUpdateLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* primal_data = bottom[1]->gpu_data();
  Dtype* conv_res_data = this->conv_res_.mutable_gpu_data();

  CUDNN_CHECK(cudnnConvolutionForward(handle_[0],
    cudnn::dataType<Dtype>::one,
    bottom_desc_, primal_data,
    filter_desc_, weight,
    conv_desc_,
    fwd_algo_[0], workspace[0], workspace_fwd_sizes_[0],
    cudnn::dataType<Dtype>::zero,
    top_desc_, conv_res_data));

  // Synchronize the work across groups, each of which went into its own
  // stream, by launching an empty kernel into the default (null) stream.
  // NOLINT_NEXT_LINE(whitespace/operators)
  sync_dual_conv_groups<<<1, 1>>>();

  Dtype* dual_before_proj_data = this->dual_before_proj_.mutable_gpu_data();
  const int count = bottom[0]->count();
  caffe_copy<Dtype>(count, bottom[0]->gpu_data(), dual_before_proj_data);
  // add conv result
  const Dtype* conv_res_data_const = this->conv_res_.gpu_data();
  caffe_gpu_axpy<Dtype>(count, Dtype(this->sigma_* 1.), conv_res_data_const, dual_before_proj_data);

  // projection to ||.|| >=1
  (this->dual_projection_layer_)->Forward(this->dual_projection_bottom_vecs_, top);
  
}

template <typename Dtype>
void CuDNNDualUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  vector<bool> down;
  down.clear();
  down.push_back(true);
  (this->dual_projection_layer_)->Backward(top, down, this->dual_projection_bottom_vecs_);
  const Dtype* dual_before_proj_diff = this->dual_before_proj_.gpu_diff();


  // compute gradient with respect to input previous dual data  
  const int count = bottom[0]->count();
  if(propagate_down[0]){
    Dtype* dual_diff = bottom[0]->mutable_gpu_diff(); 
    caffe_copy<Dtype>(count, dual_before_proj_diff, dual_diff);
  }

  Dtype* conv_res_diff = this->conv_res_.mutable_gpu_diff();
  caffe_gpu_set<Dtype>(this->conv_res_.count(), Dtype(0.), conv_res_diff);
  caffe_gpu_axpy<Dtype>(this->conv_res_.count(), Dtype(this->sigma_), dual_before_proj_diff, 
      conv_res_diff);

  
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
  }
  
  // Backward through cuDNN in parallel over groups and gradients.

  // Gradient w.r.t. weights.
  if (this->param_propagate_down_[0]) {
    const Dtype* primal_data = bottom[1]->gpu_data();
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(
      handle_[1],
      cudnn::dataType<Dtype>::one,
      bottom_desc_, primal_data,
      top_desc_,    conv_res_diff,
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
    
    Dtype* primal_diff = bottom[1]->mutable_gpu_diff();
    CUDNN_CHECK(cudnnConvolutionBackwardData(
      handle_[2],
      cudnn::dataType<Dtype>::one,
      filter_desc_, weight,
      top_desc_, conv_res_diff,
      conv_desc_,
      bwd_data_algo_[0], workspace[2],
      workspace_bwd_data_sizes_[0],
      cudnn::dataType<Dtype>::zero,
      bottom_desc_, primal_diff));
  }
    

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
  sync_dual_conv_groups<<<1, 1>>>();
  
}

INSTANTIATE_LAYER_GPU_FUNCS(CuDNNDualUpdateLayer);

}  // namespace caffe
#endif
