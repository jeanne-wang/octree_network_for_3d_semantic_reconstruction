#ifdef USE_CUDNN
#include <algorithm>
#include <vector>

#include "caffe/layers/cudnn_dual_update_layer.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
template <typename Dtype>
void CuDNNDualUpdateLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  DualUpdateLayer<Dtype>::LayerSetUp(bottom, top);
  std::cout << "Cudnn DualUpdateLayer setting up...." << std::endl;
  // Initialize CUDA streams and cuDNN.
  stream_         = new cudaStream_t[CUDNN_STREAMS_PER_GROUP];
  handle_         = new cudnnHandle_t[CUDNN_STREAMS_PER_GROUP];

  // Initialize algorithm arrays
  fwd_algo_       = new cudnnConvolutionFwdAlgo_t[1];
  bwd_filter_algo_= new cudnnConvolutionBwdFilterAlgo_t[1];
  bwd_data_algo_  = new cudnnConvolutionBwdDataAlgo_t[1];

  // initialize size arrays
  workspace_fwd_sizes_ = new size_t[1];
  workspace_bwd_filter_sizes_ = new size_t[1];
  workspace_bwd_data_sizes_ = new size_t[1];

  // workspace data
  workspaceSizeInBytes = 0;
  workspaceData = NULL;
  workspace = new void*[CUDNN_STREAMS_PER_GROUP];

  fwd_algo_[0] = (cudnnConvolutionFwdAlgo_t)0;
  bwd_filter_algo_[0] = (cudnnConvolutionBwdFilterAlgo_t)0;
  bwd_data_algo_[0] = (cudnnConvolutionBwdDataAlgo_t)0;
  
  // default algorithms don't require workspace
  workspace_fwd_sizes_[0] = 0;
  workspace_bwd_data_sizes_[0] = 0;
  workspace_bwd_filter_sizes_[0] = 0;

  for (int g = 0; g < CUDNN_STREAMS_PER_GROUP; g++) {
    CUDA_CHECK(cudaStreamCreate(&stream_[g]));
    CUDNN_CHECK(cudnnCreate(&handle_[g]));
    CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
    workspace[g] = NULL;
  }

  // Create filter descriptor.
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int kernel_h = kernel_shape_data[0];
  const int kernel_w = kernel_shape_data[1];
  const int kernel_d = kernel_shape_data[2];
  cudnn::createFilter5dDesc<Dtype>(&filter_desc_,
      this->num_output_, this->channels_,
      kernel_h, kernel_w, kernel_d);

  // Create tensor descriptor(s) for data and corresponding convolution(s).
  cudnn::createTensor5dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor5dDesc<Dtype>(&top_desc_);
  cudnn::createConvolutionDesc<Dtype>(&conv_desc_);

  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNDualUpdateLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  DualUpdateLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(3, this->num_spatial_axes_)
      << "Primal must have 3 spatial axes "
      << "(e.g., height and width and depth). "
      << "Use 'engine: CAFFE' for general ND convolution.";

  const int height = bottom[1]->shape(this->channel_axis_ + 1);
  const int width = bottom[1]->shape(this->channel_axis_ + 2);
  const int depth = bottom[1]->shape(this->channel_axis_ + 3);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int depth_out = top[0]->shape(this->channel_axis_ + 3);


  const int* start_pad_data = this->start_pad_.cpu_data();
  const int* end_pad_data = this->end_pad_.cpu_data();
  CHECK_EQ(start_pad_data[0], end_pad_data[0]) 
      << "start_pad must be equal with end_pad.";

  const int pad_h = start_pad_data[0];
  const int pad_w = start_pad_data[1];
  const int pad_d = start_pad_data[2];

  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];
  const int stride_d = stride_data[2];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  
  cudnn::setTensor5dDesc<Dtype>(&bottom_desc_,
    this->num_,
    this->channels_, height, width, depth);
  cudnn::setTensor5dDesc<Dtype>(&top_desc_,
    this->num_,
    this->num_output_, height_out, width_out, depth_out);
  cudnn::setConvolution3dDesc<Dtype>(&conv_desc_, bottom_desc_,
    filter_desc_, pad_h, pad_w, pad_d,
    stride_h, stride_w, stride_d);

  // choose forward and backward algorithms + workspace(s)
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
    bottom_desc_,
    filter_desc_,
    conv_desc_,
    top_desc_,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
    workspace_limit_bytes,
    &fwd_algo_[0]));

  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
    bottom_desc_,
    filter_desc_,
    conv_desc_,
    top_desc_,
    fwd_algo_[0],
    &(workspace_fwd_sizes_[0])));

  // choose backward algorithm for filter
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
      bottom_desc_, top_desc_, conv_desc_, filter_desc_,
      CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes, &bwd_filter_algo_[0]) );

  // get workspace for backwards filter algorithm
  CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
      bottom_desc_, top_desc_, conv_desc_, filter_desc_,
      bwd_filter_algo_[0], &workspace_bwd_filter_sizes_[0]));

    // choose backward algo for data
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
      filter_desc_, top_desc_, conv_desc_, bottom_desc_,
      CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes, &bwd_data_algo_[0]));

    // get workspace size
  CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
      filter_desc_, top_desc_, conv_desc_, bottom_desc_,
      bwd_data_algo_[0], &workspace_bwd_data_sizes_[0]) );
  

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  // get max over all operations
  size_t max_workspace = std::max(workspace_fwd_sizes_[0],
                             workspace_bwd_data_sizes_[0]);
  max_workspace = std::max(max_workspace, workspace_bwd_filter_sizes_[0]);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace * CUDNN_STREAMS_PER_GROUP;

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      
      workspace_fwd_sizes_[0] = 0;
      workspace_bwd_filter_sizes_[0] = 0;
      workspace_bwd_data_sizes_[0] = 0;
      fwd_algo_[0] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
      bwd_filter_algo_[0] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
      bwd_data_algo_[0] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      
      // NULL out all workspace pointers
      for (int g = 0; g < CUDNN_STREAMS_PER_GROUP; g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < CUDNN_STREAMS_PER_GROUP; g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }
}

template <typename Dtype>
CuDNNDualUpdateLayer<Dtype>::~CuDNNDualUpdateLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
  
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  cudaFree(workspaceData);
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNDualUpdateLayer);

}   // namespace caffe
#endif
