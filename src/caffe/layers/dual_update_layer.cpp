#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/dual_update_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DualUpdateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "DualUpdateLayer set up. ";
  // bottom[0]: dual
  // bottom[1]: primal
  // Configure the kernel size, padding, stride, and inputs.
  DualUpdateParameter dual_update_param = this->layer_param_.dual_update_param();
  channel_axis_ = bottom[1]->CanonicalAxisIndex(dual_update_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[1]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);

  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));

  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  const int num_kernel_dims = dual_update_param.kernel_size_size();
  CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
    << "kernel_size must be specified once, or once per spatial dimension "
    << "(kernel_size specified " << num_kernel_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  for (int i = 0; i < num_spatial_axes_; ++i) {
    kernel_shape_data[i] =
      dual_update_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }


  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  const int num_stride_dims = dual_update_param.stride_size();
  CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
    num_stride_dims == num_spatial_axes_)
    << "stride must be specified once, or once per spatial dimension "
    << "(stride specified " << num_stride_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  const int kDefaultStride = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
    dual_update_param.stride((num_stride_dims == 1) ? 0 : i);
    CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
  }

  // Setup pad dimensions (pad_).
  start_pad_.Reshape(spatial_dim_blob_shape);
  end_pad_.Reshape(spatial_dim_blob_shape);
  int* start_pad_data = start_pad_.mutable_cpu_data();
  int* end_pad_data = end_pad_.mutable_cpu_data();
  
  const int num_start_pad_dims = dual_update_param.start_pad_size();
  const int num_end_pad_dims = dual_update_param.end_pad_size();

  CHECK(num_start_pad_dims == 0 || num_start_pad_dims == 1 || 
    num_start_pad_dims == num_spatial_axes_)
    << "start pad must be specified once, or once per spatial dimension "
    << "(pad specified " << num_start_pad_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  const int kDefaultStartPad = 0;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    start_pad_data[i] = (num_start_pad_dims == 0) ? kDefaultStartPad :
      dual_update_param.start_pad((num_start_pad_dims == 1) ? 0 : i);
  }

  CHECK(num_end_pad_dims == 0 || num_end_pad_dims == 1 || 
    num_end_pad_dims == num_spatial_axes_)
    << "end pad must be specified once, or once per spatial dimension "
    << "(pad specified " << num_end_pad_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  const int kDefaultEndPad = 0;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    end_pad_data[i] = (num_end_pad_dims == 0) ? kDefaultEndPad :
      dual_update_param.end_pad((num_end_pad_dims == 1) ? 0 : i);
  }

  // dilation default to 0
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = kDefaultDilation;
  }


  // read parameter tau used in primal update formula
  sigma_ = this->layer_param_.dual_update_param().sigma();

  // Configure output and input channels 
  channels_ = bottom[1]->shape(channel_axis_);
  num_output_ = this->layer_param_.dual_update_param().num_output();
  CHECK_GT(num_output_, 0);
  
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  
  // Handle the parameters: weights 
  // - blobs_[0] holds the filter weights
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1, this->blobs_.size())
      << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
        << weight_shaped_blob.shape_string() << "; instead, shape was "
        << this->blobs_[0]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {

    this->blobs_.resize(1);
    // Initialize and fill the weights:
    // output channels x input channels  x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.dual_update_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get()); 


  }

  kernel_dim_ = this->blobs_[0]->count(1);

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  
  dual_before_proj_.ReshapeLike(*bottom[0]);
  dual_projection_bottom_vecs_.clear();
  dual_projection_bottom_vecs_.push_back(&dual_before_proj_);

  LayerParameter dual_projection_layer_param;
  dual_projection_layer_.reset(new DualProjectionLayer<Dtype>(dual_projection_layer_param));
  dual_projection_layer_->SetUp(dual_projection_bottom_vecs_, top);
}

template <typename Dtype>
void DualUpdateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[1]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  
  num_ = bottom[1]->count(0, channel_axis_);

  CHECK_EQ(bottom[1]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";

  // check the size of the variables
  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes())
      << "Primal and Dual variable must have the same dimensions";
  // check the spatial dimension 
  for (int i = 0; i < num_spatial_axes_; i++){
    int temp = i+first_spatial_axis;
    CHECK_EQ(bottom[0]->shape(temp), bottom[1]->shape(temp))
        <<" primal and dual must have the same spatial dimensions.";
  }

  // check the batch size
  CHECK_EQ(bottom[0]->count(0, channel_axis_), num_)
    <<"All variables must have the same number data. ";

  // Shape the tops.
  dual_shape_ = &bottom[0]->shape();
  primal_shape_ = &bottom[1]->shape();
  check_dual_shape();
  top[0]->ReshapeLike(*bottom[0]);
  conv_res_.ReshapeLike(*bottom[0]);
  dual_before_proj_.ReshapeLike(*bottom[0]);
  
  conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
 
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = bottom[1]->shape(channel_axis_ + i);
  }

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. 
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    col_buffer_shape_.push_back(dual_spatial_shape(i));
  }
  /*col_buffer_.Reshape(col_buffer_shape_); */
  dual_dim_ = top[0]->count(channel_axis_);
  primal_dim_ = bottom[1]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = primal_dim_ ;
}

template <typename Dtype>
void DualUpdateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* primal_data = bottom[1]->cpu_data();
  Dtype* conv_res_data = conv_res_.mutable_cpu_data();
  for(int n = 0; n < num_; n++){
    forward_cpu_gemm(primal_data + n * primal_dim_, weight, conv_res_data + n * dual_dim_);
  }

  
  Dtype* dual_before_proj_data = dual_before_proj_.mutable_cpu_data();
  const int count = bottom[0]->count();
  caffe_copy<Dtype>(count, bottom[0]->cpu_data(), dual_before_proj_data);
  // add conv result
  const Dtype* conv_res_data_const = conv_res_.cpu_data();
  caffe_axpy<Dtype>(count, Dtype(sigma_* 1.), conv_res_data_const, dual_before_proj_data);

  // projection to ||.|| >=1
  dual_projection_layer_->Forward(dual_projection_bottom_vecs_, top);

  

}

template <typename Dtype>
void DualUpdateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  vector<bool> down;
  down.clear();
  down.push_back(true);
  dual_projection_layer_->Backward(top, down, dual_projection_bottom_vecs_);
  const Dtype* dual_before_proj_diff = dual_before_proj_.cpu_diff();


  // compute gradient with respect to input previous dual data  
  const int count = bottom[0]->count();
  if(propagate_down[0]){
    Dtype* dual_diff = bottom[0]->mutable_cpu_diff(); 
    caffe_copy<Dtype>(count, dual_before_proj_diff, dual_diff);
  }

  // compute gradient with respect to dual term and weight.
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  if(propagate_down[1] || this->param_propagate_down_[0]){

    Dtype* conv_res_diff = conv_res_.mutable_cpu_diff();
    caffe_set<Dtype>(conv_res_.count(), Dtype(0.), conv_res_diff);
    caffe_axpy<Dtype>(conv_res_.count(), Dtype(sigma_), dual_before_proj_diff, 
      conv_res_diff);

    
    Dtype* primal_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* primal_data = bottom[1]->cpu_data();
    for(int n = 0; n < num_; n++){
      if(this->param_propagate_down_[0]){
        weight_cpu_gemm(primal_data + n * primal_dim_,
          conv_res_diff + n * dual_dim_, weight_diff);
      }

      if(propagate_down[1]){
        backward_cpu_gemm(conv_res_diff + n * dual_dim_, weight,
          primal_diff + n * primal_dim_);
      }
      
    }
  }  
}




template <typename Dtype>
void DualUpdateLayer<Dtype>::check_dual_shape(){

  const int* kernel_shape_data = kernel_shape_.cpu_data();
  const int* stride_data = stride_.cpu_data();
  const int* start_pad_data = start_pad_.cpu_data();
  const int* end_pad_data = end_pad_.cpu_data();

  for (int i = 0; i < num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = primal_spatial_shape(i);
    const int output_dim = (input_dim + start_pad_data[i] +  end_pad_data[i]
      -kernel_shape_data[i])/stride_data[i] + 1;
    CHECK_EQ(dual_spatial_shape(i), output_dim)
      << "Convolution with primal size should be compatible with dual size.";
  }
  CHECK_EQ((*dual_shape_)[channel_axis_], num_output_)
    << "Output channels should be the same as dual channels.";

}

template <typename Dtype>
void DualUpdateLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  
  if(!skip_im2col){
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
  }
  col_buff = col_buffer_.cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_, 
    conv_out_spatial_dim_, kernel_dim_, (Dtype)1., 
    weights, col_buff, (Dtype)0., output);
}


template <typename Dtype>
void DualUpdateLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
    conv_out_spatial_dim_, conv_out_channels_, (Dtype)1., 
    weights, output, (Dtype)0., col_buff); 
  
  conv_col2im_cpu(col_buff, input);
}

template <typename Dtype>
void DualUpdateLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  
  conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
  col_buff = col_buffer_.cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_,
    kernel_dim_, conv_out_spatial_dim_, (Dtype)1., 
    output, col_buff, (Dtype)1., weights);
}


#ifndef CPU_ONLY

template <typename Dtype>
void DualUpdateLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  
  if(!skip_im2col){
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
  }
  
  col_buff = col_buffer_.gpu_data();
  
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_, 
    conv_out_spatial_dim_, kernel_dim_,
    (Dtype)1., weights, col_buff, (Dtype)0., output);
}



template <typename Dtype>
void DualUpdateLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
    conv_out_spatial_dim_, conv_out_channels_, (Dtype)1., 
    weights, output, (Dtype)0., col_buff);
  
  conv_col2im_gpu(col_buff, input);
}

template <typename Dtype>
void DualUpdateLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
  col_buff = col_buffer_.gpu_data();
  
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_,
    kernel_dim_, conv_out_spatial_dim_, (Dtype)1., 
    output, col_buff, (Dtype)1., weights);
  
}


#endif  // !CPU_ONLY*/

#ifdef CPU_ONLY
STUB_GPU(DualUpdateLayer);
#endif

INSTANTIATE_CLASS(DualUpdateLayer);

}  // namespace caffe