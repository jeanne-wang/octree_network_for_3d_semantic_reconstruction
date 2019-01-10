#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/primal_update_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  // bottom[0]: primal
  // bottom[1]: dual
  // bottom[2]: lagrangian
  // bottom[3]: data cost
  // Configure the kernel size, padding, stride, and inputs.
  PrimalUpdateParameter primal_update_param = this->layer_param_.primal_update_param();
  channel_axis_ = bottom[1]->CanonicalAxisIndex(primal_update_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[1]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);

  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));

  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  const int num_kernel_dims = primal_update_param.kernel_size_size();
  CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
    << "kernel_size must be specified once, or once per spatial dimension "
    << "(kernel_size specified " << num_kernel_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  for (int i = 0; i < num_spatial_axes_; ++i) {
    kernel_shape_data[i] =
      primal_update_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }


  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  const int num_stride_dims = primal_update_param.stride_size();
  CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
    num_stride_dims == num_spatial_axes_)
    << "stride must be specified once, or once per spatial dimension "
    << "(stride specified " << num_stride_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  const int kDefaultStride = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
    primal_update_param.stride((num_stride_dims == 1) ? 0 : i);
    CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
  }

  // Setup pad dimensions (pad_).
  start_pad_.Reshape(spatial_dim_blob_shape);
  end_pad_.Reshape(spatial_dim_blob_shape);
  int* start_pad_data = start_pad_.mutable_cpu_data();
  int* end_pad_data = end_pad_.mutable_cpu_data();
  
  const int num_start_pad_dims = primal_update_param.start_pad_size();
  const int num_end_pad_dims = primal_update_param.end_pad_size();

  CHECK(num_start_pad_dims == 0 || num_start_pad_dims == 1 || 
    num_start_pad_dims == num_spatial_axes_)
    << "start pad must be specified once, or once per spatial dimension "
    << "(pad specified " << num_start_pad_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  const int kDefaultStartPad = 0;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    start_pad_data[i] = (num_start_pad_dims == 0) ? kDefaultStartPad :
      primal_update_param.start_pad((num_start_pad_dims == 1) ? 0 : i);
  }

  CHECK(num_end_pad_dims == 0 || num_end_pad_dims == 1 || 
    num_end_pad_dims == num_spatial_axes_)
    << "end pad must be specified once, or once per spatial dimension "
    << "(pad specified " << num_end_pad_dims << " times; "
    << num_spatial_axes_ << " spatial dims).";
  const int kDefaultEndPad = 0;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    end_pad_data[i] = (num_end_pad_dims == 0) ? kDefaultEndPad :
      primal_update_param.end_pad((num_end_pad_dims == 1) ? 0 : i);
  }

  // dilation default to 0
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = kDefaultDilation;
  }

  // read parameter tau used in primal update formula
  tau_ = this->layer_param_.primal_update_param().tau();

  // Configure output and input channels 
  channels_ = bottom[1]->shape(channel_axis_);
  num_output_ = this->layer_param_.primal_update_param().num_output();
  CHECK_GT(num_output_, 0);
  
  // For deconvolution, conv_out and conv_in channels are reversed
  conv_out_channels_ = channels_;
  conv_in_channels_ = num_output_;
  
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
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.primal_update_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }

  kernel_dim_ = this->blobs_[0]->count(1);

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  primal_before_proj_.ReshapeLike(*bottom[0]);
  primal_projection_bottom_vecs_.clear();
  primal_projection_bottom_vecs_.push_back(&primal_before_proj_);

  LayerParameter primal_projection_param;
  primal_projection_layer_.reset(new PrimalProjectionLayer<Dtype>(primal_projection_param));
  primal_projection_layer_->SetUp(primal_projection_bottom_vecs_, top);
}

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
  CHECK_EQ(bottom[0]->num_axes(), bottom[3]->num_axes())
      << "Primal variable and data cost term must have the same dimensions";
  CHECK_EQ(bottom[0]->num_axes(), bottom[2]->num_axes()+1)
      <<" Lagrangian has one less dimension than primal variable.";

  // check the spatial dimension 
  for (int i = 0; i < num_spatial_axes_; i++){
    int temp = i+first_spatial_axis;
    CHECK_EQ(bottom[0]->shape(temp), bottom[1]->shape(temp))
        <<" primal and dual must have the same spatial dimensions.";
    CHECK_EQ(bottom[0]->shape(temp), bottom[3]->shape(temp))
        <<" primal and data cost must have the same spatial dimensions.";
    CHECK_EQ(bottom[0]->shape(temp), bottom[2]->shape(temp-1))
        <<" primal and lagrangian must have the same spatial dimensions.";
  }

  // check the batch size
  for(int bottom_id = 1; bottom_id < bottom.size(); bottom_id++){
    CHECK_EQ(bottom[bottom_id]->shape(0), bottom[0]->shape(0))
      <<"All variables must have the same number data. ";
  }

  // Shape the tops.
  dual_shape_ = &bottom[1]->shape();
  primal_shape_ = &bottom[0]->shape();
  check_primal_shape();
  top[0]->ReshapeLike(*bottom[0]);
  deconv_res_.ReshapeLike(*bottom[0]);
  primal_before_proj_.ReshapeLike(*bottom[0]);
  
  // for deconv
  conv_out_spatial_dim_ = bottom[1]->count(first_spatial_axis);
  
 
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
  }

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. 
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    col_buffer_shape_.push_back(dual_spatial_shape(i));
  }
  /*col_buffer_.Reshape(col_buffer_shape_); */
  dual_dim_ = bottom[1]->count(channel_axis_);
  primal_dim_ = bottom[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = primal_dim_ ;
}

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* dual_data = bottom[1]->cpu_data();
  Dtype* deconv_res_data = deconv_res_.mutable_cpu_data();
  for(int n = 0; n < num_; n++){
    backward_cpu_gemm(dual_data+n*dual_dim_, weight, deconv_res_data+n*primal_dim_);
  }

  
  Dtype* primal_before_proj_data = primal_before_proj_.mutable_cpu_data();
  const Dtype* deconv_res_data_const = deconv_res_.cpu_data();
  const int count = primal_before_proj_.count();
  caffe_copy<Dtype>(count, deconv_res_data_const, primal_before_proj_data);
  // add datacost term
  caffe_axpy<Dtype>(count, Dtype(1.), bottom[3]->cpu_data(), primal_before_proj_data);

  // add lagrangian
  const int spatial_count = primal_before_proj_.count(channel_axis_+1);
  const Dtype* lagrangian_data = bottom[2]->cpu_data();
  for(int n = 0; n < num_; n++){
    for(int c = 0; c < primal_before_proj_.shape(channel_axis_); c++){   
      caffe_axpy<Dtype>(spatial_count, Dtype(1.), lagrangian_data + n * spatial_count, 
        primal_before_proj_data+ n*primal_dim_ + c * spatial_count);
    }
  }


  caffe_scal<Dtype>(count, Dtype(-1. * tau_), primal_before_proj_data);
  // add previous primal data
  caffe_axpy<Dtype>(count, Dtype(1.), bottom[0]->cpu_data(), primal_before_proj_data);

  // projection to [0, 1]
  primal_projection_layer_->Forward(primal_projection_bottom_vecs_, top); 


}

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  vector<bool> down;
  down.clear();
  down.push_back(true);
  primal_projection_layer_->Backward(top, down, primal_projection_bottom_vecs_);
  const Dtype* primal_before_proj_diff = primal_before_proj_.cpu_diff();



  // compute gradient with respect to input previous primal data  
  const int count = bottom[0]->count();
  if(propagate_down[0]){
    Dtype* primal_diff = bottom[0]->mutable_cpu_diff(); 
    caffe_copy<Dtype>(count, primal_before_proj_diff, primal_diff); 
  }
  

  // compute gradient with respect to data cost term
  if(propagate_down[3]){
    Dtype* datacost_diff = bottom[3]->mutable_cpu_diff();
    caffe_set<Dtype>(count, Dtype(0), datacost_diff);
    caffe_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, datacost_diff);
  }
  
  // compute gradient with respect to lagrangian term

  if(propagate_down[2]){
    const int spatial_count = primal_before_proj_.count(channel_axis_+1);
    Dtype* lagrangian_diff = bottom[2]->mutable_cpu_diff();
    caffe_set<Dtype>(bottom[2]->count(), Dtype(0.), lagrangian_diff);
    for(int n = 0; n < num_; n++){
      for(int c = 0; c < primal_before_proj_.shape(channel_axis_); c++){     
        caffe_axpy<Dtype>(spatial_count, Dtype(-1.*tau_), 
          primal_before_proj_diff + n * primal_dim_ + c * spatial_count,
          lagrangian_diff + n * spatial_count);
      }
    }
  }

  // compute gradient with respect to dual term and weight.
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  if(propagate_down[1] || this->param_propagate_down_[0]){


    Dtype* deconv_res_diff = deconv_res_.mutable_cpu_diff();
    caffe_set<Dtype>(count, Dtype(0.), deconv_res_diff);
    caffe_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, 
      deconv_res_diff); 

    Dtype* dual_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* dual_data = bottom[1]->cpu_data();
    for(int n = 0; n < num_; n++){
      if(this->param_propagate_down_[0]){
        weight_cpu_gemm(deconv_res_diff + n * primal_dim_,
          dual_data + n * dual_dim_, weight_diff);
      }

      if(propagate_down[1]){
        forward_cpu_gemm(deconv_res_diff + n * primal_dim_, weight,
          dual_diff + n * dual_dim_, this->param_propagate_down_[0]);
      }
      
    }
  }  
}




template <typename Dtype>
void PrimalUpdateLayer<Dtype>::check_primal_shape(){

  const int* kernel_shape_data = kernel_shape_.cpu_data();
  const int* stride_data = stride_.cpu_data();
  const int* start_pad_data = start_pad_.cpu_data();
  const int* end_pad_data = end_pad_.cpu_data();

  for (int i = 0; i < num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = dual_spatial_shape(i);
    const int output_dim = stride_data[i] * (input_dim - 1)
        + kernel_shape_data[i] - start_pad_data[i] - end_pad_data[i];
    CHECK_EQ(primal_spatial_shape(i), output_dim)
      << "Deconvolution with dual size should be compatible with primal size.";
  }

  CHECK_EQ((*primal_shape_)[channel_axis_], num_output_)
    << "Output channels should be the same as primal channels.";

}

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
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
void PrimalUpdateLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
    conv_out_spatial_dim_, conv_out_channels_, (Dtype)1., 
    weights, output, (Dtype)0., col_buff);
  
  conv_col2im_cpu(col_buff, input);
}

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
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
void PrimalUpdateLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
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
void PrimalUpdateLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
    conv_out_spatial_dim_, conv_out_channels_, (Dtype)1., 
    weights, output, (Dtype)0., col_buff);
  
  conv_col2im_gpu(col_buff, input);
}

template <typename Dtype>
void PrimalUpdateLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
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
STUB_GPU(PrimalUpdateLayer);
#endif

INSTANTIATE_CLASS(PrimalUpdateLayer);

}  // namespace caffe
