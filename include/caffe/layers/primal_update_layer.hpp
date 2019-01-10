#ifndef CAFFE_PRIMAL_UPDATE_LAYER_HPP_
#define CAFFE_PRIMAL_UPDATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/primal_projection_layer.hpp"

namespace caffe {

template <typename Dtype>
class PrimalUpdateLayer : public Layer<Dtype> {
 public:
  explicit PrimalUpdateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "PrimalUpdate"; }
  virtual inline int ExactNumBottomBlobs() const {return 4; }
  virtual inline int ExactNumTopBlobs() const {return 1;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_im2col = false);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
#endif

  void check_primal_shape();
  inline int dual_spatial_shape(int i){
    return (*dual_shape_)[channel_axis_ + i + 1];
  }
  inline int primal_spatial_shape(int i){
    return (*primal_shape_)[channel_axis_ + i + 1];
  }

  //// @ variables used in deconvolution
  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of padding from start.
  Blob<int> start_pad_;
  /// @brief The spatial dimensions of padding in the end.
  Blob<int> end_pad_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;

  Blob<int> dilation_;
  
  const vector<int>* dual_shape_;
  const vector<int>* primal_shape_;

  int conv_out_channels_;
  int conv_in_channels_;

  int num_;  // number of data, i.e., batch_size
  int dual_dim_;
  int primal_dim_;
  int num_spatial_axes_; // number of spatial axes in dual variable
  int channel_axis_;    // chanel axis in dual variable
  int channels_;      // chanels in dual variale, i.e., 3*|C|
  int num_output_;  // deconvolution output channels, i.e., |C|

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  
  int conv_out_spatial_dim_;
  int kernel_dim_; 

  Blob<Dtype> col_buffer_;

  
  Blob<Dtype> deconv_res_;
  Blob<Dtype> primal_before_proj_;

  //// parameters used in the primal update formula
  float tau_;
  // utility layer to project the primal to [0,1] itnerval.
  shared_ptr<PrimalProjectionLayer<Dtype> > primal_projection_layer_;
  vector<Blob<Dtype>*> primal_projection_bottom_vecs_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {

    im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
      col_buffer_shape_.data(), kernel_shape_.cpu_data(),
      start_pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), col_buff);
    
  }
  inline void conv_col2im_cpu(const Dtype* col_buff, Dtype* data) {

    col2im_nd_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
      col_buffer_shape_.data(), kernel_shape_.cpu_data(),
      start_pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(), data);
  }
#ifndef CPU_ONLY
  inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    
    im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
      conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
      kernel_shape_.gpu_data(), start_pad_.gpu_data(),
      stride_.gpu_data(), dilation_.gpu_data(), col_buff);
  }

  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
      conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
      kernel_shape_.gpu_data(), start_pad_.gpu_data(), stride_.gpu_data(),
      dilation_.gpu_data(), data);
  }
#endif


};

}  // namespace caffe

#endif  // CAFFE_PRIMAL_UPDATE_LAYER_HPP_
