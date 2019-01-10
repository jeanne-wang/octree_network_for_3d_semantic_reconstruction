#ifndef CAFFE_DUAL_PROJECTION_LAYER_HPP_
#define CAFFE_DUAL_PROJECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief project x such that ||x|| <= 1
 */
template <typename Dtype>
class DualProjectionLayer : public NeuronLayer<Dtype> {
 public:
 
  explicit DualProjectionLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "DualProjection"; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> dual_norm_; // the norm of each voxel along three channels in the dual variable 
  Blob<Dtype> dual_norm_proj_; // dual_norm_proj = maximum(dual_norm_, 1.0)
  int num_;
  int dual_dim_; // dual_dim_ = dual_spatial_count_* num_classes_*3
  int dual_norm_dim_;
  int dual_spatial_count_; // dual_spatial_count_ = num_rows_* num_cols_* num_slices_;
  int num_classes_;
  int num_rows_;
  int num_cols_;
  int num_slices_;

};

}  // namespace caffe

#endif  // CAFFE_DUAL_PROJECTION_LAYER_HPP_
