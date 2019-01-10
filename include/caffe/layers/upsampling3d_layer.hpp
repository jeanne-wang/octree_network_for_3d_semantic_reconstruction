#ifndef CAFFE_UPSAMPLING3D_LAYER_HPP_
#define CAFFE_UPSAMPLING3D_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Upsampling a Blob along specified dimensions.
 */
template <typename Dtype>
class UpSampling3DLayer : public Layer<Dtype> {
 public:
  explicit UpSampling3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "UpSampling3D"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int h_rep_;
  int w_rep_;
  int s_rep_;
  int first_spatial_axis_;

  Blob<Dtype> h_rep_blob_;
  Blob<Dtype> w_rep_blob_;
};

}  // namespace caffe

#endif  // CAFFE_UPSAMPLING3D_LAYER_HPP_
