#ifndef CAFFE_SEG_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_SEG_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class SegCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
   
  explicit SegCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SegCrossEntropyLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  


  int num_;
  int num_classes_;
  int num_pixels_;

  int unknown_label_;
  int ignore_label_;

};

}  // namespace caffe

#endif  // CAFFE_SEG_CROSS_ENTROPY_LOSS_LAYER_HPP_
