#ifndef CAFFE_OCT_PRIMAL_DUAL_WEIGHTED_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_OCT_PRIMAL_DUAL_WEIGHTED_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/clip_by_value_layer.hpp"

namespace caffe {

template <typename Dtype>
class OctPrimalDualWeightedCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
   
  explicit OctPrimalDualWeightedCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctPrimalDualWeightedCrossEntropyLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  shared_ptr<Layer<Dtype> > clip_by_value_layer_;
  vector<Blob<Dtype>*> clip_by_value_bottom_vec_;
  vector<Blob<Dtype>*> clip_by_value_top_vec_;
  Blob<Dtype> gt_clip_;
  Blob<Dtype> pred_clip_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> gt_log_pred_;
  Blob<Dtype> weights_;


  int num_;
  int num_classes_;
  int num_pixels_;

  int unknown_label_;
  float clip_epsilon_;

};

}  // namespace caffe

#endif  // CAFFE_PRIMAL_DUAL_CROSS_ENTROPY_LOSS_LAYER_HPP_
