#ifndef CAFFE_OCT_PRIMAL_DUAL_CROSS_ENTROPY_LOSS_LAYER_HPP_
#define CAFFE_OCT_PRIMAL_DUAL_CROSS_ENTROPY_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/clip_by_value_layer.hpp"

namespace caffe {

template <typename Dtype>
class OctPrimalDualCrossEntropyLossLayer : public LossLayer<Dtype> {
 public:
   
  explicit OctPrimalDualCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctPrimalDualCrossEntropyLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  /*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);*/
 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);*/



  /// The internal SoftmaxLayer used to map groundtruth to a distribution.
  shared_ptr<Layer<Dtype> > gt_softmax_layer_;
  shared_ptr<Layer<Dtype> > clip_by_value_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> gt_scale_;
  Blob<Dtype> gt_softmax_;
  Blob<Dtype> gt_softmax_clip_;
  Blob<Dtype> pred_softmax_clip_;
  Blob<Dtype> gt_log_pred_;

  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
  vector<Blob<Dtype>*> clip_by_value_bottom_vec_;
  vector<Blob<Dtype>*> clip_by_value_top_vec_;


  int num_;
  int num_classes_;
  int num_pixels_;
  int free_count_;
  int occu_count_;

  float softmax_scale_;
  float clip_epsilon_;
  bool freespace_weighted_;
  bool unknown_weighted_;
};

}  // namespace caffe

#endif  // CAFFE_OCT_PRIMAL_DUAL_CROSS_ENTROPY_LOSS_LAYER_HPP_

