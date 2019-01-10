#ifndef CAFFE_OCT_PROP_LOSS_PREP_LAYER_HPP_
#define CAFFE_OCT_PROP_LOSS_PREP_LAYER_HPP_

#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
namespace caffe {

/// bottom[0]: octree cell features
/// pred_key_layer: include the <key, pos> pair of octree in bottom[0]
/// gt_key_layer, include the <key, pos> pair of groundtruth octree


template <typename Dtype>
class OctPropLossPrepLayer : public Layer<Dtype> {
 public:
  explicit OctPropLossPrepLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctPropLossPrep"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

};

}  // namespace caffe

#endif  //CAFFE_OCT_PROP_LOSS_PREP_LAYER_HPP_
