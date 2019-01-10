#ifndef CAFFE_OCT_SEG_LOSS_PREP_LAYER_HPP_
#define CAFFE_OCT_SEG_LOSS_PREP_LAYER_HPP_

#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "image_tree_tools/octree.h"
namespace caffe {

/// bottom[0]: octree input (gt)
/// bottom[1]: ref input
/// ref_key_layer: include the <key, pos> pair of octree cells to be obtained from input octree
/// input_key_layer, include the <key, pos> pair of input octree


template <typename Dtype>
class OctSegLossPrepLayer : public Layer<Dtype> {
 public:
  explicit OctSegLossPrepLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctSegLossPrep"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool has_ignore_label_;
  int ignore_label_;

};

}  // namespace caffe

#endif  //CAFFE_OCT_SEG_LOSS_PREP_LAYER_HPP_
