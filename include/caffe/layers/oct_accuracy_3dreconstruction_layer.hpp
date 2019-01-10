#ifndef CAFFE_OCT_ACCURACY_3DRECONSTRUCTION_LAYER_HPP_
#define CAFFE_OCT_ACCURACY_3DRECONSTRUCTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "image_tree_tools/octree.h"
namespace caffe {


template <typename Dtype>
class OctAccuracy3DReconstructionLayer : public Layer<Dtype> {
 public:
 
  explicit OctAccuracy3DReconstructionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctAccuracy3DReconstruction"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 3; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  void arg_max(const Blob<Dtype>* input, int* output);
  int get_octree_cell_match_count(GeneralOctree<int> &pr_keys_octree, const int* labels_pred_data, unsigned int key, int label);

  
  bool has_unknown_label_;
  int freespace_label_;
  int unknown_label_;
  int max_level_;

  int batch_size_;
  int num_classes_;
  int num_elements_true_;
  int num_elements_pred_;

  Blob<unsigned int> occupied_mask_;
  Blob<int> labels_true_;
  Blob<int> labels_pred_;
};

}  // namespace caffe

#endif  // CAFFE_OCT_ACCURACY_3DRECONSTRUCTION_LAYER_HPP_
