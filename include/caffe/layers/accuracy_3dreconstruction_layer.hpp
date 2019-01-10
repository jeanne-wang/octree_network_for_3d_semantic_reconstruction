#ifndef CAFFE_ACCURACY_3DRECONSTRUCTION_LAYER_HPP_
#define CAFFE_ACCURACY_3DRECONSTRUCTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class Accuracy3DReconstructionLayer : public Layer<Dtype> {
 public:
 
  explicit Accuracy3DReconstructionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Accuracy3DReconstruction"; }
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

  int num_;
  int num_classes_;
  int num_rows_;
  int num_cols_;
  int num_slices_;
  int freespace_label_;
  int unknown_label_;
  bool has_unknown_label_;

  Blob<unsigned int> occupied_mask_;
  Blob<int> labels_true_;
  Blob<int> labels_pred_;
};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_3DRECONSTRUCTION_LAYER_HPP_
