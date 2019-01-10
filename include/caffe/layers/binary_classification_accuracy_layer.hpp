#ifndef CAFFE_BINARY_CLASSIFICATION_ACCURACY_LAYER_HPP_
#define CAFFE_BINARY_CLASSIFICATION_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the classification accuracy for a one-of-many
 *        classification task.
 */
template <typename Dtype>
class BinaryClassificationAccuracyLayer : public Layer<Dtype> {
 public:
 
  explicit BinaryClassificationAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BinaryClassificationAccuracy"; }
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

};

}  // namespace caffe

#endif  // CAFFE_BINARY_CLASSIFICATION_ACCURACY_LAYER_HPP_
