#ifndef CAFFE_PRIMAL_PROJECTION_LAYER_HPP_
#define CAFFE_PRIMAL_PROJECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief y = min(1, max(x,0))
 */
template <typename Dtype>
class PrimalProjectionLayer : public NeuronLayer<Dtype> {
 public:
 
  explicit PrimalProjectionLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "PrimalProjection"; }

 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_PRIMAL_PROJECTION_LAYER_HPP_
