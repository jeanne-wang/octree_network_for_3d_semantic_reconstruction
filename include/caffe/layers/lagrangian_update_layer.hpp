#ifndef CAFFE_LAGRANGIAN_UPDATE_LAYER_HPP_
#define CAFFE_LAGRANGIAN_UPDATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LagrangianUpdateLayer : public Layer<Dtype> {
 public:
  explicit LagrangianUpdateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LagrangianUpdate"; }
  virtual inline int ExactNumBottomBlobs() const {return 2; }
  virtual inline int ExactNumTopBlobs() const {return 1;}

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  
  //// parameters used in the primal update formula
  float sigma_;
  

};

}  // namespace caffe

#endif  // CAFFE_LAGRANGIAN_UPDATE_LAYER_HPP_
