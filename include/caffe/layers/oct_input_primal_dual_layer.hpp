#ifndef CAFFE_OCT_INPUT_PRIMAL_DUAL_LAYER_HPP_
#define CAFFE_OCT_INPUT_PRIMAL_DUAL_LAYER_HPP_

#include <vector>
#include "caffe/layers/oct_layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class OctInputPrimalDualLayer : public OctLayer<Dtype> {
 public:
  explicit OctInputPrimalDualLayer(const LayerParameter& param)
      : OctLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctInputPrimalDual"; }
  /*virtual inline int ExactNumTopBlobs() const { return 4; }*/


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
   

   int out_height_;
   int out_width_;
   int out_depth_;
   int num_classes_;
   int batch_size_;

};

}  // namespace caffe

#endif  // CAFFE_OCT_INPUT_PRIMAL_DUAL_LAYER_HPP_
