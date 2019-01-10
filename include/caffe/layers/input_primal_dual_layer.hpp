#ifndef INPUT_PRIMAL_DUAL_LAYER_HPP_
#define INPUT_PRIMAL_DUAL_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class InputPrimalDualLayer : public Layer<Dtype> {
 public:
  explicit InputPrimalDualLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InputPrimalDual"; }
  virtual inline int ExactNumTopBlobs() const { return 4; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
   

   int out_height_;
   int out_width_;
   int out_depth_;
   int num_classes_;
   int batch_size_;

};

}  // namespace caffe

#endif  // INPUT_PRIMAL_DUAL_LAYER_HPP_
