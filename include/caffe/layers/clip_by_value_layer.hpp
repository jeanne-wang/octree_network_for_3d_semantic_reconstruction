#ifndef CAFFE_CLIP_BY_VALUE_LAYER_HPP_
#define CAFFE_CLIP_BY_VALUE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

/**
 * @brief y = min(max_val, max(min_val,0))
 */
template <typename Dtype>
class ClipByValueLayer : public Layer<Dtype> {
 public:
 
  explicit ClipByValueLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "ClipByValue"; }
  virtual inline int MinBottomBlobs() const { return 1; }


 protected:
  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  float min_val_;
  float max_val_;
};

}  // namespace caffe

#endif  // CAFFE_CLIP_BY_VALUE_LAYER_HPP_
