#ifndef OCT_OUTPUT_LAYER_HPP_
#define OCT_OUTPUT_LAYER_HPP_

#include "caffe/layers/oct_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/// Assembles an octree based on the network predictions,
template <typename Dtype>
class OctOutputLayer : public OctLayer<Dtype> {
 public:
  explicit OctOutputLayer(const LayerParameter& param)
      : OctLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctOutput"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void compute_pixel_propagation(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);


  bool _done_initial_reshape;
  int _num_output_pixels;
  int _num_levels;
  int _min_level;

};

}  // namespace caffe

#endif  //OCT_OUTPUT_LAYER_HPP_
