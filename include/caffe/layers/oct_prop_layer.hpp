#ifndef CAFFE_OCT_PROP_LAYER_HPP_
#define CAFFE_OCT_PROP_LAYER_HPP_

#include "caffe/layers/oct_layer.hpp"

#include <set>

namespace caffe {

using namespace std;

/// A layer for propagating the features of the cells that are "mixed".
/// Information about the state of a cell can either be taken from the ground truth tree, or from the prediction.

/// All input variables (u, u_, m, l) must be aligned based on the key value, and they will be propagated in the same way.
/// The propagated variables are also aligned, so they share same _octree_keys

/// bottom[0]: u
/// bottom[1]: u_
/// bottom[2]: m 
/// bottom[3]: l

/// bottom[4]: groundtruth or predicted probabilities for whether to propagate or not.

template <typename Dtype>
class OctPropLayer : public OctLayer<Dtype> {
 public:
  explicit OctPropLayer(const LayerParameter& param)
      : OctLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctProp"; }
  virtual inline int ExactNumBottomBlobs() const {return 5; }
  virtual inline int ExactNumTopBlobs() const {return 4; }

 protected:

  void compute_pixel_propagation(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  bool _done_initial_reshape;
  int _num_output_pixels;

};

}  // namespace caffe

#endif  // CAFFE_OCT_PROP_LAYER_HPP_
