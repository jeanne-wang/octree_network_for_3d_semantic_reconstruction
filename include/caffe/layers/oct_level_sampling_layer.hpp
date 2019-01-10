#ifndef CAFFE_OCT_LEVEL_SAMPLING_LAYER_HPP_
#define CAFFE_OCT_LEVEL_SAMPLING_LAYER_HPP_

#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "image_tree_tools/octree.h"
namespace caffe {

/// bottom[0]: octree input 
/// ref_key_layer: include the <key, pos> pair of octree cells to be obtained from input octree
/// input_key_layer, include the <key, pos> pair of input octree


template <typename Dtype>
class OctLevelSamplingLayer : public Layer<Dtype> {

public:
	explicit OctLevelSamplingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "OctLevelSampling"; }
	virtual inline int ExactNumBottomBlobs() const { return 2; }
  

protected:
  	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
  	Dtype get_octree_cell_value(GeneralOctree<int> &input_keys_octree, 
    	const Dtype* input, unsigned int key);

  	int input_max_level_;

};

}  // namespace caffe

#endif  //CAFFE_OCT_LEVEL_SAMPLING_LAYER_HPP_
