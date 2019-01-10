#ifndef CAFFE_OCT_UPSAMPLING_LAYER_HPP_
#define CAFFE_OCT_UPSAMPLING_LAYER_HPP_

#include "caffe/layers/oct_layer.hpp"

namespace caffe {

template <typename Dtype>
class OctUpSamplingLayer : public OctLayer<Dtype> {
public:
	explicit OctUpSamplingLayer(const LayerParameter& param)
      : OctLayer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  	virtual inline const char* type() const { return "OctUpSampling"; }
	virtual inline int ExactNumBottomBlobs() const {return 4; }


protected:
  	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  	void propagate_keys_cpu();
  
 
  	int _num_input_pixels;
  	int _num_output_pixels;
  	int _batch_size;

};

}  // namespace caffe

#endif  // CAFFE_OCT_UPSAMPLING_LAYER_HPP_
