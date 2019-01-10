#ifndef CAFFE_DENSE_CONVERT_LAYER_HPP_
#define CAFFE_DENSE_CONVERT_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class DenseConvertLayer : public Layer<Dtype> {

public:
  	explicit DenseConvertLayer(const LayerParameter& param)
      	: Layer<Dtype>(param) {}
 	 virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);
 	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);

  	virtual inline const char* type() const { return "DenseConvert"; }
  	virtual inline int ExactNumBottomBlobs() const { return 1; }


protected:
  	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);
  	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
	int max_level_;
};

}  // namespace caffe

#endif  // CAFFE_DENSE_CONVERT_LAYER_HPP_
