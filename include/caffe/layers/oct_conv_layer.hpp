#ifndef CAFFE_OCT_CONV_LAYER_HPP_
#define CAFFE_OCT_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/col_buffer.hpp" // for temporary col buffer

namespace caffe {

template <typename Dtype>
class OctConvLayer : public Layer<Dtype> {
public:
  	explicit  OctConvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);
  	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);

  	virtual inline const char* type() const { return "OctConv"; }
    virtual inline int ExactNumBottomBlobs() const {return 4;}


protected:
  	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);
  	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);
  	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  	void octree2col_cpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
  		Dtype* col_buff, const int num_elements);
  	void col2octree_cpu(const Dtype* col_buff, const Dtype* neighbors_data, 
  		Dtype* bottom_diff, const int num_elements);


  	void octree2col_gpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
        Dtype* col_buff, const Dtype* num_elements);
  	void col2octree_gpu(const Dtype* col_buff, const Dtype* neighbor_of_data, 
        Dtype* bottom_diff, const Dtype* num_elements);

  	vector<int> _weight_shape;
  	vector<int> _bias_shape;
  	vector<int> _col_buffer_shape;

  	/*Blob<Dtype> _col_buffer; */// use temporary col buffer in caffe/util/col_buffer.hpp to save memory
  	Blob<Dtype> _bias_multiplier;

  	int _num_input_pixels;
  	int _num_output_pixels;
  	int _num_output_channels;
  	int _num_input_channels;
  	int _batch_size;
  	int _filter_size;
  	int _bottom_dim;
  	int _top_dim;

};

}  // namespace caffe

#endif  // CAFFE_OCT_CONV_LAYER_HPP_
