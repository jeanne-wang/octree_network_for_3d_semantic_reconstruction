#ifndef CAFFE_OCT_PRIMAL_UPDATE_LAYER_HPP_
#define CAFFE_OCT_PRIMAL_UPDATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/primal_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/col_buffer.hpp" // for temporary col buffer

namespace caffe {

template <typename Dtype>
class OctPrimalUpdateLayer : public Layer<Dtype> {
 public:
  explicit OctPrimalUpdateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctPrimalUpdate"; }
  /*virtual inline int ExactNumBottomBlobs() const {return 4; }*/
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

 
  void col2octree_cpu(const Dtype* col_buff, const Dtype* neighbors_data, 
                      Dtype* top_data, const int num_elements);
  void octree2col_cpu(const Dtype* top_diff, const Dtype* neighbors_data, 
                      Dtype* col_buff, const int num_elements);
  
  void col2octree_gpu(const Dtype* col_buff, const Dtype* neighbor_of_data, 
                      Dtype* top_data, const Dtype* num_elements);
  void octree2col_gpu(const Dtype* top_diff, const Dtype* neighbors_data, 
                      Dtype* col_buff, const Dtype* num_elements);
 

  

  vector<int> _weight_shape;
  vector<int> _col_buffer_shape;
  //Blob<Dtype> _col_buffer;


  int _num_input_pixels;
  int _num_output_pixels;
  int _num_output_channels;
  int _num_input_channels;
  int _batch_size;
  int _filter_size;
  int _primal_dim;
  int _dual_dim;

  
  Blob<Dtype> deconv_res_;
  Blob<Dtype> primal_before_proj_;

  //// parameters used in the primal update formula
  float tau_;
  // utility layer to project the primal to [0,1] itnerval.
  shared_ptr<PrimalProjectionLayer<Dtype> > primal_projection_layer_;
  vector<Blob<Dtype>*> primal_projection_bottom_vecs_;


};

}  // namespace caffe

#endif  // CAFFE_OCT_PRIMAL_UPDATE_LAYER_HPP_
