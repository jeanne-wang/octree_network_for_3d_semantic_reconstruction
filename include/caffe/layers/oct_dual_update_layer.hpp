#ifndef CAFFE_OCT_DUAL_UPDATE_LAYER_HPP_
#define CAFFE_OCT_DUAL_UPDATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/oct_dual_projection_layer.hpp"
#include "caffe/util/col_buffer.hpp" // for temporary col buffer

namespace caffe {

template <typename Dtype>
class OctDualUpdateLayer : public Layer<Dtype> {
 public:
  explicit OctDualUpdateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OctDualUpdate"; }
  /*virtual inline int ExactNumBottomBlobs() const {return 2; }*/ // add two extra bottoms for input for octree2col and col2octree.
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

  /*void im2col_octree_cpu(int batch_ind, const Blob<Dtype>& bottom);
  void col2im_octree_cpu(int batch_ind, Blob<Dtype>& bottom);*/

  void octree2col_cpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
                      Dtype* col_buff, const int num_elements);
  void col2octree_cpu(const Dtype* col_buff, const Dtype* neighbors_data, 
                      Dtype* bottom_diff, const int num_elements);


  void octree2col_gpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
                      Dtype* col_buff, const Dtype* num_elements);
  void col2octree_gpu(const Dtype* col_buff, const Dtype* neighbor_of_data, 
                      Dtype* bottom_diff, const Dtype* num_elements);

  vector<int> _weight_shape;
  vector<int> _col_buffer_shape;
  //Blob<Dtype> _col_buffer; // use temporary col buffer in caffe/util/col_buffer.hpp to save memory


  int _num_input_pixels;
  int _num_output_pixels;
  int _num_output_channels;
  int _num_input_channels;
  int _batch_size;
  int _filter_size;
  int _primal_dim;
  int _dual_dim;

  
  Blob<Dtype> conv_res_;
  Blob<Dtype> dual_before_proj_;

  //// parameters used in the primal update formula
  float sigma_;
  // utility layer to project the primal to [0,1] itnerval.
  shared_ptr<OctDualProjectionLayer<Dtype> > dual_projection_layer_;
  vector<Blob<Dtype>*> dual_projection_bottom_vecs_;


};

}  // namespace caffe

#endif  // CAFFE_OCT_DUAL_UPDATE_LAYER_HPP_
