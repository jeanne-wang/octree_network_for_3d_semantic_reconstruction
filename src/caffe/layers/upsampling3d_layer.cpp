#include <vector>

#include "caffe/layers/upsampling3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void UpSampling3DLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const UpSampling3DParameter& upsampling3d_param = this->layer_param_.upsampling3d_param();
  CHECK(upsampling3d_param.has_h_rep() && upsampling3d_param.has_w_rep() && upsampling3d_param.has_s_rep())
    << "factor of upsampling along each 3d dimension must be specified.";
  h_rep_ = upsampling3d_param.h_rep();
  w_rep_ = upsampling3d_param.w_rep();
  s_rep_ = upsampling3d_param.s_rep();
}


template <typename Dtype>
void UpSampling3DLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  CHECK_GE(bottom[0]->num_axes(), 4)
    << "The number of dimensions of the blob must be at least 4";
 
  first_spatial_axis_ = (bottom[0]->num_axes() == 5? 2 : 1); // in case of num_axes() == 4, there is no channel axis.
  vector<int> rep_blob_shape = bottom[0]->shape();
  rep_blob_shape[first_spatial_axis_] = bottom[0]->shape(first_spatial_axis_) * h_rep_;
  h_rep_blob_.Reshape(rep_blob_shape);

  rep_blob_shape[first_spatial_axis_+1] = rep_blob_shape[first_spatial_axis_+1] * w_rep_;
  w_rep_blob_.Reshape(rep_blob_shape);

  rep_blob_shape[first_spatial_axis_+2] = rep_blob_shape[first_spatial_axis_+2] * s_rep_;
  top[0]->Reshape(rep_blob_shape);

}

template <typename Dtype>
void UpSampling3DLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* h_rep_blob_data = h_rep_blob_.mutable_cpu_data();
  Dtype* w_rep_blob_data = w_rep_blob_.mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // upsampling height axis
  int outer_dim = bottom[0]->count(0, first_spatial_axis_);
  int inner_dim = bottom[0]->count(first_spatial_axis_+1);
  for(int i = 0; i < outer_dim; ++i){
    for(int j = 0; j < bottom[0]->shape(first_spatial_axis_); j++){
      for(int t = 0; t < h_rep_; t++){
        caffe_copy<Dtype>(inner_dim, bottom_data, h_rep_blob_data);
        h_rep_blob_data += inner_dim;
      }
      bottom_data += inner_dim;
    }
  }

  // upsampling width axis
  const Dtype* h_rep_blob_data_const = h_rep_blob_.cpu_data();
  outer_dim = h_rep_blob_.count(0, first_spatial_axis_+1);
  inner_dim = h_rep_blob_.count(first_spatial_axis_+2);
  for(int i = 0; i < outer_dim; ++i){
    for(int j = 0; j < h_rep_blob_.shape(first_spatial_axis_+1); j++){
      for(int t = 0; t < w_rep_; t++){
        caffe_copy<Dtype>(inner_dim, h_rep_blob_data_const, w_rep_blob_data);
        w_rep_blob_data += inner_dim;
      }
      h_rep_blob_data_const += inner_dim;
    }
  }

  // upsampling depth axis
  const Dtype* w_rep_blob_data_const = w_rep_blob_.cpu_data();
  outer_dim = w_rep_blob_.count(0, first_spatial_axis_+2);
  inner_dim = 1;
  for(int i = 0; i < outer_dim; ++i){
    for(int j = 0; j < w_rep_blob_.shape(first_spatial_axis_+2); j++){
      for(int t = 0; t < s_rep_; t++){
        caffe_copy<Dtype>(inner_dim, w_rep_blob_data_const, top_data);
        top_data += inner_dim;
      }
      w_rep_blob_data_const += inner_dim;
    }
  }
}

template <typename Dtype>
void UpSampling3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* w_rep_blob_diff = w_rep_blob_.mutable_cpu_diff();
  Dtype* h_rep_blob_diff = h_rep_blob_.mutable_cpu_diff();

  // upsampling depth axis
  caffe_set<Dtype>(w_rep_blob_.count(0), Dtype(0), w_rep_blob_diff);
  int outer_dim = w_rep_blob_.count(0, first_spatial_axis_+2);
  int inner_dim = 1;
  for(int i = 0; i < outer_dim; ++i){
    for(int j = 0; j < w_rep_blob_.shape(first_spatial_axis_+2); j++){
      for(int t = 0; t < s_rep_; t++){
        caffe_axpy<Dtype>(inner_dim, Dtype(1.), top_diff, w_rep_blob_diff);
        top_diff += inner_dim;
      }
      w_rep_blob_diff += inner_dim;
    }
  }

  const Dtype* w_rep_blob_diff_const = w_rep_blob_.cpu_diff();
  caffe_set<Dtype>(h_rep_blob_.count(0), Dtype(0), h_rep_blob_diff);
  outer_dim = h_rep_blob_.count(0, first_spatial_axis_+1);
  inner_dim = h_rep_blob_.count(first_spatial_axis_+2);
  for(int i = 0; i < outer_dim; ++i){
    for(int j = 0; j < h_rep_blob_.shape(first_spatial_axis_+1); j++){
      for(int t = 0; t < w_rep_; t++){
        caffe_axpy<Dtype>(inner_dim, Dtype(1.), w_rep_blob_diff_const, h_rep_blob_diff);
        w_rep_blob_diff_const += inner_dim;
      }
      h_rep_blob_diff += inner_dim;
    }
  }

  const Dtype* h_rep_blob_diff_const = h_rep_blob_.cpu_diff();
  caffe_set<Dtype>(bottom[0]->count(0), Dtype(0), bottom_diff);
  outer_dim = bottom[0]->count(0, first_spatial_axis_);
  inner_dim = bottom[0]->count(first_spatial_axis_+1);
  for(int i = 0; i < outer_dim; ++i){
    for(int j = 0; j < bottom[0]->shape(first_spatial_axis_); j++){
      for(int t = 0; t < h_rep_; t++){
        caffe_axpy<Dtype>(inner_dim, Dtype(1.), h_rep_blob_diff_const, bottom_diff);
        h_rep_blob_diff_const += inner_dim;
      }
      bottom_diff += inner_dim;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(UpSampling3DLayer);
#endif

INSTANTIATE_CLASS(UpSampling3DLayer);
REGISTER_LAYER_CLASS(UpSampling3D);

}  // namespace caffe
