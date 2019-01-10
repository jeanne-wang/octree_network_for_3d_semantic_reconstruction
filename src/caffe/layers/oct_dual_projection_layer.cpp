#include <algorithm>
#include <vector>

#include "caffe/layers/oct_dual_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
void OctDualProjectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
  CHECK_EQ(bottom[0]->num_axes(), 3)
    << "The dimension of octree dual variable must be 3.";

  num_ = bottom[0]->shape(0);
  int num_channels = bottom[0]->shape(1);
  CHECK_EQ(num_channels%3, 0)
    << "The number of channles must be 3 times of num_classes.";

  num_classes_ = num_channels/3;
  num_input_pixels_ = bottom[0]->shape(2);  
  dual_dim_ = bottom[0]->count(1);


  vector<int> dual_norm_shape(3);
  dual_norm_shape[0] = num_;
  dual_norm_shape[1] = num_classes_;
  dual_norm_shape[2] = num_input_pixels_;
  dual_norm_.Reshape(dual_norm_shape);
  dual_norm_proj_.Reshape(dual_norm_shape);
  dual_norm_dim_ =  dual_dim_ /3;

}


template <typename Dtype>
void OctDualProjectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* dual_norm_data = dual_norm_.mutable_cpu_data();
  // compute dual_norm
  for(int n = 0; n < num_; n++){
    for(int c = 0;  c < num_classes_; c++){
      for(int i = 0; i < num_input_pixels_; i++){
        int index = n*dual_dim_+ c* 3* num_input_pixels_ +  i;
        Dtype norm =  0;
        for(int t = 0; t < 3; t++){
          norm += bottom_data[index] * bottom_data[index];
          index += num_input_pixels_;
        }

        *(dual_norm_data++) = sqrt(norm);
      }
    }
  }

  // dual_norm_proj = maximum(dual_norm_, 1)
  const int count = dual_norm_.count();
  const Dtype* dual_norm_data_const = dual_norm_.cpu_data();
  Dtype* dual_norm_proj_data = dual_norm_proj_.mutable_cpu_data();
  for(int i = 0; i < count; i++){
    //dual_norm_proj_data[i] = dual_norm_data_const[i]; // for test
    dual_norm_proj_data[i] = std::max(dual_norm_data_const[i], Dtype(1)); 
  }

  // normalization
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* dual_norm_proj_data_const = dual_norm_proj_.cpu_data();
  for(int n = 0; n < num_; n++){
    for(int c = 0;  c < num_classes_; c++){
      int dual_offset = n * dual_dim_ + c * 3* num_input_pixels_;
      int dual_norm_proj_offset = n * dual_norm_dim_ + c * num_input_pixels_;
      for(int t = 0; t < 3; t++){
        caffe_div<Dtype>(num_input_pixels_, bottom_data + dual_offset, dual_norm_proj_data_const + dual_norm_proj_offset,
          top_data + dual_offset);
        dual_offset += num_input_pixels_;
      }
    }
  }
}

template <typename Dtype>
void OctDualProjectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* dual_norm_proj_data = dual_norm_proj_.cpu_data();
    const Dtype* dual_norm_data = dual_norm_.cpu_data();

    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* dual_norm_diff = dual_norm_.mutable_cpu_diff();
    Dtype* dual_norm_proj_diff = dual_norm_proj_.mutable_cpu_diff();


    memset(dual_norm_diff, 0, sizeof(Dtype)*dual_norm_.count(0));

    for(int n = 0; n < num_; n++){
      for(int c = 0;  c < num_classes_; c++){
        int dual_offset = n * dual_dim_ + c * 3* num_input_pixels_;
        int dual_norm_proj_offset = n * dual_norm_dim_ + c * num_input_pixels_;
        for(int t = 0; t < 3; t++){
          // gradient to the bottom diff
          caffe_div<Dtype>(num_input_pixels_, top_diff+ dual_offset, dual_norm_proj_data + dual_norm_proj_offset,
            bottom_diff + dual_offset);
          // gradient to the normalization denominatior

          caffe_mul<Dtype>(num_input_pixels_, top_diff + dual_offset, top_data + dual_offset,
            dual_norm_proj_diff + dual_norm_proj_offset);
          caffe_div<Dtype>(num_input_pixels_, dual_norm_proj_diff + dual_norm_proj_offset, dual_norm_proj_data + dual_norm_proj_offset,
            dual_norm_proj_diff + dual_norm_proj_offset);
          caffe_scal<Dtype>(num_input_pixels_, Dtype(-1), dual_norm_proj_diff+dual_norm_proj_offset);

          caffe_axpy<Dtype>(num_input_pixels_, Dtype(1), dual_norm_proj_diff + dual_norm_proj_offset,
            dual_norm_diff + dual_norm_proj_offset);
          dual_offset += num_input_pixels_;
        }
      }
    }

   

    
    const int count = dual_norm_.count(0);
    for(int i = 0; i < count; i++) dual_norm_diff[i] = dual_norm_diff[i] * (dual_norm_data[i] > 1);

    const Dtype* dual_norm_diff_const = dual_norm_.cpu_diff();
    for(int n = 0; n < num_; n++){
      for(int c = 0;  c < num_classes_; c++){
        for(int i = 0; i < num_input_pixels_; i++){
          int index = n*dual_dim_+ c* 3* num_input_pixels_ +  i;
          Dtype norm =  *(dual_norm_data++);
          Dtype norm_diff = *(dual_norm_diff_const++);
          for(int t = 0; t < 3; t++){
            bottom_diff[index] += (norm > 1? norm_diff* bottom_data[index]/norm : 0); 
            //bottom_diff[index] += norm_diff* bottom_data[index]/norm;
            index += num_input_pixels_;
          }
        }
      }
    }


  }
}


#ifdef CPU_ONLY
STUB_GPU(OctDualProjectionLayer);
#endif

INSTANTIATE_CLASS(OctDualProjectionLayer);

}  // namespace caffe
