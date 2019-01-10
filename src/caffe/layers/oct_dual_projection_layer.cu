#include <vector>

#include "caffe/layers/oct_dual_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void OctNormForward(const int nthread, const Dtype* bottom_data, const int num,
  const int num_classes, const int num_input_pixels,
  const int dual_dim, Dtype* top_data){

  CUDA_KERNEL_LOOP(index, nthread) {
    const int i = index % num_input_pixels;
    const int c = (index / num_input_pixels) % num_classes;
    const int n =  index / num_input_pixels /num_classes;
    int bottom_index = n * dual_dim + c * 3 * num_input_pixels + i;

    Dtype norm = 0;
    for(int t = 0;  t < 3; t++){
      norm += bottom_data[bottom_index] * bottom_data[bottom_index];
      bottom_index += num_input_pixels;
    }
    top_data[index] = sqrt(norm);  
  }

}

template <typename Dtype>
__global__ void OctNormTruncForward(const int nthread, const Dtype* bottom_data, Dtype* top_data){

  CUDA_KERNEL_LOOP(index, nthread) {

    top_data[index] = bottom_data[index] > 1? bottom_data[index] : 1; 
    //top_data[index] = bottom_data[index]; // for test
  }
}

template <typename Dtype>
__global__ void OctNormalizationForward(const int nthread, const Dtype* bottom_data, const Dtype* norm_data, const int num,
  const int num_classes, const int num_input_pixels,
  const int dual_norm_dim, Dtype* top_data){

  CUDA_KERNEL_LOOP(index, nthread) {
    
    const int i = index % num_input_pixels;
    const int c = (index / num_input_pixels) % (num_classes * 3);
    const int n =  index / num_input_pixels /num_classes / 3;

    int norm_data_index = n * dual_norm_dim  + (c / 3) * num_input_pixels + i;

    top_data[index] = bottom_data[index] / norm_data[norm_data_index];    
  }

}


template <typename Dtype>
void OctDualProjectionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype * dual_norm_data = dual_norm_.mutable_gpu_data();
  int nthread = dual_norm_.count();

  OctNormForward<Dtype>
      <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
        nthread, bottom_data, num_, num_classes_, num_input_pixels_,
        dual_dim_, dual_norm_data);



  const Dtype* dual_norm_data_const = dual_norm_.gpu_data();
  Dtype* dual_norm_proj_data = dual_norm_proj_.mutable_gpu_data();
  OctNormTruncForward<Dtype>
      <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
        nthread, dual_norm_data_const, dual_norm_proj_data);


  const Dtype* dual_norm_proj_data_const = dual_norm_proj_.gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  nthread =  top[0]->count();
  OctNormalizationForward<Dtype>
      <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
        nthread, bottom_data, dual_norm_proj_data_const, num_, num_classes_, num_input_pixels_,
        dual_norm_dim_, top_data);

}


template <typename Dtype>
__global__ void OctNormalizationBottomBackward(const int nthread, const Dtype* top_diff, const Dtype* norm_data, const int num,
  const int num_classes, const int num_input_pixels,
  const int dual_norm_dim, Dtype* bottom_diff){

  CUDA_KERNEL_LOOP(index, nthread) {
    const int i = index % num_input_pixels;
    const int c = (index / num_input_pixels) % (num_classes * 3);
    const int n =  index / num_input_pixels /num_classes / 3;

    int norm_data_index = n * dual_norm_dim  + (c / 3) * num_input_pixels + i;

    bottom_diff[index] = top_diff[index] / norm_data[norm_data_index];    
  }

}

template <typename Dtype>
__global__ void OctNormalizationNormBackward(const int nthread, const Dtype* top_diff, const Dtype* top_data,
  const Dtype* norm_data, const int num,
  const int num_classes, const int num_input_pixels,
  const int dual_dim, Dtype* norm_diff){

  CUDA_KERNEL_LOOP(index, nthread) {
    const int i = index % num_input_pixels;
    const int c = (index / num_input_pixels) % num_classes;
    const int n =  index / num_input_pixels /num_classes;
    int top_index = n * dual_dim + c * 3 * num_input_pixels + i;

    norm_diff[index] = 0;
    for(int t = 0;  t < 3; t++){
      norm_diff[index] -= top_diff[top_index] * top_data[top_index] / norm_data[index];
      top_index += num_input_pixels;
    }
  }

}

template <typename Dtype>
__global__ void OctNormTruncBackward(const int nthread, const Dtype* top_diff, const Dtype* bottom_data,
  Dtype* bottom_diff){

  CUDA_KERNEL_LOOP(index, nthread) {

    bottom_diff[index] = top_diff[index] * (bottom_data[index] > 1); 
    //bottom_diff[index] = top_diff[index]; //for test
  }
}


template <typename Dtype>
__global__ void OctNormBackward(const int nthread, const Dtype* top_diff, const Dtype* norm_data,
  const Dtype* bottom_data, const int num,
  const int num_classes, const int num_input_pixels,
  const int dual_norm_dim, Dtype* bottom_diff){

  CUDA_KERNEL_LOOP(index, nthread) {
    const int i = index % num_input_pixels;
    const int c = (index / num_input_pixels) % (num_classes * 3);
    const int n =  index / num_input_pixels /num_classes / 3;

    int norm_data_index = n * dual_norm_dim  + (c / 3) * num_input_pixels + i;

    bottom_diff[index] += (norm_data[norm_data_index] > 1? top_diff[norm_data_index] * bottom_data[index] / norm_data[norm_data_index] : 0); // correct comment out for debug
    //bottom_diff[index] += top_diff[norm_data_index] * bottom_data[index] / norm_data[norm_data_index]; //for test
  }

}



template <typename Dtype>
void OctDualProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* dual_norm_proj_data = dual_norm_proj_.gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int nthread = bottom[0]->count();

  OctNormalizationBottomBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(nthread, top_diff, 
        dual_norm_proj_data, num_, num_classes_, num_input_pixels_,
        dual_norm_dim_, bottom_diff);


  const Dtype* top_data = top[0]->gpu_data();
  Dtype* dual_norm_proj_diff = dual_norm_proj_.mutable_gpu_diff();
  nthread = dual_norm_proj_.count();
  OctNormalizationNormBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(nthread, top_diff, top_data,
  dual_norm_proj_data, num_, num_classes_, num_input_pixels_,
  dual_dim_, dual_norm_proj_diff);


  Dtype* dual_norm_diff = dual_norm_.mutable_gpu_diff();
  const Dtype* dual_norm_proj_diff_const = dual_norm_proj_.gpu_diff();
  const Dtype* dual_norm_data = dual_norm_.gpu_data();
  OctNormTruncBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(nthread, dual_norm_proj_diff_const,
        dual_norm_data, dual_norm_diff);


  nthread = bottom[0]->count();
  const Dtype* dual_norm_diff_const = dual_norm_.gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  OctNormBackward<Dtype>
      <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(nthread, dual_norm_diff_const, 
        dual_norm_data, bottom_data, num_, num_classes_, num_input_pixels_,
        dual_norm_dim_, bottom_diff);

}

INSTANTIATE_LAYER_GPU_FUNCS(OctDualProjectionLayer);

}  // namespace caffe