#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPool3DForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int depth, const int pooled_height,
    const int pooled_width, const int pooled_depth, const int kernel_h, const int kernel_w, const int kernel_s,
    const int stride_h, const int stride_w, const int stride_s, const int pad_h, const int pad_w, const int pad_s,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int ps = index % pooled_depth;
    const int pw = (index / pooled_depth) % pooled_width;
    const int ph = (index / pooled_depth/ pooled_width) % pooled_height;
    const int c = (index / pooled_depth /pooled_width / pooled_height) % channels;
    const int n = index / pooled_depth/ pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int sstart = ps * stride_s - pad_s;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    const int send = min(sstart + kernel_s, depth);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    sstart = max(sstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width * depth;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	      for(int s = sstart; s < send; ++s){
		      if (bottom_slice[(h * width + w)*depth + s] > maxval) {
            maxidx = (h * width + w) * depth + s;
            maxval = bottom_slice[maxidx];
          }
	      }
        
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePool3DForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int depth, const int pooled_height,
    const int pooled_width, const int pooled_depth, const int kernel_h, const int kernel_w, const int kernel_s,
    const int stride_h, const int stride_w, const int stride_s, const int pad_h, const int pad_w, const int pad_s,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int ps = index % pooled_depth;
    const int pw = (index / pooled_depth) % pooled_width;
    const int ph = (index / pooled_depth/ pooled_width) % pooled_height;
    const int c = (index / pooled_depth /pooled_width / pooled_height) % channels;
    const int n = index / pooled_depth/ pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int sstart = ps * stride_s - pad_s;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    int send = min(sstart + kernel_s, depth + pad_s);
    const int pool_size = (hend - hstart) * (wend - wstart) * (send-sstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    sstart = max(sstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    send = min(send, depth);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width * depth;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        for(int s = sstart; s < send; ++s){
          aveval += bottom_slice[(h * width + w)*depth + s];
        }      
      }
    }
    top_data[index] = aveval / pool_size;
  }
}


template <typename Dtype>
void Pooling3DLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling3d_param().pool()) {
  case Pooling3DParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPool3DForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_, kernel_h_,
        kernel_w_, kernel_s_, stride_h_, stride_w_, stride_s_, pad_h_, pad_w_, pad_s_, top_data,
        mask, top_mask);
    break;
  case Pooling3DParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePool3DForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_, kernel_h_,
        kernel_w_, kernel_s_, stride_h_, stride_w_, stride_s_, pad_h_, pad_w_, pad_s_, top_data);
    break;
  case Pooling3DParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPool3DBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width, const int depth,
    const int pooled_height, const int pooled_width, const int pooled_depth, const int kernel_h,
    const int kernel_w, const int kernel_s, const int stride_h, const int stride_w, const int stride_s, const int pad_h,
    const int pad_w, const int pad_s, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int s = index % depth;
    const int w = (index / depth) % width;
    const int h = (index / depth/ width) % height;
    const int c = (index / depth /width / height) % channels;
    const int n = index / depth /width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    const int psstart =
         (s + pad_s < kernel_s) ? 0 : (s + pad_s - kernel_s) / stride_s + 1;
    const int psend = min((s + pad_s) / stride_w + 1, pooled_depth);

    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          for(int ps = psstart; ps < psend; ++ps){
            if (mask_slice[(ph * pooled_width + pw) * pooled_depth + ps] == (h * width + w) * depth + s) {
              gradient += top_diff_slice[(ph * pooled_width + pw) * pooled_depth + ps];
            }
          }
          
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          for(int ps = psstart; ps < psend; ++ps){
            if (top_mask_slice[(ph * pooled_width + pw) * pooled_depth + ps] == (h * width + w) * depth + s) {
              gradient += top_diff_slice[(ph * pooled_width + pw) * pooled_depth + ps];
            }
          }
          
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePool3DBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int depth, const int pooled_height, const int pooled_width, const int pooled_depth,
    const int kernel_h, const int kernel_w, const int kernel_s, const int stride_h,
    const int stride_w, const int stride_s, const int pad_h, const int pad_w, const int pad_s,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int s = index % depth + pad_s;
    const int w = (index / depth) % width + pad_w;
    const int h = (index / depth/ width) % height + pad_h;
    const int c = (index / depth /width / height) % channels;
    const int n = index / depth /width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    const int psstart = (s < kernel_s) ? 0 : (s - kernel_s) / stride_s + 1;
    const int psend = min(s / stride_s + 1, pooled_depth);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        for(int ps = psstart; ps < psend; ++ps){
          // figure out the pooling size
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int sstart = ps * stride_s - pad_s;
          int hend = min(hstart + kernel_h, height + pad_h);
          int wend = min(wstart + kernel_w, width + pad_w);
          int send = min(sstart + kernel_s, depth + pad_s);
          int pool_size = (hend - hstart) * (wend - wstart) * (send-sstart);
          gradient += top_diff_slice[(ph * pooled_width + pw) * pooled_depth + ps] / pool_size;
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}



template <typename Dtype>
void Pooling3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling3d_param().pool()) {
  case Pooling3DParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPool3DBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->shape(0), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_,
        kernel_h_, kernel_w_, kernel_s_, stride_h_, stride_w_, stride_s_, pad_h_, pad_w_, pad_s_,
        bottom_diff);
    break;
  case Pooling3DParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePool3DBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->shape(0), channels_,
        height_, width_, depth_, pooled_height_, pooled_width_, pooled_depth_, kernel_h_,
        kernel_w_, kernel_s_, stride_h_, stride_w_, stride_s_, pad_h_, pad_w_, pad_s_, bottom_diff);
    break;
  case Pooling3DParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(Pooling3DLayer);


}  // namespace caffe
