#include <vector>

#include "caffe/layers/upsampling3d_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Tile(const int nthreads, const Dtype* bottom_data,
    const int tile_size, const int num_tiles, const int bottom_tile_axis,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % tile_size;
    const int b = ((index / tile_size ) % (bottom_tile_axis * num_tiles)) / num_tiles;
    const int n = index / tile_size / bottom_tile_axis / num_tiles ;
    const int bottom_index = (n * bottom_tile_axis + b) * tile_size + d;
    top_data[index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
void UpSampling3DLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  // upsampling along height axis
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* h_rep_blob_data = h_rep_blob_.mutable_gpu_data();
  int nthreads = h_rep_blob_.count();
  int tile_size = bottom[0]->count(first_spatial_axis_+1);
  int bottom_tile_axis = bottom[0]->shape(first_spatial_axis_);
  Tile<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom_data, tile_size, h_rep_, bottom_tile_axis, h_rep_blob_data);



  // upsampling along width axis
  const Dtype* h_rep_blob_data_const = h_rep_blob_.gpu_data();
  Dtype* w_rep_blob_data = w_rep_blob_.mutable_gpu_data();
  nthreads = w_rep_blob_.count();
  tile_size = h_rep_blob_.count(first_spatial_axis_+2);
  bottom_tile_axis = h_rep_blob_.shape(first_spatial_axis_+1);
  Tile<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, h_rep_blob_data_const, tile_size, w_rep_, bottom_tile_axis, w_rep_blob_data);


  // upsampling along height axis
  const Dtype* w_rep_blob_data_const = w_rep_blob_.gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  nthreads = top[0]->count();
  tile_size = 1;
  bottom_tile_axis = w_rep_blob_.shape(first_spatial_axis_+2);
  Tile<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, w_rep_blob_data_const, tile_size, s_rep_, bottom_tile_axis, top_data);


}

template <typename Dtype>
__global__ void TileBackward(const int nthreads, const Dtype* top_diff,
    const int tile_size, const int num_tiles, const int bottom_tile_axis,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % tile_size;
    const int b = (index / tile_size) % bottom_tile_axis;
    const int n = index / tile_size / bottom_tile_axis;
    bottom_diff[index] = 0;
    int top_index = (n * num_tiles * bottom_tile_axis + b * num_tiles) * tile_size + d;
    for (int t = 0; t < num_tiles; ++t) {
      bottom_diff[index] += top_diff[top_index];
      top_index += tile_size;
    }
  }
}

template <typename Dtype>
void UpSampling3DLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* w_rep_blob_diff = w_rep_blob_.mutable_gpu_diff(); 
  int tile_size = 1;
  int bottom_tile_axis = w_rep_blob_.shape(first_spatial_axis_+2);
  int nthreads = w_rep_blob_.count();
  TileBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, top_diff, tile_size, s_rep_, bottom_tile_axis, w_rep_blob_diff);

  const Dtype* w_rep_blob_diff_const = w_rep_blob_.gpu_diff();
  Dtype* h_rep_blob_diff = h_rep_blob_.mutable_gpu_diff(); 
  tile_size = h_rep_blob_.count(first_spatial_axis_+2);
  bottom_tile_axis = h_rep_blob_.shape(first_spatial_axis_+1);
  nthreads = h_rep_blob_.count();
  TileBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, w_rep_blob_diff_const, tile_size, w_rep_, bottom_tile_axis, h_rep_blob_diff);

  const Dtype* h_rep_blob_diff_const = h_rep_blob_.gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff(); 
  tile_size = bottom[0]->count(first_spatial_axis_+1);
  bottom_tile_axis = bottom[0]->shape(first_spatial_axis_);
  nthreads = bottom[0]->count();
  TileBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, h_rep_blob_diff_const, tile_size, h_rep_, bottom_tile_axis, bottom_diff);





}

INSTANTIATE_LAYER_GPU_FUNCS(UpSampling3DLayer);

}  // namespace caffe
