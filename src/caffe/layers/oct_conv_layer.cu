#include "caffe/layers/oct_conv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/octree.h"
#include "caffe/layers/oct_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void set_bias_multiplier_kernel(const int nthread, Dtype* bias_multiplier_data, 
    const Dtype* num_elements){

    CUDA_KERNEL_LOOP(index, nthread){

        if(index < (int)(*num_elements)){
            bias_multiplier_data[index] = 1;
        }else{
             bias_multiplier_data[index] = 0;
        }
   
    }

}


template <typename Dtype>
void OctConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* num_data = bottom[1]->gpu_data();
    const Dtype* neighbors_data = bottom[2]->gpu_data(); 

    Dtype* top_data = top[0]->mutable_gpu_data();
    
    //Dtype* col_buff = _col_buffer.mutable_gpu_data();
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    Dtype* col_buff = col_buffer.mutable_gpu_data();

    for (int bt = 0; bt <_batch_size; bt++){

        octree2col_gpu(bottom_data + bt * _bottom_dim, neighbors_data + bt * bottom[2]->count(1), 
                        col_buff, num_data+bt);

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
            _col_buffer_shape[1], _col_buffer_shape[0],
            (Dtype)1., weight, col_buff,
            (Dtype)0., top_data + bt * _top_dim);

        set_bias_multiplier_kernel<Dtype><<<CAFFE_GET_BLOCKS(_num_output_pixels), CAFFE_CUDA_NUM_THREADS>>>(
            _num_output_pixels, _bias_multiplier.mutable_gpu_data(), num_data+bt);
      
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _bias_shape[0], _num_output_pixels, 1,
                          (Dtype)1., this->blobs_[1]->gpu_data(), _bias_multiplier.gpu_data(),
                          (Dtype)1., top_data + bt * _top_dim);
    }

}

template <typename Dtype>
void OctConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    
    if(propagate_down[0] || this->param_propagate_down_[0] || this->param_propagate_down_[1]){
        
        const Dtype* weight = this->blobs_[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* num_data = bottom[1]->gpu_data();
        const Dtype* neighbors_data = bottom[2]->gpu_data(); 
        const Dtype* neighbor_of_data = bottom[3]->gpu_data(); 

        //Dtype* col_buff = _col_buffer.mutable_gpu_data();
        Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
        Dtype* col_buff = col_buffer.mutable_gpu_data();

        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
        Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        for (int bt = 0; bt < _batch_size; ++bt){
            
            if(this->param_propagate_down_[0]){
                
                octree2col_gpu(bottom_data + bt * _bottom_dim, neighbors_data + bt*bottom[2]->count(1), 
                        col_buff, num_data+bt);
            
                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                          _col_buffer_shape[0], _col_buffer_shape[1],
                          (Dtype)1., top_diff + bt * _top_dim, col_buff, (Dtype)1., weight_diff);
            }

            if(this->param_propagate_down_[1]){
               
                set_bias_multiplier_kernel<Dtype><<<CAFFE_GET_BLOCKS(_num_output_pixels), CAFFE_CUDA_NUM_THREADS>>>(
                    _num_output_pixels, _bias_multiplier.mutable_gpu_data(), num_data+bt);
                caffe_gpu_gemv<Dtype>(CblasNoTrans, _num_output_channels, _num_output_pixels, 1.,
                    top_diff+bt*_top_dim, _bias_multiplier.gpu_data(), (Dtype)1., bias_diff);
            }


      
            if(propagate_down[0]){
                
                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
                        _col_buffer_shape[1], _weight_shape[0],
                        (Dtype)1., weight, top_diff + bt * _top_dim, 
                        (Dtype)0., col_buff);

                col2octree_gpu(col_buffer.gpu_data(), neighbor_of_data + bt*bottom[3]->count(1), 
                        bottom_diff + bt * _bottom_dim, num_data+bt);
            }
      
        }
    }

    if(propagate_down[1] || propagate_down[2] || propagate_down[3]){
        LOG(FATAL) << "neighbors and lens input cannot be back propagated.";
    }  

}

template <typename Dtype>
__global__ void conv_octree2col_kernel2(const int nthread, const Dtype* bottom_data, const Dtype* neighbors_data,
        Dtype* col_buff, const Dtype* num_elements, const int num_neighbors, const int num_output_pixels){

    CUDA_KERNEL_LOOP(index, nthread){

        const int i = index % num_output_pixels;
        const int el = (index / num_output_pixels)% num_neighbors;
        const int ch = index/ num_output_pixels / num_neighbors;

        if(i < (int)(*num_elements)){
 
            int nbh_pos = (int)neighbors_data[i * num_neighbors + el];

            if(nbh_pos != -1){

                int col_buff_ind = (ch * num_neighbors + el) * num_output_pixels + i;
                // num_output_pixels == num_input_pixels
                int feature_ind = ch *  num_output_pixels + nbh_pos;
                col_buff[col_buff_ind] = bottom_data[feature_ind];

            }
        }
   
    }

}


// this octree2col function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctConvLayer<Dtype>::octree2col_gpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
    Dtype* col_buff, const Dtype* num_elements){
    
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_gpu_set<Dtype>(_num_input_channels * num_neighbors * _num_output_pixels, Dtype(0.), col_buff);
    
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    int nthread = num_neighbors * _num_input_channels * _num_output_pixels;
    conv_octree2col_kernel2<Dtype>
            <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
                nthread, bottom_data, neighbors_data, col_buff, num_elements, num_neighbors, _num_output_pixels);

    CUDA_POST_KERNEL_CHECK;
}
// explicit instantiation
template void OctConvLayer<double>::octree2col_gpu(const double* bottom_data, const double* neighbors_data, 
    double* col_buff, const double* num_elements);
template void OctConvLayer<float>::octree2col_gpu(const float* bottom_data, const float* neighbors_data, 
    float* col_buff, const float* num_elements);


template <typename Dtype>
__global__ void conv_col2octree_kernel2(const int nthread, const Dtype* col_buff, const Dtype* neighbor_of_data,
        Dtype* bottom_diff, const Dtype* num_elements, const int num_neighbors, const int num_output_pixels){

    CUDA_KERNEL_LOOP(index, nthread){

        // num_output_pixels == num_input_pixels
        const int i = index % num_output_pixels;
        const int ch = index/ num_output_pixels;

        if (i  < (int)(*num_elements)){

            Dtype val = 0;
            for(int el = 0; el  < num_neighbors; el++){
                int of_nbh_pos = (int)neighbor_of_data[i * num_neighbors + el];

                if(of_nbh_pos != -1){

                    int col_buff_ind = (ch * num_neighbors + el) * num_output_pixels + of_nbh_pos;
                    val += col_buff[col_buff_ind];
                }
            }

            int feature_ind = ch * num_output_pixels + i;
            bottom_diff[feature_ind] = val;
        }
   
    }

}

// this col2octree function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctConvLayer<Dtype>::col2octree_gpu(const Dtype* col_buff, const Dtype* neighbor_of_data, 
    Dtype* bottom_diff, const Dtype* num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_gpu_set<Dtype>(_num_input_channels * _num_input_pixels, Dtype(0.), bottom_diff);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    int nthread = _num_input_channels * _num_input_pixels;
    conv_col2octree_kernel2<Dtype>
            <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
                nthread, col_buff, neighbor_of_data, bottom_diff, num_elements, num_neighbors, _num_output_pixels);

    CUDA_POST_KERNEL_CHECK;

    
}

template void OctConvLayer<double>::col2octree_gpu(const double* col_buff, const double* neighbor_of_data, 
    double* bottom_diff, const double* num_elements);
template void OctConvLayer<float>::col2octree_gpu(const float* col_buff, const float* neighbor_of_data, 
    float* bottom_diff, const float* num_elements);
INSTANTIATE_LAYER_GPU_FUNCS(OctConvLayer);

}  // namespace caffe
