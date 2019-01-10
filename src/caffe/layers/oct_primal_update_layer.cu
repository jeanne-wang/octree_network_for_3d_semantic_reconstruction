#include "caffe/layers/oct_primal_update_layer.hpp"

namespace caffe{

template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* dual_data = bottom[1]->gpu_data();
    const Dtype* num_data = bottom[4]->gpu_data();
    const Dtype* neighbor_of_data = bottom[6]->gpu_data(); 

    Dtype* deconv_res_data = deconv_res_.mutable_gpu_data();
    //Dtype* col_buff = _col_buffer.mutable_gpu_data();
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    Dtype* col_buff = col_buffer.mutable_gpu_data();
    for (int bt = 0; bt <_batch_size; bt++){
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
            _col_buffer_shape[1], _weight_shape[0],
            (Dtype)1., weight, dual_data + bt * _dual_dim,
            (Dtype)0., col_buff);

        col2octree_gpu(col_buffer.gpu_data(), neighbor_of_data + bt * bottom[6]->count(1), deconv_res_data + bt * _primal_dim, num_data+bt);
    }


    Dtype* primal_before_proj_data = primal_before_proj_.mutable_gpu_data();
    const Dtype* deconv_res_data_const = deconv_res_.gpu_data();
    const int count = primal_before_proj_.count();
    caffe_copy<Dtype>(count, deconv_res_data_const, primal_before_proj_data);

    // add datacost term
    caffe_gpu_axpy<Dtype>(count, Dtype(1.), bottom[3]->gpu_data(), primal_before_proj_data);

    // add lagrangian
    const Dtype* lagrangian_data = bottom[2]->gpu_data();
    for(int n = 0; n < _batch_size; n++){
        for(int c = 0; c < primal_before_proj_.shape(1); c++){   
            caffe_gpu_axpy<Dtype>(_num_output_pixels, Dtype(1.), lagrangian_data + n * _num_output_pixels, 
            primal_before_proj_data+ n*_num_output_channels * _num_output_pixels + c * _num_output_pixels);
        }
    }


    caffe_gpu_scal<Dtype>(count, Dtype(-1. * tau_), primal_before_proj_data);
    // add previous primal data
    caffe_gpu_axpy<Dtype>(count, Dtype(1.), bottom[0]->gpu_data(), primal_before_proj_data);

    // projection to [0, 1]
    primal_projection_layer_->Forward(primal_projection_bottom_vecs_, top);  

}

template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
    vector<bool> down;
    down.clear();
    down.push_back(true);
    primal_projection_layer_->Backward(top, down, primal_projection_bottom_vecs_);
    const Dtype* primal_before_proj_diff = primal_before_proj_.gpu_diff(); 

  
    // compute gradient with respect to input previous primal data  
    const int count = bottom[0]->count();
    if(propagate_down[0]){
        Dtype* primal_diff = bottom[0]->mutable_gpu_diff(); 
        caffe_copy<Dtype>(count, primal_before_proj_diff, primal_diff); 
    }
  
    // compute gradient with respect to data cost term
    if(propagate_down[3]){
        Dtype* datacost_diff = bottom[3]->mutable_gpu_diff();
        caffe_gpu_set<Dtype>(count, Dtype(0), datacost_diff);
        caffe_gpu_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, datacost_diff);
    }
  
  
    // compute gradient with respect to lagrangian term
    if(propagate_down[2]){
        Dtype* lagrangian_diff = bottom[2]->mutable_gpu_diff();
        caffe_gpu_set<Dtype>(bottom[2]->count(), Dtype(0.), lagrangian_diff);
        for(int n = 0; n < _batch_size; n++){
            for(int c = 0; c < primal_before_proj_.shape(1); c++){     
                caffe_gpu_axpy<Dtype>(_num_output_pixels, Dtype(-1.*tau_), 
                    primal_before_proj_diff + n * _num_output_channels * _num_output_pixels + c * _num_output_pixels,
                    lagrangian_diff + n * _num_output_pixels);
            }
        }
    } 
  
 


    // compute gradient with respect to dual term and weight.
    if(propagate_down[1] || this->param_propagate_down_[0]){

        const Dtype* weight = this->blobs_[0]->gpu_data();
        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

        Dtype* deconv_res_diff = deconv_res_.mutable_gpu_diff();
        caffe_gpu_set<Dtype>(count, Dtype(0.), deconv_res_diff);
        caffe_gpu_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, 
            deconv_res_diff); 

        const Dtype* dual_data = bottom[1]->gpu_data();
        const Dtype* num_data = bottom[4]->gpu_data();
        const Dtype* neighbors_data = bottom[5]->gpu_data(); 
        //Dtype* col_buff = _col_buffer.mutable_gpu_data();
        Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
        Dtype* col_buff = col_buffer.mutable_gpu_data();
        Dtype* dual_diff =  bottom[1]->mutable_gpu_diff();
        for (int bt = 0; bt < _batch_size; ++bt){

            octree2col_gpu(deconv_res_.gpu_diff() + bt * _primal_dim, neighbors_data + bt * bottom[5]->count(1), col_buff, num_data+bt);
            if(this->param_propagate_down_[0]){

                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                    _col_buffer_shape[0], _col_buffer_shape[1],
                    (Dtype)1., dual_data + bt * _dual_dim, col_buff, (Dtype)1., weight_diff);
            }
      
            if(propagate_down[1]){

                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
                    _col_buffer_shape[1], _col_buffer_shape[0],
                    (Dtype)1., weight, col_buff,
                    (Dtype)0., dual_diff + bt * _dual_dim);
            } 
        }
    } 

    if(propagate_down[4] || propagate_down[5] || propagate_down[6]){
        LOG(FATAL) << "neighbors and lens input cannot be back propagated.";
    }   
	
}

template <typename Dtype>
__global__ void deconv_col2octree_kernel(const int nthread, const Dtype* col_buff, const Dtype* neighbor_of_data,
    Dtype* top_data, const Dtype* num_elements, const int num_neighbors, const int num_input_pixels){

    CUDA_KERNEL_LOOP(index, nthread){

        // num_output_pixels == num_input_pixels
        const int i = index % num_input_pixels;
        const int ch = index/ num_input_pixels;

        if (i  < (int)(*num_elements)){

            Dtype val = 0;
            for(int el = 0; el  < num_neighbors; el++){
                int of_nbh_pos = (int)neighbor_of_data[i * num_neighbors + el];

                if(of_nbh_pos != -1){

                    int col_buff_ind = (ch * num_neighbors + el) * num_input_pixels + of_nbh_pos;
                    val += col_buff[col_buff_ind];
                }
            }

            // num_output_pixels == num_input_pixels
            int feature_ind = ch * num_input_pixels + i;
            top_data[feature_ind] = val;            
           
        }
    }

}

// this col2octree function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::col2octree_gpu(const Dtype* col_buff, const Dtype* neighbor_of_data, 
    Dtype* top_data, const Dtype* num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_gpu_set<Dtype>(_num_output_channels * _num_output_pixels, Dtype(0.), top_data);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    int nthread = _num_output_channels * _num_output_pixels;
    deconv_col2octree_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
          nthread, col_buff, neighbor_of_data, top_data, num_elements, num_neighbors, _num_input_pixels);


    CUDA_POST_KERNEL_CHECK;
    
}

template void OctPrimalUpdateLayer<double>::col2octree_gpu(const double* col_buff, const double* neighbor_of_data, 
    double* top_data, const double* num_elements);
template void OctPrimalUpdateLayer<float>::col2octree_gpu(const float* col_buff, const float* neighbor_of_data, 
    float* top_data, const float* num_elements);


template <typename Dtype>
__global__ void deconv_octree2col_kernel(const int nthread, const Dtype* top_diff, const Dtype* neighbors_data,
    Dtype* col_buff, const Dtype* num_elements, const int num_neighbors, const int num_input_pixels){

    CUDA_KERNEL_LOOP(index, nthread){

        const int i = index % num_input_pixels;
        const int el = (index / num_input_pixels)% num_neighbors;
        const int ch = index/ num_input_pixels / num_neighbors;


        if(i < (int)(*num_elements)){

            int nbh_pos = (int)neighbors_data[i * num_neighbors + el];

            if(nbh_pos != -1){

                int col_buff_ind = (ch * num_neighbors + el) * num_input_pixels + i;
                // num_output_pixels == num_input_pixels
                int feature_ind = ch * num_input_pixels + nbh_pos;
                col_buff[col_buff_ind] = top_diff[feature_ind];
            }
        }
   
    }

}


// this col2octree function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::octree2col_gpu(const Dtype* top_diff, const Dtype* neighbors_data, 
  Dtype* col_buff, const Dtype* num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_gpu_set<Dtype>(_num_output_channels * num_neighbors * _num_input_pixels, Dtype(0.), col_buff);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    int nthread = num_neighbors * _num_output_channels * _num_input_pixels;
    deconv_octree2col_kernel<Dtype>
        <<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
          nthread,top_diff, neighbors_data, col_buff, num_elements, num_neighbors, _num_input_pixels);


    CUDA_POST_KERNEL_CHECK;
    
}

// explicit instantiation
template void OctPrimalUpdateLayer<double>::octree2col_gpu(const double* top_diff, const double* neighbors_data, 
    double* col_buff, const double* num_elements);
template void OctPrimalUpdateLayer<float>::octree2col_gpu(const float* top_diff, const float* neighbors_data, 
    float* col_buff, const float* num_elements);



INSTANTIATE_LAYER_GPU_FUNCS(OctPrimalUpdateLayer);
}