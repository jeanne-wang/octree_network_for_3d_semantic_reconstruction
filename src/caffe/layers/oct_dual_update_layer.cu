#include "caffe/layers/oct_dual_update_layer.hpp"

namespace caffe{

template <typename Dtype>
void OctDualUpdateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	
  	
  	const Dtype* weight = this->blobs_[0]->gpu_data();
    const Dtype* primal_data = bottom[1]->gpu_data();
    const Dtype* num_data = bottom[2]->gpu_data();
    const Dtype* neighbors_data = bottom[3]->gpu_data(); 
   
    Dtype* conv_res_data = conv_res_.mutable_gpu_data();
    //Dtype* col_buff = _col_buffer.mutable_gpu_data();
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    Dtype* col_buff = col_buffer.mutable_gpu_data();
    for (int bt = 0; bt <_batch_size; bt++){
    	
        octree2col_gpu(primal_data + bt * _primal_dim, neighbors_data + bt * bottom[3]->count(1),
        	col_buff, num_data+bt);
        

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
        	_col_buffer_shape[1], _col_buffer_shape[0],
			(Dtype)1., weight, col_buff,
			(Dtype)0., conv_res_data + bt * _dual_dim);
    }

    Dtype* dual_before_proj_data = dual_before_proj_.mutable_gpu_data();
    const int count = bottom[0]->count();
    caffe_copy<Dtype>(count, bottom[0]->gpu_data(), dual_before_proj_data);
    // add conv result
    const Dtype* conv_res_data_const = conv_res_.gpu_data();
    caffe_gpu_axpy<Dtype>(count, Dtype(sigma_* 1.), conv_res_data_const, dual_before_proj_data);

    // projection to ||.|| >=1
    dual_projection_layer_->Forward(dual_projection_bottom_vecs_, top); 

    


}

template <typename Dtype>
void OctDualUpdateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  	vector<bool> down;
  	down.clear();
  	down.push_back(true);
  	dual_projection_layer_->Backward(top, down, dual_projection_bottom_vecs_);
  	const Dtype* dual_before_proj_diff = dual_before_proj_.gpu_diff(); 

   

  	// compute gradient with respect to input previous dual data  
  	const int count = bottom[0]->count();
  	if(propagate_down[0]){
    	Dtype* dual_diff = bottom[0]->mutable_gpu_diff(); 
    	caffe_copy<Dtype>(count, dual_before_proj_diff, dual_diff);
  	}

  	// compute gradient with respect to dual term and weight.
    const Dtype* weight = this->blobs_[0]->gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    if(propagate_down[1] || this->param_propagate_down_[0]){

        Dtype* conv_res_diff = conv_res_.mutable_gpu_diff();
        caffe_gpu_set<Dtype>(conv_res_.count(), Dtype(0.), conv_res_diff);
        caffe_gpu_axpy<Dtype>(conv_res_.count(), Dtype(sigma_), dual_before_proj_diff, 
        conv_res_diff);


        const Dtype* primal_data = bottom[1]->gpu_data();
        const Dtype* num_data = bottom[2]->gpu_data();
        const Dtype* neighbors_data = bottom[3]->gpu_data(); 
        const Dtype* neighbor_of_data = bottom[4]->gpu_data(); 
        //Dtype* col_buff = _col_buffer.mutable_gpu_data();
        Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
        Dtype* col_buff = col_buffer.mutable_gpu_data();
        for (int bt = 0; bt < _batch_size; ++bt){
     
            if(this->param_propagate_down_[0]){
                octree2col_gpu(primal_data + bt * _primal_dim, neighbors_data + bt*bottom[3]->count(1), 
                        col_buff, num_data+bt);

                caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                          _col_buffer_shape[0], _col_buffer_shape[1],
                          (Dtype)1., conv_res_diff + bt * _dual_dim, col_buff, (Dtype)1., weight_diff);
            }
      
            if(propagate_down[1]){

                caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
                        _col_buffer_shape[1], _weight_shape[0],
                        (Dtype)1., weight, conv_res_diff + bt * _dual_dim, 
                        (Dtype)0., col_buff);

                col2octree_gpu(col_buffer.gpu_data(), neighbor_of_data + bt*bottom[4]->count(1), 
                        bottom[1]->mutable_gpu_diff()+ bt * _primal_dim, num_data+bt);

                
            }
      
        }

    }

    if(propagate_down[2] || propagate_down[3] || propagate_down[4]){
        LOG(FATAL) << "neighbors and lens input cannot be back propagated.";
    }  

	
}

template <typename Dtype>
__global__ void conv_octree2col_kernel(const int nthread, const Dtype* bottom_data, const Dtype* neighbors_data,
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
void OctDualUpdateLayer<Dtype>::octree2col_gpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
	Dtype* col_buff, const Dtype* num_elements){
  	
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
  	caffe_gpu_set<Dtype>(_num_input_channels * num_neighbors * _num_output_pixels, Dtype(0.), col_buff);
  	
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    int nthread = num_neighbors * _num_input_channels * _num_output_pixels;
    conv_octree2col_kernel<Dtype>
    		<<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
    			nthread, bottom_data, neighbors_data, col_buff, num_elements, num_neighbors, _num_output_pixels);

    CUDA_POST_KERNEL_CHECK;
}
// explicit instantiation
template void OctDualUpdateLayer<double>::octree2col_gpu(const double* bottom_data, const double* neighbors_data, 
	double* col_buff, const double* num_elements);
template void OctDualUpdateLayer<float>::octree2col_gpu(const float* bottom_data, const float* neighbors_data, 
	float* col_buff, const float* num_elements);


template <typename Dtype>
__global__ void conv_col2octree_kernel(const int nthread, const Dtype* col_buff, const Dtype* neighbor_of_data,
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
void OctDualUpdateLayer<Dtype>::col2octree_gpu(const Dtype* col_buff, const Dtype* neighbor_of_data, 
	Dtype* bottom_diff, const Dtype* num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_gpu_set<Dtype>(_num_input_channels * _num_input_pixels, Dtype(0.), bottom_diff);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    int nthread = _num_input_channels * _num_input_pixels;
    conv_col2octree_kernel<Dtype>
    		<<<CAFFE_GET_BLOCKS(nthread), CAFFE_CUDA_NUM_THREADS>>>(
    			nthread, col_buff, neighbor_of_data, bottom_diff, num_elements, num_neighbors, _num_output_pixels);

   	CUDA_POST_KERNEL_CHECK;

    
}

template void OctDualUpdateLayer<double>::col2octree_gpu(const double* col_buff, const double* neighbor_of_data, 
	double* bottom_diff, const double* num_elements);
template void OctDualUpdateLayer<float>::col2octree_gpu(const float* col_buff, const float* neighbor_of_data, 
	float* bottom_diff, const float* num_elements);


INSTANTIATE_LAYER_GPU_FUNCS(OctDualUpdateLayer);
}
