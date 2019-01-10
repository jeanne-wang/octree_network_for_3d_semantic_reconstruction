#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/oct_dual_update_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/layers/oct_layer.hpp"
#include "image_tree_tools/octree.h"
namespace caffe {

template <typename Dtype>
void OctDualUpdateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LOG(INFO) << "DualUpdateLayer set up. ";
    // bottom[0]: dual
    // bottom[1]: primal
    // bottom[2]: num
    // bottom[3]: neighbors  neighbors[(bt * num_input_pixels + i) * num_neighbors+ el]: the el-th neighbor positions  for octree cell i
    // bottom[4]: neighbor_of int j = neighbour_of[(bt * num_input_pixels + i) * num_neighbors+ el]: octree cell i is the el-th neighbor of j
  
    // convolution set up
    _filter_size = this->layer_param_.oct_dual_update_param().filter_size();
    _num_output_channels = this->layer_param_.oct_dual_update_param().output_channels();
    _num_input_channels = bottom[1]->shape(1);

    _weight_shape.push_back(_num_output_channels);
    _weight_shape.push_back(_num_input_channels);
    _weight_shape.push_back(_filter_size);
    _weight_shape.push_back(_filter_size);
    _weight_shape.push_back(_filter_size);

    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(_weight_shape));
  
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.oct_dual_update_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // Propagate gradients to the parameters (as directed by backward pass).
    this->param_propagate_down_.resize(this->blobs_.size(), true);

    // read parameter tau used in primal update formula
    sigma_ = this->layer_param_.oct_dual_update_param().sigma();

    dual_before_proj_.ReshapeLike(*bottom[0]);
    dual_projection_bottom_vecs_.clear();
    dual_projection_bottom_vecs_.push_back(&dual_before_proj_);

    LayerParameter dual_projection_layer_param;
    dual_projection_layer_.reset(new OctDualProjectionLayer<Dtype>(dual_projection_layer_param));
    dual_projection_layer_->SetUp(dual_projection_bottom_vecs_, top);
}

template <typename Dtype>
void OctDualUpdateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    _batch_size = bottom[1]->shape(0);
    _num_input_pixels = bottom[1]->shape(2);

    // check the batch size
    CHECK_EQ(bottom[0]->shape(0), _batch_size)
        <<"All variables must have the same number data. ";

    // check input voxels
    // check the size of the variables
    CHECK_EQ(bottom[0]->shape(2), _num_input_pixels)
        << "Primal and Dual variable must have the same input voxels";

    CHECK_EQ(bottom[1]->shape(1), _num_input_channels)
        << "Input size incompatible with convolution kernel.";
  
    // channels of dual should be 3 times of primal
    CHECK_EQ(bottom[0]->shape(1), _num_output_channels)
        <<"output channels must equal to number of channels in primal data";

    // newly added by xiaojuan on Sep 6th, 2018
    CHECK_EQ(_filter_size * _filter_size * _filter_size, bottom[3]->shape(2))
        << "reference neighbors matrix must comply with the filter size.";

    _num_output_pixels = _num_input_pixels;
    _dual_dim = bottom[0]->count(1);
    _primal_dim = bottom[1]->count(1);

    // Shape the tops.
    top[0]->ReshapeLike(*bottom[0]);
    conv_res_.ReshapeLike(*bottom[0]);
    dual_before_proj_.ReshapeLike(*bottom[0]);
  
    _col_buffer_shape.clear();
    _col_buffer_shape.push_back(_weight_shape[1] * _filter_size * _filter_size * _filter_size);
    _col_buffer_shape.push_back(_num_output_pixels);
    //_col_buffer.Reshape(_col_buffer_shape);

    // request col buffer from current thread
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    col_buffer.Reshape(_col_buffer_shape);  
}

template <typename Dtype>
void OctDualUpdateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* weight = this->blobs_[0]->cpu_data();
    const Dtype* primal_data = bottom[1]->cpu_data();
    const Dtype* num_data = bottom[2]->cpu_data();
    const Dtype* neighbors_data = bottom[3]->cpu_data(); 

    Dtype* conv_res_data = conv_res_.mutable_cpu_data();
    //Dtype* col_buff = _col_buffer.mutable_cpu_data();
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    Dtype* col_buff = col_buffer.mutable_cpu_data();

    for (int bt = 0; bt <_batch_size; bt++){

        octree2col_cpu(primal_data + bt * _primal_dim, neighbors_data + bt * bottom[3]->count(1), 
                        col_buff, (int)num_data[bt]);

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
        _col_buffer_shape[1], _col_buffer_shape[0],
        (Dtype)1., weight, col_buff,
        (Dtype)0., conv_res_data + bt * _dual_dim);
    }
  
    Dtype* dual_before_proj_data = dual_before_proj_.mutable_cpu_data();
    const int count = bottom[0]->count();
    caffe_copy<Dtype>(count, bottom[0]->cpu_data(), dual_before_proj_data);
    // add conv result
    const Dtype* conv_res_data_const = conv_res_.cpu_data();
    caffe_axpy<Dtype>(count, Dtype(sigma_* 1.), conv_res_data_const, dual_before_proj_data);

    // projection to ||.|| >=1
    dual_projection_layer_->Forward(dual_projection_bottom_vecs_, top); 

}

template <typename Dtype>
void OctDualUpdateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


    vector<bool> down;
    down.clear();
    down.push_back(true);
    dual_projection_layer_->Backward(top, down, dual_projection_bottom_vecs_);
    const Dtype* dual_before_proj_diff = dual_before_proj_.cpu_diff(); 

   
    // compute gradient with respect to input previous dual data  
    const int count = bottom[0]->count();
    if(propagate_down[0]){
        Dtype* dual_diff = bottom[0]->mutable_cpu_diff(); 
        caffe_copy<Dtype>(count, dual_before_proj_diff, dual_diff);
    }

    // compute gradient with respect to dual term and weight.
    const Dtype* weight = this->blobs_[0]->cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    if(propagate_down[1] || this->param_propagate_down_[0]){

        Dtype* conv_res_diff = conv_res_.mutable_cpu_diff();
        caffe_set<Dtype>(conv_res_.count(), Dtype(0.), conv_res_diff);
        caffe_axpy<Dtype>(conv_res_.count(), Dtype(sigma_), dual_before_proj_diff, 
        conv_res_diff);


        const Dtype* primal_data = bottom[1]->cpu_data();
        const Dtype* num_data = bottom[2]->cpu_data();
        const Dtype* neighbors_data = bottom[3]->cpu_data(); 
        //Dtype* col_buff = _col_buffer.mutable_cpu_data();
        Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
        Dtype* col_buff = col_buffer.mutable_cpu_data();
        for (int bt = 0; bt < _batch_size; ++bt){
     
            if(this->param_propagate_down_[0]){
                octree2col_cpu(primal_data + bt * _primal_dim, neighbors_data + bt*bottom[3]->count(1), 
                        col_buff, (int)num_data[bt]);

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                          _col_buffer_shape[0], _col_buffer_shape[1],
                          (Dtype)1., conv_res_diff + bt * _dual_dim, col_buff, (Dtype)1., weight_diff);
            }
      
            if(propagate_down[1]){

                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
                        _col_buffer_shape[1], _weight_shape[0],
                        (Dtype)1., weight, conv_res_diff + bt * _dual_dim, 
                        (Dtype)0., col_buff);

                col2octree_cpu(col_buffer.cpu_data(), neighbors_data + bt*bottom[3]->count(1), 
                        bottom[1]->mutable_cpu_diff()+ bt * _primal_dim, (int)num_data[bt]);
            }
      
        }
    }

    if(propagate_down[2] || propagate_down[3] || propagate_down[4]){
        LOG(FATAL) << "neighbors and lens input cannot be back propagated.";
    }  
}


// this octree2col function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctDualUpdateLayer<Dtype>::octree2col_cpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
                                            Dtype* col_buff, const int num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_set<Dtype>(_num_input_channels * num_neighbors * _num_output_pixels, Dtype(0.), col_buff);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    for(int i = 0; i < num_elements; i++){

        for(int ch = 0; ch < _num_input_channels; ch++){

            for(int el = 0; el < num_neighbors; el++){
  
                int nbh_pos = (int)neighbors_data[i * num_neighbors + el];

                // the neighbors existed in current octree, otherwise we simply pad zero.
                if(nbh_pos != -1){

                    int col_buff_ind = (ch * num_neighbors + el) * _num_output_pixels + i;
                    int feature_ind = ch * _num_input_pixels + nbh_pos;
                    col_buff[col_buff_ind] = bottom_data[feature_ind];

                }

            }
        }
    }
}



// this col2octree function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctDualUpdateLayer<Dtype>::col2octree_cpu(const Dtype* col_buff, const Dtype* neighbors_data, 
                                            Dtype* bottom_diff, const int num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_set<Dtype>(_num_input_channels * _num_input_pixels, Dtype(0.), bottom_diff);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    for(int i = 0; i < num_elements; i++){

        for(int ch = 0; ch < _num_input_channels; ch++){
            
            for(int el = 0; el < num_neighbors; el++){

                int nbh_pos = (int)neighbors_data[i * num_neighbors + el];

                // the neighbors existed in current octree, otherwise we simply pad zero.
                if(nbh_pos != -1){

                    int col_buff_ind = (ch * num_neighbors + el) * _num_output_pixels + i;
                    int feature_ind = ch * _num_input_pixels + nbh_pos;
                    bottom_diff[feature_ind] += col_buff[col_buff_ind];
                }

            }
        }
    }
}





#ifdef CPU_ONLY
STUB_GPU(OctDualUpdateLayer);
#endif

INSTANTIATE_CLASS(OctDualUpdateLayer);
REGISTER_LAYER_CLASS(OctDualUpdate);

}  // namespace caffe
