#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/oct_primal_update_layer.hpp"

#include "caffe/layers/oct_layer.hpp"
#include "image_tree_tools/octree.h"
namespace caffe {

template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LOG(INFO) << "PrimalUpdateLayer set up. ";
  
    // bottom[0]: primal
    // bottom[1]: dual
    // bottom[2]: lagrangian
    // bottom[3]: data cost
    // bottom[4]: num
    // bottom[5]: neighbors  neighbors[(bt * num_input_pixels + i) * num_neighbors+ el]: the el-th neighbor positions  for octree cell i
    // bottom[6]: neighbor_of  int j = neighbour_of[(bt * num_input_pixels + i) * num_neighbors+ el]: octree cell i is the el-th neighbor of j
  
    // deconvolution set up
    _filter_size = this->layer_param_.oct_primal_update_param().filter_size();
    _num_output_channels = this->layer_param_.oct_primal_update_param().output_channels();
    _num_input_channels = bottom[1]->shape(1);

    _weight_shape.push_back(_num_input_channels);
    _weight_shape.push_back(_num_output_channels);
    _weight_shape.push_back(_filter_size);
    _weight_shape.push_back(_filter_size);
    _weight_shape.push_back(_filter_size);

    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(_weight_shape));
      
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.oct_primal_update_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // Propagate gradients to the parameters (as directed by backward pass).
    this->param_propagate_down_.resize(this->blobs_.size(), true);

    // read parameter tau used in primal update formula
    tau_ = this->layer_param_.oct_primal_update_param().tau();

    primal_before_proj_.ReshapeLike(*bottom[0]);
    primal_projection_bottom_vecs_.clear();
    primal_projection_bottom_vecs_.push_back(&primal_before_proj_);

    LayerParameter primal_projection_layer_param;
    primal_projection_layer_.reset(new PrimalProjectionLayer<Dtype>(primal_projection_layer_param));
    primal_projection_layer_->SetUp(primal_projection_bottom_vecs_, top);
}

template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    _batch_size = bottom[0]->shape(0);
    _num_input_pixels = bottom[0]->shape(2);

    // check the batch size
    for(int bottom_id = 1; bottom_id < bottom.size()-3; bottom_id++){
        CHECK_EQ(bottom[bottom_id]->shape(0), _batch_size)
            <<"All variables must have the same number data. ";
    }

    // check input voxels
    for(int bottom_id = 1; bottom_id < bottom.size()-3; bottom_id++){
        CHECK_EQ(bottom[bottom_id]->shape(2), _num_input_pixels)
            << "Primal and Dual variable must have the same input voxels";
    }

    CHECK_EQ(bottom[2]->shape(1), 1)
        << "lagrangian variable has one channel";

    CHECK_EQ(bottom[1]->shape(1), _num_input_channels)
        << "Input size incompatible with convolution kernel.";
    // channels of dual should be 3 times of primal
    CHECK_EQ(bottom[0]->shape(1), _num_output_channels)
        <<"output channels must equal to number of channels in primal data";

    CHECK_EQ(bottom[0]->shape(1), bottom[3]->shape(1))
        <<"primal and datacost must have the same number of channels";

    // newly added by xiaojuan on Sep 6th, 2018
    CHECK_EQ(_filter_size * _filter_size * _filter_size, bottom[5]->shape(2))
        << "reference neighbors matrix must comply with the filter size.";


    _num_output_pixels = _num_input_pixels;
    _dual_dim = bottom[1]->count(1);
    _primal_dim = bottom[0]->count(1);

    // Shape the tops.
    top[0]->ReshapeLike(*bottom[0]);
    deconv_res_.ReshapeLike(*bottom[0]);
    primal_before_proj_.ReshapeLike(*bottom[0]);
  
    _col_buffer_shape.clear();
    _col_buffer_shape.push_back(_weight_shape[1] * _filter_size * _filter_size * _filter_size);
    _col_buffer_shape.push_back(_num_input_pixels);
    //_col_buffer.Reshape(_col_buffer_shape);

    // request col buffer from current thread
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    col_buffer.Reshape(_col_buffer_shape);  

}

template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* weight = this->blobs_[0]->cpu_data();
    const Dtype* dual_data = bottom[1]->cpu_data();
    const Dtype* num_data = bottom[4]->cpu_data();
    const Dtype* neighbors_data = bottom[5]->cpu_data(); 

    Dtype* deconv_res_data = deconv_res_.mutable_cpu_data();
    //Dtype* col_buff = _col_buffer.mutable_cpu_data();
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    Dtype* col_buff = col_buffer.mutable_cpu_data();
    for (int bt = 0; bt <_batch_size; bt++){

        caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
            _col_buffer_shape[1], _weight_shape[0],
            (Dtype)1., weight, dual_data + bt * _dual_dim,
            (Dtype)0., col_buff);

        col2octree_cpu(col_buffer.cpu_data(), neighbors_data + bt * bottom[5]->count(1), deconv_res_data + bt * _primal_dim, (int)num_data[bt]);
    }


    Dtype* primal_before_proj_data = primal_before_proj_.mutable_cpu_data();
    const Dtype* deconv_res_data_const = deconv_res_.cpu_data();
    const int count = primal_before_proj_.count();
    caffe_copy<Dtype>(count, deconv_res_data_const, primal_before_proj_data);
    // add datacost term
    caffe_axpy<Dtype>(count, Dtype(1.), bottom[3]->cpu_data(), primal_before_proj_data);

    // add lagrangian
    const Dtype* lagrangian_data = bottom[2]->cpu_data();
    for(int n = 0; n < _batch_size; n++){
        for(int c = 0; c < primal_before_proj_.shape(1); c++){   
            caffe_axpy<Dtype>(_num_output_pixels, Dtype(1.), lagrangian_data + n * _num_input_pixels, 
                primal_before_proj_data+ n * _num_output_pixels * _num_output_channels + c * _num_output_pixels);
        }
    }


    caffe_scal<Dtype>(count, Dtype(-1. * tau_), primal_before_proj_data);
    // add previous primal data
    caffe_axpy<Dtype>(count, Dtype(1.), bottom[0]->cpu_data(), primal_before_proj_data);

    // projection to [0, 1]
    primal_projection_layer_->Forward(primal_projection_bottom_vecs_, top); 

}

template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


    vector<bool> down;
    down.clear();
    down.push_back(true);
    primal_projection_layer_->Backward(top, down, primal_projection_bottom_vecs_);
    const Dtype* primal_before_proj_diff = primal_before_proj_.cpu_diff();

    // compute gradient with respect to input previous primal data  
    const int count = bottom[0]->count();
    if(propagate_down[0]){
        Dtype* primal_diff = bottom[0]->mutable_cpu_diff(); 
        caffe_copy<Dtype>(count, primal_before_proj_diff, primal_diff); 
    }
  

    // compute gradient with respect to data cost term
    if(propagate_down[3]){
        Dtype* datacost_diff = bottom[3]->mutable_cpu_diff();
        caffe_set<Dtype>(count, Dtype(0), datacost_diff);
        caffe_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, datacost_diff);
    }
  
  // compute gradient with respect to lagrangian term

    if(propagate_down[2]){
        Dtype* lagrangian_diff = bottom[2]->mutable_cpu_diff();
        caffe_set<Dtype>(bottom[2]->count(), Dtype(0.), lagrangian_diff);
        for(int n = 0; n < _batch_size; n++){
            for(int c = 0; c < primal_before_proj_.shape(1); c++){     
                caffe_axpy<Dtype>(_num_output_pixels, Dtype(-1.*tau_), 
                    primal_before_proj_diff + n * _num_output_channels * _num_output_pixels + c * _num_output_pixels,
                    lagrangian_diff + n * _num_output_pixels);
            }
        }
    }


    // compute gradient with respect to dual term and weight.
    if(propagate_down[1] || this->param_propagate_down_[0]){

        const Dtype* weight = this->blobs_[0]->cpu_data();
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

        Dtype* deconv_res_diff = deconv_res_.mutable_cpu_diff();
        caffe_set<Dtype>(count, Dtype(0.), deconv_res_diff);
        caffe_axpy<Dtype>(count, Dtype(-1.*tau_), primal_before_proj_diff, 
            deconv_res_diff); 

        const Dtype* dual_data = bottom[1]->cpu_data();
        const Dtype* num_data = bottom[4]->cpu_data();
        const Dtype* neighbors_data = bottom[5]->cpu_data(); 
        //Dtype* col_buff = _col_buffer.mutable_cpu_data();
        Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
        Dtype* col_buff = col_buffer.mutable_cpu_data();
        Dtype* dual_diff =  bottom[1]->mutable_cpu_diff();
        for (int bt = 0; bt < _batch_size; ++bt){

            octree2col_cpu(deconv_res_.cpu_diff() + bt * _primal_dim, neighbors_data + bt * bottom[5]->count(1), col_buff, (int)num_data[bt]);
            if(this->param_propagate_down_[0]){

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                    _col_buffer_shape[0], _col_buffer_shape[1],
                    (Dtype)1., dual_data + bt * _dual_dim, col_buff, (Dtype)1., weight_diff);
            }
      
            if(propagate_down[1]){

                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
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


// this col2octree function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::col2octree_cpu(const Dtype* col_buff, const Dtype* neighbors_data, 
                                            Dtype* top_data, const int num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_set<Dtype>(_num_output_channels * _num_output_pixels, Dtype(0.), top_data);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    for(int i = 0; i < num_elements; i++){

        for(int ch = 0; ch < _num_output_channels; ch++){
            
            for(int el = 0; el < num_neighbors; el++){

                int nbh_pos = (int)neighbors_data[i * num_neighbors + el];

                // the neighbors existed in current octree, otherwise we simply pad zero.
                if(nbh_pos != -1){

                    int col_buff_ind = (ch * num_neighbors + el) * _num_input_pixels + i;
                    int feature_ind = ch * _num_output_pixels + nbh_pos;
                    top_data[feature_ind] += col_buff[col_buff_ind];
                }

            }
        }
    }
}


// this octree2col function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctPrimalUpdateLayer<Dtype>::octree2col_cpu(const Dtype* top_diff, const Dtype* neighbors_data, 
                                            Dtype* col_buff, const int num_elements){
  
    const int num_neighbors = _filter_size * _filter_size * _filter_size;
    caffe_set<Dtype>(_num_output_channels * num_neighbors * _num_input_pixels, Dtype(0.), col_buff);
  
    // num_elements refers to the number of cells in current octree
    // num_output_pixels refers to the maxium number of cells in octrees in the same batch
    // num_output_pixels = num_input_pixels
    for(int i = 0; i < num_elements; i++){

        for(int ch = 0; ch < _num_output_channels; ch++){

            for(int el = 0; el < num_neighbors; el++){
  
                int nbh_pos = (int)neighbors_data[i * num_neighbors + el];

                // the neighbors existed in current octree, otherwise we simply pad zero.
                if(nbh_pos != -1){

                    int col_buff_ind = (ch * num_neighbors + el) * _num_input_pixels + i;
                    int feature_ind = ch * _num_output_pixels + nbh_pos;
                    col_buff[col_buff_ind] = top_diff[feature_ind];

                }

            }
        }
    }
}



#ifdef CPU_ONLY
STUB_GPU(OctPrimalUpdateLayer);
#endif

INSTANTIATE_CLASS(OctPrimalUpdateLayer);
REGISTER_LAYER_CLASS(OctPrimalUpdate);

}  // namespace caffe