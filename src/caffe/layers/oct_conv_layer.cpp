#include "caffe/layers/oct_conv_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void OctConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	_filter_size = this->layer_param_.oct_conv_param().filter_size();
	_num_output_channels = this->layer_param_.oct_conv_param().output_channels();
	_num_input_channels = bottom[0]->shape(1);

	this->blobs_.resize(2);

	
    _weight_shape.push_back(_num_output_channels);
    _weight_shape.push_back(_num_input_channels);
    _weight_shape.push_back(_filter_size);
    _weight_shape.push_back(_filter_size);
    _weight_shape.push_back(_filter_size);
    

    _bias_shape.push_back(_num_output_channels);

    this->blobs_[0].reset(new Blob<Dtype>(_weight_shape));
    this->blobs_[1].reset(new Blob<Dtype>(_bias_shape));

    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.oct_conv_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
        this->layer_param_.oct_conv_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());

    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void OctConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // bottom[0]: input
    // bottom[1]: num
    // bottom[2]: neighbors  neighbors[(bt * num_input_pixels + i) * num_neighbors+ el]: the el-th neighbor positions  for octree cell i
    // bottom[3]: neighbor_of int j = neighbour_of[(bt * num_input_pixels + i) * num_neighbors+ el]: octree cell i is the el-th neighbor of j
  

    CHECK_EQ(bottom[0]->num_axes(), 3)
        << "The number of axes in octconv must be 3";

    CHECK_EQ(bottom[0]->shape(1), _num_input_channels)
        << "input channels must be equal to shape(1) of bottom";

    // newly added by xiaojuan on Oct 2nd, 2018
    CHECK_EQ(_filter_size * _filter_size * _filter_size, bottom[2]->shape(2))
        << "reference neighbors matrix must comply with the filter size.";

	_batch_size = bottom[0]->shape(0);
    _num_input_pixels = bottom[0]->shape(2);
    _num_output_pixels = _num_input_pixels;

    vector<int> features_shape;
    features_shape.push_back(_batch_size);
    features_shape.push_back(_num_output_channels);
    features_shape.push_back(_num_output_pixels);
    top[0]->Reshape(features_shape);
    _bottom_dim = bottom[0]->count(1);
    _top_dim = top[0]->count(1);


    _col_buffer_shape.clear();
    _col_buffer_shape.push_back(_weight_shape[1] * _filter_size * _filter_size * _filter_size);
    _col_buffer_shape.push_back(_num_output_pixels);
    /*_col_buffer.Reshape(_col_buffer_shape); */

    // request col buffer from current thread
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    col_buffer.Reshape(_col_buffer_shape);  


    vector<int> bias_multiplier_shape;
    bias_multiplier_shape.push_back(_num_output_pixels); bias_multiplier_shape.push_back(1);
    _bias_multiplier.Reshape(bias_multiplier_shape);
}


template <typename Dtype>
void OctConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* weight = this->blobs_[0]->cpu_data();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* num_data = bottom[1]->cpu_data();
    const Dtype* neighbors_data = bottom[2]->cpu_data(); 

    Dtype* top_data = top[0]->mutable_cpu_data();
    
    //Dtype* col_buff = _col_buffer.mutable_cpu_data();
    Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
    Dtype* col_buff = col_buffer.mutable_cpu_data();

    for (int bt = 0; bt <_batch_size; bt++){

        octree2col_cpu(bottom_data + bt * _bottom_dim, neighbors_data + bt * bottom[2]->count(1), 
                        col_buff, (int)num_data[bt]);

        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _weight_shape[0],
            _col_buffer_shape[1], _col_buffer_shape[0],
            (Dtype)1., weight, col_buff,
            (Dtype)0., top_data + bt * _top_dim);

        caffe_set(_num_output_pixels, Dtype(0), _bias_multiplier.mutable_cpu_data());
        caffe_set((int)num_data[bt], Dtype(1), _bias_multiplier.mutable_cpu_data());
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, _bias_shape[0], _num_output_pixels, 1,
                          (Dtype)1., this->blobs_[1]->cpu_data(), _bias_multiplier.cpu_data(),
                          (Dtype)1., top_data + bt * _top_dim);
    }

}

template <typename Dtype>
void OctConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    
    if(propagate_down[0] || this->param_propagate_down_[0] || this->param_propagate_down_[1]){
        
        const Dtype* weight = this->blobs_[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* num_data = bottom[1]->cpu_data();
        const Dtype* neighbors_data = bottom[2]->cpu_data(); 

        //Dtype* col_buff = _col_buffer.mutable_cpu_data();
        Blob<Dtype>& col_buffer = ColBuffer::get_col_buffer(Dtype(0));
        Dtype* col_buff = col_buffer.mutable_cpu_data();

        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        for (int bt = 0; bt < _batch_size; ++bt){
            
            if(this->param_propagate_down_[0]){
                
                octree2col_cpu(bottom_data + bt * _bottom_dim, neighbors_data + bt*bottom[2]->count(1), 
                        col_buff, (int)num_data[bt]);
                
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, _weight_shape[0],
                          _col_buffer_shape[0], _col_buffer_shape[1],
                          (Dtype)1., top_diff + bt * _top_dim, col_buff, (Dtype)1., weight_diff);
            }

            if(this->param_propagate_down_[1]){
                
                caffe_set(_num_output_pixels, Dtype(0), _bias_multiplier.mutable_cpu_data());
                caffe_set((int)num_data[bt], Dtype(1), _bias_multiplier.mutable_cpu_data());
                caffe_cpu_gemv<Dtype>(CblasNoTrans, _num_output_channels, _num_output_pixels, 1.,
                    top_diff+bt*_top_dim, _bias_multiplier.cpu_data(), (Dtype)1., bias_diff);
            }


      
            if(propagate_down[0]){
                
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, _weight_shape[1] * _filter_size * _filter_size * _filter_size,
                        _col_buffer_shape[1], _weight_shape[0],
                        (Dtype)1., weight, top_diff + bt * _top_dim, 
                        (Dtype)0., col_buff);
                col2octree_cpu(col_buffer.cpu_data(), neighbors_data + bt*bottom[2]->count(1), 
                        bottom_diff + bt * _bottom_dim, (int)num_data[bt]);
            }
      
        }
    }

    if(propagate_down[1] || propagate_down[2] || propagate_down[3]){
        LOG(FATAL) << "neighbors and lens input cannot be back propagated.";
    }  



}


// this octree2col function only appicable to convolution with stride 1, and output the same size as input
template <typename Dtype>
void OctConvLayer<Dtype>::octree2col_cpu(const Dtype* bottom_data, const Dtype* neighbors_data, 
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
void OctConvLayer<Dtype>::col2octree_cpu(const Dtype* col_buff, const Dtype* neighbors_data, 
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
STUB_GPU(OctConvLayer);
#endif

INSTANTIATE_CLASS(OctConvLayer);
REGISTER_LAYER_CLASS(OctConv);

}  // namespace caffe
