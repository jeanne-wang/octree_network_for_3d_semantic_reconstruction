#include "caffe/layers/dense_grid_data_write_layer.hpp"
#include "caffe/util/rng.hpp"
namespace caffe {

using namespace std;

template <typename Dtype>
void DenseGridDataWriteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    input_source_ = this->layer_param_.dense_grid_data_write_param().input_source();
    _model_counter = 0;
    _done_initial_reshape = false;


    std::ifstream infile(input_source_.c_str());
    std::string name;
    while(infile >> name){
        std::size_t found = name.find_last_of("/\\");
        string write_file_path = name.substr(0, found);
        string input_file_name = name.substr(found+1);
        string ind = string(1, input_file_name[8]);
        string write_name = write_file_path + "/probs" + ind + ".dat";
        _write_file_names.push_back(write_name);
    }
}

template <typename Dtype>
void DenseGridDataWriteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){   
    if(!_done_initial_reshape){
        _done_initial_reshape = true;
    }else{

        data_write_file_name_ = _write_file_names[_model_counter];
        _model_counter++;
        if(_model_counter == _write_file_names.size()) _model_counter = 0;
    }
 
    CHECK_EQ(bottom[0]->shape(0), 1)
        << "only one data for write";
}

template <typename Dtype>
void DenseGridDataWriteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // for debug
    ifstream existCheck(data_write_file_name_.c_str(), ios::binary);
    if(existCheck.good()) return;
    //

    LOG(INFO)<<"writing to file "<<data_write_file_name_;

    uint8_t version = 1;
    uint8_t is_big_endian = is_byteorder_big_endian()? 1:0 ;
    uint8_t uint_size = 4;
    uint32_t elem_size = 4;
    uint32_t num_classes = bottom[0]->num_axes() == 5? bottom[0]->shape(1) : 1;
    uint32_t height = bottom[0]->shape(bottom[0]->num_axes()-3);
    uint32_t width = bottom[0]->shape(bottom[0]->num_axes()-2);
    uint32_t depth = bottom[0]->shape(bottom[0]->num_axes()-1);
   
    ofstream binaryIo(data_write_file_name_.c_str(), ios::binary);
    binaryIo.write((char *)(&version), sizeof(version));
    binaryIo.write((char *)(&is_big_endian), sizeof(is_big_endian));
    binaryIo.write((char *)(&uint_size), sizeof(uint_size));
    
    binaryIo.write((char *)(&elem_size), sizeof(elem_size));


    binaryIo.write((char *)(&num_classes), sizeof(num_classes));
    binaryIo.write((char *)(&height), sizeof(height));
    binaryIo.write((char *)(&width), sizeof(width));
    binaryIo.write((char *)(&depth), sizeof(depth));  
    int num_elems = width * height * depth * num_classes;


    
    Blob<float> data_cast;
    data_cast.Reshape(bottom[0]->shape());

    const Dtype* data = bottom[0]->cpu_data();
    float* data_cast_data = data_cast.mutable_cpu_data();
    for(int i = 0; i < bottom[0]->count(); i++){
        data_cast_data[i] = data[i];
    }

    binaryIo.write((char *)(data_cast_data), num_elems* sizeof(float));
    binaryIo.close();
   
}

template <typename Dtype>
void DenseGridDataWriteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_CLASS(DenseGridDataWriteLayer);
REGISTER_LAYER_CLASS(DenseGridDataWrite);

}  // namespace caffe
