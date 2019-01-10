#include "caffe/layers/test_input3d_layer.hpp"
#include "stdint.h"
#include <algorithm>
namespace caffe {

using namespace std;

template <typename Dtype>
void TestInput3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // read parameters

    num_classes_ = this->layer_param_.test_input3d_param().num_classes();
    row_major_ = this->layer_param_.test_input3d_param().row_major();
    num_levels_ = this->layer_param_.test_input3d_param().num_levels();

    data_file_ = this->layer_param_.test_input3d_param().data_file();
    read_gvr_datacost(data_file_, num_classes_, data_);
    
    cout << "Done reading test data from "<<data_file_<<endl;    

    
}

template <typename Dtype>
void TestInput3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    vector<int> shape(5);
    shape[0] = 1; // add extra axis
    shape[1] = data_.shape(0);
    shape[2] = 128;
    shape[3] = 128;
    shape[4] = 128;
        
    /*int mod = data_.shape(1) % (1  << (num_levels_-1));
    shape[2] = data_.shape(1) - mod ;

    mod = data_.shape(2) % (1  << (num_levels_-1));
    shape[3] = data_.shape(2) - mod ;

    mod = data_.shape(3) % (1 << (num_levels_-1));
    shape[4] = data_.shape(3) - mod ;*/

    top[0]->Reshape(shape);

    
}

template <typename Dtype>
void TestInput3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* top_data_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), 0, top_data_data);

    int hstart = 0;
    int wstart = 0;
    int dstart = 0;
    int hend = std::min(128, data_.shape(1));
    int wend = std::min(128, data_.shape(2));
    int dend = std::min(128, data_.shape(3));
    /*int hend = std::min(top[0]->shape(2), data_.shape(1));
    int wend = std::min(top[0]->shape(3), data_.shape(2));
    int dend = std::min(top[0]->shape(4), data_.shape(3));*/
    const Dtype* data_data = data_.cpu_data();

    for(int c = 0; c  < top[0]->shape(1); c++){
        for(int i = hstart; i < hend; i++){
            for(int j = wstart; j < wend; j++){
                for(int k = dstart; k < dend; k++){
                    int ind = (((c * data_.shape(1) + i) * data_.shape(2)) + j) * data_.shape(3)+ k;
                    top_data_data[((c * 128 + i-hstart) * 128 + j-wstart) * 128+ k-dstart] = data_data[ind];
                    /*top_data_data[((c * top[0]->shape(2) + i-hstart) * top[0]->shape(3) + j-wstart) * top[0]->shape(4)+ k-dstart] = data_data[ind];*/
                }
            }
        }
    }

   
}

template <typename Dtype>
void TestInput3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}


template <typename Dtype>
void TestInput3DLayer<Dtype>::read_gvr_datacost(const string& data_file_name, int num_classes, Blob<Dtype>& data){


    uint8_t version;
    uint8_t is_big_endian;
    uint8_t uint_size;
    uint32_t elem_size;
    uint32_t height;
    uint32_t width;
    uint32_t depth;
    

    ifstream binaryIo(data_file_name.c_str(), ios::binary);
    binaryIo.read((char *)(&version), sizeof(version));
    CHECK_EQ(version, 1) 
        <<" version must be 1. ";

    
    binaryIo.read((char *)(&is_big_endian), sizeof(is_big_endian));
    CHECK((is_big_endian == 1 && is_byteorder_big_endian()) || (is_big_endian == 0 && !is_byteorder_big_endian()))
        << "byteorder must be consistent. ";

    
    binaryIo.read((char *)(&uint_size), sizeof(uint_size));
    CHECK_EQ(uint_size, 4)
        << " uint_size must be 4. ";

    
    binaryIo.read((char *)(&elem_size), sizeof(elem_size));
    CHECK(elem_size == 4 || elem_size == 8)
        << "elem size must be 4 or 8. ";


    
    binaryIo.read((char *)(&height), sizeof(height));
    binaryIo.read((char *)(&width), sizeof(width));
    binaryIo.read((char *)(&depth), sizeof(depth));
    CHECK_EQ(height % num_classes, 0)
        << "width must be multiple of num_classes .";

    height = height / num_classes;
    int num_elems = width * height * depth * num_classes;


    CHECK_GT(num_elems, 0)
        <<"num_elems must be greater than 0.";

    vector<int> shape(4);
    shape[0] = num_classes;
    shape[1] = height;
    shape[2] = width;
    shape[3] = depth;
    

    data.Reshape(shape);
    Dtype* data_data = data.mutable_cpu_data();
    if(elem_size == 4){

        Blob<float> fdata(shape);
        float* fdata_data = fdata.mutable_cpu_data();
        binaryIo.read((char *)(fdata_data), num_elems* sizeof(float));

        for(int i = 0; i < data.count(); i++) data_data[i] = fdata_data[i];
        
    }else{

        Blob<double> ddata(shape);
        double* ddata_data = ddata.mutable_cpu_data();
        binaryIo.read((char *)(ddata_data), num_elems* sizeof(double));
        for(int i = 0; i < data.count(); i++) data_data[i] = ddata_data[i];
    }

    binaryIo.close();

    if(!row_major_){
        Blob<Dtype> data_trans;
        data_trans.Reshape(shape);
        Dtype* data_trans_data = data_trans.mutable_cpu_data();

        for(int k = 0; k < depth; k++){
            for(int j = 0;  j < width; j++){
                for(int i = 0; i < height; i++){
                    for(int c = 0;  c < num_classes; c++){
                        data_trans_data[((c*height + i)* width + j)* depth + k] = *(data_data++);
                    }
                }
            }
        }

        data_data = data.mutable_cpu_data();
        caffe_copy<Dtype>(data.count(), data_trans_data, data_data);


    }    

}


INSTANTIATE_CLASS(TestInput3DLayer);
REGISTER_LAYER_CLASS(TestInput3D);

}  // namespace caffe
