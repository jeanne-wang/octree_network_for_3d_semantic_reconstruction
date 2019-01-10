#include "caffe/layers/input3d_layer.hpp"
#include "caffe/util/rng.hpp"

#include "stdint.h"
#include <algorithm>
namespace caffe {

using namespace std;

template <typename Dtype>
void Input3DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    init_rand();
    model_counter_ = 0;
    // read parameters
    batch_size_ = this->layer_param_.input3d_param().batch_size();
    out_height_ = this->layer_param_.input3d_param().height();
    out_width_ = this->layer_param_.input3d_param().width();
    out_depth_ = this->layer_param_.input3d_param().depth();
    num_classes_ = this->layer_param_.input3d_param().num_classes();
    row_major_ = this->layer_param_.input3d_param().row_major();
    customized_crop_ = this->layer_param_.input3d_param().customized_crop();
    load_data_from_disk();
}

template <typename Dtype>
void Input3DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    vector<int> top_shape;
    top_shape.push_back(batch_size_);
    top_shape.push_back(num_classes_);
    top_shape.push_back(out_height_);
    top_shape.push_back(out_width_);
    top_shape.push_back(out_depth_);

    top[0]->Reshape(top_shape); // data cost
    top[1]->Reshape(top_shape); // groundtruth u
    
}

template <typename Dtype>
void Input3DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    /*vector<int> batch_indexes;
    for(int bt=0; bt<batch_size_; bt++)
    {
        batch_indexes.push_back(model_counter_++);
        if(model_counter_ == data_file_names_.size()) model_counter_ = 0;
            
    }

     generate_batch(batch_indexes, *top[0], *top[1]);*/

    generate_batch_one_scene(model_counter_, *top[0], *top[1]);
    model_counter_++;
    if(model_counter_ == data_file_names_.size()) model_counter_ = 0;

   
}

template <typename Dtype>
void Input3DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}

template <typename Dtype>
void Input3DLayer<Dtype>::load_data_from_disk()
{
    cout << "Loading training data from disk..." << endl;
    data_list_file_ = this->layer_param_.input3d_param().data_list_file();
    groundtruth_list_file_ = this->layer_param_.input3d_param().groundtruth_list_file();
    preload_data_ = this->layer_param_.input3d_param().preload_data();

    ifstream data_infile(data_list_file_.c_str());
    ifstream gt_infile(groundtruth_list_file_.c_str());
    string data_name, gt_name;

    data_file_names_.clear();
    gt_file_names_.clear();
    data_.clear();
    gt_.clear();
    while(data_infile >> data_name && gt_infile >> gt_name)
    {
        data_file_names_.push_back(data_name);
        gt_file_names_.push_back(gt_name);
        cout << data_name << " & "<< gt_name << endl;
    }

    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    vector<int> idx(data_file_names_.size());
    for(int i = 0; i < idx.size(); i++) idx[i] = i;
    shuffle(idx.begin(), idx.end(), prefetch_rng);
    reorder(data_file_names_, idx);
    reorder(gt_file_names_, idx);

    
    if(preload_data_){
        for(int i = 0; i < data_file_names_.size(); i++){
            data_.push_back(new Blob<Dtype>());
            gt_.push_back(new Blob<Dtype>());

            read_gvr_datacost(data_file_names_[i], num_classes_, *data_[i]);
            read_gvr_groundtruth(gt_file_names_[i], *gt_[i]);
            CHECK_EQ(data_[i]->shape(0), gt_[i]->shape(0))
                <<"number of channels of datacost and groundtruth must be the same.";
            CHECK_EQ(data_[i]->shape(1), gt_[i]->shape(1))
                <<"height of datacost and groundtruth must be the same.";
            CHECK_EQ(data_[i]->shape(2), gt_[i]->shape(2))
                <<"width of datacost and groundtruth must be the same.";
            CHECK_EQ(data_[i]->shape(3), gt_[i]->shape(3))
                <<"depth of datacost and groundtruth must be the same.";

            cout << "reading data from "<<data_file_names_[i]<<" & "<<gt_file_names_[i]<<endl;
            
        }            
    }

    std::cout << "Done." << std::endl;
}

template <typename Dtype>
void Input3DLayer<Dtype>::reorder(vector<string> & file_names, vector<int> order )
{
    int i,j,k;
    for(i = 0; i < order.size() - 1; ++i) {
        j = order[i];
        if(j != i) {
            for(k = i + 1; order[k] != i; ++k);
            std::swap(order[i],order[k]);
            std::swap(file_names[i],file_names[j]);
        }
    }
}

template <typename Dtype>
void Input3DLayer<Dtype>::read_gvr_datacost(const string& data_file_name, int num_classes, Blob<Dtype>& data){


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

template <typename Dtype>
void Input3DLayer<Dtype>::read_gvr_groundtruth(const string& gt_file_name, Blob<Dtype>& gt){

    uint8_t version;
    uint8_t is_big_endian;
    uint8_t uint_size;
    uint32_t elem_size;
    uint32_t num_classes;
    uint32_t height;
    uint32_t width;
    uint32_t depth;

    ifstream binaryIo(gt_file_name.c_str(), ios::binary);
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

    binaryIo.read((char *)(&num_classes), sizeof(num_classes));
    binaryIo.read((char *)(&height), sizeof(height));
    binaryIo.read((char *)(&width), sizeof(width));
    binaryIo.read((char *)(&depth), sizeof(depth));
   
    int num_elems = width * height * depth * num_classes;
    CHECK_GT(num_elems, 0)
        <<"num_elems must be greater than 0.";

    vector<int> shape(4);
    shape[0] = num_classes;
    shape[1] = height;
    shape[2] = width;
    shape[3] = depth;
    gt.Reshape(shape);


    Dtype* gt_data = gt.mutable_cpu_data();
    if(elem_size == 4){
        Blob<float> fgt(shape);
        float* fgt_data = fgt.mutable_cpu_data();
        binaryIo.read((char *)(fgt_data), num_elems* sizeof(float));

        for(int i = 0; i < gt.count(); i++) gt_data[i] = fgt_data[i];
        
    }else{
        Blob<double> dgt(shape);
        double* dgt_data = dgt.mutable_cpu_data();
        binaryIo.read((char *)(dgt_data), num_elems* sizeof(double));

        for(int i = 0; i < gt.count(); i++) gt_data[i] = dgt_data[i];
    }
    binaryIo.close();

    if(!row_major_){
        Blob<Dtype> gt_trans;
        gt_trans.Reshape(shape);
        Dtype* gt_trans_data = gt_trans.mutable_cpu_data();
        for(int k = 0; k < depth; k++){
            for(int j = 0;  j < width; j++){
                for(int i = 0; i < height; i++){
                    for(int c = 0;  c < num_classes; c++){
                        gt_trans_data[((c*height + i)* width + j)* depth + k] = *(gt_data++);

                    }
                }
            }
        }
        gt_data = gt.mutable_cpu_data();
        caffe_copy<Dtype>(gt.count(), gt_trans_data, gt_data);
    }    


}

template <typename Dtype>
void Input3DLayer<Dtype>::generate_batch_one_scene(int scene_idx, Blob<Dtype>& top_data, Blob<Dtype>& top_gt){

    Blob<Dtype>* data = new Blob<Dtype>();
    Blob<Dtype>* gt = new Blob<Dtype>();
    read_gvr_datacost(data_file_names_[scene_idx], num_classes_, *data);
    read_gvr_groundtruth(gt_file_names_[scene_idx], *gt);
            
    CHECK_EQ(data->shape(0), gt->shape(0))
        <<"number of channels of datacost and groundtruth must be the same.";
    CHECK_EQ(data->shape(1), gt->shape(1))
        <<"height of datacost and groundtruth must be the same.";
    CHECK_EQ(data->shape(2), gt->shape(2))
        <<"width of datacost and groundtruth must be the same.";
    CHECK_EQ(data->shape(3), gt->shape(3))
        <<"depth of datacost and groundtruth must be the same.";

    const Dtype* data_data = data->cpu_data();
    const Dtype* gt_data = gt->cpu_data();
    int nrows = data->shape(1);
    int ncols = data->shape(2);
    int nslices = data->shape(3); 
    int num_classes = data->shape(0);
    CHECK_EQ(num_classes_, num_classes)
        <<"number of classes in datacost must be the same as num_classes_.";


    Dtype* top_data_data = top_data.mutable_cpu_data();
    Dtype* top_gt_data = top_gt.mutable_cpu_data();
    const int inner_dim  = top_data.count(1);
    for (int bt = 0; bt < batch_size_; bt++){
        // crop
        int row_start;
        int col_start;
        int slice_start;
        if(customized_crop_){
            vector<int> surface_rows;
            vector<int> surface_cols;
            vector<int> surface_slices;
            surface_rows.clear();
            surface_cols.clear();
            surface_slices.clear();
            for(int i = 0; i < nrows; i++){
                for(int j = 0; j < ncols; j++){
                    for(int k = 0; k < nslices; k++){
                        bool res = false;
                        for(int c = 0; c < num_classes_; c++){
                            if(std::abs(data_data[ ((c*nrows + i) * ncols + j)* nslices + k]) > 3){
                                res = true;
                            }
                        }

                        if(res) {
                            surface_rows.push_back(i);
                            surface_cols.push_back(j);
                            surface_slices.push_back(k);
                        }
                    }
                }
            }
            int surface_idx = Rand(surface_rows.size());
            int surface_row_offset = Rand(out_height_);
            int surface_col_offset = Rand(out_width_);
            int surface_slice_offset = Rand(out_depth_);

            row_start = std::max(surface_rows[surface_idx]-surface_row_offset, 0);
            col_start = std::max(surface_cols[surface_idx]-surface_col_offset, 0);
            slice_start = std::max(surface_slices[surface_idx]-surface_slice_offset, 0);
        }else{
            row_start = Rand(std::max(nrows-out_height_, 0)+1);
            col_start = Rand(std::max(ncols-out_width_, 0)+1);
            slice_start = Rand(std::max(nslices-out_depth_, 0)+1);
        }
        

        int row_end = std::min(row_start+out_height_, nrows);
        int col_end = std::min(col_start+out_width_, ncols);
        int slice_end = std::min(slice_start + out_depth_, nslices);

        Blob<Dtype> crop_data;
        Blob<Dtype> crop_gt;
        vector<int> shape(4);
        shape[0] = num_classes_;
        shape[1] = out_height_;
        shape[2] = out_width_;
        shape[3] = out_depth_;
        crop_data.Reshape(shape);
        crop_gt.Reshape(shape);

        Dtype* crop_data_data = crop_data.mutable_cpu_data();
        Dtype* crop_gt_data = crop_gt.mutable_cpu_data();
        memset(crop_data_data, 0, sizeof(Dtype)*crop_data.count());
        Dtype default_gt_val = Dtype(1.)/Dtype(num_classes_);
        memset(crop_gt_data, default_gt_val, sizeof(Dtype)*crop_gt.count());
        for(int c = 0; c < num_classes_; c++){
            for(int i = row_start; i < row_end; i++){
                for(int j = col_start; j < col_end; j++){
                    for(int k = slice_start; k < slice_end; k++){
                        crop_data_data[(( c* out_height_ + i-row_start) * out_width_ + j-col_start)* out_width_ + k-slice_start]
                            = data_data[(( c* nrows + i) * ncols + j)* nslices + k];
                        crop_gt_data[(( c* out_height_ + i-row_start) * out_width_ + j-col_start)* out_width_ + k-slice_start]
                             = gt_data[(( c * nrows + i) * ncols + j)* nslices + k];
                    }
                }
            }
        }

        
        // ramdonly rotate from x axis to y axis
        int num_rot = Rand(4);
        if(num_rot == 1){
            CHECK_EQ(out_height_, out_width_);
            Blob<Dtype> rot_data;
            Blob<Dtype> rot_gt;
            rot_data.Reshape(shape);
            rot_gt.Reshape(shape);
            Dtype* rot_data_data = rot_data.mutable_cpu_data();
            Dtype* rot_gt_data = rot_gt.mutable_cpu_data();
        
            const Dtype* crop_data_const = crop_data.cpu_data();
            const Dtype* crop_gt_const = crop_gt.cpu_data();
            for(int c = 0;  c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index = ((c * out_height_ + j) * out_width_ + out_height_-1-i)* out_depth_ + k;
                            *(rot_data_data++) = crop_data_const[index];
                            *(rot_gt_data++) = crop_gt_const[index];
                        }
                    }
                }
            }

            caffe_copy<Dtype>(rot_data.count(), rot_data.cpu_data(), crop_data.mutable_cpu_data());
            caffe_copy<Dtype>(rot_gt.count(), rot_gt.cpu_data(), crop_gt.mutable_cpu_data());



        }else if(num_rot == 2){
            CHECK_EQ(out_height_, out_width_);
            Blob<Dtype> rot_data;
            Blob<Dtype> rot_gt;
            rot_data.Reshape(shape);
            rot_gt.Reshape(shape);
            Dtype* rot_data_data = rot_data.mutable_cpu_data();
            Dtype* rot_gt_data = rot_gt.mutable_cpu_data();
        
            const Dtype* crop_data_const = crop_data.cpu_data();
            const Dtype* crop_gt_const = crop_gt.cpu_data();
            for(int c = 0;  c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index = ((c * out_height_ + out_height_-1 - i) * out_width_ + out_height_-1-j)* out_depth_ + k;
                            *(rot_data_data++) = crop_data_const[index];
                            *(rot_gt_data++) = crop_gt_const[index];
                        }
                    }
                }
            }
            caffe_copy<Dtype>(rot_data.count(), rot_data.cpu_data(), crop_data.mutable_cpu_data());
            caffe_copy<Dtype>(rot_gt.count(), rot_gt.cpu_data(), crop_gt.mutable_cpu_data());

        }else if(num_rot == 3){
            CHECK_EQ(out_height_, out_width_);
            Blob<Dtype> rot_data;
            Blob<Dtype> rot_gt;
            rot_data.Reshape(shape);
            rot_gt.Reshape(shape);
            Dtype* rot_data_data = rot_data.mutable_cpu_data();
            Dtype* rot_gt_data = rot_gt.mutable_cpu_data();
        
            const Dtype* crop_data_const = crop_data.cpu_data();
            const Dtype* crop_gt_const = crop_gt.cpu_data();
            for(int c = 0;  c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index = ((c * out_height_ + out_height_-1-j) * out_width_ + i)* out_depth_ + k;
                            *(rot_data_data++) = crop_data_const[index];
                            *(rot_gt_data++) = crop_gt_const[index];
                        }
                    }
                }
            }
            caffe_copy<Dtype>(rot_data.count(), rot_data.cpu_data(), crop_data.mutable_cpu_data());
            caffe_copy<Dtype>(rot_gt.count(), rot_gt.cpu_data(), crop_gt.mutable_cpu_data());

        }
         
        // randomly flip along x and y axis
        crop_data_data = crop_data.mutable_cpu_data();
        crop_gt_data = crop_gt.mutable_cpu_data();
        int flip_axis = Rand(3);
        if(flip_axis == 0){

            for(int c = 0; c < num_classes_; c++){
                for(int i = 0; i < ceil(out_height_/2); i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index1 = ((c*out_height_ + i)*out_width_+j) *out_depth_+k;
                            int index2 = ((c*out_height_ + out_height_-1-i)*out_width_+j) *out_depth_+k;
                            Dtype temp = crop_data_data[index2];
                            crop_data_data[index2] = crop_data_data[index1];
                            crop_data_data[index1] = temp;

                            temp = crop_gt_data[index2];
                            crop_gt_data[index2] = crop_gt_data[index1];
                            crop_gt_data[index1] = temp;


                        }
                    }
                }

            }

        }

        else if(flip_axis == 1){
            for(int c = 0; c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < ceil(out_width_/2); j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index1 = ((c*out_height_ + i)*out_width_+j) * out_depth_+k;
                            int index2 = ((c*out_height_ + i)*out_width_+ out_width_-1-j) *out_depth_+k;
                            Dtype temp = crop_data_data[index2];
                            crop_data_data[index2] = crop_data_data[index1];
                            crop_data_data[index1] = temp;

                            temp = crop_gt_data[index2];
                            crop_gt_data[index2] = crop_gt_data[index1];
                            crop_gt_data[index1] = temp;


                        }
                    }
                }

            }
        }

        caffe_copy<Dtype>(inner_dim, crop_data_data, top_data_data);
        top_data_data += inner_dim;

        caffe_copy<Dtype>(inner_dim, crop_gt_data, top_gt_data);
        top_gt_data += inner_dim;
    }
    delete gt;
    delete data;
}
template <typename Dtype>
void Input3DLayer<Dtype>::generate_batch(const vector<int>& batch_indexes, Blob<Dtype>& top_data, Blob<Dtype>& top_gt){

    Dtype* top_data_data = top_data.mutable_cpu_data();
    Dtype* top_gt_data = top_gt.mutable_cpu_data();
    const int inner_dim  = top_data.count(1);
    for (int bt = 0; bt < batch_indexes.size(); bt++){


        Blob<Dtype>* data = NULL;
        Blob<Dtype>* gt = NULL;
        if(preload_data_){
            data = data_[batch_indexes[bt]];
            gt = gt_[batch_indexes[bt]];
        }else{
            data = new Blob<Dtype>();
            gt = new Blob<Dtype>();
            read_gvr_datacost(data_file_names_[batch_indexes[bt]], num_classes_, *data);
            read_gvr_groundtruth(gt_file_names_[batch_indexes[bt]], *gt);
            CHECK_EQ(data->shape(0), gt->shape(0))
                <<"number of channels of datacost and groundtruth must be the same.";
            CHECK_EQ(data->shape(1), gt->shape(1))
                <<"height of datacost and groundtruth must be the same.";
            CHECK_EQ(data->shape(2), gt->shape(2))
                <<"width of datacost and groundtruth must be the same.";
            CHECK_EQ(data->shape(3), gt->shape(3))
                <<"depth of datacost and groundtruth must be the same.";
        }

        // crop
        const Dtype* data_data = data->cpu_data();
        const Dtype* gt_data = gt->cpu_data();
        int nrows = data->shape(1);
        int ncols = data->shape(2);
        int nslices = data->shape(3); 
        int num_classes = data->shape(0);
        CHECK_EQ(num_classes_, num_classes)
            <<"number of classes in datacost must be the same as num_classes_.";

        int row_start;
        int col_start;
        int slice_start;
        if(customized_crop_){
            vector<int> surface_rows;
            vector<int> surface_cols;
            vector<int> surface_slices;
            surface_rows.clear();
            surface_cols.clear();
            surface_slices.clear();
            for(int i = 0; i < nrows; i++){
                for(int j = 0; j < ncols; j++){
                    for(int k = 0; k < nslices; k++){
                        bool res = false;
                        for(int c = 0; c < num_classes_; c++){
                            if(std::abs(data_data[ ((c*nrows + i) * ncols + j)* nslices + k]) > 3){
                                res = true;
                            }
                        }

                        if(res) {
                            surface_rows.push_back(i);
                            surface_cols.push_back(j);
                            surface_slices.push_back(k);
                        }
                    }
                }
            }
            int surface_idx = Rand(surface_rows.size());
            int surface_row_offset = Rand(out_height_);
            int surface_col_offset = Rand(out_width_);
            int surface_slice_offset = Rand(out_depth_);

            row_start = std::max(surface_rows[surface_idx]-surface_row_offset, 0);
            col_start = std::max(surface_cols[surface_idx]-surface_col_offset, 0);
            slice_start = std::max(surface_slices[surface_idx]-surface_slice_offset, 0);
        }else{
            row_start = Rand(std::max(nrows-out_height_, 0)+1);
            col_start = Rand(std::max(ncols-out_width_, 0)+1);
            slice_start = Rand(std::max(nslices-out_depth_, 0)+1);
        }
        

        int row_end = std::min(row_start+out_height_, nrows);
        int col_end = std::min(col_start+out_width_, ncols);
        int slice_end = std::min(slice_start + out_depth_, nslices);

        Blob<Dtype> crop_data;
        Blob<Dtype> crop_gt;
        vector<int> shape(4);
        shape[0] = num_classes_;
        shape[1] = out_height_;
        shape[2] = out_width_;
        shape[3] = out_depth_;
        crop_data.Reshape(shape);
        crop_gt.Reshape(shape);

        Dtype* crop_data_data = crop_data.mutable_cpu_data();
        Dtype* crop_gt_data = crop_gt.mutable_cpu_data();
        memset(crop_data_data, 0, sizeof(Dtype)*crop_data.count());
        memset(crop_gt_data, 0, sizeof(Dtype)*crop_gt.count());
        for(int c = 0; c < num_classes_; c++){
            for(int i = row_start; i < row_end; i++){
                for(int j = col_start; j < col_end; j++){
                    for(int k = slice_start; k < slice_end; k++){
                        *(crop_data_data++) = data_data[(( c* nrows + i) * ncols + j)* nslices + k];
                        *(crop_gt_data++) = gt_data[(( c * nrows + i) * ncols + j)* nslices + k];
                    }
                }
            }
        }

        
        // ramdonly rotate from x axis to y axis
        int num_rot = Rand(4);
        if(num_rot == 1){
            CHECK_EQ(out_height_, out_width_);
            Blob<Dtype> rot_data;
            Blob<Dtype> rot_gt;
            rot_data.Reshape(shape);
            rot_gt.Reshape(shape);
            Dtype* rot_data_data = rot_data.mutable_cpu_data();
            Dtype* rot_gt_data = rot_gt.mutable_cpu_data();
        
            const Dtype* crop_data_const = crop_data.cpu_data();
            const Dtype* crop_gt_const = crop_gt.cpu_data();
            for(int c = 0;  c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index = ((c * out_height_ + j) * out_width_ + out_height_-1-i)* out_depth_ + k;
                            *(rot_data_data++) = crop_data_const[index];
                            *(rot_gt_data++) = crop_gt_const[index];
                        }
                    }
                }
            }

            caffe_copy<Dtype>(rot_data.count(), rot_data.cpu_data(), crop_data.mutable_cpu_data());
            caffe_copy<Dtype>(rot_gt.count(), rot_gt.cpu_data(), crop_gt.mutable_cpu_data());



        }else if(num_rot == 2){
            CHECK_EQ(out_height_, out_width_);
            Blob<Dtype> rot_data;
            Blob<Dtype> rot_gt;
            rot_data.Reshape(shape);
            rot_gt.Reshape(shape);
            Dtype* rot_data_data = rot_data.mutable_cpu_data();
            Dtype* rot_gt_data = rot_gt.mutable_cpu_data();
        
            const Dtype* crop_data_const = crop_data.cpu_data();
            const Dtype* crop_gt_const = crop_gt.cpu_data();
            for(int c = 0;  c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index = ((c * out_height_ + out_height_-1 - i) * out_width_ + out_height_-1-j)* out_depth_ + k;
                            *(rot_data_data++) = crop_data_const[index];
                            *(rot_gt_data++) = crop_gt_const[index];
                        }
                    }
                }
            }
            caffe_copy<Dtype>(rot_data.count(), rot_data.cpu_data(), crop_data.mutable_cpu_data());
            caffe_copy<Dtype>(rot_gt.count(), rot_gt.cpu_data(), crop_gt.mutable_cpu_data());

        }else if(num_rot == 3){
            CHECK_EQ(out_height_, out_width_);
            Blob<Dtype> rot_data;
            Blob<Dtype> rot_gt;
            rot_data.Reshape(shape);
            rot_gt.Reshape(shape);
            Dtype* rot_data_data = rot_data.mutable_cpu_data();
            Dtype* rot_gt_data = rot_gt.mutable_cpu_data();
        
            const Dtype* crop_data_const = crop_data.cpu_data();
            const Dtype* crop_gt_const = crop_gt.cpu_data();
            for(int c = 0;  c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index = ((c * out_height_ + out_height_-1-j) * out_width_ + i)* out_depth_ + k;
                            *(rot_data_data++) = crop_data_const[index];
                            *(rot_gt_data++) = crop_gt_const[index];
                        }
                    }
                }
            }
            caffe_copy<Dtype>(rot_data.count(), rot_data.cpu_data(), crop_data.mutable_cpu_data());
            caffe_copy<Dtype>(rot_gt.count(), rot_gt.cpu_data(), crop_gt.mutable_cpu_data());

        }
         
        // randomly flip along x and y axis
        crop_data_data = crop_data.mutable_cpu_data();
        crop_gt_data = crop_gt.mutable_cpu_data();
        int flip_axis = Rand(3);
        if(flip_axis == 0){

            for(int c = 0; c < num_classes_; c++){
                for(int i = 0; i < ceil(out_height_/2); i++){
                    for(int j = 0; j < out_width_; j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index1 = ((c*out_height_ + i)*out_width_+j) *out_depth_+k;
                            int index2 = ((c*out_height_ + out_height_-1-i)*out_width_+j) *out_depth_+k;
                            Dtype temp = crop_data_data[index2];
                            crop_data_data[index2] = crop_data_data[index1];
                            crop_data_data[index1] = temp;

                            temp = crop_gt_data[index2];
                            crop_gt_data[index2] = crop_gt_data[index1];
                            crop_gt_data[index1] = temp;


                        }
                    }
                }

            }

        }

        else if(flip_axis == 1){
            for(int c = 0; c < num_classes_; c++){
                for(int i = 0; i < out_height_; i++){
                    for(int j = 0; j < ceil(out_width_/2); j++){
                        for(int k = 0; k < out_depth_; k++){
                            int index1 = ((c*out_height_ + i)*out_width_+j) * out_depth_+k;
                            int index2 = ((c*out_height_ + i)*out_width_+ out_width_-1-j) *out_depth_+k;
                            Dtype temp = crop_data_data[index2];
                            crop_data_data[index2] = crop_data_data[index1];
                            crop_data_data[index1] = temp;

                            temp = crop_gt_data[index2];
                            crop_gt_data[index2] = crop_gt_data[index1];
                            crop_gt_data[index1] = temp;


                        }
                    }
                }

            }
        }

        caffe_copy<Dtype>(inner_dim, crop_data_data, top_data_data);
        top_data_data += inner_dim;

        caffe_copy<Dtype>(inner_dim, crop_gt_data, top_gt_data);
        top_gt_data += inner_dim;

        // newly added
        if(!preload_data_){
            delete gt;
            delete data;
        }
        

    }
}

template <typename Dtype>
void Input3DLayer<Dtype>::init_rand() {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
}

template <typename Dtype>
int Input3DLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(Input3DLayer);
REGISTER_LAYER_CLASS(Input3D);

}  // namespace caffe
