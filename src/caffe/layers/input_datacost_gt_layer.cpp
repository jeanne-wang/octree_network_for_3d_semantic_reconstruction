#include "caffe/layers/input_datacost_gt_layer.hpp"
#include "caffe/util/rng.hpp"
#include "stdint.h"
#include <algorithm>
namespace caffe {

using namespace std;

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    init_rand();
    model_counter_ = 0;
    // read parameters
    batch_size_ = this->layer_param_.input3d_param().batch_size();
    out_height_ = this->layer_param_.input3d_param().height();
    out_width_ = this->layer_param_.input3d_param().width();
    out_depth_ = this->layer_param_.input3d_param().depth();
    num_classes_ = this->layer_param_.input3d_param().num_classes();

    freespace_label_= this->layer_param_.input3d_param().freespace_label();
    ceiling_label_ = this->layer_param_.input3d_param().ceiling_label();
    wall_label_ = this->layer_param_.input3d_param().wall_label();
    floor_label_ = this->layer_param_.input3d_param().floor_label();
    ignore_label_ = this->layer_param_.input3d_param().ignore_label();
    load_data_from_disk();
}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    vector<int> dc_shape;
    dc_shape.push_back(batch_size_);
    dc_shape.push_back(num_classes_);
    dc_shape.push_back(out_height_);
    dc_shape.push_back(out_width_);
    dc_shape.push_back(out_depth_);
    top[0]->Reshape(dc_shape); // data cost

    vector<int> gt_shape;
    gt_shape.push_back(batch_size_);
    gt_shape.push_back(out_height_);
    gt_shape.push_back(out_width_);
    gt_shape.push_back(out_depth_);
    top[1]->Reshape(gt_shape); // groundtruth u
    
}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void InputDatacostGtLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::load_data_from_disk(){

    LOG(INFO) << "Loading training data from disk...";
    data_list_file_ = this->layer_param_.input3d_param().data_list_file();
    groundtruth_list_file_ = this->layer_param_.input3d_param().groundtruth_list_file();
    

    ifstream data_infile(data_list_file_.c_str());
    ifstream gt_infile(groundtruth_list_file_.c_str());
    string data_name, gt_name;

    data_file_names_.clear();
    gt_file_names_.clear();
   
    while(data_infile >> data_name && gt_infile >> gt_name)
    {
        data_file_names_.push_back(data_name);
        gt_file_names_.push_back(gt_name);
        LOG(INFO) << data_name << " & "<< gt_name;
    }

    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    vector<int> idx(data_file_names_.size());
    for(int i = 0; i < idx.size(); i++) idx[i] = i;
    shuffle(idx.begin(), idx.end(), prefetch_rng);
    reorder(data_file_names_, idx);
    reorder(gt_file_names_, idx);

    LOG(INFO) << "Done.";
}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::reorder(vector<string> & file_names, vector<int> order )
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
void InputDatacostGtLayer<Dtype>::read_gvr_datacost(const string& data_file_name, int num_classes, Blob<Dtype>& data){


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

}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::read_gvr_groundtruth(const string& gt_file_name, Blob<Dtype>& gt){

    uint8_t version;
    uint8_t is_big_endian;
    uint8_t uint_size;
    uint32_t elem_size;
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

    binaryIo.read((char *)(&height), sizeof(height));
    binaryIo.read((char *)(&width), sizeof(width));
    binaryIo.read((char *)(&depth), sizeof(depth));
   
    int num_elems = width * height * depth;
    CHECK_GT(num_elems, 0)
        <<"num_elems must be greater than 0.";

    vector<int> shape(3);
    shape[0] = height;
    shape[1] = width;
    shape[2] = depth;
    gt.Reshape(shape);

    surface_rows_.clear();
    surface_cols_.clear();
    surface_slices_.clear();
    Dtype* gt_data = gt.mutable_cpu_data();
    if(elem_size == 4){
        Blob<float> fgt(shape);
        float* fgt_data = fgt.mutable_cpu_data();
        binaryIo.read((char *)(fgt_data), num_elems* sizeof(float));

        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                for(int k = 0; k < depth; k++){
                    int ind = (i *width + j)*depth + k;
                    gt_data[ind] = fgt_data[ind];
                    int label = static_cast<int>(gt_data[ind]);
                    if(label != freespace_label_ && label != wall_label_ && label != ceiling_label_ && label != floor_label_){
                        surface_rows_.push_back(i);
                        surface_cols_.push_back(j);
                        surface_slices_.push_back(k);
                    }
                }
            }
        }
        
        
    }else{
        Blob<double> dgt(shape);
        double* dgt_data = dgt.mutable_cpu_data();
        binaryIo.read((char *)(dgt_data), num_elems* sizeof(double));

        for(int i = 0; i < height; i++){
            for(int j = 0; j < width; j++){
                for(int k = 0; k < depth; k++){
                    int ind = (i *width + j)*depth + k;
                    gt_data[ind] = dgt_data[ind];
                    int label = static_cast<int>(gt_data[ind]);
                    if(label != freespace_label_ && label != wall_label_ && label != ceiling_label_ && label != floor_label_){
                        surface_rows_.push_back(i);
                        surface_cols_.push_back(j);
                        surface_slices_.push_back(k);
                    }
                }
            }
        }
    }
    binaryIo.close();

}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::generate_batch(const vector<int>& batch_indexes, Blob<Dtype>& top_data, Blob<Dtype>& top_gt){

    Dtype* top_data_data = top_data.mutable_cpu_data();
    Dtype* top_gt_data = top_gt.mutable_cpu_data();
    const int data_inner_dim  = top_data.count(1);
    const int gt_inner_dim = top_gt.count(1);
    for (int bt = 0; bt < batch_indexes.size(); bt++){

        Blob<Dtype>* data = new Blob<Dtype>();
        Blob<Dtype>* gt = new Blob<Dtype>();
        read_gvr_datacost(data_file_names_[batch_indexes[bt]], num_classes_, *data);
        read_gvr_groundtruth(gt_file_names_[batch_indexes[bt]], *gt);
            
        CHECK_EQ(data->shape(1), gt->shape(0))
            <<"height of datacost and groundtruth must be the same.";
        CHECK_EQ(data->shape(2), gt->shape(1))
            <<"width of datacost and groundtruth must be the same.";
        CHECK_EQ(data->shape(3), gt->shape(2))
            <<"depth of datacost and groundtruth must be the same.";
        

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
        if(surface_rows_.size() > 0){
            int surface_idx = Rand(surface_rows_.size());
            int surface_row_offset = Rand(out_height_);
            int surface_col_offset = Rand(out_width_);
            int surface_slice_offset = Rand(out_depth_);
            row_start = std::max(surface_rows_[surface_idx]-surface_row_offset, 0);
            col_start = std::max(surface_cols_[surface_idx]-surface_col_offset, 0);
            slice_start = std::max(surface_slices_[surface_idx]-surface_slice_offset, 0);
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
        vector<int> dc_shape(4);
        dc_shape[0] = num_classes_;
        dc_shape[1] = out_height_;
        dc_shape[2] = out_width_;
        dc_shape[3] = out_depth_;
        crop_data.Reshape(dc_shape);
        vector<int> gt_shape(3);
        gt_shape[0] = out_height_;
        gt_shape[1] = out_width_;
        gt_shape[2] = out_depth_;
        crop_gt.Reshape(gt_shape);

        Dtype* crop_data_data = crop_data.mutable_cpu_data();
        Dtype* crop_gt_data = crop_gt.mutable_cpu_data();
        memset(crop_data_data, 0, sizeof(Dtype)*crop_data.count());
        caffe_set<Dtype>(crop_gt.count(), ignore_label_, crop_gt_data);
        for(int c = 0; c < num_classes_; c++){
            for(int i = row_start; i < row_end; i++){
                for(int j = col_start; j < col_end; j++){
                    for(int k = slice_start; k < slice_end; k++){
                        crop_data_data[(( c* out_height_ + i-row_start) * out_width_ + j-col_start)* out_width_ + k-slice_start]
                            = data_data[(( c* nrows + i) * ncols + j)* nslices + k];
                        if(c == 0){
                            crop_gt_data[(( c* out_height_ + i-row_start) * out_width_ + j-col_start)* out_width_ + k-slice_start]
                             = gt_data[(( c * nrows + i) * ncols + j)* nslices + k];
                        }
                        
                    }
                }
            }
        }


        caffe_copy<Dtype>(data_inner_dim, crop_data.cpu_data(), top_data_data);
        top_data_data += data_inner_dim;

        caffe_copy<Dtype>(gt_inner_dim, crop_gt.cpu_data(), top_gt_data);
        top_gt_data += gt_inner_dim;

     
        
        delete gt;
        delete data;
    
    }
}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::generate_batch_one_scene(int scene_idx, Blob<Dtype>& top_data, Blob<Dtype>& top_gt){

    Blob<Dtype>* data = new Blob<Dtype>();
    Blob<Dtype>* gt = new Blob<Dtype>();
    read_gvr_datacost(data_file_names_[scene_idx], num_classes_, *data);
    read_gvr_groundtruth(gt_file_names_[scene_idx], *gt);
            
    CHECK_EQ(data->shape(1), gt->shape(0))
        <<"height of datacost and groundtruth must be the same.";
    CHECK_EQ(data->shape(2), gt->shape(1))
        <<"width of datacost and groundtruth must be the same.";
    CHECK_EQ(data->shape(3), gt->shape(2))
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
    const int data_inner_dim  = top_data.count(1);
    const int gt_inner_dim = top_gt.count(1);
    for (int bt = 0; bt < batch_size_; bt++){

        // crop
        int row_start;
        int col_start;
        int slice_start;
        if(surface_rows_.size() > 0){
            int surface_idx = Rand(surface_rows_.size());
            int surface_row_offset = Rand(out_height_);
            int surface_col_offset = Rand(out_width_);
            int surface_slice_offset = Rand(out_depth_);
            row_start = std::max(surface_rows_[surface_idx]-surface_row_offset, 0);
            col_start = std::max(surface_cols_[surface_idx]-surface_col_offset, 0);
            slice_start = std::max(surface_slices_[surface_idx]-surface_slice_offset, 0);
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
        vector<int> dc_shape(4);
        dc_shape[0] = num_classes_;
        dc_shape[1] = out_height_;
        dc_shape[2] = out_width_;
        dc_shape[3] = out_depth_;
        crop_data.Reshape(dc_shape);
        vector<int> gt_shape(3);
        gt_shape[0] = out_height_;
        gt_shape[1] = out_width_;
        gt_shape[2] = out_depth_;
        crop_gt.Reshape(gt_shape);

        Dtype* crop_data_data = crop_data.mutable_cpu_data();
        Dtype* crop_gt_data = crop_gt.mutable_cpu_data();
        memset(crop_data_data, 0, sizeof(Dtype)*crop_data.count());
        caffe_set<Dtype>(crop_gt.count(), ignore_label_, crop_gt_data);
        for(int c = 0; c < num_classes_; c++){
            for(int i = row_start; i < row_end; i++){
                for(int j = col_start; j < col_end; j++){
                    for(int k = slice_start; k < slice_end; k++){
                        crop_data_data[(( c* out_height_ + i-row_start) * out_width_ + j-col_start)* out_width_ + k-slice_start]
                            = data_data[(( c* nrows + i) * ncols + j)* nslices + k];
                        if(c == 0){
                            crop_gt_data[(( c* out_height_ + i-row_start) * out_width_ + j-col_start)* out_width_ + k-slice_start]
                             = gt_data[(( c * nrows + i) * ncols + j)* nslices + k];
                        }
                        
                    }
                }
            }
        }


        caffe_copy<Dtype>(data_inner_dim, crop_data.cpu_data(), top_data_data);
        top_data_data += data_inner_dim;

        caffe_copy<Dtype>(gt_inner_dim, crop_gt.cpu_data(), top_gt_data);
        top_gt_data += gt_inner_dim;
    }
    delete gt;
    delete data;
}

template <typename Dtype>
void InputDatacostGtLayer<Dtype>::init_rand() {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
}

template <typename Dtype>
int InputDatacostGtLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(InputDatacostGtLayer);
REGISTER_LAYER_CLASS(InputDatacostGt);

}  // namespace caffe
