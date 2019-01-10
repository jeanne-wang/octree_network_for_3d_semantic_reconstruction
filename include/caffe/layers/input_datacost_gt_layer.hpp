#ifndef CAFFE_INPUT_DATACOST_GT_LAYER_HPP_
#define CAFFE_INPUT_DATACOST_GT_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"

namespace caffe {


template <typename Dtype>
class InputDatacostGtLayer : public Layer<Dtype> {
 public:
  explicit InputDatacostGtLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "InputDatacostGt"; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
   void load_data_from_disk();
   void read_gvr_datacost(const string& data_file_name, int num_classes, Blob<Dtype>& data);
   void read_gvr_groundtruth(const string& gt_file_name, Blob<Dtype>& gt);
   void generate_batch(const vector<int>& batch_indexes, Blob<Dtype>& top_datacost, Blob<Dtype>& top_gt);
   void generate_batch_one_scene(int scene_idx, Blob<Dtype>& top_datacost, Blob<Dtype>& top_gt);
   void init_rand();
   int Rand(int n);
   bool is_byteorder_big_endian(){
      int num = 1;
      if(*(char *)&num == 1){
        return false;
      }else{
        return true;
      }
   }
   void reorder(vector<string> & file_names, vector<int> order );
   



   vector<string> data_file_names_;
   vector<string> gt_file_names_;

   string data_list_file_;
   string groundtruth_list_file_;
   int out_height_;
   int out_width_;
   int out_depth_;
   int num_classes_;
   int batch_size_;
   int  model_counter_;
   shared_ptr<Caffe::RNG> prefetch_rng_;
   shared_ptr<Caffe::RNG> rng_;

   // for crop include surface
   vector<int> surface_rows_;
   vector<int> surface_cols_;
   vector<int> surface_slices_;

   int freespace_label_;
   int ceiling_label_;
   int wall_label_;
   int floor_label_;
   int ignore_label_;
   
};

}  // namespace caffe

#endif  // CAFFE_INPUT_DATACOST_GT_LAYER_HPP_
