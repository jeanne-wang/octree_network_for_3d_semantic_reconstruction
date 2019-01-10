#ifndef INPUT3D_LAYER_HPP_
#define INPUT3D_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"

namespace caffe {


template <typename Dtype>
class Input3DLayer : public Layer<Dtype> {
 public:
  explicit Input3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Input3D"; }


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
   void generate_batch_one_scene(int scene_idx, Blob<Dtype>& top_data, Blob<Dtype>& top_gt);
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
   

   vector<Blob<Dtype>*> data_;
   vector<Blob<Dtype>*> gt_;


   vector<string> data_file_names_;
   vector<string> gt_file_names_;

   string data_list_file_;
   string groundtruth_list_file_;
   int out_height_;
   int out_width_;
   int out_depth_;
   int num_classes_;
   int batch_size_;
   bool preload_data_;
   bool row_major_;
   bool customized_crop_;
   int  model_counter_;
   shared_ptr<Caffe::RNG> prefetch_rng_;
   shared_ptr<Caffe::RNG> rng_;

};

}  // namespace caffe

#endif  // INPUT3D_LAYER_HPP_
