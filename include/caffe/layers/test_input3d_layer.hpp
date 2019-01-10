#ifndef TEST_INPUT3D_LAYER_HPP_
#define TEST_INPUT3D_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
class TestInput3DLayer : public Layer<Dtype> {
 public:
  explicit TestInput3DLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TestInput3D"; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
   void read_gvr_datacost(const string& data_file_name, int num_classes, Blob<Dtype>& data);
   bool is_byteorder_big_endian(){
      int num = 1;
      if(*(char *)&num == 1){
        return false;
      }else{
        return true;
      }
   }
   

   Blob<Dtype> data_;
   string data_file_;
   int num_levels_;
   int num_classes_;
   bool row_major_;

};

}  // namespace caffe

#endif  // INPUT3D_LAYER_HPP_
