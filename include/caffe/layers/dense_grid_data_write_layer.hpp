#ifndef DENSE_GRID_DATA_WRITE_LAYER_HPP_
#define DENSE_GRID_DATA_WRITE_LAYER_HPP_

#include <vector>
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
// writing a channel * height * width * depth blob into a file
// used in testing phase
template <typename Dtype>
class DenseGridDataWriteLayer : public Layer<Dtype> {
 public:
  explicit DenseGridDataWriteLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DenseGridDataWrite"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
   bool is_byteorder_big_endian(){
      int num = 1;
      if(*(char *)&num == 1){
        return false;
      }else{
        return true;
      }
   }
   
   vector<string> _write_file_names;
   string data_write_file_name_;
   string input_source_;
   int _model_counter;
   bool _done_initial_reshape;

};

}  // namespace caffe

#endif  // DenseGridDataWriteLayer_LAYER_HPP_
