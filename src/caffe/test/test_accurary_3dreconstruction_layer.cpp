#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/accuracy_3dreconstruction_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class Accuracy3DReconstructionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Accuracy3DReconstructionLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_free_accu_(new Blob<Dtype>()),
        blob_top_occupied_accu_(new Blob<Dtype>()),
        blob_top_semantic_accu_(new Blob<Dtype>()) {}
  virtual void SetUp(){
    vector<int> shape(5);
    shape[0] = 2;
    shape[1] = 5;
    shape[2] = 4;
    shape[3] = 4;
    shape[4] = 4;
    blob_bottom_data_->Reshape(shape);
    blob_bottom_label_->Reshape(shape);


    // fill the gt label
    Dtype* gt_data= blob_bottom_label_->mutable_cpu_data();
    caffe_set<Dtype>(blob_bottom_label_->count(), Dtype(0.), gt_data);
    for(int n = 0; n < 2; n++){
      for(int i = 0; i < 4; i++){
        for(int j = 0; j < 4; j++){
          for(int k = 0; k < 4; k++){
            int label = caffe_rng_rand() % 5;
            gt_data[(((n * 5 + label) * 4 + i) * 4 + j) * 4 + k] = 1;
          }
        }
      }
    }
  
    // fill the values
    Blob<Dtype>* data = new Blob<Dtype>(shape);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(data);

    //for(int i = 0; i < blob_bottom_label_->count(); i++) data->mutable_cpu_data()[i] = gt_data[i];

    vector<Blob<Dtype>*> softmax_bottom_vec;
    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_bottom_vec.push_back(data);
    softmax_top_vec.push_back(blob_bottom_data_);
    LayerParameter softmax_param;
    softmax_param.mutable_softmax_param()->set_axis(1);
    SoftmaxLayer<Dtype> softmax_layer(softmax_param);
    softmax_layer.SetUp(softmax_bottom_vec, softmax_top_vec);
    softmax_layer.Forward(softmax_bottom_vec, softmax_top_vec);


    
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_free_accu_);
    blob_top_vec_.push_back(blob_top_occupied_accu_);
    blob_top_vec_.push_back(blob_top_semantic_accu_);

  }
  virtual ~Accuracy3DReconstructionLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_free_accu_;
    delete blob_top_occupied_accu_;
    delete blob_top_semantic_accu_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_free_accu_;
  Blob<Dtype>* const blob_top_occupied_accu_;
  Blob<Dtype>* const blob_top_semantic_accu_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(Accuracy3DReconstructionLayerTest, TestDtypesAndDevices);

TYPED_TEST(Accuracy3DReconstructionLayerTest, TestAccuracy) {
  typedef typename TypeParam::Dtype Dtype;


  LayerParameter layer_param;
  layer_param.mutable_accuracy_3dreconstruction_param()->set_freespace_label(0);
  layer_param.mutable_accuracy_3dreconstruction_param()->set_unknown_label(4);
  Accuracy3DReconstructionLayer<Dtype> layer(layer_param);



  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  cout << "freespace accuracy is: "<< this->blob_top_vec_[0]->cpu_data()[0] <<endl;
  cout << "occupied accuracy is: "<< this->blob_top_vec_[1]->cpu_data()[0] <<endl;
  cout << "semantic accuracy is: "<< this->blob_top_vec_[2]->cpu_data()[0] <<endl;
  
}



}  // namespace caffe
