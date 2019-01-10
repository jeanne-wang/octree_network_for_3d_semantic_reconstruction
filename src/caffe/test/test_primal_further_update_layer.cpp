#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/primal_further_update_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PrimalFurtherUpdateLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PrimalFurtherUpdateLayerTest()
    : blob_bottom_(new Blob<Dtype>()),
      blob_bottom_2_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
    
  virtual void SetUp() {
    vector<int> prim_shape(5);
    prim_shape[0] = 3;
    prim_shape[1] = 6;
    prim_shape[2] = 2;
    prim_shape[3] = 2;
    prim_shape[4] = 2;

    
    blob_bottom_->Reshape(prim_shape);
    blob_bottom_2_->Reshape(prim_shape);
    
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
    filler.Fill(blob_bottom_2_);
  }

  virtual ~PrimalFurtherUpdateLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PrimalFurtherUpdateLayerTest, TestDtypesAndDevices);

TYPED_TEST(PrimalFurtherUpdateLayerTest, TestValue) {
  typedef typename TypeParam::Dtype Dtype;


  LayerParameter layer_param;
  PrimalFurtherUpdateLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* u_data = this->blob_bottom_->cpu_data();
  const Dtype* u_prev_data = this->blob_bottom_2_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype kDelta = 2e-4;

  
  int num = this->blob_top_->shape(0);
  int channel = this->blob_top_->shape(1);
  int num_rows = this->blob_top_->shape(2);
  int num_cols = this->blob_top_->shape(3);
  int num_slices = this->blob_top_->shape(4);
  for(int n = 0; n < num; n++){
    for(int c = 0; c < channel; c++){
      for(int i = 0; i < num_rows; i++){
        for(int j = 0;  j < num_cols; j++){
          for(int k = 0; k < num_slices; k++){      
            
            int ind = (((n * channel + c) * num_rows + i) * num_cols + j) * num_slices + k;
            Dtype toCompare = 2*u_data[ind]-u_prev_data[ind];
            EXPECT_NEAR(toCompare, top_data[ind], kDelta);

          }
        }
      }
    }
  }
}

TYPED_TEST(PrimalFurtherUpdateLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PrimalFurtherUpdateLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}




}  // namespace caffe
