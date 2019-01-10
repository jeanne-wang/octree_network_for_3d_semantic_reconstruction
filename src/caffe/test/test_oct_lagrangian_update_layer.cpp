#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/oct_lagrangian_update_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctLagrangianUpdateLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctLagrangianUpdateLayerTest()
    : blob_bottom_(new Blob<Dtype>()),
      blob_bottom_2_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
    
  virtual void SetUp() {
    vector<int> prim_shape(3);
    prim_shape[0] = 3;
    prim_shape[1] = 6;
    prim_shape[2] = 8;

    vector<int> lag_shape(3);
    lag_shape[0] = 3;
    lag_shape[1] = 1;
    lag_shape[2] = 8;
   

    blob_bottom_2_->Reshape(prim_shape);
    blob_bottom_->Reshape(lag_shape);
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

  virtual ~OctLagrangianUpdateLayerTest() {
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

TYPED_TEST_CASE(OctLagrangianUpdateLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctLagrangianUpdateLayerTest, TestValue) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  float sigma = 0.1;
  layer_param.mutable_lagrangian_update_param()->set_sigma(sigma);
  OctLagrangianUpdateLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* lag_data = this->blob_bottom_->cpu_data();
  const Dtype* prim_data = this->blob_bottom_2_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype kDelta = 2e-4;

  int channel = this->blob_bottom_2_->shape(1);
  int num = this->blob_top_->shape(0);
  int num_rows = this->blob_top_->shape(2);
  for(int n = 0; n < num; n++){
      for(int i = 0; i < num_rows; i++){     
            int index = n * num_rows + i; 
            Dtype sum = 0;
            for(int c = 0; c < channel; c++){
              int prim_index = (n * channel + c) * num_rows + i;
              sum += prim_data[prim_index];

            }

            sum -= 1;
            sum  = sum * sigma + lag_data[index];

            EXPECT_NEAR(sum, top_data[index], kDelta);

          
        
      }
    }
}

TYPED_TEST(OctLagrangianUpdateLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  float sigma = 0.1;
  layer_param.mutable_lagrangian_update_param()->set_sigma(sigma);
  OctLagrangianUpdateLayer<Dtype> layer(layer_param);

  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}




}  // namespace caffe
