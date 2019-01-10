#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/scaling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ScalingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ScalingLayerTest()
    : blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
    
  virtual void SetUp() {
    vector<int> in_shape(5);
    in_shape[0] = 3;
    in_shape[1] = 6;
    in_shape[2] = 2;
    in_shape[3] = 2;
    in_shape[4] = 2;

    blob_bottom_->Reshape(in_shape);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
  }

  virtual ~ScalingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ScalingLayerTest, TestDtypesAndDevices);
// all the below test was done by commenting out the code in dual projection layer for doing maximum 

TYPED_TEST(ScalingLayerTest, TestValue) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_scaling_param()->set_scale(0.01);
  ScalingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();

  const Dtype kDelta = 2e-4;
  for(int i = 0;  i < this->blob_bottom_->count(); i++){
    //EXPECT_EQ(this->blob_bottom_->cpu_data()[i] * 0.01, this->blob_top_->cpu_data()[i]);
    EXPECT_NEAR(this->blob_bottom_->cpu_data()[i] * 0.01, this->blob_top_->cpu_data()[i], kDelta);
  }



  
}

TYPED_TEST(ScalingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_scaling_param()->set_scale(0.01);
  ScalingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}




}  // namespace caffe
