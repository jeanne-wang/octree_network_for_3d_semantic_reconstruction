#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/pooling3d_layer.hpp"


#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class Pooling3DLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Pooling3DLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    vector<int> in_shape(5);
    in_shape[0] = 2;
    in_shape[1] = 3;
    in_shape[2] = 6;
    in_shape[3] = 5;
    in_shape[4] = 4;
    blob_bottom_->Reshape(in_shape);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~Pooling3DLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};
  

TYPED_TEST_CASE(Pooling3DLayerTest, TestDtypesAndDevices);

TYPED_TEST(Pooling3DLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
  pooling3d_param->set_kernel_size(3);
  pooling3d_param->set_stride(2);
  Pooling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
  EXPECT_EQ(this->blob_top_->shape(4), 2);
}

TYPED_TEST(Pooling3DLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
  pooling3d_param->set_kernel_size(3);
  pooling3d_param->set_stride(2);
  pooling3d_param->set_pad(1);
  pooling3d_param->set_pool(Pooling3DParameter_PoolMethod_AVE);
  Pooling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), this->blob_bottom_->shape(0));
  EXPECT_EQ(this->blob_top_->shape(1), this->blob_bottom_->shape(1));
  EXPECT_EQ(this->blob_top_->shape(2), 4);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 3);
}


TYPED_TEST(Pooling3DLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      for(int kernel_s = 3; kernel_s <= 4; kernel_s++){
        LayerParameter layer_param;
        Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
        pooling3d_param->set_kernel_h(kernel_h);
        pooling3d_param->set_kernel_w(kernel_w);
        pooling3d_param->set_kernel_s(kernel_s);
        pooling3d_param->set_stride(2);
        pooling3d_param->set_pad(1);
        pooling3d_param->set_pool(Pooling3DParameter_PoolMethod_MAX);
        Pooling3DLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-4, 1e-2);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);

      }
    }
  }
}

TYPED_TEST(Pooling3DLayerTest, TestForwardMaxPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
  pooling3d_param->set_kernel_size(3);
  pooling3d_param->set_stride(2);
  pooling3d_param->set_pad_h(2);
  pooling3d_param->set_pad_w(2);
  pooling3d_param->set_pad_s(1);
  pooling3d_param->set_pool(Pooling3DParameter_PoolMethod_MAX);
  vector<int> in_shape(5);
  in_shape[0] = 1;
  in_shape[1] = 1;
  in_shape[2] = 3;
  in_shape[3] = 3;
  in_shape[4] = 1;
  this->blob_bottom_->Reshape(in_shape);
  // Input:
  //     [ 1 2 4 ]
  //     [ 2 3 2 ]
  //     [ 4 2 1 ]
  this->blob_bottom_->mutable_cpu_data()[0] = 1;
  this->blob_bottom_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_->mutable_cpu_data()[2] = 4;
  this->blob_bottom_->mutable_cpu_data()[3] = 2;
  this->blob_bottom_->mutable_cpu_data()[4] = 3;
  this->blob_bottom_->mutable_cpu_data()[5] = 2;
  this->blob_bottom_->mutable_cpu_data()[6] = 4;
  this->blob_bottom_->mutable_cpu_data()[7] = 2;
  this->blob_bottom_->mutable_cpu_data()[8] = 1;
  Pooling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 1);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-8;
  // Output:
  //     [ 1 4 4 ]
  //     [ 4 4 4 ]
  //     [ 4 4 1 ]
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 1, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4, epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 1, epsilon);
}

TYPED_TEST(Pooling3DLayerTest, TestGradientMaxTopMask) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      for(int kernel_s = 3; kernel_s <= 4; kernel_s++){
        LayerParameter layer_param;
        Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
        pooling3d_param->set_kernel_h(kernel_h);
        pooling3d_param->set_kernel_w(kernel_w);
        pooling3d_param->set_kernel_s(kernel_s);
        pooling3d_param->set_stride(2);
        pooling3d_param->set_pool(Pooling3DParameter_PoolMethod_MAX);
        this->blob_top_vec_.push_back(this->blob_top_mask_);
        Pooling3DLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-4, 1e-2);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
        this->blob_top_vec_.pop_back();
      }
      
    }
  }
}

TYPED_TEST(Pooling3DLayerTest, TestForwardAve) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
  pooling3d_param->set_kernel_size(3);
  pooling3d_param->set_stride(1);
  pooling3d_param->set_pad(1);
  pooling3d_param->set_pool(Pooling3DParameter_PoolMethod_AVE);
  vector<int> in_shape(5);
  in_shape[0] = 1;
  in_shape[1] = 1;
  in_shape[2] = 3;
  in_shape[3] = 3;
  in_shape[4] = 1;
  this->blob_bottom_->Reshape(in_shape);
  FillerParameter filler_param;
  filler_param.set_value(Dtype(2));
  ConstantFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);
  Pooling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 1);
  EXPECT_EQ(this->blob_top_->shape(1), 1);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 3);
  EXPECT_EQ(this->blob_top_->shape(4), 1);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype epsilon = 1e-5;
  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / (9*3), epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / (3*3), epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / (9*3), epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / (3*3), epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0 /3   , epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / (3*3), epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / (9*3), epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / (3*3), epsilon);
  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / (9*3), epsilon);
}

TYPED_TEST(Pooling3DLayerTest, TestGradientAve) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      for(int kernel_s = 3;  kernel_s <= 4; kernel_s++){
        LayerParameter layer_param;
        Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
        pooling3d_param->set_kernel_h(kernel_h);
        pooling3d_param->set_kernel_w(kernel_w);
        pooling3d_param->set_kernel_s(kernel_s);
        pooling3d_param->set_stride(2);
        pooling3d_param->set_pool(Pooling3DParameter_PoolMethod_AVE);
        Pooling3DLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-2, 1e-2);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
      }
      
    }
  }
}

TYPED_TEST(Pooling3DLayerTest, TestGradientAvePadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
      for(int kernel_s = 3;  kernel_s <= 4; kernel_s++){
        LayerParameter layer_param;
        Pooling3DParameter* pooling3d_param = layer_param.mutable_pooling3d_param();
        pooling3d_param->set_kernel_h(kernel_h);
        pooling3d_param->set_kernel_w(kernel_w);
        pooling3d_param->set_kernel_s(kernel_s);
        pooling3d_param->set_stride(2);
        pooling3d_param->set_pad(2);
        pooling3d_param->set_pool(Pooling3DParameter_PoolMethod_AVE);
        Pooling3DLayer<Dtype> layer(layer_param);
        GradientChecker<Dtype> checker(1e-2, 1e-2);
        checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
          this->blob_top_vec_);
      }
      
    }
  }
}



}  // namespace caffe
