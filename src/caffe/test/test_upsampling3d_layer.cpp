#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/upsampling3d_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UpSampling3DLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UpSampling3DLayerTest()
    : blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
    
  virtual void SetUp() {
    vector<int> in_shape(5);
    in_shape[0] = 2;
    in_shape[1] = 3;
    in_shape[2] = 4;
    in_shape[3] = 5;
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

  virtual ~UpSampling3DLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(UpSampling3DLayerTest, TestDtypesAndDevices);

TYPED_TEST(UpSampling3DLayerTest, TestTrivialSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 1;
  layer_param.mutable_upsampling3d_param()->set_h_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_w_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_s_rep(kNumTiles);

  UpSampling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), this->blob_bottom_->num_axes());
  for (int j = 0; j < this->blob_bottom_->num_axes(); ++j) {
      EXPECT_EQ(this->blob_top_->shape(j), this->blob_bottom_->shape(j));
  }
}

TYPED_TEST(UpSampling3DLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 3;
  layer_param.mutable_upsampling3d_param()->set_h_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_w_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_s_rep(kNumTiles);

  UpSampling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->num_axes(), this->blob_bottom_->num_axes());

  int first_spatial_axis = 2; 
  for (int j = first_spatial_axis; j < this->blob_bottom_->num_axes(); ++j) {
      const int top_dim = kNumTiles  * this->blob_bottom_->shape(j);
      EXPECT_EQ(top_dim, this->blob_top_->shape(j));
  }
}


TYPED_TEST(UpSampling3DLayerTest, TestForwardOneDimesnion) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 3;
  layer_param.mutable_upsampling3d_param()->set_h_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_w_rep(1);
  layer_param.mutable_upsampling3d_param()->set_s_rep(1);


  UpSampling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->shape(0); ++n) {
    for (int c = 0; c < this->blob_top_->shape(1); ++c) {
       for (int h = 0; h < this->blob_top_->shape(2); ++h) {
         for (int w = 0; w < this->blob_top_->shape(3); ++w) {
            for(int s = 0; s < this->blob_top_->shape(4); ++s){
              const int bottom_h = h / kNumTiles;
              vector<int> ind1(5);
              ind1[0] = n;
              ind1[1] = c;
              ind1[2] = bottom_h;
              ind1[3] = w;
              ind1[4] = s;
              vector<int> ind2(ind1.begin(), ind1.end());
              ind2[2] = h;
             

              EXPECT_EQ(this->blob_bottom_->data_at(ind1),
                     this->blob_top_->data_at(ind2));
            }
         }
       }
    }
  }
}

TYPED_TEST(UpSampling3DLayerTest, TestForwardOneDimesnion4D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 3;
  layer_param.mutable_upsampling3d_param()->set_h_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_w_rep(1);
  layer_param.mutable_upsampling3d_param()->set_s_rep(1);

  this->blob_bottom_->Reshape(2,3,4,5);
  FillerParameter filler_param;
  filler_param.set_mean(0.0);
  filler_param.set_std(1.0);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);



  UpSampling3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_top_->shape(0); ++n) {
    for (int c = 0; c < this->blob_top_->shape(1); ++c) {
       for (int h = 0; h < this->blob_top_->shape(2); ++h) {
         for (int w = 0; w < this->blob_top_->shape(3); ++w) {
            
              const int bottom_c = c / kNumTiles;

              EXPECT_EQ(this->blob_bottom_->data_at(n, bottom_c, h, w ),
                     this->blob_top_->data_at(n, c, h, w));
          
         }
       }
    }
  }
}

TYPED_TEST(UpSampling3DLayerTest, TestTrivialGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 1;
  layer_param.mutable_upsampling3d_param()->set_h_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_w_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_s_rep(kNumTiles);
  UpSampling3DLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(UpSampling3DLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 2;
  layer_param.mutable_upsampling3d_param()->set_h_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_w_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_s_rep(kNumTiles);
  UpSampling3DLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(UpSampling3DLayerTest, TestGradient4D) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const int kNumTiles = 2;
  layer_param.mutable_upsampling3d_param()->set_h_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_w_rep(kNumTiles);
  layer_param.mutable_upsampling3d_param()->set_s_rep(kNumTiles);

  this->blob_bottom_->Reshape(2,3,4,5);
  FillerParameter filler_param;
  filler_param.set_mean(0.0);
  filler_param.set_std(1.0);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_);


  UpSampling3DLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
