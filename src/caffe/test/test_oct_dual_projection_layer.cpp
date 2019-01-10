#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/oct_dual_projection_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctDualProjectionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctDualProjectionLayerTest()
    : blob_bottom_(new Blob<Dtype>()),
      blob_top_(new Blob<Dtype>()) {}
    
  virtual void SetUp() {
    vector<int> in_shape(3);
    in_shape[0] = 3;
    in_shape[1] = 6;
    in_shape[2] = 8;


    blob_bottom_->Reshape(in_shape);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
  }

  virtual ~OctDualProjectionLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OctDualProjectionLayerTest, TestDtypesAndDevices);
// all the below test was done by commenting out the code in dual projection layer for doing maximum 

TYPED_TEST(OctDualProjectionLayerTest, TestDualProjectionWithoutMaximum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  OctDualProjectionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype kDelta = 2e-4;
  int num = this->blob_top_->shape(0);
  int num_classes = this->blob_top_->shape(1)/3;
  int num_rows = this->blob_top_->shape(2);
  int dual_dim = this->blob_top_->count(1);
  for(int n = 0; n < num; n++){
    for(int c = 0;  c < num_classes; c++){
      for(int i = 0; i < num_rows; i++){
              
        int index = n*dual_dim+ c* 3* num_rows +  i;
        Dtype norm =  0;
        for(int t = 0; t < 3; t++){
          norm += top_data[index] * top_data[index];
          index += num_rows;

        }

        EXPECT_NEAR(norm, 1., kDelta);
      }
    }
  }
}

TYPED_TEST(OctDualProjectionLayerTest, TestDualProjectionGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  OctDualProjectionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}




}  // namespace caffe
