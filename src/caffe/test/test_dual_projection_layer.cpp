#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dual_projection_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DualProjectionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DualProjectionLayerTest()
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

  virtual ~DualProjectionLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DualProjectionLayerTest, TestDtypesAndDevices);
// all the below test was done by commenting out the code in dual projection layer for doing maximum 

TYPED_TEST(DualProjectionLayerTest, TestDualProjectionWithoutMaximum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DualProjectionLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const Dtype* bottom_data = this->blob_bottom_->cpu_data();
  const Dtype* top_data = this->blob_top_->cpu_data();
  const Dtype kDelta = 2e-4;
  int num = this->blob_top_->shape(0);
  int num_classes = this->blob_top_->shape(1)/3;
  int num_rows = this->blob_top_->shape(2);
  int num_cols = this->blob_top_->shape(3);
  int num_slices = this->blob_top_->shape(4);
  int dual_dim = this->blob_top_->count(1);
  int dual_spatial_count = this->blob_top_->count(2);
  for(int n = 0; n < num; n++){
    for(int c = 0;  c < num_classes; c++){
      for(int i = 0; i < num_rows; i++){
        for(int j = 0;  j < num_cols; j++){
          for(int k = 0; k < num_slices; k++){      
            int index = n*dual_dim+ c* 3* dual_spatial_count +  i*num_cols * num_slices + j *num_slices + k;
            Dtype norm =  0;
            for(int t = 0; t < 3; t++){
              norm += top_data[index] * top_data[index];
              index += dual_spatial_count;

            }

            EXPECT_NEAR(norm, 1., kDelta);

          }
        }
      }
    }
  }
}

TYPED_TEST(DualProjectionLayerTest, TestDualProjectionGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DualProjectionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}




}  // namespace caffe
