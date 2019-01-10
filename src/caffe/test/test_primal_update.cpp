#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/primal_update_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {


// this test is designed for testing the conv result in dual update layer 
template <typename TypeParam>
class PrimalUpdateLayerConvTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PrimalUpdateLayerConvTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_bottom_3_(new Blob<Dtype>()),
        blob_bottom_4_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {

    vector<int> primal_shape(5);
    primal_shape[0] = 2;
    primal_shape[1] = 2;
    primal_shape[2] = 3;
    primal_shape[3] = 2;
    primal_shape[4] = 2;

    vector<int> dual_shape(primal_shape.begin(), primal_shape.end());
    dual_shape[1] = 6;

    blob_bottom_->Reshape(primal_shape);
    blob_bottom_2_->Reshape(dual_shape);
    blob_bottom_4_->Reshape(primal_shape);

    vector<int> lag_shape(4);
    lag_shape[0] = 2;
    lag_shape[1] = 3;
    lag_shape[2] = 2;
    lag_shape[3] = 2;
    blob_bottom_3_->Reshape(lag_shape);



    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    filler.Fill(this->blob_bottom_3_);
    filler.Fill(this->blob_bottom_4_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_4_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~PrimalUpdateLayerConvTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_bottom_4_;
    delete blob_top_;
  }

  

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_bottom_4_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PrimalUpdateLayerConvTest, TestDtypesAndDevices);

TYPED_TEST(PrimalUpdateLayerConvTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PrimalUpdateParameter* primal_update_param =
      layer_param.mutable_primal_update_param();

  primal_update_param->add_kernel_size(2);
  primal_update_param->add_start_pad(0);
  primal_update_param->add_end_pad(1);
  primal_update_param->add_stride(1);
  primal_update_param->set_num_output(2);
  primal_update_param->mutable_weight_filler()->set_type("gaussian");
  primal_update_param->set_tau(0.01);


  shared_ptr<Layer<Dtype> > layer(
      new PrimalUpdateLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
  EXPECT_EQ(this->blob_top_->shape(4), 2);

}


/*TYPED_TEST(PrimalUpdateLayerConvTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PrimalUpdateParameter* primal_update_param =
      layer_param.mutable_primal_update_param();

  primal_update_param->add_kernel_size(2);
  primal_update_param->add_start_pad(0);
  primal_update_param->add_end_pad(1);
  primal_update_param->add_stride(1);
  primal_update_param->set_num_output(2);
  primal_update_param->mutable_weight_filler()->set_type("gaussian");
  primal_update_param->set_tau(0.01);

  PrimalUpdateLayer<Dtype> layer(layer_param);


  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}*/





}  // namespace caffe
