#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/input_primal_dual_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class InputPrimalDualLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  InputPrimalDualLayerTest()
      : blob_top_u_(new Blob<Dtype>()),
        blob_top_u__(new Blob<Dtype>()),
        blob_top_m_(new Blob<Dtype>()),
        blob_top_l_(new Blob<Dtype>()) {}

  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_u_);
    blob_top_vec_.push_back(blob_top_u__);
    blob_top_vec_.push_back(blob_top_m_);
    blob_top_vec_.push_back(blob_top_l_);

  }

  
  virtual ~InputPrimalDualLayerTest() { 
    delete blob_top_u_; 
    delete blob_top_u__;
    delete blob_top_m_;
    delete blob_top_l_;
   }


  Blob<Dtype>* const blob_top_u_;
  Blob<Dtype>* const blob_top_u__;
  Blob<Dtype>* const blob_top_m_;
  Blob<Dtype>* const blob_top_l_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(InputPrimalDualLayerTest, TestDtypesAndDevices);



TYPED_TEST(InputPrimalDualLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_input_primal_dual_param()->set_batch_size(1);
  layer_param.mutable_input_primal_dual_param()->set_height(48);
  layer_param.mutable_input_primal_dual_param()->set_width(48);
  layer_param.mutable_input_primal_dual_param()->set_depth(48);
  layer_param.mutable_input_primal_dual_param()->set_num_classes(5);
  


  InputPrimalDualLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_u_->shape(0), 1);
  EXPECT_EQ(this->blob_top_u_->shape(1), 5);
  EXPECT_EQ(this->blob_top_u_->shape(2), 48);
  EXPECT_EQ(this->blob_top_u_->shape(3), 48);
  EXPECT_EQ(this->blob_top_u_->shape(4), 48);

  EXPECT_EQ(this->blob_top_u__->shape(0), 1);
  EXPECT_EQ(this->blob_top_u__->shape(1), 5);
  EXPECT_EQ(this->blob_top_u__->shape(2), 48);
  EXPECT_EQ(this->blob_top_u__->shape(3), 48);
  EXPECT_EQ(this->blob_top_u__->shape(4), 48);

  EXPECT_EQ(this->blob_top_m_->shape(0), 1);
  EXPECT_EQ(this->blob_top_m_->shape(1), 15);
  EXPECT_EQ(this->blob_top_m_->shape(2), 48);
  EXPECT_EQ(this->blob_top_m_->shape(3), 48);
  EXPECT_EQ(this->blob_top_m_->shape(4), 48);

  EXPECT_EQ(this->blob_top_l_->shape(0), 1);
  EXPECT_EQ(this->blob_top_l_->shape(1), 48);
  EXPECT_EQ(this->blob_top_l_->shape(2), 48);
  EXPECT_EQ(this->blob_top_l_->shape(3), 48);

  cout << this->blob_top_u_->cpu_data()[2] << endl;




}



}  // namespace caffe
