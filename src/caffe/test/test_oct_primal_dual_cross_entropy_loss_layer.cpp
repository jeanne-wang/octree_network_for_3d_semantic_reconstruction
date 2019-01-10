#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/oct_primal_dual_cross_entropy_loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class OctPrimalDualLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctPrimalDualLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_loss_(new Blob<Dtype>()) {}
  virtual void SetUp(){
    vector<int> shape(3);
    shape[0] = 2;
    shape[1] = 3;
    shape[2] = 64;

    blob_bottom_data_->Reshape(shape);
    blob_bottom_label_->Reshape(shape);

    // fill the values
    Blob<Dtype>* data = new Blob<Dtype>(shape);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(data);

    vector<Blob<Dtype>*> softmax_bottom_vec;
    vector<Blob<Dtype>*> softmax_top_vec;
    softmax_bottom_vec.push_back(data);
    softmax_top_vec.push_back(blob_bottom_data_);
    LayerParameter softmax_param;
    softmax_param.mutable_softmax_param()->set_axis(1);
    SoftmaxLayer<Dtype> softmax_layer(softmax_param);
    softmax_layer.SetUp(softmax_bottom_vec, softmax_top_vec);
    softmax_layer.Forward(softmax_bottom_vec, softmax_top_vec);


    // fill the gt label
    Dtype* gt_data= blob_bottom_label_->mutable_cpu_data();
    caffe_set<Dtype>(blob_bottom_label_->count(), Dtype(0.), gt_data);
    for(int n = 0; n < 2; n++){
      for(int i = 0; i < 64; i++){
        int label = caffe_rng_rand() % 3;
        gt_data[(n * 3 + label) * 64 + i] = 1;
      }
    }

    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_loss_);

  }
  virtual ~OctPrimalDualLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OctPrimalDualLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctPrimalDualLossLayerTest, TestLoss) {
  typedef typename TypeParam::Dtype Dtype;


  LayerParameter layer_param;
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_softmax_scale(1);
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_clip_epsilon(1e-8);
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_freespace_weighted(false);
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_unknown_weighted(false);
  OctPrimalDualCrossEntropyLossLayer<Dtype> layer(layer_param);



  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  cout << this->blob_top_vec_[0]->cpu_data()[0]<<endl;
  
}

/*TYPED_TEST(PrimalDualLossLayerTest, TestGradientFreeSpaceWeighted) {
  typedef typename TypeParam::Dtype Dtype;


  LayerParameter layer_param;
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_softmax_scale(1);
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_clip_epsilon(1e-8);
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_freespace_weighted(true);
  layer_param.mutable_primal_dual_cross_entropy_loss_param()->set_unknown_weighted(false);



  PrimalDualCrossEntropyLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}*/

/*TYPED_TEST(PrimalDualLossLayerTest, TestGradientFreeSpaceWeighted) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}*/



}  // namespace caffe
