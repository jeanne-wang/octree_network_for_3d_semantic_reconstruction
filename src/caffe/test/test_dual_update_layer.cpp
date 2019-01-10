#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/dual_update_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, 
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {

  memset(out->mutable_cpu_data(), Dtype(0.), sizeof(Dtype)* out->count());

  int kernel_h = 2;
  int kernel_w = 2;
  int kernel_d = 2;

  int start_pad_h = 0;
  int start_pad_w = 0;
  int start_pad_d = 0;

  int end_pad_h = 1;
  int end_pad_w = 1;
  int end_pad_d = 1;

  int stride_h = 1;
  int stride_w = 1;
  int stride_d = 1; 

  int dilation_h = 1;
  int dilation_w = 1;
  int dilation_d = 1;
  // Groups
  int groups = 1;
  int o_g = out->shape(1) / groups;
  int k_g = in->shape(1) / groups;
  int o_head, k_head;
  // Convolution
  vector<int> weight_offset(5);
  vector<int> in_offset(5);
  vector<int> out_offset(5);
  Dtype* out_data = out->mutable_cpu_data();

  for (int n = 0; n < out->shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < out->shape(2); z++) {
            for (int y = 0; y < out->shape(3); y++) {
              for (int x = 0; x < out->shape(4); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - start_pad_d + r * dilation_d;
                      int in_y = y * stride_h - start_pad_h + p * dilation_h;
                      int in_x = x * stride_w - start_pad_w + q * dilation_w;
                      if (in_z >= 0 && in_z < (in->shape(2))
                          && in_y >= 0 && in_y < in->shape(3)
                          && in_x >= 0 && in_x < in->shape(4)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        weight_offset[2] = r; 
                        weight_offset[3] = p;
                        weight_offset[4] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        in_offset[2] = in_z; 
                        in_offset[3] = in_y;
                        in_offset[4] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        out_offset[2] = z; 
                        out_offset[3] = y;
                        out_offset[4] = x;
                        out_data[out->offset(out_offset)] +=
                            in->data_at(in_offset)
                            * weights[0]->data_at(weight_offset);
                        
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

// this test is designed for testing the conv result in dual update layer 
template <typename TypeParam>
class DualUpdateLayerConvTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DualUpdateLayerConvTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {

    vector<int> dual_shape(5);
    dual_shape[0] = 2;
    dual_shape[1] = 6;
    dual_shape[2] = 3;
    dual_shape[3] = 2;
    dual_shape[4] = 2;

    vector<int> primal_shape(dual_shape.begin(), dual_shape.end());
    primal_shape[1] = 2;

    blob_bottom_->Reshape(dual_shape);
    blob_bottom_2_->Reshape(primal_shape);


    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);

    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~DualUpdateLayerConvTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DualUpdateLayerConvTest, TestDtypesAndDevices);

TYPED_TEST(DualUpdateLayerConvTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DualUpdateParameter* dual_update_param =
      layer_param.mutable_dual_update_param();

  dual_update_param->add_kernel_size(2);
  dual_update_param->add_start_pad(0);
  dual_update_param->add_end_pad(1);
  dual_update_param->add_stride(1);
  dual_update_param->set_num_output(6);
  dual_update_param->mutable_weight_filler()->set_type("gaussian");
  dual_update_param->set_sigma(0.01);


  shared_ptr<Layer<Dtype> > layer(
      new DualUpdateLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 6);
  EXPECT_EQ(this->blob_top_->shape(2), 3);
  EXPECT_EQ(this->blob_top_->shape(3), 2);
  EXPECT_EQ(this->blob_top_->shape(4), 2);

}



TYPED_TEST(DualUpdateLayerConvTest, TestConvolution) {
  typedef typename TypeParam::Dtype Dtype;
  
  
  LayerParameter layer_param;
  DualUpdateParameter* dual_update_param =
      layer_param.mutable_dual_update_param();

  dual_update_param->add_kernel_size(2);
  dual_update_param->add_start_pad(0);
  dual_update_param->add_end_pad(1);
  dual_update_param->add_stride(1);
  dual_update_param->set_num_output(6);
  dual_update_param->mutable_weight_filler()->set_type("gaussian");
  dual_update_param->set_sigma(0.01);

  shared_ptr<Layer<Dtype> > layer(
      new DualUpdateLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_2_, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }






}


TYPED_TEST(DualUpdateLayerConvTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  DualUpdateParameter* dual_update_param =
      layer_param.mutable_dual_update_param();

  dual_update_param->add_kernel_size(2);
  dual_update_param->add_start_pad(0);
  dual_update_param->add_end_pad(1);
  dual_update_param->add_stride(1);
  dual_update_param->set_num_output(6);
  dual_update_param->mutable_weight_filler()->set_type("gaussian");
  dual_update_param->set_sigma(0.01);
  DualUpdateLayer<Dtype> layer(layer_param);


  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}





}  // namespace caffe
