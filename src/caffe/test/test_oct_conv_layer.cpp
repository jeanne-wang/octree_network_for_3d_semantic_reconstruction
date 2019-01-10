#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/oct_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "image_tree_tools/octree.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class OctConvLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctConvLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_bottom_3_(new Blob<Dtype>()),
        blob_bottom_4_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {

    vector<int> shape(3);
    shape[0] = 2;
    shape[1] = 6;
    shape[2] = 64;
  
    blob_bottom_->Reshape(shape);
   
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);


    // 
    // newly added for efficient conv in GPU later
    vector<GeneralOctree<int> > _octree_keys;
    for(int bt=0; bt<2; bt++)
    {
        GeneralOctree<int> octree_keys;
        for(int x=0; x<4; x++)
        {
            for(int y=0; y<4; y++)
            {
                for(int z=0; z<4; z++)
                {
                    OctreeCoord c;
                    c.x = x; c.y = y; c.z = z; c.l = 2;
                    unsigned int key = GeneralOctree<int>::compute_key(c);
                    octree_keys.add_element(key, x*4*4+ y*4 + z);
                }
            }
        }
        _octree_keys.push_back(octree_keys);
    }


    /****** newly added for efficient conv in GPU later***********/
    // num_[i]: number of octree cells in the ith octree in the batch
    int num_output_pixels = 4*4*4;
    vector<int> num_shape(1);
    num_shape[0] = 2;
    blob_bottom_2_->Reshape(num_shape);
    caffe_set<Dtype>(2, num_output_pixels, blob_bottom_2_->mutable_cpu_data());

    vector<int> nbh_shape(3);
    nbh_shape[0] = 2;
    nbh_shape[1] = num_output_pixels;
    nbh_shape[2] = 2*2*2;
    blob_bottom_3_->Reshape(nbh_shape);
    blob_bottom_4_->Reshape(nbh_shape);

    int batch_size = blob_bottom_3_->shape(0);
    int num_pixels = blob_bottom_3_->shape(1);

    Dtype* neighbors_data = blob_bottom_3_->mutable_cpu_data();
    Dtype* neighbor_of_data = blob_bottom_4_->mutable_cpu_data();
    caffe_set<Dtype>(blob_bottom_4_->count(), -1, neighbor_of_data);

        for(int bt = 0; bt < batch_size; bt++){

            for(typename GeneralOctree<int>::iterator it=_octree_keys[bt].begin(); it!=_octree_keys[bt].end(); it++){
                
                unsigned int key= it->first;
                vector<unsigned int> neighbors = _octree_keys[bt].get_neighbor_keys(key, 2);
                int num_neighbors = neighbors.size();

                for(int el = 0; el < num_neighbors; el++){

                    if(neighbors[el] != GeneralOctree<int>::INVALID_KEY()){
                        
                        unsigned int nbh_key = neighbors[el];
                        int nbh_pos = _octree_keys[bt].get_value(nbh_key);
                        neighbors_data[(bt * num_pixels + it->second) * num_neighbors + el] =  nbh_pos;
                        neighbor_of_data[(bt * num_pixels + nbh_pos) * num_neighbors + el] = it->second;
                      
                    }else{
                       neighbors_data[(bt * num_pixels + it->second) * num_neighbors + el]  = -1;
                    }
                }
            }
        }


    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_4_);

    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~OctConvLayerTest() {
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

TYPED_TEST_CASE(OctConvLayerTest, TestDtypesAndDevices);

TYPED_TEST(OctConvLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  OctConvParameter* oct_conv_param =
      layer_param.mutable_oct_conv_param();

  oct_conv_param->set_filter_size(2);
  oct_conv_param->set_output_channels(12);
  oct_conv_param->mutable_weight_filler()->set_type("gaussian");
  oct_conv_param->mutable_bias_filler()->set_type("gaussian");
 

  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);


  shared_ptr<Layer<Dtype> > layer(
      new OctConvLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 12);
  EXPECT_EQ(this->blob_top_->shape(2), 64);

}



/*TYPED_TEST(DualUpdateLayerConvTest, TestConvolution) {
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






}*/


TYPED_TEST(OctConvLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  
  LayerParameter layer_param;
  OctConvParameter* oct_conv_param =
      layer_param.mutable_oct_conv_param();

  oct_conv_param->set_filter_size(2);
  oct_conv_param->set_output_channels(12);
  oct_conv_param->mutable_weight_filler()->set_type("gaussian");
  oct_conv_param->mutable_bias_filler()->set_type("gaussian");
 

  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);


  OctConvLayer<Dtype> layer(layer_param);


  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}





}  // namespace caffe
