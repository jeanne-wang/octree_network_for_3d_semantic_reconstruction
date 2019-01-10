#include <vector>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/oct_primal_update_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "image_tree_tools/octree.h"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

/*// Reference convolution for checking results:
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
*/
// this test is designed for testing the conv result in dual update layer 
template <typename TypeParam>
class OctPrimalUpdateLayerConvTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OctPrimalUpdateLayerConvTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_bottom_2_(new Blob<Dtype>()),
        blob_bottom_3_(new Blob<Dtype>()),
        blob_bottom_4_(new Blob<Dtype>()),
        blob_bottom_5_(new Blob<Dtype>()),
        blob_bottom_6_(new Blob<Dtype>()),
        blob_bottom_7_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {

    vector<int> primal_shape(3);
    primal_shape[0] = 2;
    primal_shape[1] = 2;
    primal_shape[2] = 64;
   
    vector<int> dual_shape(primal_shape.begin(), primal_shape.end());
    dual_shape[1] = 6;

    blob_bottom_->Reshape(primal_shape); // orimal
    blob_bottom_2_->Reshape(dual_shape); // dual

    vector<int> lag_shape(primal_shape.begin(), primal_shape.end());
    lag_shape[1] = 1;
    blob_bottom_6_->Reshape(lag_shape); //lag
    blob_bottom_7_->Reshape(primal_shape); //datacost


    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom_);
    filler.Fill(blob_bottom_2_);
    filler.Fill(blob_bottom_6_);
    filler.Fill(blob_bottom_7_);


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
    blob_bottom_3_->Reshape(num_shape);
    caffe_set<Dtype>(2, num_output_pixels, blob_bottom_3_->mutable_cpu_data());

    vector<int> nbh_shape(3);
    nbh_shape[0] = 2;
    nbh_shape[1] = num_output_pixels;
    nbh_shape[2] = 2*2*2;
    blob_bottom_4_->Reshape(nbh_shape);
    blob_bottom_5_->Reshape(nbh_shape);

    int batch_size = blob_bottom_4_->shape(0);
    int num_pixels = blob_bottom_4_->shape(1);

    Dtype* neighbors_data = blob_bottom_4_->mutable_cpu_data();
    Dtype* neighbor_of_data = blob_bottom_5_->mutable_cpu_data();
    caffe_set<Dtype>(blob_bottom_5_->count(), -1, neighbor_of_data);

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
    blob_bottom_vec_.push_back(blob_bottom_6_);
    blob_bottom_vec_.push_back(blob_bottom_7_);
    blob_bottom_vec_.push_back(blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_4_);
    blob_bottom_vec_.push_back(blob_bottom_5_);
    blob_top_vec_.push_back(blob_top_);
 
  }

  virtual ~OctPrimalUpdateLayerConvTest() {
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_bottom_4_;
    delete blob_bottom_5_;
    delete blob_bottom_6_;
    delete blob_bottom_7_;
    delete blob_top_;
  }

 

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_bottom_4_;
  Blob<Dtype>* const blob_bottom_5_;
  Blob<Dtype>* const blob_bottom_6_;
  Blob<Dtype>* const blob_bottom_7_;
  Blob<Dtype>* const blob_top_;
  

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(OctPrimalUpdateLayerConvTest, TestDtypesAndDevices);

TYPED_TEST(OctPrimalUpdateLayerConvTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  OctPrimalUpdateParameter* primal_update_param =
      layer_param.mutable_oct_primal_update_param();

  primal_update_param->set_filter_size(2);
  primal_update_param->set_output_channels(2);
  primal_update_param->mutable_weight_filler()->set_type("gaussian");
  primal_update_param->set_tau(0.01);

  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);


  shared_ptr<Layer<Dtype> > layer(
      new OctPrimalUpdateLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->shape(0), 2);
  EXPECT_EQ(this->blob_top_->shape(1), 2);
  EXPECT_EQ(this->blob_top_->shape(2), 64);

}





TYPED_TEST(OctPrimalUpdateLayerConvTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  
  OctPrimalUpdateParameter* primal_update_param =
      layer_param.mutable_oct_primal_update_param();

  primal_update_param->set_filter_size(2);
  primal_update_param->set_output_channels(2);
  primal_update_param->mutable_weight_filler()->set_type("gaussian");
  primal_update_param->set_tau(0.01);

  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(true);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);
  layer_param.add_propagate_down(false);

  OctPrimalUpdateLayer<Dtype> layer(layer_param);


  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 1);
}





}  // namespace caffe
