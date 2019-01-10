#include <string>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/input3d_layer.hpp"
#include "caffe/layers/dense_grid_data_write_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

using boost::scoped_ptr;

template <typename TypeParam>
class Input3DLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  Input3DLayerTest()
      : blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()){}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  
  virtual ~Input3DLayerTest() { delete blob_top_data_; delete blob_top_label_; }


  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(Input3DLayerTest, TestDtypesAndDevices);



TYPED_TEST(Input3DLayerTest, TestReadBinaryTestPhase) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TEST);
  layer_param.mutable_input3d_param()->set_batch_size(1);
  layer_param.mutable_input3d_param()->set_height(48);
  layer_param.mutable_input3d_param()->set_width(48);
  layer_param.mutable_input3d_param()->set_depth(48);
  layer_param.mutable_input3d_param()->set_num_classes(5);
  layer_param.mutable_input3d_param()->set_row_major(false);
  layer_param.mutable_input3d_param()->set_data_list_file("/home/xiaojwan/thesis/experiment/octree_primal_dual/run/train_southbuilding_datacost.txt");
  layer_param.mutable_input3d_param()->set_groundtruth_list_file("/home/xiaojwan/thesis/experiment/octree_primal_dual/run/train_southbuilding_gt.txt");
  layer_param.mutable_input3d_param()->set_preload_data(true);
  layer_param.mutable_input3d_param()->set_customized_crop(true);


  Input3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  cout << this->blob_top_data_->shape(0) << endl;
  cout << this->blob_top_data_->shape(1) << endl;
  cout << this->blob_top_data_->shape(2) << endl;
  cout << this->blob_top_data_->shape(3) << endl;
  cout << this->blob_top_data_->shape(4) << endl;

  cout << this->blob_top_label_->shape(0) << endl;
  cout << this->blob_top_label_->shape(1) << endl;
  cout << this->blob_top_label_->shape(2) << endl;
  cout << this->blob_top_label_->shape(3) << endl;
  cout << this->blob_top_label_->shape(4) << endl;


  /*LayerParameter layer_param_2;
  layer_param_2.mutable_dense_grid_data_write_param()->set_data_write_file("/home/xiaojwan/thesis/southbuilding_gt.dat");
  DenseGridDataWriteLayer<Dtype> layer2(layer_param_2);
  
  vector<Blob<Dtype>*> top_vec;
  vector<Blob<Dtype>*> bottom_vec;
  bottom_vec.clear();
  bottom_vec.push_back(this->blob_top_label_);
  layer2.SetUp(bottom_vec, top_vec);
  layer2.Forward(bottom_vec, top_vec);*/

}

TYPED_TEST(Input3DLayerTest, TestReadBinaryTrainPhase) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.set_phase(TRAIN);
  layer_param.mutable_input3d_param()->set_batch_size(2);
  layer_param.mutable_input3d_param()->set_height(48);
  layer_param.mutable_input3d_param()->set_width(48);
  layer_param.mutable_input3d_param()->set_depth(48);
  layer_param.mutable_input3d_param()->set_num_classes(5);
  layer_param.mutable_input3d_param()->set_row_major(false);
  layer_param.mutable_input3d_param()->set_data_list_file("/home/xiaojwan/thesis/experiment/octree_primal_dual/run/train_southbuilding_datacost.txt");
  layer_param.mutable_input3d_param()->set_groundtruth_list_file("/home/xiaojwan/thesis/experiment/octree_primal_dual/run/train_southbuilding_gt.txt");
  layer_param.mutable_input3d_param()->set_preload_data(true);
  layer_param.mutable_input3d_param()->set_customized_crop(true);


  Input3DLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  EXPECT_EQ(this->blob_top_data_->shape(1), 5);
  EXPECT_EQ(this->blob_top_data_->shape(2), 48);
  EXPECT_EQ(this->blob_top_data_->shape(3), 48);
  EXPECT_EQ(this->blob_top_data_->shape(4), 48);

  EXPECT_EQ(this->blob_top_label_->shape(1), 5);
  EXPECT_EQ(this->blob_top_label_->shape(2), 48);
  EXPECT_EQ(this->blob_top_label_->shape(3), 48);
  EXPECT_EQ(this->blob_top_label_->shape(4), 48);


}


/*TYPED_TEST(Input3DLayerTest, TestReshape) {
  this->TestReshape(DataParameter_DB_LMDB);
}

TYPED_TEST(Input3DLayerTest, TestReadCropTrain) {
  const bool unique_pixels = true;  // all images the same; pixels different
  this->Fill(unique_pixels, DataParameter_DB_LMDB);
  this->TestReadCrop(TRAIN);
}*/


}  // namespace caffe
