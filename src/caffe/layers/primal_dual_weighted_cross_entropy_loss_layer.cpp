#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/primal_dual_weighted_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrimalDualWeightedCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  // bottom [0] gt
  // bottom [1] pred probs
  // read parameter
  PrimalDualWeightedCrossEntropyLossParameter param = this->layer_param_.primal_dual_weighted_cross_entropy_loss_param();
  softmax_scale_ = param.softmax_scale();
  clip_epsilon_ = param.clip_epsilon();
  unknown_label_ = param.unknown_label();


  LayerParameter softmax_param;
  softmax_param.mutable_softmax_param()->set_axis(1);
  gt_softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));

  gt_scale_.ReshapeLike(*bottom[0]);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(&gt_scale_);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&gt_softmax_);
  gt_softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  
  LayerParameter clip_by_value_param;
  clip_by_value_param.mutable_clip_by_value_param()->set_min_val(clip_epsilon_);
  clip_by_value_param.mutable_clip_by_value_param()->set_max_val(1.0-clip_epsilon_);
  clip_by_value_layer_.reset(new ClipByValueLayer<Dtype>(clip_by_value_param));
  clip_by_value_bottom_vec_.clear();
  clip_by_value_bottom_vec_.push_back(&gt_softmax_);
  clip_by_value_bottom_vec_.push_back(bottom[1]);
  clip_by_value_top_vec_.clear();
  clip_by_value_top_vec_.push_back(&gt_softmax_clip_);
  clip_by_value_top_vec_.push_back(&pred_softmax_clip_);
  clip_by_value_layer_->SetUp(clip_by_value_bottom_vec_, clip_by_value_top_vec_);
}

template <typename Dtype>
void PrimalDualWeightedCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  LossLayer<Dtype>::Reshape(bottom, top);

  gt_scale_.ReshapeLike(*bottom[0]);
  gt_softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  clip_by_value_layer_->Reshape(clip_by_value_bottom_vec_, clip_by_value_top_vec_);
  gt_log_pred_.ReshapeLike(*bottom[0]);


  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
    << "The shape of groundtruth data and the predicted probs must the the same.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
    << "The shape of groundtruth data and the predicted probs must the the same.";
  CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))
    << "The shape of groundtruth data and the predicted probs must the the same.";
  CHECK_EQ(bottom[0]->shape(3), bottom[1]->shape(3))
    << "The shape of groundtruth data and the predicted probs must the the same.";
  CHECK_EQ(bottom[0]->shape(4), bottom[1]->shape(4))
    << "The shape of groundtruth data and the predicted probs must the the same.";


  num_ = bottom[0]->shape(0);
  num_classes_ = bottom[0]->shape(1);
  num_rows_ = bottom[0]->shape(2);
  num_cols_ = bottom[0]->shape(3);
  num_slices_ = bottom[0]->shape(4);
  weights_.Reshape(num_, num_rows_, num_cols_, num_slices_);
  
}


template <typename Dtype>
void PrimalDualWeightedCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
  const int count = bottom[0]->count();

  // scale ground truth data
  const Dtype* gt_data = bottom[0]->cpu_data();
  Dtype* gt_scale_data = gt_scale_.mutable_cpu_data();
  caffe_set<Dtype>(count, Dtype(0.), gt_scale_data);
  caffe_axpy<Dtype>(count, Dtype(softmax_scale_), gt_data, gt_scale_data);

  // softmax on groundtruth data
  gt_softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  // clip value
  clip_by_value_layer_->Forward(clip_by_value_bottom_vec_, clip_by_value_top_vec_);
  const Dtype* pred_clip_data = pred_softmax_clip_.cpu_data();
  const Dtype* gt_clip_data = gt_softmax_clip_.cpu_data();

  

  Dtype* gt_log_pred_data = gt_log_pred_.mutable_cpu_data();  
  caffe_log<Dtype>(count, pred_clip_data, gt_log_pred_data);
  caffe_mul<Dtype>(count, gt_log_pred_data, gt_clip_data, gt_log_pred_data);

  // weights
  Dtype* weights_data = weights_.mutable_cpu_data();
  Dtype weights_sum = 0;
  for(int n = 0; n < num_; n++){
    for(int i = 0; i < num_rows_; i++){
      for(int j = 0; j < num_cols_; j++){
        for(int k = 0; k < num_slices_; k++){
          int ind = ((n * num_rows_ + i) * num_cols_ + j)* num_slices_ + k;
          weights_data[ind] = std::log(Dtype(num_classes_));

          for(int c = 0;  c < num_classes_; c++){
            int ind2 = (((n*num_classes_ + c) * num_rows_ + i)*num_cols_ + j)* num_slices_ + k;
            weights_data[ind] += gt_clip_data[ind2] * std::log(gt_clip_data[ind2]);
          }

          int ind3 = (((n*num_classes_ + unknown_label_) * num_rows_ + i)*num_cols_ + j)* num_slices_ + k;
          weights_data[ind] *= (Dtype(1.) - gt_clip_data[ind3]);

          weights_data[ind] = std::max(Dtype(clip_epsilon_), weights_data[ind]);
          weights_sum += weights_data[ind];

        }
      }
    }
  }

  caffe_scal<Dtype>(weights_.count(), Dtype(1.0)/ weights_sum, weights_data);

  Dtype loss = 0;
  for(int n = 0; n < num_; n++){
    for(int i = 0; i < num_rows_; i++){
      for(int j = 0; j < num_cols_; j++){
        for(int k = 0; k < num_slices_; k++){
          int ind = ((n * num_rows_ + i) * num_cols_ + j)* num_slices_ + k;
          for(int c = 0;  c < num_classes_; c++){
            int ind2 = (((n*num_classes_ + c) * num_rows_ + i)*num_cols_ + j)* num_slices_ + k;
            loss -= gt_log_pred_data[ind2] * weights_data[ind];
          }

        }
      }
    }
  }

  top[0]->mutable_cpu_data()[0] = loss;
  
  
}

template <typename Dtype>
void PrimalDualWeightedCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


  if (propagate_down[0]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  const int count = bottom[0]->count();

  if (propagate_down[1]) {
    Dtype* pred_clip_diff = pred_softmax_clip_.mutable_cpu_diff();
    const Dtype* pred_clip_data = pred_softmax_clip_.cpu_data();
    const Dtype* gt_clip_data = gt_softmax_clip_.cpu_data();

    caffe_div<Dtype>(count, gt_clip_data, pred_clip_data, pred_clip_diff);
    const Dtype* weights_data = weights_.cpu_data();

    for(int n = 0; n < num_; n++){
      for(int i = 0; i < num_rows_; i++){
        for(int j = 0; j < num_cols_; j++){
          for(int k = 0; k < num_slices_; k++){
              int ind = ((n * num_rows_ + i) * num_cols_ + j)* num_slices_ + k;
              for(int c = 0; c < num_classes_; c++){
                  int ind2 = (((n * num_classes_ + c) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k;
                  pred_clip_diff[ind2] *= Dtype(-1.) * weights_data[ind];
              }
            }
          }
        }
      }

    

    vector<bool> down(2);
    down[0] = false;
    down[1] = true;
    clip_by_value_layer_->Backward(clip_by_value_top_vec_, down, clip_by_value_bottom_vec_);

  }
}

#ifdef CPU_ONLY
STUB_GPU(PrimalDualWeightedCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(PrimalDualWeightedCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(PrimalDualWeightedCrossEntropyLoss);

}  // namespace caffe
