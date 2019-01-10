#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/primal_dual_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrimalDualCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  // bottom [0] gt
  // bottom [1] pred probs
  // read parameter
  PrimalDualCrossEntropyLossParameter param = this->layer_param_.primal_dual_cross_entropy_loss_param();
  softmax_scale_ = param.softmax_scale();
  clip_epsilon_ = param.clip_epsilon();
  freespace_weighted_ = param.freespace_weighted();
  unknown_weighted_ = param.unknown_weighted();

  CHECK((!freespace_weighted_ && !unknown_weighted_) || (freespace_weighted_ && ! unknown_weighted_) 
    ||(!freespace_weighted_ && unknown_weighted_))
    << "must choose only one of the weighting method.";


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
void PrimalDualCrossEntropyLossLayer<Dtype>::Reshape(
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
  
}


template <typename Dtype>
void PrimalDualCrossEntropyLossLayer<Dtype>::Forward_cpu(
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
  if(freespace_weighted_){

    Dtype free_loss = 0;
    Dtype occu_loss = 0;
    free_count_ = 0;
    occu_count_ = 0;

    for(int n = 0; n < num_; n++){
      for(int i = 0; i < num_rows_; i++){
        for(int j = 0; j < num_cols_; j++){
          for(int k = 0; k < num_slices_; k++){

            Dtype max_val = gt_clip_data[(((n * num_classes_) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k];
            int max_ind = 0; // init the max_val and max_ind in the first channel
            Dtype sum = -1 * gt_log_pred_data[(((n * num_classes_) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k];
            for(int c = 1; c < num_classes_; c++){

              int ind = (((n * num_classes_ + c) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k;
              if(gt_clip_data[ind] > max_val)
              {
                max_val = gt_clip_data[ind];
                max_ind = c;
              }

              sum -= gt_log_pred_data[ind];
            }

            if(max_ind == 0){
              free_count_++;
              free_loss += sum;
              
            }
            else{
              occu_count_++;
              occu_loss += sum;
            }
          }
        }
      }
    }

    free_loss /= (free_count_ + clip_epsilon_);
    occu_loss /= (occu_count_ + clip_epsilon_);

    top[0]->mutable_cpu_data()[0] = 2*free_loss + occu_loss;


  }else {
    Dtype loss = 0;
    for(int i = 0; i < count; i++){
      loss -= gt_log_pred_data[i];
    }
    top[0]->mutable_cpu_data()[0] = loss / count;
  }
  
}

template <typename Dtype>
void PrimalDualCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

    if(freespace_weighted_){
      caffe_div<Dtype>(count, gt_clip_data, pred_clip_data, pred_clip_diff);

      Dtype  free_loss_weight = -2 * top[0]->cpu_diff()[0] / free_count_;
      Dtype  occu_loss_weight = -1 * top[0]->cpu_diff()[0] / occu_count_;

      for(int n = 0; n < num_; n++){
        for(int i = 0; i < num_rows_; i++){
          for(int j = 0; j < num_cols_; j++){
            for(int k = 0; k < num_slices_; k++){

              Dtype max_val = gt_clip_data[(((n * num_classes_) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k];
              int max_ind = 0; // init the max_val and max_ind in the first channel
              for(int c = 1; c < num_classes_; c++){
                int ind = (((n * num_classes_ + c) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k;
                if(gt_clip_data[ind] > max_val)
                {
                  max_val = gt_clip_data[ind];
                  max_ind = c;
                }

              }

              for(int c = 0; c < num_classes_; c++){
                  int ind = (((n * num_classes_ + c) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k;
                  if(max_ind == 0){
                    pred_clip_diff[ind] /= free_loss_weight;
                  }else{
                    pred_clip_diff[ind] /= occu_loss_weight;
                  }
                  
              }
            }
          }
        }
      }

    }else{
      Dtype loss_weight = -1 * top[0]->cpu_diff()[0] /count;
      caffe_div<Dtype>(count, gt_clip_data, pred_clip_data, pred_clip_diff);
      caffe_scal<Dtype>(count, Dtype(loss_weight), pred_clip_diff);
    }

    vector<bool> down(2);
    down[0] = false;
    down[1] = true;
    clip_by_value_layer_->Backward(clip_by_value_top_vec_, down, clip_by_value_bottom_vec_);

  }
}

#ifdef CPU_ONLY
STUB_GPU(PrimalDualCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(PrimalDualCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(PrimalDualCrossEntropyLoss);

}  // namespace caffe
