#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/oct_primal_dual_weighted_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OctPrimalDualWeightedCrossEntropyLossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::LayerSetUp(bottom, top);
  
    // bottom [0] gt
    // bottom [1] pred probs
    // read parameter
    unknown_label_ = this->layer_param_.primal_dual_weighted_cross_entropy_loss_param().unknown_label();
    clip_epsilon_ = this->layer_param_.primal_dual_weighted_cross_entropy_loss_param().clip_epsilon();

    LayerParameter clip_by_value_param;
    clip_by_value_param.mutable_clip_by_value_param()->set_min_val(clip_epsilon_);
    clip_by_value_param.mutable_clip_by_value_param()->set_max_val(1.0-clip_epsilon_);
    clip_by_value_layer_.reset(new ClipByValueLayer<Dtype>(clip_by_value_param));
    clip_by_value_bottom_vec_.clear();
    clip_by_value_bottom_vec_.push_back(bottom[0]);
    clip_by_value_bottom_vec_.push_back(bottom[1]);
    clip_by_value_top_vec_.clear();
    clip_by_value_top_vec_.push_back(&gt_clip_);
    clip_by_value_top_vec_.push_back(&pred_clip_);
    clip_by_value_layer_->SetUp(clip_by_value_bottom_vec_, clip_by_value_top_vec_);

 
}

template <typename Dtype>
void OctPrimalDualWeightedCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::Reshape(bottom, top);
  


    CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
        << "The shape of groundtruth data and the predicted probs must the the same.";
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
        << "The shape of groundtruth data and the predicted probs must the the same.";
    CHECK_EQ(bottom[0]->shape(2), bottom[1]->shape(2))
        << "The shape of groundtruth data and the predicted probs must the the same.";

    clip_by_value_layer_->Reshape(clip_by_value_bottom_vec_, clip_by_value_top_vec_);
    gt_log_pred_.ReshapeLike(*bottom[0]);
    num_ = bottom[0]->shape(0);
    num_classes_ = bottom[0]->shape(1);
    num_pixels_ = bottom[0]->shape(2);

    vector<int> shape;
    shape.push_back(num_);
    shape.push_back(num_pixels_);
    weights_.Reshape(shape);
  
}


template <typename Dtype>
void OctPrimalDualWeightedCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  

    // clip value
    clip_by_value_layer_->Forward(clip_by_value_bottom_vec_, clip_by_value_top_vec_);

    const int count = bottom[0]->count();

    const Dtype* pred_data = pred_clip_.cpu_data();
    const Dtype* gt_data = gt_clip_.cpu_data();

  

    Dtype* gt_log_pred_data = gt_log_pred_.mutable_cpu_data();  
    caffe_log<Dtype>(count, pred_data, gt_log_pred_data);
    caffe_mul<Dtype>(count, gt_log_pred_data, gt_data, gt_log_pred_data);

    // weights
    Dtype* weights_data = weights_.mutable_cpu_data();
    Dtype weights_sum = 0;
    for(int n = 0; n < num_; n++){
        for(int i = 0; i < num_pixels_; i++){
      
            int ind = n * num_pixels_ + i;
            weights_data[ind] = std::log(Dtype(num_classes_));

            for(int c = 0;  c < num_classes_; c++){
                int ind2 = (n*num_classes_ + c) * num_pixels_ + i;
                weights_data[ind] += gt_data[ind2] * std::log(gt_data[ind2]);
            }

            int ind3 = (n*num_classes_ + unknown_label_) * num_pixels_ + i;
            weights_data[ind] *= (Dtype(1.) - gt_data[ind3]);

            weights_data[ind] = std::max(Dtype(clip_epsilon_), weights_data[ind]);
            weights_sum += weights_data[ind];
        }
    }

    
    caffe_scal<Dtype>(weights_.count(), Dtype(1.0)/ weights_sum, weights_data);
    
    

    Dtype loss = 0;
    for(int n = 0; n < num_; n++){
        for(int i = 0; i < num_pixels_; i++){
      
            int ind = n * num_pixels_ + i;
            for(int c = 0;  c < num_classes_; c++){
                int ind2 = (n*num_classes_ + c) * num_pixels_ + i;
                loss -= gt_log_pred_data[ind2] * weights_data[ind];
            }

        }
    }

    top[0]->mutable_cpu_data()[0] = loss;
  
}

template <typename Dtype>
void OctPrimalDualWeightedCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (propagate_down[0]) {
        LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
    }

    Dtype top_diff = top[0]->cpu_diff()[0];
    const int count = bottom[0]->count();

    if (propagate_down[1]) {
        Dtype* pred_diff = pred_clip_.mutable_cpu_diff();
        const Dtype* pred_data = pred_clip_.cpu_data();
        const Dtype* gt_data = gt_clip_.cpu_data();

        caffe_div<Dtype>(count, gt_data, pred_data, pred_diff);
        const Dtype* weights_data = weights_.cpu_data();

        for(int n = 0; n < num_; n++){
            for(int i = 0; i < num_pixels_; i++){
       
                int ind = n * num_pixels_ + i;
                for(int c = 0; c < num_classes_; c++){
                    int ind2 = (n * num_classes_ + c) * num_pixels_ + i;
                    pred_diff[ind2] *= Dtype(-1.) * weights_data[ind] * top_diff;
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
STUB_GPU(OctPrimalDualWeightedCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(OctPrimalDualWeightedCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(OctPrimalDualWeightedCrossEntropyLoss);

}  // namespace caffe
