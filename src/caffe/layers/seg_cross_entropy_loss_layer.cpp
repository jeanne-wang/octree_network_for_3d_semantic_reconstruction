#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/seg_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SegCrossEntropyLossLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    LossLayer<Dtype>::LayerSetUp(bottom, top);
  
    // bottom [0] gt labels
    // bottom [1] pred probs
    // read parameter
    unknown_label_ = this->layer_param_.seg_cross_entropy_loss_param().unknown_label();
    ignore_label_ = this->layer_param_.seg_cross_entropy_loss_param().ignore_label(); 
}

template <typename Dtype>
void SegCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::Reshape(bottom, top);
  


    CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
        << "The batch_suze of groundtruth data and the predicted probs must the the same.";
    CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(2))
        << "The num_pixels_ of groundtruth data and the predicted probs must the the same.";
    num_ = bottom[1]->shape(0);
    num_classes_ = bottom[1]->shape(1);
    num_pixels_ = bottom[1]->shape(2);

  
}


template <typename Dtype>
void SegCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  

    const Dtype* gt_label = bottom[0]->cpu_data();
    const Dtype* pred_prob = bottom[1]->cpu_data();


    Dtype loss = 0;
    int valid_count = 0;
    for(int n = 0; n < num_; n++){
        for(int i = 0; i < num_pixels_; i++){

            const int label_value = static_cast<int>(gt_label[n * num_pixels_ + i]);
            if(label_value == unknown_label_ || label_value == ignore_label_){
                continue;
            }

            DCHECK_GE(label_value, 0);
            DCHECK_LT(label_value, num_classes_);
            loss -= log(std::max(pred_prob[(n*num_classes_ + label_value) * num_pixels_ + i], Dtype(FLT_MIN)));
            valid_count++;

        }
    }

    top[0]->mutable_cpu_data()[0] = loss/std::max(Dtype(valid_count), Dtype(1.0));
  
}

template <typename Dtype>
void SegCrossEntropyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    if (propagate_down[0]) {
        LOG(FATAL) << this->type()
            << " Layer cannot backpropagate to label inputs.";
    }

    

    if (propagate_down[1]) {

        const Dtype* gt_label = bottom[0]->cpu_data();
        const Dtype* pred_prob = bottom[1]->cpu_data();
        Dtype* pred_diff = bottom[1]->mutable_cpu_diff();
        caffe_set<Dtype>(bottom[1]->count(), 0, pred_diff);

        int valid_count = 0;
        for(int n = 0; n < num_; n++){
            for(int i = 0; i < num_pixels_; i++){

                const int label_value = static_cast<int>(gt_label[n * num_pixels_ + i]);
                if(label_value == unknown_label_ || label_value == ignore_label_){
                    continue;
                }

                pred_diff[(n*num_classes_ + label_value) * num_pixels_ + i] = Dtype(-1.0)/std::max(pred_prob[(n*num_classes_ + label_value) * num_pixels_ + i], Dtype(FLT_MIN));
                valid_count++;
          
            }
        }

        Dtype loss_weight = top[0]->cpu_diff()[0]/std::max(Dtype(valid_count), Dtype(1.0));
        caffe_scal<Dtype>(bottom[1]->count(), loss_weight, pred_diff);

    }
}


INSTANTIATE_CLASS(SegCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SegCrossEntropyLoss);

}  // namespace caffe
