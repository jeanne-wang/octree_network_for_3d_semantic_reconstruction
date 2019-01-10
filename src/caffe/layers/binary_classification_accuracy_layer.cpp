#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/binary_classification_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "image_tree_tools/oct_prop_util.h"
namespace caffe {

template <typename Dtype>
void BinaryClassificationAccuracyLayer<Dtype>::LayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void BinaryClassificationAccuracyLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
    top[0]->Reshape(top_shape);  
    top[1]->Reshape(top_shape);   
    top[2]->Reshape(top_shape);    
}


template <typename Dtype>
void BinaryClassificationAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  

    const Dtype* gt_label = bottom[0]->cpu_data();
    const Dtype* pred_prob = bottom[1]->cpu_data();

    int num = bottom[0]->shape(0);
    int num_pixels = bottom[0]->count()/num;

    
    int valid_count = 0;
    int positive_count = 0;
    int negative_count = 0;
    
    int correct_count = 0;
    int positive_correct_count = 0;
    int negative_correct_count = 0;
    for(int n = 0; n < num; n++){
        for(int i = 0; i < num_pixels; i++){

            const int label_value = static_cast<int>(gt_label[n * num_pixels + i]);
            if(label_value == PROP_IGNORE_CLASS){
                continue;
            }

            DCHECK_GE(label_value, 0);
            DCHECK_LE(label_value, 1);
            int pred_label_value = pred_prob[n * num_pixels + i] >= 0.5? 1: 0;
            if(label_value == 1){
                positive_correct_count += (pred_label_value == label_value);
                positive_count++;
            }else{
                negative_correct_count += (pred_label_value == label_value);
                negative_count++;
            }
            correct_count += (pred_label_value == label_value);
            valid_count++;

        }
    }
    CHECK_EQ(positive_count+negative_count, valid_count)
        <<"the labels should either be positive or negative.";

    top[0]->mutable_cpu_data()[0] = (Dtype)correct_count/std::max(Dtype(valid_count), Dtype(1.0));
    top[1]->mutable_cpu_data()[0] = (Dtype)positive_correct_count/std::max(Dtype(positive_count), Dtype(1.0));
    top[2]->mutable_cpu_data()[0] = (Dtype)negative_correct_count/std::max(Dtype(negative_count), Dtype(1.0));
  
}


INSTANTIATE_CLASS(BinaryClassificationAccuracyLayer);
REGISTER_LAYER_CLASS(BinaryClassificationAccuracy);

}  // namespace caffe
