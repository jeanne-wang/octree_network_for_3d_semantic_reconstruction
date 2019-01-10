#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/accuracy_3dreconstruction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Accuracy3DReconstructionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    freespace_label_ = this->layer_param_.accuracy_3dreconstruction_param().freespace_label();
    has_unknown_label_ = this->layer_param_.accuracy_3dreconstruction_param().has_unknown_label();
    if(has_unknown_label_){
        unknown_label_ = this->layer_param_.accuracy_3dreconstruction_param().unknown_label();
    }
}

template <typename Dtype>
void Accuracy3DReconstructionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
    vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
    top[0]->Reshape(top_shape); // freespace accuracy
    top[1]->Reshape(top_shape); // semantic accuracy
    top[2]->Reshape(top_shape); // overall accuracy

    // bottom[0] gt
    // bottom[1] pred
    num_ = bottom[0]->shape(0);
    num_classes_ = bottom[0]->shape(1);
    num_rows_ = bottom[0]->shape(2);
    num_cols_ = bottom[0]->shape(3);
    num_slices_ = bottom[0]->shape(4);

    CHECK_EQ(bottom[1]->shape(0), num_)
        << "gt and label must have the same num.";
    CHECK_EQ(bottom[1]->shape(1), num_classes_)
        << "gt and label must have the same num_classes.";
    CHECK_EQ(bottom[1]->shape(2), num_rows_)
        << "gt and label must have the same num_rows.";
    CHECK_EQ(bottom[1]->shape(3), num_cols_)
        << "gt and label must have the same num_cols.";
    CHECK_EQ(bottom[1]->shape(4), num_slices_)
        << "gt and label must have the same num_slices.";

    vector<int> shape(4);
    shape[0] = num_;
    shape[1] = num_rows_;
    shape[2] = num_cols_;
    shape[3] = num_slices_;
    labels_true_.Reshape(shape);
    labels_pred_.Reshape(shape);
    occupied_mask_.Reshape(shape);

}

template <typename Dtype>
void Accuracy3DReconstructionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    const Dtype* bottom_label = bottom[0]->cpu_data();
    const Dtype* bottom_data = bottom[1]->cpu_data();

    int* labels_true_data = labels_true_.mutable_cpu_data();
    int* labels_pred_data = labels_pred_.mutable_cpu_data();
    for(int n = 0; n < num_; n++){
        for(int i = 0; i < num_rows_; i++){
            for(int j = 0; j < num_cols_; j++){
                for(int k = 0; k < num_slices_; k++){
                    int labels_true_max_ind = 0;
                    Dtype labels_true_max_val = bottom_label[ (((n * num_classes_) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k];

                    int labels_pred_max_ind = 0;
                    Dtype labels_pred_max_val = bottom_data[ (((n * num_classes_) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k];

                    for(int c = 1; c  < num_classes_; c++){
                        int ind = (((n * num_classes_ + c) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k;
                        if(bottom_label[ind] > labels_true_max_val){
                            labels_true_max_val = bottom_label[ind];
                            labels_true_max_ind = c;
                        }

                        if(bottom_data[ind] > labels_pred_max_val){
                            labels_pred_max_val = bottom_data[ind];
                            labels_pred_max_ind = c;
                        }
                    }

                    int index = ((n * num_rows_ + i) * num_cols_ + j) * num_slices_ + k;
                    labels_true_data[index] = labels_true_max_ind;
                    labels_pred_data[index] = labels_pred_max_ind;
                }
            }
        }
    }

    const int count = labels_true_.count();
    unsigned int* occupied_mask_data = occupied_mask_.mutable_cpu_data();
    for(int i  = 0; i < count; i++){

        occupied_mask_data[i] = (labels_true_data[i] != freespace_label_);
        if(has_unknown_label_) occupied_mask_data[i] &= (labels_true_data[i] != unknown_label_);
    }


    for(int n = 0; n < num_; n++){
        for(int i = 0; i < num_rows_; i++){
            for(int j = 0; j < num_cols_; j++){
                for(int k = 0; k < num_slices_; k++){
          
                    bool observed = false;
                    for(int c = 0; c  < num_classes_; c++){
                        int ind = (((n * num_classes_ + c) * num_rows_ + i) * num_cols_ + j) * num_slices_ + k;
                        if(bottom_label[ind] > 0.5){
                            observed = true;
                            break;
                        }

                    }
                    occupied_mask_data[ ((n * num_rows_ + i) * num_cols_ + j) * num_slices_ + k] &= observed;
                }
            }
        }
    }

  

    int freespace_total_count = 0;
    int freespace_correct_count = 0;
    int occupied_total_count = 0;
    int occupied_correct_count = 0;

    for(int i = 0; i < count; i++){
        if(labels_true_data[i] == freespace_label_){
            freespace_total_count++;
            freespace_correct_count += (labels_true_data[i] == labels_pred_data[i]);
        }else if(occupied_mask_data[i]){
            occupied_total_count++;
            occupied_correct_count += (labels_true_data[i] == labels_pred_data[i]);
         /* if(freespace_label_ == 0){
            occupied_accuracy_correct_count += ((labels_true_data[i] > freespace_label_) && (labels_pred_data[i] > freespace_label_));
          }else{
             occupied_accuracy_correct_count += ((labels_true_data[i] < freespace_label_) && (labels_pred_data[i] < freespace_label_));
          }*/
      
        }
    }
    
    Dtype freespace_accuracy = 0;
    Dtype semantic_accuracy = 0;
    Dtype overall_accuracy = 0;

    if(freespace_total_count != 0){
        freespace_accuracy = Dtype(freespace_correct_count)/Dtype(freespace_total_count);
    }

    if(occupied_total_count != 0){
        semantic_accuracy = Dtype(occupied_correct_count)/Dtype(occupied_total_count);
    }

    int total_count = freespace_total_count + occupied_total_count;

    if(total_count != 0){
        overall_accuracy = Dtype(freespace_correct_count + occupied_correct_count)/Dtype(total_count);
    }
  

    top[0]->mutable_cpu_data()[0] = freespace_accuracy;
    top[1]->mutable_cpu_data()[0] = semantic_accuracy;
    top[2]->mutable_cpu_data()[0] = overall_accuracy;

}

INSTANTIATE_CLASS(Accuracy3DReconstructionLayer);
REGISTER_LAYER_CLASS(Accuracy3DReconstruction);

}  // namespace caffe
