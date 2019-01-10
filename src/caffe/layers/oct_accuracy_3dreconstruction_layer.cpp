#include <functional>
#include <utility>
#include <vector>
#include <queue>

#include "caffe/layers/oct_accuracy_3dreconstruction_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/oct_layer.hpp"

namespace caffe {

template <typename Dtype>
void OctAccuracy3DReconstructionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    freespace_label_ = this->layer_param_.oct_accuracy_3dreconstruction_param().freespace_label();
    has_unknown_label_ = this->layer_param_.oct_accuracy_3dreconstruction_param().has_unknown_label();
    if(has_unknown_label_){
        unknown_label_ = this->layer_param_.oct_accuracy_3dreconstruction_param().unknown_label();
    }
    max_level_ = this->layer_param_.oct_accuracy_3dreconstruction_param().max_level();

}

template <typename Dtype>
void OctAccuracy3DReconstructionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
    top[0]->Reshape(top_shape); // freespace accuracy
    top[1]->Reshape(top_shape); // semantic accuracy
    top[2]->Reshape(top_shape); // overall accuracy

    // bottom[0] probs_true
    // bottom[1] probs_pred
    batch_size_ = bottom[0]->shape(0);
    num_classes_ = bottom[0]->shape(1);
 
    CHECK_EQ(bottom[1]->shape(0), batch_size_)
        << "gt and label must have the same num.";
    CHECK_EQ(bottom[1]->shape(1), num_classes_)
        << "gt and label must have the same num_classes.";


    num_elements_true_ = bottom[0]->shape(2);
    num_elements_pred_ = bottom[1]->shape(2);

    vector<int> shape(2);
    shape[0] = batch_size_;
    shape[1] = num_elements_true_;
    labels_true_.Reshape(shape);
    occupied_mask_.Reshape(shape);


    shape[1] = num_elements_pred_;
    labels_pred_.Reshape(shape);
  

}

template <typename Dtype>
void OctAccuracy3DReconstructionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    // get the argmax along classes axis
    int* labels_true_data = labels_true_.mutable_cpu_data();
    int* labels_pred_data = labels_pred_.mutable_cpu_data();
    arg_max(bottom[0], labels_true_data);
    arg_max(bottom[1], labels_pred_data);

   
    // get occupied mask
    unsigned int* occupied_mask_data = occupied_mask_.mutable_cpu_data();
    for(int i  = 0; i <labels_true_.count(); i++){
        occupied_mask_data[i] = (labels_true_data[i] != freespace_label_);
        if(has_unknown_label_) occupied_mask_data[i] &= (labels_true_data[i] != unknown_label_);
    }

    const Dtype* probs_true_data = bottom[0]->cpu_data();
    for(int bt = 0; bt < batch_size_; bt++){
        for(int i = 0; i < num_elements_true_; i++){
            bool observed = false;
            for(int c = 0; c  < num_classes_; c++){

                int ind = (bt * num_classes_+ c) * num_elements_true_ + i;
                if(probs_true_data[ind] > 0.5){
                    observed = true;
                    break;
                }

            }
            occupied_mask_data[ bt * num_elements_true_ + i] &= observed;
        }
    }

    int freespace_total_count = 0;
    int occupied_total_count = 0;
    int freespace_correct_count = 0;
    int occupied_correct_count = 0;
    
    const string gt_key_layer_name = this->layer_param_.oct_accuracy_3dreconstruction_param().gt_key_layer();
    const string pr_key_layer_name = this->layer_param_.oct_accuracy_3dreconstruction_param().pr_key_layer();

    shared_ptr<Layer<Dtype> > gt_raw_ptr = this->parent_net()->layer_by_name(gt_key_layer_name);
    shared_ptr<Layer<Dtype> > pr_raw_ptr = this->parent_net()->layer_by_name(pr_key_layer_name);

    shared_ptr<OctLayer<Dtype> > gt_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(gt_raw_ptr);
    shared_ptr<OctLayer<Dtype> > pr_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(pr_raw_ptr);

    for(int bt = 0; bt < batch_size_; bt++){
       
        GeneralOctree<int> &pr_keys_octree = pr_key_layer->get_keys_octree(bt);
        GeneralOctree<int> &gt_keys_octree = gt_key_layer->get_keys_octree(bt);
        for(GeneralOctree<int>::iterator it=gt_keys_octree.begin(); it!=gt_keys_octree.end(); it++){
            
            unsigned int key = it->first;
            int label = labels_true_data[bt * num_elements_true_ + it->second];
            int key_level = GeneralOctree<int>::compute_level(key);
            int count = std::pow(8, max_level_- key_level);

            int match_count = get_octree_cell_match_count(pr_keys_octree,  labels_pred_data + bt * num_elements_pred_, key, label);
            if(labels_true_data[bt * num_elements_true_ + it->second] == freespace_label_){

                freespace_total_count += count;
                freespace_correct_count += match_count;
            }else if(occupied_mask_data[bt * num_elements_true_ + it->second]){

                occupied_total_count+= count;
                occupied_correct_count += match_count;
            }
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


template <typename Dtype>
void OctAccuracy3DReconstructionLayer<Dtype>::arg_max(const Blob<Dtype>* input, int* output){

    const Dtype* input_data  = input->cpu_data(); 
    int num_elements  =  input->shape(2);

    for(int bt = 0; bt < batch_size_; bt++){

        for(int i = 0; i < num_elements; i++){

            int max_ind = 0;
            Dtype max_val = input_data[(bt * num_classes_) *  num_elements  + i];

            for(int c = 1; c  < num_classes_; c++){

                int ind = (bt * num_classes_ + c) * num_elements + i;
                if(input_data[ind] > max_val){

                    max_val = input_data[ind];
                    max_ind = c;
                }
            }

            output[bt * num_elements + i] = max_ind;
        
        }
    }
    
}

template <typename Dtype>
int OctAccuracy3DReconstructionLayer<Dtype>::get_octree_cell_match_count(GeneralOctree<int> &pr_keys_octree, 
    const int* labels_pred_data, unsigned int key, int label){
    int match_count = 0;
    int level =  GeneralOctree<int>::compute_level(key);

    // breadth first traversel of the octree and only compare the label of the leaf nodes
    std::queue<unsigned int> nodeQueue;
    nodeQueue.push(key);
    while(!nodeQueue.empty()){

        int num = nodeQueue.size();
        for(int n = 0; n < num; n++){

            unsigned int cur_key = nodeQueue.front();
            nodeQueue.pop();
            int pos = pr_keys_octree.get_value(cur_key, true);
            if(pos != -1){

                if(labels_pred_data[pos] == label) match_count += std::pow(8, max_level_-level);

            }else{
                int key_level = GeneralOctree<int>::compute_level(cur_key);
                const string gt_key_layer_name = this->layer_param_.oct_accuracy_3dreconstruction_param().gt_key_layer();
                const string pr_key_layer_name = this->layer_param_.oct_accuracy_3dreconstruction_param().pr_key_layer();
                CHECK_NE(key_level, max_level_) 
                    << "It is impossible for cell in the finest level in " 
                    << gt_key_layer_name << " to not be found in " << pr_key_layer_name;


                cur_key <<= 3;
                for(int i = 0; i < 8; i++){
                    nodeQueue.push(cur_key | i);
                }

            }

        }
        

        level++;
    }

    return match_count;
}

INSTANTIATE_CLASS(OctAccuracy3DReconstructionLayer);
REGISTER_LAYER_CLASS(OctAccuracy3DReconstruction);

}  // namespace caffe
