#include "caffe/layers/oct_seg_loss_prep_layer.hpp"

#include "caffe/net.hpp"
#include "caffe/layers/oct_layer.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void OctSegLossPrepLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    has_ignore_label_ = this->layer_param_.oct_seg_loss_prep_param().has_ignore_label();
    if(has_ignore_label_){
        ignore_label_ = this->layer_param_.oct_seg_loss_prep_param().ignore_label();
    } // in this case, the gt is label instead of probability distribution
}

template <typename Dtype>
void OctSegLossPrepLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // bottom[0] octree input data (gt)
    // bottom[1] ref data
    CHECK_EQ(bottom[1]->num_axes(), 3)
        << "The number of dimensions of octree features should be 3.";
    
    // this is for gt in one-hot encoding
    if(has_ignore_label_){
        vector<int> shape(2);
        shape[0] = bottom[1]->shape(0);
        shape[1] = bottom[1]->shape(2);
        top[0]->Reshape(shape);
    }else{
        top[0]->ReshapeLike(*bottom[1]);
    }
    
}

template <typename Dtype>
void OctSegLossPrepLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    int batch_size = bottom[1]->shape(0);
    int channels = bottom[1]->shape(1);
    int num_input_pixels = bottom[0]->shape(bottom[0]->num_axes()-1);
    int num_output_pixels = bottom[1]->shape(2);

    const string input_key_layer_name = this->layer_param_.oct_seg_loss_prep_param().input_key_layer();
    const string ref_key_layer_name = this->layer_param_.oct_seg_loss_prep_param().ref_key_layer();

    shared_ptr<Layer<Dtype> > input_raw_ptr = this->parent_net()->layer_by_name(input_key_layer_name);
    shared_ptr<Layer<Dtype> > ref_raw_ptr = this->parent_net()->layer_by_name(ref_key_layer_name);

    shared_ptr<OctLayer<Dtype> > input_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(input_raw_ptr);
    shared_ptr<OctLayer<Dtype> > ref_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(ref_raw_ptr);


    const Dtype* input = bottom[0]->cpu_data();
    Dtype* output = top[0]->mutable_cpu_data();
    if(!has_ignore_label_){
        Dtype uni_prob = Dtype(1.)/Dtype(channels);
        caffe_set<Dtype>(top[0]->count(), uni_prob, output); // setting defualt to uniform probs, this will be ignored in loss computation
    }else{
        caffe_set<Dtype>(top[0]->count(), ignore_label_, output); // setting defualt to ignored label, this will be ignored in loss computation
    }
    
    for(int bt = 0; bt<batch_size; bt++){
       
        GeneralOctree<int> &ref_keys_octree = ref_key_layer->get_keys_octree(bt);
        GeneralOctree<int> &input_keys_octree = input_key_layer->get_keys_octree(bt);

        for(GeneralOctree<int>::iterator it=ref_keys_octree.begin(); it!=ref_keys_octree.end(); it++){    

            int pos = input_keys_octree.get_value(it->first, true);
            if(pos != -1){
                if(!has_ignore_label_){
                    for (int ch = 0; ch < channels; ch++){
                        output[(bt * channels + ch) * num_output_pixels + it->second]
                            = input[(bt * channels + ch) * num_input_pixels + pos];
                    }
                }
                else{
                    
                    output[bt * num_output_pixels + it->second] = input[bt * num_input_pixels + pos];
                }
            }
            
        }
        
    }
}

template <typename Dtype>
void OctSegLossPrepLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}




INSTANTIATE_CLASS(OctSegLossPrepLayer);
REGISTER_LAYER_CLASS(OctSegLossPrep);

}  // namespace caffe
