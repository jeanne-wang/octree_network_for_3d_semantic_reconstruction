#include "caffe/layers/oct_output_layer.hpp"

#include "caffe/net.hpp"
#include "image_tree_tools/octree.h"

namespace caffe {

using namespace std;

template <typename Dtype>
void OctOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    _done_initial_reshape = false;
    _min_level = this->layer_param_.oct_output_param().min_level();
}

template <typename Dtype>
void OctOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // bottom[i] u
    // bottom[i+1] pred/gt probs for propagation
    // the last level do not have pred/gt probs for propagation
    // so bottom.size()%2 == 1

    int batch_size = bottom[0]->shape(0);
    int num_classes = bottom[0]->shape(1);
    _num_levels = bottom.size()/2+1;
   
    for(int i = 0; i < bottom.size(); i+=2){
        CHECK_EQ(bottom[i]->shape(0), batch_size)
            <<"all bottoms should have the same batch_size.";
        CHECK_EQ(bottom[i]->shape(1), num_classes)
            <<"all bottoms u should have the same number of classes.";

        if(i < bottom.size()-1){
            CHECK_EQ(bottom[i+1]->shape(0), batch_size)
                <<"pred/gt should have the same batch_size with u.";
            CHECK_EQ(bottom[i]->shape(2), bottom[i+1]->shape(2))
                <<"pred/gt should have the same input pixels with u.";
            CHECK_EQ(bottom[i+1]->shape(1), 1)
                <<"pred/gt should have one channel.";
        }
    }
	
    if(!_done_initial_reshape){
        _num_output_pixels = 1;
        _done_initial_reshape = true;
    }else{

        compute_pixel_propagation(bottom, top);
    }

	if(!_num_output_pixels) _num_output_pixels = 1;

	
	vector<int> shape_features;
    shape_features.push_back(batch_size); 
    shape_features.push_back(num_classes); 
    shape_features.push_back(_num_output_pixels);
    top[0]->Reshape(shape_features);
}

template <typename Dtype>
void OctOutputLayer<Dtype>::compute_pixel_propagation(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){


    this->_octree_keys.clear();
    _num_output_pixels = 0;

    const int batch_size = bottom[0]->shape(0);

    for(int bt=0; bt<batch_size; bt++){

        int counter_top = 0;
        GeneralOctree<int> octree_keys;

        for(int l = 0; l < _num_levels; l++){

            const Dtype* input_values = NULL;
            if(l < _num_levels-1) input_values = bottom[2*l+1]->cpu_data();
            const int num_input_pixels = bottom[2*l]->shape(2);

            string key_layer_name = this->layer_param_.oct_output_param().key_layer(l);
            boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
            boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);    

            for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(bt).begin(); it!=l_ptr->get_keys_octree(bt).end(); it++){

                int v;
                OctOutputParameter_PropagationMode prop_mode = this->layer_param().oct_output_param().prop_mode();
                if(l == _num_levels-1){
                    v = 0;
                }else if(prop_mode == OctOutputParameter_PropagationMode_PROP_PRED){    

                    if(input_values[bt * num_input_pixels + it->second] >= 0.5){
                        v = 1;
                    }else{
                        v = 0;
                    }

                }else if(prop_mode == OctOutputParameter_PropagationMode_PROP_KNOWN){
                    v = input_values[bt * num_input_pixels + it->second];
                }

                if(v == 0){
                    octree_keys.add_element(it->first, counter_top);
                    counter_top++;
                }
            } 
        }


        if(counter_top > _num_output_pixels) _num_output_pixels = counter_top;
        this->_octree_keys.push_back(octree_keys);       
    }
}

template <typename Dtype>
void OctOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    Dtype* output_features = top[0]->mutable_cpu_data();
    const int batch_size = top[0]->shape(0);
    const int channels = top[0]->shape(1);
    memset(output_features, 0, sizeof(Dtype)*top[0]->count());

    for(int bt=0; bt<batch_size; bt++){

        for(typename GeneralOctree<int>::iterator it = this->_octree_keys[bt].begin(); it != this->_octree_keys[bt].end(); it++){

            int l = GeneralOctree<int>::compute_level(it->first)-_min_level; 
            string key_layer_name = this->layer_param_.oct_output_param().key_layer(l);
            boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
            boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);   

            const int num_input_pixels =  bottom[2*l]->shape(2);
            const Dtype* input_features = bottom[2*l]->cpu_data() + bt * channels * num_input_pixels;
            
            
            for(int ch=0; ch<channels; ch++){
                output_features[bt * channels * _num_output_pixels + ch * _num_output_pixels + it->second] =
                    input_features[ch * num_input_pixels + l_ptr->get_keys_octree(bt).get_value(it->first)];
            }
        }
    }
}

template <typename Dtype>
void OctOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    LOG(FATAL) << "Backward not implemented.";

}

INSTANTIATE_CLASS(OctOutputLayer);
REGISTER_LAYER_CLASS(OctOutput);

}  // namespace caffe
