#include "caffe/net.hpp"
#include "caffe/layers/oct_prop_layer.hpp"

#include "image_tree_tools/octree.h"


namespace caffe {

using namespace std;

template <typename Dtype>
void OctPropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    _done_initial_reshape = false;
}

template <typename Dtype>
void OctPropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // bottom[0] u
    // bottom[1] u_
    // bottom[2] m
	// bottom[3] l
    // bottom[4] gt or pred

	// check the number of dimension
    for(int i = 0; i < bottom.size(); i++){
        CHECK_EQ(bottom[i]->num_axes(), 3)
            <<"All input blob must have dimension of 3. ";
    }


	// check the batch size
    for(int i = 1; i < bottom.size(); i++){
        CHECK_EQ(bottom[i]->shape(0), bottom[0]->shape(0))
            <<"All variables must have the same number data. ";
    }

    // check input voxels
    for(int i = 1; i < bottom.size(); i++){
        CHECK_EQ(bottom[i]->shape(2), bottom[0]->shape(2))
            << "All variables must have the same input voxels";
    }
	
    if(!_done_initial_reshape){
        _num_output_pixels = 1;
        _done_initial_reshape = true;
    }else{

        compute_pixel_propagation(bottom, top);
       // LOG(INFO) << _num_output_pixels << " cells are propagated to next level.";

    }

	if(!_num_output_pixels) _num_output_pixels = 1;

	for(int i = 0; i < bottom.size()-1; i++){
		vector<int> shape_features;
    	shape_features.push_back(bottom[i]->shape(0)); 
    	shape_features.push_back(bottom[i]->shape(1)); 
    	shape_features.push_back(_num_output_pixels);
     	top[i]->Reshape(shape_features);
	}
}

template <typename Dtype>
void OctPropLayer<Dtype>::compute_pixel_propagation(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

    this->_octree_keys.clear();
    _num_output_pixels = 0;

    const Dtype* input_values = bottom[4]->cpu_data();
    const int batch_size = bottom[0]->shape(0);
    const int num_input_pixels = bottom[0]->shape(2);

    std::string key_layer_name = this->layer_param_.oct_prop_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);
    
    for(int bt=0; bt<batch_size; bt++){

        int counter_top = 0;
        GeneralOctree<int> octree_keys;

        for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(bt).begin(); it!=l_ptr->get_keys_octree(bt).end(); it++){

            int v;
            OctPropParameter_PropagationMode prop_mode = this->layer_param().oct_prop_param().prop_mode();

            if(prop_mode == OctPropParameter_PropagationMode_PROP_PRED){	

                if(input_values[bt * num_input_pixels + it->second] >= 0.5){
                    v = 1;
				}else{
					v = 0;
				}

			}else if(prop_mode == OctPropParameter_PropagationMode_PROP_KNOWN){
				v = input_values[bt * num_input_pixels + it->second];
			}

            if(v == 1){
                octree_keys.add_element(it->first, counter_top);
                counter_top++;
            }
        }  

        if(counter_top > _num_output_pixels) _num_output_pixels = counter_top;
        this->_octree_keys.push_back(octree_keys);
    }
}

template <typename Dtype>
void OctPropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	for(int i = 0; i < bottom.size()-1; i++){
		const Dtype* input_features = bottom[i]->cpu_data();
		Dtype* output_features = top[i]->mutable_cpu_data();

		const int batch_size = bottom[i]->shape(0);
        const int channels = bottom[i]->shape(1);
        const int num_input_pixels = bottom[i]->shape(2);

        memset(output_features, 0, sizeof(Dtype)*batch_size*channels*_num_output_pixels);

        std::string key_layer_name = this->layer_param_.oct_prop_param().key_layer();
        boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
        boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);

        for(int bt=0; bt<batch_size; bt++){
            for(typename GeneralOctree<int>::iterator it=this->_octree_keys[bt].begin(); it!=this->_octree_keys[bt].end(); it++){
                for(int ch=0; ch<channels; ch++){
                    output_features[bt * channels * _num_output_pixels + ch * _num_output_pixels + it->second] =
                        input_features[bt * channels * num_input_pixels + ch * num_input_pixels + l_ptr->get_keys_octree(bt).get_value(it->first)];
                }
            }
        }
	}
}

template <typename Dtype>
void OctPropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


	for(int i = 0; i < bottom.size()-1; i++){
		const int batch_size = bottom[i]->shape(0);
        const int channels = bottom[i]->shape(1);
        const int num_input_pixels = bottom[i]->shape(2);

        const Dtype* top_diff = top[i]->cpu_diff();
        Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

        memset(bottom_diff, 0, sizeof(Dtype)*batch_size*channels*num_input_pixels);

        std::string key_layer_name = this->layer_param_.oct_prop_param().key_layer();
        boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
        boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);

        for(int bt=0; bt<batch_size; bt++){
            for(typename GeneralOctree<int>::iterator it=this->_octree_keys[bt].begin(); it!=this->_octree_keys[bt].end(); it++){
                for(int ch=0; ch<channels; ch++){
                    bottom_diff[bt * channels * num_input_pixels + ch * num_input_pixels + l_ptr->get_keys_octree(bt).get_value(it->first)] +=
                        top_diff[bt * channels * _num_output_pixels + ch * _num_output_pixels + it->second];
                }
            }
        }

	}	
}

INSTANTIATE_CLASS(OctPropLayer);
REGISTER_LAYER_CLASS(OctProp);

}  // namespace caffe
