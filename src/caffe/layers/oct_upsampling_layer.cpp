#include "caffe/layers/oct_upsampling_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/net.hpp"

#include "image_tree_tools/octree.h"

namespace caffe {

template <typename Dtype>
void OctUpSamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    this->nbh_size_ = this->layer_param_.oct_upsampling_param().nbh_size();
}

template <typename Dtype>
void OctUpSamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    int num_bottoms = bottom.size();
    for(int i = 0; i < num_bottoms; i++){
        CHECK_EQ(bottom[i]->num_axes(), 3)
        << "input dimension must be 3";
    }

    for(int i = 1; i < num_bottoms; i++){
    	CHECK_EQ(bottom[i]->shape(0), bottom[0]->shape(0))
        << "All variables must have the same number of input data";
        CHECK_EQ(bottom[i]->shape(2), bottom[0]->shape(2))
        << "All variables must have the same number of voxels";
    }

    
        
	_batch_size = bottom[0]->shape(0);
    _num_input_pixels = bottom[0]->shape(2);

    _num_output_pixels = 8 * _num_input_pixels;
   

   	for(int i = 0; i < num_bottoms; i++){
   		vector<int> features_shape;
    	features_shape.push_back(_batch_size);
    	features_shape.push_back(bottom[i]->shape(1));
    	features_shape.push_back(_num_output_pixels);
    	top[i]->Reshape(features_shape);
   	}

    // newly added for efficient conv in GPU later by xiaojuan on Sep. 9, 2018
    vector<int> num_shape(1);
    num_shape[0] = _batch_size;
    this->num_.Reshape(num_shape);


    vector<int> nbh_shape(3);
    nbh_shape[0] = _batch_size;
    nbh_shape[1] = _num_output_pixels;
    nbh_shape[2] = this->nbh_size_ * this->nbh_size_ * this->nbh_size_;
    this->neighbors_.Reshape(nbh_shape);
    this->neighbor_of_.Reshape(nbh_shape);

    top[num_bottoms]->ReshapeLike(this->num_); 
    top[num_bottoms+1]->ReshapeLike(this->neighbors_);
    top[num_bottoms+2]->ReshapeLike(this->neighbor_of_);


}

template <typename Dtype>
void OctUpSamplingLayer<Dtype>::propagate_keys_cpu(){
	

	this->_octree_keys.clear();

    std::string key_layer_name = this->layer_param_.oct_upsampling_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);

    // newly added for efficient conv in GPU later by xiaojuan on Sep. 9, 2018
    Dtype* num_data = this->num_.mutable_cpu_data();
    //

    for (int bt = 0; bt < _batch_size; bt++){
        GeneralOctree<int> octree_keys;

        int output_counter = 0;
        for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(bt).begin(); it!=l_ptr->get_keys_octree(bt).end(); it++){       

            unsigned int key = it->first;
            for(int i=0; i<8; i++){

                unsigned int new_key = (key << 3) | i;
                octree_keys.add_element(new_key, output_counter);
                output_counter++;

            }
        }
        num_data[bt] = output_counter; // newly added to record the length for each octree in the batch.
        this->_octree_keys.push_back(octree_keys);
       
    }

    this->calc_neighbors();
}

template <typename Dtype>
void OctUpSamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	propagate_keys_cpu();

	std::string key_layer_name = this->layer_param_.oct_upsampling_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);

    int num_bottoms = bottom.size();
	for(int i = 0; i < num_bottoms; i++){
		const Dtype* input = bottom[i]->cpu_data();
		Dtype* output = top[i]->mutable_cpu_data();

		int channels = bottom[i]->shape(1);
		memset(output, 0, sizeof(Dtype)*top[i]->count());
		for(int n =0; n < _batch_size; n++){

            for(typename GeneralOctree<int>::iterator it=this->_octree_keys[n].begin(); it!=this->_octree_keys[n].end(); it++){
                
                unsigned int ori_key = (it->first) >> 3;
                for(int ch=0; ch<channels; ch++){
                    
                    output[n * channels * _num_output_pixels + ch * _num_output_pixels + it->second] =
                        input[n * channels * _num_input_pixels + ch * _num_input_pixels + l_ptr->get_keys_octree(n).get_value(ori_key)];
                }
            }
        }

	}

    // newly added
    top[num_bottoms]->ShareData(this->num_);
    top[num_bottoms+1]->ShareData(this->neighbors_);
    top[num_bottoms+2]->ShareData(this->neighbor_of_);

}

template <typename Dtype>
void OctUpSamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	std::string key_layer_name = this->layer_param_.oct_upsampling_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);

    int num_bottoms = bottom.size();
	for(int i = 0; i < num_bottoms; i++){
		
		const Dtype* top_diff = top[i]->cpu_diff();
		Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

		int channels = bottom[i]->shape(1);
		memset(bottom_diff, 0, sizeof(Dtype)* bottom[i]->count());
		for(int n =0; n < _batch_size; n++){

            for(typename GeneralOctree<int>::iterator it=this->_octree_keys[n].begin(); it!=this->_octree_keys[n].end(); it++){
                
                unsigned int ori_key = (it->first) >> 3;
                for(int ch=0; ch<channels; ch++){
                    
                    bottom_diff[n * channels * _num_input_pixels + ch * _num_input_pixels + l_ptr->get_keys_octree(n).get_value(ori_key)]
                    	+= top_diff[n * channels * _num_output_pixels + ch * _num_output_pixels + it->second];
                }
            }
        }

	}

}


INSTANTIATE_CLASS(OctUpSamplingLayer);
REGISTER_LAYER_CLASS(OctUpSampling);

}  // namespace caffe
