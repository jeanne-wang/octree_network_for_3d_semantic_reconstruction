#include "caffe/layers/oct_level_sampling_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/oct_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include<queue>
namespace caffe {

using namespace std;

template <typename Dtype>
void OctLevelSamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    input_max_level_ = this->layer_param_.oct_level_sampling_param().input_max_level();
}

template <typename Dtype>
void OctLevelSamplingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // bottom[0] octree input data
    // bottom[1] ref data
    CHECK_EQ(bottom[0]->num_axes(), 3)
        << "The number of dimensions of octree features should be 3.";
    CHECK_EQ(bottom[1]->num_axes(), 3)
        << "The number of dimensions of octree features should be 3.";
   
    top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void OctLevelSamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    int batch_size = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int num_input_pixels = bottom[0]->shape(2);
    int num_output_pixels = top[0]->shape(2);

    const string input_key_layer_name = this->layer_param_.oct_level_sampling_param().input_key_layer();
    const string ref_key_layer_name = this->layer_param_.oct_level_sampling_param().ref_key_layer();

    shared_ptr<Layer<Dtype> > input_raw_ptr = this->parent_net()->layer_by_name(input_key_layer_name);
    shared_ptr<Layer<Dtype> > ref_raw_ptr = this->parent_net()->layer_by_name(ref_key_layer_name);

    shared_ptr<OctLayer<Dtype> > input_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(input_raw_ptr);
    shared_ptr<OctLayer<Dtype> > ref_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(ref_raw_ptr);


    const Dtype* input = bottom[0]->cpu_data();
    Dtype* output = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), Dtype(0.), output);

    for(int bt = 0; bt<batch_size; bt++){
       
        GeneralOctree<int> &ref_keys_octree = ref_key_layer->get_keys_octree(bt);
        GeneralOctree<int> &input_keys_octree = input_key_layer->get_keys_octree(bt);

        for(GeneralOctree<int>::iterator it=ref_keys_octree.begin(); it!=ref_keys_octree.end(); it++){           

            for (int ch = 0; ch < channels; ch++){
                output[(bt * channels + ch) * num_output_pixels + it->second]
                    = get_octree_cell_value(input_keys_octree, input + (bt * channels + ch) * num_input_pixels, it->first);
            }
        }
        
    }
}

template <typename Dtype>
void OctLevelSamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


template <typename Dtype>
Dtype OctLevelSamplingLayer<Dtype>::get_octree_cell_value( GeneralOctree<int> &input_keys_octree, 
    const Dtype* input, unsigned int key){


    int level = 0;
    Dtype res = 0;

    // breadth first traversel of the octree and only add the value of the leaf nodes
    std::queue<unsigned int> nodeQueue;
    nodeQueue.push(key);
    while(!nodeQueue.empty()){

        int num = nodeQueue.size();
        for(int n = 0; n < num; n++){

            unsigned int cur_key = nodeQueue.front();
            nodeQueue.pop();
            int pos = input_keys_octree.get_value(cur_key, true);
            if(pos != -1){
                Dtype avg_weight = std::pow(8, level);
                res += input[pos]/avg_weight;
            }else{
                int key_level = GeneralOctree<int>::compute_level(cur_key);
                const string input_key_layer_name = this->layer_param_.oct_level_sampling_param().input_key_layer();
                const string ref_key_layer_name = this->layer_param_.oct_level_sampling_param().ref_key_layer();
                CHECK_NE(key_level, this->input_max_level_) 
                    << "It is impossible for cell in the finest level in " 
                    << ref_key_layer_name << " to not be found in " << input_key_layer_name;

                cur_key <<= 3;
                for(int i = 0; i < 8; i++){
                    nodeQueue.push(cur_key | i);
                }

            }

        }
        

        level++;
    }

    return res;
    
}

/*template <typename Dtype>
Dtype OctLevelSamplingLayer<Dtype>::get_octree_cell_value( GeneralOctree<int> &input_keys_octree, 
    const Dtype* input, unsigned int key){


    int pos = input_keys_octree.get_value(key, true);
    if(pos != -1) return input[pos];

    int level = GeneralOctree<int>::compute_level(key);
    const string input_key_layer_name = this->layer_param_.oct_level_sampling_param().input_key_layer();
    const string ref_key_layer_name = this->layer_param_.oct_level_sampling_param().ref_key_layer();
    
    CHECK_NE(level, this->input_max_level_) 
        << "It is impossible for cell in the finest level in " 
        << ref_key_layer_name << " to not be found in " << input_key_layer_name;
   
    Dtype res = 0;
    key <<= 3;
    for(int i = 0; i < 8; i++){
        unsigned int inner_key = key | i;

        res += get_octree_cell_value(input_keys_octree, inner_key);
    }
    res =  res / Dtype(8.);
    return res;
}*/

INSTANTIATE_CLASS(OctLevelSamplingLayer);
REGISTER_LAYER_CLASS(OctLevelSampling);

}  // namespace caffe
