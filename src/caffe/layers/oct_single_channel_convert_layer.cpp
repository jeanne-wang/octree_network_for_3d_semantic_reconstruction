#include "caffe/layers/oct_single_channel_convert_layer.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void OctSingleChannelConvertLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    _done_initial_reshape = false; 
    _min_level = this->layer_param_.oct_convert_param().min_level();
}

template <typename Dtype>
void OctSingleChannelConvertLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){


    _batch_size = bottom[0]->shape(0);
    
    std::vector<int> values_shape;

    if(!_done_initial_reshape){

        values_shape.push_back(_batch_size); 
        values_shape.push_back(1);

        _done_initial_reshape = true;

    }else{

        
        int num_elements = batch_octree_convert(bottom);
        values_shape.push_back(_batch_size); 
        values_shape.push_back(num_elements);

    }

    top[0]->Reshape(values_shape);
}

template <typename Dtype>
void OctSingleChannelConvertLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    this->_octree_keys.clear();

    int num_elements = top[0]->shape(1);

    Dtype* top_values = top[0]->mutable_cpu_data();
    memset(top_values, 0, sizeof(Dtype) * top[0]->count());

    for(int bt=0; bt< _batch_size; bt++)
    {
        GeneralOctree<int> octree_keys;
        int counter = 0;
        for(typename GeneralOctree<Dtype>::iterator it=_batch_octrees[bt].begin(); it!=_batch_octrees[bt].end(); it++)
        {
            
            top_values[bt*num_elements + counter] = (it->second);
            octree_keys.add_element(it->first, counter);
            counter++;
        }


        this->_octree_keys.push_back(octree_keys);
    }
}

template <typename Dtype>
void OctSingleChannelConvertLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
}

template <typename Dtype>
int OctSingleChannelConvertLayer<Dtype>::batch_octree_convert(const vector<Blob<Dtype>*>& bottom)
{
   
    int num_elements = 0;
    _batch_octrees.clear();

    CHECK_EQ(bottom[0]->shape(1), bottom[0]->shape(2))
        <<"dense input must be square.";
    CHECK_EQ(bottom[0]->shape(2), bottom[0]->shape(3))
        <<"dense input must be square.";

    int resolution = bottom[0]->shape(1);
    const Dtype* bottom_data = bottom[0]->cpu_data();
    for(int bt=0; bt<_batch_size; bt++)
    {
        int len = 0;
       
        GeneralOctree<Dtype> tree;
        GeneralVoxelGrid<Dtype> vg(resolution, resolution,resolution,  bottom_data + bt * bottom[0]->count(1));
       
        tree.from_voxel_grid(vg, _min_level);
        
        
        _batch_octrees.push_back(tree);
        len = tree.num_elements();
        if(len > num_elements) num_elements = len;
    }
    return num_elements;
}


INSTANTIATE_CLASS(OctSingleChannelConvertLayer);
REGISTER_LAYER_CLASS(OctSingleChannelConvert);

}  // namespace caffe
