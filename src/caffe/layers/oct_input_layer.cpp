#include "caffe/layers/oct_input_layer.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void OctInputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    _source  = this->layer_param_.oct_input_param().source();
    _preload_data = this->layer_param_.oct_input_param().preload_data();

    _num_classes = this->layer_param_.oct_input_param().num_classes();
    _batch_size = this->layer_param_.oct_input_param().batch_size();

    _model_counter = 0;
    _done_initial_reshape = false;

    load_data_from_disk();
}

template <typename Dtype>
void OctInputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
   

    std::vector<int> values_shape;

    if(!_done_initial_reshape){

        values_shape.push_back(_batch_size); 
        values_shape.push_back(_num_classes);
        values_shape.push_back(1);

        _done_initial_reshape = true;

    }else{

        std::vector<int> batch_elements;
        for(int bt=0; bt< _batch_size; bt++){
           
            batch_elements.push_back(_model_counter++);
            if(_model_counter == _file_names.size()) _model_counter = 0;
            
           
        }
        
        int num_elements = select_next_batch_models(batch_elements);
        values_shape.push_back(_batch_size); 
        values_shape.push_back(_num_classes);
        values_shape.push_back(num_elements);

    }

    top[0]->Reshape(values_shape);
}

template <typename Dtype>
void OctInputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    this->_octree_keys.clear();

    int num_elements = top[0]->shape(2);

    Dtype* top_values = top[0]->mutable_cpu_data();
    memset(top_values, 0, sizeof(Dtype) * top[0]->count());

    for(int bt=0; bt< _batch_size; bt++)
    {
        GeneralOctree<int> octree_keys;
        int counter = 0;
        for(typename MultiChannelOctree<Dtype>::iterator it=_batch_octrees[bt].begin(); it!=_batch_octrees[bt].end(); it++)
        {
            for(int ch = 0; ch <_num_classes; ch++){
                int top_index = (bt * _num_classes + ch) * num_elements + counter;
                top_values[top_index] = (it->second)[ch];
            }

            octree_keys.add_element(it->first, counter);
            counter++;
        }


        this->_octree_keys.push_back(octree_keys);
    }
}

template <typename Dtype>
void OctInputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}

template <typename Dtype>
int OctInputLayer<Dtype>::select_next_batch_models(vector<int> labels)
{
   
    int num_elements = 0;
    _batch_octrees.clear();

    for(int bt=0; bt<labels.size(); bt++)
    {
        int len = 0;
        if(_preload_data)
        {
            _batch_octrees.push_back(_octrees[labels[bt]]);
            len = _octrees[labels[bt]].num_elements();
        }
        else
        {
            MultiChannelOctree<Dtype> tree(_num_classes);
            tree.from_bin_file(_file_names[labels[bt]]);
            _batch_octrees.push_back(tree);
            len = tree.num_elements();
        }
        if(len > num_elements) num_elements = len;
    }
    return num_elements;
}


template <typename Dtype>
void OctInputLayer<Dtype>::load_data_from_disk(){

    std::cout << "Loading training data from disk..." << endl;
    
    std::ifstream infile(_source.c_str());
    std::string name;
    int counter = 0;
    while(infile >> name){

        _file_names.push_back(name);
        if(_preload_data){
            
            MultiChannelOctree<Dtype> tree(_num_classes);
            tree.from_bin_file(name);
            _octrees.push_back(tree);
            std::cout << "read octree from "<< name << endl;
        }
        counter++;
    }

    std::cout << "Done." << std::endl;
}

INSTANTIATE_CLASS(OctInputLayer);
REGISTER_LAYER_CLASS(OctInput);

}  // namespace caffe
