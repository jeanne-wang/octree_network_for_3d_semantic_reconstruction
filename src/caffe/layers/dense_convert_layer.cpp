#include "caffe/layers/dense_convert_layer.hpp"

#include "caffe/net.hpp"
#include "caffe/layers/oct_layer.hpp"
#include "image_tree_tools/octree.h"
#include "image_tree_tools/zindex.h"
#include "image_tree_tools/common_util.h"
namespace caffe {

using namespace std;

template <typename Dtype>
void DenseConvertLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // should be read from parameters
    max_level_ = this->layer_param_.dense_convert_param().max_level();


}

template <typename Dtype>
void DenseConvertLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){


    int batch_size = bottom[0]->shape(0);
    int num_classes = bottom[0]->shape(1);
    int resolution = std::pow(2, max_level_);

    
    vector<int> shape;
    shape.push_back(batch_size);
    shape.push_back(num_classes);
    shape.push_back(resolution);
    shape.push_back(resolution);
    shape.push_back(resolution);
    top[0]->Reshape(shape);
}

template <typename Dtype>
void DenseConvertLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    const Dtype* input_data = bottom[0]->cpu_data();
    Dtype* output_data = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), Dtype(0.), output_data);

    int batch_size = bottom[0]->shape(0);
    int num_classes = bottom[0]->shape(1);
    int resolution = std::pow(2, max_level_);
    
    const string key_layer_name = this->layer_param_.dense_convert_param().key_layer();
    boost::shared_ptr<Layer<Dtype> > base_ptr = this->parent_net()->layer_by_name(key_layer_name);
    boost::shared_ptr<OctLayer<Dtype> > l_ptr = boost::dynamic_pointer_cast<OctLayer<Dtype> >(base_ptr);

    for(int bt = 0; bt < batch_size; bt++){

        for(typename GeneralOctree<int>::iterator it=l_ptr->get_keys_octree(bt).begin(); it!=l_ptr->get_keys_octree(bt).end(); it++){
            
            int level = GeneralOctree<int>::compute_level(it->first);

            unsigned int code = (it->first & ~((unsigned int)(1) << level * 3)) << (max_level_ - level) * 3;
            unsigned int x, y, z;
            inverse_morton_3d(x, y, z, code);
            int len = int(pow(2, max_level_ - level));
            for(int i=0; i < len; i++){
                for(int j=0; j < len; j++){
                    for(int k=0; k < len; k++){
                        for(int ch = 0; ch < num_classes; ch++){

                            int nbh_x = x+i;
                            int nbh_y = y+j;
                            int nbh_z = z+k;
                            int ind = (((bt * num_classes + ch) * resolution + nbh_x)* resolution + nbh_y) * resolution + nbh_z;
                            output_data[ind] = input_data[bt * bottom[0]->count(1) + ch * bottom[0]->count(2) + it->second];
                            //ret.set_element(ch, x + i, y + j, z + k, input_data[bt * bottom[0]->count(1) + ch * bottom[0]->count(2) + it->second]);
                        }
                    }
                }
            }
        }



    }
    

        


}

template <typename Dtype>
void DenseConvertLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){

}

INSTANTIATE_CLASS(DenseConvertLayer);
REGISTER_LAYER_CLASS(DenseConvert);

}  // namespace caffe
