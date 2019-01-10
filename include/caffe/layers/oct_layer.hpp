#ifndef CAFFE_OCT_LAYER_HPP_
#define CAFFE_OCT_LAYER_HPP_

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "image_tree_tools/octree.h"

namespace caffe {

template <typename Dtype>
class OctLayer : public Layer<Dtype> {

public:
    
    explicit OctLayer(const LayerParameter& param)
        : Layer<Dtype>(param) {}

    //TODO: make these references constant
    GeneralOctree<int>& get_keys_octree(int batch_ind){
        return _octree_keys[batch_ind];
    }

    int get_current_batch_size(){
        return _octree_keys.size();
    }

    int get_level() {return _level;}

    void calc_neighbors(){

        int batch_size = neighbors_.shape(0);
        int num_pixels = neighbors_.shape(1);

        Dtype* neighbors_data = neighbors_.mutable_cpu_data();
        Dtype* neighbor_of_data = neighbor_of_.mutable_cpu_data();
        caffe_set<Dtype>(neighbor_of_.count(), -1, neighbor_of_data);

        for(int bt = 0; bt < batch_size; bt++){

            for(typename GeneralOctree<int>::iterator it=_octree_keys[bt].begin(); it!=_octree_keys[bt].end(); it++){
                
                unsigned int key= it->first;
                vector<unsigned int> neighbors = _octree_keys[bt].get_neighbor_keys(key, nbh_size_);
                int num_neighbors = neighbors.size();

                for(int el = 0; el < num_neighbors; el++){

                    if(neighbors[el] != GeneralOctree<int>::INVALID_KEY()){
                        
                        unsigned int nbh_key = neighbors[el];
                        int nbh_pos = _octree_keys[bt].get_value(nbh_key);
                        neighbors_data[(bt * num_pixels + it->second) * num_neighbors + el] =  nbh_pos;
                        neighbor_of_data[(bt * num_pixels + nbh_pos) * num_neighbors + el] = it->second;
                      
                    }else{
                       neighbors_data[(bt * num_pixels + it->second) * num_neighbors + el]  = -1;
                    }
                }
            }
        }
    }

protected:

    std::vector<GeneralOctree<int> > _octree_keys;
    int _level;

    // for implementation on GPU
    Blob<Dtype> neighbors_; // int
    Blob<Dtype> neighbor_of_; //int
    Blob<Dtype> num_; // int  //num_[i]: number of octree cells in the ith octree in the batch
    int nbh_size_;

};

}  // namespace caffe

#endif  // CAFFE_OCT_LAYER_HPP_
