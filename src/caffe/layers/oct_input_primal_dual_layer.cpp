#include "caffe/layers/oct_input_primal_dual_layer.hpp"
#include "image_tree_tools/octree.h"

namespace caffe {

using namespace std;

template <typename Dtype>
void OctInputPrimalDualLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    
    // read parameters
    batch_size_ = this->layer_param_.oct_input_primal_dual_param().batch_size();
    out_height_ = this->layer_param_.oct_input_primal_dual_param().height();
    out_width_ = this->layer_param_.oct_input_primal_dual_param().width();
    out_depth_ = this->layer_param_.oct_input_primal_dual_param().depth();
    num_classes_ = this->layer_param_.oct_input_primal_dual_param().num_classes();
    
    this->nbh_size_ = this->layer_param_.oct_input_primal_dual_param().nbh_size();
    this->_level = ceil(log(out_height_) / log(2));
    this->_octree_keys.clear();

    for(int bt=0; bt<batch_size_; bt++)
    {
        GeneralOctree<int> octree_keys;
        for(int x=0; x<out_height_; x++)
        {
            for(int y=0; y<out_width_; y++)
            {
                for(int z=0; z<out_depth_; z++)
                {
                    OctreeCoord c;
                    c.x = x; c.y = y; c.z = z; c.l = this->_level;
                    unsigned int key = GeneralOctree<int>::compute_key(c);
                    octree_keys.add_element(key, x*out_width_*out_depth_+ y*out_depth_ + z);
                }
            }
        }
        this->_octree_keys.push_back(octree_keys);
    }


    /****** newly added for efficient conv in GPU later***********/
    // num_[i]: number of octree cells in the ith octree in the batch
    int num_output_pixels = out_height_ * out_depth_ * out_width_;
    vector<int> num_shape(1);
    num_shape[0] = batch_size_;
    this->num_.Reshape(num_shape);
    caffe_set<Dtype>(batch_size_, num_output_pixels, this->num_.mutable_cpu_data());

    vector<int> nbh_shape(3);
    nbh_shape[0] = batch_size_;
    nbh_shape[1] = num_output_pixels;
    nbh_shape[2] = this->nbh_size_ * this->nbh_size_ * this->nbh_size_;
    this->neighbors_.Reshape(nbh_shape);
    this->neighbor_of_.Reshape(nbh_shape);
    this->calc_neighbors();
   
}

template <typename Dtype>
void OctInputPrimalDualLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
    

    vector<int> top_shape(3);
    top_shape[0] = batch_size_;
    top_shape[1] = num_classes_;
    top_shape[2] = out_height_ * out_width_ * out_depth_;
    top[0]->Reshape(top_shape); // u
    top[1]->Reshape(top_shape); // u_

    top_shape[1] *= 3;
    top[2]->Reshape(top_shape); // m

    top_shape[1] = 1; 
    top[3]->Reshape(top_shape); // l

    top[4]->ReshapeLike(this->num_); 
    top[5]->ReshapeLike(this->neighbors_);
    top[6]->ReshapeLike(this->neighbor_of_);

    top[4]->ShareData(this->num_);
    top[5]->ShareData(this->neighbors_);
    top[6]->ShareData(this->neighbor_of_);
    
}

template <typename Dtype>
void OctInputPrimalDualLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    
    Dtype* u_data = top[0]->mutable_cpu_data();
    Dtype* u__data = top[1]->mutable_cpu_data();
    Dtype* m_data = top[2]->mutable_cpu_data();
    Dtype* l_data = top[3]->mutable_cpu_data();

    caffe_set<Dtype>(top[0]->count(), Dtype(1./num_classes_), u_data);
    caffe_set<Dtype>(top[1]->count(), Dtype(1./num_classes_), u__data);
    caffe_set<Dtype>(top[2]->count(), Dtype(0.), m_data);
    caffe_set<Dtype>(top[3]->count(), Dtype(0.), l_data);
}

template <typename Dtype>
void OctInputPrimalDualLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}


INSTANTIATE_CLASS(OctInputPrimalDualLayer);
REGISTER_LAYER_CLASS(OctInputPrimalDual);

}  // namespace caffe
