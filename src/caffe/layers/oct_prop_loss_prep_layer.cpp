#include "caffe/layers/oct_prop_loss_prep_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/layers/oct_layer.hpp"
#include "image_tree_tools/octree.h"
#include "caffe/util/math_functions.hpp"

#include "image_tree_tools/oct_prop_util.h"
namespace caffe {

using namespace std;

template <typename Dtype>
void OctPropLossPrepLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void OctPropLossPrepLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    CHECK_EQ(bottom[0]->num_axes(), 3)
        << "The number of dimensions of octree features should be 3.";

    int batch_size = bottom[0]->shape(0);
    int num_pixels = bottom[0]->shape(2);

    vector<int> output_shape;
    output_shape.push_back(batch_size);
    output_shape.push_back(1);
    output_shape.push_back(num_pixels);
   
    top[0]->Reshape(output_shape);
}

template <typename Dtype>
void OctPropLossPrepLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    int batch_size = bottom[0]->shape(0);
    int num_pixels = bottom[0]->shape(2);


    const string gt_key_layer_name = this->layer_param_.oct_prop_loss_prep_param().gt_key_layer();
    const string pr_key_layer_name = this->layer_param_.oct_prop_loss_prep_param().pr_key_layer();
    const bool use_voxel_grid = this->layer_param_.oct_prop_loss_prep_param().use_voxel_grid();

    shared_ptr<Layer<Dtype> > gt_raw_ptr = this->parent_net()->layer_by_name(gt_key_layer_name);
    shared_ptr<Layer<Dtype> > pr_raw_ptr = this->parent_net()->layer_by_name(pr_key_layer_name);

    shared_ptr<OctLayer<Dtype> > gt_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(gt_raw_ptr);
    shared_ptr<OctLayer<Dtype> > pr_key_layer = boost::dynamic_pointer_cast<OctLayer<Dtype> >(pr_raw_ptr);


    Dtype* output_classification = top[0]->mutable_cpu_data();
    caffe_set<Dtype>(top[0]->count(), (Dtype)PROP_IGNORE_CLASS, output_classification);

    for(int bt = 0; bt<batch_size; bt++){
       
        GeneralOctree<int> &pr_keys_octree = pr_key_layer->get_keys_octree(bt);
        GeneralOctree<int> &gt_keys_octree = gt_key_layer->get_keys_octree(bt);

        for(GeneralOctree<int>::iterator it=pr_keys_octree.begin(); it!=pr_keys_octree.end(); it++){
            
            int gt_value;
            int gt_ind = gt_keys_octree.get_value(it->first, use_voxel_grid);
            if(gt_ind != -1) gt_value = 0; // prop_false
            else gt_value = 1; //prop_true

            output_classification[bt * num_pixels + it->second] = gt_value;
            
        }
        
    }
}

template <typename Dtype>
void OctPropLossPrepLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(OctPropLossPrepLayer);
REGISTER_LAYER_CLASS(OctPropLossPrep);

}  // namespace caffe
