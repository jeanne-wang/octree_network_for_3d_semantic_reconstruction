#ifndef CAFFE_OCT_INPUT_LAYER_HPP_
#define CAFFE_OCT_INPUT_LAYER_HPP_

#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/oct_layer.hpp"
#include "image_tree_tools/multi_channel_octree.h"
#include <vector>

namespace caffe {

template <typename Dtype>
class OctInputLayer : public OctLayer<Dtype> {
public:
    explicit OctInputLayer(const LayerParameter& param)
      : OctLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "OctInput"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:

    void load_data_from_disk();
    int select_next_batch_models(std::vector<int> labels);

    std::vector<MultiChannelOctree<Dtype> > _octrees;
    std::vector<MultiChannelOctree<Dtype> > _batch_octrees;
    
    std::string _source;
    std::vector<std::string> _file_names;


    int _num_classes;
    int _batch_size;
    bool _preload_data;
   
    bool _done_initial_reshape;
    int _model_counter;

};

}  // namespace caffe

#endif  // CAFFE_OCT_INPUT_LAYER_HPP_
