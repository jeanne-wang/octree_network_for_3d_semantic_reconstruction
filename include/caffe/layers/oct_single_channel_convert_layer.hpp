#ifndef CAFFE_OCT_SINGLE_CHANNEL_CONVERT_LAYER_HPP_
#define CAFFE_OCT_SINGLE_CHANNEL_CONVERT_LAYER_HPP_

#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/oct_layer.hpp"
#include "image_tree_tools/octree.h"
#include "image_tree_tools/voxel_grid.h"
#include <vector>

namespace caffe {

template <typename Dtype>
class OctSingleChannelConvertLayer : public OctLayer<Dtype> {
public:
    explicit OctSingleChannelConvertLayer(const LayerParameter& param)
      : OctLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "OctSingleChannelConvert"; }

protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

private:
    int batch_octree_convert(const vector<Blob<Dtype>*>& bottom);

    std::vector<GeneralOctree<Dtype> > _batch_octrees;

    int _batch_size;
   
    bool _done_initial_reshape;
    int _min_level;
};

}  // namespace caffe

#endif  // CAFFE_OCT_SINGLE_CHANNEL_CONVERT_LAYER_HPP_
