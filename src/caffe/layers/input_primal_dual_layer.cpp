#include "caffe/layers/input_primal_dual_layer.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void InputPrimalDualLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    
    // read parameters
    batch_size_ = this->layer_param_.input_primal_dual_param().batch_size();
    out_height_ = this->layer_param_.input_primal_dual_param().height();
    out_width_ = this->layer_param_.input_primal_dual_param().width();
    out_depth_ = this->layer_param_.input_primal_dual_param().depth();
    num_classes_ = this->layer_param_.input_primal_dual_param().num_classes();
}

template <typename Dtype>
void InputPrimalDualLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    vector<int> top_shape(5);
    top_shape[0] = batch_size_;
    top_shape[1] = num_classes_;
    top_shape[2] = out_height_;
    top_shape[3] = out_width_;
    top_shape[4] = out_depth_;
    top[0]->Reshape(top_shape); // u
    top[1]->Reshape(top_shape); // u_

    top_shape[1] *= 3;
    top[2]->Reshape(top_shape); // m

    vector<int> lag_shape(4);
    lag_shape[0] = batch_size_;
    lag_shape[1] = out_height_;
    lag_shape[2] = out_width_;
    lag_shape[3] = out_depth_;
    top[3]->Reshape(lag_shape);


    
    
}

template <typename Dtype>
void InputPrimalDualLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
void InputPrimalDualLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}


INSTANTIATE_CLASS(InputPrimalDualLayer);
REGISTER_LAYER_CLASS(InputPrimalDual);

}  // namespace caffe
