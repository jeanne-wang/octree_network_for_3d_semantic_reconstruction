#include "caffe/layers/oct_input_primal_dual_layer.hpp"

namespace caffe {

template <typename Dtype>
void OctInputPrimalDualLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    
    Dtype* u_data = top[0]->mutable_gpu_data();
    Dtype* u__data = top[1]->mutable_gpu_data();
    Dtype* m_data = top[2]->mutable_gpu_data();
    Dtype* l_data = top[3]->mutable_gpu_data();

    caffe_gpu_set<Dtype>(top[0]->count(), Dtype(1./num_classes_), u_data);
    caffe_gpu_set<Dtype>(top[1]->count(), Dtype(1./num_classes_), u__data);
    caffe_gpu_set<Dtype>(top[2]->count(), Dtype(0.), m_data);
    caffe_gpu_set<Dtype>(top[3]->count(), Dtype(0.), l_data);

}

template <typename Dtype>
void OctInputPrimalDualLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        LOG(FATAL) << "Backward not implemented";
}


INSTANTIATE_LAYER_GPU_FUNCS(OctInputPrimalDualLayer);
}  // namespace caffe
