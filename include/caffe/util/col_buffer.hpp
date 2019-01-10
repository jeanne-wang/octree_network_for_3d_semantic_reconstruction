#ifndef CAFFE_UTIL_COL_BUFFER_HPP_
#define CAFFE_UTIL_COL_BUFFER_HPP_


#include "caffe/blob.hpp"


namespace caffe {

// A singleton class to hold global colbuffer and neighbor info for conv/deconv
class ColBuffer{

public:
		
	// Thread local context for ColBuffer. 
	static ColBuffer& Get();
	static Blob<float>& get_col_buffer(float) { return Get().col_buffer_; }
	static Blob<double>& get_col_buffer(double) { return Get().col_buffer_d_; }
	
	
protected:

	// col buffer is used as the temporary buffer of 
	// gemm in octree conv/deconv to save memory.
	Blob<float> col_buffer_;
	Blob<double> col_buffer_d_;

    


private:
		// The private constructor to avoid duplicate instantiation.
	ColBuffer() : col_buffer_(), col_buffer_d_(){}

};

}
#endif // CAFFE_UTIL_COL_BUFFER_HPP