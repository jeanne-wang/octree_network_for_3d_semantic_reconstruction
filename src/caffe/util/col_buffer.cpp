#include <boost/thread.hpp>
#include "caffe/util/col_buffer.hpp"


namespace caffe{
	// Make sure each thread can have different values.
	static boost::thread_specific_ptr<ColBuffer> thread_col_buffer_instance_;

	ColBuffer& ColBuffer::Get(){

		if (!thread_col_buffer_instance_.get()){

			thread_col_buffer_instance_.reset(new ColBuffer());
		}
		
		return *(thread_col_buffer_instance_.get());
	}
}
