#include <boost/program_options.hpp>
#include <iostream>
#include "image_tree_tools/multi_channel_voxel_grid.h"
#include "image_tree_tools/multi_channel_octree.h"
#include "image_tree_tools/common_util.h"
#include <vector>
#include <utility>
#include <sstream>
#include <assert.h>
std::string input_file, output_file;
int min_level;
int max_level;
int num_classes;

int register_cmd_options(int argc, char* argv[]) {
    try {
        boost::program_options::options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i", boost::program_options::value<std::string>(&input_file)->required(), "Input file name for conversion")
            ("output,o", boost::program_options::value<std::string>(&output_file)->required(), "Output file name for conversion")
            ("min_level", boost::program_options::value<int>(&min_level)->required(), "Minimum octree level")
            ("max_level", boost::program_options::value<int>(&max_level)->required(), "Maximum octree level")
            ("num_classes", boost::program_options::value<int>(&num_classes)->required(), "number of classes in multi channel octree")
        ;

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

        if ( vm.count("help") ) {
            std::cout << desc << std::endl;
        } else if ( !vm.count("input") || !vm.count("output") || !vm.count("min_level") || !vm.count("max_level") 
            || !vm.count("num_classes")) {
            std::cout << desc << std::endl;
            return -1;
        }
        
        input_file = vm["input"].as<std::string>();
        output_file = vm["output"].as<std::string>();
        min_level = vm["min_level"].as<int>();
        max_level = vm["max_level"].as<int>();
        num_classes = vm["num_classes"].as<int>();
    } catch( boost::program_options::required_option& e ) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        return -1;
    }
    return 0;
}
std::string num2string( int number ){
    std::ostringstream ss;
    ss << number;
    return ss.str();
}
bool is_byteorder_big_endian(){
    int num = 1;
    if(*(char *)&num == 1){
        return false;
    }else{
        return true;
    }
}

std::pair<int, int> determine_crop_start(int crop_ind, int num_crop, int resolution, int dimension){
    assert(crop_ind < num_crop);
    int crop_start;
    if(crop_ind == 0) return std::make_pair(0, std::min(resolution, dimension));
    else if(crop_ind == num_crop-1){
        crop_start = dimension-resolution;
        assert(crop_start >= 0);
        return std::make_pair(crop_start, dimension);
    }else{
        int res = dimension-2*resolution;
        int pad = (resolution-res)/2;
        assert(resolution-pad>=0);
        assert(2*resolution-pad < dimension);
        return std::make_pair(resolution-pad, 2*resolution-pad);

    }


}
int main(int argc, char* argv[]) {
    if ( !register_cmd_options(argc, argv) ) {

        std::cout << "Input file: " << input_file << std::endl;
        std::cout << "Output file: " << output_file << std::endl;
        std::cout << "Minimum level: " << min_level << std::endl;
        std::cout << "Maximum level: " << max_level << std::endl;
        std::cout << "number of classes: " << num_classes << std::endl;

        std::string input_ext = get_file_extension(input_file);


        int resolution = pow(2, max_level);
        
        
        if(input_ext == "dat"){

            // read scene size info
            uint8_t version;
            uint8_t is_big_endian;
            uint8_t uint_size;
            uint32_t elem_size;
            uint32_t num_classes_;
            uint32_t height;
            uint32_t width;
            uint32_t depth;

            std::ifstream binaryIo(input_file.c_str(), std::ios::binary);
            binaryIo.read((char *)(&version), sizeof(version));
            CHECK_EQ(version, 1) 
                <<" version must be 1. ";
            binaryIo.read((char *)(&is_big_endian), sizeof(is_big_endian));
            CHECK((is_big_endian == 1 && is_byteorder_big_endian()) || (is_big_endian == 0 && !is_byteorder_big_endian()))
                << "byteorder must be consistent. ";
            binaryIo.read((char *)(&uint_size), sizeof(uint_size));
            CHECK_EQ(uint_size, 4)
                << " uint_size must be 4. ";

            binaryIo.read((char *)(&elem_size), sizeof(elem_size));
            CHECK(elem_size == 4 || elem_size == 8)
                << "elem size must be 4 or 8. ";
            binaryIo.read((char *)(&num_classes_), sizeof(num_classes_));
            binaryIo.read((char *)(&height), sizeof(height));
            binaryIo.read((char *)(&width), sizeof(width));
            binaryIo.read((char *)(&depth), sizeof(depth));
            binaryIo.close();
            //

            std::vector<std::pair<int,int> > crop;
            crop.resize(3);
            crop[2] = std::make_pair(0, std::min(resolution, (int)depth));
            int num_h = height/resolution+1;
            int num_w = width/resolution+1;
            assert(num_h <= 3);
            assert(num_w <= 3);
            int count = 0;
            for (int x = 0; x < num_h; x++){
                for(int y = 0; y < num_w; y++){
                    float default_value = 1.0/float(num_classes_);
                    MultiChannelVoxelGrid<float> vg(num_classes, resolution, resolution,resolution, default_value);
                    crop[0] = determine_crop_start(x, num_h, resolution, height);
                    crop[1] = determine_crop_start(y, num_w, resolution, width);
                    
                    vg.read_groundtruth_probs(input_file, crop);
                    MultiChannelOctree<float> octree(num_classes, max_level);            
                    octree.from_voxel_grid(vg, min_level);
                    std::cout << "grid to octree conversion finished,  there are "<< octree.num_elements() <<" octree cells\n";
                    octree.to_bin_file(output_file + num2string(count) + ".ot");

                    count++;
                    
                }
            }
           
        }else if(input_ext =="ot"){
            
            MultiChannelOctree<float> octree(num_classes, max_level);  
            octree.from_bin_file(input_file);
           
        
            MultiChannelVoxelGrid<float> vg = octree.to_voxel_grid();
            std::cout << "converted voxel grid is of height " << vg.height()<<", width "<< vg.width() 
                    << ", depth "<< vg.depth() << ", and classes " << vg.num_classes()<<std::endl;
            vg.write_groundtruth_probs(output_file);
           
        }

    }
    return 0;
}
