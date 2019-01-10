#include <boost/program_options.hpp>
#include <iostream>
#include <tr1/unordered_map>
#include "image_tree_tools/multi_channel_voxel_grid.h"
#include "image_tree_tools/multi_channel_octree.h"
#include "image_tree_tools/common_util.h"


std::string input_file, output_file;
int min_level;
int max_level;

int register_cmd_options(int argc, char* argv[]) {
    try {
        boost::program_options::options_description desc("Options");
        desc.add_options()
            ("help,h", "Show help")
            ("input,i", boost::program_options::value<std::string>(&input_file)->required(), "Input file name for conversion")
            ("output,o", boost::program_options::value<std::string>(&output_file)->required(), "Output file name for conversion")
            ("min_level", boost::program_options::value<int>(&min_level)->required(), "Minimum octree level")
            ("max_level", boost::program_options::value<int>(&max_level)->required(), "Maximum octree level")
        ;

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

        if ( vm.count("help") ) {
            std::cout << desc << std::endl;
        } else if ( !vm.count("input") || !vm.count("output") || !vm.count("min_level") || !vm.count("max_level")) {
            std::cout << desc << std::endl;
            return -1;
        }
        
        input_file = vm["input"].as<std::string>();
        output_file = vm["output"].as<std::string>();
        min_level = vm["min_level"].as<int>();
        max_level = vm["max_level"].as<int>();
    } catch( boost::program_options::required_option& e ) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        return -1;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    if ( !register_cmd_options(argc, argv) ) {

        std::cout << "Input file: " << input_file << std::endl;
        std::cout << "Output file: " << output_file << std::endl;
        std::cout << "Minimum level: " << min_level << std::endl;
        std::cout << "Maximum level: " << max_level << std::endl;

        std::string input_ext = get_file_extension(input_file);

        int ignore_label = 35;
        int resolution = pow(2, max_level);
        
        
        if(input_ext == "dat"){
            // convert from voxel grid to octree
            std::tr1::unordered_map<int, string> crop_mode;
            crop_mode[0] = "00";
            crop_mode[1] = "01";
            crop_mode[2] = "10";
            crop_mode[3] = "11";
            for(int i = 0; i < 4; i++){

                float default_value = ignore_label;
                GeneralVoxelGrid<float> vg(resolution, resolution,resolution, default_value);
                vg.read(input_file, i);

                GeneralOctree<float> octree(max_level);            
                octree.from_voxel_grid(vg, min_level);
                std::cout << "grid to octree conversion finished,  there are "<< octree.num_elements() <<" octree cells\n";
                octree.to_file(output_file + crop_mode[i] + ".ot");
                
            }
        }else if(input_ext =="ot"){
            
            GeneralOctree<float> octree(max_level);  
            octree.from_file(input_file);
           
        
            GeneralVoxelGrid<float> vg = octree.to_voxel_grid();
            std::cout << "converted voxel grid is of height " << vg.height()<<", width "<< vg.width() 
                    << ", depth "<< vg.depth() <<std::endl;
            vg.write(output_file);
           
        }

    }
    return 0;
}
