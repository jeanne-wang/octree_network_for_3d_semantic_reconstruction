#ifndef MULTI_CHANNEL_OCTREE_H_
#define MULTI_CHANNEL_OCTREE_H_

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>

#include <tr1/unordered_map>
#include <vector>
#include <math.h>
#include <glog/logging.h>

#include "zindex.h"
#include "multi_channel_voxel_grid.h"
#include "common_util.h"
#include "octree.h"


template <class VALUE>
class MultiChannelOctree{

public:
    typedef unsigned int KEY;

private:
    
    typedef std::tr1::unordered_map<KEY, vector<VALUE> > HashTable;
    HashTable _hash_table;
    int _max_level;
    int _num_channels;

public:

    static int MIN_LEVEL() { return 0; }
    static int MAX_LEVEL() { return (sizeof(KEY) * 8) / 3; }
    static KEY INVALID_KEY() { return 0; }
    static bool IS_VALID_COORD(const OctreeCoord& c){
        if( c.l > MAX_LEVEL() ||
            c.x < 0 || c.x >= (int(1) << c.l) ||
            c.y < 0 || c.y >= (int(1) << c.l) ||
            c.z < 0 || c.z >= (int(1) << c.l) ) return false;
        return true;
    }
    static bool IS_VALID_KEY(const KEY& key){
        if(key == INVALID_KEY()) return false;
        int lz = __builtin_clz(key) - 1;
        if(lz % 3) return false;
        return true;
    }

    MultiChannelOctree(int num_channels, int max_level = -1){
        _max_level = max_level;
        _num_channels = num_channels;
    }

    typedef typename HashTable::iterator iterator;
    typedef typename HashTable::const_iterator const_iterator;
    iterator begin() { return _hash_table.begin(); }
    iterator end() { return _hash_table.end(); }

    static int resolution_from_level(int level){
        return pow(2, level);
    }

    static int compute_level(const KEY& key){
        return (MAX_LEVEL() * 3 - __builtin_clz(key) + 1) / 3;
    }

    static KEY compute_key(const OctreeCoord& c){
      
        if(!IS_VALID_COORD(c)) return INVALID_KEY();
        return morton_3d(KEY(c.x), KEY(c.y), KEY(c.z)) | (KEY(1) << 3 * c.l);
    }

    static OctreeCoord compute_coord(KEY key){
        OctreeCoord c;
        c.l = compute_level(key);

        KEY x,y,z;
        inverse_morton_3d(x, y, z, key & ~(KEY(1) << c.l * 3));
        c.x = x;
        c.y = y;
        c.z = z;
        return c;
    }

    int num_elements() {return _hash_table.size();}

    int find_maxium(const VALUE* arr, int len){
        int ind = 0;
        VALUE maxium = arr[0];
        for(int i = 1; i < len; i++){
            if(arr[i] > maxium){
                ind = i;
                maxium = arr[i];
            }

        }

        return ind;
    }

    bool values_equal(const VALUE* arr1, const VALUE* arr2, int len){

        for(int i = 0; i < len; i++){
            if(arr1[i] != arr2[i]){
                return false;
            }
        }

        return true;
    }
    
    MultiChannelVoxelGrid<VALUE> to_voxel_grid(){
        int resolution = pow(2, _max_level);
        MultiChannelVoxelGrid<VALUE> ret(_num_channels, resolution, resolution, resolution);

        for(typename std::tr1::unordered_map<KEY, vector<VALUE> >::iterator iter = _hash_table.begin(); iter != _hash_table.end(); ++iter){
            int level = compute_level(iter->first);

            KEY code = (iter->first & ~(KEY(1) << level * 3)) << (_max_level - level) * 3;
            KEY x, y, z;
            inverse_morton_3d(x, y, z, code);
            int len = int(pow(2, _max_level - level));
            for(int i=0; i < len; i++){
                for(int j=0; j < len; j++){
                    for(int k=0; k < len; k++){
                        for(int ch = 0; ch < _num_channels; ch++){
                            ret.set_element(ch, x + i, y + j, z + k, (iter->second)[ch]);
                        }
                    }
                }
            }
         
        }
        return ret;
    }

    void from_voxel_grid(MultiChannelVoxelGrid<VALUE>& vg, int min_level){
      
        int level = log2((float)vg.depth());
        int dim = vg.depth();

        _max_level = level;
        CHECK_EQ(_num_channels, vg.num_classes())
            << "the number of channels in octree must equal to the number of classes in the grid.";


        KEY *keys_arr = new KEY[dim*dim*dim];
        VALUE *values_arr = new VALUE[dim*dim*dim*_num_channels];

        //initially fill the hash map
        for(unsigned int i=0; i<vg.height(); i++){
            for(unsigned int j=0; j<vg.width(); j++){
                for(unsigned int k=0; k<vg.depth(); k++){    
                    OctreeCoord crd;
                    crd.x = i; crd.y = j; crd.z = k; crd.l = level;
                    KEY key = compute_key(crd);
                    keys_arr[i*dim*dim + j*dim + k] = key;
                    for(unsigned int c = 0; c < _num_channels; c++){
                        VALUE val = vg.get_element(c, i, j, k);
                        values_arr[(i * dim * dim + j * dim + k) * _num_channels + c] = val;
                    }
                }
            }
        }

        while(level > min_level){

            int step = pow(2, _max_level - level + 1);
            for(unsigned int i=0; i<vg.height(); i+=step){
                
                for(unsigned int j=0; j<vg.width(); j+=step){
                    
                    for(unsigned int k=0; k<vg.depth(); k+=step){

                        KEY key = keys_arr[i*dim*dim + j*dim + k];
                        if(compute_level(key) == level){

                            int label = find_maxium(values_arr + (i * dim * dim + j * dim + k)*_num_channels, _num_channels);
                            int count = 0;

                            for(int ii=0; ii<2; ii++){
                                for(int jj=0; jj<2; jj++){
                                    for(int kk=0; kk<2; kk++){
                                        KEY comp_key = keys_arr[(i+ii*step/2)*dim*dim + (j+jj*step/2)*dim + k+kk*step/2];
                                        int comp_lev = compute_level(comp_key);
                                        if(comp_lev == level){
                                        
                                            int label_comp = find_maxium(values_arr + ((i+ii*step/2)*dim*dim + (j+jj*step/2)*dim + k+kk*step/2) *_num_channels,
                                                _num_channels);
                                
                                            if(label_comp == label) count++;
                                        }
                                    }
                                }
                            }

                            if(count==8 ){
                                KEY new_key = key >> 3;
                                vector<VALUE> avg_vals(_num_channels, 0);
                                for(int ii=0; ii<step; ii++){
                                    for(int jj=0; jj<step; jj++){
                                        for(int kk=0; kk<step; kk++){
                                            keys_arr[(i+ii)*dim*dim + (j+jj)*dim + k+kk] = new_key; 
                                            for(int c = 0; c < _num_channels; c++){
                                                avg_vals[c] += values_arr[((i+ii)*dim*dim + (j+jj)*dim + k+kk) * _num_channels + c];
                                            }

                                        }
                                    }
                                }

                                for(int ii=0; ii<step; ii++){
                                    for(int jj=0; jj<step; jj++){
                                        for(int kk=0; kk<step; kk++){
                                            for(int c = 0; c < _num_channels; c++){
                                                values_arr[((i+ii)*dim*dim + (j+jj)*dim + k+kk) * _num_channels + c] = avg_vals[c]/VALUE(step * step * step);
                                            }

                                        }
                                    }
                                }

                            }
                        }
                    }
                }
            }
            level--;
        }

        for(unsigned int i=0; i<vg.height(); i++){
            for(unsigned int j=0; j<vg.width(); j++){
                for(unsigned int k=0; k<vg.depth(); k++){
                    KEY key = keys_arr[i*dim*dim + j*dim + k];
                    if(_hash_table.find(key) != _hash_table.end()) continue;
                    _hash_table.insert(std::make_pair(key, vector<VALUE>(_num_channels)));
                    for(unsigned int c = 0;  c < _num_channels; c++){
                        _hash_table[key][c] = values_arr[i * dim * dim * _num_channels + j * dim * _num_channels + k * _num_channels + c];
                    }
                    
                }
            }
        }

        delete[] keys_arr;
        delete[] values_arr;
    }

    void from_voxel_grid_strict(MultiChannelVoxelGrid<VALUE>& vg, int min_level){
      
        int level = log2((float)vg.depth());
        int dim = vg.depth();

        _max_level = level;
        CHECK_EQ(_num_channels, vg.num_classes())
            << "the number of channels in octree must equal to the number of classes in the grid.";


        KEY *keys_arr = new KEY[dim*dim*dim];
        VALUE *values_arr = new VALUE[dim*dim*dim*_num_channels];

        //initially fill the hash map
        for(unsigned int i=0; i<vg.height(); i++){
            for(unsigned int j=0; j<vg.width(); j++){
                for(unsigned int k=0; k<vg.depth(); k++){    
                    OctreeCoord crd;
                    crd.x = i; crd.y = j; crd.z = k; crd.l = level;
                    KEY key = compute_key(crd);
                    keys_arr[i*dim*dim + j*dim + k] = key;
                    for(unsigned int c = 0; c < _num_channels; c++){
                        VALUE val = vg.get_element(c, i, j, k);
                        values_arr[(i * dim * dim + j * dim + k) * _num_channels + c] = val;
                    }
                }
            }
        }

        while(level > min_level){

            int step = pow(2, _max_level - level + 1);
            for(unsigned int i=0; i<vg.height(); i+=step){
                
                for(unsigned int j=0; j<vg.width(); j+=step){
                    
                    for(unsigned int k=0; k<vg.depth(); k+=step){

                        KEY key = keys_arr[i*dim*dim + j*dim + k];
                        if(compute_level(key) == level){
                            
                            const VALUE* values = values_arr + (i * dim * dim + j * dim + k)*_num_channels;
                           
                            int count = 0;
                            for(int ii=0; ii<2; ii++){
                                for(int jj=0; jj<2; jj++){
                                    for(int kk=0; kk<2; kk++){
                                        KEY comp_key = keys_arr[(i+ii*step/2)*dim*dim + (j+jj*step/2)*dim + k+kk*step/2];
                                        int comp_lev = compute_level(comp_key);
                                        if(comp_lev == level){

                                            const VALUE* values_comp = values_arr + ((i+ii*step/2)*dim*dim + (j+jj*step/2)*dim + k+kk*step/2) *_num_channels;

                                            if(values_equal(values, values_comp, _num_channels)) count++;
                                        }
                                    }
                                }
                            }

                            if(count==8 ){
                                KEY new_key = key >> 3;
                                for(int ii=0; ii<step; ii++){

                                    for(int jj=0; jj<step; jj++){

                                        for(int kk=0; kk<step; kk++){

                                            keys_arr[(i+ii)*dim*dim + (j+jj)*dim + k+kk] = new_key; 

                                        }
                                    }
                                }

                            }
                        }
                    }
                }
            }
            level--;
        }

        for(unsigned int i=0; i<vg.height(); i++){
            for(unsigned int j=0; j<vg.width(); j++){
                for(unsigned int k=0; k<vg.depth(); k++){
                    KEY key = keys_arr[i*dim*dim + j*dim + k];
                    if(_hash_table.find(key) != _hash_table.end()) continue;
                    _hash_table.insert(std::make_pair(key, vector<VALUE>(_num_channels)));
                    for(unsigned int c = 0;  c < _num_channels; c++){
                        _hash_table[key][c] = values_arr[i * dim * dim * _num_channels + j * dim * _num_channels + k * _num_channels + c];
                    }
                    
                }
            }
        }

        delete[] keys_arr;
        delete[] values_arr;
    }


    bool is_byteorder_big_endian(){
        int num = 1;
        if(*(char *)&num == 1){
            return false;
        }else{
            return true;
        }
    }

    void to_csv_file(std::string fname){
        std::ofstream ff(fname.c_str(), std::ios_base::out);
        for(typename std::tr1::unordered_map<KEY, vector<VALUE> >::iterator it=_hash_table.begin(); it!=_hash_table.end(); it++){
            ff <<  it->first << ",";
            for(int ch = 0; ch < _num_channels-1; ch++){
                ff << (it->second)[ch] << ",";
            }

            ff << (it->second)[_num_channels-1] << "\n";
        }

        ff.close();
    }

    void to_bin_file(std::string fname){

        uint8_t version = 1;
        uint8_t is_big_endian = is_byteorder_big_endian()? 1:0 ;
        uint8_t uint_size = 4;
        uint32_t elem_size = 4;
        uint32_t num_classes = _num_channels;
        uint32_t num_elements = this->num_elements();


        ofstream binaryIo(fname.c_str(), ios::binary);
        binaryIo.write((char *)(&version), sizeof(version));
        binaryIo.write((char *)(&is_big_endian), sizeof(is_big_endian));
        binaryIo.write((char *)(&uint_size), sizeof(uint_size));
    
        binaryIo.write((char *)(&elem_size), sizeof(elem_size));


        binaryIo.write((char *)(&num_classes), sizeof(num_classes));
        binaryIo.write((char *)(&num_elements), sizeof(num_elements));
       
       

        
        for(typename std::tr1::unordered_map<KEY, vector<VALUE> >::iterator it=_hash_table.begin(); it!=_hash_table.end(); it++){

            uint32_t key = it->first; 
            binaryIo.write((char*)(&key), sizeof(key));

            float* write_arr = new float[_num_channels];
            for(int ch = 0; ch < _num_channels; ch++){
                write_arr[ch] = (it->second)[ch];
            }

            binaryIo.write((char*)(write_arr), _num_channels*sizeof(float));
            delete [] write_arr;
        }

        binaryIo.close();


    }

    void from_csv_file(std::string fname){

        std::ifstream ff(fname.c_str(), std::ios_base::in);
        while (ff){

            string line;
            if (!std::getline(ff, line )) break;

            istringstream ss(line);
            vector<string>  temp;
            while (ss){
                string s;
                if (!std::getline( ss, s, ',' )) break;
                temp.push_back( s );
            }

            CHECK_EQ(temp.size(), _num_channels+1)
                << "number of elements in each line in csv file must equal to num_class + 1";


            KEY key;
            std::vector<VALUE> values;
            stringstream key_str(temp[0]);
            key_str >> key;
            for(int i = 1; i < temp.size(); i++){
                VALUE value;
                stringstream value_str(temp[i]);
                value_str >> value;
                values.push_back(value);

            }
            CHECK_EQ(values.size(), _num_channels)
                << "number of classes read in csv must equal to _num_channels";

            _hash_table[key] = values;
            int level = compute_level(key);
            if(level > _max_level) _max_level = level;



        }

        if (!ff.eof()){
            cerr << "read error!\n";
        }

        ff.close();

    }

   

    /*void from_csv_file(std::string fname){
        
        FILE* in = fopen(fname.c_str(), "r");
        CHECK(in != NULL)
            << "error fopen(): Failed to open input file.";
        KEY key;
        while(fscanf(in,"%u", &key) != EOF){

            vector<VALUE> values;
            float value;
            
            while( !feof(in) && fscanf(in,",%f", &value)>0 ){
                values.push_back(VALUE(value));
            }
           
            CHECK_EQ(values.size(), _num_channels)
                << "number of classes read in csv must equal to _num_channels";

            _hash_table[key] = values;
            int level = compute_level(key);
            if(level > _max_level) _max_level = level;

        }

        fclose(in);
    }*/

    void from_bin_file(std::string fname){

        uint8_t version;
        uint8_t is_big_endian;
        uint8_t uint_size;
        uint32_t elem_size;
        uint32_t num_classes;
        uint32_t num_elements;

        ifstream binaryIo(fname.c_str(), ios::binary);
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

        binaryIo.read((char *)(&num_classes), sizeof(num_classes));
        binaryIo.read((char *)(&num_elements), sizeof(num_elements));
               
        CHECK_EQ(num_classes, _num_channels) 
            << "num_classes must equal to _num_classes.";

        for(int i = 0; i < num_elements; i++){

            uint32_t key;
            vector<VALUE> values(_num_channels);
            binaryIo.read((char*)(&key), sizeof(key));


            if(elem_size == 4){
                float* read_arr = new float[_num_channels];
                binaryIo.read((char*)(read_arr), _num_channels*sizeof(float));
                for(int ch = 0; ch < _num_channels; ch++){
                    values[ch] = read_arr[ch];
                }
                delete [] read_arr;

                
            }else{
                double* read_arr = new double[_num_channels];
                binaryIo.read((char*)(read_arr), _num_channels*sizeof(double));
                for(int ch = 0; ch < _num_channels; ch++){
                    values[ch] = read_arr[ch];
                }
                delete [] read_arr;

            }


            _hash_table[key] = values;
            int level = compute_level(key);
            if(level > _max_level) _max_level = level;

        }

    }

};

#endif //MULTI_CHANNEL_OCTREE_H_
