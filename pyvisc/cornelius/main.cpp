#include "hypersf.h"

int main( int argc, char** argv ){
    std::string pathin;
    double OLD_TIME;
    double NEW_TIME;
    int header_flag = 0;
    int corona;
    int vorticity;
    if(argc == 7){
        pathin = std::string(argv[1]);
        OLD_TIME = atof(argv[2]);
        NEW_TIME = atof(argv[3]);
        header_flag = atoi(argv[4]);
        corona = atoi(argv[5]);
        vorticity = atoi(argv[6]);
        
        Hypersf hypersf(pathin,OLD_TIME,NEW_TIME,header_flag,corona,vorticity);
        if (corona == 0){
           
          hypersf.get_hypersf();
        }
        
    }
  else{
      std::cout <<" usage: ./main path old_time new_time header_flag corona_flag vorticity_flag"<<std::endl;
      exit(1);
  }

    return 0;

    

}
