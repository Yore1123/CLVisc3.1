#include "cl_spec.h"

int main(int argc, char** argv){
    
    std::chrono::time_point<std::chrono::system_clock> start, end; 
    start = std::chrono::system_clock::now(); 

    std::string pathin;
    int DECAY_ON = 0;
    int GPU_ID = 0;
    int NSAMPLING = 2000;
    int VORTICITY_ON = 0;
    std::string MODEL;
    std::string eos_type;
    if (argc == 8) {
        pathin = std::string(argv[1]);
        std::string opt3(argv[2]);
        if ( opt3 == "true" || opt3 == "1" || opt3 == "True" || opt3 == "TRUE" ) {
            DECAY_ON = 1;
        }
        eos_type = std::string(argv[3]);
	    NSAMPLING = atoi(argv[4]);
        GPU_ID = atoi(argv[5]);
        MODEL =  std::string(argv[6]);
        std::string opt4(argv[7]);
        if ( opt4 == "true" || opt4 == "1" || opt4 == "True" || opt4 == "TRUE" ) {
            VORTICITY_ON = 1;
        }

    } else {
        std::cerr << "Usage: ./spec hypersf_directory viscous_on decay_on gpu_id model" << std::endl;
        std::cerr << "Example: ./spec /home/name/results/event0 true true 0" << std::endl;
    }

    Spec spec(pathin, DECAY_ON, GPU_ID, eos_type, NSAMPLING,MODEL,VORTICITY_ON);
    spec.Sample_particle();

    
    end = std::chrono::system_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end - start; 
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "finished computation at " << std::ctime(&end_time) 
              << "elapsed time: " << elapsed_seconds.count() << "s\n"; 
    

}
