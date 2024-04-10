#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf_coupling.h>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include<chrono>

#include "include/sampler.h"
#include "include/constants.h"
#include "include/fourvector.h"
#include "include/integrate.h"

using namespace Smash;


/** Calc total number of species i from freeze out hypersurface in T.hirano's
 *  mthod. This equals to \f[ n*\sum_i u_i\cdot d\Sigma_i \f]*/
namespace hirano_method {
    constexpr double acu = 1.0E-8;
    /// dN/dp3
    inline double f(double p, double dS0, double dSi,
            double mass, double Tfrz, double muB, double lam){
        double E = sqrt(p*p+mass*mass);
        double f = 0.0;

        if( dS0 > 0.0 ){
            f += 4.0*M_PI*p*p/( exp((E-muB)/Tfrz)
                    + lam ) * dS0;
        }

        double vsig = dS0 / fmax(acu,dSi);

        //The second part in Ntotal calculation
        if ( fabs(vsig) < 1.0 && fabs(dSi)>acu && p > (mass*fabs(vsig)/std::sqrt(1-vsig*vsig)) ) {
            f += M_PI*( dSi*p*(E*E*vsig*vsig + p*p)/E/( exp((E-muB)/Tfrz) + lam) \
                    - 2.0*fabs( dS0 )*p*p/( exp((E-muB)/Tfrz) + lam) );
        }
        return f;
    }

    const double prefactor = 1.0/pow(twopi*hbarc,3.0);

    /// return: dN total number of hadrons from freeze out hyper surface dS^{mu}
    inline double get(double dS0, double dSi, double mass, double Tfrz, double muB, double lam){
        //Integrate over |p| to get dN
        Integrator integrate;

        return prefactor * integrate(0.0, 50.0*Tfrz,
                [&](double x){ return f(x, dS0, dSi, mass, Tfrz, muB, lam);});
    }
} // end namespace hirano_method




int main(int argc, char ** argv) {
    if ( argc != 7 ) {
        std::cerr << "usage:" << std::endl;
        std::cerr << "./main hypersf_directory viscous_on_ force_decay\
              number_of_sampling" << std::endl;
        std::cerr << "hypersf_directory: directory that has";
        std::cerr << "hypersf.dat and pimnsf.dat" << std::endl;
        std::cerr << "viscous_on: true to use viscous corrections" << std::endl;
        std::cerr << "force_decay: true to force decay" << std::endl;
        std::cerr << "num_of_sampling: type int" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string path(argv[1]);

    // switch for force resonance decay
    bool force_decay = false;
    std::string opt3(argv[2]);
    if ( opt3 == "true" || opt3 == "1" || opt3 == "True" || opt3 == "TRUE" ) {
        force_decay = true;
    }
    

    std::string eos_type(argv[3]);

    int number_of_events = std::atoi(argv[4]);
    std::string opt6(argv[5]);
    bool only_decay = false;
    if (opt6 == "true" || opt6 == "True" || opt6 == "TRUE" || opt6 == "1"){
        only_decay = true;
    }
    std::string model(argv[6]);
    


    auto start = std::chrono::system_clock::now();
    Sampler sampler(path, force_decay,only_decay,eos_type);
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::clog << "initialize finished! It costs " <<  elapsed_seconds.count() << " s" <<std::endl;
    

    int num_of_pion_plus = 0;

    if(only_decay){
        sampler.only_force_decay(path);
    }

    else{

    if( (!force_decay) && (model == "URQMD"))
    {
        std::stringstream fname_particle_list;
        fname_particle_list << path << "/mc_particle_list0";
        std::ofstream fpmag(fname_particle_list.str());
         double NTOT = 0.0;
        for ( int nevent=0; nevent < number_of_events; nevent++ ) {
            sampler.sample_particles_from_hypersf();
            
            NTOT = NTOT+ sampler.particles_.size();
            

            std::clog << nevent << "...";
            int particle_number = 0;
            fpmag<<"# "<< sampler.particles_.size()<<std::endl;
            for ( const auto & par : sampler.particles_ ) {
                int nid = sampler.newpid[par.pdgcode];
                fpmag << std::setprecision(6);
                fpmag << std::scientific;
                fpmag << par.pdgcode
                      << " "<<par.position.x0()
                      << " " << par.position.x1()
                      << " " << par.position.x2()
                      << " " << par.position.x3();
                fpmag << std::setprecision(16);
                fpmag << " " << par.momentum.x0()
                      << " " << par.momentum.x1()
                      << " " << par.momentum.x2()
                      << " " << par.momentum.x3()<< std::endl;;
       }
       

       sampler.particles_.clear();
       std::cout << "#finished" << std::endl;
       }
       fpmag.close();


    }
    else{

    
    


    
    std::stringstream fname_particle_list;
    fname_particle_list << path << "/mc_particle_list0";
    std::ofstream fpmag(fname_particle_list.str());
    fpmag << "#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge" <<std::endl;
    fpmag << "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none none" <<std::endl;
     double NTOT = 0.0;
    for ( int nevent=0; nevent < number_of_events; nevent++ ) {
       sampler.sample_particles_from_hypersf();
       
       NTOT = NTOT+ sampler.particles_.size();
       std::clog << nevent << "...";
       int particle_number = 0;
       
       for ( const auto & par : sampler.particles_ ) {
           int nid = sampler.newpid[par.pdgcode];
           if ( sampler.list_hadrons_.at(nid).stable &&
                sampler.list_hadrons_.at(nid).charge ) {
           FourVector momentum = par.momentum;
           double pmag = std::sqrt(momentum.sqr3());
           double pseudo_rapidity = 0.5*(std::log(pmag+momentum.x3())-
                       std::log(pmag-momentum.x3())); 

           double rapidity = 0.5*(std::log(momentum.x0()+momentum.x3())
                     - std::log(momentum.x0()-momentum.x3()));
           
           }
           if ( nid == 1 ) num_of_pion_plus ++;

           // write the output to mc_particle_list0
           particle_number ++;
           {
               fpmag << std::setprecision(6);
               fpmag << std::scientific;
               fpmag << par.position.x0()
                     << ' ' << par.position.x1()
                     << ' ' << par.position.x2()
                     << ' ' << par.position.x3();

               fpmag << std::setprecision(16);
               fpmag << ' ' << sampler.list_hadrons_.at(nid).mass
                     << ' ' << par.momentum.x0()
                     << ' ' << par.momentum.x1()
                     << ' ' << par.momentum.x2()
                     << ' ' << par.momentum.x3()
                     << ' ' << par.pdgcode
                     << ' ' << particle_number
                     << ' ' << sampler.list_hadrons_.at(nid).charge << std::endl;
           }
       }
       

       sampler.particles_.clear();
       std::cout << "#finished" << std::endl;
       fpmag << "# event "<<nevent<<" end"<<std::endl;
       }
       fpmag.close();
    




    std::clog << std::endl;

    std::clog << "ntot for pion+ from sample=" << num_of_pion_plus/ \
       static_cast<float>(number_of_events)  << std::endl;
    //if(!baryon_switch){

    double pion_mass = 0.13957;
    double baryon_chemical_potential = sampler.muB_[211];
    double fermion_boson_factor = -1.0;
    double freezeout_temperature = sampler.freezeout_temperature_;
    double pionplus_density = sampler.densities_.at(1);

    double ntotal_pion_plus_from_nudotsigma = 0.0;
    double ntotal_pion_plus_from_hirano = 0.0;

    for ( const auto & ele : sampler.elements_ ) {
       FourVector sigma_lrf = ele.dsigma.LorentzBoost(ele.velocity);
       double dSi = std::sqrt(fmax(really_small*really_small, \
                   sigma_lrf.sqr()));
       
       ntotal_pion_plus_from_nudotsigma += pionplus_density*sigma_lrf.x0();

       ntotal_pion_plus_from_hirano += hirano_method::get(sigma_lrf.x0(), dSi, \
               pion_mass, freezeout_temperature, baryon_chemical_potential,\
               fermion_boson_factor);
     }

    std::clog << "ntot for pion+ from udotsigma=" << 
       ntotal_pion_plus_from_nudotsigma << std::endl;

    std::clog << "ntot for pion+ from hirano (no viscous correction)=" << 
      ntotal_pion_plus_from_hirano << std::endl;
    }
    
    }
    
    
}

