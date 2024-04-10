/*
 *
 *    Copyright (c) 2014
 *      SMASH Team
 *
 *    GNU General Public License (GPLv3 or later)
 *
 */

#ifndef SRC_INCLUDE_SAMPLER_H_
#define SRC_INCLUDE_SAMPLER_H_

#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <functional>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <iomanip>

#include "ars.h"
#include "fourvector.h"
#include "forwarddeclarations.h"
#include "particletype.h"

namespace Smash {
/// Sampled particle information
typedef struct {
    int pdgcode;
    FourVector position;
    FourVector momentum;
} ParticleData;

/// Information of each piece of freeze out hypersurface
typedef struct {
    FourVector dsigma;      // freeze out hypersurface dSigma^{\mu}
    FourVector nmutp;       // freeze out baryon number,chemical potential,temperature, pressure+efrz
    FourVector qb;
    ThreeVector velocity;   // fluid velocity, vx, vy, vetas
    double etas;            // space-time rapidity
    FourVector position;    // position (tau, x, y, etas)
    /** shear viscous tensor in comoving frame of fluid
     * pi00, pi01, pi02, pi03, pi11, pi12, pi13, pi22, pi23, pi33*/
    double pimn[10];
    /** sqrt(pi_{mu nu} pi^{mu nu}) in comoving frame */
    double pimn_max;
    double qb_max;
    std::vector<double> sfdensities;
    double sfdntot;
    Random::discrete_dist<double> sfdraw_hadron_type;

    
}VolumnElement;

class Sampler {
 public:
    /// Information of all particle types
    //
    //
    struct LoadFailure : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    /// constant temperature for freezeout hypersurface calculation
    double freezeout_temperature_;
    double temperature0_;
    double chemical_potential_;
    // 1/(2T^2(e+P)) on freeze out hypersf for viscous hydro
    double one_over_2TsqrEplusP_;

    // total thermal equilibrium density
    double dntot_;

    // discrete distribution to determine sampled particle type
    Random::discrete_dist<double> draw_hadron_type_;

    bool only_decay_;
    bool flag_mu_pce;

    // True to force the resonance decay to stable particles
    bool force_decay_;

    int TLENGTH;
    int MULENGTH;

    double T0;
    double MU0;
    double MU_STEP;
    double T_STEP;
    std::vector<double> deltaf_qmu; 

    std::map<int, int> newpid;

    // effective chemical potential from PCE-EOS
    std::map<int, double> muB_;

    // freeze out hypersf
    std::deque<VolumnElement> elements_;


    /**Store the information of sampled hadrons from
     *freezeout hypersurface in hydrodynamics.
     *The particles are used in List modus
     */
    std::deque<ParticleData> particles_;

    /// Equilibrium density from Bose-Einstein or Fermi-Dirac distributions
    std::vector<double> densities_;

    /* e+-, mu+-, photon will not be considered in this MC sampler;
     * densities_ and particle_dist_ vectors have different indices compared
     * with ParticleType::list_all(); Store ParticleTypePtr for the hadrons
     * in order to retrieve the information of hadrons later by
     * ParticleTypePtr.lookup()
     */
    std::vector<ParticleType> list_hadrons_;
    

    /// Adaptive rejection sampler for each species
    std::vector<Rejection::AdaptiveRejectionSampler> particle_dist_;

    /// No default constructor
    Sampler() = delete;

    /// Construct Sampler with hypersurface data file directory 
    // and viscous_on option to readin 10 pi^{mu nu} terms
    Sampler(const std::string & fpath,  bool force_decay,bool only_decay,const std::string & eos_type);

    /** Read freeze out hypersurface from data file
    * if VISCOUS_CORRECTION==true, information of pi^{mu nu} are needed
    */
    void read_hypersurface(const std::string & hypersf_directory);

    /** read pimn on freeze out hypersf */
    void read_pimn_on_sf(const std::string & pimn_directory);

    /** read qb on freeze out hypersf */
    void read_qb_on_sf(const std::string & qb_directory);

    void Readdeltaf(const std::string & deltaf_directory);
    double get_deltaf_qmn(double T, double mu);
    /** read nmtp on freeze out hypersf */
    void read_nmtp_on_sf(const std::string & nmtp_directory);

    void read_chemical_potential(const std::string & muB_directory);

    /** Read particle data table pdg05.txt */
    void read_pdg_table(const std::string & fpath);
    /** Calculate equilibrium density and initialize
     * adaptive rejection sampler for each species*/
    void get_equilibrium_properties();
    void get_equilibrium_properties2();

    /** Sample 4 momentum from freeze out hyper surface*/
    void sample_particles_from_hypersf();

    /** only do force decay*/
    void only_force_decay(const std::string& fpath);
    
    bool neglect_partices(int pdgid);
    /** Force the resonance to decay to stable particles */
    void force_decay(const ParticleData & resonance);

    void two_body_decay(const ParticleData & reso_data,
                        const DecayChannel & channel,
                        std::vector<ParticleData> & unstable);

    void three_body_decay(const ParticleData & reso_data,
                        const DecayChannel & channel,
                        std::vector<ParticleData> & unstable);

    /** Get the pion+ yield ratio between after reso ( calculated from
     * thermal equilbirum densities and branch ratios to pion+ ) and 
     * before reso */
    double pion_ratio_after_reso_vs_before();

};
}  // end namespace Smash

#endif  // SRC_INCLUDE_SAMPLER_H_
