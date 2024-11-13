#ifndef SIMULATION_N_BODY_OPENMP_HPP_
#define SIMULATION_N_BODY_OPENMP_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyOpenMP : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    float softSquared;
    std::vector<float> mG;

  public:
    SimulationNBodyOpenMP(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOpenMP() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */