#ifndef SIMULATION_N_BODY_GPUOPENCL_HPP_
#define SIMULATION_N_BODY_GPUOPENCL_HPP_

#include <string>
#include <cstring>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "CL/cl.h"

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyGPUOpenCL : public SimulationNBodyInterface {
  protected:
    // std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    accSoA_t<float> accelerations;
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id chosen_device;
    // cl_mem acc_buf;
    cl_mem ax_buf;
    cl_mem ay_buf;
    cl_mem az_buf;
    // cl_mem bodies_buf;
    cl_mem qx_buf;
    cl_mem qy_buf;
    cl_mem qz_buf;
    cl_mem m_buf;
    cl_mem test_buf;
    cl_kernel kernel;
    unsigned int DIM;
    float softSquared;
    size_t GPU_global_size;

  public:
    SimulationNBodyGPUOpenCL(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyGPUOpenCL() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
