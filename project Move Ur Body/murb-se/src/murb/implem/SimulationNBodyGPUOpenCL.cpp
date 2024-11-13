#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyGPUOpenCL.hpp"

#define MAX_PLATFORMS 3
#define MAX_DEVICES 5

/*
    Command test = ./bin/murb -n 1000 -i 100 -v --im gpu+ocl

    Utilisation of rsqrt in the cl file instead of sqrt for earning some 
    fps.
*/

SimulationNBodyGPUOpenCL::SimulationNBodyGPUOpenCL(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());

    // Create memory buffers on the device for each vector
    DIM = this->getBodies().getN();
    softSquared = soft * soft;

    GPU_global_size = this->getBodies().getN();

    if (this->getBodies().getN() < 64) {
        GPU_global_size = 64;
    }

    cl_platform_id pf[MAX_PLATFORMS];
    cl_device_id devices[MAX_DEVICES];

    // Get platform and device information
    cl_uint num_platforms;
    cl_uint num_devices;
    cl_int clStatus;

    //Set up the Platform
    clStatus = clGetPlatformIDs(MAX_PLATFORMS, pf, &num_platforms);

    if (clStatus != CL_SUCCESS) {
        printf("cl Status get platform ids = %d\n", clStatus);
    }

    clStatus = clGetDeviceIDs (pf[0], CL_DEVICE_TYPE_GPU, MAX_DEVICES, devices,
                           &num_devices);

    if (clStatus != CL_SUCCESS) {
        printf("cl Status get device ids = %d\n", clStatus);
    }

    chosen_device = devices[0];

    // Create one OpenCL context for each device in the platform
    this->context = clCreateContext(NULL, 1, &chosen_device, NULL, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create context = %d\n", clStatus);
    }

    // Create a command queue
    this->command_queue = clCreateCommandQueue(context, chosen_device, CL_QUEUE_PROFILING_ENABLE, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create command queue = %d\n", clStatus);
    }

    
    // Open and read the source code from the 
    std::string path = __FILE__;
    std::string final_path = path.substr(0, path.size() - strlen("SimulationNBodyGPUOpenCL.cpp")).append("SimulationNBodyGPUOpenCL.cl");
    FILE *kernels_file = fopen(final_path.c_str(), "r");
    fseek(kernels_file, 0, SEEK_END);
    size_t file_size = ftell(kernels_file);
    fseek(kernels_file, 0, SEEK_SET);

    char *kernels_source = (char*) malloc((file_size + 1) * sizeof(char));
    fread(kernels_source, sizeof(char), file_size, kernels_file);
    fclose(kernels_file);
    kernels_source[file_size] = '\0';

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &kernels_source, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create prog = %d\n", clStatus);
    }

    // Build the program
    // const char *option = "-cl-std=CLC++";
    clStatus = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status build prog = %d\n", clStatus);
    }

    // Create the OpenCL kernel
    kernel = clCreateKernel(program, "computeBodiesAccelerationOcl", &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create kernel = %d\n", clStatus);
    }

    char result[4096];
    size_t size;
    clStatus = clGetProgramBuildInfo(program, chosen_device, CL_PROGRAM_BUILD_LOG, sizeof(result), NULL, &size);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status program build info = %d\n", clStatus);
    }

    if (size > 2 && size <= 2048) {
      char buffer[size];

      fprintf (stderr, "--- OpenCL Compiler log ---\n");
      clGetProgramBuildInfo (program, chosen_device, CL_PROGRAM_BUILD_LOG,
                             sizeof (buffer), buffer, NULL);
      fprintf (stderr, "%s\n", buffer);
      fprintf (stderr, "---------------------------\n");
    }
}

void SimulationNBodyGPUOpenCL::initIteration()
{
    cl_int clStatus;

    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax[iBody] = 0.f;
        this->accelerations.ay[iBody] = 0.f;
        this->accelerations.az[iBody] = 0.f;
    }

    const dataSoA_t<float> d = this->getBodies().getDataSoA();

    const std::vector<float> vqx = d.qx;
    const std::vector<float> vqy = d.qy;
    const std::vector<float> vqz = d.qz;
    const std::vector<float> vm = d.m;

    qx_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DIM, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create debug_buf = %d\n", clStatus);
    }
    qy_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DIM, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create debug_buf = %d\n", clStatus);
    }
    qz_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DIM, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create debug_buf = %d\n", clStatus);
    }
    m_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DIM, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create debug_buf = %d\n", clStatus);
    }

    ax_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DIM, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create debug_buf = %d\n", clStatus);
    }
    ay_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DIM, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create debug_buf = %d\n", clStatus);
    }
    az_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * DIM, NULL, &clStatus);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status create debug_buf = %d\n", clStatus);
    }


    // Copy the Buffer A and B to the device

    clStatus = clEnqueueWriteBuffer(command_queue, qx_buf, CL_TRUE, 0, sizeof(float) * DIM, &vqx[0], 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status write buf debug = %d\n", clStatus);
    }
    clStatus = clEnqueueWriteBuffer(command_queue, qy_buf, CL_TRUE, 0, sizeof(float) * DIM, &vqy[0], 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status write buf debug = %d\n", clStatus);
    }
    clStatus = clEnqueueWriteBuffer(command_queue, qz_buf, CL_TRUE, 0, sizeof(float) * DIM, &vqz[0], 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status write buf debug = %d\n", clStatus);
    }
    clStatus = clEnqueueWriteBuffer(command_queue, m_buf, CL_TRUE, 0, sizeof(float) * DIM, &vm[0], 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status write buf debug = %d\n", clStatus);
    }

    clStatus = clEnqueueWriteBuffer(command_queue, ax_buf, CL_TRUE, 0, sizeof(float) * DIM, &(this->accelerations.ax[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status write buf debug = %d\n", clStatus);
    }
    clStatus = clEnqueueWriteBuffer(command_queue, ay_buf, CL_TRUE, 0, sizeof(float) * DIM, &(this->accelerations.ay[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status write buf debug = %d\n", clStatus);
    }
    clStatus = clEnqueueWriteBuffer(command_queue, az_buf, CL_TRUE, 0, sizeof(float) * DIM, &(this->accelerations.az[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status write buf debug = %d\n", clStatus);
    }
}

void SimulationNBodyGPUOpenCL::computeBodiesAcceleration()
{
    cl_int clStatus;

    // Execute the OpenCL kernel on the list
    size_t global_size[2] = {GPU_global_size, GPU_global_size}; // Process the entire lists
    // size_t global_size[1] = {GPU_global_size * GPU_global_size};
    // size_t global_size[2] = {100, 100}; 
    size_t local_size[2] = {1, 1}; 

    clStatus = clSetKernelArg(kernel, 0, sizeof(float), (void *)&(this->G));
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 0 = %d\n", clStatus);
    }
    clStatus = clSetKernelArg(kernel, 1, sizeof(float), (void *)&softSquared);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 1 = %d\n", clStatus);
    }

    clStatus = clSetKernelArg(kernel, 2, sizeof(qx_buf), (void *)&qx_buf);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 2 = %d\n", clStatus);
    }
    clStatus = clSetKernelArg(kernel, 3, sizeof(qy_buf), (void *)&qy_buf);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 3 = %d\n", clStatus);
    }
    clStatus = clSetKernelArg(kernel, 4, sizeof(qz_buf), (void *)&qz_buf);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 4 = %d\n", clStatus);
    }
    clStatus = clSetKernelArg(kernel, 5, sizeof(m_buf), (void *)&m_buf);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 5 = %d\n", clStatus);
    }
    clStatus = clSetKernelArg(kernel, 6, sizeof(ax_buf), (void *)&ax_buf);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 6 = %d\n", clStatus);
    }
    clStatus = clSetKernelArg(kernel, 7, sizeof(ay_buf), (void *)&ay_buf);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 7 = %d\n", clStatus);
    }
    clStatus = clSetKernelArg(kernel, 8, sizeof(az_buf), (void *)&az_buf);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status arg 8 = %d\n", clStatus);
    }

    clStatus = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status enqueue range kernel = %d\n", clStatus);
    }

    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status finish = %d\n", clStatus);
    }

    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(command_queue, ax_buf, CL_TRUE, 0, DIM * sizeof(float), &(this->accelerations.ax[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status read buf acc = %d\n", clStatus);
    }
    clStatus = clEnqueueReadBuffer(command_queue, ay_buf, CL_TRUE, 0, DIM * sizeof(float), &(this->accelerations.ay[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status read buf acc = %d\n", clStatus);
    }
    clStatus = clEnqueueReadBuffer(command_queue, az_buf, CL_TRUE, 0, DIM * sizeof(float), &(this->accelerations.az[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status read buf acc = %d\n", clStatus);
    }
}

void SimulationNBodyGPUOpenCL::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
