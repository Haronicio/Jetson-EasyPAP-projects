#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyHetero.hpp"

#define MAX_PLATFORMS 3
#define MAX_DEVICES 5

// Factor to decide which part gpu and cpu are going to take
#define BALANCE_GPU_CPU 3

/*
    Command test = ./bin/murb -n 1000 -i 100 -v --im hetero
    OMP_NUM_THREADS=8 OMP_SCHEDULE="dynamic,16"

    Just the GPU and CPU part working together, the repartition of the index 
    is 1/3 for GPU and 2/3 for the CPU part, more for CPU part because faster. 
*/

SimulationNBodyHetero::SimulationNBodyHetero(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());

    // Create memory buffers on the device for each vector
    DIM = this->getBodies().getN();
    softSquared = this->soft * this->soft;

    GPU_global_size = this->getBodies().getN();

    if (this->getBodies().getN() < 64) {
        GPU_global_size = 64;
        GPU_CPU_balance = 0;
    } else {
        GPU_CPU_balance = GPU_global_size / BALANCE_GPU_CPU;
        if (GPU_CPU_balance < 64) {
            GPU_CPU_balance = 64;
        }
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
    std::string final_path = path.substr(0, path.size() - strlen("SimulationNBodyHetero.cpp")).append("SimulationNBodyGPUOpenCL.cl");
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

void SimulationNBodyHetero::initIteration()
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

void SimulationNBodyHetero::computeBodiesAcceleration()
{
    // GPU
    cl_int clStatus;

    // Execute the OpenCL kernel on the list
    size_t global_size[2] = {GPU_CPU_balance, GPU_global_size}; // Process the entire lists
    size_t local_size[2] = {1, 1}; // Process one item at a time

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

    // CPU
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    std::vector<float> ms(this->getBodies().getN(), 0);

#pragma omp parallel 
    {
        #pragma omp for schedule(runtime)
        for (unsigned long i = 0; i < this->getBodies().getN(); i++) {
            ms[i] = this->G * d[i].m;
        }
    }

    float sumAx = 0.f;
    float sumAy = 0.f;
    float sumAz = 0.f;

    // flops = n² * 20
    #pragma omp parallel firstprivate(sumAx,sumAy,sumAz)
    {
        #pragma omp for schedule(runtime)
        for (unsigned long iBody = GPU_CPU_balance; iBody < this->getBodies().getN(); iBody++) {
            const float x = d[iBody].qx;
            const float y = d[iBody].qy;
            const float z = d[iBody].qz;

            sumAx = 0.f;
            sumAy = 0.f;
            sumAz = 0.f;

            // flops = n * 19
            for (unsigned long jBody = 0; jBody < this->getBodies().getN(); jBody++) {
                const float rijx = d[jBody].qx - x;
                const float rijy = d[jBody].qy - y;
                const float rijz = d[jBody].qz - z;

                const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

                float powR = rijSquared + softSquared;
                powR *= std::sqrt(powR);
                const float ai = ms[jBody] / powR;

                sumAx += ai * rijx;
                sumAy += ai * rijy;
                sumAz += ai * rijz;
            }
            this->accelerations.ax[iBody] += sumAx;
            this->accelerations.ay[iBody] += sumAy;
            this->accelerations.az[iBody] += sumAz;
        }
    }

    // Clean up and wait for all the comands to complete.
    clStatus = clFlush(command_queue);
    clStatus = clFinish(command_queue);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status finish = %d\n", clStatus);
    }

    // Read the cl memory C_clmem on device to the host variable C
    clStatus = clEnqueueReadBuffer(command_queue, ax_buf, CL_TRUE, 0, GPU_CPU_balance * sizeof(float), &(this->accelerations.ax[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status read buf acc = %d\n", clStatus);
    }
    clStatus = clEnqueueReadBuffer(command_queue, ay_buf, CL_TRUE, 0, GPU_CPU_balance * sizeof(float), &(this->accelerations.ay[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status read buf acc = %d\n", clStatus);
    }
    clStatus = clEnqueueReadBuffer(command_queue, az_buf, CL_TRUE, 0, GPU_CPU_balance * sizeof(float), &(this->accelerations.az[0]), 0, NULL, NULL);
    if (clStatus != CL_SUCCESS) {
        printf("cl Status read buf acc = %d\n", clStatus);
    }
}

void SimulationNBodyHetero::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
