/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/

#include "easypap.h"

//EASY PAP BUG
#include "graphics.h"

///////////////////////////// Simple sequential version (seq)
unsigned sample_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        cur_img (i, j) = 0xFFFF00FF;

  }

  return 0;
}

#ifdef ENABLE_OPENCL

//////////////////////////////////////////////////////////////////////////
///////////////////////////// OpenCL version
// Suggested cmdlines:
// CPU    ./run -k sample --variant seq -i 1
// GPU    ./run -k sample -n --gpu --variant ocl -i 1
//
unsigned sample_invoke_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};   // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);

    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  // cpy data from GPU to CPU
  ocl_retrieve_data(); 
  // store the PNG image on the HDD
  graphics_dump_image_to_file("result_gpu_ocl.png");

  return 0;
}


// GPU    ./run -k sample -n --gpu --variant ocl_grad -i 1 -s 256
unsigned sample_invoke_ocl_grad (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};   // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);

    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  // cpy data from GPU to CPU
  ocl_retrieve_data(); 
  // store the PNG image on the HDD
  graphics_dump_image_to_file("result_gpu_ocl_grad.png");

  return 0;
}

// GPU    ./run -k sample -n --gpu --variant ocl_grad_ydiv2 -i 1 -s 256
unsigned sample_invoke_ocl_grad_ydiv2 (unsigned nb_iter)
{
  size_t ydiv2_size = (size_t)((float)(GPU_SIZE_Y)/2.0);
  size_t global[2] = {GPU_SIZE_X, ydiv2_size};   // global domain size for our calculation
  size_t local[2]  = {TILE_W, TILE_H}; // local domain size for our calculation
  cl_int err;

  uint64_t clock = monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);

    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

  }

  clFinish (queue);

  monitoring_end_tile (clock, 0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  // cpy data from GPU to CPU
  ocl_retrieve_data(); 
  // store the PNG image on the HDD
  graphics_dump_image_to_file("result_gpu_ocl_grad_ydiv2.png");

  return 0;
}

#endif
