/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/

#include "kernel/ocl/common.cl"


__kernel void sample_ocl (__global unsigned *img)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  unsigned color = 0xFF0000FF; // opacity

  img [y * DIM + x] = color;
}

__kernel void sample_ocl_grad (__global unsigned *img)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  unsigned color = 0xFF; // opacity
  color |= (x & 255) << 24; // the greater x, the more red we use
  color |= (y & 255) << 8; // the greater y, the more blue we us

  img [y * DIM + x] = color;
}

__kernel void sample_ocl_grad_ydiv2 (__global unsigned *img)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  unsigned color = 0xFF;
  color |= (x & 255) << 24; 
  color |= (y & 255) << 8; 

  img [y * DIM + x] = color;
  img [(int)(y + (float)(DIM/2.0)) * DIM + x] = color;
}

