/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/

#include "easypap.h"

#include <omp.h>


static unsigned compute_one_pixel (int i, int j);
static void zoom (void);


int mandel_do_tile_default (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = compute_one_pixel (i, j);

  return 0;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --kernel mandel
//
unsigned mandel_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile (0, 0, DIM, DIM, 0);

    zoom ();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline:
// ./run -k mandel -v tiled -ts 64
//
unsigned mandel_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, 0);

    zoom ();
  }

  return 0;
}


// best time 2145ms with 10 iteration
// (dynamic, 500), 8 threads
// ./run -k mandel -v omp_tiled -i 10 -tw 8 -th 8 -s 1024 -t
/*
  Calcul of memory access and arithmetic operation
  memory access : (128*128) * (8 * 4) = 131 072 * 4 = 524 288
  arithmetic operation : (128*128) * 37 = 606 208

  OI = 606 208 / 524 288

  min(187, 170 * 1.2)
  min(187, 204) = 187

  compute bound
*/
unsigned mandel_compute_omp_tiled(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    #pragma omp parallel 
    {
      #pragma omp for schedule(runtime) collapse(2)
      for (int y = 0; y < DIM; y += TILE_H)
        for (int x = 0; x < DIM; x += TILE_W)
          do_tile (x, y, TILE_W, TILE_H, 0);
    }

    zoom ();
  }

  return 0;
}

/////////////// Mandelbrot basic computation

#define MAX_ITERATIONS 4096
#define ZOOM_SPEED -0.01

static float leftX   = -0.2395;
static float rightX  = -0.2275;
static float topY    = .660;
static float bottomY = .648;

static float xstep;
static float ystep;

void mandel_init ()
{
  // check tile size's conformity with respect to CPU vector width
  // easypap_check_vectorization (VEC_TYPE_FLOAT, DIR_HORIZONTAL);

  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

static unsigned iteration_to_color (unsigned iter)
{
  unsigned r = 0, g = 0, b = 0;

  if (iter < MAX_ITERATIONS) {
    if (iter < 64) {
      r = iter * 2; /* 0x0000 to 0x007E */
    } else if (iter < 128) {
      r = (((iter - 64) * 128) / 126) + 128; /* 0x0080 to 0x00C0 */
    } else if (iter < 256) {
      r = (((iter - 128) * 62) / 127) + 193; /* 0x00C1 to 0x00FF */
    } else if (iter < 512) {
      r = 255;
      g = (((iter - 256) * 62) / 255) + 1; /* 0x01FF to 0x3FFF */
    } else if (iter < 1024) {
      r = 255;
      g = (((iter - 512) * 63) / 511) + 64; /* 0x40FF to 0x7FFF */
    } else if (iter < 2048) {
      r = 255;
      g = (((iter - 1024) * 63) / 1023) + 128; /* 0x80FF to 0xBFFF */
    } else {
      r = 255;
      g = (((iter - 2048) * 63) / 2047) + 192; /* 0xC0FF to 0xFFFF */
    }
  }
  return rgba (r, g, b, 255);
}

static void zoom (void)
{
  float xrange = (rightX - leftX);
  float yrange = (topY - bottomY);

  leftX += ZOOM_SPEED * xrange;
  rightX -= ZOOM_SPEED * xrange;
  topY -= ZOOM_SPEED * yrange;
  bottomY += ZOOM_SPEED * yrange;

  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

static unsigned compute_one_pixel (int i, int j)
{
  float cr = leftX + xstep * j;
  float ci = topY - ystep * i;
  float zr = 0.0, zi = 0.0;

  int iter;

  // Pour chaque pixel, on calcule les termes d'une suite, et on
  // s'arrête lorsque |Z| > 2 ou lorsqu'on atteint MAX_ITERATIONS
  for (iter = 0; iter < MAX_ITERATIONS; iter++) {
    float x2 = zr * zr;
    float y2 = zi * zi;

    /* Stop iterations when |Z| > 2 */
    if (x2 + y2 > 4.0)
      break;

    float twoxy = (float)2.0 * zr * zi;
    /* Z = Z^2 + C */
    zr = x2 - y2 + cr;
    zi = twoxy + ci;
  }

  return iteration_to_color (iter);
}

