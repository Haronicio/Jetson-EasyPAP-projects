/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/

#include "easypap.h"

#include <omp.h>

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v seq -si
//
int blur2_do_tile_default(int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
    {
      unsigned r = 0, g = 0, b = 0, a = 0, n = 0;

      int i_d = (i > 0) ? i - 1 : i;
      int i_f = (i < DIM - 1) ? i + 1 : i;
      int j_d = (j > 0) ? j - 1 : j;
      int j_f = (j < DIM - 1) ? j + 1 : j;

      for (int yloc = i_d; yloc <= i_f; yloc++)
        for (int xloc = j_d; xloc <= j_f; xloc++)
        {
          unsigned c = cur_img(yloc, xloc);
          r += extract_red(c);
          g += extract_green(c);
          b += extract_blue(c);
          a += extract_alpha(c);
          n += 1;
        }

      r /= n;
      g /= n;
      b /= n;
      a /= n;

      next_img(i, j) = rgba(r, g, b, a);
    }

  return 0;
}

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v seq
//
unsigned blur2_compute_seq(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {

    do_tile(0, 0, DIM, DIM, 0);

    swap_images();
  }

  return 0;
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v tiled -ts 32 -m si
//
unsigned blur2_compute_tiled(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile(x, y, TILE_W, TILE_H, 0);

    swap_images();
  }

  return 0;
}


// best time with ((dynamic, 1),th = 128, tw = 256, 8 threads) = 230ms for 100 iterations
// export OMP_NUM_THREADS=8 (just a little more than our cores)
// export OMP_SCHEDULE="dynamic,1"
// ./run -k blur2 -l images/1024.png -v omp_tiled -i 100 -wt urrot2 -th 128 -tw 256 -m -t
/*
  Calcul of every memory access using urrot2 with the parameters I choose before :
  ((1024/size_tile_height * 1024/size_tile_width) * memory_access + compute_borders_access) * size_data_type 
  ((1024/128) * (1024/256)) * (127 * (32 + 257 * 36)) + (127 * 6 * 2 + 9 * 127) = 150 931 372 * 8 = 1207450976

  with graphs of bandwitch we produce earlier we can say we made most of the loads and store in the RAM
  because 1 207 450 KB >> 5 092 KB

  For each pixel : 1207450976 / 1024 = 1 179 151, 1179KB < 5092KB, nearly any access to the ram for each pixel 

  I should now calculate the number of ops in the same code

  OI = 515 178 / 1 179 151 = 2.3

  Roofline model = min(57+130, 170 * 0.43)
                 = min(187, 73) = 73

  memory bound
*/
unsigned blur2_compute_omp_tiled(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++)
  {
    #pragma omp parallel 
    {
      #pragma omp for schedule(runtime) collapse(2)
      for (int y = 0; y < DIM; y += TILE_H) {
       for (int x = 0; x < DIM; x += TILE_W) {
          do_tile(x, y, TILE_W, TILE_H, omp_get_thread_num());
        }
      }
    }
    swap_images();
  }

  return 0;
}

static void compute_borders(int x, int y, int width, int height, int bsize)
{
  assert(bsize > 0);
  // left -------------------------------------------------------------------------------------------------------------
  for (int i = y + 1; i < y + height - 1; i++)
  {
    uint16_t r = 0, g = 0, b = 0, a = 0;
    unsigned c_0_1 = cur_img(i - 1, x + 0), c_1_1 = cur_img(i + 0, x + 0), c_2_1 = cur_img(i + 1, x + 0);
    unsigned c_0_2 = cur_img(i - 1, x + 1), c_1_2 = cur_img(i + 0, x + 1), c_2_2 = cur_img(i + 1, x + 1);
    r += extract_red(c_0_1);
    g += extract_green(c_0_1);
    b += extract_blue(c_0_1);
    a += extract_alpha(c_0_1);
    r += extract_red(c_1_1);
    g += extract_green(c_1_1);
    b += extract_blue(c_1_1);
    a += extract_alpha(c_1_1);
    r += extract_red(c_2_1);
    g += extract_green(c_2_1);
    b += extract_blue(c_2_1);
    a += extract_alpha(c_2_1);
    r += extract_red(c_0_2);
    g += extract_green(c_0_2);
    b += extract_blue(c_0_2);
    a += extract_alpha(c_0_2);
    r += extract_red(c_1_2);
    g += extract_green(c_1_2);
    b += extract_blue(c_1_2);
    a += extract_alpha(c_1_2);
    r += extract_red(c_2_2);
    g += extract_green(c_2_2);
    b += extract_blue(c_2_2);
    a += extract_alpha(c_2_2);
    r /= 6;
    g /= 6;
    b /= 6;
    a /= 6;
    next_img(i, 0) = rgba(r, g, b, a);
  }

  for (int i = y + 1; i < y + height - 1; i++)
  {
    for (int j = x + 1; j < x + bsize; j++)
    {
      uint16_t r = 0, g = 0, b = 0, a = 0;
      unsigned c_0_0 = cur_img(i - 1, j - 1), c_1_0 = cur_img(i + 0, j - 1), c_2_0 = cur_img(i + 1, j - 1);
      unsigned c_0_1 = cur_img(i - 1, j + 0), c_1_1 = cur_img(i + 0, j + 0), c_2_1 = cur_img(i + 1, j + 0);
      unsigned c_0_2 = cur_img(i - 1, j + 1), c_1_2 = cur_img(i + 0, j + 1), c_2_2 = cur_img(i + 1, j + 1);
      r += extract_red(c_0_0);
      g += extract_green(c_0_0);
      b += extract_blue(c_0_0);
      a += extract_alpha(c_0_0);
      r += extract_red(c_1_0);
      g += extract_green(c_1_0);
      b += extract_blue(c_1_0);
      a += extract_alpha(c_1_0);
      r += extract_red(c_2_0);
      g += extract_green(c_2_0);
      b += extract_blue(c_2_0);
      a += extract_alpha(c_2_0);
      r += extract_red(c_0_1);
      g += extract_green(c_0_1);
      b += extract_blue(c_0_1);
      a += extract_alpha(c_0_1);
      r += extract_red(c_1_1);
      g += extract_green(c_1_1);
      b += extract_blue(c_1_1);
      a += extract_alpha(c_1_1);
      r += extract_red(c_2_1);
      g += extract_green(c_2_1);
      b += extract_blue(c_2_1);
      a += extract_alpha(c_2_1);
      r += extract_red(c_0_2);
      g += extract_green(c_0_2);
      b += extract_blue(c_0_2);
      a += extract_alpha(c_0_2);
      r += extract_red(c_1_2);
      g += extract_green(c_1_2);
      b += extract_blue(c_1_2);
      a += extract_alpha(c_1_2);
      r += extract_red(c_2_2);
      g += extract_green(c_2_2);
      b += extract_blue(c_2_2);
      a += extract_alpha(c_2_2);
      r /= 9;
      g /= 9;
      b /= 9;
      a /= 9;
      next_img(i, j) = rgba(r, g, b, a);
    }
  }

  // right ------------------------------------------------------------------------------------------------------------
  for (int i = y + 1; i < y + height - 1; i++)
  {
    uint16_t r = 0, g = 0, b = 0, a = 0;
    unsigned c_0_0 = cur_img(i - 1, x + width - 2), c_1_0 = cur_img(i + 0, x + width - 2), c_2_0 = cur_img(i + 1, x + width - 2);
    unsigned c_0_1 = cur_img(i - 1, x + width - 1), c_1_1 = cur_img(i + 0, x + width - 1), c_2_1 = cur_img(i + 1, x + width - 1);
    r += extract_red(c_0_1);
    g += extract_green(c_0_1);
    b += extract_blue(c_0_1);
    a += extract_alpha(c_0_1);
    r += extract_red(c_1_1);
    g += extract_green(c_1_1);
    b += extract_blue(c_1_1);
    a += extract_alpha(c_1_1);
    r += extract_red(c_2_1);
    g += extract_green(c_2_1);
    b += extract_blue(c_2_1);
    a += extract_alpha(c_2_1);
    r += extract_red(c_0_0);
    g += extract_green(c_0_0);
    b += extract_blue(c_0_0);
    a += extract_alpha(c_0_0);
    r += extract_red(c_1_0);
    g += extract_green(c_1_0);
    b += extract_blue(c_1_0);
    a += extract_alpha(c_1_0);
    r += extract_red(c_2_0);
    g += extract_green(c_2_0);
    b += extract_blue(c_2_0);
    a += extract_alpha(c_2_0);
    r /= 6;
    g /= 6;
    b /= 6;
    a /= 6;
    next_img(i, x + width - 1) = rgba(r, g, b, a);
  }

  for (int i = y + 1; i < y + height - 1; i++)
  {
    for (int j = x + width - bsize; j < x + width - 1; j++)
    {
      uint16_t r = 0, g = 0, b = 0, a = 0;
      unsigned c_0_0 = cur_img(i - 1, j - 1), c_1_0 = cur_img(i + 0, j - 1), c_2_0 = cur_img(i + 1, j - 1);
      unsigned c_0_1 = cur_img(i - 1, j + 0), c_1_1 = cur_img(i + 0, j + 0), c_2_1 = cur_img(i + 1, j + 0);
      unsigned c_0_2 = cur_img(i - 1, j + 1), c_1_2 = cur_img(i + 0, j + 1), c_2_2 = cur_img(i + 1, j + 1);
      r += extract_red(c_0_0);
      g += extract_green(c_0_0);
      b += extract_blue(c_0_0);
      a += extract_alpha(c_0_0);
      r += extract_red(c_1_0);
      g += extract_green(c_1_0);
      b += extract_blue(c_1_0);
      a += extract_alpha(c_1_0);
      r += extract_red(c_2_0);
      g += extract_green(c_2_0);
      b += extract_blue(c_2_0);
      a += extract_alpha(c_2_0);
      r += extract_red(c_0_1);
      g += extract_green(c_0_1);
      b += extract_blue(c_0_1);
      a += extract_alpha(c_0_1);
      r += extract_red(c_1_1);
      g += extract_green(c_1_1);
      b += extract_blue(c_1_1);
      a += extract_alpha(c_1_1);
      r += extract_red(c_2_1);
      g += extract_green(c_2_1);
      b += extract_blue(c_2_1);
      a += extract_alpha(c_2_1);
      r += extract_red(c_0_2);
      g += extract_green(c_0_2);
      b += extract_blue(c_0_2);
      a += extract_alpha(c_0_2);
      r += extract_red(c_1_2);
      g += extract_green(c_1_2);
      b += extract_blue(c_1_2);
      a += extract_alpha(c_1_2);
      r += extract_red(c_2_2);
      g += extract_green(c_2_2);
      b += extract_blue(c_2_2);
      a += extract_alpha(c_2_2);
      r /= 9;
      g /= 9;
      b /= 9;
      a /= 9;
      next_img(i, j) = rgba(r, g, b, a);
    }
  }

  // top & bottom -----------------------------------------------------------------------------------------------------
  for (int i = y; i < y + height; i += height - 1)
  {
    for (int j = x; j < x + width; j++)
    {
      uint16_t r = 0, g = 0, b = 0, a = 0, n = 0;
      int i_d = (i > 0) ? i - 1 : i;
      int i_f = (i < DIM - 1) ? i + 1 : i;
      int j_d = (j > 0) ? j - 1 : j;
      int j_f = (j < DIM - 1) ? j + 1 : j;
      for (int yloc = i_d; yloc <= i_f; yloc++)
        for (int xloc = j_d; xloc <= j_f; xloc++)
        {
          unsigned c = cur_img(yloc, xloc);
          r += extract_red(c);
          g += extract_green(c);
          b += extract_blue(c);
          a += extract_alpha(c);
          n += 1;
        }
      r /= n;
      g /= n;
      b /= n;
      a /= n;
      next_img(i, j) = rgba(r, g, b, a);
    }
  }
}

///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v seq -si --wt urrot1
//
/*

  Denver  A57
  749     1303
*/
int blur2_do_tile_urrot1(int x, int y, int width, int height)
{
  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++)
  {

    uint16_t c_0_r = 0, c_0_g = 0, c_0_b = 0, c_0_a = 0; // col 0 -> 4 color components {r,g,b,a}
    uint16_t c_1_r = 0, c_1_g = 0, c_1_b = 0, c_1_a = 0; // col 1 -> 4 color components {r,g,b,a}

    // read 3 pixels of column 0
    unsigned c_0_0 = cur_img(i - 1, x + 0), c_1_0 = cur_img(i + 0, x + 0), c_2_0 = cur_img(i + 1, x + 0);
    // read 3 pixels of column 1
    unsigned c_0_1 = cur_img(i - 1, x + 1), c_1_1 = cur_img(i + 0, x + 1), c_2_1 = cur_img(i + 1, x + 1);

    // reduction of the pixels of column 0 (per components {r,g,b,a})
    c_0_r += extract_red(c_0_0);
    c_0_g += extract_green(c_0_0);
    c_0_b += extract_blue(c_0_0);
    c_0_a += extract_alpha(c_0_0);
    c_0_r += extract_red(c_1_0);
    c_0_g += extract_green(c_1_0);
    c_0_b += extract_blue(c_1_0);
    c_0_a += extract_alpha(c_1_0);
    c_0_r += extract_red(c_2_0);
    c_0_g += extract_green(c_2_0);
    c_0_b += extract_blue(c_2_0);
    c_0_a += extract_alpha(c_2_0);

    // reduction of the pixels of column 1 (per components {r,g,b,a})
    c_1_r += extract_red(c_0_1);
    c_1_g += extract_green(c_0_1);
    c_1_b += extract_blue(c_0_1);
    c_1_a += extract_alpha(c_0_1);
    c_1_r += extract_red(c_1_1);
    c_1_g += extract_green(c_1_1);
    c_1_b += extract_blue(c_1_1);
    c_1_a += extract_alpha(c_1_1);
    c_1_r += extract_red(c_2_1);
    c_1_g += extract_green(c_2_1);
    c_1_b += extract_blue(c_2_1);
    c_1_a += extract_alpha(c_2_1);

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width - 1; j++)
    {
      uint16_t c_2_r = 0, c_2_g = 0, c_2_b = 0, c_2_a = 0; // col 2 -> 4 color components {r,g,b,a}

      // read 3 pixels of column 2
      unsigned c_0_2 = cur_img(i - 1, j + 1);
      unsigned c_1_2 = cur_img(i + 0, j + 1);
      unsigned c_2_2 = cur_img(i + 1, j + 1);

      // reduction of the pixels of column 2 (per components {r,g,b,a})
      c_2_r += extract_red(c_0_2);
      c_2_g += extract_green(c_0_2);
      c_2_b += extract_blue(c_0_2);
      c_2_a += extract_alpha(c_0_2);
      c_2_r += extract_red(c_1_2);
      c_2_g += extract_green(c_1_2);
      c_2_b += extract_blue(c_1_2);
      c_2_a += extract_alpha(c_1_2);
      c_2_r += extract_red(c_2_2);
      c_2_g += extract_green(c_2_2);
      c_2_b += extract_blue(c_2_2);
      c_2_a += extract_alpha(c_2_2);

      // compute the sum of all the columns 0,1,2 per components {r,g,b,a}
      uint16_t r = 0, g = 0, b = 0, a = 0;
      r = c_0_r + c_1_r + c_2_r;
      g = c_0_g + c_1_g + c_2_g;
      b = c_0_b + c_1_b + c_2_b;
      a = c_0_a + c_1_a + c_2_a;
      // compute the average (sum = sum / 9)
      r /= 9;
      g /= 9;
      b /= 9;
      a /= 9;

      // variables rotations (col0 <- col1 and col1 <- col2)
      c_0_r = c_1_r;
      c_0_g = c_1_g;
      c_0_b = c_1_b;
      c_0_a = c_1_a;
      c_1_r = c_2_r;
      c_1_g = c_2_g;
      c_1_b = c_2_b;
      c_1_a = c_2_a;

      // write the current pixel
      next_img(i, j) = rgba(r, g, b, a);
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

/*
  Denver  A57
*/

/*
int blur2_do_tile_urrot1 (int x, int y, int width, int height)
{
  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++) {

    // uint8x16_t rc_r = {};
    // uint8x16_t rc_g = {};
    // uint8x16_t rc_b = {};
    // uint8x16_t rc_r = {};

    uint16_t c_0_r = 0, c_0_g = 0, c_0_b = 0, c_0_a = 0; // col 0 -> 4 color components {r,g,b,a}
    uint16_t c_1_r = 0, c_1_g = 0, c_1_b = 0, c_1_a = 0; // col 1 -> 4 color components {r,g,b,a}


    // uint8x16_t c_0 = {cur_img(i - 1, x + 0), c_1_0 = cur_img(i + 0, x + 0), c_2_0 = cur_img(i + 1, x + 0), 0};
    // uint16x4_t c_1 = {cur_img(i - 1, x + 1), c_1_1 = cur_img(i + 0, x + 1), c_2_1 = cur_img(i + 1, x + 1), 0};


    // read 3 pixels of column 0
    unsigned c_0_0 = cur_img(i - 1, x + 0), c_1_0 = cur_img(i + 0, x + 0), c_2_0 = cur_img(i + 1, x + 0);
    // read 3 pixels of column 1
    unsigned c_0_1 = cur_img(i - 1, x + 1), c_1_1 = cur_img(i + 0, x + 1), c_2_1 = cur_img(i + 1, x + 1);

    // reduction of the pixels of column 0 (per components {r,g,b,a})
    c_0_r += extract_red(c_0_0); c_0_g += extract_green(c_0_0); c_0_b += extract_blue(c_0_0); c_0_a += extract_alpha(c_0_0);
    c_0_r += extract_red(c_1_0); c_0_g += extract_green(c_1_0); c_0_b += extract_blue(c_1_0); c_0_a += extract_alpha(c_1_0);
    c_0_r += extract_red(c_2_0); c_0_g += extract_green(c_2_0); c_0_b += extract_blue(c_2_0); c_0_a += extract_alpha(c_2_0);

    // reduction of the pixels of column 1 (per components {r,g,b,a})
    c_1_r += extract_red(c_0_1); c_1_g += extract_green(c_0_1); c_1_b += extract_blue(c_0_1); c_1_a += extract_alpha(c_0_1);
    c_1_r += extract_red(c_1_1); c_1_g += extract_green(c_1_1); c_1_b += extract_blue(c_1_1); c_1_a += extract_alpha(c_1_1);
    c_1_r += extract_red(c_2_1); c_1_g += extract_green(c_2_1); c_1_b += extract_blue(c_2_1); c_1_a += extract_alpha(c_2_1);

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width - 1; j++) {

      // column
      // uint8x16x4_t c_1_2 = vld4_u8((uint8_t*)& cur_img(i, j+16));

      // uint8x16_t r_1_2 =
      // uint8x16_t g_1_2 =
      // uint8x16_t b_1_2 =
      // uint8x16_t a_1_2 =

      uint16_t c_2_r = 0, c_2_g = 0, c_2_b = 0, c_2_a = 0; // col 2 -> 4 color components {r,g,b,a}

      // read 3 pixels of column 2
      unsigned c_0_2 = cur_img(i - 1, j + 1);
      unsigned c_1_2 = cur_img(i + 0, j + 1);
      unsigned c_2_2 = cur_img(i + 1, j + 1);

      // reduction of the pixels of column 2 (per components {r,g,b,a})
      c_2_r += extract_red(c_0_2); c_2_g += extract_green(c_0_2); c_2_b += extract_blue(c_0_2); c_2_a += extract_alpha(c_0_2);
      c_2_r += extract_red(c_1_2); c_2_g += extract_green(c_1_2); c_2_b += extract_blue(c_1_2); c_2_a += extract_alpha(c_1_2);
      c_2_r += extract_red(c_2_2); c_2_g += extract_green(c_2_2); c_2_b += extract_blue(c_2_2); c_2_a += extract_alpha(c_2_2);

      // compute the sum of all the columns 0,1,2 per components {r,g,b,a}
      uint16_t r = 0, g = 0, b = 0, a = 0;
      r = c_0_r+c_1_r+c_2_r; g = c_0_g+c_1_g+c_2_g; b = c_0_b+c_1_b+c_2_b; a = c_0_a+c_1_a+c_2_a;
      // compute the average (sum = sum / 9)
      r /= 9; g /= 9; b /= 9; a /= 9;

      // variables rotations (col0 <- col1 and col1 <- col2)
      c_0_r = c_1_r; c_0_g = c_1_g; c_0_b = c_1_b; c_0_a = c_1_a;
      c_1_r = c_2_r; c_1_g = c_2_g; c_1_b = c_2_b; c_1_a = c_2_a;

      // write the current pixel
      next_img(i, j) = rgba (r, g, b, a);
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}
*/
///////////////////////////// Sequential version (tiled)
// Suggested cmdline(s):
// ./run -l images/1024.png -k blur2 -v seq -si --wt urrot2
//
int blur2_do_tile_urrot2(int x, int y, int width, int height)
{
  // small arrays (4 elements each) to store all the components of 1 pixel
  uint8_t c_0_0[4], c_1_0[4], c_2_0[4];
  uint8_t c_0_1[4], c_1_1[4], c_2_1[4];
  uint8_t c_0_2[4], c_1_2[4], c_2_2[4];

  // loop over y (start from +1, end at -1 => no border)
  for (int i = y + 1; i < y + height - 1; i++)
  {
    uint8_t *cur_img_ptr;
    // load the 3 pixels of the column 0 (no extraction of the components)
    cur_img_ptr = (uint8_t *)&cur_img(i - 1, x + 0);
    for (int c = 0; c < 4; c++)
      c_0_0[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t *)&cur_img(i + 0, x + 0);
    for (int c = 0; c < 4; c++)
      c_1_0[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t *)&cur_img(i + 1, x + 0);
    for (int c = 0; c < 4; c++)
      c_2_0[c] = cur_img_ptr[c];

    // load the 3 pixels of the column 1 (no extraction of the components)
    cur_img_ptr = (uint8_t *)&cur_img(i - 1, x + 1);
    for (int c = 0; c < 4; c++)
      c_0_1[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t *)&cur_img(i + 0, x + 1);
    for (int c = 0; c < 4; c++)
      c_1_1[c] = cur_img_ptr[c];
    cur_img_ptr = (uint8_t *)&cur_img(i + 1, x + 1);
    for (int c = 0; c < 4; c++)
      c_2_1[c] = cur_img_ptr[c];

    // column 0 and column 1 reduction
    uint16_t c_0[4], c_1[4];
    for (int c = 0; c < 4; c++)
      c_0[c] = c_0_0[c] + c_1_0[c] + c_2_0[c];
    for (int c = 0; c < 4; c++)
      c_1[c] = c_0_1[c] + c_1_1[c] + c_2_1[c];

    // loop over x (start from +1, end at -1 => no border)
    for (int j = x + 1; j < x + width - 1; j++)
    {
      // load the 3 pixels of the column 2 (no extraction of the components)
      cur_img_ptr = (uint8_t *)&cur_img(i - 1, j + 1);
      for (int c = 0; c < 4; c++)
        c_0_2[c] = cur_img_ptr[c];
      cur_img_ptr = (uint8_t *)&cur_img(i + 0, j + 1);
      for (int c = 0; c < 4; c++)
        c_1_2[c] = cur_img_ptr[c];
      cur_img_ptr = (uint8_t *)&cur_img(i + 1, j + 1);
      for (int c = 0; c < 4; c++)
        c_2_2[c] = cur_img_ptr[c];

      // column 2 reduction
      uint16_t c_2[4] = {0, 0, 0, 0};
      for (int c = 0; c < 4; c++)
        c_2[c] += c_0_2[c] + c_1_2[c] + c_2_2[c];

      // add column 0, 1 and 2 and compute the avg (div9)
      uint16_t avg[4] = {0, 0, 0, 0};
      for (int c = 0; c < 4; c++)
        avg[c] = (c_0[c] + c_1[c] + c_2[c]) / 9;

      // variables rotations
      for (int c = 0; c < 4; c++)
        c_0[c] = c_1[c];
      for (int c = 0; c < 4; c++)
        c_1[c] = c_2[c];

      // store the resulting pixel (no need for the 'rgba' function)
      uint8_t *next_img_ptr = (uint8_t *)&next_img(i, j);
      for (int c = 0; c < 4; c++)
        next_img_ptr[c] = avg[c];
    }
  }

  // left-right borders size
  uint32_t bsize = 1;
  // compute the borders
  compute_borders(x, y, width, height, bsize);

  return 0;
}

#if defined(ENABLE_VECTO) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include <arm_neon.h>

void print_reg_u8(const uint8x16_t r, const char *name)
{
  printf("%s = [", name);
  printf("%u, ", vgetq_lane_u8(r, 0));
  printf("%u, ", vgetq_lane_u8(r, 1));
  printf("%u, ", vgetq_lane_u8(r, 2));
  printf("%u, ", vgetq_lane_u8(r, 3));
  printf("%u, ", vgetq_lane_u8(r, 4));
  printf("%u, ", vgetq_lane_u8(r, 5));
  printf("%u, ", vgetq_lane_u8(r, 6));
  printf("%u, ", vgetq_lane_u8(r, 7));
  printf("%u, ", vgetq_lane_u8(r, 8));
  printf("%u, ", vgetq_lane_u8(r, 9));
  printf("%u, ", vgetq_lane_u8(r, 10));
  printf("%u, ", vgetq_lane_u8(r, 11));
  printf("%u, ", vgetq_lane_u8(r, 12));
  printf("%u, ", vgetq_lane_u8(r, 13));
  printf("%u, ", vgetq_lane_u8(r, 14));
  printf("%u", vgetq_lane_u8(r, 15));
  printf("]\n");
}

void print_reg_u16(const uint16x8_t r, const char *name)
{
  printf("%s = [", name);
  printf("%u, ", vgetq_lane_u16(r, 0));
  printf("%u, ", vgetq_lane_u16(r, 1));
  printf("%u, ", vgetq_lane_u16(r, 2));
  printf("%u, ", vgetq_lane_u16(r, 3));
  printf("%u, ", vgetq_lane_u16(r, 4));
  printf("%u, ", vgetq_lane_u16(r, 5));
  printf("%u, ", vgetq_lane_u16(r, 6));
  printf("%u", vgetq_lane_u16(r, 7));
  printf("]\n");
}

void print_reg_u32(const uint32x4_t r, const char *name)
{
  printf("%s = [", name);
  printf("%u, ", vgetq_lane_u32(r, 0));
  printf("%u, ", vgetq_lane_u32(r, 1));
  printf("%u, ", vgetq_lane_u32(r, 2));
  printf("%u", vgetq_lane_u32(r, 3));
  printf("]\n");
}

void print_reg_f32(const float32x4_t r, const char *name)
{
  printf("%s = [", name);
  printf("%f, ", vgetq_lane_f32(r, 0));
  printf("%f, ", vgetq_lane_f32(r, 1));
  printf("%f, ", vgetq_lane_f32(r, 2));
  printf("%f", vgetq_lane_f32(r, 3));
  printf("]\n");
}

void print_reg_u8x4(const uint8x16x4_t r, const char *name)
{
  printf("%s = [\n", name);
  print_reg_u8(r.val[0], "0");
  print_reg_u8(r.val[1], "1");
  print_reg_u8(r.val[2], "2");
  print_reg_u8(r.val[3], "3");
  printf("]\n");
}

void print_reg_u16x4(const uint16x8x4_t r, const char *name)
{
  printf("%s = [\n", name);
  print_reg_u16(r.val[0], "0");
  print_reg_u16(r.val[1], "1");
  print_reg_u16(r.val[2], "2");
  print_reg_u16(r.val[3], "3");
  printf("]\n");
}

int blur2_do_tile_test(int x, int y, int width, int height)
{
  printf("x %d y %d\n", x, y);
  uint8x16_t raw_col_0 = vld1q_u8((uint8_t *)&cur_img(0, 0));
  print_reg_u8(raw_col_0, "raw");

  uint8x16x4_t col_0 = vld4q_u8((uint8_t *)&cur_img(0, 0));
  print_reg_u8(col_0.val[0], "col_a");
  print_reg_u8(col_0.val[1], "col_b");
  print_reg_u8(col_0.val[2], "col_g");
  print_reg_u8(col_0.val[3], "col_r");

  for (size_t i = 0; i < 4; i++)
  {
    printf("%u %u %u %u \n", extract_red(cur_img(0, i)), extract_green(cur_img(0, i)), extract_blue(cur_img(0, i)), extract_alpha(cur_img(0, i)));
  }

  return 0;
}

//TODO DES FONCTIONS POUR MOIN DE CODE

int blur2_do_tile_urrot1_simd_div9_f32(int x, int y, int width, int height)
{
  x = x+16;
  for (int i = y + 1; i < y + height - 1; i++)
  {
    // Étape 1 : Prologue
    // on charge on promouvoie et on somme (les canals) des colonnes 0 et 1

    uint8x16x4_t u8_col0_0 = vld4q_u8((uint8_t *)&cur_img(i - 1, x - 16));
    uint8x16x4_t u8_col0_1 = vld4q_u8((uint8_t *)&cur_img(i + 0, x - 16));
    uint8x16x4_t u8_col0_2 = vld4q_u8((uint8_t *)&cur_img(i + 1, x - 16));

    uint16x8x4_t u16_col0_0_low, u16_col0_0_high, u16_col0_1_low, u16_col0_1_high, u16_col0_2_low, u16_col0_2_high;

    u16_col0_0_low.val[0] = vmovl_u8(vget_low_u8(u8_col0_0.val[0]));
    u16_col0_0_low.val[1] = vmovl_u8(vget_low_u8(u8_col0_0.val[1]));
    u16_col0_0_low.val[2] = vmovl_u8(vget_low_u8(u8_col0_0.val[2]));
    u16_col0_0_low.val[3] = vmovl_u8(vget_low_u8(u8_col0_0.val[3]));

    u16_col0_0_high.val[0] = vmovl_u8(vget_high_u8(u8_col0_0.val[0]));
    u16_col0_0_high.val[1] = vmovl_u8(vget_high_u8(u8_col0_0.val[1]));
    u16_col0_0_high.val[2] = vmovl_u8(vget_high_u8(u8_col0_0.val[2]));
    u16_col0_0_high.val[3] = vmovl_u8(vget_high_u8(u8_col0_0.val[3]));

    u16_col0_1_low.val[0] = vmovl_u8(vget_low_u8(u8_col0_1.val[0]));
    u16_col0_1_low.val[1] = vmovl_u8(vget_low_u8(u8_col0_1.val[1]));
    u16_col0_1_low.val[2] = vmovl_u8(vget_low_u8(u8_col0_1.val[2]));
    u16_col0_1_low.val[3] = vmovl_u8(vget_low_u8(u8_col0_1.val[3]));

    u16_col0_1_high.val[0] = vmovl_u8(vget_high_u8(u8_col0_1.val[0]));
    u16_col0_1_high.val[1] = vmovl_u8(vget_high_u8(u8_col0_1.val[1]));
    u16_col0_1_high.val[2] = vmovl_u8(vget_high_u8(u8_col0_1.val[2]));
    u16_col0_1_high.val[3] = vmovl_u8(vget_high_u8(u8_col0_1.val[3]));

    u16_col0_2_low.val[0] = vmovl_u8(vget_low_u8(u8_col0_2.val[0]));
    u16_col0_2_low.val[1] = vmovl_u8(vget_low_u8(u8_col0_2.val[1]));
    u16_col0_2_low.val[2] = vmovl_u8(vget_low_u8(u8_col0_2.val[2]));
    u16_col0_2_low.val[3] = vmovl_u8(vget_low_u8(u8_col0_2.val[3]));

    u16_col0_2_high.val[0] = vmovl_u8(vget_high_u8(u8_col0_2.val[0]));
    u16_col0_2_high.val[1] = vmovl_u8(vget_high_u8(u8_col0_2.val[1]));
    u16_col0_2_high.val[2] = vmovl_u8(vget_high_u8(u8_col0_2.val[2]));
    u16_col0_2_high.val[3] = vmovl_u8(vget_high_u8(u8_col0_2.val[3]));

    uint16x8x4_t u16_col0_low, u16_col0_high;

    u16_col0_low.val[0] = vaddq_u16(u16_col0_1_low.val[0], vaddq_u16(u16_col0_0_low.val[0], u16_col0_2_low.val[0]));
    u16_col0_low.val[1] = vaddq_u16(u16_col0_1_low.val[1], vaddq_u16(u16_col0_0_low.val[1], u16_col0_2_low.val[1]));
    u16_col0_low.val[2] = vaddq_u16(u16_col0_1_low.val[2], vaddq_u16(u16_col0_0_low.val[2], u16_col0_2_low.val[2]));
    u16_col0_low.val[3] = vaddq_u16(u16_col0_1_low.val[3], vaddq_u16(u16_col0_0_low.val[3], u16_col0_2_low.val[3]));

    u16_col0_high.val[0] = vaddq_u16(u16_col0_1_high.val[0], vaddq_u16(u16_col0_0_high.val[0], u16_col0_2_high.val[0]));
    u16_col0_high.val[1] = vaddq_u16(u16_col0_1_high.val[1], vaddq_u16(u16_col0_0_high.val[1], u16_col0_2_high.val[1]));
    u16_col0_high.val[2] = vaddq_u16(u16_col0_1_high.val[2], vaddq_u16(u16_col0_0_high.val[2], u16_col0_2_high.val[2]));
    u16_col0_high.val[3] = vaddq_u16(u16_col0_1_high.val[3], vaddq_u16(u16_col0_0_high.val[3], u16_col0_2_high.val[3]));

     print_reg_u16x4(u16_col0_low,"u16_col0_low");
    print_reg_u16x4(u16_col0_high,"u16_col0_high");


    uint8x16x4_t u8_col1_0 = vld4q_u8((uint8_t *)&cur_img(i - 1, x));
    uint8x16x4_t u8_col1_1 = vld4q_u8((uint8_t *)&cur_img(i + 0,x));
    uint8x16x4_t u8_col1_2 = vld4q_u8((uint8_t *)&cur_img(i + 1, x));

    uint16x8x4_t u16_col1_0_low, u16_col1_0_high, u16_col1_1_low, u16_col1_1_high, u16_col1_2_low, u16_col1_2_high;

    u16_col1_0_low.val[0] = vmovl_u8(vget_low_u8(u8_col1_0.val[0]));
    u16_col1_0_low.val[1] = vmovl_u8(vget_low_u8(u8_col1_0.val[1]));
    u16_col1_0_low.val[2] = vmovl_u8(vget_low_u8(u8_col1_0.val[2]));
    u16_col1_0_low.val[3] = vmovl_u8(vget_low_u8(u8_col1_0.val[3]));

    u16_col1_0_high.val[0] = vmovl_u8(vget_high_u8(u8_col1_0.val[0]));
    u16_col1_0_high.val[1] = vmovl_u8(vget_high_u8(u8_col1_0.val[1]));
    u16_col1_0_high.val[2] = vmovl_u8(vget_high_u8(u8_col1_0.val[2]));
    u16_col1_0_high.val[3] = vmovl_u8(vget_high_u8(u8_col1_0.val[3]));

    u16_col1_1_low.val[0] = vmovl_u8(vget_low_u8(u8_col1_1.val[0]));
    u16_col1_1_low.val[1] = vmovl_u8(vget_low_u8(u8_col1_1.val[1]));
    u16_col1_1_low.val[2] = vmovl_u8(vget_low_u8(u8_col1_1.val[2]));
    u16_col1_1_low.val[3] = vmovl_u8(vget_low_u8(u8_col1_1.val[3]));

    u16_col1_1_high.val[0] = vmovl_u8(vget_high_u8(u8_col1_1.val[0]));
    u16_col1_1_high.val[1] = vmovl_u8(vget_high_u8(u8_col1_1.val[1]));
    u16_col1_1_high.val[2] = vmovl_u8(vget_high_u8(u8_col1_1.val[2]));
    u16_col1_1_high.val[3] = vmovl_u8(vget_high_u8(u8_col1_1.val[3]));

    u16_col1_2_low.val[0] = vmovl_u8(vget_low_u8(u8_col1_2.val[0]));
    u16_col1_2_low.val[1] = vmovl_u8(vget_low_u8(u8_col1_2.val[1]));
    u16_col1_2_low.val[2] = vmovl_u8(vget_low_u8(u8_col1_2.val[2]));
    u16_col1_2_low.val[3] = vmovl_u8(vget_low_u8(u8_col1_2.val[3]));

    u16_col1_2_high.val[0] = vmovl_u8(vget_high_u8(u8_col1_2.val[0]));
    u16_col1_2_high.val[1] = vmovl_u8(vget_high_u8(u8_col1_2.val[1]));
    u16_col1_2_high.val[2] = vmovl_u8(vget_high_u8(u8_col1_2.val[2]));
    u16_col1_2_high.val[3] = vmovl_u8(vget_high_u8(u8_col1_2.val[3]));

    uint16x8x4_t u16_col1_low, u16_col1_high;

    u16_col1_low.val[0] = vaddq_u16(u16_col1_1_low.val[0], vaddq_u16(u16_col1_0_low.val[0], u16_col1_2_low.val[0]));
    u16_col1_low.val[1] = vaddq_u16(u16_col1_1_low.val[1], vaddq_u16(u16_col1_0_low.val[1], u16_col1_2_low.val[1]));
    u16_col1_low.val[2] = vaddq_u16(u16_col1_1_low.val[2], vaddq_u16(u16_col1_0_low.val[2], u16_col1_2_low.val[2]));
    u16_col1_low.val[3] = vaddq_u16(u16_col1_1_low.val[3], vaddq_u16(u16_col1_0_low.val[3], u16_col1_2_low.val[3]));

    u16_col1_high.val[0] = vaddq_u16(u16_col1_1_high.val[0], vaddq_u16(u16_col1_0_high.val[0], u16_col1_2_high.val[0]));
    u16_col1_high.val[1] = vaddq_u16(u16_col1_1_high.val[1], vaddq_u16(u16_col1_0_high.val[1], u16_col1_2_high.val[1]));
    u16_col1_high.val[2] = vaddq_u16(u16_col1_1_high.val[2], vaddq_u16(u16_col1_0_high.val[2], u16_col1_2_high.val[2]));
    u16_col1_high.val[3] = vaddq_u16(u16_col1_1_high.val[3], vaddq_u16(u16_col1_0_high.val[3], u16_col1_2_high.val[3]));

     print_reg_u16x4(u16_col1_low,"u16_col1_low");
    print_reg_u16x4(u16_col1_high,"u16_col1_high");

    for (int j = x + 16; j < x + width - 16; j += 16)
    {
      // Étape 2 : Chargement des pixels
      // +
      // Etape 3 :
      // Deintervaling
      // Deinterleaving automatique avec vld4q_u8

      uint8x16x4_t u8_col2_0 = vld4q_u8((uint8_t *)&cur_img(i - 1, j + 16));
      uint8x16x4_t u8_col2_1 = vld4q_u8((uint8_t *)&cur_img(i + 0, j + 16));
      uint8x16x4_t u8_col2_2 = vld4q_u8((uint8_t *)&cur_img(i + 1, j + 16));

      // print_reg_u8x4(u8_col2_0,"u8_col2_0");

      // Étape 4 : Promotion à u16 (car registre 8 bits pas assez grand pour l'accumulation des composantes)
      uint16x8x4_t u16_col2_0_low, u16_col2_0_high, u16_col2_1_low, u16_col2_1_high, u16_col2_2_low, u16_col2_2_high;

      u16_col2_0_low.val[0] = vmovl_u8(vget_low_u8(u8_col2_0.val[0]));
      u16_col2_0_low.val[1] = vmovl_u8(vget_low_u8(u8_col2_0.val[1]));
      u16_col2_0_low.val[2] = vmovl_u8(vget_low_u8(u8_col2_0.val[2]));
      u16_col2_0_low.val[3] = vmovl_u8(vget_low_u8(u8_col2_0.val[3]));

      u16_col2_0_high.val[0] = vmovl_u8(vget_high_u8(u8_col2_0.val[0]));
      u16_col2_0_high.val[1] = vmovl_u8(vget_high_u8(u8_col2_0.val[1]));
      u16_col2_0_high.val[2] = vmovl_u8(vget_high_u8(u8_col2_0.val[2]));
      u16_col2_0_high.val[3] = vmovl_u8(vget_high_u8(u8_col2_0.val[3]));

      u16_col2_1_low.val[0] = vmovl_u8(vget_low_u8(u8_col2_1.val[0]));
      u16_col2_1_low.val[1] = vmovl_u8(vget_low_u8(u8_col2_1.val[1]));
      u16_col2_1_low.val[2] = vmovl_u8(vget_low_u8(u8_col2_1.val[2]));
      u16_col2_1_low.val[3] = vmovl_u8(vget_low_u8(u8_col2_1.val[3]));

      u16_col2_1_high.val[0] = vmovl_u8(vget_high_u8(u8_col2_1.val[0]));
      u16_col2_1_high.val[1] = vmovl_u8(vget_high_u8(u8_col2_1.val[1]));
      u16_col2_1_high.val[2] = vmovl_u8(vget_high_u8(u8_col2_1.val[2]));
      u16_col2_1_high.val[3] = vmovl_u8(vget_high_u8(u8_col2_1.val[3]));

      u16_col2_2_low.val[0] = vmovl_u8(vget_low_u8(u8_col2_2.val[0]));
      u16_col2_2_low.val[1] = vmovl_u8(vget_low_u8(u8_col2_2.val[1]));
      u16_col2_2_low.val[2] = vmovl_u8(vget_low_u8(u8_col2_2.val[2]));
      u16_col2_2_low.val[3] = vmovl_u8(vget_low_u8(u8_col2_2.val[3]));

      u16_col2_2_high.val[0] = vmovl_u8(vget_high_u8(u8_col2_2.val[0]));
      u16_col2_2_high.val[1] = vmovl_u8(vget_high_u8(u8_col2_2.val[1]));
      u16_col2_2_high.val[2] = vmovl_u8(vget_high_u8(u8_col2_2.val[2]));
      u16_col2_2_high.val[3] = vmovl_u8(vget_high_u8(u8_col2_2.val[3]));

      // print_reg_u16x4(u16_col2_0_low,"u16_col2_0_low");
      // print_reg_u16x4(u16_col2_0_high,"u16_col2_0_high");

      // Etape 5 : Reduction (pour tous les channels, reduire les 3 lignes a une seul)

      uint16x8x4_t u16_col2_low, u16_col2_high;

      u16_col2_low.val[0] = vaddq_u16(u16_col2_1_low.val[0], vaddq_u16(u16_col2_0_low.val[0], u16_col2_2_low.val[0]));
      u16_col2_low.val[1] = vaddq_u16(u16_col2_1_low.val[1], vaddq_u16(u16_col2_0_low.val[1], u16_col2_2_low.val[1]));
      u16_col2_low.val[2] = vaddq_u16(u16_col2_1_low.val[2], vaddq_u16(u16_col2_0_low.val[2], u16_col2_2_low.val[2]));
      u16_col2_low.val[3] = vaddq_u16(u16_col2_1_low.val[3], vaddq_u16(u16_col2_0_low.val[3], u16_col2_2_low.val[3]));

      u16_col2_high.val[0] = vaddq_u16(u16_col2_1_high.val[0], vaddq_u16(u16_col2_0_high.val[0], u16_col2_2_high.val[0]));
      u16_col2_high.val[1] = vaddq_u16(u16_col2_1_high.val[1], vaddq_u16(u16_col2_0_high.val[1], u16_col2_2_high.val[1]));
      u16_col2_high.val[2] = vaddq_u16(u16_col2_1_high.val[2], vaddq_u16(u16_col2_0_high.val[2], u16_col2_2_high.val[2]));
      u16_col2_high.val[3] = vaddq_u16(u16_col2_1_high.val[3], vaddq_u16(u16_col2_0_high.val[3], u16_col2_2_high.val[3]));

      print_reg_u16x4(u16_col2_low,"u16_col2_low");
      print_reg_u16x4(u16_col2_high,"u16_col2_high");


return 0;
      // Étape 6 : Addition entre les colonnes 0, 1, 2 (left right pattern)
      

      // Étape 7-8 : Promotion à f32, division par 9
      // TODO
      // float32x4x4_t avg;
      // avg.val[0] = vdivq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(sum_r))), vdupq_n_f32(9.0f));

      // Étape 9-11 : Conversion en arrière à u32, u16 et u8
      // TODO
      // uint8x16x4_t result;
      // result.val[0] = vcombine_u8(vqmovun_s16(vmovn_u32(vcvtq_u32_f32(avg.val[0]))), vqmovun_s16(vmovn_u32(vcvtq_u32_f32(avg.val[0]))));

      // Étape 12 : Stockage
      // TODO
      // vst4q_u8((uint8_t *)&next_img(i, j), result);

      // Étape 13 : Rotation des colonnes
      // TODO
    }
  }

  // Etape 14 : Bordures
  //  TODO
  //  uint32_t bsize = 2*16;
  //  compute_borders(x, y, width, height, bsize);
  return 0;
}

int blur2_do_tile_urrot1_simd_div9_u16(int x, int y, int width, int height)
{
  // TODO
  return 0;
}

int blur2_do_tile_urrot2_simd_div9_f32(int x, int y, int width, int height)
{
  // TODO
  return 0;
}

int blur2_do_tile_urrot2_simd_div9_u16(int x, int y, int width, int height)
{
  // TODO
  return 0;
}

int blur2_do_tile_urrot2_simd_div8_u16(int x, int y, int width, int height)
{
  // TODO
  return 0;
}

#endif