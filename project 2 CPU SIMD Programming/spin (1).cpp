/*
  ALBISSON Damien
  DAUVET-DIAKHATE Haron
*/

#include <math.h>
#include <omp.h>

#include "global.h"
#include "img_data.h"
#include "cppdefs.h"
#include "mipp.h"

static void rotate(void);

#ifdef ENABLE_VECTO
#include <iostream>
static bool is_printed = false;
static void print_simd_info() {
  if (!is_printed) {
    std::cout << "SIMD infos:" << std::endl;
    std::cout << " - Instr. type:       " << mipp::InstructionType << std::endl;
    std::cout << " - Instr. full type:  " << mipp::InstructionFullType << std::endl;
    std::cout << " - Instr. version:    " << mipp::InstructionVersion << std::endl;
    std::cout << " - Instr. size:       " << mipp::RegisterSizeBit << " bits"
              << std::endl;
    std::cout << " - Instr. lanes:      " << mipp::Lanes << std::endl;
    std::cout << " - 64-bit support:    " << (mipp::Support64Bit ? "yes" : "no")
              << std::endl;
    std::cout << " - Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no")
              << std::endl;
    auto ext = mipp::InstructionExtensions();
    if (ext.size() > 0) {
      std::cout << " - Instr. extensions: {";
      for (auto i = 0; i < (int)ext.size(); i++)
        std::cout << ext[i] << (i < ((int)ext.size() - 1) ? ", " : "");
      std::cout << "}" << std::endl;
    }
    std::cout << std::endl;
    is_printed = true;
  }
}
#endif

// Global variables
static float base_angle = 0.f;
static int color_a_r = 255, color_a_g = 255, color_a_b = 0, color_a_a = 255;
static int color_b_r = 0, color_b_g = 0, color_b_b = 255, color_b_a = 255;

// ----------------------------------------------------------------------------
// -------------------------------------------------------- INITIAL SEQ VERSION
// ----------------------------------------------------------------------------

// The image is a two-dimension array of size of DIM x DIM. Each pixel is of
// type 'unsigned' and store the color information following a RGBA layout (4
// bytes). Pixel at line 'l' and column 'c' in the current image can be accessed
// using cur_img (l, c).

// The kernel returns 0, or the iteration step at which computation has
// completed (e.g. stabilized).

// Computation of one pixel
/*


 taskset -c 1 ./run -k spin -n -i 50 -v seq

            DEN     AS7

-O0         8720   10346

-O1         7404   7211

-O2         7301   6644

-O3         7454   6605

-ffast-math 5534   5572

// 
*/
static unsigned compute_color(int i, int j) {

  // mipp::Reg<float> rAtan = 

  float atan2f_in1 = (float)DIM / 2.f - (float)i;
  float atan2f_in2 = (float)j - (float)DIM / 2.f;
  float angle = atan2f(atan2f_in1, atan2f_in2) + M_PI + base_angle;

  float ratio = fabsf((fmodf(angle, M_PI / 4.f) - (M_PI / 8.f)) / (M_PI / 8.f));

  int r = color_a_r * ratio + color_b_r * (1.f - ratio);
  int g = color_a_g * ratio + color_b_g * (1.f - ratio);
  int b = color_a_b * ratio + color_b_b * (1.f - ratio);
  int a = color_a_a * ratio + color_b_a * (1.f - ratio);

  return rgba(r, g, b, a);
}

static void rotate(void) {
  base_angle = fmodf(base_angle + (1.f / 180.f) * M_PI, M_PI);
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run --size 1024 --kernel spin --variant seq
// or
// ./run -s 1024 -k spin -v seq
//
EXTERN unsigned spin_compute_seq(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j++)
        cur_img(i, j) = compute_color(i, j);

    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// --------------------------------------------------------- APPROX SEQ VERSION
// ----------------------------------------------------------------------------

// arctangent, result between -pi/2 et pi/2 radians
static float atanf_approx(float x)
{
  return x * M_PI / 4.f + 0.273f * x * (1.f - fabsf(x));
}
// arctangent, result between -pi et pi radians
static float atan2f_approx(float y, float x)
{
  float ay = fabsf(y);
  float ax = fabsf(x);
  int invert = ay > ax;
  float z = invert ? ax / ay : ay / ax; // [0,1]
  float th = atanf_approx(z);           // [0,pi/4]
  if (invert)
    th = M_PI_2 - th; // [0,pi/2]
  if (x < 0)
    th = M_PI - th; // [0,pi]
  if (y < 0)
    th = -th;
  return th;
}
// remainder of floating-point division
static float fmodf_approx(float x, float y)
{
  return x - trunc(x / y) * y;
}

// Computation of one pixel
/*
taskset -c 0 ./run -k spin -n -i 50 -v approx

atan cost many time to compute, so we reduce the call to an simple operations 
idem for the remainder of euclidian div
less precision -> faster

  DEN     AS7

  2000    2635

*/
static unsigned compute_color_approx(int i, int j) {
  float atan2f_in1 = (float)DIM / 2.f - (float)i;
  float atan2f_in2 = (float)j - (float)DIM / 2.f;
  float angle = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;

  float ratio = fabsf((fmodf_approx(angle, M_PI / 4.f) - (M_PI / 8.f)) / (M_PI / 8.f));

  int r = color_a_r * ratio + color_b_r * (1.f - ratio);
  int g = color_a_g * ratio + color_b_g * (1.f - ratio);
  int b = color_a_b * ratio + color_b_b * (1.f - ratio);
  int a = color_a_a * ratio + color_b_a * (1.f - ratio);

  return rgba(r, g, b, a);
}

EXTERN unsigned spin_compute_approx(unsigned nb_iter) {
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j++)
        cur_img(i, j) = compute_color_approx(i, j);

    rotate();  // Slightly increase the base angle
  }

  return 0;
}

#ifdef ENABLE_VECTO
// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 0
// ----------------------------------------------------------------------------

/*
  taskset -c 1 ./run -k spin -n -i 50 -v simd_v0

  DEN     AS7

  4470    3000

It's slower than the approx version with the two kind of core, it can be explained here because
we use the vector register only to make it sequential like before (no real interest) ...
The fact the denver cores is slower is surely due to the fact, A57 work better with this kind of register
(or maybe it just doesn't have this kind of register and simulate the actions with normal register ???)
*/

// Computation of one pixel
static mipp::Reg<int> compute_color_simd_v0(mipp::Reg<int> r_i,
                                            mipp::Reg<int> r_j)
{

  int tab_result[mipp::N<int>()];

  for (int i = 0; i < mipp::N<int>(); i++) {
    float atan2f_in1 = (float)DIM / 2.f - (float)r_i[i];
    float atan2f_in2 = (float)r_j[i] - (float)DIM / 2.f;
    float angle = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;

    float ratio = fabsf((fmodf_approx(angle, M_PI / 4.f) - (M_PI / 8.f)) / (M_PI / 8.f));

    int r = color_a_r * ratio + color_b_r * (1.f - ratio);
    int g = color_a_g * ratio + color_b_g * (1.f - ratio);
    int b = color_a_b * ratio + color_b_b * (1.f - ratio);
    int a = color_a_a * ratio + color_b_a * (1.f - ratio);

    tab_result[i] = rgba(r, g, b, a);
  }

  return mipp::Reg<int>(tab_result);
}

EXTERN unsigned spin_compute_simd_v0(unsigned nb_iter) {
  print_simd_info();
  int tab_j[mipp::N<int>()];

  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v0(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }

  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 1
// ----------------------------------------------------------------------------

static inline mipp::Reg<float> fmodf_approx_simd(mipp::Reg<float> r_x,
                                                 mipp::Reg<float> r_y) {
  return r_x - mipp::trunc(r_x / r_y) * r_y;
}

/*

Version with the simd of fmodf_approx

  Denver    A57
   3300     2153

Not bad we gain 1 sec on the kind of cores
*/

// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v1(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  int tab_result[mipp::N<int>()];
  float tab_angle[mipp::N<float>()];

  for (int i = 0; i < mipp::N<int>(); i++) {
    float atan2f_in1 = (float)DIM / 2.f - (float)r_i[i];
    float atan2f_in2 = (float)r_j[i] - (float)DIM / 2.f;
    float angle = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;
    tab_angle[i] = angle;
  }

  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(tab_angle), mipp::Reg<float>(M_PI / 4.f)) - (M_PI / 8.f)) / (M_PI / 8.f));

  for (int i = 0; i < mipp::N<int>(); i++) {
    int r = color_a_r * ratio[i] + color_b_r * (1.f - ratio[i]);
    int g = color_a_g * ratio[i] + color_b_g * (1.f - ratio[i]);
    int b = color_a_b * ratio[i] + color_b_b * (1.f - ratio[i]);
    int a = color_a_a * ratio[i] + color_b_a * (1.f - ratio[i]);

    tab_result[i] = rgba(r, g, b, a);
  }

  return mipp::Reg<int>(tab_result);
}

EXTERN unsigned spin_compute_simd_v1(unsigned nb_iter) {
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v1(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 2
// ----------------------------------------------------------------------------

static inline mipp::Reg<int> rgba_simd(mipp::Reg<int> r_r, mipp::Reg<int> r_g,
                                       mipp::Reg<int> r_b, mipp::Reg<int> r_a) {
  return mipp::orb(mipp::orb(mipp::orb((mipp::lshift(r_r, 24)), mipp::lshift(r_g, 16)), mipp::lshift(r_b, 8)), r_a);
}

/*

  Denver     A57
   2690     2000

Again little bit faster, because we parallelize our operations with the help of vector register 
and SIMD operations on them
*/
// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v2(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  float tab_angle[mipp::N<float>()];

  for (int i = 0; i < mipp::N<int>(); i++) {
    float atan2f_in1 = (float)DIM / 2.f - (float)r_i[i];
    float atan2f_in2 = (float)r_j[i] - (float)DIM / 2.f;
    float angle = atan2f_approx(atan2f_in1, atan2f_in2) + M_PI + base_angle;
    tab_angle[i] = angle;
  }

  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(mipp::Reg<float>(tab_angle), mipp::Reg<float>(M_PI / 4.f)) - (M_PI / 8.f)) / (M_PI / 8.f));

  mipp::Reg<int> r = mipp::cvt<float,int>(mipp::Reg<float>(color_a_r) * ratio + mipp::Reg<float>(color_b_r) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> g = mipp::cvt<float,int>(mipp::Reg<float>(color_a_g) * ratio + mipp::Reg<float>(color_b_g) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> b = mipp::cvt<float,int>(mipp::Reg<float>(color_a_b) * ratio + mipp::Reg<float>(color_b_b) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> a = mipp::cvt<float,int>(mipp::Reg<float>(color_a_a) * ratio + mipp::Reg<float>(color_b_a) * (mipp::Reg<float>(1.f) - ratio));

  return rgba_simd(r, g, b, a);
}

EXTERN unsigned spin_compute_simd_v2(unsigned nb_iter) {
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v2(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 3
// ----------------------------------------------------------------------------

static inline mipp::Reg<float> atanf_approx_simd(mipp::Reg<float> r_z) {
  return r_z * mipp::Reg<float>(M_PI) / mipp::Reg<float>(4.f) + mipp::Reg<float>(0.273f) * r_z * (mipp::Reg<float>(1.f) - mipp::abs(r_z));
}

static inline mipp::Reg<float> atan2f_approx_simd(mipp::Reg<float> r_y,
                                                  mipp::Reg<float> r_x) {
  mipp::Reg<float> ay = mipp::abs(r_y);
  mipp::Reg<float> ax = mipp::abs(r_x);
  mipp::Msk<mipp::N<float>()> invert = ay > ax;
  mipp::Reg<float> r_z = mipp::blend(ax/ay, ay/ax, invert);

  mipp::Reg<float> th = atanf_approx_simd(r_z);           // [0,pi/4]
  th = mipp::blend(mipp::Reg<float>(M_PI_2) - th, th, invert);

  mipp::Msk<mipp::N<float>()> x0 = r_x < mipp::Reg<float>((float) 0);
  th = mipp::blend(mipp::Reg<float>(M_PI) - th, th, x0);

  mipp::Msk<mipp::N<float>()> y0 = r_y < mipp::Reg<float>((float) 0);
  th = mipp::blend(-th, th, y0);

  return th;
}

/*
  Denver   A57
   1385    549

Pretty impressive improvement 
*/
// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v3(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  mipp::Reg<float> atan2f_in1 = mipp::Reg<float>((float)DIM / 2.f) - mipp::cvt<int, float>(r_i);
  mipp::Reg<float> atan2f_in2 = mipp::cvt<int, float>(r_j) - mipp::Reg<float>((float)DIM / 2.f);
  mipp::Reg<float> angle = atan2f_approx_simd(atan2f_in1, atan2f_in2) + mipp::Reg<float>(M_PI) + base_angle;

  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(angle, mipp::Reg<float>(M_PI / 4.f)) - (M_PI / 8.f)) / (M_PI / 8.f));

  mipp::Reg<int> r = mipp::cvt<float,int>(mipp::Reg<float>(color_a_r) * ratio + mipp::Reg<float>(color_b_r) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> g = mipp::cvt<float,int>(mipp::Reg<float>(color_a_g) * ratio + mipp::Reg<float>(color_b_g) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> b = mipp::cvt<float,int>(mipp::Reg<float>(color_a_b) * ratio + mipp::Reg<float>(color_b_b) * (mipp::Reg<float>(1.f) - ratio));
  mipp::Reg<int> a = mipp::cvt<float,int>(mipp::Reg<float>(color_a_a) * ratio + mipp::Reg<float>(color_b_a) * (mipp::Reg<float>(1.f) - ratio));

  return rgba_simd(r, g, b, a);
}

EXTERN unsigned spin_compute_simd_v3(unsigned nb_iter) {
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v3(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 4
// ----------------------------------------------------------------------------

/*
  Denver   A57
   1380    550 

No difference in execution time but a lot of pleasure, maybe because the inlining 
was already done by the -O3 optimization, and the vector register was already use 
when it was usefull and not just when a object mipp::Reg was created 
*/
// Computation of one pixel
static inline mipp::Reg<int> compute_color_simd_v4(mipp::Reg<int> r_i,
                                                   mipp::Reg<int> r_j)
{
  float dim_div_2f = (float)DIM / 2.f;
  float mpi_div_8f = M_PI / 8.f;

  mipp::Reg<float> atan2f_in1 = -mipp::cvt<int, float>(r_i) + dim_div_2f;
  mipp::Reg<float> atan2f_in2 = mipp::cvt<int, float>(r_j) - dim_div_2f;

  // inlining of atanf_approx_simd and atan2f_approx_simd func

  mipp::Reg<float> ay = mipp::abs(atan2f_in1);
  mipp::Reg<float> ax = mipp::abs(atan2f_in2);
  mipp::Msk<mipp::N<float>()> invert = ay > ax;
  mipp::Reg<float> r_z = mipp::blend(ax/ay, ay/ax, invert);

  mipp::Reg<float> th = r_z * ((float) M_PI) / 4.f + r_z * 0.273f * (mipp::abs(r_z) - 1.f);
  th = mipp::blend(-th + M_PI_2, th, invert);

  mipp::Msk<mipp::N<float>()> x0 = atan2f_in2 < 0.f;
  th = mipp::blend(-th + M_PI, th, x0);

  mipp::Msk<mipp::N<float>()> y0 = atan2f_in1 < 0.f;
  th = mipp::blend(-th, th, y0);

  // end inlining

  mipp::Reg<float> angle = th + M_PI + base_angle;
  mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(angle, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

  mipp::Reg<float> neg_ration_add_1 = (-ratio + 1.f);
  mipp::Reg<int> r = mipp::cvt<float,int>(ratio * color_a_r + neg_ration_add_1 * color_b_r);
  mipp::Reg<int> g = mipp::cvt<float,int>(ratio * color_a_g + neg_ration_add_1 * color_b_g);
  mipp::Reg<int> b = mipp::cvt<float,int>(ratio * color_a_b + neg_ration_add_1 * color_b_b);
  mipp::Reg<int> a = mipp::cvt<float,int>(ratio * color_a_a + neg_ration_add_1 * color_b_a);

  return rgba_simd(r, g, b, a);
}

EXTERN unsigned spin_compute_simd_v4(unsigned nb_iter) {
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++)
      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);
        mipp::Reg<int> r_result =
            compute_color_simd_v4(mipp::Reg<int>(i), mipp::Reg<int>(tab_j));

        r_result.store(img_out_ptr);
      }

    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 5
// ----------------------------------------------------------------------------

/*
  Denver  A57
   1300   520

Maybe a little bit faster, but not so much
*/
EXTERN unsigned spin_compute_simd_v5(unsigned nb_iter) {

  float dim_div_2f = (float)DIM / 2.f;
  float mpi_div_8f = M_PI / 8.f;
  mipp::Reg<float> r_mpi_div_4f = M_PI / 4.f;

  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++) {

      mipp::Reg<float> atan2f_in1 = -mipp::cvt<int, float>(mipp::Reg<int>(i)) + dim_div_2f;
      mipp::Reg<float> ay = mipp::abs(atan2f_in1);
      mipp::Msk<mipp::N<float>()> y0 = atan2f_in1 < 0.f;

      for (unsigned j = 0; j < DIM; j += mipp::N<float>()) {
        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr = (int*)&cur_img(i, j);

        mipp::Reg<float> atan2f_in2 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j)) - dim_div_2f;

        // inlining of atanf_approx_simd and atan2f_approx_simd func

        mipp::Reg<float> ax = mipp::abs(atan2f_in2);
        mipp::Msk<mipp::N<float>()> invert = ay > ax;
        mipp::Reg<float> r_z = mipp::blend(ax/ay, ay/ax, invert);

        mipp::Reg<float> th = mipp::fmadd(r_z, r_mpi_div_4f, r_z * 0.273f * (mipp::abs(r_z) - 1.f));
        th = mipp::blend(-th + M_PI_2, th, invert);

        mipp::Msk<mipp::N<float>()> x0 = atan2f_in2 < 0.f;
        th = mipp::blend(-th + M_PI, th, x0);

        th = mipp::blend(-th, th, y0);

        // end inlining

        mipp::Reg<float> angle = th + M_PI + base_angle;
        mipp::Reg<float> ratio = mipp::abs((fmodf_approx_simd(angle, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

        mipp::Reg<float> neg_ration_add_1 = (-ratio + 1.f);
        mipp::Reg<int> r = mipp::cvt<float,int>(mipp::fmadd(ratio, mipp::Reg<float>(color_a_r), neg_ration_add_1 * color_b_r));
        mipp::Reg<int> g = mipp::cvt<float,int>(mipp::fmadd(ratio, mipp::Reg<float>(color_a_g), neg_ration_add_1 * color_b_g));
        mipp::Reg<int> b = mipp::cvt<float,int>(mipp::fmadd(ratio, mipp::Reg<float>(color_a_b), neg_ration_add_1 * color_b_b));
        mipp::Reg<int> a = mipp::cvt<float,int>(mipp::fmadd(ratio, mipp::Reg<float>(color_a_a), neg_ration_add_1 * color_b_a));

        rgba_simd(r, g, b, a).store(img_out_ptr);
      }
    }
    rotate();  // Slightly increase the base angle
  }
  return 0;
}

// ----------------------------------------------------------------------------
// ------------------------------------------------------------- SIMD VERSION 6
// ----------------------------------------------------------------------------

EXTERN unsigned spin_compute_simd_v6u2(unsigned nb_iter) {

  float dim_div_2f = (float)DIM / 2.f;
  float mpi_div_8f = M_PI / 8.f;
  mipp::Reg<float> r_mpi_div_4f = M_PI / 4.f;
  
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++) {

      mipp::Reg<float> atan2f_in1 = -mipp::cvt<int, float>(mipp::Reg<int>(i)) + dim_div_2f;
      mipp::Reg<float> ay = mipp::abs(atan2f_in1);
      mipp::Msk<mipp::N<float>()> y0 = atan2f_in1 < 0.f;

      for (unsigned j = 0; j < DIM; j += mipp::N<float>()*2) {

        // first loop

        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr_j0 = (int*)&cur_img(i, j);

        mipp::Reg<float> atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j)) - dim_div_2f;

        mipp::Reg<float> ax_j0 = mipp::abs(atan2f_in2_j0);
        mipp::Msk<mipp::N<float>()> invert_j0 = ay > ax_j0;
        mipp::Reg<float> r_z_j0 = mipp::blend(ax_j0/ay, ay/ax_j0, invert_j0);

        mipp::Reg<float> th_j0 = mipp::fmadd(r_z_j0, r_mpi_div_4f, r_z_j0 * 0.273f * (mipp::abs(r_z_j0) - 1.f));
        th_j0 = mipp::blend(-th_j0 + M_PI_2, th_j0, invert_j0);

        mipp::Msk<mipp::N<float>()> x0_j0 = atan2f_in2_j0 < 0.f;
        th_j0 = mipp::blend(-th_j0 + M_PI, th_j0, x0_j0);

        th_j0 = mipp::blend(-th_j0, th_j0, y0);

        mipp::Reg<float> angle_j0 = th_j0 + M_PI + base_angle;
        mipp::Reg<float> ratio_j0 = mipp::abs((fmodf_approx_simd(angle_j0, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

        mipp::Reg<float> neg_ration_add_1 = (-ratio_j0 + 1.f);
        mipp::Reg<int> r = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_r), neg_ration_add_1 * color_b_r));
        mipp::Reg<int> g = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_g), neg_ration_add_1 * color_b_g));
        mipp::Reg<int> b = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_b), neg_ration_add_1 * color_b_b));
        mipp::Reg<int> a = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_a), neg_ration_add_1 * color_b_a));

        rgba_simd(r, g, b, a).store(img_out_ptr_j0);

        // second loop

        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + 4 + jj;
        img_out_ptr_j0 = (int*)&cur_img(i, j+4);

        atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j)) - dim_div_2f;

        ax_j0 = mipp::abs(atan2f_in2_j0);
        invert_j0 = ay > ax_j0;
        r_z_j0 = mipp::blend(ax_j0/ay, ay/ax_j0, invert_j0);

        th_j0 = mipp::fmadd(r_z_j0, r_mpi_div_4f, r_z_j0 * 0.273f * (mipp::abs(r_z_j0) - 1.f));
        th_j0 = mipp::blend(-th_j0 + M_PI_2, th_j0, invert_j0);

        x0_j0 = atan2f_in2_j0 < 0.f;
        th_j0 = mipp::blend(-th_j0 + M_PI, th_j0, x0_j0);

        th_j0 = mipp::blend(-th_j0, th_j0, y0);

        angle_j0 = th_j0 + M_PI + base_angle;
        ratio_j0 = mipp::abs((fmodf_approx_simd(angle_j0, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

        neg_ration_add_1 = (-ratio_j0 + 1.f);
        r = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_r), neg_ration_add_1 * color_b_r));
        g = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_g), neg_ration_add_1 * color_b_g));
        b = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_b), neg_ration_add_1 * color_b_b));
        a = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_a), neg_ration_add_1 * color_b_a));

        rgba_simd(r, g, b, a).store(img_out_ptr_j0);
      }
    }
    rotate();  // Slightly increase the base angle
  }
  return 0;
}

/*
  Denver   A57
   1300    480

We gain again a little bit of time with the A57 cores but not with the Denver
Surely because the denver already made these operations inside with the translator
the same reason who cause some strange result during TP1
*/
EXTERN unsigned spin_compute_simd_v6u4(unsigned nb_iter) {

  float dim_div_2f = (float)DIM / 2.f;
  float mpi_div_8f = M_PI / 8.f;
  mipp::Reg<float> r_mpi_div_4f = M_PI / 4.f;
  
  int tab_j[mipp::N<int>()];
  for (unsigned it = 1; it <= nb_iter; it++) {
    for (unsigned i = 0; i < DIM; i++) {

      mipp::Reg<float> atan2f_in1 = -mipp::cvt<int, float>(mipp::Reg<int>(i)) + dim_div_2f;
      mipp::Reg<float> ay = mipp::abs(atan2f_in1);
      mipp::Msk<mipp::N<float>()> y0 = atan2f_in1 < 0.f;

      for (unsigned j = 0; j < DIM; j += mipp::N<float>()*4) {

        // first loop

        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + jj;
        int* img_out_ptr_j0 = (int*)&cur_img(i, j);

        mipp::Reg<float> atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j)) - dim_div_2f;

        mipp::Reg<float> ax_j0 = mipp::abs(atan2f_in2_j0);
        mipp::Msk<mipp::N<float>()> invert_j0 = ay > ax_j0;
        mipp::Reg<float> r_z_j0 = mipp::blend(ax_j0/ay, ay/ax_j0, invert_j0);

        mipp::Reg<float> th_j0 = mipp::fmadd(r_z_j0, r_mpi_div_4f, r_z_j0 * 0.273f * (mipp::abs(r_z_j0) - 1.f));
        th_j0 = mipp::blend(-th_j0 + M_PI_2, th_j0, invert_j0);

        mipp::Msk<mipp::N<float>()> x0_j0 = atan2f_in2_j0 < 0.f;
        th_j0 = mipp::blend(-th_j0 + M_PI, th_j0, x0_j0);

        th_j0 = mipp::blend(-th_j0, th_j0, y0);

        mipp::Reg<float> angle_j0 = th_j0 + M_PI + base_angle;
        mipp::Reg<float> ratio_j0 = mipp::abs((fmodf_approx_simd(angle_j0, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

        mipp::Reg<float> neg_ration_add_1 = (-ratio_j0 + 1.f);
        mipp::Reg<int> r = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_r), neg_ration_add_1 * color_b_r));
        mipp::Reg<int> g = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_g), neg_ration_add_1 * color_b_g));
        mipp::Reg<int> b = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_b), neg_ration_add_1 * color_b_b));
        mipp::Reg<int> a = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_a), neg_ration_add_1 * color_b_a));

        rgba_simd(r, g, b, a).store(img_out_ptr_j0);

        // loop 2

        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + 4 + jj;
        img_out_ptr_j0 = (int*)&cur_img(i, j+4);

        atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j)) - dim_div_2f;

        ax_j0 = mipp::abs(atan2f_in2_j0);
        invert_j0 = ay > ax_j0;
        r_z_j0 = mipp::blend(ax_j0/ay, ay/ax_j0, invert_j0);

        th_j0 = mipp::fmadd(r_z_j0, r_mpi_div_4f, r_z_j0 * 0.273f * (mipp::abs(r_z_j0) - 1.f));
        th_j0 = mipp::blend(-th_j0 + M_PI_2, th_j0, invert_j0);

        x0_j0 = atan2f_in2_j0 < 0.f;
        th_j0 = mipp::blend(-th_j0 + M_PI, th_j0, x0_j0);

        th_j0 = mipp::blend(-th_j0, th_j0, y0);

        angle_j0 = th_j0 + M_PI + base_angle;
        ratio_j0 = mipp::abs((fmodf_approx_simd(angle_j0, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

        neg_ration_add_1 = (-ratio_j0 + 1.f);
        r = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_r), neg_ration_add_1 * color_b_r));
        g = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_g), neg_ration_add_1 * color_b_g));
        b = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_b), neg_ration_add_1 * color_b_b));
        a = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_a), neg_ration_add_1 * color_b_a));

        rgba_simd(r, g, b, a).store(img_out_ptr_j0);

        // loop 3

        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + 8 + jj;
        img_out_ptr_j0 = (int*)&cur_img(i, j+8);

        atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j)) - dim_div_2f;

        ax_j0 = mipp::abs(atan2f_in2_j0);
        invert_j0 = ay > ax_j0;
        r_z_j0 = mipp::blend(ax_j0/ay, ay/ax_j0, invert_j0);

        th_j0 = mipp::fmadd(r_z_j0, r_mpi_div_4f, r_z_j0 * 0.273f * (mipp::abs(r_z_j0) - 1.f));
        th_j0 = mipp::blend(-th_j0 + M_PI_2, th_j0, invert_j0);

        x0_j0 = atan2f_in2_j0 < 0.f;
        th_j0 = mipp::blend(-th_j0 + M_PI, th_j0, x0_j0);

        th_j0 = mipp::blend(-th_j0, th_j0, y0);

        angle_j0 = th_j0 + M_PI + base_angle;
        ratio_j0 = mipp::abs((fmodf_approx_simd(angle_j0, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

        neg_ration_add_1 = (-ratio_j0 + 1.f);
        r = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_r), neg_ration_add_1 * color_b_r));
        g = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_g), neg_ration_add_1 * color_b_g));
        b = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_b), neg_ration_add_1 * color_b_b));
        a = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_a), neg_ration_add_1 * color_b_a));

        rgba_simd(r, g, b, a).store(img_out_ptr_j0);

        // loop 4

        for (unsigned jj = 0; jj < mipp::N<float>(); jj++) tab_j[jj] = j + 12 + jj;
        img_out_ptr_j0 = (int*)&cur_img(i, j+12);

        atan2f_in2_j0 = mipp::cvt<int, float>(mipp::Reg<int>(tab_j)) - dim_div_2f;

        ax_j0 = mipp::abs(atan2f_in2_j0);
        invert_j0 = ay > ax_j0;
        r_z_j0 = mipp::blend(ax_j0/ay, ay/ax_j0, invert_j0);

        th_j0 = mipp::fmadd(r_z_j0, r_mpi_div_4f, r_z_j0 * 0.273f * (mipp::abs(r_z_j0) - 1.f));
        th_j0 = mipp::blend(-th_j0 + M_PI_2, th_j0, invert_j0);

        x0_j0 = atan2f_in2_j0 < 0.f;
        th_j0 = mipp::blend(-th_j0 + M_PI, th_j0, x0_j0);

        th_j0 = mipp::blend(-th_j0, th_j0, y0);

        angle_j0 = th_j0 + M_PI + base_angle;
        ratio_j0 = mipp::abs((fmodf_approx_simd(angle_j0, M_PI / 4.f) - mpi_div_8f) / mpi_div_8f);

        neg_ration_add_1 = (-ratio_j0 + 1.f);
        r = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_r), neg_ration_add_1 * color_b_r));
        g = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_g), neg_ration_add_1 * color_b_g));
        b = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_b), neg_ration_add_1 * color_b_b));
        a = mipp::cvt<float,int>(mipp::fmadd(ratio_j0, mipp::Reg<float>(color_a_a), neg_ration_add_1 * color_b_a));

        rgba_simd(r, g, b, a).store(img_out_ptr_j0);
      }
    }
    rotate();  // Slightly increase the base angle
  }
  return 0;
}

#endif /* ENABLE_VECTO */
