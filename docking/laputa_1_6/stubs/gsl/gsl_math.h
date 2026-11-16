// KANM docking stub for GSL math helpers.
#ifndef KANM_STUB_GSL_MATH_H
#define KANM_STUB_GSL_MATH_H
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
inline int gsl_isnan(double x) { return std::isnan(x) ? 1 : 0; }
inline int gsl_isinf(double x) { return std::isinf(x) ? (x > 0 ? 1 : -1) : 0; }
#endif
