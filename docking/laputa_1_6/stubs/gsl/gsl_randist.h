// KANM docking stub for the GSL densities and samplers used by Laputa 1.6.
// Densities follow the GSL definitions (see harness.cpp).
#ifndef KANM_STUB_GSL_RANDIST_H
#define KANM_STUB_GSL_RANDIST_H
#include "gsl/gsl_rng.h"
double gsl_ran_beta_pdf(double x, double a, double b);
double gsl_ran_gaussian_pdf(double x, double sigma);
unsigned int gsl_ran_binomial(const gsl_rng* r, double p, unsigned int n);
#endif
