// KANM docking stub for GSL random numbers. The harness supplies a seeded
// std::mt19937_64 generator. Deterministic docking configurations use chances
// of exactly 0 or 1, for which any generator on [0, 1) gives identical results.
#ifndef KANM_STUB_GSL_RNG_H
#define KANM_STUB_GSL_RNG_H
struct gsl_rng;
double gsl_rng_uniform(const gsl_rng* r);
unsigned long gsl_rng_get(const gsl_rng* r);
unsigned long gsl_rng_uniform_int(const gsl_rng* r, unsigned long n);
#endif
