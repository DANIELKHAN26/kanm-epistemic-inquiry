// KANM docking harness for the Laputa 1.6 model core.
//
// Copyright (C) 2026 KANM contributors. GPL-3.0-only (see LICENSE).
//
// This program links, unmodified, the model files of the archived Laputa 1.6
// source (Society.cpp, Inquirer.cpp, Link.cpp, Trust.cpp, Amount.h,
// Distribution.cpp, MetaDistribution.cpp, Parameters.cpp, StatisticsBlock.cpp)
// and drives Society::evolve headlessly. Nothing in this file implements the
// model's credence, trust or messaging rules; those come from the Laputa files.
//
// What this file does supply:
//   * stubs for the GUI/application layer (App, fl_alert, societyWindow);
//   * the few GSL functions the core calls (see the GSL section below);
//   * Simulation::Simulation and two Simulation members that only affect
//     logging and statistics (addToLog, individualEValue);
//   * resetSimulation(), which re-states the per-run initialisation that
//     Laputa performs in Simulation::reset (Simulation.cpp lines 126-134 of the
//     1.6 archive): evidence counters are set to -1 and, for links with a
//     new-evidence policy, the initial link message is derived from the
//     source's starting belief with *strict* threshold comparisons.
//
// Input (whitespace separated, from stdin):
//   mode run
//   seed <unsigned integer>
//   steps <n>
//   inquirers <k>
//     k lines: belief inquiryChance inquiryAccuracy updateInquiryTrust(0/1) v0 .. v48
//   links <m>
//     m lines: source target listenChance threshold updateTrust(0/1)
//              evidencePolicy(0 none, 1 inquiry, 2 any) countPriorAsEvidence(0/1) v0 .. v48
// or
//   mode preset <value> <sharpness>        (TrustFunction::setValueSharpness)
//   mode metadefault <distr_dir> <weight>  (default batch trust metadistribution)
//
// Output: one JSON object per line.

#include "App.h"
#include "Society.h"
#include "Inquirer.h"
#include "Link.h"
#include "Trust.h"
#include "Distribution.h"
#include "MetaDistribution.h"
#include "SocietySetup.h"
#include "Simulation.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

//----------------------------------------------------------------------------
// GSL replacements.
//
// Random numbers: a seeded std::mt19937_64. Laputa's own generator is not
// reproduced, so stochastic runs are compared statistically, never draw by
// draw. Deterministic docking uses chances of exactly 0 or 1, where the
// comparison `uniform < chance` has the same outcome for every generator with
// values in [0, 1).
//
// Densities: written from the standard definitions used by GSL
// (gsl_ran_beta_pdf, gsl_ran_gaussian_pdf, gsl_cdf_gaussian_P).
//----------------------------------------------------------------------------

struct gsl_rng { std::mt19937_64 engine; };
static gsl_rng harnessRng;
gsl_rng *rng = &harnessRng;

double gsl_rng_uniform(const gsl_rng *r) {
	gsl_rng *m = const_cast<gsl_rng *>(r);
	return (double)(m->engine() >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
}
unsigned long gsl_rng_get(const gsl_rng *r) {
	return (unsigned long)(const_cast<gsl_rng *>(r)->engine() & 0xFFFFFFFFul);
}
unsigned long gsl_rng_uniform_int(const gsl_rng *r, unsigned long n) {
	return (unsigned long)(gsl_rng_uniform(r) * (double)n);
}
unsigned int gsl_ran_binomial(const gsl_rng *r, double p, unsigned int n) {
	unsigned int k = 0;
	for (unsigned int i = 0; i < n; ++i) if (gsl_rng_uniform(r) < p) ++k;
	return k;
}
double gsl_ran_beta_pdf(double x, double a, double b) {
	if (x < 0 || x > 1) return 0;
	double lognorm = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
	if (x == 0.0 || x == 1.0) {
		if (a > 1.0 && b > 1.0) return 0.0;
		return std::exp(lognorm) * std::pow(x, a - 1) * std::pow(1 - x, b - 1);
	}
	return std::exp(lognorm + (a - 1) * std::log(x) + (b - 1) * std::log1p(-x));
}
double gsl_ran_gaussian_pdf(double x, double sigma) {
	double u = x / std::fabs(sigma);
	return (1.0 / (std::sqrt(2.0 * M_PI) * std::fabs(sigma))) * std::exp(-u * u / 2.0);
}
double gsl_cdf_gaussian_P(double x, double sigma) {
	return 0.5 * std::erfc(-x / (sigma * std::sqrt(2.0)));
}
static double betaContinuedFraction(double a, double b, double x) {
	const double tiny = 1e-300;
	double c = 1.0, d = 1.0 - (a + b) * x / (a + 1.0);
	if (std::fabs(d) < tiny) d = tiny;
	d = 1.0 / d;
	double h = d;
	for (int m = 1; m < 10000; ++m) {
		double m2 = 2.0 * m;
		double aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2));
		d = 1.0 + aa * d; if (std::fabs(d) < tiny) d = tiny;
		c = 1.0 + aa / c; if (std::fabs(c) < tiny) c = tiny;
		d = 1.0 / d; h *= d * c;
		aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0));
		d = 1.0 + aa * d; if (std::fabs(d) < tiny) d = tiny;
		c = 1.0 + aa / c; if (std::fabs(c) < tiny) c = tiny;
		d = 1.0 / d;
		double delta = d * c; h *= delta;
		if (std::fabs(delta - 1.0) < 1e-15) break;
	}
	return h;
}
double gsl_cdf_beta_P(double x, double a, double b) {
	if (x <= 0) return 0;
	if (x >= 1) return 1;
	double lbt = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x);
	if (x < (a + 1.0) / (a + b + 2.0)) return std::exp(lbt) * betaContinuedFraction(a, b, x) / a;
	return 1.0 - std::exp(lbt) * betaContinuedFraction(b, a, 1.0 - x) / b;
}

//----------------------------------------------------------------------------
// Application-layer stubs.
//----------------------------------------------------------------------------

static Simulation *harnessSimulation = 0;
App *app = 0;
Simulation *App::getCurSimulation(void) { return harnessSimulation; }
static HarnessView harnessView = {0, 0};
static HarnessSocietyWindow harnessWindow = {&harnessView};
HarnessSocietyWindow *societyWindow = &harnessWindow;
// curSociety itself is defined in Society.cpp.

void fl_alert(const char *fmt, ...) {
	va_list args;
	va_start(args, fmt);
	std::fprintf(stderr, "laputa alert: ");
	std::vfprintf(stderr, fmt, args);
	std::fprintf(stderr, "\n");
	va_end(args);
	std::exit(3);
}

// Simulation members whose definitions live in the GUI-bound Simulation.cpp.
// They only touch logging and statistics, never the model state.
Simulation::Simulation() {
	soc = 0;
	curStep = 0;
	eValue = eValueDelta = eValueTotal = eValueDeltaTotal = startEValue = 0;
	polarisation = polarisationDelta = startPolarisation = 0;
	msgSent = inqResults = 0;
	bwTowardsP = bwTowardsNotP = 0;
	inqOverriddenTowardsP = inqOverriddenTowardsNotP = 0;
	logLevel = LOG_NONE;
	val.applicationMethod = APPLY_TO_MAJORITY;  // statistics only; avoids individualEValue
	val.eValues[0] = 0; val.eValues[1] = 0.5; val.eValues[2] = 1;
	val.exponent = 1;
	val.majorityAmt = 0.5; val.majorityPCert = 0.5; val.majorityNotPCert = 0.5;
	val.amtStrictlyGreater = val.blfPStrictlyGreater = val.blfNotPStrictlyLess = true;
	logMsgSize = 0;
}
t_float Simulation::individualEValue(t_float blf) { return blf; }

// Text formatting helper from the GUI-bound Utility.cpp; used only to build
// human-readable distribution descriptions, never by the model dynamics.
const char *DoubleToString(double v, t_int digits) {
	static char str[64];
	std::snprintf(str, sizeof str, "%.*f", (int)digits, v);
	return str;
}
void Simulation::addToLog(const char *) {}

//----------------------------------------------------------------------------
// Driver.
//----------------------------------------------------------------------------

static void fail(const std::string &what) {
	std::fprintf(stderr, "harness input error: %s\n", what.c_str());
	std::exit(2);
}

template <typename T> static T readValue(std::istream &in, const char *what) {
	T v;
	if (!(in >> v)) fail(what);
	return v;
}

static void expectKeyword(std::istream &in, const char *keyword) {
	std::string word = readValue<std::string>(in, keyword);
	if (word != keyword) fail(std::string("expected '") + keyword + "', got '" + word + "'");
}

static void readTrust(std::istream &in, TrustFunction &tf) {
	for (int i = 0; i <= TRUST_FUNCTION_RESOLUTION; ++i) tf.values[i] = readValue<float>(in, "trust value");
	tf.expValid = false;
}

static void printTrust(TrustFunction &tf) {
	std::printf("[");
	for (int i = 0; i <= TRUST_FUNCTION_RESOLUTION; ++i) std::printf(i ? ",%.9g" : "%.9g", (double)tf.values[i]);
	std::printf("]");
}

// Re-statement of Simulation::reset lines 126-134 (see header comment).
static void resetSimulation(Society &soc, Simulation &sim) {
	for (t_int i = 0; i < (t_int)soc.people.size(); ++i) soc.people[i].lastInquiryResult = -1;
	for (LinkIterator link = soc.links.begin(); link != soc.links.end(); ++link) {
		link->second.lastUsed = -1;
		if (link->second.evidencePolicy > NEW_EVIDENCE_NONE) {
			link->second.message = MSG_SAY_NOTHING;
			double b = soc.people[link->second.source].belief.v();
			if (b > 0.5 && b > link->second.threshold) link->second.message = MSG_SAY_P;
			else if (b < 0.5 && b < 1.0 - link->second.threshold) link->second.message = MSG_SAY_NOT_P;
		}
	}
	sim.curStep = 0;
	sim.msgSent = sim.inqResults = 0;
}

static void printState(Society &soc, Simulation &sim, int step) {
	std::printf("{\"step\":%d,\"beliefs\":[", step);
	for (size_t i = 0; i < soc.people.size(); ++i) std::printf(i ? ",%.17g" : "%.17g", soc.people[i].belief.v());
	std::printf("],\"inquiry_trust_expectation\":[");
	for (size_t i = 0; i < soc.people.size(); ++i)
		std::printf(i ? ",%.9g" : "%.9g", (double)soc.people[i].inquiryTrust.expectation());
	std::printf("],\"link_trust_expectation\":{");
	bool first = true;
	for (LinkIterator l = soc.links.begin(); l != soc.links.end(); ++l) {
		std::printf("%s\"%d>%d\":%.9g", first ? "" : ",", l->second.source, l->second.target,
		            (double)l->second.trust.expectation());
		first = false;
	}
	std::printf("},\"messages_delivered\":%d,\"inquiry_results\":%d}\n", sim.msgSent, sim.inqResults);
}

static int runMode(std::istream &in) {
	expectKeyword(in, "seed");
	unsigned long long seed = readValue<unsigned long long>(in, "seed");
	harnessRng.engine.seed(seed);
	expectKeyword(in, "steps");
	int steps = readValue<int>(in, "steps");
	if (steps < 0) fail("steps must be non-negative");

	Society soc;
	curSociety = &soc;
	expectKeyword(in, "inquirers");
	int k = readValue<int>(in, "inquirer count");
	if (k < 1 || k > 0xFFFF) fail("inquirer count out of range");
	for (int i = 0; i < k; ++i) {
		Inquirer inq(0, 0, 0);
		inq.name[0] = 0;
		double belief = readValue<double>(in, "belief");
		if (!(belief >= 0 && belief <= 1)) fail("belief out of range");
		inq.belief.set(belief);
		inq.newBelief = inq.belief;
		inq.inquiryChance = readValue<double>(in, "inquiry chance");
		inq.inquiryAccuracy = readValue<double>(in, "inquiry accuracy");
		inq.updateInquiryTrust = readValue<int>(in, "updateInquiryTrust") != 0;
		readTrust(in, inq.inquiryTrust);
		soc.people.push_back(inq);
	}
	expectKeyword(in, "links");
	int m = readValue<int>(in, "link count");
	for (int j = 0; j < m; ++j) {
		int source = readValue<int>(in, "source"), target = readValue<int>(in, "target");
		if (source < 0 || source >= k || target < 0 || target >= k || source == target) fail("bad link endpoints");
		Link link(source, target, (SocietySetup *)0);
		link.listenChance = readValue<double>(in, "listen chance");
		link.threshold = readValue<double>(in, "threshold");
		link.updateTrust = readValue<int>(in, "updateTrust") != 0;
		link.evidencePolicy = (char)readValue<int>(in, "evidence policy");
		link.countPriorAsEvidence = readValue<int>(in, "countPriorAsEvidence") != 0;
		link.message = MSG_SAY_NOTHING;
		link.lastUsed = -1;
		readTrust(in, link.trust);
		if (soc.links.count(COUPLE(source, target))) fail("duplicate link");
		soc.links.insert(std::pair<const unsigned t_int, Link>(COUPLE(source, target), link));
	}
	soc.recalculateListeners();

	Simulation sim;
	harnessSimulation = &sim;
	sim.soc = &soc;
	resetSimulation(soc, sim);
	printState(soc, sim, 0);
	for (int t = 0; t < steps; ++t) {
		soc.evolve(&sim);   // Laputa 1.6: Simulation::step calls evolve, then ++curStep
		++sim.curStep;
		printState(soc, sim, t + 1);
	}
	return 0;
}

int main(int argc, char **argv) {
	std::istream &in = std::cin;
	expectKeyword(in, "mode");
	std::string mode = readValue<std::string>(in, "mode name");
	if (mode == "run") return runMode(in);
	if (mode == "preset") {
		double value = readValue<double>(in, "value"), sharpness = readValue<double>(in, "sharpness");
		TrustFunction tf;
		tf.setValueSharpness(value, sharpness);
		std::printf("{\"values\":");
		printTrust(tf);
		std::printf(",\"expectation\":%.9g}\n", (double)tf.expectation());
		return 0;
	}
	if (mode == "metadefault") {
		std::string dir = readValue<std::string>(in, "distribution directory");
		double weight = readValue<double>(in, "weight");
		if (!dir.empty() && dir[dir.size() - 1] != '/') dir += '/';
		LoadDefaultDistributions(dir.c_str());
		MetaDistribution meta;
		meta.setDefault();
		meta.precalculateForTrust();  // as SocietySetup does before a batch run
		TrustFunction tf;
		meta.setTrustFunction(&tf, (float)weight);
		std::printf("{\"values\":");
		printTrust(tf);
		std::printf(",\"expectation\":%.9g}\n", (double)tf.expectation());
		return 0;
	}
	fail("unknown mode " + mode);
	return 2;
}
