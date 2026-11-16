// KANM docking stub: replaces Laputa 1.6 headers/App.h for a headless build.
// It declares only what the model core (Society, Inquirer, Link, Trust,
// Distribution, MetaDistribution, Parameters) references. No model logic here.
#ifndef __APP_H__
#define __APP_H__
#include "Prefix.h"
#include "Distribution.h"
#include "Simulation.h"
#include "gsl/gsl_rng.h"
#include "gsl/gsl_math.h"
#include <stdio.h>
#include <string>
using namespace std;
#define LAPUTA_VERSION 160
#define MIN_LAPUTA_VERSION 150
// From headers/DistributionView.h (a GUI header); used only by image-drawing code.
#define DISTRIBUTION_VIEW_HEIGHT 4.0
class App {
public:
	Simulation* getCurSimulation(void);
};
extern App *app;
extern gsl_rng *rng;
struct HarnessView { t_float xMid, yMid; };
struct HarnessSocietyWindow { HarnessView *view; };
extern HarnessSocietyWindow *societyWindow;
void fl_alert(const char *fmt, ...);
#endif
