// KANM docking stub: minimal widget base so that TrustView (a GUI class in
// Trust.cpp) compiles. TrustView is never instantiated by the harness.
#ifndef KANM_STUB_FL_BUTTON_H
#define KANM_STUB_FL_BUTTON_H
#include "FL/Fl.H"
class Fl_Widget {
public:
	Fl_Widget(int x, int y, int w, int h, const char* l = 0) : x_(x), y_(y), w_(w), h_(h) {}
	virtual ~Fl_Widget() {}
	virtual void draw() {}
	void deactivate() {}
	void activate() {}
	int active() const { return 1; }
	void redraw() {}
	int x() const { return x_; } int y() const { return y_; }
	int w() const { return w_; } int h() const { return h_; }
private:
	int x_, y_, w_, h_;
};
class Fl_Button : public Fl_Widget {
public:
	Fl_Button(int x, int y, int w, int h, const char* l = 0) : Fl_Widget(x, y, w, h, l) {}
	int value() const { return 0; }
};
#endif
