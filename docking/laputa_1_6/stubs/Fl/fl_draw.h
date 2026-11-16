// KANM docking stub: no-op drawing primitives (never called by the harness).
#ifndef KANM_STUB_FL_DRAW_H
#define KANM_STUB_FL_DRAW_H
enum { FL_FLAT_BOX = 1, FL_WHITE = 7, FL_INACTIVE_COLOR = 8, FL_HELVETICA = 0, FL_BOLD = 1, FL_BLACK = 0, FL_RED = 1, FL_SELECTION_COLOR = 15 };
inline void fl_draw_image(const unsigned char*, int, int, int, int, int = 3, int = 0) {}
inline void fl_draw_box(int, int, int, int, int, int) {}
inline void fl_font(int, int) {}
inline void fl_color(int) {}
inline void fl_color(unsigned char, unsigned char, unsigned char) {}
inline void fl_draw(const char*, int, int) {}
inline double fl_width(const char*) { return 0; }
inline void fl_line(int, int, int, int) {}
inline void fl_rectf(int, int, int, int) {}
#endif
