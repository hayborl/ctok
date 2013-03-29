#ifndef TIMER_H
#define TIMER_H

#include "opencv2/opencv.hpp"
using namespace cv;

class My_Timer : public TickMeter
{
public:
	enum TimeUnit{MICRO, MILLI, SEC};
	void start()
	{
		TickMeter::reset();
		TickMeter::start();
	}
	double stop(TimeUnit unit = SEC)
	{
		TickMeter::stop();
		switch (unit)
		{
		case MICRO:
			return getTimeMicro();
		case MILLI:
			return getTimeMilli();
		case SEC:
		default:
			return getTimeSec();
		}
	}
};

#define RUNANDTIME(timer, fun, output, s) timer.start(); fun; \
	if(output) cout << timer.stop() << "s " << s << endl;

static My_Timer global_timer;

#endif