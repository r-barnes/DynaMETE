#ifndef _tps_hpp_
#define _tps_hpp_

#include <chrono>
#include <stdexcept>

///@brief Used to time how intervals in code.
///
///Such as how long it takes a given function to run, or how long I/O has taken.
class TimePerSecond {
  private:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::duration<double, std::ratio<1> > second;

    std::chrono::time_point<clock> start_clock;  ///< Last time the timer was started
    double start_time  = 0;                      ///< Time at which clock was last activated

    ///Number of (fractional) seconds between two time objects
    double clockDiff(const std::chrono::time_point<clock> &start, const std::chrono::time_point<clock> &end){
      return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    }

  public:
    ///Creates a Timer which is not running and has no accumulated time
    TimePerSecond() = default;

    double operator()(const double time){
      const auto   now_clock  = clock::now();
      const double times_diff = time-start_time;
      const double clock_diff = clockDiff(start_clock,now_clock);
      start_time              = time;
      start_clock             = now_clock;
      return times_diff/clock_diff;
    }
};

#endif
