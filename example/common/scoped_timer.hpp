#ifndef PROCEDURALPLACEMENTLIB_SCOPED_TIMER_HPP
#define PROCEDURALPLACEMENTLIB_SCOPED_TIMER_HPP

#include <chrono>
#include <iostream>

struct ScopedTimer
{
    using clock = std::chrono::steady_clock;

    explicit ScopedTimer(const char* tag) : start_time(clock::now()), tag(tag) {}

    ~ScopedTimer()
    {
        const auto dt = clock::now() - start_time;
        std::cout << "[" << tag << "] : " << dt.count() << "ns\n";
    }

    clock::time_point start_time;
    const char* tag;
};


#endif //PROCEDURALPLACEMENTLIB_SCOPED_TIMER_HPP
