#pragma once

#include <chrono>

namespace hpc {

class CpuTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    void start() {
        m_start = Clock::now();
        m_running = true;
    }

    void stop() {
        m_end = Clock::now();
        m_running = false;
    }

    double elapsed_ms() const {
        TimePoint endTime = m_running ? Clock::now() : m_end;
        return std::chrono::duration<double, std::milli>(endTime - m_start).count();
    }

    double elapsed_seconds() const {
        return elapsed_ms() / 1000.0;
    }

private:
    TimePoint m_start;
    TimePoint m_end;
    bool m_running = false;
};

} // namespace hpc
