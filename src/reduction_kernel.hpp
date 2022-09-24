#ifndef PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP

#include <glutils/guard.hpp>
#include <glutils/program.hpp>

namespace placement {

    class ReductionKernel
    {
    public:
    private:
        glutils::Guard<glutils::Program> m_program;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
