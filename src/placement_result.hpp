#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_RESULT_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_RESULT_HPP

#include "glutils/buffer.hpp"
#include "glutils/sync.hpp"

#include "glm/vec3.hpp"

#include <chrono>
#include <utility>
#include <vector>

namespace placement {

/**
 * @brief Wraps a buffer containing placement results.
 * A result buffer is composed of "count" and "value" sections. The count section specifies the number of valid elements
 * generated by the placement operation, while the value section contains said elements. Note that the size of these
 * two sections does not necessarily add up to the size of the buffer; there is unused space after the value section.
 *
 * The count section begins at the start of the buffer, and consists of an array of 32-bit unsigned integers containing
 * num_classes elements, one for each placement class. These values represent the number of valid elements for each
 * class, that is, the value at index i is the number of valid elements in class i.
 *
 * The value section is an array of valid elements, each one composed of a 3-element vector of 32-bit floating point
 * values (vec3) followed by a single 32-bit unsigned integer (uint). The vector corresponds to the position of the
 * element in world space, while the integer is the class index. The array is sorted in ascending order of class index.
 * This means that the first element of class 0 is at position 0 and the last element is at position count[0] - 1 of the
 * array. Elements of class 1 are located in the range [count[0], count[0] + count[1]), and so on for each additional
 * class.
 */
class ResultBuffer final
{
public:
    using GLsizeiptr = GL::GLsizeiptr;

    GLsizeiptr size;        //<! Total size of the buffer, in bytes.
    GLsizeiptr num_classes; //<! Number of placement classes in the buffer.
    GL::Buffer gl_object;   //<! GL buffer object.
};

/**
 * @brief Contains the results of a placement operation.
 */
class Result
{
public:
    using vec3 = glm::vec3;
    using uint = GL::GLuint;
    using GLintptr = GL::GLintptr;
    using GLsizeiptr = GL::GLsizeiptr;

    struct Element
    {
        vec3 position;
        uint class_index;
    };

    static constexpr GLsizeiptr element_size = sizeof(Element);

    /// Access the result buffer struct.
    [[nodiscard]]
    const ResultBuffer& getBuffer() const { return m_buffer; }

    /// Get the number of placement classes in this result.
    [[nodiscard]]
    uint getNumClasses() const { return m_buffer.num_classes; }

    /// Get the total number of elements in the buffer.
    [[nodiscard]]
    uint getTotalElementCount() const { return m_index_offsets.back(); }

    /// Get the range within the buffer that contains all elements.
    GL::Buffer::Range getElementBufferRange() const { return {m_base_offset, m_index_offsets.back() * element_size}; }

    /// Get the number of elements in a given placement class.
    [[nodiscard]]
    uint getClassElementCount(uint class_index) const
    { return m_index_offsets[class_index + 1] - m_index_offsets[class_index]; }

    /// Get the element index offset for a specific placement class.
    [[nodiscard]] uint getClassIndexOffset(uint class_index) const { return m_index_offsets[class_index]; }

    [[nodiscard]] GLintptr getClassOffset(uint class_index) const
    { return m_base_offset + getClassIndexOffset(class_index) * element_size; }

    [[nodiscard]] GLsizeiptr getClassSize(uint class_index) const
    { return getClassElementCount(class_index) * element_size; }

    /// Get the range within the buffer that contains all elements of a specific class.
    [[nodiscard]] GL::Buffer::Range getClassBufferRange(uint class_index) const
    { return {getClassOffset(class_index), getClassSize(class_index)}; }

private:
    ResultBuffer m_buffer;
    GLintptr m_base_offset;
    std::vector<uint> m_index_offsets;
};

/// Contains the results of a placement operation which may not have finished execution yet.
class FutureResult final
{
public:
    FutureResult(ResultBuffer&& result_buffer, GL::Sync&& sync);

    /// Check if results are ready.
    [[nodiscard]]
    bool isReady() const { return wait(std::chrono::nanoseconds::zero()); }

    /// Wait until results are ready, or until the timeout expires.
    [[nodiscard]]
    bool wait(std::chrono::nanoseconds timeout) const;

    /**
     * @brief Access the results of the placement operation.
     * It is valid to access the results _before_ computation has finished (isReady() or wait() return true), but doing
     * so may have significant CPU-side latency and performance cost, depending on the operation. As a general rule,
     * GPU-only operations (e.g. using the buffer for vertex attributes or copying to another buffer) can be executed
     * with no additional cost, whereas "read back" operations (e.g. using glBufferSubData to read the buffer's contents
     * ) will likely cause a CPU-GPU sync and a stall in the pipeline.
     *
     * Note that for any of these operations to produce correct results, the relevant memory barriers must be
     * issued.
     *
     * @see ResultBuffer
     * @return A const reference to the ResultBuffer struct.
     */
    [[nodiscard]]
    const ResultBuffer& getResult() const { return m_buffer; }

    /// Move out the result buffer struct and, consequently, transfer ownership of the contained GL buffer object.
    [[nodiscard]]
    ResultBuffer moveResult() { return std::move(m_buffer); }

private:
    ResultBuffer m_buffer;
    GL::Sync m_sync;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_RESULT_HPP
