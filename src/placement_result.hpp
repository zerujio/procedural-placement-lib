#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_RESULT_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_RESULT_HPP

#include "glutils/buffer.hpp"
#include "glutils/sync.hpp"

#include "glm/vec3.hpp"

#include <chrono>
#include <utility>
#include <vector>
#include <memory>

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

    unsigned int num_classes;   ///< Number of placement classes in the buffer.
    GLsizeiptr size;            ///< Total size of the buffer, in bytes.
    GL::Buffer gl_object;       ///< GL buffer object.
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

    explicit Result(ResultBuffer &&buffer);

    /// Get the number of placement classes in the result buffer.
    [[nodiscard]]
    uint getNumClasses() const noexcept
    { return m_buffer.num_classes; }

    /// Total number of elements in the element array.
    [[nodiscard]]
    uint getElementArrayLength() const noexcept
    { return m_index_offset.back(); }

    /// Get the byte offset within the result buffer at which the element array starts.
    [[nodiscard]]
    GLintptr getElementArrayBufferOffset() const noexcept
    { return m_buffer.num_classes * static_cast<GLintptr>(sizeof(uint)); }

    /**
     * @brief Access the index offsets for each placement class.
     * @return A const reference to a vector containing the index offsets for each class.
     *
     * The index offset of a class indicates where the elements of a class start within the element array. The index
     * offset vector contains num_classes + 1 elements, where the each element corresponds to the index offset of the
     * class with the same index. The last value of the vector is the length of the array.
     */
    [[nodiscard]]
    const std::vector<uint> &getIndexOffsets() const noexcept
    { return m_index_offset; }

    /// Same as `getIndexOffsets()[class_index]`.
    [[nodiscard]]
    uint getClassIndexOffset(uint class_index) const noexcept
    { return m_index_offset[class_index]; }

    /// Get the number of elements in a given placement class.
    [[nodiscard]]
    uint getClassElementCount(uint class_index) const noexcept
    { return getClassRangeElementCount(class_index, class_index + 1); }

    /**
     * @brief Get the element count of multiple consecutive classes.
     * @param begin_class first class of the range.
     * @param end_class Last class of the range, non inclusive.
     * @return The sum of all the element counts of each class in the range [begin_class, end_class).
     */
    [[nodiscard]]
    uint getClassRangeElementCount(uint begin_class, uint end_class) const noexcept
    { return m_index_offset[end_class] - m_index_offset[begin_class]; }

    /**
     * @brief Copy elements of classes in range [begin_class, end_class) from the element array to another buffer.
     * @param begin_class The start of the class range.
     * @param end_class The end of the class range, not included in it.
     * @param buffer A handle to a GL buffer object.
     * @param offset offset into @p at which to begin copying the data.
     * @return the number of elements copied. This value can be calculated beforehand with getClassRangeElementCount().
     */
    uint copyClassRange(uint begin_class, uint end_class, GL::BufferHandle buffer, GLintptr offset = 0) const;

    uint copyClassRange(uint begin_class, uint end_class, uint buffer, GLintptr offset = 0) const
    { return copyClassRange(begin_class, end_class, static_cast<GL::BufferHandle>(buffer), buffer); }

    /**
     * @brief Copy elements of class in range [begin_class, end_class) to CPU memory.
     * @tparam Iter An output iterator type, such that its value type is copy assignable from an instance of Element.
     * @param begin_class index of the first class in the range.
     * @param end_class index of the last class in the range. This class class is not included in it.
     * @param out_iter An iterator at which to start writing the values read from the element array.
     * @return the number of elements copied. This value can be calculated beforehand with getClassRangeElementCount().
     */
    template<typename Iter>
    uint copyClassRange(uint begin_class, uint end_class, Iter out_iter) const
    {
        constexpr GLsizeiptr element_size = sizeof(Element);
        const uint element_count = getClassRangeElementCount(begin_class, end_class);
        const GL::Buffer::Range map_range
        {
            getElementArrayBufferOffset() + getClassIndexOffset(begin_class) * element_size,
            element_count * element_size
        };
        auto ptr = static_cast<const Element*>(m_buffer.gl_object.mapRange(map_range, GL::Buffer::AccessFlags::read));

        for (auto in_iter = ptr; in_iter != ptr + element_count;)
            *out_iter++ = *in_iter++;

        m_buffer.gl_object.unmap();

        return element_count;
    }

    /**
     * @brief Copy all valid elements from the result buffer to another GPU buffer.
     * @param buffer A handle to a GL buffer object.
     * @param offset Byte offset into the write buffer.
     * @return number of elements copied
     */
    uint copyAll(GL::BufferHandle buffer, GLintptr offset = 0) const
    { return copyClassRange(0, m_buffer.num_classes, buffer, offset); }

    uint copyAll(uint buffer, GLintptr offset = 0) const
    { return copyAll(static_cast<GL::BufferHandle>(buffer), offset); }

    /// Copy all elements to host.
    template<typename Iter>
    uint copyAll(Iter out_iter) const
    { return copyClassRange(0, m_buffer.num_classes, out_iter); }

    /// Copy all elements of a specific class to another buffer.
    uint copyClass(uint class_index, GL::BufferHandle buffer, GLintptr offset = 0) const
    { return copyClassRange(class_index, class_index + 1, buffer, offset); }

    uint copyClass(uint class_index, uint buffer, GLintptr offset = 0) const
    { return copyClass(class_index, static_cast<GL::BufferHandle>(buffer), offset); }

    /// copy all elements of a specific class to host.
    template<typename Iter>
    uint copyClass(uint class_index, Iter out_iter) const
    { return copyClassRange(class_index, class_index + 1, out_iter); }

private:
    ResultBuffer m_buffer;
    std::vector<uint> m_index_offset;
};

/// Contains the results of a placement operation which may not have finished execution yet.
class FutureResult final
{
public:
    FutureResult(ResultBuffer &&result_buffer, GL::Sync &&sync);

    /// Check if results are available.
    [[nodiscard]]
    bool isReady() const
    { return wait(std::chrono::nanoseconds::zero()); }

    /// Wait until results are ready, or until the timeout expires.
    [[nodiscard]]
    bool wait(std::chrono::nanoseconds timeout) const;

    /**
     * @brief Read results, if available, or block execution until they are.
     * This operation moves out the ResultBuffer, leaving this object in an empty state.
     * @return a Result structure that contains the buffer and information about the layout of the data.
     */
    [[nodiscard]] Result readResult()
    { return Result(moveResultBuffer()); }

    /**
     * @brief Access the results of the placement operation.
     * Assuming that the appropriate memory barriers have been issued, it is valid behavior to operate on the result
     * buffer before computation of the results has finished. However, it is important to keep in mind that most
     * operations which move data from device to host, such as reading the buffer with glGetBufferSubData(), will
     * block the CPU thread in order to wait for the data to be available. On the other hand, using the data from within
     * the GPU, such as when reading vertex attributes from the buffer, will not block execution on the calling thread.
     *
     * @see ResultBuffer
     * @return A const reference to the ResultBuffer struct.
     */
    [[nodiscard]]
    const ResultBuffer &getResultBuffer() const
    { return m_buffer; }

    [[nodiscard]] ResultBuffer moveResultBuffer()
    { return std::move(m_buffer); }

private:
    ResultBuffer m_buffer;
    GL::Sync m_sync;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_RESULT_HPP
