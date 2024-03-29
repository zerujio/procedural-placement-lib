#include "placement/placement_result.hpp"

#include "gl_context.hpp"

namespace placement {

constexpr GLintptr uint_size = sizeof(GLuint);

Result::Result(ResultBuffer &&buffer) : m_buffer(std::move(buffer))
{
    using clock = std::chrono::steady_clock;

    const uint num_classes = m_buffer.num_classes;

    m_index_offset.reserve(num_classes + 1);
    m_index_offset.emplace_back(0);

    //m_buffer.gl_object.read(0, read_size, m_index_offset.data() + 1);
    m_index_offset.insert(m_index_offset.end(), m_buffer.getCountDataBegin(), m_buffer.getCountDataEnd());

    uint sum = 0;
    for (uint &index: m_index_offset)
    {
        index += sum;
        sum = index;
    }
}

Result::uint Result::copyClassRange(Result::uint begin_class, Result::uint end_class, GL::BufferHandle buffer,
                                    GLintptr offset) const
{
    constexpr GLsizeiptr element_size = sizeof(Element);
    const auto element_count = getClassRangeElementCount(begin_class, end_class);

    GL::Buffer::copy(m_buffer.gl_object,
                     buffer,
                     getElementArrayBufferOffset() + getClassIndexOffset(begin_class) * element_size,
                     offset,
                     element_count * element_size);

    return element_count;
}

std::vector<Result::Element> Result::copyAllToHost() const
{
    std::vector<Element> vector {getElementArrayLength()};

    copyAllToHost(vector.begin());

    return vector;
}

std::vector<Result::Element> Result::copyClassToHost(Result::uint class_index) const
{
    std::vector<Element> vector {getClassElementCount(class_index)};

    copyClassToHost(class_index, vector.begin());

    return vector;
}

FutureResult::FutureResult(ResultBuffer &&result_buffer, GL::Sync &&sync) : m_buffer(std::move(result_buffer)),
                                                                            m_sync(std::move(sync))
{}

bool FutureResult::wait(std::chrono::nanoseconds timeout) const
{
    const auto status = m_sync.clientWait(false, timeout);
    return status == GL::Sync::Status::already_signaled || status == GL::Sync::Status::condition_satisfied;
}

Result FutureResult::readResult()
{
    while (!wait(std::chrono::nanoseconds::max()))
        /* wait */;

    return Result(moveResultBuffer());
}

} // placement