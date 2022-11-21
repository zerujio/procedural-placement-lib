#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP

#include "generation_kernel.hpp"
#include "reduction_kernel.hpp"

#include "glutils/buffer.hpp"

#include "glm/glm.hpp"

#include <vector>

namespace placement {

    class PlacementPipeline
    {
    public:
        PlacementPipeline();

        /**
         * @brief Compute placement for the specified area.
         * Elements will be placed in the area specified by @p lower_bound and @p upperbound . The placement area is
         * defined by all points (x, y) such that
         *
         *      lower_bound.x <= x < upper_bound.x
         *      lower_bound.y <= y < upper_bound.y
         *
         * i.e. the half open [lower_bound, upper_bound) range. Consequently, if lower_bound is not less than upper_bound,
         * the placement region will have zero area and no elements will be placed. In that case no error will occur and an
         * empty vector will be returned.
         *
         * @param footprint collision radius for the placed objects.
         * @param lower_bound minimum x and y coordinates of the placement area
         * @param upper_bound maximum x and y coordinates of the placement area
         */
         void computePlacement(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound);

        /**
         * @brief The heightmap texture.
         * @param tex integer name for a 2D OpenGL texture object.
         */
        void setHeightTexture(unsigned int tex);

        [[nodiscard]]
        auto getHeightTexture() const -> unsigned int;

        /**
         * @brief The density map texture.
         * @param tex integer name for a 2D OpenGL texture object.
         */
        void setDensityTexture(unsigned int tex);

        [[nodiscard]]
        auto getDensityTexture() const -> unsigned int;

        /**
         * @brief The scale of the world.
         * @param scale
         */
        void setWorldScale(const glm::vec3& scale);

        [[nodiscard]]
        auto getWorldScale() const -> const glm::vec3&;

        /**
         * @brief set the seed for the random number generator.
         * For a given set of heightmap, densitymap and world scale, the random seed completely determines placement.
         */
        void setRandomSeed() const;

        /// The number of different texture units used by the placement compute shaders
        static constexpr auto required_texture_units = 2u;

        /**
         * @brief Configures the texture units the pipeline will use
         * @param index the index of a texture unit such that indices in the range [index, index + required_texture_units)
         *      are all valid texture unit indices.
         */
        void setBaseTextureUnit(glutils::GLuint index);

        /// The number of different shader storage buffer binding points used by the placement compute shaders.
        static constexpr auto required_shader_storage_binding_points = 3u;

        /**
         * @brief Configures the shader storage buffer binding points the pipeline will use.
         * @param index An index such that elements in the range [index, index + required_shader_storage_binding_points)
         *      are valid shader storage buffer binding points.
         */
        void setBaseShaderStorageBindingPoint(glutils::GLuint index);

        /// Copy results from GPU buffer to CPU memory
        [[nodiscard]] std::vector<glm::vec3> copyResultsToCPU() const;

        [[nodiscard]] std::size_t getResultsSize() const {return m_valid_count;}

        /**
         * @brief Copy results to another buffer using glCopyNamedBufferSubData()
         * @param buffer the integer name of an opengl buffer. Its size must be at
         * least @p offset + getResultsSize() * sizeof(vec4).
         * @param offset a byte offset into @p buffer
         */
        void copyResultsToGPUBuffer(glutils::GLuint buffer, glutils::GLsizeiptr offset = 0u) const;

    private:

        void m_setCandidateBufferBindingIndex(glutils::GLuint index);
        void m_setIndexBufferBindingIndex(glutils::GLuint index);
        void m_setPositionBufferBindingIndex(glutils::GLuint index);

        struct WorldData
        {
            unsigned int height_tex {0};
            unsigned int density_tex {0};
            glm::vec3 scale {1.0f};
        } m_world_data;

        GenerationKernel m_generation_kernel;
        IndexAssignmentKernel m_assignment_kernel;
        IndexedCopyKernel m_copy_kernel;

        /// number of valid candidates. Total candidates are equal to m_buffer.getSize()
        glutils::GLsizeiptr m_valid_count = 0;

        class Buffer
        {
        public:
            void resize(glutils::GLsizeiptr candidate_count);
            void reserve(glutils::GLsizeiptr candidate_count);

            [[nodiscard]] glutils::GLsizeiptr getSize() const {return m_size;}
            [[nodiscard]] glutils::GLsizeiptr getCapacity() const {return m_capacity;}

            [[nodiscard]] glutils::BufferRange getCandidateRange() const;
            [[nodiscard]] glutils::BufferRange getIndexRange() const;
            [[nodiscard]] glutils::BufferRange getPositionRange() const;

        private:
            glutils::Guard<glutils::Buffer> m_buffer {};
            glutils::GLsizeiptr m_capacity {0};
            glutils::GLsizeiptr m_size {0};

            struct Range
            {
                glutils::GLintptr offset;
                glutils::GLsizeiptr size;
            };

            Range m_candidate_range;
            Range m_index_range;
            Range m_position_range;

            static constexpr glutils::GLsizeiptr s_min_capacity = 64;
            static glutils::GLsizeiptr s_calculateSize(glutils::GLsizeiptr capacity);

            class Allocator;
        } m_buffer;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
