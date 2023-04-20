# procedural-placement-lib
A GPU-based runtime procedural placement library, inspired by Horizon: Zero Dawn's [excellent system](https://youtu.be/ToCozpl1sYY).

Version 0.2

## Requirements
- OpenGL 4.5+
- C++ 17 capable compiler

### Dependencies
The library itself requires only an OpenGL loader (glad, currently) and glm. The examples additionally depend on GLFW, ImGui, and stb_image. All of these dependencies are included as git submodules and can be cloned using `git submodule update --init --recursive`.

### OS
All dependencies are multiplatform, but the project itself has only been tested on Linux.

## Build
The project can be built or integrated into other projects using CMake. The main target is `procedural-placement-lib`, which will build a static library. The `pplib-examples` target will build the example programs.

To build standalone, from the main directory of this repository:
```bash
mkdir build
cd build
cmake ..
cmake --build . --target procedural-placement-lib
```

To integrate into another project as a statically linked library:
```cmake
add_subdirectory(path/to/cloned/repo)
target_link_libraries(your-target procedural-placement-lib)
```
The libraries' include directories are part of the target, so it's not necessary to specify them separately.

## Usage
### Briefly
```cpp
// load the GL context into the placement lib
placement::loadGLContext(glfwGetProcAddress)

// instantiate the placement pipeline. This will load and compile the required compute shaders.
placement::PlacementPipeline pipeline;

const placement::WorldData world_data {/*scale=*/{500.0f, 1000.0f, 150.0f}, /*heightmap=*/height_texture};
const placement::LayerData layer_data {/*footprint=*/2.5f, /*textures=*/{{density_texture_0}, {density_texture_1}, {density_texture_2}}};

const glm::vec2 lower_bound{glm::vec2(world_data.scale) / 2.0f};
const glm::vec2 upper_bound{lower_bound + 100.0f};

// dispatch compute shaders
placement::FutureResult future_result = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound);

// wait for results to be ready
const placement::Result results = future_result.readResult();

// option 1: copy results to CPU
std::vector<placement::Result::Element> elements = results.copyAllToHost();
for (auto [position, class_index] : elements)
    doSomething(position, class_index);

// option 2: copy to another buffer
result.copyAll(some_buffer);
doSomethingInGPU(some_buffer);

// option 3: use directly
GLuint result_buffer = results.getBuffer().gl_object.getName();
doSomethingElse(result_buffer);

```
### In more detail
Before any other operation can be performed, the library needs to be granted access to the current OpenGL context. This is done by passing a loader function to `loadGLContext()`. For example, when using GLFW:
```cpp
placement::loadGLContext(glfwGetProcAddress);
```
Placement operations are performed using the `computePlacement()` member function of the `PlacementPipeline` class. This function is declared as:
```cpp
FutureResult PlacementPipeline::computePlacement(const WorldData& world_data, const LayerData& layer_data, vec2 lower_bound, vec2 upper_bound);
```

#### World data
The first argument, `world_data`, is a struct that specifies the characteristics of the terrain on which the objects will be placed.
```cpp
struct WorldData
{ 
    vec3 scale;
    uint heightmap;
};
```
`scale` is a 3-component floating point vector that determines the dimensions of the world in each axis. The units for this scale are arbitrary, and the placement results will be expressed in relation to this scale. The `heightmap` is an unsigned integer identifier (or "name") for an OpenGL 2D texture object. The texture will be used to determine the height of the terrain by sampling its red channel, and will be scaled according to `scale`. That is, the texture coordinate (0, 0) will be mapped to the world origin, whereas the (1, 1) texcoord will correspond to (`scale.x`, `scale.y`) in world coordinates. Similarly, a color value of `0.0` in the red channel will translate into a terrain height of `0.0`, while a value of `1.0` will become `scale.z`.

#### Layer data
The second argument, `layer_data`, is used to specify one or more "placement layers" and the footprint used to generate them.
```cpp
struct LayerData
{ 
    float footprint;
    std::vector<DensityMap> densitymaps;
};
```
The `footprint` is the minimum separation between any two objects placed by the same call to `computePlacement`. It is named so because it approximates the area effectively occupied by these objects. The footprint is interpreted in terms of the world coordinate space, the one used for the `scale` in `world_data`.

The `densitymaps` are a series of grayscale textures that define the distribution of objects over the landscape, somewhat like probability density function. Much like the `world_data`'s `heightmap`, these are grayscale 2D textures, and are mapped onto the terrain in the same way (and using the same `scale`). The difference is that the grayscale value is interpreted differently. A value of `0.0` indicates the absence of any objects in that specific layer, while a value of `1.0` corresponds to the maximum possible density. In other words, a full black texture will produce no objects, while a full white texture will saturate the available space with them.

```cpp
struct DensityMap
{ 
    GLuint texture;
    
    float scale = 1.0f;
    float offset = 0.0f;
    float min_value = 0.0f;
    float max_value = 1.0f;
};
```

The `texture` data member of the DensityMap is, again, the unsigned integer identifier of an OpenGL texture object containing the grayscale image. The other numeric parameters are used to modify the way in which this texture is sampled. Specifically, the final density value is defined as `clamp(sample(texture, uv).r * scale + offset, min_value, max_value)`.

#### Placement region
The last two arguments, `lower_bound` and `upper_bound`, define the region of the world for which object placement will be computed. The position of all generated objects will be such that `lower_bound.x <= x < upper_bound.x` and `lower_bound.y <= y < upper_bound.y`. These values must define a valid rectangle _within_ the confines defined by `world_data.scale`. That is, the minimum value for the lower bound is (0, 0), the maximum for the upper bound is (`world_data.scale.x`, `world_data.scale.y`), and these two must be such that `lower_bound.x < upper_bound.x` and `lower_bound.y < upper_bound.y`.

#### FutureResult and Result
The `computePlacement` function returns a `FutureResult` object, which manages the lifetime of the OpenGL buffer that will contain the results once the compute shaders have finished their execution. The state of the operation can be queried by calling the `isReady` function on the `FutureResult` instance.
```cpp
FutureResult future_result = pipeline.computePlacement(...);

// do something else ... 

Result result = future_result.readResult();
```

Finally, calling the `readResult` function will construct a new `Result` object and transfer ownership of the GL buffer to it, invalidating the `FutureResult` in the process. A `Result` object can be used to consult the number of elements generated per layer and in total, as well as to copy these results to another GL buffer or directly to the CPU.

```cpp
// 1. copy all elements to an array in host memory
auto elements = result.copyAllToHost();

// 2. copy only elements generated by the third densitymap (indices have base 0)
auto specific_elements = result.copyClassToHost(2);

// 3. same as 1 and 2, but copy values to a different GL buffer
result.copyAll(another_buffer);
result.copyClass(2, some_other_buffer);

// 4. query the number of densitymaps used to generate these results.
result.getNumClasses();

// 5. get the total number of elements
std::size_t total_count = result.getElementArrayLength();

// 6. get the number of elements from layer 2 (third densitymap), specifically 
std::size_t layer_count = result.getClassElementCount(2);
```

### More examples
For more detailed examples, including all the boilerplate, see the `example` directory.