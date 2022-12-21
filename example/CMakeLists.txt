file(CREATE_LINK ${ASSETS_DIRECTORY} ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)

add_library(example-common STATIC
        example-common.cpp
        common/glfw_wrapper.cpp
        common/camera_controller.cpp
        common/imgui_wrapper.cpp)
target_link_libraries(example-common
        PUBLIC
        procedural-placement-lib
        simple-renderer
        glutils
        glfw
        PRIVATE
        stb_image
        imgui)
target_include_directories(example-common PUBLIC $<TARGET_PROPERTY:imgui,INTERFACE_INCLUDE_DIRECTORIES>)

add_executable(01-basic-placement 01-basic-placement.cpp common/camera_controller.cpp)
target_link_libraries(01-basic-placement example-common)

add_executable(02-basic-visualization 02-basic-visualization.cpp)
target_link_libraries(02-basic-visualization example-common)

add_executable(03-interactive-placement 03-interactive-placement.cpp)
target_link_libraries(03-interactive-placement example-common)

add_custom_target(pplib-examples)
add_dependencies(pplib-examples
        01-basic-placement
        02-basic-visualization
        03-interactive-placement)