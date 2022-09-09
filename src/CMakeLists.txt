add_library(procedural-placement-lib STATIC placement.cpp)

target_include_directories(procedural-placement-lib
        PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(procedural-placement-lib
        PUBLIC glm
        PRIVATE glutils)