add_subdirectory(simple-renderer)
add_subdirectory(fast_obj)

set(BUILD_SHARED_LIBS ON)
option(TBB_TEST "" OFF)
option(TBBMALLOC_BUILD "" OFF)
add_subdirectory(oneTBB)