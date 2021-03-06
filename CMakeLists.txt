cmake_minimum_required(VERSION 3.15)
project(nettest)

set(CMAKE_PREFIX_PATH D:/Cpplibtorchopencv/libtorch)
set(OpenCV_DIR D:/Cpplibtorchopencv/opencv)
set(CMAKE_CXX_STANDARD 11)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(nettest main.cpp)
target_link_libraries(nettest "${TORCH_LIBRARIES}")
target_link_libraries(nettest "${OpenCV_LIBS}")
set_property(TARGET nettest PROPERTY CXX_STANDARD 11)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET nettest
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:nettest>)
endif (MSVC)