cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(dlprim_backend)

set(Python_FIND_VIRTUALENV FIRST)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(INSTALL_TORCH)
	# CMake is a Rube Goldberg machine platform, I don't care what people say.
	# This is all because pyproject.toml has no way of specifying the package index for the build environment packages,
	# and I don't want the build system to install a bunch of CUDA bloat just to have torch's cmake files.
	if(NOT DEFINED BUILD_TORCH_PACKAGE_NAME)
		set(BUILD_TORCH_PACKAGE_NAME "torch==2.4.0")
	endif()
	if(NOT DEFINED BUILD_TORCH_PACKAGE_INDEX)
		set(BUILD_TORCH_PACKAGE_INDEX "https://download.pytorch.org/whl/cpu")
	endif()
	execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install ${BUILD_TORCH_PACKAGE_NAME} --index-url ${BUILD_TORCH_PACKAGE_INDEX})
endif()

if(INSTALL_PYBIND11)
	execute_process(COMMAND ${Python3_EXECUTABLE} -m pip install pybind11 pybind11-global)
endif()

# This will get the cmake prefix of torch
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; import os; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))" OUTPUT_VARIABLE TORCH_CMAKE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${TORCH_CMAKE_DIR})

find_package(Torch REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
option(BUILD_CORE_ONLY "Core Build" ON)
option(DLPRIM_STATIC_BUILD "Build static version only" ON)
option(BUILD_CPLUS_EXAMPLE "Build C++ examples" OFF)

add_subdirectory(dlprimitives)
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dlprimitives)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dlprimitives/include)
include_directories(${Python3_INCLUDE_DIRS})   # Python include
include_directories(${pybind11_INCLUDE_DIRS})  # pybind11 include

message(" Pytorch libraries: ${TORCH_LIBRARIES}")
message(" Python libraries: ${Python3_LIBRARIES}")
message(" Python include: ${Python3_INCLUDE_DIRS}")

if(1)
	if(NOT WIN32)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall")
	endif()

	# Define our library target
	add_library(pt_vk SHARED 
			src/pointwise_ops.cpp
			src/tensor_ops.cpp
			src/vision_ops.cpp
			src/conv.cpp
			src/im2col.cpp
			src/softmax_impl.cpp
			src/loss_ops.cpp
			src/random_ops.cpp
			src/norm_ops.cpp
			src/utils.cpp
			src/registeration.cpp
			src/CLTensor.cpp
			src/guard.cpp
			src/python_iface.cpp
			src/fill.cpp
			)

	target_compile_features(pt_vk PRIVATE cxx_std_17)
	set_target_properties(pt_vk PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

	# Link against LibTorch
	target_link_libraries(pt_vk ${TORCH_LIBRARIES} ${OCL_LIB} dlprim_core ${Python3_LIBRARIES})
	if(WIN32)
		target_link_libraries(pt_vk ${Python3_LIBRARIES})
	endif()
	set_target_properties(pt_vk PROPERTIES PREFIX "" OUTPUT_NAME "pt_vk")
	if(NOT WIN32)
		set_target_properties(pt_vk PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_vk/")
	else()
		set_target_properties(pt_vk PROPERTIES
			RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/pytorch_vk/"
			SUFFIX ".pyd"
		)
		 
	endif()

		


	foreach(PYNAME __init__.py)
		configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_vk/${PYNAME} "${CMAKE_BINARY_DIR}/pytorch_vk/${PYNAME}" COPYONLY)
		install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/python/pytorch_vk/${PYNAME} DESTINATION python/pytorch_vk)
	endforeach()

	install(TARGETS pt_vk 
			LIBRARY DESTINATION pytorch_vk
			RUNTIME DESTINATION pytorch_vk
		)
else()
	# This is attempt #2 at proper pybind11 usage with this project.
	find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
	find_package(pybind11 CONFIG REQUIRED)
	pybind11_add_module(pt_vk
		src/pointwise_ops.cpp
		src/tensor_ops.cpp
		src/vision_ops.cpp
		src/conv.cpp
		src/im2col.cpp
		src/loss_ops.cpp
		src/random_ops.cpp
		src/norm_ops.cpp
		src/utils.cpp
		src/registeration.cpp
		src/CLTensor.cpp
		src/guard.cpp
		src/python_iface.cpp)
	target_link_libraries(pt_vk ${TORCH_LIBRARIES} ${OCL_LIB} dlprim_core ${Python3_LIBRARIES} pybind11::headers)
	set_target_properties(pt_vk PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
	install(TARGETS pt_vk DESTINATION "${SKBUILD_PLATLIB_DIR}")
endif()

# Add C++ examples
if(BUILD_CPLUS_EXAMPLE)
    add_subdirectory(examples/cplus)
endif()
