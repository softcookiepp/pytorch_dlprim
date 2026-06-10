# examples/CMakeLists.txt
cmake_minimum_required(VERSION 3.1)

# find sub-directories
file(GLOB CHILD_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(dir ${CHILD_DIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
        # collect .cpp files in the sub-directory
        file(GLOB SRC_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.cpp)
        if(SRC_FILES)
            # Executable name is the same as the sub-directory name
            set(TARGET_NAME ${dir})

            add_executable(${TARGET_NAME} ${SRC_FILES})

            # Include main project headers
            target_include_directories(${TARGET_NAME} PRIVATE
                ${PROJECT_SOURCE_DIR}/dlprimitives/include
                ${PROJECT_SOURCE_DIR}/include
                ${OpenCV_INCLUDE_DIRS}
            )

            # Link main library and dependencies
            target_link_libraries(${TARGET_NAME} PRIVATE pt_vk ${TORCH_LIBRARIES} ${OCL_LIB} )

            # Set C++17
            target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

            message(STATUS "Added example: ${TARGET_NAME} -> ${SRC_FILES}")
        endif()
    endif()
endforeach()
