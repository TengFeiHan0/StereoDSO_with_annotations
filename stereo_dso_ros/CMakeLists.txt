cmake_minimum_required(VERSION 2.4.6)

#set(OpenCV_DIR "/usr/local/opencv3.4/share/OpenCV")
#include_directories("/usr/local/opencv3.4/include")

project(stereo_dso_ros)

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

find_package(OpenCV REQUIRED)
#message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
#message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  cv_bridge
  message_filters
)

set(STEREO_DSO_PATH /home/mm/Downloads/stereoDSO_ws/src/stereo_dso)
message("\n\n---- LOOKING FOR STEREO_DSO at\n\"${STEREO_DSO_PATH}\"")


find_package(Pangolin 0.2 REQUIRED)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_module)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)
find_library(STEREO_DSO_LIBRARY dso ${STEREO_DSO_PATH}/build/lib)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    sensor_msgs
    cv_bridge
    message_filters
)

###########
## Build ##
###########

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -march=native -fno-omit-frame-pointer"
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${STEREO_DSO_PATH}/src
  ${STEREO_DSO_PATH}/thirdparty/Sophus
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(stero_dso_ros src/stereo_dso_ros.cpp)

target_link_libraries(stero_dso_ros
    ${STEREO_DSO_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES})
