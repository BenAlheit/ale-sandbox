##
#  CMake script for the step-1 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "ale")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC})
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
        ${TARGET}.cc
        )
#execute_process (
#        COMMAND bash -c "cat $HOME/deal-dir"
#        OUTPUT_VARIABLE outVar
#)
#string(STRIP ${outVar} outVar)

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 9.3.2
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR} ${outVar}
        )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
          "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
          "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
          "or set an environment variable \"DEAL_II_DIR\" that contains this path."
          )
ENDIF()
#find_package(yaml-cpp REQUIRED)
#include_directories(${YAML_CPP_INCLUDE_DIRS})
#include_directories(${YAML_CPP_INCLUDE_DIR})

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()
#target_link_libraries(${TARGET} ${YAML_CPP_LIBRARIES})