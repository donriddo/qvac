# stable-diffusion.cpp vcpkg overlay port
#
# Fetches stable-diffusion.cpp from GitHub (including the ggml submodule)
# and builds it as a static library.
#
# The port installs:
#   - include/stable-diffusion.h       (main C API)
#   - include/stb_image.h              (stb image loading)
#   - include/stb_image_write.h        (stb PNG encoding)
#   - lib/libstable-diffusion.a        (static library)
#   - share/stable-diffusion-cpp/      (CMake config)
#
# GPU backend selection is controlled via vcpkg features:
#   - metal   -> -DGGML_METAL=ON   (macOS/iOS, auto-enabled on Apple)
#   - vulkan  -> -DGGML_VULKAN=ON
#   - cuda    -> -DGGML_CUDA=ON
#   - opencl  -> -DGGML_OPENCL=ON
#
# NOTE: This port uses vcpkg_from_git which clones the repo so that
# git submodule init/update works for ggml.
# Update REF to pin a specific commit for reproducible builds.

set(SD_CPP_REPO "https://github.com/leejet/stable-diffusion.cpp.git")
set(SOURCE_PATH "${CURRENT_BUILDTREES_DIR}/src/stable-diffusion-cpp-master")

# vcpkg_from_git cannot fetch by SHA on GitHub (server restriction) and
# HEAD_REF-only ports require --head. Use a direct clone with --recurse-submodules
# so that the bundled ggml submodule is also initialised in one step.
# TODO: switch to vcpkg_from_github once stable-diffusion.cpp publishes versioned tarballs.
if(NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    vcpkg_execute_required_process(
        COMMAND "${GIT}" clone
            --depth 1
            --recurse-submodules
            --shallow-submodules
            -b master
            "${SD_CPP_REPO}"
            "${SOURCE_PATH}"
        WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}"
        LOGNAME "git-clone-stable-diffusion-cpp"
    )
endif()

# --- Determine GPU feature flags ---
set(SD_GGML_METAL   OFF)
set(SD_GGML_VULKAN  OFF)
set(SD_GGML_CUDA    OFF)
set(SD_GGML_OPENCL  OFF)
set(SD_FLASH_ATTN   OFF)

if("metal" IN_LIST FEATURES)
    set(SD_GGML_METAL ON)
elseif(VCPKG_TARGET_IS_OSX OR VCPKG_TARGET_IS_IOS)
    # Auto-enable Metal on Apple platforms even without the feature flag
    set(SD_GGML_METAL ON)
endif()

if("vulkan" IN_LIST FEATURES)
    set(SD_GGML_VULKAN ON)
endif()

if("cuda" IN_LIST FEATURES)
    set(SD_GGML_CUDA ON)
endif()

if("opencl" IN_LIST FEATURES)
    set(SD_GGML_OPENCL ON)
endif()

if("flash-attn" IN_LIST FEATURES)
    set(SD_FLASH_ATTN ON)
endif()

# --- Configure and build ---
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_SHARED_LIBS=OFF
        -DSD_BUILD_EXAMPLES=OFF
        -DSD_BUILD_SHARED_LIBS=OFF
        -DSD_METAL=${SD_GGML_METAL}
        -DSD_VULKAN=${SD_GGML_VULKAN}
        -DSD_CUDA=${SD_GGML_CUDA}
        -DSD_OPENCL=${SD_GGML_OPENCL}
        -DSD_FLASH_ATTN=${SD_FLASH_ATTN}
)

vcpkg_cmake_install()

# --- Install stb headers for PNG encode/decode in consumer code ---
if(EXISTS "${SOURCE_PATH}/thirdparty/stb/stb_image.h")
    file(INSTALL "${SOURCE_PATH}/thirdparty/stb/stb_image.h"
         DESTINATION "${CURRENT_PACKAGES_DIR}/include")
    file(INSTALL "${SOURCE_PATH}/thirdparty/stb/stb_image_write.h"
         DESTINATION "${CURRENT_PACKAGES_DIR}/include")
elseif(EXISTS "${SOURCE_PATH}/thirdparty/stb_image.h")
    file(INSTALL "${SOURCE_PATH}/thirdparty/stb_image.h"
         DESTINATION "${CURRENT_PACKAGES_DIR}/include")
    file(INSTALL "${SOURCE_PATH}/thirdparty/stb_image_write.h"
         DESTINATION "${CURRENT_PACKAGES_DIR}/include")
endif()

# --- Create CMake config for find_package(stable-diffusion-cpp CONFIG REQUIRED) ---
set(CONFIG_DIR "${CURRENT_PACKAGES_DIR}/share/stable-diffusion-cpp")
file(MAKE_DIRECTORY "${CONFIG_DIR}")

file(WRITE "${CONFIG_DIR}/stable-diffusion-cppConfig.cmake" [=[
get_filename_component(_SD_CPP_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

find_library(STABLE_DIFFUSION_LIBRARY
    NAMES stable-diffusion
    PATHS "${_SD_CPP_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

find_path(STABLE_DIFFUSION_INCLUDE_DIR
    NAMES stable-diffusion.h
    PATHS "${_SD_CPP_INSTALL_PREFIX}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

if(NOT TARGET stable-diffusion::stable-diffusion)
    add_library(stable-diffusion::stable-diffusion STATIC IMPORTED)
    set_target_properties(stable-diffusion::stable-diffusion PROPERTIES
        IMPORTED_LOCATION             "${STABLE_DIFFUSION_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${STABLE_DIFFUSION_INCLUDE_DIR}"
    )
endif()
]=])

file(WRITE "${CONFIG_DIR}/stable-diffusion-cppConfigVersion.cmake" [=[
set(PACKAGE_VERSION "0.0.1")
if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
]=])

# Remove debug include dir (no debug headers needed)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# Install license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

set(VCPKG_BUILD_TYPE release)
