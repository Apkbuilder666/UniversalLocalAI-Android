include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

if(ANDROID AND LOCALAI_ENABLE_VULKAN)
    find_path(LOCALAI_VULKAN_HPP_SYSTEM_INCLUDE vulkan/vulkan.hpp
        PATHS /usr/include NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH REQUIRED)
    set(LOCALAI_VULKAN_HPP_INCLUDE "${CMAKE_BINARY_DIR}/localai-vulkan-headers")
    file(MAKE_DIRECTORY "${LOCALAI_VULKAN_HPP_INCLUDE}")
    foreach(LOCALAI_VULKAN_HEADER_TREE IN ITEMS vulkan vk_video spirv)
        if(NOT EXISTS "${LOCALAI_VULKAN_HPP_SYSTEM_INCLUDE}/${LOCALAI_VULKAN_HEADER_TREE}")
            message(FATAL_ERROR
                "Required Khronos header tree is missing: ${LOCALAI_VULKAN_HPP_SYSTEM_INCLUDE}/${LOCALAI_VULKAN_HEADER_TREE}")
        endif()
        file(COPY "${LOCALAI_VULKAN_HPP_SYSTEM_INCLUDE}/${LOCALAI_VULKAN_HEADER_TREE}"
            DESTINATION "${LOCALAI_VULKAN_HPP_INCLUDE}")
    endforeach()
    include_directories(SYSTEM "${LOCALAI_VULKAN_HPP_INCLUDE}")
endif()

if(LOCALAI_ENABLE_LLAMA)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
    set(LLAMA_SUBPROCESS OFF CACHE BOOL "" FORCE)
    set(MTMD_VIDEO OFF CACHE BOOL "" FORCE)
    set(GGML_CUDA ${LOCALAI_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(GGML_HIP ${LOCALAI_ENABLE_HIP} CACHE BOOL "" FORCE)
    set(GGML_VULKAN ${LOCALAI_ENABLE_VULKAN} CACHE BOOL "" FORCE)
    set(GGML_METAL ${LOCALAI_ENABLE_METAL} CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
            "${CMAKE_BINARY_DIR}/third-party/llama.cpp")
    else()
        FetchContent_Declare(llama_cpp
            GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
            GIT_TAG 08659901c43b51de735740f1cf61bb82fbe0c4e4
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(llama_cpp)
    endif()
endif()

if(LOCALAI_ENABLE_IMAGE_GENERATION)
    include(ExternalProject)
    set(SD_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/sd-install")
    if(WIN32)
        set(SD_RUNTIME_LIBRARY "${SD_INSTALL_PREFIX}/bin/stable-diffusion${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(SD_LINK_LIBRARY "${SD_INSTALL_PREFIX}/lib/stable-diffusion${CMAKE_IMPORT_LIBRARY_SUFFIX}")
        set(SD_BYPRODUCTS "${SD_RUNTIME_LIBRARY}" "${SD_LINK_LIBRARY}")
    else()
        set(SD_RUNTIME_LIBRARY "${SD_INSTALL_PREFIX}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}stable-diffusion${CMAKE_SHARED_LIBRARY_SUFFIX}")
        set(SD_LINK_LIBRARY "${SD_RUNTIME_LIBRARY}")
        set(SD_BYPRODUCTS "${SD_RUNTIME_LIBRARY}")
    endif()
    set(SD_EXTERNAL_SOURCE_ARGS
        GIT_REPOSITORY https://github.com/leejet/stable-diffusion.cpp.git
        GIT_TAG c6beeef35526c6dc94b74a7fb69f9d2e6a2a7a12
        GIT_SHALLOW TRUE
        GIT_SUBMODULES ggml)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp/CMakeLists.txt")
        set(SD_EXTERNAL_SOURCE_ARGS
            SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stable-diffusion.cpp"
            DOWNLOAD_COMMAND ""
            UPDATE_COMMAND "")
    endif()
    set(SD_PLATFORM_CMAKE_ARGS)
    if(CMAKE_MAKE_PROGRAM)
        list(APPEND SD_PLATFORM_CMAKE_ARGS
            -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM})
    endif()
    if(ANDROID)
        list(APPEND SD_PLATFORM_CMAKE_ARGS
            -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
            -DCMAKE_ANDROID_NDK=${CMAKE_ANDROID_NDK}
            -DANDROID_NDK=${CMAKE_ANDROID_NDK}
            -DANDROID_ABI=${ANDROID_ABI}
            -DANDROID_PLATFORM=${ANDROID_PLATFORM}
            -DANDROID_STL=${ANDROID_STL}
            -DSPIRV-Headers_DIR=/usr/share/cmake/SPIRV-Headers
            -DVulkan_INCLUDE_DIR=${LOCALAI_VULKAN_HPP_INCLUDE})
    endif()
    ExternalProject_Add(stable_diffusion_external
        ${SD_EXTERNAL_SOURCE_ARGS}
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${SD_INSTALL_PREFIX}
            -DSD_BUILD_EXAMPLES=OFF
            -DSD_BUILD_SHARED_LIBS=ON
            -DSD_WEBM=OFF
            -DSD_WEBP=OFF
            -DSD_CUDA=${LOCALAI_ENABLE_CUDA}
            -DSD_HIPBLAS=${LOCALAI_ENABLE_HIP}
            -DSD_VULKAN=${LOCALAI_ENABLE_VULKAN}
            -DSD_METAL=${LOCALAI_ENABLE_METAL}
            ${SD_PLATFORM_CMAKE_ARGS}
        UPDATE_DISCONNECTED TRUE
        BUILD_BYPRODUCTS ${SD_BYPRODUCTS}
    )
    ExternalProject_Get_Property(stable_diffusion_external SOURCE_DIR)
    file(MAKE_DIRECTORY "${SOURCE_DIR}/include")
    add_library(stable-diffusion SHARED IMPORTED GLOBAL)
    set_target_properties(stable-diffusion PROPERTIES
        IMPORTED_LOCATION "${SD_RUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SOURCE_DIR}/include"
    )
    if(WIN32)
        set_target_properties(stable-diffusion PROPERTIES IMPORTED_IMPLIB "${SD_LINK_LIBRARY}")
    endif()
    add_dependencies(stable-diffusion stable_diffusion_external)
    if(ANDROID)
        list(APPEND LOCALAI_ANDROID_EXTRA_LIBS "${SD_RUNTIME_LIBRARY}")
    endif()
    if(WIN32)
        install(FILES "${SD_RUNTIME_LIBRARY}" DESTINATION bin)
    elseif(APPLE)
        install(FILES "${SD_RUNTIME_LIBRARY}" DESTINATION lib)
        if(LOCALAI_BUILD_GUI)
            install(FILES "${SD_RUNTIME_LIBRARY}" DESTINATION "localai-desktop.app/Contents/Frameworks")
        endif()
    else()
        install(FILES "${SD_RUNTIME_LIBRARY}" DESTINATION lib)
    endif()
endif()

if(LOCALAI_ENABLE_WHISPER)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_CUDA ${LOCALAI_ENABLE_CUDA} CACHE BOOL "" FORCE)
    set(WHISPER_HIPBLAS ${LOCALAI_ENABLE_HIP} CACHE BOOL "" FORCE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp"
            "${CMAKE_BINARY_DIR}/third-party/whisper.cpp")
    else()
        FetchContent_Declare(whisper_cpp
            GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
            GIT_TAG 592feef04a1802b18cbeffd0fd0eb5d02570c2ec
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(whisper_cpp)
    endif()
endif()

if(LOCALAI_ENABLE_ONNX)
    if(ANDROID)
        set(LOCALAI_ONNXRUNTIME_ROOT
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/android/onnxruntime" CACHE PATH "ONNX Runtime Android SDK root")
        set(ONNXRUNTIME_INCLUDE_DIR "${LOCALAI_ONNXRUNTIME_ROOT}/include")
        set(ONNXRUNTIME_LIBRARY "${LOCALAI_ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so")
        if(NOT EXISTS "${ONNXRUNTIME_LIBRARY}")
            message(FATAL_ERROR "The bundled ONNX Runtime library does not support Android ABI ${ANDROID_ABI}")
        endif()
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
        list(APPEND LOCALAI_ANDROID_EXTRA_LIBS "${ONNXRUNTIME_LIBRARY}")
    else()
        find_package(onnxruntime CONFIG QUIET)
    endif()
    if(NOT TARGET onnxruntime::onnxruntime)
        set(LOCALAI_ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime native SDK root")
        find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            HINTS "${LOCALAI_ONNXRUNTIME_ROOT}"
            PATH_SUFFIXES include include/onnxruntime/core/session)
        find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
            HINTS "${LOCALAI_ONNXRUNTIME_ROOT}"
            PATH_SUFFIXES lib lib64 bin Release)
        if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
            message(FATAL_ERROR "ONNX Runtime SDK was not found. Set LOCALAI_ONNXRUNTIME_ROOT or ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY.")
        endif()
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
    endif()
endif()

if(LOCALAI_ENABLE_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

if(LOCALAI_ENABLE_LIBTORCH)
    find_package(Torch REQUIRED)
endif()

if(LOCALAI_ENABLE_LITERT)
    if(ANDROID)
        set(LOCALAI_LITERT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/android/litert"
            CACHE PATH "LiteRT Android SDK root")
        set(LITERT_INCLUDE_DIR "${LOCALAI_LITERT_ROOT}/include")
        if(NOT EXISTS "${LITERT_INCLUDE_DIR}/tensorflow/lite/c/c_api.h")
            message(FATAL_ERROR "The bundled LiteRT C API headers were not found under ${LITERT_INCLUDE_DIR}")
        endif()
    else()
        set(LOCALAI_LITERT_ROOT "" CACHE PATH "LiteRT/TFLite native SDK root")
        find_path(LITERT_INCLUDE_DIR tensorflow/lite/c/c_api.h
            HINTS "${LOCALAI_LITERT_ROOT}" PATH_SUFFIXES include REQUIRED)
    endif()
    if(ANDROID)
        set(LITERT_LIBRARY "${LOCALAI_LITERT_ROOT}/lib/${ANDROID_ABI}/libtensorflowlite_jni.so")
        set(LITERT_GPU_LIBRARY "${LOCALAI_LITERT_ROOT}/lib/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so")
        if(NOT EXISTS "${LITERT_LIBRARY}" OR NOT EXISTS "${LITERT_GPU_LIBRARY}")
            message(FATAL_ERROR "The bundled LiteRT libraries do not support Android ABI ${ANDROID_ABI}")
        endif()
        list(APPEND LOCALAI_ANDROID_EXTRA_LIBS "${LITERT_LIBRARY}" "${LITERT_GPU_LIBRARY}")
    else()
        find_library(LITERT_LIBRARY NAMES tensorflowlite_c tensorflow-lite
            HINTS "${LOCALAI_LITERT_ROOT}" PATH_SUFFIXES lib lib64 bin REQUIRED)
    endif()
    add_library(LiteRT::Runtime INTERFACE IMPORTED)
    set_target_properties(LiteRT::Runtime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LITERT_INCLUDE_DIR}")
endif()

if(LOCALAI_ENABLE_TTS)
    set(LOCALAI_SHERPA_DEPS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sherpa-deps")
    if(EXISTS "${LOCALAI_SHERPA_DEPS}/kaldi-native-fbank/CMakeLists.txt")
        set(FETCHCONTENT_SOURCE_DIR_KALDI_NATIVE_FBANK "${LOCALAI_SHERPA_DEPS}/kaldi-native-fbank")
        set(FETCHCONTENT_SOURCE_DIR_KISSFFT "${LOCALAI_SHERPA_DEPS}/kissfft")
        set(FETCHCONTENT_SOURCE_DIR_KALDI_DECODER "${LOCALAI_SHERPA_DEPS}/kaldi-decoder")
        set(FETCHCONTENT_SOURCE_DIR_KALDIFST "${LOCALAI_SHERPA_DEPS}/kaldifst")
        set(FETCHCONTENT_SOURCE_DIR_OPENFST "${LOCALAI_SHERPA_DEPS}/openfst")
        set(FETCHCONTENT_SOURCE_DIR_EIGEN "${LOCALAI_SHERPA_DEPS}/eigen")
        set(FETCHCONTENT_SOURCE_DIR_SIMPLE-SENTENCEPIECE "${LOCALAI_SHERPA_DEPS}/simple-sentencepiece")
        set(FETCHCONTENT_SOURCE_DIR_JSON "${LOCALAI_SHERPA_DEPS}/json")
        set(FETCHCONTENT_SOURCE_DIR_ESPEAK_NG "${LOCALAI_SHERPA_DEPS}/espeak-ng")
        set(FETCHCONTENT_SOURCE_DIR_PIPER_PHONEMIZE "${LOCALAI_SHERPA_DEPS}/piper-phonemize")
    endif()
    set(LOCALAI_APPLICATION_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
    set(CMAKE_CXX_STANDARD 17)
    set(SHERPA_ONNX_ENABLE_C_API ON CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_BINARY OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_BUILD_C_API_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_TTS ON CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_PORTAUDIO OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_JNI OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_ENABLE_PYTHON OFF CACHE BOOL "" FORCE)
    set(SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY OFF CACHE BOOL "" FORCE)
    if(ANDROID AND LOCALAI_ENABLE_ONNX)
        set(ENV{SHERPA_ONNXRUNTIME_INCLUDE_DIR} "${ONNXRUNTIME_INCLUDE_DIR}")
        get_filename_component(SHERPA_ANDROID_ORT_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
        set(ENV{SHERPA_ONNXRUNTIME_LIB_DIR} "${SHERPA_ANDROID_ORT_LIB_DIR}")
    endif()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sherpa-onnx/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/sherpa-onnx"
            "${CMAKE_BINARY_DIR}/third-party/sherpa-onnx")
    else()
        FetchContent_Declare(sherpa_onnx
            GIT_REPOSITORY https://github.com/k2-fsa/sherpa-onnx.git
            GIT_TAG 634265c9b57642fdd158120148785c89aa281c4b
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(sherpa_onnx)
    endif()
    set(CMAKE_CXX_STANDARD "${LOCALAI_APPLICATION_CXX_STANDARD}")
endif()
