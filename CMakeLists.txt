cmake_minimum_required(VERSION 3.24)

project(UniversalLocalAI VERSION 0.3.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build dependencies as static libraries unless isolated" FORCE)

if(ANDROID)
    set(LOCALAI_DESKTOP_DEFAULT OFF)
    set(LOCALAI_ANDROID_ACCELERATION_DEFAULT ON)
else()
    set(LOCALAI_DESKTOP_DEFAULT ON)
    set(LOCALAI_ANDROID_ACCELERATION_DEFAULT OFF)
endif()
option(LOCALAI_BUILD_GUI "Build the Qt desktop or Android application" ON)
option(LOCALAI_BUILD_CLI "Build the command-line application" ${LOCALAI_DESKTOP_DEFAULT})
option(LOCALAI_BUILD_TESTS "Build unit tests" ${LOCALAI_DESKTOP_DEFAULT})
set(LOCALAI_TEST_MODELS_DIR "" CACHE PATH "Directory containing optional real smoke-test models")
option(LOCALAI_ENABLE_LLAMA "Enable llama.cpp GGUF inference" ON)
option(LOCALAI_ENABLE_ONNX "Enable ONNX Runtime" ${LOCALAI_ANDROID_ACCELERATION_DEFAULT})
option(LOCALAI_ENABLE_LITERT "Enable LiteRT/TFLite" ${LOCALAI_ANDROID_ACCELERATION_DEFAULT})
option(LOCALAI_ENABLE_LIBTORCH "Enable LibTorch/TorchScript" OFF)
option(LOCALAI_ENABLE_OPENVINO "Enable OpenVINO devices" OFF)
option(LOCALAI_ENABLE_IMAGE_GENERATION "Enable stable-diffusion.cpp still-image generation" ON)
option(LOCALAI_ENABLE_WHISPER "Enable whisper.cpp speech recognition" ON)
option(LOCALAI_ENABLE_TTS "Enable sherpa-onnx speech synthesis" ${LOCALAI_ANDROID_ACCELERATION_DEFAULT})
option(LOCALAI_ENABLE_CUDA "Enable CUDA in compatible native engines" OFF)
option(LOCALAI_ENABLE_HIP "Enable HIP/ROCm in compatible native engines" OFF)
option(LOCALAI_ENABLE_VULKAN "Enable Vulkan in compatible native engines" ${LOCALAI_ANDROID_ACCELERATION_DEFAULT})
option(LOCALAI_ENABLE_METAL "Enable Metal on macOS" ${APPLE})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(Dependencies)

add_library(localai_core STATIC
    src/core/backend.cpp
    src/core/benchmark.cpp
    src/core/log.cpp
    src/core/model.cpp
    src/core/scheduler.cpp
    src/core/settings.cpp
    src/media/audio/wav.cpp
    src/platform/hardware.cpp
)
add_library(LocalAI::Core ALIAS localai_core)
target_include_directories(localai_core PUBLIC include)
target_compile_features(localai_core PUBLIC cxx_std_23)

if(WIN32)
    target_link_libraries(localai_core PRIVATE wbemuuid ole32)
endif()
if(ANDROID)
    target_sources(localai_core PRIVATE src/platform/android_hardware.cpp)
    target_link_libraries(localai_core PRIVATE android log vulkan)
    target_compile_definitions(localai_core PUBLIC LOCALAI_ANDROID=1)
endif()

if(LOCALAI_ENABLE_LLAMA)
    target_sources(localai_core PRIVATE src/backends/llama_backend.cpp src/backends/vlm_backend.cpp)
    target_link_libraries(localai_core PRIVATE llama mtmd)
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_LLAMA=1)
endif()

if(LOCALAI_ENABLE_ONNX)
    target_sources(localai_core PRIVATE src/backends/onnx_backend.cpp)
    target_link_libraries(localai_core PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_ONNX=1)
endif()

if(LOCALAI_ENABLE_LITERT)
    target_sources(localai_core PRIVATE src/backends/litert_backend.cpp)
    target_link_libraries(localai_core PRIVATE LiteRT::Runtime ${CMAKE_DL_LIBS})
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_LITERT=1
        PRIVATE LOCALAI_LITERT_COMPILED_LIBRARY="${LITERT_LIBRARY}")
    if(LITERT_GPU_LIBRARY)
        target_compile_definitions(localai_core PRIVATE
            LOCALAI_LITERT_GPU_LIBRARY="${LITERT_GPU_LIBRARY}")
    endif()
endif()

if(LOCALAI_ENABLE_LIBTORCH)
    target_sources(localai_core PRIVATE src/backends/libtorch_backend.cpp)
    target_link_libraries(localai_core PRIVATE "${TORCH_LIBRARIES}")
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_LIBTORCH=1)
endif()

if(LOCALAI_ENABLE_OPENVINO)
    target_sources(localai_core PRIVATE src/providers/openvino_provider.cpp)
    target_link_libraries(localai_core PRIVATE openvino::runtime)
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_OPENVINO=1)
endif()

if(LOCALAI_ENABLE_IMAGE_GENERATION)
    target_sources(localai_core PRIVATE src/backends/image_backend.cpp)
    target_link_libraries(localai_core PRIVATE stable-diffusion)
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_IMAGE_GENERATION=1)
    if(LOCALAI_ENABLE_CUDA OR LOCALAI_ENABLE_HIP OR LOCALAI_ENABLE_VULKAN OR LOCALAI_ENABLE_METAL)
        target_compile_definitions(localai_core PRIVATE LOCALAI_IMAGE_ACCELERATOR_ENABLED=1)
    endif()
endif()

if(LOCALAI_ENABLE_WHISPER)
    target_sources(localai_core PRIVATE src/backends/whisper_backend.cpp)
    target_link_libraries(localai_core PRIVATE whisper)
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_WHISPER=1)
    if(LOCALAI_ENABLE_CUDA OR LOCALAI_ENABLE_HIP OR LOCALAI_ENABLE_VULKAN OR LOCALAI_ENABLE_METAL)
        target_compile_definitions(localai_core PRIVATE LOCALAI_WHISPER_ACCELERATOR_ENABLED=1)
    endif()
endif()

if(LOCALAI_ENABLE_TTS)
    target_sources(localai_core PRIVATE src/backends/tts_backend.cpp)
    target_link_libraries(localai_core PRIVATE sherpa-onnx-c-api)
    target_compile_definitions(localai_core PUBLIC LOCALAI_HAS_TTS=1)
endif()

if(LOCALAI_BUILD_CLI)
    add_executable(localai-cli src/cli/main.cpp)
    target_link_libraries(localai-cli PRIVATE LocalAI::Core)
    install(TARGETS localai-cli RUNTIME DESTINATION bin)
endif()

if(LOCALAI_BUILD_GUI)
    find_package(Qt6 6.4 REQUIRED COMPONENTS Core Gui Widgets Multimedia)
    qt_standard_project_setup()
    qt_add_executable(localai-desktop
        src/app/main.cpp
        src/ui/main_window.cpp
        src/ui/main_window.hpp
    )
    target_include_directories(localai-desktop PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(localai-desktop PRIVATE LocalAI::Core Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Multimedia)
    if(ANDROID)
        set_property(TARGET localai-desktop PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
            "${CMAKE_CURRENT_SOURCE_DIR}/android")
        set_property(TARGET localai-desktop PROPERTY QT_ANDROID_PACKAGE_NAME "ai.local.universal")
        set_property(TARGET localai-desktop PROPERTY QT_ANDROID_APP_NAME "Universal Local AI")
        set_property(TARGET localai-desktop PROPERTY QT_ANDROID_VERSION_CODE 300)
        set_property(TARGET localai-desktop PROPERTY QT_ANDROID_VERSION_NAME "0.3.0")
        set_property(TARGET localai-desktop PROPERTY QT_ANDROID_MIN_SDK_VERSION 28)
        set_property(TARGET localai-desktop PROPERTY QT_ANDROID_TARGET_SDK_VERSION 35)
        if(LOCALAI_ANDROID_EXTRA_LIBS)
            set_property(TARGET localai-desktop PROPERTY QT_ANDROID_EXTRA_LIBS
                "${LOCALAI_ANDROID_EXTRA_LIBS}")
        endif()
    endif()
    set_target_properties(localai-desktop PROPERTIES
        WIN32_EXECUTABLE TRUE
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_BUNDLE_NAME "Universal Local AI"
        MACOSX_BUNDLE_GUI_IDENTIFIER "ai.local.universal"
        INSTALL_RPATH "$<$<PLATFORM_ID:Darwin>:@executable_path/../Frameworks>;$<$<PLATFORM_ID:Linux>:$ORIGIN/../lib>"
    )
    if(NOT ANDROID)
        install(TARGETS localai-desktop
            BUNDLE DESTINATION .
            RUNTIME DESTINATION bin
        )
    endif()
endif()

if(LOCALAI_BUILD_CLI)
    set_target_properties(localai-cli PROPERTIES
        INSTALL_RPATH "$<$<PLATFORM_ID:Darwin>:@executable_path/../lib>;$<$<PLATFORM_ID:Linux>:$ORIGIN/../lib>"
    )
endif()

if(LOCALAI_BUILD_TESTS)
    enable_testing()
    add_executable(localai-tests tests/core_tests.cpp)
    target_link_libraries(localai-tests PRIVATE LocalAI::Core)
    add_test(NAME localai-core-tests COMMAND localai-tests)
    if(LOCALAI_TEST_MODELS_DIR AND LOCALAI_ENABLE_LLAMA AND LOCALAI_ENABLE_WHISPER)
        add_executable(localai-model-smoke tests/model_smoke.cpp)
        target_link_libraries(localai-model-smoke PRIVATE LocalAI::Core)
        add_test(NAME localai-real-model-smoke COMMAND localai-model-smoke "${LOCALAI_TEST_MODELS_DIR}")
        set_tests_properties(localai-real-model-smoke PROPERTIES TIMEOUT 300)
    endif()
endif()

install(FILES README.md LICENSE THIRD_PARTY_NOTICES.md DESTINATION share/UniversalLocalAI)
if(LOCALAI_ENABLE_LITERT)
    if(WIN32)
        install(FILES "${LITERT_LIBRARY}" DESTINATION bin)
    else()
        install(FILES "${LITERT_LIBRARY}" DESTINATION lib)
    endif()
endif()
if(UNIX AND NOT APPLE AND LOCALAI_BUILD_GUI)
    install(FILES resources/ai.local.universal.desktop DESTINATION share/applications)
    install(FILES resources/universal-local-ai.svg DESTINATION share/icons/hicolor/scalable/apps)
endif()

include(Packaging)
