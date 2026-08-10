#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${LOCALAI_ANDROID_BUILD_DIR:-${project_dir}/build-android-arm64}"
qt_cmake="${LOCALAI_QT_ANDROID_CMAKE:-}"

if [[ -z "${qt_cmake}" ]]; then
    if command -v qt-cmake >/dev/null 2>&1; then
        qt_cmake="$(command -v qt-cmake)"
    elif [[ -n "${QT_ANDROID_ARM64:-}" && -x "${QT_ANDROID_ARM64}/bin/qt-cmake" ]]; then
        qt_cmake="${QT_ANDROID_ARM64}/bin/qt-cmake"
    else
        echo "Set LOCALAI_QT_ANDROID_CMAKE to the arm64 Qt for Android qt-cmake executable." >&2
        exit 2
    fi
fi

if [[ -z "${ANDROID_SDK_ROOT:-}" || -z "${ANDROID_NDK_ROOT:-}" ]]; then
    echo "ANDROID_SDK_ROOT and ANDROID_NDK_ROOT must point to installed Android SDK/NDK directories." >&2
    exit 2
fi

"${qt_cmake}" -S "${project_dir}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DLOCALAI_BUILD_GUI=ON \
    -DLOCALAI_BUILD_CLI=OFF \
    -DLOCALAI_BUILD_TESTS=OFF \
    -DLOCALAI_ENABLE_LLAMA=ON \
    -DLOCALAI_ENABLE_VULKAN=ON \
    -DLOCALAI_ENABLE_ONNX=ON \
    -DLOCALAI_ENABLE_LITERT=ON \
    -DLOCALAI_ENABLE_IMAGE_GENERATION=ON \
    -DLOCALAI_ENABLE_WHISPER=ON \
    -DLOCALAI_ENABLE_TTS=ON

cmake --build "${build_dir}" --parallel
cmake --build "${build_dir}" --target apk

echo "APK output directory: ${build_dir}/android-build/build/outputs/apk"
