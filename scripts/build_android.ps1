$ErrorActionPreference = "Stop"

$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BuildDir = if ($env:LOCALAI_ANDROID_BUILD_DIR) { $env:LOCALAI_ANDROID_BUILD_DIR } else { Join-Path $ProjectDir "build-android-arm64" }
$QtCMake = $env:LOCALAI_QT_ANDROID_CMAKE

if (-not $QtCMake) {
    if ($env:QT_ANDROID_ARM64) {
        $QtCMake = Join-Path $env:QT_ANDROID_ARM64 "bin\qt-cmake.bat"
    }
}
if (-not $QtCMake -or -not (Test-Path $QtCMake)) {
    throw "Set LOCALAI_QT_ANDROID_CMAKE to the arm64 Qt for Android qt-cmake.bat file."
}
if (-not $env:ANDROID_SDK_ROOT -or -not $env:ANDROID_NDK_ROOT) {
    throw "ANDROID_SDK_ROOT and ANDROID_NDK_ROOT must point to installed Android SDK/NDK directories."
}

& $QtCMake -S $ProjectDir -B $BuildDir -G Ninja `
    -DCMAKE_BUILD_TYPE=Release `
    -DANDROID_ABI=arm64-v8a `
    -DANDROID_PLATFORM=android-28 `
    -DLOCALAI_BUILD_GUI=ON `
    -DLOCALAI_BUILD_CLI=OFF `
    -DLOCALAI_BUILD_TESTS=OFF `
    -DLOCALAI_ENABLE_LLAMA=ON `
    -DLOCALAI_ENABLE_VULKAN=ON `
    -DLOCALAI_ENABLE_ONNX=ON `
    -DLOCALAI_ENABLE_LITERT=ON `
    -DLOCALAI_ENABLE_IMAGE_GENERATION=ON `
    -DLOCALAI_ENABLE_WHISPER=ON `
    -DLOCALAI_ENABLE_TTS=ON

cmake --build $BuildDir --parallel
cmake --build $BuildDir --target apk
Write-Host "APK output directory: $BuildDir\android-build\build\outputs\apk"
