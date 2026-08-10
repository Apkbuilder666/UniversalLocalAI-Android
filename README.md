# Universal Local AI for Android

Universal Local AI is a native C++23 and Qt 6 Android application that runs supported text, still-image, and audio models locally. It targets 64-bit Android phones from Android 9 (API 28) onward, including the Samsung Galaxy A52s. The APK does not request Internet permission and inference never uses a cloud service.

Version 0.3.0 adds an Android touch layout, Storage Access Framework imports, runtime microphone permission, Vulkan and NNAPI discovery, real accelerated provider initialization, and automatic CPU recovery.

## Android execution paths

| Model | Android accelerated path | Verified fallback |
| --- | --- | --- |
| GGUF text and paired-projector VLM | llama.cpp/GGML Vulkan layer offload | llama.cpp CPU/ARM NEON |
| ONNX numeric models and SqueezeNet 1.1 | ONNX Runtime NNAPI execution provider | ONNX Runtime CPU execution provider |
| TFLite/LiteRT numeric models and MobileNet v1 | LiteRT NNAPI delegate or GPU delegate | LiteRT CPU interpreter |
| still-image diffusion models | stable-diffusion.cpp compiled with Vulkan | stable-diffusion.cpp CPU |
| whisper.cpp speech recognition | compiled GGML accelerator selection | whisper.cpp CPU |
| sherpa-onnx speech synthesis bundles | native sherpa-onnx C API | ONNX Runtime CPU |

NNAPI is Android's graph-routing API. A successful NNAPI session proves the NNAPI provider accepted and executed the model, but ONNX Runtime and LiteRT do not expose exact per-operation placement. The app therefore reports `NNAPI provider executed; exact GPU/NPU placement unavailable` instead of claiming a particular NPU. If the phone exposes no compatible accelerator driver, the selected graph is unsupported, or allocation fails, the app records the failure and retries on CPU.

On the Galaxy A52s, Vulkan is the primary GGUF GPU route. ONNX and LiteRT try the phone's NNAPI drivers. Other Samsung, Qualcomm, MediaTek, Google, and non-Samsung phones use the same capability-based logic; the manufacturer name is never used as evidence of acceleration.

## Implemented workspaces

- Chat: streaming generation, conversations, system prompt, stop strings, edit/regenerate, cancellation, token counts, first-token time, and tokens per second.
- Vision: still-image file or clipboard input, paired GGUF projector VLMs, SqueezeNet ONNX classification, and MobileNet LiteRT classification.
- Image: text-to-image, image-to-image, mask inpainting, negative prompt, seed, steps, guidance, dimensions, sampler, scheduler, progress, cancellation, preview, and save.
- Audio: WAVE input, microphone capture, whisper.cpp transcription, sherpa-onnx TTS playback, and WAVE saving.
- Models: inspection, format/capability classification, memory estimate, persistent catalog, compatible runtime filtering, load, unload, and removal from the catalog.
- Hardware: SoC/product identity, ARM capabilities, RAM, Vulkan physical devices, NNAPI GPU/NPU drivers, and compiled runtime state.
- Benchmarks, settings, structured logs, and execution verification.

There is no moving-image workspace, camera capture, moving-image import, playback, generation, or transcoding path.

## Bundled native dependencies

The source package contains the exact arm64 runtime files needed by the Android target:

- llama.cpp `08659901c43b51de735740f1cf61bb82fbe0c4e4`
- whisper.cpp `592feef04a1802b18cbeffd0fd0eb5d02570c2ec`
- stable-diffusion.cpp `c6beeef35526c6dc94b74a7fb69f9d2e6a2a7a12`
- sherpa-onnx `634265c9b57642fdd158120148785c89aa281c4b`
- sherpa-onnx's pinned kaldi-native-fbank, KissFFT, kaldi-decoder, kaldifst,
  OpenFST, Eigen, simple-sentencepiece, nlohmann/json, espeak-ng, and
  piper-phonemize source dependencies
- ONNX Runtime Android 1.22.0 arm64 binary and native headers
- LiteRT 0.1.0 arm64 runtime, GPU delegate, and native headers

The three prebuilt runtime checksums are in `third_party/android/SHA256SUMS`. Upstream licenses remain beside the code or binary SDK. Qt itself is installed through the official Qt installer because its installation and license selection are developer-specific.

## Build an APK

Install:

- Qt 6.8.3 for Android `arm64-v8a`, including Qt Multimedia, plus its matching desktop host tools
- Android SDK platform 35 and build tools 35.0.0
- Android NDK 26.1.10909125
- CMake 3.24 or newer, Ninja, JDK 17, and `glslc`

Linux/macOS:

```bash
export ANDROID_SDK_ROOT=/path/to/Android/Sdk
export ANDROID_NDK_ROOT="$ANDROID_SDK_ROOT/ndk/26.1.10909125"
export QT_ANDROID_ARM64=/path/to/Qt/6.8.3/android_arm64_v8a
./scripts/build_android.sh
```

Windows PowerShell:

```powershell
$env:ANDROID_SDK_ROOT = "C:\Android\Sdk"
$env:ANDROID_NDK_ROOT = "$env:ANDROID_SDK_ROOT\ndk\26.1.10909125"
$env:QT_ANDROID_ARM64 = "C:\Qt\6.8.3\android_arm64_v8a"
.\scripts\build_android.ps1
```

The scripts configure every shipped Android runtime, build the native library, run `androiddeployqt`, and create an APK under `build-android-arm64/android-build/build/outputs/apk`. The GitHub Actions workflow performs the same release build and uploads the APK artifact.

## Import and run models

1. Open Models and choose **Import model or bundle**.
2. Select a model through Android's document picker.
3. The app copies document-provider content into private app storage so access remains valid after restart. Direct filesystem paths are referenced without another copy.
4. Select Auto, CPU, GPU, or NPU in Settings. Only discovered device classes are shown.
5. Select the model, choose Auto or a compatible runtime, and load it.
6. Run the matching workspace and inspect the Hardware or result metrics for the real provider outcome.

Large model copies consume internal storage. Android document URIs cannot safely be persisted as ordinary C++ filesystem paths across every provider, so the private copy is deliberate. Removing a catalog entry does not delete a user-owned source model.

## Models recognized by the adapters

- GGUF language models supported by the bundled llama.cpp revision.
- Paired multimodal GGUF plus `mmproj` files recognized by libmtmd, including the supplied SmolVLM 256M pair.
- whisper.cpp GGML speech models such as `ggml-tiny.en-q5_1.bin`.
- Exact SqueezeNet 1.1 ONNX image classifier signature and generic single-input float tensor ONNX graphs.
- Exact MobileNet v1 224 float TFLite image classifier signature and generic single-input float tensor TFLite graphs.
- stable-diffusion.cpp compatible GGUF/SafeTensors still-image models.
- sherpa-onnx Kokoro, KittenTTS, and VITS bundle layouts recognized by the native TTS adapter.

An ONNX or TFLite container alone does not identify preprocessing. Unknown numeric graphs can run through the tensor adapter, while a user-facing vision task is enabled only for a recognized signature. Python `.pt` checkpoints and state dictionaries are rejected; only executable TorchScript is accepted in LibTorch-enabled desktop builds.

## Memory behavior

Balanced mode defaults to four worker threads, a 2K token context, a 128-token batch, and 384×384 image generation on Android. When total RAM is below 6 GiB or currently available RAM is below 3 GiB, model loading further limits context, batch, and initial GPU offload. Accelerator allocation failures trigger a clean CPU reload instead of terminating the app.

For the Galaxy A52s, start with the supplied Stories GGUF or SmolVLM 256M pair. Still-image diffusion models are much heavier; use a compact quantized model and 256–384 pixel output on 6 GiB devices.

## Security and privacy

- No Internet permission, telemetry, cloud inference, local server, Python runtime, browser UI, or subprocess inference.
- Android document-provider input is treated as untrusted and inspected with bounded parsers.
- Model repositories and scripts are never executed.
- Microphone access is requested only when recording starts.
- Logs, settings, benchmark records, and persistent imports remain in the app's private data directory.

## Validation

Run host-side core tests without optional runtimes:

```bash
cmake -S . -B build-core -G Ninja \
  -DLOCALAI_BUILD_GUI=OFF \
  -DLOCALAI_ENABLE_LLAMA=OFF \
  -DLOCALAI_ENABLE_IMAGE_GENERATION=OFF \
  -DLOCALAI_ENABLE_WHISPER=OFF \
  -DLOCALAI_ENABLE_ONNX=OFF \
  -DLOCALAI_ENABLE_LITERT=OFF \
  -DLOCALAI_ENABLE_TTS=OFF
cmake --build build-core --parallel
ctest --test-dir build-core --output-on-failure
```

For device validation, install the APK, import the supplied small models, run one inference through each selected provider, and verify the provider plus fallback messages in Logs. Hardware detection alone is never displayed as proof of model execution.
