# Validation record

Validation performed on 2026-08-10:

- Clean CPU-only core configure, build, and unit test: passed.
- llama.cpp plus libmtmd application integration build: passed.
- Combined llama.cpp, libmtmd, and whisper.cpp application integration build: passed.
- Isolated stable-diffusion.cpp shared-library build plus application image backend link: passed.
- Unit suite under each compiled host profile: passed.
- Android ONNX Runtime, LiteRT, and sherpa-onnx/TTS dependency graph configure using the bundled arm64 runtimes and vendored source dependencies: passed.
- Android and host syntax checks for ONNX, LiteRT, and sherpa-onnx adapter branches: passed.
- Android runtime SHA-256 verification: passed.
- Android manifest and resource XML parsing: passed.
- GitHub Actions workflow YAML parsing: passed.
- Android shell build-script syntax check: passed.
- Application-owned source unfinished-marker scan: clean.
- Application-owned source moving-image/codec dependency scan: clean.

The validation container did not contain Qt for Android, an Android SDK, or an NDK, so APK creation was not executed there. `build-android.yml` installs the pinned toolchain, compiles all Android runtimes, invokes `androiddeployqt`, and requires an APK before artifact upload.

The build profiles were kept outside the delivered source tree. Vendored upstream projects retain their own internal tests, comments, and implementation history; application-owned completion scans intentionally distinguish those unmodified dependencies from Universal Local AI source.
