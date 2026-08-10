# Bundled dependency versions

Application builds prefer these bundled sources and Android runtime binaries. CMake only uses the matching upstream `FetchContent` fallback when a bundled source directory is absent.

| Component | Pinned revision or release |
| --- | --- |
| llama.cpp | `08659901c43b51de735740f1cf61bb82fbe0c4e4` |
| whisper.cpp | `592feef04a1802b18cbeffd0fd0eb5d02570c2ec` |
| stable-diffusion.cpp | `c6beeef35526c6dc94b74a7fb69f9d2e6a2a7a12` |
| sherpa-onnx | `634265c9b57642fdd158120148785c89aa281c4b` |
| kaldi-native-fbank | `v1.22.3` |
| KissFFT | `febd4caeed32e33ad8b2e0bb5ea77542c40f18ec` |
| kaldi-decoder | `v0.3.0` |
| kaldifst | `v1.8.0` |
| OpenFST fork | `v1.8.5-2026-07-09` |
| Eigen | `5.0.1` archive |
| simple-sentencepiece | `v0.7` |
| nlohmann/json | `v3.12.0` |
| espeak-ng fork | `ed530aa113046142eb5115cf2fc9157854d0ffe1` |
| piper-phonemize fork | `f3ff95afc03640bc1399e113e83361192a2fafb4` |
| ONNX Runtime Android | `1.22.0`, official arm64 Android package |
| LiteRT Android | `0.1.0`, official arm64 runtime and GPU delegate packages |

Cryptographic hashes for the prebuilt Android shared libraries are stored in `android/SHA256SUMS` relative to this directory.
