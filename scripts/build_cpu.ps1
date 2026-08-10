$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$BuildDir = if ($args.Count -gt 0) { $args[0] } else { Join-Path $ProjectDir "build" }

cmake -S $ProjectDir -B $BuildDir -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DLOCALAI_ENABLE_CUDA=OFF `
  -DLOCALAI_ENABLE_HIP=OFF `
  -DLOCALAI_ENABLE_VULKAN=OFF
cmake --build $BuildDir --parallel
ctest --test-dir $BuildDir --output-on-failure
