list(APPEND CPACK_SOURCE_IGNORE_FILES
  /\.git/
  /\.gitignore
  /\.github/
  /\.tx/
  /\.travis.yml
  /_layouts/
  /android/
  /build/
  /chromium_extension/
  /data/
  /docs/
  /emscripten/
  /fastlane/
  /tools/
  /vim/
)

set(PACKAGE_NAME ${PROJECT_NAME})
set(VERSION ${PROJECT_VERSION})

set(prefix ${CMAKE_INSTALL_PREFIX})
set(exec_prefix ${CMAKE_INSTALL_PREFIX})
set(libdir "\${exec_prefix}/lib")
set(includedir "\${prefix}/include")
