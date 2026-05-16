#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
THIRD_LIBRARY_DIR="${ROOT_DIR}/3rdlibrary"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-24}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${THIRD_LIBRARY_DIR}/android-ndk-r26c}"
ANDROID_TOOLCHAIN_FILE="${ANDROID_TOOLCHAIN_FILE:-${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake}"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/android-${ANDROID_ABI}}"
OPENCV_DIR="${OpenCV_DIR:-${THIRD_LIBRARY_DIR}/OpenCV-android-sdk/sdk/native/jni}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:-${THIRD_LIBRARY_DIR}/v2.44.0.260225/qairt/2.44.0.260225}"

repair_ndk_clang_wrappers() {
  local llvm_bin="${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin"
  local clang17="${llvm_bin}/clang-17"
  local lld="${llvm_bin}/lld"

  if [[ ! -x "${clang17}" ]]; then
    echo "NDK clang-17 not found: ${clang17}" >&2
    exit 1
  fi

  if [[ ! -x "${lld}" ]]; then
    echo "NDK lld not found: ${lld}" >&2
    exit 1
  fi

  # 部分本地 NDK 压缩包会把 clang/clang++ 解成纯文本占位文件。
  # Android toolchain 需要这两个路径是可执行 wrapper。
  cat > "${llvm_bin}/clang" <<'EOF'
#!/usr/bin/env bash
exec -a "$(basename "$0")" "$(dirname "$0")/clang-17" "$@"
EOF
  chmod +x "${llvm_bin}/clang"

  cat > "${llvm_bin}/clang++" <<'EOF'
#!/usr/bin/env bash
exec -a "$(basename "$0")" "$(dirname "$0")/clang-17" "$@"
EOF
  chmod +x "${llvm_bin}/clang++"

  for linker in ld ld.lld ld64.lld lld-link; do
    if [[ -f "${llvm_bin}/${linker}" ]]; then
      cat > "${llvm_bin}/${linker}" <<'EOF'
#!/usr/bin/env bash
exec -a "$(basename "$0")" "$(dirname "$0")/lld" "$@"
EOF
      chmod +x "${llvm_bin}/${linker}"
    fi
  done

  declare -A llvm_tool_targets=(
    [llvm-ranlib]=llvm-ar
    [llvm-strip]=llvm-objcopy
    [llvm-addr2line]=llvm-symbolizer
    [llvm-readelf]=llvm-readobj
  )

  for tool in "${!llvm_tool_targets[@]}"; do
    local target="${llvm_tool_targets[${tool}]}"
    if [[ -f "${llvm_bin}/${tool}" && -x "${llvm_bin}/${target}" ]]; then
      cat > "${llvm_bin}/${tool}" <<EOF
#!/usr/bin/env bash
exec -a "\$(basename "\$0")" "\$(dirname "\$0")/${target}" "\$@"
EOF
      chmod +x "${llvm_bin}/${tool}"
    fi
  done
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --build-dir <path>       CMake build directory, default: ./build/android-arm64-v8a
  --opencv-dir <path>      Directory containing OpenCVConfig.cmake
                           default: ./3rdlibrary/OpenCV-android-sdk/sdk/native/jni
  --qnn-sdk-root <path>    QAIRT/QNN SDK root
                           default: ./3rdlibrary/v2.44.0.260225/qairt/2.44.0.260225
  --ndk-root <path>        Android NDK root, default: ./3rdlibrary/android-ndk-r26c
  --abi <name>             Android ABI, default: arm64-v8a
  --platform <name>        Android platform, default: android-24
  --debug                  Build Debug instead of Release
  --clean                  Remove build directory before configuring
  -j, --jobs <n>           Parallel build jobs
  -h, --help              Show this help

Examples:
  ./build.sh
  ./build.sh --clean
  ./build.sh --abi arm64-v8a --platform android-24
EOF
}

CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --opencv-dir)
      OPENCV_DIR="$2"
      shift 2
      ;;
    --qnn-sdk-root)
      QNN_SDK_ROOT="$2"
      shift 2
      ;;
    --ndk-root)
      ANDROID_NDK_ROOT="$2"
      ANDROID_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake"
      shift 2
      ;;
    --abi)
      ANDROID_ABI="$2"
      shift 2
      ;;
    --platform)
      ANDROID_PLATFORM="$2"
      shift 2
      ;;
    --debug)
      BUILD_TYPE="Debug"
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    -j|--jobs)
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${CLEAN}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi

if [[ ! -f "${ANDROID_TOOLCHAIN_FILE}" ]]; then
  echo "Android toolchain not found: ${ANDROID_TOOLCHAIN_FILE}" >&2
  exit 1
fi

if [[ ! -f "${OPENCV_DIR}/OpenCVConfig.cmake" ]]; then
  echo "OpenCVConfig.cmake not found under: ${OPENCV_DIR}" >&2
  exit 1
fi

if [[ ! -f "${QNN_SDK_ROOT}/include/QNN/QnnInterface.h" ]]; then
  echo "QNN SDK root is invalid: ${QNN_SDK_ROOT}" >&2
  exit 1
fi

repair_ndk_clang_wrappers

export PATH="${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/bin:${PATH}"

CMAKE_ARGS=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_TOOLCHAIN_FILE}"
  -DANDROID_ABI="${ANDROID_ABI}"
  -DANDROID_PLATFORM="${ANDROID_PLATFORM}"
  -DOpenCV_DIR="${OPENCV_DIR}"
  -DQNN_SDK_ROOT="${QNN_SDK_ROOT}"
)

cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"

echo
echo "Build finished: ${BUILD_DIR}/test_classify"
