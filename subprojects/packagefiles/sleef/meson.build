project('sleef', meson_version: '>=1.1')

cmake = find_program('cmake')
ninja = find_program('ninja', 'make', required: false)
cc = meson.get_compiler('c')

sleef_build_dir = 'sleef_build'
sleef_install_dir = 'sleef_install'

# turning off parallel build in windows
parallel_flag = ['--parallel']
build_config_flag = []
if host_machine.system() == 'windows'
  parallel_flag = []
  # On Windows with MSVC (multi-config generator), must specify --config Release
  # to match the main project's release build and avoid runtime library mismatches
  build_config_flag = ['--config', 'Release']
endif

# For building sleef with TSan, delete the sleef subproject and follow the README instructions to build sleef externally.
# Enable SIMD extensions that are OFF by default but required by qblas (will change in future)
sleef_simd_flags = []
sleef_purecfma_flag = []

# Check for force-disable FMA option (for cross-compilation or emulation scenarios)
force_disable_fma = get_option('disable_fma')

if host_machine.cpu_family() == 'x86_64' or host_machine.cpu_family() == 'x86'
  sleef_simd_flags = ['-DSLEEF_ENABLE_SSE2=ON']
  
  if force_disable_fma
    # User explicitly requested no FMA
    message('FMA explicitly disabled via option - disabling PURECFMA scalar for x86-64-v2 compatibility')
    sleef_purecfma_flag = ['-DSLEEF_DISABLE_PURECFMA_SCALAR=ON']
  else
    # Auto-detect FMA support at configure time by actually running FMA code
    fma_test_result = cc.run('''
      #include <immintrin.h>
      int main(void) {
        __m128 a = _mm_set1_ps(1.0f);
        __m128 b = _mm_set1_ps(2.0f);
        __m128 c = _mm_set1_ps(3.0f);
        __m128 r = _mm_fmadd_ps(a, b, c);
        (void)r;
        return 0;
      }
    ''', args: ['-mfma'], name: 'FMA instruction runtime support')
    
    has_fma = fma_test_result.compiled() and fma_test_result.returncode() == 0
    
    if not has_fma
      message('FMA not supported at runtime - disabling PURECFMA scalar code path')
      sleef_purecfma_flag = ['-DSLEEF_DISABLE_PURECFMA_SCALAR=ON']
    else
      message('FMA supported - enabling PURECFMA scalar code path')
    endif
  endif
endif

sleef_configure = run_command([
    cmake, 
    '-S', meson.current_source_dir(),
    '-B', meson.current_build_dir() / sleef_build_dir,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DSLEEF_BUILD_QUAD=ON',
    '-DSLEEF_BUILD_SHARED_LIBS=OFF',
    '-DSLEEF_BUILD_TESTS=OFF',
    '-DSLEEF_BUILD_INLINE_HEADERS=OFF',
    '-DSLEEF_ENABLE_TLFLOAT=OFF', # this is only used for testing in SLEEF, not runtime
    '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
    '-DCMAKE_INSTALL_PREFIX=' + meson.current_build_dir() / sleef_install_dir
] + sleef_simd_flags + sleef_purecfma_flag, check: false, capture: true)

if sleef_configure.returncode() != 0
    error('SLEEF CMake configuration failed: ' + sleef_configure.stderr())
endif

sleef_build_target = custom_target('sleef_build',
    command: [cmake, '--build', meson.current_build_dir() / sleef_build_dir, '--target', 'install'] + build_config_flag + parallel_flag,
    output: 'sleef_built.stamp',
    console: true,
    build_always_stale: true,
    build_by_default: true
)

sleef_include_path = meson.current_build_dir() / sleef_install_dir / 'include'

sleef_build_dep = declare_dependency(sources: [sleef_build_target])

sleef_static_define = ''
if host_machine.system() == 'windows'
  sleef_static_define = '-DSLEEF_STATIC_LIBS'
endif

compile_args_list = ['-I' + sleef_include_path]
if sleef_static_define != ''
  compile_args_list += sleef_static_define
endif

sleef_dep = declare_dependency(
    dependencies: [sleef_build_dep],
    compile_args: compile_args_list,
    link_args: ['-L' + meson.current_build_dir() / sleef_install_dir / 'lib', '-L' + meson.current_build_dir() / sleef_install_dir / 'lib64', '-lsleef'] #both lib and lib64 because of ubuntu vs redhat compatibility
)

sleefquad_dep = declare_dependency(
    dependencies: [sleef_build_dep, sleef_dep],
    compile_args: compile_args_list,
    link_args: ['-L' + meson.current_build_dir() / sleef_install_dir / 'lib', '-L' + meson.current_build_dir() / sleef_install_dir / 'lib64', '-lsleefquad']
)