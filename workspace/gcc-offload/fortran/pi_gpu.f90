! pi with the midpoint rule in an OpenMP target region on the GPU. GCC maps OpenMP threads to whole
! warps of the GPU; only "simd" spreads the work over the 32 lanes of a warp.
! Run with OMP_TARGET_OFFLOAD=MANDATORY, so that the program fails instead of falling back to the CPU.
program pi_gpu
  use common
  use omp_lib
  implicit none
  real(real64) :: pi, start, seconds
  logical :: on_host
  !$omp target map(from: on_host)
  on_host = omp_is_initial_device()
  !$omp end target
  pi = pi_target(1000_int64)  ! warm-up: initializes the GPU and loads the GPU code
  start = seconds_now()
  pi = pi_target(pi_steps)
  seconds = seconds_now() - start
  if (on_host) error stop "the target region ran on the host"
  if (pi_report("GPU (" // trim(compiler()) // " OpenMP target)", pi, pi_steps, seconds) /= 0) error stop 1

contains

  real(real64) function pi_target(n)
    integer(int64), intent(in) :: n
    integer(int64) :: i
    real(real64) :: h, x, s
    h = 1.0_real64 / n
    s = 0
    !$omp target teams distribute parallel do simd reduction(+: s) private(x) map(tofrom: s)
    do i = 0, n - 1
      x = (i + 0.5_real64) * h
      s = s + 4 / (1 + x * x)
    end do
    pi_target = h * s
  end function

end program
