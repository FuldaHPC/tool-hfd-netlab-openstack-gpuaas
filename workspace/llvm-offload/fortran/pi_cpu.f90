! pi with the midpoint rule in an OpenMP parallel do loop with a reduction on the CPU threads.
program pi_cpu
  use common
  use omp_lib
  implicit none
  real(real64) :: pi, start, seconds
  character(100) :: label
  pi = pi_omp(1000_int64)  ! warm-up: starts the OpenMP threads
  start = seconds_now()
  pi = pi_omp(pi_steps)
  seconds = seconds_now() - start
  write (label, '(a, " OpenMP, ", i0, " threads")') trim(compiler()), omp_get_max_threads()
  if (pi_report("CPU (" // trim(label) // ")", pi, pi_steps, seconds) /= 0) error stop 1

contains

  real(real64) function pi_omp(n)
    integer(int64), intent(in) :: n
    integer(int64) :: i
    real(real64) :: h, x, s
    h = 1.0_real64 / n
    s = 0
    !$omp parallel do reduction(+: s) private(x)
    do i = 0, n - 1
      x = (i + 0.5_real64) * h
      s = s + 4 / (1 + x * x)
    end do
    pi_omp = h * s
  end function

end program
