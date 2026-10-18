! pi with the midpoint rule in a do concurrent loop with a reduce clause (standard Fortran), which
! nvfortran -stdpar=gpu runs on the GPU.
program pi_stdpar
  use common
  implicit none
  real(real64) :: pi, start, seconds
  pi = pi_concurrent(1000_int64)  ! warm-up: initializes the GPU and loads the GPU code
  start = seconds_now()
  pi = pi_concurrent(pi_steps)
  seconds = seconds_now() - start
  if (pi_report("GPU (" // trim(compiler()) // " do concurrent)", pi, pi_steps, seconds) /= 0) error stop 1

contains

  real(real64) function pi_concurrent(n)
    integer(int64), intent(in) :: n
    integer(int64) :: i
    real(real64) :: h, x, s
    h = 1.0_real64 / n
    s = 0
    do concurrent (i = 0:n - 1) local(x) reduce(+: s)
      x = (i + 0.5_real64) * h
      s = s + 4 / (1 + x * x)
    end do
    pi_concurrent = h * s
  end function

end program
