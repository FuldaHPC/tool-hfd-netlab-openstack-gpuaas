! pi with the midpoint rule in a plain loop on one CPU core (sequential), as reference for the
! parallel versions.
program pi_seq
  use common
  implicit none
  integer(int64) :: i
  real(real64) :: h, x, s, start
  h = 1.0_real64 / pi_steps
  s = 0
  start = seconds_now()
  do i = 0, pi_steps - 1
    x = (i + 0.5_real64) * h
    s = s + 4 / (1 + x * x)
  end do
  if (pi_report("CPU sequential (" // trim(compiler()) // ")", h * s, pi_steps, seconds_now() - start) /= 0) error stop 1
end program
