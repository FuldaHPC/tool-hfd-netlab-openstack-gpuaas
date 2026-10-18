! pi with the midpoint rule in an OpenACC parallel loop with a reduction on the GPU; the program fails
! unless the OpenACC device is an NVIDIA GPU.
program pi_acc
  use common
  use openacc
  implicit none
  real(real64) :: pi, start, seconds
  pi = pi_openacc(1000_int64)  ! warm-up: initializes the GPU and loads the GPU code
  start = seconds_now()
  pi = pi_openacc(pi_steps)
  seconds = seconds_now() - start
  if (acc_get_device_type() /= acc_device_nvidia) error stop "OpenACC did not use the GPU"
  if (pi_report("GPU (" // trim(compiler()) // " OpenACC)", pi, pi_steps, seconds) /= 0) error stop 1

contains

  real(real64) function pi_openacc(n)
    integer(int64), intent(in) :: n
    integer(int64) :: i
    real(real64) :: h, x, s
    h = 1.0_real64 / n
    s = 0
    !$acc parallel loop reduction(+: s) private(x)
    do i = 0, n - 1
      x = (i + 0.5_real64) * h
      s = s + 4 / (1 + x * x)
    end do
    pi_openacc = h * s
  end function

end program
