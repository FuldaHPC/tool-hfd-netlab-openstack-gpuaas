! Matrix product C = A B in an OpenACC parallel loop on the GPU: the columns of C are distributed over
! the gangs (blocks) and each column over the vector lanes (threads), so neighbouring threads take
! neighbouring rows, which are neighbours in memory. copyin/copyout move the matrices; the time
! includes it. The program fails unless the OpenACC device is an NVIDIA GPU.
program matmul_acc
  use common
  use openacc
  implicit none
  real(real32), allocatable :: a(:, :), b(:, :), c(:, :)
  real(real64) :: start, seconds
  allocate (a(matrix_n, matrix_n), b(matrix_n, matrix_n), c(matrix_n, matrix_n))
  call matrix_init(a, b)
  c = 0  ! writes C once, so that the page faults of fresh memory do not fall into the timed region
  call matmul_openacc(a(1:16, 1:16), b(1:16, 1:16), c(1:16, 1:16))  ! warm-up: GPU and GPU code
  start = seconds_now()
  call matmul_openacc(a, b, c)
  seconds = seconds_now() - start
  if (acc_get_device_type() /= acc_device_nvidia) error stop "OpenACC did not use the GPU"
  if (matrix_report("GPU (" // trim(compiler()) // " OpenACC)", a, b, c, seconds) /= 0) error stop 1

contains

  subroutine matmul_openacc(a, b, c)
    real(real32), intent(in), contiguous :: a(:, :), b(:, :)
    real(real32), intent(out), contiguous :: c(:, :)
    integer :: i, j, k, n
    real(real32) :: s
    n = size(a, 1)
    !$acc parallel loop gang copyin(a, b) copyout(c) private(i, k, s)
    do j = 1, n
      !$acc loop vector private(k, s)
      do i = 1, n
        s = 0
        do k = 1, n
          s = s + a(i, k) * b(k, j)
        end do
        c(i, j) = s
      end do
    end do
  end subroutine

end program
