! Matrix product C = A B in a do concurrent loop over all elements of C (standard Fortran), which
! nvfortran -stdpar=gpu runs on the GPU; the arrays move between host and GPU as CUDA managed memory
! when one side touches them. The time includes moving A and B to the GPU and, by summing C on the CPU,
! moving C back, as the other GPU versions copy C back.
program matmul_stdpar
  use common
  implicit none
  real(real32), allocatable :: a(:, :), b(:, :), c(:, :)
  real(real64) :: start, seconds
  real(real64), volatile :: checksum  ! volatile: the sum must be computed
  allocate (a(matrix_n, matrix_n), b(matrix_n, matrix_n), c(matrix_n, matrix_n))
  call matrix_init(a, b)
  c = 0  ! writes C once, so that the page faults of fresh memory do not fall into the timed region
  call matmul_concurrent(a(1:16, 1:16), b(1:16, 1:16), c(1:16, 1:16))  ! warm-up: GPU and GPU code
  start = seconds_now()
  call matmul_concurrent(a, b, c)
  checksum = sum(c)  ! reads C on the CPU, which moves it back from the GPU
  seconds = seconds_now() - start
  if (matrix_report("GPU (" // trim(compiler()) // " do concurrent)", a, b, c, seconds) /= 0) error stop 1

contains

  subroutine matmul_concurrent(a, b, c)
    real(real32), intent(in), contiguous :: a(:, :), b(:, :)
    real(real32), intent(out), contiguous :: c(:, :)
    integer :: i, j, k, n
    real(real32) :: s
    n = size(a, 1)
    do concurrent (j = 1:n, i = 1:n) local(k, s)
      s = 0
      do k = 1, n
        s = s + a(i, k) * b(k, j)
      end do
      c(i, j) = s
    end do
  end subroutine

end program
