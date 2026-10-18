! Matrix product C = A B with an OpenMP parallel do loop over the columns of C on the CPU threads.
! Fortran stores matrices by columns, so this matches distributing rows in C, and the loop order
! j-k-i lets the innermost loop run along columns.
program matmul_cpu
  use common
  use omp_lib
  implicit none
  real(real32), allocatable :: a(:, :), b(:, :), c(:, :)
  real(real64) :: start, seconds
  character(100) :: label
  allocate (a(matrix_n, matrix_n), b(matrix_n, matrix_n), c(matrix_n, matrix_n))
  call matrix_init(a, b)
  c = 0  ! writes C once, so that the page faults of fresh memory do not fall into the timed region
  call matmul_omp(a(1:16, 1:16), b(1:16, 1:16), c(1:16, 1:16))  ! warm-up: starts the OpenMP threads
  start = seconds_now()
  call matmul_omp(a, b, c)
  seconds = seconds_now() - start
  write (label, '(a, " OpenMP, ", i0, " threads")') trim(compiler()), omp_get_max_threads()
  if (matrix_report("CPU (" // trim(label) // ")", a, b, c, seconds) /= 0) error stop 1

contains

  subroutine matmul_omp(a, b, c)
    real(real32), intent(in), contiguous :: a(:, :), b(:, :)
    real(real32), intent(out), contiguous :: c(:, :)
    integer :: i, j, k, n
    n = size(a, 1)
    !$omp parallel do private(i, k)
    do j = 1, n
      c(:, j) = 0
      do k = 1, n
        do i = 1, n
          c(i, j) = c(i, j) + a(i, k) * b(k, j)
        end do
      end do
    end do
  end subroutine

end program
