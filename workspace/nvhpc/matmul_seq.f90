! Matrix product C = A B in plain loops on one CPU core (sequential), as reference for the parallel
! versions. Fortran stores matrices by columns, so the loop order j-k-i lets the innermost loop run
! along columns.
program matmul_seq
  use common
  implicit none
  real(real32), allocatable :: a(:, :), b(:, :), c(:, :)
  real(real64) :: start
  integer :: i, j, k, n
  n = matrix_n
  allocate (a(n, n), b(n, n), c(n, n))
  call matrix_init(a, b)
  c = 0  ! writes C once, so that the page faults of fresh memory do not fall into the timed region
  start = seconds_now()
  do j = 1, n
    c(:, j) = 0
    do k = 1, n
      do i = 1, n
        c(i, j) = c(i, j) + a(i, k) * b(k, j)
      end do
    end do
  end do
  if (matrix_report("CPU sequential (" // trim(compiler()) // ")", a, b, c, seconds_now() - start) /= 0) error stop 1
end program
