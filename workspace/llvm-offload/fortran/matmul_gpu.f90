! Matrix product C = A B in an OpenMP target region on the GPU with one iteration per element of C
! (collapse(2); gfortran needs another mapping, see gcc-offload/fortran).
! map(to:) copies A and B to the GPU and map(from:) copies C back; the time includes it.
! Run with OMP_TARGET_OFFLOAD=MANDATORY, so that the program fails instead of falling back to the CPU.
program matmul_gpu
  use common
  use omp_lib
  implicit none
  real(real32), allocatable :: a(:, :), b(:, :), c(:, :)
  real(real64) :: start, seconds
  logical :: on_host
  !$omp target map(from: on_host)
  on_host = omp_is_initial_device()
  !$omp end target
  allocate (a(matrix_n, matrix_n), b(matrix_n, matrix_n), c(matrix_n, matrix_n))
  call matrix_init(a, b)
  c = 0  ! writes C once, so that the page faults of fresh memory do not fall into the timed region
  call matmul_target(a(1:16, 1:16), b(1:16, 1:16), c(1:16, 1:16))  ! warm-up: GPU and GPU code
  start = seconds_now()
  call matmul_target(a, b, c)
  seconds = seconds_now() - start
  if (on_host) error stop "the target region ran on the host"
  if (matrix_report("GPU (" // trim(compiler()) // " OpenMP target)", a, b, c, seconds) /= 0) error stop 1

contains

  subroutine matmul_target(a, b, c)
    real(real32), intent(in), contiguous :: a(:, :), b(:, :)
    real(real32), intent(out), contiguous :: c(:, :)
    integer :: i, j, k, n
    real(real32) :: s
    n = size(a, 1)
    !$omp target teams distribute parallel do collapse(2) map(to: a, b) map(from: c) private(k, s)
    do j = 1, n
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
