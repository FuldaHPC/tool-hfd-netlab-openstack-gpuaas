! Shared by the Fortran examples of this folder: problem sizes, test matrices, timer and result checks,
! so that the programs themselves only show the programming model.
module common
  use iso_fortran_env, only: int64, real32, real64, compiler_version
  implicit none
  integer(int64), parameter :: pi_steps = 1000000000_int64  ! steps of the midpoint rule for pi
  integer, parameter :: pi_flops_per_step = 6  ! x = (i + 0.5) * h; s = s + 4 / (1 + x * x)
  integer, parameter :: matrix_n = 2048  ! the matrices are matrix_n x matrix_n real32

contains

  ! Name of the compiler for the labels: the first word of compiler_version() (flang, nvfortran), or
  ! gfortran for GCC. A fixed-length result, because programs built with nvfortran -stdpar=gpu allocate
  ! allocatable variables in CUDA managed memory and could not free a string allocated here.
  function compiler() result(name)
    character(16) :: name
    character(200) :: version
    version = compiler_version()
    name = version(1:index(version, ' ') - 1)
    if (name == "GCC") name = "gfortran"
  end function

  function seconds_now() result(t)
    real(real64) :: t
    integer(int64) :: count, rate
    call system_clock(count, rate)
    t = real(count, real64) / real(rate, real64)
  end function

  function num(x, digits) result(s)
    real(real64), intent(in) :: x
    integer, intent(in) :: digits
    character(:), allocatable :: s
    character(40) :: buffer, form
    write (form, '("(f40.", i0, ")")') digits
    write (buffer, form) x
    s = trim(adjustl(buffer))
  end function

  ! pi is the integral of 4/(1+x^2) over [0,1]; the midpoint rule sums 4/(1+x^2) at x = (i+0.5)/n.
  ! Rounding makes the error about 1e-13; one missing step would add at least 2e-9.
  integer function pi_report(label, pi, n, seconds) result(status)
    character(*), intent(in) :: label
    real(real64), intent(in) :: pi, seconds
    integer(int64), intent(in) :: n
    real(real64) :: error
    character(12) :: e
    error = abs(pi - acos(-1.0_real64))
    write (e, '(es8.1)') error
    if (index(e, 'E') > 0) e(index(e, 'E'):index(e, 'E')) = 'e'  ! 2.8e-14 as in the other languages
    print '(a)', label // ": pi = " // num(pi, 12) // " (error " // trim(adjustl(e)) // ") in " // &
      num(seconds, 3) // " s (" // num(pi_flops_per_step * real(n, real64) / seconds / 1e9_real64, 1) // " GFLOP/s)"
    status = merge(0, 1, error < 1e-9_real64)
  end function

  ! Small integers, so that every product and sum in C = A B is exact in single precision
  ! (A(i, j) = mod(i + 2 j, 8) and B(i, j) = mod(3 i + j, 8) with indices from 0, as in the other languages).
  subroutine matrix_init(a, b)
    real(real32), intent(out) :: a(:, :), b(:, :)
    integer :: i, j
    do j = 1, size(a, 2)
      do i = 1, size(a, 1)
        a(i, j) = real(mod((i - 1) + 2 * (j - 1), 8), real32)
        b(i, j) = real(mod(3 * (i - 1) + (j - 1), 8), real32)
      end do
    end do
  end subroutine

  ! Checks C x == A (B x) for x = (1, 2, ..., n) in O(n^2); exact, as all values are integers below
  ! 2^53. x has no zero, so that every element of C counts.
  logical function matrix_correct(a, b, c) result(ok)
    real(real32), intent(in) :: a(:, :), b(:, :), c(:, :)
    real(real64), allocatable :: x(:), bx(:), cx(:), abx(:)
    integer :: j, n
    n = size(a, 1)
    x = [(real(j, real64), j = 1, n)]
    bx = matmul(real(b, real64), x)
    cx = matmul(real(c, real64), x)
    abx = matmul(real(a, real64), bx)
    ok = all(cx == abx)
  end function

  integer function matrix_report(label, a, b, c, seconds) result(status)
    character(*), intent(in) :: label
    real(real32), intent(in) :: a(:, :), b(:, :), c(:, :)
    real(real64), intent(in) :: seconds
    logical :: ok
    integer :: n
    n = size(a, 1)
    ok = matrix_correct(a, b, c)
    print '(a, ": ", i0, "x", i0, " matrix product ", a, " in ", a, " s (", a, " GFLOP/s)")', label, n, n, &
      trim(merge("correct", "WRONG  ", ok)), num(seconds, 3), num(2 * real(n, real64)**3 / seconds / 1e9_real64, 1)
    status = merge(0, 1, ok)
  end function

end module
