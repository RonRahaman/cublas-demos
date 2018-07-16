! Doesn't work.  c is just zero
program main
  use ISO_C_BINDING
  use cudafor
  implicit none
  integer, parameter :: n = 10
  integer(c_int) :: a(n), b(n), c(n)
  integer(c_int), device :: a_d(n), b_d(n), c_d(n)
  integer :: i

  interface
    attributes(global) subroutine add(a, b, c) bind(C, name='add_wrapper')
      use ISO_C_BINDING
      integer(c_int) :: a(*), b(*), c(*)
    end subroutine add
  end interface

  do i = 1, n
    a(i) = -(i-1)
    b(i) = (i-1)**2
  enddo

  a_d = a
  b_d = b

  call add<<<n,1>>>(a_d, b_d, c_d)

  c = c_d

  do i = 1, n
    print *, a(i), ' + ', b(i), ' = ', c(i)
  enddo

end program main
