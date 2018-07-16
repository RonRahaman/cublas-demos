! Doesn't work.  c is just zero
program main
  use ISO_C_BINDING
  implicit none
  integer, parameter :: n = 10
  integer :: a(n), b(n), c(n)
  integer :: i

  interface
    subroutine add_wrapper(a, b, c) bind(C, name='add_wrapper')
      integer :: a(:), b(:), c(:)
    end subroutine add_wrapper
  end interface

  do i = 1, n
    a(i) = -(i-1)
    b(i) = (i-1)**2
  enddo

!acc data copyin(a, b) copyout(c)
!acc host_data use_device(a, b, c)
   call add_wrapper(a, b, c)
!acc end data
!acc end host_data

  do i = 1, n
    print *, a(i), ' + ', b(i), ' = ', c(i)
  enddo

end program main
