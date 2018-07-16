program main
  use ISO_C_BINDING
  use cudafor
  !use add_kernel
  implicit none
  integer, parameter :: n = 10
  integer(c_int) :: a(n), b(n), c(n)
  integer :: i

  interface
    attributes(global) subroutine add(a, b, c) bind(C, name='add')
      use ISO_C_BINDING
      integer(c_int), intent(inout) :: a(*), b(*), c(*)
    end subroutine add
  end interface

  do i = 1, n
    a(i) = -(i-1)
    b(i) = (i-1)**2
  enddo

!$acc data copyin(a, b) copyout(c)
!$acc host_data use_device(a, b, c)
  call add<<<n,1>>>(a, b, c)
!$acc end host_data
!$acc end data

  do i = 1, n
    print *, a(i), ' + ', b(i), ' = ', c(i)
  enddo

end program main
