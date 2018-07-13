program dot_driver
  use ISO_C_BINDING
  implicit none

  interface
    subroutine dot(A, B, C, N, blocksPerGrid, threadsPerBlock) BIND(C, name='f_dot')
        use ISO_C_BINDING
        real(C_PTR) :: A, B, C
        integer, value :: N, blocksPerGrid, threadsPerBlock
    end subroutine dot

  integer, parameter :: gridDim = 4
  integer, parameter :: blockDim = 4
  integer, parameter :: N = gridDim * blockDim

  real, target, dimension(N) :: A, B, C
  integer :: i

  do i = 1, N
    A(i) = i
    B(i) = -i
    C(i) = 0
  enddo

!$acc data copyin(A, B) copyout(C)
!$acc host_data use_device(A, B, C)
  f_dot(C_LOC(A), C_LOC(B), C_LOC(C), N, gridDim, blockDim)
!$acc end host_data
!$acc end data

  do i = 1, N
    print *, C(i)
  enddo
end program dot_driver
