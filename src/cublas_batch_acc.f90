program main
  use cudafor
  use cublas
  implicit none

  integer :: mdim, stat, i, j, k, batch_count
  real(8), allocatable, dimension(:,:,:) :: A, B, C
  type(c_devptr), allocatable, dimension(:) :: devptr_A, devptr_B, devptr_C
  real(8) :: alpha, beta, idx, mysum
  type(cublasHandle) :: handle
  character(len=128) :: argv
  real :: clock_start, clock_end

  ! First arg is size of matrix
  call get_command_argument(1,argv)
  if (len_trim(argv) > 0) then
    read (argv, '(i)') mdim
  else
    mdim = 8
  endif

  ! Second arg is size of batch
  call get_command_argument(2,argv)
  if (len_trim(argv) > 0) then
    read (argv, '(i)') batch_count
  else
    batch_count = 1024
  endif

  write (*,'(A,I15)'),    'Matrix dim:       ', mdim
  write (*,'(A,I15)'),    'Batch count:      ', batch_count

  ! Allocate storage for host data and device pointers
  allocate(A(mdim,mdim,batch_count), B(mdim,mdim,batch_count), C(mdim,mdim,batch_count))
  allocate(devptr_A(batch_count), devptr_B(batch_count), devptr_C(batch_count))

  ! Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
  ! Matrices are arranged column major
  do k=1,batch_count
    do j=1,mdim
      do i=1,mdim
        if (i==j) then
          idx = real(j*mdim + i)
          A(i,j,k) = k*sin(idx)
          B(i,j,k) = sin(idx)
          C(i,j,k) = k*cos(idx) * cos(idx)
        else
          A(i,j,k) = 0.0
          B(i,j,k) = 0.0
          C(i,j,k) = 0.0
        endif
      enddo ! i
    enddo ! j
  enddo ! k

  ! Set matrix coefficients
  alpha = 1.0
  beta = 1.0

  ! Create cublas instance
  stat = cublasCreate(handle)

!$acc data copy(A, B, C) create(devptr_A, devptr_B, devptr_C)

!$acc host_data use_device(A, B, C)
  ! Set device pointers to device arrays
  do i = 1, batch_count
    devptr_A(i) = c_devloc(A(1,1,i))
    devptr_B(i) = c_devloc(B(1,1,i))
    devptr_C(i) = c_devloc(C(1,1,i))
  enddo
!$acc end host_data

!$acc update device(devptr_A, devptr_B, devptr_C)

  stat = cudaDeviceSynchronize()
  call cpu_time(clock_start)

!$acc host_data use_device(devptr_A, devptr_B, devptr_C)
  ! batched DGEMM: C = alpha*A*B + beta*C
  stat = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
      mdim, mdim, mdim, &
      alpha,         &
      devptr_A, mdim, &
      devptr_B, mdim, &
      beta,          &
      devptr_C, mdim, &
      batch_count)
!$acc end host_data

      stat = cudaDeviceSynchronize()
      call cpu_time(clock_end)

!$acc end data

  ! Simple sanity test, mysum up all elements
  mysum = 0.0
  do k=1,batch_count
    do j=1,mdim
      do i=1,mdim
        mysum = mysum + C(i,j,k)
      enddo
    enddo
  enddo
  print *, ''
  write (*,'(A,F15.3)'),  'Should be:        ', float(mdim*(batch_count)*(batch_count+1)/2)
  write (*,'(A,F15.3)'),  'Sum is:           ', mysum

  ! Report times, etc
  print *, ''
  write (*,'(A,ES15.3)'), 'Expect FLOP count:', real(batch_count * (2*mdim**3 + 3*mdim**2))
  write (*,'(A,ES15.3)'), 'Time (s):         ', clock_end - clock_start

  ! Cleanup
  deallocate(A)
  deallocate(B)
  deallocate(C)
  deallocate(devptr_A)
  deallocate(devptr_B)
  deallocate(devptr_C)
  stat =cublasDestroy(handle)

end program
