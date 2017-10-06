program main
  use cudafor
  use cublas
  implicit none

  integer, parameter :: mdim = 8, batch_count = 1024, iter_count = 10
  real(8), dimension(mdim,mdim,batch_count) :: A, B, C
  type(c_devptr), dimension(batch_count) :: devptr_A, devptr_B, devptr_C
  real, dimension(iter_count) :: times

  integer :: stat, i, j, k, iter
  real(8) :: alpha, beta, idx, mysum
  type(cublasHandle) :: handle
  character(len=128) :: argv
  real :: clock_start, clock_end

  write (*,'(A,I15)'),    'Matrix dim:         ', mdim
  write (*,'(A,I15)'),    'Batch count:        ', batch_count

  do iter=1,iter_count

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
        times(iter) = clock_end - clock_start

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
    write (*,'(A,I15)'),    'Iter:               ', iter
    write (*,'(A,F15.3)'),  'Sum is:             ', mysum
    write (*,'(A,F15.3)'),  'Should be:          ', float(mdim*(batch_count)*(batch_count+1)/2)

  enddo ! iter_count

  ! Report times, etc
  print *, ''
  write (*,'(A,ES15.3)'), 'Expect FLOP count:    ', real(batch_count * (2*mdim**3 + 3*mdim**2))
  write (*,'(A,ES15.3,ES11.3,ES11.3)'), 'Avg/min/max time (s): ', &
      SUM(times)/SIZE(times), MINVAL(times), MAXVAL(times)

  ! Cleanup
  stat =cublasDestroy(handle)

end program
