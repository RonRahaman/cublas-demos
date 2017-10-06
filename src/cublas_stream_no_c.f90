program main
  use cudafor
  use cublas_v2
  implicit none

  integer, parameter :: mdim = 8, batch_count = 1024, iter_count = 10
  real(8), dimension(mdim,mdim,batch_count) :: A, B, C
  real(8), device, dimension(mdim,mdim,batch_count) :: d_A, d_B, d_C
  integer(cuda_stream_kind), dimension(batch_count) :: streams
  real, dimension(iter_count) :: times

  integer :: stat, i, j, k, iter
  real(8) :: alpha, beta, idx, mysum
  type(cublasHandle) :: handle
  real :: clock_start, clock_end

  write (*,'(A,I15)'),    'Matrix dim:         ', mdim
  write (*,'(A,I15)'),    'Batch count:        ', batch_count

  ! Create cublas instance
  stat = cublasCreate(handle)
  ! Create a stream for every DGEMM operation
  do i=1,batch_count
    stat = cudaStreamCreate(streams(i))
  enddo

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

    d_A = A
    d_B = B
    d_C = C

    ! Set matrix coefficients
    alpha = 1.0
    beta = 1.0

    ! Launch each DGEMM operation in own CUDA stream
    stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)

    stat = cudaDeviceSynchronize()
    call cpu_time(clock_start)

    do i=1,batch_count
      ! Set CUDA stream
      stat = cublasSetStream(handle, streams(i))
      ! DGEMM: C = alpha*A*B + beta*C
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
          mdim, mdim, mdim,  &
          alpha,             &
          d_A(:,:,i), mdim,  &
          d_B(:,:,i), mdim,  &
          beta,              &
          d_C(:,:,i), mdim)
    enddo

    stat = cudaDeviceSynchronize()
    call cpu_time(clock_end)
    times(iter) = clock_end - clock_start

    ! Retrieve result matrix from device
    C = d_C

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
      sum(times)/size(times), minval(times), maxval(times)

  stat = cublasDestroy(handle)

end program
