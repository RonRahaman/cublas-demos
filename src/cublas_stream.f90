program main
  use ISO_C_BINDING
  use cublas_f
  implicit none

  integer, parameter :: mdim = 8, batch_count = 1024, iter_count = 10
  integer :: stat, i, j, k, iter
  integer(8) :: bytes
  real(8),dimension(:,:,:), pointer:: A, B, C
  real(8) :: alpha, beta, idx, mysum
  type(C_PTR), dimension(:), pointer :: d_A, d_B, d_C, streams
  type(C_PTR) :: handle
  real :: clock_start, clock_end
  real, dimension(iter_count) :: times

  integer, parameter :: sizeof_double = 8

  write (*,'(A,I15)'),    'Matrix dim:         ', mdim
  write (*,'(A,I15)'),    'Batch count:        ', batch_count

  ! Create cublas instance
  stat = cublasCreate(handle)

  ! Allocate host storage for A,B,C square matrices
  allocate(A(mdim,mdim,batch_count))
  allocate(B(mdim,mdim,batch_count))
  allocate(C(mdim,mdim,batch_count))

  ! Allocate device storage for A,B,C
  allocate(d_A(batch_count))
  allocate(d_B(batch_count))
  allocate(d_C(batch_count))

  bytes = int(mdim*mdim*sizeof_double, 8)
  do i=1,batch_count
    stat = cudaMalloc(d_A(i), bytes)
    stat = cudaMalloc(d_B(i), bytes)
    stat = cudaMalloc(d_C(i), bytes)
  enddo

  ! Create a stream for every DGEMM operation
  allocate(streams(batch_count))
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

    ! Set input matrices on device
    do i=1,batch_count
      stat = cublasSetMatrix(mdim, mdim, sizeof_double, C_LOC(A(1,1,i)), mdim, d_A(i), mdim)
      stat = cublasSetMatrix(mdim, mdim, sizeof_double, C_LOC(B(1,1,i)), mdim, d_B(i), mdim)
      stat = cublasSetMatrix(mdim, mdim, sizeof_double, C_LOC(C(1,1,i)), mdim, d_C(i), mdim)
    enddo

    ! Set matrix coefficients
    alpha = 1.0
    beta = 1.0

    call cudaDeviceSynchronize()
    call cpu_time(clock_start)

    ! Launch each DGEMM operation in own CUDA stream
    do i=1,batch_count
      ! Set CUDA stream
      stat = cublasSetStream(handle, streams(i))
      ! DGEMM: C = alpha*A*B + beta*C
      stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
          mdim, mdim, mdim, &
          alpha,         &
          d_A(i), mdim,   &
          d_B(i), mdim,   &
          beta,          &
          d_C(i), mdim)
    enddo

    call cudaDeviceSynchronize()
    call cpu_time(clock_end)
    times(iter) = clock_end - clock_start

    ! Retrieve result matrix from device
    do i=1,batch_count
      stat = cublasGetMatrix(mdim, mdim, sizeof_double, d_C(i), mdim, C_LOC(C(1,1,i)), mdim)
    enddo

    ! Simple sanity test, sum up all elements
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

  do i=1,batch_count
    stat = cudaStreamDestroy(streams(i))
    call cudaFree(d_A(i))
    call cudaFree(d_B(i))
    call cudaFree(d_C(i))
  enddo

  deallocate(A)
  deallocate(B)
  deallocate(C)
  call cublasDestroy(handle)

end program
