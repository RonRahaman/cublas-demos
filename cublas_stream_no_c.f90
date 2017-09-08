program main
  use cudafor
  use cublas_v2
  implicit none

  integer :: dim, stat, i, j, k, m, n, batch_count
  real(8), allocatable, dimension(:,:,:) :: A, B, C
  real(8), allocatable, device, dimension(:,:,:) :: d_A, d_B, d_C
  real(8) :: alpha, beta, index, mysum
  type(cublasHandle) :: handle
  integer(cuda_stream_kind), allocatable, dimension(:) :: streams
  integer :: clock_start, clock_end, clock_rate

  ! Create cublas instance
  stat = cublasCreate(handle)

  print *, "dim,", "batch_count,", "sec_per_dgemm,"

  do m = 0,15  ! Batch sizes, 2**m, m=0..14

    batch_count = 2**m

    ! Allocate streams
    allocate(streams(batch_count))

    ! Create a stream for every DGEMM operation
    do i=1,batch_count
      stat = cudaStreamCreate(streams(i))
    enddo

    do n=1,5  ! Matrix sizes, 2**n x 2**n, n=0..5

      dim = 2**n

      ! Allocate host storage for A,B,C square matrices
      allocate(A(dim,dim,batch_count))
      allocate(B(dim,dim,batch_count))
      allocate(C(dim,dim,batch_count))
      allocate(d_A(dim,dim,batch_count))
      allocate(d_B(dim,dim,batch_count))
      allocate(d_C(dim,dim,batch_count))

      ! Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
      ! Matrices are arranged column major
      do k=1,batch_count
        do j=1,dim
          do i=1,dim
            if (i==j) then
              index = real(j*dim + i)
              A(i,j,k) = k*sin(index)
              B(i,j,k) = sin(index)
              C(i,j,k) = k*cos(index) * cos(index)
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
      call system_clock(clock_start, clock_rate)

      do i=1,batch_count
        ! Set CUDA stream
        stat = cublasSetStream(handle, streams(i))
        ! DGEMM: C = alpha*A*B + beta*C
        stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
            dim, dim, dim,    &
            alpha,            &
            d_A(:,:,i), dim,  &
            d_B(:,:,i), dim,  &
            beta,             &
            d_C(:,:,i), dim)
      enddo

      stat = cudaDeviceSynchronize()
      call system_clock(clock_end)

      ! Retrieve result matrix from device
      C = d_C

      ! Simple sanity test, mysum up all elements
      ! mysum = 0.0
      ! do k=1,batch_count
      !   do j=1,dim
      !     do i=1,dim
      !       mysum = mysum + C(i,j,k)
      !     enddo
      !   enddo
      ! enddo
      ! print *, "Sum is:", mysum, "should be: ", int(dim,8)*(batch_count)*(batch_count+1)/2

      ! Print columns: "dim, batch_count, sec_per_dgemm,"
      print *, dim, ",", batch_count, ",", real(clock_end - clock_start) / clock_rate / batch_count, ","

      deallocate(A)
      deallocate(B)
      deallocate(C)
      deallocate(d_A)
      deallocate(d_B)
      deallocate(d_C)

    enddo  ! Matrix sizes, 2**n x 2**n, n=0..5

    do i=1,batch_count
      stat = cudaStreamDestroy(streams(i))
    enddo

    deallocate(streams)

  enddo  ! Batch sizes, 2**m, m=0..14

  stat = cublasDestroy(handle)

end program
