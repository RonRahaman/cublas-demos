program main
  use cudafor
  use cublas
  implicit none

  integer :: dim, stat, i, j, k, batch_count
  real(8), allocatable, dimension(:,:,:) :: A, B, C
  real(8), device, allocatable, dimension(:,:,:) :: d_A, d_B, d_C
  type(c_devptr), allocatable, dimension(:) :: h_devptr_A, h_devptr_B, h_devptr_C
  type(c_devptr), device, allocatable, dimension(:) :: d_devptr_A, d_devptr_B, d_devptr_C
  real(8) :: alpha, beta, index, mysum
  type(cublasHandle) :: handle

  !Linear dimension of matrices
  dim = 100

  ! Number of A,B,C matrix sets
  batch_count = 1000

  ! Allocate host storage for A,B,C square matrices
  allocate(A(dim,dim,batch_count))
  allocate(B(dim,dim,batch_count))
  allocate(C(dim,dim,batch_count))
  allocate(d_A(dim,dim,batch_count))
  allocate(d_B(dim,dim,batch_count))
  allocate(d_C(dim,dim,batch_count))
  allocate(h_devptr_A(batch_count))
  allocate(h_devptr_B(batch_count))
  allocate(h_devptr_C(batch_count))
  allocate(d_devptr_A(batch_count))
  allocate(d_devptr_B(batch_count))
  allocate(d_devptr_C(batch_count))

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

  ! Copy host arrays to device
  d_A = A
  d_B = B
  d_C = C

  ! Set device pointers to device arrays
  do i = 1, batch_count
    h_devptr_A(i) = c_devloc(d_A(1,1,i))
    h_devptr_B(i) = c_devloc(d_B(1,1,i))
    h_devptr_C(i) = c_devloc(d_C(1,1,i))
  enddo
  d_devptr_A = h_devptr_A
  d_devptr_B = h_devptr_B
  d_devptr_C = h_devptr_C

  ! Create cublas instance
  stat = cublasCreate(handle)

  ! Set matrix coefficients
  alpha = 1.0
  beta = 1.0

  ! batched DGEMM: C = alpha*A*B + beta*C
  stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)
  stat = cublasDgemmBatched_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
      dim, dim, dim, &
      alpha,         &
      d_devptr_A, dim, &
      d_devptr_B, dim, &
      beta,          &
      d_devptr_C, dim, &
      batch_count)

  ! Retrieve result matrix from device
  C = d_C

  ! Simple sanity test, mysum up all elements
  mysum = 0.0
  do k=1,batch_count
    do j=1,dim
      do i=1,dim
        mysum = mysum + C(i,j,k)
      enddo
    enddo
  enddo
  print *, "Sum is:", mysum, "should be: ", dim*(batch_count)*(batch_count+1)/2

  ! Cleanup
  deallocate(A)
  deallocate(B)
  deallocate(C)
  deallocate(d_A)
  deallocate(d_B)
  deallocate(d_C)
  deallocate(d_devptr_A)
  deallocate(d_devptr_B)
  deallocate(d_devptr_C)
  stat =cublasDestroy(handle)

end program