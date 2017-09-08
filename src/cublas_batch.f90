program main
  use ISO_C_BINDING
  use cublas_f
  implicit none

  integer :: dim, stat, i, j, k, batch_count
  integer(8) :: bytes
  real(8),dimension(:,:,:), pointer:: A, B, C
  real(8) :: alpha, beta, index, sum
  type(C_PTR) :: d_A, d_B, d_C
  type(C_PTR), dimension(:), pointer :: h_d_A, h_d_B, h_d_C, streams
  type(C_PTR) :: handle

  integer :: sizeof_double
  parameter (sizeof_double=8)

  !Linear dimension of matrices
  dim = 100

  ! Number of A,B,C matrix sets
  batch_count = 1000

  ! Allocate host storage for A,B,C square matrices
  allocate(A(dim,dim,batch_count))
  allocate(B(dim,dim,batch_count))
  allocate(C(dim,dim,batch_count))

  ! Create host pointer array to device matrix storage
  allocate(h_d_A(batch_count))
  allocate(h_d_B(batch_count))
  allocate(h_d_C(batch_count))
  bytes = dim*dim*sizeof_double

  do i=1,batch_count
    stat = cudaMalloc(h_d_A(i), bytes)
    stat = cudaMalloc(h_d_B(i), bytes)
    stat = cudaMalloc(h_d_C(i), bytes)
  enddo

  ! Copy the host array of device pointers to the device
  bytes = batch_count*sizeof_double ! 8 byte pointers
  stat = cudaMalloc(d_A, bytes)
  stat = cudaMalloc(d_B, bytes)
  stat = cudaMalloc(d_C, bytes)

  stat = cudaMemcpy(d_A, C_LOC(h_d_A(1)), bytes, cudaMemcpyHostToDevice);
  stat = cudaMemcpy(d_B, C_LOC(h_d_B(1)), bytes, cudaMemcpyHostToDevice);
  stat = cudaMemcpy(d_C, C_LOC(h_d_C(1)), bytes, cudaMemcpyHostToDevice);

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

  ! Create cublas instance
  stat = cublasCreate(handle)

  ! Set input matrices on device
  do i=1,batch_count
    stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(A(1,1,i)), dim, h_d_A(i), dim)
    stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(B(1,1,i)), dim, h_d_B(i), dim)
    stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(C(1,1,i)), dim, h_d_C(i), dim)
  enddo

  ! Set matrix coefficients
  alpha = 1.0
  beta = 1.0

  ! batched DGEMM: C = alpha*A*B + beta*C
  stat = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
      dim, dim, dim, &
      alpha,         &
      d_A, dim,      &
      d_B, dim,      &
      beta,          &
      d_C, dim,      &
      batch_count)

  ! Retrieve result matrix from device
  do i=1,batch_count
    stat = cublasGetMatrix(dim, dim, sizeof_double, h_d_C(i), dim, C_LOC(C(1,1,i)), dim)
  enddo

  ! Simple sanity test, sum up all elements
  sum = 0.0
  do k=1,batch_count
    do j=1,dim
      do i=1,dim
        sum = sum + C(i,j,k)
      enddo
    enddo
  enddo
  print *, "Sum is:", sum, "should be: ", dim*(batch_count)*(batch_count+1)/2

  do i=1,batch_count
    call cudaFree(h_d_A(i))
    call cudaFree(h_d_B(i))
    call cudaFree(h_d_C(i))
  enddo

  deallocate(A)
  deallocate(B)
  deallocate(C)
  call cublasDestroy(handle)

end program
