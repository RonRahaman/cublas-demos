program main
  use cudafor
  use cublas
  implicit none

  integer :: n, stat, i, j
  real(8), allocatable, dimension(:,:) :: A, B, C
  real(8), allocatable, device, dimension(:,:) :: d_A, d_B, d_C
  real(8) :: alpha, beta, idx, mysum
  real(8), device :: d_alpha, d_beta
  type(cublasHandle) :: handle

  n = 8

  ! Create cublas instance
  stat = cublasCreate(handle)

  ! Allocate host storage for A,B,C square matrices
  allocate(A(n,n))
  allocate(B(n,n))
  allocate(C(n,n))
  allocate(d_A(n,n))
  allocate(d_B(n,n))
  allocate(d_C(n,n))

  ! Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
  ! Matrices are arranged column major
  do j=1,n
    do i=1,n
      if (i==j) then
        idx = real(j*n + i)
        A(i,j) = sin(idx)
        B(i,j) = sin(idx)
        C(i,j) = cos(idx) * cos(idx)
      else
        A(i,j) = 0.0
        B(i,j) = 0.0
        C(i,j) = 0.0
      endif
    enddo ! i
  enddo ! j

  ! Set matrix coefficients
  alpha = 1.0
  beta = 1.0

  d_A = A
  d_B = B
  d_C = C
  d_alpha = alpha
  d_beta = beta

  stat = cublasDgemm_v2( &
      handle, &
      CUBLAS_OP_N,  &
      CUBLAS_OP_N,  &
      n, n, n,      &
      d_alpha,        &
      d_A, n,  &
      d_B, n,  &
      d_beta,         &
      d_C, n)

  ! Retrieve result matrix from device
  C = d_C

  ! Simple sanity test, mysum up all elements
  mysum = 0.0
  do j=1,n
    do i=1,n
      mysum = mysum + C(i,j)
    enddo
  enddo
  print *, "Sum is:", mysum, "should be: ", int(n,8)*1/2

  deallocate(A)
  deallocate(B)
  deallocate(C)
  deallocate(d_A)
  deallocate(d_B)
  deallocate(d_C)

  stat = cublasDestroy(handle)

end program
