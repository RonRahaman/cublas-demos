program main
  use cudafor
  use cublas
  implicit none

  integer :: dim, stat, i, j, k, batch_count
  real(8), allocatable, dimension(:,:,:) :: A, B, C
  type(c_devptr), allocatable, dimension(:) :: devptr_A, devptr_B, devptr_C
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
  allocate(devptr_A(batch_count))
  allocate(devptr_B(batch_count))
  allocate(devptr_C(batch_count))

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

!$acc host_data use_device(devptr_A, devptr_B, devptr_C)
  ! batched DGEMM: C = alpha*A*B + beta*C
  stat = cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
      dim, dim, dim, &
      alpha,         &
      devptr_A, dim, &
      devptr_B, dim, &
      beta,          &
      devptr_C, dim, &
      batch_count)
!$acc end host_data

!$acc end data

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
  deallocate(devptr_A)
  deallocate(devptr_B)
  deallocate(devptr_C)
  stat =cublasDestroy(handle)

end program
