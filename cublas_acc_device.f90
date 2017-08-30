module mtests
  integer, parameter :: n = 100
contains
  subroutine testcu(a,b)
      use openacc_cublas
      real :: a(n), b(n)
      type(cublasHandle)  :: h
      !$acc parallel num_gangs(1) copy(a,b,h)
      j = cublasCreate(h)
      j = cublasSswap(h,n,a,1,b,1)
      j = cublasDestroy(h)
      !$acc end parallel
      return
  end subroutine
end module mtests

program cublas_acc_device
  use mtests
  real :: a(n), b(n), c(n)
  logical :: passing
  a = 1.0
  b = 2.0
  passing = .true.
  call testcu(a,b)
  print *, "Should all be 2.0"
  print *, a
  passing = passing .and. all(a == 2.0)
  print *, "Should all be 1.0"
  print *, b
  passing = passing .and. all(b == 1.0)
  if (passing) then
    print *, "Test PASSED"
  else
    print *, "Test FAILED"
  endif
end program cublas_acc_device


