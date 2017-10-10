program main
  implicit none

  integer, parameter :: ldim = 8, nbatch = 1024, niter = 10
  integer, parameter :: nblock = 16
  integer, parameter :: nblock_sq = nblock**2
  real(8), dimension(ldim, ldim, nbatch) :: Abatch, Bbatch, Cbatch
  real(8), dimension(ldim, ldim, 0:3) :: Ablock, Bblock
  real, dimension(niter) :: times

  integer :: i, j, k, iiter, ibatch, iblock
  real(8) :: idx, Ctemp, mysum
  real :: clock_start, clock_end

  write (*,'(A,I15)'),    'Matrix dim:         ', ldim
  write (*,'(A,I15)'),    'Batch count:        ', nbatch

  do iiter = 1, niter

    ! Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
    ! Matrices are arranged column major
    do ibatch = 1, nbatch
      do j = 1, ldim
        do i = 1, ldim
          if (i==j) then
            idx = real(j*ldim + i)
            Abatch(i,j,ibatch) = ibatch*sin(idx)
            Bbatch(i,j,ibatch) = sin(idx)
            Cbatch(i,j,ibatch) = ibatch*cos(idx) * cos(idx)
          else
            Abatch(i,j,ibatch) = 0.0
            Bbatch(i,j,ibatch) = 0.0
            Cbatch(i,j,ibatch) = 0.0
          endif
        enddo ! i
      enddo ! j
    enddo ! k

    !$acc data copyin(A, B) copy(C)

    !$acc wait
    call cpu_time(clock_start)

    !$acc parallel vector_length(nblock_sq)
    !$acc loop gang private(Ablock, Bblock)
    do ibatch = 1, nbatch, 4

      !$acc cache(Ablock, Bblock)

      !$acc loop vector collapse(3)
      do iblock = 0, 3
        do j = 1, ldim
          do i = 1, ldim
            Ablock(i,j,iblock) = Abatch(i,j,ibatch+iblock)
          enddo
        enddo
      enddo

      !$acc loop vector collapse(3)
      do iblock = 0, 3
        do j = 1, ldim
          do i = 1, ldim
            Bblock(i,j,iblock) = Bbatch(i,j,ibatch+iblock)
          enddo
        enddo
      enddo

      !$acc loop vector collapse(3)
      do iblock = 0, 3
        do j = 1, ldim
          do i = 1, ldim
            Ctemp = 0.0
            !$acc loop seq
            do k = 1, ldim
              Ctemp = Ctemp + Ablock(i,k,iblock) * Bblock(k,j,iblock)
            enddo
            Cbatch(i,j,ibatch+iblock) = Cbatch(i,j,ibatch+iblock) + Ctemp
          enddo
        enddo
      enddo

    enddo ! ibatch
    !$acc end parallel

    !$acc wait
    call cpu_time(clock_end)
    times(iiter) = clock_end - clock_start

    !$acc end data

    ! Simple sanity test, mysum up all elements
    mysum = 0.0
    do k=1,nbatch
      do j=1,ldim
        do i=1,ldim
          mysum = mysum + Cbatch(i,j,k)
        enddo
      enddo
    enddo
    print *, ''
    write (*,'(A,I15)'),    'Iter:               ', iiter
    write (*,'(A,F15.3)'),  'Sum is:             ', mysum
    write (*,'(A,F15.3)'),  'Should be:          ', float(ldim*(nbatch)*(nbatch+1)/2)

  enddo ! iiter

  ! Report times, etc
  print *, ''
  write (*,'(A,ES15.3)'), 'Expect FLOP count:    ', real(nbatch * (2*ldim**3 + 3*ldim**2))
  write (*,'(A,ES15.3,ES11.3,ES11.3)'), 'Avg/min/max time (s): ', &
      SUM(times)/SIZE(times), MINVAL(times), MAXVAL(times)

end program
