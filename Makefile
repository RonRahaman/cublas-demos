FC = pgfortran
CC = nvcc
CFLAGS = -g --gpu-architecture=compute_50
FFLAGS = -g -ta=tesla:cc50,cuda8.0 -Mcuda=cc50,cuda8.0,keepgpu -Minfo=accel,ftn 

cublas_acc_device: cublas_acc_device.f90
	$(FC) $(FFLAGS) $^ -o $@ -lcublas_device

cublas_fortran.o: cublas_fortran.cu
	$(CC) $(CFLAGS) -c $^

cublas_stream: cublas_fortran_iso.f90 cublas_stream.f90 cublas_fortran.o 
	$(FC) $(FFLAGS) $^ -o $@

clean:
	rm -f cublas_acc_device cublas_stream *.mod *.o

.PHONY: clean
