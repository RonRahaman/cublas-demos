FC = pgfortran
CC = nvcc
CFLAGS = -O3 --gpu-architecture=compute_50
FFLAGS = -O3 -ta=tesla:cc50,cuda8.0 -Mcuda=cc50,cuda8.0,keepgpu,keepptx -Minfo=accel,ftn 
OBJ = cublas_acc_device cublas_stream cublas_stream_no_c cublas_batch cublas_batch_no_c cublas_batch_acc cublas_single cublas_strided_batch

all: $(OBJ)

cublas_acc_device: cublas_acc_device.f90
	$(FC) $(FFLAGS) $^ -o $@ -lcublas_device

cublas_fortran.o: cublas_fortran.cu
	$(CC) $(CFLAGS) -c $^

cublas_stream: cublas_fortran_iso.f90 cublas_stream.f90 cublas_fortran.o 
	$(FC) $(FFLAGS) $^ -o $@

cublas_stream_no_c: cublas_stream_no_c.f90 
	$(FC) $(FFLAGS) $^ -o $@

cublas_batch: cublas_fortran_iso.f90 cublas_batch.f90 cublas_fortran.o 
	$(FC) $(FFLAGS) $^ -o $@

cublas_batch_no_c: cublas_batch_no_c.f90
	$(FC) $(FFLAGS) $^ -o $@

cublas_batch_acc: cublas_batch_acc.f90
	$(FC) -acc $(FFLAGS) $^ -o $@

cublas_single: cublas_single.f90
	$(FC) $(FFLAGS) -Mcudalib=cublas $^ -o $@

cublas_strided_batch: cublas_fortran_iso.f90 cublas_strided_batch.f90 cublas_fortran.o 
	$(FC) $(FFLAGS) $^ -o $@

clean:
	rm -f $(OBJ) *.mod *.o *.gpu *.ptx

.PHONY: clean 
