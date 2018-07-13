PGI = /opt/pgi/linux86-64/2016/
FC = $(PGI)/bin/pgfortran
CC = nvcc
CFLAGS = -ccbin $(PGI)/bin/pgc++ -O3 --gpu-architecture=compute_50
FFLAGS = -O3 -ta=tesla:cc50,cuda8.0 -Mcuda=cc50,cuda8.0,keepgpu,keepptx -Minfo=accel,ftn 

dot_driver: dot_driver.f90 dot.o
	$(FC) -acc $(FFLAGS $^ -o $@

dot.o: dot.cu
	$(CC) $(CFLAGS) -c $^ -o $@
