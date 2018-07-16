CXX = pgc++
FC = pgfortran
NVCC = nvcc

CXXFLAGS = -O3
FFLAGS = -O3 
NVCCFLAGS = -ccbin $(CXX) -O3

CUDA_DIR ?= /soft/compilers/cuda/cuda-9.0.176
LDFLAGS = -L$(CUDA_DIR)/lib64
LIBS = -lcudart

# ===============================================

all: driver_f_acc driver_f driver_acc driver_cu

driver_f_acc: driver_f_acc.o add_loop_gpu.o
	$(FC) $^ -Mcuda=cc60,cuda9.0 -acc -ta=tesla:cc60,cuda9.0 -Minfo=acc -o $@ -lstdc++

driver_f: driver_f.o add_loop_gpu.o
	$(FC) $^ -Mcuda -o $@ -lstdc++

driver_acc: driver_acc.o add_loop_gpu.o
	$(CXX) -acc -Minfo=acc $^ -o $@ $(LDFLAGS) $(LIBS) 

driver_cu: driver_cu.o add_loop_gpu.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS) 

# ===============================================

driver_f_acc.o: driver_f_acc.f90
	$(FC) -Mcuda=cc60,cuda9.0 -acc -ta=tesla:cc60,cuda9.0 -Minfo=acc $(FFLAGS) $^ -c -o $@

driver_f.o: driver_f.f90
	$(FC) -Mcuda $(FFLAGS) $^ -c -o $@

driver_acc.o: driver_acc.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -acc -Xcompiler -Minfo=acc $^ -c -o $@

driver_cu.o: driver.cu
	$(NVCC) $(NVCCFLAGS) $^ -c -o $@

add_loop_gpu.o: add_loop_gpu.cu
	$(NVCC) $(NVCCFLAGS) $^ -c -o $@

# ===============================================

clean:
	rm -rf add_loop_gpu.o driver_cu.o driver_acc.o driver_f_acc.o driver_cu driver_acc driver_f

.PHONY: clean
