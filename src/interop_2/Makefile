CXX = pgc++
FC = pgfortran
NVCC = nvcc

CXXFLAGS = -O3
FFLAGS = -O3 
NVCCFLAGS = -ccbin $(CXX) -O3

LDFLAGS = -L$(CUDA_DIR)/lib64
LIBS = -lcudart

# ===============================================

all: driver_f driver_acc driver_cu

driver_f: driver_f.o add_loop_gpu.o
	$(FC) $^ -Mcuda -o $@

driver_acc: driver_acc.o add_loop_gpu.o
	$(CXX) -acc -Minfo=acc $^ -o $@ $(LDFLAGS) $(LIBS) 

driver_cu: driver_cu.o add_loop_gpu.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LIBS) 

# ===============================================

driver_f.o: driver_f.f90
	$(FC) -Mcuda $(FFLAGS) $^ -c -o $@

driver_acc.o: driver_acc.c 
	$(CXX) -acc -Minfo=acc $(CXXFLAGS) $^ -c -o $@

driver_cu.o: driver.cu
	$(NVCC) $(NVCCFLAGS) $^ -c -o $@

add_loop_gpu.o: add_loop_gpu.cu
	$(NVCC) $(NVCCFLAGS) $^ -c -o $@

# ===============================================

clean:
	rm -rf add_loop_gpu.o driver_cu.o driver_acc.o driver_cu driver_acc

.PHONY: clean
