FC = pgfortran
FFLAGS = -g -ta=tesla:cc50,cuda8.0 -Mcuda=cc50,cuda8.0,keepgpu -Minfo=accel,ftn 

cublas_acc_device: cublas_acc_device.f90
	$(FC) $(FFLAGS) $^ -o $@ -lcublas_device

clean:
	rm -f cublas_acc_device *.mod *.gpu

.PHONY: clean
