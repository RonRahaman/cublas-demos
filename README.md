cuBLAS Demos
============

This repo contains cuBLAS demos from several sources of documentation.  

* **cublas\_acc\_device** calls `cublasSswap` from an OpenACC device kernel.  It is from
  Section 6.2 of PGI's [Fortran CUDA Library Interfaces, v.  2017](https://www.pgroup.com/doc/pgi17cudaint.pdf).

* **cublas\_stream** calls `cublasDgemm` from the host using multiple streams.
  It is from OLCF's tutorial, [Concurrent Kernels II: Batched Library Calls](https://www.olcf.ornl.gov/tutorials/concurrent-kernels-ii-batched-library-calls/#Streams_1).
  Note that it uses a custom Fortran interface to the C cuBLAS v2 functions. It
  appears that, when the tutorial was written, NVIDIA did not provide a Fortran
  interface to cuBLAS v2.  

* **cublas\_stream\_no\_c** is a version of cublas\_stream that uses NVIDIA's
  current Fortran interfaces to cuBLAS v2.  It was written by me, Ron Rahaman.  

* **cublas\_batch** calls `cublasDgemmBatched` to launch multiple dgemm operations with one call.  
  It is also from OLCF's tutorial, [Concurrent Kernels II: Batched Library Calls](https://www.olcf.ornl.gov/tutorials/concurrent-kernels-ii-batched-library-calls/#Batched_1).
  Like cublas\_stream,  it uses a custom Fortran interface to the C cuBLAS v2 functions.
