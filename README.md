cuBLAS Demos
============

This repo contains cuBLAS demos from several sources of documentation.  

* **cublas-acc-device** calls `cublasSswap` from an OpenACC device kernel.  It is from
  Section 6.2 of PGI's [Fortran CUDA Library Interfaces, v.  2017](https://www.pgroup.com/doc/pgi17cudaint.pdf).
* **cublas-stream** calls `cublasDgemm` from the host using multiple streams.  It is
  from OLCF's tutorial, [Concurrent Kernels II: Batched Library Calls](https://www.olcf.ornl.gov/tutorials/concurrent-kernels-ii-batched-library-calls/#Streams_1).  
