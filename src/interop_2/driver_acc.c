#include "common.h"

int main( void ) {
    int a[N], b[N], c[N];

    // fill the arrays 'a' and 'b' on the CPU
    for (int i=0; i<N; i++) {
        a[i] = -i;
        b[i] = i * i;
    }

#pragma acc data copyin(a, b) copyout(c)
    {
#pragma acc host_data use_device(a, b, c)
      {
        add_wrapper(a, b, c);
      }
    }

    // display the results
    for (int i=0; i<N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }

    return 0;
}

