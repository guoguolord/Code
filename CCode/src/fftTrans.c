//
// Created by 果果lord on 2024/10/16.
//

#include <math.h>
#include <complex.h>

void fft(complex float* x, int n){
    if (n <= 1)
        return;

    complex float even[n/2];
    complex float odd[n/2];

    for (int i = 0; i < n/2; i++)
    {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }
    fft(even, n/2);
    fft(odd, n/2);
    for (int i=0; i<n/2; i++){
        complex float t = cexp(-2*M_PI*i/n)*odd[i];
        x[i] = even[i] + t;
        x[i+n/2] = even[i] - t;
    }
}

void ifft(complex float* x, int n){
    for (int i = 0; i < n; i++)
        x[i] = conj(x[i]);

    fft(x, n);
    for (int i = 0; i < n; i++)
        x[i] = conj(x[i]) / n;
}