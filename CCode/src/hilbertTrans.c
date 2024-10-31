//
// Created by 果果lord on 2024/10/16.
//

#include <complex.h>
#include "../include/fftTrans.h"

void hilbert(float* input, complex float* output, int n){
    // Convert input signal to complex type
    complex float x[n];
    for (int i = 0; i < n; i++) {
        x[i] = input[i] + 0.0*I;
    }

    // Perform FFT
    fft(x, n);

    // Create the Hilbert transform filter in frequency domain
    for (int i = 1; i < n/2; i++) {
        x[i] *= 2.0; // Multiply positive frequencies by 2
    }
    for (int i = n/2 + 1; i < n; i++) {
        x[i] = 0.0 + 0.0*I; // Set negative frequencies to 0
    }

    // Perform IFFT
    ifft(x, n);

    // Copy result to output
    for (int i = 0; i < n; i++) {
        output[i] = x[i];
    }
}