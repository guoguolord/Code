#include <stdio.h>
#include <stdlib.h>
#include "../ReadWave.h"
#include "../include/hilbertTrans.h"
#include "../include/fftTrans.h"
#include "../include/filterTrans.h"
#include <complex.h>
#include <math.h>


// wave头结构，共44个字节
typedef struct {
    char riff_header[4];       // 存储'riff'字符串，表示文件类型,4个字节
    unsigned int wav_size;              // 存储整个wave文件的大小减去8字节, 4个字节
    char wave_header[4];       // 存储"WAVE"字符串，表示文件格式, 4个字节
    char fmt_header[4];        // 存储"fmt "字符串，表示格式块的开始, 4个字节
    unsigned int fmt_chunk_size;        // 存储格式块的大小,4个字节
    unsigned short audio_format;        // 存储音频格式，1表示PCM格式, 2个字节
    unsigned short num_channels;        // 存储声道数， 2个字节
    unsigned int sample_rate;           // 采样率, 4个字节
    unsigned int byte_rate;             // 存储每秒数据传输速度(SampleRate * NumChannels * BitsPerSample) / 8, 4个字节
    unsigned short sample_alignment;    // 存储每个样本的对齐字节数(NumChannels * BitsPerSample / 8), 2个字节
    unsigned short bit_depth;           // 位深度, 2个字节
    char data_header[4];       // 存储"data"字符串，表示数据块的开始, 4个字节
    unsigned int data_bytes;            // 存储数据块的字节数, 4个字节
} WAVHeader;

void print_fft_magnitude(complex float* x, int n);

int main() {
    FILE *file;
    WAVHeader header;
    int n = 4096;
    int duration = 1;
    complex float hilbert_output[n];
    // 采样频率Fs，频率范围Fc1到Fc2
    unsigned short Fc1 = 15000; // 15kHz
    unsigned short Fc2 = 40000; // 40kHz
    //读取wave文件
    const char *filename = "F://Gitlab//CPPLearn//CCode//911_20240112_140614_smallface.wav";
    file = fopen(filename, "rb");
    if (file == NULL) {
        printf("文件无法打开\n");
        return 1;
    }
    // 读取WAV头信息
    fread(&header, sizeof(WAVHeader), 1, file);
    unsigned int sr = header.sample_rate ; // 采样率

    // 打印文件头信息，确认音频格式
    printf("sample rate: %d Hz\n", sr);

    // 数据的采样点数，采样率*所需时间
    int num_per_samples = sr * duration;
    int total_samples = header.byte_rate / (header.bit_depth / 8);
    // 分配内存
    float *buffer = (float *) malloc(num_per_samples * sizeof(float));
    if (buffer == NULL) {
        printf("Memory allocation error!\n");
        fclose(file);
        return 1;
    }
    // 跳过头部并读取音频数据
    fseek(file, 44, SEEK_SET);
    fread(buffer, sizeof(float), total_samples, file);

    //    // 打印前10个浮点型样本
    //    for (int i = 0; i < num_per_samples; i++) {
    //        printf("sample No.%d: %f\n", i, buffer[i]);
    //    }
    // 带通滤波
    Biquad bandpass;
    init_biquad(&bandpass, Fc1, Fc2, sr);
    float output_signal[num_per_samples];
    for (int i = 0; i < num_per_samples; i++) {
        output_signal[i] = process_biqaud(&bandpass, buffer[i]);
        // printf("Input: %f, Output: %f\n", buffer[i], output_signal[i]);
    }
    // 希尔伯特变换
    hilbert(output_signal, hilbert_output, n);
    // 傅里叶变换
    fft(hilbert_output, n);
    // 输出FFT结果
    printf("FFT result:\n");
    for (int i = 0; i < n / 2; i++) {
        // 计算幅度
        float magnitude = cabsf(hilbert_output[i]);
        // float magnitude = sqrtf(crealf(hilbert_output[i]) * crealf(hilbert_output[i]) + cimagf(hilbert_output[i]) * cimagf(hilbert_output[i]));
        // 对应的频率
        float frequency = i * sr / n;
        printf("freq: %f Hz, magnitude: %f\n", frequency, magnitude);
    }
    //释放内存
    free(buffer);

    // 关闭文件
    fclose(file);
    return 0;
}

void print_fft_magnitude(complex float* x, int n)
{
    WAVHeader header;
    float freq;
    for (int i = 0; i < n; i++) {
        // 打印幅值信息
        printf("-------------------------------------------\n");
        printf("Frequency bin %d: Magnitude = %.2f\n", i, cabs(x[i]));
        printf("-------------------------------------------\n");
        // 打印频率信息
        freq = i * header.sample_rate / n;
        printf("Frequency bin %d: freq = %.2f\n", i, freq);
    }
}