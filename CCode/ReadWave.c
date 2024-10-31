//
// Created by 果果lord on 2024/10/16.
//

#include <stdlib.h>
#include <stdio.h>

// WAV file header structure
typedef struct {
    char riff_header[4];       // 存储'riff'字符串，表示文件类型
    int wav_size;              // 存储整个wave文件的大小减去8字节
    char wave_header[4];       // 存储"WAVE"字符串，表示文件格式
    char fmt_header[4];        // 存储"fmt "字符串，表示格式块的开始
    int fmt_chunk_size;        // 存储格式块的大小
    short audio_format;        // 存储音频格式，1表示PCM格式
    short num_channels;        // 存储声道数
    int sample_rate;           // 采样率
    int byte_rate;             // 存储每秒数据传输速度(SampleRate * NumChannels * BitsPerSample) / 8
    short sample_alignment;    // 存储每个样本的对齐字节数(NumChannels * BitsPerSample / 8)
    short bit_depth;           // 位深度
    char data_header[4];       // 存储"data"字符串，表示数据块的开始
    int data_bytes;            // 存储数据块的字节数
} WAVHeader;

int Wave(const char* filename) {
    FILE *wav_file = fopen(filename, "rb");
    char *buffer;
    size_t byte_read;
    WAVHeader header;

    if (wav_file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    // 读取wave文件头
    fread(&header, sizeof(WAVHeader), 1, wav_file);

    // Check if the file is a valid WAV file
    if (header.riff_header[0] != 'R' || header.riff_header[1] != 'I' ||
        header.riff_header[2] != 'F' || header.riff_header[3] != 'F' ||
        header.wave_header[0] != 'W' || header.wave_header[1] != 'A' ||
        header.wave_header[2] != 'V' || header.wave_header[3] != 'E') {
        printf("This is not a valid WAV file.\n");
        fclose(wav_file);
        return 1;
    }

    // 分配1秒wav文件字节的内存
    buffer = (char *) malloc((header.byte_rate));
    if (buffer == NULL) {
        printf("Memory allocation error!\n");
        fclose(wav_file);
        return 1;
    }

    byte_read = fread(buffer, 1, header.byte_rate, wav_file);
    free(buffer);
    fclose(wav_file);

    return 0;
}
//    unsigned char* data = (unsigned char*)malloc(header.data_bytes);
//    if (data == NULL) {
//        printf("Memory allocation error!\n");
//        fclose(wav_file);
//        return 1;
//    }

    // Read audio data
//    fread(data, header.data_bytes, 1, wav_file);
//    fclose(wav_file);

    // Do something with the audio data (e.g., process, play, analyze)
    // For now, we just print the first few bytes
//    printf("First few bytes of audio data:\n");
//    for (int i = 0; i < 20 && i < header.data_bytes; i++) {
//        printf("%02x ", data[i]);
//    }
//    printf("\n");
//
//    // Free the allocated memory
//    free(data);
//
//    return 0;


