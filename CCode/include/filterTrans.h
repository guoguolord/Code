//
// Created by 果果lord on 2024/10/17.
//

#ifndef CCODE_FILTERTRANS_H
#define CCODE_FILTERTRANS_H
// 定义滤波器结构体
typedef struct {
    double a[3]; // 滤波器系数a
    double b[3]; // 滤波器系数b
    double z[2]; // 滤波器的延迟元素
}Biquad;
float process_biqaud(Biquad *bq, float x);
void init_biquad(Biquad *bq, unsigned short Fc1, unsigned short Fc2, unsigned int Fs);
#endif //CCODE_FILTERTRANS_H
