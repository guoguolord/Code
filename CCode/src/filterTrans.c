//
// Created by 果果lord on 2024/10/17.
//
#include <math.h>

// 定义滤波器结构体
typedef struct {
    float a[3]; // 滤波器系数a
    float b[3]; // 滤波器系数b
    float z[2]; // 滤波器的延迟元素
}Biquad;

// 双二阶节滤波器处理函数
float process_biqaud(Biquad *bq, float x)
{
    float output = bq->b[0] * x + bq->z[0];
    bq->z[0] = bq->b[1] * x - bq->a[1] * output + bq->z[1];
    bq->z[1] = bq->b[2] * x - bq->a[2] * output;
    return output;
}

// 初始化双二阶节滤波器
void init_biquad(Biquad *bq, unsigned short Fc1, unsigned short Fc2, unsigned int Fs)
{
    double w0 = 2.0 * M_PI * sqrt(Fc1 * Fc2) / Fs;
    double bw = 2.0 * M_PI * (Fc2 - Fc1) / Fs;
    double alpha = sin(w0) * sinh(log(2.0) / 2.0 * bw * w0 / sin(w0));
    double cos_w0 = cos(w0);

    bq->b[0] = alpha;
    bq->b[1] = 0;
    bq->b[2] = -alpha;

    bq->a[0] = 1 + alpha;
    bq->a[1] = -2 * cos_w0;
    bq->a[2] = 1 - alpha;

    // 归一化滤波器系数
    bq->b[0] /= bq->a[0];
    bq->b[1] /= bq->a[0];
    bq->b[2] /= bq->a[0];

    bq->a[1] /= bq->a[0];
    bq->a[2] /= bq->a[0];

    // 初始化延迟
    bq->z[0] = 0;
    bq->z[1] = 0;
}
