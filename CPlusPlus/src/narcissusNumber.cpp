#include<iostream>
#include<math.h>

using namespace std;

int narcissusNumber(){
    int num = 100;
    do
    {
        // 取出3位数的个位、十位和百位
        int i = num / 100;
        int j = num / 10 % 10;
        int k = num % 10;
        // 计算每个位数的立方和原数相比
        int third_num = pow(i,3) + pow(j, 3) + pow(k, 3);
        if (num == third_num){
            cout << num <<endl;
        }
        num++;
    } while (num < 1000);   
}