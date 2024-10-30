#include<iostream>
#include "../Header/twoDimensionArray.h"
using namespace std;

void twoArray(){
    int sorces[3][3] = 
    {{100, 100, 100},
    {90, 50, 100},
    {60, 70, 80}};
    string names[3] = {"zhansan", "lisi", "wangwu"};
    
    for (int i=0; i<3; i++){
        int sum = 0;
        for (int j = 0; j<3; j++)
        {
            int sorce = sorces[i][j];
            sum += sorce;
        }
        cout << "The names is:"<< names[i] << ",all sorces is:" << sum << endl;
    }
}