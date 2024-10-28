#include<iostream>
using namespace std;

int knockDesk(){
    int i = 1;
    for (i; i < 101; i++){
        // 逢7的倍数，个位是7的，十位是7的，敲桌子；否则，输出数字本身
        if ((i % 7 == 0) or (i / 10 == 7) or (i % 10 == 7)){
            cout << "knock desk" << endl;
        }
        else{
            cout << i << endl;
        }
    }
    return 0;
}