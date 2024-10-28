#include<iostream>
#include "../Header/reverseOrder.h"
using namespace std;

void reverse(){
    int array[5] = {1,3,2,5,4};
    int new_array[5];
    for (int i=0; i<5; i++){
        new_array[i] = array[4-i];
    }
    for (int i=0; i<5; i++){
        cout << new_array[i] << endl;
    }
}