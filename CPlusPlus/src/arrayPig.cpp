#include<iostream>
#include "../include/arrayPig.h"
using namespace std;

void pig(){
    int array[5] = {300, 350, 200, 400, 250};
    int min_array = array[0];
    for (int i=0; i<5;i++){
        if (min_array > array[i]){
            min_array = array[i]; 
        }
    }
    cout << "min num is :" << min_array <<endl;
}