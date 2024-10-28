#include<iostream>

using namespace std;

int max(int a, int b);
void bubble_sort(){
    int array[9] = {4,2,8,0,5,7,1,3,9};
    for (int i=0; i<9; i++){
        for (int j=0; j < 9-i; j++){
            if (max(array[j], array[j+1])){
                int temp = array[j];
                array[j] = array[j+1];
                array[j+1] = temp;
            }
        }
    }
    for (int i=0; i<9; i++){
        cout << array[i];
    }
}

int max(int a, int b){
    
    return a>b? true:false;
}