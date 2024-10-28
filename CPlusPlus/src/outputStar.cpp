#include<iostream>

using namespace std;

int outputstar(){
    int i, j;
    for (i=0;i<10;i++){
        for (j=0;j<10;j++){
            cout << "*";
        }
        cout << endl;
    }
    for (i=0;i<10;i++){
        for (j=0;j<i;j++){
            cout << "*";
        }
        cout << endl;
    }
    return 0;
}