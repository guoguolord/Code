#include<iostream>

using namespace std;

int multiplicatonTable(){
    int i,j;
    for(i=1;i<10;i++) {
        for (j=1;j<(i+1);j++)
        {
            cout << j << "*" << i << "=" << i*j << "\t";
        }
        cout << endl;   
    }
    return 0;
}