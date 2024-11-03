#include<iostream>
using namespace std;

struct hero
{
    string hName;
    int age;
    // 1表示男，0表示女
    bool gender;
};



void bubbleSort(hero arr[], int len){
    for (int i=0; i<len-1; i++){
        for (int j=0; j< len-1-i; j++){
            if (arr[j].age > arr[j+1].age){
                hero temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

void printHero(){
    struct hero arr[] = {
        {"liubei", 23, 1},
        {"guanyu", 22, 1},
        {"zhangfei",20, 1},
        {"zhaoyun", 21, 1},
        {"diaochan", 19, 0}
    }; 
    int len = sizeof(arr) / sizeof(hero);
    printf("%d\n",len);
    bubbleSort(arr, len);
    for (int i=0; i<len; i++){
        cout <<"the name is\t" << arr[i].hName << "\tage is\t" << arr[i].age << "\tgender is\t" << (arr[i].gender? "Male":"Female") << endl;
    }
}