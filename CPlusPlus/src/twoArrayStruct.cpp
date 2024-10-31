#include<iostream>
#include "../include/twoArrayStruct.h"
using namespace std;

struct sorces
{
    string name;
    int ChineseSorce;
    int MathSorce;
    int EnglishSorce;
}sorc;

void arrayStruct(){
    // sorces s1 = {"zhangsan", 100, 100, 100};
    // struct sorces s2;
    // s2.name = "lisi";
    // s2.ChineseSorce = 90;
    // s2.MathSorce = 50;
    // s2.EnglishSorce = 100;
    // sorces s3 = {"wangwu", 60, 70, 80};
    sorces stduent[] = {
        {"zhangsan", 100, 100, 100},
        {"lisi", 90, 50, 100},
        {"wangwu", 60, 70, 80}
    };
    int stuent_num = sizeof(stduent) / sizeof(stduent[0]);
    for (int i=0; i<stuent_num; i++){
        int total_sorces = stduent[i].ChineseSorce + stduent[i].MathSorce + stduent[i].EnglishSorce; 
        cout << "The name is" << stduent[i].name << ",all sorces is:" << total_sorces <<endl;
    }
  
}