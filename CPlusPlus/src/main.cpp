#include<iostream>
//#include "../include/peopleSort.h"

int main(){
    int a = 10;
    int b = 20;
    int &c = a;
    c = b;
    std::cout << "a=" << a << std::endl;
    std::cout << "b=" << b << std::endl;
    std::cout << "c=" << c << std::endl;
    a = 90;
    std::cout << "a=" << a << std::endl;
    std::cout << "b=" << b << std::endl;
    std::cout << "c=" << c << std::endl;
    return 0;
}

