#include <iostream>
#include <ctime>
using namespace std;
struct stuent
{
    string name;
    int score;
};

struct teacher
{
    string name;
    stuent sArray[5];
};
void allocateSpace(teacher tArray[], int len){
    string tName = "teacher";
    string sName = "student";
    string nameSeed = "ABCDE";
    for (int i=0; i<len; i++){
        tArray[i].name = tName + nameSeed[i];
        for (int j=0; j<5; j++){
            tArray[i].sArray[j].name = sName + nameSeed[j];
            tArray[i].sArray[j].score = rand() % 61 + 50;
        }
    }
}

void printTeachers(teacher tArray[], int len){
    for (int i = 0; i < len; i++)
	{
		cout << tArray[i].name << endl;
		for (int j = 0; j < 5; j++)
		{
			cout << "\tthe name is:" << tArray[i].sArray[j].name << " ,score is:" << tArray[i].sArray[j].score << endl;
		}
	}
}

void allProcess(){
    // 随机数种子函数srand
    srand((unsigned int)time(NULL));
    teacher tArray[3];
    int len = sizeof(tArray) / sizeof(teacher);
    allocateSpace(tArray, len);
    printTeachers(tArray, len);

    system("pause");
    
}