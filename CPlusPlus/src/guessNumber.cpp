#include <iostream>
#include <random>

int guess() {
    std::mt19937 gen; // 标准梅森旋转算法生成器
    std::uniform_int_distribution<int> dis(1, 100); // 定义1到100之间的均匀分布

    int secretNumber = dis(gen); // 生成一个1到100之间的随机数
    int guess;

    std::cout << dis(gen) << std::endl;

    // 使用 while 循环来读取用户的猜测
    while (true) { // 无限循环，直到用户猜对数字
        std::cout << "Please input your guess number:";
        // std::cin >> guess;

        // 检查输入是否有效
        while (!(std::cin >> guess)) {
            std::cin.clear(); // 清除错误标志
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // 忽略错误输入直到下一个换行符
            std::cout << "Invalid input:";
            std::cin >> guess;
        }

        if (guess == secretNumber) {
            std::cout << "Congratulations on guessing correctly:" << secretNumber << "!" << std::endl;
            break; // 如果猜对了，打印恭喜信息并退出循环
        } else if (guess < secretNumber) {
            std::cout << "too small, try again" << std::endl;
        } else {
            std::cout << "too big, try again" << std::endl;
        }
    }

    return 0;
}