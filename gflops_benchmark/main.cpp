#include <iostream> 

extern "C" void func1(int);

int main() {
// 1.7e8 * 8 instructions = 13.8 gflops
    func1(1.7e8);
//  func2(1.7e8);
}
