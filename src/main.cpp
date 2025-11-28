#include <seat_state_judger.hpp>
#include <data_structures.hpp>
#include <json.hpp>
#include <iostream>
#include <thread>
#include <iomanip>

using json = nlohmann::json;

int main(int argc, char* argv[]) {
    // 忽略未使用参数
    (void)argc;
    (void)argv;

    SeatStateJudger judger;

    // 命令行参数说明：
    // 用法1：./b_module_standalone → 监听 last_frame.json（原功能）
    // 用法2：./b_module_standalone [JSONL路径] → 测试批量数据
    if (argc == 2) {
        string jsonl_path = argv[1];
        judger.run(jsonl_path);  // 传入JSONL路径，测试批量数据
    } else {
        judger.run();  // 无参数，监听单个JSON
    }

    return 0;
}