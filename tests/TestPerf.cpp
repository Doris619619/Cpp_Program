// Minimal perf test executable entry point.
#include <iostream>
#include <chrono>

int main() {
	auto start = std::chrono::high_resolution_clock::now();
	// TODO: add performance measurement invoking Vision components.
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "test_perf stub elapsed ms: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
			  << "\n";
	return 0;
}
