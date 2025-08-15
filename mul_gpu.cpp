//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp>
#include <iomanip>
#include <vector>
#include <chrono> // For measuring execution time
#include "dpc_common.hpp"
#include "Complex.hpp"

using namespace sycl;
using namespace std;
using namespace std::chrono;

// Number of complex numbers passing to the SYCL code
static const int num_elements = 1000000; // Increase if necessary

class CustomDeviceSelector {
 public:
  CustomDeviceSelector(std::string vendorName) : vendorName_(vendorName) {}
  int operator()(const device &dev) const {
    int device_rating = 0;
    if (dev.is_gpu() & (dev.get_info<info::device::name>().find(vendorName_) != std::string::npos))
      device_rating = 3;
    else if (dev.is_gpu())
      device_rating = 2;
    else if (dev.is_cpu())
      device_rating = 1;
    return device_rating;
  };

 private:
  std::string vendorName_;
};

void SYCLParallel(queue &q, std::vector<Complex2> &in_vect1, std::vector<Complex2> &in_vect2, std::vector<Complex2> &out_vect) {
  auto R = range(in_vect1.size());
  if (in_vect2.size() != in_vect1.size() || out_vect.size() != in_vect1.size()) { 
	  std::cout << "ERROR: Vector sizes do not match\n";
	  return;
  }
  
  buffer bufin_vect1(in_vect1);
  buffer bufin_vect2(in_vect2);
  buffer bufout_vect(out_vect);

  std::cout << "Target Device: " << q.get_device().get_info<info::device::name>() << "\n";

  q.submit([&](auto &h) {
    accessor V1(bufin_vect1, h, read_only);
    accessor V2(bufin_vect2, h, read_only);
    accessor V3(bufout_vect, h, write_only);
    h.parallel_for(R, [=](auto i) {
      V3[i] = V1[i].complex_mul(V2[i]);
    });
  });
  q.wait_and_throw();
}

void Scalar(std::vector<Complex2> &in_vect1, std::vector<Complex2> &in_vect2, std::vector<Complex2> &out_vect) {
  if ((in_vect2.size() != in_vect1.size()) || (out_vect.size() != in_vect1.size())) {
	  std::cout << "ERROR: Vector sizes do not match\n";
	  return;
  }		 
  for (int i = 0; i < in_vect1.size(); i++) {
    out_vect[i] = in_vect1[i].complex_mul(in_vect2[i]);
  }
}

int Compare(std::vector<Complex2> &v1, std::vector<Complex2> &v2) {
  int ret_code = 1;
  if (v1.size() != v2.size()) {
	  ret_code = -1;
  }
  for (int i = 0; i < v1.size(); i++) {
    if (v1[i] != v2[i]) {
      ret_code = -1;
      break;
    }
  }
  return ret_code;
}

int main() {
  vector<Complex2> input_vect1;
  vector<Complex2> input_vect2;
  vector<Complex2> out_vect_parallel;
  vector<Complex2> out_vect_scalar;

  for (int i = 0; i < num_elements; i++) {
    input_vect1.push_back(Complex2(i + 2, i + 4));
    input_vect2.push_back(Complex2(i + 4, i + 6));
    out_vect_parallel.push_back(Complex2(0, 0));
    out_vect_scalar.push_back(Complex2(0, 0));
  }

  try {
    std::string vendor_name = "Intel";
    CustomDeviceSelector selector(vendor_name);
    queue q(selector);

    // Measure parallel execution time
    auto start_parallel = high_resolution_clock::now();
    SYCLParallel(q, input_vect1, input_vect2, out_vect_parallel);
    auto end_parallel = high_resolution_clock::now();
    auto duration_parallel = duration_cast<microseconds>(end_parallel - start_parallel);
    std::cout << "Parallel execution time on device: " << duration_parallel.count() << " µs\n";

    // Measure scalar execution time with 100 repetitions
    int loop_count = 100; // Number of repetitions for the scalar function
    auto start_scalar = high_resolution_clock::now();
    for (int i = 0; i < loop_count; i++) {
      Scalar(input_vect1, input_vect2, out_vect_scalar);
    }
    auto end_scalar = high_resolution_clock::now();
    auto duration_scalar = duration_cast<microseconds>(end_scalar - start_scalar) / loop_count;
    std::cout << "Average Scalar execution time on CPU (100 runs): " << duration_scalar.count() << " µs\n";

  } catch (...) {
    std::cout << "Failure" << std::endl;
    std::terminate();
  }

  int indices[]{0, 1, 2, 3, 4, (num_elements - 1)};
  constexpr size_t indices_size = sizeof(indices) / sizeof(int);

  for (int i = 0; i < indices_size; i++) {
    int j = indices[i];
    if (i == indices_size - 1) std::cout << "...\n";
    std::cout << "[" << j << "] " << input_vect1[j] << " * " << input_vect2[j] << " = " << out_vect_parallel[j] << "\n";
  }

  int ret_code = Compare(out_vect_parallel, out_vect_scalar);
  if (ret_code == 1) {
    std::cout << "Complex multiplication successfully run on the device\n";
  } else
    std::cout << "Verification Failed. Results are not matched\n";

  return 0;
}
