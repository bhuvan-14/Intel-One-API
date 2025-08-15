**1)**Overview****

This project compares the execution performance of complex number multiplication across different computing devices using Intel oneAPI SYCL.
The devices tested are:

**CPU – Central Processing Unit

GPU – Graphics Processing Unit

FPGA – Field-Programmable Gate Array (optional if available)**

The main objective is to measure how the parallel processing capabilities of each device affect execution speed compared to traditional scalar execution.

**2)**Description****
The program:

Generates large arrays of complex numbers.

**3)**Performs multiplication using:**
Parallel execution on the selected device (CPU/GPU/FPGA)

Scalar execution on the CPU for baseline comparison

Measures execution time for both methods.

Compares results to ensure correctness.

Two versions are implemented:

CPU version – Uses SYCL device selector to run on CPU.

GPU version – Uses SYCL device selector to target Intel GPU.

**4)Purpose**

Demonstrate how to write parallel code using SYCL.

Understand device selection in Intel oneAPI.

Analyze performance differences between CPU, GPU, and FPGA.

Explore the benefits of heterogeneous computing.

**5)Expected Output**

Name of the device used for execution.

Parallel execution time.

Average scalar execution time.

Sample multiplication results.

Verification message indicating whether results match.

**References**

**Intel oneAPI Base Toolkit – https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html

SYCL Specification – https://www.khronos.org/sycl/**
