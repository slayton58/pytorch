#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_fp16.h>
#include <c10/macros/Macros.h>

#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>

namespace {

int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

// The softmax_warp_* methods perform softmax forward and backward propagation on samples spanning the fast dimension.
// Each sample contains element_count scalar elements. element_count can be any integer value <= 1024.
// The template arguments have the following meaning:
// One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples.
// WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small.
// A "WARP" contains "C10_WARPS_SIZE" threads, these treads are guaranteed to belong to the same warp.
// This is important because it means only __shfl_ instructions are required for reductions.
// Note that this means WARP_SIZE must be a power of two and <= architecture warp size.
// CUDA warp size is 32 for all existing GPU architectures, but there is no guarantee this will not change for future arch.
// ROCm warp size is 64 for all currently ROCm-supported GPU architectures, but this may change for future archs.
// is_log_softmax is a flag indicating whether SoftMax or LogSoftMax should be computed.
// The template can be instantiated with any floating point type for the type arguments input_t, output_t and acc_t.
// This allows SoftMax to be fused with a cast immediately following the SoftMax.
// For instance:
// input_t=half,  acc_t=float, output_t=half  => read half tensor, float accumulators, write half tensor.
// input_t=half,  acc_t=float, output_t=float => read half tensor, float accumulators, write float tensor.
// input_t_float, acc_t=float, output_t=half  => read float tensor, float accumulators, write half tensor.

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax, int INPUT_VEC, int OUTPUT_VEC>
__global__ void softmax_warp_forward(output_t *dst, input_t *src, int batch_size, int stride, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = std::max(next_power_of_two / WARP_SIZE / INPUT_VEC, 1);
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    using loadT = at::native::memory::aligned_vector<input_t, INPUT_VEC>;
    using storeT = at::native::memory::aligned_vector<output_t, OUTPUT_VEC>;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    // alias pointer into the src & dst buffers
    // Move forward to the first batch this warp will handle,
    // and take into account threadIdx and vector size.
    // first_batch * stride indexes into row
    // local_idx * INPUT_VEC indexes into position within that row.
    src += first_batch * stride + local_idx * INPUT_VEC;
    dst += first_batch * stride + local_idx * INPUT_VEC;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    // Each iteration is going to read INPUT_VEC elements at a time
    // with WARP_ITERATIONS adjusted accordingly
    // internal elements array is acc_t so we need a separate input_t buffer to read into
    input_t input_vec[INPUT_VEC];
    acc_t elements[WARP_BATCH][WARP_ITERATIONS][INPUT_VEC];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        // how many elements in this batch need to be handled?
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        // if (threadIdx.x == 0) printf("i(%d, %d): %d - batch_element_count: %d\n", blockIdx.x, blockIdx.y, i, batch_element_count);

        // For each iteration we'll grab INPUT_VEC values
        for (int it = 0;  it < WARP_ITERATIONS;  it++) {
            // it * WARP_SIZE is multiples of WARP_SIZE, local_idx \in [0, 32)
            // So we need to multiply this by INPUT_VEC
            // local_idx is threadIdx, it*WARP_SIZE is how many strides we've taken, INPUT_VEC is size of those strides
            int element_index = (local_idx + it * WARP_SIZE) * INPUT_VEC;
            // vectorized path - can this thread read a full vector
            if (element_index + INPUT_VEC <= batch_element_count) {
                // perform vectorized read
                // alias into the register elements buffer
                loadT *input = reinterpret_cast<loadT*>(&input_vec[0]);
                // loadT *input = reinterpret_cast<loadT*>(&elements[i][it][0]);
                // Read from the global buffer
                // Note: src already takes into account the threadIdx and first batch offset
                // i * element_count is offset into batch number
                // it * WARP_SIZE * INPUT_VEC is the offset within that batch
                *input = *reinterpret_cast<const loadT*>(&src[i*element_count+it*WARP_SIZE*INPUT_VEC]);
                // now cast (register->register) to the from input buffer to elements
                #pragma unroll
                for (int v = 0; v < INPUT_VEC; ++v) {
                  elements[i][it][v] = input_vec[v];
                }
            } else {
                // can't do a vectorized load, load what we can
                int vec_idx = 0;
                #pragma unroll
                for (int j = element_index ; j < element_index + INPUT_VEC; ++j) {
                    if (j < batch_element_count) {
                        elements[i][it][vec_idx] = src[i*element_count+it*WARP_SIZE*INPUT_VEC + j];
                    } else {
                        elements[i][it][vec_idx] = -std::numeric_limits<acc_t>::infinity();
                    }
                    vec_idx++;
                }
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        max_value[i] = elements[i][0][0];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            #pragma unroll
            for (int v = 0; v < INPUT_VEC; ++v) {
                max_value[i] = (max_value[i] > elements[i][it][v]) ? max_value[i] : elements[i][it][v];
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = 0.f;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            #pragma unroll
            for (int v = 0; v < INPUT_VEC; ++v) {
                if (is_log_softmax) {
                  sum[i] += std::exp(elements[i][it][v] - max_value[i]);
                } else {
                  elements[i][it][v] = std::exp(elements[i][it][v] - max_value[i]);
                  sum[i] += elements[i][it][v];
                }
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    output_t output_vec[INPUT_VEC];
    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        if (is_log_softmax) sum[i] = max_value[i] + std::log(sum[i]);
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS; it++) {
            int element_index = (local_idx + it * WARP_SIZE) * INPUT_VEC;

            if (element_index + INPUT_VEC < element_count) {
              // need to perform the final calculation in registers first as we'll then
              // write out vectorized
              // Note: need a copy in the output_t typed buffer for writing
              #pragma unroll
              for (int v = 0; v < INPUT_VEC; ++v) {
                  if (is_log_softmax) {
                    elements[i][it][v] -= sum[i];
                  } else {
                    elements[i][it][v] /= sum[i];
                  }
                  // need a copy in our actual output buffer
                  output_vec[v] = elements[i][it][v];
              }
              // perform vectorized write
              storeT *output = reinterpret_cast<storeT*>(&dst[i*element_count+it*WARP_SIZE*INPUT_VEC]);
              // *output = *reinterpret_cast<storeT*>(&elements[i][it][0]);
              *output = *reinterpret_cast<storeT*>(&output_vec[0]);
            } else {
                  #pragma unroll
                  for (int vec_idx = 0; vec_idx < INPUT_VEC; ++vec_idx) {
                      if (element_index + vec_idx < element_count) {
                          if (is_log_softmax) {
                              dst[i*element_count+it*WARP_SIZE*INPUT_VEC + vec_idx] = elements[i][it][vec_idx] - sum[i];
                          } else {
                              dst[i*element_count+it*WARP_SIZE*INPUT_VEC + vec_idx] = elements[i][it][vec_idx] / sum[i];
                          }
                      }
                  }
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t, int log2_elements, bool is_log_softmax>
__global__ void softmax_warp_backward(output_t *gradInput, const input_t *grad, const input_t *output, int batch_size, int stride, int element_count)
{
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;

    int first_batch = (blockDim.y * blockIdx.x + threadIdx.y) * WARP_BATCH;

    // batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x % WARP_SIZE;

    // the first element to process by the current thread
    int thread_offset = first_batch * stride + local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // The nested loops over WARP_BATCH and then WARP_ITERATIONS can be simplified to one loop,
    // but I think doing so would obfuscate the logic of the algorithm, thus I chose to keep
    // the nested loops.
    // This should have no impact on performance because the loops are unrolled anyway.

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS];
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS];
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : element_count;
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                grad_reg[i][it] = grad[i*element_count+it*WARP_SIZE];
                output_reg[i][it] = output[i*element_count+it*WARP_SIZE];
            } else {
                grad_reg[i][it] = acc_t(0);
                output_reg[i][it] = acc_t(0);
            }
        }
    }

    acc_t sum[WARP_BATCH];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = grad_reg[i][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            int element_index = local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                if (is_log_softmax) {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - std::exp(output_reg[i][it]) * sum[i]);
                } else {
                    gradInput[i*element_count+it*WARP_SIZE] = (grad_reg[i][it] - output_reg[i][it] * sum[i]);
                }
            }
        }
    }
}

} // end of anonymous namespace

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax, int INPUT_VEC, int OUTPUT_VEC>
void dispatch_softmax_forward_elements(output_t *dst, input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 1024 );
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_forward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // dim3 threads(warp_size, 1, 1);
        // printf("threads: %d, %d blocks: %d\n", threads.x, threads.y, blocks);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0: // 1
                softmax_warp_forward<input_t, output_t, acc_t, 0, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 1: // 2
                softmax_warp_forward<input_t, output_t, acc_t, 1, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 2: // 4
                softmax_warp_forward<input_t, output_t, acc_t, 2, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 3: // 8
                softmax_warp_forward<input_t, output_t, acc_t, 3, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 4: // 16
                softmax_warp_forward<input_t, output_t, acc_t, 4, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 5: // 32
                softmax_warp_forward<input_t, output_t, acc_t, 5, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 6: // 64
                softmax_warp_forward<input_t, output_t, acc_t, 6, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 7: // 128
                softmax_warp_forward<input_t, output_t, acc_t, 7, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 8: // 256
                softmax_warp_forward<input_t, output_t, acc_t, 8, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 9: // 512
                softmax_warp_forward<input_t, output_t, acc_t, 9, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 10: // 1024
                softmax_warp_forward<input_t, output_t, acc_t, 10, is_log_softmax, INPUT_VEC, OUTPUT_VEC>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst, src, batch_count, softmax_elements_stride, softmax_elements);
                break;
            default:
                break;
        }
    }
}

template <typename input_t, typename output_t>
int get_vector_size(const input_t *input_ptr, const output_t *output_ptr, int stride) {
  int proposed_vec = std::min(
      at::native::memory::can_vectorize_up_to<input_t>((char*)input_ptr),
      at::native::memory::can_vectorize_up_to<output_t>((char*)output_ptr)
  );

  // This is tricky - want to make sure that each row of the input is aligned to proposed vector length
  if (stride % proposed_vec != 0) proposed_vec = 1;

  return proposed_vec;
}

template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_forward(output_t *dst, input_t *src, int softmax_elements, int softmax_elements_stride, int batch_count) {
  int vec_size = get_vector_size<input_t, output_t>(src, dst, softmax_elements_stride);

  switch (vec_size) {
   case 1:
    dispatch_softmax_forward_elements<input_t, output_t, acc_t, is_log_softmax, 1, 1>(dst, src, softmax_elements, softmax_elements_stride, batch_count);
    break;
   case 2:
    dispatch_softmax_forward_elements<input_t, output_t, acc_t, is_log_softmax, 2, 2>(dst, src, softmax_elements, softmax_elements_stride, batch_count);
    break;
   case 4:
    dispatch_softmax_forward_elements<input_t, output_t, acc_t, is_log_softmax, 4, 4>(dst, src, softmax_elements, softmax_elements_stride, batch_count);
    break;
  }

}


template<typename input_t, typename output_t, typename acc_t, bool is_log_softmax>
void dispatch_softmax_backward(output_t *grad_input, const input_t *grad, const input_t *output, int softmax_elements, int softmax_elements_stride, int batch_count)
{
    TORCH_INTERNAL_ASSERT( softmax_elements >= 0 && softmax_elements <= 1024 );
    if (softmax_elements == 0) {
       return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;

        // This value must match the WARP_SIZE constexpr value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        int blocks = (batch_count + batches_per_block - 1) / batches_per_block;
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0: // 1
                softmax_warp_backward<input_t, output_t, acc_t, 0, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 1: // 2
                softmax_warp_backward<input_t, output_t, acc_t, 1, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 2: // 4
                softmax_warp_backward<input_t, output_t, acc_t, 2, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 3: // 8
                softmax_warp_backward<input_t, output_t, acc_t, 3, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 4: // 16
                softmax_warp_backward<input_t, output_t, acc_t, 4, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 5: // 32
                softmax_warp_backward<input_t, output_t, acc_t, 5, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 6: // 64
                softmax_warp_backward<input_t, output_t, acc_t, 6, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 7: // 128
                softmax_warp_backward<input_t, output_t, acc_t, 7, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 8: // 256
                softmax_warp_backward<input_t, output_t, acc_t, 8, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 9: // 512
                softmax_warp_backward<input_t, output_t, acc_t, 9, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            case 10: // 1024
                softmax_warp_backward<input_t, output_t, acc_t, 10, is_log_softmax>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, batch_count, softmax_elements_stride, softmax_elements);
                break;
            default:
                break;
        }
    }
}

