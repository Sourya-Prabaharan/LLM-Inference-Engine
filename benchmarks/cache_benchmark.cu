#include "kv_cache.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <vector>
using namespace llm;
struct Allocation {
    size_t requested, free_before, free_after, total;
};
Allocation measure(size_t bytes) {
    Allocation result{bytes, 0, 0, 0};
    CUDA_CHECK(cudaMemGetInfo(&result.free_before, &result.total));
    {
        Tensor pool({bytes}, DType::I8, Device::CUDA);
        CUDA_CHECK(cudaMemset(pool.data(), 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemGetInfo(&result.free_after, &result.total));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}
int main() {
    try {
        const std::vector<size_t> lengths = {17, 31, 65, 127, 255, 511, 1023};
        constexpr size_t context = 2048, block_size = 16, bytes_per_token = 22 * 4 * 64 * 2 * 4;
        size_t assigned_blocks = 0,
               used = std::accumulate(lengths.begin(), lengths.end(), size_t(0));
        for (auto n : lengths)
            assigned_blocks += (n + block_size - 1) / block_size;
        PagedKVCache cache(assigned_blocks, block_size);
        for (size_t r = 0; r < lengths.size(); ++r) {
            cache.allocate_request(int(r));
            cache.append_tokens(int(r), lengths[r]);
        }
        size_t assigned = 0;
        for (size_t r = 0; r < lengths.size(); ++r)
            assigned += cache.reserved_tokens(int(r));
        CUDA_CHECK(cudaFree(nullptr)); // Initialize the context before measuring either allocation.
        auto contiguous = measure(lengths.size() * context * bytes_per_token);
        auto paged = measure(assigned * bytes_per_token);
        cudaDeviceProp prop{};
        int device, version;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        CUDA_CHECK(cudaRuntimeGetVersion(&version));
        size_t live = used * bytes_per_token, cw = contiguous.requested - live,
               pw = paged.requested - live;
        std::cout << "{\"backend\":\"CUDA\",\"gpu\":\"" << prop.name
                  << "\",\"cuda_runtime\":" << version
                  << ",\"dtype\":\"fp32\",\"context\":" << context
                  << ",\"block_size\":" << block_size
                  << ",\"lengths\":[17,31,65,127,255,511,1023],\"live_bytes\":" << live
                  << ",\"contiguous_pool_bytes\":" << contiguous.requested
                  << ",\"paged_pool_bytes\":" << paged.requested
                  << ",\"contiguous_driver_free_delta\":"
                  << double(contiguous.free_before) - double(contiguous.free_after)
                  << ",\"paged_driver_free_delta\":"
                  << double(paged.free_before) - double(paged.free_after)
                  << ",\"contiguous_waste_bytes\":" << cw << ",\"paged_waste_bytes\":" << pw
                  << ",\"waste_reduction_fraction\":" << 1 - double(pw) / cw
                  << ",\"pool_sizing\":\"minimum blocks for the identical live workload; no extra "
                     "free capacity\"}\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
