#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "api.h"
#include "hash.cuh"


/**
 * Insert keys into the hashmap
 * 
 * @param N         number of elements in the hashmap
 * @param M         number of keys to be inserted
 * @param hashmap   [2N] uint32 tensor containing the hashmap (key-value pairs)
 * @param keys      [M] uint32 tensor containing the keys to be inserted
 * @param values    [M] uint32 tensor containing the values to be inserted
 */
__global__ void hashmap_insert_cuda_kernel(
    const uint32_t N,
    const int64_t M,
    uint32_t* __restrict__ hashmap,
    const uint32_t* __restrict__ keys,
    const uint32_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M)
    {
        uint32_t key = keys[thread_id];
        uint32_t value = values[thread_id];
        linear_probing_insert(hashmap, key, value, N);
    }
}


/**
 * Insert 64-bit keys into the hashmap
 */
__global__ void hashmap_insert_cuda_kernel_64(
    const uint64_t N,
    const int64_t M,
    uint64_t* __restrict__ hashmap,
    const uint64_t* __restrict__ keys,
    const uint64_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        uint64_t key = keys[thread_id];
        uint64_t value = values[thread_id];
        linear_probing_insert_64(hashmap, key, value, N);
    }
}


/**
 * Insert keys into the hashmap
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param keys      [M] uint32/uint64 tensor containing the keys to be inserted
 * @param values    [M] uint32/uint64 tensor containing the values to be inserted
 */
void hashmap_insert_cuda(
    torch::Tensor& hashmap,
    const torch::Tensor& keys,
    const torch::Tensor& values
) {
    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_insert_cuda_kernel<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint32_t>(),
            keys.data_ptr<uint32_t>(),
            values.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_insert_cuda_kernel_64<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint64_t>(),
            keys.data_ptr<uint64_t>(),
            values.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}


/**
 * Lookup keys in the hashmap
 * 
 * @param N         number of elements in the hashmap
 * @param M         number of keys to be looked up
 * @param hashmap   [2N] uint32 tensor containing the hashmap (key-value pairs)
 * @param keys      [M] uint32 tensor containing the keys to be looked up
 * @param values    [M] uint32 tensor containing the values to be looked up
 */
__global__ void hashmap_lookup_cuda_kernel(
    const uint32_t N,
    const int64_t M,
    const uint32_t* __restrict__ hashmap,
    const uint32_t* __restrict__ keys,
    uint32_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        uint32_t key = keys[thread_id];
        values[thread_id] = linear_probing_lookup(hashmap, key, N);
    }
}


/**
 * Lookup 64-bit keys in the hashmap
 */
__global__ void hashmap_lookup_cuda_kernel_64(
    const uint64_t N,
    const int64_t M,
    const uint64_t* __restrict__ hashmap,
    const uint64_t* __restrict__ keys,
    uint64_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        uint64_t key = keys[thread_id];
        values[thread_id] = linear_probing_lookup_64(hashmap, key, N);
    }
}


/**
 * Lookup keys in the hashmap
 * 
 * @param hashmap   [N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param keys      [M] uint32/uint64 tensor containing the keys to be looked up
 * @return          [M] uint32/uint64 tensor containing the values of the keys
 */
torch::Tensor hashmap_lookup_cuda(
    const torch::Tensor& hashmap,
    const torch::Tensor& keys
) {
    // Allocate output tensor
    auto output = torch::empty_like(keys);

    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_lookup_cuda_kernel<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint32_t>(),
            keys.data_ptr<uint32_t>(),
            output.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_lookup_cuda_kernel_64<<<
            (keys.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            keys.size(0),
            hashmap.data_ptr<uint64_t>(),
            keys.data_ptr<uint64_t>(),
            output.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return output;
}


/**
 * Insert 3D coordinates into the hashmap
 * 
 * @param N         number of elements in the hashmap
 * @param M         number of keys to be inserted
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 * @param hashmap   [2N] uint32 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be inserted
 * @param values    [M] uint32 tensor containing the values to be inserted
 */
__global__ void hashmap_insert_3d_cuda_kernel(
    const uint32_t N,
    const int64_t M,
    const int W,
    const int H,
    const int D,
    uint32_t* __restrict__ hashmap,
    const int32_t* __restrict__ coords,
    const uint32_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        uint32_t key = static_cast<uint32_t>((((b * W + x) * H + y) * D + z));
        uint32_t value = values[thread_id];
        linear_probing_insert(hashmap, key, value, N);
    }
}


__global__ void hashmap_insert_3d_cuda_kernel_64(
    const uint64_t N,
    const int64_t M,
    const int W,
    const int H,
    const int D,
    uint64_t* __restrict__ hashmap,
    const int32_t* __restrict__ coords,
    const uint64_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        uint64_t key = (((((uint64_t)b * W) + x) * H + y) * D + z);
        uint64_t value = values[thread_id];
        linear_probing_insert_64(hashmap, key, value, N);
    }
}


/**
 * Insert 3D coordinates into the hashmap
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be inserted
 * @param values    [M] uint32/uint64 tensor containing the values to be inserted
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 */
void hashmap_insert_3d_cuda(
    torch::Tensor& hashmap,
    const torch::Tensor& coords,
    const torch::Tensor& values,
    int W,
    int H,
    int D
) {
    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_insert_3d_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            values.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_insert_3d_cuda_kernel_64<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint64_t>(),
            coords.data_ptr<int32_t>(),
            values.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}


/**
 * Lookup 3D coordinates in the hashmap
 * 
 * @param N         number of elements in the hashmap
 * @param M         number of keys to be looked up
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 * @param hashmap   [2N] uint32 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be looked up
 * @param values    [M] uint32 tensor containing the values to be looked up
 */
__global__ void hashmap_lookup_3d_cuda_kernel(
    const uint32_t N,
    const int64_t M,
    const int W,
    const int H,
    const int D,
    const uint32_t* __restrict__ hashmap,
    const int32_t* __restrict__ coords,
    uint32_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
            values[thread_id] = K_EMPTY;
            return;
        }
        uint32_t key = static_cast<uint32_t>((((b * W + x) * H + y) * D + z));
        values[thread_id] = linear_probing_lookup(hashmap, key, N);
    }
}


__global__ void hashmap_lookup_3d_cuda_kernel_64(
    const uint64_t N,
    const int64_t M,
    const int W,
    const int H,
    const int D,
    const uint64_t* __restrict__ hashmap,
    const int32_t* __restrict__ coords,
    uint64_t* __restrict__ values
) {
    int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
            values[thread_id] = K_EMPTY_64;
            return;
        }
        uint64_t key = (((((uint64_t)b * W) + x) * H + y) * D + z);
        values[thread_id] = linear_probing_lookup_64(hashmap, key, N);
    }
}


/**
 * Lookup 3D coordinates in the hashmap
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be looked up
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 * 
 * @return          [M] uint32/uint64 tensor containing the values of the keys
 */
torch::Tensor hashmap_lookup_3d_cuda(
    const torch::Tensor& hashmap,
    const torch::Tensor& coords,
    int W,
    int H,
    int D
) {
    // Allocate output tensor
    auto output = torch::empty({coords.size(0)}, torch::dtype(hashmap.dtype()).device(hashmap.device()));

    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_lookup_3d_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>(),
            output.data_ptr<uint32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_lookup_3d_cuda_kernel_64<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint64_t>(),
            coords.data_ptr<int32_t>(),
            output.data_ptr<uint64_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    return output;
}


/**
 * Insert 3D coordinates into the hashmap using index as value
 * 
 * @param N         number of elements in the hashmap
 * @param M         number of 3d coordinates
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 * @param hashmap   [2N] uint32 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be inserted
 */
__global__ void hashmap_insert_3d_idx_as_val_cuda_kernel(
    const uint32_t N,
    const uint32_t M,
    const int W,
    const int H,
    const int D,
    uint32_t* __restrict__ hashmap,
    const int32_t* __restrict__ coords
) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        uint32_t key = static_cast<uint32_t>((((b * W + x) * H + y) * D + z));
        linear_probing_insert(hashmap, key, thread_id, N);
    }
}


__global__ void hashmap_insert_3d_idx_as_val_cuda_kernel_64(
    const uint64_t N,
    const uint64_t M,
    const int W,
    const int H,
    const int D,
    uint64_t* __restrict__ hashmap,
    const int32_t* __restrict__ coords
) {
    uint64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < M) {
        int4 coord = reinterpret_cast<const int4*>(coords)[thread_id];
        int b = coord.x;
        int x = coord.y;
        int y = coord.z;
        int z = coord.w;
        uint64_t key = (((((uint64_t)b * W) + x) * H + y) * D + z);
        linear_probing_insert_64(hashmap, key, thread_id, N);
    }
}


/**
 * Insert 3D coordinates into the hashmap using index as value
 * 
 * @param hashmap   [2N] uint32/uint64 tensor containing the hashmap (key-value pairs)
 * @param coords    [M, 4] int32 tensor containing the keys to be inserted
 * @param W         the number of width dimensions
 * @param H         the number of height dimensions
 * @param D         the number of depth dimensions
 */
void hashmap_insert_3d_idx_as_val_cuda(
    torch::Tensor& hashmap,
    const torch::Tensor& coords,
    int W,
    int H,
    int D
) {
    // Dispatch to 32-bit or 64-bit kernel
    if (hashmap.dtype() == torch::kUInt32) {
        hashmap_insert_3d_idx_as_val_cuda_kernel<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint32_t>(),
            coords.data_ptr<int32_t>()
        );
    } else if (hashmap.dtype() == torch::kUInt64) {
        hashmap_insert_3d_idx_as_val_cuda_kernel_64<<<
            (coords.size(0) + BLOCK_SIZE - 1) / BLOCK_SIZE,
            BLOCK_SIZE
        >>>(
            hashmap.size(0) / 2,
            coords.size(0),
            W,
            H,
            D,
            hashmap.data_ptr<uint64_t>(),
            coords.data_ptr<int32_t>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }
}
