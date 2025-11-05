#define K_EMPTY 0xffffffff
#define K_EMPTY_64 0xffffffffffffffffULL


// 32 bit Murmur3 hash
__forceinline__ __device__ uint32_t hash(uint32_t k, uint32_t N) {
    k ^= k >> 16;
    k *= 0x85ebca6b;
    k ^= k >> 13;
    k *= 0xc2b2ae35;
    k ^= k >> 16;
    return k % N;
}


// 64 bit Murmur3 hash
__forceinline__ __device__ uint64_t hash_64(uint64_t k, uint64_t N) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k % N;
}


__forceinline__ __device__ void linear_probing_insert(
    uint32_t* hashmap,
    const uint32_t key,
    const uint32_t values,
    const int64_t N
) {
    uint32_t slot = hash(key, N);
    while (true) {
        uint32_t prev = atomicCAS(&hashmap[slot], K_EMPTY, key);
        if (prev == K_EMPTY || prev == key) {
            hashmap[slot + N] = values;
            return;
        }
        slot = (slot + 1) % N;
    }
}


__forceinline__ __device__ uint32_t linear_probing_lookup(
    const uint32_t* hashmap,
    const uint32_t key,
    const int64_t N
) {
    uint32_t slot = hash(key, N);
    while (true) {
        uint32_t prev = hashmap[slot];
        if (prev == K_EMPTY) {
            return K_EMPTY;
        }
        if (prev == key) {
            return hashmap[slot + N];
        }
        slot = (slot + 1) % N;
    }
}


__forceinline__ __device__ void linear_probing_insert_64(
    uint64_t* hashmap,
    const uint64_t key,
    const uint64_t value,
    const int64_t N
) {
    uint64_t slot = hash_64(key, N);
    while (true) {
        uint64_t prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&hashmap[slot]),
            static_cast<unsigned long long>(K_EMPTY_64),
            static_cast<unsigned long long>(key)
        );
        if (prev == K_EMPTY_64 || prev == key) {
            hashmap[slot + N] = value;
            return;
        }
        slot = (slot + 1) % N;
    }
}


__forceinline__ __device__ uint64_t linear_probing_lookup_64(
    const uint64_t* hashmap,
    const uint64_t key,
    const int64_t N
) {
    uint64_t slot = hash_64(key, N);
    while (true) {
        uint64_t prev = hashmap[slot];
        if (prev == K_EMPTY_64) {
            return K_EMPTY_64;
        }
        if (prev == key) {
            return hashmap[slot + N];
        }
        slot = (slot + 1) % N;
    }
}
