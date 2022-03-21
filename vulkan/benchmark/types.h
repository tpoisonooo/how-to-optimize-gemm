#pragma once
#include <vector>

template <class T>
struct AlignAllocator {
    typedef T value_type;

    AlignAllocator() = default;
    template <class U>
    constexpr AlignAllocator(const AlignAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        if (void* p = std::aligned_alloc(64, n * sizeof(T))) {
            return static_cast<T*>(p);
        }

        throw std::bad_alloc();
    }

    void deallocate(T* p, std::size_t n) noexcept { std::free(p); }
};

template <class T, class U>
bool operator==(const AlignAllocator<T>&, const AlignAllocator<U>&) {
    return true;
}

template <class T, class U>
bool operator!=(const AlignAllocator<T>&, const AlignAllocator<U>&) {
    return false;
}

using AlignVector = std::vector<float, AlignAllocator<float> >;
