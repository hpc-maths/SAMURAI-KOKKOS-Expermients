#pragma once

template <typename DstView, typename SrcView>
inline void partial_deep_copy(DstView& dst, const SrcView& src, std::size_t size)
{
    static_assert(std::is_same_v<typename DstView::data_type,
                                 typename SrcView::data_type>,
                  "Source and destination views must have the same data type");

    using DataType       = typename SrcView::data_type;
    using SrcMemSpace    = typename SrcView::memory_space;
    using DstMemSpace    = typename DstView::memory_space;

    Kokkos::View<DataType, SrcMemSpace, Kokkos::MemoryUnmanaged> src_sub(src.data(), size);

    Kokkos::View<DataType, DstMemSpace, Kokkos::MemoryUnmanaged> dst_sub(dst.data(), size);

    Kokkos::deep_copy(dst_sub, src_sub);
}
