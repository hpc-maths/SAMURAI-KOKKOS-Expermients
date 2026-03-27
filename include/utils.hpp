#pragma once

template <typename DstView, typename SrcView>
void partial_deep_copy(DstView& dst, const SrcView& src, std::size_t size)
{
    static_assert(std::is_same_v<typename DstView::data_type,
                                 typename SrcView::data_type>,
                  "Source and destination views must have the same data type");

    using DataType       = typename SrcView::data_type;
    using SrcMemSpace    = typename SrcView::memory_space;
    using DstMemSpace    = typename DstView::memory_space;
    using SrcLayout      = typename SrcView::array_layout;
    using DstLayout      = typename DstView::array_layout;

    Kokkos::View<DataType, SrcLayout, SrcMemSpace, Kokkos::MemoryUnmanaged> src_sub(
        src.data(), size);

    Kokkos::View<DataType, DstLayout, DstMemSpace, Kokkos::MemoryUnmanaged> dst_sub(
        dst.data(), size);

    Kokkos::deep_copy(dst_sub, src_sub);
}
