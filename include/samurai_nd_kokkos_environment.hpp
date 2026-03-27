#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

template<std::size_t NOffsets>
class SamuraiNDKokkosEnvironment
{
public:
	using DeviceOffsets       = Kokkos::View<std::size_t*[NOffsets]>;
	using DeviceIntervalSizes = Kokkos::View<std::size_t*>;
	using HostOffsets         = typename DeviceOffsets::host_mirror_type;
	using HostIntervalSizes   = typename DeviceIntervalSizes::host_mirror_type;
	
	void clear() { m_size = 0; }
	
	void reserve(const std::size_t new_size);
	
	void add_offset_and_interval_size(const std::array<std::size_t, NOffsets>& offset, const std::size_t intervalSize);
	
	void copy_data_to_host();
	
	std::size_t size() const { return m_size; }
	
	DeviceOffsets get_device_offsets() { return m_device_offsets; }
	
	DeviceIntervalSizes get_device_interval_sizes() { return m_device_interval_sizes; }
private:
	HostOffsets         m_host_offsets;
	HostIntervalSizes   m_host_interval_sizes;
	DeviceOffsets       m_device_offsets;
	DeviceIntervalSizes m_device_interval_sizes;
	std::size_t         m_size            = {};
	std::size_t         m_host_capacity   = {}; // capacity of the host views
	std::size_t         m_device_capacity = {}; // capacity of the host views
};

////////////////////////////////////////////////////////////////////////
//// Method implementation
////////////////////////////////////////////////////////////////////////

#include <utils.hpp>

template<std::size_t NOffsets>
void SamuraiNDKokkosEnvironment<NOffsets>::reserve(const std::size_t new_size)
{
	if (new_size > m_host_capacity)
	{
		Kokkos::resize(m_host_offsets,        new_size);
		Kokkos::resize(m_host_interval_sizes, new_size);
		
		m_host_capacity = new_size;
	}
}

template<std::size_t NOffsets>
void SamuraiNDKokkosEnvironment<NOffsets>::add_offset_and_interval_size(const std::array<std::size_t, NOffsets>& offset, const std::size_t intervalSize)
{
	if (m_size == m_host_capacity)
	{
		const std::size_t new_capacity = 1 + 2*m_host_capacity;
		
		Kokkos::resize(m_host_offsets,        new_capacity);
		Kokkos::resize(m_host_interval_sizes, new_capacity);
		
		m_host_capacity = new_capacity;
	}
	for (std::size_t offsetId = 0;  offsetId != NOffsets; ++offsetId)
	{
		m_host_offsets(m_size, offsetId) = offset[offsetId];
	}
	m_host_interval_sizes[m_size] = intervalSize;
	
	++m_size;
}

template<std::size_t NOffsets>
void SamuraiNDKokkosEnvironment<NOffsets>::copy_data_to_host()
{	
	if (m_size > m_device_capacity)
	{
		Kokkos::resize(m_device_offsets,        m_size);
		Kokkos::resize(m_device_interval_sizes, m_size);
		
		m_device_capacity = m_size;
	}
	
	partial_deep_copy(m_device_offsets, m_host_offsets, m_size);
	partial_deep_copy(m_device_interval_sizes, m_host_interval_sizes, m_size);
}
