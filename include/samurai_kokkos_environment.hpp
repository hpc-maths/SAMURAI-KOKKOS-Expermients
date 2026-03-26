#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

class SamuraiKokkosEnvironment
{
public:
	using DeviceOffsets       = Kokkos::View<int*>;
	using DeviceIntervalSizes = Kokkos::View<std::size_t*>;
	using HostOffsets         = typename DeviceOffsets::host_mirror_type;
	using HostIntervalSizes   = typename DeviceIntervalSizes::host_mirror_type;
	
	void clear() { m_size = 0; }
	
	void reserve(const std::size_t new_size);
	
	void add_offset_and_interval_size(const int offset, const std::size_t intervalSize);
	
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
