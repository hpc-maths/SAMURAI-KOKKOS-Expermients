#pragma once

#include <vector>

#include <Kokkos_Core.hpp>

class SamuraiKokkosAllOffsetsEnvironment
{
public:
	using DeviceOffsets = Kokkos::View<int*>;
	using HostOffsets   = typename DeviceOffsets::host_mirror_type;
	
	void clear() { m_size = 0; }
	
	void reserve(const std::size_t new_size);
	
	void add_offset(const int offset);
	
	void copy_data_to_host();
	
	std::size_t size() const { return m_size; }
	
	DeviceOffsets get_device_offsets() { return m_device_offsets; }
private:
	HostOffsets   m_host_offsets;
	DeviceOffsets m_device_offsets;
	std::size_t   m_size            = {};
	std::size_t   m_host_capacity   = {}; // capacity of the host views
	std::size_t   m_device_capacity = {}; // capacity of the host views
};
