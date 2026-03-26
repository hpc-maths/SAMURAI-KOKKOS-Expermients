#include <samurai_kokkos_all_offsets_environment.hpp>

#include <fmt/format.h>

void SamuraiKokkosAllOffsetsEnvironment::reserve(const std::size_t new_size)
{
	if (new_size > m_host_capacity)
	{
		Kokkos::resize(m_host_offsets, new_size);
		
		m_host_capacity = new_size;
	}
}

void SamuraiKokkosAllOffsetsEnvironment::add_offset(const int offset)
{
	if (m_size == m_host_capacity)
	{
		const std::size_t new_capacity = 1 + 2*m_host_capacity;
		
		Kokkos::resize(m_host_offsets, new_capacity);
		
		m_host_capacity = new_capacity;
	}
	m_host_offsets[m_size] = offset;
	
	++m_size;
}

void SamuraiKokkosAllOffsetsEnvironment::copy_data_to_host()
{	
	if (m_size > m_device_capacity)
	{
		Kokkos::resize(m_device_offsets, m_size);
		
		m_device_capacity = m_size;
	}
	auto devive_offsets_subview = Kokkos::subview(m_device_offsets, Kokkos::make_pair(std::size_t(), m_size));
	auto host_offsets_subview   = Kokkos::subview(m_host_offsets,   Kokkos::make_pair(std::size_t(), m_size));
	
	Kokkos::deep_copy(devive_offsets_subview, host_offsets_subview);
}
