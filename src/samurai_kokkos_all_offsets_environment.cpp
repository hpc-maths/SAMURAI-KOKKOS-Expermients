#include <samurai_kokkos_all_offsets_environment.hpp>
#include <utils.hpp>

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
	
	partial_deep_copy(m_device_offsets, m_host_offsets, m_size);
}
