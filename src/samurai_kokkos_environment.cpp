#include <samurai_kokkos_environment.hpp>

void SamuraiKokkosEnvironment::reserve(const std::size_t new_size)
{
	if (new_size > m_host_capacity)
	{
		Kokkos::resize(m_host_offsets,        new_size);
		Kokkos::resize(m_host_interval_sizes, new_size);
		
		m_host_capacity = new_size;
	}
}

void SamuraiKokkosEnvironment::add_offset_and_interval_size(const int offset, const std::size_t intervalSize)
{
	if (m_size == m_host_capacity)
	{
		const std::size_t new_capacity = 1 + 2*m_host_capacity;
		
		Kokkos::resize(m_host_offsets,        new_capacity);
		Kokkos::resize(m_host_interval_sizes, new_capacity);
		
		m_host_capacity = new_capacity;
	}
	m_host_offsets[m_size]        = offset;
	m_host_interval_sizes[m_size] = intervalSize;
	
	++m_size;
}

void SamuraiKokkosEnvironment::copy_data_to_host()
{
	if (m_size > m_device_capacity)
	{
		Kokkos::resize(m_device_offsets,        m_size);
		Kokkos::resize(m_device_interval_sizes, m_size);
		
		m_device_capacity = m_size;
	}
	auto devive_offsets_subview = Kokkos::subview(m_device_offsets, Kokkos::make_pair(std::size_t(), m_size));
	auto host_offsets_subview   = Kokkos::subview(m_host_offsets,   Kokkos::make_pair(std::size_t(), m_size));
	
	auto device_interval_sizes_subview = Kokkos::subview(m_device_interval_sizes, Kokkos::make_pair(std::size_t(), m_size));
	auto host_interval_sizes_subview   = Kokkos::subview(m_host_interval_sizes,   Kokkos::make_pair(std::size_t(), m_size));
	
	Kokkos::deep_copy(devive_offsets_subview,        host_offsets_subview);
	Kokkos::deep_copy(device_interval_sizes_subview, host_interval_sizes_subview);
}
