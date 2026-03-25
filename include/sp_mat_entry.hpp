#pragma once

#include <utility>
#include <concepts>

template<typename T, std::integral Int>
class SpMatEntry
{
public:
	constexpr SpMatEntry() = default;

	constexpr SpMatEntry(const Int i, const Int j, const T& value) : m_i(i), m_j(j), m_value(value) {}
	
	constexpr SpMatEntry(const Int i, const Int j, T&& value) : m_i(i), m_j(j), m_value(std::move(value)) {}
	
	constexpr Int row() const { return m_i; }
	
	constexpr Int col() const { return m_j; }
	
	constexpr const T& value() const { return m_value; }
private:
	Int m_i;
	Int m_j;
	T	m_value;
};

