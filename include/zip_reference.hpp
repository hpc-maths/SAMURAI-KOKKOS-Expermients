#pragma once

#include <tuple>
#include <utility>

template<typename... Refs>
class ZipReference
{
public:
	ZipReference(Refs... refs) : m_refs(refs...) {}

	// Convert to value_type (tuple of values)
	constexpr operator std::tuple<std::remove_reference_t<Refs>...>() const
	{
		return std::apply([](const auto&... refs) 
		{
			return std::tuple<std::remove_reference_t<Refs>...>(refs...);
		}, m_refs);
	}

	// Assignment from another ZipReference
	template<typename... OtherRefs>
	constexpr ZipReference& operator=(const ZipReference<OtherRefs...>& other)
	{
		std::apply([&](auto&... lhs) 
		{
			std::apply([&](const auto&... rhs) 
			{ 
				((lhs = rhs), ...); 
			}, other.refs);
		}, m_refs);
	
		return *this;
	}

	// Assignment from value_type
	constexpr ZipReference& operator=(const std::tuple<std::remove_reference_t<Refs>...>& val)
	{
		std::apply([&](auto&... lhs) 
		{
			std::apply([&](const auto&... rhs) { ((lhs = rhs), ...); }, val);
		}, m_refs);
	
		return *this;
	}

	friend constexpr void swap(ZipReference a, ZipReference b)
	{
		std::apply([&](auto&... lhs) 
		{
			std::apply([&](auto&... rhs) 
			{ 
				((std::swap(lhs, rhs)), ...); 
			}, b.m_refs);
		}, a.m_refs);
	}

	template<std::size_t I>
	constexpr decltype(auto) get(std::integral_constant<std::size_t, I>) { return std::get<I>(m_refs); } 

	template<std::size_t I>
	constexpr decltype(auto) get(std::integral_constant<std::size_t, I>) const { return std::get<I>(m_refs); } 

private:
	std::tuple<Refs...> m_refs;
};


namespace std 
{

template<std::size_t I, typename... Refs>
constexpr decltype(auto) get(ZipReference<Refs...>& zipRefs) { return zipRefs.get(std::integral_constant<std::size_t, I>{}); }

template<std::size_t I, typename... Refs>
constexpr decltype(auto) get(const ZipReference<Refs...>& zipRefs) { return zipRefs.get(std::integral_constant<std::size_t, I>{}); }

} // namespace std
