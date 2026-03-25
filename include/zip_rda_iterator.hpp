#pragma once

#include <zip_reference.hpp>

#include <tuple>
#include <iterator>
#include <utility>
#include <concepts>

template<std::random_access_iterator... InnerIts>
class ZipRDAIterator 
{
public:
	using iterator_category = std::random_access_iterator_tag;
	using value_type		= std::tuple<std::iter_value_t<InnerIts>...>;
	using difference_type   = std::common_type_t<std::iter_difference_t<InnerIts>...>;
	using pointer		   = void;
	using reference		 = ZipReference<std::iter_reference_t<InnerIts>...>;

	constexpr ZipRDAIterator() = default;

	constexpr ZipRDAIterator(InnerIts... innerIts) : m_innerIts(std::move(innerIts)...) {}

	constexpr reference operator*() const 
	{
		return std::apply([](auto&... its) 
		{
			return reference(*its...);
		}, m_innerIts);
	}

	constexpr reference operator[](difference_type n) const { return *(*this + n); }

	constexpr ZipRDAIterator& operator++() 
	{
		std::apply([](auto&... its) { (++its, ...); }, m_innerIts);
		return *this;
	}

	constexpr ZipRDAIterator operator++(int) 
	{
		ZipRDAIterator tmp = *this;
		++(*this);
		return tmp;
	}

	constexpr ZipRDAIterator& operator--() 
	{
		std::apply([](auto&... its) { (--its, ...); }, m_innerIts);
		return *this;
	}

	constexpr ZipRDAIterator operator--(int) 
	{
		ZipRDAIterator tmp = *this;
		--(*this);
		return tmp;
	}

	constexpr ZipRDAIterator& operator+=(difference_type n) 
	{
		std::apply([n](auto&... its) { ((its += n), ...); }, m_innerIts);
		return *this;
	}

	constexpr ZipRDAIterator& operator-=(difference_type n) 
	{
		return *this += -n;
	}

	friend constexpr ZipRDAIterator operator+(ZipRDAIterator it, difference_type n) 
	{
		return it += n;
	}

	friend constexpr ZipRDAIterator operator+(difference_type n, ZipRDAIterator it) 
	{
		return it += n;
	}

	friend constexpr ZipRDAIterator operator-(ZipRDAIterator it, difference_type n) 
	{
		return it -= n;
	}

	friend constexpr difference_type operator-(const ZipRDAIterator& lhs, const ZipRDAIterator& rhs) 
	{
		return std::get<0>(lhs.m_innerIts) - std::get<0>(rhs.m_innerIts);
	}

	friend constexpr bool operator==(const ZipRDAIterator& lhs, const ZipRDAIterator& rhs) 
	{
		return std::get<0>(lhs.m_innerIts) == std::get<0>(rhs.m_innerIts);
	}

	friend constexpr auto operator<=>(const ZipRDAIterator& lhs, const ZipRDAIterator& rhs) 
	{
		return std::get<0>(lhs.m_innerIts) <=> std::get<0>(rhs.m_innerIts);
	}

	friend constexpr void iter_swap(const ZipRDAIterator& a, const ZipRDAIterator& b) 
	{
		std::apply([&](auto&... itsA) 
		{
			std::apply([&](auto&... itsB) {
				((std::iter_swap(itsA, itsB)), ...);
			}, b.m_innerIts);
		}, a.m_innerIts);
	}

private:
	std::tuple<InnerIts...> m_innerIts;
};

template<typename... Iterators>
constexpr ZipRDAIterator<Iterators...> make_zip_rda_iterator(Iterators... its) { return ZipRDAIterator<Iterators...>(its...); }
