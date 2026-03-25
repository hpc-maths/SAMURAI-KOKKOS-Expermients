#pragma once

#include <sp_mat_entry.hpp>

#include <span>
#include <vector>
#include <concepts>

template<typename Range, typename T>
concept InputRangeOf = std::ranges::input_range<Range> and std::same_as<std::ranges::range_value_t<Range>, T>;

template<typename T, std::integral Offset = std::ptrdiff_t, std::integral Size = std::size_t>
class CsrMatrix
{
public:
	using Entry = SpMatEntry<T, Offset>;

	CsrMatrix(const Size nRows) : m_row_ptr(nRows+1) {}
	
	CsrMatrix(const Size nRows, const Size nnz) : m_row_ptr(nRows+1), m_col_idx(nnz), m_values(nnz) {}
	
	template<InputRangeOf<Entry> Entries> 
	void initFromEntries(const Entries& entries);
	
	template<InputRangeOf<Entry> Entries> 
	void initFromEntriesWithoutReallocate(const Entries& entries);
	
	Size nRows() const { return Size(m_row_ptr.size()-1); }
	
	Size nnz() const  { return m_row_ptr.back(); }
	
	std::span<Size> row_ptr() { return std::span(m_row_ptr); }
	
	std::span<Offset> col_idx() { return std::span(m_col_idx); }
	
	std::span<T> values_data() { return std::span(m_values); }

	std::span<const Size> row_ptr() const { return std::span(m_row_ptr); }
	
	std::span<const Offset> col_idx() const { return std::span(m_col_idx); }
	
	std::span<const T> values() const { return std::span(m_values); }
private:
	std::vector<Size>   m_row_ptr;
	std::vector<Offset> m_col_idx;
	std::vector<T>      m_values;
	
	std::vector<Size> m_work;
};

////////////////////////////////////////////////////////////////////////
//// Method implementations
////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cassert>
#include <ranges>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <zip_rda_iterator.hpp>

extern template class CsrMatrix<double, std::ptrdiff_t, std::size_t>;
extern template void  CsrMatrix<double, std::ptrdiff_t, std::size_t>::initFromEntries(const std::vector<typename CsrMatrix<double>::Entry>&);
extern template void  CsrMatrix<double, std::ptrdiff_t, std::size_t>::initFromEntriesWithoutReallocate(const std::vector<typename CsrMatrix<double>::Entry>&);

template<typename T, std::integral Offset, std::integral Size> template<InputRangeOf<typename CsrMatrix<T, Offset, Size>::Entry> Entries> 
void CsrMatrix<T, Offset, Size>::initFromEntries(const Entries& entries)
{
	m_col_idx.resize(entries.size());
	m_values.resize(entries.size());
	
	initFromEntriesWithoutReallocate(entries);
}

template<typename T, std::integral Offset, std::integral Size> template<InputRangeOf<typename CsrMatrix<T, Offset, Size>::Entry> Entries> 
void CsrMatrix<T, Offset, Size>::initFromEntriesWithoutReallocate(const Entries& entries)
{
	assert(m_col_idx.size() >= entries.size());
	assert(m_values.size()  >= entries.size());
	
	// 1. Prepare row_ptr with zeros
	std::ranges::fill(m_row_ptr, std::ptrdiff_t{});

	// 2. First pass: Count the number of non-zeros in each row
	for (const Entry& entry : entries) { m_row_ptr[Size(entry.row() + 1)] += 1; }

	// 3. Prefix sum: Transform counts into starting offsets
	// After this, row_ptr[i] is the starting index in col_idx/values for row i
	for (Size i = 0; i != nRows(); ++i) { m_row_ptr[i + 1] += m_row_ptr[i]; }

	// 4. Second pass: Place entries into the CSR arrays
	// We use a temporary copy of row_ptr to track the current insertion position for each row
	m_work.assign(std::ranges::begin(m_row_ptr), std::ranges::end(m_row_ptr));

	for (const Entry& entry : entries) 
	{
		Size& current_row_pos = m_work[Size(entry.row())];
	
		Size dest_idx = current_row_pos;
		++current_row_pos;
	
		m_col_idx[dest_idx] = entry.col();
		m_values[dest_idx]  = entry.value();
	}
	
	const auto isRowNotEmpty = [row_offset = m_row_ptr.data()](const Size i) -> bool
	{
		return row_offset[i] != row_offset[i+1];
	};
	
	for (Size i : std::views::iota(Size{}, nRows()) | std::views::filter(isRowNotEmpty))
	{
		Size row_start = m_row_ptr[i];
		Size row_end   = m_row_ptr[i + 1];
	
		const auto start = make_zip_rda_iterator(std::ranges::begin(m_col_idx) + Offset(row_start), std::ranges::begin(m_values) + Offset(row_start));
		const auto end   = make_zip_rda_iterator(std::ranges::begin(m_col_idx) + Offset(row_end),   std::ranges::begin(m_values) + Offset(row_end));
	
		std::sort(start, end, [](const auto& a, const auto& b) -> bool
		{
			return std::get<0>(a) < std::get<0>(b);
		});
	
		Size write_idx = row_start;
		for (Size read_idx = row_start+1; read_idx !=row_end; ++read_idx)
		{
			assert(static_cast<std::size_t>(write_idx) < m_col_idx.size());
			assert(static_cast<std::size_t>(write_idx) < m_values.size());
			assert(static_cast<std::size_t>(read_idx) < m_col_idx.size());
			assert(static_cast<std::size_t>(read_idx) < m_values.size());
			
			if (m_col_idx[write_idx] == m_col_idx[read_idx]) { m_values[write_idx] += m_values[read_idx]; } 
			else
			{
				++write_idx;
				m_col_idx[write_idx] = m_col_idx[read_idx];
				m_values[write_idx]  = m_values[read_idx];
			}
		}
		// Update the next row's start to reflect the removed duplicates
		m_row_ptr[i + 1] = write_idx + 1;
	}
}
