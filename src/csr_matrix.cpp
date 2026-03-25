#include <csr_matrix.hpp>

template class CsrMatrix<double, std::ptrdiff_t, std::size_t>;
template void  CsrMatrix<double, std::ptrdiff_t, std::size_t>::initFromEntries(const std::vector<typename CsrMatrix<double>::Entry>&);
template void  CsrMatrix<double, std::ptrdiff_t, std::size_t>::initFromEntriesWithoutReallocate(const std::vector<typename CsrMatrix<double>::Entry>&);
