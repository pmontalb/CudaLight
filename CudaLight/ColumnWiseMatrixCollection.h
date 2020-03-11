#pragma once

#include <string>
#include <memory>

#include <IBuffer.h>
#include <ColumnWiseMatrix.h>
#include <Types.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class ColumnWiseMatrixCollection
	{
	public:
		explicit ColumnWiseMatrixCollection(const std::vector<std::pair<size_t, size_t>>& sizes_)
			: data(static_cast<unsigned>(std::accumulate(sizes_.begin(), sizes_.end(), 0ul, [](const auto& x, const auto& y) { return x + y.first * y.second; })), 0.0),
			  sizes(sizes_)
		{
			size_t startOffset = 0;
			size_t endOffset = 0;
			for (size_t i = 0; i < sizes.size(); ++i)
			{
				startOffset = endOffset;
				endOffset += sizes[i].first * sizes[i].second;
				
				matrices.emplace_back(data, startOffset, sizes[i].first, sizes[i].second);
				assert(!matrices.back().OwnsMemory());
			}
		}
		
		Vector<memorySpace, mathDomain>& Get() noexcept { return data; }
		ColumnWiseMatrix<memorySpace, mathDomain>& operator[](const size_t i) noexcept { return matrices[i]; }
		ColumnWiseMatrix<memorySpace, mathDomain>& front() noexcept { return matrices.front(); }
		ColumnWiseMatrix<memorySpace, mathDomain>& back() noexcept { return matrices.back(); }
		
	private:
		Vector<memorySpace, mathDomain> data;
		const std::vector<std::pair<size_t, size_t>> sizes;
		std::vector<ColumnWiseMatrix<memorySpace, mathDomain>> matrices {};
	};
}

#include <Vector.tpp>

