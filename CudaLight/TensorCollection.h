#pragma once

#include <string>
#include <memory>

#include <IBuffer.h>
#include <Tensor.h>
#include <Types.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class TensorCollection
	{
	public:
		explicit TensorCollection(const std::vector<std::tuple<size_t, size_t, size_t>>& sizes_)
			: data(static_cast<unsigned>(std::accumulate(sizes_.begin(), sizes_.end(), 0ul, [](const auto& x, const auto& y) { return x + std::get<0>(y) * std::get<1>(y) * std::get<2>(y); })), 0.0),
			  sizes(sizes_)
		{
			size_t startOffset = 0;
			size_t endOffset = 0;
			for (size_t i = 0; i < sizes.size(); ++i)
			{
				startOffset = endOffset;
				endOffset += std::get<0>(sizes[i]) * std::get<1>(sizes[i]) * std::get<2>(sizes[i]);
				
				tensors.emplace_back(data, startOffset, std::get<0>(sizes[i]), std::get<1>(sizes[i]), std::get<2>(sizes[i]));
				assert(!tensors.back().OwnsMemory());
			}
		}
		
		Vector<memorySpace, mathDomain>& Get() noexcept { return data; }
		Tensor<memorySpace, mathDomain>& operator[](const size_t i) noexcept { return tensors[i]; }
		Tensor<memorySpace, mathDomain>& front() noexcept { return tensors.front(); }
		Tensor<memorySpace, mathDomain>& back() noexcept { return tensors.back(); }
		
	private:
		Vector<memorySpace, mathDomain> data;
		const std::vector<std::tuple<size_t, size_t, size_t>> sizes;
		std::vector<Tensor<memorySpace, mathDomain>> tensors;
	};
}

#include <Vector.tpp>

