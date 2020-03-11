#pragma once

#include <string>
#include <memory>

#include <IBuffer.h>
#include <Vector.h>
#include <Types.h>

namespace cl
{
	template<MemorySpace memorySpace = MemorySpace::Device, MathDomain mathDomain = MathDomain::Float>
	class VectorCollection
	{
	public:
		explicit VectorCollection(const std::vector<size_t>& sizes_)
			: data(static_cast<unsigned>(std::accumulate(sizes_.begin(), sizes_.end(), 0ul)), 0.0), sizes(sizes_)
		{
			size_t startOffset = 0;
			size_t endOffset = 0;
			for (size_t i = 0; i < sizes.size(); ++i)
			{
				startOffset = endOffset;
				endOffset += sizes[i];
				
				vectors.emplace_back(data, startOffset, endOffset);
				assert(!vectors.back().OwnsMemory());
			}
		}
		
		Vector<memorySpace, mathDomain>& Get() noexcept { return data; }
		Vector<memorySpace, mathDomain>& operator[](const size_t i) noexcept { return vectors[i]; }
		Vector<memorySpace, mathDomain>& front() noexcept { return vectors.front(); }
		Vector<memorySpace, mathDomain>& back() noexcept { return vectors.back(); }
		
	private:
		Vector<memorySpace, mathDomain> data;
		const std::vector<size_t> sizes;
		std::vector<Vector<memorySpace, mathDomain>> vectors {};
	};
}

#include <Vector.tpp>

