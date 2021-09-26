#pragma once

#include <cstddef>

#ifndef FORCE_32_BIT
typedef size_t ptr_t;
#else
typedef unsigned ptr_t;
#endif
