#pragma once

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
//  do nothing and hope for the best?
	#define EXPORT
	#define IMPORT
	#error Unknown dynamic link import/export semantics.
#endif

#include <cstddef>

#define EXTERN_C extern "C"

#define RESTRICT __restrict__

#ifndef FORCE_32_BIT
typedef size_t ptr_t;
#else
typedef unsigned ptr_t;
#endif