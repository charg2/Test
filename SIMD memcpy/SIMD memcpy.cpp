#include <iostream>
#include <chrono>
#include <cstring>
#include <immintrin.h>
#include <vector>
#include <memory>
#include <iomanip>
#include <thread>
#include <functional>
//#include "Memcpy.h"
constexpr size_t TEST_CASE{ 1'0000 };
constexpr size_t THREAD_COUNT{ 1 };

void avx2_memcpy( void* dest, const void* src, size_t size )
{
    size_t i = 0;
    for ( ; i + 32 <= size; i += 32 )
    {
        __m256i data = _mm256_loadu_si256( reinterpret_cast<const __m256i*>( static_cast<const char*>( src ) + i ) );
        _mm256_storeu_si256( reinterpret_cast<__m256i*>( static_cast<char*>( dest ) + i ), data );
    }

    std::memcpy( static_cast<char*>( dest ) + i, static_cast<const char*>( src ) + i, size - i );
}

void avx512_memcpy( void* dest, const void* src, size_t size )
{
    size_t i = 0;
    for ( ; i + 64 <= size; i += 64 )
    {
        __m512i data = _mm512_loadu_si512( reinterpret_cast<const __m512i*>( static_cast<const char*>( src ) + i ) );
        _mm512_storeu_si512( reinterpret_cast<__m512i*>( static_cast<char*>( dest ) + i ), data );
    }

    std::memcpy( static_cast<char*>( dest ) + i, static_cast<const char*>( src ) + i, size - i );
}

void sse_memcpy( void* dest, const void* src, size_t size )
{
    size_t i = 0;
    for ( ; i + 16 <= size; i += 16 )
    {
        __m128i data = _mm_loadu_si128( reinterpret_cast<const __m128i*>( static_cast<const char*>( src ) + i ) );
        _mm_storeu_si128( reinterpret_cast<__m128i*>( static_cast<char*>( dest ) + i ), data );
    }

    std::memcpy( static_cast<char*>( dest ) + i, static_cast<const char*>( src ) + i, size - i );
}

template< std::invocable TTask >
void DoTest( TTask&& func, const char* label )
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();

	/// 시간 단위에 맞춰 duration을 출력
	if ( std::chrono::duration_cast<std::chrono::seconds>( end - start ).count() > 100 )
	{
		auto duration = std::chrono::duration_cast<std::chrono::seconds>( end - start ).count();
		std::cout << label << "                   " << duration << " sec" << std::endl;
		return;
	}
	if ( std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count() > 100 )
	{
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
		std::cout << label << "                   " << duration << " ms" << std::endl;
		return;
	}
	if ( std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count() > 100 )
	{
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();
		std::cout << label << "                   " << duration << " us" << std::endl;
		return;
	}
    if ( std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count() > 100 )
    {
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();
        std::cout << label << "                   " << duration << " ns" << std::endl;
        return;
    }
}

template< std::invocable TTask >
void DoTestUsingMultiThread( TTask&& func, const char* label )
{
    DoTest( [ & ]()
        {
            std::vector< std::thread > threads;
			threads.reserve( std::thread::hardware_concurrency() );

			for ( int i = 0; i < std::thread::hardware_concurrency(); ++i )
				threads.emplace_back( func );

			for ( auto& thread : threads )
				thread.join();
        },
        label );
}

void benchmark_memcpy( void* dest, const void* src, size_t size, const char* label, int threadCount = 1 )
{
    if ( threadCount == 1 )
    {
        DoTest( [ dest, src, size ]()
            {
                for ( int i = 0; i < TEST_CASE; ++i )
                {
                    std::memcpy( dest, src, size );
                }
            },
            label );
    }
    else
    {
        DoTestUsingMultiThread( [ dest, src, size ]()
            {
                for ( int i = 0; i < TEST_CASE; ++i )
                {
                    std::memcpy( dest, src, size );
                }
            },
            label );
    }
}

void benchmark_avx2_memcpy( void* dest, const void* src, size_t size, const char* label, int threadCount = 1 )
{
    if ( threadCount == 1 )
    {
        DoTest( [ dest, src, size ]()
            {
                for ( int i = 0; i < TEST_CASE; ++i )
                {
                    avx2_memcpy( dest, src, size );
                }
            },
            label );
    }
    else
    {
		DoTestUsingMultiThread( [ dest, src, size ]()
			{
				for ( int i = 0; i < TEST_CASE; ++i )
				{
					avx2_memcpy( dest, src, size );
				}
			},
			label );
    }
}


void benchmark_avx512_memcpy( unsigned char* dest, const unsigned char* src, size_t size, const char* label, int threadCount = 1 )
{
    if ( threadCount == 1 )
    {
        DoTest( [ dest, src, size ]()
            {
                for ( int i = 0; i < TEST_CASE; ++i )
                {
                    avx512_memcpy( dest, src, size );
                }
            },
            label );
    }
    else
    {
        DoTestUsingMultiThread( [ dest, src, size ]()
            {
                for ( int i = 0; i < TEST_CASE; ++i )
                {
                    avx512_memcpy( dest, src, size );
                }
            },
            label );
    }
}

void benchmark_sse_memcpy( void* dest, const void* src, size_t size, const char* label, int threadCount = 1 )
{
    if ( threadCount == 1 )
    {
        DoTest( [ dest, src, size ]()
            {
                for ( int i = 0; i < TEST_CASE; ++i )
                {
                    sse_memcpy( dest, src, size );
                }
            },
            label );
    }
    else
    {
		DoTestUsingMultiThread( [ dest, src, size ]()
			{
				for ( int i = 0; i < TEST_CASE; ++i )
				{
					sse_memcpy( dest, src, size );
				}
			},
			label );
    }
}


int main()
{
    std::vector< size_t > sizes;
    for ( size_t size = 16; size <= 1024 * 1024 * 100; size *= 2 )
        sizes.push_back( size + ( rand() % size ) );

    for ( const auto& size : sizes )
    {
        auto src = std::make_unique< unsigned char[] >( size );
        auto dest1 = std::make_unique< char[] >( size );
        auto dest2 = std::make_unique< char[] >( size );
        auto dest3 = std::make_unique< unsigned char[] >( size );
        auto dest4 = std::make_unique< char[] >( size );

        // src 배열을 초기화합니다.
        for ( size_t i = 0; i < size; ++i )
            src[ i ] = static_cast<char>( i );

        std::cout << "Benchmarking memcpy with size ";
        if ( size < 1024 )
            std::cout << size << " bytes" << std::endl;
        else if ( size < 1024 * 1024 )
            std::cout << std::fixed << std::setprecision( 2 ) << static_cast<double>( size ) / 1024 << " KB" << std::endl;
        else if ( size < 1024 * 1024 * 1024 )
            std::cout << std::fixed << std::setprecision( 2 ) << static_cast<double>( size ) / ( 1024 * 1024 ) << " MB" << std::endl;
        else
            std::cout << std::fixed << std::setprecision( 2 ) << static_cast<double>( size ) / ( 1024 * 1024 * 1024 ) << " GB" << std::endl;

        // 일반 memcpy 벤치마크
        benchmark_memcpy( dest1.get(), src.get(), size, "Standard memcpy", THREAD_COUNT );
        // AVX2를 사용한 memcpy 벤치마크
        benchmark_avx2_memcpy( dest2.get(), src.get(), size, "AVX2 memcpy", THREAD_COUNT );
        // AVX-512를 사용한 memcpy 벤치마크
        benchmark_avx512_memcpy( dest3.get(), src.get(), size, "AVX-512 memcpy", THREAD_COUNT );
        // SSE를 사용한 memcpy 벤치마크
        benchmark_sse_memcpy( dest4.get(), src.get(), size, "SSE memcpy", THREAD_COUNT );

		std::cout << std::endl;
    }

    return 0;
}