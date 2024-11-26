#include <iostream>
#include <chrono>
#include <array>
#include <fstream>
#include <cstring>
#include <immintrin.h>
#include <vector>
#include <memory>
#include <iomanip>
#include <thread>
#include <intrin.h>

constexpr size_t TEST_CASE{ 100 };
size_t EACH_SIZE{};
size_t THREAD_COUNT{ std::thread::hardware_concurrency() / 2 };

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


std::atomic< unsigned long long > totalNs{};


auto Out( std::chrono::nanoseconds result, std::string_view label )
{
    using namespace std::chrono;

    /// 시간 단위에 맞춰 duration을 출력
    if ( duration_cast<seconds>( result ).count() > 100 )
    {
        auto duration = duration_cast<seconds>( result ).count();
        std::cout << label << "                   " << duration << " sec" << std::endl;
        return std::format( "{} sec", duration );
    }
    if ( duration_cast<milliseconds>( result ).count() > 100 )
    {
        auto duration = duration_cast<milliseconds>( result ).count();
        std::cout << label << "                   " << duration << " ms" << std::endl;
        return std::format( "{} ms", duration );
    }
    if ( duration_cast<microseconds>( result ).count() > 100 )
    {
        auto duration = duration_cast<microseconds>( result ).count();
        std::cout << label << "                   " << duration << " us" << std::endl;
        return std::format( "{} us", duration );
    }
    if ( duration_cast<nanoseconds>( result ).count() > 100 )
    {
        auto duration = duration_cast<nanoseconds>( result ).count();
        std::cout << label << "                   " << duration << " ns" << std::endl;
        return std::format( "{} ns", duration );
    }
}

template< std::invocable TTask >
auto DoTestInternal( TTask&& task, std::string_view label )
{
    totalNs = 0;

    using namespace std::chrono;

    auto start{ high_resolution_clock::now() };
    task();
    auto end{ high_resolution_clock::now() };
    auto result{ end - start };
    totalNs += result.count();
    return Out( totalNs.load() * 1ns, label );
}


template< std::invocable TTask >
auto DoTestUsingMultiThread( TTask&& task, std::string_view label, int threadCount )
{
    using namespace std::chrono;

    std::vector< std::thread > threads;
    threads.reserve( threadCount );

    totalNs = 0;

    for ( int i = 0; i < threadCount; ++i )
        threads.emplace_back( [ task, i ]()
            {
                auto start{ high_resolution_clock::now() };
                task();
                auto end{ high_resolution_clock::now() };
                auto result{ end - start };
                totalNs += result.count();
            } );

    for ( auto& thread : threads )
        thread.join();

    return Out( totalNs.load() * 1ns, label );
}

template< std::invocable TTask >
auto DoTestOnThisThread( TTask&& task, std::string_view label )
{
    return DoTestInternal( task, label );
}

template< std::invocable TTask >
auto DoTest( TTask&& task, std::string_view label, int threadCount )
{
    if ( threadCount == 1 )
        return DoTestOnThisThread( task, label );
    else
        return DoTestUsingMultiThread( task, label, threadCount );
}

auto benchmark_memcpy( void* dest, const void* src, size_t size, std::string_view label, int threadCount = 1 )
{
    return DoTest( [ dest, src, size ]
        {
            for ( int i = 0; i < TEST_CASE; ++i )
            {
                std::memcpy( dest, src, size );
            }
        },
        label,
        threadCount );
}

auto benchmark_avx2_memcpy( void* dest, const void* src, size_t size, std::string_view label, int threadCount = 1 )
{
    return DoTest( [ dest, src, size ]
        {
            for ( int i = 0; i < TEST_CASE; ++i )
            {
                avx2_memcpy( dest, src, size );
            }
        },
        label,
        threadCount );
}

auto benchmark_avx512_memcpy( unsigned char* dest, const unsigned char* src, size_t size, std::string_view label, int threadCount = 1 )
{
    return DoTest( [ dest, src, size ]
        {
            for ( int i = 0; i < TEST_CASE; ++i )
            {
                avx512_memcpy( dest, src, size );
            }
        },
        label,
        threadCount );
}

auto benchmark_sse_memcpy( void* dest, const void* src, size_t size, std::string_view label, int threadCount = 1 )
{
    return DoTest( [ dest, src, size ]
        {
            for ( int i = 0; i < TEST_CASE; ++i )
            {
                sse_memcpy( dest, src, size );
            }
        },
        label,
        threadCount );
}

std::string ToSizeFormatString( size_t size )
{
    if ( size < 1024 )
        return std::format( "{} bytes", size );
    else if ( size < 1024 * 1024 )
        return std::format( "{:.2f} KB", static_cast<double>( size ) / 1024 );
    else if ( size < 1024 * 1024 * 1024 )
        return std::format( "{:.2f} MB", static_cast<double>( size ) / ( 1024 * 1024 ) );
    else
        return std::format( "{:.2f} GB", static_cast<double>( size ) / ( 1024 * 1024 * 1024 ) );
}

void WriteFile( const std::vector< std::string >& vec, const std::string& filename )
{
    auto outFile{ std::ofstream( filename ) };
    if ( !outFile )
    {
        std::cerr << "파일을 열 수 없습니다: " << filename << std::endl;
        return;
    }

    for ( const auto& str : vec )
    {
        outFile << str << std::endl;
    }

    outFile.close();
}

// 문자열의 왼쪽 공백을 제거하는 함수
void ltrim( std::string& str )
{
    str.erase( 0, str.find_first_not_of( " \t\n\r\f\v" ) );
}

// 문자열의 오른쪽 공백을 제거하는 함수
void rtrim( std::string& str )
{
    str.erase( str.find_last_not_of( " \t\n\r\f\v" ) + 1 );
}

// 문자열의 양쪽 공백을 제거하는 함수
void trim( std::string& str )
{
    ltrim( str );
    rtrim( str );
}


std::pair< std::string, std::string > GetCPUInfo()
{
    std::array<int, 4> cpui{};
    std::array<int, 4> cpuiExt{};
    char vendor[ 0x20 ];
    char brand[ 0x40 ];

    // CPU Vendor
    __cpuid( cpui.data(), 0 );
    *reinterpret_cast<int*>( vendor ) = cpui[ 1 ];
    *reinterpret_cast<int*>( vendor + 4 ) = cpui[ 3 ];
    *reinterpret_cast<int*>( vendor + 8 ) = cpui[ 2 ];
    vendor[ 12 ] = '\0';

    // CPU Brand
    __cpuid( cpui.data(), 0x80000000 );
    int nExIds = cpui[ 0 ];
    if ( nExIds >= 0x80000004 )
    {
        memset( brand, 0, sizeof( brand ) );
        for ( int i = 0x80000002; i <= 0x80000004; ++i ) {
            __cpuid( cpuiExt.data(), i );
            memcpy( brand + ( i - 0x80000002 ) * 16, cpuiExt.data(), sizeof( cpuiExt ) );
        }
    }

    std::string vendorStr{ vendor };
    std::string brandStr{ brand };

    trim( vendorStr );
    trim( brandStr );

    return { std::move( vendorStr ), std::move( brandStr ) };
}

inline thread_local std::unique_ptr< unsigned char[] > src;
inline thread_local std::unique_ptr< char[] > dest1;
inline thread_local std::unique_ptr< char[] > dest2;
inline thread_local std::unique_ptr< unsigned char[] > dest3;
inline thread_local std::unique_ptr< char[] > dest4;

int main()
{
    auto cpuInfo = GetCPUInfo();

    std::vector< std::string > csvResult
    {
        //std::format( "THREAD {}, N {}, {}, {}", THREAD_COUNT, TEST_CASE, cpuInfo.first, cpuInfo.second ),
        "size,fmt size,std::memcpy,AVX2 mempcy,AVX512 mempcy,SSE mempcy"
    };

    std::vector< size_t > sizes;
    for ( size_t size = 16; size <= 1024 * 1024 * 100; size *= 2 )
        sizes.push_back( size + ( rand() % size ) );

    for ( const auto& size : sizes )
    {
        std::string result{ std::to_string( size ) + "," };

        EACH_SIZE = size;

        src = std::make_unique< unsigned char[] >( size );
        dest1 = std::make_unique< char[] >( size );
        dest2 = std::make_unique< char[] >( size );
        dest3 = std::make_unique< unsigned char[] >( size );
        dest4 = std::make_unique< char[] >( size );

        std::cout << std::format( "memcpy size {}\n", ToSizeFormatString( size ) );
        result += std::format( "{},", ToSizeFormatString( size ) );
        // 일반 memcpy 벤치마크
        result += std::format( "{},", benchmark_memcpy( dest1.get(), src.get(), size, "std::memcpy", THREAD_COUNT ) );
        // AVX2를 사용한 memcpy 벤치마크
        result += std::format( "{},", benchmark_avx2_memcpy( dest2.get(), src.get(), size, "AVX2 memcpy", THREAD_COUNT ) );
        // AVX-512를 사용한 memcpy 벤치마크
        result += std::format( "{},", benchmark_avx512_memcpy( dest3.get(), src.get(), size, "AVX-512 memcpy", THREAD_COUNT ) );
        // SSE를 사용한 memcpy 벤치마크
        result += std::format( "{}", benchmark_sse_memcpy( dest4.get(), src.get(), size, "SSE memcpy", THREAD_COUNT ) );

        csvResult.emplace_back( std::move( result ) );

        std::cout << std::endl;
    }

    WriteFile( csvResult, std::format( "Result_THREAD{}_N{}_{}.csv", THREAD_COUNT, TEST_CASE, cpuInfo.second ) );

    return 0;
}