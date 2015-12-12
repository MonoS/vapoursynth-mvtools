#ifndef __OVERLAP__
#define __OVERLAP__

#include <math.h>
#include <stdint.h>
#ifndef M_PI
#define M_PI       3.14159265358979323846f
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b)            (((a) < (b)) ? (b) : (a))
#endif

// top, middle, botom and left, middle, right windows
#define OW_TL 0
#define OW_TM 1
#define OW_TR 2
#define OW_ML 3
#define OW_MM 4
#define OW_MR 5
#define OW_BL 6
#define OW_BM 7
#define OW_BR 8

class OverlapWindows
{
    int nx; // window sizes
    int ny;
    int ox; // overap sizes
    int oy;
    int size; // full window size= nx*ny

    short * Overlap9Windows;

    float *fWin1UVx;
    float *fWin1UVxfirst;
    float *fWin1UVxlast;
    float *fWin1UVy;
    float *fWin1UVyfirst;
    float *fWin1UVylast;
    public :

    OverlapWindows(int _nx, int _ny, int _ox, int _oy);
    ~OverlapWindows();

    inline int Getnx() const { return nx; }
    inline int Getny() const { return ny; }
    inline int GetSize() const { return size; }
    inline short *GetWindow(int i) const { return Overlap9Windows + size*i; }
};

typedef void (*OverlapsFunction)(uint8_t *pDst, intptr_t nDstPitch,
        const uint8_t *pSrc, intptr_t nSrcPitch,
        short *pWin, intptr_t nWinPitch);


template <int blockWidth, int blockHeight, typename PixelType2, typename PixelType>
void Overlaps_C(uint8_t *pDst8, intptr_t nDstPitch, const uint8_t *pSrc8, intptr_t nSrcPitch, short *pWin, intptr_t nWinPitch)
{
    // pWin from 0 to 2048
    for (int j=0; j<blockHeight; j++)
    {
        for (int i=0; i<blockWidth; i++)
        {
            PixelType2 *pDst = (PixelType2 *)pDst8;
            const PixelType *pSrc = (const PixelType *)pSrc8;

            pDst[i] += ((pSrc[i] * pWin[i]) >> 6);
        }
        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        pWin += nWinPitch;
    }
}

#define MK_CFUNC(functionname) extern "C" void functionname (uint8_t *pDst, intptr_t nDstPitch, const uint8_t *pSrc, intptr_t nSrcPitch, short *pWin, intptr_t nWinPitch)

MK_CFUNC(mvtools_Overlaps2x2_sse2);
MK_CFUNC(mvtools_Overlaps2x4_sse2);
MK_CFUNC(mvtools_Overlaps4x2_sse2);
MK_CFUNC(mvtools_Overlaps4x4_sse2);
MK_CFUNC(mvtools_Overlaps4x8_sse2);
MK_CFUNC(mvtools_Overlaps8x1_sse2);
MK_CFUNC(mvtools_Overlaps8x2_sse2);
MK_CFUNC(mvtools_Overlaps8x4_sse2);
MK_CFUNC(mvtools_Overlaps8x8_sse2);
MK_CFUNC(mvtools_Overlaps8x16_sse2);
MK_CFUNC(mvtools_Overlaps16x1_sse2);
MK_CFUNC(mvtools_Overlaps16x2_sse2);
MK_CFUNC(mvtools_Overlaps16x4_sse2);
MK_CFUNC(mvtools_Overlaps16x8_sse2);
MK_CFUNC(mvtools_Overlaps16x16_sse2);
MK_CFUNC(mvtools_Overlaps16x32_sse2);
MK_CFUNC(mvtools_Overlaps32x8_sse2);
MK_CFUNC(mvtools_Overlaps32x16_sse2);
MK_CFUNC(mvtools_Overlaps32x32_sse2);

#undef MK_CFUNC


typedef void (*ToPixelsFunction)(uint8_t *pDst, int nDstPitch,
        const uint8_t *pSrc, int nSrcPitch,
        int width, int height, int bitsPerSample);


template <typename PixelType2, typename PixelType>
void ToPixels(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, int nWidth, int nHeight, int bitsPerSample)
{
    int pixelMax = (1 << bitsPerSample) - 1;

    for (int h=0; h<nHeight; h++)
    {
        for (int i=0; i<nWidth; i++)
        {
            const PixelType2 *pSrc = (const PixelType2 *)pSrc8;
            PixelType *pDst = (PixelType *)pDst8;

            int a = (pSrc[i] + 16)>>5;
            if (sizeof(PixelType) == 1)
                pDst[i] = a | ((255-a) >> (sizeof(int)*8-1));
            else
                pDst[i] = min(pixelMax, a);
        }
        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
    }
}

__inline __m256i _mm256_loadu2_m128i(__m128i *low, __m128i *high)
{
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((__m128i*)low)),
                                                          _mm_loadu_si128((__m128i*)high),1);

}

void ToPixels_AVX2_16bit(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, int nWidth, int nHeight, int bitsPerSample)
{
	static __m256i const sixteen  = _mm256_set1_epi32(16);
	static __m128i const five     = _mm_set1_epi64x(5);
    for (int h=0; h<nHeight; h++)
    {
		const uint32_t *pSrc = (const uint32_t *)pSrc8;
        uint16_t *pDst = (uint16_t *)pDst8;

        nWidth_16 = (nWidth & ~15);
        for (int i = 0; i < nWidth_16; i += 16)
        {
            __m256i src_1 = _mm256_loadu2_m128i((__m128i* )(pSrc + i), (__m128i* )(pSrc + i + 8));
            __m256i src_2 = _mm256_loadu2_m128i((__m128i* )(pSrc + i + 4), (__m128i* )(pSrc + i + 12));

			__m256i a1 = _mm256_srl_epi32(_mm256_add_epi32(src_1, sixteen), five);
			__m256i a2 = _mm256_srl_epi32(_mm256_add_epi32(src_2, sixteen), five);
			_mm256_storeu_si256((__m256i* )(pDst + i), _mm256_packus_epi32(a1, a2));

        }

        for (int i = nWidth_16; nWidth_16 < nWidth; i++)
        {
            int a = (pSrc[i] + 16)>>5;
            pDst[i] = min(pixelMax, a);
        }
        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
    }
}

void ToPixels_SSE2_16bit(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, int nWidth, int nHeight, int bitsPerSample)
{
	static __m128i const sixteen  = _mm_set1_epi32(16);
	static __m128i const five     = _mm_set1_epi64x(5);
    for (int h=0; h<nHeight; h++)
    {
		const uint32_t *pSrc = (const uint32_t *)pSrc8;
        uint16_t *pDst = (uint16_t *)pDst8;

        nWidth_8 = (nWidth & ~7);
        for (int i = 0; i < nWidth_8; i += 8)
        {
            __m128i src_1 = _mm_lddqu_si128((__m128i* )(pSrc + i));
            __m128i src_2 = _mm_lddqu_si128((__m128i* )(pSrc + i + 4));

			__m128i a1 = _mm_srl_epi32(_mm_add_epi32(src_1, sixteen), five);
			__m128i a2 = _mm_srl_epi32(_mm_add_epi32(src_2, sixteen), five);
			_mm_storeu_si128((__m128i* )(pDst + i), _mm_packus_epi32(a1, a2));
        }

        for (int i = nWidth_8; nWidth_8 < nWidth; i++)
        {
            int a = (pSrc[i] + 16)>>5;
            pDst[i] = min(pixelMax, a);
        }
        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
    }
}

#endif
