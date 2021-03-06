#ifndef __MV_DEGRAINS__
#define __MV_DEGRAINS__

#include <cstdint>
#include <cstring>

#include <emmintrin.h>
#include <immintrin.h>
#include <avx2intrin.h>

#include "MVFrame.h"

enum VectorOrder {
    Backward1 = 0,
    Forward1,
    Backward2,
    Forward2,
    Backward3,
    Forward3
};


typedef void (*DenoiseFunction)(uint8_t *pDst, int nDstPitch, const uint8_t *pSrc, int nSrcPitch, const uint8_t **_pRefs, const int *nRefPitches, int WSrc, const int *WRefs);


// XXX Moves the pointers passed in pRefs. This is okay because they are not
// used after this function is done with them.
template <int radius, int blockWidth, int blockHeight, typename PixelType>
void Degrain_C(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, const uint8_t **pRefs8, const int *nRefPitches, int WSrc, const int *WRefs) {
    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            PixelType *pDst = (PixelType *)pDst8;

            int sum = 128 + pSrc[x] * WSrc;

            for (int r = 0; r < radius*2; r++) {
                const PixelType *pRef = (const PixelType *)pRefs8[r];
                sum += pRef[x] * WRefs[r];
            }

            pDst[x] = sum >> 8;
        }

        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
        for (int r = 0; r < radius*2; r++)
            pRefs8[r] += nRefPitches[r];
    }
}


// XXX Moves the pointers passed in pRefs. This is okay because they are not
// used after this function is done with them.
template <int radius, int blockWidth, int blockHeight>
void Degrain_sse2(uint8_t *pDst, int nDstPitch, const uint8_t *pSrc, int nSrcPitch, const uint8_t **pRefs, const int *nRefPitches, int WSrc, const int *WRefs) {
    __m128i zero = _mm_setzero_si128();
    __m128i wsrc = _mm_set1_epi16(WSrc);
    __m128i wrefs[6];
    wrefs[0] = _mm_set1_epi16(WRefs[0]);
    wrefs[1] = _mm_set1_epi16(WRefs[1]);
    if (radius > 1) {
        wrefs[2] = _mm_set1_epi16(WRefs[2]);
        wrefs[3] = _mm_set1_epi16(WRefs[3]);
    }
    if (radius > 2) {
        wrefs[4] = _mm_set1_epi16(WRefs[4]);
        wrefs[5] = _mm_set1_epi16(WRefs[5]);
    }

    __m128i src, refs[6];

    for (int y = 0; y < blockHeight; y++) {
        for (int x = 0; x < blockWidth; x += 8) {
            // pDst[x] = (pRefF[x]*WRefF + pSrc[x]*WSrc + pRefB[x]*WRefB + pRefF2[x]*WRefF2 + pRefB2[x]*WRefB2 + pRefF3[x]*WRefF3 + pRefB3[x]*WRefB3 + 128)>>8;

            if (blockWidth == 4) {
                src   = _mm_cvtsi32_si128(*(const int *)pSrc);
                refs[0] = _mm_cvtsi32_si128(*(const int *)pRefs[0]);
                refs[1] = _mm_cvtsi32_si128(*(const int *)pRefs[1]);
                if (radius > 1) {
                    refs[2] = _mm_cvtsi32_si128(*(const int *)pRefs[2]);
                    refs[3] = _mm_cvtsi32_si128(*(const int *)pRefs[3]);
                }
                if (radius > 2) {
                    refs[4] = _mm_cvtsi32_si128(*(const int *)pRefs[4]);
                    refs[5] = _mm_cvtsi32_si128(*(const int *)pRefs[5]);
                }
            } else {
                src   = _mm_loadl_epi64((const __m128i *)(pSrc + x));
                refs[0] = _mm_loadl_epi64((const __m128i *)(pRefs[0] + x));
                refs[1] = _mm_loadl_epi64((const __m128i *)(pRefs[1] + x));
                if (radius > 1) {
                    refs[2] = _mm_loadl_epi64((const __m128i *)(pRefs[2] + x));
                    refs[3] = _mm_loadl_epi64((const __m128i *)(pRefs[3] + x));
                }
                if (radius > 2) {
                    refs[4] = _mm_loadl_epi64((const __m128i *)(pRefs[4] + x));
                    refs[5] = _mm_loadl_epi64((const __m128i *)(pRefs[5] + x));
                }
            }

            src   = _mm_unpacklo_epi8(src, zero);
            refs[0] = _mm_unpacklo_epi8(refs[0], zero);
            refs[1] = _mm_unpacklo_epi8(refs[1], zero);
            if (radius > 1) {
                refs[2] = _mm_unpacklo_epi8(refs[2], zero);
                refs[3] = _mm_unpacklo_epi8(refs[3], zero);
            }
            if (radius > 2) {
                refs[4] = _mm_unpacklo_epi8(refs[4], zero);
                refs[5] = _mm_unpacklo_epi8(refs[5], zero);
            }

            src   = _mm_mullo_epi16(src, wsrc);
            refs[0] = _mm_mullo_epi16(refs[0], wrefs[0]);
            refs[1] = _mm_mullo_epi16(refs[1], wrefs[1]);
            if (radius > 1) {
                refs[2] = _mm_mullo_epi16(refs[2], wrefs[2]);
                refs[3] = _mm_mullo_epi16(refs[3], wrefs[3]);
            }
            if (radius > 2) {
                refs[4] = _mm_mullo_epi16(refs[4], wrefs[4]);
                refs[5] = _mm_mullo_epi16(refs[5], wrefs[5]);
            }

            __m128i accum = _mm_set1_epi16(128);

            accum = _mm_add_epi16(accum, src);
            accum = _mm_add_epi16(accum, refs[0]);
            accum = _mm_add_epi16(accum, refs[1]);
            if (radius > 1) {
                accum = _mm_add_epi16(accum, refs[2]);
                accum = _mm_add_epi16(accum, refs[3]);
            }
            if (radius > 2) {
                accum = _mm_add_epi16(accum, refs[4]);
                accum = _mm_add_epi16(accum, refs[5]);
            }

            accum = _mm_srli_epi16(accum, 8);
            accum = _mm_packus_epi16(accum, zero);

            if (blockWidth == 4)
                *(int *)pDst = _mm_cvtsi128_si32(accum);
            else
                _mm_storel_epi64((__m128i *)(pDst + x), accum);
        }
        pDst += nDstPitch;
        pSrc += nSrcPitch;
        pRefs[0] += nRefPitches[0];
        pRefs[1] += nRefPitches[1];
        if (radius > 1) {
            pRefs[2] += nRefPitches[2];
            pRefs[3] += nRefPitches[3];
        }
        if (radius > 2) {
            pRefs[4] += nRefPitches[4];
            pRefs[5] += nRefPitches[5];
        }
    }
}

__forceinline __m256i _mm256_loadl_epi128(__m128i const * a)
{
    static __m128i const zero = _mm_setzero_si128();
    return _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128(a)), zero, 1);
}

template <int radius, int blockWidth, int blockHeight>
void Degrain_8to32xX_AVX2_16bit(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, const uint8_t **pRefs8, const int *nRefPitches, int WSrc, const int *WRefs)
{
	__m256i refs[6];
	__m256i wrefs[6];
	__m256i wSrcX8 = _mm256_set1_epi32(WSrc);

	uint16_t *pSrc16 = (uint16_t*)pSrc8;
	ptrdiff_t nSrcPitch16 = nSrcPitch / 2;
	uint16_t *pDst16 = (uint16_t*)pDst8;
	ptrdiff_t nDstPitch16 = nDstPitch / 2;

	uint16_t *pRefs16[6];
	ptrdiff_t nRefPitches16[6];

	pRefs16[0]       = (uint16_t*)pRefs8[0];
	nRefPitches16[0] = nRefPitches[0] / 2;
	wrefs[0]         = _mm256_set1_epi32(WRefs[0]);

	pRefs16[1] = (uint16_t*)pRefs8[1];
	nRefPitches16[1] = nRefPitches[1] / 2;
	wrefs[1]         = _mm256_set1_epi32(WRefs[1]);
	if(radius > 1)
	{
		pRefs16[2] = (uint16_t*)pRefs8[2];
		nRefPitches16[2] = nRefPitches[2] / 2;
		wrefs[2]         = _mm256_set1_epi32(WRefs[2]);

		pRefs16[3] = (uint16_t*)pRefs8[3];
		nRefPitches16[3] = nRefPitches[3] / 2;
		wrefs[3]         = _mm256_set1_epi32(WRefs[3]);
	}
	if(radius > 2)
	{
		pRefs16[4] = (uint16_t*)pRefs8[4];
		nRefPitches16[4] = nRefPitches[4] / 2;
		wrefs[4]         = _mm256_set1_epi32(WRefs[4]);

		pRefs16[5] = (uint16_t*)pRefs8[5];
		nRefPitches16[5] = nRefPitches[5] / 2;
		wrefs[5]         = _mm256_set1_epi32(WRefs[5]);
	}

	for(int y = 0; y < blockHeight; y++)
	{
		for(int x = 0; x < blockWidth; x += 8)
		{
			__m256i accum = _mm256_set1_epi32(128);

			__m256i src = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pSrc16 + x)));
			src         = _mm256_mullo_epi32(src, wSrcX8);
			accum       = _mm256_add_epi32(accum, src);

			refs[0] = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pRefs16[0] + x)));
			refs[0] = _mm256_mullo_epi32(refs[0], wrefs[0]);
			accum   = _mm256_add_epi32(accum, refs[0]);

			refs[1] = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pRefs16[1] + x)));
			refs[1] = _mm256_mullo_epi32(refs[1], wrefs[1]);
			accum   = _mm256_add_epi32(accum, refs[1]);

			if(radius == 2)
			{
				refs[2] = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pRefs16[2] + x)));
				refs[2] = _mm256_mullo_epi32(refs[2], wrefs[2]);
				accum   = _mm256_add_epi32(accum, refs[2]);

				refs[3] = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pRefs16[3] + x)));
				refs[3] = _mm256_mullo_epi32(refs[3], wrefs[3]);
				accum   = _mm256_add_epi32(accum, refs[3]);
			}
			if(radius == 3)
			{
				refs[4] = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pRefs16[4] + x)));
				refs[4] = _mm256_mullo_epi32(refs[4], wrefs[4]);
				accum   = _mm256_add_epi32(accum, refs[4]);

				refs[5] = _mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)(pRefs16[5] + x)));
				refs[5] = _mm256_mullo_epi32(refs[5], wrefs[5]);
				accum   = _mm256_add_epi32(accum, refs[5]);
			}

			accum = _mm256_srli_epi32(accum, 8);
			/*
			|X1|X2|X3|X4|X5|X6|X7|X8|
					   AND
			| 1| 2| 3| 4| 5| 6| 7| 8|
					 SHUFFLE
			| 1| 2| 3| 4| 5| 6| 7| 8|
			| 5| 6| 7| 8|  |  |  |  |
					 PACKUS
			|12|34|56|78|  |  |  |  |
			*/
            __m256i mask = _mm256_set1_epi32(0x0000FFFF);

            accum = _mm256_and_si256(accum, mask);
			__m256i shuffled = _mm256_permute4x64_epi64(accum, 0x0000000E);

			_mm_storeu_si128((__m128i *)(pDst16 + x), _mm256_extracti128_si256(_mm256_packus_epi32(accum, shuffled), 0));
		}

		pSrc16 += nSrcPitch16;
		pDst16 += nDstPitch16;
		pRefs16[0] += nRefPitches16[0];
        pRefs16[1] += nRefPitches16[1];
        if (radius > 1)
        {
            pRefs16[2] += nRefPitches16[2];
            pRefs16[3] += nRefPitches16[3];
        }
        if (radius > 2)
        {
            pRefs16[4] += nRefPitches16[4];
            pRefs16[5] += nRefPitches16[5];
		}
	}
}

//This version is actually slowe than the SSE 4.1 version, use that instead
#if 0
__inline __m128i _mm_loadu2_m64(__m64 *low, __m64 *high)
{
	__m128 a;
	
	a = _mm_loadl_pi(a, low);
	a = _mm_loadh_pi(a, high);
	
	return _mm_castps_si128(a);
}
template <int radius, int blockWidth, int blockHeight>
void Degrain_4xX_AVX2_16bit(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, const uint8_t **pRefs8, const int *nRefPitches, int WSrc, const int *WRefs)
{
	__m256i refs[6];
	__m256i wrefs[6];
	__m256i wSrcX8 = _mm256_set1_epi32(WSrc);

	uint16_t *pSrc16 = (uint16_t*)pSrc8;
	ptrdiff_t nSrcPitch16 = nSrcPitch / 2;
	uint16_t *pDst16 = (uint16_t*)pDst8;
	ptrdiff_t nDstPitch16 = nDstPitch / 2;

	uint16_t *pRefs16[6];
	ptrdiff_t nRefPitches16[6];

	pRefs16[0]       = (uint16_t*)pRefs8[0];
	nRefPitches16[0] = nRefPitches[0] / 2;
	wrefs[0]         = _mm256_set1_epi32(WRefs[0]);

	pRefs16[1] = (uint16_t*)pRefs8[1];
	nRefPitches16[1] = nRefPitches[1] / 2;
	wrefs[1]         = _mm256_set1_epi32(WRefs[1]);
	if(radius > 1)
	{
		pRefs16[2] = (uint16_t*)pRefs8[2];
		nRefPitches16[2] = nRefPitches[2] / 2;
		wrefs[2]         = _mm256_set1_epi32(WRefs[2]);

		pRefs16[3] = (uint16_t*)pRefs8[3];
		nRefPitches16[3] = nRefPitches[3] / 2;
		wrefs[3]         = _mm256_set1_epi32(WRefs[3]);
	}
	if(radius > 2)
	{
		pRefs16[4] = (uint16_t*)pRefs8[4];
		nRefPitches16[4] = nRefPitches[4] / 2;
		wrefs[4]         = _mm256_set1_epi32(WRefs[4]);

		pRefs16[5] = (uint16_t*)pRefs8[5];
		nRefPitches16[5] = nRefPitches[5] / 2;
		wrefs[5]         = _mm256_set1_epi32(WRefs[5]);
	}

	for(int y = 0; y < blockHeight; y += 2)
	{
		__m256i accum = _mm256_set1_epi32(128);

		__m256i src = _mm256_cvtepu16_epi32(_mm_loadu2_m64((__m64*)pSrc16,(__m64*)(pSrc16 + nSrcPitch16)));
		src         = _mm256_mullo_epi32(src, wSrcX8);
		accum       = _mm256_add_epi32(accum, src);

		refs[0] = _mm256_cvtepu16_epi32(_mm_loadu2_m64((__m64*)pRefs16[0],(__m64*)(pRefs16[0] + nRefPitches16[0])));
		refs[0] = _mm256_mullo_epi32(refs[0], wrefs[0]);
		accum   = _mm256_add_epi32(accum, refs[0]);

		refs[1] = _mm256_cvtepu16_epi32(_mm_loadu2_m64((__m64*)pRefs16[1],(__m64*)(pRefs16[1] + nRefPitches16[1])));
		refs[1] = _mm256_mullo_epi32(refs[1], wrefs[1]);
		accum   = _mm256_add_epi32(accum, refs[1]);

		if(radius == 2)
		{
			refs[2] = _mm256_cvtepu16_epi32(_mm_loadu2_m64((__m64*)pRefs16[2],(__m64*)(pRefs16[2] + nRefPitches16[2])));
			refs[2] = _mm256_mullo_epi32(refs[2], wrefs[2]);
			accum   = _mm256_add_epi32(accum, refs[2]);

			refs[3] = _mm256_cvtepu16_epi32(_mm_loadu2_m64((__m64*)pRefs16[3],(__m64*)(pRefs16[3] + nRefPitches16[3])));
			refs[3] = _mm256_mullo_epi32(refs[3], wrefs[3]);
			accum   = _mm256_add_epi32(accum, refs[3]);
		}
		if(radius == 3)
		{
			refs[4] = _mm256_cvtepu16_epi32(_mm_loadu2_m64((__m64*)pRefs16[4],(__m64*)(pRefs16[4] + nRefPitches16[4])));
			refs[4] = _mm256_mullo_epi32(refs[4], wrefs[4]);
			accum   = _mm256_add_epi32(accum, refs[4]);

			refs[5] = _mm256_cvtepu16_epi32(_mm_loadu2_m64((__m64*)pRefs16[5],(__m64*)(pRefs16[5] + nRefPitches16[5])));
			refs[5] = _mm256_mullo_epi32(refs[5], wrefs[5]);
			accum   = _mm256_add_epi32(accum, refs[5]);
		}

		accum = _mm256_srli_epi32(accum, 8);
		
		/*
		|X1|X2|X3|X4|X5|X6|X7|X8|
				   AND
		| 1| 2| 3| 4| 5| 6| 7| 8|
				 SHUFFLE
		| 1| 2| 3| 4| 5| 6| 7| 8|
		| 5| 6| 7| 8|  |  |  |  |
				 PACKUS
		|12|34|56|78|  |  |  |  |
		*/
		__m256i mask = _mm256_set1_epi32(0x0000FFFF);

		accum = _mm256_and_si256(accum, mask);
		__m256i shuffled = _mm256_permute4x64_epi64(accum, 0x0000000E);

		__m128 extracted = _mm_castsi128_ps(_mm256_extracti128_si256(_mm256_packus_epi32(accum, shuffled), 0));
		
		_mm_storel_pi((__m64*) pDst16               , extracted);
		_mm_storeh_pi((__m64*)(pDst16 + nDstPitch16), extracted);
		
		
		pSrc16 += nSrcPitch16 * 2;
		pDst16 += nDstPitch16 * 2;
		pRefs16[0] += nRefPitches16[0] * 2;
        pRefs16[1] += nRefPitches16[1] * 2;
        if (radius > 1)
        {
            pRefs16[2] += nRefPitches16[2] * 2;
            pRefs16[3] += nRefPitches16[3] * 2;
        }
        if (radius > 2)
        {
            pRefs16[4] += nRefPitches16[4] * 2;
            pRefs16[5] += nRefPitches16[5] * 2;
		}
	}
}
#endif

//In total it should have a latency of 9 cicles and a throughput of 3 cicles
//assuming the shift and unpack ports are different
__inline __m128i _mm_mullo_epi32_SSE2(__m128i a, __m128i b)
{
	return _mm_unpacklo_epi32(
						   	  _mm_shuffle_epi32(_mm_mul_epu32(a, b), 0x08),
						   	  _mm_shuffle_epi32(_mm_mul_epu32(
						   					                  _mm_srli_epi64(a, 32),
						   					                  _mm_srli_epi64(b, 32)), 0x08));
}
template <int radius, int blockWidth, int blockHeight>
void Degrain_4xX_SSE2_16bit(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, const uint8_t **pRefs8, const int *nRefPitches, int WSrc, const int *WRefs)
{
	__m128i refs[6];
	__m128i wrefs[6];
	__m128i wSrcX4 = _mm_set1_epi32(WSrc);

	uint16_t *pSrc16 = (uint16_t*)pSrc8;
	ptrdiff_t nSrcPitch16 = nSrcPitch / 2;
	uint16_t *pDst16 = (uint16_t*)pDst8;
	ptrdiff_t nDstPitch16 = nDstPitch / 2;

	uint16_t *pRefs16[6];
	ptrdiff_t nRefPitches16[6];

	pRefs16[0]       = (uint16_t*)pRefs8[0];
	nRefPitches16[0] = nRefPitches[0] / 2;
	wrefs[0]         = _mm_set1_epi32(WRefs[0]);

	pRefs16[1] = (uint16_t*)pRefs8[1];
	nRefPitches16[1] = nRefPitches[1] / 2;
	wrefs[1]         = _mm_set1_epi32(WRefs[1]);
	if(radius > 1)
	{
		pRefs16[2] = (uint16_t*)pRefs8[2];
		nRefPitches16[2] = nRefPitches[2] / 2;
		wrefs[2]         = _mm_set1_epi32(WRefs[2]);

		pRefs16[3] = (uint16_t*)pRefs8[3];
		nRefPitches16[3] = nRefPitches[3] / 2;
		wrefs[3]         = _mm_set1_epi32(WRefs[3]);
	}
	if(radius > 2)
	{
		pRefs16[4] = (uint16_t*)pRefs8[4];
		nRefPitches16[4] = nRefPitches[4] / 2;
		wrefs[4]         = _mm_set1_epi32(WRefs[4]);

		pRefs16[5] = (uint16_t*)pRefs8[5];
		nRefPitches16[5] = nRefPitches[5] / 2;
		wrefs[5]         = _mm_set1_epi32(WRefs[5]);
	}

	for(int y = 0; y < blockHeight; y++)
	{
		for(int x = 0; x < blockWidth; x += 4)
		{
			__m128i accum = _mm_set1_epi32(128);

			__m128i src = _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)(pSrc16 + x)), _mm_setzero_di());
			src         = _mm_mullo_epi32_SSE2(src, wSrcX4);
			accum       = _mm_add_epi32(accum, src);

			refs[0] = _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)(pRefs16[0] + x)), _mm_setzero_di());
			refs[0] = _mm_mullo_epi32_SSE2(refs[0], wrefs[0]);
			accum   = _mm_add_epi32(accum, refs[0]);

			refs[1] = _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)(pRefs16[1] + x)), _mm_setzero_di());
			refs[1] = _mm_mullo_epi32_SSE2(refs[1], wrefs[1]);
			accum   = _mm_add_epi32(accum, refs[1]);

			if(radius == 2)
			{
				refs[2] = _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)(pRefs16[2] + x)), _mm_setzero_di());
				refs[2] = _mm_mullo_epi32_SSE2(refs[2], wrefs[2]);
				accum   = _mm_add_epi32(accum, refs[2]);

				refs[3] = _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)(pRefs16[3] + x)), _mm_setzero_di());
				refs[3] = _mm_mullo_epi32_SSE2(refs[3], wrefs[3]);
				accum   = _mm_add_epi32(accum, refs[3]);
			}
			if(radius == 3)
			{
				refs[4] = _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)(pRefs16[4] + x)), _mm_setzero_di());
				refs[4] = _mm_mullo_epi32_SSE2(refs[4], wrefs[4]);
				accum   = _mm_add_epi32(accum, refs[4]);

				refs[5] = _mm_unpacklo_epi16(_mm_loadl_epi64((__m128i*)(pRefs16[5] + x)), _mm_setzero_di());
				refs[5] = _mm_mullo_epi32_SSE2(refs[5], wrefs[5]);
				accum   = _mm_add_epi32(accum, refs[5]);
			}

			accum = _mm_srli_epi32(accum, 8);

			__m128i mask = _mm_set1_epi32(0x0000FFFF);

			accum = _mm_and_si128(accum, mask);

			accum = _mm_shufflelo_epi16(accum, 0x8);
			accum = _mm_shufflehi_epi16(accum, 0x8);
			accum = _mm_shuffle_epi32(accum, 0x8);
			_mm_store_sd((double*)(pDst16 + x), _mm_castsi128_pd(accum));
		}

		pSrc16 += nSrcPitch16;
		pDst16 += nDstPitch16;
		pRefs16[0] += nRefPitches16[0];
        pRefs16[1] += nRefPitches16[1];
        if (radius > 1)
        {
            pRefs16[2] += nRefPitches16[2];
            pRefs16[3] += nRefPitches16[3];
        }
        if (radius > 2)
        {
            pRefs16[4] += nRefPitches16[4];
            pRefs16[5] += nRefPitches16[5];
		}
	}
}

template <int radius, int blockWidth, int blockHeight>
void Degrain_4xX_SSE41_16bit(uint8_t *pDst8, int nDstPitch, const uint8_t *pSrc8, int nSrcPitch, const uint8_t **pRefs8, const int *nRefPitches, int WSrc, const int *WRefs)
{
	__m128i refs[6];
	__m128i wrefs[6];
	__m128i wSrcX4 = _mm_set1_epi32(WSrc);

	uint16_t *pSrc16 = (uint16_t*)pSrc8;
	ptrdiff_t nSrcPitch16 = nSrcPitch / 2;
	uint16_t *pDst16 = (uint16_t*)pDst8;
	ptrdiff_t nDstPitch16 = nDstPitch / 2;

	uint16_t *pRefs16[6];
	ptrdiff_t nRefPitches16[6];

	pRefs16[0]       = (uint16_t*)pRefs8[0];
	nRefPitches16[0] = nRefPitches[0] / 2;
	wrefs[0]         = _mm_set1_epi32(WRefs[0]);

	pRefs16[1] = (uint16_t*)pRefs8[1];
	nRefPitches16[1] = nRefPitches[1] / 2;
	wrefs[1]         = _mm_set1_epi32(WRefs[1]);
	if(radius > 1)
	{
		pRefs16[2] = (uint16_t*)pRefs8[2];
		nRefPitches16[2] = nRefPitches[2] / 2;
		wrefs[2]         = _mm_set1_epi32(WRefs[2]);

		pRefs16[3] = (uint16_t*)pRefs8[3];
		nRefPitches16[3] = nRefPitches[3] / 2;
		wrefs[3]         = _mm_set1_epi32(WRefs[3]);
	}
	if(radius > 2)
	{
		pRefs16[4] = (uint16_t*)pRefs8[4];
		nRefPitches16[4] = nRefPitches[4] / 2;
		wrefs[4]         = _mm_set1_epi32(WRefs[4]);

		pRefs16[5] = (uint16_t*)pRefs8[5];
		nRefPitches16[5] = nRefPitches[5] / 2;
		wrefs[5]         = _mm_set1_epi32(WRefs[5]);
	}

	for(int y = 0; y < blockHeight; y++)
	{
		for(int x = 0; x < blockWidth; x += 4)
		{
			__m128i accum = _mm_set1_epi32(128);

			__m128i src = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(pSrc16 + x)));
			src         = _mm_mullo_epi32(src, wSrcX4);
			accum       = _mm_add_epi32(accum, src);

			refs[0] = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(pRefs16[0] + x)));
			refs[0] = _mm_mullo_epi32(refs[0], wrefs[0]);
			accum   = _mm_add_epi32(accum, refs[0]);

			refs[1] = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(pRefs16[1] + x)));
			refs[1] = _mm_mullo_epi32(refs[1], wrefs[1]);
			accum   = _mm_add_epi32(accum, refs[1]);

			if(radius == 2)
			{
				refs[2] = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(pRefs16[2] + x)));
				refs[2] = _mm_mullo_epi32(refs[2], wrefs[2]);
				accum   = _mm_add_epi32(accum, refs[2]);

				refs[3] = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(pRefs16[3] + x)));
				refs[3] = _mm_mullo_epi32(refs[3], wrefs[3]);
				accum   = _mm_add_epi32(accum, refs[3]);
			}
			if(radius == 3)
			{
				refs[4] = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(pRefs16[4] + x)));
				refs[4] = _mm_mullo_epi32(refs[4], wrefs[4]);
				accum   = _mm_add_epi32(accum, refs[4]);

				refs[5] = _mm_cvtepu16_epi32(_mm_loadl_epi64((__m128i*)(pRefs16[5] + x)));
				refs[5] = _mm_mullo_epi32(refs[5], wrefs[5]);
				accum   = _mm_add_epi32(accum, refs[5]);
			}

			accum = _mm_srli_epi32(accum, 8);

            __m128i mask = _mm_set1_epi32(0x0000FFFF);

			accum = _mm_and_si128(accum, mask);

			_mm_store_sd((double*)(pDst16 + x), _mm_castsi128_pd(_mm_packus_epi32(accum, _mm_setzero_si128())));
		}

		pSrc16 += nSrcPitch16;
		pDst16 += nDstPitch16;
		pRefs16[0] += nRefPitches16[0];
        pRefs16[1] += nRefPitches16[1];
        if (radius > 1)
        {
            pRefs16[2] += nRefPitches16[2];
            pRefs16[3] += nRefPitches16[3];
        }
        if (radius > 2)
        {
            pRefs16[4] += nRefPitches16[4];
            pRefs16[5] += nRefPitches16[5];
		}
	}
}

typedef void (*LimitFunction)(uint8_t *pDst, intptr_t nDstPitch, const uint8_t *pSrc, intptr_t nSrcPitch, intptr_t nWidth, intptr_t nHeight, intptr_t nLimit);


extern "C" void mvtools_LimitChanges_sse2(uint8_t *pDst, intptr_t nDstPitch, const uint8_t *pSrc, intptr_t nSrcPitch, intptr_t nWidth, intptr_t nHeight, intptr_t nLimit);


template <typename PixelType>
static void LimitChanges_C(uint8_t *pDst8, intptr_t nDstPitch, const uint8_t *pSrc8, intptr_t nSrcPitch, intptr_t nWidth, intptr_t nHeight, intptr_t nLimit) {
    for (int h = 0; h < nHeight; h++) {
        for (int i = 0; i < nWidth; i++) {
            const PixelType *pSrc = (const PixelType *)pSrc8;
            PixelType *pDst = (PixelType *)pDst8;

            pDst[i] = (PixelType)VSMIN(VSMAX(pDst[i], (pSrc[i] - nLimit)), (pSrc[i] + nLimit));
        }
        pDst8 += nDstPitch;
        pSrc8 += nSrcPitch;
    }
}


inline int DegrainWeight(int64_t thSAD, int64_t blockSAD) {
    if (blockSAD >= thSAD)
        return 0;

    return int((thSAD - blockSAD) * (thSAD + blockSAD) * 256 / (thSAD * thSAD + blockSAD * blockSAD));
}


inline void useBlock(const uint8_t * &p, int &np, int &WRef, bool isUsable, const MVClipBalls *mvclip, int i, const MVPlane *pPlane, const uint8_t **pSrcCur, int xx, const int *nSrcPitch, int nLogPel, int plane, int xSubUV, int ySubUV, const int *thSAD) {
    if (isUsable) {
        const FakeBlockData &block = mvclip->GetBlock(0, i);
        int blx = (block.GetX() << nLogPel) + block.GetMV().x;
        int bly = (block.GetY() << nLogPel) + block.GetMV().y;
        p = pPlane->GetPointer(plane ? blx >> xSubUV : blx, plane ? bly >> ySubUV : bly);
        np = pPlane->GetPitch();
        int blockSAD = block.GetSAD();
        WRef = DegrainWeight(thSAD[plane], blockSAD);
    } else {
        p = pSrcCur[plane] + xx;
        np = nSrcPitch[plane];
        WRef = 0;
    }
}


template <int radius>
static inline void normaliseWeights(int &WSrc, int *WRefs) {
    // normalise weights to 256
    WSrc = 256;
    int WSum = WSrc + 1;
    for (int r = 0; r < radius*2; r++)
        WSum += WRefs[r];

    for (int r = 0; r < radius*2; r++) {
        WRefs[r] = WRefs[r] * 256 / WSum;
        WSrc -= WRefs[r];
    }
}


#endif