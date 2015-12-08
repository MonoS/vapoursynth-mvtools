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

template <int radius, int blockWidth, int blockHeight>
void Degrain_AVX2(uint8_t *pDst, int nDstPitch, const uint8_t *pSrc, int nSrcPitch, const uint8_t **pRefs, const int *nRefPitches, int WSrc, const int *WRefs)
{
    __m256i zero = _mm256_set1_epi32(0);
    __m128i zero128 = _mm_setzero_si128();

	__m256i wsrc = _mm256_set1_epi32(WSrc);

    __m256i wrefs[6];
    wrefs[0] = _mm256_set1_epi32(WRefs[0]);
    wrefs[1] = _mm256_set1_epi32(WRefs[1]);

    if (radius > 1)
    {
        wrefs[2] = _mm256_set1_epi32(WRefs[2]);
        wrefs[3] = _mm256_set1_epi32(WRefs[3]);
    }
    if (radius > 2)
    {
        wrefs[4] = _mm256_set1_epi32(WRefs[4]);
        wrefs[5] = _mm256_set1_epi32(WRefs[5]);
    }
    __m256i src, refs[6];

    for (int y = 0; y < blockHeight; y++)
    {
        for (int x = 0; x < blockWidth; x += 8)
        {
            if (blockWidth == 4)
            {
                src     = _mm256_broadcastsi128_si256(_mm_cvtsi64_si128(*(const int *)pSrc));
                refs[0] = _mm256_broadcastsi128_si256(_mm_cvtsi64_si128(*(const int *)pRefs[0]));
                refs[1] = _mm256_broadcastsi128_si256(_mm_cvtsi64_si128(*(const int *)pRefs[1]));
                if (radius > 1)
                {
                    refs[2] = _mm256_broadcastsi128_si256(_mm_cvtsi64_si128(*(const int *)pRefs[2]));
                    refs[3] = _mm256_broadcastsi128_si256(_mm_cvtsi64_si128(*(const int *)pRefs[3]));
                }
                if (radius > 2)
                {
                    refs[4] = _mm256_broadcastsi128_si256(_mm_cvtsi64_si128(*(const int *)pRefs[4]));
                    refs[5] = _mm256_broadcastsi128_si256(_mm_cvtsi64_si128(*(const int *)pRefs[5]));
                }
            }
            else
            {
                src     = _mm256_loadl_epi128((const __m128i *)(pSrc + x));
                refs[0] = _mm256_loadl_epi128((const __m128i *)(pRefs[0] + x));
                refs[1] = _mm256_loadl_epi128((const __m128i *)(pRefs[1] + x));
                if (radius > 1)
                {
                    refs[2] = _mm256_loadl_epi128((const __m128i *)(pRefs[2] + x));
                    refs[3] = _mm256_loadl_epi128((const __m128i *)(pRefs[3] + x));
                }
                if (radius > 2)
                {
                    refs[4] = _mm256_loadl_epi128((const __m128i *)(pRefs[4] + x));
                    refs[5] = _mm256_loadl_epi128((const __m128i *)(pRefs[5] + x));
                }
            }


            src     = _mm256_unpacklo_epi16(src, zero);
            refs[0] = _mm256_unpacklo_epi16(refs[0], zero);
            refs[1] = _mm256_unpacklo_epi16(refs[1], zero);
            if (radius > 1)
            {
                refs[2] = _mm256_unpacklo_epi16(refs[2], zero);
                refs[3] = _mm256_unpacklo_epi16(refs[3], zero);
            }
            if (radius > 2)
            {
                refs[4] = _mm256_unpacklo_epi16(refs[4], zero);
                refs[5] = _mm256_unpacklo_epi16(refs[5], zero);
            }


            src = _mm256_mullo_epi32(src, wsrc);
            refs[0] = _mm256_mullo_epi32(refs[0], wrefs[0]);
            refs[1] = _mm256_mullo_epi32(refs[1], wrefs[1]);
            if (radius > 1)
            {
                refs[2] = _mm256_mullo_epi32(refs[2], wrefs[2]);
                refs[3] = _mm256_mullo_epi32(refs[3], wrefs[3]);
            }
            if (radius > 2)
            {
                refs[4] = _mm256_mullo_epi32(refs[4], wrefs[4]);
                refs[5] = _mm256_mullo_epi32(refs[5], wrefs[5]);
            }

            __m256i accum = _mm256_set1_epi32(128);
            accum = _mm256_add_epi32(accum, src);
            accum = _mm256_add_epi32(accum, refs[0]);
            accum = _mm256_add_epi32(accum, refs[1]);
            if (radius > 1)
            {
                accum = _mm256_add_epi32(accum, refs[2]);
                accum = _mm256_add_epi32(accum, refs[3]);
            }
            if (radius > 2)
            {
                accum = _mm256_add_epi32(accum, refs[4]);
                accum = _mm256_add_epi32(accum, refs[5]);
            }

            accum = _mm256_srli_epi32(accum, 8);
            accum = _mm_packus_epi16(accum, zero);
            if (blockWidth == 4)
                *(int64_t *)pDst = _mm256_extract_epi64(accum, 0);
            else
                *(__m128i *)(pDst + x) = _mm256_extracti128_si256(accum, 0);
        }

        pDst += nDstPitch;
        pSrc += nSrcPitch;
        pRefs[0] += nRefPitches[0];
        pRefs[1] += nRefPitches[1];
        if (radius > 1)
        {
            pRefs[2] += nRefPitches[2];
            pRefs[3] += nRefPitches[3];
        }
        if (radius > 2)
        {
            pRefs[4] += nRefPitches[4];
            pRefs[5] += nRefPitches[5];
        }
    }
}

template <int radius>
void Degrain_AVX2_2x2(uint8_t *pDst, int nDstPitch, const uint8_t *pSrc, int nSrcPitch, const uint8_t **pRefs, const int *nRefPitches, int WSrc, const int *WRefs)
{
    __m256i zero = _mm256_set1_epi32(0);
    __m128i vindex = _mm_set_epi32(0, 1, 0, 1);
	__m256i wsrc = _mm256_set1_epi32(WSrc);

    __m256i wrefs[6];
    wrefs[0] = _mm256_set1_epi32(WRefs[0]);
    wrefs[1] = _mm256_set1_epi32(WRefs[1]);

    if (radius > 1)
    {
        wrefs[2] = _mm256_set1_epi32(WRefs[2]);
        wrefs[3] = _mm256_set1_epi32(WRefs[3]);
    }
    if (radius > 2)
    {
        wrefs[4] = _mm256_set1_epi32(WRefs[4]);
        wrefs[5] = _mm256_set1_epi32(WRefs[5]);
    }
    __m256i src, refs[6];

    src     = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pSrc    , vindex, nSrcPitch));
    refs[0] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[0], vindex, nRefPitches[0]));
    refs[1] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[1], vindex, nRefPitches[1]));
    if (radius > 1)
	{
        refs[2] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[2], vindex, nRefPitches[2]));
        refs[3] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[3], vindex, nRefPitches[3]));
    }
    if (radius > 2)
    {
        refs[4] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[4], vindex, nRefPitches[4]));
        refs[5] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[5], vindex, nRefPitches[5]));
    }

    src     = _mm256_unpacklo_epi16(src, zero);
    refs[0] = _mm256_unpacklo_epi16(refs[0], zero);
    refs[1] = _mm256_unpacklo_epi16(refs[1], zero);
    if (radius > 1)
    {
        refs[2] = _mm256_unpacklo_epi16(refs[2], zero);
        refs[3] = _mm256_unpacklo_epi16(refs[3], zero);
    }
    if (radius > 2)
    {
        refs[4] = _mm256_unpacklo_epi16(refs[4], zero);
        refs[5] = _mm256_unpacklo_epi16(refs[5], zero);
    }


    src = _mm256_mullo_epi32(src, wsrc);
    refs[0] = _mm256_mullo_epi32(refs[0], wrefs[0]);
    refs[1] = _mm256_mullo_epi32(refs[1], wrefs[1]);
    if (radius > 1)
    {
        refs[2] = _mm256_mullo_epi32(refs[2], wrefs[2]);
        refs[3] = _mm256_mullo_epi32(refs[3], wrefs[3]);
    }
    if (radius > 2)
    {
        refs[4] = _mm256_mullo_epi32(refs[4], wrefs[4]);
        refs[5] = _mm256_mullo_epi32(refs[5], wrefs[5]);
    }

    __m256i accum = _mm256_set1_epi32(128);
    accum = _mm256_add_epi32(accum, src);
    accum = _mm256_add_epi32(accum, refs[0]);
    accum = _mm256_add_epi32(accum, refs[1]);
    if (radius > 1)
    {
        accum = _mm256_add_epi32(accum, refs[2]);
        accum = _mm256_add_epi32(accum, refs[3]);
    }
    if (radius > 2)
    {
        accum = _mm256_add_epi32(accum, refs[4]);
        accum = _mm256_add_epi32(accum, refs[5]);
    }

    accum = _mm256_srli_epi32(accum, 8);
    accum = _mm_packus_epi16(accum, zero);

    *(int *)pDst = _mm256_extract_epi32(accum, 0);
    *(int *)(pDst + nDstPitch) = _mm256_extract_epi32(accum, 1);
}

template <int radius>
void Degrain_AVX2_2x4(uint8_t *pDst, int nDstPitch, const uint8_t *pSrc, int nSrcPitch, const uint8_t **pRefs, const int *nRefPitches, int WSrc, const int *WRefs)
{
    __m256i zero = _mm256_set1_epi32(0);
    __m128i vindex = _mm_set_epi32(0, 1, 2, 3);
	__m256i wsrc = _mm256_set1_epi32(WSrc);

    __m256i wrefs[6];
    wrefs[0] = _mm256_set1_epi32(WRefs[0]);
    wrefs[1] = _mm256_set1_epi32(WRefs[1]);

    if (radius > 1)
    {
        wrefs[2] = _mm256_set1_epi32(WRefs[2]);
        wrefs[3] = _mm256_set1_epi32(WRefs[3]);
    }
    if (radius > 2)
    {
        wrefs[4] = _mm256_set1_epi32(WRefs[4]);
        wrefs[5] = _mm256_set1_epi32(WRefs[5]);
    }
    __m256i src, refs[6];

    src     = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pSrc    , vindex, nSrcPitch));
    refs[0] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[0], vindex, nRefPitches[0]));
    refs[1] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[1], vindex, nRefPitches[1]));
    if (radius > 1)
	{
        refs[2] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[2], vindex, nRefPitches[2]));
        refs[3] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[3], vindex, nRefPitches[3]));
    }
    if (radius > 2)
    {
        refs[4] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[4], vindex, nRefPitches[4]));
        refs[5] = _mm256_broadcastsi128_si256(_mm_i32gather_epi32(*(const int *)pRefs[5], vindex, nRefPitches[5]));
    }

    src     = _mm256_unpacklo_epi16(src, zero);
    refs[0] = _mm256_unpacklo_epi16(refs[0], zero);
    refs[1] = _mm256_unpacklo_epi16(refs[1], zero);
    if (radius > 1)
    {
        refs[2] = _mm256_unpacklo_epi16(refs[2], zero);
        refs[3] = _mm256_unpacklo_epi16(refs[3], zero);
    }
    if (radius > 2)
    {
        refs[4] = _mm256_unpacklo_epi16(refs[4], zero);
        refs[5] = _mm256_unpacklo_epi16(refs[5], zero);
    }


    src = _mm256_mullo_epi32(src, wsrc);
    refs[0] = _mm256_mullo_epi32(refs[0], wrefs[0]);
    refs[1] = _mm256_mullo_epi32(refs[1], wrefs[1]);
    if (radius > 1)
    {
        refs[2] = _mm256_mullo_epi32(refs[2], wrefs[2]);
        refs[3] = _mm256_mullo_epi32(refs[3], wrefs[3]);
    }
    if (radius > 2)
    {
        refs[4] = _mm256_mullo_epi32(refs[4], wrefs[4]);
        refs[5] = _mm256_mullo_epi32(refs[5], wrefs[5]);
    }

    __m256i accum = _mm256_set1_epi32(128);
    accum = _mm256_add_epi32(accum, src);
    accum = _mm256_add_epi32(accum, refs[0]);
    accum = _mm256_add_epi32(accum, refs[1]);
    if (radius > 1)
    {
        accum = _mm256_add_epi32(accum, refs[2]);
        accum = _mm256_add_epi32(accum, refs[3]);
    }
    if (radius > 2)
    {
        accum = _mm256_add_epi32(accum, refs[4]);
        accum = _mm256_add_epi32(accum, refs[5]);
    }

    accum = _mm256_srli_epi32(accum, 8);
    accum = _mm_packus_epi16(accum, zero);

    *(int *)pDst = _mm256_extract_epi32(accum, 0);
    *(int *)(pDst + nDstPitch) = _mm256_extract_epi32(accum, 1);
    *(int *)(pDst + nDstPitch* 2) = _mm256_extract_epi32(accum, 2);
    *(int *)(pDst + nDstPitch* 3) = _mm256_extract_epi32(accum, 3);
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
