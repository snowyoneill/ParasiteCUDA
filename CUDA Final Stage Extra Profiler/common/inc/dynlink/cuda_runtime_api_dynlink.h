/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#if !defined(__CUDA_RUNTIME_API_DYNLINK_H__)
#define __CUDA_RUNTIME_API_DYNLINK_H__

/*******************************************************************************
*                                                                              *
* CUDA runtime API version number 2.1                                          *
*                                                                              *
*******************************************************************************/
 
#define CUDART_VERSION \
        2010

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "host_defines.h"
#include "builtin_types.h"

#if !defined(__dv)

#if defined(__cplusplus)

#define __dv(v) \
        = v

#else /* __cplusplus */

#define __dv(v)

#endif /* __cplusplus */

#endif /* !__dv */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#if defined(__cplusplus)

namespace dyn
{

extern "C" {
#endif /* __cplusplus */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaMalloc3D(struct cudaPitchedPtr* pitchDevPtr, struct cudaExtent extent);
typedef __host__ cudaError_t CUDARTAPI tcudaMalloc3DArray(struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent);
typedef __host__ cudaError_t CUDARTAPI tcudaMemset3D(struct cudaPitchedPtr pitchDevPtr, int value, struct cudaExtent extent);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy3D(const struct cudaMemcpy3DParms *p);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p, cudaStream_t stream);


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaMalloc(void **devPtr, size_t size);
typedef __host__ cudaError_t CUDARTAPI tcudaMallocHost(void **ptr, size_t size);
typedef __host__ cudaError_t CUDARTAPI tcudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height);
typedef __host__ cudaError_t CUDARTAPI tcudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc, size_t width, size_t height /*__dv(1)*/);
typedef __host__ cudaError_t CUDARTAPI tcudaFree(void *devPtr);
typedef __host__ cudaError_t CUDARTAPI tcudaFreeHost(void *ptr);
typedef __host__ cudaError_t CUDARTAPI tcudaFreeArray(struct cudaArray *array);


/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyFromArray(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind /*kind __dv(cudaMemcpyDeviceToDevice)*/);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy2DFromArray(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy2DArrayToArray(struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind /*kind __dv(cudaMemcpyDeviceToDevice)*/);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyToSymbol(const char *symbol, const void *src, size_t count, size_t offset /*__dv(0)*/, enum cudaMemcpyKind kind /*__dv(cudaMemcpyHostToDevice)*/);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyFromSymbol(void *dst, const char *symbol, size_t count, size_t offset /*__dv(0)*/, enum cudaMemcpyKind kind /*__dv(cudaMemcpyDeviceToHost)*/);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyFromArrayAsync(void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy2DToArrayAsync(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyToSymbolAsync(const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaMemcpyFromSymbolAsync(void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaMemset(void *mem, int c, size_t count);
typedef __host__ cudaError_t CUDARTAPI tcudaMemset2D(void *mem, size_t pitch, int c, size_t width, size_t height);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaGetSymbolAddress(void **devPtr, const char *symbol);
typedef __host__ cudaError_t CUDARTAPI tcudaGetSymbolSize(size_t *size, const char *symbol);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaGetDeviceCount(int *count);
typedef __host__ cudaError_t CUDARTAPI tcudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
typedef __host__ cudaError_t CUDARTAPI tcudaChooseDevice(int *device, const struct cudaDeviceProp *prop);
typedef __host__ cudaError_t CUDARTAPI tcudaSetDevice(int device);
typedef __host__ cudaError_t CUDARTAPI tcudaGetDevice(int *device);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaBindTexture(size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size /*__dv(UINT_MAX)*/);
typedef __host__ cudaError_t CUDARTAPI tcudaBindTextureToArray(const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc);
typedef __host__ cudaError_t CUDARTAPI tcudaUnbindTexture(const struct textureReference *texref);
typedef __host__ cudaError_t CUDARTAPI tcudaGetTextureAlignmentOffset(size_t *offset, const struct textureReference *texref);
typedef __host__ cudaError_t CUDARTAPI tcudaGetTextureReference(const struct textureReference **texref, const char *symbol);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array);
typedef __host__ struct cudaChannelFormatDesc CUDARTAPI tcudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaGetLastError(void);
typedef __host__ const char* CUDARTAPI tcudaGetErrorString(cudaError_t error);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem /*__dv(0)*/, cudaStream_t stream /*__dv(0)*/);
typedef __host__ cudaError_t CUDARTAPI tcudaSetupArgument(const void *arg, size_t size, size_t offset);
typedef __host__ cudaError_t CUDARTAPI tcudaLaunch(const char *symbol);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaStreamCreate(cudaStream_t *stream);
typedef __host__ cudaError_t CUDARTAPI tcudaStreamDestroy(cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaStreamSynchronize(cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaStreamQuery(cudaStream_t stream);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaEventCreate(cudaEvent_t *event);
typedef __host__ cudaError_t CUDARTAPI tcudaEventRecord(cudaEvent_t event, cudaStream_t stream);
typedef __host__ cudaError_t CUDARTAPI tcudaEventQuery(cudaEvent_t event);
typedef __host__ cudaError_t CUDARTAPI tcudaEventSynchronize(cudaEvent_t event);
typedef __host__ cudaError_t CUDARTAPI tcudaEventDestroy(cudaEvent_t event);
typedef __host__ cudaError_t CUDARTAPI tcudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaSetDoubleForDevice(double *d);
typedef __host__ cudaError_t CUDARTAPI tcudaSetDoubleForHost(double *d);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __host__ cudaError_t CUDARTAPI tcudaThreadExit(void);
typedef __host__ cudaError_t CUDARTAPI tcudaThreadSynchronize(void);

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

extern tcudaMalloc3D                   *cudaMalloc3D;
extern tcudaMalloc3DArray              *cudaMalloc3DArray;
extern tcudaMemset3D                   *cudaMemset3D;
extern tcudaMemcpy3D                   *cudaMemcpy3D;
extern tcudaMemcpy3DAsync              *cudaMemcpy3DAsync;
extern tcudaMalloc                     *cudaMalloc;
extern tcudaMallocHost                 *cudaMallocHost;
extern tcudaMallocPitch                *cudaMallocPitch;
extern tcudaMallocArray                *cudaMallocArray;
extern tcudaFree                       *cudaFree;
extern tcudaFreeHost                   *cudaFreeHost;
extern tcudaFreeArray                  *cudaFreeArray;
extern tcudaMemcpy                     *cudaMemcpy;
extern tcudaMemcpyToArray              *cudaMemcpyToArray;
extern tcudaMemcpyFromArray            *cudaMemcpyFromArray;
extern tcudaMemcpyArrayToArray         *cudaMemcpyArrayToArray;
extern tcudaMemcpy2D                   *cudaMemcpy2D;
extern tcudaMemcpy2DToArray            *cudaMemcpy2DToArray;
extern tcudaMemcpy2DFromArray          *cudaMemcpy2DFromArray;
extern tcudaMemcpy2DArrayToArray       *cudaMemcpy2DArrayToArray;
extern tcudaMemcpyToSymbol             *cudaMemcpyToSymbol;
extern tcudaMemcpyFromSymbol           *cudaMemcpyFromSymbol;
extern tcudaMemcpyAsync                *cudaMemcpyAsync;
extern tcudaMemcpyToArrayAsync         *cudaMemcpyToArrayAsync;
extern tcudaMemcpyFromArrayAsync       *cudaMemcpyFromArrayAsync;
extern tcudaMemcpy2DAsync              *cudaMemcpy2DAsync;
extern tcudaMemcpy2DToArrayAsync       *cudaMemcpy2DToArrayAsync;
extern tcudaMemcpy2DFromArrayAsync     *cudaMemcpy2DFromArrayAsync;
extern tcudaMemcpyToSymbolAsync        *cudaMemcpyToSymbolAsync;
extern tcudaMemcpyFromSymbolAsync      *cudaMemcpyFromSymbolAsync;
extern tcudaMemset                     *cudaMemset;
extern tcudaMemset2D                   *cudaMemset2D;
extern tcudaGetSymbolAddress           *cudaGetSymbolAddress;
extern tcudaGetSymbolSize              *cudaGetSymbolSize;
extern tcudaGetDeviceCount             *cudaGetDeviceCount;
extern tcudaGetDeviceProperties        *cudaGetDeviceProperties;
extern tcudaChooseDevice               *cudaChooseDevice;
extern tcudaSetDevice                  *cudaSetDevice;
extern tcudaGetDevice                  *cudaGetDevice;
extern tcudaBindTexture                *cudaBindTexture;
extern tcudaBindTextureToArray         *cudaBindTextureToArray;
extern tcudaUnbindTexture              *cudaUnbindTexture;
extern tcudaGetTextureAlignmentOffset  *cudaGetTextureAlignmentOffset;
extern tcudaGetTextureReference        *cudaGetTextureReference;
extern tcudaGetChannelDesc             *cudaGetChannelDesc;
extern tcudaCreateChannelDesc          *cudaCreateChannelDesc;
extern tcudaGetLastError               *cudaGetLastError;
extern tcudaGetErrorString             *cudaGetErrorString;
extern tcudaConfigureCall              *cudaConfigureCall;
extern tcudaSetupArgument              *cudaSetupArgument;
extern tcudaLaunch                     *cudaLaunch;
extern tcudaStreamCreate               *cudaStreamCreate;
extern tcudaStreamDestroy              *cudaStreamDestroy;
extern tcudaStreamSynchronize          *cudaStreamSynchronize;
extern tcudaStreamQuery                *cudaStreamQuery;
extern tcudaEventCreate                *cudaEventCreate;
extern tcudaEventRecord                *cudaEventRecord;
extern tcudaEventQuery                 *cudaEventQuery;
extern tcudaEventSynchronize           *cudaEventSynchronize;
extern tcudaEventDestroy               *cudaEventDestroy;
extern tcudaEventElapsedTime           *cudaEventElapsedTime;
extern tcudaSetDoubleForDevice         *cudaSetDoubleForDevice;
extern tcudaSetDoubleForHost           *cudaSetDoubleForHost;
extern tcudaThreadExit                 *cudaThreadExit;
extern tcudaThreadSynchronize          *cudaThreadSynchronize;

extern __host__ cudaError_t CUDARTAPI cudaRuntimeDynload(void);


#if defined(__cplusplus)

} /* namespace dyn */

}
#endif /* __cplusplus */

#undef __dv

#endif /* !__CUDA_RUNTIME_API_DYNLINK_H__ */
