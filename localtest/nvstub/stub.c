/* No-GPU stubs for the NVIDIA libraries the Pandia WebRTC binary links against
 * (BIND_NOW needs every symbol at load time). Every call reports "no device"
 * (CUDA_ERROR_NO_DEVICE = 100 / NV_ENC_ERR_NO_ENCODE_DEVICE = 1), so the
 * binary must run with NVENC/NVDEC disabled (software codecs). */
#define STUB(name) int name() { return 100; }
#ifdef LIBCUDA
STUB(cuInit) STUB(cuDeviceGet) STUB(cuDeviceGetName) STUB(cuGetErrorName)
STUB(cuCtxCreate_v2) STUB(cuCtxPopCurrent_v2) STUB(cuCtxPushCurrent_v2)
STUB(cuMemAllocPitch_v2) STUB(cuMemAlloc_v2) STUB(cuMemFree_v2)
STUB(cuMemcpy2DAsync_v2) STUB(cuMemcpy2DUnaligned_v2) STUB(cuMemcpy2D_v2)
STUB(cuStreamCreate) STUB(cuStreamSynchronize)
int cuDeviceGetCount(int *n) { if (n) *n = 0; return 100; }
#endif
#ifdef LIBENC
int NvEncodeAPICreateInstance() { return 1; }
int NvEncodeAPIGetMaxSupportedVersion(unsigned *v) { if (v) *v = 0; return 1; }
#endif
#ifdef LIBCUVID
STUB(cuvidCreateDecoder) STUB(cuvidCreateVideoParser) STUB(cuvidCtxLockCreate)
STUB(cuvidCtxLockDestroy) STUB(cuvidDecodePicture) STUB(cuvidDestroyDecoder)
STUB(cuvidDestroyVideoParser) STUB(cuvidGetDecodeStatus) STUB(cuvidGetDecoderCaps)
STUB(cuvidMapVideoFrame64) STUB(cuvidParseVideoData) STUB(cuvidReconfigureDecoder)
STUB(cuvidUnmapVideoFrame64)
#endif
