## Initialization and Cleanup
- [ ] nvmlInit_v3

## System Queries
- [ ] nvmlSystemGetCudaDriverVersion
- [ ] nvmlSystemGetCudaDriverVersion_v2

## Device Queries
- [ ] nvmlDeviceGetBoardPartNumber
- [ ] nvmlDeviceGetClock
- [x] nvmlDeviceGetCudaComputeCapability
- [ ] nvmlDeviceGetEncoderCapacity
- [ ] nvmlDeviceGetEncoderSessions
- [ ] nvmlDeviceGetEncoderStats
- [ ] nvmlDeviceGetFBCSessions
- [ ] nvmlDeviceGetFBCStats
- [ ] nvmlDeviceGetMaxCustomerBoostClock
- [ ] nvmlDeviceGetP2PStatus
- [ ] nvmlDeviceGetTotalEnergyConsumption

## Device Commands
- [ ] nvmlDeviceResetGpuLockedClocks
- [ ] nvmlDeviceSetGpuLockedClocks

## NvLink Methods
- [ ] nvmlDeviceFreezeNvLinkUtilizationCounter
- [ ] nvmlDeviceGetNvLinkCapability
- [ ] nvmlDeviceGetNvLinkErrorCounter
- [ ] nvmlDeviceGetNvLinkRemotePciInfo
- [ ] nvmlDeviceGetNvLinkState
- [ ] nvmlDeviceGetNvLinkUtilizationControl
- [ ] nvmlDeviceGetNvLinkUtilizationCounter
- [ ] nvmlDeviceGetNvLinkVersion
- [ ] nvmlDeviceResetNvLinkErrorCounters
- [ ] nvmlDeviceResetNvLinkUtilizationCounter
- [ ] nvmlDeviceSetNvLinkUtilizationControl

## Drain states
- [ ] nvmlDeviceDiscoverGpus
- [ ] nvmlDeviceModifyDrainState
- [ ] nvmlDeviceQueryDrainState
- [ ] nvmlDeviceRemoveGpu 

## Field Value Queries
- [ ] nvmlDeviceGetFieldValues

## GRID Virtualization APIs 
- [ ] nvmlDeviceGetGridLicensableFeatures
- [ ] nvmlDeviceGetHostVgpuMode
- [ ] nvmlDeviceGetProcessUtilization
- [ ] nvmlDeviceGetVirtualizationMode
- [ ] nvmlDeviceSetVirtualizationMode

## GRID vGPU Management
- [ ] nvmlDeviceGetActiveVgpus
- [ ] nvmlDeviceGetCreatableVgpus
- [ ] nvmlDeviceGetSupportedVgpus
- [ ] nvmlVgpuInstanceGetEccMode
- [ ] nvmlVgpuInstanceGetEncoderCapacity
- [ ] nvmlVgpuInstanceGetEncoderSessions
- [ ] nvmlVgpuInstanceGetEncoderStats
- [ ] nvmlVgpuInstanceGetFBCSessions
- [ ] nvmlVgpuInstanceGetFBCStats
- [ ] nvmlVgpuInstanceGetFbUsage
- [ ] nvmlVgpuInstanceGetFrameRateLimit
- [ ] nvmlVgpuInstanceGetLicenseStatus
- [ ] nvmlVgpuInstanceGetType
- [ ] nvmlVgpuInstanceGetUUID
- [ ] nvmlVgpuInstanceGetVmDriverVersion
- [ ] nvmlVgpuInstanceGetVmID
- [ ] nvmlVgpuInstanceSetEncoderCapacity
- [ ] nvmlVgpuTypeGetClass
- [ ] nvmlVgpuTypeGetDeviceID
- [ ] nvmlVgpuTypeGetFrameRateLimit
- [ ] nvmlVgpuTypeGetFramebufferSize
- [ ] nvmlVgpuTypeGetLicense
- [ ] nvmlVgpuTypeGetMaxInstances
- [ ] nvmlVgpuTypeGetMaxInstancesPerVm
- [ ] nvmlVgpuTypeGetName
- [ ] nvmlVgpuTypeGetNumDisplayHeads
- [ ] nvmlVgpuTypeGetResolution
 
 ## GRID Virtualization Migration
 nvmlReturn_t nvmlDeviceGetPgpuMetadataString ( nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize )
nvmlReturn_t nvmlDeviceGetVgpuMetadata ( nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize )
nvmlReturn_t nvmlGetVgpuCompatibility ( nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo )
nvmlReturn_t nvmlGetVgpuVersion ( nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current )
nvmlReturn_t nvmlSetVgpuVersion ( nvmlVgpuVersion_t* vgpuVersion )
nvmlReturn_t nvmlVgpuInstanceGetMetadata ( nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize )

## GRID Virtualization Utilization and Accounting

nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization ( nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples )
nvmlReturn_t nvmlDeviceGetVgpuUtilization ( nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples )
nvmlReturn_t nvmlVgpuInstanceGetAccountingMode ( nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode )
nvmlReturn_t nvmlVgpuInstanceGetAccountingPids ( nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids )
nvmlReturn_t nvmlVgpuInstanceGetAccountingStats ( nvmlVgpuInstance_t vgpuInstance, unsigned int  pid, nvmlAccountingStats_t* stats ) 

## GPU Blacklist Queries
nvmlReturn_t nvmlGetBlacklistDeviceCount ( unsigned int* deviceCount )
nvmlReturn_t nvmlGetBlacklistDeviceInfoByIndex ( unsigned int  index, nvmlBlacklistDeviceInfo_t* info )

## Deprecated List
 
Class nvmlEccErrorCounts_t

    Different GPU families can have different memory error counters See nvmlDeviceGetMemoryErrorCounter

Global nvmlEccBitType_t

    See nvmlMemoryErrorType_t for a more flexible type

Global NVML_SINGLE_BIT_ECC

    Mapped to NVML_MEMORY_ERROR_TYPE_CORRECTED

Global NVML_DOUBLE_BIT_ECC

    Mapped to NVML_MEMORY_ERROR_TYPE_UNCORRECTED

Global nvmlDeviceGetHandleBySerial

    Since more than one GPU can exist on a single board this function is deprecated in favor of nvmlDeviceGetHandleByUUID. For dual GPU boards this function will return NVML_ERROR_INVALID_ARGUMENT.

Global nvmlDeviceGetDetailedEccErrors

    This API supports only a fixed set of ECC error locations On different GPU architectures different locations are supported See nvmlDeviceGetMemoryErrorCounter

Global nvmlClocksThrottleReasonUserDefinedClocks

    Renamed to nvmlClocksThrottleReasonApplicationsClocksSetting as the name describes the situation more accurately.

## missing constants

#define nvmlClocksThrottleReasonAll
#define nvmlClocksThrottleReasonApplicationsClocksSetting 0x0000000000000002LL
#define nvmlClocksThrottleReasonDisplayClockSetting 0x0000000000000100LL
#define nvmlClocksThrottleReasonGpuIdle 0x0000000000000001LL
#define nvmlClocksThrottleReasonHwPowerBrakeSlowdown 0x0000000000000080LL
#define nvmlClocksThrottleReasonHwSlowdown 0x0000000000000008LL
#define nvmlClocksThrottleReasonHwThermalSlowdown 0x0000000000000040LL
#define nvmlClocksThrottleReasonNone 0x0000000000000000LL
#define nvmlClocksThrottleReasonSwPowerCap 0x0000000000000004LL
#define nvmlClocksThrottleReasonSwThermalSlowdown 0x0000000000000020LL
#define nvmlClocksThrottleReasonSyncBoost 0x0000000000000010LL
#define nvmlClocksThrottleReasonUserDefinedClocks



_nvml(.*)_t

$1.C_TYPE

_nvml(.*)_t = c_uint
class $1(UIntEnum):