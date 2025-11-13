/*
 * input_gen_tcn_129.h
 *
 * Code generation for model "input_gen_tcn_129".
 *
 * Model version              : 7.10
 * Simulink Coder version : 25.2 (R2025b) 28-Jul-2025
 * C source code generated on : Thu Nov 13 16:58:10 2025
 *
 * Target selection: grt.tlc
 * Note: GRT includes extra infrastructure and instrumentation for prototyping
 * Embedded hardware selection: Intel->x86-64 (Windows64)
 * Code generation objectives: Unspecified
 * Validation result: Not run
 */

#ifndef input_gen_tcn_129_h_
#define input_gen_tcn_129_h_
#ifndef input_gen_tcn_129_COMMON_INCLUDES_
#define input_gen_tcn_129_COMMON_INCLUDES_
#include <stdlib.h>
#include "rtwtypes.h"
#include "rtw_continuous.h"
#include "rtw_solver.h"
#include "rt_logging.h"
#include "rt_nonfinite.h"
#include "math.h"
#endif                                 /* input_gen_tcn_129_COMMON_INCLUDES_ */

#include "input_gen_tcn_129_types.h"
#include <float.h>
#include <stddef.h>
#include <string.h>

/* Macros for accessing real-time model data structure */
#ifndef rtmGetBlockIO
#define rtmGetBlockIO(rtm)             ((rtm)->blockIO)
#endif

#ifndef rtmSetBlockIO
#define rtmSetBlockIO(rtm, val)        ((rtm)->blockIO = (val))
#endif

#ifndef rtmGetFinalTime
#define rtmGetFinalTime(rtm)           ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetRTWLogInfo
#define rtmGetRTWLogInfo(rtm)          ((rtm)->rtwLogInfo)
#endif

#ifndef rtmGetRootDWork
#define rtmGetRootDWork(rtm)           ((rtm)->dwork)
#endif

#ifndef rtmSetRootDWork
#define rtmSetRootDWork(rtm, val)      ((rtm)->dwork = (val))
#endif

#ifndef rtmGetStepSize
#define rtmGetStepSize(rtm)            ((rtm)->Timing.stepSize)
#endif

#ifndef rtmGetU
#define rtmGetU(rtm)                   ((rtm)->inputs)
#endif

#ifndef rtmSetU
#define rtmSetU(rtm, val)              ((rtm)->inputs = (val))
#endif

#ifndef rtmGetY
#define rtmGetY(rtm)                   ((rtm)->outputs)
#endif

#ifndef rtmSetY
#define rtmSetY(rtm, val)              ((rtm)->outputs = (val))
#endif

#ifndef rtmGetErrorStatus
#define rtmGetErrorStatus(rtm)         ((rtm)->errorStatus)
#endif

#ifndef rtmSetErrorStatus
#define rtmSetErrorStatus(rtm, val)    ((rtm)->errorStatus = (val))
#endif

#ifndef rtmGetStopRequested
#define rtmGetStopRequested(rtm)       ((rtm)->Timing.stopRequestedFlag)
#endif

#ifndef rtmSetStopRequested
#define rtmSetStopRequested(rtm, val)  ((rtm)->Timing.stopRequestedFlag = (val))
#endif

#ifndef rtmGetStopRequestedPtr
#define rtmGetStopRequestedPtr(rtm)    (&((rtm)->Timing.stopRequestedFlag))
#endif

#ifndef rtmGetT
#define rtmGetT(rtm)                   ((rtm)->Timing.taskTime0)
#endif

#ifndef rtmGetTFinal
#define rtmGetTFinal(rtm)              ((rtm)->Timing.tFinal)
#endif

#ifndef rtmGetTPtr
#define rtmGetTPtr(rtm)                (&(rtm)->Timing.taskTime0)
#endif

#define input_gen_tcn_129_M_TYPE       RT_MODEL_input_gen_tcn_129_T

/* Block signals (default storage) */
typedef struct {
  real32_T weightsNumeric[16384];
  real32_T weightsPermuted[16384];
  real32_T weightsNumeric_m[16384];
  real32_T weightsPermuted_c[16384];
  real32_T weightsNumeric_k[16384];
  real32_T weightsPermuted_cx[16384];
  real32_T weightsNumeric_b[16384];
  real32_T weightsPermuted_p[16384];
  real32_T weightsNumeric_c[16384];
  real32_T weightsPermuted_f[16384];
  real32_T weightsNumeric_g[16384];
  real32_T weightsPermuted_g[16384];
  real32_T weightsNumeric_me[16384];
  real32_T weightsPermuted_n[16384];
  real32_T fv[4736];
  real32_T fv1[4736];
  real32_T x_tcn_network_ne_51[4736];
  real32_T x_tcn_network_ne_59[4736];
  real32_T fv2[3968];
  real32_T fv3[3968];
  real32_T x_tcn_network_ne_35[3968];
  real32_T x_tcn_network_ne_43[3968];
  real32_T fv4[3584];
  real32_T fv5[3584];
  real32_T x_tcn_network_ne_19[3584];
  real32_T x_tcn_network_ne_27[3584];
  real32_T fv6[3392];
  real32_T fv7[3392];
  real32_T x_tcn_network_ne_3[3392];
  real32_T x_tcn_network_ne_11[3392];
  real32_T outT_f11_0_f1[3200];
  real32_T fv8[3200];
  real32_T fv9[3200];
  real32_T fv10[3200];
  real32_T objdata[3200];
  real32_T b_X[3200];
  real32_T objdata_p[3200];
  real32_T objdata_l[3200];
  real32_T objdata_j[3200];
  real32_T objdata_d[3200];
  real32_T objdata_g[3200];
  real32_T objdata_ld[3200];
  real32_T weightsNumeric_d[3072];
  real32_T weightsPermuted_d[3072];
} B_input_gen_tcn_129_T;

/* Block states (default storage) for system '<Root>' */
typedef struct {
  real32_T Delay1_DSTATE[600];         /* '<Root>/Delay1' */
  boolean_T icLoad;                    /* '<Root>/Delay1' */
} DW_input_gen_tcn_129_T;

/* External inputs (root inport signals with default storage) */
typedef struct {
  real32_T In1[12];                    /* '<Root>/In1' */
} ExtU_input_gen_tcn_129_T;

/* External outputs (root outports fed by signals with default storage) */
typedef struct {
  real32_T Out1;                       /* '<Root>/Out1' */
} ExtY_input_gen_tcn_129_T;

/* Real-time Model Data Structure */
struct tag_RTM_input_gen_tcn_129_T {
  const char_T *errorStatus;
  RTWLogInfo *rtwLogInfo;
  B_input_gen_tcn_129_T *blockIO;
  ExtU_input_gen_tcn_129_T *inputs;
  ExtY_input_gen_tcn_129_T *outputs;
  DW_input_gen_tcn_129_T *dwork;

  /*
   * Timing:
   * The following substructure contains information regarding
   * the timing information for the model.
   */
  struct {
    time_T stepSize;
    time_T taskTime0;
    uint32_T clockTick0;
    uint32_T clockTickH0;
    time_T stepSize0;
    time_T tFinal;
    boolean_T stopRequestedFlag;
  } Timing;
};

/* External data declarations for dependent source files */
extern const char_T *RT_MEMORY_ALLOCATION_ERROR;

/* Model entry point functions */
extern RT_MODEL_input_gen_tcn_129_T *input_gen_tcn_129(void);
extern void input_gen_tcn_129_initialize(RT_MODEL_input_gen_tcn_129_T *const
  input_gen_tcn_129_M);
extern void input_gen_tcn_129_step(RT_MODEL_input_gen_tcn_129_T *const
  input_gen_tcn_129_M);
extern void input_gen_tcn_129_terminate(RT_MODEL_input_gen_tcn_129_T
  * input_gen_tcn_129_M);

/*-
 * These blocks were eliminated from the model due to optimizations:
 *
 * Block '<Root>/Data Type Conversion' : Eliminate redundant data type conversion
 */

/*-
 * The generated code includes comments that allow you to trace directly
 * back to the appropriate location in the model.  The basic format
 * is <system>/block_name, where system is the system number (uniquely
 * assigned by Simulink) and block_name is the name of the block.
 *
 * Use the MATLAB hilite_system command to trace the generated code back
 * to the model.  For example,
 *
 * hilite_system('<S3>')    - opens system 3
 * hilite_system('<S3>/Kp') - opens and selects block Kp which resides in S3
 *
 * Here is the system hierarchy for this model
 *
 * '<Root>' : 'input_gen_tcn_129'
 * '<S1>'   : 'input_gen_tcn_129/MATLAB Function1'
 * '<S2>'   : 'input_gen_tcn_129/Predict'
 * '<S3>'   : 'input_gen_tcn_129/Predict/MLFB'
 */
#endif                                 /* input_gen_tcn_129_h_ */
