/*
 * @file inscan.c
 * @brief Implementation of collective sum_inscan operation
 *
 */

#include "shcoll.h"
#include <shmem/api_types.h>
#include "util/bithacks.h"
#include "../tests/util/debug.h"

#include "shmem.h"

#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>


/*
   @brief Helper macro to define linear sum_inscan operations
 * essentially the naive implementation of the sum_inscan collective
 *
 * Implements a linear sum_inscan algorithm where each PE gets and adds its required
 * information
 *
 * @param _name Typename of sum_inscan operation
 * @param _type Data type to operate on
 */
#define SUM_INSCAN_HELPER_LINEAR(_name, _type)                                     \
  int sum_inscan_helper_##_name##_linear(                                          \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * workBuffer = (_type *)shmem_calloc(nelems, sizeof(_type));           \
  if (workBuffer == NULL)                                                      \
    return 1;                                                                  \
  /* If dest and source are the same, need a buffer to store intermediate */   \
  /* results */                                                                \
  _type * result;                                                              \
  if ( (void *) dest == (void *) source){                                      \
    result = (_type *)calloc(nelems, sizeof(_type));                           \
    if (result == NULL){                                                       \
      shmem_free(workBuffer);                                                  \
      return 1;                                                                \
    }                                                                          \
  }                                                                            \
  else {                                                                       \
    /* zero out dest array */                                                  \
    memset(dest, 0, nelems * sizeof(_type));                                   \
    result = (_type *) dest;                                                   \
  }                                                                            \
                                                                               \
  shcoll_barrier_linear(PE_start, logPE_stride, PE_size, pSync);               \
  /* Do insum scan */                                                          \
  for (int pe = 0; pe <= me_as; pe++){                                         \
    int src_pe = shmemc_team_pe_to_world(team, pe);                            \
    shmem_getmem(workBuffer, source, nelems * sizeof(_type), src_pe);          \
    /* add results */                                                          \
    for (size_t i = 0; i < nelems; i ++){                                      \
      result[i] = result[i] + workBuffer[i];                                   \
    }                                                                          \
  }                                                                            \
  shcoll_barrier_linear(PE_start, logPE_stride, PE_size, pSync);               \
                                                                               \
  /* If dest and source are same, override array with results */               \
  if ( (void *) dest == (void *) source){                                      \
    for (int i = 0; i < nelems; i ++)                                          \
      dest[i] = result[i];                                                     \
    free(result);                                                              \
  }                                                                            \
                                                                               \
  shmem_free(workBuffer);                                                      \
  return 0;                                                                    \
}

/* Implement the ring algorithm here */                                        
#define SUM_INSCAN_HELPER_RING(_name, _type)                                       \
  int sum_inscan_helper_##_name##_ring(                                            \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * workBuffer;                                                          \
  if ((void *) dest == (void *) source){                                       \
    workBuffer = (_type *)shmem_malloc(nelems * sizeof(_type));                \
    if (workBuffer == NULL)                                                    \
      return 1;                                                                \
  } else {                                                                     \
    /* if different src / dst arrays, save memory and use dest */              \
    /* as our work buffer */                                                   \
    workBuffer = dest;                                                         \
  }                                                                            \
  uint64_t * received = shmem_calloc(1,sizeof(uint64_t));                      \
  if (received == NULL){                                                       \
    shmem_free(workBuffer);                                                    \
    return 1;                                                                  \
  }                                                                            \
  if (me_as == 0 && dest != source)                                            \
    memcpy(dest, source, nelems * sizeof(_type));                              \
                                                                               \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
                                                                               \
  /* do shmem_sum_inscan */                                                        \
  /* wait on results and take sum (except for first PE) */                     \
  if (me_as != 0){                                                             \
    shmem_wait_until(received, 1, SHMEM_CMP_EQ);                               \
    for (size_t i = 0; i < nelems; i ++)                                       \
      dest[i] = workBuffer[i] + source[i];                                     \
  }                                                                            \
  if (me_as != PE_size - 1){                                                   \
    /* put results to next PE in line (except for last PE) */                  \
    int dest_pe = shmemc_team_pe_to_world(team, me_as + 1);                    \
    shmem_put8_signal(workBuffer, dest, nelems * sizeof(_type), received, 1,   \
        SHMEM_SIGNAL_SET, dest_pe);                                            \
  }                                                                            \
                                                                               \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
                                                                               \
  if ((void *) dest == (void *) source)                                        \
    shmem_free(workBuffer);                                                    \
  shmem_free(received);                                                        \
  return 0;                                                                    \
}

/* based off of Hillis and Steele's algorithm  */
#define SUM_INSCAN_HELPER_LOGARITHMIC(_name, _type)                                \
  int sum_inscan_helper_##_name##_logarithmic(                                     \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * workBuffer = (_type *)shmem_malloc(nelems * sizeof(_type));          \
  if (workBuffer == NULL)                                                      \
    return 1;                                                                  \
  /* If not src / dst not same array, copy over elements */                    \
  if ( (void *) source != (void *) dest){                                      \
    memcpy(dest, source, nelems * sizeof(_type));                              \
  }                                                                            \
                                                                               \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
  for (int stride = 1; stride <= PE_size; stride <<= 1){                       \
    if (me_as >= stride){                                                      \
      int target = shmemc_team_pe_to_world(team, me_as - stride);              \
      shmem_getmem(workBuffer, dest, nelems * sizeof(_type), target);          \
    }                                                                          \
    /* need two barriers to prevent race conditions */                         \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
    if (me_as >= stride){                                                      \
      for (int elem = 0; elem < nelems; elem ++){                              \
        dest[elem] += workBuffer[elem];                                        \
      }                                                                        \
    }                                                                          \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
  }                                                                            \
                                                                               \
  shmem_free(workBuffer);                                                      \
  return 0;                                                                    \
}

#define SUM_INSCAN_HELPER_RECURSIVE_DOUBLING(_name, _type)                         \
  int sum_inscan_helper_##_name##_recursive_doubling(                              \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * partialResBuf = (_type *) shmem_malloc(nelems * sizeof(_type));     \
  if (partialResBuf == NULL)                                                   \
    return 1;                                                                  \
  _type * recvBuf = (_type *) shmem_malloc(nelems * sizeof(_type));           \
  if (recvBuf == NULL){                                                        \
    shmem_free(partialResBuf);                                                 \
    return 1;                                                                  \
  }                                                                            \
  memcpy(partialResBuf, source, nelems * sizeof(_type)); \
  memcpy(dest, source, nelems * sizeof(_type)); \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
                                                                               \
  for (int stride = 1; stride < PE_size; stride <<= 1){                        \
    int target = stride ^ me_as;                                               \
    if (target < PE_size){                                                     \
      int world_target = shmemc_team_pe_to_world(team, target);                \
      shmem_getmem(recvBuf, partialResBuf, nelems * sizeof(_type),             \
          world_target);                                                       \
      shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);    \
      if (me_as > target){                                                     \
        for (size_t i = 0; i < nelems; i ++)                                   \
          partialResBuf[i] = partialResBuf[i] + recvBuf[i];                    \
        for (size_t i = 0; i < nelems; i ++)                                   \
          dest[i] = dest[i] + recvBuf[i];                    \
      } else {                                                                 \
        for (size_t i = 0; i < nelems; i ++)                                   \
          partialResBuf[i] = partialResBuf[i] + recvBuf[i];                    \
      }                                                                        \
    }                                                                          \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
  }                                                                            \
  shmem_free(partialResBuf);                                                   \
  shmem_free(recvBuf);                                                         \
  return 0;                                                                    \
}

#define DECLARE_SUM_INSCAN_HELPER(_type, _typename)                               \
  SUM_INSCAN_HELPER_LINEAR(_typename, _type) \
  SUM_INSCAN_HELPER_RING(_typename, _type) \
  SUM_INSCAN_HELPER_LOGARITHMIC(_typename, _type) \
  SUM_INSCAN_HELPER_RECURSIVE_DOUBLING(_typename, _type)

SHMEM_REDUCE_ARITH_TYPE_TABLE(DECLARE_SUM_INSCAN_HELPER)
#undef DECLARE_SUM_INSCAN_HELPER

/*
   @brief Helper macro to define linear sum_exscan operations
 *
 * Implements a linear sum_exscan algorithm where each PE gets and adds its required
 * information
 *
 * @param _name Typename of sum_exscan operation
 * @param _type Data type to operate on
 */
#define SUM_EXSCAN_HELPER_LINEAR(_name, _type)                                     \
  int sum_exscan_helper_##_name##_linear(                                          \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * workBuffer = (_type *)shmem_calloc(nelems, sizeof(_type));           \
  if (workBuffer == NULL)                                                      \
    return 1;                                                                  \
  /* If dest and source are the same, need a buffer to store intermediate */   \
  /* results */                                                                \
  _type * result;                                                              \
  if ( (void *) dest == (void *) source){                                      \
    result = (_type *)calloc(nelems, sizeof(_type));                           \
    if (result == NULL){                                                       \
      shmem_free(workBuffer);                                                  \
      return 1;                                                                \
    }                                                                          \
  }                                                                            \
  else {                                                                       \
    /* zero out dest array */                                                  \
    memset(dest, 0, nelems * sizeof(_type));                                   \
    result = (_type *) dest;                                                   \
  }                                                                            \
                                                                               \
  shcoll_barrier_linear(PE_start, logPE_stride, PE_size, pSync);               \
  /* Do insum scan */                                                          \
  for (int pe = 0; pe <= me_as - 1; pe++){                                     \
    int src_pe = shmemc_team_pe_to_world(team, pe);                            \
    shmem_getmem(workBuffer, source, nelems * sizeof(_type), src_pe);          \
    /* add results */                                                          \
    for (int i = 0; i < nelems; i ++){                                         \
      result[i] = result[i] + workBuffer[i];                                   \
    }                                                                          \
  }                                                                            \
  shcoll_barrier_linear(PE_start, logPE_stride, PE_size, pSync);               \
                                                                               \
  /* If dest and source are same, override array with results */               \
  if ( (void *) dest == (void *) source){                                      \
    for (int i = 0; i < nelems; i ++)                                          \
      dest[i] = result[i];                                                     \
    free(result);                                                              \
  }                                                                            \
                                                                               \
  shmem_free(workBuffer);                                                      \
  return 0;                                                                    \
}

/* Implement the ring algorithm here */                                        
#define SUM_EXSCAN_HELPER_RING(_name, _type)                                       \
  int sum_exscan_helper_##_name##_ring(                                          \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * workBuffer;                                                          \
  if ((void *) dest == (void *) source){                                       \
    workBuffer = (_type *)shmem_malloc(nelems * sizeof(_type));                \
    if (workBuffer == NULL)                                                    \
      return 1;                                                                \
  } else {                                                                     \
    /* if different src / dst arrays, save memory and use dest */              \
    /* as our work buffer */                                                   \
    workBuffer = dest;                                                         \
  }                                                                            \
  uint64_t * received = shmem_calloc(1,sizeof(uint64_t));                      \
  if (received == NULL){                                                       \
    if ((void *) dest == (void *) source) shmem_free(workBuffer);              \
    return 1;                                                                  \
  }                                                                            \
  if (me_as == 0 && dest != source)                                            \
    memcpy(dest, source, nelems * sizeof(_type));                              \
                                                                               \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
                                                                               \
  /* do shmem_sum_inscan */                                                        \
  /* wait on results and take sum (except for first PE) */                     \
  if (me_as != 0){                                                             \
    shmem_wait_until(received, 1, SHMEM_CMP_EQ);                               \
    for (size_t i = 0; i < nelems; i ++)                                       \
      dest[i] = workBuffer[i] + source[i];                                     \
  }                                                                            \
  if (me_as != PE_size - 1){                                                   \
    /* put results to next PE in line (except for last PE) */                  \
    int dest_pe = shmemc_team_pe_to_world(team, me_as + 1);                    \
    shmem_put8_signal(workBuffer, dest, nelems * sizeof(_type), received, 1,   \
        SHMEM_SIGNAL_SET, dest_pe);                                            \
  }                                                                            \
                                                                               \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
                                                                               \
  /* shift phase for sum_exscan */                                                 \
  _type * shiftBuf = (_type *)shmem_malloc(nelems * sizeof(_type));            \
  if (shiftBuf != NULL) {                                                      \
    if (me_as > 0) {                                                           \
      int target = shmemc_team_pe_to_world(team, me_as - 1);                   \
      shmem_getmem(shiftBuf, dest, nelems * sizeof(_type), target);            \
    }                                                                          \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
    if (me_as > 0) {                                                           \
      memcpy(dest, shiftBuf, nelems * sizeof(_type));                          \
    } else {                                                                   \
      memset(dest, 0, nelems * sizeof(_type));                                 \
    }                                                                          \
    shmem_free(shiftBuf);                                                      \
  }                                                                            \
                                                                               \
  if ((void *) dest == (void *) source)                                        \
    shmem_free(workBuffer);                                                    \
  shmem_free(received);                                                        \
  return (shiftBuf == NULL) ? 1 : 0;                                           \
}

#define SUM_EXSCAN_HELPER_LOGARITHMIC(_name, _type)                                \
  int sum_exscan_helper_##_name##_logarithmic(                                          \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * workBuffer = (_type *)shmem_malloc(nelems * sizeof(_type));          \
  if (workBuffer == NULL)                                                      \
    return 1;                                                                  \
  if ( (void *) source != (void *) dest){                                      \
    memcpy(dest, source, nelems * sizeof(_type));                              \
  }                                                                            \
                                                                               \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
  for (int stride = 1; stride <= PE_size; stride <<= 1){                       \
    if (me_as >= stride){                                                      \
      int target = shmemc_team_pe_to_world(team, me_as - stride);              \
      shmem_getmem(workBuffer, dest, nelems * sizeof(_type), target);          \
    }                                                                          \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
    if (me_as >= stride){                                                      \
      for (int elem = 0; elem < nelems; elem ++){                              \
        dest[elem] += workBuffer[elem];                                        \
      }                                                                        \
    }                                                                          \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
  }                                                                            \
                                                                               \
  /* shift phase for sum_exscan */                                                 \
  _type * shiftBuf = (_type *)shmem_malloc(nelems * sizeof(_type));            \
  if (shiftBuf != NULL) {                                                      \
    if (me_as > 0) {                                                           \
      int target = shmemc_team_pe_to_world(team, me_as - 1);                   \
      shmem_getmem(shiftBuf, dest, nelems * sizeof(_type), target);            \
    }                                                                          \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
    if (me_as > 0) {                                                           \
      memcpy(dest, shiftBuf, nelems * sizeof(_type));                          \
    } else {                                                                   \
      memset(dest, 0, nelems * sizeof(_type));                                 \
    }                                                                          \
    shmem_free(shiftBuf);                                                      \
  }                                                                            \
                                                                               \
  shmem_free(workBuffer);                                                      \
  return (shiftBuf == NULL) ? 1 : 0;                                           \
}

#define SUM_EXSCAN_HELPER_RECURSIVE_DOUBLING(_name, _type)                                \
  int sum_exscan_helper_##_name##_recursive_doubling(                                          \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, _type *pWrk, long *pSync) {   \
  _type * partialResBuf = (_type *) shmem_malloc(nelems * sizeof(_type));     \
  if (partialResBuf == NULL)                                                   \
    return 1;                                                                  \
  _type * recvBuf = (_type *) shmem_malloc(nelems * sizeof(_type));           \
  if (recvBuf == NULL){                                                        \
    shmem_free(partialResBuf);                                                 \
    return 1;                                                                  \
  }                                                                            \
  memcpy(partialResBuf, source, nelems * sizeof(_type)); \
  memset(dest, 0, nelems * sizeof(_type)); \
  shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);        \
                                                                               \
  for (int stride = 1; stride < PE_size; stride <<= 1){                        \
    int target = stride ^ me_as;                                               \
    if (target < PE_size){                                                     \
      int world_target = shmemc_team_pe_to_world(team, target);                \
      shmem_getmem(recvBuf, partialResBuf, nelems * sizeof(_type),             \
          world_target);                                                       \
      shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);    \
      if (me_as > target){                                                     \
        for (size_t i = 0; i < nelems; i ++)                                   \
          partialResBuf[i] = partialResBuf[i] + recvBuf[i];                    \
        for (size_t i = 0; i < nelems; i ++)                                   \
          dest[i] = dest[i] + recvBuf[i];                    \
      } else {                                                                 \
        for (size_t i = 0; i < nelems; i ++)                                   \
          partialResBuf[i] = partialResBuf[i] + recvBuf[i];                    \
      }                                                                        \
    }                                                                          \
    shcoll_barrier_binomial_tree(PE_start, logPE_stride, PE_size, pSync);      \
  }                                                                            \
  shmem_free(partialResBuf);                                                   \
  shmem_free(recvBuf);                                                         \
  return 0;                                                                    \
}

#define DECLARE_SUM_EXSCAN_HELPER(_type, _typename)                               \
  SUM_EXSCAN_HELPER_LINEAR(_typename, _type) \
  SUM_EXSCAN_HELPER_RING(_typename, _type) \
  SUM_EXSCAN_HELPER_LOGARITHMIC(_typename, _type) \
  SUM_EXSCAN_HELPER_RECURSIVE_DOUBLING(_typename, _type)

SHMEM_REDUCE_ARITH_TYPE_TABLE(DECLARE_SUM_EXSCAN_HELPER)
#undef DECLARE_SUM_EXSCAN_HELPER

/*
 * @brief Macro to define team-based sum_inscan operations
 */
#define SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, _algo)                      \
  int shcoll_##_typename##_sum_inscan_##_algo(                                     \
      shmem_team_t team, _type *dest, const _type *source, size_t nelems) {    \
    SHMEMU_CHECK_INIT();                                                       \
    SHMEMU_CHECK_TEAM_VALID(team);                                             \
    SHMEMU_CHECK_SYMMETRIC(dest, "dest");                                      \
    SHMEMU_CHECK_SYMMETRIC(source, "source");                                  \
    shmemc_team_h team_h = (shmemc_team_h)team;                                \
    SHMEMU_CHECK_NULL(shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE),  \
                     "team_h->pSyncs[COLLECTIVE]");                            \
                                                                               \
    _type *pWrk =                                                              \
        shmem_malloc(SHCOLL_REDUCE_MIN_WRKDATA_SIZE * sizeof(_type));          \
    int me = shmemc_my_pe();                                                   \
    int me_as = shmemc_team_translate_pe(SHMEM_TEAM_WORLD, me, team);          \
                                                                               \
    int success = sum_inscan_helper_##_typename##_##_algo(                         \
        dest, source, nelems, me_as, team, team_h->start,                      \
        (team_h->stride > 0) ? (int)log2((double)team_h->stride) : 0,          \
        team_h->nranks, pWrk,                                                  \
        shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE));               \
                                                                               \
    shmemc_team_reset_psync(team_h, SHMEMC_PSYNC_COLLECTIVE);                  \
    shmem_free(pWrk);                                                          \
    return success;                                                            \
  }

/*
 * @brief Macro to define team-based sum_inscan operations
 */
#define SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, _algo)                      \
  int shcoll_##_typename##_sum_exscan_##_algo(                                     \
      shmem_team_t team, _type *dest, const _type *source, size_t nelems) {    \
    SHMEMU_CHECK_INIT();                                                       \
    SHMEMU_CHECK_TEAM_VALID(team);                                             \
    SHMEMU_CHECK_SYMMETRIC(dest, "dest");                                      \
    SHMEMU_CHECK_SYMMETRIC(source, "source");                                  \
    shmemc_team_h team_h = (shmemc_team_h)team;                                \
    SHMEMU_CHECK_NULL(shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE),  \
                     "team_h->pSyncs[COLLECTIVE]");                            \
    /* TODO: check that either same src & dest, or non-overlapping */          \
    _type *pWrk =                                                              \
        shmem_malloc(SHCOLL_REDUCE_MIN_WRKDATA_SIZE * sizeof(_type));          \
    int me = shmemc_my_pe();                                                   \
    int me_as = shmemc_team_translate_pe(SHMEM_TEAM_WORLD, me, team);          \
                                                                               \
    int success = sum_exscan_helper_##_typename##_##_algo(                         \
        dest, source, nelems, me_as, team, team_h->start,                      \
        (team_h->stride > 0) ? (int)log2((double)team_h->stride) : 0,          \
        team_h->nranks, pWrk,                                                  \
        shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE));               \
                                                                               \
    shmemc_team_reset_psync(team_h, SHMEMC_PSYNC_COLLECTIVE);                  \
    shmem_free(pWrk);                                                          \
    return success;                                                            \
  }

#define X(_type, _typename)                                       \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, linear)              \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, ring)              \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, logarithmic)              \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, recursive_doubling)              \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, linear)  \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, ring) \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, logarithmic) \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, recursive_doubling)
  SHMEM_REDUCE_ARITH_TYPE_TABLE(X)
#undef X


