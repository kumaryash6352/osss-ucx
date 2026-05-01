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
#define SUM_INSCAN_HELPER_LINEAR(_name, _type)                                 \
  static inline int sum_inscan_helper_##_name##_linear(                        \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
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

/*
   @brief Helper macro to define ring sum_inscan operations
 *
 * Implements a ring sum_inscan algorithm where each PE waits for the partial
 * prefix sum from its predecessor via a signal, adds its own source values,
 * and forwards the result to its successor using shmem_put8_signal.  PE 0
 * seeds the chain with its own values.  Two binomial-tree barriers bookend
 * the data-exchange phase: the first ensures all source data is committed
 * before the chain starts; the second ensures all destinations are stable
 * before returning.
 *
 * @param _name Typename of sum_inscan operation
 * @param _type Data type to operate on
 */
#define SUM_INSCAN_HELPER_RING(_name, _type)                                   \
  static inline int sum_inscan_helper_##_name##_ring(                          \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
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
  /* do shmem_sum_inscan */                                                    \
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

/*
   @brief Helper macro to define logarithmic (Hillis-Steele) sum_inscan operations
 *
 * Implements the Hillis-Steele parallel prefix scan.  In each of log2(PE_size)
 * rounds (strides 1, 2, 4, ...), every PE whose rank is at least stride fetches
 * the current dest from the PE at rank (me - stride) and accumulates it into
 * its own dest.  Two barriers per round prevent read-after-write races across
 * PEs: one after the fetch and one after the update.  After all rounds, PE i
 * holds the inclusive sum of elements 0..i.
 *
 * @param _name Typename of sum_inscan operation
 * @param _type Data type to operate on
 */
#define SUM_INSCAN_HELPER_LOGARITHMIC(_name, _type)                            \
  static inline int sum_inscan_helper_##_name##_logarithmic(                   \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
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

/*
   @brief Helper macro to define recursive-doubling sum_inscan operations
 *
 * Implements an inclusive prefix scan via recursive doubling.  Two symmetric
 * buffers are maintained: partialResBuf (the group-reduction partial sum for
 * this PE's subtree) and dest (the running inclusive prefix sum).  In each of
 * log2(PE_size) rounds, each PE XOR-pairs with a partner at distance stride
 * and fetches that partner's partialResBuf.  The higher-ranked PE in each pair
 * accumulates the received data into both partialResBuf and dest; the lower-
 * ranked PE accumulates only into partialResBuf.  Two barriers per round
 * (before and after the accumulate) prevent races.  After all rounds, dest
 * holds the inclusive prefix sum for each PE.
 *
 * @param _name Typename of sum_inscan operation
 * @param _type Data type to operate on
 */
#define SUM_INSCAN_HELPER_RECURSIVE_DOUBLING(_name, _type)                     \
  static inline int sum_inscan_helper_##_name##_rec_dbl(                       \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
  _type * partialResBuf = (_type *) shmem_malloc(nelems * sizeof(_type));      \
  if (partialResBuf == NULL)                                                   \
    return 1;                                                                  \
  _type * recvBuf = (_type *) shmem_malloc(nelems * sizeof(_type));            \
  if (recvBuf == NULL){                                                        \
    shmem_free(partialResBuf);                                                 \
    return 1;                                                                  \
  }                                                                            \
  memcpy(partialResBuf, source, nelems * sizeof(_type));                       \
  memcpy(dest, source, nelems * sizeof(_type));                                \
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
          dest[i] = dest[i] + recvBuf[i];                                      \
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

/*
   @brief Instantiates all sum_inscan algorithm variants for a given type
 *
 * Expands all four sum_inscan helper macros (linear, ring, logarithmic, and
 * recursive doubling) for the given (_type, _typename) pair.  Intended to be
 * passed as the callback to SHMEM_SCAN_ARITH_TYPE_TABLE so that one
 * instantiation covers every supported arithmetic type.
 *
 * @param _type    C type of the scan elements (e.g. int, float)
 * @param _typename Stringified type name used in function-name mangling
 */
#define DECLARE_SUM_INSCAN_HELPER(_type, _typename)                            \
  SUM_INSCAN_HELPER_LINEAR(_typename, _type)                                   \
  SUM_INSCAN_HELPER_RING(_typename, _type)                                     \
  SUM_INSCAN_HELPER_LOGARITHMIC(_typename, _type)                              \
  SUM_INSCAN_HELPER_RECURSIVE_DOUBLING(_typename, _type)

SHMEM_SCAN_ARITH_TYPE_TABLE(DECLARE_SUM_INSCAN_HELPER)
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
#define SUM_EXSCAN_HELPER_LINEAR(_name, _type)                                 \
  static inline int sum_exscan_helper_##_name##_linear(                        \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
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

/*
   @brief Helper macro to define ring sum_exscan operations
 *
 * Implements an exclusive prefix scan using the ring algorithm.  The data-
 * exchange phase is identical to SUM_INSCAN_HELPER_RING: a signal-driven chain
 * propagates partial sums from PE 0 to PE N-1.  After the second barrier, an
 * additional shift phase fetches each PE's inclusive result from its
 * predecessor (PE 0 writes zeros), converting the inclusive chain result into
 * an exclusive prefix sum.  A third barrier ensures the shifted values are
 * visible before the function returns.
 *
 * @param _name Typename of sum_exscan operation
 * @param _type Data type to operate on
 */
#define SUM_EXSCAN_HELPER_RING(_name, _type)                                   \
  static inline int sum_exscan_helper_##_name##_ring(                          \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
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
  /* do shmem_sum_inscan */                                                    \
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
  /* shift phase for sum_exscan */                                             \
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

/*
   @brief Helper macro to define logarithmic (Hillis-Steele) sum_exscan operations
 *
 * Runs the same log2(PE_size)-round Hillis-Steele sweep as
 * SUM_INSCAN_HELPER_LOGARITHMIC to build an inclusive prefix sum in dest, then
 * applies a shift phase: each PE (me > 0) fetches dest from its predecessor
 * and overwrites its own dest with that value; PE 0 zeroes its dest.  A barrier
 * after the shift ensures the exclusive results are stable before returning.
 *
 * @param _name Typename of sum_exscan operation
 * @param _type Data type to operate on
 */
#define SUM_EXSCAN_HELPER_LOGARITHMIC(_name, _type)                            \
  static inline int sum_exscan_helper_##_name##_logarithmic(                   \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
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
  /* shift phase for sum_exscan */                                             \
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

/*
   @brief Helper macro to define recursive-doubling sum_exscan operations
 *
 * Produces an exclusive prefix scan using the same XOR-pairing structure as
 * SUM_INSCAN_HELPER_RECURSIVE_DOUBLING.  partialResBuf is initialised from
 * source and dest is zeroed.  In each round the higher-ranked PE of an XOR
 * pair accumulates the received partial sum into both partialResBuf and dest;
 * the lower-ranked PE accumulates only into partialResBuf.  Because dest starts
 * at zero and is only updated by PEs that are strictly greater than their
 * partner, dest converges to the exclusive prefix sum (the sum of all elements
 * with rank strictly less than the calling PE) after all log2(PE_size) rounds.
 *
 * @param _name Typename of sum_exscan operation
 * @param _type Data type to operate on
 */
#define SUM_EXSCAN_HELPER_RECURSIVE_DOUBLING(_name, _type)                     \
  static inline int sum_exscan_helper_##_name##_rec_dbl(                       \
    _type *dest, const _type *source, int nelems, int me_as, shmem_team_t team,\
    int PE_start, int logPE_stride, int PE_size, long *pSync) {                \
  _type * partialResBuf = (_type *) shmem_malloc(nelems * sizeof(_type));      \
  if (partialResBuf == NULL)                                                   \
    return 1;                                                                  \
  _type * recvBuf = (_type *) shmem_malloc(nelems * sizeof(_type));            \
  if (recvBuf == NULL){                                                        \
    shmem_free(partialResBuf);                                                 \
    return 1;                                                                  \
  }                                                                            \
  memcpy(partialResBuf, source, nelems * sizeof(_type));                       \
  memset(dest, 0, nelems * sizeof(_type));                                     \
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
          dest[i] = dest[i] + recvBuf[i];                                      \
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

/*
   @brief Instantiates all sum_exscan algorithm variants for a given type
 *
 * Expands all four sum_exscan helper macros (linear, ring, logarithmic, and
 * recursive doubling) for the given (_type, _typename) pair.  Intended to be
 * passed as the callback to SHMEM_SCAN_ARITH_TYPE_TABLE so that one
 * instantiation covers every supported arithmetic type.
 *
 * @param _type    C type of the scan elements (e.g. int, float)
 * @param _typename Stringified type name used in function-name mangling
 */
#define DECLARE_SUM_EXSCAN_HELPER(_type, _typename)                            \
  SUM_EXSCAN_HELPER_LINEAR(_typename, _type)                                   \
  SUM_EXSCAN_HELPER_RING(_typename, _type)                                     \
  SUM_EXSCAN_HELPER_LOGARITHMIC(_typename, _type)                              \
  SUM_EXSCAN_HELPER_RECURSIVE_DOUBLING(_typename, _type)

SHMEM_SCAN_ARITH_TYPE_TABLE(DECLARE_SUM_EXSCAN_HELPER)
#undef DECLARE_SUM_EXSCAN_HELPER

/*
 * @brief Macro to define a team-based sum_inscan public API entry point
 *
 * Generates the shcoll_<typename>_sum_inscan_<algo> function for one
 * (_typename, _type, _algo) triple.  The generated function validates its
 * arguments, translates the caller's team-local rank, delegates to the
 * corresponding sum_inscan_helper, and resets the collective pSync slot
 * before returning.
 *
 * @param _typename Stringified type name used in function-name mangling
 * @param _type     C type of the scan elements
 * @param _algo     Algorithm suffix (linear, ring, logarithmic, or rec_dbl)
 */
#define SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, _algo)                  \
  int shcoll_##_typename##_sum_inscan_##_algo(                                 \
      shmem_team_t team, _type *dest, const _type *source, size_t nelems) {    \
    SHMEMU_CHECK_INIT();                                                       \
    SHMEMU_CHECK_TEAM_VALID(team);                                             \
    SHMEMU_CHECK_SYMMETRIC(dest, "dest");                                      \
    SHMEMU_CHECK_SYMMETRIC(source, "source");                                  \
    shmemc_team_h team_h = (shmemc_team_h)team;                                \
    SHMEMU_CHECK_NULL(shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE),  \
                     "team_h->pSyncs[COLLECTIVE]");                            \
    SHMEMU_CHECK_BUFFER_OVERLAP(dest, source,                                  \
                      sizeof(_type) * nelems,sizeof(_type) * nelems);          \
                                                                               \
    int me = shmemc_my_pe();                                                   \
    int me_as = shmemc_team_translate_pe(SHMEM_TEAM_WORLD, me, team);          \
                                                                               \
    int success = sum_inscan_helper_##_typename##_##_algo(                     \
        dest, source, nelems, me_as, team, team_h->start,                      \
        (team_h->stride > 0) ? (int)log2((double)team_h->stride) : 0,          \
        team_h->nranks,                                                        \
        shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE));               \
                                                                               \
    shmemc_team_reset_psync(team_h, SHMEMC_PSYNC_COLLECTIVE);                  \
    return success;                                                            \
  }

/*
 * @brief Macro to define a team-based sum_exscan public API entry point
 *
 * Generates the shcoll_<typename>_sum_exscan_<algo> function for one
 * (_typename, _type, _algo) triple.  The generated function validates its
 * arguments, translates the caller's team-local rank, delegates to the
 * corresponding sum_exscan_helper, and resets the collective pSync slot
 * before returning.
 *
 * @param _typename Stringified type name used in function-name mangling
 * @param _type     C type of the scan elements
 * @param _algo     Algorithm suffix (linear, ring, logarithmic, or rec_dbl)
 */
#define SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, _algo)                  \
  int shcoll_##_typename##_sum_exscan_##_algo(                                 \
      shmem_team_t team, _type *dest, const _type *source, size_t nelems) {    \
    SHMEMU_CHECK_INIT();                                                       \
    SHMEMU_CHECK_TEAM_VALID(team);                                             \
    SHMEMU_CHECK_SYMMETRIC(dest, "dest");                                      \
    SHMEMU_CHECK_SYMMETRIC(source, "source");                                  \
    shmemc_team_h team_h = (shmemc_team_h)team;                                \
    SHMEMU_CHECK_NULL(shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE),  \
                     "team_h->pSyncs[COLLECTIVE]");                            \
    SHMEMU_CHECK_BUFFER_OVERLAP(dest, source,                                  \
                      sizeof(_type) * nelems,sizeof(_type) * nelems);          \
                                                                               \
    int me = shmemc_my_pe();                                                   \
    int me_as = shmemc_team_translate_pe(SHMEM_TEAM_WORLD, me, team);          \
                                                                               \
    int success = sum_exscan_helper_##_typename##_##_algo(                     \
        dest, source, nelems, me_as, team, team_h->start,                      \
        (team_h->stride > 0) ? (int)log2((double)team_h->stride) : 0,          \
        team_h->nranks,                                                        \
        shmemc_team_get_psync(team_h, SHMEMC_PSYNC_COLLECTIVE));               \
                                                                               \
    shmemc_team_reset_psync(team_h, SHMEMC_PSYNC_COLLECTIVE);                  \
    return success;                                                            \
  }

/*
 * @brief Macro to emit all public sum_inscan and sum_exscan entry points for one type
 *
 * Expands SHCOLL_SUM_INSCAN_DEFINITION and SHCOLL_SUM_EXSCAN_DEFINITION for
 * every algorithm variant (linear, ring, logarithmic, rec_dbl) for the given
 * (_type, _typename) pair.  Passed as the callback to
 * SHMEM_SCAN_ARITH_TYPE_TABLE so that the full API surface is instantiated for
 * every supported arithmetic type in a single table expansion.
 *
 * @param _type     C type of the scan elements
 * @param _typename Stringified type name used in function-name mangling
 */
#define X(_type, _typename)                                                    \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, linear)                       \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, ring)                         \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, logarithmic)                  \
  SHCOLL_SUM_INSCAN_DEFINITION(_typename, _type, rec_dbl)                      \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, linear)                       \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, ring)                         \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, logarithmic)                  \
  SHCOLL_SUM_EXSCAN_DEFINITION(_typename, _type, rec_dbl)
  SHMEM_SCAN_ARITH_TYPE_TABLE(X)
#undef X
