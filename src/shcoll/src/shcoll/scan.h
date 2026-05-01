/**
 * @file scan.h
 * @brief Header file containing declarations for scan operations
 *
 */

#ifndef _SHCOLL_SCAN_H
#define _SHCOLL_SCAN_H 1

#include <shmem/teams.h>
#include "shmemu.h"
#include <shmem/api_types.h>

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Macro to declare a single sum_inscan operation
 *
 * @param _typename Name of datatype
 * @param _type Data type to operate on
 * @param _algo Algorithm implementation to use
 */
#define SHCOLL_SUM_INSCAN_DECLARE(_typename, _type, _algo)                      \
  int shcoll_##_typename##_sum_inscan_##_algo(                                  \
      shmem_team_t team, _type *dest, const _type *source, size_t nelems);

/**
 * @brief Macro to declare a single sum_exscan operation
 *
 * @param _typename Name of datatype
 * @param _type Data type to operate on
 * @param _algo Algorithm implementation to use
 */
#define SHCOLL_SUM_EXSCAN_DECLARE(_typename, _type, _algo)                      \
  int shcoll_##_typename##_sum_exscan_##_algo(                                  \
      shmem_team_t team, _type *dest, const _type *source, size_t nelems);

#define DECLARE_SUM_INSCAN(_type, _typename)                                    \
  SHCOLL_SUM_INSCAN_DECLARE(_typename, _type, linear)                           \
  SHCOLL_SUM_INSCAN_DECLARE(_typename, _type, ring)                             \
  SHCOLL_SUM_INSCAN_DECLARE(_typename, _type, logarithmic)                      \
  SHCOLL_SUM_INSCAN_DECLARE(_typename, _type, rec_dbl)

SHMEM_SCAN_ARITH_TYPE_TABLE(DECLARE_SUM_INSCAN)
#undef DECLARE_SUM_INSCAN

#define DECLARE_SUM_EXSCAN(_type, _typename)                                    \
  SHCOLL_SUM_EXSCAN_DECLARE(_typename, _type, linear)                           \
  SHCOLL_SUM_EXSCAN_DECLARE(_typename, _type, ring)                             \
  SHCOLL_SUM_EXSCAN_DECLARE(_typename, _type, logarithmic)                      \
  SHCOLL_SUM_EXSCAN_DECLARE(_typename, _type, rec_dbl)


SHMEM_SCAN_ARITH_TYPE_TABLE(DECLARE_SUM_EXSCAN)
#undef DECLARE_SUM_EXSCAN

#endif /* ! _SHCOLL_SCAN_H */
