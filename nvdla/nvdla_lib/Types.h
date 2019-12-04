//===- Types.h ------------------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef NVDLA_NVDLA_LIB_TYPES_H_INCLUDED
#define NVDLA_NVDLA_LIB_TYPES_H_INCLUDED

#include <cstdint>
#include "NvDlaDefine.h"

namespace nvdla {

enum class ConfigSet : unsigned
{
  nv_full = 0,
};

enum class ExecutionMode : unsigned
{
  direct = 0,
};

} // namespace nvdla

#ifndef DEFINE_WEIGHT_TYPE
#  define DEFINE_WEIGHT_TYPE(config_set, weight_type) \
    template <>                                       \
    struct get_weight_type<config_set>                \
    {                                                 \
      using type = weight_type;                       \
    }
#else
#  error "macro DEFINE_WEIGHT_TYPE has been define in other file"
#endif

template <nvdla::ConfigSet CS>
struct get_weight_type;

template <nvdla::ConfigSet CS>
using nv_weight_t = typename get_weight_type<CS>::type;

DEFINE_WEIGHT_TYPE(nvdla::ConfigSet::nv_full, std::uint16_t);

#undef DEFINE_WEIGHT_TYPE

#endif
