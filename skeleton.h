#ifndef SKELETON_H
#define SKELETON_H

#include <stdint.h>
#include <tree_sitter/api.h>

/* Capture index names — must match order in QUERY_SOURCE */
enum {
    CAP_FUNC_DEF = 0,
    CAP_STRUCT_DEF,
    CAP_ENUM_DEF,
    CAP_TYPEDEF,
    CAP_GLOBAL_VAR,
    CAP_PREPROC_DEF,
    CAP_PREPROC_FUNC,
    CAP_INCLUDE,
};

char *skeleton_text(const char *source, TSNode node, uint32_t cap_idx);
void strip_struct_refs(char *text);

#endif /* SKELETON_H */
