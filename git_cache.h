#ifndef GIT_CACHE_H
#define GIT_CACHE_H

#include "perf_analysis.h"  /* struct file_entry, struct symbol */

typedef struct archmap_cache archmap_cache;

/* Build the options string used for cache invalidation.
   Caller must free the returned string. */
char *cache_make_opts_str(int show_calls, char **defines, int n_defines,
                          char **strip_macros, int n_strip,
                          char **inc_paths, int n_inc);

/* Open (or create) cache for the repo containing work_dir.
   Returns NULL if not in a git repo or on error (graceful degradation). */
archmap_cache *cache_open(const char *work_dir, const char *opts_str);

/* Look up a cached file entry by absolute path.
   Returns: 1=hit (out_fe populated, caller owns), 0=miss, -1=error.
   On hit, *out_fe is a deep copy; *out_tags is a malloc'd array of
   strdup'd tag names (caller frees each + array). */
int cache_lookup(archmap_cache *c, const char *abs_path,
                 struct file_entry *out_fe,
                 char ***out_tags, int *out_n_tags);

/* Store a file entry in the cache (keyed by blob OID of abs_path). */
void cache_store(archmap_cache *c, const char *abs_path,
                 const struct file_entry *fe,
                 char **tag_names, int n_tags);

/* Write cache to disk (if dirty) and free all resources. */
void cache_close(archmap_cache *c);

#endif /* GIT_CACHE_H */
