#ifndef SYMRES_H
#define SYMRES_H

#include <stdint.h>
#include <stddef.h>
#include <elfutils/libdwfl.h>
#include <capstone/capstone.h>
#include <libelf.h>

struct arena;

struct sym_resolver {
    Dwfl *dwfl;
    Dwfl_Module *mod;
    csh cs_handle;
    int cs_ok;
    Elf *elf;
    int elf_fd;
    uint64_t addr_bias;  /* subtract from runtime IP -> dwfl address */
};

uint64_t read_load_base(pid_t pid, const char *binary_path);
int symres_init(struct sym_resolver *sr, const char *binary_path,
                uint64_t load_base);
const char *symres_func_name(struct sym_resolver *sr, uint64_t addr);
int symres_srcline(struct sym_resolver *sr, uint64_t addr,
                   const char **file, int *line);
int symres_func_range(struct sym_resolver *sr, const char *name,
                      uint64_t *start, uint8_t **bytes, size_t *len,
                      struct arena *a);
int symres_has_debug(struct sym_resolver *sr);
void symres_free(struct sym_resolver *sr);

#endif /* SYMRES_H */
