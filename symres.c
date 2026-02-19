/* symres.c — Symbol resolution via libdwfl + capstone */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <fcntl.h>
#include <gelf.h>

#include "arena.h"
#include "symres.h"

static const Dwfl_Callbacks offline_callbacks = {
    .find_elf = dwfl_build_id_find_elf,
    .find_debuginfo = dwfl_standard_find_debuginfo,
    .section_address = dwfl_offline_section_address,
};

/* Read /proc/pid/maps to find the load base of the binary.
   Returns 0 on failure (child already exited or binary not found). */
uint64_t read_load_base(pid_t pid, const char *binary_path) {
    char maps_path[64];
    snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
    FILE *fp = fopen(maps_path, "r");
    if (!fp) return 0;

    char *real_bin = realpath(binary_path, NULL);

    uint64_t load_base = 0;
    char line[1024];
    while (fgets(line, sizeof(line), fp)) {
        uint64_t start;
        uint64_t offset;
        char pathname[512] = {0};

        if (sscanf(line, "%" SCNx64 "-%*x %*s %" SCNx64 " %*s %*s %511[^\n]",
                   &start, &offset, pathname) >= 2) {
            char *p = pathname;
            while (*p == ' ') p++;
            /* Strip " (deleted)" suffix */
            char *del = strstr(p, " (deleted)");
            if (del) *del = '\0';

            if (*p && ((real_bin && strcmp(p, real_bin) == 0) ||
                       strcmp(p, binary_path) == 0)) {
                load_base = start - offset;
                break;
            }
        }
    }
    fclose(fp);
    free(real_bin);
    return load_base;
}

int symres_init(struct sym_resolver *sr, const char *binary_path,
                        uint64_t load_base) {
    memset(sr, 0, sizeof(*sr));
    sr->elf_fd = -1;

    elf_version(EV_CURRENT);

    sr->dwfl = dwfl_begin(&offline_callbacks);
    if (!sr->dwfl) return -1;

    sr->mod = dwfl_report_offline(sr->dwfl, "", binary_path, -1);
    dwfl_report_end(sr->dwfl, NULL, NULL);
    if (!sr->mod) {
        dwfl_end(sr->dwfl);
        sr->dwfl = NULL;
        return -1;
    }

    /* Open ELF directly for raw section data */
    sr->elf_fd = open(binary_path, O_RDONLY);
    if (sr->elf_fd >= 0)
        sr->elf = elf_begin(sr->elf_fd, ELF_C_READ, NULL);

    /* Compute address bias for PIE binaries */
    if (sr->elf && load_base > 0) {
        GElf_Ehdr ehdr;
        if (gelf_getehdr(sr->elf, &ehdr) && ehdr.e_type == ET_DYN) {
            Dwarf_Addr mod_low;
            dwfl_module_info(sr->mod, NULL, &mod_low, NULL,
                             NULL, NULL, NULL, NULL);
            sr->addr_bias = load_base - mod_low;
        }
    }

    /* Init capstone */
    if (cs_open(CS_ARCH_X86, CS_MODE_64, &sr->cs_handle) == CS_ERR_OK) {
        cs_option(sr->cs_handle, CS_OPT_SYNTAX, CS_OPT_SYNTAX_ATT);
        sr->cs_ok = 1;
    }

    return 0;
}

const char *symres_func_name(struct sym_resolver *sr, uint64_t addr) {
    if (!sr->dwfl) return NULL;
    uint64_t dwfl_addr = addr - sr->addr_bias;
    Dwfl_Module *mod = dwfl_addrmodule(sr->dwfl, dwfl_addr);
    if (!mod) return NULL;
    GElf_Off offset;
    GElf_Sym sym;
    return dwfl_module_addrinfo(mod, dwfl_addr, &offset, &sym,
                                NULL, NULL, NULL);
}

int symres_srcline(struct sym_resolver *sr, uint64_t addr,
                           const char **file, int *line) {
    if (!sr->dwfl) return -1;
    uint64_t dwfl_addr = addr - sr->addr_bias;
    Dwfl_Module *mod = dwfl_addrmodule(sr->dwfl, dwfl_addr);
    if (!mod) return -1;
    Dwfl_Line *ln = dwfl_module_getsrc(mod, dwfl_addr);
    if (!ln) return -1;
    *file = dwfl_lineinfo(ln, NULL, line, NULL, NULL, NULL);
    return *file ? 0 : -1;
}

/* Returns runtime start address, raw bytes (caller must free), and length.
   Uses the raw symbol name (including .constprop.N etc). */
int symres_func_range(struct sym_resolver *sr, const char *name,
                              uint64_t *start, uint8_t **bytes, size_t *len,
                              struct arena *a) {
    if (!sr->mod || !sr->elf) return -1;

    /* Find symbol via dwfl (for the runtime-adjusted address) */
    int nsyms = dwfl_module_getsymtab(sr->mod);
    GElf_Sym best_sym;
    GElf_Addr best_addr = 0;
    int found = 0;

    for (int i = 0; i < nsyms; i++) {
        GElf_Sym sym;
        GElf_Addr addr;
        const char *sname = dwfl_module_getsym_info(
            sr->mod, i, &sym, &addr, NULL, NULL, NULL);
        if (!sname || GELF_ST_TYPE(sym.st_info) != STT_FUNC) continue;
        if (strcmp(sname, name) == 0 && sym.st_size > 0) {
            best_sym = sym;
            best_addr = addr;
            found = 1;
            break;
        }
    }
    if (!found) return -1;

    *start = best_addr + sr->addr_bias;  /* convert to runtime address */
    *len = best_sym.st_size;

    /* Extract raw bytes from ELF using the original (non-biased) address */
    Elf_Scn *scn = NULL;
    while ((scn = elf_nextscn(sr->elf, scn)) != NULL) {
        GElf_Shdr shdr;
        gelf_getshdr(scn, &shdr);
        if (shdr.sh_type != SHT_PROGBITS) continue;
        if (!(shdr.sh_flags & SHF_EXECINSTR)) continue;

        if (best_sym.st_value >= shdr.sh_addr &&
            best_sym.st_value + best_sym.st_size <=
                shdr.sh_addr + shdr.sh_size) {
            Elf_Data *d = elf_getdata(scn, NULL);
            if (!d) return -1;
            uint64_t offset = best_sym.st_value - shdr.sh_addr;
            if (offset + best_sym.st_size > d->d_size) return -1;
            *bytes = arena_alloc(a, best_sym.st_size);
            memcpy(*bytes, (uint8_t *)d->d_buf + offset, best_sym.st_size);
            return 0;
        }
    }
    return -1;
}

int symres_has_debug(struct sym_resolver *sr) {
    if (!sr->dwfl || !sr->mod) return 0;
    Dwarf_Addr bias;
    Dwarf *dbg = dwfl_module_getdwarf(sr->mod, &bias);
    return dbg != NULL;
}

void symres_free(struct sym_resolver *sr) {
    if (sr->cs_ok) cs_close(&sr->cs_handle);
    if (sr->elf) elf_end(sr->elf);
    if (sr->elf_fd >= 0) close(sr->elf_fd);
    if (sr->dwfl) dwfl_end(sr->dwfl);
    memset(sr, 0, sizeof(*sr));
    sr->elf_fd = -1;
}
