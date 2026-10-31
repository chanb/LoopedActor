// 15-puzzle (4x4) tools for data_scripts/build_slidingpuzzle_banks.py. Goal: tile i+1 at cell i (row-major), blank (0)
// at cell 15. A board is a u64: nibble i = tile at cell i.
//   bfs <maxd> <dir>      exact BFS layers d=1..maxd -> <dir>/dNN.u64 (sorted raw u64). The graph is bipartite, so
//                         layer d+1 = unique(neighbours(layer d)) \ layer d-1.
//   sample <seed> <quota_file> <max_total>
//                         uniform random solvable boards, solved optimally (IDA*, additive 6-6-3 pattern databases with
//                         transpose reflection); keeps a board if its optimal length d has quota left. quota_file: 81
//                         ints, quota per d (-1 = no per-d limit). Stops when all finite quotas are met and (if
//                         max_total > 0) max_total boards are kept, or when only unlimited bins remain and max_total is
//                         reached. Prints "<hex> <d>" per kept board. Deterministic given seed (batches are solved in
//                         parallel, then kept in board-index order; duplicates are skipped).
// PDBs are cached in ./pdb663.bin (32 MB) next to the binary's working directory.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
typedef uint64_t u64; typedef uint8_t u8; typedef uint32_t u32;

static int nbr[16][4], nn[16];

// ---------------------------------------------------------------- BFS
static void rsort(u64 *a, u64 *tmp, size_t n) {
  for (int p = 0; p < 64; p += 8) { size_t cnt[256] = {0};
    for (size_t i = 0; i < n; i++) cnt[(a[i] >> p) & 255]++;
    size_t s = 0; for (int b = 0; b < 256; b++) { size_t c = cnt[b]; cnt[b] = s; s += c; }
    for (size_t i = 0; i < n; i++) tmp[cnt[(a[i] >> p) & 255]++] = a[i];
    u64 *t = a; a = tmp; tmp = t; }  // 8 passes: result ends in the original array
}
static inline int blank_of(u64 s) { for (int i = 0; i < 16; i++) if (((s >> (4 * i)) & 15) == 0) return i; return -1; }
static inline u64 slide(u64 s, int b, int c) { u64 v = (s >> (4 * c)) & 15; return (s & ~(15ULL << (4 * c))) | (v << (4 * b)); }
static u64 goal_board(void) { u64 g = 0; for (int i = 0; i < 15; i++) g |= (u64)(i + 1) << (4 * i); return g; }

static int bfs(int maxd, const char *dir) {
  u64 *prev = malloc(8), *cur = malloc(8); size_t np = 0, nc = 1; cur[0] = goal_board();
  for (int d = 1; d <= maxd; d++) {
    size_t cap = nc * 4; u64 *nb = malloc(cap * 8);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nc; i++) { u64 s = cur[i]; int b = blank_of(s);
      for (int j = 0; j < 4; j++) nb[i * 4 + j] = j < nn[b] ? slide(s, b, nbr[b][j]) : ~0ULL; }
    u64 *tmp = malloc(cap * 8); rsort(nb, tmp, cap); free(tmp);
    size_t m = 0, k = 0;
    for (size_t i = 0; i < cap && nb[i] != ~0ULL; i++) { u64 v = nb[i]; if (m && nb[m - 1] == v) continue;
      while (k < np && prev[k] < v) k++; if (k < np && prev[k] == v) continue; nb[m++] = v; }
    free(prev); prev = cur; np = nc; cur = realloc(nb, (m ? m : 1) * 8); nc = m;
    char fn[1024]; snprintf(fn, sizeof fn, "%s/d%02d.u64", dir, d); FILE *f = fopen(fn, "wb");
    if (!f || fwrite(cur, 8, nc, f) != nc) return 1; fclose(f);
    fprintf(stderr, "bfs d=%d %zu\n", d, nc);
  }
  return 0;
}

// ---------------------------------------------------------------- pattern databases
static const int PAT[3][6] = {{1, 2, 5, 6, 9, 13}, {3, 4, 7, 8, 11, 12}, {10, 14, 15, 0, 0, 0}}; static const int PK[3] = {6, 6, 3};
static u8 *pdb[3]; static int grp[16], slot[16];  // tile -> pattern, slot within the pattern
static int T[16], TT[16];                         // transpose of a cell; tile whose goal cell is the transpose of t's
static int goalcell(int t) { return t == 0 ? 15 : t - 1; }
// 0-1 BFS over (pattern tile cells, blank cell), packed 4 bits each; only moves of pattern tiles cost 1.
static void build(int p) {
  int k = PK[p]; size_t N = 1ULL << (4 * (k + 1)); u8 *d = malloc(N); memset(d, 255, N);
  u32 *q = malloc(sizeof(u32) * N / 2), *nq = malloc(sizeof(u32) * N / 2); size_t qn = 0, nqn = 0;
  u32 s0 = 15u << (4 * k); for (int i = 0; i < k; i++) s0 |= (u32)goalcell(PAT[p][i]) << (4 * i);
  d[s0] = 0; q[qn++] = s0;
  for (int lvl = 0; qn; lvl++) {
    for (size_t h = 0; h < qn; h++) { u32 s = q[h]; int b = (s >> (4 * k)) & 15, occ[16]; memset(occ, -1, sizeof occ);
      for (int i = 0; i < k; i++) occ[(s >> (4 * i)) & 15] = i;
      for (int j = 0; j < nn[b]; j++) { int c = nbr[b][j]; u32 t = s;
        if (occ[c] >= 0) t = (t & ~(15u << (4 * occ[c]))) | ((u32)b << (4 * occ[c]));
        t = (t & ~(15u << (4 * k))) | ((u32)c << (4 * k));
        if (occ[c] < 0) { if (d[t] <= lvl) continue; d[t] = lvl; q[qn++] = t; }
        else { if (d[t] != 255) continue; d[t] = lvl + 1; nq[nqn++] = t; } } }
    u32 *tq = q; q = nq; nq = tq; qn = nqn; nqn = 0;
  }
  size_t M = 1ULL << (4 * k); pdb[p] = malloc(M); memset(pdb[p], 255, M);
  for (size_t s = 0; s < N; s++) if (d[s] != 255 && d[s] < pdb[p][s & (M - 1)]) pdb[p][s & (M - 1)] = d[s];
  free(d); free(q); free(nq);
}
static void load_pdbs(void) {
  FILE *f = fopen("pdb663.bin", "rb");
  if (f) { for (int p = 0; p < 3; p++) { size_t M = 1ULL << (4 * PK[p]); pdb[p] = malloc(M); if (fread(pdb[p], 1, M, f) != M) exit(1); } fclose(f); return; }
  for (int p = 0; p < 3; p++) build(p);
  f = fopen("pdb663.bin", "wb"); for (int p = 0; p < 3; p++) fwrite(pdb[p], 1, 1ULL << (4 * PK[p]), f); fclose(f);
}

// ---------------------------------------------------------------- IDA*
typedef struct { int tile[16], blk; u32 ix[3], mx[3]; } Search;
static inline int H(const Search *S) {
  int a = pdb[0][S->ix[0]] + pdb[1][S->ix[1]] + pdb[2][S->ix[2]], b = pdb[0][S->mx[0]] + pdb[1][S->mx[1]] + pdb[2][S->mx[2]];
  return a > b ? a : b;
}
static int dfs(Search *S, int g, int bound, int from, int *next) {
  int h = H(S), f = g + h; if (f > bound) { if (f < *next) *next = f; return 0; } if (h == 0) return 1;
  int b = S->blk;
  for (int j = 0; j < nn[b]; j++) { int c = nbr[b][j]; if (c == from) continue; int t = S->tile[c];
    int p = grp[t], mt = TT[t], mp = grp[mt]; u32 o = S->ix[p], mo = S->mx[mp];  // tile t slides c -> b
    S->ix[p] += ((u32)b - (u32)c) << (4 * slot[t]); S->mx[mp] += ((u32)T[b] - (u32)T[c]) << (4 * slot[mt]);
    S->tile[b] = t; S->tile[c] = 0; S->blk = c;
    int r = dfs(S, g + 1, bound, b, next);
    S->tile[c] = t; S->tile[b] = 0; S->blk = b; S->ix[p] = o; S->mx[mp] = mo;
    if (r) return 1; }
  return 0;
}
static void init_search(Search *S, u64 s) {
  memset(S, 0, sizeof *S);
  for (int i = 0; i < 16; i++) { int t = (s >> (4 * i)) & 15; S->tile[i] = t; if (!t) { S->blk = i; continue; }
    S->ix[grp[t]] |= (u32)i << (4 * slot[t]); S->mx[grp[TT[t]]] |= (u32)T[i] << (4 * slot[TT[t]]); }
}
// Optimal length, or -1 if it exceeds cap.
static int solve(u64 s, int cap) {
  Search S; init_search(&S, s);
  for (int bound = H(&S); bound <= cap;) { int next = 1000; if (dfs(&S, 0, bound, -1, &next)) return bound; bound = next; }
  return -1;
}

// ---------------------------------------------------------------- uniform random boards
static inline u64 splitmix(u64 x) { x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31); }
static u64 random_board(u64 seed, u64 i) {
  int p[16]; for (int k = 0; k < 16; k++) p[k] = k;
  u64 r = splitmix(seed * 0x100000001b3ULL ^ i);
  for (int k = 15; k > 0; k--) { r = splitmix(r); int j = (int)(r % (u64)(k + 1)); int t = p[k]; p[k] = p[j]; p[j] = t; }
  int inv = 0, b = 0; for (int x = 0; x < 16; x++) { if (!p[x]) { b = x; continue; } for (int y = x + 1; y < 16; y++) inv += p[y] && p[x] > p[y]; }
  if ((inv + 3 - b / 4) & 1) { int x = p[0] ? 0 : 2, y = p[1] ? 1 : 2; int t = p[x]; p[x] = p[y]; p[y] = t; }  // fix parity: swap two tiles
  u64 s = 0; for (int k = 0; k < 16; k++) s |= (u64)p[k] << (4 * k); return s;
}
// open-addressing set of kept boards (boards are never 0)
static u64 *set; static size_t set_mask;
static int set_add(u64 v) { size_t h = splitmix(v) & set_mask; while (set[h]) { if (set[h] == v) return 0; h = (h + 1) & set_mask; } set[h] = v; return 1; }

static int sample(u64 seed, const char *quota_file, long max_total) {
  long quota[81]; FILE *f = fopen(quota_file, "r"); for (int d = 0; d <= 80; d++) if (fscanf(f, "%ld", &quota[d]) != 1) return 1; fclose(f);
  long kept = 0, cnt[81] = {0}, est = max_total; for (int d = 0; d <= 80; d++) if (quota[d] > 0) est += quota[d];
  set_mask = 1; while (set_mask < (size_t)est * 4) set_mask <<= 1; set = calloc(set_mask, 8); set_mask--;
  const size_t B = 1 << 16; u64 *bd = malloc(B * 8); int *dd = malloc(B * sizeof(int));
  for (u64 base = 0;; base += B) {
    int cap = -1, finite_left = 0;  // deepest bin that can still take boards
    for (int d = 0; d <= 80; d++) { int open = quota[d] < 0 ? (max_total <= 0 || kept < max_total) : cnt[d] < quota[d];
      if (open) cap = d; if (quota[d] > 0 && cnt[d] < quota[d]) finite_left = 1; }
    if (cap < 0 || (!finite_left && max_total > 0 && kept >= max_total)) break;
    #pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < B; i++) { bd[i] = random_board(seed, base + i); dd[i] = solve(bd[i], cap); }
    for (size_t i = 0; i < B; i++) { int d = dd[i]; if (d < 0) continue;
      if (quota[d] == 0 || (quota[d] > 0 && cnt[d] >= quota[d]) || (quota[d] < 0 && max_total > 0 && kept >= max_total)) continue;
      if (!set_add(bd[i])) continue;
      cnt[d]++; kept++; printf("%016llx %d\n", (unsigned long long)bd[i], d); }
    fflush(stdout); fprintf(stderr, "sample boards=%llu kept=%ld cap=%d\n", (unsigned long long)(base + B), kept, cap);
  }
  return 0;
}

int main(int argc, char **argv) {
  for (int i = 0; i < 16; i++) { int r = i / 4, c = i % 4; nn[i] = 0;
    if (r > 0) nbr[i][nn[i]++] = i - 4; if (c < 3) nbr[i][nn[i]++] = i + 1; if (r < 3) nbr[i][nn[i]++] = i + 4; if (c > 0) nbr[i][nn[i]++] = i - 1;
    T[i] = c * 4 + r; }
  for (int t = 1; t < 16; t++) TT[t] = T[t - 1] + 1;
  for (int p = 0; p < 3; p++) for (int i = 0; i < PK[p]; i++) { grp[PAT[p][i]] = p; slot[PAT[p][i]] = i; }
  if (argc == 4 && !strcmp(argv[1], "bfs")) return bfs(atoi(argv[2]), argv[3]);
  if (argc == 5 && !strcmp(argv[1], "sample")) { load_pdbs(); return sample(strtoull(argv[2], 0, 10), argv[3], atol(argv[4])); }
  if (argc == 2 && !strcmp(argv[1], "solve")) { load_pdbs(); char line[64];  // stdin hex boards -> "<hex> <d>"
    while (fgets(line, sizeof line, stdin)) { u64 s = strtoull(line, 0, 16); if (s) printf("%016llx %d\n", (unsigned long long)s, solve(s, 1000)); }
    return 0; }
  fprintf(stderr, "usage: %s bfs <maxd> <dir> | sample <seed> <quota_file> <max_total> | solve\n", argv[0]); return 2;
}
