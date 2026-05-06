/* warp-room.c — Subroutine-threaded tile classifier with P48 exact encoding */
#define _GNU_SOURCE
#include "warp-room.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <ctype.h>

/* ---- Shared memory segment ---- */
static struct room_table *g_table = NULL;
static int g_shm_fd = -1;

/* ---- P48 inline helpers ---- */
static void float_to_p48_component(float val, int index, uint64_t *vec) {
    int v = (int)(val * 63.0f + 0.5f);
    if (v < 0) v = 0;
    if (v > 63) v = 63;
    int vec_idx = index / 8;
    int bit_off = (index % 8) * 6;
    vec[vec_idx] &= ~((uint64_t)0x3F << bit_off);
    vec[vec_idx] |= ((uint64_t)v & 0x3F) << bit_off;
}

static inline int p48_dot(const uint64_t *a, const uint64_t *b, int n) {
    int sum = 0;
    for (int v = 0; v < n; v++) {
        uint64_t pa = a[v], pb = b[v];
        for (int i = 0; i < 8; i++) {
            int ca = (pa >> (6 * i)) & 0x3F;
            int cb = (pb >> (6 * i)) & 0x3F;
            sum += ca * cb;
        }
    }
    return sum;
}

static inline int p48_dist_sq(const uint64_t *a, const uint64_t *b, int n) {
    int sum = 0;
    for (int v = 0; v < n; v++) {
        uint64_t pa = a[v], pb = b[v];
        for (int i = 0; i < 8; i++) {
            int ca = (pa >> (6 * i)) & 0x3F;
            int cb = (pb >> (6 * i)) & 0x3F;
            int d = ca - cb;
            sum += d * d;
        }
    }
    return sum;
}

/* ---- Bag-of-features extraction with real keywords ---- */

/* Keyword-to-index mapping: ~90 keywords, each gets a dimension */
/* Edge keywords (indices 0-25) */
static const char *edge_kw[] = {
    "jetson", "cpu", "gpu", "memory", "temperature", "load", "uptime",
    "disk", "thermal", "fan", "power", "nvidia", "cuda", "nvcc",
    "arm64", "aarch64", "swap", "network", "interface", "sensor",
    "telemetry", "hardware", "clock", "throttle", "edge", "device"
};
#define EDGE_KW_COUNT 26

/* Research keywords (indices 26-52) */
static const char *research_kw[] = {
    "research", "paper", "study", "findings", "analysis", "experiment",
    "benchmark", "performance", "test", "comparison", "evaluation",
    "learn", "training", "dataset", "model", "inference", "llm",
    "neural", "embedding", "vector", "similarity", "tile",
    "investigation", "methodology", "result", "conclusion", "algorithm"
};
#define RESEARCH_KW_COUNT 27

/* Fleet keywords (indices 53-76) */
static const char *fleet_kw[] = {
    "fleet", "agent", "oracle", "forge", "vessel", "bottle", "matrix",
    "heartbeat", "sync", "mesh", "iron", "coordination", "bridge",
    "pki", "cert", "trust", "deadman", "migration", "protocol",
    "lighthouse", "beacon", "dm", "conduit", "message"
};
#define FLEET_KW_COUNT 24

/* JC1 keywords (indices 77-89) */
static const char *jc1_kw[] = {
    "jc1", "jetsonclaw", "plato", "evennia", "flato", "mythos",
    "cocapn", "libllama", "gguf", "sovereign", "infer", "think", "vessel"
};
#define JC1_KW_COUNT 13

/* Total = 26 + 27 + 24 + 13 = 90 (within VOCAB_DIM = 97) */
#define TOTAL_KW_COUNT (EDGE_KW_COUNT + RESEARCH_KW_COUNT + FLEET_KW_COUNT + JC1_KW_COUNT)

static int keyword_index(const char *word) {
    /* Linear search through keyword lists */
    for (int i = 0; i < EDGE_KW_COUNT; i++)
        if (strcmp(word, edge_kw[i]) == 0) return i;
    for (int i = 0; i < RESEARCH_KW_COUNT; i++)
        if (strcmp(word, research_kw[i]) == 0) return EDGE_KW_COUNT + i;
    for (int i = 0; i < FLEET_KW_COUNT; i++)
        if (strcmp(word, fleet_kw[i]) == 0) return EDGE_KW_COUNT + RESEARCH_KW_COUNT + i;
    for (int i = 0; i < JC1_KW_COUNT; i++)
        if (strcmp(word, jc1_kw[i]) == 0)
            return EDGE_KW_COUNT + RESEARCH_KW_COUNT + FLEET_KW_COUNT + i;
    return -1;
}

static void text_features(const char *text, float *vec) {
    memset(vec, 0, VOCAB_DIM * sizeof(float));

    if (!text || !*text) return;

    /* Tokenize: split on whitespace and punctuation */
    char *buf = strdup(text);
    if (!buf) return;

    /* Collect term frequencies */
    int tf[VOCAB_DIM] = {0};
    int total_terms = 0;

    char *tok = strtok(buf, " \t\n\r.,!?;:\"'()[]{}<>/\\-@#$%^&*+=~`|");
    while (tok) {
        /* Lowercase */
        for (char *p = tok; *p; p++) *p = tolower((unsigned char)*p);
        int idx = keyword_index(tok);
        if (idx >= 0 && idx < VOCAB_DIM) {
            tf[idx]++;
            total_terms++;
        }
        tok = strtok(NULL, " \t\n\r.,!?;:\"'()[]{}<>/\\-@#$%^&*+=~`|");
    }
    free(buf);

    if (total_terms == 0) return;

    /* TF weighting: log-normalized */
    const float idf = 1.0f; /* Placeholder — we'd compute from corpus */
    for (int i = 0; i < VOCAB_DIM; i++) {
        if (tf[i] > 0) {
            vec[i] = (1.0f + logf((float)tf[i])) * idf;
        }
    }
}

/* ---- Training: online learning via EMA, stored as P48 ---- */
void wr_train(const char *text, enum room_id room) {
    if (!g_table || room >= NUM_ROOMS) return;
    float fvec[VOCAB_DIM];
    text_features(text, fvec);

    float norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += fvec[i] * fvec[i];
    if (norm < 1e-8f) return;
    norm = sqrtf(norm);
    for (int i = 0; i < VOCAB_DIM; i++) fvec[i] /= norm;

    /* Decode current P48 back to float for EMA */
    float current[VOCAB_DIM] = {0};
    struct room *r = &g_table->rooms[room];
    for (int v = 0; v < P48_DIMS; v++) {
        uint64_t p = r->vector[v];
        for (int i = 0; i < 8; i++) {
            int idx = v * 8 + i;
            if (idx >= VOCAB_DIM) break;
            current[idx] = (p >> (6 * i)) & 0x3F;
            current[idx] /= 63.0f;
        }
    }

    /* EMA update in float space */
    const float lr = 0.3f;
    float updated[VOCAB_DIM];
    for (int i = 0; i < VOCAB_DIM; i++)
        updated[i] = (1.0f - lr) * current[i] + lr * fvec[i];

    norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += updated[i] * updated[i];
    norm = sqrtf(norm);
    if (norm > 1e-8f)
        for (int i = 0; i < VOCAB_DIM; i++) updated[i] /= norm;

    /* Quantize back to P48 */
    memset(r->vector, 0, P48_DIMS * sizeof(uint64_t));
    for (int i = 0; i < VOCAB_DIM; i++)
        float_to_p48_component(updated[i], i, r->vector);

    __sync_fetch_and_add(&g_table->version, 1);
}

/* ---- Float-based classification (legacy) ---- */
enum room_id wr_classify(const char *text, float *confidence) {
    if (!g_table) return ROOM_EDGE;

    float vec[VOCAB_DIM];
    text_features(text, vec);

    float norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += vec[i] * vec[i];
    if (norm < 1e-8f) return ROOM_EDGE;
    norm = sqrtf(norm);
    for (int i = 0; i < VOCAB_DIM; i++) vec[i] /= norm;

    enum room_id best = ROOM_EDGE;
    float best_score = -1.0f;
    for (int ri = 0; ri < NUM_ROOMS; ri++) {
        float current[VOCAB_DIM] = {0};
        for (int v = 0; v < P48_DIMS; v++) {
            uint64_t p = g_table->rooms[ri].vector[v];
            for (int i = 0; i < 8; i++) {
                int idx = v * 8 + i;
                if (idx >= VOCAB_DIM) break;
                current[idx] = (p >> (6 * i)) & 0x3F;
                current[idx] /= 63.0f;
            }
        }
        float dot = 0.0f;
        for (int j = 0; j < VOCAB_DIM; j++)
            dot += vec[j] * current[j];
        if (dot > best_score) {
            best_score = dot;
            best = (enum room_id)ri;
        }
    }
    if (confidence) *confidence = best_score;
    return best;
}

/* ---- P48 exact classification (scalar) ---- */
enum room_id wr_classify_p48(const char *text, int *exact_dist) {
    if (!g_table) return ROOM_EDGE;

    float fvec[VOCAB_DIM];
    text_features(text, fvec);

    float norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += fvec[i] * fvec[i];
    if (norm < 1e-8f) return ROOM_EDGE;
    norm = sqrtf(norm);
    for (int i = 0; i < VOCAB_DIM; i++) fvec[i] /= norm;

    /* Quantize to P48 */
    uint64_t qvec[P48_DIMS] = {0};
    for (int i = 0; i < VOCAB_DIM; i++)
        float_to_p48_component(fvec[i], i, qvec);

    /* Exact integer nearest-neighbor */
    enum room_id best = ROOM_EDGE;
    int best_dist = INT32_MAX;
    for (int ri = 0; ri < NUM_ROOMS; ri++) {
        int d = p48_dist_sq(qvec, g_table->rooms[ri].vector, P48_DIMS);
        if (d < best_dist) {
            best_dist = d;
            best = (enum room_id)ri;
        }
    }
    if (exact_dist) *exact_dist = best_dist;
    return best;
}

/* ---- P48 exact classification with NEON batch ---- */
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#include "p48_neon.h"

enum room_id wr_classify_p48_neon(const char *text) {
    if (!g_table) return ROOM_EDGE;

    float fvec[VOCAB_DIM];
    text_features(text, fvec);

    float norm = 0.0f;
    for (int i = 0; i < VOCAB_DIM; i++) norm += fvec[i] * fvec[i];
    if (norm < 1e-8f) return ROOM_EDGE;
    norm = sqrtf(norm);
    for (int i = 0; i < VOCAB_DIM; i++) fvec[i] /= norm;

    /* Quantize query to P48 */
    uint64_t qvec_packed[P48_DIMS] = {0};
    for (int i = 0; i < VOCAB_DIM; i++)
        float_to_p48_component(fvec[i], i, qvec_packed);

    /* Pre-unpack query to bytes */
    uint8_t query_bytes[P48_DIMS * 8];
    p48_unpack_bytes(qvec_packed, query_bytes, P48_DIMS);

    /* Pre-unpack all room vectors to bytes (cache: compute once per classify) */
    uint8_t room_bytes[NUM_ROOMS * P48_DIMS * 8];
    for (int ri = 0; ri < NUM_ROOMS; ri++) {
        p48_unpack_bytes(g_table->rooms[ri].vector,
                         room_bytes + ri * P48_DIMS * 8, P48_DIMS);
    }

    /* NEON batch NN */
    int best = p48_neon_batch_nn(query_bytes, room_bytes,
                                  NUM_ROOMS, P48_DIMS * 8);
    return (enum room_id)best;
}
#endif

/* ---- init: create or open shared memory ---- */
void wr_init(void) {
    g_shm_fd = shm_open("/warp-room-vectors", O_CREAT | O_RDWR, 0666);
    if (g_shm_fd < 0) { perror("shm_open"); return; }
    ftruncate(g_shm_fd, sizeof(struct room_table));

    g_table = mmap(NULL, sizeof(struct room_table),
                   PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (g_table == MAP_FAILED) { perror("mmap"); g_table = NULL; return; }

    if (g_table->version == 0) {
        g_table->num_rooms = NUM_ROOMS;

        strcpy(g_table->rooms[ROOM_EDGE].name,     "edge");
        strcpy(g_table->rooms[ROOM_RESEARCH].name, "research");
        strcpy(g_table->rooms[ROOM_FLEET].name,    "fleet");
        strcpy(g_table->rooms[ROOM_JC1].name,      "jc1");

        /* Seed with keyword vectors, L2-normalized (same normalization as wr_train) */
        {
            float fvec[VOCAB_DIM];
            float norm;

            /* Edge: high weight on tokens 0-25 */
            memset(fvec, 0, sizeof(fvec));
            for (int i = 0; i < EDGE_KW_COUNT; i++) fvec[i] = 1.0f;
            norm = 0.0f;
            for (int i = 0; i < VOCAB_DIM; i++) norm += fvec[i] * fvec[i];
            norm = sqrtf(norm);
            for (int i = 0; i < VOCAB_DIM; i++) fvec[i] /= norm;
            for (int i = 0; i < VOCAB_DIM; i++)
                float_to_p48_component(fvec[i], i, g_table->rooms[ROOM_EDGE].vector);

            /* Research: high weight on tokens 26-52 */
            memset(fvec, 0, sizeof(fvec));
            for (int i = 0; i < RESEARCH_KW_COUNT; i++) fvec[EDGE_KW_COUNT + i] = 1.0f;
            norm = 0.0f;
            for (int i = 0; i < VOCAB_DIM; i++) norm += fvec[i] * fvec[i];
            norm = sqrtf(norm);
            for (int i = 0; i < VOCAB_DIM; i++) fvec[i] /= norm;
            for (int i = 0; i < VOCAB_DIM; i++)
                float_to_p48_component(fvec[i], i, g_table->rooms[ROOM_RESEARCH].vector);

            /* Fleet: high weight on tokens 53-76 */
            memset(fvec, 0, sizeof(fvec));
            for (int i = 0; i < FLEET_KW_COUNT; i++) fvec[EDGE_KW_COUNT + RESEARCH_KW_COUNT + i] = 1.0f;
            norm = 0.0f;
            for (int i = 0; i < VOCAB_DIM; i++) norm += fvec[i] * fvec[i];
            norm = sqrtf(norm);
            for (int i = 0; i < VOCAB_DIM; i++) fvec[i] /= norm;
            for (int i = 0; i < VOCAB_DIM; i++)
                float_to_p48_component(fvec[i], i, g_table->rooms[ROOM_FLEET].vector);

            /* JC1: high weight on tokens 77-89 */
            memset(fvec, 0, sizeof(fvec));
            for (int i = 0; i < JC1_KW_COUNT; i++) fvec[EDGE_KW_COUNT + RESEARCH_KW_COUNT + FLEET_KW_COUNT + i] = 1.0f;
            norm = 0.0f;
            for (int i = 0; i < VOCAB_DIM; i++) norm += fvec[i] * fvec[i];
            norm = sqrtf(norm);
            for (int i = 0; i < VOCAB_DIM; i++) fvec[i] /= norm;
            for (int i = 0; i < VOCAB_DIM; i++)
                float_to_p48_component(fvec[i], i, g_table->rooms[ROOM_JC1].vector);
        }

        g_table->version = 1;
        printf("warp-room: created fresh P48 shared memory (version 1)\n");
    } else {
        printf("warp-room: attached to existing shared memory (version %u)\n",
               g_table->version);
    }
}

/* ---- CLI ---- */
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    wr_init();

    if (argc >= 3 && strcmp(argv[1], "--neon-gpu") == 0) {
        /* Shell out to nvidia-smi for GPU telemetry — like rightnow-cli */
        FILE *fp = popen("nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1", "r");
        if (!fp) { printf("{\"error\": \"nvidia-smi failed\"}\n"); return 1; }
        char buf[256];
        if (!fgets(buf, sizeof(buf), fp)) { buf[0] = 0; }
        pclose(fp);
        printf("%s", buf);
    } else if (argc >= 3 && strcmp(argv[1], "--infer") == 0) {
        float conf = 0.0f;
        enum room_id r = wr_classify(argv[2], &conf);
        printf("{\"room\": \"%s\", \"confidence\": %.4f}\n",
               g_table ? g_table->rooms[r].name : "unknown", conf);
    } else if (argc >= 3 && strcmp(argv[1], "--infer-neon") == 0) {
#if defined(__aarch64__) || defined(__ARM_NEON)
        enum room_id r = wr_classify_p48_neon(argv[2]);
        printf("{\"room\": \"%s\", \"method\": \"neon\"}\n",
               g_table ? g_table->rooms[r].name : "unknown");
#else
        printf("{\"error\": \"NEON not available\"}\n");
#endif
    } else if (argc >= 3 && strcmp(argv[1], "--infer-p48") == 0) {
        int dist;
        enum room_id r = wr_classify_p48(argv[2], &dist);
        printf("{\"room\": \"%s\", \"exact_distance\": %d}\n",
               g_table ? g_table->rooms[r].name : "unknown", dist);
    } else if (argc >= 3 && strcmp(argv[1], "--train") == 0) {
        if (argc < 4) { fprintf(stderr, "usage: --train <text> <room>\n"); return 1; }
        enum room_id id = ROOM_EDGE;
        for (int i = 0; i < NUM_ROOMS; i++)
            if (strcmp(argv[3], g_table->rooms[i].name) == 0) id = (enum room_id)i;
        wr_train(argv[2], id);
        printf("trained on room %s\n", argv[3]);
    } else {
        printf("Usage:\n");
        printf("  warp-room --infer <text>       (float cosine similarity)\n");
        printf("  warp-room --infer-p48 <text>   (P48 exact integer distance)\n");
        printf("  warp-room --infer-neon <text>  (NEON P48 batch NN)\n");
        printf("  warp-room --neon-gpu <text>    (NEON + GPU telemetry)\n");
        printf("  warp-room --train <text> <room>\n");
    }
    return 0;
}
