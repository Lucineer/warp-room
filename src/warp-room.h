#ifndef WARP_ROOM_H
#define WARP_ROOM_H

#include <stdint.h>
#include <stddef.h>

/* ---- Room profiles ---- */

#define NUM_ROOMS    4
#define VOCAB_DIM    97   /* unigrams + bigrams (kept for text features) */

/* P48: 8 components per uint64. We use ceiling(VOCAB_DIM/8) = 13 vectors */
#define P48_DIMS    13    /* 13 × 8 = 104 components, 97 used, 7 spare */

/* Room IDs */
enum room_id {
    ROOM_EDGE     = 0,
    ROOM_RESEARCH = 1,
    ROOM_FLEET    = 2,
    ROOM_JC1      = 3,
};

/* A room's profile: P48-encoded + handler */
struct room {
    enum room_id id;
    char         name[32];
    uint64_t     vector[P48_DIMS];  /* P48 exact encoding, not floats */
    void         (*handler)(const char *query, char *reply, size_t rlen);
};

/* Shared memory header (mmap'd) */
struct room_table {
    volatile uint32_t version; /* incremented on update */
    uint32_t          num_rooms;
    struct room       rooms[NUM_ROOMS];
};

/* ---- API ---- */
void        wr_init(void);          /* open mmap segment or create */
void        wr_train(const char *text, enum room_id room);
enum room_id wr_classify(const char *text, float *confidence);
void        wr_stats(void);         /* print to stdout */

/* ---- P48 exact classification (scalar, exact integer distance) ---- */
enum room_id wr_classify_p48(const char *text, int *exact_dist);

/* ---- NEON SIMD batch nearest-neighbor (ARM64 only) ---- */
#ifdef __aarch64__
enum room_id wr_classify_p48_neon(const char *text);
#endif

#endif /* WARP_ROOM_H */ /* WARP_ROOM_H */
