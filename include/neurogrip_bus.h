#ifndef NEUROGRIP_BUS_H
#define NEUROGRIP_BUS_H

/* Neurogrip vision protocol v1. This header performs no actuator/bus I/O.
 * Wire bytes are little endian; float encoding is IEEE-754 binary32.
 * Do not send this C structure as a packed frame. Decode the explicit payload.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#define NG_SCHEMA_VERSION 1u
#define NG_CAN_FD_PAYLOAD_SIZE 48u
#define NG_MODE_SIMULATION 0u
#define NG_MODE_LIVE 1u
#define NG_DIRECTION_BACKWARDS 0u
#define NG_DIRECTION_UPRIGHT 1u
#define NG_DIRECTION_FORWARDS 2u
#define NG_DIRECTION_UNKNOWN 255u
#define NG_FLAG_STRIKE_PERMIT 0x01u
#define NG_FLAG_SCALE_VALID 0x02u
#define NG_GAP_UNKNOWN_MM (-1.0f)

#define NG_BLOCK_STALE_FRAME (UINT32_C(1) << 0)
#define NG_BLOCK_UNKNOWN_DEPTH (UINT32_C(1) << 1)
#define NG_BLOCK_UNCALIBRATED (UINT32_C(1) << 2)
#define NG_BLOCK_LOW_QUALITY (UINT32_C(1) << 3)
#define NG_BLOCK_TARGET_MISSING (UINT32_C(1) << 4)
#define NG_BLOCK_FINGER_MISSING (UINT32_C(1) << 5)
#define NG_BLOCK_UNREACHABLE (UINT32_C(1) << 6)
#define NG_BLOCK_SIMULATION (UINT32_C(1) << 7)
#define NG_BLOCK_DIRECTION_UNKNOWN (UINT32_C(1) << 8)
#define NG_BLOCK_CONTROLLER_UNARMED (UINT32_C(1) << 9)
#define NG_BLOCK_INVALID_GEOMETRY (UINT32_C(1) << 10)
#define NG_BLOCK_OCCLUDED (UINT32_C(1) << 11)
#define NG_BLOCK_CALIBRATION_UNVALIDATED (UINT32_C(1) << 12)
#define NG_BLOCK_NO_STROKE_CALIBRATION (UINT32_C(1) << 13)
#define NG_BLOCK_CAMERA_MOVED (UINT32_C(1) << 14)
#define NG_BLOCK_DEPTH_OUT_OF_RANGE (UINT32_C(1) << 15)
#define NG_KNOWN_REASON_MASK UINT32_C(0x0000FFFF)

typedef struct {
    uint32_t session_id;
    uint32_t sequence;
    uint64_t timestamp_ms;
    uint16_t capture_age_ms;
    uint16_t valid_for_ms;
    uint8_t mode;
    uint8_t direction;
    bool strike_permit;
    bool scale_valid;
    bool gap_known;
    float gap_mm;
    uint8_t quality_u8; /* Heuristic only; quality = quality_u8 / 255. */
    uint32_t block_reasons;
} ng_vision_telemetry_t;

typedef struct {
    bool paired;
    bool sequence_seen;
    uint32_t session_id;
    uint32_t last_sequence;
} ng_sequence_tracker_t;

static inline uint16_t ng_read_u16_le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static inline uint32_t ng_read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t ng_read_u64_le(const uint8_t *p) {
    return (uint64_t)ng_read_u32_le(p) | ((uint64_t)ng_read_u32_le(p + 4) << 32);
}

/* Same CRC-32/ISO-HDLC as Python zlib.crc32, not authentication. */
static inline uint32_t ng_crc32(const uint8_t *data, size_t length) {
    uint32_t crc = UINT32_C(0xFFFFFFFF);
    size_t i;
    unsigned bit;
    for (i = 0; i < length; ++i) {
        crc ^= data[i];
        for (bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ ((crc & 1u) ? UINT32_C(0xEDB88320) : 0u);
        }
    }
    return crc ^ UINT32_C(0xFFFFFFFF);
}

static inline bool ng_decode_vision(const uint8_t *p, size_t length,
                                    ng_vision_telemetry_t *out) {
    uint32_t gap_bits;
    ng_vision_telemetry_t decoded;
    if (p == NULL || out == NULL || length != NG_CAN_FD_PAYLOAD_SIZE || sizeof(float) != 4u)
        return false;
    if (memcmp(p, "NGV1", 4) != 0 || p[4] != NG_SCHEMA_VERSION ||
        p[5] > NG_MODE_LIVE || (p[7] & ~0x03u) != 0u)
        return false;
    if (p[6] != NG_DIRECTION_BACKWARDS && p[6] != NG_DIRECTION_UPRIGHT &&
        p[6] != NG_DIRECTION_FORWARDS && p[6] != NG_DIRECTION_UNKNOWN)
        return false;
    if (p[33] || p[34] || p[35] || p[40] || p[41] || p[42] || p[43])
        return false;
    if (ng_read_u32_le(p + 44) != ng_crc32(p, 44))
        return false;
    decoded.mode = p[5];
    decoded.direction = p[6];
    decoded.strike_permit = (p[7] & NG_FLAG_STRIKE_PERMIT) != 0u;
    decoded.scale_valid = (p[7] & NG_FLAG_SCALE_VALID) != 0u;
    decoded.session_id = ng_read_u32_le(p + 8);
    decoded.sequence = ng_read_u32_le(p + 12);
    decoded.timestamp_ms = ng_read_u64_le(p + 16);
    decoded.capture_age_ms = ng_read_u16_le(p + 24);
    decoded.valid_for_ms = ng_read_u16_le(p + 26);
    gap_bits = ng_read_u32_le(p + 28);
    memcpy(&decoded.gap_mm, &gap_bits, sizeof(decoded.gap_mm));
    decoded.gap_known = decoded.gap_mm != NG_GAP_UNKNOWN_MM;
    decoded.quality_u8 = p[32];
    decoded.block_reasons = ng_read_u32_le(p + 36);
    if (!isfinite(decoded.gap_mm) ||
        (decoded.gap_known && (decoded.gap_mm < 0.0f || decoded.gap_mm > 1000000.0f)) ||
        decoded.valid_for_ms == 0u ||
        (decoded.block_reasons & ~NG_KNOWN_REASON_MASK) != 0u)
        return false;
    if (decoded.strike_permit &&
        (decoded.mode != NG_MODE_LIVE || decoded.direction == NG_DIRECTION_UNKNOWN ||
         !decoded.scale_valid || !decoded.gap_known || decoded.quality_u8 == 0u ||
         decoded.block_reasons != 0u || decoded.capture_age_ms >= decoded.valid_for_ms))
        return false;
    *out = decoded;
    return true;
}

/* A true result is a vision condition only. It does not authorize actuation.
 * Add a VERIFIED bound for transport delay, not a guessed zero for hardware.
 * Source timestamp_ms must not be compared directly with receiver uptime.
 */
static inline bool ng_vision_is_actionable(const ng_vision_telemetry_t *packet,
                                          uint32_t elapsed_since_receipt_ms,
                                          uint32_t transport_delay_budget_ms,
                                          uint16_t local_max_age_ms) {
    uint64_t age;
    uint16_t age_limit;
    if (packet == NULL || local_max_age_ms == 0u)
        return false;
    age = (uint64_t)packet->capture_age_ms + elapsed_since_receipt_ms + transport_delay_budget_ms;
    age_limit = packet->valid_for_ms < local_max_age_ms ? packet->valid_for_ms : local_max_age_ms;
    return packet->mode == NG_MODE_LIVE && packet->strike_permit &&
           packet->direction != NG_DIRECTION_UNKNOWN && packet->scale_valid &&
           packet->gap_known && packet->quality_u8 > 0u && packet->block_reasons == 0u &&
           age < age_limit;
}

/* Call only after manual bench pairing or an explicit fresh-session handshake.
 * A reboot/session change must leave the local actuator disarmed.
 */
static inline void ng_sequence_reset(ng_sequence_tracker_t *state, uint32_t session_id) {
    state->paired = true;
    state->sequence_seen = false;
    state->session_id = session_id;
    state->last_sequence = 0u;
}

static inline bool ng_sequence_accept(ng_sequence_tracker_t *state,
                                      const ng_vision_telemetry_t *packet) {
    uint32_t delta;
    if (state == NULL || packet == NULL || !state->paired || state->session_id != packet->session_id)
        return false;
    if (state->sequence_seen) {
        delta = packet->sequence - state->last_sequence;
        if (delta == 0u || delta >= UINT32_C(0x80000000))
            return false;
    }
    state->last_sequence = packet->sequence;
    state->sequence_seen = true;
    return true;
}

#endif /* NEUROGRIP_BUS_H */
