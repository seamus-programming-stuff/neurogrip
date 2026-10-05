#ifndef NEUROGRIP_SHM_H
#define NEUROGRIP_SHM_H

/* Local-PC shared-memory envelope v1. Motor I/O is deliberately absent.
 * Readers must take the publisher's named data mutex before copying all96B.
 * CRC checks alone do not replace that actual interprocess synchronization.
 */
#include "neurogrip_bus.h"

#define NG_SHM_SIZE 96u
#define NG_SHM_HEADER_SIZE 44u
#define NG_SHM_PAYLOAD_OFFSET 44u
#define NG_SHM_CRC_OFFSET 92u
#define NG_SHM_DEFAULT_NAME_W L"neurogrip_vision_v1"
#define NG_SHM_DEFAULT_DATA_MUTEX_W L"Local\\NeurogripData_6a19e1474facf1da4beb667e38374ac0384c3700034618a0ef1bf25f70952caa"

typedef struct {
    ng_vision_telemetry_t packet;
    uint32_t publisher_age_ms;
    uint32_t write_counter;
    uint32_t publisher_pid;
    bool actionable; /* Vision condition only; controller must separately arm. */
} ng_shm_snapshot_t;

static inline bool ng_decode_shm_snapshot(const uint8_t *p, size_t length,
                                          uint64_t local_monotonic_ns,
                                          uint16_t max_age_ms,
                                          ng_shm_snapshot_t *out) {
    ng_shm_snapshot_t value;
    uint64_t published_ns, age_ns, age_ms;
    uint32_t session;
    if (p == NULL || out == NULL || length != NG_SHM_SIZE || max_age_ms == 0u)
        return false;
    if (memcmp(p, "NGSHM1\0\0", 8) != 0 || ng_read_u16_le(p + 8) != 1u ||
        ng_read_u16_le(p + 10) != NG_SHM_HEADER_SIZE ||
        ng_read_u16_le(p + 12) != NG_CAN_FD_PAYLOAD_SIZE || ng_read_u16_le(p + 14) != 1u)
        return false;
    if (p[36] || p[37] || p[38] || p[39] || p[40] || p[41] || p[42] || p[43])
        return false;
    if (ng_read_u32_le(p + NG_SHM_CRC_OFFSET) != ng_crc32(p, NG_SHM_CRC_OFFSET))
        return false;
    session = ng_read_u32_le(p + 16);
    published_ns = ng_read_u64_le(p + 24);
    if (published_ns == 0u || local_monotonic_ns < published_ns)
        return false;
    age_ns = local_monotonic_ns - published_ns;
    age_ms = age_ns / UINT64_C(1000000) + (age_ns % UINT64_C(1000000) != 0u);
    if (age_ms >= max_age_ms)
        return false;
    if (!ng_decode_vision(p + NG_SHM_PAYLOAD_OFFSET, NG_CAN_FD_PAYLOAD_SIZE, &value.packet) ||
        value.packet.session_id != session ||
        age_ms + value.packet.capture_age_ms >= value.packet.valid_for_ms ||
        age_ms + value.packet.capture_age_ms >= max_age_ms)
        return false;
    value.publisher_age_ms = (uint32_t)age_ms;
    value.write_counter = ng_read_u32_le(p + 20);
    value.publisher_pid = ng_read_u32_le(p + 32);
    value.actionable = ng_vision_is_actionable(&value.packet, value.publisher_age_ms, 0u, max_age_ms);
    *out = value;
    return true;
}

#ifdef _WIN32
#include <windows.h>

typedef struct {
    HANDLE mapping;
    HANDLE data_mutex;
    const uint8_t *view;
} ng_shm_windows_reader_t;

/* Pass mutex_names(custom_name)[0] from Python for a custom mapping name. */
static inline bool ng_shm_windows_open(ng_shm_windows_reader_t *reader,
                                      const wchar_t *mapping_name,
                                      const wchar_t *data_mutex_name) {
    memset(reader, 0, sizeof(*reader));
    reader->data_mutex = CreateMutexW(NULL, FALSE, data_mutex_name);
    if (reader->data_mutex == NULL)
        return false;
    reader->mapping = OpenFileMappingW(FILE_MAP_READ, FALSE, mapping_name);
    if (reader->mapping == NULL) {
        CloseHandle(reader->data_mutex);
        reader->data_mutex = NULL;
        return false;
    }
    reader->view = (const uint8_t *)MapViewOfFile(reader->mapping, FILE_MAP_READ, 0, 0, NG_SHM_SIZE);
    if (reader->view == NULL) {
        CloseHandle(reader->mapping);
        CloseHandle(reader->data_mutex);
        reader->mapping = reader->data_mutex = NULL;
        return false;
    }
    return true;
}

/* Matches Python3.10+ Windows time.monotonic_ns (QueryPerformanceCounter). */
static inline bool ng_shm_windows_monotonic_ns(uint64_t *result) {
    LARGE_INTEGER ticks, frequency;
    uint64_t count, rate;
    if (!QueryPerformanceCounter(&ticks) || !QueryPerformanceFrequency(&frequency) ||
        ticks.QuadPart < 0 || frequency.QuadPart <= 0)
        return false;
    count = (uint64_t)ticks.QuadPart;
    rate = (uint64_t)frequency.QuadPart;
    if (rate > UINT64_MAX / UINT64_C(1000000000))
        return false;
    *result = (count / rate) * UINT64_C(1000000000) +
              ((count % rate) * UINT64_C(1000000000)) / rate;
    return true;
}

static inline bool ng_shm_windows_read(ng_shm_windows_reader_t *reader,
                                      uint16_t max_age_ms,
                                      ng_shm_snapshot_t *out) {
    uint8_t copy[NG_SHM_SIZE];
    uint64_t now_ns;
    DWORD wait_result;
    if (reader == NULL || reader->view == NULL || reader->data_mutex == NULL)
        return false;
    wait_result = WaitForSingleObject(reader->data_mutex, 50u);
    if (wait_result == WAIT_ABANDONED) {
        ReleaseMutex(reader->data_mutex);
        return false; /* A publisher died inside its protected copy. */
    }
    if (wait_result != WAIT_OBJECT_0)
        return false;
    memcpy(copy, reader->view, NG_SHM_SIZE);
    if (!ReleaseMutex(reader->data_mutex) || !ng_shm_windows_monotonic_ns(&now_ns))
        return false;
    return ng_decode_shm_snapshot(copy, NG_SHM_SIZE, now_ns, max_age_ms, out);
}

static inline void ng_shm_windows_close(ng_shm_windows_reader_t *reader) {
    if (reader == NULL)
        return;
    if (reader->view != NULL) UnmapViewOfFile(reader->view);
    if (reader->mapping != NULL) CloseHandle(reader->mapping);
    if (reader->data_mutex != NULL) CloseHandle(reader->data_mutex);
    memset(reader, 0, sizeof(*reader));
}
#endif /* _WIN32 */

#endif /* NEUROGRIP_SHM_H */
