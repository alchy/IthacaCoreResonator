#pragma once
/*
 * sysex.h
 * ────────
 * IthacaCoreResonator MIDI SysEx codec.
 * Full protocol spec: docs/SYSEX.md
 *
 * Header:  F0 7D 49 43 52  [type]  [payload]  F7
 * Mfr ID:  0x7D  (non-commercial / educational)
 * Sig:     0x49 0x43 0x52  ("ICR")
 *
 * Message types:
 *   0x01 SET_PARAM       — set one parameter (float, 7-bit encoded)
 *   0x02 GET_PARAM       — request one parameter value
 *   0x03 PARAM_RESPONSE  — response to GET_PARAM (device → host)
 *   0x04 SET_ALL         — set all R/W parameters (full SynthConfig dump)
 *   0x05 REQUEST_ALL     — request full dump
 *   0x06 ALL_PARAMS_DUMP — response: full SynthConfig (device → host)
 *
 * Param ID (16 bit):
 *   bits 15–12: category (0–9)
 *   bits 11– 0: index within category
 *   Encoded as 3 × 7-bit bytes (id0/id1/id2).
 *
 * Float encoding:
 *   IEEE 754 uint32 split into 5 × 7-bit bytes (v0..v4).
 *   Int params (pitch_glide_vel_thresh) encoded as float cast.
 */

#include "synth_config.h"
#include "voice_manager.h"
#include <cstdint>
#include <cstring>
#include <vector>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr uint8_t SYSEX_MFR   = 0x7D;
static constexpr uint8_t SYSEX_SIG0  = 0x49;  // 'I'
static constexpr uint8_t SYSEX_SIG1  = 0x43;  // 'C'
static constexpr uint8_t SYSEX_SIG2  = 0x52;  // 'R'

static constexpr uint8_t SYSEX_SET_PARAM      = 0x01;
static constexpr uint8_t SYSEX_GET_PARAM      = 0x02;
static constexpr uint8_t SYSEX_PARAM_RESPONSE = 0x03;
static constexpr uint8_t SYSEX_SET_ALL        = 0x04;
static constexpr uint8_t SYSEX_REQUEST_ALL    = 0x05;
static constexpr uint8_t SYSEX_ALL_PARAMS_DUMP= 0x06;

// Param IDs — cat << 12 | index
// 0x3 STEREO
static constexpr uint16_t PID_PAN_SPREAD           = 0x3000;
static constexpr uint16_t PID_STEREO_DECORR        = 0x3001;
static constexpr uint16_t PID_STEREO_BOOST         = 0x3002;
static constexpr uint16_t PID_PAN_TILT             = 0x3003;
static constexpr uint16_t PID_DECORR_MIDI_LO       = 0x3004;
static constexpr uint16_t PID_DECORR_MIDI_HI       = 0x3005;
static constexpr uint16_t PID_DECORR_MAX           = 0x3006;
// 0x4 TIMBRE
static constexpr uint16_t PID_BEAT_SCALE           = 0x4000;
static constexpr uint16_t PID_HARMONIC_BRIGHTNESS  = 0x4001;
static constexpr uint16_t PID_EQ_STRENGTH          = 0x4002;
static constexpr uint16_t PID_EQ_FREQ_MIN          = 0x4003;
static constexpr uint16_t PID_PITCH_GLIDE          = 0x4004;
static constexpr uint16_t PID_PITCH_GLIDE_TAU_MS   = 0x4005;
static constexpr uint16_t PID_PITCH_GLIDE_VEL_THRESH= 0x4006;  // int, encoded as float
// 0x5 LEVEL/ENV
static constexpr uint16_t PID_TARGET_RMS           = 0x5000;
static constexpr uint16_t PID_VEL_GAMMA            = 0x5001;
static constexpr uint16_t PID_NOISE_LEVEL          = 0x5002;
static constexpr uint16_t PID_ONSET_MS             = 0x5003;
static constexpr uint16_t PID_LONGITUDINAL_PRECURSOR= 0x5004;
static constexpr uint16_t PID_RENDER_REF_DURATION   = 0x5005;

// ── Codec helpers ─────────────────────────────────────────────────────────────

// Encode uint16 param_id → 3 × 7-bit bytes [id0, id1, id2]
inline void encodeParamId(uint16_t id, uint8_t& id0, uint8_t& id1, uint8_t& id2) {
    id0 = (id >> 14) & 0x03;
    id1 = (id >>  7) & 0x7F;
    id2 =  id        & 0x7F;
}

// Decode 3 × 7-bit bytes → uint16 param_id
inline uint16_t decodeParamId(uint8_t id0, uint8_t id1, uint8_t id2) {
    return (uint16_t)((id0 & 0x03) << 14)
         | (uint16_t)((id1 & 0x7F) <<  7)
         | (uint16_t) (id2 & 0x7F);
}

// Encode IEEE 754 float → 5 × 7-bit bytes [v0..v4]
inline void encodeFloat(float f, uint8_t v[5]) {
    uint32_t raw;
    memcpy(&raw, &f, 4);
    v[0] =  raw        & 0x7F;
    v[1] = (raw >>  7) & 0x7F;
    v[2] = (raw >> 14) & 0x7F;
    v[3] = (raw >> 21) & 0x7F;
    v[4] = (raw >> 28) & 0x0F;
}

// Decode 5 × 7-bit bytes → float
inline float decodeFloat(const uint8_t v[5]) {
    uint32_t raw = (uint32_t)(v[0] & 0x7F)
                 | ((uint32_t)(v[1] & 0x7F) <<  7)
                 | ((uint32_t)(v[2] & 0x7F) << 14)
                 | ((uint32_t)(v[3] & 0x7F) << 21)
                 | ((uint32_t)(v[4] & 0x0F) << 28);
    float f;
    memcpy(&f, &raw, 4);
    return f;
}

// ── API ───────────────────────────────────────────────────────────────────────

// Validate SysEx header. Returns false if not an ICR message.
bool sysexValidate(const std::vector<uint8_t>& msg);

// Apply an incoming SysEx message to the voice manager.
// Handles SET_PARAM and SET_ALL. Returns true if recognized.
bool sysexApply(const std::vector<uint8_t>& msg, ResonatorVoiceManager& vm);

// Apply one (param_id, float) pair directly.
bool sysexApplyParam(uint16_t param_id, float value, ResonatorVoiceManager& vm);

// Read one param value from current SynthConfig.
// Returns false if param_id is unknown or write-only.
bool sysexReadParam(uint16_t param_id, const SynthConfig& cfg, float& out_value);

// Build SET_PARAM message (15 bytes).
std::vector<uint8_t> sysexBuildSetParam(uint16_t param_id, float value);

// Build GET_PARAM message (9 bytes).
std::vector<uint8_t> sysexBuildGetParam(uint16_t param_id);

// Build PARAM_RESPONSE message (15 bytes).
std::vector<uint8_t> sysexBuildParamResponse(uint16_t param_id, float value);

// Build SET_ALL dump of all R/W SynthConfig params.
std::vector<uint8_t> sysexBuildSetAll(const SynthConfig& cfg);

// Build REQUEST_ALL (7 bytes).
std::vector<uint8_t> sysexBuildRequestAll();

// Build ALL_PARAMS_DUMP response.
std::vector<uint8_t> sysexBuildAllParamsDump(const SynthConfig& cfg);

// List of all R/W param IDs (for building SET_ALL / REQUEST_ALL)
const std::vector<uint16_t>& sysexRWParamIds();
