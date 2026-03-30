/*
 * sysex.cpp
 * ──────────
 * IthacaCoreResonator MIDI SysEx codec implementation.
 */

#include "sysex.h"
#include <cstring>
#include <algorithm>

static float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ── R/W param ID list ─────────────────────────────────────────────────────────

static const std::vector<uint16_t> kRWParamIds = {
    // 0x3 STEREO
    PID_PAN_SPREAD, PID_STEREO_DECORR, PID_STEREO_BOOST,
    PID_PAN_TILT, PID_DECORR_MIDI_LO, PID_DECORR_MIDI_HI, PID_DECORR_MAX,
    // 0x4 TIMBRE
    PID_BEAT_SCALE, PID_HARMONIC_BRIGHTNESS,
    PID_EQ_STRENGTH, PID_EQ_FREQ_MIN,
    PID_PITCH_GLIDE, PID_PITCH_GLIDE_TAU_MS, PID_PITCH_GLIDE_VEL_THRESH,
    // 0x5 LEVEL/ENV
    PID_TARGET_RMS, PID_VEL_GAMMA, PID_NOISE_LEVEL,
    PID_ONSET_MS, PID_LONGITUDINAL_PRECURSOR, PID_RENDER_REF_DURATION,
};

const std::vector<uint16_t>& sysexRWParamIds() { return kRWParamIds; }

// ── Validate ──────────────────────────────────────────────────────────────────

bool sysexValidate(const std::vector<uint8_t>& msg) {
    if (msg.size() < 7)         return false;
    if (msg.front() != 0xF0)    return false;
    if (msg.back()  != 0xF7)    return false;
    if (msg[1] != SYSEX_MFR)    return false;
    if (msg[2] != SYSEX_SIG0)   return false;
    if (msg[3] != SYSEX_SIG1)   return false;
    if (msg[4] != SYSEX_SIG2)   return false;
    return true;
}

// ── Read one param from SynthConfig ──────────────────────────────────────────

bool sysexReadParam(uint16_t id, const SynthConfig& c, float& v) {
    switch (id) {
        case PID_PAN_SPREAD:             v = c.pan_spread;               return true;
        case PID_STEREO_DECORR:          v = c.stereo_decorr;            return true;
        case PID_STEREO_BOOST:           v = c.stereo_boost;             return true;
        case PID_PAN_TILT:               v = c.pan_tilt;                 return true;
        case PID_DECORR_MIDI_LO:         v = c.stereo_decorr_midi_lo;    return true;
        case PID_DECORR_MIDI_HI:         v = c.stereo_decorr_midi_hi;    return true;
        case PID_DECORR_MAX:             v = c.stereo_decorr_max;        return true;
        case PID_BEAT_SCALE:             v = c.beat_scale;               return true;
        case PID_HARMONIC_BRIGHTNESS:    v = c.harmonic_brightness;      return true;
        case PID_EQ_STRENGTH:            v = c.eq_strength;              return true;
        case PID_EQ_FREQ_MIN:            v = c.eq_freq_min;              return true;
        case PID_PITCH_GLIDE:            v = c.pitch_glide;              return true;
        case PID_PITCH_GLIDE_TAU_MS:     v = c.pitch_glide_tau_ms;       return true;
        case PID_PITCH_GLIDE_VEL_THRESH: v = (float)c.pitch_glide_vel_thresh; return true;
        case PID_TARGET_RMS:             v = c.target_rms;               return true;
        case PID_VEL_GAMMA:              v = c.vel_gamma;                return true;
        case PID_NOISE_LEVEL:            v = c.noise_level;              return true;
        case PID_ONSET_MS:               v = c.onset_ms;                 return true;
        case PID_LONGITUDINAL_PRECURSOR: v = c.longitudinal_precursor;   return true;
        case PID_RENDER_REF_DURATION:    v = c.render_ref_duration_s;   return true;
        default: return false;
    }
}

// ── Apply one param to VoiceManager ──────────────────────────────────────────

bool sysexApplyParam(uint16_t id, float v, ResonatorVoiceManager& vm) {
    switch (id) {
        // 0x3 STEREO
        case PID_PAN_SPREAD:             vm.setSynthPanSpread(clamp(v,0.f,1.57f));          return true;
        case PID_STEREO_DECORR:          vm.setSynthStereoDecorr(clamp(v,0.f,1.f));         return true;
        case PID_STEREO_BOOST:           vm.setSynthStereoBoost(clamp(v,0.f,4.f));          return true;
        case PID_PAN_TILT:               vm.setSynthPanTilt(clamp(v,0.f,1.f));              return true;
        case PID_DECORR_MIDI_LO:         vm.setSynthStereoDecorrMidiLo(clamp(v,0.f,127.f)); return true;
        case PID_DECORR_MIDI_HI:         vm.setSynthStereoDecorrMidiHi(clamp(v,0.f,127.f)); return true;
        case PID_DECORR_MAX:             vm.setSynthStereoDecorrMax(clamp(v,0.f,1.f));      return true;
        // 0x4 TIMBRE
        case PID_BEAT_SCALE:             vm.setSynthBeatScale(clamp(v,0.f,5.f));            return true;
        case PID_HARMONIC_BRIGHTNESS:    vm.setSynthHarmonicBrightness(clamp(v,-2.f,4.f));  return true;
        case PID_EQ_STRENGTH:            vm.setSynthEqStrength(clamp(v,0.f,1.f));           return true;
        case PID_EQ_FREQ_MIN:            vm.setSynthEqFreqMin(clamp(v,20.f,2000.f));        return true;
        case PID_PITCH_GLIDE:            vm.setSynthPitchGlide(clamp(v,0.f,0.05f));         return true;
        case PID_PITCH_GLIDE_TAU_MS:     vm.setSynthPitchGlideTauMs(clamp(v,1.f,500.f));    return true;
        case PID_PITCH_GLIDE_VEL_THRESH: vm.setSynthPitchGlideVelThresh(
                                             (int)clamp(v,0.f,127.f));                      return true;
        // 0x5 LEVEL/ENV
        case PID_TARGET_RMS:             vm.setSynthTargetRms(clamp(v,0.001f,0.5f));        return true;
        case PID_VEL_GAMMA:              vm.setSynthVelGamma(clamp(v,0.1f,3.f));            return true;
        case PID_NOISE_LEVEL:            vm.setSynthNoiseLevel(clamp(v,0.f,4.f));           return true;
        case PID_ONSET_MS:               vm.setSynthOnsetMs(clamp(v,0.f,50.f));             return true;
        case PID_LONGITUDINAL_PRECURSOR: vm.setSynthLongitudinalPrecursor(clamp(v,0.f,1.f));return true;
        case PID_RENDER_REF_DURATION:    vm.setSynthRenderRefDuration(clamp(v,0.1f,60.f));  return true;
        default: return false;
    }
}

// ── Parse and apply incoming SysEx ───────────────────────────────────────────

bool sysexApply(const std::vector<uint8_t>& msg, ResonatorVoiceManager& vm) {
    if (!sysexValidate(msg)) return false;

    uint8_t type = msg[5];
    const uint8_t* p = msg.data() + 6;
    const int payload_len = (int)msg.size() - 7;  // exclude header(6) + F7(1)

    if (type == SYSEX_SET_PARAM && payload_len >= 8) {
        // 3 bytes ID + 5 bytes value
        uint16_t id = decodeParamId(p[0], p[1], p[2]);
        float    v  = decodeFloat(p + 3);
        return sysexApplyParam(id, v, vm);
    }

    if (type == SYSEX_SET_ALL) {
        // Repeated 8-byte blocks: 3 bytes ID + 5 bytes value
        bool any = false;
        for (int i = 0; i + 7 < payload_len + 1; i += 8) {
            uint16_t id = decodeParamId(p[i], p[i+1], p[i+2]);
            float    v  = decodeFloat(p + i + 3);
            any |= sysexApplyParam(id, v, vm);
        }
        return any;
    }

    return false;
}

// ── Build helpers ─────────────────────────────────────────────────────────────

static std::vector<uint8_t> makeHeader(uint8_t type) {
    return { 0xF0, SYSEX_MFR, SYSEX_SIG0, SYSEX_SIG1, SYSEX_SIG2, type };
}

std::vector<uint8_t> sysexBuildSetParam(uint16_t param_id, float value) {
    auto msg = makeHeader(SYSEX_SET_PARAM);
    uint8_t id0, id1, id2;
    encodeParamId(param_id, id0, id1, id2);
    uint8_t v[5];
    encodeFloat(value, v);
    msg.insert(msg.end(), {id0, id1, id2});
    msg.insert(msg.end(), v, v + 5);
    msg.push_back(0xF7);
    return msg;
}

std::vector<uint8_t> sysexBuildGetParam(uint16_t param_id) {
    auto msg = makeHeader(SYSEX_GET_PARAM);
    uint8_t id0, id1, id2;
    encodeParamId(param_id, id0, id1, id2);
    msg.insert(msg.end(), {id0, id1, id2});
    msg.push_back(0xF7);
    return msg;
}

std::vector<uint8_t> sysexBuildParamResponse(uint16_t param_id, float value) {
    auto msg = makeHeader(SYSEX_PARAM_RESPONSE);
    uint8_t id0, id1, id2;
    encodeParamId(param_id, id0, id1, id2);
    uint8_t v[5];
    encodeFloat(value, v);
    msg.insert(msg.end(), {id0, id1, id2});
    msg.insert(msg.end(), v, v + 5);
    msg.push_back(0xF7);
    return msg;
}

static void appendParamBlock(std::vector<uint8_t>& msg, uint16_t id, float value) {
    uint8_t id0, id1, id2;
    encodeParamId(id, id0, id1, id2);
    uint8_t v[5];
    encodeFloat(value, v);
    msg.insert(msg.end(), {id0, id1, id2});
    msg.insert(msg.end(), v, v + 5);
}

std::vector<uint8_t> sysexBuildSetAll(const SynthConfig& cfg) {
    auto msg = makeHeader(SYSEX_SET_ALL);
    for (uint16_t id : kRWParamIds) {
        float v = 0.f;
        sysexReadParam(id, cfg, v);
        appendParamBlock(msg, id, v);
    }
    msg.push_back(0xF7);
    return msg;
}

std::vector<uint8_t> sysexBuildRequestAll() {
    auto msg = makeHeader(SYSEX_REQUEST_ALL);
    msg.push_back(0xF7);
    return msg;
}

std::vector<uint8_t> sysexBuildAllParamsDump(const SynthConfig& cfg) {
    auto msg = makeHeader(SYSEX_ALL_PARAMS_DUMP);
    for (uint16_t id : kRWParamIds) {
        float v = 0.f;
        sysexReadParam(id, cfg, v);
        appendParamBlock(msg, id, v);
    }
    msg.push_back(0xF7);
    return msg;
}
