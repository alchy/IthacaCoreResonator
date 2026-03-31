/*
 * resonator_gui.cpp
 * ──────────────────
 * Dear ImGui + GLFW + OpenGL3 GUI for IthacaCoreResonator.
 *
 * Layout:
 *   Top bar   — MIDI port selector, connect button, active-voice + sustain badge,
 *               MIDI activity LEDs, output level
 *   Center    — Clickable piano keyboard (C2..C7, 5 octaves), active notes lit
 *   Bottom-L  — Master mix, LFO pan, Limiter, BBE controls
 *   Bottom-R  — Data-driven core param sliders + last-note visualization
 *
 * The right panel is fully data-driven via ISynthCore::describeParams() and
 * getVizState() — no hard-coded SynthConfig references.
 */

#include "resonator_gui.h"
#include "../synth/midi_input.h"
#include "../dsp/dsp_chain.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <chrono>

static uint64_t guiNowMs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── Piano key layout constants ────────────────────────────────────────────────
static constexpr int   PIANO_MIDI_LOW  = 36;   // C2
static constexpr int   PIANO_MIDI_HIGH = 96;   // C7
static constexpr float WHITE_W  = 22.f;
static constexpr float WHITE_H  = 90.f;
static constexpr float BLACK_W  = 14.f;
static constexpr float BLACK_H  = 56.f;

static bool isBlack(int midi) {
    int n = midi % 12;
    return n == 1 || n == 3 || n == 6 || n == 8 || n == 10;
}
static int whitesBefore(int midi) {
    int count = 0;
    for (int m = PIANO_MIDI_LOW; m < midi; m++)
        if (!isBlack(m)) count++;
    return count;
}
static const char* noteName(int midi) {
    static const char* n[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    return n[midi % 12];
}

// ── GUI state ─────────────────────────────────────────────────────────────────
struct GuiState {
    std::vector<std::string> ports;
    int  selected_port   = 0;
    bool midi_connected  = false;

    bool active_notes[128] = {};
    int  mouse_held_note   = -1;

    // Master mix
    uint8_t master_gain  = 100;
    uint8_t pan          = 64;
    uint8_t lfo_speed    = 0;
    uint8_t lfo_depth    = 0;

    // Limiter
    uint8_t limiter_thr     = 100;
    uint8_t limiter_rel     = 50;
    bool    limiter_enabled = false;

    // BBE
    uint8_t bbe_def     = 0;
    uint8_t bbe_bass    = 0;
    bool    bbe_enabled = false;

    // Stats
    int  active_voices  = 0;
    bool sustain_on     = false;
};

// ── GLFW error callback ───────────────────────────────────────────────────────
static Logger* g_glfw_logger = nullptr;
static void glfwErrorCb(int /*err*/, const char* desc) {
    if (g_glfw_logger) g_glfw_logger->log("GLFW", LogSeverity::Error, desc);
}

// ── Piano keyboard widget ─────────────────────────────────────────────────────
static int drawPiano(GuiState& gs, CoreEngine& engine) {
    ImDrawList* dl  = ImGui::GetWindowDrawList();
    ImVec2 origin   = ImGui::GetCursorScreenPos();

    int total_white = 0;
    for (int m = PIANO_MIDI_LOW; m <= PIANO_MIDI_HIGH; m++)
        if (!isBlack(m)) total_white++;
    float total_w = total_white * WHITE_W;
    ImGui::Dummy(ImVec2(total_w, WHITE_H + 4.f));

    bool   lmb  = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    ImVec2 mp   = ImGui::GetMousePos();
    int    hit  = -1;

    // White keys
    for (int midi = PIANO_MIDI_LOW; midi <= PIANO_MIDI_HIGH; midi++) {
        if (isBlack(midi)) continue;
        float x  = origin.x + whitesBefore(midi) * WHITE_W;
        ImVec2 tl = {x + 1.f, origin.y};
        ImVec2 br = {x + WHITE_W - 1.f, origin.y + WHITE_H};
        bool   h  = lmb && mp.x >= tl.x && mp.x <= br.x
                        && mp.y >= tl.y && mp.y <= br.y;
        if (h) hit = midi;
        ImU32 col = gs.active_notes[midi] ? IM_COL32(120,160,255,255)
                  : h                     ? IM_COL32(200,220,255,255)
                                          : IM_COL32(240,240,240,255);
        dl->AddRectFilled(tl, br, col, 2.f);
        dl->AddRect(tl, br, IM_COL32(80,80,80,200), 2.f);
        if (midi % 12 == 0) {
            char buf[8]; snprintf(buf, sizeof(buf), "C%d", midi / 12 - 1);
            dl->AddText({tl.x + 2.f, br.y - 14.f}, IM_COL32(60,60,60,200), buf);
        }
    }
    // Black keys (drawn on top)
    for (int midi = PIANO_MIDI_LOW; midi <= PIANO_MIDI_HIGH; midi++) {
        if (!isBlack(midi)) continue;
        int prev = midi - 1;
        while (isBlack(prev)) prev--;
        float x  = origin.x + whitesBefore(prev) * WHITE_W + WHITE_W - BLACK_W * 0.5f;
        ImVec2 tl = {x, origin.y};
        ImVec2 br = {x + BLACK_W, origin.y + BLACK_H};
        bool   h  = lmb && mp.x >= tl.x && mp.x <= br.x
                        && mp.y >= tl.y && mp.y <= br.y;
        if (h) hit = midi;
        ImU32 col = gs.active_notes[midi] ? IM_COL32(80,120,255,255)
                  : h                     ? IM_COL32(60,80,140,255)
                                          : IM_COL32(30,30,30,255);
        dl->AddRectFilled(tl, br, col, 2.f);
        dl->AddRect(tl, br, IM_COL32(0,0,0,255), 2.f);
    }

    // Mouse note-on/off
    if (lmb && hit >= 0) {
        if (gs.mouse_held_note != hit) {
            if (gs.mouse_held_note >= 0)
                engine.noteOff((uint8_t)gs.mouse_held_note);
            engine.noteOn((uint8_t)hit, gs.master_gain);
            gs.mouse_held_note = hit;
        }
    } else {
        if (gs.mouse_held_note >= 0) {
            engine.noteOff((uint8_t)gs.mouse_held_note);
            gs.mouse_held_note = -1;
        }
    }
    return hit;
}

// ── Right panel: data-driven core params + last-note detail ──────────────────
static void drawCorePanel(CoreEngine& engine) {
    ISynthCore* core = engine.core();
    if (!core) {
        ImGui::TextDisabled("(no core loaded)");
        return;
    }

    // ── Core params — grouped sliders ────────────────────────────────────────
    ImGui::SeparatorText(("CORE PARAMS  [" + core->coreName() + "]").c_str());

    auto params = core->describeParams();

    // Group params by their group field
    std::string cur_group;
    for (auto& p : params) {
        if (p.group != cur_group) {
            cur_group = p.group;
            ImGui::SeparatorText(cur_group.c_str());
        }

        float val = p.value;
        std::string id  = "##cp_" + p.key;
        std::string lbl = p.unit.empty() ? p.label
                                         : p.label + " (" + p.unit + ")";

        ImGui::SetNextItemWidth(-1);
        bool changed;
        if (p.is_int) {
            int iv = (int)std::round(val);
            changed = ImGui::SliderInt(id.c_str(), &iv, (int)p.min, (int)p.max,
                                       (lbl + " %d").c_str());
            if (changed) core->setParam(p.key, (float)iv);
        } else {
            changed = ImGui::SliderFloat(id.c_str(), &val, p.min, p.max,
                                         (lbl + " %.4f").c_str());
            if (changed) core->setParam(p.key, val);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();

    // ── Last note visualization ───────────────────────────────────────────────
    CoreVizState viz = core->getVizState();
    if (!viz.last_note_valid) {
        ImGui::TextDisabled("(no note played yet)");
        return;
    }

    const CoreVoiceViz& ln = viz.last_note;
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255,220,100,255));
    ImGui::Text("LAST NOTE  %s%d  (MIDI %d)  vel %d  %.2f Hz",
        noteName(ln.midi), ln.midi / 12 - 1,
        ln.midi, ln.vel, ln.f0_hz);
    ImGui::PopStyleColor();

    // Show resonator-style detail only if the core provides it
    if (ln.n_partials > 0 || ln.n_strings > 0) {
        constexpr ImGuiTableFlags mf =
            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame;
        if (ImGui::BeginTable("##notemeta", 3, mf)) {
            ImGui::TableSetupColumn("STRUCTURE");
            ImGui::TableSetupColumn("NOISE");
            ImGui::TableSetupColumn("SPECTRAL EQ");
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("strings  %d", ln.n_strings);
            ImGui::Text("partials %d", ln.n_partials);
            ImGui::Text("width    %.3f", ln.width_factor);

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("centroid  %.0f Hz", ln.noise_centroid_hz);
            ImGui::Text("floor_rms %.4f",    ln.noise_floor_rms);
            ImGui::Text("tau       %.3f s",  ln.noise_tau_s);

            ImGui::TableSetColumnIndex(2);
            if (!ln.eq_gains_db.empty()) {
                float eq_min = *std::min_element(ln.eq_gains_db.begin(), ln.eq_gains_db.end());
                float eq_max = *std::max_element(ln.eq_gains_db.begin(), ln.eq_gains_db.end());
                float eq_sum = 0.f;
                for (float g : ln.eq_gains_db) eq_sum += g;
                ImGui::Text("points   %d", (int)ln.eq_gains_db.size());
                ImGui::Text("min      %.1f dB", eq_min);
                ImGui::Text("max      %.1f dB", eq_max);
                ImGui::Text("mean     %.1f dB", eq_sum / ln.eq_gains_db.size());
            } else {
                ImGui::TextDisabled("(none)");
            }
            ImGui::EndTable();
        }
    }

    // Partials table
    if (!ln.partials.empty()) {
        ImGui::Spacing();
        ImGui::SeparatorText("PARTIALS");
        constexpr ImGuiTableFlags ptf =
            ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_RowBg |
            ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY;
        float row_h = ImGui::GetTextLineHeightWithSpacing();
        float tbl_h = 12.5f * row_h;

        if (ImGui::BeginTable("##partials", 8, ptf, {0.f, tbl_h})) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("k",       ImGuiTableColumnFlags_WidthFixed, 26.f);
            ImGui::TableSetupColumn("f_hz",    ImGuiTableColumnFlags_WidthFixed, 66.f);
            ImGui::TableSetupColumn("A0",      ImGuiTableColumnFlags_WidthFixed, 66.f);
            ImGui::TableSetupColumn("tau1",    ImGuiTableColumnFlags_WidthFixed, 48.f);
            ImGui::TableSetupColumn("tau2",    ImGuiTableColumnFlags_WidthFixed, 48.f);
            ImGui::TableSetupColumn("a1",      ImGuiTableColumnFlags_WidthFixed, 44.f);
            ImGui::TableSetupColumn("beat_hz", ImGuiTableColumnFlags_WidthFixed, 58.f);
            ImGui::TableSetupColumn("mo",      ImGuiTableColumnFlags_WidthFixed, 22.f);
            ImGui::TableHeadersRow();

            for (const auto& pp : ln.partials) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0); ImGui::Text("%d", pp.k);
                ImGui::TableSetColumnIndex(1); ImGui::Text("%.2f", pp.f_hz);
                ImGui::TableSetColumnIndex(2); ImGui::Text("%.5f", pp.A0);
                ImGui::TableSetColumnIndex(3); ImGui::Text("%.2f", pp.tau1);
                ImGui::TableSetColumnIndex(4);
                if (pp.a1 < 1.f - 1e-5f) ImGui::Text("%.2f", pp.tau2);
                else                      ImGui::TextDisabled("-");
                ImGui::TableSetColumnIndex(5); ImGui::Text("%.3f", pp.a1);
                ImGui::TableSetColumnIndex(6);
                if (pp.beat_hz > 1e-6f) ImGui::Text("%.4f", pp.beat_hz);
                else                    ImGui::TextDisabled("0");
                ImGui::TableSetColumnIndex(7);
                ImGui::TextDisabled(pp.mono ? "y" : "n");
            }
            ImGui::EndTable();
        }
    }
}

// ── Main GUI loop ─────────────────────────────────────────────────────────────

int runResonatorGui(CoreEngine& engine, Logger& logger) {
    g_glfw_logger = &logger;
    logger.log("GUI", LogSeverity::Info, "Starting GLFW + ImGui");

    glfwSetErrorCallback(glfwErrorCb);
    if (!glfwInit()) {
        logger.log("GUI", LogSeverity::Error, "glfwInit failed");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(1350, 680,
        "IthacaCoreResonator", nullptr, nullptr);
    if (!win) {
        logger.log("GUI", LogSeverity::Error, "glfwCreateWindow failed");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().IniFilename = nullptr;
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 4.f;
    ImGui::GetStyle().FrameRounding  = 3.f;
    ImGui::GetStyle().GrabRounding   = 3.f;

    ImGui_ImplGlfw_InitForOpenGL(win, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    GuiState gs;
    gs.ports = MidiInput::listPorts();
    MidiInput midi_in;

    if (!gs.ports.empty()) {
        midi_in.open(engine, 0);
        gs.midi_connected = midi_in.isOpen();
        logger.log("GUI", LogSeverity::Info,
            gs.midi_connected ? "Auto-connected: " + gs.ports[0]
                              : "MIDI open failed");
    }
    logger.log("GUI", LogSeverity::Info, "GUI loop started");

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // Update active notes from viz state
        if (engine.core()) {
            CoreVizState viz = engine.core()->getVizState();
            std::memset(gs.active_notes, 0, sizeof(gs.active_notes));
            for (int m : viz.active_midi_notes)
                if (m >= 0 && m < 128) gs.active_notes[m] = true;
            gs.active_voices = viz.active_voice_count;
            gs.sustain_on    = viz.sustain_active;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        int fb_w, fb_h;
        glfwGetFramebufferSize(win, &fb_w, &fb_h);
        ImGui::SetNextWindowPos({0, 0});
        ImGui::SetNextWindowSize({(float)fb_w, (float)fb_h});
        ImGui::Begin("##main", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove     | ImGuiWindowFlags_NoScrollbar);

        // ── Top bar ───────────────────────────────────────────────────────────
        ImGui::Text("MIDI Port:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(280.f);
        const char* preview = gs.ports.empty() ? "(none)"
                            : gs.ports[gs.selected_port].c_str();
        if (ImGui::BeginCombo("##port", preview)) {
            for (int i = 0; i < (int)gs.ports.size(); i++) {
                bool sel = (i == gs.selected_port);
                if (ImGui::Selectable(gs.ports[i].c_str(), sel))
                    gs.selected_port = i;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (gs.midi_connected) {
            ImGui::PushStyleColor(ImGuiCol_Button, IM_COL32(60,150,60,255));
            if (ImGui::Button("Disconnect")) {
                midi_in.close();
                gs.midi_connected = false;
            }
            ImGui::PopStyleColor();
        } else {
            if (ImGui::Button("Connect") && !gs.ports.empty()) {
                midi_in.open(engine, gs.selected_port);
                gs.midi_connected = midi_in.isOpen();
            }
        }
        ImGui::SameLine(0, 20.f);
        ImGui::TextColored(
            gs.midi_connected ? ImVec4(0.4f,1.f,0.4f,1.f) : ImVec4(1.f,0.4f,0.4f,1.f),
            gs.midi_connected ? "MIDI: connected" : "MIDI: not connected");
        ImGui::SameLine(0, 20.f);
        ImGui::Text("Voices: %d", gs.active_voices);
        if (gs.lfo_speed > 0 && gs.lfo_depth > 0) {
            ImGui::SameLine(0, 14.f);
            float t = (float)ImGui::GetTime();
            float pulse = 0.5f + 0.5f * std::sin(t * 2.f * 3.14159f *
                          (2.f * gs.lfo_speed / 127.f));
            ImGui::TextColored({0.3f + 0.7f*pulse, 0.8f, 1.f, 1.f}, "LFO");
        }
        if (gs.sustain_on) {
            ImGui::SameLine(0, 10.f);
            ImGui::TextColored({1.f,0.9f,0.2f,1.f}, "[SUSTAIN]");
        }
        ImGui::SameLine(0, 20.f);
        if (ImGui::SmallButton("Refresh")) {
            gs.ports = MidiInput::listPorts();
            gs.selected_port = 0;
        }

        ImGui::Separator();

        // ── MIDI activity LEDs ────────────────────────────────────────────────
        {
            const uint64_t now   = guiNowMs();
            const uint64_t flash = 80;
            const auto& act      = midi_in.activity();

            auto midiLed = [&](const char* label, uint64_t last_ms) {
                float  r   = 5.f;
                float  th  = ImGui::GetTextLineHeight();
                ImVec2 p   = ImGui::GetCursorScreenPos();
                bool   lit = (last_ms > 0) && ((now - last_ms) < flash);
                ImU32  col = lit ? IM_COL32(50,230,80,255) : IM_COL32(35,65,35,220);
                ImU32  rim = lit ? IM_COL32(120,255,140,180) : IM_COL32(60,90,60,160);
                ImGui::GetWindowDrawList()->AddCircleFilled({p.x+r, p.y+th*0.5f}, r, col);
                ImGui::GetWindowDrawList()->AddCircle({p.x+r, p.y+th*0.5f}, r, rim, 12, 1.f);
                ImGui::Dummy({r*2.f+2.f, th});
                ImGui::SameLine(0, 3.f);
                if (lit) ImGui::TextColored({0.4f,1.f,0.5f,1.f}, "%s", label);
                else     ImGui::TextDisabled("%s", label);
            };
            auto ledSep = [&]() {
                ImGui::SameLine(0, 10.f);
                ImVec2 pos = ImGui::GetCursorScreenPos();
                float cy = pos.y + ImGui::GetTextLineHeight() * 0.5f;
                ImGui::GetWindowDrawList()->AddLine({pos.x, cy}, {pos.x+14.f, cy},
                    ImGui::GetColorU32(ImGuiCol_Separator), 1.f);
                ImGui::Dummy({14.f, ImGui::GetTextLineHeight()});
                ImGui::SameLine(0, 10.f);
            };

            midiLed("MIDI DATA",   act.any_ms.load(std::memory_order_relaxed));
            ledSep();
            midiLed("SysEx",       act.sysex_ms.load(std::memory_order_relaxed));
            ledSep();
            midiLed("Note ON",     act.note_on_ms.load(std::memory_order_relaxed));
            ledSep();
            midiLed("Note OFF",    act.note_off_ms.load(std::memory_order_relaxed));
            ledSep();
            midiLed("Pedal",       act.pedal_ms.load(std::memory_order_relaxed));

            // Output level LED
            static constexpr float CLIP_THRESH = 0.3548f;
            float peak_lin = engine.getOutputPeakLin();
            bool  over     = (peak_lin > CLIP_THRESH);
            {
                ledSep();
                float  r   = 5.f;
                float  th  = ImGui::GetTextLineHeight();
                ImVec2 p   = ImGui::GetCursorScreenPos();
                ImU32  col = over ? IM_COL32(230,40,40,255) : IM_COL32(65,30,30,220);
                ImU32  rim = over ? IM_COL32(255,120,120,200) : IM_COL32(90,50,50,160);
                ImGui::GetWindowDrawList()->AddCircleFilled({p.x+r, p.y+th*0.5f}, r, col);
                ImGui::GetWindowDrawList()->AddCircle({p.x+r, p.y+th*0.5f}, r, rim, 12, 1.f);
                ImGui::Dummy({r*2.f+2.f, th});
                ImGui::SameLine(0, 3.f);
                float db = (peak_lin > 1e-9f) ? 20.f * std::log10(peak_lin) : -99.f;
                if (over) ImGui::TextColored({1.f,0.3f,0.3f,1.f}, "LEVEL %.1f dB", db);
                else      ImGui::TextDisabled("LEVEL %.1f dB", db);
            }
        }

        ImGui::Separator();

        // ── Two-column layout ─────────────────────────────────────────────────
        {
            int nw = 0;
            for (int m = PIANO_MIDI_LOW; m <= PIANO_MIDI_HIGH; m++)
                if (!isBlack(m)) nw++;
            float piano_px = nw * WHITE_W;
            float pad      = ImGui::GetStyle().WindowPadding.x;
            float left_w   = piano_px + pad * 2.f;

            // ── Left: piano + controls ────────────────────────────────────────
            ImGui::BeginChild("##left", {left_w, 0.f}, false,
                              ImGuiWindowFlags_NoScrollbar);

            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {0.f, 4.f});
            drawPiano(gs, engine);
            ImGui::PopStyleVar();

            ImGui::Separator();

            DspChain* dsp = engine.getDspChain();

            auto labeledSlider = [](const char* id, const char* label,
                                    const char* desc, int* val, int lo, int hi) -> bool {
                ImGui::Text("%s", label);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(-1);
                bool changed = ImGui::SliderInt(id, val, lo, hi);
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(140,140,140,255));
                ImGui::TextUnformatted(desc);
                ImGui::PopStyleColor();
                ImGui::Spacing();
                return changed;
            };

            constexpr ImGuiTableFlags tf =
                ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersInnerV |
                ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_PadOuterX;

            if (ImGui::BeginTable("ctrl", 2, tf)) {
                ImGui::TableNextRow();

                // MIX column
                ImGui::TableSetColumnIndex(0);
                ImGui::SeparatorText("MIX");
                {
                    int v = gs.master_gain;
                    char desc[48]; snprintf(desc, sizeof(desc), "Level: %d/127", v);
                    if (labeledSlider("##gain", "Gain", desc, &v, 0, 127)) {
                        gs.master_gain = (uint8_t)v;
                        engine.setMasterGain(gs.master_gain, engine.getLogger());
                    }
                }
                {
                    int v = gs.pan;
                    char desc[48];
                    float pp = (v - 64) / 64.f;
                    if (std::abs(pp) < 0.02f) snprintf(desc, sizeof(desc), "Balance: center");
                    else snprintf(desc, sizeof(desc), "Balance: %.0f%% %s",
                                  std::abs(pp)*100.f, pp < 0 ? "L" : "R");
                    if (labeledSlider("##pan", "Pan ", desc, &v, 0, 127)) {
                        gs.pan = (uint8_t)v;
                        engine.setMasterPan(gs.pan);
                    }
                }

                // LFO PAN column
                ImGui::TableSetColumnIndex(1);
                ImGui::SeparatorText("LFO PAN");
                {
                    int v = gs.lfo_speed;
                    char desc[48]; snprintf(desc, sizeof(desc), "Rate: %.2f Hz",
                                            2.f * (v / 127.f));
                    if (labeledSlider("##lfospd", "Speed", desc, &v, 0, 127)) {
                        gs.lfo_speed = (uint8_t)v;
                        engine.setPanSpeed(gs.lfo_speed);
                    }
                }
                {
                    int v = gs.lfo_depth;
                    char desc[48]; snprintf(desc, sizeof(desc), "Sweep: %.0f%%",
                                            100.f * (v / 127.f));
                    if (labeledSlider("##lfodep", "Depth", desc, &v, 0, 127)) {
                        gs.lfo_depth = (uint8_t)v;
                        engine.setPanDepth(gs.lfo_depth);
                    }
                }
                // LFO indicator
                {
                    bool active = gs.lfo_speed > 0 && gs.lfo_depth > 0;
                    float bar_w = ImGui::GetContentRegionAvail().x;
                    ImVec2 bp   = ImGui::GetCursorScreenPos();
                    float bar_h = 10.f;
                    ImDrawList* dl2 = ImGui::GetWindowDrawList();
                    dl2->AddRectFilled(bp, {bp.x+bar_w, bp.y+bar_h},
                                       IM_COL32(40,40,40,200), 3.f);
                    float pos_x = bp.x + bar_w * 0.5f;
                    if (active) {
                        float hz  = 2.f * (gs.lfo_speed / 127.f);
                        float dep = gs.lfo_depth / 127.f;
                        float lv  = dep * std::sin((float)ImGui::GetTime() *
                                                   2.f * 3.14159f * hz);
                        pos_x = bp.x + bar_w * 0.5f * (1.f + lv);
                    }
                    dl2->AddCircleFilled({pos_x, bp.y+bar_h*0.5f}, 5.f,
                        active ? IM_COL32(80,200,255,255) : IM_COL32(80,80,80,180));
                    dl2->AddText({bp.x, bp.y}, IM_COL32(120,120,120,200), "L");
                    dl2->AddText({bp.x+bar_w-8.f, bp.y}, IM_COL32(120,120,120,200), "R");
                    ImGui::Dummy({bar_w, bar_h + 2.f});
                }

                // LIMITER row
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                {
                    bool ena = gs.limiter_enabled;
                    if (ImGui::Checkbox("##limon", &ena)) {
                        gs.limiter_enabled = ena;
                        if (dsp) dsp->setLimiterEnabled(ena ? 127 : 0);
                    }
                    ImGui::SameLine(); ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("LIMITER");
                }
                {
                    int v = gs.limiter_thr;
                    float db = -40.f + 40.f * (v / 127.f);
                    char desc[48]; snprintf(desc, sizeof(desc), "Ceiling: %.1f dB", db);
                    if (labeledSlider("##limthr", "Threshold", desc, &v, 0, 127)) {
                        gs.limiter_thr = (uint8_t)v;
                        engine.setLimiterThreshold(gs.limiter_thr);
                    }
                }
                {
                    int v = gs.limiter_rel;
                    float ms = 10.f + 1990.f * (v / 127.f);
                    char desc[48]; snprintf(desc, sizeof(desc), "Recovery: %.0f ms", ms);
                    if (labeledSlider("##limrel", "Release  ", desc, &v, 0, 127)) {
                        gs.limiter_rel = (uint8_t)v;
                        engine.setLimiterRelease(gs.limiter_rel);
                    }
                }
                if (dsp) {
                    float gr = gs.limiter_enabled
                        ? std::max(0.f, std::min(-dsp->limiter().gainReductionDb() / 40.f, 1.f))
                        : 0.f;
                    char ovl[32];
                    snprintf(ovl, sizeof(ovl), gs.limiter_enabled
                        ? "GR  %.1f dB" : "GR  (off)",
                        dsp->limiter().gainReductionDb());
                    ImGui::PushStyleColor(ImGuiCol_PlotHistogram,
                        gs.limiter_enabled ? IM_COL32(220,60,60,200)
                                           : IM_COL32(70,70,70,150));
                    ImGui::ProgressBar(gr, {-1.f, 14.f}, ovl);
                    ImGui::PopStyleColor();
                }

                // BBE column
                ImGui::TableSetColumnIndex(1);
                {
                    bool ena = gs.bbe_enabled;
                    if (ImGui::Checkbox("##bbeon", &ena)) {
                        gs.bbe_enabled = ena;
                        if (dsp) {
                            dsp->setBBEDefinition(ena ? gs.bbe_def  : 0);
                            dsp->setBBEBassBoost (ena ? gs.bbe_bass : 0);
                        }
                    }
                    ImGui::SameLine(); ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("BBE Sonic Maximizer");
                }
                {
                    int v = gs.bbe_def;
                    char desc[48]; snprintf(desc, sizeof(desc), "5kHz presence: +%.1f dB",
                                            12.f * (v / 127.f));
                    if (labeledSlider("##bbedef", "Definition", desc, &v, 0, 127)) {
                        gs.bbe_def = (uint8_t)v;
                        gs.bbe_enabled = (v > 0 || gs.bbe_bass > 0);
                        engine.setBBEDefinition(gs.bbe_def);
                    }
                }
                {
                    int v = gs.bbe_bass;
                    char desc[48]; snprintf(desc, sizeof(desc), "180Hz warmth: +%.1f dB",
                                            10.f * (v / 127.f));
                    if (labeledSlider("##bbebas", "Bass Boost", desc, &v, 0, 127)) {
                        gs.bbe_bass = (uint8_t)v;
                        gs.bbe_enabled = (v > 0 || gs.bbe_def > 0);
                        engine.setBBEBassBoost(gs.bbe_bass);
                    }
                }

                ImGui::EndTable();
            }

            ImGui::EndChild();  // ##left

            // ── Separator ─────────────────────────────────────────────────────
            ImGui::SameLine(0, 0);
            {
                ImVec2 p = ImGui::GetCursorScreenPos();
                float  h = ImGui::GetContentRegionAvail().y;
                ImGui::GetWindowDrawList()->AddLine(p, {p.x, p.y + h},
                    ImGui::GetColorU32(ImGuiCol_Separator), 1.f);
                ImGui::SetCursorScreenPos({p.x + 1.f, p.y});
            }
            ImGui::SameLine(0, 8.f);

            // ── Right: data-driven core params + last note ─────────────────────
            ImGui::BeginChild("##right", {0.f, 0.f}, false,
                              ImGuiWindowFlags_NoScrollbar);
            drawCorePanel(engine);
            ImGui::EndChild();
        }

        // ── Bottom bar ────────────────────────────────────────────────────────
        ImGui::Separator();
        ImGui::Text("Spacebar = sustain  |  A-K = C4-B4  |  Click keys to play");

        // Sustain toggle
        if (ImGui::IsKeyPressed(ImGuiKey_Space)) {
            static bool sus = false;
            sus = !sus;
            engine.sustainPedal(sus ? 127 : 0);
            gs.sustain_on = sus;
        }
        // Keyboard shortcuts (QWERTY piano)
        const ImGuiKey qkeys[] = {
            ImGuiKey_A, ImGuiKey_W, ImGuiKey_S, ImGuiKey_E,
            ImGuiKey_D, ImGuiKey_F, ImGuiKey_T, ImGuiKey_G,
            ImGuiKey_Y, ImGuiKey_H, ImGuiKey_U, ImGuiKey_J
        };
        const int qmidis[] = { 60,61,62,63,64,65,66,67,68,69,70,71 };
        for (int i = 0; i < 12; i++) {
            if (ImGui::IsKeyPressed(qkeys[i], false)) {
                engine.noteOn((uint8_t)qmidis[i], gs.master_gain);
                gs.active_notes[qmidis[i]] = true;
            }
            if (ImGui::IsKeyReleased(qkeys[i])) {
                engine.noteOff((uint8_t)qmidis[i]);
                gs.active_notes[qmidis[i]] = false;
            }
        }

        ImGui::End();

        // Render
        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}
