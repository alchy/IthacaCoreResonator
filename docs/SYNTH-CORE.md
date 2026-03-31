# Pluggable Synth Core — návrh architektury

Tento dokument navrhuje refaktoring IthacaCoreResonator na modulární architekturu
kde lze zaměnit jádro syntézy bez úpravy enginu, GUI nebo offline rendereru.

---

## 1. Proč to děláme — a proč to není triviální

Současný stav: `ResonatorVoice` implementuje fyzikální model pianové struny.
Chceme ho nahradit alternativními jádry (jednoduchý oscilátor, sample přehrávač,
FM synth…) aniž bychom přepisovali engine, GUI, nebo render server.

### Co se zdá jednoduché, ale není

**Virtuální funkce v audio callbacku.**
Pokud uděláme `ISynthVoice` s `virtual processBlock()`, volí 88 hlasů ×
96 parciálů virtuální dispatch per blok — cache missy v RT vlákně.
Řešení: virtualita na úrovni **celého core** (88 hlasů najednou), ne per hlas.
Core řeší interně vše — polyphonie, envelopy, mix — a vrátí hotový stereo mix.

**Panning není post-processing.**
V ResonatorVoice jsou pan úhly součástí fyzikálního modelu (struny pianových
tónů jdou do prostoru). Nelze jednoduše "přidat panning za core". Core musí
sám definovat co je jeho "přirozený" výstup (mono / stereo).
Rozhodnutí: **core vždy outputuje stereo** (L, R). Mono core pošle stejný
signál na oba kanály.

**Konfigurace je per-core.**
`SynthConfig` je dnes struct se specifickými fieldy pianového synthu
(beat_scale, eq_strength…). Jiné jádro má jiné parametry.
GUI musí být **data-driven** — core popíše svoje parametry, GUI je zobrazí.
Hardcoded sliders pro SynthConfig musí zmizet.

**Polyphonie.**
Každé jádro může mít jinou strategii voice stealingu. Klavír: oldest note.
Synth: priority na high velocity. Necheme generický pool co by všechna jádra
omezoval. Řešení: **polyphonie je uvnitř core**, ne ve wrapperu.

---

## 2. Hranice Core — co dovnitř, co ven

```
┌─────────────────────────────────────────────────────────┐
│                    ISynthCore                           │
│                                                         │
│  IN:  noteOn(midi, vel)  noteOff(midi)  sustain(bool)  │
│       processBlock(L*, R*, n) — RT volání              │
│       setParam(key, value) — z GUI / SysEx / config    │
│                                                         │
│  UVNITŘ: polyphonie, envelopy, oscilátory, EQ, noise,  │
│          panning strun, detuning                        │
│                                                         │
│  OUT: stereo PCM float (surový, nenormalizovaný)        │
│       getVizState() — pro GUI (aktivní hlasy, partials) │
│       describeParams() — seznam parametrů pro GUI       │
└─────────────────────────────────────────────────────────┘
         │ stereo PCM out
         ▼
┌─────────────────────┐
│    CoreEngine       │   (není součástí core)
│  - normalizace      │
│  - master gain      │
│  - DspChain         │
│    (limiter, BBE)   │
│  - audio device     │
└─────────────────────┘
```

**Co je záměrně mimo core:**
- Normalizace výstupu (master level)
- Limiter, BBE — master bus efekty
- Audio zařízení (miniaudio)
- MIDI port management
- SysEx protokol (ten zůstane v enginu, mapuje na `setParam`)

**Proč panning strun je uvnitř:**
Fyzikální model definuje prostorové rozmístění strun (bas vlevo, diskant vpravo).
To je součást zvukového charakteru jádra, ne master bus operace. Jiné jádro
(FM synth) může mít úplně jiné prostorové zpracování.

---

## 3. Rozhraní ISynthCore

```cpp
// synth/i_synth_core.h

#pragma once
#include <string>
#include <vector>
#include <cstdint>

// ── Vizualizační data — co core vrátí GUI ─────────────────────────────────────

struct CoreParamDesc {
    std::string key;        // identifikátor (např. "beat_scale")
    std::string label;      // zobrazované jméno ("Beat Scale")
    std::string group;      // skupina pro GUI panel ("Timbre", "Stereo", ...)
    std::string unit;       // jednotka pro zobrazení ("Hz", "ms", "")
    float       value;      // aktuální hodnota
    float       min;        // minimum slideru
    float       max;        // maximum slideru
    bool        is_int;     // zobrazit jako celé číslo
};

struct CorePartialViz {
    int   k;
    float f_hz;
    float A0;
    float tau1, tau2, a1;
    float beat_hz;
    bool  mono;
};

struct CoreVoiceViz {
    int   midi;
    int   vel;
    float f0_hz;
    float B;
    int   n_strings;
    int   n_partials;
    float width_factor;
    float noise_centroid_hz;
    float noise_floor_rms;
    float noise_tau_s;
    std::vector<CorePartialViz> partials;  // max ~96, pro GUI tabulku
};

struct CoreVizState {
    int                       active_voice_count;
    bool                      sustain_active;
    std::vector<CoreVoiceViz> active_voices;   // aktuálně znějící hlasy
    CoreVoiceViz              last_note;        // poslední zahraná nota (pro detail panel)
    bool                      last_note_valid;
};

// ── Hlavní rozhraní ───────────────────────────────────────────────────────────

class ISynthCore {
public:
    virtual ~ISynthCore() = default;

    // Inicializace — načte parametry a připraví voice pool.
    // params_path: cesta k JSON souboru specifickému pro dané jádro.
    // sr: vzorkovací frekvence.
    // Vrátí false při chybě (soubor nenalezen, neplatný formát).
    virtual bool load(const std::string& params_path,
                      float              sr,
                      Logger&            logger) = 0;

    // Změní vzorkovací frekvenci po inicializaci (přepočítá koeficienty).
    virtual void setSampleRate(float sr) = 0;

    // ── MIDI vstup ────────────────────────────────────────────────────────────
    // Vše thread-safe (volatelné z MIDI vlákna i GUI vlákna).
    virtual void noteOn (uint8_t midi, uint8_t velocity) = 0;
    virtual void noteOff(uint8_t midi)                   = 0;
    virtual void sustainPedal(bool down)                 = 0;
    virtual void allNotesOff()                           = 0;

    // ── Audio rendering — RT vlákno ───────────────────────────────────────────
    // Přidá výstup core do out_l / out_r (ADDITIVE — volající nuluje buffery).
    // Vrátí true pokud jsou aktivní hlasy.
    // NESMÍ alokovat paměť, zamykat mutexy, ani volat IO.
    virtual bool processBlock(float* out_l, float* out_r,
                              int n_samples) noexcept = 0;

    // ── Parametry — generický key/value přístup ───────────────────────────────
    // Nastaví parametr podle klíče. Vrátí false pokud klíč neexistuje.
    // Thread-safe vůči processBlock (atomic float zápis).
    virtual bool setParam(const std::string& key, float value) = 0;
    virtual bool getParam(const std::string& key, float& out)  const = 0;

    // Popis všech parametrů core — GUI je zobrazí jako sliders.
    // Volá se při inicializaci a po každém setParam pro aktualizaci hodnot.
    virtual std::vector<CoreParamDesc> describeParams() const = 0;

    // ── Vizualizace — GUI vlákno ──────────────────────────────────────────────
    // Snapshot aktuálního stavu pro render. Smí alokovat (mimo RT vlákno).
    virtual CoreVizState getVizState() const = 0;

    // ── Meta informace ────────────────────────────────────────────────────────
    virtual std::string coreName()    const = 0;  // "ResonatorCore", "FMCore", ...
    virtual std::string coreVersion() const = 0;  // "1.0"
    virtual bool        isLoaded()    const = 0;
};
```

### Proč `setParam(string key, float value)` místo konkrétních setterů?

Původní VoiceManager má 20+ individuálních setterů (`setSynthBeatScale`,
`setSynthNoiseLevel`…). Problém: každé nové jádro by přidalo další sadu setterů
do enginu, GUI a SysEx handleru.

Generický `setParam("beat_scale", 1.5f)` umožní:
- GUI zobrazit sliders automaticky z `describeParams()`
- SysEx handler mapovat parametr ID → klíč bez změny kódu per core
- RenderServer přijmout `{"cmd":"set_config","params":{"beat_scale":1.5}}`
  bez změny protokolu

**Nevýhoda:** ztráta typové bezpečnosti, string lookup overhead.
Řešení: implementace si interně mapuje string → index při inicializaci
(hash map s `unordered_map<string,int>` — setup cost, ne RT cost).

### Proč `processBlock` additivní (ne replace)?

Engine nuluje buffery jednou, pak volá `processBlock` core. Pokud bychom
v budoucnu měli víc core (layering), každý přidá do stejných bufferů.
Aktuálně jeden core, ale návyk dobrý.

---

## 4. Registrace a factory

```cpp
// synth/synth_core_registry.h

#pragma once
#include "i_synth_core.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

// Factory funkce pro vytvoření core instance.
using SynthCoreFactory = std::function<std::unique_ptr<ISynthCore>()>;

class SynthCoreRegistry {
public:
    static SynthCoreRegistry& instance();

    void registerCore(const std::string& name, SynthCoreFactory factory);
    std::unique_ptr<ISynthCore> create(const std::string& name) const;

    // Seznam registrovaných jmen pro CLI / GUI výběr.
    std::vector<std::string> availableCores() const;

private:
    std::unordered_map<std::string, SynthCoreFactory> factories_;
};

// Macro pro snadnou registraci (volat v .cpp souboru core):
//   REGISTER_SYNTH_CORE("ResonatorCore", ResonatorCore)
#define REGISTER_SYNTH_CORE(name, CoreClass) \
    static bool _registered_##CoreClass = []{ \
        SynthCoreRegistry::instance().registerCore( \
            name, []{ return std::make_unique<CoreClass>(); }); \
        return true; \
    }();
```

### Proč static registry místo compile-time list?

Chceme přidat core bez úpravy CMakeLists nebo enginu. Stačí přidat nový
`.cpp` soubor s `REGISTER_SYNTH_CORE(...)` — linker ho zahrne a static
inicializátor ho zaregistruje. Engine se nezměnit.

**Riziko:** static initialization order fiasco — registry musí existovat
před voláním `registerCore`. Řešení: Meyers singleton (`static instance`
v těle funkce garantuje pořadí inicializace v C++11+).

---

## 5. ResonatorCore — refaktoring stávajícího kódu

Stávající `ResonatorVoice` + `ResonatorVoiceManager` se zabalí do:

```cpp
// synth/cores/resonator_core.h

#pragma once
#include "../i_synth_core.h"
#include "../note_params.h"
#include "../synth_config.h"
#include "../resonator_voice.h"
#include "../note_lut.h"
#include <array>
#include <atomic>

class ResonatorCore final : public ISynthCore {
public:
    bool load(const std::string& params_path, float sr, Logger& logger) override;
    void setSampleRate(float sr) override;

    void noteOn(uint8_t midi, uint8_t velocity) override;
    void noteOff(uint8_t midi) override;
    void sustainPedal(bool down) override;
    void allNotesOff() override;

    bool processBlock(float* out_l, float* out_r, int n_samples) noexcept override;

    bool setParam(const std::string& key, float value) override;
    bool getParam(const std::string& key, float& out) const override;
    std::vector<CoreParamDesc> describeParams() const override;

    CoreVizState getVizState() const override;

    std::string coreName()    const override { return "ResonatorCore"; }
    std::string coreVersion() const override { return "1.0"; }
    bool        isLoaded()    const override { return initialized_; }

private:
    NoteLUT                                           lut_;
    std::array<ResonatorVoice, MIDI_COUNT>           voices_;
    SynthConfig                                       cfg_;
    std::unordered_map<std::string, std::size_t>     param_index_;

    float    sample_rate_  = 44100.f;
    bool     initialized_  = false;
    Logger*  logger_       = nullptr;

    // MIDI state (thread-safe)
    std::atomic<bool>   sustain_{false};
    std::array<bool, 128> delayed_offs_{};

    void buildParamIndex();
    void handleNoteOn(uint8_t midi, uint8_t vel) noexcept;
    void handleNoteOff(uint8_t midi) noexcept;
};

REGISTER_SYNTH_CORE("ResonatorCore", ResonatorCore)
```

`SynthConfig` zůstane jako interní struct ResonatorCore — není součástí
obecného API. `describeParams()` vrátí jeho fieldy jako `CoreParamDesc`.

### Co se ODSTRANÍ z VoiceManager

- DSP chain (přejde do CoreEngine)
- LFO panning (přejde do CoreEngine nebo zůstane jako CoreParam)
- Přímé gettery/settery SynthConfig (nahradí setParam/getParam)
- Interleaved output varianta (engine ji nepotřebuje)

### Co zůstane ve VoiceManager (přejmenovaném na ResonatorCore)

- NoteLUT a načítání params.json
- Pole 88 ResonatorVoice
- Voice stealing logika
- MIDI sustain handling

---

## 6. CoreEngine — obal nad ISynthCore

```cpp
// synth/core_engine.h

#pragma once
#include "i_synth_core.h"
#include "../dsp/dsp_chain.h"
#include "../sampler/core_logger.h"
#include <memory>
#include <string>

class CoreEngine {
public:
    // Vytvoří a inicializuje core podle jména.
    bool initialize(const std::string& core_name,
                    const std::string& params_path,
                    const std::string& config_json_path,
                    float              sr,
                    Logger&            logger);

    // RT: zpracuje blok a aplikuje master DSP.
    // Tato funkce nuluje buffery, volá core->processBlock, pak DspChain.
    void processBlock(float* out_interleaved, uint32_t frame_count);

    // Předá MIDI dál do core (thread-safe).
    void noteOn(uint8_t midi, uint8_t velocity);
    void noteOff(uint8_t midi);
    void sustainPedal(uint8_t val);

    ISynthCore* core()     { return core_.get(); }
    DspChain*   dspChain() { return &dsp_;       }

private:
    std::unique_ptr<ISynthCore> core_;
    DspChain                    dsp_;
    float*                      buf_l_ = nullptr;
    float*                      buf_r_ = nullptr;
    int                         block_size_ = 256;
};
```

`CoreEngine` nahrazuje `ResonatorEngine` — je generický, neví nic o
konkrétním core. `ResonatorEngine` se stane thin wrapperem nad `CoreEngine`
pro zpětnou kompatibilitu nebo zmizí.

---

## 7. CLI a výběr core

```
IthacaCoreResonator --core ResonatorCore
                    --params soundbanks/params-ks-grand-ft.json
                    --config soundbanks/params-ks-grand-ft.synth_config.json
                    --port 0
```

Výchozí `--core ResonatorCore` (pro zpětnou kompatibilitu).

`--core` název se předá `SynthCoreRegistry::instance().create(name)`.
Pokud neexistuje, engine vypíše dostupná jádra a skončí s chybou.

---

## 8. GUI — data-driven panel

GUI přestane používat `getSynthConfig()` přímo. Místo toho:

```cpp
// Při inicializaci nebo po setParam:
auto params = engine->core()->describeParams();

// Render sliderů:
for (auto& p : params) {
    if (ImGui::SliderFloat(p.label.c_str(), &p.value, p.min, p.max)) {
        engine->core()->setParam(p.key, p.value);
    }
}
```

Skupiny (`p.group`) definují záložky nebo sekce v panelu.

### Vizualizace not

```cpp
CoreVizState viz = engine->core()->getVizState();

// Aktivní hlasy v piano vizualizaci:
for (auto& v : viz.active_voices)
    drawActiveKey(v.midi);

// Detail poslední noty — partials tabulka:
if (viz.last_note_valid)
    drawPartialTable(viz.last_note.partials);
```

GUI se nezměnit při přidání nového core — zobrazí jeho parametry automaticky.

---

## 9. Kritická místa a potenciální úzká hrdla

### 9.1 Thread safety setParam ↔ processBlock

`processBlock` běží v RT vlákně (miniaudio callback). GUI volá `setParam`
z GUI vlákna. Problém: `SynthConfig` není thread-safe — zápis floatu není
atomický na všech platformách (není garantovaný load-store order).

**Současné řešení (VoiceManager):** přímý zápis `synth_cfg_.beat_scale = v`
— funguje v praxi (x86 float write je atomický), ale není správné C++.

**Navržené řešení:** lock-free ring buffer příkazů (stejný pattern jako
MIDI queue). GUI zapíše `{key, value}` do fronty; RT vlákno ji vyčte
na začátku `processBlock`. Overhead minimální, správné chování zaručeno.

```cpp
struct ParamChange { int param_idx; float value; };
// Ring buffer (64 položek) — dost pro burst ze GUI
```

### 9.2 getVizState alokuje paměť

`getVizState()` vrátí `vector<CoreVoiceViz>` s `vector<CorePartialViz>`.
Alokace na heap ze GUI vlákna — OK, ale vyžaduje přístup k voice datům
která se mění v RT vlákně.

Řešení: **snapshot** — core zkopíruje stav hlasů do přednastaveného
bufferu (na stacku nebo statický) pod krátkým spinlockem. GUI nikdy
nevidí konzistentní RT stav, jen snapshot z posledního volání.

Alternativa: double-buffer bez locku (RT vlákno píše do jednoho,
GUI čte z druhého, atomic swap). Složitější, ale lockfree.

### 9.3 SysEx mapování

SysEx protokol dnes mapuje pevné ID → SynthConfig field. Při pluggable core
se ID musí mapovat dynamicky na `setParam(key, value)`.

Dvě možnosti:
- **Pevná tabulka per core** — každé jádro registruje své SysEx mapování
- **Obecný SET_PARAM příkaz** — SysEx pošle string klíč + float (větší packet)

Doporučení: udržet zpětnou kompatibilitu pevné tabulky pro ResonatorCore,
přidat obecný `SET_PARAM_BY_NAME` SysEx opcode pro ostatní.

### 9.4 Offline renderer a RenderServer

Aktuálně `OfflineRenderer` obsahuje vlastní `ResonatorVoiceManager`.
Po refaktoringu použije `ISynthCore` přímo — bez audio zařízení, bez DspChain.

```cpp
class OfflineRenderer {
    std::unique_ptr<ISynthCore> core_;
    // render loop: noteOn → processBlock × N → collect output
};
```

RenderServer zůstane beze změny — jen bude používat `OfflineRenderer`
s `ISynthCore` uvnitř.

### 9.5 Zpětná kompatibilita params formátu

Soubor `params-ks-grand-nn.json` je specifický pro ResonatorCore (má sekce
`samples`, `partials`, `eq_gains_db`…). Jiné core bude mít jiný formát.
CLI `--params` předá cestu bez validace formátu — validace je na core.

Pokud uživatel omylem předá ResonatorCore params jinému jádru, dostane
srozumitelnou chybu z `load()`. Žádný crash.

---

## 10. Navržená struktura souborů

```
synth/
    i_synth_core.h              ← ISynthCore + VizState + CoreParamDesc
    synth_core_registry.h/.cpp  ← factory registrace
    core_engine.h/.cpp          ← generický engine (nahrazuje resonator_engine)
    cores/
        resonator_core.h/.cpp   ← ResonatorCore (refaktoring VoiceManager)
        ← budoucí: fm_core.h/.cpp, sampler_core.h/.cpp, …
    resonator_voice.h/.cpp      ← beze změny (ResonatorCore ho používá interně)
    note_params.h               ← beze změny
    synth_config.h              ← beze změny (interní pro ResonatorCore)
    synth_config_io.h           ← beze změny
    voice_manager.h/.cpp        ← deprecated, odstraní se po migraci
```

---

## 11. Migrace — pořadí kroků

Doporučené pořadí aby byla vždy buildovatelná verze:

1. Přidat `i_synth_core.h` (interface + VizState structs) — bez změny buildů
2. Přidat `synth_core_registry.h/.cpp` — bez změny buildů
3. Napsat `ResonatorCore` jako wrapper nad stávajícím `VoiceManager`
   (deleguje, nepřepisuje) — ověřit funkčnost
4. Přidat `CoreEngine` vedle stávajícího `ResonatorEngine` (paralelně)
5. Přepnout `main.cpp` na `CoreEngine` — první produkční test
6. Přepnout `gui_main.cpp` — data-driven GUI panel
7. Přepnout `OfflineRenderer` — ověřit RenderServer
8. Smazat `VoiceManager` (nebo ponechat jako `ResonatorVoiceManager`
   interní implementační detail `ResonatorCore`)
9. Napsat první alternativní core (SimpleOscCore) jako ověření API

---

## 12. Co tento návrh NEŘEŠÍ (vědomě)

- **Core layering** (víc core najednou, additivní mix) — pro budoucnost
- **Plugin formát** (VST/CLAP/LV2) — mimo scope, ale API je kompatibilní
- **Hot-swap core za běhu** — vyžaduje careful lifetime management, odloženo
- **Serializace stavu core** (preset systém) — `describeParams()` jako základ
- **Per-partial DSP** (convolution reverb per partial) — zůstane uvnitř core
