# IthacaCoreResonator — Historie vývoje

Chronologický přehled toho, co bylo uděláno, proč a s jakými výsledky.

---

## Phase 1 — Signal Chain Parity

**Proč:** C++ syntetizér a Python referenční implementace (`physics_synth.py`) produkovaly různý zvuk přesto, že měly implementovat stejnou fyziku. Bez parity nelze C++ použít jako verifikační nástroj ani jako proxy v training loopu.

**Co bylo uděláno:**

- Opraven pořadí signal chain — sjednocen s Pythonem: `decorr → EQ → M/S → onset_ramp`. Původní C++ pořadí bylo jiné.
- SIMD downgrade z AVX2 na AVX pro kompatibilitu s AMD Steamroller (Athlon X4 870K).
- Post-hoc RMS normalizace v offline rendereru — místo per-block kalibrace se celý buffer normalizuje až po dokončení syntézy. Odpovídá chování Pythonu.
- Opraven vel_band round-trip: `(vel*127+6)/7` místo `vel*127/7`. Stará formula mapovala band=3 → MIDI vel 54 → band zpět = 2, tedy špatná velocity vrstva.
- `compare_cpp_python.py`: přidán vel_gain do Python syntézy, aby odpovídal training loop konvenci.

**Výsledky (params-ks-grand-ft.json):**

| Nota | MRSTFT (C++) | Stoch. floor | Status |
|---|---|---|---|
| A0  MIDI 21 vel 3 | 2.36 | 0.83 | šumová nota |
| C2  MIDI 36 vel 3 | 1.09 | 1.42 | konvergováno |
| C3  MIDI 48 vel 3 | 1.30 | ~1.3 | konvergováno |
| C4  MIDI 60 vel 3 | 1.27 | 1.10 | konvergováno |
| C5  MIDI 72 vel 3 | 1.21 | ~1.2 | konvergováno |
| C7  MIDI 96 vel 3 | 1.41 | ~1.4 | konvergováno |

Střed klavíru (C2–C7, mezzo velocity) konvergoval na stochastický floor. Divergence jen u šumem dominovaných not (A0, C8) — vyřešeno v Phase 3.

---

## Phase 2 — Modulární refaktor C++ kódu

**Proč:** Původní kód měl všechny soubory v jednom `synth/` adresáři bez struktury. S přibývajícím počtem souborů bylo obtížné orientovat se a rozlišit, co patří k real-time enginu a co k offline rendereru.

**Co bylo uděláno:**

Kód přeskupen do tří modulů:

```
synth/core/       — sdílené typy bez audio závislostí
                    (NoteParams, SynthConfig, BiquadEQ, NoteLUT)
synth/realtime/   — real-time engine
                    (ResonatorVoice, VoiceManager, ResonatorEngine, MidiInput, Sysex)
synth/offline/    — offline renderer a RenderServer
                    (OfflineRenderer, RenderServer)
```

CMakeLists.txt aktualizován — přidány include paths pro všechny tři subadresáře, takže interní `#include "file.h"` direktivy uvnitř synth/ nebylo nutné měnit. Opraveny všechny externí include cesty v `main.cpp`, `gui_main.cpp`, `server_main.cpp` a `gui/`.

Build ověřen pro všechny tři targety.

---

## Phase 3 — Kvalita a tooling

### KI-1: Noise formula fix

**Proč:** Offline renderer produkoval špatný poměr šum/signál. Noise vrstva byla pre-scaled konstantou `target_rms * vel_gain`, ale post-hoc RMS normalizace pak celý buffer rescalovala — noise se přeškáloval dvakrát.

**Co bylo uděláno:**

Přidán flag `SynthConfig.offline_mode`. Offline path používá absolutní úroveň šumu (`floor_rms * noise_level`), post-hoc normalizace pak nastaví finální level a poměr noise/signal zůstane správný. Real-time path zůstala beze změny (pre-scaled, kauzální).

### GUI live slidery

**Proč:** GUI zobrazovalo SynthConfig parametry jen read-only. Nebylo možné experimentovat s parametry za běhu.

**Co bylo uděláno:**

`beat_scale` (0–3), `eq_strength` (0–1) a `noise_level` (0–2) jsou nyní interaktivní slidery v GUI. Změna se okamžitě projeví na zvuku přes `ResonatorEngine::setSynthBeatScale/EqStrength/NoiseLevel()`.

### Regresní baseline pro compare_cpp_python.py

**Proč:** Ověření C++ vs Python parity bylo čistě manuální — nebylo poznat, jestli nová změna v kódu C++ paritu zhorší.

**Co bylo uděláno:**

Přidány flagy `--save-baseline` a `--check`. Baseline se uloží do `analysis/regression_baseline.json` s 0.3 margin. `--check` pak dá PASS/FAIL per nota a vrátí exit code 1 při selhání.

### NaN guard v closed_loop_finetune.py

**Proč:** Při předchozím fine-tuningu explodoval loss na NaN v epoch 120. Gradient clipping (max_norm=1.0) situaci nevyřešil, protože loss byl NaN ještě před `backward()`.

**Co bylo uděláno:**

Přidána kontrola `torch.isfinite(loss_i)` před `backward()`. Problematický batch se přeskočí, trénink pokračuje.

---

## Trénink první generace (profile.pt)

**Proč:** Potřebujeme soundbanku s fyzikálními parametry pro 88 not × 8 velocity vrstev. Parametry se nedají změřit ručně — je jich příliš mnoho.

**Pipeline:**
1. `train_instrument_profile.py` — supervisory trénink na extrahovaných parametrech z nahrávek
2. `closed_loop_finetune.py` — MRSTFT fine-tuning pomocí diferenciabilního Python proxy syntetizéru

**Výsledek:** `params-ks-grand-ft.json` — první generace soundbanky, stále v použití jako referenční profil.

---

## Trénink druhé generace (profile-v2.pt)

**Proč:** Původní model (`profile.pt`) byl trénován s 1-string proxy syntetizérem. Nový `torch_synth.py` implementuje 2-string model s `phi_net` — sub-sítí predikující relativní fázi mezi strunami (`phi_diff`). Stará váhy jsou nekompatibilní s novou architekturou.

**Co bylo uděláno:**

Celý pipeline spuštěn od nuly s novou architekturou:

| Krok | Výsledek |
|---|---|
| Supervisory training, 800 epoch | eval loss 1.298 → `profile-v2.pt` |
| MRSTFT fine-tuning, 300 epoch | mean MRSTFT 8.369 → `profile-v2-finetuned.pt` |
| Generování params JSON | 88×8 banka → `params-ks-grand-v2-nn.json` |

**Poznámka k číslům:** MRSTFT ~8.4 je zde měřen vůči originálním nahrávkám (syntetizér vs skutečný klavír) — to je inherentně těžší úkol než C++ vs Python porovnání (kde dosahujeme ~1.3). Fyzikální model není perfektní replika nástroje.

---

## Aktuální stav a co zbývá

| Úkol | Stav |
|---|---|
| Signal chain parity C++ ↔ Python | ✅ |
| Modulární refaktor kódu | ✅ |
| KI-1 noise fix, GUI slidery, regresní baseline | ✅ |
| V2 model natrénován | ✅ |
| Ověření v2 profilu přes compare_cpp_python --batch | ⏳ |
| Generování finetuned params JSON z profile-v2-finetuned.pt | ⏳ |
| GUI slidery pro pan_spread, stereo_decorr, onset_ms, vel_gamma | ⏳ |
