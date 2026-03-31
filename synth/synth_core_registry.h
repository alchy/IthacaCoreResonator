#pragma once
/*
 * synth_core_registry.h
 * ──────────────────────
 * Meyers-singleton factory registry for ISynthCore implementations.
 *
 * Usage — registering a core (place in the core's .cpp file, NOT header):
 *   REGISTER_SYNTH_CORE("SineCore", SineCore)
 *
 * Usage — creating a core by name:
 *   auto core = SynthCoreRegistry::instance().create("SineCore");
 *
 * The REGISTER_SYNTH_CORE macro uses a static-initializer trick: the core
 * registers itself when its translation unit is linked.  No modifications to
 * any central list are needed when a new core is added.
 *
 * Thread safety: registerCore() must only be called during static
 * initialization (before main()).  create() and availableCores() are
 * read-only after that and are safe to call concurrently.
 */

#include "i_synth_core.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using SynthCoreFactory = std::function<std::unique_ptr<ISynthCore>()>;

class SynthCoreRegistry {
public:
    // Meyers singleton — safe across TU boundaries (C++11+).
    static SynthCoreRegistry& instance();

    // Called by REGISTER_SYNTH_CORE at static init time.
    void registerCore(const std::string& name, SynthCoreFactory factory);

    // Instantiate a core by name; returns nullptr if name not found.
    std::unique_ptr<ISynthCore> create(const std::string& name) const;

    // Sorted list of registered core names.
    std::vector<std::string> availableCores() const;

private:
    std::unordered_map<std::string, SynthCoreFactory> factories_;
};

// ── Registration macro ────────────────────────────────────────────────────────
// Place once in the core's .cpp file (NOT in a header — would cause multiple
// registrations if the header is included from multiple translation units).
//
//   REGISTER_SYNTH_CORE("MyCore", MyCore)
//
// Expands to a static bool that calls registerCore() at static-init time.
// The lambda captures nothing; core instances are created on demand via
// std::make_unique<CoreClass>().

// Use an anonymous namespace so the static variable has internal linkage
// and the declaration is self-contained (no trailing semicolon needed at
// the call site, though one may be added without harm).
#define REGISTER_SYNTH_CORE(name, CoreClass)                              \
    namespace {                                                           \
        static const bool _synth_core_reg_##CoreClass = []() -> bool {   \
            SynthCoreRegistry::instance().registerCore(                   \
                name,                                                     \
                [] { return std::unique_ptr<ISynthCore>(new CoreClass()); }); \
            return true;                                                  \
        }();                                                              \
    }
