/*
 * synth_core_registry.cpp
 * ────────────────────────
 */
#include "synth_core_registry.h"
#include <algorithm>

SynthCoreRegistry& SynthCoreRegistry::instance() {
    static SynthCoreRegistry inst;
    return inst;
}

void SynthCoreRegistry::registerCore(const std::string& name,
                                      SynthCoreFactory   factory) {
    factories_[name] = std::move(factory);
}

std::unique_ptr<ISynthCore>
SynthCoreRegistry::create(const std::string& name) const {
    auto it = factories_.find(name);
    if (it == factories_.end()) return nullptr;
    return it->second();
}

std::vector<std::string> SynthCoreRegistry::availableCores() const {
    std::vector<std::string> names;
    names.reserve(factories_.size());
    for (const auto& p : factories_)
        names.push_back(p.first);
    std::sort(names.begin(), names.end());
    return names;
}
