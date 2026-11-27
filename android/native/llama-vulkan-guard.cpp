#include "llama.h"

#include <android/set_abort_message.h>

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace {

[[noreturn]] void abort_inference(const char *operation, const char *error) noexcept {
    char message[1024];
    std::snprintf(message, sizeof(message), "%s failed (%s)", operation, error);
    android_set_abort_message(message);
    // A failed backend can leave outstanding GPU work. Restart the inference process with a fresh CPU model.
    std::abort();
}

template <typename Call>
decltype(auto) guard_inference(const char *operation, Call call) noexcept {
    try {
        return call();
    } catch (const std::exception &error) {
        abort_inference(operation, error.what());
    } catch (...) {
        abort_inference(operation, "unknown C++ exception");
    }
}

} // namespace

extern "C" decltype(llama_decode) __real_llama_decode;
extern "C" decltype(ggml_backend_sched_synchronize) __real_ggml_backend_sched_synchronize;

extern "C" int32_t __wrap_llama_decode(llama_context *context, llama_batch batch) noexcept {
    return guard_inference("llama_decode", [&] { return __real_llama_decode(context, batch); });
}

extern "C" void __wrap_ggml_backend_sched_synchronize(ggml_backend_sched_t scheduler) noexcept {
    guard_inference("ggml_backend_sched_synchronize", [&] { __real_ggml_backend_sched_synchronize(scheduler); });
}
