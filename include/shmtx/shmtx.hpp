#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <new>
#include <thread>

#define SHMTX_ARCH_X86 0
#define SHMTX_ARCH_ARM 0

#if defined(__i386__) || defined(__x86_64__)
#undef SHMTX_ARCH_X86
#define SHMTX_ARCH_X86 1
#elif defined(__arm__) || defined(__aarch64__)
#undef SHMTX_ARCH_ARM
#define SHMTX_ARCH_ARM 1
#endif

#define SHMTX_OS_MACOS 0
#if defined(__APPLE__)
#undef SHMTX_OS_MACOS
#define SHMTX_OS_MACOS 1
#endif

#define SHMTX_COMP_GCC 0
#if defined(__GNUC__) && !defined(__clang__)
#undef SHMTX_COMP_GCC
#define SHMTX_COMP_GCC __GNUC__
#endif

#if SHMTX_ARCH_X86
#include <emmintrin.h>
#elif SHMTX_ARCH_ARM && !SHMTX_COMP_GCC
#include <arm_acle.h>
#endif

// app defines

#if !defined(SHMTX_SPIN_RETRY)
#define SHMTX_SPIN_RETRY 0
#endif

#if !defined(SHMTX_CACHELINE_SIZE)
#define SHMTX_CACHELINE_SIZE 0
#endif

namespace shmtx {

namespace arch {
#if SHMTX_ARCH_X86
inline void spin_pause() { _mm_pause(); }
#elif SHMTX_ARCH_ARM
#if SHMTX_COMP_GCC
inline void spin_pause() {
  asm volatile("yield" ::: "memory");}
  //asm volatile("sevl; wfe" ::: "memory");
#else
inline void spin_pause() {
  __yield();
  //__sevl();
  //__wfe();
}
#endif
#else
inline void spin_pause() {}
#endif

#if SHMTX_SPIN_RETRY
constexpr inline int spin_retry = SHMTX_SPIN_RETRY;
#else
constexpr inline int spin_retry = 64;
#endif

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void spin_relax(T &n, int yield = spin_retry) {
  if (n++ > yield) {
    n = T{};
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
  }
  spin_pause();
}

#if SHMTX_CACHELINE_SIZE
constexpr inline size_t cacheline = SHMTX_CACHELINE_SIZE;
#elif SHMTX_OS_MACOS && SHMTX_ARCH_ARM
// Overwrite for Apple Silicon: GCC somehow reports 256
// std::hardware_destructive_interference_size
constexpr inline size_t cacheline = 128;
#elif defined(__cpp_lib_hardware_interference_size)
constexpr inline size_t cacheline = std::hardware_destructive_interference_size;
#else
constexpr inline size_t cacheline = 64;
#endif
} // namespace arch

namespace detail {
struct alignas(arch::cacheline) shmtx_slot {
  constexpr static size_t write = size_t(1) << (sizeof(size_t) * 8 - 1);
  constexpr static size_t write_clear = ~write;
  std::atomic<size_t> state;

  size_t get() const noexcept { return state.load(std::memory_order_relaxed); }
  size_t acq() const noexcept { return state.load(std::memory_order_acquire); }

  // write_enter/leave are NOT sync points:
  void write_enter() noexcept {
    state.fetch_or(write, std::memory_order_relaxed);
  }
  void write_leave() noexcept {
    state.fetch_and(write_clear, std::memory_order_relaxed);
  }

  void pub(size_t st = 0) noexcept {
    state.store(st, std::memory_order_release);
  }
  size_t dec() noexcept {
    return state.fetch_sub(1, std::memory_order_release);
  }
  bool cas(size_t &observed, size_t desired) noexcept {
    return state.compare_exchange_weak(observed, desired,
                                       std::memory_order_acquire,
                                       std::memory_order_relaxed);
  }
  size_t xchg(size_t val) noexcept {
    return state.exchange(val, std::memory_order_acquire);
  }
};

struct shmtx_impl {
  template <typename Iter> static void wlock(Iter slot) noexcept {
    auto retry = 0;
    while (slot->xchg(1))
      while (slot->get())
        arch::spin_relax(retry);
  }

  template <typename Iter> static bool try_wlock(Iter slot) noexcept {
    size_t expected = 0;
    return slot->cas(expected, 1); // acq
  }

  template <typename Iter> static void wunlock(Iter slot) noexcept {
    slot->pub(0); // rel
  }

  template <typename Iter> static void lock_sh(Iter slot) noexcept {
    auto retry = 0;
    while (true) {
      auto state = slot->get();
      if (state & shmtx_slot::write) {
        arch::spin_relax(retry);
      } else if (slot->cas(state, state + 1)) {
        // acq
        break;
      }
    }
  }

  template <typename Iter> static bool trylock_sh(Iter slot) noexcept {
    auto state = slot->get();
    if (state & shmtx_slot::write) {
      return false;
    } else {
      return slot->cas(state, state + 1); // acq
    }
  }

  template <typename Iter> static void unlock_sh(Iter slot) noexcept {
    slot->dec(); // rel
  }

  template <typename Iter> static void lock_ex(Iter first, Iter last) noexcept {
    wlock(last);
    for (auto slot = first; slot != last; ++slot) {
      slot->write_enter(); // nosync
    }
    auto retry = 0;
    for (auto slot = first; slot != last; ++slot) {
      while (slot->acq() != shmtx_slot::write) {
        // acq
        arch::spin_relax(retry);
      }
    }
  }

  template <typename Iter>
  static bool trylock_ex(Iter first, Iter last) noexcept {
    if (!try_wlock(last))
      return false;
    for (auto slot = first; slot != last; ++slot) {
      slot->write_enter(); // nosync
    }
    for (auto slot = first; slot != last; ++slot) {
      if (slot->acq() != shmtx_slot::write) {
        // acq
        goto FAILED_UNLOCK;
      }
    }
    return true;
  FAILED_UNLOCK:
    for (auto slot = first; slot != last; ++slot)
      slot->write_leave(); // nosync
    wunlock(last);         // rel
    return false;
  }

  template <typename Iter>
  static void unlock_ex(Iter first, Iter last) noexcept {
    for (auto slot = first; slot != last; ++slot)
      slot->pub(0); // rel
    wunlock(last);
  }

  template <typename Iter>
  static void upgrade_sh(Iter first, Iter last, Iter hold) noexcept {
    wlock(last);
    for (auto slot = first; slot != last; ++slot) {
      slot->write_enter();
    }
    unlock_sh(hold);
    auto retry = 0;
    for (auto slot = first; slot != last; ++slot) {
      while (slot->acq() != shmtx_slot::write) {
        arch::spin_relax(retry);
      }
    }
  }

  template <typename Iter>
  static void downgrade_ex(Iter first, Iter last, Iter hold) noexcept {
    for (auto slot = first; slot != last; ++slot)
      if (slot == hold)
        slot->pub(1);
      else
        slot->pub(0);
    wunlock(last);
  }

  // NOTE: try_upgrade_sh will most likely to fail and starve writer,
  // split to multiple stages for finer grained control.

  // try to acquire write lock then block new readers
  template <typename Iter>
  static bool prepare_upgrade(Iter first, Iter last) noexcept {
    if (!try_wlock(last))
      return false;
    for (auto slot = first; slot != last; ++slot) {
      slot->write_enter();
    }
    return true;
  }

  // try to acquire exclusive lock
  template <typename Iter>
  static bool commit_upgrade(Iter first, Iter last, Iter hold) noexcept {
    for (auto slot = first; slot != last; ++slot) {
      auto state = slot->acq();
      if ((slot == hold && state != (shmtx_slot::write + 1)) ||
          (slot != hold && state != shmtx_slot::write))
        return false;
    }
    unlock_sh(hold);
    return true;
  }

  // abort upgrade, unblock readers
  template <typename Iter>
  static void abort_upgrade(Iter first, Iter last) noexcept {
    for (auto slot = first; slot != last; ++slot)
      slot->write_leave();
    wunlock(last);
  }
};
} // namespace detail

/**
 * @brief A shared mutex
 *
 * @note shmtx::shared_mutex uses on thread_local storage to index each thread.
 * Since threads do not contend for a single shared lock, this mutex should be
 * faster in a mostly shared lock scenario. However, since it relies on
 * thread_local storage, it is not suitable for a thread pool and WILL NOT WORK
 * if execution context is rescheduled to a different thread. Use
 * shmtx::shared_mutex_pool to avoid this issue.
 *
 * @tparam N number of slots to reduce contention
 */
template <size_t N> class shared_mutex final : private detail::shmtx_impl {
  std::array<detail::shmtx_slot, N + 1> slots{};
  static std::atomic<size_t> nthread;
  // this thread's slot index
  static size_t my_idx() {
    // thread_local idx is only initialized once, so we don't care about the
    // slow modulo here
    static thread_local size_t idx =
        nthread.fetch_add(1, std::memory_order_relaxed) % N;
    return idx;
  }

public:
  constexpr shared_mutex() = default;

  // shared lock
  void lock_shared() noexcept { this->lock_sh(slots.data() + my_idx()); }
  void unlock_shared() noexcept { this->unlock_sh(slots.data() + my_idx()); }
  bool try_lock_shared() noexcept {
    return this->trylock_sh(slots.data() + my_idx());
  }

  // exclusive lock
  void lock() noexcept { this->lock_ex(slots.data(), slots.data() + N); }
  void unlock() noexcept { this->unlock_ex(slots.data(), slots.data() + N); }
  bool try_lock() noexcept {
    return this->trylock_ex(slots.data(), slots.data() + N);
  }

  // upgrade lock
  void upgrade() noexcept {
    this->upgrade_sh(slots.data(), slots.data() + N, slots.data() + my_idx());
  }

  // downgrade lock
  void downgrade() noexcept {
    this->downgrade_ex(slots.data(), slots.data() + N, slots.data() + my_idx());
  }
};

template <size_t N> std::atomic<size_t> shared_mutex<N>::nthread{0};

class shared_mutex_mgr {
  unsigned n;
  std::atomic<unsigned> next;
  std::unique_ptr<detail::shmtx_slot[]> slot;
  unsigned next_index() noexcept {
    return next.fetch_add(1, std::memory_order_relaxed) % n;
  }

public:
  class [[nodiscard]] mutex_type final : private detail::shmtx_impl {
    friend class shared_mutex_mgr;
    unsigned n, id;
    detail::shmtx_slot *base;

    mutex_type(size_t n, size_t id, detail::shmtx_slot *base) noexcept
        : n(n), id(id), base(base) {}

  public:
    void lock_shared() noexcept { this->lock_sh(base + id); }
    void unlock_shared() noexcept { this->unlock_sh(base + id); }
    bool try_lock_shared() noexcept { return this->trylock_sh(base + id); }

    void lock() noexcept { this->lock_ex(base, base + n); }
    void unlock() noexcept { this->unlock_ex(base, base + n); }
    bool try_lock() noexcept { return this->trylock_ex(base, base + n); }

    void upgrade() noexcept { this->upgrade_sh(base, base + n, base + id); }
    void downgrade() noexcept { this->downgrade_ex(base, base + n, base + id); }
  };

  explicit shared_mutex_mgr(unsigned n) noexcept
      : n(n), next(0), slot(std::make_unique<detail::shmtx_slot[]>(n + 1)) {}
  mutex_type create() noexcept {
    return mutex_type(n, next_index(), slot.get());
  }
  mutex_type get(unsigned id) const noexcept {
    return mutex_type(n, id, slot.get());
  }
};

class spsc_mutex final {
  using slot_t = detail::shmtx_slot;
  slot_t slot;

public:
  void lock_shared() noexcept {
    auto retry = 0;
    while (true) {
      auto state = slot.get();
      if (state & slot_t::write) {
        arch::spin_relax(retry);
      } else if (slot.cas(state, state + 1)) {
        break;
      }
    }
  }

  void unlock_shared() noexcept {
    slot.dec();
  }

  bool trylock_shared() noexcept {
    auto state = slot.get();
    if (state & slot_t::write) {
      return false;
    } else {
      return slot.cas(state, state + 1);
    }
  }

  void lock() noexcept {
    slot.write_enter();
    auto retry = 0;
    while (slot.acq() != slot_t::write) {
      arch::spin_relax(retry);
    }
  }

  void unlock() noexcept {
    slot.pub(0);
  }

  bool try_lock() noexcept {
    slot.write_enter();
    if (slot.acq() != slot_t::write) {
      slot.write_leave();
      return false;
    } else {
      return true;
    }
  }

  void upgrade() noexcept {
    slot.write_enter();
    slot.dec();
    auto retry = 0;
    while (slot.acq() != slot_t::write) {
      arch::spin_relax(retry);
    }
  }

  void downgrade() noexcept {
    slot.pub(1);
  }
};

} // namespace shmtx
