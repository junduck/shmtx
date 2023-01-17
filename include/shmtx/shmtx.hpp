#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <new>
#include <thread>

#include <boost/predef.h>
#if BOOST_ARCH_X86 || BOOST_ARCH_X86_64
#include <emmintrin.h>
#elif BOOST_ARCH_ARM || BOOST_ARCH_ARM64
#include <arm_acle.h>
#endif

#if !defined(SHMTX_MAXSPIN)
#define SHMTX_MAXSPIN 32
#endif

namespace shmtx {

namespace arch {

#if BOOST_ARCH_X86 || BOOST_ARCH_X86_64
inline void spin_pause() { _mm_pause(); }
#elif BOOST_ARCH_ARM || BOOST_ARCH_ARM64
#if BOOST_COMP_GNUC
// GCC does not provide __yield() in arm_acle.h
inline void spin_pause() { asm volatile("yield" ::: "memory"); }
#else
inline void spin_pause() { __yield(); }
#endif // BOOST_COMP_GNUC
#else
inline void spin_pause() {}
#endif // BOOST_ARCH_X86 || BOOST_ARCH_X86_64

// implementing spinlock in userland is a bad idea,
// let's relax it a little bit and hope for the best
inline void spin_relax(int &nspin) {
  if (nspin++ == SHMTX_MAXSPIN) {
    nspin = 0;
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
  }
  spin_pause();
}

#if defined(SHMTX_CACHELINE)
constexpr size_t cacheline = SHMTX_CACHELINE;
#elif defined(__cpp_lib_hardware_interference_size)
constexpr size_t cacheline = std::hardware_destructive_interference_size;
#else
constexpr size_t cacheline = 64;
#endif // SHMTX_CACHELINE

} // namespace arch

namespace detail {

template <size_t N> class shmtx_impl {
  struct alignas(arch::cacheline) state_t {
    // Highest bit is reserved to hold exclusive lock so in theory we can have
    // 2^63 shared locks per slot in a 64-bit system
    constexpr static size_t exclusive = size_t(1) << (sizeof(size_t) * 8 - 1);
    constexpr static size_t unlocked = size_t(0);
    std::atomic<size_t> state;
  };

  std::array<state_t, N> slot{};
  auto get_state(size_t idx) -> std::atomic<size_t> & {
    return slot[idx].state;
  }

public:
  constexpr shmtx_impl() = default;
  ~shmtx_impl() = default;

  // shared lock

  // when locking shared, we only need to spin on owned slot
  void lock_slot_shared(size_t idx) noexcept {
    auto retry = 0;
    while (true) {
      auto state = get_state(idx).load(std::memory_order_relaxed);
      // exclusive lock is held on this slot, now spin
      if (state == state_t::exclusive) {
        arch::spin_relax(retry);
        continue;
      }
      // An exclusive lock could be acquired during this gap
      // so we will need to reload the state if CAS fails
      if (get_state(idx).compare_exchange_weak(state, state + 1,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
        break;
      }
    }
  }

  void unlock_slot_shared(size_t idx) noexcept {
    // since shared lock is held, we can just decrement shared count
    // there is no sanity check here, it's the caller's responsibility
    // to make sure the correct lock is held on the correct slot
    get_state(idx).fetch_sub(1, std::memory_order_release);
  }

  bool try_lock_slot_shared(size_t idx) noexcept {
    auto state = get_state(idx).load(std::memory_order_relaxed);
    // exclusive lock is held on this slot, can't acquire shared lock
    if (state == state_t::exclusive) {
      return false;
    }
    // we could spuriously fail here but that's how try_lock works right?
    return get_state(idx).compare_exchange_weak(
        state, state + 1, std::memory_order_acquire, std::memory_order_relaxed);
  }

  // exclusive lock

  // when locking exclusively, we need to spin on all slots
  void lock_exclusive() noexcept {
    for (size_t i = 0; i < N; ++i) {
      auto retry = 0;
      while (true) {
        auto state = get_state(i).load(std::memory_order_relaxed);
        // A shared or exclusive lock is held on this slot, now spin
        if (state) {
          arch::spin_relax(retry);
          continue;
        }
        // A lock could be acquired in this gap
        // we will need to reload the state if CAS fails
        if (get_state(i).compare_exchange_weak(state, state_t::exclusive,
                                               std::memory_order_acquire,
                                               std::memory_order_relaxed)) {
          break;
        }
      }
    }
  }

  void unlock_exclusive() noexcept {
    for (size_t i = N; i--;) {
      // since this is an exclusive lock, we can just set the state to
      // state_t::unlocked
      get_state(i).store(state_t::unlocked, std::memory_order_release);
    }
  }

  bool try_lock_exclusive() noexcept {
    for (size_t i = 0; i < N; ++i) {
      auto state = get_state(i).load(std::memory_order_relaxed);
      if (state) {
        // we can't acquire a lock on this slot, unlock all the locks we
        // previously acquired before returning
        goto FAILED_UNLOCK;
      }
      if (get_state(i).compare_exchange_weak(state, state_t::exclusive,
                                             std::memory_order_acquire,
                                             std::memory_order_relaxed)) {
        continue;
      }
      // CAS failed, unlock all the locks we previously acquired before
      // returning
    FAILED_UNLOCK:
      for (size_t j = i; j--;) {
        get_state(j).store(state_t::unlocked, std::memory_order_release);
      }
      return false;
    }
    return true;
  }
};

// check power of 2
constexpr bool is_pow2(size_t n) { return n && ((n & (n - 1)) == 0); }

// round up to the next power of 2
constexpr size_t roundup_pow2(size_t n) {
  if (is_pow2(n)) {
    return n;
  }
  size_t i = 1;
  while (i < n) {
    i <<= 1;
  }
  return i;
}

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
template <size_t N> class shared_mutex {
  detail::shmtx_impl<N> impl{};
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
  ~shared_mutex() = default;

  // non-copyable
  shared_mutex(const shared_mutex &) = delete;
  shared_mutex &operator=(const shared_mutex &) = delete;

  // shared lock
  void lock_shared() noexcept { impl.lock_slot_shared(my_idx()); }
  void unlock_shared() noexcept { impl.unlock_slot_shared(my_idx()); }
  bool try_lock_shared() noexcept {
    return impl.try_lock_slot_shared(my_idx());
  }

  // exclusive lock
  void lock() noexcept { impl.lock_exclusive(); }
  void unlock() noexcept { impl.unlock_exclusive(); }
  bool try_lock() noexcept { return impl.try_lock_exclusive(); }
};

template <size_t N> std::atomic<size_t> shared_mutex<N>::nthread{0};

/**
 * @brief A shared mutex pool
 *
 * @tparam N number of slots to reduce contention
 */
template <size_t N> class shared_mutex_pool {
  constexpr static size_t nslot = detail::roundup_pow2(N);
  std::shared_ptr<detail::shmtx_impl<nslot>> impl =
      std::make_shared<detail::shmtx_impl<nslot>>();
  std::atomic<size_t> next_slot{0};

  // round robin slot allocation
  size_t request_slot() {
    return next_slot.fetch_add(1, std::memory_order_relaxed) & (nslot - 1);
  }

public:
  class worker_mutex {
    friend class shared_mutex_pool;

    std::shared_ptr<detail::shmtx_impl<nslot>> impl;
    size_t idx;

    worker_mutex() = default;

  public:
    ~worker_mutex() = default;

    // shared lock
    void lock_shared() noexcept { impl->lock_slot_shared(idx); }
    void unlock_shared() noexcept { impl->unlock_slot_shared(idx); }
    bool try_lock_shared() noexcept { return impl->try_lock_slot_shared(idx); }

    // exclusive lock
    void lock() noexcept { impl->lock_exclusive(); }
    void unlock() noexcept { impl->unlock_exclusive(); }
    bool try_lock() noexcept { return impl->try_lock_exclusive(); }
  };

  using mutex_type = worker_mutex;

  /**
   * @brief Create a mutex object for a thread pool worker
   *
   * @details Create an owned mutex object for a worker. All worker_mutex
   * created by this function will share the same underlying mutex. However this
   * mutex can be safely used even if the worker is rescheduled to a different
   * thread.
   *
   * @return worker_mutex
   */
  auto create_mutex() -> worker_mutex {
    worker_mutex m;
    m.impl = impl;
    m.idx = request_slot();
    return m;
  }
};

} // namespace shmtx
