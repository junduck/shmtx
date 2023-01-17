# A faster shared_mutex for mostly shared access

## Usage

shmtx is a header-only library. Just include the header file and you're good to go.

## Dependencies

shmtx depends on C++11 and Boost::predef.

## Template parameter

Both `shmtx::shared_mutex` and `shmtx::shared_mutex_pool` have a template parameter `N`. It specifies the number of shared locking slots. Normally you would want it to be set to the number of threads that contend for the mutex, but it is totally fine if set to a smaller number.

## shmtx::shared_mutex

`shmtx::shared_mutex` has the same interface as `std::shared_mutex`. You can use it as a drop-in replacement for `std::shared_mutex`.

```cpp
#include <shmtx/shared_mutex.hpp>
#include <shared_mutex>

int main() {
    shmtx::shared_mutex<8> m;

    {
      std::shared_lock lock(m);
      // do something read-only
    }

    {
      std::unique_lock lock(m);
      // exclusive access
    }
}
```

## shmtx::shared_mutex_pool

`shtmx::shared_mutex` uses thread_local storage to assign shared locking slots for each thread. This should not cause any problem for usual use cases. However, if you have workers running in a thread pool, in case of the worker being suspended and rescheduled to another thread while holding a shared lock, you will have undefined behaviour and crashing is the best outcome you can expect. To avoid this, you can use `shmtx::shared_mutex_pool` instead. It creates and manages mutexes that are not bound to specific threads and all mutexes created by the same pool share the same underlying shared_mutex.

```cpp

#include <shmtx/shared_mutex.hpp>

using mutex_pool_type = shmtx::shared_mutex_pool<8>;
using mutex_type = typename mutex_pool_type::mutex_type;

struct worker_t {
  mutex_type mutex;

  worker_t(mutex_pool_type& pool) : mutex(pool.create_mutex()) {}

  void run() {
    {
      std::shared_lock lock(mutex);
      // do something read-only
    }

    {
      std::unique_lock lock(mutex);
      // exclusive access
    }

    // worker can be safely suspended and resumed on another thread, even while holding a lock
  }
};

```

## Reentrancy

`shmtx::shared_mutex` and `shmtx::shared_mutex_pool` are not reentrant and do not perform any checks for reentrancy. If a thread tries to aquire a shared lock while already holding an exclusive lock or vice versa, deadlock will occur. However a thread can acquire a shared lock multiple times but I don't why anyone would want to do that.

## Implementation details

In case of no lock being held, acquiring an exclusive lock is N atomic loads and CAS operations, where N is the number of shared locking slots. Releasing an exclusive lock is N atomic stores. In case of no exclusive lock being held, acquiring a shared lock is one atomic load and one CAS operation. Releasing a shared lock is one fetch and set operation. `shmtx::shared_mutex` spins on contention and the spin is relxed after certain number of iterations. To fine tune the behavior, see next section.

## Tuning

You should always benchmark your code and milage may vary. However, there are some knobs you can turn to fine tune the behavior of `shmtx::shared_mutex` and `shmtx::shared_mutex_pool`.

1. Template parameter `N` specifies the number of shared locking slots. The more slots you have, the less contention you will have on shared locks. However, the more slots you have, the more memory is used and slower exclusive lock performs. You should benchmark it and see what works best for you.

2. SHMTX_MAXSPIN macro defines every SHMTX_MAXSPIN number of iterations to spin on contention before nano-sleeping. Default is 32. The spinloop DOES NOT yield() or sched_yield().

3. SHMTX_CACHELINE macro defines the cacheline size of the CPU. It tries to get the number from `std::hardware_destructive_interference_size` but if it fails, it defaults to 64. You can override it by defining the macro before including the header file.

## License

shmtx is licensed under the MIT License.
