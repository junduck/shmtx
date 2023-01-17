#include "shmtx/shmtx.hpp"
#include "gtest/gtest.h"

#include <shared_mutex>
#include <thread>
#include <vector>

struct test_t {
  size_t v1, v2;
};

constexpr size_t nloops = 1000000;
constexpr size_t xrate_low = 10000;
constexpr size_t xrate_high = 10;

constexpr size_t nthreads_low = 4;
constexpr size_t nthreads_high = 32;
constexpr size_t nslot_low = 1;
constexpr size_t nslot_high = nthreads_low;
constexpr size_t nslot_more = nthreads_high * 2;

template <typename MUTEX>
auto create_worker(size_t nloops, size_t xrate, MUTEX &mtx, test_t &data,
                   std::atomic<bool> &start) {
  return [nloops, xrate, &mtx, &data, &start]() {
    while (!start.load(std::memory_order_relaxed))
      ;
    for (size_t i = 0; i < nloops; ++i) {
      if (i % xrate == 0) {
        std::unique_lock<MUTEX> lock(mtx);
        data.v1 += 1;
        data.v2 += 2;
      } else {
        std::shared_lock<MUTEX> lock(mtx);
        EXPECT_EQ(data.v1 * 2, data.v2);
      }
    }
  };
}

TEST(shmtx, std_shared_mutex) {
  test_t data{};
  std::shared_mutex mtx;
  std::atomic<bool> start{false};

  std::vector<std::thread> workers;
  for (size_t i = 0; i < nthreads_low; ++i) {
    workers.emplace_back(create_worker(nloops, xrate_low, mtx, data, start));
  }
  start.store(true, std::memory_order_relaxed);

  for (auto &worker : workers) {
    worker.join();
  }
}

TEST(shmtx, shmtx_shared_mutex_lowx_matched_slots) {
  test_t data{};
  shmtx::shared_mutex<nslot_low> mtx;
  std::atomic<bool> start{false};

  std::vector<std::thread> workers;
  for (size_t i = 0; i < nthreads_low; ++i) {
    workers.emplace_back(create_worker(nloops, xrate_low, mtx, data, start));
  }
  start.store(true, std::memory_order_relaxed);

  for (auto &worker : workers) {
    worker.join();
  }
}

TEST(shmtx, shmtx_shared_mutex_highx_matched_slots) {
  test_t data{};
  shmtx::shared_mutex<nslot_low> mtx;
  std::atomic<bool> start{false};

  std::vector<std::thread> workers;
  for (size_t i = 0; i < nthreads_low; ++i) {
    workers.emplace_back(create_worker(nloops, xrate_high, mtx, data, start));
  }
  start.store(true, std::memory_order_relaxed);

  for (auto &worker : workers) {
    worker.join();
  }
}

TEST(shmtx, shmtx_shared_mutex_lowx_low_slots) {
  test_t data{};
  shmtx::shared_mutex<nslot_low> mtx;
  std::atomic<bool> start{false};

  std::vector<std::thread> workers;
  for (size_t i = 0; i < nthreads_low; ++i) {
    workers.emplace_back(create_worker(nloops, xrate_low, mtx, data, start));
  }
  start.store(true, std::memory_order_relaxed);

  for (auto &worker : workers) {
    worker.join();
  }
}

TEST(shmtx, shmtx_shared_mutex_lowx_low_contention) {
  test_t data{};
  shmtx::shared_mutex<nslot_more> mtx;
  std::atomic<bool> start{false};

  std::vector<std::thread> workers;
  for (size_t i = 0; i < nthreads_high; ++i) {
    workers.emplace_back(create_worker(nloops, xrate_low, mtx, data, start));
  }
  start.store(true, std::memory_order_relaxed);

  for (auto &worker : workers) {
    worker.join();
  }
}

TEST(shmtx, shmtx_shared_mutex_lowx_high_contention) {
  test_t data{};
  shmtx::shared_mutex<nslot_high> mtx;
  std::atomic<bool> start{false};

  std::vector<std::thread> workers;
  for (size_t i = 0; i < nthreads_high; ++i) {
    workers.emplace_back(create_worker(nloops, xrate_low, mtx, data, start));
  }
  start.store(true, std::memory_order_relaxed);

  for (auto &worker : workers) {
    worker.join();
  }
}

TEST(shmtx, shmtx_shared_mutex_lowx_higher_contention) {
  test_t data{};
  shmtx::shared_mutex<nslot_low> mtx;
  std::atomic<bool> start{false};

  std::vector<std::thread> workers;
  for (size_t i = 0; i < nthreads_high; ++i) {
    workers.emplace_back(create_worker(nloops, xrate_low, mtx, data, start));
  }
  start.store(true, std::memory_order_relaxed);

  for (auto &worker : workers) {
    worker.join();
  }
}

template <size_t N> struct worker_t {
  using pool_type = shmtx::shared_mutex_pool<N>;
  using mutex_type = typename pool_type::mutex_type;

  size_t nloops, xrate;
  test_t *data;
  std::atomic<bool> *start;
  pool_type *mtx_pool;
  mutex_type mtx;

  worker_t(size_t nloops, size_t xrate, test_t *data, std::atomic<bool> *start,
           pool_type *mtx_pool)
      : nloops(nloops), xrate(xrate), data(data), start(start),
        mtx_pool(mtx_pool), mtx(mtx_pool->create_mutex()) {}

  void operator()() {
    while (!start->load(std::memory_order_relaxed))
      ;
    for (size_t i = 0; i < nloops; ++i) {
      if (i % xrate == 0) {
        std::unique_lock lock(mtx);
        data->v1 += 1;
        data->v2 += 2;
      } else {
        std::shared_lock lock(mtx);
        EXPECT_EQ(data->v1 * 2, data->v2);
      }
    }
  }
};

TEST(shmtx, shmtx_shared_mutex_pool) {
  test_t data{};
  shmtx::shared_mutex_pool<nslot_high> mtx_pool;
  std::atomic<bool> start{false};

  std::vector<worker_t<nslot_high>> workers;
  for (size_t i = 0; i < nthreads_high; ++i) {
    workers.emplace_back(nloops, xrate_low, &data, &start, &mtx_pool);
  }

  std::vector<std::thread> threads;
  for (auto &worker : workers) {
    threads.emplace_back(worker);
  }

  start.store(true, std::memory_order_relaxed);

  for (auto &thread : threads) {
    thread.join();
  }

  EXPECT_EQ(data.v1, nloops * nthreads_high / xrate_low);

  start.store(false, std::memory_order_relaxed);

  // simulate workers being rescheduled on different threads

  for (auto &worker : workers) {
    threads.emplace_back(worker);
  }

  start.store(true, std::memory_order_relaxed);

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  EXPECT_EQ(data.v1, nloops * nthreads_high / xrate_low * 2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
