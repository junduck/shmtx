#include "shmtx/shmtx.hpp"
#include "gtest/gtest.h"

#include <shared_mutex>
#include <thread>
#include <vector>

constexpr size_t work_size = 50'000;
constexpr size_t slots = 32;
constexpr size_t reader = 8;
constexpr size_t writer = 2;

using mutex_type = shmtx::shared_mutex<slots>;

struct work_type_vector {
  size_t n;
  std::vector<size_t> pop, push;

  explicit work_type_vector(size_t n) : n(n) {
    pop.reserve(n);
    push.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      pop.push_back(i);
    }
  }

  bool done() const { return push.size() == n; }
};

struct worker_reader {
  work_type_vector &work;
  mutex_type &mtx;
  std::atomic<bool> &start;

  explicit worker_reader(work_type_vector &w, mutex_type &m,
                         std::atomic<bool> &s)
      : work(w), mtx(m), start(s) {}

  void operator()() {
    while (!start.load(std::memory_order_relaxed))
      ;
    while (true) {
      std::shared_lock lock(mtx);
      if (work.done()) {
        break;
      }
      for (size_t j = 0; j < work.push.size(); ++j) {
        auto wp = work.push[j];
        ptrdiff_t diff = wp - j;
        EXPECT_EQ(diff, 0);
      }
    }
  }
};

struct worker_writer {
  work_type_vector &work;
  mutex_type &mtx;
  std::atomic<bool> &start;

  explicit worker_writer(work_type_vector &w, mutex_type &m,
                         std::atomic<bool> &s)
      : work(w), mtx(m), start(s) {}

  void operator()() {
    while (!start.load(std::memory_order_relaxed))
      ;
    while (true) {
      std::lock_guard lock(mtx);
      if (work.done()) {
        break;
      }
      work.push.push_back(work.pop[work.push.size()]);
    }
  }
};

TEST(shmtx, basic) {
  work_type_vector work(work_size);
  std::atomic<bool> start(false);
  mutex_type mtx{};
  std::vector<std::function<void()>> workers;
  std::vector<std::thread> th;

  for (size_t i = 0; i < reader; ++i) {
    workers.emplace_back(
        std::function<void()>(worker_reader(work, mtx, start)));
  }
  for (size_t i = 0; i < writer; ++i) {
    workers.emplace_back(
        std::function<void()>(worker_writer(work, mtx, start)));
  }

  for (size_t i = 0; i < (reader + writer); ++i) {
    th.emplace_back(workers[i]);
  }

  start.store(true, std::memory_order_relaxed);
  for (auto &t : th) {
    t.join();
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
