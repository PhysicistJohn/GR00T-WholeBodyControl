/**
 * @file sanitizer_state_rollback.hpp
 * @brief Transactional rollback for command-sanitizer state.
 *
 * The low-command sanitizer updates its slew reference and accounting before
 * the DDS writer reports whether a local dispatch actually succeeded. A
 * rejected candidate or failed local dispatch must leave that state exactly as
 * it was so the next command is constrained relative to the last command that
 * was really published.
 */
#pragma once

#include <cstdint>

template <typename QTargetsT, typename TimePointT> class SanitizerStateRollback {
  public:
    SanitizerStateRollback(QTargetsT& last_sent_q_target, bool& has_last_sent_q_target,
                           std::uint64_t& sanitize_events_total, TimePointT& last_sanitize_log_time)
      : last_sent_q_target_(last_sent_q_target),
        has_last_sent_q_target_(has_last_sent_q_target),
        sanitize_events_total_(sanitize_events_total),
        last_sanitize_log_time_(last_sanitize_log_time),
        saved_last_sent_q_target_(last_sent_q_target),
        saved_has_last_sent_q_target_(has_last_sent_q_target),
        saved_sanitize_events_total_(sanitize_events_total),
        saved_last_sanitize_log_time_(last_sanitize_log_time) {}

    SanitizerStateRollback(const SanitizerStateRollback&) = delete;
    SanitizerStateRollback& operator=(const SanitizerStateRollback&) = delete;

    ~SanitizerStateRollback() {
      if (active_) { Restore(); }
    }

    /// Retain sanitizer mutations after a successful local DDS Write().
    void Commit() noexcept { active_ = false; }

    /// Restore the pre-sanitization state immediately and make rollback idempotent.
    void Restore() {
      last_sent_q_target_ = saved_last_sent_q_target_;
      has_last_sent_q_target_ = saved_has_last_sent_q_target_;
      sanitize_events_total_ = saved_sanitize_events_total_;
      last_sanitize_log_time_ = saved_last_sanitize_log_time_;
      active_ = false;
    }
  private:
    QTargetsT& last_sent_q_target_;
    bool& has_last_sent_q_target_;
    std::uint64_t& sanitize_events_total_;
    TimePointT& last_sanitize_log_time_;

    const QTargetsT saved_last_sent_q_target_;
    const bool saved_has_last_sent_q_target_;
    const std::uint64_t saved_sanitize_events_total_;
    const TimePointT saved_last_sanitize_log_time_;
    bool active_ = true;
};
