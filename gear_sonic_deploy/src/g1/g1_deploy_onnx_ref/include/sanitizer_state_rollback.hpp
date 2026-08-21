/**
 * @file sanitizer_state_rollback.hpp
 * @brief Transactional rollback for command-sanitizer state.
 *
 * The low-command sanitizer updates its slew reference before the DDS writer
 * reports whether a local dispatch actually succeeded. A rejected candidate
 * or failed local dispatch must leave that pre-write state exactly as it was
 * so the next command is constrained relative to the last command that was
 * really published. Telemetry is deliberately accounted only after a
 * successful local write and therefore is not rollback state.
 */
#pragma once

template <typename QTargetsT, typename TimePointT> class SanitizerStateRollback {
  public:
    SanitizerStateRollback(QTargetsT& last_sent_q_target, bool& has_last_sent_q_target,
                           TimePointT& last_sent_q_target_time)
      : last_sent_q_target_(last_sent_q_target),
        has_last_sent_q_target_(has_last_sent_q_target),
        last_sent_q_target_time_(last_sent_q_target_time),
        saved_last_sent_q_target_(last_sent_q_target),
        saved_has_last_sent_q_target_(has_last_sent_q_target),
        saved_last_sent_q_target_time_(last_sent_q_target_time) {}

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
      last_sent_q_target_time_ = saved_last_sent_q_target_time_;
      active_ = false;
    }
  private:
    QTargetsT& last_sent_q_target_;
    bool& has_last_sent_q_target_;
    TimePointT& last_sent_q_target_time_;

    const QTargetsT saved_last_sent_q_target_;
    const bool saved_has_last_sent_q_target_;
    const TimePointT saved_last_sent_q_target_time_;
    bool active_ = true;
};
