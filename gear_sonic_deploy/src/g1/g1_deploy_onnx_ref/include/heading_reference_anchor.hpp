#pragma once

#include <array>

/**
 * Keeps the reference-motion orientation used to anchor policy heading.
 *
 * Streamed motion commonly consists of a succession of one-frame windows, so
 * playback frame zero is not evidence of a new stream.  The anchor is captured
 * once and remains stable until the caller explicitly requests a reset.
 */
class HeadingReferenceAnchor {
public:
    using Quaternion = std::array<double, 4>;

    void CaptureIfNeeded(const Quaternion& orientation) {
        if (!initialized_) {
            Capture(orientation);
        }
    }

    void Capture(const Quaternion& orientation) {
        orientation_ = orientation;
        initialized_ = true;
    }

    [[nodiscard]] bool IsInitialized() const {
        return initialized_;
    }

    [[nodiscard]] const Quaternion& Orientation() const {
        return orientation_;
    }

private:
    Quaternion orientation_{1.0, 0.0, 0.0, 0.0};
    bool initialized_ = false;
};
