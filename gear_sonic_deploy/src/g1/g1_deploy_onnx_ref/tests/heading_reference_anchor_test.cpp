#include "heading_reference_anchor.hpp"

#include <array>
#include <cassert>
#include <iostream>

int main() {
    HeadingReferenceAnchor anchor;
    const std::array<double, 4> initial{1.0, 0.0, 0.0, 0.0};
    const std::array<double, 4> next_streamed_frame{0.9239, 0.0, 0.0, 0.3827};
    const std::array<double, 4> explicit_reset{0.7071, 0.0, 0.0, 0.7071};

    assert(!anchor.IsInitialized());
    anchor.CaptureIfNeeded(initial);
    assert(anchor.IsInitialized());
    assert(anchor.Orientation() == initial);

    // A one-frame streaming window reports playback frame zero on every
    // packet. Normal tick processing must not turn each packet into a new
    // physical heading anchor.
    anchor.CaptureIfNeeded(next_streamed_frame);
    assert(anchor.Orientation() == initial);

    // An explicit operator/protocol reset is allowed to recapture the anchor.
    anchor.Capture(explicit_reset);
    assert(anchor.Orientation() == explicit_reset);

    std::cout << "heading reference anchor policy passed\n";
    return 0;
}
