#pragma once
#include "Types.h"
#include <functional>

namespace vision {

using PublishCallback = std::function<void(const std::vector<SeatFrameState>&)>;

class Publisher {
public:
    void setCallback(PublishCallback cb) { cb_ = std::move(cb); }
    void publish(const std::vector<SeatFrameState>& states) {
        if (cb_) cb_(states);
    }
private:
    PublishCallback cb_;
};

} // namespace vision