#ifndef REGISTER_H
#define REGISTER_H

#include "Video.h"

namespace {

static auto registerVideo =
    torch::class_<Video>("torchvision", "Video")
        .def(torch::init<std::string, std::string, bool>())
        .def("get_current_stream", &Video::getCurrentStream)
        .def("duration", &Video::getDuration)
        .def("fps", &Video::getFPS)
        .def("seek", &Video::Seek)
        .def("next", &Video::Next)
        .def("nextDebug", &Video::NextDebug)
        .def("nextDebugNoReturn", &Video::nextDebugNR)
        .def("debug_video", &Video::debugReadVideo)
        .def("debug_video_tensor", &Video::debugReadVideoToTensor);


} //namespace
#endif
