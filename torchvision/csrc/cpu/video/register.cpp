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
        .def("next_list", &Video::Next)
        .def("next_tensor", &Video::NextNoPTS)
        .def("next_list_dummy_tensor", &Video::NextListDummyTensor)
        .def("next_tensor_dummy_tensor", &Video::NextDummyTensorOnly)
        .def("next_tensor_dummy_noalloc", &Video::NextDummyTensorOnlyNoAlloc)
        .def("next_int_numframes", &Video::nextDebugNoReturn)
        .def("fullvideo_tensor", &Video::debugReadVideoTensor)
        .def("fullvideo_numframes", &Video::debugReadVideoNumFrames)
        .def("tb", &Video::tbTest)
        .def("tbTensor", &Video::tbTestTensor);


} //namespace
#endif
