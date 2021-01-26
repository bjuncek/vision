#pragma once

#include "defs.h"

namespace ffmpeg {

/**
 * FFMPEG library utility functions.
 */

namespace Util {
void SaveAvFrame(AVFrame* avFrame, std::string str);
void SaveYComponent(
    unsigned char* buf,
    int wrap,
    int xsize,
    int ysize,
    char* filename);
std::string generateErrorDesc(int errorCode);
size_t serialize(const AVSubtitle& sub, ByteStorage* out);
bool deserialize(const ByteStorage& buf, AVSubtitle* sub);
size_t size(const AVSubtitle& sub);
void setFormatDimensions(
    size_t& destW,
    size_t& destH,
    size_t userW,
    size_t userH,
    size_t srcW,
    size_t srcH,
    size_t minDimension,
    size_t maxDimension,
    size_t cropImage);
bool validateVideoFormat(const VideoFormat& format);
} // namespace Util
} // namespace ffmpeg
