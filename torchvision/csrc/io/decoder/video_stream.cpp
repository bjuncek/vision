#include "video_stream.h"
#include <c10/util/Logging.h>
#include "stream.h"
#include "util.h"

using namespace std;
namespace ffmpeg {

void SaveAFrame(AVFrame* avFrame) {
  // This is a nead and serialised way to dump YUV420P data
  // into a readable tensor

  // accessing code:
  //   YUV = np.fromfile("filename.binary", dtype=np.uint8)
  //   Y = YUV[0:w*h].reshape(h,w)
  //   #Take next px / 4 samples as U
  //   U = YUV[px:(px*5)//4].reshape(h//2,w//2)
  //   #Take next px / 4 samples as V
  //   V = YUV[(px*5)//4:(px*6)//4].reshape(h//2,w//2)

  FILE* fDump = fopen("dumpedAVFrame.binary", "ab");

  uint32_t pitchY = avFrame->linesize[0];
  uint32_t pitchU = avFrame->linesize[1];
  uint32_t pitchV = avFrame->linesize[2];

  uint8_t* avY = avFrame->data[0];
  uint8_t* avU = avFrame->data[1];
  uint8_t* avV = avFrame->data[2];

  for (uint32_t i = 0; i < avFrame->height; i++) {
    fwrite(avY, avFrame->width, 1, fDump);
    avY += pitchY;
  }

  for (uint32_t i = 0; i < avFrame->height / 2; i++) {
    fwrite(avU, avFrame->width / 2, 1, fDump);
    avU += pitchU;
  }

  for (uint32_t i = 0; i < avFrame->height / 2; i++) {
    fwrite(avV, avFrame->width / 2, 1, fDump);
    avV += pitchV;
  }

  fclose(fDump);
}

namespace {
bool operator==(const VideoFormat& x, const AVFrame& y) {
  return x.width == y.width && x.height == y.height && x.format == y.format;
}

bool operator==(const VideoFormat& x, const AVCodecContext& y) {
  return x.width == y.width && x.height == y.height && x.format == y.pix_fmt;
}

VideoFormat& toVideoFormat(VideoFormat& x, const AVFrame& y) {
  x.width = y.width;
  x.height = y.height;
  x.format = y.format;
  return x;
}

VideoFormat& toVideoFormat(VideoFormat& x, const AVCodecContext& y) {
  x.width = y.width;
  x.height = y.height;
  x.format = y.pix_fmt;
  return x;
}
} // namespace

VideoStream::VideoStream(
    AVFormatContext* inputCtx,
    int index,
    bool convertPtsToWallTime,
    const VideoFormat& format,
    int64_t loggingUuid)
    : Stream(
          inputCtx,
          MediaFormat::makeMediaFormat(format, index),
          convertPtsToWallTime,
          loggingUuid) {}

VideoStream::~VideoStream() {
  if (sampler_) {
    sampler_->shutdown();
    sampler_.reset();
  }
}

int VideoStream::initFormat() {
  // set output format
  if (!Util::validateVideoFormat(format_.format.video)) {
    LOG(ERROR) << "Invalid video format"
               << ", width: " << format_.format.video.width
               << ", height: " << format_.format.video.height
               << ", format: " << format_.format.video.format
               << ", minDimension: " << format_.format.video.minDimension
               << ", crop: " << format_.format.video.cropImage;
    return -1;
  }

  // keep aspect ratio
  Util::setFormatDimensions(
      format_.format.video.width,
      format_.format.video.height,
      format_.format.video.width,
      format_.format.video.height,
      codecCtx_->width,
      codecCtx_->height,
      format_.format.video.minDimension,
      format_.format.video.maxDimension,
      0);

  if (format_.format.video.format == AV_PIX_FMT_NONE) {
    format_.format.video.format = codecCtx_->pix_fmt;
  }
  return format_.format.video.width != 0 && format_.format.video.height != 0 &&
          format_.format.video.format != AV_PIX_FMT_NONE
      ? 0
      : -1;
}

int VideoStream::copyFrameBytes(ByteStorage* out, bool flush) {
  if (!sampler_) {
    sampler_ = std::make_unique<VideoSampler>(SWS_AREA, loggingUuid_);
  }

  int fsize = avpicture_get_size(
      (AVPixelFormat)frame_->format, frame_->width, frame_->height);

  // check if input format gets changed
  if (flush ? !(sampler_->getInputFormat().video == *codecCtx_)
            : !(sampler_->getInputFormat().video == *frame_)) {
    // - reinit sampler
    SamplerParameters params;
    params.type = format_.type;
    params.out = format_.format;
    params.in = FormatUnion(0);
    flush ? toVideoFormat(params.in.video, *codecCtx_)
          : toVideoFormat(params.in.video, *frame_);
    if (!sampler_->init(params)) {
      return -1;
    }

    int fsize2 = avpicture_get_size(
        (AVPixelFormat)format_.format.video.format,
        params.in.video.width,
        params.in.video.height);

    LOG(ERROR) << "Extimated FSIZE of FRAME format: " << fsize;
    LOG(ERROR) << "Extimated FSIZE of FORMAT format : " << fsize2;

    LOG(ERROR) << "Set input video sampler format"
               << ", width: " << params.in.video.width
               << ", height: " << params.in.video.height
               << ", format: " << params.in.video.format
               << " : output video sampler format"
               << ", width: " << format_.format.video.width
               << ", height: " << format_.format.video.height
               << ", format: " << format_.format.video.format
               << ", minDimension: " << format_.format.video.minDimension
               << ", linesize (frame): " << frame_->linesize[0]
               << ", crop: " << format_.format.video.cropImage;
  }

  int resislav = sampler_->sample(flush ? nullptr : frame_, out);
  LOG(ERROR) << "expected out after sampler: " << resislav;
  LOG(ERROR) << "FSIZE of the OUT: " << out->length();

  // Please note: this saves YUV420P file in a planar form.
  // can be accessed (from python as follows)

  //   YUV = np.fromfile("inVideoStreamTestStuff.binary", dtype=np.uint8)
  //   Y = YUV[0:w*h].reshape(h,w)
  //   #Take next px / 4 samples as U
  //   U = YUV[px:(px*5)//4].reshape(h//2,w//2)
  //   #Take next px / 4 samples as V
  //   V = YUV[(px*5)//4:(px*6)//4].reshape(h//2,w//2)
  SaveAFrame(frame_);

  // this binary dump should be RGB
  FILE* pFile;
  pFile = fopen("inVideoStreamPostTransform.binary", "wb");
  fwrite(out->data(), 1, out->length(), pFile);
  fclose(pFile);

  return resislav;
}

void VideoStream::setHeader(DecoderHeader* header, bool flush) {
  Stream::setHeader(header, flush);
  if (!flush) { // no frames for video flush
    header->keyFrame = frame_->key_frame;
    header->fps = av_q2d(av_guess_frame_rate(
        inputCtx_, inputCtx_->streams[format_.stream], nullptr));
  }
}

} // namespace ffmpeg
