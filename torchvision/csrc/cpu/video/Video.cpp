
#include "Video.h"
#include <c10/util/Logging.h>
#include <torch/script.h>
#include "defs.h"
#include "memory_buffer.h"
#include "sync_decoder.h"

using namespace std;
using namespace ffmpeg;

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension
// #ifdef _WIN32
// #if PY_MAJOR_VERSION < 3
// PyMODINIT_FUNC init_video_reader(void) {
//   // No need to do anything.
//   return NULL;
// }
// #else
// PyMODINIT_FUNC PyInit_video_reader(void) {
//   // No need to do anything.
//   return NULL;
// }
// #endif
// #endif

const size_t decoderTimeoutMs = 600000;
const AVSampleFormat defaultAudioSampleFormat = AV_SAMPLE_FMT_FLT;
// A jitter can be added to the end of the range to avoid conversion/rounding
// error, small value 100us won't be enough to select the next frame, but enough
// to compensate rounding error due to the multiple conversions.
const size_t timeBaseJitterUs = 100;

// returns number of written bytes
template <typename T>
size_t fillTensorList(
    DecoderOutputMessage& msgs,
    torch::Tensor& frame,
    torch::Tensor& framePts) {
  // set up PTS data
  const auto& msg = msgs;

  float* framePtsData = framePts.data_ptr<float>();

  float pts_s = float(float(msg.header.pts) * 1e-6);
  framePtsData[0] = pts_s;

  T* frameData = frame.numel() > 0 ? frame.data_ptr<T>() : nullptr;

  if (frameData) {
    auto sizeInBytes = msg.payload->length();
    memcpy(frameData, msg.payload->data(), sizeInBytes);
  }
  return sizeof(T);
}

size_t fillVideoTensor(
    DecoderOutputMessage& msgs,
    torch::Tensor& videoFrame,
    torch::Tensor& videoFramePts) {
  return fillTensorList<uint8_t>(msgs, videoFrame, videoFramePts);
}

size_t fillAudioTensor(
    DecoderOutputMessage& msgs,
    torch::Tensor& audioFrame,
    torch::Tensor& audioFramePts) {
  return fillTensorList<float>(msgs, audioFrame, audioFramePts);
}

std::string parse_type_to_string(const std::string& stream_string) {
  static const std::array<std::pair<std::string, MediaType>, 4> types = {{
      {"video", TYPE_VIDEO},
      {"audio", TYPE_AUDIO},
      {"subtitle", TYPE_SUBTITLE},
      {"cc", TYPE_CC},
  }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [stream_string](const std::pair<std::string, MediaType>& p) {
        return p.first == stream_string;
      });
  if (device != types.end()) {
    return device->first;
  }
  AT_ERROR("Expected one of [audio, video, subtitle, cc] ", stream_string);
}

MediaType parse_type_to_mt(const std::string& stream_string) {
  static const std::array<std::pair<std::string, MediaType>, 4> types = {{
      {"video", TYPE_VIDEO},
      {"audio", TYPE_AUDIO},
      {"subtitle", TYPE_SUBTITLE},
      {"cc", TYPE_CC},
  }};
  auto device = std::find_if(
      types.begin(),
      types.end(),
      [stream_string](const std::pair<std::string, MediaType>& p) {
        return p.first == stream_string;
      });
  if (device != types.end()) {
    return device->second;
  }
  AT_ERROR("Expected one of [audio, video, subtitle, cc] ", stream_string);
}

std::tuple<std::string, long> _parseStream(const std::string& streamString) {
  TORCH_CHECK(!streamString.empty(), "Stream string must not be empty");
  static const std::regex regex("([a-zA-Z_]+)(?::([1-9]\\d*|0))?");
  std::smatch match;

  TORCH_CHECK(
      std::regex_match(streamString, match, regex),
      "Invalid stream string: '",
      streamString,
      "'");

  std::string type_ = "video";
  type_ = parse_type_to_string(match[1].str());
  long index_ = -1;
  if (match[2].matched) {
    try {
      index_ = c10::stoi(match[2].str());
    } catch (const std::exception&) {
      AT_ERROR(
          "Could not parse device index '",
          match[2].str(),
          "' in device string '",
          streamString,
          "'");
    }
  }
  return std::make_tuple(type_, index_);
}

void Video::_getDecoderParams(
    int64_t videoStartS,
    int64_t getPtsOnly,
    std::string stream,
    long stream_id = -1,
    bool all_streams = false,
    double seekFrameMarginUs = 10) {
  int64_t videoStartUs = int64_t(videoStartS * 1e6);

  params.timeoutMs = decoderTimeoutMs;
  params.startOffset = videoStartUs;
  params.seekAccuracy = 10;
  params.headerOnly = false;

  params.preventStaleness = false; // not sure what this is about

  if (all_streams == true) {
    MediaFormat format;
    format.stream = -2;
    format.type = TYPE_AUDIO;
    params.formats.insert(format);

    format.type = TYPE_VIDEO;
    format.stream = -2;
    format.format.video.width = 0;
    format.format.video.height = 0;
    format.format.video.cropImage = 0;
    params.formats.insert(format);

    format.type = TYPE_SUBTITLE;
    format.stream = -2;
    params.formats.insert(format);

    format.type = TYPE_CC;
    format.stream = -2;
    params.formats.insert(format);
  } else {
    // parse stream type
    MediaType stream_type = parse_type_to_mt(stream);

    // TODO: reset params.formats
    std::set<MediaFormat> formats;
    params.formats = formats;
    // Define new format
    MediaFormat format;
    format.type = stream_type;
    format.stream = stream_id;
    if (stream_type == TYPE_VIDEO) {
      format.format.video.width = 0;
      format.format.video.height = 0;
      format.format.video.cropImage = 0;
    }
    params.formats.insert(format);
  }

} // _get decoder params

Video::Video(std::string videoPath, std::string stream, bool isReadFile) {
  // parse stream information
  current_stream = _parseStream(stream);
  // note that in the initial call we want to get all streams
  Video::_getDecoderParams(
      0, // video start
      0, // headerOnly
      get<0>(current_stream), // stream info - remove that
      long(-1), // stream_id parsed from info above change to -2
      true // read all streams
  );

  std::string logMessage, logType;

  // TODO: add read from memory option
  params.uri = videoPath;
  logType = "file";
  logMessage = videoPath;

  std::vector<double> audioFPS, videoFPS, ccFPS, subsFPS;
  std::vector<double> audioDuration, videoDuration, ccDuration, subsDuration;
  std::vector<double> audioTB, videoTB, ccTB, subsTB;

  // calback and metadata defined in struct
  succeeded = decoder.init(params, std::move(callback), &metadata);
  if (succeeded) {
    for (const auto& header : metadata) {
      double fps = double(header.fps);
      double timeBase = double(header.num) / double(header.den);
      double duration = double(header.duration) * 1e-6; // * timeBase;

      if (header.format.type == TYPE_VIDEO) {
        videoMetadata = header;
        videoFPS.push_back(fps);
        videoDuration.push_back(duration);

        int oh = header.format.format.video.height;
        int ow = header.format.format.video.width;
        int nc = 3;
        dummy = torch::ones({nc, oh, ow}, torch::kByte);
      } else if (header.format.type == TYPE_AUDIO) {
        audioFPS.push_back(fps);
        audioDuration.push_back(duration);
      } else if (header.format.type == TYPE_CC) {
        ccFPS.push_back(fps);
        ccDuration.push_back(duration);
      } else if (header.format.type == TYPE_SUBTITLE) {
        subsFPS.push_back(fps);
        subsDuration.push_back(duration);
      };
    }
  }
  streamFPS.insert({{"video", videoFPS}, {"audio", audioFPS}});
  streamDuration.insert({{"video", videoDuration}, {"audio", audioDuration}});

  succeeded = Video::_setCurrentStream();
  LOG(INFO) << "\nDecoder inited with: " << succeeded << "\n";
  if (get<1>(current_stream) != -1) {
    LOG(INFO)
        << "Stream index set to " << get<1>(current_stream)
        << ". If you encounter trouble, consider switching it to automatic stream discovery. \n";
  }
} // video

bool Video::_setCurrentStream() {
  double ts = 0;
  if (seekTS > 0) {
    ts = seekTS;
  }

  _getDecoderParams(
      ts, // video start
      0, // headerOnly
      get<0>(current_stream), // stream
      long(get<1>(
          current_stream)), // stream_id parsed from info above change to -2
      false // read all streams
  );

  // calback and metadata defined in Video.h
  return (decoder.init(params, std::move(callback), &metadata));
}

std::tuple<std::string, int64_t> Video::getCurrentStream() const {
  return current_stream;
}

std::vector<double> Video::getFPS(std::string stream) const {
  // add safety check
  if (stream.empty()) {
    stream = get<0>(current_stream);
  }
  auto stream_tpl = _parseStream(stream);
  std::string stream_str = get<0>(stream_tpl);
  // check if the stream exists
  return streamFPS.at(stream_str);
}

std::vector<double> Video::getDuration(std::string stream) const {
  // add safety check
  if (stream.empty()) {
    stream = get<0>(current_stream);
  }
  auto stream_tpl = _parseStream(stream);
  std::string stream_str = get<0>(stream_tpl);
  // check if the stream exists
  return streamDuration.at(stream_str);
}

void Video::Seek(double ts, bool any_frame = false) {
  // initialize the class variables used for seeking and retrurn
  video_any_frame = any_frame;
  seekTS = ts;
  doSeek = true;
}

// next returns the torch list
torch::List<torch::Tensor> Video::Next(std::string stream) {

  bool newInit = false;
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
      // cout << "Reinitializing the decoder again \n";
    }
  }

  // if failing to decode simply return 0 (note, maybe
  // raise an exeption otherwise)
  torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);

  // first decode the frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;

    // then initialize the output variables based on type
    size_t expectedWrittenBytes = 0;

    if (format.type == TYPE_VIDEO) {
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      outFrame = torch::zeros({outHeight, outWidth, numChannels}, torch::kByte);
      expectedWrittenBytes = outHeight * outWidth * numChannels;
      // std::cout << expectedWrittenBytes;
    } else if (format.type == TYPE_AUDIO) {
      int outAudioChannels = format.format.audio.channels;
      int bytesPerSample = av_get_bytes_per_sample(
          static_cast<AVSampleFormat>(format.format.audio.format));
      int frameSizeTotal = out.payload->length();

      CHECK_EQ(frameSizeTotal % (outAudioChannels * bytesPerSample), 0);
      int numAudioSamples =
          frameSizeTotal / (outAudioChannels * bytesPerSample);

      outFrame =
          torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);

      expectedWrittenBytes = numAudioSamples * outAudioChannels * sizeof(float);
    }

    // std::cout << "Successfully allocated tensors to the dimension \n";
    // if not in seek mode or only looking at the keyframes,
    // return the immediate next frame
    if ((seekTS == -1) || (video_any_frame == false)) {
      // std::cout << "In non-seek mode stuff is happening \n";
      if (format.type == TYPE_VIDEO) {
        auto numberWrittenBytes = fillVideoTensor(out, outFrame, framePTS);
      } else {
        auto numberWrittenBytes = fillAudioTensor(out, outFrame, framePTS);
      }
      out.payload.reset();
    }
  } else {
    LOG(ERROR) << "Decoder run into a last iteration or has failed";
  }

  torch::List<torch::Tensor> result;
  result.push_back(outFrame);
  result.push_back(framePTS);
  return result;
}

//////// DELETE ALL UNDER THIS - DEBUGing issues

// next returns the tensor only
torch::Tensor Video::NextNoPTS(std::string stream) {

  bool newInit = false;
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
      // cout << "Reinitializing the decoder again \n";
    }
  }

  // if failing to decode simply return 0 (note, maybe
  // raise an exeption otherwise)
  torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);

  // first decode the frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;

    // then initialize the output variables based on type
    size_t expectedWrittenBytes = 0;

    if (format.type == TYPE_VIDEO) {
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      outFrame = torch::zeros({outHeight, outWidth, numChannels}, torch::kByte);
      expectedWrittenBytes = outHeight * outWidth * numChannels;
      // std::cout << expectedWrittenBytes;
    } else if (format.type == TYPE_AUDIO) {
      int outAudioChannels = format.format.audio.channels;
      int bytesPerSample = av_get_bytes_per_sample(
          static_cast<AVSampleFormat>(format.format.audio.format));
      int frameSizeTotal = out.payload->length();

      CHECK_EQ(frameSizeTotal % (outAudioChannels * bytesPerSample), 0);
      int numAudioSamples =
          frameSizeTotal / (outAudioChannels * bytesPerSample);

      outFrame =
          torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);

      expectedWrittenBytes = numAudioSamples * outAudioChannels * sizeof(float);
    }

    // std::cout << "Successfully allocated tensors to the dimension \n";
    // if not in seek mode or only looking at the keyframes,
    // return the immediate next frame
    if ((seekTS == -1) || (video_any_frame == false)) {
      // std::cout << "In non-seek mode stuff is happening \n";
      if (format.type == TYPE_VIDEO) {
        auto numberWrittenBytes = fillVideoTensor(out, outFrame, framePTS);
      } else {
        auto numberWrittenBytes = fillAudioTensor(out, outFrame, framePTS);
      }
      out.payload.reset();
    }
  } else {
    LOG(ERROR) << "Decoder run into a last iteration or has failed";
  }
  return outFrame;
}


torch::List<torch::Tensor> Video::NextWithMove(std::string stream) {

  bool newInit = false;
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
      // cout << "Reinitializing the decoder again \n";
    }
  }

  // if failing to decode simply return 0 (note, maybe
  // raise an exeption otherwise)
  torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);

  // first decode the frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;

    // then initialize the output variables based on type
    size_t expectedWrittenBytes = 0;

    // init PTS
    float* framePtsData = framePTS.data_ptr<float>();
    float pts_s = float(float(header.pts) * 1e-6);
    framePtsData[0] = pts_s;

    if (format.type == TYPE_VIDEO) {
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      outFrame = torch::zeros({outHeight, outWidth, numChannels}, torch::kByte);
      expectedWrittenBytes = outHeight * outWidth * numChannels;
      // std::cout << expectedWrittenBytes;

      auto options = torch::TensorOptions().dtype(torch::kByte);
      torch::IntArrayRef sizes = {outHeight, outWidth, numChannels};
      // void *dataPayload = (void*) out.payload->data();
      // if (out.payload->data()){
      //   cout << "ISSUe arising";
      // }
      outFrame = torch::from_blob((void*) out.payload->data(), sizes, options);
      // outFrame = torch::from_blob(const_cast<void*>(out.payload->data()), sizes, options);
    }
    // } else if (format.type == TYPE_AUDIO) {
    //   int outAudioChannels = format.format.audio.channels;
    //   int bytesPerSample = av_get_bytes_per_sample(
    //       static_cast<AVSampleFormat>(format.format.audio.format));
    //   int frameSizeTotal = out.payload->length();

    //   CHECK_EQ(frameSizeTotal % (outAudioChannels * bytesPerSample), 0);
    //   int numAudioSamples =
    //       frameSizeTotal / (outAudioChannels * bytesPerSample);

    //   outFrame =
    //       torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);
      
    //   auto options = torch::TensorOptions().dtype(torch::kFloat);
    //   torch::IntArrayRef sizes = {numAudioSamples, outAudioChannels};
    //   void *dataPayload = (void*) out.payload->data();
    //   outFrame = torch::from_blob(dataPayload, sizes);
    //   expectedWrittenBytes = numAudioSamples * outAudioChannels * sizeof(float);
    // }

    // std::cout << "Successfully allocated tensors to the dimension \n";
    // if not in seek mode or only looking at the keyframes,
    // return the immediate next frame
    // if ((seekTS == -1) || (video_any_frame == false)) {
    //   std::cout << "In non-seek mode stuff is happening \n";
      
    //   // if (format.type == TYPE_VIDEO) {
    //   //   auto numberWrittenBytes = fillVideoTensor(out, outFrame, framePTS);
    //   // } else {
    //   //   auto numberWrittenBytes = fillAudioTensor(out, outFrame, framePTS);
    //   // }
    //   // out.payload.reset();
    // }
  } else {
    LOG(ERROR) << "Decoder run into a last iteration or has failed";
  }
  torch::List<torch::Tensor> result;
  result.push_back(outFrame);
  result.push_back(framePTS);
  return result;
}


torch::Tensor Video::NextNoPTSWithMove(std::string stream) {

  bool newInit = false;
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
      // cout << "Reinitializing the decoder again \n";
    }
  }

  // if failing to decode simply return 0 (note, maybe
  // raise an exeption otherwise)
  torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);

  // first decode the frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;

    // then initialize the output variables based on type
    size_t expectedWrittenBytes = 0;

    // init PTS
    float* framePtsData = framePTS.data_ptr<float>();
    float pts_s = float(float(header.pts) * 1e-6);
    framePtsData[0] = pts_s;

    if (format.type == TYPE_VIDEO) {
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      outFrame = torch::zeros({outHeight, outWidth, numChannels}, torch::kByte);
      expectedWrittenBytes = outHeight * outWidth * numChannels;
      // std::cout << expectedWrittenBytes;

      auto options = torch::TensorOptions().dtype(torch::kInt8);
      torch::IntArrayRef sizes = {outHeight, outWidth, numChannels};
      // void *dataPayload = (void*) out.payload->data();
      // if (out.payload->data()){
      //   cout << "ISSUe arising";
      // }
      outFrame = torch::from_blob((void*) out.payload->data(), sizes, options);
      // outFrame = torch::from_blob(const_cast<void*>(out.payload->data()), sizes, options);
    }
    // } else if (format.type == TYPE_AUDIO) {
    //   int outAudioChannels = format.format.audio.channels;
    //   int bytesPerSample = av_get_bytes_per_sample(
    //       static_cast<AVSampleFormat>(format.format.audio.format));
    //   int frameSizeTotal = out.payload->length();

    //   CHECK_EQ(frameSizeTotal % (outAudioChannels * bytesPerSample), 0);
    //   int numAudioSamples =
    //       frameSizeTotal / (outAudioChannels * bytesPerSample);

    //   outFrame =
    //       torch::zeros({numAudioSamples, outAudioChannels}, torch::kFloat);
      
    //   auto options = torch::TensorOptions().dtype(torch::kFloat);
    //   torch::IntArrayRef sizes = {numAudioSamples, outAudioChannels};
    //   void *dataPayload = (void*) out.payload->data();
    //   outFrame = torch::from_blob(dataPayload, sizes);
    //   expectedWrittenBytes = numAudioSamples * outAudioChannels * sizeof(float);
    // }

    // std::cout << "Successfully allocated tensors to the dimension \n";
    // if not in seek mode or only looking at the keyframes,
    // return the immediate next frame
    // if ((seekTS == -1) || (video_any_frame == false)) {
    //   std::cout << "In non-seek mode stuff is happening \n";
      
    //   // if (format.type == TYPE_VIDEO) {
    //   //   auto numberWrittenBytes = fillVideoTensor(out, outFrame, framePTS);
    //   // } else {
    //   //   auto numberWrittenBytes = fillAudioTensor(out, outFrame, framePTS);
    //   // }
    //   // out.payload.reset();
    // }
  } else {
    LOG(ERROR) << "Decoder run into a last iteration or has failed";
  }
  return outFrame;
}






torch::List<torch::Tensor> Video::NextListDummyTensor(std::string stream) {

  bool newInit = false;
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
      // cout << "Reinitializing the decoder again \n";
    }
  }

  // if failing to decode simply return 0 (note, maybe
  // raise an exeption otherwise)
  torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);
  torch::List<torch::Tensor> result;
  // first decode the frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;
    // then initialize the output variables based on type
    size_t expectedWrittenBytes = 0;
    if (format.type == TYPE_VIDEO) {
          int outHeight = format.format.video.height;
          int outWidth = format.format.video.width;
          int numChannels = 3;
          outFrame = torch::ones({outHeight, outWidth, numChannels}, torch::kByte);
  }
  } else {
    LOG(ERROR) << "Decoder run into a last iteration or has failed";
    
  }
  result.push_back(outFrame);
  result.push_back(framePTS);
  return result;
}


torch::Tensor Video::NextDummyTensorOnly(std::string stream) {

  bool newInit = false;
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
      // cout << "Reinitializing the decoder again \n";
    }
  }

  // if failing to decode simply return 0 (note, maybe
  // raise an exeption otherwise)
  torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
  torch::Tensor outFrame = torch::zeros({0}, torch::kByte);
  // first decode the frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;
    // then initialize the output variables based on type
    size_t expectedWrittenBytes = 0;
    if (format.type == TYPE_VIDEO) {
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      outFrame = torch::ones({outHeight, outWidth, numChannels}, torch::kByte);
    }
  } else {
    LOG(ERROR) << "Decoder run into a last iteration or has failed";
  }
  return outFrame;
}

// this benchmark doesn't really make sense anymore
// torch::Tensor Video::NextDummyTensorOnlyNoAlloc(std::string stream) {

//   bool newInit = false;
//   if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
//       current_stream = _parseStream(stream);
//       newInit = true;
//   }

//   if ((seekTS != -1) && (doSeek == true)) {
//       newInit = true;
//       doSeek = false;
//   }

//   if (newInit){
//     succeeded = Video::_setCurrentStream();
//     if (succeeded) {
//       newInit = false;
//       // cout << "Reinitializing the decoder again \n";
//     }
//   }

//   // if failing to decode simply return 0 (note, maybe
//   // raise an exeption otherwise)
//   torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);
//   torch::Tensor outFrame = torch::zeros({0}, torch::kByte);
//   // first decode the frame
//   DecoderOutputMessage out;
//   int64_t res = decoder.decode(&out, decoderTimeoutMs);
//   if (res == 0) {
//     auto header = out.header;
//     const auto& format = header.format;
//     // then initialize the output variables based on type
//     size_t expectedWrittenBytes = 0;
//     if (format.type == TYPE_VIDEO) {
//       // fill the random but preallocated tensor
//       dummy = torch::ones({nc, oh, ow}, torch::kByte);
//       outFrame = dummy;
//     }
//   } else {
//     LOG(ERROR) << "Decoder run into a last iteration or has failed";
//   }
//   return outFrame;
// }

int64_t Video::nextDebugNoReturn(std::string stream) {

  bool newInit = false;
  if ((!stream.empty()) && (_parseStream(stream) != current_stream)) {
      current_stream = _parseStream(stream);
      newInit = true;
  }

  if ((seekTS != -1) && (doSeek == true)) {
      newInit = true;
      doSeek = false;
  }

  if (newInit){
    succeeded = Video::_setCurrentStream();
    if (succeeded) {
      newInit = false;
      // cout << "Reinitializing the decoder again \n";
    }
  }

  // first decode the frame
  DecoderOutputMessage out;
  int64_t res = decoder.decode(&out, decoderTimeoutMs);
  if (res == 0) {
    auto header = out.header;
    const auto& format = header.format;

    // then initialize the output variables based on type
    size_t expectedWrittenBytes = 0;
    torch::Tensor outFrame = torch::zeros({0}, torch::kByte);
    torch::Tensor framePTS = torch::zeros({1}, torch::kFloat);

    if (format.type == TYPE_VIDEO) {
      int outHeight = format.format.video.height;
      int outWidth = format.format.video.width;
      int numChannels = 3;
      // outFrame = torch::zeros({outHeight, outWidth, numChannels}, torch::kByte);
      // expectedWrittenBytes = outHeight * outWidth * numChannels;
      
      // auto numberWrittenBytes = fillVideoTensor(out, outFrame, framePTS);
      out.payload.reset();
      return int64_t(1);
    } else {
      return int64_t(0); 
    }
    
  } else {
    return int64_t(0);
  }
}


// returns number of written bytes
template <typename T>
size_t fillTensor(
    std::vector<DecoderOutputMessage>& msgs,
    torch::Tensor& frame,
    torch::Tensor& framePts,
    int64_t num,
    int64_t den) {
  if (msgs.empty()) {
    return 0;
  }
  T* frameData = frame.numel() > 0 ? frame.data_ptr<T>() : nullptr;
  int64_t* framePtsData = framePts.data_ptr<int64_t>();
  CHECK_EQ(framePts.size(0), msgs.size());
  size_t avgElementsInFrame = frame.numel() / msgs.size();

  size_t offset = 0;
  for (size_t i = 0; i < msgs.size(); ++i) {
    const auto& msg = msgs[i];
    // convert pts into original time_base
    AVRational avr = {(int)num, (int)den};
    framePtsData[i] = av_rescale_q(msg.header.pts, AV_TIME_BASE_Q, avr);
    VLOG(2) << "PTS type: " << sizeof(T) << ", us: " << msg.header.pts
            << ", original: " << framePtsData[i];

    if (frameData) {
      auto sizeInBytes = msg.payload->length();
      memcpy(frameData + offset, msg.payload->data(), sizeInBytes);
      if (sizeof(T) == sizeof(uint8_t)) {
        // Video - move by allocated frame size
        offset += avgElementsInFrame / sizeof(T);
      } else {
        // Audio - move by number of samples
        offset += sizeInBytes / sizeof(T);
      }
    }
  }
  return offset * sizeof(T);
}

size_t fillVideoTensorDBG(
    std::vector<DecoderOutputMessage>& msgs,
    torch::Tensor& videoFrame,
    torch::Tensor& videoFramePts,
    int64_t num,
    int64_t den) {
  return fillTensor<uint8_t>(msgs, videoFrame, videoFramePts, num, den);
}




int64_t Video::debugReadVideoNumFrames() {
  DecoderOutputMessage out;
  std::vector<DecoderOutputMessage> videoMessages;

  size_t audioFrames = 0, videoFrames = 0, totalBytes = 0;
  while (0 == decoder.decode(&out, 10000)) {
    if (out.header.format.type == TYPE_VIDEO) {
      ++videoFrames;
      videoMessages.push_back(std::move(out));

    } 
  LOG(INFO) << "Decoded audio frames: " << audioFrames
            << ", video frames: " << videoFrames
            << ", total bytes: " << totalBytes;
  
  }
  
  torch::Tensor videoFrame = torch::zeros({0}, torch::kByte);
  torch::Tensor videoFramePts = torch::zeros({0}, torch::kLong);
  int numVideoFrames = 0;
  if (!videoMessages.empty()) {
     auto header = videoMetadata;
     const auto& format = header.format.format.video;
      numVideoFrames = videoMessages.size();
      int outHeight = format.height;
      int outWidth = format.width;
      int numChannels = 3; // decoder guarantees the default AV_PIX_FMT_RGB24
      videoFrame = torch::zeros(
            {numVideoFrames, outHeight, outWidth, numChannels}, torch::kByte);
      videoFramePts = torch::zeros({numVideoFrames}, torch::kLong);
      auto numberWrittenBytes = fillVideoTensorDBG(
          videoMessages, videoFrame, videoFramePts, header.num, header.den);
      // LOG(ERROR) << "Writen bytes: " << numberWrittenBytes;
  }
  return int64_t(numVideoFrames);
}

torch::Tensor Video::debugReadVideoTensor() {
  DecoderOutputMessage out;
  std::vector<DecoderOutputMessage> videoMessages;

  size_t audioFrames = 0, videoFrames = 0, totalBytes = 0;
  while (0 == decoder.decode(&out, 10000)) {
    if (out.header.format.type == TYPE_VIDEO) {
      ++videoFrames;
      videoMessages.push_back(std::move(out));

    } 
  LOG(INFO) << "Decoded audio frames: " << audioFrames
            << ", video frames: " << videoFrames
            << ", total bytes: " << totalBytes;
  
  }
  
  torch::Tensor videoFrame = torch::zeros({0}, torch::kByte);
  torch::Tensor videoFramePts = torch::zeros({0}, torch::kLong);
  if (!videoMessages.empty()) {
     auto header = videoMetadata;
     const auto& format = header.format.format.video;
      int numVideoFrames = videoMessages.size();
      int outHeight = format.height;
      int outWidth = format.width;
      int numChannels = 3; // decoder guarantees the default AV_PIX_FMT_RGB24
      videoFrame = torch::zeros(
            {numVideoFrames, outHeight, outWidth, numChannels}, torch::kByte);
      videoFramePts = torch::zeros({numVideoFrames}, torch::kLong);
      auto numberWrittenBytes = fillVideoTensorDBG(
          videoMessages, videoFrame, videoFramePts, header.num, header.den);
      // LOG(ERROR) << "Writen bytes: " << numberWrittenBytes;
  }
  return videoFrame;
}

int64_t Video::tbTest() {
  torch::Tensor videoFrame = torch::ones(
            {73, 3, 224, 224}, torch::kByte);
  return videoFrame.numel();
}

torch::Tensor Video::tbTestTensor() {
  torch::Tensor videoFrame = torch::ones(
            {73, 3, 224, 224}, torch::kByte);
  return videoFrame;
}

Video::~Video() {
//   delete params; // does not have destructor
//   delete metadata; // struct does not have destructor
//   delete decoder; // should be fine
//   delete streamFPS; // should be fine
//   delete streamDuration; // should be fine
}
