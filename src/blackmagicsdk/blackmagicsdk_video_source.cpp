#include "blackmagicsdk_video_source.h"

namespace gg
{

VideoSourceBlackmagicSDK::VideoSourceBlackmagicSDK()
    : IVideoSource()
    , _frame_rate(0.0)
    , _video_buffer(nullptr)
    , _video_buffer_length(0)
    , _cols(0)
    , _rows(0)
    // This constructor should never be called,
    // therefore setting I420 arbitrarily for the
    // buffer video frame here.
    , _buffer_video_frame(VideoFrame(I420))
    , _running(false)
{
    // nop
}


VideoSourceBlackmagicSDK::VideoSourceBlackmagicSDK(size_t deck_link_index,
                                                   ColourSpace colour)
    : IVideoSource(colour)
    , _frame_rate(0.0)
    , _video_buffer(nullptr)
    , _video_buffer_length(0)
    , _cols(0)
    , _rows(0)
    , _buffer_video_frame(VideoFrame(colour, false)) // TODO manage data?
    , _deck_link(nullptr)
    , _deck_link_input(nullptr)
    , _running(false)
{
    switch(_colour)
    {
    case BGRA:
        break;
    case I420:
    default:
        release_deck_link();
        throw VideoSourceError(
            "BlackmagicSDK video source supports only BGRA"
        );
    }

    // Get an iterator through the available DeckLink ports
    IDeckLinkIterator * deck_link_iterator = CreateDeckLinkIteratorInstance();
    if (deck_link_iterator == nullptr)
    {
        release_deck_link();
        throw VideoSourceError("DeckLink drivers do not appear to be installed");
    }

    HRESULT res;

    // Get the desired DeckLink index (port)
    int idx = deck_link_index;
    while ((res = deck_link_iterator->Next(&_deck_link)) == S_OK)
    {
        if (idx == 0)
            break;
        --idx;

        _deck_link->Release();
    }
    if (deck_link_iterator != nullptr)
        deck_link_iterator->Release();
    // No glory: release everything and throw exception
    if (res != S_OK or _deck_link == nullptr)
    {
        release_deck_link();
        throw VideoSourceError(
            std::string("Could not get DeckLink device ").append(std::to_string(deck_link_index))
        );
    }

    // Get the input interface of connected DeckLink port
    res = _deck_link->QueryInterface(IID_IDeckLinkInput, reinterpret_cast<void **>(&_deck_link_input));
    // No glory: release everything and throw exception
    if (res != S_OK)
    {
        release_deck_link();
        throw VideoSourceError("Could not connect to any Blackmagic DeckLink device");
    }

    // Set this object (IDeckLinkInputCallback instance) as callback
    res = _deck_link_input->SetCallback(this);
    // No glory: release everything and throw exception
    if (res != S_OK)
    {
        release_deck_link();
        throw VideoSourceError("Could not set the callback of Blackmagic DeckLink device");
    }

    // Try to set the first available of the following modes (in descending frame-rate order)
    std::vector<BMDDisplayMode> display_modes =
        { bmdModeHD1080p6000, bmdModeHD1080p5994, bmdModeHD1080p50,
          bmdModeHD1080p30, bmdModeHD1080p2997, bmdModeHD1080p25,
          bmdModeHD1080p24, bmdModeHD1080p2398 };
    // and BGRA as pixel format
    BMDPixelFormat pixel_format = bmdFormat8BitBGRA;
    // and these video flags
    BMDVideoInputFlags video_input_flags = bmdVideoInputFlagDefault | bmdVideoInputEnableFormatDetection;
    // These two are output variables
    BMDDisplayModeSupport display_mode_support;
    IDeckLinkDisplayMode * deck_link_display_mode = nullptr;
    // Flag for indicating the result of the video input enable attempt
    bool enabled_video_input = false;
    // And an error string to be set according to any failing intermediate step
    std::string error_msg = "";

    // Now loop through the set of modes
    for (BMDDisplayMode display_mode : display_modes)
    {
        // Check whether the mode is supported
        res = _deck_link_input->DoesSupportVideoMode(
            display_mode, pixel_format, video_input_flags,
            &display_mode_support, &deck_link_display_mode
        );
        // No glory (could not even check mode support)
        if (res != S_OK or deck_link_display_mode == nullptr)
        {
            error_msg = "Could not check video mode support of Blackmagic DeckLink device";
            break;
        }

        // If mode supported, set it and exit loop
        if (display_mode_support == bmdDisplayModeSupported)
        {
            // Get frame rate of DeckLink device
            BMDTimeValue frame_rate_duration, frame_rate_scale;
            res = deck_link_display_mode->GetFrameRate(&frame_rate_duration, &frame_rate_scale);
            // No glory
            if (res != S_OK)
            {
                error_msg = "Could not infer frame rate of Blackmagic DeckLink device";
                break;
            }
            _frame_rate = (double) frame_rate_scale / (double) frame_rate_duration;

            // Enable video input
            res = _deck_link_input->EnableVideoInput(display_mode,
                                                     pixel_format,
                                                     video_input_flags);
            // No glory
            if (res != S_OK)
            {
                error_msg = "Could not enable video input of Blackmagic DeckLink device";
                break;
            }

            enabled_video_input = true;
        }

        // Release the DeckLink display mode object at each iteration
        if (deck_link_display_mode != nullptr)
        {
            deck_link_display_mode->Release();
            deck_link_display_mode = nullptr;
        }

        if (enabled_video_input)
            break;
    }

    // Release the DeckLink display mode object in case loop pre-maturely broken
    if (deck_link_display_mode != nullptr)
        deck_link_display_mode->Release();

    // No glory (loop exited without success): release everything and throw exception
    if (not enabled_video_input)
    {
        release_deck_link();
        // If all modes checked, and none is supported!
        if (error_msg.empty())
            error_msg = "Could not enable video input of Blackmagic DeckLink device";
        // Else: an intermediate step went wrong, so put that into the exception
        throw VideoSourceError(error_msg);
    }

    // Disable audio input
    res = _deck_link_input->DisableAudioInput();
    // No glory: release everything and throw exception
    if (res != S_OK)
    {
        release_deck_link();
        throw VideoSourceError("Could not disable audio input of Blackmagic DeckLink device");
    }

    // Start streaming
    _running = true;
    res = _deck_link_input->StartStreams();
    // No glory: release everything and throw exception
    if (res != S_OK)
    {
        _running = false;
        release_deck_link();
        throw VideoSourceError("Could not start streaming from the Blackmagic DeckLink device");
    }
}


VideoSourceBlackmagicSDK::~VideoSourceBlackmagicSDK()
{
    // Make sure streamer thread not trying to access buffer
    std::lock_guard<std::mutex> data_lock_guard(_data_lock);
    _running = false;

    // Stop streaming and disable enabled inputs
    _deck_link_input->StopStreams();

    // Release DeckLink members
    release_deck_link();
}


bool VideoSourceBlackmagicSDK::get_frame_dimensions(int & width, int & height)
{
    // TODO
    if (_cols <= 0 or _rows <= 0)
        return false;

    width = _cols;
    height = _rows;
    return true;
}


bool VideoSourceBlackmagicSDK::get_frame(VideoFrame & frame)
{
    // TODO
    return false;
}


double VideoSourceBlackmagicSDK::get_frame_rate()
{
    return _frame_rate;
}


void VideoSourceBlackmagicSDK::set_sub_frame(int x, int y, int width, int height)
{
    // TODO
}


void VideoSourceBlackmagicSDK::get_full_frame()
{
    // TODO
}


HRESULT STDMETHODCALLTYPE VideoSourceBlackmagicSDK::QueryInterface(REFIID iid, LPVOID * ppv)
{
    // TODO
}


ULONG STDMETHODCALLTYPE VideoSourceBlackmagicSDK::AddRef(void)
{
    // TODO
}


ULONG STDMETHODCALLTYPE VideoSourceBlackmagicSDK::Release(void)
{
    // TODO
}


HRESULT STDMETHODCALLTYPE VideoSourceBlackmagicSDK::VideoInputFormatChanged(
    BMDVideoInputFormatChangedEvents events,
    IDeckLinkDisplayMode * display_mode,
    BMDDetectedVideoInputFormatFlags format_flags
)
{
    // TODO
}


HRESULT STDMETHODCALLTYPE VideoSourceBlackmagicSDK::VideoInputFrameArrived(
    IDeckLinkVideoInputFrame * video_frame,
    IDeckLinkAudioInputPacket * audio_packet
)
{
    // TODO
}


void VideoSourceBlackmagicSDK::release_deck_link() noexcept
{
    this->Release();

    if (_deck_link_input != nullptr)
    {
        _deck_link_input->Release();
        _deck_link_input = nullptr;
    }

    if (_deck_link != nullptr)
        _deck_link->Release();
}

}
