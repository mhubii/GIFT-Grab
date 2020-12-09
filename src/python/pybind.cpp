#include <videosourcefactory.h>
#include <videotargetfactory.h>
#include <except.h>
#include <videoframe.h>
#include <device.h>
#include <codec.h>
#include <iobservable.h>
#include <iobserver.h>
#ifdef USE_OPENCV
#include <opencv_video_source.h>
#include <opencv_video_target.h>
#endif
#ifdef USE_EPIPHANSDK
#include "epiphansdk_video_source.h"
#endif
#ifdef USE_LIBVLC
#include "vlc_video_source.h"
#endif
#ifdef USE_BLACKMAGICSDK
#include "blackmagicsdk_video_source.h"
#endif
#ifdef USE_FFMPEG
#include "ffmpeg_video_source.h"
#include "ffmpeg_video_target.h"
#endif
#include <pybind11/pybind11.h>
#ifdef USE_NUMPY
#include <pybind11/numpy.h>
#endif

namespace py = pybind11;

// 3 wrap videoframe into numpy
// 4 wrap virtual functions

// exception tranlation
// https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html#registering-custom-translators
// https://docs.python.org/3/c-api/exceptions.html#raising-exceptions

// numpy, difficult
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html#

// boost virtual functions
// https://www.boost.org/doc/libs/1_38_0/libs/python/doc/v2/wrapper.html

// pybind virtual functions
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html#combining-virtual-functions-and-inheritance

// global interpreter lock (gil)
// exchange gil.h with py::gil_scoped_release and py::gil_scoped_acquire
// https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil

class VideoFrameNumPyWrapper : public gg::VideoFrame {
protected:
    //!
    //! \brief Wrapped \c VideoFrame object
    //!
    gg::VideoFrame * _frame;

    //!
    //! \brief \c false indicates the wrapped
    //! \c VideoFrame object is managed externally,
    //! \c true indicates it was created and is
    //! managed by this class
    //! \sa _frame
    //!
    bool _manage_frame;

public:
    //!
    //! \brief Very thinly wrap passed \c frame
    //! \param frame it is the caller's responsibility
    //! to ensure the lifetime of the object pointed to
    //! by this pointer
    //!
    VideoFrameNumPyWrapper(gg::VideoFrame * frame)
        : _frame(frame)
        , _manage_frame(false) {
        _manage_data = false;
        sync_specs();
    }

    //!
    //! \brief Copy constructor needs to be defined
    //! here as well for compatibility with exposed
    //! interface
    //! \param rhs
    //!
    VideoFrameNumPyWrapper(const gg::VideoFrame & rhs)
        : _frame(new gg::VideoFrame(rhs))
        , _manage_frame(true) {
        _manage_data = false;
        sync_specs();
    }

    //!
    //! \brief This constructor needs to be defined
    //! here as well for compatibility with exposed
    //! interface
    //! \param colour
    //! \param cols
    //! \param rows
    //!
    VideoFrameNumPyWrapper(enum gg::ColourSpace colour,
                           size_t cols, size_t rows)
        : _frame(new gg::VideoFrame(colour, cols, rows))
        , _manage_frame(true) {
        _manage_data = false;
        sync_specs();
    }

    //!
    //! \brief This constructor needs to be defined
    //! here as well for compatibility with exposed
    //! interface
    //! \param colour
    //! \param cols
    //! \param rows
    //!
    VideoFrameNumPyWrapper(enum gg::ColourSpace colour,
                           size_t cols, size_t rows,
                           size_t stereo_count)
        : _frame(new gg::VideoFrame(colour, cols, rows,
                                    stereo_count))
        , _manage_frame(true) {
        _manage_data = false;
        sync_specs();
    }

    //!
    //! \brief This constructor needs to be defined
    //! here as well for compatibility with exposed
    //! interface
    //! \param colour
    //! \param manage_data
    //!
    VideoFrameNumPyWrapper(enum gg::ColourSpace colour,
                           bool manage_data)
        : _frame(new gg::VideoFrame(colour, manage_data))
        , _manage_frame(true) {
        _manage_data = false;
        sync_specs();
    }

    ~VideoFrameNumPyWrapper() {
        if (_manage_frame)
            delete _frame;
    }

#ifdef USE_NUMPY
    //!
    //! \brief Create a NumPy array referencing gg::VideoFrame::data()
    //! \param structured
    //! \param stereo_index
    //! \return a flat NumPy array if not \c structured; otherwise one
    //! that conforms to the shape SciPy routines expect: (height,
    //! width, channels), e.g. (9, 16, 4) for BGRA data of a 16 x 9
    //! image
    //! \throw gg::BasicException if wrapped gg::VideoFrame has colour
    //! other than BGRA (currently only BGRA data supported for
    //! structured ndarray exposure)
    //! \throw std::out_of_range if stereo index invalid
    //!
    py::array_t<uint8_t> stereo_data_as_ndarray(bool structured, size_t stereo_index) const {
        py::array::ShapeContainer shape;
        py::array::StridesContainer strides;
        if (structured) {
            switch(colour())
            {
            case gg::BGRA:
                shape = py::array::ShapeContainer({_frame->rows(), _frame->cols(), 4}); // NOTE shrinking conversion here
                strides = py::array::ShapeContainer({_frame->cols() * 4 * sizeof(uint8_t),
                                     4 * sizeof(uint8_t),
                                     sizeof(uint8_t)});
                break;
            // TODO: see issue #155
            case gg::I420:
            case gg::UYVY:
            default:
                throw gg::BasicException("Structured NumPy arrays"
                                         " supported only for BGRA");
                break;
            }
        }
        else {
            shape = py::array::ShapeContainer({_frame->data_length(stereo_index)});
            strides = py::array::ShapeContainer({sizeof(uint8_t)});
        }

        return py::array_t<uint8_t>(
            shape,
            strides,
            _frame->data(stereo_index),
            py::none() // https://github.com/pybind/pybind11/issues/2271
        );
    }
#endif

protected:
    void sync_specs() {
        _colour = _frame->colour();
        _cols = _frame->cols();
        _rows = _frame->rows();
        _data = _frame->data();
        _data_length = _frame->data_length()*_frame->stereo_count(); // bug where _data_length was only 1/stereo_count of desired size
	    _stereo_count = _frame->stereo_count();
    }
};

// previously IObservableWrapper, trampolines, not meant for construction
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html
class PyIObservable : public gg::IObservable {
    public:
        // Trampolines
        void attach(gg::IObserver& observer) override {
            PYBIND11_OVERRIDE(
                void,
                gg::IObservable,
                attach,
                observer
            );
        }

        void detach(gg::IObserver& observer) override {
            PYBIND11_OVERRIDE(
                void,
                gg::IObservable,
                detach,
                observer
            );
        }
};

// previously IObserverWrapper, trampolines, not meant for construction
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html
class PyIObserver : public gg::IObserver {
    public:
        // Trampolines
        void update(gg::VideoFrame& frame) override {
            py::gil_scoped_acquire acquire;
            VideoFrameNumPyWrapper wrapped_frame(&frame);

            PYBIND11_OVERRIDE_PURE(
                void,
                gg::IObserver,
                update,
                wrapped_frame
            );
        }
};

// previously IObservableObserverWrapper
// https://pybind11.readthedocs.io/en/stable/advanced/classes.html
class PyIObservableObserver : public PyIObservable, public PyIObserver {
    public:
        PyIObservableObserver() = default;
        ~PyIObservableObserver() = default;

        // Trampolines
        void attach(gg::IObserver& observer) override {
            PYBIND11_OVERRIDE(
                void,
                PyIObservable,
                attach,
                observer
            );
        }

        void detach(gg::IObserver& observer) override {
            PYBIND11_OVERRIDE(
                void,
                PyIObservable,
                detach,
                observer
            );
        }

        void update(gg::VideoFrame& frame) override {
            py::gil_scoped_acquire acquire;
            VideoFrameNumPyWrapper wrapped_frame(&frame);

            PYBIND11_OVERRIDE_PURE(
                void,
                PyIObserver,
                update,
                wrapped_frame
            );
            notify(frame);
        }
};

PYBIND11_MODULE(pygiftgrab, m) {
    // exceptions // NOTE helper functions default to what(), default already translated translated by pybind
    py::register_exception<gg::BasicException>(m, "BasicException", PyExc_RuntimeError);
    py::register_exception<gg::VideoSourceError>(m, "VideoSourceError", PyExc_RuntimeError);
    py::register_exception<gg::DeviceAlreadyConnected>(m, "DeviceAlreadyConnected", PyExc_IOError);
    py::register_exception<gg::DeviceNotFound>(m, "DeviceNotFound", PyExc_IOError);
    py::register_exception<gg::DeviceOffline>(m, "DeviceOffline", PyExc_IOError);
    py::register_exception<gg::NetworkSourceUnavailable>(m, "NetworkSourceUnavailable", PyExc_IOError);
    py::register_exception<gg::VideoTargetError>(m, "VideoTargetError", PyExc_RuntimeError);
    py::register_exception<gg::ObserverError>(m, "ObserverError", PyExc_RuntimeError);

    // enums
    py::enum_<gg::ColourSpace> (m, "ColourSpace")
        .value("BGRA", gg::ColourSpace::BGRA)
        .value("I420", gg::ColourSpace::I420)
        .value("UYVY", gg::ColourSpace::UYVY)
        .export_values();

    py::enum_<gg::Device> (m, "Device")
        .value("DVI2PCIeDuo_SDI", gg::Device::DVI2PCIeDuo_SDI)
        .value("DVI2PCIeDuo_DVI", gg::Device::DVI2PCIeDuo_DVI)
        .value("DeckLinkSDI4K", gg::Device::DeckLinkSDI4K)
        .value("DeckLink4KExtreme12G", gg::Device::DeckLink4KExtreme12G)
        .export_values();

    py::enum_<gg::Codec> (m, "Codec")
        .value("HEVC", gg::Codec::HEVC)
        .value("Xvid", gg::Codec::Xvid)
        .value("VP9", gg::Codec::VP9)
        .export_values();

    // classes
    py::class_<VideoFrameNumPyWrapper>(m, "VideoFrame")
        .def(py::init<gg::ColourSpace, bool>())
        .def(py::init<gg::ColourSpace, const size_t, const size_t>())
        .def(py::init<gg::ColourSpace, const size_t, const size_t, const size_t>()) // NOTE constructors
        .def("colour", &VideoFrameNumPyWrapper::colour)
        .def("rows", &VideoFrameNumPyWrapper::rows)
        .def("cols", &VideoFrameNumPyWrapper::cols)
        .def("data_length", &VideoFrameNumPyWrapper::data_length, py::arg("stereo_index") = 0) // NOTE default value
        .def("stereo_count", &VideoFrameNumPyWrapper::stereo_count)
        .def_static("required_data_length", &VideoFrameNumPyWrapper::required_data_length)
        .def_static("required_pixel_length", &VideoFrameNumPyWrapper::required_pixel_length)
#ifdef USE_NUMPY
        .def("data", &VideoFrameNumPyWrapper::stereo_data_as_ndarray, py::arg("structured") = false, py::arg("stereo_index") = 0, py::return_value_policy::move)
#endif
        ;

    py::class_<gg::IObservable, PyIObservable>(m, "IObservable")
        .def("attach", &gg::IObservable::attach)
        .def("detach", &gg::IObservable::detach);

    py::class_<gg::IObserver, PyIObserver>(m, "IObserver")
        .def("update", &gg::IObserver::update);

    py::class_<PyIObservableObserver, PyIObservable, PyIObserver>(m, "IObservableObserver", py::multiple_inheritance())
        .def(py::init<>())
        .def("attach", &gg::IObservable::attach)
        .def("detach", &gg::IObservable::detach)
        .def("update", &gg::IObserver::update);

    // left out wrapper for gg:IVideoTarget here, what is the use case of it?

    py::class_<gg::VideoSourceFactory>(m, "VideoSourceFactory")
        .def("get_device", &gg::VideoSourceFactory::get_device, py::return_value_policy::reference)
        .def("free_device", &gg::VideoSourceFactory::free_device)
        .def("connect_network_source", &gg::VideoSourceFactory::connect_network_source, py::return_value_policy::take_ownership)
        .def_static("get_instance", &gg::VideoSourceFactory::get_instance, py::return_value_policy::reference)
        .def("create_file_reader", &gg::VideoSourceFactory::create_file_reader, py::return_value_policy::take_ownership);

    py::class_<gg::VideoTargetFactory>(m, "VideoTargetFactory")
        .def("create_file_writer", &gg::VideoTargetFactory::create_file_writer, py::return_value_policy::take_ownership)
        .def_static("get_instance", &gg::VideoTargetFactory::get_instance, py::return_value_policy::reference);

// NOTE: these should be extending the core rather than being appended here
#ifdef USE_OPENCV
    // NOTE: why not using appropriate namespace?
    py::class_<VideoSourceOpenCV, IVideoSource, PyIObservable>(m, "VideoSourceOpenCV")
        .def(py::init<>())
        .def(py::init<char*>())
        .def("get_frame", &VideoSourceOpenCV::get_frame)
        .def("get_frame_dimensions", &VideoSourceOpenCV::get_frame_dimensions)
        .def("get_frame_rate", &VideoSourceOpenCV::get_frame_rate)
        .def("set_sub_frame", &VideoSourceOpenCV::set_sub_frame)
        .def("get_full_frame", &VideoSourceOpenCV::get_full_frame)
        .def("attach", &gg::IObservable::attach)
        .def("detach", &gg::IObservable::detach);

    py::class_<gg::VideoTargetOpenCV, gg::VideoTarget>(m, VideoTargetOpenCV)
        .def(py::init<std::string, std::string, float>())
        .def("append", &gg::VideoTargetOpenCV::append);
#endif

#ifdef USE_EPIPHANSDK
    py::class_<gg::VideoSourceEpiphanSDK, IVideoSource, PyIObservable>(m, "VideoSourceEpiphaSDK")
        .def(py::init<const std::string, const V2U_INT32>())
        .def("get_frame", &gg::VideoSourceEpiphanSDK::get_frame)
        .def("get_frame_dimensions", &gg::VideoSourceEpiphanSDK::get_frame_dimensions)
        .def("get_frame_rate", &gg::VideoSourceEpiphanSDK::get_frame_rate)
        .def("set_sub_frame", &gg::VideoSourceEpiphanSDK::set_sub_frame)
        .def("get_full_frame", &gg::VideoSourceEpiphanSDK::get_full_frame)
        .def("attach", &gg::IObservable::attach)
        .def("detach", &gg::IObservable::detach);
#endif

#ifdef USE_LIBVLC
    py::class_<gg::VIdeoSOurceVLC, IVideoSource, PyIObservable>(m, "VideoSourceVLC")
        .def(py::init<const std::string>())
        .def("get_frame", &gg::VideoSourceVLC::get_frame)
        .def("get_frame_dimensions", &gg::VideoSourceVLC::get_frame_dimensions)
        .def("get_frame_rate", &gg::VideoSourceVLC::get_frame_rate)
        .def("set_sub_frame", &gg::VideoSourceVLC::set_sub_frame)
        .def("get_full_frame", &gg::VideoSourceVLC::get_full_frame)
        .def("attach", &gg::IObservable::attach)
        .def("detach", &gg::IObservable::detach);
#endif

#ifdef USE_BLACKMAGICSDK
    py::class_<gg::VideoSourceBlackmagicSDK, IVideoSource, PyIObservable>(m, "VideoSOurceBlackmagicSDK")
        .def(py::init<size_t, gg::CoulourSpace>())
        .def("get_frame", &gg::VideoSourceBlackmagicSDK::get_frame)
        .def("get_frame_dimensions", &gg::VideoSourceBlackmagicSDK::get_frame_dimensions)
        .def("get_frame_rate", &gg::VideoSourceBlackmagicSDK::get_frame_rate)
        .def("set_sub_frame", &gg::VideoSourceBlackmagicSDK::set_sub_frame)
        .def("get_full_frame", &gg::VideoSourceBlackmagicSDK::get_full_frame)
        .def("attach", &gg::IObservable::attach)
        .def("detach", &gg::IObservable::detach);
#endif

#ifdef USE_FFMPEG
    py::class_<gg::VideoSOurceFFmpgeg, IVideoSource, PyIObservable>(m, "VideoSourceFFmpgeg")
        .def(py::init<std::string, gg::ColourSpace, bool>())
        .def("get_frame", &gg::VideoSourceFFmpeg::get_frame)
        .def("get_frame_dimensions", &gg::VideoSourceFFmpeg::get_frame_dimensions)
        .def("get_frame_rate", &gg::VideoSourceFFmpeg::get_frame_rate)
        .def("set_sub_frame", &gg::VideoSourceFFmpeg::set_sub_frame)
        .def("get_full_frame", &gg::VideoSourceFFmpeg::get_full_frame)
        .def("attach", &gg::IObservable::attach)
        .def("detach", &gg::IObservable::detach);
    
    py::class_<gg::VideoTargetFFmpeg, gg::IVideoTarget>(m, "VideoTargetFFmpeg")
        .def(py::init<std::string, std::string, float>())
        .def("append", &gg::VideoTargetFFmpeg::append);
#endif
}
