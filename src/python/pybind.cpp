// #include <videosourcefactory.h>
// #include <videotargetfactory.h>
// #include <except.h>
// #include <iobservable.h>
#include <except.h>
#include <videoframe.h>
#include <device.h>
#include <codec.h>
// #ifdef USE_OPENCV
// #include <opencv_video_source.h>
// #include <opencv_video_target.h>
// #endif
// #ifdef USE_EPIPHANSDK
// #include "epiphansdk_video_source.h"
// #endif
// #ifdef USE_LIBVLC
// #include "vlc_video_source.h"
// #endif
// #ifdef USE_BLACKMAGICSDK
// #include "blackmagicsdk_video_source.h"
// #endif
// #ifdef USE_FFMPEG
// #include "ffmpeg_video_source.h"
// #include "ffmpeg_video_target.h"
// #endif
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

class VideoFrameNumPyWrapper : public gg::VideoFrame
{
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
        , _manage_frame(false)
    {
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
        , _manage_frame(true)
    {
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
        , _manage_frame(true)
    {
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
        , _manage_frame(true)
    {
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
        , _manage_frame(true)
    {
        _manage_data = false;
        sync_specs();
    }

    ~VideoFrameNumPyWrapper()
    {
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
    py::array_t<uint8_t> stereo_data_as_ndarray(
        bool structured, size_t stereo_index) const
    {
        py::array::ShapeContainer shape;
        py::array::StridesContainer strides;
        if (structured)
        {
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
        else
        {
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
    void sync_specs()
    {
        _colour = _frame->colour();
        _cols = _frame->cols();
        _rows = _frame->rows();
        _data = _frame->data();
        _data_length = _frame->data_length()*_frame->stereo_count(); // bug where _data_length was only 1/stereo_count of desired size
	    _stereo_count = _frame->stereo_count();
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
        .def("data", &VideoFrameNumPyWrapper::stereo_data_as_ndarray, py::arg("structured") = true, py::arg("stereo_index") = 0, py::return_value_policy::move)
#endif
        ;

    // py::class_<>
    
}
