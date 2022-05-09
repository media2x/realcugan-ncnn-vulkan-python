%module real_cugan_ncnn_vulkan_wrapper

%include "cpointer.i"
%include "carrays.i"
%include "std_string.i"
%include "std_wstring.i"
%include "stdint.i"
%include "pybuffer.i"

%pybuffer_mutable_string(unsigned char *d);
%pointer_functions(std::string, str_p);
%pointer_functions(std::wstring, wstr_p);

%{
    #include "real_cugan.h"
    #include "real_cugan_wrapped.h"
%}

class RealCUGAN
{
    public:
        RealCUGAN(int gpuid, bool tta_mode = false, int num_threads = 1);
        ~RealCUGAN();

    public:
        // Real-CUGAN parameters
        int noise;
        int scale;
        int tilesize;
        int prepadding;
        int syncgap;
};

%include "real_cugan_wrapped.h"
