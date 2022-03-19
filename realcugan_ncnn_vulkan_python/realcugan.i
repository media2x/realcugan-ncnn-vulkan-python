%module realcugan_ncnn_vulkan_wrapper

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
    #include "realcugan.h"
    #include "realcugan_wrapped.h"
%}

class RealCUGAN
{
    public:
        RealCUGAN(int gpuid, bool tta_mode = false, int num_threads = 1);
        ~RealCUGAN();

    public:
        // realcugan parameters
        int noise;
        int scale;
        int tilesize;
        int prepadding;
        int syncgap;
};

%include "realcugan_wrapped.h"
