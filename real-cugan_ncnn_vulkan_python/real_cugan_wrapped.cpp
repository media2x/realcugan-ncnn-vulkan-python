#include "real_cugan_wrapped.h"

int RealCUGANWrapped::process(const Image &inimage, Image &outimage) const
{
    int c = inimage.elempack;
    ncnn::Mat inimagemat =
        ncnn::Mat(inimage.w, inimage.h, (void *)inimage.data, (size_t)c, c);
    ncnn::Mat outimagemat =
        ncnn::Mat(outimage.w, outimage.h, (void *)outimage.data, (size_t)c, c);
    return RealCUGAN::process(inimagemat, outimagemat);
}

int RealCUGANWrapped::process_cpu(const Image &inimage, Image &outimage) const
{
    int c = inimage.elempack;
    ncnn::Mat inimagemat =
        ncnn::Mat(inimage.w, inimage.h, (void *)inimage.data, (size_t)c, c);
    ncnn::Mat outimagemat =
        ncnn::Mat(outimage.w, outimage.h, (void *)outimage.data, (size_t)c, c);
    return RealCUGAN::process_cpu(inimagemat, outimagemat);
}

RealCUGANWrapped::RealCUGANWrapped(int gpuid, bool tta_mode, int num_threads)
    : RealCUGAN(gpuid, tta_mode, num_threads)
{
    this->gpuid = gpuid;
}

uint32_t RealCUGANWrapped::get_heap_budget()
{
    return ncnn::get_gpu_device(this->gpuid)->get_heap_budget();
}

int RealCUGANWrapped::load(const StringType &parampath,
                           const StringType &modelpath)
{
#if _WIN32
    return RealCUGAN::load(*parampath.wstr, *modelpath.wstr);
#else
    return RealCUGAN::load(*parampath.str, *modelpath.str);
#endif
}

int get_gpu_count() { return ncnn::get_gpu_count(); }

void destroy_gpu_instance() { ncnn::destroy_gpu_instance(); }
