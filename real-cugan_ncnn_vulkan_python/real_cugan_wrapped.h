#ifndef REALCUGAN_NCNN_VULKAN_REALCUGAN_WRAPPED_H
#define REALCUGAN_NCNN_VULKAN_REALCUGAN_WRAPPED_H
#include "real_cugan.h"

// wrapper class of ncnn::Mat
typedef struct Image {
    unsigned char *data;
    int w;
    int h;
    int elempack;
    Image(unsigned char *d, int w, int h, int channels)
    {
        this->data = d;
        this->w = w;
        this->h = h;
        this->elempack = channels;
    }

} Image;

union StringType {
    std::string *str;
    std::wstring *wstr;
};

class RealCUGANWrapped : public RealCUGAN
{
  public:
    RealCUGANWrapped(int gpuid, bool tta_mode = false, int num_threads = 1);
    int load(const StringType &parampath, const StringType &modelpath);
    int process(const Image &inimage, Image &outimage) const;
    int process_cpu(const Image &inimage, Image &outimage) const;
    uint32_t get_heap_budget();

  private:
    int gpuid;
};

int get_gpu_count();
void destroy_gpu_instance();
#endif // REALCUGAN_NCNN_VULKAN_REALCUGAN_WRAPPED_H
