//
// Created by st003780 on 2021/8/16.
//
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"

using namespace std;
using namespace cv;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id


using namespace nvinfer1;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 3;

const char *INPUT_BLOB_NAME = "input";
const char *OUTPUT_BLOB_NAME = "identity_probs";


static Logger gLogger;


float *blobFromImage(cv::Mat &img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    float *blob = new float[2 * img.total() * 3];

    int channels = 3;
    int img_h = 224;
    int img_w = 224;
    std::vector<float> mean = {0.00, 0.0, 0.0};
    std::vector<float> std = {1.0, 1.0, 1.0};
//    std::vector<float> mean = {0.485, 0.456, 0.406};
//    std::vector<float> std = {0.229, 0.224, 0.225};
    for (int idx = 0; idx < 2; idx++)
    {
        for (size_t c = 0; c < channels; c++)
        {
            for (size_t h = 0; h < img_h; h++)
            {
                for (size_t w = 0; w < img_w; w++)
                {
                    blob[idx * channels * img_w * img_h + c * img_w * img_h + h * img_w + w] =
                            (((float) img.at<cv::Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
                }
            }
        }
    }
    return blob;
}


void doInferencev2(IExecutionContext &context, float *input, float *output, int batch_size)
{
    const ICudaEngine &engine = context.getEngine();

    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);

    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);


    CHECK(cudaMalloc(&buffers[inputIndex], batch_size * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batch_size * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batch_size, buffers, stream, nullptr);
//    context.enqueueV2(buffers, stream, nullptr);

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batch_size * OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost,
                          stream));
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));

}


void doInference(IExecutionContext &context, float *input, float *output, int batch_size)
{
    const ICudaEngine &engine = context.getEngine();

    assert(engine.getNbBindings() == 2);
    void *buffers[2];

    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);

    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

    CHECK(cudaMalloc(&buffers[inputIndex], batch_size * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batch_size * OUTPUT_SIZE * sizeof(float)));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batch_size * 3 * INPUT_H * INPUT_W * sizeof(float),
                          cudaMemcpyHostToDevice, stream));
    context.enqueue(batch_size, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batch_size * OUTPUT_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost,
                          stream));
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


int main(int argc, char **argv)
{
    // set guu id
    cudaSetDevice(DEVICE);

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 4 && std::string(argv[2]) == "-i")
    {
        const std::string engine_file_path{argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else
    {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./resnet ../model_trt.engine -i ../../../assets/dog.jpg " << std::endl;
        return -1;
    }

    const std::string input_image_path{argv[3]};

    IRuntime *runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    static float prob[2 * OUTPUT_SIZE];

    cv::Mat img = cv::imread(input_image_path);
    int img_w = img.cols;
    int img_h = img.rows;

    cv::Mat resize_img;
    cv::resize(img, resize_img, cv::Size(224, 224), cv::INTER_LINEAR);


    float *blob;
    blob = blobFromImage(resize_img);

//    for (int j = 0; j < 224 * 224 * 3; j++)
//    {
//        std::cout << blob[j] << std::endl;
//    }

//    std::cout << sizeof(&blob) / sizeof(blob[0]) << "sss" << std::endl;

    for (size_t i = 0; i < 10; i++)
    {
        auto start = std::chrono::system_clock::now();
        doInference(*context, blob, prob, 2);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::cout << "\nOutput:\n\n";
        for (unsigned int i = 0; i < 6; i++)
        {
            std::cout << prob[i] << ", ";
        }

    }


    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}