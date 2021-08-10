//
// Created by st003780 on 2021/8/3.
//
// Resnet ncnn forward , the softmax is in the network

#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include "net.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


int classifier(const ncnn::Net &model, const cv::Mat &bgr, std::vector<float> &cls_scores)
{

    // step 2, process img
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
    const float norm_vals[3] = {1 / 255., 1 / 255., 1 / 255.};
    in.substract_mean_normalize(0, norm_vals);

    // step 3, create extractor
    ncnn::Extractor ex = model.create_extractor();

    // step 4, forward
    ex.input("input", in);
    ncnn::Mat out;
    ex.extract("probs", out);

    // step 5, postprocess
    out = out.reshape(out.w * out.h * out.c);
    cls_scores.resize(out.w);
    for (int i = 0; i < out.w; i++)
    {
        cls_scores[i] = out[i];
    }
    return 1;
}

int print_topk(const std::vector<float> &cls_scores, int topk)
{
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;  //std::pair 结构体模板，一个单元存放两个相异的对象
    vec.resize(size);
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<float, int>>());

    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int class_idx = vec[i].second;
        fprintf(stderr, "class_idx: %d,score: %.3f\n", class_idx, score);

    }
}


int main(int argc, char **argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "Error param %s", argv[0]);
        return -1;
    }


    const char *image_path = argv[1];
    const char *model_param_path = argv[2];
    const char *model_bin_path = argv[3];

    cv::Mat img = cv::imread(image_path, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s error \n", image_path);
        return -1;
    }

    // step 1, init net
    ncnn::Net model;
    model.opt.use_vulkan_compute = false;
    model.load_param(model_param_path);
    model.load_model(model_bin_path);

    struct timeval t1, t2;
    double using_time;

    std::vector<float> cls_scores;
    gettimeofday(&t1, NULL);
    int tag = classifier(model, img, cls_scores);
    gettimeofday(&t2, NULL);
    using_time = 1000 * (t2.tv_sec - t1.tv_sec) + (double) (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout << "using time(ms) = " << using_time << std::endl;


    if (tag)
    {
        std::cout << "Inference success" << std::endl;
    }
    print_topk(cls_scores, 3);


    return 0;

}