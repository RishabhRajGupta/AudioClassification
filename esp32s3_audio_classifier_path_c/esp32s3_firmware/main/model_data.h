#pragma once
#include <stdint.h>
#include <stddef.h>

extern const char* kCategoryLabels[4];
extern const float kFeatureNormMean[80];
extern const float kFeatureNormStd[80];
extern const float kInt8WeightScale;
extern const float kClassifierBias[4];
extern const int8_t kInt8StudentWeights[80 * 4];
extern const unsigned char g_student_model_tflite[];
extern const unsigned int g_student_model_tflite_len;
