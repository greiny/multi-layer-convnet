#pragma once
#include "general_settings.h"
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

void save2txt(const Mat&, string, string);
void save2txt3ch(const Mat&, string, string);
void saveConvKernel(const vector<Cvl>&, string);
void saveConvKernelGradient(const vector<Cvl>&, string);
void saveConvImage(const int nsample,const int nkernel,const Mat conv, string);
