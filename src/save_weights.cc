#include "save_weights.h"

#define AT3D at<cv::Vec3d>
using namespace cv;
using namespace std;

void 
save2txt(const Mat &data, string path, string str){
    string tmp = path + str;
    FILE *pOut = fopen(tmp.c_str(), "w");
    for(int i = 0; i < data.rows; i++){
        for(int j = 0; j < data.cols; j++){
            fprintf(pOut, "%lf", data.ATD(i, j));
            if(j == data.cols - 1){
                fprintf(pOut, ",\n");
            } 
            else{
                fprintf(pOut, ",");
            } 
        }
    }
    fclose(pOut);
}

void
save2txt3ch(const Mat &data, string path, string str){
    string str_r = path + str + "ch_0";
    string str_g = path + str + "ch_1";
    string str_b = path + str + "ch_2";
    str_r += ".csv";
    str_g += ".csv";
    str_b += ".csv";
    FILE *pOut_r = fopen(str_r.c_str(), "w");
    FILE *pOut_g = fopen(str_g.c_str(), "w");
    FILE *pOut_b = fopen(str_b.c_str(), "w");
    for(int i=0; i<data.rows; i++){
        for(int j=0; j<data.cols; j++){
            Vec3d tmp = data.AT3D(i, j);
            fprintf(pOut_r, "%lf", tmp[0]);
            fprintf(pOut_g, "%lf", tmp[1]);
            fprintf(pOut_b, "%lf", tmp[2]);
            if(j == data.cols - 1){
                fprintf(pOut_r, "\n");
                fprintf(pOut_g, "\n");
                fprintf(pOut_b, "\n");
            } 
            else{
                fprintf(pOut_r, ",");
                fprintf(pOut_g, ",");
                fprintf(pOut_b, ",");
            } 
        }
    }
    fclose(pOut_r);
    fclose(pOut_g);
    fclose(pOut_b);
}

void
saveConvKernel(const vector<Cvl> &CLayers, string path){
    string tmp = path;
    mkdir(tmp.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    int layers = CLayers.size();
    for(int i = 0; i < layers; i++){
        string str = tmp + "/layer_" + std::to_string(i);
        mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        int kernels = convConfig[i].KernelAmount;
        for(int j = 0; j < kernels; j++){
            string str2 = str + "/kernel_" + std::to_string(j);
            mkdir(str2.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            str2 += "/";
            save2txt(CLayers[i].layer[j].W, str2, "ch_0.csv");
        }
    }
}

void
saveConvImage(const int ncls, const int nsample, const int nkernel, const Mat conv, string path){
    string tmp = path;
    mkdir(tmp.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	string str = tmp + "/layer_" + std::to_string(ncls);
	mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	string str2 = str + "/kernel_" + std::to_string(nkernel);
	mkdir(str2.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	str2 += "/";
	string str3 = "sample_" + std::to_string(nsample) + ".csv";
	save2txt(conv, str2, str3);
}

void
saveConvKernelGradient(const vector<Cvl> &CLayers, string path){
    string tmp = path + "/kernels_gradient";
    mkdir(tmp.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    int layers = CLayers.size();
    for(int i = 0; i < layers; i++){
        string str = tmp + "/layer_" + std::to_string(i);
        mkdir(str.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        int kernels = convConfig[i].KernelAmount;
        for(int j = 0; j < kernels; j++){
            string str2 = str + "/kernel_" + std::to_string(j);
            mkdir(str2.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            str2 += "/";
            save2txt(CLayers[i].layer[j].Wgrad, str2, "ch_0.txt");
            save2txt(CLayers[i].layer[j].Wd2, str2, "Wd2_ch_0.txt");
        }
    }
}
