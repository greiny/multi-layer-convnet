#include "general_settings.h"
#include "convolution.h"

using namespace cv;
using namespace std;

std::vector<ConvLayerConfig> convConfig;
std::vector<FullConnectLayerConfig> fcConfig;
SoftmaxLayerConfig softmaxConfig;
vector<int> sample_vec;

///////////////////////////////////
// General parameters
///////////////////////////////////
bool is_gradient_checking = false;
bool use_log = false;
int log_iter = 0;
int batch_size = 100;
int pooling_method = 0;
int non_linearity = 2;
int training_epochs = 0;
double lrate_w = 0.0;
double lrate_b = 0.0;
int iter_per_epo = 0;

void
run(){
    vector<Mat> trainX;
    vector<Mat> testX;
    Mat trainY, testY;
    read_MNIST_data(trainX, testX, trainY, testY);

    int imgDim = trainX[0].rows*trainX[0].cols;
    int nsamples = trainX.size();

    for(int i = 0; i < nsamples; i++){
        sample_vec.push_back(i);
    }

    vector<Cvl> ConvLayers;
    vector<Fcl> HiddenLayers;
    Smr smr;
    readConfigFile("config.txt");
    ConvNetInitPrarms(ConvLayers, HiddenLayers, smr, imgDim, nsamples);
    // Train network using Back Propogation
    trainNetwork(trainX, trainY, ConvLayers, HiddenLayers, smr, testX, testY);
    ConvLayers.clear();
    HiddenLayers.clear();

}

//void* video_thread(void*);

int 
main(int argc, char** argv){
    string str = "clean_log";
    if(argv[1] && !str.compare(argv[1])){
        system("rm -rf log");
        cout<<"Cleaning log ..."<<endl;
        return 0;
    }
    long start, end;
    start = clock();


//int status = 0;
//pthread_t id;
//if(pthread_create(&id,NULL,video_thread,NULL)) printf("UDP thread create error\n");

    run();

//pthread_join(id,(void**)&status);

    end = clock();
    cout<<"Totally used time: "<<((double)(end - start)) / CLOCKS_PER_SEC<<" second"<<endl;
    return 0;
}

/*void* video_thread(void*){
	UDP.udp();
	return 0;
}*/
