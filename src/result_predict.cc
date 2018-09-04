#include "result_predict.h"

using namespace cv;
using namespace std;

Mat 
resultPredict(const vector<Mat> &x, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){
 
    int nsamples = x.size();
    // Conv & Pooling
    vector<vector<Mat> > conved;
    convAndPooling(x, CLayers, conved);
    Mat convolvedX = concatenateMat(conved);

    // full connected layers
    vector<Mat> hidden;
    hidden.push_back(convolvedX);
    for(int i = 1; i <= fcConfig.size(); i++){
        Mat tmpacti = hLayers[i - 1].W * hidden[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
//        tmpacti = sigmoid(tmpacti);
        tmpacti = ReLU(tmpacti);
        if(fcConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(fcConfig[i - 1].DropoutRate);
        hidden.push_back(tmpacti);
    }
    Mat M = smr.W * hidden[hidden.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat result = Mat::zeros(softmaxConfig.NumClasses, M.cols, CV_64FC1);

    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD((int)maxLoc.y, i) = 1;
    }
    // destructor
    for(int i = 0; i < conved.size(); i++){
        conved[i].clear();
    }
    conved.clear();
    vector<vector<Mat> >().swap(conved);
    M.release();
    hidden.clear();
    vector<Mat>().swap(hidden);
    convolvedX.release();
    return result;
}

Mat
resultPredict4test(const vector<Mat> &x, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){

    int nsamples = x.size();
    // Conv & Pooling
    vector<vector<Mat> > conved;
    convAndPooling4Test(x, CLayers, conved);
    Mat convolvedX = concatenateMat(conved);

    // full connected layers
    vector<Mat> hidden;
    hidden.push_back(convolvedX);
    for(int i = 1; i <= fcConfig.size(); i++){
        Mat tmpacti = hLayers[i - 1].W * hidden[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
//        tmpacti = sigmoid(tmpacti);
        tmpacti = ReLU(tmpacti);
        if(fcConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(fcConfig[i - 1].DropoutRate);
        hidden.push_back(tmpacti);
    }
    Mat M = smr.W * hidden[hidden.size() - 1] + repeat(smr.b, 1, nsamples);
    Mat result = Mat::zeros(softmaxConfig.NumClasses, M.cols, CV_64FC1);

    save2txt(hLayers[0].W,"log/","weight.txt");
    save2txt(hLayers[0].b,"log/","weight_bias.txt");
    save2txt(smr.W,"log/","smr.txt");
    save2txt(smr.b,"log/","smr_bias.txt");
    saveConvKernel(CLayers, "log/");
    saveConvKernelGradient(CLayers, "log/");

    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD((int)maxLoc.y, i) = 1;
    }
    // destructor
    for(int i = 0; i < conved.size(); i++){
        conved[i].clear();
    }
    conved.clear();
    vector<vector<Mat> >().swap(conved);
    M.release();
    hidden.clear();
    vector<Mat>().swap(hidden);
    convolvedX.release();
    return result;
}

void 
testNetwork(const vector<Mat> &testX, const Mat &testY, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){   
    // Test use test set
    // Because it may leads to lack of memory if testing the whole dataset at 
    // one time, so separate the dataset into small pieces of batches (say, batch size = 100).

    int batchSize = 100;
    Mat result = Mat::zeros(testY.rows, testX.size(), CV_64FC1);
    vector<Mat> tmpBatch;
    int batch_amount = testX.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
//        cout<<"processing batch No. "<<i<<endl;
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(testX[i * batchSize + j]);
        }
        Mat resultBatch = resultPredict(tmpBatch, CLayers, hLayers, smr);
        Rect roi = Rect(i * batchSize, 0, batchSize, testY.rows);
        resultBatch.copyTo(result(roi));
        tmpBatch.clear();
    }
    if(testX.size() % batchSize){
//        cout<<"processing batch No. "<<batch_amount<<endl;
        for(int j = 0; j < testX.size() % batchSize; j++){
            tmpBatch.push_back(testX[batch_amount * batchSize + j]);
        }
        Mat resultBatch = resultPredict(tmpBatch, CLayers, hLayers, smr);
        Rect roi = Rect(batch_amount * batchSize, 0, testX.size() % batchSize, testY.rows);
        resultBatch.copyTo(result(roi));
        ++ batch_amount;
        tmpBatch.clear();
    }

    Mat err;
    testY.copyTo(err);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
        for(int j=0; j<err.rows; j++){
        	if(err.ATD(j, i) == -1 ) --correct;
        }
    }
    cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
    result.release();
    err.release();
}

void
testNetwork4test(const vector<Mat> &testX, const Mat &testY, const vector<Cvl> &CLayers, const vector<Fcl> &hLayers, const Smr &smr){
    // Test use test set
    // Because it may leads to lack of memory if testing the whole dataset at
    // one time, so separate the dataset into small pieces of batches (say, batch size = 100).
    //

    int batchSize = testX.size();
    Mat result = Mat::zeros(testY.rows, testX.size(), CV_64FC1);
    vector<Mat> tmpBatch;
    int batch_amount = testX.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
//        cout<<"processing batch No. "<<i<<endl;
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(testX[i * batchSize + j]);
        }
        Mat resultBatch = resultPredict4test(tmpBatch, CLayers, hLayers, smr);
        Rect roi = Rect(i * batchSize, 0, batchSize, testY.rows);
        resultBatch.copyTo(result(roi));
        tmpBatch.clear();
    }
    if(testX.size() % batchSize){
//        cout<<"processing batch No. "<<batch_amount<<endl;
        for(int j = 0; j < testX.size() % batchSize; j++){
            tmpBatch.push_back(testX[batch_amount * batchSize + j]);
        }
        Mat resultBatch = resultPredict4test(tmpBatch, CLayers, hLayers, smr);
        Rect roi = Rect(batch_amount * batchSize, 0, testX.size() % batchSize, testY.rows);
        resultBatch.copyTo(result(roi));
        ++ batch_amount;
        tmpBatch.clear();
    }
    Mat err;
    testY.copyTo(err);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
         for(int j=0; j<err.rows; j++){
         	if(err.ATD(j, i) == -1 ) --correct;
         }
     }

    cout<<"correct: "<<correct<<", total: "<<err.cols<<", accuracy: "<<double(correct) / (double)(err.cols)<<endl;
    result.release();
    err.release();
}
