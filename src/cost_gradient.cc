#include "cost_gradient.h"

using namespace cv;
using namespace std;


void
getNetworkCost(vector<Mat> &x, Mat &y, vector<Cvl> &CLayers, vector<Fcl> &hLayers, Smr &smr){

    int nsamples = x.size();
    // Conv & Pooling
    unordered_map<string, Mat> cpmap;
    unordered_map<string, vector<Point> > locmap;
    convAndPooling(x, CLayers, cpmap, locmap);
    vector<Mat> P;
    vector<string> vecstr = getLayerKey(nsamples, CLayers.size() - 1, KEY_POOL);
    for(int i = 0; i<vecstr.size(); i++){
        P.push_back(cpmap.at(vecstr[i]));
    }
    Mat convolvedX = concatenateMat(P, nsamples);
    P.clear();

    // full connected layers
    vector<Mat> nonlin;
    vector<Mat> hidden;
    vector<Mat> acti;
    vector<double> factor;
    hidden.push_back(convolvedX);
    acti.push_back(convolvedX);
    vector<Mat> bernoulli;
    for(int i = 1; i <= fcConfig.size(); i++){
        Mat tmpacti = hLayers[i - 1].W * hidden[i - 1] + repeat(hLayers[i - 1].b, 1, convolvedX.cols);
        nonlin.push_back(tmpacti);
//        tmpacti = sigmoid(tmpacti);
        tmpacti = ReLU(tmpacti);
        if(fcConfig[i - 1].DropoutRate < 1.0){
            Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, fcConfig[i - 1].DropoutRate);
            hidden.push_back(tmpacti.mul(bnl));
            bernoulli.push_back(bnl);
        }else hidden.push_back(tmpacti);
        acti.push_back(tmpacti);
    }

    Mat M = smr.W * hidden[hidden.size() - 1] + repeat(smr.b, 1, nsamples);
    M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
    M = exp(M);
    Mat p = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));

    Mat groundTruth = Mat::zeros(softmaxConfig.NumClasses, nsamples, CV_64FC1);
    for(int i = 0; i < nsamples; i++){
        groundTruth.ATD(y.ATD(0, i), i) = 1.0;
    }
    double J1 = - sum1(groundTruth.mul(log(p))) / nsamples;
    double J2 = sum1(pow(smr.W, 2.0)) * softmaxConfig.WeightDecay / 2;
    double J3 = 0.0; 
    double J4 = 0.0;
    for(int hl = 0; hl < hLayers.size(); hl++){
        J3 += sum1(pow(hLayers[hl].W, 2.0)) * fcConfig[hl].WeightDecay / 2;
    }
    for(int cl = 0; cl < CLayers.size(); cl++){        
        for(int i = 0; i < convConfig[cl].KernelAmount; i++){
            J4 += sum1(pow(CLayers[cl].layer[i].W, 2.0)) * convConfig[cl].WeightDecay / 2;
        }
    }
    smr.cost = J1 + J2 + J3 + J4;
    if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<smr.cost<<endl;
    // bp - softmax
    smr.Wgrad =  - (groundTruth - p) * hidden[hidden.size() - 1].t() / nsamples + softmaxConfig.WeightDecay * smr.W;
    smr.bgrad = - reduce((groundTruth - p), 1, CV_REDUCE_SUM) / nsamples;
    smr.Wd2 = pow((groundTruth - p), 2.0) * pow(hidden[hidden.size() - 1].t(), 2.0);
    smr.Wd2 = smr.Wd2 / nsamples + softmaxConfig.WeightDecay;
    smr.bd2 = reduce(pow((groundTruth - p), 2.0), 1, CV_REDUCE_SUM) / nsamples;
    // bp - full connected
    vector<Mat> delta(hidden.size());
    vector<Mat> deltad2(hidden.size());
    delta[delta.size() - 1] = -smr.W.t() * (groundTruth - p);
//    delta[delta.size() - 1] = delta[delta.size() -1].mul(dsigmoid_a(acti[acti.size() - 1]));
    deltad2[deltad2.size() - 1] = pow(smr.W.t(), 2) * pow((groundTruth - p), 2.0);
    if(!fcConfig.empty()){
        delta[delta.size() - 1] = delta[delta.size() -1].mul(dReLU(nonlin[nonlin.size() -1]));
        deltad2[deltad2.size() - 1] = deltad2[deltad2.size() -1].mul(pow(dReLU(nonlin[nonlin.size() -1]), 2.0));
    }
//    deltad2[deltad2.size() - 1] = deltad2[deltad2.size() -1].mul(pow(dsigmoid_a(acti[acti.size() -1]), 2.0));
    if(!fcConfig.empty() && fcConfig[fcConfig.size() - 1].DropoutRate < 1.0){
        delta[delta.size() - 1] = delta[delta.size() -1].mul(bernoulli[bernoulli.size() - 1]);
        deltad2[deltad2.size() - 1] = deltad2[deltad2.size() -1].mul(pow(bernoulli[bernoulli.size() - 1], 2.0));
    } 
    for(int i = delta.size() - 2; i >= 0; i--){
        delta[i] = hLayers[i].W.t() * delta[i + 1];
        deltad2[i] = pow(hLayers[i].W.t(), 2.0) * deltad2[i + 1];
        if(i > 0){
            //delta[i] = delta[i].mul(dsigmoid_a(acti[i]));
            delta[i] = delta[i].mul(dReLU(nonlin[i - 1]));
            //deltad2[i] = deltad2[i].mul(pow(dsigmoid_a(acti[i]), 2.0));
            deltad2[i] = deltad2[i].mul(pow(dReLU(nonlin[i - 1]), 2.0));
            if(fcConfig[i - 1].DropoutRate < 1.0){
                delta[i] = delta[i].mul(bernoulli[i - 1]);
                deltad2[i] = deltad2[i].mul(pow(bernoulli[i - 1], 2.0));
            }
        }else{
            ; // 1st layer, do nothing
        }             
    }
    for(int i = fcConfig.size() - 1; i >= 0; i--){
        hLayers[i].Wgrad = delta[i + 1] * (hidden[i]).t();
        hLayers[i].Wgrad = hLayers[i].Wgrad / nsamples + fcConfig[i].WeightDecay * hLayers[i].W;
        hLayers[i].bgrad = reduce(delta[i + 1], 1, CV_REDUCE_SUM) / nsamples;
        hLayers[i].Wd2 = deltad2[i + 1] * pow((hidden[i]).t(), 2.0) / nsamples + fcConfig[i].WeightDecay;
        hLayers[i].bd2 = reduce(deltad2[i + 1], 1, CV_REDUCE_SUM) / nsamples;
    }
    //bp - Conv layer
    hashDelta(delta[0], cpmap, CLayers.size(), HASH_DELTA);
    hashDelta(deltad2[0], cpmap, CLayers.size(), HASH_HESSIAN);
    for(int cl = CLayers.size() - 1; cl >= 0; cl --){
        int pDim = convConfig[cl].PoolingDim;
        vector<string> deltaKey = getLayerKey(nsamples, cl, KEY_DELTA);
        for(int k = 0; k < deltaKey.size(); k ++){
            string locstr = deltaKey[k].substr(0, deltaKey[k].length() - 1);
            string deltaStr = locstr + "D";
            string hessianStr = locstr + "H";
            string upDstr = locstr + "UD";
            string upHstr = locstr + "UH";
            Mat upDelta = UnPooling(cpmap.at(deltaStr), pDim, pDim, pooling_method, locmap.at(locstr));
            Mat upHessian = UnPooling(cpmap.at(hessianStr), pDim, pDim, pooling_method, locmap.at(locstr));
            cpmap[upDstr] = upDelta;
            cpmap[upHstr] = upHessian;
        }
        for(int k = 0; k < deltaKey.size(); k ++){
            string locstr = deltaKey[k].substr(0, deltaKey[k].length() - 1);
            string convstr = locstr.substr(0, locstr.length() - 1);
            // Local response normalization 
            string upDstr = locstr + "UD";
            string upHstr = locstr + "UH";
            if(convConfig[cl].useLRN){
                Mat dLRN = dlocalResponseNorm(cpmap, upDstr); 
                Mat dhLRN = dlocalResponseNorm(cpmap, upHstr);
                dLRN = dLRN.mul(dnonLinearity(cpmap[convstr]));
                dhLRN = dhLRN.mul(pow(dnonLinearity(cpmap[convstr]), 2.0));
                cpmap[upDstr] = dLRN;
                cpmap[upHstr] = dhLRN;
            }else{
                Mat upDelta;
                Mat upHessian;
                cpmap.at(upDstr).copyTo(upDelta);
                upDelta = upDelta.mul(dnonLinearity(cpmap.at(convstr)));
                cpmap[upDstr] = upDelta;
                cpmap.at(upHstr).copyTo(upHessian);
                upHessian = upHessian.mul(pow(dnonLinearity(cpmap.at(convstr)), 2.0));
                cpmap[upHstr] = upHessian;
            }
        }

        if(cl > 0){
            for(int k = 0; k < convConfig[cl - 1].KernelAmount; k ++){
                vector<string> prevUD = getSpecKeys(nsamples, cl, cl - 1, k, KEY_UP_DELTA);
                vector<string> prevUH = getSpecKeys(nsamples, cl, cl - 1, k, KEY_UP_HESSIAN);
                for(int i = 0; i < prevUD.size(); i++){
                    string strd = getPreviousLayerKey(prevUD[i], KEY_DELTA);
                    string strh = getPreviousLayerKey(prevUH[i], KEY_HESSIAN);
                    if(cpmap.find(strd) == cpmap.end()){
                        string psize = getPreviousLayerKey(prevUD[i], KEY_POOL);
                        Mat zero1 = Mat::zeros(cpmap.at(psize).rows, cpmap.at(psize).cols, CV_64FC1);
                        Mat zero2 = Mat::zeros(cpmap.at(psize).rows, cpmap.at(psize).cols, CV_64FC1);
                        cpmap[strd] = zero1;
                        cpmap[strh] = zero2;
                    }
                    int currentKernel = getCurrentKernelNum(prevUD[i]);
                    cpmap.at(strd) += convCalc(cpmap.at(prevUD[i]), CLayers[cl].layer[currentKernel].W, CONV_SAME);
                    cpmap.at(strh) += convCalc(cpmap.at(prevUH[i]), pow(CLayers[cl].layer[currentKernel].W, 2.0), CONV_SAME);
                }
            }
        }
        for(int j = 0; j < convConfig[cl].KernelAmount; j ++){
            Mat tpgradW = Mat::zeros(convConfig[cl].KernelSize, convConfig[cl].KernelSize, CV_64FC1);
            double tpgradb = 0.0;
            Mat tpWd2 = Mat::zeros(tpgradW.size(), CV_64FC1);
            double tpbd2 = 0.0;
            vector<string> convKey = getKeys(nsamples, cl, j, KEY_UP_DELTA);
            for(int m = 0; m < convKey.size(); m ++){
                Mat temp = rot90(cpmap.at(convKey[m]), 2);
                if(cl == 0){
                    tpgradW += convCalc(x[getSampleNum(convKey[m])], temp, CONV_VALID);
                }

                else{
                    string strprev = getPreviousLayerKey(convKey[m], KEY_POOL);
                    tpgradW += convCalc(cpmap.at(strprev), temp, CONV_VALID);
                }
                tpgradb += sum(temp)[0];
            }
            vector<string> convKey2 = getKeys(nsamples, cl, j, KEY_UP_HESSIAN);
            for(int m = 0; m < convKey2.size(); m ++){
                Mat temp = rot90(cpmap.at(convKey2[m]), 2);
                if(cl == 0){
                    tpWd2 += convCalc(pow(x[getSampleNum(convKey2[m])], 2.0), temp, CONV_VALID);
                }else{
                    string strprev = getPreviousLayerKey(convKey2[m], KEY_POOL);
                    tpWd2 += convCalc(pow(cpmap.at(strprev), 2.0), temp, CONV_VALID);
                }
                tpbd2 +=sum(temp)[0];
            }
            CLayers[cl].layer[j].Wgrad = tpgradW / convKey.size() + convConfig[cl].WeightDecay * CLayers[cl].layer[j].W;
            CLayers[cl].layer[j].bgrad = tpgradb / convKey.size();
            CLayers[cl].layer[j].Wd2 = tpWd2 / convKey.size() + convConfig[cl].WeightDecay;
            CLayers[cl].layer[j].bd2 = tpbd2 / convKey.size();
            tpgradW.release();
        }
    }
    // destructor
    p.release();
    M.release();
    groundTruth.release();
    cpmap.clear();
    unordered_map<string, Mat>().swap(cpmap);
    locmap.clear();
    unordered_map<string, vector<Point> >().swap(locmap);
    acti.clear();
    vector<Mat>().swap(acti);
    delta.clear();
    vector<Mat>().swap(delta);
    bernoulli.clear();
    vector<Mat>().swap(bernoulli);
    nonlin.clear();
    vector<Mat>().swap(nonlin);
}
