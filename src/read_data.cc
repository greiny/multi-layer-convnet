#include "read_data.h"

using namespace cv;
using namespace std;

int 
ReverseInt (int i){
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void
read_MNIST_data(vector<Mat> &trainX, vector<Mat> &testX, Mat &trainY, Mat &testY){


	readImage(trainX, trainY,testX, testY);
    //readData(trainX, trainY, "mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", 60000);
    //readData(testX, testY, "mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", 10000);
    //readcsv(trainX,trainY,testX,testY);
    preProcessing(trainX, testX);
    dataEnlarge(trainX, trainY);

    cout<<"****************************************************************************"<<endl
        <<"**                        READ DATASET COMPLETE                             "<<endl
        <<"****************************************************************************"<<endl<<endl;
    cout<<"The training data has "<<trainX.size()<<" images, each images has "<<trainX[0].cols<<" columns and "<<trainX[0].rows<<" rows."<<endl;
    cout<<"The testing data has "<<testX.size()<<" images, each images has "<<testX[0].cols<<" columns and "<<testX[0].rows<<" rows."<<endl;
    cout<<"There are "<<trainY.cols<<" training labels and "<<testY.cols<<" testing labels."<<endl<<endl;
}

void 
read_Mnist(string filename, vector<Mat> &vec){
    ifstream file(filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i){
            Mat tpmat = Mat::zeros(n_rows, n_cols, CV_8UC1);
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tpmat.at<uchar>(r, c) = (int) temp;
                }
            }
            vec.push_back(tpmat);
        }
    }
}

void 
read_Mnist_Label(string filename, Mat &mat)
{
    ifstream file(filename, ios::binary);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i){
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            mat.ATD(0, i) = (double)temp;
        }
        file.close();
    }
}

void
readData(vector<Mat> &x, Mat &y, string xpath, string ypath, int number_of_images){

    //read MNIST iamge into OpenCV Mat vector
    read_Mnist(xpath, x);
    for(int i = 0; i < x.size(); i++){
        x[i].convertTo(x[i], CV_64FC1, 1.0/255, 0);
    }
    //read MNIST label into double vector
    y = Mat::zeros(1, number_of_images, CV_64FC1);
    read_Mnist_Label(ypath, y);
}

void
readcsv(vector<Mat> &trainX, Mat &trainY, vector<Mat> &testX, Mat &testY)
{
	vector<double*> data;
	//open file for reading
	fstream inputFile;
	inputFile.open("final_example.csv", ios::in);

	if ( inputFile.is_open() )
	{
		string line = "";
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);
			if (line.length() > 2 )
			{
				//store inputs
				char* cstr = new char[line.size()+1];
				char* t;
				strcpy(cstr, line.c_str());

				//tokenise
				int i = 0;
				t=strtok (cstr,",");
				double* dat = new double[(int)2];
				while ( t!=NULL && i < 2 ) // image_name and steering angle
				{
					if ( i < 1 ) dat[0] = atof(t);
					else dat[1] = atof(t);
					//move token onwards
					t = strtok(NULL,",");
					i++;
				}
				data.push_back(dat);
				//free memory
				delete[] cstr;
			}
		}
		cout << "Read Complete: " << data.size() << " Patterns Loaded"  << endl;
		inputFile.close();
	}
	else cout << endl << "Error - input file could not be opened: " << endl;
#if 1
	//set steering angle
	Mat prev = imread("center/(1).jpg",0);
	equalizeHist(prev,prev);
	resize(prev,prev,Size(),0.5,0.5);
	for (int k=0; k < data.size(); k++ )
	{
		char file_name[255];
		sprintf(file_name,"center/(%.0f).jpg",(float)data[k][0]);
		Mat next = imread(file_name,0);
		resize(next,next,Size(),0.5,0.5);
		equalizeHist(next,next);
		Mat buf = next-prev;
		sprintf(file_name,"proc/(%.0f).jpg",(float)(k+1));
		imwrite(file_name,buf);
		//calcOpticalFlowFarneback(prev, next, none, 0.5, 3, 15, 3, 5, 1.2, 0);
		next.copyTo(prev);
		next.release();
		buf.release();
	}
	prev.release();
#endif
	random_shuffle(data.begin(), data.end());
	vector<double> strangle; strangle.reserve(data.size()-1);
	for (int k=0; k < data.size(); k++ )
	{
		if (data[k][0]==1) continue;
		char file_name[255];
		sprintf(file_name,"proc/(%.0f).jpg",(float)data[k][0]);
		Mat buf = imread(file_name,0);
		buf.convertTo(buf, CV_64FC1, 1.0/255, 0);
		if (k<(int)(data.size()*0.8)) trainX.push_back(buf);
			else testX.push_back(buf);
		strangle.push_back(data[k][1]);
		buf.release();
	}
	for (int k=0; k < data.size(); k++ ) delete[] data[k];
	data.clear();

	trainY = Mat::zeros(8, trainX.size(), CV_64FC1);
	testY = Mat::zeros(8,testX.size(), CV_64FC1);
	for(int i = 0; i < trainX.size()+testX.size(); i++){
		double temp = 1;
		if (i<trainX.size()) {
			if (strangle[i] > 0.1) trainY.ATD(0, i) = temp;
			else if (0.05 < strangle[i] && strangle[i]< 0.1) trainY.ATD(1, i) = temp;
			else if (0.02 < strangle[i] && strangle[i]< 0.05) trainY.ATD(2, i) = temp;
			else if (0 < strangle[i] && strangle[i] < 0.02) trainY.ATD(3, i) = temp;
			else if (-0.02 < strangle[i] && strangle[i] < 0) trainY.ATD(4, i) = temp;
			else if (-0.05 < strangle[i] && strangle[i] < -0.02) trainY.ATD(5, i) = temp;
			else if (-0.1 < strangle[i] && strangle[i] < -0.05) trainY.ATD(6, i) = temp;
			else if (strangle[i] < -0.1 ) trainY.ATD(7, i) = temp;
		}
		else{
			if (strangle[i] > 0.1) testY.ATD(0, (i-trainX.size())) = temp;
			else if (0.05 < strangle[i] && strangle[i]< 0.1) testY.ATD(1, (i-trainX.size())) = temp;
			else if (0.02 < strangle[i] && strangle[i]< 0.05) testY.ATD(2, (i-trainX.size())) = temp;
			else if (0 < strangle[i] && strangle[i] < 0.02) testY.ATD(3, (i-trainX.size())) = temp;
			else if (-0.02 < strangle[i] && strangle[i] < 0) testY.ATD(4, (i-trainX.size())) = temp;
			else if (-0.05 < strangle[i] && strangle[i] < -0.02) testY.ATD(5, (i-trainX.size())) = temp;
			else if (-0.1 < strangle[i] && strangle[i] < -0.05) testY.ATD(6, (i-trainX.size())) = temp;
			else if (strangle[i] < -0.1 ) testY.ATD(7, (i-trainX.size())) = temp;
		}
	}
	strangle.clear();
}

void
readImage(vector<Mat> &trainX, Mat &trainY,vector<Mat> &testX, Mat &testY){
	vector<Mat> dataO,dataX;
	for ( int i=1 ; i <= 1486 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"test/correct/correct(%d).png",i);
		buf = imread(file_name,0);
		Mat kk = Mat::ones(6, 6, CV_8UC1);
		erode(buf, buf, kk);
		dataO.push_back(buf);
		buf.release();
	}
	for ( int i=0 ; i < dataO.size() ; i++ )
	{
		if (i<(int)(dataO.size()*0.7)) trainX.push_back(dataO[i]);
		else testX.push_back(dataO[i]);
	}
	for ( int i=0 ; i <= 1665 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"test/incorrect/incorrect(%d).png",i);
		buf = imread(file_name,0);
		Mat kk = Mat::ones(6, 6, CV_8UC1);
		erode(buf, buf, kk);
		dataX.push_back(buf);
		buf.release();
	}
	for ( int i=0 ; i < dataX.size() ; i++ )
	{
		if (i<(int)(dataX.size()*0.7)) trainX.push_back(dataX[i]);
		else testX.push_back(dataX[i]);
	}

	trainY = Mat::zeros(2, trainX.size(), CV_64FC1);
	testY = Mat::zeros(2,testX.size(), CV_64FC1);
	for(int j = 0; j < trainX.size(); j++){
		if (j<(int)(dataO.size()*0.7)) trainY.ATD(0, j) = (double)1;
		else trainY.ATD(1, j) = (double)1;
	}
	for(int j = 0; j < testX.size(); j++){
		if (j<(int)(dataO.size()*0.3)) testY.ATD(0, j) = (double)1;
		else testY.ATD(1, j) = (double)1;
	}
	dataO.clear();
	dataX.clear();

	/*vector<Mat> dataO,dataX;
	trainY = Mat::zeros(2, 845+338-300, CV_64FC1);
	testY = Mat::zeros(2,200+100, CV_64FC1);

	for ( int i=1 ; i <= 845 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"target/o/o(%d).png",i);
		buf = imread(file_name,0);
		//resize(buf,buf,Size(32,32));
		dataO.push_back(buf);
		buf.release();
	}
	random_shuffle(dataO.begin(), dataO.end());
	for ( int i=0 ; i < dataO.size() ; i++ )
	{
		if (i<200) testX.push_back(dataO[i]);
		else trainX.push_back(dataO[i]);
	}

	for ( int i=1 ; i <= 338 ; i++ )
	{
		Mat buf;
		char file_name[255];
		sprintf(file_name,"target/x/x(%d).png",i);
		buf = imread(file_name,0);
		//resize(buf,buf,Size(32,32));
		dataX.push_back(buf);
		buf.release();
	}
	random_shuffle(dataX.begin(), dataX.end());
	for ( int i=0 ; i < dataX.size() ; i++ )
	{
		if (i<100) testX.push_back(dataX[i]);
		else trainX.push_back(dataX[i]);
	}

	for(int j = 0; j < 845+338-300; j++){
		unsigned char temp = 1;
		if (j<645) trainY.ATD(0, j) = (double)temp;
			else trainY.ATD(1, j) = (double)temp;
	}
	for(int j = 0; j < 200+100; j++){
		unsigned char temp = 1;
		if (j<200) testY.ATD(0, j) = (double)temp;
			else testY.ATD(1, j) = (double)temp;
	}
	//for(int i = 0; i < trainX.size(); i++) trainX[i].convertTo(trainX[i], CV_64FC1, 1.0/255, 0);
	//for(int i = 0; i < testX.size(); i++) testX[i].convertTo(testX[i], CV_64FC1, 1.0/255, 0);

	dataO.clear();
	dataX.clear();*/
}


Mat 
concat(const vector<Mat> &vec){
    int height = vec[0].rows * vec[0].cols;
    int width = vec.size();
    Mat res = Mat::zeros(height, width, CV_64FC1);
    for(int i = 0; i < vec.size(); i++){
        Rect roi = Rect(i, 0, 1, height);
        Mat subView = res(roi);
        Mat ptmat = vec[i].reshape(0, height);
        ptmat.copyTo(subView);
    }
    return res;
}

void
preProcessing(vector<Mat> &trainX, vector<Mat> &testX){

    for(int i = 0; i < trainX.size(); i++){
        trainX[i].convertTo(trainX[i], CV_64FC1, 1.0/255, 0);
    }
    for(int i = 0; i < testX.size(); i++){
        testX[i].convertTo(testX[i], CV_64FC1, 1.0/255, 0);
    }

    // first convert vec of mat into a single mat
    Mat tmp = concat(trainX);
    Mat tmp2 = concat(testX);
    Mat alldata = Mat::zeros(tmp.rows, tmp.cols + tmp2.cols, CV_64FC1);

    tmp.copyTo(alldata(Rect(0, 0, tmp.cols, tmp.rows)));
    tmp2.copyTo(alldata(Rect(tmp.cols, 0, tmp2.cols, tmp.rows)));

    Scalar mean;
    Scalar stddev;
    meanStdDev (alldata, mean, stddev);

    for(int i = 0; i < trainX.size(); i++){
        divide(trainX[i] - mean[0], stddev[0], trainX[i]);
    }
    for(int i = 0; i < testX.size(); i++){
        divide(testX[i] - mean[0], stddev[0], testX[i]);
    }
    tmp.release();
    tmp2.release();
    alldata.release();
}





