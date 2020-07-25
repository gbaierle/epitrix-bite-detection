/** 
    UNISC - Computer Engineering Final Project

    Name:       Gabriel Augusto Baierle
    Advisor:    Rolf Fredi Molz, Andreas Kohler
    
    This project aims to segment and estimate the number os Epitrix
    bites in tobacco leaves, measuring a crop infestation level.
 **/

/*
    OpenCV Libraries
*/
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

/*
    Standard C++ libraries
*/
#include <iostream>
#include <fstream>
#include <math.h>
#include <string.h>
#include <cstdlib>

/*
    Namespaces
*/
using namespace cv;
using namespace std;

//########## Global Variables ##########

/*
    Region diameter to be considered as a bite.
 */
double biteDiameterReference = 99999;

/*
    Image index counter
*/
int imageCounter = 0;

/*
    Allows ptinting debug logs
*/
bool debugMode = true;

/*
    File with the keypoints diameter
*/
ofstream bitesDiameterFile;

//void openCvVersion();


/*
    Metodo estatico para obter a versao do OpenCV que sera utilizada.
 */
void openCvVersion() {
    if (debugMode) cout << "Using OpenCV version " << CV_VERSION << "\n" << endl;
}

/*
    
*/
class EpitrixDetection {

    public:
        int id;
        Mat image;
        Mat imageSegmented;
        vector<KeyPoint> keypoints;
        Mat imageBlobs;
    
    EpitrixDetection(
            int id,
            Mat image,
            Mat imageSegmented,
            vector<KeyPoint> keypoints,
            Mat imageBlobs) {

        this->id              = id;
        this->image           = image;
        this->imageSegmented  = imageSegmented;
        this->keypoints       = keypoints;
        this->imageBlobs      = imageBlobs;
    }

    ~EpitrixDetection() {}
};

/*
    Plot the images with a secific resolution, making the results visualization easier.
*/
void showImage(string name, Mat image) {
    Mat imageResized;
    resize(image, imageResized, Size(650,650), 0, 0, INTER_LINEAR);
    imshow(name + to_string(imageCounter), imageResized);
}


/*
    Retorna uma lista de keypoints representando os objetos fechados encontraados na imagem.

*/
vector<KeyPoint> detectBlobsIn(Mat image) {

    // Inicializa o detector com parâmetros padrão.
    // SimpleBlobDetector detector;

    // Inicializa o detector com diferentes parâmetros.
    SimpleBlobDetector::Params parameters;
     
    // Limiares de segmentação.
    parameters.minThreshold = 0;
    parameters.maxThreshold = 255;
     
    // Filtro por área.
    parameters.filterByArea = true;
    parameters.minArea = 15;// 30
    //parameters.maxArea = 200;
     
    // Filtro por circularidade
    parameters.filterByCircularity = false;
    parameters.minCircularity = 0.3;
    parameters.maxCircularity = 1;
     
    // Filtro de convexidade
    parameters.filterByConvexity = false;
    parameters.minConvexity = 0.5;
    parameters.maxConvexity = 1;
     
    // Filtro de inércia
    parameters.filterByInertia = false;
    parameters.minInertiaRatio = 0.01;
     
    // Cria o objeto detector com os parâmetros definidos acima
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(parameters);
     
    // Chamada do método de detecção.
    vector<KeyPoint> keypoints;
    detector->detect(image, keypoints);
     
    for (int i = 0; i < keypoints.size(); i++) {

        if (keypoints[i].size < biteDiameterReference) {
            biteDiameterReference = keypoints[i].size;
        }
    }

    return keypoints;
}

/*
    
*/
Mat drawLabeledKeypoints(Mat image, vector<KeyPoint> keypoints) {

    Mat auxiliar = image;

    if (debugMode) cout << "Bite reference = " << biteDiameterReference << endl;

    for (int i = 0; i < keypoints.size(); i++) {

        if (debugMode) cout << "keypoint " << i << " diameter = " << keypoints[i].size << "\n";

        vector<KeyPoint> keypointsAlone;
        Mat aux;

        keypointsAlone.push_back(keypoints[i]);
        //drawKeypoints( image, keypointsAlone, aux, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
        drawKeypoints( auxiliar, keypointsAlone, auxiliar, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

        string text = to_string(i);
        int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontScale = 0.8;
        int thickness = 2;
        putText(auxiliar, text, Point(keypointsAlone[0].pt.x,keypointsAlone[0].pt.y), fontFace, fontScale,Scalar(0,0,255), thickness, 8);
    }

    namedWindow( "BLOBS", CV_WINDOW_NORMAL );
    showImage("BLOBS", auxiliar);

    return auxiliar;
}
/*
    
*/
EpitrixDetection processUsingHChannelFromHsi(Mat image) {

    Mat imageHSI, hsi[3], imageOriginalWithBlobs, imageBlobs;

    imageHSI = bgrToHsi(image);
    
    showImage("hsi ", imageHSI);

    split(imageHSI, hsi); // Separar canais H, S e I

    //GaussianBlur(hsi[0], hsi[0], Size(15,15), 1, 1);
    
    showImage("gaussian ", hsi[0]);
    bitwise_not(hsi[0], hsi[0]); // Inversão dos pixels
    morphologyEx(hsi[0], hsi[0], MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(15, 15), Point(1, 1)));
    showImage("close ", hsi[0]);
    

    vector<KeyPoint> keypoints = detectBlobsIn(hsi[0]);
    if (debugMode) cout << keypoints.size() << " keypoints found hsi \n";

    Mat sHsv;
    cvtColor(image, sHsv, CV_BGR2HSV);
    Mat hsv[3];
    split(sHsv, hsv); // Separar canais H, S e I

    

    morphologyEx(hsv[0], hsv[0], MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(15, 15), Point(1, 1)));
    bitwise_not(hsv[0], hsv[0]); // Inversão dos pixels

    

    vector<KeyPoint> keypoints2 = detectBlobsIn(hsv[0]);
    if (debugMode) cout << keypoints2.size() << " keypoints found hsv \n";

    Mat auxiliar = drawLabeledKeypoints(image, keypoints);

    EpitrixDetection epitrixDetection(imageCounter, image, hsi[0], keypoints, auxiliar);

    return epitrixDetection;

}

/*

*/
EpitrixDetection processUsingChannelIFromHsi(Mat image) {

    Mat imageHSI, hsi[3], imageOriginalWithBlobs, imageBlobs;

    imageHSI = bgrToHsi(image);

    showImage("HSI", image);

    split(imageHSI, hsi); // Separar canais H, S e I

    split(imageHSI,hsi);//split source  
    showImage("HSI", imageHSI);
    showImage("H from HSI", hsi[0]);
    showImage("S from HSI", hsi[1]);
    showImage("I from HSI", hsi[2]);

    Mat graythresh;
    Mat imggray = hsi[2];
    threshold( imggray, graythresh, 210, 255,CV_THRESH_BINARY | CV_THRESH_OTSU );
    showImage("OTSU thresh", graythresh);

    imggray = hsi[2];
    threshold( imggray, graythresh, 210, 255,CV_THRESH_BINARY | CV_THRESH_TRIANGLE );
    showImage("TRIANGLE thresh", graythresh);

    imggray = hsi[2];
    adaptiveThreshold(imggray, graythresh, 210, ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 5, 0);
    showImage("Adaptive MEAN thresh", graythresh);

    imggray = hsi[2];
    adaptiveThreshold(imggray, graythresh, 210, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 5, 0);
    showImage("adaptive GAUSSIAN thresh", graythresh);


    //GaussianBlur(hsi[0], hsi[0], Size(3,3), 2, 2);
    morphologyEx(hsi[2], hsi[2], MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5), Point(1, 1)));
    
    bitwise_not(hsi[2],hsi[2]);
    showImage("Inverted I", hsi[2]);
    

    vector<KeyPoint> keypoints = detectBlobsIn(hsi[2]);
    if (debugMode) cout << keypoints.size() << " keypoints found\n";

    drawKeypoints(hsi[1], keypoints, imageBlobs, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(image, keypoints, imageOriginalWithBlobs, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    Mat auxiliar = drawLabeledKeypoints(image, keypoints);

    namedWindow( "BLOBS", CV_WINDOW_NORMAL );
    showImage("BLOBS", auxiliar);

    EpitrixDetection epitrixDetection(imageCounter, image, hsi[0], keypoints, auxiliar);

    return epitrixDetection;
}

/*

*/
EpitrixDetection processUsingBinaryThresholding(Mat image) {

    Mat imggray, graythresh;
    
    cvtColor(image, imggray, CV_BGR2GRAY);
    
    showImage("before gaussian ", imggray);

    GaussianBlur( imggray, imggray, Size( 77,77 ), 5, 5 );

    showImage("gaussian ", imggray);

    double min, max;
    Point min_loc, max_loc;
    minMaxLoc(imggray, &min, &max, &min_loc, &max_loc);   

    if (debugMode) cout << "min = " << min << ", max = " << max << "\n";

    int limiar = max-min;
    if (limiar < 220) limiar = max-5;

    if (debugMode) cout << "limiar = " << limiar << "\n";
    threshold( imggray, graythresh, limiar, 255, CV_THRESH_BINARY);
    
    showImage("before morph ", graythresh);
    morphologyEx(graythresh, graythresh, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(1, 1)));
    showImage("after morph ", graythresh);
    //dilate( graythresh, graythresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(1, 1)) );

    //showImage("ïmage after closing",graythresh);

    bitwise_not(graythresh,graythresh);

    showImage("graythresh", graythresh);

    vector<KeyPoint> keypoints = detectBlobsIn(graythresh);

    Mat auxiliar = drawLabeledKeypoints(image, keypoints);

    showImage("BLOBS", auxiliar);

    EpitrixDetection epitrixDetection(imageCounter, image, graythresh, keypoints, auxiliar);

    return epitrixDetection;
    
}

/*

*/
EpitrixDetection processUsingBinaryThresholdingWithFixedValue(Mat image) {

    Mat imggray, graythresh;
    
    cvtColor(image, imggray, CV_BGR2GRAY);
    
    GaussianBlur( imggray, imggray, Size( 15,15 ), 0, 0 );

    showImage("gaussian", imggray);
    
    threshold( imggray, graythresh, 230, 255, CV_THRESH_BINARY);

    morphologyEx(graythresh, graythresh, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(1, 1)));
    //dilate( graythresh, graythresh, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(1, 1)) );

    //showImage("ïmage after closing",graythresh);

    bitwise_not(graythresh,graythresh);

    showImage("graythresh", graythresh);

    vector<KeyPoint> keypoints = detectBlobsIn(graythresh);

    Mat auxiliar = drawLabeledKeypoints(image, keypoints);

    showImage("BLOBS", auxiliar);

    EpitrixDetection epitrixDetection(imageCounter, image, graythresh, keypoints, auxiliar);

    return epitrixDetection;
    
}

/*

*/
EpitrixDetection processUsingBarbedoMethod(Mat image) {

    GaussianBlur( image, image, Size( 15,15 ), 2, 2 );

    Mat imageBright, imageDark, imageResult;

    image.copyTo(imageBright);
    image.copyTo(imageDark);
    image.copyTo(imageResult);

    float r1, r2, M1, M2, M3, M4, Ma, Mb, M;
    float b, g, r;

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {

            b = image.at<Vec3b>(i, j)[0];
            g = image.at<Vec3b>(i, j)[1];
            r = image.at<Vec3b>(i, j)[2];

            r1 = r/(g+0.001);
            r2 = b/(g+0.001);

            if (r1 > 1) {
                M1 = 1; 
            } else {
                M1 = 0; 
            }

            if (r2 > 0.8) {
                M2 = 1; 
            } else {
                M2 = 0; 
            }

            if (r1 > 0.9) {
                M3 = 1; 
            } else {
                M3 = 0; 
            }

            if (r2 > 0.7) {
                M4 = 1; 
            } else {
                M4 = 0; 
            }

            Ma = M1 || M2;
            Mb = M3 && M4;

            M = Ma || Mb;

            imageBright.at<Vec3b>(i, j) = image.at<Vec3b>(i, j) * Mb;
            imageDark.at<Vec3b>(i, j) = image.at<Vec3b>(i, j) * Ma;
        }
    }

    showImage("Partes claras", imageBright);
    //showImage("Partes escuras", imageDark);

    Mat imggray, graythresh;

    cvtColor(imageBright, imggray, CV_BGR2GRAY);
    
    bitwise_not(imggray,imggray);    

    morphologyEx(imggray, imggray, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(1, 1)));

    showImage("ROI gray", imggray);

    vector<KeyPoint> keypointsBarbedo = detectBlobsIn(imggray);

    if (debugMode) cout << "keypoints " << keypointsBarbedo.size() << "\n";

    Mat auxiliar = drawLabeledKeypoints(image, keypointsBarbedo);

    EpitrixDetection epitrixDetection(imageCounter, image, imggray, keypointsBarbedo, auxiliar);

    return epitrixDetection;

}

/*
    Main function.
*/
int main(int argc, char** argv) {

    openCvVersion();

    Mat image;

    vector<EpitrixDetection> epitrixDetectionVector;

    VideoCapture cap("Samples/%03d.jpg");
    cap.read(image);

    const int dir= system("mkdir -p out");

    while(!image.empty()) {
        
        Mat imageResized = image;

        imageCounter++;
        cout << "\n\n#################### Processing image " << imageCounter << " ####################\n";

        Mat imageROI = imageResized; 
        /*
            Choose and uncomment the method to be used to process the images.
        */
        EpitrixDetection epitrixDetection = processUsingBinaryThresholding(imageROI);
        //EpitrixDetection epitrixDetection = processUsingBinaryThresholdingWithFixedValue(imageROI);
        //EpitrixDetection epitrixDetection = processUsingBarbedoMethod(imageROI);
        //EpitrixDetection epitrixDetection = processUsingHChannelFromHsi(imageROI);
        //EpitrixDetection epitrixDetection2 = processUsingChannelIFromHsi(imageROI);

        epitrixDetectionVector.push_back(epitrixDetection);

        string name = "./out/image"+to_string(imageCounter)+".jpg";
        imwrite(name, epitrixDetection.imageBlobs); // A JPG FILE IS BEING SAVED

        name = "./out/image"+to_string(imageCounter)+"Thresh"+".jpg";
        imwrite(name, epitrixDetection.imageSegmented); // A JPG FILE IS BEING SAVED

        cap.read(image);

        waitKey();
    }

    if (debugMode) cout << "Minimal bite number = " << biteDiameterReference << "\n";

    bitesDiameterFile.open ("bitesDiameter.csv");
    bitesDiameterFile << "index,keypoints,totalDiam,bites\n";

    for (int i = 0; i < epitrixDetectionVector.size(); i++) {
        
        long keypoints = epitrixDetectionVector[i].keypoints.size();
        double totalDiam = 0;
        for (int j = 0; j < keypoints; j++) {
            totalDiam += epitrixDetectionVector[i].keypoints[j].size;
        }

        double bites = totalDiam / biteDiameterReference;

        bitesDiameterFile << (i+1) << "," << keypoints << "," << totalDiam << "," <<bites << "\n";

        if (debugMode) cout << (i+1) << "," << keypoints << "," << totalDiam << "," <<bites << "\n";
        
    }
    bitesDiameterFile.close();

}