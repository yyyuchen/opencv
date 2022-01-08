#include <iostream>
using namespace std;

#include "opencv2/opencv.hpp"
using namespace cv;

void readImage();
void drawImage();
void changeImage();
void split_merge();
void changeColor();
void imageAdd();
void imageMix();
void resizeImage();
void moveImage();
void rotateImage();
void erode_dilateImage();
void open_close();
void top_black_hat();
void mean_blur();
void gaussian_blur();
void median_blur();
void calcHist();
void yanmo();
void zhifangtujunhenghua();
void zishiyingjunhenghua();
void sobel();
void laplacian();
void canny();
void match();
void hough();
void hough_circle();
void harris();
void shi_tomas();
void sift();
void fast();
void orb();

int main() {
//    readImage();
//    drawImage();
//    changeImage();
//    split_merge();
//    changeColor();
//    imageAdd();
//    imageMix();
//    resizeImage();
//    moveImage();
//    rotateImage();
//    erode_dilateImage();
//    open_close();
//    top_black_hat();
//    mean_blur();
//    gaussian_blur();
//    median_blur();
//    calcHist();
//    yanmo();
//    zhifangtujunhenghua();
//    zishiyingjunhenghua();
    sobel();
//    laplacian();
//    canny();
//    match();
//    hough();
//    hough_circle();
//    harris();
//    shi_tomas();
//    sift();
//    fast();
//    orb();

}

void readImage(){
    Mat img = imread("F:\\picture\\boy.png");
    imshow("image",img);
    waitKey(0);
//    imwrite("G:\\test.png", img);
}

void drawImage(){
    Mat img = imread("F:\\picture\\y.png", 1);
    Point pt1 = Point(0, 0);
    Point pt2 = Point(100, 100);
    line(img, pt1, pt2, Scalar(255, 0, 0), 5);

    Point pt3 = Point(150, 0);
    Point pt4 = Point(300, 150);
    rectangle(img, pt3, pt4, Scalar(0, 255, 0), -1);

    Point pt5 = Point(500, 300);
    circle(img, pt5, 50, Scalar(0, 0, 255), 2);

    Point pt6 = Point(100, 300);
    putText(img, "openCV", pt6, FONT_HERSHEY_SIMPLEX, 1, Scalar(155, 155, 155), 4);
    imshow("image", img);
    waitKey(0);
}

void changeImage(){

    Mat img = imread("F:\\picture\\boy.png");
    Vec3b &bgr = img.at<Vec3b>(100, 100);
    int blue = int(bgr[1]);
    bgr[0] = 0;
    cout<<"================"<<bgr<<"============="<<blue<<endl;
}

//
void split_merge(){

    Mat img = Mat(500, 500, CV_8UC3, Scalar(133, 18, 206));
    cout<<img.size<<"\n"<<img.flags<<"\n"<<img.cols<<"\n"<<img.rows<<"\n"<<img.dims<<"\n"<<img.channels()<<endl;
    vector<Mat> planes;
    split(img, planes);
    planes[0] = 0;
    planes[1] = 0;
    planes[2] = 255;
    Mat dst;
    merge(planes,dst);
    imshow("merge", dst);
    waitKey(0);
}

void changeColor(){
    Mat img = imread("F:\\picture\\boy.png");
    Mat HSV, gray, Lab, YUV;

    cvtColor(img, HSV, COLOR_BGR2HSV);
    cvtColor(img, gray, COLOR_BGR2GRAY);
    cvtColor(img, Lab, COLOR_BGR2Lab);
    cvtColor(img, YUV, COLOR_BGR2YUV);
    imshow("img", img);
    imshow("HSV", HSV);
    imshow("gray", gray);
    imshow("Lab", Lab);
    imshow("YUV", YUV);
    waitKey(0);
}

void imageAdd(){
    Mat img1 = Mat(500, 500, CV_8UC3, Scalar(155, 0, 0));
    Mat img2 = Mat(500, 500, CV_8UC3, Scalar(0, 155, 0));
    Mat img3 = Mat(500, 500, CV_8UC3, Scalar(0, 0, 155));
    Mat img;
    add(img1, img3,img);
    imshow("img", img);
    waitKey(0);
}

void imageMix(){
    Mat img1 = Mat(500, 500, CV_8UC3, Scalar(155, 0, 0));
    Mat img2 = Mat(500, 500, CV_8UC3, Scalar(0, 0, 155));
    Mat img;
    addWeighted(img1, 0.3, img2, 0.7,50,img);
    imshow("img", img);
    waitKey(0);
}

void resizeImage(){
    Mat img = imread("F:\\picture\\boy.png");
    int cols = img.cols;
    int rows = img.rows;
    Mat dst;
//    Size dsize = Size(cols * 2, rows * 2);
    Size dsize;
//    resize(img, dst, dsize, CV_INTER_CUBIC);
    resize(img, dst, dsize, 0.5, 0.5, CV_INTER_CUBIC);
    imshow("origin", img);
    imshow("img", dst);
    waitKey(0);
}

void moveImage(){
    Mat img = imread("F:\\picture\\boy.png");
    Mat dst;
    Mat move_mat = Mat::zeros(2, 3, CV_32FC1);
    move_mat.at<float>(0, 0) = 1;
    move_mat.at<float>(0,2) = 100;
    move_mat.at<float>(1, 1) = 1;
    move_mat.at<float>(1, 2) = 50;
    Size dsize = Size(0, 0);
    warpAffine(img, dst, move_mat, dsize);
    imshow("move", dst);
    waitKey(0);
}

void rotateImage(){
    Mat img = imread("F:\\picture\\boy.png");
    Mat dst;
    Point center = Point(img.cols / 2, img.rows / 2);
    Mat mat = getRotationMatrix2D(center,45,1);
    Size dsize = Size(img.cols, img.rows);
    warpAffine(img, dst, mat, dsize);
    imshow("rotate", dst);
    waitKey(0);
}

void erode_dilateImage(){
    Mat img, ero, dil, element;
    img = imread("F:\\picture\\number.png");
    element = getStructuringElement(MORPH_RECT, Size(10, 10));
    erode(img, ero, element);
    dilate(img, dil, element, Point(-1, -1));
    imshow("origin", img);
    imshow("erode", ero);
    imshow("dilate", dil);
    waitKey(0);
}

void open_close(){
    Mat img1, img2, open, close,element;
    img1 = imread("F:\\picture\\boy.png");
    img2 = imread("F:\\picture\\number.png");
    element = getStructuringElement(MORPH_RECT, Size(10, 10));
    morphologyEx(img1, open, MORPH_OPEN, element);
    morphologyEx(img1, close, MORPH_CLOSE, element);
    imshow("origin", img1);
    imshow("open", open);
    imshow("close", close);
    waitKey(0);
}

void top_black_hat(){
    Mat img1, img2, tophat, blackhat,element;
    img1 = imread("F:\\picture\\butterfly.png");
    element = getStructuringElement(MORPH_RECT, Size(10, 10));
    morphologyEx(img1, tophat, MORPH_TOPHAT, element);
    morphologyEx(img1, blackhat, MORPH_BLACKHAT, element);
    imshow("origin", img1);
    imshow("tophat", tophat);
    imshow("blackhat", blackhat);
    waitKey(0);
}

void mean_blur(){
    Mat img, dst;
    img = imread("F:\\picture\\girl.png");
    blur(img, dst, Size(5, 5));
    imshow("origin", img);
    imshow("meanBlur", dst);
    waitKey(0);
}

void gaussian_blur(){
    Mat img,dst;
    img = imread("F:\\picture\\noise_25.png");
    GaussianBlur(img, dst, Size(5, 5), 5);
    imshow("origin", img);
    imshow("gaussianBlur", dst);
    waitKey(0);
}

void median_blur(){
    Mat img, dst;
    img = imread("F:\\picture\\girl.png");
    medianBlur(img, dst, 3);
    imshow("origin", img);
    imshow("medianBlur", dst);
    waitKey(0);
}

void calcHist(){
    Mat src, dst;
    src = imread("F:\\picture\\butterfly.png");

    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    int histSize = 256;
    float range[] = { 0,256 };
    const float *histRanges = { range };

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRanges, true, false);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRanges, true, false);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRanges, true, false);

    int hist_h = 400;
    int hist_w = 512;
    int bin_w = hist_w / histSize;
    Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());

    for (int i = 0; i < histSize; i++)
    {
        line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(b_hist.at<float>(i - 1))),
             Point((i)*bin_w, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, LINE_AA);
        line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(g_hist.at<float>(i - 1))),
             Point((i)*bin_w, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, LINE_AA);
        line(histImage, Point((i - 1)*bin_w, hist_h - cvRound(r_hist.at<float>(i - 1))),
             Point((i)*bin_w, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, LINE_AA);
    }

    imshow("histImage", histImage);
    waitKey(0);

}

void yanmo(){
    Mat img, src;
    img = imread("F:\\picture\\butterfly.png");
    Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    filter2D(img, src, img.depth(), kernel);
    imshow("origin", img);
    imshow("filter2D", src);
    waitKey(0);
}

void zhifangtujunhenghua(){
    Mat img, src;
    img = imread("F:\\picture\\diaosu.png",0);
    equalizeHist(img, src);
    imshow("origin", img);
    imshow("equalizeHist", src);
    waitKey(0);
}

void zishiyingjunhenghua(){
    Mat img, dst;
    img = imread("F:\\picture\\diaosu.png", 0);
    Ptr<CLAHE> ptr = createCLAHE(5.0, Size(8, 8));
    ptr->apply(img, dst);
    imshow("origin", img);
    imshow("zishiying", dst);
    waitKey(0);
}

void sobel(){
    Mat img, dst_x, dst_y, abs_x, abs_y, src, dst_x1, dst_y1, abs_x1, abs_y1, src1;
    int ddepth = CV_16S;
    img = imread("F:\\picture\\castle.png");
    cvtColor(img,img,CV_BGR2GRAY);
    Sobel(img, dst_x, ddepth, 1, 0, 3);
    convertScaleAbs(dst_x, abs_x);
    Sobel(img, dst_y, ddepth, 0, 1, 3);
    convertScaleAbs(dst_y, abs_y);
    addWeighted(abs_x, 0.5, abs_y, 0.5, 0, src);

    Scharr(img, dst_x1, ddepth, 1, 0, 3);
    convertScaleAbs(dst_x1, abs_x1);
    Scharr(img, dst_y1, ddepth, 0, 1, 3);
    convertScaleAbs(dst_y1, abs_y1);
    addWeighted(abs_x1, 0.5, abs_y1, 0.5, 0, src1);

    imshow("origin", img);
    imshow("sobel", src);
    imshow("scharr", src1);
    waitKey(0);

}

void laplacian(){
    Mat img, dst,abs;
    int ddepth = CV_16S;
    img = imread("F:\\picture\\horse.png", 0);
    Laplacian(img, dst, ddepth, 3);
    convertScaleAbs(dst, abs);
    imshow("laplacian", abs);
    waitKey(0);
}

void canny(){
    Mat img,dst;
    img = imread("F:\\picture\\horse.png");
    Canny(img, dst, 0, 60);
    imshow("origin", img);
    imshow("canny", dst);
    waitKey(0);
}

void match(){
    Mat img1, img2, result;
    double minVal,maxVal;
    Point minLoc, maxLoc;
    img1 = imread("F:\\picture\\na.png");
    img2 = imread("F:\\picture\\eye.png");
    matchTemplate(img1, img2, result, TM_SQDIFF);
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    cout << "==============>" << minVal << "==============>" << maxVal << "==============>" << minLoc << "==============>" << maxLoc << endl;
    Point pt1 = Point(minLoc.x, minLoc.y);
    Point pt2 = Point(minLoc.x + img2.cols, minLoc.y + img2.rows);
    rectangle(img1, pt1, pt2, Scalar(0, 255, 0), 2);
    imshow("matchTemplate", img1);
    waitKey(0);
}

void hough(){
    Mat img, gray, dst;
    img = imread("F:\\picture\\calendar.png");
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Canny(gray, dst, 0, 60, 3);
    vector<Vec4f> lines;
    HoughLinesP(dst, lines, 1.0, CV_PI / 180, 150, 0, 3);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4f planes = lines[i];
        line(img, Point(planes[0], planes[1]), Point(planes[2], planes[3]), Scalar(0, 0, 255), 2);
    }
    imshow("houghLines", img);
    waitKey(0);
}

void hough_circle(){
    Mat img, gray, dst;
    img = imread("F:\\picture\\circle.png");
    medianBlur(img, dst, 3);
    cvtColor(dst, gray, COLOR_BGR2GRAY);
    vector<Vec3f> planes;
    HoughCircles(gray, planes, CV_HOUGH_GRADIENT, 1, 100, 40, 100, 20, 1000);
    for (size_t i = 0; i < planes.size(); ++i) {
        Vec3f circles = planes[i];
        circle(img, Point(circles[0], circles[1]), circles[2], Scalar(0, 255, 0), 2);
    }
    imshow("hough_circle", img);
    waitKey(0);
}

void harris(){
    Mat img, grayImg, norImg, absImg;
    img = imread("F:\\picture\\check.png");
    Mat dstImg = Mat::zeros(img.size(), CV_32FC1);
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    cornerHarris(grayImg, dstImg, 2, 3, 0.04, BORDER_DEFAULT);
    normalize(dstImg, norImg, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(norImg, absImg);
    Mat resultImg = img.clone();
    for(int i = 0; i < resultImg.rows; i++){
        uchar *currentRow = absImg.ptr(i);
        for(int j = 0; j < resultImg.cols; j++){
            int value = (int) *currentRow;
            if(value > 125){
                circle(resultImg, Point(j,i), 2, Scalar(0, 255, 0), 2, 8, 0);
            }
            currentRow++;
        }
    }
    imshow("harris", resultImg);
    waitKey(0);
}

void shi_tomas(){
    Mat img, grayImg;
    img = imread("F:\\picture\\buildings.png");
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    vector<Point2f> corners;
    goodFeaturesToTrack(grayImg, corners, 1000, 0.05, 10);
    for (int i = 0; i < corners.size(); ++i) {
        circle(img, corners[i], 2, Scalar(0, 0, 255), 2);
    }
    imshow("shi-tomas", img);
    waitKey(0);
}

void sift(){
    Mat img, grayImg;
    img = imread("F:\\picture\\buildings.png");
    Ptr<SIFT> ptr = SIFT::create();
    vector<KeyPoint> keyPoints;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);
    ptr->detect(grayImg, keyPoints, Mat());
    drawKeypoints(img, keyPoints, img, Scalar(0, 0, 255),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    imshow("sift", img);
    waitKey(0);
}

void fast(){
    Mat img;
    img = imread("F:\\picture\\castle.png");
    Ptr<FastFeatureDetector> ptr = FastFeatureDetector::create(35);
    vector<KeyPoint> keyPoints;
    ptr->detect(img, keyPoints, Mat());
    drawKeypoints(img, keyPoints, img, Scalar(0, 0, 255));
    imshow("fast", img);
    waitKey(0);
}

void orb(){
    Mat img;
    img = imread("F:\\picture\\castle.png");
    Ptr<ORB> ptr = ORB::create(3000);
    vector<KeyPoint> keyPoints;
    ptr->detect(img, keyPoints, Mat());
    drawKeypoints(img, keyPoints, img, Scalar(0, 0, 255));
    imshow("orb", img);
    waitKey(0);
}

