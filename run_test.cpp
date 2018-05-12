#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <sys/types.h>
#include <dirent.h>
#include <fstream>
#include <algorithm>
#include <limits>
#include <vector>
#include <iostream>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


#define elif else if

#define LEFT_DIR "data/left/"
#define RIGHT_DIR "data/right/"
#define ORIG_DIR "data/GT/"
#define TRUTH_FILE "data/truth_table.csv"


struct GT{
    int id;
    int o;
    double r;
    double s;
};


struct entry{
    string l;
    string r;
    string orig;
};


size_t EXTRACTION_METHOD;


//prototypes
void display(Mat src);
void open_data(vector<entry> &data, vector<GT> truth);
void stitch(Mat& im_0, const Mat& im_1);
void load_truth(vector<GT> &truth);
void batch_test(int n);





void open_data(vector<entry> &data, vector<GT> truth){
    DIR *dir;
    vector<string> l, r;
    struct dirent *dp;
    string s;

    //left images
    dir = opendir(LEFT_DIR);
    while (dir != NULL){
        if ((dp = readdir(dir)) != NULL){
            s = string(dp->d_name);
            if (s != "." && s != ".."){
                s = LEFT_DIR + s;
                l.push_back(s);
            }
        } else {
            closedir(dir);
            break;
        }
    }
    //right images
    dir = opendir(RIGHT_DIR);
    while (dir != NULL){
        if ((dp = readdir(dir)) != NULL){
            s = string(dp->d_name);
            if (s != "." && s != ".."){
                s = RIGHT_DIR + s;
                r.push_back(s);
            }
        } else {
            closedir(dir);
            break;
        }
    }
    struct entry e;
    if (l.size() == r.size()){
        for (size_t i = 0; i < l.size(); i++){
            e.l = l[i];
            e.r = r[i];
            e.orig = ORIG_DIR + to_string(truth[i].id) + ".png";
            data.push_back(e);
        }
    } else {
        cout << "Data corrupt!\nExiting..." << endl; 
        exit(1);
    }
}


void stitch(Mat& im_0, const Mat& im_1) {
    //perform translate on im_1 and paste into larger frame
    int offsetx = 0;
    int offsety = 0;
    Mat T = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
    warpAffine(im_0, im_0, T, Size(2 * im_0.cols, 1 * im_0.rows));

    Ptr<Feature2D> f2d;
    if (EXTRACTION_METHOD == 0){
        f2d = BRISK::create();
    } elif (EXTRACTION_METHOD == 1) {
        f2d = MSER::create();
    } elif (EXTRACTION_METHOD == 2) {
        f2d = FREAK::create();
    } elif (EXTRACTION_METHOD == 3) {
        f2d = SIFT::create();
    } elif (EXTRACTION_METHOD == 4) {
        f2d = SURF::create();
    } elif (EXTRACTION_METHOD == 5) {
        f2d = ORB::create();
    } elif (EXTRACTION_METHOD == 6) {
        f2d = KAZE::create();
    } else {
        cout << "FATAL ERROR" << endl;
        exit(0);
    }
	
	vector<KeyPoint> keypoints_0, keypoints_1;
	f2d->detect(im_0, keypoints_0);
	f2d->detect(im_1, keypoints_1);

	Ptr<DescriptorExtractor> extractor;
	Mat descriptors_0, descriptors_1;

	f2d->compute(im_0, keypoints_0, descriptors_0);
	f2d->compute(im_1, keypoints_1, descriptors_1);

	BFMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors_0, descriptors_1, matches);


	Mat index;
	int nbMatch = int(matches.size());
	Mat tab(nbMatch, 1, CV_32F);
	for (size_t i = 0; i < nbMatch; i++)
		tab.at<float>(i, 0) = matches[i].distance;
	sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
	vector<DMatch> bestMatches;

	for (int i = 0; i < min(200, nbMatch); i++)
		bestMatches.push_back(matches[index.at<int>(i, 0)]);


	vector<Point2f> dst_pts, src_pts;                   

    for (size_t i = 0; i < bestMatches.size(); i++){
		dst_pts.push_back(keypoints_0[bestMatches[i].queryIdx].pt);
		src_pts.push_back(keypoints_1[bestMatches[i].trainIdx].pt);
	}

	Mat img_matches;
	drawMatches(im_0, keypoints_0, im_1, keypoints_1,
	            bestMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
	            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	
    imwrite("out.png", img_matches);
	//display(img_matches);

	Mat H = findHomography(src_pts, dst_pts, CV_RANSAC);

	Mat wim_1;
	warpPerspective(im_1, wim_1, H, im_0.size());

    //copy pixels over
	for (int i = 0; i < im_0.cols; i++){
		for (int j = 0; j < im_0.rows; j++) {
			Vec3b color_im0 = im_0.at<Vec3b>(Point(i, j));
			Vec3b color_im1 = wim_1.at<Vec3b>(Point(i, j));
			if (norm(color_im0) == 0)
				im_0.at<Vec3b>(Point(i, j)) = color_im1;

		}
    }
    imwrite("res.png", im_0);
}


void load_truth(vector<GT> &truth){
    ifstream in;
    in.open(TRUTH_FILE);
    string line, word;
    stringstream ss;
    struct GT t;
    while (getline(in, line)){
        ss = stringstream(line);
        getline(ss, word, ',');
        t.id = stoi(word);
        getline(ss, word, ',');
        t.o = stoi(word);
        getline(ss, word, ',');
        t.r = stof(word);
        getline(ss, word, '\n');
        t.s = stof(word);
        truth.push_back(t);
    }
}

void display(Mat src){
    namedWindow("test", 0);
    imshow("test", src);
    waitKey(0);
}


double evaluate(Mat a, Mat b, GT t){
    //steps:
    //isolate specific region from righthand side of Mat b
    //transform according to the ground truth
    //check similarity with corresponding patch from mat A
    //using matchTemplate function
	a = a(Rect(0, 0, 1000, 600));
    if ( a.rows > 0 && a.rows == b.rows && a.cols > 0 && a.cols == b.cols ) {
		// Calculate the L2 relative error between images.
		double errorL2 = norm(a, b, CV_L2 );
		// Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
		double sim = errorL2 / (double)( a.rows * a.cols );
		return sim;
	}
	else {
		//Images have a different size
		return 100000000.0;  // Return a bad value
	}

}



void batch_test(int n){
    vector<GT> truth;
    vector<entry> data;
    load_truth(truth);
    open_data(data, truth);

    Mat im_0, im_1, im_2;
    double err = 0.0, total_err = 0.0, max_err = 0.0;

    for (int i = 0; i < min((int)data.size(), n); i++){
		cout << i+1 << "/" << min((int)data.size(), n) << endl;
        im_0 = imread(data[i].l, CV_LOAD_IMAGE_COLOR);
        im_1 = imread(data[i].r, CV_LOAD_IMAGE_COLOR);
        im_2 = imread(data[i].orig, CV_LOAD_IMAGE_COLOR);
        stitch(im_0, im_1);
        //imwrite("data/res/" + to_string(i) + ".png", im_0);
        err = evaluate(im_0, im_2, truth[i]);
        total_err += err;
        max_err = max(err, max_err);
    }
	cout << "total error is: " << total_err << endl;

}


int main (int argc, char **argv){
    if (argc != 3){
        cout << "Specify method of evaluation!" << endl;
    } else {
        EXTRACTION_METHOD = (size_t)stoi(argv[1]);
        int n = stoi(argv[2]);
        batch_test(n);
    }
    return 0;
}
