#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <dirent.h>


using namespace std;
using namespace cv;


#define DATASET_PATH "data/GT/"
#define LEFT_DIR "data/left/"
#define RIGHT_DIR "data/right/"


void get_filenames(vector<string> &f_name);
void convert(const Mat m, Mat &a, Mat &b, int o, double r, double s);
void batch_convert(vector<string> f_name);




void get_filenames(vector<string> &f_name){
    DIR *dir; 
    struct dirent *dp;
    string s;

    dir = opendir(DATASET_PATH);
    while (dir != NULL) {
        if ((dp = readdir(dir)) != NULL) {
            s = string(dp->d_name);
            if (s != "." && s != ".."){
                s = DATASET_PATH + s;
                f_name.push_back(s);
            }
        } else {
            closedir(dir);
            break;
        }
    }
}


void batch_convert(vector<string> f_name){
    int o, c = 0;
    double r, s;
    Mat src, a, b;
    ofstream out;
    string info, id;
    out.open("data/truth_table.csv", std::ofstream::out);
    for (unsigned i = 0; i < f_name.size(); i++){
        src = imread(f_name[i], CV_LOAD_IMAGE_COLOR);
        if (src.cols != 1000 && src.rows != 600){
            cout << f_name[i] << ":" << src.cols << ":" << src.rows << endl;
            cout << "Wrong image dimensions" << endl;
            continue;
        }
        cout << "Processing: " << f_name[i] << " " << i+1 << "/" << f_name.size() << endl;
        id = f_name[i];
        id = id.erase(0, strlen(DATASET_PATH));
        id = id.erase(id.length()-4); 

        //apply different transformations
        for (r = -15; r <= 15; r += 1.5){
            for (o = 200; o <= 400; o += 50){
                for (s = 1.0; s >= 0.5; s -= 0.1){
                    convert(src, a, b, o, r, s);
                    info = id + ", " + to_string(o) + ", " + to_string(r) + ", " + to_string(s);
                    out << info << endl;
                    imwrite(LEFT_DIR + to_string(c) + ".png", a);
                    imwrite(RIGHT_DIR + to_string(c) + ".png", b);
                    c++;
                }
            }
        }
    }
    out.close();
}

void display(Mat a){
    namedWindow("test", 0);
    imshow("test", a);
    waitKey(0);
}


//o = overlap, r = rotation, s = scaling
void convert(const Mat m, Mat &a, Mat &b, int o, double r, double s){
    // overlap
    a = m(Rect(0, 0, 600, 600)).clone();
    b = m(Rect(600 - o, 0, 600, 600)).clone();
    // rotate
    Point c = Point(b.cols/2, b.rows/2);
    Mat r_m = getRotationMatrix2D(c, r, 1.0);
    warpAffine(b, b, r_m, b.size());
    // scale
    resize(b, b, Size(), s, s, CV_INTER_LINEAR);
}



int main (int argc, char** argv){
    vector<string> f_name;
    cout << "getting files..." << endl;
    get_filenames(f_name);
    cout << "creating data..." << endl;
    batch_convert(f_name);
    cout << "done!" << endl;
    return 0;
}
