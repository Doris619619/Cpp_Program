// roi_annotator.cpp
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

using namespace cv; using namespace std;

Mat img, img_copy;
vector<Rect> rois;
Point startPt;
bool drawing = false;

void onMouse(int event, int x, int y, int, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        drawing = true;
        startPt = Point(x, y);
    }
    else if (event == EVENT_MOUSEMOVE && drawing) {
        img_copy = img.clone();
        rectangle(img_copy, startPt, Point(x, y), Scalar(0, 255, 0), 2);
        imshow("roi annotator", img_copy);
    }
    else if (event == EVENT_LBUTTONUP && drawing) {
        drawing = false;
        Rect r = Rect(startPt, Point(x, y));
        // 规范化
        r = Rect(min(r.x, r.x + r.width), min(r.y, r.y + r.height), abs(r.width), abs(r.height));
        if (r.width > 5 && r.height > 5) {
            rois.push_back(r);
            rectangle(img, r, Scalar(0, 255, 0), 2);
            imshow("roi annotator", img);
        }
    }
}
int main()
{
    ///此处改变截图提取的路径
    string imgPath = "D:\\3002code\\c++\\3002Project1\\3002Project1\\screenshots\\screenshot_20.0s_600.jpg";

    img = imread(imgPath);
    if (img.empty()) {
        cerr << "无法打开图片: " << imgPath << endl;
        return -1;
    }

    img_copy = img.clone();
    namedWindow("roi annotator", WINDOW_AUTOSIZE);
    setMouseCallback("roi annotator", onMouse, 0);
    imshow("roi annotator", img);

    cout << "Instructions:\n - Drag mouse to draw ROI\n - Press 's' to save ROIs to rois.csv\n - Press 'q' to quit without saving\n";
    while (true) {
        char c = (char)waitKey(0);
        if (c == 's') {
            ofstream fout("rois.csv");
            for (int i = 0; i < (int)rois.size(); ++i) {
                Rect r = rois[i];
                fout << (i + 1) << "," << r.x << "," << r.y << "," << r.width << "," << r.height << "\n";
            }
            fout.close();
            cout << "Saved " << rois.size() << " ROIs to rois.csv\n";
            break;
        }
        else if (c == 'q' || c == 27) {
            cout << "Exit without saving\n";
            break;
        }
    }
    destroyAllWindows();
    return 0;
}
