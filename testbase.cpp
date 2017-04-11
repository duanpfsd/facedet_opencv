#include <iostream>
#include <fstream>
#include <io.h>
#include <string>
#include <vector>
#include <algorithm>
#include "conio.h"

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>

using namespace std;
using namespace cv;

String face_cascade_name = "lbpcascade_frontalface.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";

RNG rng(12345);

int main(int argc, char* argv[]) {
  if (!face_cascade.load(face_cascade_name)){ printf("--(!)1Error loading\n"); return -1; };
  if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)2Error loading\n"); return -1; };

  std::string path = "E:\\MyDownloads\\Download\\fktpxzq\\pic\\ztest\\testspeed\\";
  int showimg = 0;

  string pathName;
  long hFile = 0;
  struct _finddata_t fileInfo;
  int ncount = 0;

  std::ofstream infoout;
  infoout.open("./infoout/opencv_80.txt", ofstream::out);

  double t = cvGetTickCount();
  // Read picture files and store Face Information.
  if ((hFile = _findfirst(pathName.assign(path).append("/*").c_str(), &fileInfo)) == -1)
  {
    return -1;
  }
  while (_findnext(hFile, &fileInfo) == 0)
  {
    if (++ncount % 20 == 0)
    {
      cout << ncount << endl;
    }
    string imgname = pathName.assign(path).append(fileInfo.name).c_str();
    cv::Mat img_color = cv::imread(imgname, 1);
    if (img_color.empty())
    {
      //fprintf(stderr,"empty.\n");
      infoout << 0 << "  empty." << endl;
      continue;
    }
    cv::Mat img_gray;
    cv::cvtColor(img_color, img_gray, CV_BGR2GRAY);

    /*
      todo detect and landmark
      a list of int number is given, which indicate faces in each imagine.
    */


    std::vector<Rect> faces;
    equalizeHist(img_gray, img_gray);
    face_cascade.detectMultiScale(img_gray, faces, 1.1, 2, 0, Size(80, 80));

    infoout << faces.size() << "  good  ";
    cout << faces.size() << "  good  ";
    infoout << endl;
    cout << endl;

    if (showimg)
    {

      for (size_t i = 0; i < faces.size(); i++)
      {
        Mat faceROI = img_gray(faces[i]);
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
        if (eyes.size() == 2)
        {
          //-- Draw the face
          Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
          ellipse(img_color, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);

          for (size_t j = 0; j < eyes.size(); j++)
          { //-- Draw the eyes
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            circle(img_color, eye_center, radius, Scalar(255, 0, 255), 3, 8, 0);
          }
        }
      }
      cvWaitKey(0);
    }
  }
  t = cvGetTickCount() - t;
  cout << "Face detection and landmark consuming:" << t / (cvGetTickFrequency() * 1000) << "ms" << endl;
  infoout.close();
  _getch();

  return 0;
}


