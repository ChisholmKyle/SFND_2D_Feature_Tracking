/* INCLUDES FOR THIS PROJECT */
#include <ctime>
#include <deque>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

using namespace std;

static void WriteHeaderToCsv(std::ofstream &csvFs)
{
    csvFs << "'Case'";
    csvFs << ","
          << "'Image'";

    csvFs << ","
          << "'Detector'";
    csvFs << ","
          << "'Descriptor'";
    csvFs << ","
          << "'Matcher'";
    csvFs << ","
          << "'Selector'";

    csvFs << ","
          << "'Detector Duration (ms)'";
    csvFs << ","
          << "'Descriptor Duration (ms)'";
    csvFs << ","
          << "'Matching Duration (ms)'";

    csvFs << ","
          << "'Keypoints'";
    csvFs << ","
          << "'Mean Size'";
    csvFs << ","
          << "'StdDev Size'";

    csvFs << ","
          << "'Matches'";

    csvFs << endl;
}

static void AppendToCsv(const MatchingTestResults &results, const MatchingTestParameters &parameters, std::ofstream &csvFs)
{
    csvFs << results.testCase;
    csvFs << "," << results.imageNumber;

    csvFs << "," << parameters.keypointDetectorType;
    csvFs << "," << parameters.keypointDescriptorType;
    csvFs << "," << parameters.matcherType;
    csvFs << "," << parameters.selectorType;

    csvFs << "," << results.detectorDurationMs;
    csvFs << "," << results.descriptorDurationMs;
    csvFs << "," << results.matchingDurationMs;

    csvFs << "," << results.frame.keypoints.size();

    if (!results.frame.keypoints.empty())
    {

        std::vector<double> keyPointSizes;
        for (const auto &pnt : results.frame.keypoints)
        {
            keyPointSizes.push_back(pnt.size);
        }
        std::vector<double> keyPointMean;
        std::vector<double> keyPointStdDev;
        cv::meanStdDev(keyPointSizes, keyPointMean, keyPointStdDev);

        csvFs << "," << keyPointMean[0];
        csvFs << "," << keyPointStdDev[0];
    }
    else
    {
        csvFs << "," << -1;
        csvFs << "," << -1;
    }

    csvFs << "," << results.frame.kptMatches.size();

    csvFs << endl;
}

static void GenerateTest(const MatchingTestParameters &parameters, const int testCase, std::ofstream &csvFs)
{

    // camera
    string imgBasePath = parameters.dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;           // no. of images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {

        // results
        MatchingTestResults result;
        result.testCase = testCase;
        result.imageNumber = imgIndex;

        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.push_back(frame);
        while (dataBuffer.size() > dataBufferSize)
        {
            dataBuffer.pop_front();
        }

        //// EOF STUDENT ASSIGNMENT
        // cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        string detectorType = parameters.keypointDetectorType;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, result.detectorDurationMs, parameters.visualizeResults);
        }
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray, result.detectorDurationMs, parameters.visualizeResults);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, result.detectorDurationMs, parameters.visualizeResults);
        }
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (parameters.focusOnVehicle)
        {
            std::vector<cv::KeyPoint> filteredKeypoints;
            for (const auto &key : keypoints)
            {
                if (vehicleRect.contains(key.pt))
                {
                    filteredKeypoints.push_back(key);
                }
            }
            std::swap(keypoints, filteredKeypoints);
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        if (parameters.limitKeyPoints)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT
        //// -> SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = parameters.keypointDescriptorType;

        try
        {
            descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType, result.descriptorDurationMs);
        }
        catch( cv::Exception& e )
        {
            const char* err_msg = e.what();
            cout << std::endl << "{ERROR} DETECTOR: " << detectorType << ", DESCRIPTOR: " << descriptorType << std::endl;
            std::cout << "exception caught: " << err_msg << std::endl;
            return;
        }
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        // cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_" + parameters.matcherType;   // MAT_BF, MAT_FLANN
            string selectorType = "SEL_" + parameters.selectorType; // SEL_NN, SEL_KNN

            string matchDescriptorType = "DES_BINARY"; // DES_BINARY, DES_HOG
            if (descriptorType.compare("SIFT") == 0)
            {
                matchDescriptorType = "DES_HOG";
            }

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, matchDescriptorType, matcherType, selectorType, result.matchingDurationMs);

            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            // cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            if (parameters.visualizeResults)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }

            // append to CSV
            result.frame = *(dataBuffer.end() - 1);
            AppendToCsv(result, parameters, csvFs);
        }

    } // eof loop over all images
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES */

    // test data tables
    std::vector<std::string> keypointDetectorList =
        {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    std::vector<std::string> keypointDescriptorList =
        {"BRISK", "BRIEF", "ORB", "FREAK", "SIFT"};

    std::vector<MatchingTestParameters> testParameters;
    for (const auto &detectorType : keypointDetectorList)
    {
        for (const auto &descriptorType : keypointDescriptorList)
        {
            MatchingTestParameters testParameter;
            testParameter.keypointDetectorType = detectorType;
            testParameter.keypointDescriptorType = descriptorType;
            // add to list
            testParameters.push_back(testParameter);
        }
    }
    // AKAZE descriptor requires AKAZE detector
    MatchingTestParameters testParameter;
    testParameter.keypointDetectorType = "AKAZE";
    testParameter.keypointDescriptorType = "AKAZE";
    testParameters.push_back(testParameter);

    // timestamp for csv filename
    time_t t = std::time(0);
    struct tm *now = std::localtime(&t);
    char timeString[80];
    std::strftime(timeString, 80, "%Y-%m-%d_%Hh%mm%Ss", now);

    // open CSV file
    std::ofstream csvFs;
    std::string csvFilePath = "output_" + std::string(timeString) + ".csv";
    csvFs.open(csvFilePath);
    WriteHeaderToCsv(csvFs);

    // generate results
    for (int testCase = 0; testCase < testParameters.size(); ++testCase)
    {
        GenerateTest(testParameters[testCase], testCase, csvFs);
    }

    csvFs.close();
    return 0;
}
