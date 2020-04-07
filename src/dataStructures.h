#ifndef dataStructures_h
#define dataStructures_h

#include <opencv2/core.hpp>
#include <vector>

struct DataFrame { // represents the available sensor information at the same
                   // time instance

  cv::Mat cameraImg; // camera image

  std::vector<cv::KeyPoint> keypoints; // 2D keypoints within camera image
  cv::Mat descriptors;                 // keypoint descriptors
  std::vector<cv::DMatch>
      kptMatches; // keypoint matches between previous and current frame
};

struct MatchingTestParameters {
  bool focusOnVehicle;
  bool visualizeResults;
  bool limitKeyPoints;
  std::string dataPath;
  /**
   * @brief Options: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
   */
  std::string keypointDetectorType;
  /**
   * @brief Options: BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
   */
  std::string keypointDescriptorType;
  /**
   * @brief Options: BF, FLANN
   */
  std::string matcherType;

  /**
   * @brief Options: NN, KNN
   */
  std::string selectorType;
  MatchingTestParameters()
      : focusOnVehicle(true), visualizeResults(false), limitKeyPoints(false),
        dataPath(DATA_ROOT),
        keypointDetectorType("BRISK"), keypointDescriptorType("BRISK"),
        matcherType("BF"), selectorType("KNN") {}
};

struct MatchingTestResults {
  int testCase;
  int imageNumber;
  double detectorDurationMs;
  double descriptorDurationMs;
  double matchingDurationMs;
  DataFrame frame;
  MatchingTestResults()
      : testCase(0), imageNumber(0), detectorDurationMs(0.0),
        descriptorDurationMs(0.0), matchingDurationMs(0.0),
        frame() {}
};

#endif /* dataStructures_h */
