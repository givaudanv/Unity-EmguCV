using System.Collections;
using System.Collections.Generic;
using System;
using System.Drawing;
using System.IO;
using UnityEngine;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public class EmguCVTest : MonoBehaviour
{
    [Range(0, 179)]
    public double lowerHue = 0;
    [Range(0, 179)]
    public double upperHue = 179;
    [Range(0, 255)]
    public double lowerIntensity = 0;
    [Range(0, 255)]
    public double upperIntensity = 255;
    [Range(0, 255)]
    public double lowerValue = 0;
    [Range(0, 255)]
    public double upperValue = 255;


    private Hsv lowerHSV;
    private Hsv upperHSV;

    private VideoCapture capture;

    void Start()
    {
        capture = new VideoCapture(0);
    }

    void Update()
    {
        lowerHSV = new Hsv(lowerHue, lowerIntensity, lowerValue);
        upperHSV = new Hsv(upperHue, upperIntensity, upperValue);

        Mat image;
        image = capture.QueryFrame();
        Mat hsvImg = image.Clone();

        CvInvoke.CvtColor(image, hsvImg, ColorConversion.Bgr2Hsv);
        Image<Hsv, byte> hsvConverted = hsvImg.ToImage<Hsv, byte>();
        Image<Gray, byte> hsvThreshold = hsvConverted.InRange(lowerHSV, upperHSV);

        Image<Gray, byte> eroded = hsvThreshold.Clone();
        Image<Gray, byte> dilated = hsvThreshold.Clone();
        Mat structuringElement = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(-1, -1));
        CvInvoke.Erode(hsvThreshold, eroded, structuringElement, new Point(-1, -1), 2, BorderType.Constant, new MCvScalar(0));
        CvInvoke.Dilate(eroded, dilated, structuringElement, new Point(-1, -1), 2, BorderType.Constant, new MCvScalar(0));

        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        VectorOfPoint biggestContour = new VectorOfPoint();
        int biggestContourIndex = -1;
        double biggestContourArea = 0;
        Mat hierarchy = new Mat();
        Mat contourImg = image.Clone();

        CvInvoke.FindContours(dilated, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxNone);

        for (int i = 0; i < contours.Size; i++)
        {
            if (CvInvoke.ContourArea(contours[i]) > biggestContourArea)
            {
                biggestContour = contours[i];
                biggestContourIndex = i;
                biggestContourArea = CvInvoke.ContourArea(contours[i]);
            }
        }

        CvInvoke.DrawContours(contourImg, contours, -1, new MCvScalar(0, 255, 0), 2);
        if (biggestContourIndex > -1)
        {
            CvInvoke.DrawContours(contourImg, contours, biggestContourIndex, new MCvScalar(0, 0, 255), 2);

            Moments moments = CvInvoke.Moments(contours[biggestContourIndex]);
            Point centroid = new Point((int)(moments.M10 / moments.M00), (int)(moments.M01 / moments.M00));
            CvInvoke.Circle(contourImg, centroid, 8, new MCvScalar(0, 0, 255), -1);
        }

        CvInvoke.Imshow("Original", image);
        CvInvoke.Imshow("Contour", contourImg);
        CvInvoke.Imshow("Seuillage", hsvThreshold);
        CvInvoke.Imshow("Ouverture", dilated);
        CvInvoke.WaitKey(24);
    }

    private void OnDestroy()
    {
        capture.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}