using System.Collections;
using System.Collections.Generic;
using System;
using System.Drawing;
using System.IO;
using UnityEngine;
using Emgu.CV;
using Emgu.CV.Aruco;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

public class ArucoMarkers : MonoBehaviour
{
    private VideoCapture capture;
    private Dictionary dico;
    private GridBoard grid;
    private DetectorParameters arucoParam = new DetectorParameters();

    void Start()
    {
        capture = new VideoCapture(0);
        capture.ImageGrabbed += HandleGrab;
        dico = new Dictionary(Dictionary.PredefinedDictionaryName.Dict4X4_50);
        grid = new GridBoard(4, 4, 80, 30, dico);
        arucoParam = DetectorParameters.GetDefault();
    }

    void Update()
    {
        if (capture.IsOpened) { capture.Grab(); }
    }

    private void HandleGrab(object sender, EventArgs e)
    {
        Mat image = new Mat();
        if (capture.IsOpened) capture.Retrieve(image);
        if (image.IsEmpty) return;
        Mat grayImg = image.Clone();

        CvInvoke.CvtColor(image, grayImg, ColorConversion.Bgr2Gray);
        CvInvoke.AdaptiveThreshold(grayImg, grayImg, 255, AdaptiveThresholdType.MeanC, ThresholdType.BinaryInv, 21, 11);

        VectorOfInt ids = new VectorOfInt();
        VectorOfVectorOfPointF corners = new VectorOfVectorOfPointF();
        VectorOfVectorOfPointF rejected = new VectorOfVectorOfPointF();
        ArucoInvoke.DetectMarkers(image, dico, corners, ids, arucoParam, rejected);

        if (ids.Size > 0) ArucoInvoke.DrawDetectedMarkers(image, corners, ids, new MCvScalar(0, 0, 255));

        CvInvoke.Imshow("Original", image);
        CvInvoke.Imshow("Gray", grayImg);
    }

    private void OnDestroy()
    {
        capture.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}