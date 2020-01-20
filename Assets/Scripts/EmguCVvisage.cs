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

public class EmguCVvisage : MonoBehaviour
{
    private VideoCapture capture;
    private CascadeClassifier classifier;
    private Rectangle[] frontFaces;

    [Range(10, 500)]
    public int MIN_FACE_SIZE = 10;
    [Range(10, 500)]
    public int MAX_FACE_SIZE = 500;

    void Start()
    {
        capture = new VideoCapture(0);
        capture.ImageGrabbed += HandleGrab;
        classifier = new CascadeClassifier("Assets/Resources/haarcascade_frontalface_default.xml");
    }

    void Update()
    {
        if (capture.IsOpened) { capture.Grab(); }
    }

    private void HandleGrab(object sender, EventArgs e)
    {
        Mat img = new Mat();

        if (capture.IsOpened) capture.Retrieve(img);
        if (img.IsEmpty) return;

        Mat grayImg = img.Clone();
        CvInvoke.CvtColor(img, grayImg, ColorConversion.Bgr2Gray);
        frontFaces = classifier.DetectMultiScale(img, 1.1, 5, new Size(MIN_FACE_SIZE, MIN_FACE_SIZE), new Size(MAX_FACE_SIZE, MAX_FACE_SIZE));

        foreach (Rectangle rec in frontFaces)
        {
            CvInvoke.Rectangle(img, rec, new MCvScalar(0, 0, 255), 2);
        }


        CvInvoke.Imshow("Orignal", img);
    }

    private void OnDestroy()
    {
        capture.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}
