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

public class EmguCVMarker : MonoBehaviour
{
    private VideoCapture capture;

    void Start()
    {
        capture = new VideoCapture(0);
        capture.ImageGrabbed += HandleGrab;
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

        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        VectorOfPoint approx = new VectorOfPoint();
        VectorOfVectorOfPoint candidates = new VectorOfVectorOfPoint();
        Mat hierarchy = new Mat();

        CvInvoke.FindContours(grayImg, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);

        for (int i = 0; i < contours.Size; i++)
        {
            CvInvoke.ApproxPolyDP(contours[i], approx, CvInvoke.ArcLength(contours[i], true) * 0.05, true);
            if (approx.Size == 4)
            {
                if (CvInvoke.ContourArea(contours[i]) > 300)
                {
                    var rect = CvInvoke.BoundingRectangle(approx);
                    if (rect.Height > 0.95 * rect.Width || rect.Height < 0.95 * rect.Width)
                    {
                        candidates.Push(approx);
                        CvInvoke.DrawContours(image, contours, i, new MCvScalar(0, 0, 255), 4);
                        CvInvoke.Rectangle(image, rect, new MCvScalar(0, 255, 0), 3);
                    }
                }
            }
        }

        for (int i = 0; i < candidates.Size; i++)
        {
            System.Drawing.PointF[] pts = new System.Drawing.PointF[4];
            pts[0] = new System.Drawing.PointF(0, 0);
            pts[1] = new System.Drawing.PointF(64 - 1, 0);
            pts[2] = new System.Drawing.PointF(64 - 1, 64 - 1);
            pts[3] = new System.Drawing.PointF(0, 64 - 1);
            VectorOfPointF perfect = new VectorOfPointF(pts);
            System.Drawing.PointF[] sample_pts = new System.Drawing.PointF[4];
            for (int j = 0; j < 4; j++)
                sample_pts[j] = new System.Drawing.PointF(candidates[i][j].X, candidates[i][j].Y);
            VectorOfPointF sample = new VectorOfPointF(sample_pts);

            var tf = CvInvoke.GetPerspectiveTransform(sample, perfect);

            Mat warped = new Mat();
            CvInvoke.WarpPerspective(image, warped, tf, new System.Drawing.Size(64, 64));

            CvInvoke.Imshow("Warped " + i, warped);
        }

        CvInvoke.Imshow("Original", image);
        CvInvoke.Imshow("Gray", grayImg);
    }

    private void OnDestroy()
    {
        capture.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}