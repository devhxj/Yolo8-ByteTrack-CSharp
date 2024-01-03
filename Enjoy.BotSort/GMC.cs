using NumpyDotNet;
using OpenCvSharp;
using OpenCvSharp.Features2D;
using System.Diagnostics;

namespace Enjoy.BotSort;

public class GMC()
{
    string? _method;
    double _downscale;
    bool _initializedFirstFrame = false;
    Feature2D? _detector;
    Feature2D? _extractor;
    DescriptorMatcher? _matcher;
    int _numberOfIterations;
    double _terminationEps;
    MotionTypes _warpMode;
    TermCriteria _criteria;
    Feature _features = new();
    Mat _prevFrame = new();
    Array? _prevKeyPoints ;
    Mat _prevDescriptors = new();
    StreamReader? _gmcFile;

    public GMC(string method = "sparseOptFlow", double downscale = 2, object[]? verbose = null) : this()
    {
        
        _method = method;
        _downscale = Math.Max(1, downscale);

        if (_method == "orb")
        {
            _detector = FastFeatureDetector.Create(20);
            _extractor = ORB.Create();
            _matcher = BFMatcher.Create("BruteForce-Hamming");
        }
        else if (_method == "sift")
        {
            _detector = SIFT.Create(nOctaveLayers: 3, contrastThreshold: 0.02, edgeThreshold: 20);
            _extractor = SIFT.Create(nOctaveLayers: 3, contrastThreshold: 0.02, edgeThreshold: 20);
            _matcher = BFMatcher.Create("BruteForce");
        }
        else if (_method == "ecc")
        {
            _numberOfIterations = 5000;
            _terminationEps = 1e-6;
            _warpMode = MotionTypes.Euclidean;
            _criteria = new(CriteriaTypes.Eps | CriteriaTypes.Count, _numberOfIterations, _terminationEps);
        }
        else if (_method == "sparseOptFlow")
        {
            _features = new(
                MaxCorners: 1000,
                QualityLevel: 0.01,
                MinDistance: 1,
                BlockSize: 3,
                UseHarrisDetector: false,
                K: 0.04);

            // gmc_file = open("GMC_results.txt", "w")
        }
        else if (_method == "file" || _method == "files")
        {
            var seqName = verbose?.Length > 1 ? verbose[0].ToString() : null;
            var ablation = verbose?.Length > 2 ? verbose[1] : null;
            var filePath = ablation?.Equals(true) == true
                ? "tracker/GMC_files/MOT17_ablation"
                : "tracker/GMC_files/MOTChallenge";

            if (string.IsNullOrWhiteSpace(seqName))
            {

            }
            else if (seqName.Contains("-FRCNN"))
            {
                seqName = seqName[..^6];
            }
            else if (seqName.Contains("-DPM") || seqName.Contains("-SDP"))
            {
                seqName = seqName[..^4];
            }

            _gmcFile = new StreamReader($@"{filePath}/GMC-{seqName}.txt");

            if (_gmcFile is null)
            {
                throw new Exception($"Error: Unable to open GMC file in directory:{filePath}");
            }
            else if (_method.Equals("none", StringComparison.CurrentCultureIgnoreCase))
            {
                _method = "none";
            }
            else
            {
                throw new Exception($"Error: Unknown CMC method:{method}");
            }
        }
    }

    /// <summary>
    /// Apply object detection on a raw frame using specified method.
    /// </summary>
    /// <param name="rawFrame"></param>
    /// <param name="detections"></param>
    public Mat Apply(Mat rawFrame, IEnumerable<Rect>? detections = null)
    {
        if (_method == "orb" || _method == "sift")
            return ApplyFeatures(rawFrame, detections);

        else if (_method == "ecc")
            return ApplyEcc(rawFrame, detections);

        else if (_method == "sparseOptFlow")
            return ApplySparseOptFlow(rawFrame, detections);

        else if (_method == "file")
            return ApplyFile(rawFrame, detections);

        else if (_method == "none")
            return Mat.Eye(2, 3, MatType.CV_64FC1);

        else
            return Mat.Eye(2, 3, MatType.CV_64FC1);
    }

    /// <summary>
    /// Initialize.
    /// </summary>
    /// <param name="rawFrame"></param>
    /// <param name="detections"></param>
    /// <returns></returns>
    Mat ApplyEcc(Mat rawFrame, IEnumerable<Rect>? detections = null)
    {
        using Mat frame = new();
        var (height, width) = (rawFrame.Height, rawFrame.Width);
        Cv2.CvtColor(rawFrame, frame, ColorConversionCodes.BGR2GRAY);
        using var H = Mat.Eye(2,3, MatType.CV_32FC1).ToMat();

        // Downscale image (TODO: consider using pyramids)
        if (_downscale > 1.0)
        {
            width = (int)Math.Floor(width / _downscale);
            height = (int)Math.Floor(height / _downscale);
            Cv2.GaussianBlur(frame, frame, new(3, 3), 1.5);
            Cv2.Resize(frame, frame, new(width, height));
        }

        // Handle first frame
        if (!_initializedFirstFrame)
        {
            // Initialize data
            frame.CopyTo(_prevFrame);
            // Initialization done
            _initializedFirstFrame = true;
            return H;
        }

        // Run the ECC algorithm. The results are stored in warp_matrix.
        // (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria)
        try
        {
            var cc = Cv2.FindTransformECC(_prevFrame!, frame, H, _warpMode, _criteria, null, 1);
        }
        catch (Exception ex)
        {
            Trace.TraceWarning($"WARNING: find transform failed. Set warp as identity {ex.Message}");
        }

        return H;
    }

    /// <summary>
    /// Initialize.
    /// </summary>
    /// <param name="rawFrame"></param>
    /// <param name="detections"></param>
    /// <returns></returns>
    Mat ApplyFeatures(Mat rawFrame,IEnumerable<Rect>? detections = null)
    {
        if (_detector is null)
        {
            throw new NullReferenceException(nameof(_detector));
        }

        using Mat frame = new();
        var (height, width) = (rawFrame.Height, rawFrame.Width);
        Cv2.CvtColor(rawFrame, frame, ColorConversionCodes.BGR2GRAY);
        var H = Mat.Eye(2, 3, MatType.CV_64FC1).ToMat();

        // Downscale image (TODO: consider using pyramids)
        if (_downscale > 1.0)
        {
            width = (int)Math.Floor(width / _downscale);
            height = (int)Math.Floor(height / _downscale);
            //frame = cv2.gaussianBlur(frame, new(3, 3), 1.5);
            Cv2.Resize(frame, frame, new(width, height));
        }

        // Find the keypoints
        using var mask =  Mat.Zeros(rawFrame.Rows, rawFrame.Cols, rawFrame.Type()).ToMat();
        mask[(int)(0.02 * height), (int)(0.98 * height), (int)(0.02 * width), (int)(0.98 * width)] += 255;
        if (detections is not null) 
        {
            foreach (var det in detections)
            {
                var tlbr = new int[]{
                    (int)(det.Top / _downscale),
                    (int)(det.Left / _downscale),
                    (int)(det.Bottom / _downscale),
                    (int)(det.Right / _downscale)
                };
                mask[tlbr[1], tlbr[3], tlbr[0], tlbr[2]] *= 0;
            }
        }

        var keypoints = _detector.Detect(frame, mask);
        // Compute the descriptors
        using var descriptors = new Mat();
        _extractor?.Compute(frame, ref keypoints!, descriptors);

        // Handle first frame
        if (!_initializedFirstFrame)
        {
            // Initialize data
            frame.CopyTo(_prevFrame);
            if (keypoints.Length > 0)
            {
                _prevKeyPoints = new KeyPoint[keypoints.Length];
                keypoints.CopyTo(_prevKeyPoints, 0);
            }
            _prevDescriptors = descriptors.Clone();
            // Initialization done
            _initializedFirstFrame = true;
            return H;
        }

        // Match descriptors.
        var knnMatches = _matcher?.KnnMatch(_prevDescriptors, descriptors, 2);

        // Handle empty matches case
        if (knnMatches is null || knnMatches.Length == 0)
        {
            // Store to next iteration
            frame.CopyTo(_prevFrame);
            _prevKeyPoints = new KeyPoint[keypoints.Length];
            keypoints.CopyTo(_prevKeyPoints, 0);
            _prevDescriptors = descriptors.Clone();
            return H;
        }

        var maxSpatialDistance = new double[] { 0.25 * width, 0.25 * height };

        // Filtered matches based on smallest spatial distance
        List<DMatch> matches = [];
        var spatialDistances = np.empty((knnMatches.Length, 2), np.Float32);

        var prevKeyPoints = _prevKeyPoints as KeyPoint[] ?? Array.Empty<KeyPoint>();
        for (var index = 0; index < knnMatches.Length; index++)
        {
            var m = knnMatches[index][0];
            var n = knnMatches[index][1];
            if (m.Distance < 0.9 * n.Distance)
            {
                var prevKeyPointLocation = prevKeyPoints[m.QueryIdx].Pt;
                var currKeyPointLocation = keypoints[m.TrainIdx].Pt;
                var spatialDistance = (prevKeyPointLocation.X - currKeyPointLocation.X,
                                       prevKeyPointLocation.Y - currKeyPointLocation.Y);

                if (Math.Abs(spatialDistance.Item1) < maxSpatialDistance[0] && Math.Abs(spatialDistance.Item2) < maxSpatialDistance[1])
                {
                    spatialDistances[0] = spatialDistance.AsArray<float>();
                    matches.Add(m);
                }
            }
        }

        var meanSpatialDistances = np.mean(spatialDistances, 0);
        var stdSpatialDistances = np.std(spatialDistances, 0);
        var inliers = (spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances;

        List<Point2f> prevPoints = [];
        List<Point2f> currPoints = [];

        for (var index = 0; index < matches.Count; index++)
        {
            if ((bool)inliers[index, 0] && (bool)inliers[index, 1])
            {
                prevPoints.Add(prevKeyPoints[matches[index].QueryIdx].Pt);
                currPoints.Add(keypoints[matches[index].TrainIdx].Pt);
            }
        }

        // Find rigid matrix
        if (prevPoints.Count > 4 && prevPoints.Count == prevPoints.Count)
        {
            H = Cv2.EstimateAffinePartial2D(InputArray.Create(prevPoints),
                                            InputArray.Create(currPoints),
                                            null,
                                            RobustEstimationAlgorithms.RANSAC);
            if (H is null) throw new NullReferenceException(nameof(H));
            // Handle downscale
            if (_downscale > 1.0)
            {
                H.Set(0, 2, H.At<double>(0, 2) * _downscale);
                H.Set(1, 2, H.At<double>(1, 2) * _downscale);
            }
        }

        else Trace.TraceWarning("WARNING: not enough matching points");

        // Store to next iteration
        if (keypoints.Length > 0)
        {
            _prevKeyPoints = new KeyPoint[keypoints.Length];
            keypoints.CopyTo(_prevKeyPoints, 0);
        }
        _prevFrame = frame.Clone();
        _prevDescriptors = descriptors.Clone();

        return H;
    }

    /// <summary>
    /// Initialize.
    /// <param name="rawFrame"></param>
    /// <param name="detections"></param>
    /// <returns></returns>
    Mat ApplySparseOptFlow(Mat rawFrame, IEnumerable<Rect>? detections = null)
    {
        // t0 = time.time()
        using Mat frame = new();
        var (height, width) = (rawFrame.Height, rawFrame.Width);
        Cv2.CvtColor(rawFrame, frame, ColorConversionCodes.BGR2GRAY);
        var H = Mat.Eye(2, 3, MatType.CV_64FC1).ToMat();

        // Downscale image
        if (_downscale > 1.0)
        {
            width = (int)Math.Floor(width / _downscale);
            height = (int)Math.Floor(height / _downscale);
            Cv2.Resize(frame, frame, new(width, height));
        }

        // Find the keypoints
        var keypoints = Cv2.GoodFeaturesToTrack(
            frame,
            mask: frame.EmptyClone(),
            maxCorners: _features.MaxCorners,
            qualityLevel: _features.QualityLevel,
            minDistance: _features.MinDistance,
            blockSize:  _features.BlockSize,
            useHarrisDetector:  _features.UseHarrisDetector,
            k: _features.K);

        // Handle first frame
        if (!_initializedFirstFrame)
        {
            // Initialize data
            _prevFrame = frame.Clone();
            if (keypoints.Length > 0)
            {
                _prevKeyPoints = new Point2f[keypoints.Length];
                keypoints.CopyTo(_prevKeyPoints, 0);
            }
            // Initialization done
            _initializedFirstFrame = true;
            return H;
        }


        // Find correspondences
        var prevKeyPoints = _prevKeyPoints as Point2f[] ?? Array.Empty<Point2f>();
        var matchedKeypoints = new Point2f[prevKeyPoints.Length];
        Cv2.CalcOpticalFlowPyrLK(_prevFrame, frame, prevKeyPoints, ref matchedKeypoints, out var status, out var err);

        // Leave good correspondences only
        List<Point2f> prevPoints = [];
        List<Point2f> currPoints = [];

        for (var i = 0; i < status.Length; i++)
        {
            if (Convert.ToBoolean(status[i]))
            {
                prevPoints.Add(prevKeyPoints[i]);
                currPoints.Add(matchedKeypoints[i]);
            }
        }

        // Find rigid matrix
        if ((prevPoints.Count > 4) && (prevPoints.Count == prevPoints.Count))
        {
            H = Cv2.EstimateAffinePartial2D(InputArray.Create(prevPoints),
                                            InputArray.Create(currPoints),
                                            null,
                                            RobustEstimationAlgorithms.RANSAC)!;
            // Handle downscale
            if (_downscale > 1.0)
            {
                H.Set(0, 2, H.At<double>(0, 2) * _downscale);
                H.Set(1, 2, H.At<double>(1, 2) * _downscale);
            }
        }
        else
        {
            Trace.TraceWarning("WARNING: not enough matching points");
        }

        // Store to next iteration
        _prevFrame = frame.Clone();
        if (keypoints.Length > 0)
        {
            _prevKeyPoints = new Point2f[keypoints.Length];
            keypoints.CopyTo(_prevKeyPoints, 0);
        }

        // gmc_line = str(1000 * (time.time() - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        //     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        // self.gmc_file.write(gmc_line)

        return H;
    }

    Mat ApplyFile(Mat rawFrame, IEnumerable<Rect>? detections = null)
    {
        if (_gmcFile is null)
        {
            throw new NullReferenceException(nameof(_gmcFile));
        }

        //Return the homography matrix based on the GCPs in the next line of the input GMC file.
        var line = _gmcFile.ReadLine();
        if (string.IsNullOrWhiteSpace(line))
        {
            throw new NullReferenceException(nameof(_gmcFile));
        }

        var tokens = line.Split('\t');
        var H = Mat.Eye(2, 3, MatType.CV_64FC1).ToMat();

        H.Set(0, 0, H.At<double>(0, 0) * double.Parse(tokens[1]));
        H.Set(0, 1, H.At<double>(0, 1) * double.Parse(tokens[2]));
        H.Set(0, 2, H.At<double>(0, 2) * double.Parse(tokens[3]));
        H.Set(1, 0, H.At<double>(1, 0) * double.Parse(tokens[4]));
        H.Set(1, 1, H.At<double>(1, 1) * double.Parse(tokens[5]));
        H.Set(1, 2, H.At<double>(1, 2) * double.Parse(tokens[6]));
        return H;
    }
}
