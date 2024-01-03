using Enjoy.Algorithm;
using MathNet.Numerics.LinearAlgebra;
using NumpyDotNet;
using System.Diagnostics.Metrics;

namespace Enjoy.BotSort;

/// <summary>
/// A simple Kalman filter for tracking bounding boxes in image space.
/// For BoT-SORT
/// The 8-dimensional state space
///    x, y, w, h, vx, vy, vw, vh
/// contains the bounding box center position (x, y), width w, height h,
/// and their respective velocities.
/// Object motion follows a constant velocity model.The bounding box location
/// (x, y, w, h) is taken as direct observation of the state space (linear
/// observation model).
/// </summary>
public class BotKalmanFilter: KalmanFilter
{
    public override void Initiate(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, float[] measurement)
    {
        var m2 = measurement[2];
        var m3 = measurement[3];
        f1x8Mean.SetSubMatrix(0, 1, 0, 4, M.Dense(1, 4, measurement));
        f1x8Mean.SetSubMatrix(0, 1, 4, 4, M.Dense(1, 4, 0));
        var std = V.Dense([
            2 * _stdWeightPosition * m2,
            2 * _stdWeightPosition * m3,
            2 * _stdWeightPosition * m2,
            2 * _stdWeightPosition * m3,
            10 * _stdWeightVelocity * m2,
            10 * _stdWeightVelocity * m3,
            10 * _stdWeightVelocity * m2,
            10 * _stdWeightVelocity * m3
        ]);
        f8x8Covariance = M.DenseOfDiagonalVector(std.PointwiseMultiply(std));
    }

    public override void Predict(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance)
    {
        var m2 = f1x8Mean[0, 2];
        var m3 = f1x8Mean[0, 3];
        var std = V.Dense([
            _stdWeightPosition * m2,
            _stdWeightPosition * m3,
            _stdWeightPosition * m2,
            _stdWeightPosition * m3,

            _stdWeightVelocity * m2,
            _stdWeightVelocity * m3,
            _stdWeightVelocity * m2,
            _stdWeightVelocity * m3
        ]);

        var motionCov = M.DenseOfDiagonalVector(std.PointwiseMultiply(std));
        f1x8Mean = (_f8x8MotionMat * f1x8Mean.Transpose()).Transpose();
        f8x8Covariance = _f8x8MotionMat * f8x8Covariance * _f8x8MotionMat.Transpose() + motionCov;
    }

    protected override void Project(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, out Matrix<float> f1x4ProjectedMean, out Matrix<float> f4x4ProjectedCov)
    {
        var m2 = f1x8Mean[0, 2];
        var m3 = f1x8Mean[0, 3];
        var std = V.Dense([
            _stdWeightPosition * m2,
            _stdWeightPosition * m3,
            _stdWeightPosition * m2,
            _stdWeightPosition * m3
        ]);

        var diag = M.DenseOfDiagonalVector(std.PointwiseMultiply(std));
        f1x4ProjectedMean = (_f4x8UpdateMat * f1x8Mean.Transpose()).Transpose();
        f4x4ProjectedCov = _f4x8UpdateMat * f8x8Covariance * _f4x8UpdateMat.Transpose() + diag;
    }

    /// <summary>
    /// Compute gating distance between state distribution and measurements.A suitable distance threshold can be obtained from `chi2inv95`. If `only_position` is False, the chi-square distribution has 4 degrees of freedom, otherwise 2.
    /// </summary>
    /// <param name="f1x8Mean"></param>
    /// <param name="f8x8Covariance"></param>
    /// <param name="measurement"></param>
    /// <param name="only_position"></param>
    /// <param name="metric"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public float GatingDistance(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, float[] measurement, bool only_position = false)
    {
        Project(ref f1x8Mean, ref f8x8Covariance, out var f1x4ProjectedMean, out var f4x4ProjectedCov);
        if (only_position)
        {
            f1x4ProjectedMean.Clear();
            f4x4ProjectedCov.SubMatrix(2, 2, 2, 2).Clear();
        }

        var diff = M.Dense(1, 4, measurement) - f1x4ProjectedMean;
        var choleskyFactor = f4x4ProjectedCov.Cholesky().Factor;
        var mahalanobisDistance = choleskyFactor.LowerTriangle().Solve(diff).Transpose();
        var array = mahalanobisDistance.PointwiseMultiply(mahalanobisDistance).ToRowMajorArray();

        return M.DenseOfDiagonalArray(array).Trace();
    }
}
