using MathNet.Numerics.LinearAlgebra;
namespace Enjoy.Algorithm;

public sealed class KalmanFilter
{
    static readonly MatrixBuilder<float> M = Matrix<float>.Build;
    static readonly VectorBuilder<float> V = Vector<float>.Build;

    readonly float _stdWeightPosition;
    readonly float _stdWeightVelocity;
    readonly Matrix<float> _f8x8MotionMat;
    readonly Matrix<float> _f4x8UpdateMat;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="stdWeightPosition"></param>
    /// <param name="stdWeightVelocity"></param>
    public KalmanFilter(float stdWeightPosition = 1.0f / 20, float stdWeightVelocity = 1.0f / 160)
    {   
        _stdWeightPosition = stdWeightPosition;
        _stdWeightVelocity = stdWeightVelocity;
        int ndim = 4;
        float dt = 1;

        _f8x8MotionMat = M.DenseIdentity(8, 8);
        _f4x8UpdateMat = M.DenseIdentity(4, 8);

        for (var i = 0; i < ndim; i++)
            _f8x8MotionMat[i, ndim + i] = dt;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="f1x8Mean"></param>
    /// <param name="f8x8Covariance"></param>
    /// <param name="measurement"></param>
    public void Initiate(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, float[] measurement)
    {
        var m = measurement[3];
        f1x8Mean.SetSubMatrix(0, 1, 0, 4, M.Dense(1, 4, measurement));
        f1x8Mean.SetSubMatrix(0, 1, 4, 4, M.Dense(1, 4, 0));
        var std = V.Dense([
            2 * _stdWeightPosition * m,
            2 * _stdWeightPosition * m,
            1e-2f,
            2 * _stdWeightPosition * m,
            10 * _stdWeightVelocity * m,
            10 * _stdWeightVelocity * m,
            1e-5f,
            10 * _stdWeightVelocity * m
        ]);
        f8x8Covariance = M.DenseOfDiagonalVector(std.PointwiseMultiply(std));
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="f1x8Mean"></param>
    /// <param name="f8x8Covariance"></param>
    public void Predict(ref Matrix<float> f1x8Mean,ref Matrix<float> f8x8Covariance)
    {
        var m = f1x8Mean[0, 3];
        var std = V.Dense([
            _stdWeightPosition * m,
            _stdWeightPosition * m,
            1e-2f,
            _stdWeightPosition * m,
            _stdWeightVelocity * m,
            _stdWeightVelocity * m,
            1e-5f,
            _stdWeightVelocity* m
        ]);

        var motionCov = M.DenseOfDiagonalVector(std.PointwiseMultiply(std));
        f1x8Mean = (_f8x8MotionMat * f1x8Mean.Transpose()).Transpose();
        f8x8Covariance = _f8x8MotionMat * f8x8Covariance * _f8x8MotionMat.Transpose() + motionCov;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="f1x8Mean"></param>
    /// <param name="f8x8Covariance"></param>
    /// <param name="measurement"></param>
    public void Update(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, float[] measurement)
    {
        Project(ref f1x8Mean, ref f8x8Covariance, out var f1x4ProjectedMean, out var f4x4ProjectedCov);

        var f4x8B = (f8x8Covariance * _f4x8UpdateMat.Transpose()).Transpose();
        var f8x4KalmanGain = f4x4ProjectedCov.Cholesky().Solve(f4x8B).Transpose();
        var f1x4Innovation = M.Dense(1, 4, measurement) - f1x4ProjectedMean;

        var tmp = f1x4Innovation * f8x4KalmanGain.Transpose();
        f1x8Mean += tmp;
        f8x8Covariance -= f8x4KalmanGain * f4x4ProjectedCov * f8x4KalmanGain.Transpose();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="f1x8Mean"></param>
    /// <param name="f8x8Covariance"></param>
    /// <param name="f1x4ProjectedMean"></param>
    /// <param name="f4x4ProjectedCov"></param>
    void Project(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, out Matrix<float> f1x4ProjectedMean, out Matrix<float> f4x4ProjectedCov)
    {
        var m = f1x8Mean[0, 3];
        var std = V.Dense([
            _stdWeightPosition * m,
            _stdWeightPosition * m,
            1e-1f,
            _stdWeightPosition * m
        ]);

        var diag = M.DenseOfDiagonalVector(std.PointwiseMultiply(std));
        f1x4ProjectedMean = (_f4x8UpdateMat * f1x8Mean.Transpose()).Transpose();
        f4x4ProjectedCov = _f4x8UpdateMat * f8x8Covariance * _f4x8UpdateMat.Transpose() + diag;
    }
}
