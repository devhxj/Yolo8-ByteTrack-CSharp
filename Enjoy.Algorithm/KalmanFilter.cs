using MathNet.Numerics.LinearAlgebra;
namespace Enjoy.Algorithm;

public class KalmanFilter
{
    protected static readonly MatrixBuilder<float> M = Matrix<float>.Build;
    protected static readonly VectorBuilder<float> V = Vector<float>.Build;
    protected readonly float _stdWeightPosition;
    protected readonly float _stdWeightVelocity;
    protected readonly Matrix<float> _f8x8MotionMat;
    protected readonly Matrix<float> _f4x8UpdateMat;

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
    /// Create track from unassociated measurement
    /// </summary>
    /// <param name="f1x8Mean">the mean matrix(1x8 dimensional) of the new track.</param>
    /// <param name="f8x8Covariance">the covariance matrix(8x8 dimensional) of the new track.</param>
    /// <param name="measurement">Bounding box coordinates(x, y, a, h) with center position(x, y), aspect ratio a, and height h.</param>
    public virtual void Initiate(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, float[] measurement)
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
    /// Run Kalman filter prediction step.
    /// </summary>
    /// <param name="f1x8Mean">the mean matrix(1x8 dimensional) of the new track.</param>
    /// <param name="f8x8Covariance">the covariance matrix(8x8 dimensional) of the new track.</param>
    public virtual void Predict(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance)
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
    /// Run Kalman filter correction step.
    /// </summary>
    /// <param name="f1x8Mean">The predicted state's mean matrix (1x8 dimensional).</param>
    /// <param name="f8x8Covariance">The state's covariance matrix (8x8 dimensional).</param>
    /// <param name="measurement">The 4 dimensional measurement vector(x, y, a, h), where(x, y) is the center position, a the aspect ratio, and h the height of the bounding box.</param>
    public virtual void Update(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, float[] measurement)
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
    /// Project state distribution to measurement space.
    /// </summary>
    /// <param name="f1x8Mean">The state's mean matrix (1x8 dimensional).</param>
    /// <param name="f8x8Covariance">The state's covariance matrix (8x8 dimensional).</param>
    /// <param name="f1x4ProjectedMean">Returns the projected mean matrix of the given state estimate.</param>
    /// <param name="f4x4ProjectedCov">Returns the projected covariance matrix of the given state estimate.</param>
    protected virtual void Project(ref Matrix<float> f1x8Mean, ref Matrix<float> f8x8Covariance, out Matrix<float> f1x4ProjectedMean, out Matrix<float> f4x4ProjectedCov)
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
