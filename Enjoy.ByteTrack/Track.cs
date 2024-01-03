using MathNet.Numerics.LinearAlgebra;
using System.Diagnostics.CodeAnalysis;

namespace Enjoy.ByteTrack;

/// <summary>
/// 
/// </summary>
/// <param name="rect"></param>
/// <param name="score"></param>
public sealed class Track
{
    readonly Dictionary<string, object?>? _attributes;
    readonly KalmanFilter _filter = new();
    Matrix<float> _f1x8Mean = Matrix<float>.Build.Dense(1, 8);
    Matrix<float> _f4x8Covariance = Matrix<float>.Build.Dense(4, 8);
    RectBox _rectBox;
    TrackState _state = TrackState.New;
    bool _isActivated = false;
    float _score;
    int _trackId = 0;
    int _frameId = 0;
    int _startFrameId = 0;
    int _trackletLen = 0;

    public Track(RectBox rectBox, float score, params (string Name, object? Value)[] attributes)
    {
        _rectBox = rectBox;
        _score = score;
        if (attributes.Length > 0)
        {
            _attributes = [];
            foreach (var (name, value) in attributes)
                _attributes.TryAdd(name, value);
        }
    }

    /// <summary>
    /// 
    /// </summary>
    public RectBox RectBox => _rectBox;

    /// <summary>
    /// 
    /// </summary>
    public TrackState State => _state;

    /// <summary>
    /// 
    /// </summary>
    public bool IsActivated => _isActivated;

    /// <summary>
    /// 
    /// </summary>
    public float Score => _score;

    /// <summary>
    /// 
    /// </summary>
    public int TrackId => _trackId;

    /// <summary>
    /// 
    /// </summary>
    public int FrameId => _frameId;

    /// <summary>
    /// 
    /// </summary>
    public int StartFrameId => _startFrameId;

    /// <summary>
    /// 
    /// </summary>
    public int TrackletLength => _trackletLen;

    public object? this[string key]
    {
        get
        {
            if (_attributes is null)
                return null;

            return _attributes.TryGetValue(key, out var value) ? value : null;
        }
    }

    void UpdateRect()
    {
        _rectBox.Width = _f1x8Mean[0, 2] * _f1x8Mean[0, 3];
        _rectBox.Height = _f1x8Mean[0, 3];
        _rectBox.X = _f1x8Mean[0,0] - _rectBox.Width / 2;
        _rectBox.Y = _f1x8Mean[0,1] - _rectBox.Height / 2;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="frameId"></param>
    /// <param name="trackId"></param>
    public void Activate(int frameId, int trackId)
    {
        _filter.Initiate(ref _f1x8Mean, ref _f4x8Covariance, _rectBox.ToXYAH());

        UpdateRect();

        _state = TrackState.Tracked;
        if (frameId == 1) _isActivated = true;
        _trackId = trackId;
        _frameId = frameId;
        _startFrameId = frameId;
        _trackletLen = 0;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="newTrack"></param>
    /// <param name="frameId"></param>
    /// <param name="newTrackId"></param>
    public void ReActivate(Track newTrack, int frameId, int newTrackId = -1)
    {
        _filter.Update(ref _f1x8Mean, ref _f4x8Covariance, newTrack.RectBox.ToXYAH());

        UpdateRect();

        _state = TrackState.Tracked;
        _isActivated = true;
        _score = newTrack.Score;
        if (0 <= newTrackId) _trackId = newTrackId;
        _frameId = frameId;
        _trackletLen = 0;
    }

    /// <summary>
    /// 
    /// </summary>
    public void Predict()
    {
        if (_state != TrackState.Tracked) 
            _f1x8Mean[0, 7] = 0;
        _filter.Predict(ref _f1x8Mean, ref _f4x8Covariance);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="newTrack"></param>
    /// <param name="frameId"></param>
    public void Update(Track newTrack, int frameId)
    {
        _filter.Update(ref _f1x8Mean, ref _f4x8Covariance, newTrack.RectBox.ToXYAH());

        UpdateRect();

        _state = TrackState.Tracked;
        _isActivated = true;
        _score = newTrack.Score;
        _frameId = frameId;
        _trackletLen++;
    }

    /// <summary>
    /// 
    /// </summary>
    public void MarkAsLost() => _state = TrackState.Lost;

    /// <summary>
    /// 
    /// </summary>
    public void MarkAsRemoved() => _state = TrackState.Removed;
}
