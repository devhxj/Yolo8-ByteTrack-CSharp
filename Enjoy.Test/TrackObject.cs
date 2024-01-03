using Enjoy.ByteTrack;

namespace Enjoy.Test;

/// <summary>
/// 
/// </summary>
/// <param name="rectBox"></param>
/// <param name="label"></param>
/// <param name="prob"></param>
public class TrackObject(RectBox rectBox, int label, float prob) : IObject
{
    readonly RectBox _rectBox = rectBox;
    readonly int _label = label;
    readonly float _prob = prob;

    public RectBox RectBox => _rectBox;

    public int Label => _label;

    public float Prob => _prob;

    public Track ToTrack() => new(_rectBox, _prob);
}