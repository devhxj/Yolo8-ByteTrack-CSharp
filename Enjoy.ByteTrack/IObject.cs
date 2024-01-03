namespace Enjoy.ByteTrack;

public interface IObject
{
    public RectBox RectBox { get; }

    public int Label { get; }

    public float Prob { get; }

    public Track ToTrack();
}
