using OpenCvSharp;

namespace Enjoy.ByteTrack;

/// <summary>
/// 
/// </summary>
public static class Extensions
{
    public static Rect ToRect(this RectBox rect) => new((int)rect.X, (int)rect.Y, (int)rect.Width, (int)rect.Height);
    public static Rect2f ToRect2f(this RectBox rect) => new(rect.X, rect.Y, rect.Width, rect.Height);
}
