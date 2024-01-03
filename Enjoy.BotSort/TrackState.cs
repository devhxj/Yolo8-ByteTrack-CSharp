namespace Enjoy.Tracker.Trackers;

/// <summary>
/// Enumeration of possible object tracking states.
/// </summary>
public enum TrackState : short
{
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
}
