using Enjoy.Tracker.Trackers.Utils;
using NumpyDotNet;
using System.Runtime.CompilerServices;

namespace Enjoy.Tracker.Trackers;

public abstract class BaseTrack
{
    protected static int Count = 0;
    public int track_id = 0;
    public bool is_activated = false;
    protected TrackState _state = TrackState.New;
    protected SortedDictionary<int, object> history = new();
    protected List<Feature> features = [];
    protected Feature? curr_feature = null;
    public double _score = 0;
    public int start_frame = 0;
    public int _frame_id = 0;
    protected int time_since_update = 0;

    // Multi-camera
    ITuple location = (np.Inf, np.Inf);

    public TrackState State => _state;

    /// <summary>
    /// Return the last frame ID of the track.
    /// </summary>
    /// <returns></returns>
    public int end_frame => _frame_id;

    /// <summary>
    /// Increment and return the global track ID counter.
    /// </summary>
    /// <returns></returns>
    public static int next_id()
    {
        BaseTrack.Count += 1;
        return BaseTrack.Count;
    }

    /// <summary>
    /// Activate the track with the provided arguments.
    /// </summary>
    public virtual void activate(params object[] args)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Predict the next state of the track.
    /// </summary>
    /// <exception cref="NotImplementedException"></exception>
    public virtual void predict()
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Update the track with new observations.
    /// </summary>
    /// <exception cref="NotImplementedException"></exception>
    public virtual void update()
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Mark the track as lost.
    /// </summary>
    /// <exception cref="NotImplementedException"></exception>
    public virtual void mark_lost()
    {
        _state = TrackState.Lost;
    }

    /// <summary>
    /// Mark the track as removed.
    /// </summary>
    /// <exception cref="NotImplementedException"></exception>
    public virtual void mark_removed()
    {
        _state = TrackState.Removed;
    }

    /// <summary>
    /// Reset the global track ID counter.
    /// </summary>
    public static void reset_id()
    {
        BaseTrack.Count = 0;
    }
}
