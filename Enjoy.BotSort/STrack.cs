using Enjoy.Tracker.Trackers.Utils;
using NumpyDotNet;

namespace Enjoy.Tracker.Trackers;

public class STrack : BaseTrack
{
    protected ndarray _tlwh;
    static SimpleKalmanFilter shared_kalman = new SimpleKalmanFilter();
    protected SimpleKalmanFilter? _kalman_filter;
    protected ndarray idx;
    int tracklet_len;
    int _cls;

    public STrack(ndarray tlwh, double score, int cls)
    {
        _tlwh = np.asarray(tlbr_to_tlwh((ndarray)tlwh[":-1"]), dtype: np.Float32);
        _kalman_filter = null;
        Mean = null;
        Covariance = null;
        is_activated = false;

        _score = score;
        tracklet_len = 0;
        _cls = cls;
        idx = (ndarray)tlwh[-1];
    }
    public ndarray? Mean { get; set; }
    public ndarray? Covariance { get; set; }

    public ndarray? pred_bbox { get; }

    /// <summary>
    /// Get current position in bounding box format `(top left x, top left y, width, height)`.
    /// </summary>
    public virtual ndarray tlwh
    {
        get
        {
            if (Mean is null)
                return _tlwh.Copy();

            var ret = ((ndarray)Mean[":4"]).Copy();
            ret[2] *= (ndarray)ret[3];
            ret[":2"] -= (ndarray)ret["2:"] / 2;
            return ret;
        }
    }

    /// <summary>
    /// Convert bounding box to format `(min x, min y, max x, max y)`, i.e., `(top left, bottom right)`.
    /// </summary>
    public ndarray tlbr
    {
        get
        {
            var ret = tlwh.Copy();
            ret["2:"] += (ndarray)ret[":2"];
            return ret;
        }
    }

    /// <summary>
    /// Perform multi-object predictive tracking using Kalman filter for given stracks.
    /// </summary>
    /// <param name="stracks"></param>
    public static void multi_predict(IList<STrack> stracks)
    {
        if (stracks.Count <= 0)
            return;

        var multi_mean = np.asarray(stracks.Select(x => x.Mean.Copy()).ToArray());
        var multi_covariance = np.asarray(stracks.Select(x => x.Covariance).ToArray());

        for (var i = 0; i < stracks.Count; i++)
        {
            var st = stracks[i];
            if (st._state != TrackState.Tracked)
                ((ndarray)multi_mean[i])[7] = 0;
        }

        (multi_mean, multi_covariance) = STrack.shared_kalman.MultiPredict(multi_mean, multi_covariance);

        var j = 0;
        foreach (var (mean, cov) in Enumerable.Zip(multi_mean, multi_covariance))
        {
            stracks[j].Mean = (ndarray)mean;
            stracks[j].Covariance = (ndarray)cov;
            j++;
        }
    }

    /// <summary>
    /// Update state tracks positions and covariances using a homography matrix.
    /// </summary>
    /// <param name="stracks"></param>
    /// <param name="H"></param>
    /// <returns></returns>
    public static void multi_gmc(IList<STrack> stracks, ndarray? H = null)
    {
        if (stracks.Count <= 0)
            return;

        H ??= np.eye(2, 3);
        var multi_mean = np.asarray(stracks.Select(x => x.Mean.Copy()).ToArray());
        var multi_covariance = np.asarray(stracks.Select(x => x.Covariance).ToArray());

        var R = (ndarray)H[":2", ":2"];
        var R8x8 = np.kron(np.eye(4, dtype: np.Float32), R);
        var t = (ndarray)H[":2", 2];

        var i = 0;
        foreach (var (mean, cov) in Enumerable.Zip(multi_mean, multi_covariance))
        {
            var mean1 = R8x8.dot((ndarray)mean);
            mean1[":2"] += t;
            var cov1 = R8x8.dot(cov).dot(R8x8.Transpose());
            stracks[i].Mean = (ndarray)mean;
            stracks[i].Covariance = (ndarray)cov;
            i++;
        }
    }

    /// <summary>
    /// Convert bounding box to format `(center x, center y, aspect ratio,height)`, where the aspect ratio is `width / height`.
    /// </summary>
    /// <param name="tlwh"></param>
    /// <returns></returns>
    public static ndarray tlwh_to_xyah(ndarray tlwh)
    {
        var ret = np.asarray(tlwh).Copy();
        ret[":2"] += (ndarray)ret["2:"] / 2;
        ret[2] /= (ndarray)ret[3];
        return ret;
    }

    /// <summary>
    /// Converts top-left bottom-right format to top-left width height format.
    /// </summary>
    /// <param name="tlbr"></param>
    /// <returns></returns>
    public static ndarray tlbr_to_tlwh(ndarray tlbr)
    {
        var ret = np.asarray(tlbr).Copy();
        ret["2:"] -= (ndarray)ret[":2"];
        return ret;
    }

    /// <summary>
    /// Converts tlwh bounding box format to tlbr format.
    /// </summary>
    /// <param name="tlwh"></param>
    /// <returns></returns>
    public static ndarray tlwh_to_tlbr(ndarray tlwh)
    {
        var ret = np.asarray(tlwh).Copy();
        ret["2:"] += (ndarray)ret[":2"];
        return ret;
    }

    /// <summary>
    /// Predicts mean and covariance using Kalman filter.
    /// </summary>
    public override void predict()
    {
        var mean_state = Mean?.Copy();
        if (_state != TrackState.Tracked)
        {
            mean_state[7] = 0;
        }

        (Mean, Covariance) = _kalman_filter!.Predict(mean_state, Covariance);
    }

    /// <summary>
    /// Start a new tracklet.
    /// </summary>
    /// <param name="kalman_filter"></param>
    /// <param name="frame_id"></param>
    void activate(SimpleKalmanFilter kalman_filter, int frame_id)
    {
        _kalman_filter = kalman_filter;
        track_id = next_id();
        (Mean, Covariance) = _kalman_filter.Initiate(convert_coords(_tlwh));

        tracklet_len = 0;
        _state = TrackState.Tracked;
        if (frame_id == 1)
            is_activated = true;

        _frame_id = frame_id;
        start_frame = frame_id;
    }

    /// <summary>
    /// Reactivates a previously lost track with a new detection.
    /// </summary>
    /// <param name="new_track"></param>
    /// <param name="frame_id"></param>
    /// <param name="new_id"></param>
    public void re_activate(STrack new_track, int frame_id, bool new_id = false)
    {
        (Mean, Covariance) = _kalman_filter.Update(Mean, Covariance, convert_coords(new_track.tlwh));

        tracklet_len = 0;
        _state = TrackState.Tracked;
        is_activated = true;
        _frame_id = frame_id;

        if (new_id)
            track_id = next_id();

        _score = new_track._score;
        _cls = new_track._cls;
        idx = new_track.idx;
    }

    /// <summary>
    /// Update a matched track
    /// </summary>
    /// <param name="new_track"></param>
    /// <param name="frame_id"></param>
    void update(STrack new_track, int frame_id)
    {
        _frame_id = frame_id;
        tracklet_len += 1;

        var new_tlwh = new_track.tlwh;
        (Mean, Covariance) = _kalman_filter.Update(Mean, Covariance, convert_coords(new_tlwh));
        _state = TrackState.Tracked;
        is_activated = true;

        _score = new_track._score;
        _cls = new_track._cls;
        idx = new_track.idx;
    }

    /// <summary>
    /// Convert a bounding box's top-left-width-height format to its x-y-angle-height equivalent.
    /// </summary>
    /// <param name="tlwh"></param>
    /// <returns></returns>
    ndarray convert_coords(ndarray tlwh) => tlwh_to_xyah(tlwh);

    /// <summary>
    /// Return a string representation of the BYTETracker object with start and end frames and track ID.
    /// </summary>
    /// <returns></returns>
    public override string ToString() => $"OT_{track_id}_({start_frame}-{end_frame})";
}
