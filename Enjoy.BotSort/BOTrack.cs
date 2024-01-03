using NumpyDotNet;
using static NumpyDotNet.Numpy.Binder;

namespace Enjoy.Tracker.Trackers;

public class BOTrack: STrack
{

    static Utils.SimpleKalmanFilter shared_kalman = new Utils.SimpleKalmanFilter(false);

    ndarray? smooth_feat;
    ndarray? curr_feat;
    double alpha;

    /// <summary>
    /// Initialize YOLOv8 object with temporal parameters, such as feature history, alpha and current features.
    /// </summary>
    /// <param name="tlwh"></param>
    /// <param name="score"></param>
    /// <param name="cls"></param>
    /// <param name="feat"></param>
    /// <param name="feat_history"></param>
    public BOTrack(ndarray tlwh, double score, int cls, ndarray? feat = null,int feat_history= 50):base(tlwh, score, cls)
    {
        smooth_feat = null;
        curr_feat = null;
        if (feat is not null)
            update_features(feat);

        features = new(feat_history);
        alpha = 0.9;
    }

    /// <summary>
    /// Get current position in bounding box format `(top left x, top left y, width, height)`.
    /// </summary>
    public override ndarray tlwh
    {
        get
        {
            if (Mean is null)
                return _tlwh.Copy();

            var ret = ((ndarray)Mean[":4"]).Copy();
            ret[":2"] -= (ndarray)ret["2:"] / 2;
            return ret;
        }
    }

    /// <summary>
    /// Convert bounding box to format `(center x, center y, width,height)`.
    /// </summary>
    /// <param name="tlwh"></param>
   static ndarray tlwh_to_xywh(ndarray tlwh) 
    {
        var ret = np.asarray(tlwh).Copy();
        ret[":2"] += (ndarray)ret["2:"] / 2;
        return ret;
    }

    /// <summary>
    /// Predicts the mean and covariance of multiple object tracks using shared Kalman filter.
    /// </summary>
    /// <param name="stracks"></param>
    static void multi_predict(IList<STrack> stracks) 
    {
        if (stracks.Count <= 0)
            return;

        var multi_mean = np.asarray(stracks.Select(st => st.Mean.Copy()));
        var multi_covariance = np.asarray(stracks.Select(x => x.Covariance).ToArray());

        for (var i = 0; i < stracks.Count; i++)
        {
            var st = stracks[i];
            if (st.State != TrackState.Tracked) {

                ((ndarray)multi_mean[i])[6] = 0;
                ((ndarray)multi_mean[i])[7] = 0;
            }
        }

        (multi_mean, multi_covariance) = BOTrack.shared_kalman.MultiPredict(multi_mean, multi_covariance);

        var j = 0;
        foreach (var (mean, cov) in Enumerable.Zip(multi_mean, multi_covariance))
        {
            stracks[j].Mean = (ndarray)mean;
            stracks[j].Covariance = (ndarray)cov;
            j++;
        }
    }

    /// <summary>
    /// Converts Top-Left-Width-Height bounding box coordinates to X-Y-Width-Height format.
    /// </summary>
    /// <param name="tlwh"></param>
    /// <returns></returns>
    ndarray convert_coords(ndarray tlwh) => tlwh_to_xywh(tlwh);

    /// <summary>
    /// Update features vector and smooth it using exponential moving average.
    /// </summary>
    /// <param name="feat"></param>
    void update_features(ndarray feat)
    {
        feat /= nx.linalg.norm(feat);
        curr_feat = feat;

        if (smooth_feat is null)
            smooth_feat = feat;
        else
            smooth_feat = alpha * smooth_feat + (1 - alpha) * feat;

        features.Add(feat);
        smooth_feat /= nx.linalg.norm(smooth_feat);
    }

    /// <summary>
    /// Predicts the mean and covariance using Kalman filter.
    /// </summary>
    void predict()
    {
        var mean_state = Mean.Copy();

        if (_state != TrackState.Tracked)
        {
            mean_state[6] = 0;
            mean_state[7] = 0;
        }

        (Mean, Covariance) = _kalman_filter.Predict(mean_state, Covariance);
    }

    /// <summary>
    /// Reactivates a track with updated features and optionally assigns a new ID.
    /// </summary>
    /// <param name="new_track"></param>
    /// <param name="frame_id"></param>
    /// <param name="new_id"></param>
    void re_activate(STrack new_track,int frame_id,bool new_id= false) {
        if (new_track.curr_feat is not null) {

            update_features(new_track.curr_feat);
        }
        base.re_activate(new_track, frame_id, new_id);
    }

    /// <summary>
    /// Update the YOLOv8 instance with new track and frame ID.
    /// </summary>
    /// <param name="new_track"></param>
    /// <param name="frame_id"></param>
    void update(STrack new_track,int frame_id) {
        if (new_track.curr_feat is not null)
            update_features(new_track.curr_feat);

        base.update(new_track, frame_id);
    }
}
