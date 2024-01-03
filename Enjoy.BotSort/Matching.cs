using NumpyDotNet;
using OpenCvSharp;
using System.Runtime.CompilerServices;
using static NumpyDotNet.Scipy.Binder;

namespace Enjoy.BotSort;

public record Detection(double Score,Rect2f XYAH,int curr_feat);

public sealed class Matching
{
    static Dictionary<int, double> chi2inv95 = new Dictionary<int, double>()
    {
        { 1,3.8415 },
        { 2, 5.9915 },
        { 3,7.8147 },
        { 4, 9.4877 },
        { 5, 11.070 },
        { 6, 12.592 },
        { 7, 14.067 },
        { 8, 15.507 },
        { 9, 16.919 },
    };

    /// <summary>
    /// Merge two sets of matches and return matched and unmatched indices.
    /// </summary>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <param name=""></param>
    public static void merge_matches(ndarray m1, ndarray m2, shape shape)
    {
        throw new NotImplementedException();
        //shape O, P, Q = shape;
        //m1 = np.asarray(m1);
        //m2 = np.asarray(m2);

        //var M1 = scipy.sparse.coo_matrix((np.ones(np.alen(m1)), (m1[:, 0], m1[:, 1])), shape = (O, P))
        //var  M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape = (P, Q))

        //var mask = M1 * M2;
        //var match = mask.nonzero();
        //match = list(zip(match[0], match[1]));
        //var   unmatched_O = tuple(set(range(O)) - { i for i, j in match});
        //var unmatched_Q = tuple(set(range(Q)) - { j for i, j in match});

        //return (match, unmatched_O, unmatched_Q);
    }

    /// <summary>
    /// _indices_to_matches: Return matched and unmatched indices given a cost matrix, indices, and a threshold.
    /// </summary>
    /// <param name="cost_matrix"></param>
    /// <param name="indices"></param>
    /// <param name="thresh"></param>
    /// <returns></returns>
    static (ndarray, ndarray, ndarray) _indices_to_matches(ndarray cost_matrix, ndarray indices, double thresh)
    {
        throw new NotImplementedException();
        //var matched_cost = cost_matrix[tuple(stdlib.zip(indices))];
        //var matched_mask = (matched_cost <= thresh);

        //var matches = indices[matched_mask];
        //var unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]));
        //var unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]));

        //return (matches, unmatched_a, unmatched_b)
    }


    /// <summary>
    /// Linear assignment implementations with scipy and lap.lapjv.
    /// </summary>
    /// <param name="cost_matrix"></param>
    /// <param name="thresh"></param>
    /// <param name="use_lap"></param>
    /// <returns></returns>
    public static (ndarray, ndarray, ndarray) linear_assignment(ndarray cost_matrix, double thresh, bool use_lap = true)
    {
        if (cost_matrix.size == 0)
            return (
                np.empty((0, 2), dtype: np.Int32),
                np.arange(cost_matrix.shape[0]),
                np.arange(cost_matrix.shape[1]));

        if (use_lap)
        {
            var (x, y) = Lapjv.Assign((double[,])cost_matrix.ToSystemArray(), extendCost: true, costLimit: thresh);
            //matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
            var matches = Enumerable
                .Range(0, x.Length)
                .Select(i => (ix: i, mx: x[i]))
                .Where(i => i.mx > 0)
                .Cast<ITuple>()
                .AsArray();
            //unmatched_a = np.where(x < 0)[0]
            //unmatched_b = np.where(y < 0)[0]
            var unmatched_a = (ndarray)((ndarray)np.where(np.array<int>(x) < 0))[0];
            var unmatched_b = (ndarray)((ndarray)np.where(np.array<int>(y) < 0))[0];

            return (matches, unmatched_a, unmatched_b);
        }
        else
        {
            // Scipy linear sum assignment is NOT working correctly, DO NOT USE
            var (y, x) = scipy.optimize.linear_sum_assignment(cost_matrix);  // row y, col x
            //matches = np.asarray([[i, x] for i, x in enumerate(x) if cost_matrix[i, x] <= thresh])
            var matches = Enumerable
                .Range(0, x.Length)
                .Select(i => (i, xi: x[i]))
                .Where(i => (double)cost_matrix[i.i, i.xi] <= thresh);
            var unmatched = np.ones(cost_matrix.shape);

            foreach (var (i, xi) in matches)
            {
                unmatched[i, xi] = 0.0;
            }
            //unmatched_a = np.where(unmatched.all(1))[0]
            //unmatched_b = np.where(unmatched.all(0))[0]
            var unmatched_a = (ndarray)((ndarray)np.where((bool)unmatched.All(1)))[0];
            var unmatched_b = (ndarray)((ndarray)np.where(unmatched.All(0)))[0];

            return (matches.Cast<ITuple>().AsArray(), unmatched_a, unmatched_b);
        }
    }

    /// <summary>
    /// Compute cost based on IoU
    /// </summary>
    /// <param name=""></param>
    /// <param name=""></param>
    public static ndarray ious(ndarray atlbrs, ndarray btlbrs)
    {
        var ious = np.zeros((np.alen(atlbrs), np.alen(btlbrs)), dtype: np.Float32);
        if (ious.size == 0)
            return ious;

        ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype: np.Float32), np.ascontiguousarray(btlbrs, dtype: np.Float32));
        return ious;
    }

    /// <summary>
    /// Compute cost based on IoU
    /// </summary>
    /// <param name="atracks">list[STrack]</param>
    /// <param name="btracks">list[STrack]</param>
    /// <returns></returns>
    public static ndarray iou_distance(IList<STrack> atracks, IList<STrack> btracks)
    {
        var atlbrs = np.concatenate(atracks.Select(track => track.tlbr));
        var btlbrs = np.concatenate(btracks.Select(track => track.tlbr));
        var _ious = ious(atlbrs, btlbrs);
        return 1 - _ious;  // cost matrix
    }

    /// <summary>
    /// Compute cost based on IoU
    /// </summary>
    /// <param name="atracks">list[STrack]</param>
    /// <param name="btracks">list[STrack]</param>
    /// <returns></returns>
    public static ndarray iou_distance(ndarray atlbrs, ndarray btlbrs)
    {
        var _ious = ious(atlbrs, btlbrs);
        return 1 - _ious;  // cost matrix
    }

    /// <summary>
    /// Compute cost based on IoU
    /// </summary>
    /// <param name="atracks">list[STrack]</param>
    /// <param name="btracks">list[STrack]</param>
    /// <returns></returns>
    public static ndarray v_iou_distance(IList<STrack> atracks, IList<STrack> btracks)
    {
        var atlbrs = np.concatenate(atracks.Select(track => STrack.tlwh_to_tlbr(track.pred_bbox)));
        var btlbrs = np.concatenate(btracks.Select(track => STrack.tlwh_to_tlbr(track.pred_bbox)));
        var _ious = ious(atlbrs, btlbrs);
        return 1 - _ious;  // cost matrix
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="tracks">list[STrack]</param>
    /// <param name="detections">list[BaseTrack]</param>
    /// <param name="metric"></param>
    /// <returns></returns>
    public static ndarray embedding_distance(IList<Track> tracks, IList<Detection> detections,string metric= "cosine") 
    {
        var cost_matrix = np.zeros((tracks.Count, detections.Count), dtype: np.Float32);
        if (cost_matrix.size == 0)
            return cost_matrix;

        var det_features = np.asarray(detections.Select(x => x.curr_feat).ToArray(), dtype : np.Float32) ;

        // for i, track in enumerate(tracks):
        // cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
        var track_features = np.asarray(detections.Select(x => x.curr_feat).ToArray(), dtype:np.Float32) ;
        cost_matrix = np.maximum(0.0, scipy.spatial.distance.cdist(track_features, det_features, metric));  // Normalized features
        return cost_matrix;
    }

    /// <summary>
    /// Apply gating to the cost matrix based on predicted tracks and detected objects.
    /// </summary>
    /// <param name="kf"></param>
    /// <param name="cost_matrix"></param>
    /// <param name="tracks"></param>
    /// <param name="detections"></param>
    /// <param name="only_position"></param>
    /// <returns></returns>
    public static ndarray gate_cost_matrix(BotKalmanFilter kf, ndarray cost_matrix, IList<Track> tracks, IList<Detection> detections, bool only_position = false)
    {
        if (cost_matrix.size == 0)
            return cost_matrix;

        var gating_dim = only_position ? 2 : 4;
        var gating_threshold = chi2inv95[gating_dim];
        var measurements = np.asarray(detections.Select(x => x.XYAH).ToArray());

        for (var row = 0; row < tracks.Count; row++)
        {
            var track = tracks[row];
            var gating_distance = kf.GatingDistance(track.Mean, track.Covariance, measurements, only_position);
            cost_matrix[row, gating_distance > gating_threshold] = float.PositiveInfinity;
        }
        return cost_matrix;
    }

    /// <summary>
    /// Fuse motion between tracks and detections with gating and Kalman filtering.
    /// </summary>
    /// <param name="kf"></param>
    /// <param name="cost_matrix"></param>
    /// <param name="tracks"></param>
    /// <param name="detections"></param>
    /// <param name="only_position"></param>
    /// <param name="lambda_"></param>
    /// <returns></returns>
    public static ndarray fuse_motion(BotKalmanFilter kf, ndarray cost_matrix, IList<Track> tracks, IList<Detection> detections, bool only_position = false, float lambda_ = 0.98f)
    {
        if (cost_matrix.size == 0)
            return cost_matrix;

        var gating_dim = only_position ? 2 : 4;
        var gating_threshold = chi2inv95[gating_dim];
        var measurements = np.asarray(detections.Select(x => x.XYAH).ToArray());

        for (var row = 0; row < tracks.Count; row++)
        {
            var track = tracks[row];
            var gating_distance = kf.GatingDistance(track.Mean, track.Covariance, measurements, only_position, metric: "maha");
            cost_matrix[row, gating_distance > gating_threshold] = np.Inf;
            cost_matrix[row] = lambda_ * (ndarray)cost_matrix[row] + (1 - lambda_) * gating_distance;
        }
        return cost_matrix;
    }

    /// <summary>
    /// Fuses ReID and IoU similarity matrices to yield a cost matrix for object tracking.
    /// </summary>
    /// <param name="cost_matrix"></param>
    /// <param name="tracks"></param>
    /// <param name="detections"></param>
    /// <returns></returns>
    public static ndarray fuse_iou(ndarray cost_matrix, IList<STrack> tracks, IList<STrack> detections)
    {
        if (cost_matrix.size == 0)
            return cost_matrix;

        var reid_sim = 1 - cost_matrix;
        var iou_dist = iou_distance(tracks, detections);
        var iou_sim = 1 - iou_dist;
        var fuse_sim = reid_sim * (1 + iou_sim) / 2;
        // det_scores = np.array([det.score for det in detections])
        // det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
        return 1 - fuse_sim;  // fuse cost
    }

    /// <summary>
    /// Fuses cost matrix with detection scores to produce a single similarity matrix.
    /// </summary>
    /// <param name=""></param>
    /// <param name=""></param>
    /// <returns></returns>
    public static ndarray fuse_score(ndarray cost_matrix, IList<STrack> detections)
    {
        if (cost_matrix.size == 0)
            return cost_matrix;

        var iou_sim = 1 - cost_matrix;
        var det_scores = np.array(detections.Select(x => x._score).ToArray());
        det_scores = np.expand_dims(det_scores, axis: 0).Repeat(cost_matrix.shape[0], axis: 0);

        var fuse_sim = iou_sim * det_scores;
        return 1 - fuse_sim;  // fuse_cost
    }

    /// <summary>
    /// Calculate the Intersection over Union (IoU) between pairs of bounding boxes.
    /// The bounding box coordinates are expected to be in the format (x1, y1, x2, y2).
    /// </summary>
    /// <param name="box1">A numpy array of shape (n, 4) representing 'n' bounding boxes.Each row is in the format (x1, y1, x2, y2).</param>
    /// <param name="box2">A numpy array of shape (m, 4) representing 'm' bounding boxes.Each row is in the format (x1, y1, x2, y2).</param>
    /// <param name="eps">A small constant to prevent division by zero. Defaults to 1e-7.</param>
    /// <returns>A numpy array of shape (n, m) representing the IoU scores for each pair of bounding boxes from box1 and box2.</returns>
    public static ndarray bbox_ious(ndarray box1, ndarray box2, float eps = 1e-7f)
    {
        ndarray b1_x1, b1_y1, b1_x2, b1_y2, b2_x1, b2_y1, b2_x2, b2_y2;
        b1_x1 = b1_y1 = b1_x2 = b1_y2 = box1.T;
        b2_x1 = b2_y1 = b2_x2 = b2_y2 = box2.T;

        // Intersection area
        var inter_area = (
                np.minimum((ndarray)b1_x2[":", null], b2_x2) -
                np.maximum((ndarray)b1_x1[":", null], b2_x1)
            ).clip(0, 10000) * (
                np.minimum((ndarray)b1_y2[":", null], b2_y2) -
                np.maximum((ndarray)b1_y1[":", null], b2_y1)
            ).clip(0, 10000);

        // box2 area
        var box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1);
        var box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1);
        return inter_area / (box2_area + box1_area[":", null] - inter_area + eps);
    }
}
