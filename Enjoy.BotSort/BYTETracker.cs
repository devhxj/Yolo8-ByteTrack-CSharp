using Enjoy.Tracker.Trackers.Utils;
using NumpyDotNet;
using OpenCvSharp.Dnn;
using OpenCvSharp;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.PortableExecutable;
using System.Text.RegularExpressions;
using System.Threading;
using static NumpyDotNet.Python.Binder;
using static OpenCvSharp.ML.DTrees;
using static System.Formats.Asn1.AsnWriter;
using System.Diagnostics;

namespace Enjoy.Tracker.Trackers;

public class BYTETracker
{
    List<STrack> tracked_stracks;
    List<STrack> lost_stracks;
    List<STrack> removed_stracks;

    int frame_id = 0;
    (double track_high_thresh, double track_low_thresh, double track_buffer) args;
    int max_time_lost;
    object kalman_filter;

    /// <summary>
    /// Initialize a YOLOv8 object to track objects with given arguments and frame rate.
    /// </summary>
    /// <param name="args"></param>
    /// <param name="frame_rate"></param>
    public BYTETracker((double track_high_thresh,double track_low_thresh,double track_buffer) args1, int frame_rate = 30)
    {
        tracked_stracks = [];
        lost_stracks = [];
        removed_stracks = [];

        frame_id = 0;
        args = args1;
        max_time_lost = (int)(frame_rate / 30.0 * args.track_buffer);
        kalman_filter = this.get_kalmanfilter();
        reset_id();
    }

    /// <summary>
    /// Returns a Kalman filter object for tracking bounding boxes.
    /// </summary>
    /// <returns></returns>
    SimpleKalmanFilter get_kalmanfilter() => new SimpleKalmanFilter();

    /// <summary>
    /// Initialize object tracking with detections and scores using STrack algorithm.
    /// </summary>
    /// <param name="dets"></param>
    /// <param name="scores"></param>
    /// <param name="cls"></param>
    /// <param name="img"></param>
    IList<STrack> init_track(ndarray[] dets, double[] scores, int[] cls, object? img = null)
    {
        if (dets.Length == 0)
            return Array.Empty<STrack>();

        return stdlib
            .zip(dets, scores, cls)
            .Select(a => new STrack((ndarray)a[0], (double)a[1], (int)a[2]))
            .ToArray(); // detections    
    }

    /// <summary>
    /// Calculates the distance between tracks and detections using IOU and fuses scores.
    /// </summary>
    /// <param name="tracks"></param>
    /// <param name="detections"></param>
    /// <returns></returns>
    ndarray get_dists(IList<STrack> tracks, IList<STrack> detections) 
    {
        var dists = Matching.iou_distance(tracks, detections);
        // TODO: mot20
        // if not this.args.mot20:
        dists = Matching.fuse_score(dists, detections);
        return dists;
    }

    /// <summary>
    /// Returns the predicted tracks using the YOLOv8 network.
    /// </summary>
    /// <param name="tracks"></param>
    /// <returns></returns>
    void multi_predict(IList<STrack> tracks) => STrack.multi_predict(tracks);

    /// <summary>
    /// Resets the ID counter of STrack.
    /// </summary>
    /// <returns></returns>
    void reset_id() => STrack.reset_id();

    /// <summary>
    /// Updates object tracker with new detections and returns tracked object bounding boxes.
    /// </summary>
    /// <param name="results"></param>
    /// <param name="img"></param>
    void update(object results,object? img = null)
    {
        frame_id += 1;
        List<STrack> activated_stracks = [];
        List<STrack> refind_stracks = [];
        List<STrack> lost_stracks = [];
        List<STrack> removed_stracks = [];

        var scores = results.conf;
        var bboxes = results.xyxy;

        // Add index
        bboxes = np.concatenate([bboxes, np.arange(np.alen(bboxes)).reshape(-1, 1)], axis: -1);
        int[] cls = results.cls;

        var remain_inds = scores > args.track_high_thresh;
        var inds_low = scores > args.track_low_thresh;
        var inds_high = scores < args.track_high_thresh;

        var inds_second = np.logical_and(inds_low, inds_high);
        var dets_second = bboxes[inds_second];
        var dets = bboxes[remain_inds];
        var scores_keep = scores[remain_inds];
        var scores_second = scores[inds_second];
        var cls_keep = cls[remain_inds];
        var cls_second = cls[inds_second];

        var detections = init_track(dets, scores_keep, cls_keep, img);

        // Add newly detected tracklets to tracked_stracks
        List<STrack> unconfirmed = [];
        List<STrack> tracked_stracks = [];
        foreach (var track in this.tracked_stracks)
            if (!track.is_activated)
                unconfirmed.Add(track);
            else
                tracked_stracks.Add(track);

        // Step 2: First association, with high score detection boxes
        var strack_pool = joint_stracks(tracked_stracks, lost_stracks);
        // Predict the current location with KF
        multi_predict(strack_pool);

        var method = this.GetType().GetMethod("gmc");
        if (method is not null && img is not null)
        {
            var warp = (ndarray)method.Invoke(this, img, dets);
            STrack.multi_gmc(strack_pool, warp);
            STrack.multi_gmc(unconfirmed, warp);
        }


        var dists = get_dists(strack_pool, detections);
        var (matches, u_track, u_detection) = Matching.linear_assignment(dists, thresh: args.match_thresh);

        foreach ((int itracked, int idet) item in matches)
        {
            var track = strack_pool[item.itracked];
            var det = detections[item.idet];

            if (track.State == TrackState.Tracked)
            {
                track.update(det, frame_id);
                activated_stracks.Add(track);
            }
            else {
                track.re_activate(det, frame_id, new_id: false);
                refind_stracks.Add(track);
            }
        }

        // Step 3: Second association, with low score detection boxes
        // association the untrack to the low score detections
        var detections_second = init_track(dets_second, scores_second, cls_second, img);
        var r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        // TODO
        dists = Matching.iou_distance(r_tracked_stracks, detections_second);
        var (matches, u_track, u_detection_second) = Matching.linear_assignment(dists, thresh: 0.5);

        foreach ((int itracked, int idet) item in matches)
        {
            var track = r_tracked_stracks[item.itracked];
            var det = detections_second[item.idet];
            if (track.State == TrackState.Tracked)
            {
                track.update(det, frame_id);
                activated_stracks.Add(track);
            }
            else
            {
                track.re_activate(det, frame_id, new_id: false);
                refind_stracks.Add(track);
            }
        }
        foreach (var it in u_track)
        {
            var track = r_tracked_stracks[it];
            if (track.State != TrackState.Lost)
            { 
                track.mark_lost();
                lost_stracks.Add(track);
            }
        }

        // Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = u_detection.AsObjectArray().Cast<STrack>().ToArray();
        dists = get_dists(unconfirmed, detections);

        var (matches, u_unconfirmed, u_detection) = Matching.linear_assignment(dists, thresh: 0.7);
        foreach ((int itracked, int idet) item in matches)
        {
            unconfirmed[item.itracked].update(detections[item.idet], frame_id);
            activated_stracks.Add(unconfirmed[item.itracked]);
        }
        foreach (int it in u_unconfirmed) 
        {
            var track = unconfirmed[it];
            track.mark_removed();
            removed_stracks.Add(track);
                }
        // Step 4: Init new stracks
        foreach (int inew in u_detection) {
            var track = detections[inew];
            if (track._score < args.new_track_thresh)
                continue;
            track.activate(kalman_filter, frame_id);
            activated_stracks.Add(track);
        }

        // Step 5: Update state
        foreach (var track in lost_stracks)
        {
            if (frame_id - track.end_frame > max_time_lost) 
            {
                track.mark_removed();
                removed_stracks.Add(track);
            }
        }

        this.tracked_stracks = [t for t in this.tracked_stracks if t.state == TrackState.Tracked]
        this.tracked_stracks = this.joint_stracks(this.tracked_stracks, activated_stracks)
        this.tracked_stracks = this.joint_stracks(this.tracked_stracks, refind_stracks)
        this.lost_stracks = this.sub_stracks(this.lost_stracks, this.tracked_stracks)
        this.lost_stracks.extend(lost_stracks)
        this.lost_stracks = this.sub_stracks(this.lost_stracks, this.removed_stracks)
        this.tracked_stracks, this.lost_stracks = this.remove_duplicate_stracks(this.tracked_stracks, this.lost_stracks)
        this.removed_stracks.extend(removed_stracks)
        if (this.removed_stracks.Count > 1000)
            this.removed_stracks = this.removed_stracks[^999]; // clip remove stracks to 1000 maximum

        this.tracked_stracks.Where(x=> x.is_activated).Select(x=>new object[x.track_id,x.Score])

        return np.asarray(
            [x.tlbr.tolist() + [x.track_id, x.score, x.cls, x.idx] for x in this.tracked_stracks if x.is_activated],
            dtype: np.Float32)
    }

    /// <summary>
    /// Combine two lists of stracks into a single one.
    /// </summary>
    /// <param name="tlista"></param>
    /// <param name="tlistb"></param>
    /// <returns></returns>
    static IList<STrack> joint_stracks(IList<STrack> tlista, IList<STrack> tlistb)
    {
        var exists = new Dictionary<int, int>();
        List<STrack> res = [];
        foreach (var t in tlista)
        {
            exists.Add(t.track_id, 1);
            res.Add(t);
        }
        foreach (var t in tlistb)
        {
            var tid = t.track_id;
            if (!exists.ContainsKey(tid)) {
                exists.Add(tid, 1);
                res.Add(t);
            }
        }
        return res;
    }

    int[] sub_stracks(IList<STrack> tlista, IList<STrack> tlistb) {
        //DEPRECATED CODE in https://github.com/ultralytics/ultralytics/pull/1890/
        //stracks = {t.track_id: t for t in tlista}
        //for t in tlistb:
        //    tid = t.track_id
        //    if stracks.get(tid, 0):
        //        del stracks[tid]
        //return list(stracks.values())

        var track_ids_b = tlistb.Select(t => t.track_id);
        return tlistb
            .Where(t => track_ids_b.Contains(t.track_id))
            .Select(t => t.track_id)
            .ToArray();
    }

    /// <summary>
    /// Remove duplicate stracks with non-maximum IOU distance.
    /// </summary>
    /// <param name="stracksa"></param>
    /// <param name="stracksb"></param>
    /// <returns></returns>
    (IList<STrack>, IList<STrack>) remove_duplicate_stracks(IList<STrack> stracksa, IList<STrack> stracksb)
    {
        var pdist = Matching.iou_distance(stracksa, stracksb);
        var pairs = (ndarray)np.where(pdist < 0.15);
        List<int> dupa = [];
        List<int> dupb = [];

        foreach (var t in stdlib.zip(pairs))
        {
            if (t.Length > 1)
            {
                var p = (int)t[0];
                var q = (int)t[1];
                var timep = stracksa[p]._frame_id - stracksa[p].start_frame;
                var timeq = stracksb[q]._frame_id - stracksb[q].start_frame;
                if (timep > timeq)
                    dupb.Add(q);
                else
                    dupa.Add(p);
            }
        }

        var resa = Enumerable
             .Range(0, stracksa.Count)
             .Where(i => !dupa.Contains(i))
             .Select(i => stracksa[i])
             .ToArray();

        var resb = Enumerable
            .Range(0, stracksb.Count)
            .Where(i => !dupb.Contains(i))
            .Select(i => stracksb[i])
            .ToArray();
        return (resa, resb);
    }
}
