namespace Enjoy.ByteTrack;

/// <summary>
/// 
/// </summary>
/// <param name="frameRate">帧率</param>
/// <param name="trackBuffer"></param>
/// <param name="trackThresh"></param>
/// <param name="highThresh"></param>
/// <param name="matchThresh"></param>
public sealed class ByteTracker(int frameRate = 30, int trackBuffer = 30, float trackThresh = 0.5f, float highThresh = 0.6f, float matchThresh = 0.8f)
{
    readonly float _trackThresh = trackThresh;
    readonly float _highThresh = highThresh;
    readonly float _matchThresh = matchThresh;
    readonly int _maxTimeLost = (int)(frameRate / 30.0 * trackBuffer);

    int _frameId = 0;
    int _trackIdCount = 0;

    readonly List<Track> _trackedTracks = new(100);
    readonly List<Track> _lostTracks = new(100);
    List<Track> _removedTracks = new(100);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="objects"></param>
    /// <returns></returns>
    public IList<Track> Update(IList<IObject> objects)
    {
        #region Step 1: Get detections 
        _frameId++;

        // Create new Tracks using the result of object detection
        List<Track> detTracks = [];
        List<Track> detLowTracks = [];

        foreach (var obj in objects)
        {
            var strack = obj.ToTrack();
            if (obj.Prob >= _trackThresh)
            {
                detTracks.Add(strack);
            }
            else
            {
                detLowTracks.Add(strack);
            }
        }

        // Create lists of existing STrack
        List<Track> activeTracks = [];
        List<Track> nonActiveTracks = [];

        foreach (var trackedTrack in _trackedTracks)
        {
            if (!trackedTrack.IsActivated)
            {
                nonActiveTracks.Add(trackedTrack);
            }
            else
            {
                activeTracks.Add(trackedTrack);
            }
        }

        var trackPool = activeTracks.Union(_lostTracks).ToArray();

        // Predict current pose by KF
        foreach (var track in trackPool)
        {
            track.Predict();
        }
        #endregion

        #region Step 2: First association, with IoU 
        List<Track> currentTrackedTracks = [];
        Track[] remainTrackedTracks;
        Track[] remainDetTracks;
        List<Track> refindTracks = [];
        {
            var dists = CalcIouDistance(trackPool, detTracks);
            LinearAssignment(dists, trackPool.Length, detTracks.Count, _matchThresh,
                out var matchesIdx,
                out var unmatchTrackIdx,
                out var unmatchDetectionIdx);

            foreach (var matchIdx in matchesIdx)
            {
                var track = trackPool[matchIdx[0]];
                var det = detTracks[matchIdx[1]];
                if (track.State == TrackState.Tracked)
                {
                    track.Update(det, _frameId);
                    currentTrackedTracks.Add(track);
                }
                else
                {
                    track.ReActivate(det, _frameId);
                    refindTracks.Add(track);
                }
            }

            remainDetTracks = unmatchDetectionIdx.Select(unmatchIdx=> detTracks[unmatchIdx]).ToArray();
            remainTrackedTracks = unmatchTrackIdx
                .Where(unmatchIdx => trackPool[unmatchIdx].State == TrackState.Tracked)
                .Select(unmatchIdx => trackPool[unmatchIdx])
                .ToArray();
        }
        #endregion

        #region Step 3: Second association, using low score dets 
        List<Track> currentLostTracks = [];
        {
            var dists = CalcIouDistance(remainTrackedTracks, detLowTracks);
            LinearAssignment(dists, remainTrackedTracks.Length, detLowTracks.Count, 0.5f,
                             out var matchesIdx,
                             out var unmatchTrackIdx,
                             out var unmatchDetectionIdx);

            foreach (var matchIdx in matchesIdx)
            {
                var track = remainTrackedTracks[matchIdx[0]];
                var det = detLowTracks[matchIdx[1]];
                if (track.State == TrackState.Tracked)
                {
                    track.Update(det, _frameId);
                    currentTrackedTracks.Add(track);
                }
                else
                {
                    track.ReActivate(det, _frameId);
                    refindTracks.Add(track);
                }
            }

            foreach (var unmatchTrack in unmatchTrackIdx)
            {
                var track = remainTrackedTracks[unmatchTrack];
                if (track.State != TrackState.Lost)
                {
                    track.MarkAsLost();
                    currentLostTracks.Add(track);
                }
            }
        }
        #endregion

        #region Step 4: Init new tracks 
        List<Track> currentRemovedTracks = [];
        {
            // Deal with unconfirmed tracks, usually tracks with only one beginning frame
            var dists = CalcIouDistance(nonActiveTracks, remainDetTracks);
            LinearAssignment(dists, nonActiveTracks.Count, remainDetTracks.Length, 0.7f,
                             out var matchesIdx,
                             out var unmatchUnconfirmedIdx,
                             out var unmatchDetectionIdx);

            foreach (var matchIdx in matchesIdx)
            {
                nonActiveTracks[matchIdx[0]].Update(remainDetTracks[matchIdx[1]], _frameId);
                currentTrackedTracks.Add(nonActiveTracks[matchIdx[0]]);
            }

            foreach (var unmatchIdx in unmatchUnconfirmedIdx)
            {
                var track = nonActiveTracks[unmatchIdx];
                track.MarkAsRemoved();
                currentRemovedTracks.Add(track);
            }

            // Add new stracks
            foreach (var unmatchIdx in unmatchDetectionIdx)
            {
                var track = remainDetTracks[unmatchIdx];
                if (track.Score < _highThresh)
                    continue;
                
                _trackIdCount++;
                track.Activate(_frameId, _trackIdCount);
                currentTrackedTracks.Add(track);
            }
        }
        #endregion

        #region Step 5: Update state
        foreach (var lostTrack in _lostTracks)
        {
            if (_frameId - lostTrack.FrameId > _maxTimeLost)
            {
                lostTrack.MarkAsRemoved();
                currentRemovedTracks.Add(lostTrack);
            }
        }

        var trackedTracks = currentTrackedTracks.Union(refindTracks).ToArray();
        var lostTracks = _lostTracks.Except(trackedTracks).Union(currentLostTracks).Except(_removedTracks).ToArray();
        _removedTracks = _removedTracks.Union(currentRemovedTracks).ToList();
        RemoveDuplicateStracks(trackedTracks, lostTracks);
        #endregion

        return _trackedTracks.Where(track => track.IsActivated).ToArray();
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="aTracks"></param>
    /// <param name="bTracks"></param>
    /// <param name="aResults"></param>
    /// <param name="bResults"></param>
    void RemoveDuplicateStracks(IList<Track> aTracks, IList<Track> bTracks)
    {
        _trackedTracks.Clear();
        _lostTracks.Clear();

        List<(int, int)> overlappingCombinations;
        var ious = CalcIouDistance(aTracks, bTracks);

        if (ious is null)
            overlappingCombinations = [];
        else
        {
            var rows = ious.GetLength(0);
            var cols = ious.GetLength(1);
            overlappingCombinations = new(rows * cols / 2);
            for (var i = 0; i < rows; i++)
                for (var j = 0; j < cols; j++)
                    if (ious[i, j] < 0.15f)
                        overlappingCombinations.Add((i, j));
        }

        var aOverlapping = aTracks.Select(x => false).ToArray();
        var bOverlapping = bTracks.Select(x => false).ToArray();

        foreach (var (aIdx, bIdx) in overlappingCombinations)
        {
            var timep = aTracks[aIdx].FrameId - aTracks[aIdx].StartFrameId;
            var timeq = bTracks[bIdx].FrameId - bTracks[bIdx].StartFrameId;
            if (timep > timeq)
                bOverlapping[bIdx] = true;
            else
                aOverlapping[aIdx] = true;
        }

        for (var ai = 0; ai < aTracks.Count; ai++)
            if (!aOverlapping[ai])
                _trackedTracks.Add(aTracks[ai]);

        for (var bi = 0; bi < bTracks.Count; bi++)
            if (!bOverlapping[bi])
                _lostTracks.Add(bTracks[bi]);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="costMatrix"></param>
    /// <param name="costMatrixSize"></param>
    /// <param name="costMatrixSizeSize"></param>
    /// <param name="thresh"></param>
    /// <param name="matches"></param>
    /// <param name="aUnmatched"></param>
    /// <param name="bUnmatched"></param>
    void LinearAssignment(float[,]? costMatrix, int costMatrixSize, int costMatrixSizeSize, float thresh, out IList<int[]> matches, out IList<int> aUnmatched, out IList<int> bUnmatched)
    {
        matches = new List<int[]>();
        if (costMatrix is null)
        {
            aUnmatched = Enumerable.Range(0, costMatrixSize).ToArray();
            bUnmatched = Enumerable.Range(0, costMatrixSizeSize).ToArray();
            return;
        }

        bUnmatched = new List<int>();
        aUnmatched = new List<int>();

        var (rowsol, colsol) = Lapjv.Exec(costMatrix, true, thresh);

        for (var i = 0; i < rowsol.Length; i++)
        {
            if (rowsol[i] >= 0)
                matches.Add([i, rowsol[i]]);
            else
                aUnmatched.Add(i);
        }

        for (var i = 0; i < colsol.Length; i++)
            if (colsol[i] < 0)
                bUnmatched.Add(i);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="aRects"></param>
    /// <param name="bRects"></param>
    /// <returns></returns>
    static float[,]? CalcIous(IList<RectBox> aRects, IList<RectBox> bRects)
    {
        if (aRects.Count * bRects.Count == 0) return null;

        var ious = new float[aRects.Count, bRects.Count];
        for (var bi = 0; bi < bRects.Count; bi++)
            for (var ai = 0; ai < aRects.Count; ai++)
                ious[ai, bi] = bRects[bi].CalcIoU(aRects[ai]);

        return ious;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="aTtracks"></param>
    /// <param name="bTracks"></param>
    /// <returns></returns>
    static float[,]? CalcIouDistance(IEnumerable<Track> aTtracks, IEnumerable<Track> bTracks)
    {
        var aRects = aTtracks.Select(x => x.RectBox).ToArray();
        var bRects = bTracks.Select(x => x.RectBox).ToArray();

        var ious = CalcIous(aRects, bRects);
        if (ious is null) return null;

        var rows = ious.GetLength(0);
        var cols = ious.GetLength(1);
        var matrix = new float[rows, cols];
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                matrix[i, j] = 1 - ious[i, j];

        return matrix;
    }
}
