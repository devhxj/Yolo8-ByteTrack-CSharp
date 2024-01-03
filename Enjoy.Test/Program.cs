using Enjoy.ByteTrack;
using Enjoy.Test;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Diagnostics;
using System.Text.Json;

const double EPS = 1e-2;
const string D_RESULTS_FILE = @".\data\detection_results.json";
const string T_RESULTS_FILE = @".\data\tracking_results.json";

var detectResults = JsonSerializer.Deserialize<Detection>(File.ReadAllText(D_RESULTS_FILE))!;
var trackingResults = JsonSerializer.Deserialize<Tracking>(File.ReadAllText(T_RESULTS_FILE))!;

if (detectResults.Name != trackingResults.Name)
{
    Console.WriteLine($@"The name of the tests are different: [detection_results_name: {detectResults.Name}, tracking_results_name: {trackingResults.Name}]");
}

var inputs = detectResults.Results
    .GroupBy(x => x.FrameId)
    .ToDictionary(x => x.Key, x => x
        .Select(z => new TrackObject(new(z.X, z.Y, z.Width, z.Height), 0, z.Prob))
        .ToArray());

var outputs = trackingResults.Results
    .GroupBy(x => x.FrameId)
    .ToDictionary(x => x.Key, x => x
        .ToDictionary(y => y.TrackId, y => new RectBox(y.X, y.Y, y.Width, y.Height)));


Stopwatch stopwatch = new();
stopwatch.Start();
var tracker = new ByteTracker(detectResults.FPS, detectResults.TrackBuffer);
foreach (var (frameId, objects) in inputs)
{
    var trackOutputs = tracker.Update(objects);

    // Verify between the reference data and the output of the ByteTracker impl
    Assert.AreEqual(trackOutputs.Count, outputs[frameId].Count);
    foreach (var frame in trackOutputs)
    {
        var output = outputs[frameId][frame.TrackId];
        Assert.IsTrue(output.X - frame.RectBox.X < EPS);
        Assert.IsTrue(output.Y - frame.RectBox.Y < EPS);
        Assert.IsTrue(output.Width - frame.RectBox.Width < EPS);
        Assert.IsTrue(output.Height - frame.RectBox.Height < EPS);
    }
    Console.WriteLine($"{(frameId + 1)} passed.");
}

var postprocessTime = stopwatch.Elapsed.TotalMilliseconds;
stopwatch.Stop();

Console.WriteLine($"verify passed,using {postprocessTime} ms.");
Console.ReadLine();