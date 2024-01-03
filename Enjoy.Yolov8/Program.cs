using Enjoy.ByteTrack;
using Enjoy.Yolo8;
using OpenCvSharp;
using Sdcb.OpenVINO;
using Sdcb.OpenVINO.Extensions.OpenCvSharp4;
using System.Diagnostics;
using System.Xml.Linq;
using System.Xml.XPath;

var range = new string[] { "person" };//{ "car", "bus", "truck" };
var model = @".\yolov8n_openvino_model\yolov8n.xml";
var dicts = XDocument.Load(model)
    .XPathSelectElement(@"/net/rt_info/model_info/labels")!.Attribute("value")!.Value
    .Split(' ');

using var vc = new VideoCapture(@".\data\palace.mp4");//"D:\\test.mp4"
using var rawModel = OVCore.Shared.ReadModel(model);
using var pp = rawModel.CreatePrePostProcessor();
using var inputInfo = pp.Inputs.Primary;
inputInfo.TensorInfo.Layout = Layout.NHWC;
inputInfo.ModelInfo.Layout = Layout.NCHW;

using var m = pp.BuildModel();
using var cm = OVCore.Shared.CompileModel(m, "GPU");
using var ir = cm.CreateInferRequest();

var inputShape = m.Inputs.Primary.Shape;
var sizeRatio = new Size2f(1f * vc.FrameWidth / inputShape[2], 1f * vc.FrameHeight / inputShape[1]);
var tracker = new ByteTracker((int)vc.Fps, 240);
while (vc.Grab())
{
    using var src = vc.RetrieveMat();
    if (src.Empty()) continue;

    Stopwatch stopwatch = new();
    using var resized = src.Resize(new Size(inputShape[2], inputShape[1]));
    using var f32 = new Mat();
    resized.ConvertTo(f32, MatType.CV_32FC3, 1.0 / 255);

    using var input = f32.AsTensor();
    ir.Inputs.Primary = input;

    var preprocessTime = stopwatch.Elapsed.TotalMilliseconds;
    stopwatch.Restart();

    ir.Run();
    var inferTime = stopwatch.Elapsed.TotalMilliseconds;
    stopwatch.Restart();

    using var output = ir.Outputs.Primary;
    var data = output.GetData<float>();
    var results = DetectionResult.FromYolov8DetectionResult(data, output.Shape, sizeRatio, dicts);
    var postprocessTime = stopwatch.Elapsed.TotalMilliseconds;
    stopwatch.Stop();
    var totalTime = preprocessTime + inferTime + postprocessTime;

    Cv2.PutText(src, $"Preprocess: {preprocessTime:F2}ms", new Point(10, 20), HersheyFonts.HersheyPlain, 1, Scalar.Red);
    Cv2.PutText(src, $"Infer: {inferTime:F2}ms", new Point(10, 40), HersheyFonts.HersheyPlain, 1, Scalar.Red);
    Cv2.PutText(src, $"Postprocess: {postprocessTime:F2}ms", new Point(10, 60), HersheyFonts.HersheyPlain, 1, Scalar.Red);
    Cv2.PutText(src, $"Total: {totalTime:F2}ms", new Point(10, 80), HersheyFonts.HersheyPlain, 1, Scalar.Red);

    var trackOutputs = tracker.Update(results);
    foreach (var t in trackOutputs)
    {
        if (range.Length > 0 && !range.Contains(t["name"])) continue;

        var rect = t.RectBox.ToRect();
        Cv2.PutText(src, $"{t.TrackId}:{t.Score:P0}", rect.TopLeft, HersheyFonts.HersheyTriplex, 1, Scalar.Blue);
        Cv2.Rectangle(src, rect, Scalar.Blue, thickness: 2);
    }

    Cv2.ImShow("frame", src);
    Cv2.WaitKey(1);
}
