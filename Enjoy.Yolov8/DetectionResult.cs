using Enjoy.ByteTrack;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using Sdcb.OpenVINO;

namespace Enjoy.Yolo8;

public class DetectionResult(RectBox box,string name, int label, float prob) : IObject
{
    readonly RectBox _box = box;
    readonly int _label = label;
    readonly float _prob = prob;
    readonly string _name = name;

    public RectBox RectBox => _box;

    public int Label => _label;

    public float Prob => _prob;

    public string Name => _name;

    public static DetectionResult[] FromYolov8DetectionResult(ReadOnlySpan<float> tensorData, Shape shape, Size2f sizeRatio, string[] dicts)
    {
        // tensorData: 1x84x8400=705600xF32
        // shape: 1x84x8400, 84=(x, y, width, height)+80 class confidences, 8400=possible object count(code should for loop 8400 first)
        var t = Transpose(tensorData, shape[1], shape[2]);
        List<DetectionResult> detResults = [];

        var objectCount = shape[2];
        var clsRowCount = shape[1];
        if (dicts.Length != clsRowCount - 4) 
            throw new ArgumentException($"dicts length {dicts.Length} does not match shape cls row count{clsRowCount}.");
        for (var i = 0; i < objectCount; i++)
        {
            var rectData = t.AsSpan()[(i * clsRowCount)..(i * clsRowCount + 4)];
            var confidenceInfo = t.AsSpan()[(i * clsRowCount + 4)..(i * clsRowCount + clsRowCount)];
            var maxConfidenceClsId = IndexOfMax(confidenceInfo);
            var confidence = confidenceInfo[maxConfidenceClsId];
            var centerX = rectData[0] * sizeRatio.Width;
            var centerY = rectData[1] * sizeRatio.Height;
            var width = rectData[2] * sizeRatio.Width;
            var height = rectData[3] * sizeRatio.Height;
            detResults.Add(new(
                new RectBox(centerX - width / 2, centerY - height / 2, width, height),
                dicts[maxConfidenceClsId],
                maxConfidenceClsId,
                confidence));
        }

        CvDnn.NMSBoxes(
            detResults.Select(x => x.RectBox.ToRect()), 
            detResults.Select(x => x.Prob), 
            scoreThreshold: 0.5f, 
            nmsThreshold: 0.5f, 
            out int[] indices);
        return detResults.Where((x, i) => indices.Contains(i)).ToArray();
    }

    static int IndexOfMax(ReadOnlySpan<float> data)
    {
        if (data.Length == 0) throw new ArgumentException("The provided data span is null or empty.");

        // 初始化最大值及其索引
        int maxIndex = 0;
        float maxValue = data[0];

        // 遍历跨度查找最大值及其索引
        for (int i = 1; i < data.Length; i++)
        {
            if (data[i] > maxValue)
            {
                maxValue = data[i];
                maxIndex = i;
            }
        }

        // 返回最大值及其索引
        return maxIndex;
    }

    static unsafe float[] Transpose(ReadOnlySpan<float> tensorData, int rows, int cols)
    {
        float[] transposedTensorData = new float[tensorData.Length];

        fixed (float* pTensorData = tensorData)
        {
            fixed (float* pTransposedData = transposedTensorData)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        // Index in the original tensor
                        int index = i * cols + j;

                        // Index in the transposed tensor
                        int transposedIndex = j * rows + i;

                        pTransposedData[transposedIndex] = pTensorData[index];
                    }
                }
            }
        }

        return transposedTensorData;
    }

    public Track ToTrack() => new(_box, _prob, ("label", _label), ("name", Name));
}
