using System.Text.Json.Serialization;

namespace Enjoy.Test;

[JsonNumberHandling(JsonNumberHandling.AllowReadingFromString)]
public class DetectionResult
{
    [JsonPropertyName("frame_id")]
    public int FrameId { get; set; }

    [JsonPropertyName("prob")]
    public float Prob { get; set; }

    [JsonPropertyName("x")]
    public float X { get; set; }

    [JsonPropertyName("y")]
    public float Y { get; set; }

    [JsonPropertyName("width")]
    public float Width { get; set; }

    [JsonPropertyName("height")]
    public float Height { get; set; }
}
