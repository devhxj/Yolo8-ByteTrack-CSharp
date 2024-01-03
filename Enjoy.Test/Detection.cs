﻿using System.Text.Json.Serialization;

namespace Enjoy.Test;

public class Detection
{
    [JsonPropertyName("name")]
    public string? Name { get; set; }

    [JsonPropertyName("fps")]
    public int FPS { get; set; }

    [JsonPropertyName("track_buffer")]
    public int TrackBuffer { get; set; }

    [JsonPropertyName("results")]
    public DetectionResult[] Results { get; set; } = [];
}
