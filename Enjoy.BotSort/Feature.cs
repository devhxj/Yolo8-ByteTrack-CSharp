namespace Enjoy.BotSort;

public record Feature(int MaxCorners = 1000,
                double QualityLevel = 0.01,
                int MinDistance = 1,
                int BlockSize = 3,
                bool UseHarrisDetector = false,
                double K = 0.04)
{

    public double[] ToArray()
    {
        return new double[] 
        {
            MaxCorners,
            QualityLevel,
            MinDistance,
            BlockSize,
            UseHarrisDetector?1:0,
            K
        };
    }
}
