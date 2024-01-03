using NumpyDotNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Enjoy.Tracker.Trackers;

public class Track
{
    public Track(ndarray mean, ndarray covariance)
    {
        Mean = mean;
        Covariance = covariance;
    }

    public int smooth_feat { get; }
    public  ndarray Mean { get; }
    public ndarray Covariance { get; }
}
