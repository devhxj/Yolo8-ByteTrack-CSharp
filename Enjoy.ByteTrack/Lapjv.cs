namespace Enjoy.ByteTrack;

/// <summary>
/// lapjv算法，线性任务分配
/// </summary>
public sealed class Lapjv
{
    /// <summary>
    /// 分配
    /// </summary>
    /// <param name="cost"></param>
    /// <param name="extendCost"></param>
    /// <param name="costLimit">分配限制</param>
    /// <returns></returns>
    public static (int[] x, int[] y) Exec(float[,] cost,bool extendCost = false, float costLimit = float.PositiveInfinity) 
    {
        var rows = cost.GetLength(0);
        var cols = cost.GetLength(1);
        var n = rows;

        if (extendCost)
        {
            n = rows + cols;
        }

        cost = ExtendCostMatrix(cost, extendCost, costLimit);
        LapjvInternal(n, cost, out int[] x, out int[] y);

        if (n != rows)
        {
            for (int i = 0; i < n; i++)
            {
                if (x[i] >= cols)
                    x[i] = -1;
                if (y[i] >= rows)
                    y[i] = -1;
            }
        }
        Array.Resize(ref x, rows);
        Array.Resize(ref y, cols);

        return (x, y);
    }

    static float[,] ExtendCostMatrix(float[,] cost, bool extendCost, float costLimit)
    {
        var nRows = cost.GetLength(0);
        var nCols = cost.GetLength(1);
        if (extendCost || costLimit < float.PositiveInfinity)
        {
            var n = nRows + nCols;
            var extendedCost = new float[n, n];
            var defaultValue = costLimit < float.PositiveInfinity ? costLimit / 2.0f : MaxValue(cost) + 1;
            for (var i = 0; i < n; i++)
            {
                for (var j = 0; j < n; j++)
                {
                    extendedCost[i, j] = defaultValue;
                }
            }
            for (var i = nRows; i < n; i++)
            {
                for (var j = nCols; j < n; j++)
                {
                    extendedCost[i, j] = 0;
                }
            }
            for (var i = 0; i < nRows; i++)
            {
                for (var j = 0; j < nCols; j++)
                {
                    extendedCost[i, j] = cost[i, j];
                }
            }
            return extendedCost;
        }
        return cost;
    }

    static int LapjvInternal(int n, float[,] cost, out int[] x, out int[] y)
    {
        int ret;
        var freeRows = new int[n];
        var v = new float[n];

        x = new int[n];
        y = new int[n];

        ret = CcrrtDense(n, cost, freeRows, x, y, v);

        int i = 0;
        while (ret > 0 && i < 2)
        {
            ret = CarrDense(n, cost, ret, freeRows, x, y, v);
            i++;
        }

        if (ret > 0)
        {
            ret = CaDense(n, cost, ret, freeRows, x, y, v);
        }

        return ret;
    }

    static float MaxValue(float[,] matrix)
    {
        var maxVal = float.MinValue;
        for (var i = 0; i < matrix.GetLength(0); i++)
        {
            for (var j = 0; j < matrix.GetLength(1); j++)
            {
                if (matrix[i, j] > maxVal)
                {
                    maxVal = matrix[i, j];
                }
            }
        }
        return maxVal;
    }

    static int CcrrtDense(int n, float[,] cost, int[] freeRows, int[] x, int[] y, float[] v)
    {
        var nFreeRows = 0;
        var unique = new bool[n];

        for (var i = 0; i < n; i++)
        {
            x[i] = -1;
            v[i] = float.MaxValue;
            y[i] = 0;
        }

        for (var i = 0; i < n; i++)
        {
            for (var j = 0; j < n; j++)
            {
                var c = cost[i, j];
                if (c < v[j])
                {
                    v[j] = c;
                    y[j] = i;
                }
            }
        }

        for (var i = n - 1; i >= 0; i--)
        {
            var j = y[i];
            if (x[j] < 0)
            {
                x[j] = i;
            }
            else
            {
                unique[j] = false;
                y[i] = -1;
            }
        }

        for (var i = 0; i < n; i++)
        {
            if (x[i] < 0)
            {
                freeRows[nFreeRows++] = i;
            }
            else if (unique[i])
            {
                var j = x[i];
                var min = float.MaxValue;
                for (var j2 = 0; j2 < n; j2++)
                {
                    if (j2 == j) continue;
                    var c = cost[i, j2] - v[j2];
                    if (c < min)
                    {
                        min = c;
                    }
                }
                v[j] -= min;
            }
        }

        return nFreeRows;
    }

    static int CarrDense(int n, float[,] cost, int nFreeRows, int[] freeRows, int[] x, int[] y, float[] v)
    {
        var current = 0;
        var newFreeRows = 0;
        var rrCnt = 0;
        while (current < nFreeRows)
        {
            var freeI = freeRows[current++];
            var j1 = 0;
            var v1 = cost[freeI, 0] - v[0];
            var j2 = -1;
            var v2 = float.MaxValue;
            for (var j = 1; j < n; j++)
            {
                var c = cost[freeI, j] - v[j];
                if (c < v2)
                {
                    if (c >= v1)
                    {
                        v2 = c;
                        j2 = j;
                    }
                    else
                    {
                        v2 = v1;
                        v1 = c;
                        j2 = j1;
                        j1 = j;
                    }
                }
            }

            var i0 = y[j1];
            var v1New = v[j1] - (v2 - v1);
            var v1Lowers = v1New < v[j1];

            if (rrCnt < current * n)
            {
                if (v1Lowers)
                {
                    v[j1] = v1New;
                }
                else if (i0 >= 0 && j2 >= 0)
                {
                    j1 = j2;
                    i0 = y[j2];
                }

                if (i0 >= 0)
                {
                    if (v1Lowers)
                    {
                        freeRows[--current] = i0;
                    }
                    else
                    {
                        freeRows[newFreeRows++] = i0;
                    }
                }
            }
            else
            {
                if (i0 >= 0)
                {
                    freeRows[newFreeRows++] = i0;
                }
            }

            x[freeI] = j1;
            y[j1] = freeI;
            rrCnt++;
        }

        return newFreeRows;
    }

    static int CaDense(int n, float[,] cost, int nFreeRows, int[] freeRows, int[] x, int[] y, float[] v)
    {
        var pred = new int[n];

        for (var index = 0; index < nFreeRows; index++)
        {
            var freeI = freeRows[index];
            var i = -1;
            var k = 0;

            var j = FindPathDense(n, cost, freeI, y, v, pred);
            if (j < 0 || j >= n)
            {
                throw new InvalidOperationException("断言失败：j超出范围");
            }

            while (i != freeI)
            {
                i = pred[j];
                y[j] = i;
                SwapIndices(ref j, ref x[i]);
                k++;
                if (k >= n)
                {
                    throw new InvalidOperationException("断言失败: k >= n");
                }
            }
        }

        return 0;
    }

    static void SwapIndices(ref int a, ref int b) => (b, a) = (a, b);

    static int FindPathDense(int n, float[,] cost, int startI, int[] y, float[] v, int[] pred)
    {
        int lo = 0, hi = 0;
        var finalJ = -1;
        var nReady = 0;
        var cols = new int[n];
        var d = new float[n];

        for (var i = 0; i < n; i++)
        {
            cols[i] = i;
            pred[i] = startI;
            d[i] = cost[startI, i] - v[i];
        }

        while (finalJ == -1)
        {
            if (lo == hi)
            {
                nReady = lo;
                hi = FindDense(n, lo, d, cols, y);

                for (int k = lo; k < hi; k++)
                {
                    int j = cols[k];
                    if (y[j] < 0)
                    {
                        finalJ = j;
                    }
                }
            }
            if (finalJ == -1)
            {
                finalJ = ScanDense(n, cost, ref lo, ref hi, d, cols, pred, y, v);
            }
        }

        var mind = d[cols[lo]];
        for (var k = 0; k < nReady; k++)
        {
            var j = cols[k];
            v[j] += d[j] - mind;
        }

        return finalJ;
    }

    static int FindDense(int n, int lo, float[] d, int[] cols, int[] y)
    {
        var hi = lo + 1;
        var mind = d[cols[lo]];

        for (var k = hi; k < n; k++)
        {
            var j = cols[k];
            if (d[j] <= mind)
            {
                if (d[j] < mind)
                {
                    hi = lo;
                    mind = d[j];
                }
                cols[k] = cols[hi];
                cols[hi++] = j;
            }
        }

        return hi;
    }

    static int ScanDense(int n, float[,] cost, ref int lo, ref int hi, float[] d, int[] cols, int[] pred, int[] y, float[] v)
    {
        float h, cred_ij;

        while (lo != hi)
        {
            var j = cols[lo++];
            var i = y[j];
            var mind = d[j];
            h = cost[i, j] - v[j] - mind;

            for (var k = hi; k < n; k++)
            {
                j = cols[k];
                cred_ij = cost[i, j] - v[j] - h;
                if (cred_ij < d[j])
                {
                    d[j] = cred_ij;
                    pred[j] = i;
                    if (cred_ij == mind)
                    {
                        if (y[j] < 0)
                        {
                            return j;
                        }
                        cols[k] = cols[hi];
                        cols[hi++] = j;
                    }
                }
            }
        }
        return -1;
    }
}
