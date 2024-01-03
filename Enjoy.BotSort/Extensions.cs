using NumpyDotNet;
using NumpyLib;
using OpenCvSharp;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Enjoy.BotSort;

public static class Extensions
{
    static readonly Dictionary<(int Channel, NPY_TYPES TypeNum), MatType> NPYTypeMappings = new()
    {
        {(1, NPY_TYPES.NPY_UBYTE) , MatType.CV_8UC1},
        {(1, NPY_TYPES.NPY_UINT16), MatType.CV_16UC1},
        {(1, NPY_TYPES.NPY_BYTE) , MatType.CV_8SC1},
        {(1, NPY_TYPES.NPY_INT16) , MatType.CV_16SC1},
        {(1, NPY_TYPES.NPY_INT32) , MatType.CV_32SC1},
        {(1, NPY_TYPES.NPY_FLOAT) , MatType.CV_32FC1},
        {(1, NPY_TYPES.NPY_DOUBLE) , MatType.CV_64FC1},

        {(2, NPY_TYPES.NPY_UBYTE ), MatType.CV_8UC2},
        {(2, NPY_TYPES.NPY_UINT16 ), MatType.CV_16UC2},
        {(2, NPY_TYPES.NPY_BYTE ), MatType.CV_8SC2},
        {(2, NPY_TYPES.NPY_INT16 ), MatType.CV_16SC2},
        {(2, NPY_TYPES.NPY_INT32 ), MatType.CV_32SC2},
        {(2, NPY_TYPES.NPY_FLOAT ), MatType.CV_32FC2},
        {(2, NPY_TYPES.NPY_DOUBLE ), MatType.CV_64FC2},

        {(3, NPY_TYPES.NPY_UBYTE ), MatType.CV_8UC3},
        {(3, NPY_TYPES.NPY_UINT16 ), MatType.CV_16UC3},
        {(3, NPY_TYPES.NPY_BYTE ), MatType.CV_8SC3},
        {(3, NPY_TYPES.NPY_INT16 ), MatType.CV_16SC3},
        {(3, NPY_TYPES.NPY_INT32 ), MatType.CV_32SC3},
        {(3, NPY_TYPES.NPY_FLOAT ), MatType.CV_32FC3},
        {(3, NPY_TYPES.NPY_DOUBLE ), MatType.CV_64FC3},

        {(4, NPY_TYPES.NPY_UBYTE ), MatType.CV_8UC4},
        {(4, NPY_TYPES.NPY_UINT16 ), MatType.CV_16UC4},
        {(4, NPY_TYPES.NPY_BYTE ), MatType.CV_8SC4},
        {(4, NPY_TYPES.NPY_INT16 ), MatType.CV_16SC4},
        {(4, NPY_TYPES.NPY_INT32 ), MatType.CV_32SC4},
        {(4, NPY_TYPES.NPY_FLOAT ), MatType.CV_32FC4},
        {(4, NPY_TYPES.NPY_DOUBLE ), MatType.CV_64FC4},
    };

    /// <summary>
    /// 多维数组转OpenCV图像矩阵
    /// </summary>
    /// <param name="array"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException">最低1通道，最多4通道</exception>
    /// <exception cref="NotSupportedException">行列数不能超0x7fffffff</exception>
    public static Mat ToImageMat(this ndarray array) 
    {
        var rank = array.shape.iDims is null ? -1 : array.shape.iDims.Length;
        if (rank <= 1 || rank >= 4)
            throw new ArgumentOutOfRangeException($"Converting from ndarray to Mat with shape with rank {rank} has not been supported.");

        var rows = (int)array.shape[0];
        var cols = (int)array.shape[1];
        if (rows > int.MaxValue || cols > int.MaxValue)
            throw new ArgumentOutOfRangeException($"The shape {array.shape} is too large to convert to ndarray");

        var format = (Channel: rank == 2 ? 1 : (int)array.shape[2], array.TypeNum);
        if (NPYTypeMappings.TryGetValue(format, out var matType))
        {
            var mat = new Mat(rows, cols, matType);
            switch (array.TypeNum)
            {
                case NPY_TYPES.NPY_BYTE:
                    mat.SetArray(array.AsSByteArray());
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    mat.SetArray(array.AsByteArray());
                    break;
                case NPY_TYPES.NPY_INT16:
                    mat.SetArray(array.AsInt16Array());
                    break;
                case NPY_TYPES.NPY_UINT16:
                    mat.SetArray(array.AsUInt16Array());
                    break;
                case NPY_TYPES.NPY_INT32:
                    mat.SetArray(array.AsInt32Array());
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    mat.SetArray(array.AsFloatArray());
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    mat.SetArray(array.AsDoubleArray());
                    break;
                default:
                    throw new NotSupportedException($"Type {array.Dtype.name} is not supported to convert to Mat.");
            }
            return mat;
        }
        else
            throw new NotSupportedException($"{format.Channel} channels data is not supported by opencv.");
    }

    /// <summary>
    /// 多维数组转OpenCV多维矩阵
    /// </summary>
    /// <param name="array"></param>
    /// <returns>异常时为空矩阵</returns>
    public static Mat AsMat(this ndarray array)
    {
        var rank = array.shape.iDims is null ? -1 : array.shape.iDims.Length;
        if (rank <= 1 || rank >= 4)
            return new Mat();

        var rows = (int)array.shape[0];
        var cols = (int)array.shape[1];
        if (rows > int.MaxValue || cols > int.MaxValue)
            return new Mat();

        var format = (Channel: rank == 2 ? 1 : (int)array.shape[2], array.TypeNum);
        if (NPYTypeMappings.TryGetValue(format, out var matType))
        {
            var mat = new Mat(rows, cols, matType);
            switch (array.TypeNum)
            {
                case NPY_TYPES.NPY_BYTE:
                    mat.SetArray(array.AsSByteArray());
                    break;
                case NPY_TYPES.NPY_UBYTE:
                    mat.SetArray(array.AsByteArray());
                    break;
                case NPY_TYPES.NPY_INT16:
                    mat.SetArray(array.AsInt16Array());
                    break;
                case NPY_TYPES.NPY_UINT16:
                    mat.SetArray(array.AsUInt16Array());
                    break;
                case NPY_TYPES.NPY_INT32:
                    mat.SetArray(array.AsInt32Array());
                    break;
                case NPY_TYPES.NPY_FLOAT:
                    mat.SetArray(array.AsFloatArray());
                    break;
                case NPY_TYPES.NPY_DOUBLE:
                    mat.SetArray(array.AsDoubleArray());
                    break;
                default:
                    return new Mat();
            }
            return mat;
        }

        return new Mat();
    }

    public static ndarray ToNDArray(this Mat mat)
    {
        switch (mat.Type().Depth)
        {
            case MatType.CV_8U:
                if (mat.GetRectangularArray<sbyte>(out var cv8u))
                {
                    return np.array(cv8u);
                }
                throw new NotSupportedException("Cannot convert to sbyte martix.");
            case MatType.CV_8S:
                if (mat.GetRectangularArray<byte>(out var cv8s))
                {
                    return np.array(cv8s);
                }
                throw new NotSupportedException("Cannot convert to byte martix.");
            case MatType.CV_16U:
                if (mat.GetRectangularArray<ushort>(out var cv16u))
                {
                    return np.array(cv16u);
                }
                throw new NotSupportedException("Cannot convert to ushort martix.");
            case MatType.CV_16S:
                if (mat.GetRectangularArray<short>(out var cv16s))
                {
                    return np.array(cv16s);
                }
                throw new NotSupportedException("Cannot convert to short martix.");
            case MatType.CV_32S:
                if (mat.GetRectangularArray<int>(out var cv32s))
                {
                    return np.array(cv32s);
                }
                throw new NotSupportedException("Cannot convert to int martix.");
            case MatType.CV_32F:
                if (mat.GetRectangularArray<float>(out var cv32f))
                {
                    return np.array(cv32f);
                }
                throw new NotSupportedException("Cannot convert to float martix.");
            case MatType.CV_64F:
                if (mat.GetRectangularArray<double>(out var cv64f))
                {
                    return np.array(cv64f);
                }
                throw new NotSupportedException("Cannot convert to double martix.");
            default:
                throw new NotSupportedException($"{mat.Type()} type is not supported by opencv adapter.");
        }
    }

    public static InputArray ToInputArray(this ndarray? array)
    {
        ArgumentNullException.ThrowIfNull(array);
        return (InputArray)array.ToImageMat();
    }

    public static ndarray AsArray(this IEnumerable<ITuple> tuples, dtype? dtype = null)
    {
        var size = tuples.Min(x => x.Length);
        var count = tuples.Count();
        var data = new object?[count, size];
        var i = 0;
        foreach (var tuple in tuples)
        {
            for (var j = 0; j < size; j++)
                data[i, j] = tuple[j];
            i++;
        }

        return np.array(data, dtype);
    }

    public static T[] AsArray<T>(this ITuple tuple, T nullValue, (int Start, int Count)? slice = null) where T : IParsable<T>
    {
        var size = slice?.Count ?? tuple.Length;
        if (size < 1)
            return [];

        var index = slice?.Start ?? 0;
        var end = index + size;
        var result = new T[size];
        for (; index < end; index++)
        {
            var t = tuple[index];
            if (t is null)
                result[index] = nullValue;
            else if (t is T v1)
                result[index] = v1;
            else if (T.TryParse(Convert.ToString(t, CultureInfo.InvariantCulture), CultureInfo.InvariantCulture, out var v2))
                result[index] = v2;
            else
                result[index] = nullValue;
        }
        return result;
    }

    public static T?[] AsArray<T>(this ITuple tuple, (int Start, int Count)? slice = null) where T : IParsable<T>
    {
        var size = slice?.Count ?? tuple.Length;
        if (size < 1)
            return [];

        var index = slice?.Start ?? 0;
        var end = index + size;
        var result = new T?[size];
        for (; index < end; index++)
        {
            var t = tuple[index];
            if (t is null)
                result[index] = default;
            else if (t is T v1)
                result[index] = v1;
            else if (T.TryParse(Convert.ToString(t, CultureInfo.InvariantCulture), CultureInfo.InvariantCulture, out var v2))
                result[index] = v2;
            else
                result[index] = default;
        }
        return result;
    }

    public static T? At<T>(this ITuple tuple, int index) where T : IParsable<T>
    {
        var t = tuple[index];
        if (t is null)
            return default;
        else if (t is T v1)
            return v1;
        else if (T.TryParse(Convert.ToString(t, CultureInfo.InvariantCulture), CultureInfo.InvariantCulture, out var v2))
            return v2;
        else
            return default;
    }

    public static T At<T>(this ITuple tuple, int index, T nullValue) where T : IParsable<T>
    {
        var t = tuple[index];
        if (t is null)
            return nullValue;
        else if (t is T v1)
            return v1;
        else if (T.TryParse(Convert.ToString(t, CultureInfo.InvariantCulture), CultureInfo.InvariantCulture, out var v2))
            return v2;
        else
            return nullValue;
    }
}
