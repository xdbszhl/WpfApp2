using System;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;
using Microsoft.Win32;
using System.Drawing;
using System.Drawing.Imaging;
using OpenCvSharp;  // 引入OpenCVSharp命名空间
using System.Text;
using LiveCharts;
using LiveCharts.Wpf;
using System.Threading.Tasks;
using System.Linq;
using System.Collections.Generic;
using System.Data.SqlClient;
using WpfApp2;
using Size = OpenCvSharp.Size;



namespace ImageSegmentation
{
    public partial class MainWindow : System.Windows.Window
    {
        private Bitmap originalBitmap;
        private Bitmap copyBitmap;

        private int typeIs = -1;

        private int userID;
        private string connectionString = "Server=localhost;DataBase=ImageSegmentationDB;Uid=sa;Pwd=123456";
        private System.Windows.Threading.DispatcherTimer throttleTimer;
        private bool isSegmenting = false;
        public MainWindow(int userID)
        {
            this.userID = userID;
            InitializeComponent();

            // 初始化节流定时器
            throttleTimer = new System.Windows.Threading.DispatcherTimer();
            throttleTimer.Interval = TimeSpan.FromMilliseconds(300); // 300 毫秒的节流间隔
            throttleTimer.Tick += ThrottleTimer_Tick;
        }

        private async void ThrottleTimer_Tick(object sender, EventArgs e)
        {
            throttleTimer.Stop();

            if (!isSegmenting)
            {
                isSegmenting = true;
                await Task.Run(() =>
                {
                    // 在后台线程中执行分割操作
                    Dispatcher.Invoke(() => ApplySegmentation());
                });
                isSegmenting = false;
            }
        }



        private void ViewHistoryButton_Click(object sender, RoutedEventArgs e)
        {
            List<HistoryRecord> historyRecords = new List<HistoryRecord>();

            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                string query = "SELECT OriginalImagePath, SegmentedImagePath FROM SegmentationHistory WHERE UserID = @UserID";
                SqlCommand command = new SqlCommand(query, connection);
                command.Parameters.AddWithValue("@UserID", userID);

                try
                {
                    connection.Open();
                    SqlDataReader reader = command.ExecuteReader();
                    while (reader.Read())
                    {
                        string originalImagePath = reader.GetString(0);
                        string segmentedImagePath = reader.GetString(1);

                        // 输出日志，确认图片路径
                        System.Diagnostics.Debug.WriteLine($"Original Image Path: {originalImagePath}");
                        System.Diagnostics.Debug.WriteLine($"Segmented Image Path: {segmentedImagePath}");

                        historyRecords.Add(new HistoryRecord
                        {
                            OriginalImagePath = originalImagePath,
                            SegmentedImagePath = segmentedImagePath
                        });
                    }
                    reader.Close();
                }
                catch (SqlException ex)
                {
                    MessageBox.Show($"连接数据库时出现错误: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"出现未知错误: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }

            if (historyRecords.Count > 0)
            {
                HistoryWindow historyWindow = new HistoryWindow(historyRecords);
                historyWindow.Show();
            }
            else
            {
                MessageBox.Show("没有找到分割历史记录。", "提示", MessageBoxButton.OK, MessageBoxImage.Information);
            }
        }
        private void LoadImage_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.Filter = "图像文件 (*.jpg;*.png;*.bmp)|*.jpg;*.png;*.bmp";
            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    // 清空之前的分割结果、分类图像和聚类统计信息
                    Class2Image.Source = null;
                    Class3Image.Source = null;
                    ClusterImageListBox.Items.Clear();
                    ClusterStatisticsPieChart.Series.Clear();

                    originalBitmap = new Bitmap(openFileDialog.FileName);
                    Class1Image.Source = ConvertBitmapToBitmapSource(originalBitmap);
                    ApplyPreprocessing();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"加载图片时出现错误:{ex.Message}", "错误");
                }
            }
        }

        private void ApplyPreprocessing()
        {
            if (this.Dispatcher.CheckAccess())
            {
                if (originalBitmap != null)
                {
                    double grayScaleIntensity = GrayScaleIntensitySlider.Value;
                    double contrast = ContrastSlider.Value;

                    Bitmap processedBitmap = originalBitmap.Clone() as Bitmap;

                    // 灰度化处理
                    processedBitmap = ApplyGrayScale(processedBitmap, grayScaleIntensity);

                    // 增强对比度处理，扩大调整范围
                    contrast = Math.Max(0.1, contrast); // 确保对比度不小于 0.1
                    processedBitmap = AdjustContrast(processedBitmap, contrast);

                    // 去噪类型
                    if (typeIs == 0)
                    {
                        processedBitmap = MeanFilter(processedBitmap, 5);
                    }
                    else if (typeIs == 1)
                    {
                        processedBitmap = GaussianFilter(processedBitmap, 5, 1.0);
                    }
                    else if (typeIs == 2)
                    {
                        processedBitmap = BilateralFilter(processedBitmap, 9, 75, 75); // 双边滤波
                    }

                    copyBitmap = processedBitmap;
                    Class2Image.Source = ConvertBitmapToBitmapSource(processedBitmap);
                }
            }
            else
            {
                this.Dispatcher.Invoke(ApplyPreprocessing);
            }
        }



        //private void ApplySegmentation()
        //{
        //    if (copyBitmap == null) return;

        //    Mat src = BitmapToMat(copyBitmap);
        //    int clusterCount = int.Parse(ClusterCountTextBox.Text);

        //    Mat samples = src.Reshape(1, src.Rows * src.Cols);
        //    samples.ConvertTo(samples, MatType.CV_32F);

        //    Mat labels = new Mat();
        //    Mat centers = new Mat();
        //    Cv2.Kmeans(
        //        samples,
        //        clusterCount,
        //        labels,
        //        new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 10, 1.0),
        //        3,
        //        KMeansFlags.RandomCenters,
        //        centers
        //    );

        //    Mat segmentedImage = new Mat(src.Size(), src.Type());
        //    for (int i = 0; i < src.Rows; i++)
        //    {
        //        for (int j = 0; j < src.Cols; j++)
        //        {
        //            int label = labels.Get<int>(i * src.Cols + j);
        //            Vec3b color = new Vec3b((byte)centers.Get<float>(label, 0), (byte)centers.Get<float>(label, 1), (byte)centers.Get<float>(label, 2));
        //            segmentedImage.Set(i, j, color);
        //        }
        //    }

        //    // 将OpenCV的Mat对象转换为Bitmap
        //    Bitmap segmentedBitmap = MatToBitmap(segmentedImage);

        //    // 显示分割后的图像
        //    Class3Image.Source = ConvertBitmapToBitmapSource(segmentedBitmap);
        //    // 统计每个聚类的像素数量
        //    int[] clusterPixelCounts = new int[clusterCount];
        //    for (int i = 0; i < labels.Rows; i++)
        //    {
        //        int label = labels.Get<int>(i, 0);
        //        clusterPixelCounts[label]++;
        //    }

        //    // 显示聚类统计信息
        //    for (int i = 0; i < clusterCount; i++)
        //    {
        //        double percentage = (double)clusterPixelCounts[i] / (src.Rows * src.Cols) * 100;
        //        Console.WriteLine($"聚类 {i}：像素数量 = {clusterPixelCounts[i]}, 占比 = {percentage:0.00}%");
        //    }

        //    // 显示聚类统计到界面
        //    DisplayClusterStatistics(clusterPixelCounts, clusterCount, src.Rows * src.Cols);

        //    // 创建聚类图像
        //    Mat[] clusterMats = new Mat[clusterCount];
        //    for (int i = 0; i < clusterCount; i++)
        //    {
        //        clusterMats[i] = new Mat(src.Size(), src.Type(), Scalar.Black);
        //    }

        //    for (int y = 0; y < src.Rows; y++)
        //    {
        //        for (int x = 0; x < src.Cols; x++)
        //        {
        //            int index = y * src.Cols + x;
        //            int label = labels.Get<int>(index);
        //            Vec3b color = src.At<Vec3b>(y, x); // 保留原始颜色
        //            clusterMats[label].Set(y, x, color);
        //        }
        //    }

        //    // 添加图像到界面
        //    ClusterImageListBox.Items.Clear();
        //    for (int i = 0; i < clusterCount; i++)
        //    {
        //        Bitmap bmp = MatToBitmap(clusterMats[i]);
        //        System.Windows.Controls.Image image = new System.Windows.Controls.Image
        //        {
        //            Source = ConvertBitmapToBitmapSource(bmp),
        //            Width = 200,
        //            Margin = new Thickness(5)
        //        };
        //        ClusterImageListBox.Items.Add(image);
        //    }
        //}

        public class ClusterInfo
        {
            public int Index { get; set; }
            public string Title { get; set; }
            public BitmapSource ImageSource { get; set; }
            public double Percentage { get; set; }
        }
        private void ApplySegmentation()
        {
            if (copyBitmap == null) return;

            Mat src = BitmapToMat(copyBitmap);
            int clusterCount = int.Parse(ClusterCountTextBox.Text);

            Mat samples = src.Reshape(1, src.Rows * src.Cols);
            samples.ConvertTo(samples, MatType.CV_32F);

            Mat labels = new Mat();
            Mat centers = new Mat();
            Cv2.Kmeans(
                samples,
                clusterCount,
                labels,
                new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 10, 1.0),
                3,
                KMeansFlags.RandomCenters,
                centers
            );

            Mat segmentedImage = new Mat(src.Size(), src.Type());
            for (int i = 0; i < src.Rows; i++)
            {
                for (int j = 0; j < src.Cols; j++)
                {
                    int label = labels.Get<int>(i * src.Cols + j);
                    Vec3b color = new Vec3b((byte)centers.Get<float>(label, 0), (byte)centers.Get<float>(label, 1), (byte)centers.Get<float>(label, 2));
                    segmentedImage.Set(i, j, color);
                }
            }

            // 将OpenCV的Mat对象转换为Bitmap
            Bitmap segmentedBitmap = MatToBitmap(segmentedImage);

            // 显示分割后的图像
            Class3Image.Source = ConvertBitmapToBitmapSource(segmentedBitmap);

            // 统计每个聚类的像素数量
            int[] clusterPixelCounts = new int[clusterCount];
            for (int i = 0; i < labels.Rows; i++)
            {
                int label = labels.Get<int>(i, 0);
                clusterPixelCounts[label]++;
            }

            // 显示聚类统计信息
            for (int i = 0; i < clusterCount; i++)
            {
                double percentage = (double)clusterPixelCounts[i] / (src.Rows * src.Cols) * 100;
                Console.WriteLine($"聚类 {i}：像素数量 = {clusterPixelCounts[i]}, 占比 = {percentage:0.00}%");
            }

            // 显示聚类统计到界面
            DisplayClusterStatistics(clusterPixelCounts, clusterCount, src.Rows * src.Cols);

            // 创建聚类图像
            Mat[] clusterMats = new Mat[clusterCount];
            for (int i = 0; i < clusterCount; i++)
            {
                clusterMats[i] = new Mat(src.Size(), src.Type(), Scalar.Black);
            }

            for (int y = 0; y < src.Rows; y++)
            {
                for (int x = 0; x < src.Cols; x++)
                {
                    int index = y * src.Cols + x;
                    int label = labels.Get<int>(index);
                    Vec3b color = src.At<Vec3b>(y, x); // 保留原始颜色
                    clusterMats[label].Set(y, x, color);
                }
            }

            // 计算每个聚类的占比并存储在列表中
            List<ClusterInfo> clusterInfos = new List<ClusterInfo>();
            for (int i = 0; i < clusterCount; i++)
            {
                double percentage = (double)clusterPixelCounts[i] / (src.Rows * src.Cols) * 100;
                Bitmap bmp = MatToBitmap(clusterMats[i]);
                BitmapSource bitmapSource = ConvertBitmapToBitmapSource(bmp);

                clusterInfos.Add(new ClusterInfo
                {
                    Index = i,
                    Title = $"聚类 {i + 1} ({percentage:0.00}%)",
                    ImageSource = bitmapSource,
                    Percentage = percentage
                });
            }

            // 按占比从大到小排序
            clusterInfos.Sort((a, b) => b.Percentage.CompareTo(a.Percentage));

            // 添加图像到界面
            ClusterImageListBox.Items.Clear();
            foreach (var clusterInfo in clusterInfos)
            {
                ClusterImageListBox.Items.Add(clusterInfo);
            }

            // 保存历史分割记录
            SaveSegmentationHistory();
        }

        private void SaveSegmentationHistory()
        {
            if (originalBitmap == null || Class3Image.Source == null) return;

            string originalImagePath = System.IO.Path.GetFullPath("OriginalImages/" + Guid.NewGuid().ToString() + ".bmp");
            string segmentedImagePath = System.IO.Path.GetFullPath("SegmentedImages/" + Guid.NewGuid().ToString() + ".bmp");

            // 保存原始图像
            Directory.CreateDirectory("OriginalImages");
            originalBitmap.Save(originalImagePath, ImageFormat.Bmp);

            // 保存分割后的图像
            Directory.CreateDirectory("SegmentedImages");
            BitmapSource bitmapSource = (BitmapSource)Class3Image.Source;
            BitmapEncoder encoder = new BmpBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
            using (FileStream fileStream = new FileStream(segmentedImagePath, FileMode.Create))
            {
                encoder.Save(fileStream);
            }

            // 插入记录到数据库
            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                string query = "INSERT INTO SegmentationHistory (UserID, OriginalImagePath, SegmentedImagePath) VALUES (@UserID, @OriginalImagePath, @SegmentedImagePath)";
                SqlCommand command = new SqlCommand(query, connection);
                command.Parameters.AddWithValue("@UserID", userID);
                command.Parameters.AddWithValue("@OriginalImagePath", originalImagePath);
                command.Parameters.AddWithValue("@SegmentedImagePath", segmentedImagePath);

                try
                {
                    connection.Open();
                    command.ExecuteNonQuery();
                }
                catch (SqlException ex)
                {
                    MessageBox.Show($"保存历史记录时数据库错误: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"保存历史记录时出现未知错误: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }
        // RBF 核函数
        private double RBFKernel(Mat x, Mat y, double gamma)
        {
            Mat diff = x - y;
            double norm = Cv2.Norm(diff);
            return Math.Exp(-gamma * norm * norm);
        }

        // 多项式核函数
        private double PolynomialKernel(Mat x, Mat y, double degree)
        {
            Mat dotProduct = x.T() * y;
            return Math.Pow(dotProduct.Get<double>(0, 0) + 1, degree);
        }

        // 核 K-means 算法
        private int[] KernelKMeans(Mat samples, int clusterCount, Func<Mat, Mat, double, double> kernelFunction, double kernelParameter)
        {
            int nSamples = samples.Rows;
            int[] labels = new int[nSamples];
            Random random = new Random();

            // 随机初始化聚类中心
            int[] centers = new int[clusterCount];
            for (int i = 0; i < clusterCount; i++)
            {
                centers[i] = random.Next(nSamples);
            }

            bool changed = true;
            while (changed)
            {
                changed = false;
                // 分配样本到最近的聚类中心
                for (int i = 0; i < nSamples; i++)
                {
                    double minDistance = double.MaxValue;
                    int newLabel = -1;
                    for (int j = 0; j < clusterCount; j++)
                    {
                        double distance = kernelFunction(samples.Row(i), samples.Row(centers[j]), kernelParameter);
                        if (distance < minDistance)
                        {
                            minDistance = distance;
                            newLabel = j;
                        }
                    }
                    if (newLabel != labels[i])
                    {
                        labels[i] = newLabel;
                        changed = true;
                    }
                }

                // 更新聚类中心
                for (int j = 0; j < clusterCount; j++)
                {
                    double maxSimilarity = double.MinValue;
                    int newCenter = -1;
                    for (int i = 0; i < nSamples; i++)
                    {
                        if (labels[i] == j)
                        {
                            double similarity = 0;
                            for (int k = 0; k < nSamples; k++)
                            {
                                if (labels[k] == j)
                                {
                                    similarity += kernelFunction(samples.Row(i), samples.Row(k), kernelParameter);
                                }
                            }
                            if (similarity > maxSimilarity)
                            {
                                maxSimilarity = similarity;
                                newCenter = i;
                            }
                        }
                    }
                    if (newCenter != -1)
                    {
                        centers[j] = newCenter;
                    }
                }
            }

            return labels;
        }

        private void DisplayClusterStatistics(int[] clusterPixelCounts, int clusterCount, int totalPixels)
        {
            var pieChartValues = new ChartValues<double>();

            // 计算每个聚类的百分比并填充 ChartValues
            for (int i = 0; i < clusterCount; i++)
            {
                double percentage = (double)clusterPixelCounts[i] / totalPixels * 100;
                pieChartValues.Add(percentage);
            }

            // 清空当前 Series
            ClusterStatisticsPieChart.Series.Clear();

            // 动态生成 PieSeries
            for (int i = 0; i < clusterCount; i++)
            {
                var pieSeries = new PieSeries
                {
                    Title = $"聚类 {i + 1}",
                    Values = new ChartValues<double> { pieChartValues[i] },
                    DataLabels = true,
                    LabelPoint = chartPoint => $"{chartPoint.Y:0.00}%" // 格式化显示
                };

                // 添加到图表
                ClusterStatisticsPieChart.Series.Add(pieSeries);
            }
        }

        //private Bitmap ApplyGrayScale(Bitmap image, double intensity)
        //{
        //    Bitmap newImage = new Bitmap(image.Width, image.Height);
        //    for (int x = 0; x < image.Width; x++)
        //    {
        //        for (int y = 0; y < image.Height; y++)
        //        {
        //            Color originalColor = image.GetPixel(x, y);
        //            // 计算灰度值
        //            int fullGray = (int)(originalColor.R * 0.299 + originalColor.G * 0.587 + originalColor.B * 0.114);
        //            // 线性插值
        //            int newR = (int)(originalColor.R * (1 - intensity) + fullGray * intensity);
        //            int newG = (int)(originalColor.G * (1 - intensity) + fullGray * intensity);
        //            int newB = (int)(originalColor.B * (1 - intensity) + fullGray * intensity);

        //            // 确保颜色值在 0 到 255 之间
        //            newR = Math.Max(0, Math.Min(255, newR));
        //            newG = Math.Max(0, Math.Min(255, newG));
        //            newB = Math.Max(0, Math.Min(255, newB));

        //            Color newColor = Color.FromArgb(originalColor.A, newR, newG, newB);
        //            newImage.SetPixel(x, y, newColor);
        //        }
        //    }
        //    return newImage;
        //}
        private Bitmap ApplyGrayScale(Bitmap image, double intensity)
        {
            // 将 Bitmap 转换为 OpenCV 的 Mat 对象
            Mat src = BitmapToMat(image);

            // 转换为灰度图像
            Mat gray = new Mat();
            Cv2.CvtColor(src, gray, ColorConversionCodes.BGR2GRAY);

            // 创建一个新的 Mat 对象，用于存储结果
            Mat result = new Mat();
            src.CopyTo(result);

            // 遍历每个像素并应用线性插值
            for (int y = 0; y < src.Rows; y++)
            {
                for (int x = 0; x < src.Cols; x++)
                {
                    Vec3b originalColor = src.Get<Vec3b>(y, x);
                    byte grayValue = gray.Get<byte>(y, x);

                    // 线性插值
                    int newR = (int)(originalColor.Item2 * (1 - intensity) + grayValue * intensity);
                    int newG = (int)(originalColor.Item1 * (1 - intensity) + grayValue * intensity);
                    int newB = (int)(originalColor.Item0 * (1 - intensity) + grayValue * intensity);

                    // 确保颜色值在 0 到 255 之间
                    newR = Math.Max(0, Math.Min(255, newR));
                    newG = Math.Max(0, Math.Min(255, newG));
                    newB = Math.Max(0, Math.Min(255, newB));

                    result.Set(y, x, new Vec3b((byte)newB, (byte)newG, (byte)newR));
                }
            }

            // 将处理后的 Mat 对象转换回 Bitmap
            return MatToBitmap(result);
        }

        //private Bitmap AdjustContrast(Bitmap image, double contrast)
        //{
        //    Bitmap newImage = new Bitmap(image.Width, image.Height);
        //    Graphics g = Graphics.FromImage(newImage);

        //    // 对比度调整因子，范围从 0 到 2，1 表示无变化
        //    float factor = (float)contrast;

        //    ColorMatrix colorMatrix = new ColorMatrix(new float[][]
        //    {
        //new float[] { factor, 0, 0, 0, 0 },
        //new float[] { 0, factor, 0, 0, 0 },
        //new float[] { 0, 0, factor, 0, 0 },
        //new float[] { 0, 0, 0, 1, 0 },
        //new float[] { 0, 0, 0, 0, 1 }
        //    });

        //    ImageAttributes attributes = new ImageAttributes();
        //    attributes.SetColorMatrix(colorMatrix);

        //    g.DrawImage(image, new Rectangle(0, 0, image.Width, image.Height), 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, attributes);

        //    g.Dispose();
        //    return newImage;
        //}
        private Bitmap AdjustContrast(Bitmap image, double contrast)
        {
            // 将 Bitmap 转换为 OpenCV 的 Mat 对象
            Mat src = BitmapToMat(image);

            // 调整对比度
            Mat dst = new Mat();
            src.ConvertTo(dst, src.Type(), contrast, 0);

            // 将处理后的 Mat 对象转换回 Bitmap
            return MatToBitmap(dst);
        }
        private BitmapSource ConvertBitmapToBitmapSource(Bitmap bitmap)
        {
            using (MemoryStream memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;
                BitmapImage bitmapImage = new BitmapImage();
                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                return bitmapImage;
            }
        }


        // 在 MainWindow.xaml.cs 中添加双边滤波方法
        private Bitmap BilateralFilter(Bitmap sourceImage, int diameter, double sigmaColor, double sigmaSpace)
        {
            Mat src = BitmapToMat(sourceImage);
            Mat dst = new Mat();
            Cv2.BilateralFilter(src, dst, diameter, sigmaColor, sigmaSpace);
            return MatToBitmap(dst);
        }
        // 均值滤波
        //public static Bitmap MeanFilter(Bitmap sourceImage, int kernelSize)
        //{
        //    Bitmap resultImage = new Bitmap(sourceImage.Width, sourceImage.Height);
        //    int halfKernel = kernelSize / 2;

        //    for (int y = 0; y < sourceImage.Height; y++)
        //    {
        //        for (int x = 0; x < sourceImage.Width; x++)
        //        {
        //            int rSum = 0, gSum = 0, bSum = 0;
        //            int count = 0;

        //            for (int ky = -halfKernel; ky <= halfKernel; ky++)
        //            {
        //                for (int kx = -halfKernel; kx <= halfKernel; kx++)
        //                {
        //                    int newX = x + kx;
        //                    int newY = y + ky;

        //                    if (newX >= 0 && newX < sourceImage.Width && newY >= 0 && newY < sourceImage.Height)
        //                    {
        //                        Color pixelColor = sourceImage.GetPixel(newX, newY);
        //                        rSum += pixelColor.R;
        //                        gSum += pixelColor.G;
        //                        bSum += pixelColor.B;
        //                        count++;
        //                    }
        //                }
        //            }

        //            int r = rSum / count;
        //            int g = gSum / count;
        //            int b = bSum / count;

        //            resultImage.SetPixel(x, y, Color.FromArgb(r, g, b));
        //        }
        //    }

        //    return resultImage;
        //}
        private Bitmap MeanFilter(Bitmap sourceImage, int kernelSize)
        {
            // 将 Bitmap 转换为 OpenCV 的 Mat 对象
            Mat src = BitmapToMat(sourceImage);
            // 创建一个新的 Mat 对象用于存储滤波后的图像
            Mat dst = new Mat();
            // 调用 OpenCV 的 Blur 方法进行均值滤波
            Cv2.Blur(src, dst, new Size(kernelSize, kernelSize));
            // 将处理后的 Mat 对象转换回 Bitmap
            return MatToBitmap(dst);
        }

        // 高斯滤波
        //public static Bitmap GaussianFilter(Bitmap sourceImage, int kernelSize, double sigma)
        //{
        //    Bitmap resultImage = new Bitmap(sourceImage.Width, sourceImage.Height);
        //    int halfKernel = kernelSize / 2;
        //    double[,] kernel = new double[kernelSize, kernelSize];

        //    // 计算高斯核
        //    double sum = 0;
        //    for (int y = -halfKernel; y <= halfKernel; y++)
        //    {
        //        for (int x = -halfKernel; x <= halfKernel; x++)
        //        {
        //            double value = Math.Exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma);
        //            kernel[y + halfKernel, x + halfKernel] = value;
        //            sum += value;
        //        }
        //    }

        //    // 归一化高斯核
        //    for (int y = 0; y < kernelSize; y++)
        //    {
        //        for (int x = 0; x < kernelSize; x++)
        //        {
        //            kernel[y, x] /= sum;
        //        }
        //    }

        //    for (int y = 0; y < sourceImage.Height; y++)
        //    {
        //        for (int x = 0; x < sourceImage.Width; x++)
        //        {
        //            double rSum = 0, gSum = 0, bSum = 0;

        //            for (int ky = -halfKernel; ky <= halfKernel; ky++)
        //            {
        //                for (int kx = -halfKernel; kx <= halfKernel; kx++)
        //                {
        //                    int newX = x + kx;
        //                    int newY = y + ky;

        //                    if (newX >= 0 && newX < sourceImage.Width && newY >= 0 && newY < sourceImage.Height)
        //                    {
        //                        Color pixelColor = sourceImage.GetPixel(newX, newY);
        //                        rSum += pixelColor.R * kernel[ky + halfKernel, kx + halfKernel];
        //                        gSum += pixelColor.G * kernel[ky + halfKernel, kx + halfKernel];
        //                        bSum += pixelColor.B * kernel[ky + halfKernel, kx + halfKernel];
        //                    }
        //                }
        //            }

        //            int r = (int)rSum;
        //            int g = (int)gSum;
        //            int b = (int)bSum;

        //            resultImage.SetPixel(x, y, Color.FromArgb(r, g, b));
        //        }
        //    }

        //    return resultImage;
        //}
        private Bitmap GaussianFilter(Bitmap sourceImage, int kernelSize, double sigma)
        {
            // 将 Bitmap 转换为 OpenCV 的 Mat 对象
            Mat src = BitmapToMat(sourceImage);
            // 创建一个新的 Mat 对象用于存储滤波后的图像
            Mat dst = new Mat();
            // 调用 OpenCV 的 GaussianBlur 方法进行高斯滤波
            Cv2.GaussianBlur(src, dst, new Size(kernelSize, kernelSize), sigma);
            // 将处理后的 Mat 对象转换回 Bitmap
            return MatToBitmap(dst);
        }


        private void DenoisingComboBox_SelectionChanged_1(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
        {
            typeIs = DenoisingComboBox.SelectedIndex;
            ApplyPreprocessing();
            ApplySegmentation();
        }


        private async void GrayScaleIntensitySlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            // 直接调用 ApplyPreprocessing，因为它内部会处理线程问题
            if (originalBitmap == null) return;

            // 直接调用 ApplyPreprocessing，因为它内部会处理线程问题
            ApplyPreprocessing();

            // 如果正在分割，重置定时器
            if (throttleTimer != null && throttleTimer.IsEnabled)
            {
                throttleTimer.Stop();
            }

            // 启动定时器
            if (throttleTimer != null)
            {
                throttleTimer.Start();
            }
        }

        private async void ContrastSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            // 检查相关对象是否为空
            if (originalBitmap == null) return;

            // 直接调用 ApplyPreprocessing，因为它内部会处理线程问题
            ApplyPreprocessing();

            // 如果正在分割，重置定时器
            if (throttleTimer != null && throttleTimer.IsEnabled)
            {
                throttleTimer.Stop();
            }

            // 启动定时器
            if (throttleTimer != null)
            {
                throttleTimer.Start();
            }
        }

        // 将Bitmap转换为OpenCV的Mat对象
        private Mat BitmapToMat(Bitmap bitmap)
        {
            MemoryStream ms = new MemoryStream();
            bitmap.Save(ms, ImageFormat.Bmp);
            byte[] byteArray = ms.ToArray();
            Mat mat = Cv2.ImDecode(byteArray, ImreadModes.Color);
            return mat;
        }

        // 将OpenCV的Mat对象转换为Bitmap
        private Bitmap MatToBitmap(Mat mat)
        {
            Cv2.ImEncode(".bmp", mat, out byte[] buffer);
            using (MemoryStream ms = new MemoryStream(buffer))
            {
                using (Bitmap temp = new Bitmap(ms))
                {
                    return new Bitmap(temp); // 返回副本，避免 GDI+ 使用已释放流
                }
            }
        }


        private void Button_Click(object sender, RoutedEventArgs e)
        {
            ApplySegmentation(); // 调用图像分割方法
        }

        private void Button_Click_1(object sender, RoutedEventArgs e)
        {

        }

        private void ExportSegmentedImage_Click(object sender, RoutedEventArgs e)
        {
            if (Class3Image.Source == null)
            {
                MessageBox.Show("还没有分割后的图像可供导出，请先进行图像分割。", "提示");
                return;
            }

            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.Filter = "PNG Image|*.png|JPEG Image|*.jpg|Bitmap Image|*.bmp";
            saveFileDialog.Title = "保存分割后的图像";
            if (saveFileDialog.ShowDialog() == true)
            {
                try
                {
                    BitmapSource bitmapSource = (BitmapSource)Class3Image.Source;
                    BitmapEncoder encoder;
                    switch (System.IO.Path.GetExtension(saveFileDialog.FileName).ToLower())
                    {
                        case ".png":
                            encoder = new PngBitmapEncoder();
                            break;
                        case ".jpg":
                            encoder = new JpegBitmapEncoder();
                            break;
                        case ".bmp":
                            encoder = new BmpBitmapEncoder();
                            break;
                        default:
                            encoder = new PngBitmapEncoder();
                            break;
                    }

                    encoder.Frames.Add(BitmapFrame.Create(bitmapSource));
                    using (System.IO.FileStream fileStream = new System.IO.FileStream(saveFileDialog.FileName, System.IO.FileMode.Create))
                    {
                        encoder.Save(fileStream);
                    }

                    MessageBox.Show("分割后的图像已成功导出。", "成功");
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"导出图像时出现错误: {ex.Message}", "错误");
                }
            }
        }
    }
}