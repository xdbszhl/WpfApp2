using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace WpfApp2
{
    /// <summary>
    /// HistoryWindow.xaml 的交互逻辑
    /// </summary>
    public partial class HistoryWindow : Window
    {
        public HistoryWindow(List<HistoryRecord> historyRecords)
        {
            InitializeComponent();
            HistoryListView.ItemsSource = historyRecords;
        }
    }

    public class HistoryRecord
    {
        public string OriginalImagePath { get; set; }
        public string SegmentedImagePath { get; set; }
    }
}
