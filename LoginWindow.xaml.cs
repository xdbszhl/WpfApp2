using System;
using System.Data.SqlClient;
using System.Windows;

namespace ImageSegmentation
{
    public partial class LoginWindow : Window
    {
        // 请根据实际情况修改数据库连接字符串
        private string connectionString = "Server=localhost;DataBase=ImageSegmentationDB;Uid=sa;Pwd=123456";

        public LoginWindow()
        {
            InitializeComponent();
        }

        private void LoginButton_Click(object sender, RoutedEventArgs e)
        {
            string username = UsernameTextBox.Text;
            string password = PasswordBox.Password;

            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                string query = "SELECT UserID FROM Users WHERE Username = @Username AND Password = @Password";
                SqlCommand command = new SqlCommand(query, connection);
                command.Parameters.AddWithValue("@Username", username);
                command.Parameters.AddWithValue("@Password", password);

                try
                {
                    connection.Open();
                    SqlDataReader reader = command.ExecuteReader();
                    if (reader.Read())
                    {
                        int userID = reader.GetInt32(0);
                        // 登录成功，打开主窗口并传入 userID
                        MainWindow mainWindow = new MainWindow(userID);
                        mainWindow.Show();
                        this.Close();
                    }
                    else
                    {
                        MessageBox.Show("用户名或密码错误，请重试。", "登录失败", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                    reader.Close();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"登录时出现错误: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }
        private void RegisterButton_Click(object sender, RoutedEventArgs e)
        {
            string username = UsernameTextBox.Text;
            string password = PasswordBox.Password;

            if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password))
            {
                MessageBox.Show("用户名和密码不能为空，请输入完整信息。", "注册失败", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            using (SqlConnection connection = new SqlConnection(connectionString))
            {
                string query = "INSERT INTO Users (Username, Password) VALUES (@Username, @Password)";
                SqlCommand command = new SqlCommand(query, connection);
                command.Parameters.AddWithValue("@Username", username);
                command.Parameters.AddWithValue("@Password", password);

                try
                {
                    connection.Open();
                    command.ExecuteNonQuery();
                    MessageBox.Show("注册成功，请登录。", "注册成功", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (SqlException ex)
                {
                    if (ex.Number == 2627) // 违反唯一约束，用户名已存在
                    {
                        MessageBox.Show("该用户名已被使用，请选择其他用户名。", "注册失败", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                    else
                    {
                        MessageBox.Show($"注册时出现错误: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                    }
                }
            }
        }
    }
}