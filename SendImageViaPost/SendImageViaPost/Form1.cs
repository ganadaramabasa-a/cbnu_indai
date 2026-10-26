using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Net.WebRequestMethods;

namespace SendImageViaPost
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private async void simpleButton1_Click(object sender, EventArgs e)
        {
            // OCR(Text Spotting) 서버 주소는 환경변수 OCR_SERVER_URL로 지정, 없으면 로컬 서버 사용
            string url = Environment.GetEnvironmentVariable("OCR_SERVER_URL") ?? "http://localhost:5013/textSpotting2";

            OpenFileDialog dialog = new OpenFileDialog();
            dialog.Filter = "Image Files|*.jpg;*.jpeg;*.png;*.bmp";
            if (dialog.ShowDialog() != DialogResult.OK)
                return;
            string imgPath = dialog.FileName;

            HttpClient httpClient = new HttpClient();
            FileStream fileStream = new FileStream(imgPath, FileMode.Open);
            byte[] fileBytes = new byte[fileStream.Length];

            await fileStream.ReadAsync(fileBytes, 0, fileBytes.Length);

            //Create JSON Request
            var content = new { image = Convert.ToBase64String(fileBytes) }; // Encoding image to Base64 and include to JSON request
            string json = JsonSerializer.Serialize(content);

            // Create HTTP request
            HttpRequestMessage requestMessage = new HttpRequestMessage(HttpMethod.Post, url);
            requestMessage.Content = new StringContent(json, Encoding.UTF8, "application/json");

            HttpResponseMessage responseMessage = await httpClient.SendAsync(requestMessage);
            responseMessage.EnsureSuccessStatusCode(); // Ensure the response is success
            string responseBody = await responseMessage.Content.ReadAsStringAsync();

        }
    }
}
