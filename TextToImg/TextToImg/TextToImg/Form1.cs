using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace TextToImg
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            DataTable dt = new DataTable();
            DataRow row = null;
            dt.Columns.Add(new DataColumn("WAFER_NO"));
            dt.Columns.Add(new DataColumn("ROW_NO"));
            dt.Columns.Add(new DataColumn("DETAIL"));
            row = dt.NewRow();
            row["WAFER_NO"] = "1";
            row["ROW_NO"] = 1;
            row["DETAIL"] = "010";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "1";
            row["ROW_NO"] = 2;
            row["DETAIL"] = "111";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "1";
            row["ROW_NO"] = 3;
            row["DETAIL"] = "020";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "2";
            row["ROW_NO"] = 1;
            row["DETAIL"] = "030";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "2";
            row["ROW_NO"] = 2;
            row["DETAIL"] = "211";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "2";
            row["ROW_NO"] = 3;
            row["DETAIL"] = "020";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "3";
            row["ROW_NO"] = 1;
            row["DETAIL"] = "030";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "3";
            row["ROW_NO"] = 2;
            row["DETAIL"] = "211";
            dt.Rows.Add(row);

            row = dt.NewRow();
            row["WAFER_NO"] = "3";
            row["ROW_NO"] = 3;
            row["DETAIL"] = "020";
            dt.Rows.Add(row);

            gridControl1.DataSource = dt;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string path = System.IO.Path.Combine(Application.StartupPath, "wafer_map.png");
            DrawText(path);
            pictureBox1.ImageLocation = path;
            pictureBox1.Refresh();
        }

        private void DrawText(string v)
        {
            Font font = new Font("Consolas", 20);
            StringFormat sf = new StringFormat();
            sf.Trimming = StringTrimming.Word;

            Image img = new Bitmap(100, 100);
            Graphics drawing = Graphics.FromImage(img);
            SizeF textSize = drawing.MeasureString("■", font, 100);
            drawing.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            drawing.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBilinear;
            drawing.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
            drawing.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;

            float width = 0;
            float height = 0;
            float default_cnt = 20;
            string oldWafer = "";
            drawing.Clear(Color.Transparent);
            for (int row = 0; row < gridView1.RowCount; row++)
            {
                Brush textBrush = null;
                string text = gridView1.GetRowCellValue(row, "DETAIL").ToString();
                string waferNo = gridView1.GetRowCellValue(row, "WAFER_NO").ToString();
                int rowNo = Convert.ToInt32(gridView1.GetRowCellValue(row, "ROW_NO"));



                for (int idx = 0; idx < text.Length; idx++)
                {
                    if (text[idx] == '0')
                        textBrush = new SolidBrush(Color.White);
                    else if (text[idx] == '1')
                        textBrush = new SolidBrush(Color.Red);
                    else if (text[idx] == '2')
                        textBrush = new SolidBrush(Color.Blue);
                    else if (text[idx] == '3')
                        textBrush = new SolidBrush(Color.Black);

                    drawing.DrawString("■", font, textBrush, new RectangleF(width, height, 100, 100), sf);
                }
                oldWafer = gridView1.GetRowCellValue(row, "WAFER_NO").ToString();
            }
            img.Save(v);
            img.Dispose();
        }
    }
}
