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
    public partial class Form2 : Form
    {
        public Form2()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            short a;
            a = (short)Convert.ToInt32(textBox1.Text);
            textBox2.Text = a.ToString();

            do
            {

            } while (true);

            while (true)
            {

            }
        }
    }
}
