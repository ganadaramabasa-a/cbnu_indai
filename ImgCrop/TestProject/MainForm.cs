using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Media;
using System.Windows.Forms;

namespace TestProject
{
    /// <summary>
    /// 메인 폼
    /// </summary>
    public partial class MainForm : Form
    {
        //////////////////////////////////////////////////////////////////////////////////////////////////// Field
        ////////////////////////////////////////////////////////////////////////////////////////// Private

        #region Field

        /// <summary>
        /// 소스 비트맵
        /// </summary>
        private Bitmap sourceBitmap = null;

        /// <summary>
        /// X0
        /// </summary>
        private int x0;
        
        /// <summary>
        /// Y0
        /// </summary>
        private int y0;
        
        /// <summary>
        /// X1
        /// </summary>
        private int x1;
        
        /// <summary>
        /// Y1
        /// </summary>
        private int y1;

        /// <summary>
        /// 영역 선택 여부
        /// </summary>
        private bool selectingArea = false;

        /// <summary>
        /// 선택 비트맵
        /// </summary>
        private Bitmap selectedBitmap = null;

        /// <summary>
        /// 선택 그래픽스
        /// </summary>
        private Graphics selectedGraphics = null;

        /// <summary>
        /// 선택 사각형
        /// </summary>
        private Rectangle selectedRectangle;

        /// <summary>
        /// 선택 완료 여부
        /// </summary>
        private bool slectionCompleted = false;

        #endregion

        //////////////////////////////////////////////////////////////////////////////////////////////////// Constructor
        ////////////////////////////////////////////////////////////////////////////////////////// Public

        #region 생성자 - MainForm()

        /// <summary>
        /// 생성자
        /// </summary>
        public MainForm()
        {
            InitializeComponent();

            this.pictureBox.Image = Image.FromFile("sample.jpg");

            #region 이벤트를 설정한다.

            Load                                       += Form_Load;
            this.exitMenuItem.Click                    += exitMenuItem_Click;
            this.copyMenuItem.Click                    += copyMenuItem_Click;
            this.cutMenuItem.Click                     += cutMenuItem_Click;
            this.pasteCenterMenuItem.Click             += pasteCenterMenuItem_Click;
            this.pasteStretchMenuItem.Click            += pasteStretchMenuItem_Click;
            this.sizeModeComboBox.SelectedIndexChanged += sizeModeComboBox_SelectedIndexChanged;
            this.pictureBox.MouseDown                  += pictureBox_MouseDown;
            this.pictureBox.MouseMove                  += pictureBox_MouseMove;
            this.pictureBox.MouseUp                    += pictureBox_MouseUp;

            #endregion
        }

        #endregion

        //////////////////////////////////////////////////////////////////////////////////////////////////// Method
        ////////////////////////////////////////////////////////////////////////////////////////// Private
        //////////////////////////////////////////////////////////////////////////////// Event

        #region 폼 로드시 처리하기 - Form_Load(sender, e)

        /// <summary>
        /// 폼 로드시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void Form_Load(object sender, EventArgs e)
        {
            this.sizeModeComboBox.SelectedIndex = 1;

            this.sourceBitmap = new Bitmap(this.pictureBox.Image);

            KeyPreview = true;
        }

        #endregion
        #region 폼 키 PRESS 처리하기 - Form_KeyPress(sender, e)

        /// <summary>
        /// 폼 키 PRESS 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void Form_KeyPress(object sender, KeyPressEventArgs e)
        {
            if(e.KeyChar == 27)
            {
                if(!this.selectingArea)
                {
                    return;
                }

                this.selectingArea = false;

                this.selectedBitmap   = null;
                this.selectedGraphics = null;

                this.pictureBox.Image = this.sourceBitmap;

                this.pictureBox.Refresh();

                this.slectionCompleted = false;

                EnableMenuItems();
            }
        }

        #endregion

        #region 종료 메뉴 항목 클릭시 처리하기 - exitMenuItem_Click(sender, e)

        /// <summary>
        /// 종료 메뉴 항목 클릭시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void exitMenuItem_Click(object sender, EventArgs e)
        {
            Close();
        }

        #endregion
        #region 복사 메뉴 항목 클릭시 처리하기 - copyMenuItem_Click(sender, e)

        /// <summary>
        /// 복사 메뉴 항목 클릭시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void copyMenuItem_Click(object sender, EventArgs e)
        {
            CopyToClipboard(this.selectedRectangle);

            SystemSounds.Beep.Play();
        }

        #endregion
        #region 잘라내기 메뉴 항목 클릭시 처리하기 - cutMenuItem_Click(sender, e)

        /// <summary>
        /// 잘라내기 메뉴 항목 클릭시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void cutMenuItem_Click(object sender, EventArgs e)
        {
            CopyToClipboard(this.selectedRectangle);

            using(Graphics graphics = Graphics.FromImage(this.sourceBitmap))
            {
                using(SolidBrush brush = new SolidBrush(this.pictureBox.BackColor))
                {
                    graphics.FillRectangle(brush, this.selectedRectangle);
                }
            }

            this.selectedBitmap = new Bitmap(this.sourceBitmap);

            this.pictureBox.Image = this.selectedBitmap;

            EnableMenuItems();

            this.selectedBitmap    = null;
            this.selectedGraphics  = null;
            this.slectionCompleted = false;

            SystemSounds.Beep.Play();
        }

        #endregion
        #region 중앙 붙여넣기 메뉴 항목 클릭시 처리하기 - pasteCenterMenuItem_Click(sender, e)

        /// <summary>
        /// 중앙 붙여넣기 메뉴 항목 클릭시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void pasteCenterMenuItem_Click(object sender, EventArgs e)
        {
            if(!Clipboard.ContainsImage())
            {
                return;
            }

            Image clipboardImage = Clipboard.GetImage();

            int centerX = this.selectedRectangle.X + (this.selectedRectangle.Width  - clipboardImage.Width ) / 2;
            int centerY = this.selectedRectangle.Y + (this.selectedRectangle.Height - clipboardImage.Height) / 2;

            Rectangle targetRectangle = new Rectangle
            (
                centerX,
                centerY,
                clipboardImage.Width,
                clipboardImage.Height
            );

            using(Graphics graphics = Graphics.FromImage(this.sourceBitmap))
            {
                graphics.DrawImage(clipboardImage, targetRectangle);
            }

            this.pictureBox.Image = this.sourceBitmap;

            this.pictureBox.Refresh();

            this.selectedBitmap    = null;
            this.selectedGraphics  = null;
            this.slectionCompleted = false;
        }

        #endregion
        #region 확장 붙여넣기 메뉴 항목 클릭시 처리하기 - pasteStretchMenuItem_Click(sender, e)

        /// <summary>
        /// 확장 붙여넣기 메뉴 항목 클릭시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void pasteStretchMenuItem_Click(object sender, EventArgs e)
        {
            if(!Clipboard.ContainsImage())
            {
                return;
            }

            Image clipboardImage = Clipboard.GetImage();

            Rectangle sourceRectangle = new Rectangle
            (
                0,
                0,
                clipboardImage.Width,
                clipboardImage.Height
            );

            using(Graphics graphics = Graphics.FromImage(this.sourceBitmap))
            {
                graphics.DrawImage
                (
                    clipboardImage,
                    this.selectedRectangle,
                    sourceRectangle,
                    GraphicsUnit.Pixel
                );
            }

            this.pictureBox.Image = this.sourceBitmap;

            this.pictureBox.Refresh();

            this.selectedBitmap    = null;
            this.selectedGraphics  = null;
            this.slectionCompleted = false;
        }

        #endregion

        #region 크기 모드 콤보 박스 선택 인덱스 변경시 처리하기 - sizeModeComboBox_SelectedIndexChanged(sender, e)

        /// <summary>
        /// 크기 모드 콤보 박스 선택 인덱스 변경시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void sizeModeComboBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            switch(this.sizeModeComboBox.SelectedItem.ToString())
            {
                case "AutoSize" :

                    this.pictureBox.SizeMode = PictureBoxSizeMode.AutoSize;

                    ClientSize = new Size(this.pictureBox.Right + 10, this.pictureBox.Bottom + 10);

                    break;

                case "Normal" :

                    this.pictureBox.SizeMode = PictureBoxSizeMode.Normal;

                    break;

                case "CenterImage" :

                    this.pictureBox.SizeMode = PictureBoxSizeMode.CenterImage;

                    break;

                case "StretchImage" :

                    this.pictureBox.SizeMode = PictureBoxSizeMode.StretchImage;

                    break;

                case "Zoom" :

                    this.pictureBox.SizeMode = PictureBoxSizeMode.Zoom;

                    break;
            }
        }

        #endregion

        #region 픽처 박스 마우스 DOWN 처리하기 - pictureBox_MouseDown(sender, e)

        /// <summary>
        /// 픽처 박스 마우스 DOWN 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void pictureBox_MouseDown(object sender, MouseEventArgs e)
        {
            this.selectingArea = true;

            ConvertCoordinates(this.pictureBox, out this.x0, out this.y0, e.X, e.Y);

            this.selectedBitmap = new Bitmap(this.sourceBitmap);

            this.selectedGraphics = Graphics.FromImage(this.selectedBitmap);

            this.pictureBox.Image = this.selectedBitmap;
        }

        #endregion
        #region 픽처 박스 마우스 이동시 처리하기 - pictureBox_MouseMove(sender, e)

        /// <summary>
        /// 픽처 박스 마우스 이동시 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void pictureBox_MouseMove(object sender, MouseEventArgs e)
        {
            if(!this.selectingArea)
            {
                return;
            }

            ConvertCoordinates(this.pictureBox, out this.x1, out this.y1, e.X, e.Y);

            this.selectedGraphics.DrawImage(this.sourceBitmap, 0, 0);

            using(Pen pen = new Pen(Color.Red))
            {
                pen.DashStyle = DashStyle.Dash;

                Rectangle rectangle = GetRectangle(this.x0, this.y0, this.x1, this.y1);

                this.selectedGraphics.DrawRectangle(pen, rectangle);
            }

            this.pictureBox.Refresh();
        }

        #endregion
        #region 픽처 박스 마우스 UP 처리하기 - pictureBox_MouseUp(sender, e)

        /// <summary>
        /// 픽처 박스 마우스 UP 처리하기
        /// </summary>
        /// <param name="sender">이벤트 발생자</param>
        /// <param name="e">이벤트 인자</param>
        private void pictureBox_MouseUp(object sender, MouseEventArgs e)
        {
            if(!this.selectingArea)
            {
                return;
            }

            this.selectingArea = false;

            this.selectedGraphics = null;

            this.selectedRectangle = GetRectangle(this.x0, this.y0, this.x1, this.y1);

            this.slectionCompleted = ((this.selectedRectangle.Width > 0) && (this.selectedRectangle.Height > 0));

            EnableMenuItems();
        }

        #endregion

        //////////////////////////////////////////////////////////////////////////////// Function

        #region 좌표계 변환하기 - ConvertCoordinates(pictureBox, x0, y0, x1, y1)

        /// <summary>
        /// 좌표계 변환하기
        /// </summary>
        /// <param name="pictureBox">픽처 박스</param>
        /// <param name="targetX">타겟 X 좌표</param>
        /// <param name="targetY">타겟 Y 좌표</param>
        /// <param name="sourceX">소스 X 좌표</param>
        /// <param name="sourceY">소스 Y 좌표</param>
        private void ConvertCoordinates(PictureBox pictureBox, out int targetX, out int targetY, int sourceX, int sourceY)
        {
            int pictureBoxWidth  = pictureBox.ClientSize.Width;
            int pictureBoxHeight = pictureBox.ClientSize.Height;
            int imageWidth       = pictureBox.Image.Width;
            int imageHeight      = pictureBox.Image.Height;

            targetX = sourceX;
            targetY = sourceY;

            switch(pictureBox.SizeMode)
            {
                case PictureBoxSizeMode.AutoSize :
                case PictureBoxSizeMode.Normal   :

                    break;

                case PictureBoxSizeMode.CenterImage :

                    targetX = sourceX - (pictureBoxWidth  - imageWidth ) / 2;
                    targetY = sourceY - (pictureBoxHeight - imageHeight) / 2;

                    break;

                case PictureBoxSizeMode.StretchImage :

                    targetX = (int)(imageWidth  * sourceX / (float)pictureBoxWidth );
                    targetY = (int)(imageHeight * sourceY / (float)pictureBoxHeight);

                    break;

                case PictureBoxSizeMode.Zoom :

                    float pictureBoxAspectRatio = pictureBoxWidth / (float)pictureBoxHeight;
                    float imageAspectRatio      = imageWidth / (float)imageHeight;

                    if(pictureBoxAspectRatio > imageAspectRatio)
                    {
                        targetY = (int)(imageHeight * sourceY / (float)pictureBoxHeight);

                        float scaledWidth = imageWidth * pictureBoxHeight / imageHeight;

                        float deltaX = (pictureBoxWidth - scaledWidth) / 2;

                        targetX = (int)((sourceX - deltaX) * imageHeight / (float)pictureBoxHeight);
                    }
                    else
                    {
                        targetX = (int)(imageWidth * sourceX / (float)pictureBoxWidth);

                        float scaledHeight = imageHeight * pictureBoxWidth / imageWidth;

                        float deltaY = (pictureBoxHeight - scaledHeight) / 2;

                        targetY = (int)((sourceY - deltaY) * imageWidth / pictureBoxWidth);
                    }

                    break;
            }
        }

        #endregion
        #region 메뉴 항목 활성화 하기 - EnableMenuItems()

        /// <summary>
        /// 메뉴 항목 활성화 하기
        /// </summary>
        private void EnableMenuItems()
        {
            this.copyMenuItem.Enabled         = this.slectionCompleted;
            this.cutMenuItem.Enabled          = this.slectionCompleted;
            this.pasteStretchMenuItem.Enabled = this.slectionCompleted;
            this.pasteCenterMenuItem.Enabled  = this.slectionCompleted;
        }

        #endregion
        #region 사각형 구하기 - GetRectangle(x0, y0, x1, y1)

        /// <summary>
        /// 사각형 구하기
        /// </summary>
        /// <param name="x0">X0 좌표</param>
        /// <param name="y0">Y0 좌표</param>
        /// <param name="x1">X1 좌표</param>
        /// <param name="y1">Y1 좌표</param>
        /// <returns>사각형</returns>
        private Rectangle GetRectangle(int x0, int y0, int x1, int y1)
        {
            return new Rectangle
            (
                Math.Min(x0, x1),
                Math.Min(y0, y1),
                Math.Abs(x0 - x1),
                Math.Abs(y0 - y1)
            );
        }

        #endregion
        #region 클립보드 복사하기 - CopyToClipboard(sourceRectangle)

        /// <summary>
        /// 클립보드 복사하기
        /// </summary>
        /// <param name="sourceRectangle">소스 사각형</param>
        private void CopyToClipboard(Rectangle sourceRectangle)
        {
            Bitmap bitmap = new Bitmap(sourceRectangle.Width, sourceRectangle.Height);

            using(Graphics graphics = Graphics.FromImage(bitmap))
            {
                Rectangle targetRectangle = new Rectangle(0, 0, sourceRectangle.Width, sourceRectangle.Height);

                graphics.DrawImage(this.sourceBitmap, targetRectangle, sourceRectangle, GraphicsUnit.Pixel);
            }

            Clipboard.SetImage(bitmap);
        }

        #endregion
    }
}