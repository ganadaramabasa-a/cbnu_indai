namespace TestProject
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.menuStrip = new System.Windows.Forms.MenuStrip();
            this.fileMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.exitMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.eidtMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.copyMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.cutMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.pasteCenterMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.pasteStretchMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.sizeModeLabel = new System.Windows.Forms.Label();
            this.sizeModeComboBox = new System.Windows.Forms.ComboBox();
            this.pictureBox = new System.Windows.Forms.PictureBox();
            this.menuStrip.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).BeginInit();
            this.SuspendLayout();
            // 
            // menuStrip
            // 
            this.menuStrip.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileMenuItem,
            this.eidtMenuItem});
            this.menuStrip.Location = new System.Drawing.Point(0, 0);
            this.menuStrip.Name = "menuStrip";
            this.menuStrip.Padding = new System.Windows.Forms.Padding(7, 2, 0, 2);
            this.menuStrip.Size = new System.Drawing.Size(784, 24);
            this.menuStrip.TabIndex = 0;
            this.menuStrip.Text = "menuStrip";
            // 
            // fileMenuItem
            // 
            this.fileMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.exitMenuItem});
            this.fileMenuItem.Name = "fileMenuItem";
            this.fileMenuItem.Size = new System.Drawing.Size(57, 20);
            this.fileMenuItem.Text = "파일(&F)";
            // 
            // exitMenuItem
            // 
            this.exitMenuItem.Name = "exitMenuItem";
            this.exitMenuItem.Size = new System.Drawing.Size(113, 22);
            this.exitMenuItem.Text = "종료(&X)";
            // 
            // eidtMenuItem
            // 
            this.eidtMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.copyMenuItem,
            this.cutMenuItem,
            this.pasteCenterMenuItem,
            this.pasteStretchMenuItem});
            this.eidtMenuItem.Name = "eidtMenuItem";
            this.eidtMenuItem.Size = new System.Drawing.Size(57, 20);
            this.eidtMenuItem.Text = "편집(&E)";
            // 
            // copyMenuItem
            // 
            this.copyMenuItem.Name = "copyMenuItem";
            this.copyMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.C)));
            this.copyMenuItem.Size = new System.Drawing.Size(207, 22);
            this.copyMenuItem.Text = "복사(&C)";
            this.copyMenuItem.ToolTipText = "Copy the selected area to the clipboard";
            // 
            // cutMenuItem
            // 
            this.cutMenuItem.Name = "cutMenuItem";
            this.cutMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.X)));
            this.cutMenuItem.Size = new System.Drawing.Size(207, 22);
            this.cutMenuItem.Text = "잘라내기(&U)";
            this.cutMenuItem.ToolTipText = "Copy the selected area to the clipboard and clear the area";
            // 
            // pasteCenterMenuItem
            // 
            this.pasteCenterMenuItem.Name = "pasteCenterMenuItem";
            this.pasteCenterMenuItem.ShortcutKeys = ((System.Windows.Forms.Keys)((System.Windows.Forms.Keys.Control | System.Windows.Forms.Keys.V)));
            this.pasteCenterMenuItem.Size = new System.Drawing.Size(207, 22);
            this.pasteCenterMenuItem.Text = "중앙 붙여넣기(&P)";
            this.pasteCenterMenuItem.ToolTipText = "Paste the image on the clipboard to the selected area, centering it in the select" +
    "ed area";
            // 
            // pasteStretchMenuItem
            // 
            this.pasteStretchMenuItem.Name = "pasteStretchMenuItem";
            this.pasteStretchMenuItem.Size = new System.Drawing.Size(207, 22);
            this.pasteStretchMenuItem.Text = "화장 붙여넣기(&S)";
            this.pasteStretchMenuItem.ToolTipText = "Paste the image on the clipboard to the selected area, stretching it to fit";
            // 
            // sizeModeLabel
            // 
            this.sizeModeLabel.Location = new System.Drawing.Point(10, 30);
            this.sizeModeLabel.Name = "sizeModeLabel";
            this.sizeModeLabel.Size = new System.Drawing.Size(80, 24);
            this.sizeModeLabel.TabIndex = 1;
            this.sizeModeLabel.Text = "크기 모드";
            this.sizeModeLabel.TextAlign = System.Drawing.ContentAlignment.MiddleLeft;
            // 
            // sizeModeComboBox
            // 
            this.sizeModeComboBox.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.sizeModeComboBox.FormattingEnabled = true;
            this.sizeModeComboBox.Items.AddRange(new object[] {
            "AutoSize",
            "Normal",
            "CenterImage",
            "StretchImage",
            "Zoom"});
            this.sizeModeComboBox.Location = new System.Drawing.Point(90, 30);
            this.sizeModeComboBox.Name = "sizeModeComboBox";
            this.sizeModeComboBox.Size = new System.Drawing.Size(150, 24);
            this.sizeModeComboBox.TabIndex = 2;
            // 
            // pictureBox
            // 
            this.pictureBox.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.pictureBox.BackColor = System.Drawing.Color.LightBlue;
            this.pictureBox.Cursor = System.Windows.Forms.Cursors.Cross;
            this.pictureBox.Location = new System.Drawing.Point(10, 70);
            this.pictureBox.Name = "pictureBox";
            this.pictureBox.Size = new System.Drawing.Size(760, 480);
            this.pictureBox.TabIndex = 5;
            this.pictureBox.TabStop = false;
            // 
            // MainForm
            // 
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.None;
            this.ClientSize = new System.Drawing.Size(784, 561);
            this.Controls.Add(this.sizeModeComboBox);
            this.Controls.Add(this.sizeModeLabel);
            this.Controls.Add(this.pictureBox);
            this.Controls.Add(this.menuStrip);
            this.Font = new System.Drawing.Font("나눔고딕코딩", 12F);
            this.Name = "MainForm";
            this.Text = "PictureBox 클래스 : SizeMode 속성에 따라 이미지 구하기";
            this.menuStrip.ResumeLayout(false);
            this.menuStrip.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox;
        private System.Windows.Forms.MenuStrip menuStrip;
        private System.Windows.Forms.ToolStripMenuItem fileMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitMenuItem;
        private System.Windows.Forms.ToolStripMenuItem eidtMenuItem;
        private System.Windows.Forms.ToolStripMenuItem copyMenuItem;
        private System.Windows.Forms.ToolStripMenuItem cutMenuItem;
        private System.Windows.Forms.ToolStripMenuItem pasteCenterMenuItem;
        private System.Windows.Forms.ToolStripMenuItem pasteStretchMenuItem;
        private System.Windows.Forms.Label sizeModeLabel;
        private System.Windows.Forms.ComboBox sizeModeComboBox;
    }
}

