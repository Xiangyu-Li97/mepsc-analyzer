# 傻瓜级使用教程 (Step-by-Step Guide for Beginners)

这篇教程为没有任何编程经验的科研人员设计。我们会手把手教您如何使用本工具。

This guide is designed for researchers with **zero programming experience**. We will walk you through every step.

---

##  English Guide (Python Version)

We recommend the Python version because it's free and works on any computer (Windows, Mac, Linux).

### Step 0: What You Need

1.  **Python**: A free programming language.
2.  **A Terminal**: A command-line program that's already on your computer.

**How to check if you have Python:**

-   **On Windows**: Open the Start Menu, type `cmd` and press Enter. In the black window that appears, type `python --version` and press Enter.
-   **On Mac**: Open Launchpad, find and open the "Terminal" app. In the window, type `python3 --version` and press Enter.

If you see a version number (e.g., `Python 3.11.0`), you're good to go! If not, please download and install Python from [python.org](https://www.python.org/downloads/).

### Step 1: Download the Tool

1.  Go to the GitHub page: [https://github.com/Xiangyu-Li97/mepsc-analyzer](https://github.com/Xiangyu-Li97/mepsc-analyzer)
2.  Click the green **`< > Code`** button.
3.  In the dropdown menu, click **`Download ZIP`**.

![Download ZIP](https://i.imgur.com/iB4jA2O.png)

### Step 2: Unzip the Folder

Find the downloaded `mepsc-analyzer-main.zip` file (usually in your `Downloads` folder) and unzip it. You will get a folder named `mepsc-analyzer-main`.

### Step 3: Open the Folder in a Terminal

This is the trickiest part, but you can do it!

-   **On Windows**:
    1.  Open the `mepsc-analyzer-main` folder.
    2.  Click on the address bar at the top of the window.
    3.  Type `cmd` and press Enter. A black window will pop up, already in the correct folder.

-   **On Mac**:
    1.  Open the "Terminal" app.
    2.  Type `cd ` (note the space after `cd`).
    3.  Drag the `mepsc-analyzer-main` folder from Finder and drop it into the Terminal window.
    4.  Press Enter.

### Step 4: Install Required Packages

In the terminal window you just opened, copy and paste the following command, then press Enter:

```bash
pip install -r requirements.txt
```

This will automatically download and install all the necessary dependencies. It might take a few minutes.

### Step 5: Place Your Data File

Copy your `.abf` data file into the `mepsc-analyzer-main` folder. Let's say your file is named `cell01.abf`.

### Step 6: Run the Analysis!

In the same terminal window, type the following command and press Enter. **Remember to replace `cell01.abf` with your actual file name.**

```bash
python mepsc_analyzer.py cell01.abf
```

The analysis will start. You will see text updates on the screen. When it's done, it will say "分析完成!" (Analysis Complete!).

### Step 7: Check Your Results

Look inside the `mepsc-analyzer-main` folder. You will find three new files:

1.  **`cell01_analysis_... .png`**: This is an image file with all the summary graphs. You can put this directly into your presentation or paper!
2.  **`cell01_events_... .csv`**: This is a spreadsheet file (can be opened with Excel) that lists **every single event** detected and all its properties (amplitude, rise time, etc.).
3.  **`cell01_summary_... .csv`**: A simple spreadsheet with the summary statistics for the entire recording (average frequency, average amplitude, etc.).

**You're done! You've successfully analyzed your data without any manual clicking!**

---

## 中文教程 (Python 版本)

我们推荐使用 Python 版本，因为它是完全免费的，并且可以在任何电脑上运行（Windows, Mac, Linux）。

### 第 0 步：您需要什么

1.  **Python**: 一个免费的编程语言。
2.  **终端 (Terminal)**: 一个您的电脑上自带的命令行程序。

**如何检查您是否安装了 Python：**

-   **在 Windows 上**: 打开开始菜单，输入 `cmd` 并按回车。在弹出的黑色窗口中，输入 `python --version` 并按回车。
-   **在 Mac 上**: 打开 "启动台"，找到并打开 "终端" (Terminal) 应用。在窗口中，输入 `python3 --version` 并按回车。

如果您看到了版本号 (例如 `Python 3.11.0`)，那就说明已经安装好了！如果没有，请从 [python.org](https://www.python.org/downloads/) 下载并安装 Python。

### 第 1 步：下载工具

1.  前往 GitHub 页面: [https://github.com/Xiangyu-Li97/mepsc-analyzer](https://github.com/Xiangyu-Li97/mepsc-analyzer)
2.  点击绿色的 **`< > Code`** 按钮。
3.  在下拉菜单中，点击 **`Download ZIP`**。

![Download ZIP](https://i.imgur.com/iB4jA2O.png)

### 第 2 步：解压文件夹

找到下载的 `mepsc-analyzer-main.zip` 文件（通常在您的 "下载" 文件夹中）并解压。您会得到一个名为 `mepsc-analyzer-main` 的文件夹。

### 第 3 步：在终端中打开文件夹

这是最关键的一步，但别担心，很简单！

-   **在 Windows 上**:
    1.  打开 `mepsc-analyzer-main` 文件夹。
    2.  点击窗口顶部的地址栏。
    3.  输入 `cmd` 并按回车。一个黑色窗口会弹出，它已经自动定位到了正确的文件夹。

-   **在 Mac 上**:
    1.  打开 "终端" (Terminal) 应用。
    2.  输入 `cd ` (注意 `cd` 后面有一个空格)。
    3.  将 `mepsc-analyzer-main` 文件夹从访达 (Finder) 拖拽到终端窗口中。
    4.  按回车。

### 第 4 步：安装所需的依赖包

在您刚刚打开的终端窗口中，复制并粘贴以下命令，然后按回车：

```bash
pip install -r requirements.txt
```

这个命令会自动下载并安装所有必需的软件库。可能需要几分钟时间。

### 第 5 步：放入您的数据文件

将您的 `.abf` 数据文件复制到 `mepsc-analyzer-main` 文件夹中。假设您的文件名是 `cell01.abf`。

### 第 6 步：运行分析！

在同一个终端窗口中，输入以下命令，然后按回车。**请记得将 `cell01.abf` 替换成您自己的文件名。**

```bash
python mepsc_analyzer.py cell01.abf
```

分析将会开始，您会在屏幕上看到文字更新。当分析结束时，会显示 "分析完成!"。

### 第 7 步：查看您的结果

查看 `mepsc-analyzer-main` 文件夹，您会发现三个新文件：

1.  **`cell01_analysis_... .png`**: 这是一个图片文件，包含了所有的汇总图表。您可以直接把它放进您的 PPT 或论文里！
2.  **`cell01_events_... .csv`**: 这是一个可以用 Excel 打开的表格文件，里面列出了**每一个**被检测到的事件和它的所有参数（振幅、上升时间等）。
3.  **`cell01_summary_... .csv`**: 一个简单的表格，包含了整个记录的汇总统计信息（平均频率、平均振幅等）。

**恭喜您！您已经成功地在没有手动点击的情况下完成了数据分析！**
