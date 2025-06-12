@echo off
REM 自动分析FedDCA实验结果的批处理脚本
REM 适用于Windows PowerShell环境

echo ====================================
echo FedDCA实验结果自动分析工具
echo ====================================
echo.

REM 设置默认路径
set "DEFAULT_OUTPUT_DIR=system\feddca_direct_commands_output_*"
set "DEFAULT_RESULTS_DIR=analysis_results"

REM 检查是否提供了输出目录参数
if "%1"=="" (
    echo 正在搜索实验输出目录...
    
    REM 查找最新的输出目录
    for /f "delims=" %%i in ('dir /b /ad /o-d "system\feddca_direct_commands_output_*" 2^>nul ^| head -1') do (
        set "OUTPUT_DIR=system\%%i"
        goto :found_dir
    )
    
    echo 错误: 未找到实验输出目录
    echo 请确保已运行实验脚本或手动指定输出目录:
    echo   analyze_results.bat ^<输出目录路径^>
    echo.
    echo 示例:
    echo   analyze_results.bat system\feddca_direct_commands_output_20240613_143022
    pause
    exit /b 1
    
    :found_dir
    echo 找到输出目录: %OUTPUT_DIR%
) else (
    set "OUTPUT_DIR=%1"
    echo 使用指定的输出目录: %OUTPUT_DIR%
)

REM 检查输出目录是否存在
if not exist "%OUTPUT_DIR%" (
    echo 错误: 输出目录不存在: %OUTPUT_DIR%
    pause
    exit /b 1
)

REM 检查是否提供了结果目录参数
if "%2"=="" (
    set "RESULTS_DIR=%DEFAULT_RESULTS_DIR%"
) else (
    set "RESULTS_DIR=%2"
)

echo 结果将保存到: %RESULTS_DIR%
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    echo 请确保Python已安装并添加到PATH环境变量中
    pause
    exit /b 1
)

echo 检查并安装依赖包...
pip install matplotlib seaborn pandas numpy pathlib >nul 2>&1

echo.
echo 开始分析实验结果...
echo 输出目录: %OUTPUT_DIR%
echo 结果目录: %RESULTS_DIR%
echo.

REM 运行分析脚本
python analyze_experiment_results.py --output_dir "%OUTPUT_DIR%" --results_dir "%RESULTS_DIR%"

if errorlevel 1 (
    echo.
    echo 分析过程中出现错误
    pause
    exit /b 1
)

echo.
echo ====================================
echo 分析完成！
echo ====================================
echo.
echo 结果文件位置:
echo - 图表: %RESULTS_DIR%\figures\
echo - 数据表: %RESULTS_DIR%\tables\
echo - 报告: %RESULTS_DIR%\experiment_report.md
echo.

REM 询问是否打开结果目录
set /p "OPEN_DIR=是否打开结果目录? (y/n): "
if /i "%OPEN_DIR%"=="y" (
    start "" "%RESULTS_DIR%"
)

echo.
echo 按任意键退出...
pause >nul
