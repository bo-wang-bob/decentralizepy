$start_time = Get-Date

# 获取脚本当前目录的绝对路径
$script_path = (Get-Location).Path
Write-Host "Script path is $script_path"

# 定义路径和变量
$graph = Join-Path $script_path "fullyConnected_32.edges" # 图文件的绝对路径
$run_path = Join-Path $script_path "../../eval/data" # 运行路径的绝对路径
$config_file = Join-Path $script_path "config_EL.ini"

# 确保运行路径存在
if (-Not (Test-Path $run_path)) {
    New-Item -ItemType Directory -Force -Path $run_path | Out-Null
}

# 将图文件和配置文件复制到运行路径
Copy-Item -Path $graph -Destination $run_path -Force
Copy-Item -Path $config_file -Destination $run_path -Force

# 环境设置
$env_python = "python" # Python 可执行文件路径，推荐使用 conda 环境
$machines = 1 # 运行时的机器数量
$iterations = 200
$test_after = 20
$eval_file = Join-Path $script_path "testingEL_Local.py" # 分布式驱动代码（每台机器上运行）
$log_level = "DEBUG" # 可选值：DEBUG | INFO | WARN | CRITICAL

# 机器 ID
$m = 0 # 与 ip.json 一致的机器 ID
Write-Host "M is $m"

# 每台机器的进程数
$malicous_nodes = 4 # 每台机器的恶意节点数量
$procs_per_machine = 32 # 每台机器的进程数 这个好像是节点数
Write-Host "Procs per machine is $procs_per_machine"

# 创建日志目录
$timestamp = (Get-Date -Format "yyyy-MM-dd_HH-mm") # 替换冒号为有效字符
$log_dir = Join-Path $run_path "$timestamp/machine$m" # 在 eval 文件夹中

# 确保日志路径合法并创建
if (-Not (Test-Path $log_dir)) {
    New-Item -ItemType Directory -Force -Path $log_dir | Out-Null
}

# 构造命令字符串
$command = "& $env_python $eval_file " +
"-ro 0 " +
"-tea $test_after " +
"-ld $log_dir " +
"-mid $m " +
"-ps $procs_per_machine " +
"-ms $machines " +
"-is $iterations " +
"-gf `"$run_path/$(Split-Path -Leaf $graph)`" " +
"-ta $test_after " +
"-cf `"$run_path/$(Split-Path -Leaf $config_file)`" " +
"-ll $log_level " +
"-wsd $log_dir" +
"-mals $malicous_nodes"

# 输出最终的命令形式
Write-Host "Executing Command: $command"


# 执行 Python 脚本
& $env_python $eval_file `
    -ro 0 `
    -tea $test_after `
    -ld $log_dir `
    -mid $m `
    -ps $procs_per_machine `
    -ms $machines `
    -is $iterations `
    -gf "$run_path/$(Split-Path -Leaf $graph)" `
    -ta $test_after `
    -cf "$run_path/$(Split-Path -Leaf $config_file)" `
    -ll $log_level `
    -wsd $log_dir `
    -mals $malicous_nodes

$end_time = Get-Date
$duration = $end_time - $start_time
# 计算总小时数、分钟数和剩余秒数
$total_hours = [math]::Floor($duration.TotalHours)
$total_minutes = [math]::Floor($duration.TotalMinutes % 60)
$remaining_seconds = [math]::Floor($duration.TotalSeconds % 60)

# 输出结果
Write-Host "Script Execution Time: $total_hours hours, $total_minutes minutes, and $remaining_seconds seconds"