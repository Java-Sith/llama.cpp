catch {::common::set_param -quiet hls.xocc.mode csynth};
# 
# HLS run script generated by v++ compiler
# 

open_project matmul
set_top matmul
# v++ -g, -D, -I, --advanced.prop kernel.matmul.kernel_flags
add_files "/pub/scratch/lprieto/llama.cpp/ecas-scripts/XRT-MatMul/HW/matmul.cpp" -cflags " -I /pub/scratch/lprieto/llama.cpp/ecas-scripts/XRT-MatMul/HW/ -DUSE_$::env(DATATYPE) -DBUS=$::env(BUS) -DB_COLS=$::env(B_COLS) -DC_COLS=$::env(C_COLS) "
add_files -tb "/pub/scratch/lprieto/llama.cpp/ecas-scripts/XRT-MatMul/HW/matmul_tb.cc" -cflags " -I /pub/scratch/lprieto/llama.cpp/ecas-scripts/XRT-MatMul/HW/ -DUSE_$::env(DATATYPE)"
open_solution -flow_target vitis solution
set_part xilinx_u250_gen3x16_xdma_4_1_202210_1
create_clock -period 300MHz -name default
# v++ --advanced.param compiler.hlsDataflowStrictMode
config_dataflow -strict_mode warning
# v++ --advanced.param compiler.deadlockDetection
config_rtl -deadlock_detection sim
# v++ --advanced.param compiler.axiDeadLockFree
config_interface -m_axi_conservative_mode=1
config_interface -m_axi_addr64
# v++ --hls.max_memory_ports
config_interface -m_axi_auto_max_ports=0
config_export -format xo -ipname matmul
#csim_design -clean 
csynth_design
#cosim_design
#export_design
close_project
puts "HLS completed successfully"
exit