# ==============================================================================
# 1. CLOCK DEFINITION (Slowed to 10 MHz / 100ns to pass Implementation Timing)
# ==============================================================================
set_property PACKAGE_PIN W5 [get_ports clk]							
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -add -name sys_clk_pin -period 100.00 -waveform {0 50} [get_ports clk]

# ==============================================================================
# 2. RESET SIGNAL
# ==============================================================================
set_property PACKAGE_PIN U18 [get_ports rst]						
set_property IOSTANDARD LVCMOS33 [get_ports rst]

# ==============================================================================
# 3. ANALOG XADC INPUT (From ADXL1002 Voltage Divider)
# ==============================================================================
# Pin J3 is the "P" side of the analog pair on the JXADC header
set_property PACKAGE_PIN J3 [get_ports vauxp6]
set_property IOSTANDARD LVCMOS33 [get_ports vauxp6]

# Pin K3 is the "N" side (Ground) of the analog pair on the JXADC header
set_property PACKAGE_PIN K3 [get_ports vauxn6]
set_property IOSTANDARD LVCMOS33 [get_ports vauxn6]
# Note: Analog pins do NOT require an IOSTANDARD declaration in Vivado.
# ==============================================================================
# 4. MASTER ALARM OUTPUT (Mapped to a PMOD port or standalone pin)
# ==============================================================================
# Ensures it does not conflict with the 16 LEDs used for the spike train
set_property PACKAGE_PIN J1 [get_ports master_alarm]					
set_property IOSTANDARD LVCMOS33 [get_ports master_alarm]

# ==============================================================================
# 5. 16-BIT SPIKE TRAIN OUTPUT (Mapped to 16 physical LEDs)
# ==============================================================================
set_property PACKAGE_PIN U16 [get_ports {spike_train_out[0]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[0]}]
set_property PACKAGE_PIN E19 [get_ports {spike_train_out[1]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[1]}]
set_property PACKAGE_PIN U19 [get_ports {spike_train_out[2]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[2]}]
set_property PACKAGE_PIN V19 [get_ports {spike_train_out[3]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[3]}]
set_property PACKAGE_PIN W18 [get_ports {spike_train_out[4]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[4]}]
set_property PACKAGE_PIN U15 [get_ports {spike_train_out[5]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[5]}]
set_property PACKAGE_PIN U14 [get_ports {spike_train_out[6]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[6]}]
set_property PACKAGE_PIN V14 [get_ports {spike_train_out[7]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[7]}]
set_property PACKAGE_PIN V13 [get_ports {spike_train_out[8]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[8]}]
set_property PACKAGE_PIN V3  [get_ports {spike_train_out[9]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[9]}]
set_property PACKAGE_PIN W3  [get_ports {spike_train_out[10]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[10]}]
set_property PACKAGE_PIN U3  [get_ports {spike_train_out[11]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[11]}]
set_property PACKAGE_PIN P3  [get_ports {spike_train_out[12]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[12]}]
set_property PACKAGE_PIN N3  [get_ports {spike_train_out[13]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[13]}]
set_property PACKAGE_PIN P1  [get_ports {spike_train_out[14]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[14]}]
set_property PACKAGE_PIN L1  [get_ports {spike_train_out[15]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {spike_train_out[15]}]
# ==============================================================================
# 6. EXTERNAL I/O TIMING CONSTRAINTS (To resolve TIMING-18 Methodology Warnings)
# ==============================================================================
# Assuming a standard external ADC with a max 15ns propagation delay, and min 2ns delay.
# Our clock period is 100ns, so we have massive amounts of margin to absorb this safely.



set_input_delay -clock [get_clocks sys_clk_pin] -max 15.000 [get_ports rst]
set_input_delay -clock [get_clocks sys_clk_pin] -min 2.000  [get_ports rst]

# Output Delays (FPGA to External Relays/LEDs)
# Assuming the receiving device requires data to be stable 10ns before its clock edge
set_output_delay -clock [get_clocks sys_clk_pin] -max 10.000 [get_ports master_alarm]
set_output_delay -clock [get_clocks sys_clk_pin] -min 1.000  [get_ports master_alarm]
set_output_delay -clock [get_clocks sys_clk_pin] -max 10.000 [get_ports {spike_train_out[*]}]
set_output_delay -clock [get_clocks sys_clk_pin] -min 1.000  [get_ports {spike_train_out[*]}]
# MICRO ALARM OUTPUT (Mapped to spare PMOD pin L2)
set_property PACKAGE_PIN L2 [get_ports micro_alarm_out]
set_property IOSTANDARD LVCMOS33 [get_ports micro_alarm_out]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]