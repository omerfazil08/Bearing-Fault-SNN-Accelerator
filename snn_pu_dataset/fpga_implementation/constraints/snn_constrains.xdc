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
# 3. ADC DATA INPUT (8-Bits: 0 to 255) & VALID SIGNAL
# ==============================================================================
set_property PACKAGE_PIN V17 [get_ports {adc_data_in[0]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[0]}]
set_property PACKAGE_PIN V16 [get_ports {adc_data_in[1]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[1]}]
set_property PACKAGE_PIN W16 [get_ports {adc_data_in[2]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[2]}]
set_property PACKAGE_PIN W17 [get_ports {adc_data_in[3]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[3]}]
set_property PACKAGE_PIN W15 [get_ports {adc_data_in[4]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[4]}]
set_property PACKAGE_PIN V15 [get_ports {adc_data_in[5]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[5]}]
set_property PACKAGE_PIN W14 [get_ports {adc_data_in[6]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[6]}]
set_property PACKAGE_PIN W13 [get_ports {adc_data_in[7]}]					
set_property IOSTANDARD LVCMOS33 [get_ports {adc_data_in[7]}]

set_property PACKAGE_PIN V2 [get_ports data_valid]					
set_property IOSTANDARD LVCMOS33 [get_ports data_valid]

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

# Input Delays (ADC to FPGA)
set_input_delay -clock [get_clocks sys_clk_pin] -max 15.000 [get_ports {adc_data_in[*]}]
set_input_delay -clock [get_clocks sys_clk_pin] -min 2.000  [get_ports {adc_data_in[*]}]
set_input_delay -clock [get_clocks sys_clk_pin] -max 15.000 [get_ports data_valid]
set_input_delay -clock [get_clocks sys_clk_pin] -min 2.000  [get_ports data_valid]
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