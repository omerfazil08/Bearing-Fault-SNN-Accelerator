----------------------------------------------------------------------------------
-- Testbench for SNN Neuromorphic Core (REAL DATA VERIFICATION)
-- Reads 8-bit ADC integers directly from Python-generated text file.
-- Includes Simulation Phase string for waveform presentation and Crash Protection.
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use STD.TEXTIO.ALL; 

entity tb_neuromorphic_core is
end tb_neuromorphic_core;

architecture behavior of tb_neuromorphic_core is 

    component neuromorphic_core
    port(
         clk             : in  std_logic;
         rst             : in  std_logic;
         adc_data_in     : in  std_logic_vector(7 downto 0);
         data_valid      : in  std_logic;
         master_alarm    : out std_logic;
         micro_alarm_out : out std_logic; -- ? NEW
         spike_train_out : out std_logic_vector(15 downto 0)
        );
    end component;

    -- Signal Declarations
    signal clk             : std_logic := '0';
    signal rst             : std_logic := '0';
    signal adc_data_in     : std_logic_vector(7 downto 0) := (others => '0');
    signal data_valid      : std_logic := '0';
    signal master_alarm    : std_logic;
    signal micro_alarm_out : std_logic;
    signal spike_train_out : std_logic_vector(15 downto 0);
    
    -- Simulation Phase Tracker for the Waveform (Exactly 15 chars)
    signal sim_phase       : string(1 to 15) := "INITIALIZING   ";

    -- Clock period definition (100 MHz)
    constant clk_period : time := 10 ns;

begin

    uut: neuromorphic_core port map (
          clk             => clk,
          rst             => rst,
          adc_data_in     => adc_data_in,
          data_valid      => data_valid,
          master_alarm    => master_alarm,
          micro_alarm_out => micro_alarm_out,
          spike_train_out => spike_train_out
        );

    clk_process :process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process;

stim_proc: process
        -- File reading variables
        file text_file : text open read_mode is "C:/Users/omer-/Desktop/normal_data_down_12k/vivado_down_1/adc_presentation_vectors.txt";
        variable text_line : line;
        variable file_adc_val : integer;
        variable read_ok : boolean; 
        variable sample_count : integer := 0; 
        
        -- Macro-Checking Variables (Every 2400 samples)
        variable TP : integer := 0;
        variable TN : integer := 0;
        variable FP : integer := 0;
        variable FN : integer := 0;
        variable acc_int : integer := 0;

        -- NEW: Micro-Checking Variables (Every 32 samples)
        variable micro_TP : integer := 0;
        variable micro_TN : integer := 0;
        variable micro_FP : integer := 0;
        variable micro_FN : integer := 0;
        variable micro_acc_int : integer := 0;
        variable micro_decision : std_logic;

        variable expected_alarm : std_logic := '0';
    begin		
        -- 1. Global Reset
        rst <= '1';
        wait for 100 ns;	
        rst <= '0';
        wait for clk_period * 10;

        report "--- STARTING ORCHESTRATED SELF-CHECKING SIMULATION ---";

        -- 2. Read File Loop
        while not endfile(text_file) loop
            readline(text_file, text_line);
            read(text_line, file_adc_val, read_ok);
            
            if read_ok then
                
                -- EXACT PHASE TRACKING (32,000 samples per phase)
-- EXACT PHASE TRACKING (16,800 samples per phase)
                if sample_count < 16800 then
                    sim_phase <= "1. HEALTHY     ";
                    expected_alarm := '0';
                elsif sample_count < 33600 then
                    sim_phase <= "2. INNER FAULT ";
                    expected_alarm := '1';
                elsif sample_count < 50400 then
                    sim_phase <= "3. RECOVERY    ";
                    expected_alarm := '0';
                elsif sample_count < 67200 then
                    sim_phase <= "4. BALL FAULT  ";
                    expected_alarm := '1';
                elsif sample_count < 84000 then
                    sim_phase <= "5. RECOVERY    ";
                    expected_alarm := '0';
                else
                    sim_phase <= "6. OUTER FAULT ";
                    expected_alarm := '1';
                end if;
                
                sample_count := sample_count + 1;
                
                -- Push Data
                adc_data_in <= std_logic_vector(to_unsigned(file_adc_val, 8));
                data_valid <= '1';
                wait for clk_period;
                
                data_valid <= '0';
                wait for clk_period * 5; 
                
                -- NEW: EVALUATE MICRO-ACCURACY (Every 32 samples)
                if sample_count > 0 and (sample_count mod 32) = 0 then
                    -- The LSB of the spike train holds the immediate decision
                    micro_decision := micro_alarm_out; 
                    
                    if micro_decision = '1' and expected_alarm = '1' then
                        micro_TP := micro_TP + 1;
                    elsif micro_decision = '0' and expected_alarm = '0' then
                        micro_TN := micro_TN + 1;
                    elsif micro_decision = '1' and expected_alarm = '0' then
                        micro_FP := micro_FP + 1;
                    elsif micro_decision = '0' and expected_alarm = '1' then
                        micro_FN := micro_FN + 1;
                    end if;
                end if;

                -- EVALUATE MACRO-ACCURACY (Every 2400 samples)
                if sample_count > 0 and (sample_count mod 2400) = 0 then
                    if master_alarm = '1' and expected_alarm = '1' then
                        TP := TP + 1;
                    elsif master_alarm = '0' and expected_alarm = '0' then
                        TN := TN + 1;
                    elsif master_alarm = '1' and expected_alarm = '0' then
                        FP := FP + 1;
                    elsif master_alarm = '0' and expected_alarm = '1' then
                        FN := FN + 1;
                    end if;
                end if;
                
            end if;
        end loop;

        wait for clk_period * 10;
        
        -- PRINT THE FINAL ACCURACY TO THE TCL CONSOLE
        report "=================================================";
        report "? VIVADO HARDWARE SELF-CHECKING RESULTS ?";
        report "=================================================";
        
        if (micro_TP + micro_TN + micro_FP + micro_FN) > 0 then
            micro_acc_int := ((micro_TP + micro_TN) * 10000) / (micro_TP + micro_TN + micro_FP + micro_FN);
            report "Micro-Window Accuracy (2.6ms) : " & integer'image(micro_acc_int / 100) & "." & integer'image(micro_acc_int mod 100) & "%";
            report "Micro-Counts -> TP: " & integer'image(micro_TP) & " | TN: " & integer'image(micro_TN) & " | FP: " & integer'image(micro_FP) & " | FN: " & integer'image(micro_FN);
        end if;
        
        report "-------------------------------------------------";

        if (TP + TN + FP + FN) > 0 then
            acc_int := ((TP + TN) * 100) / (TP + TN + FP + FN);
            report "Macro-Window Accuracy (100ms) : " & integer'image(acc_int) & "%";
            report "Macro-Counts -> TP: " & integer'image(TP) & " | TN: " & integer'image(TN) & " | FP: " & integer'image(FP) & " | FN: " & integer'image(FN);
        end if;
        
        report "=================================================";
        report "--- END OF SIMULATION ---";
        
        std.env.stop;
    end process;
end Behavior;  