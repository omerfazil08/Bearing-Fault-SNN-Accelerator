----------------------------------------------------------------------------------
-- Testbench for SNN Neuromorphic Core (Avionics UART Wrapper)
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use STD.TEXTIO.ALL; 

entity tb_neuromorphic_core is
end tb_neuromorphic_core;

architecture behavior of tb_neuromorphic_core is 

    component snn_avionics_top
    Port (
        clk              : in  STD_LOGIC;
        rst              : in  STD_LOGIC;
        adc_data_in      : in  STD_LOGIC_VECTOR(7 downto 0);
        data_valid       : in  STD_LOGIC;
        master_alarm_led : out STD_LOGIC;
        micro_alarm_out  : out STD_LOGIC;
        spike_train_leds : out STD_LOGIC_VECTOR(15 downto 0);
        uart_tx_pin      : out STD_LOGIC
    );
    end component;

    signal clk              : std_logic := '0';
    signal rst              : std_logic := '0';
    signal adc_data_in      : std_logic_vector(7 downto 0) := (others => '0');
    signal data_valid       : std_logic := '0';
    signal master_alarm     : std_logic;
    signal micro_alarm_out  : std_logic;
    signal spike_train_out  : std_logic_vector(15 downto 0);
    signal uart_tx_signal   : std_logic; 

    signal sim_phase        : string(1 to 15) := "INITIALIZING   ";
    
    -- UPDATED: 100 ns period = 10 MHz Clock (Required for UART timing)
    constant clk_period     : time := 100 ns;

begin

    uut: snn_avionics_top port map (
          clk              => clk,
          rst              => rst,
          adc_data_in      => adc_data_in,
          data_valid       => data_valid,
          master_alarm_led => master_alarm,
          micro_alarm_out  => micro_alarm_out,
          spike_train_leds => spike_train_out,
          uart_tx_pin      => uart_tx_signal
        );

    -- THE MISSING CLOCK GENERATOR
    clk_process :process
    begin
        clk <= '0'; wait for clk_period/2;
        clk <= '1'; wait for clk_period/2;
    end process;

stim_proc: process
        -- UPDATE THIS PATH TO YOUR CSV
        file text_file : text open read_mode is "C:/Users/omer-/Desktop/ders/grad/sdp_2/snn_vivado/PU_dataset_work/snn_res_hel/vivado/3/multi_domain_test_vectors.csv";
        variable text_line : line;
        variable file_adc_val : integer;
        variable read_ok : boolean; 
        variable sample_count : integer := 0; 
        
        variable TP, TN, FP, FN : integer := 0;
        variable acc_int : integer := 0;

        variable micro_TP, micro_TN, micro_FP, micro_FN : integer := 0;
        variable micro_acc_int : integer := 0;
        variable micro_decision : std_logic;

        variable expected_alarm : std_logic := '0'; 
    begin       
        rst <= '1'; wait for 500 ns;   
        rst <= '0'; wait for clk_period * 10;

        report "--- STARTING 6-PHASE ORCHESTRATED SIMULATION ---";

        while not endfile(text_file) loop
            readline(text_file, text_line);
            read(text_line, file_adc_val, read_ok);
            
            if read_ok then
                
                -- EXACT PHASE TRACKING (49,152 samples per phase)
                if sample_count < 49152 then
                    sim_phase <= "1. N15_HEALTHY "; expected_alarm := '0';
                elsif sample_count < 98304 then
                    sim_phase <= "2. N15_INNER   "; expected_alarm := '1';
                elsif sample_count < 147456 then
                    sim_phase <= "3. N15_OUTER   "; expected_alarm := '1';
                elsif sample_count < 196608 then
                    sim_phase <= "4. N09_HEALTHY "; expected_alarm := '0';
                elsif sample_count < 245760 then
                    sim_phase <= "5. N09_INNER   "; expected_alarm := '1';
                else
                    sim_phase <= "6. N09_OUTER   "; expected_alarm := '1';
                end if;
                
                sample_count := sample_count + 1;
                
                adc_data_in <= std_logic_vector(to_unsigned(file_adc_val, 8));
                data_valid <= '1';
                wait for clk_period;
                
                data_valid <= '0';
                wait for clk_period * 5; 
                
                -- EVALUATE MICRO-ACCURACY (Every 2048 samples)
                if sample_count > 0 and (sample_count mod 2048) = 0 then
                    micro_decision := micro_alarm_out; 
                    if micro_decision = '1' and expected_alarm = '1' then micro_TP := micro_TP + 1;
                    elsif micro_decision = '0' and expected_alarm = '0' then micro_TN := micro_TN + 1;
                    elsif micro_decision = '1' and expected_alarm = '0' then micro_FP := micro_FP + 1;
                    elsif micro_decision = '0' and expected_alarm = '1' then micro_FN := micro_FN + 1;
                    end if;
                end if;

                -- EVALUATE MACRO-ACCURACY (Every 24576 samples = 12 * 2048)
                if sample_count > 0 and (sample_count mod 24576) = 0 then
                    if master_alarm = '1' and expected_alarm = '1' then TP := TP + 1;
                    elsif master_alarm = '0' and expected_alarm = '0' then TN := TN + 1;
                    elsif master_alarm = '1' and expected_alarm = '0' then FP := FP + 1;
                    elsif master_alarm = '0' and expected_alarm = '1' then FN := FN + 1;
                    end if;
                end if;
                
            end if;
        end loop;

        wait for clk_period * 10;
        
        report "=================================================";
        report "   VIVADO HARDWARE SELF-CHECKING RESULTS   ";
        report "=================================================";
        
        if (micro_TP + micro_TN + micro_FP + micro_FN) > 0 then
            micro_acc_int := ((micro_TP + micro_TN) * 10000) / (micro_TP + micro_TN + micro_FP + micro_FN);
            report "Micro-Window Accuracy (32.0ms) : " & integer'image(micro_acc_int / 100) & "." & integer'image(micro_acc_int mod 100) & "%";
            report "Micro-Counts -> TP: " & integer'image(micro_TP) & " | TN: " & integer'image(micro_TN) & " | FP: " & integer'image(micro_FP) & " | FN: " & integer'image(micro_FN);
        end if;
        
        report "-------------------------------------------------";

        if (TP + TN + FP + FN) > 0 then
            acc_int := ((TP + TN) * 100) / (TP + TN + FP + FN);
            report "Macro-Window Accuracy (384ms)  : " & integer'image(acc_int) & "%";
            report "Macro-Counts -> TP: " & integer'image(TP) & " | TN: " & integer'image(TN) & " | FP: " & integer'image(FP) & " | FN: " & integer'image(FN);
        end if;
        
        report "=================================================";
        report "--- END OF SIMULATION ---";
        
        std.env.stop;
    end process;
end Behavior;