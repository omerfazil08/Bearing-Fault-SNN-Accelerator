----------------------------------------------------------------------------------
-- Module Name: neuromorphic_core
-- Description: Temporal Spiking Neural Network for Early Fault Detection
-- Architecture: 1-8-1 LIF Network (Hardware Quantized)
-- Debouncer: 75-Window Macro-Relay (Trips at >= 25 Micro-Alarms)
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.snn_weights_pkg.all;

entity neuromorphic_core is
    Port (
        clk             : in  STD_LOGIC;
        rst             : in  STD_LOGIC;
        adc_data_in     : in  STD_LOGIC_VECTOR(7 downto 0);
        data_valid      : in  STD_LOGIC;
        master_alarm    : out STD_LOGIC;
        micro_alarm_out : out STD_LOGIC; -- ? NEW DEBUG PIN
        spike_train_out : out STD_LOGIC_VECTOR(15 downto 0)
    );
end neuromorphic_core;

architecture Behavioral of neuromorphic_core is

    -- Temporal SNN Counters (Updated for 32-sample window)
    signal step_count   : integer range 0 to 31 := 0; 
    signal macro_count  : integer range 0 to 74 := 0;

    -- LIF Membrane Voltages
    type mem1_array is array (0 to HIDDEN_SIZE-1) of signed(31 downto 0);
    signal mem1         : mem1_array := (others => (others => '0'));
    signal mem2         : signed(31 downto 0) := (others => '0');

    -- Relay and Shift Register Logic
    signal micro_alarm_flag : STD_LOGIC := '0';
    signal macro_alarms     : integer range 0 to 75 := 0;
    signal shift_reg        : STD_LOGIC_VECTOR(15 downto 0) := (others => '0');

begin

    process(clk)
        -- Process Variables (Updates immediately in the same clock cycle)
        variable adc_val              : signed(8 downto 0);
        variable cur1, cur2           : signed(31 downto 0);
        variable spk1                 : std_logic_vector(HIDDEN_SIZE-1 downto 0);
        variable spk2                 : std_logic;
        variable mem1_leaked          : signed(31 downto 0);
        variable mem2_leaked          : signed(31 downto 0);
        variable current_window_alarm : integer range 0 to 1;
    begin
        if rising_edge(clk) then
            if rst = '1' then
                step_count <= 0;
                macro_count <= 0;
                mem1 <= (others => (others => '0'));
                mem2 <= (others => '0');
                micro_alarm_flag <= '0';
                macro_alarms <= 0;
                master_alarm <= '0';
                micro_alarm_out <= '0';
                shift_reg <= (others => '0');
                spike_train_out <= (others => '0');
                
            elsif data_valid = '1' then
                -- 1. Read ADC (Zero-extend to 9 bits to ensure a positive signed integer)
                adc_val := signed('0' & adc_data_in);

                -- 2. Process Layer 1 (Hidden Neurons)
                spk1 := (others => '0');
                for i in 0 to HIDDEN_SIZE-1 loop
                    cur1 := resize(adc_val * FC1_WEIGHTS(i), 32);
                    mem1_leaked := shift_right(mem1(i), 1);

                    if (mem1_leaked + cur1) > THRESHOLD_L1 then
                        spk1(i) := '1';
                        mem1(i) <= (others => '0'); 
                    else
                        mem1(i) <= mem1_leaked + cur1; 
                    end if;
                end loop;

                -- 3. Process Layer 2 (Output Neuron)
                cur2 := (others => '0');
                for i in 0 to HIDDEN_SIZE-1 loop
                    if spk1(i) = '1' then
                        cur2 := cur2 + resize(FC2_WEIGHTS(i), 32);
                    end if;
                end loop;

                mem2_leaked := shift_right(mem2, 1);
                
                spk2 := '0';
                if (mem2_leaked + cur2) > THRESHOLD_L2 then
                    spk2 := '1';
                    mem2 <= (others => '0');
                    micro_alarm_flag <= '1'; 
                else
                    mem2 <= mem2_leaked + cur2;
                end if;

                -- Shift the current spike into the tracking register
                shift_reg <= shift_reg(14 downto 0) & spk2;

                -- 4. Temporal Window Management (THE FIX: 31 instead of 15)
                if step_count = 31 then
                    step_count <= 0;
                    mem1 <= (others => (others => '0'));
                    mem2 <= (others => '0');
                    micro_alarm_flag <= '0'; 

                    -- Latch the completed 16-bit train to the physical output port
                    spike_train_out <= shift_reg(14 downto 0) & spk2;

                    -- Check if the 2.6ms window caught a fault
                    current_window_alarm := 0;
                    if micro_alarm_flag = '1' or spk2 = '1' then
                        current_window_alarm := 1;
                        micro_alarm_out <= '1';
                    else
                        micro_alarm_out <= '0';    
                    end if;

                    -- 5. The Hardware Debouncer (100ms Revolution Check)
                    if macro_count = 74 then
                        macro_count <= 0;
                        
                        -- MASTER TRIP LOGIC
                        if (macro_alarms + current_window_alarm) >= 25 then
                            master_alarm <= '1'; 
                        else
                            master_alarm <= '0'; 
                        end if;
                        
                        macro_alarms <= 0; 
                    else
                        macro_count <= macro_count + 1;
                        macro_alarms <= macro_alarms + current_window_alarm;
                    end if;
                    
                else
                    step_count <= step_count + 1;
                end if;
            end if;
        end if;
    end process;

end Behavioral;