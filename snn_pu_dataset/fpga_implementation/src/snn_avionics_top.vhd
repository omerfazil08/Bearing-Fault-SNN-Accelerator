library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity snn_avionics_top is
    Port (
        clk              : in  STD_LOGIC;
        rst              : in  STD_LOGIC;
        adc_data_in      : in  STD_LOGIC_VECTOR(7 downto 0);
        data_valid       : in  STD_LOGIC;
        
        -- Physical SNN Debug Outputs
        master_alarm_led : out STD_LOGIC;
        micro_alarm_out  : out STD_LOGIC;  -- ADD THIS BACK
        spike_train_leds : out STD_LOGIC_VECTOR(15 downto 0);
        
        -- Avionics Telemetry
        uart_tx_pin      : out STD_LOGIC
    );
end snn_avionics_top;

architecture Behavioral of snn_avionics_top is

    -- Component Declarations
    component neuromorphic_core is
        Port (
            clk             : in  STD_LOGIC;
            rst             : in  STD_LOGIC;
            adc_data_in     : in  STD_LOGIC_VECTOR(7 downto 0);
            data_valid      : in  STD_LOGIC;
            master_alarm    : out STD_LOGIC;
            micro_alarm_out : out STD_LOGIC; 
            spike_train_out : out STD_LOGIC_VECTOR(15 downto 0)
        );
    end component;

    component uart_tx is
        Generic ( CLKS_PER_BIT : integer := 87 );
        Port (
            clk         : in  STD_LOGIC;
            rst         : in  STD_LOGIC;
            tx_start    : in  STD_LOGIC;
            tx_data     : in  STD_LOGIC_VECTOR(7 downto 0);
            tx_active   : out STD_LOGIC;
            tx_serial   : out STD_LOGIC;
            tx_done     : out STD_LOGIC
        );
    end component;

    -- Internal Signals
    signal snn_master_alarm  : STD_LOGIC;
    signal snn_micro_alarm   : STD_LOGIC;
    
    -- UART Telemetry Logic
    signal telemetry_start   : STD_LOGIC := '0';
    signal telemetry_data    : STD_LOGIC_VECTOR(7 downto 0) := (others => '0');
    signal uart_active       : STD_LOGIC;
    
    -- Edge Detection to prevent spamming the UART bus
    signal prev_master_alarm : STD_LOGIC := '0';

begin

    -- 1. Instantiate the SNN Core
    u_snn_core : neuromorphic_core
    port map (
            clk             => clk,
            rst             => rst,
            adc_data_in     => adc_data_in,
            data_valid      => data_valid,
            master_alarm    => snn_master_alarm,
            micro_alarm_out => micro_alarm_out, -- MAP IT HERE
            spike_train_out => spike_train_leds
    );
    
    master_alarm_led <= snn_master_alarm;

    -- 2. Instantiate the UART Telemetry Transmitter
    u_telemetry_tx : uart_tx
    generic map ( CLKS_PER_BIT => 87 ) -- 10 MHz / 115200 Baud
    port map (
        clk       => clk,
        rst       => rst,
        tx_start  => telemetry_start,
        tx_data   => telemetry_data,
        tx_active => uart_active,
        tx_serial => uart_tx_pin,
        tx_done   => open
    );

    -- 3. Avionics Event Manager (Edge Detection & Packet Formatting)
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                prev_master_alarm <= '0';
                telemetry_start   <= '0';
                telemetry_data    <= (others => '0');
            else
                telemetry_start <= '0'; -- Default state
                
                -- Detect the RISING EDGE of the SNN's anomaly detection
                if snn_master_alarm = '1' and prev_master_alarm = '0' then
                    if uart_active = '0' then
                        -- Send 0xFF (CRITICAL ALARM / JAMMER DETECTED)
                        telemetry_data  <= x"FF";
                        telemetry_start <= '1';
                    end if;
                end if;
                
                prev_master_alarm <= snn_master_alarm;
            end if;
        end if;
    end process;

end Behavioral;