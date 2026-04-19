----------------------------------------------------------------------------------
-- Module: main_system (Top-Level Wrapper)
-- Description: Integrates the XADC, 12kHz Timer, DC Offset, and the 1-8-1 SNN Core.
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity main_system is
    Port (
        clk             : in  STD_LOGIC;
        rst             : in  STD_LOGIC;
        vauxp6          : in  STD_LOGIC; -- Analog Input + from ADXL1002
        vauxn6          : in  STD_LOGIC; -- Analog Input - (Ground)
        master_alarm    : out STD_LOGIC;
        micro_alarm_out : out STD_LOGIC;
        spike_train_out : out STD_LOGIC_VECTOR(15 downto 0)
    );
end main_system;

architecture Behavioral of main_system is

    -- 1. Component Declaration for the XADC IP
component xadc_wiz_0
        port (
            di_in       : in  STD_LOGIC_VECTOR(15 downto 0);
            daddr_in    : in  STD_LOGIC_VECTOR(6 downto 0);
            den_in      : in  STD_LOGIC;
            dwe_in      : in  STD_LOGIC;
            drdy_out    : out STD_LOGIC;
            do_out      : out STD_LOGIC_VECTOR(15 downto 0);
            dclk_in     : in  STD_LOGIC;
            reset_in    : in  STD_LOGIC;
            vp_in       : in  STD_LOGIC;  -- ADDED THIS
            vn_in       : in  STD_LOGIC;  -- ADDED THIS
            vauxp6      : in  STD_LOGIC;
            vauxn6      : in  STD_LOGIC;
            channel_out : out STD_LOGIC_VECTOR(4 downto 0);
            eoc_out     : out STD_LOGIC;
            alarm_out   : out STD_LOGIC;
            eos_out     : out STD_LOGIC;
            busy_out    : out STD_LOGIC
        );
    end component;

    -- 2. Component Declaration for your Frozen SNN Core
    component neuromorphic_core
        port (
            clk             : in  STD_LOGIC;
            rst             : in  STD_LOGIC;
            adc_data_in     : in  STD_LOGIC_VECTOR(7 downto 0);
            data_valid      : in  STD_LOGIC;
            master_alarm    : out STD_LOGIC;
            micro_alarm_out : out STD_LOGIC;
            spike_train_out : out STD_LOGIC_VECTOR(15 downto 0)
        );
    end component;

    -- Constants
    -- 100 MHz clock / 12,000 Hz = 8333 clock cycles per sample
    constant TICK_12KHZ_LIMIT : integer := 8333; 
    -- The resting state of the voltage divider (0.38V mapped to 8-bit)
    constant RESTING_OFFSET   : unsigned(7 downto 0) := to_unsigned(97, 8); 

    -- Signals
    signal clk_counter  : integer range 0 to 8333 := 0;
    signal tick_12khz   : STD_LOGIC := '0';
    
    signal xadc_den     : STD_LOGIC := '0';
    signal xadc_drdy    : STD_LOGIC;
    signal xadc_do      : STD_LOGIC_VECTOR(15 downto 0);
    
    signal xadc_8bit    : unsigned(7 downto 0);
    signal abs_amp      : unsigned(7 downto 0);
    
    signal snn_data_in  : STD_LOGIC_VECTOR(7 downto 0) := (others => '0');
    signal snn_valid    : STD_LOGIC := '0';

begin

    -- Instantiate the XADC
     xadc_inst : xadc_wiz_0
        port map (
            di_in       => (others => '0'),
            daddr_in    => "0010110", 
            den_in      => xadc_den,
            dwe_in      => '0',
            drdy_out    => xadc_drdy,
            do_out      => xadc_do,
            dclk_in     => clk,
            reset_in    => rst,
            vp_in       => '0',       -- TIED TO GROUND
            vn_in       => '0',       -- TIED TO GROUND
            vauxp6      => vauxp6,
            vauxn6      => vauxn6,
            channel_out => open,
            eoc_out     => open,
            alarm_out   => open,
            eos_out     => open,
            busy_out    => open
        );

    -- Instantiate the Frozen SNN Core
    snn_core : neuromorphic_core
        port map (
            clk             => clk,
            rst             => rst,
            adc_data_in     => snn_data_in,
            data_valid      => snn_valid,
            master_alarm    => master_alarm,
            micro_alarm_out => micro_alarm_out,
            spike_train_out => spike_train_out
        );

    -- 12 kHz Timing & ADC Reading Process
    process(clk)
    begin
        if rising_edge(clk) then
            if rst = '1' then
                clk_counter <= 0;
                xadc_den <= '0';
                snn_valid <= '0';
            else
                -- Step 1: 12 kHz Clock Divider
                if clk_counter = TICK_12KHZ_LIMIT then
                    clk_counter <= 0;
                    xadc_den <= '1'; -- Request data from XADC
                else
                    clk_counter <= clk_counter + 1;
                    xadc_den <= '0';
                end if;

                -- Step 2: Receive XADC Data and Apply DC Offset Math
                if xadc_drdy = '1' then
                    -- Extract top 8 bits from the 12-bit MSB-aligned XADC data
                    xadc_8bit <= unsigned(xadc_do(15 downto 8));
                    
                    -- Perform Python's np.abs() equivalent based on resting offset
                    if unsigned(xadc_do(15 downto 8)) > RESTING_OFFSET then
                        abs_amp <= unsigned(xadc_do(15 downto 8)) - RESTING_OFFSET;
                    else
                        abs_amp <= RESTING_OFFSET - unsigned(xadc_do(15 downto 8));
                    end if;
                    
                    snn_valid <= '1';
                else
                    snn_valid <= '0';
                end if;
                
                -- Step 3: Route final data to the SNN
                snn_data_in <= std_logic_vector(abs_amp);
                
            end if;
        end if;
    end process;

end Behavioral;