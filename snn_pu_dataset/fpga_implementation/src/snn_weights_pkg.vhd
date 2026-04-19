library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package snn_weights_pkg is

    constant INPUT_SIZE  : integer := 1;
    constant HIDDEN_SIZE : integer := 32;

    constant THRESHOLD_L1 : signed(31 downto 0) := to_signed(2925, 32);
    constant THRESHOLD_L2 : signed(31 downto 0) := to_signed(709, 32);

    type weight_array is array (0 to HIDDEN_SIZE-1) of signed(7 downto 0);
    type leak_array_type is array (0 to HIDDEN_SIZE-1) of integer range 0 to 7;
    -- W1 (Input to L1 Hidden Layer Weights)
    constant FC1_WEIGHTS : weight_array := (
        to_signed(46, 8), to_signed(-25, 8), to_signed(47, 8), to_signed(-16, 8), 
        to_signed(46, 8), to_signed(-86, 8), to_signed(-66, 8), to_signed(52, 8), 
        to_signed(118, 8), to_signed(22, 8), to_signed(-36, 8), to_signed(78, 8), 
        to_signed(-96, 8), to_signed(-73, 8), to_signed(8, 8), to_signed(-15, 8), 
        to_signed(-17, 8), to_signed(109, 8), to_signed(-22, 8), to_signed(16, 8), 
        to_signed(-4, 8), to_signed(28, 8), to_signed(127, 8), to_signed(-66, 8), 
        to_signed(15, 8), to_signed(81, 8), to_signed(111, 8), to_signed(-103, 8), 
        to_signed(59, 8), to_signed(-22, 8), to_signed(-52, 8), to_signed(-72, 8)
    );

    -- W2 (L1 to Output Neuron Weights)
    constant FC2_WEIGHTS : weight_array := (
        to_signed(127, 8), to_signed(-62, 8), to_signed(-88, 8), to_signed(-127, 8), 
        to_signed(-99, 8), to_signed(-11, 8), to_signed(95, 8), to_signed(88, 8), 
        to_signed(10, 8), to_signed(3, 8), to_signed(47, 8), to_signed(4, 8), 
        to_signed(116, 8), to_signed(127, 8), to_signed(80, 8), to_signed(-51, 8), 
        to_signed(-100, 8), to_signed(-66, 8), to_signed(73, 8), to_signed(-33, 8), 
        to_signed(-3, 8), to_signed(-8, 8), to_signed(127, 8), to_signed(74, 8), 
        to_signed(-39, 8), to_signed(127, 8), to_signed(118, 8), to_signed(-106, 8), 
        to_signed(50, 8), to_signed(-74, 8), to_signed(20, 8), to_signed(49, 8)
    );

    -- LEAK SHIFTS (Bit-shift values for decay)
    constant L1_LEAK_SHIFTS : leak_array_type := (
        3, 3, 2, 4, 3, 0, 2, 2, 0, 2, 0, 1, 0, 0, 3, 2, 
        1, 2, 1, 4, 2, 1, 3, 1, 1, 2, 2, 0, 2, 0, 2, 0
    );

end package snn_weights_pkg;