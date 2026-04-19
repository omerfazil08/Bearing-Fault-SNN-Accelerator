library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

package snn_weights_pkg is
    constant INPUT_SIZE  : integer := 1;
    constant HIDDEN_SIZE : integer := 8;

    constant THRESHOLD_L1 : signed(31 downto 0) := to_signed(1318, 32);
    constant THRESHOLD_L2 : signed(31 downto 0) := to_signed(70, 32);

    type weight_array is array (0 to HIDDEN_SIZE-1) of signed(7 downto 0);

    constant FC1_WEIGHTS : weight_array := (
        to_signed(-110, 8),
        to_signed(-127, 8),
        to_signed(-108, 8),
        to_signed(-41, 8),
        to_signed(-94, 8),
        to_signed(126, 8),
        to_signed(-22, 8),
        to_signed(82, 8)
    );
    constant FC2_WEIGHTS : weight_array := (
        to_signed(1, 8),
        to_signed(65, 8),
        to_signed(96, 8),
        to_signed(41, 8),
        to_signed(61, 8),
        to_signed(70, 8),
        to_signed(-75, 8),
        to_signed(32, 8)
    );
end package snn_weights_pkg;
