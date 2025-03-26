NAME          SET_PARTITION
ROWS
 N  COST
 E  COVER1
 E  COVER2
 E  COVER3
 E  COVER4
 E  COVER5
 E  COVER6

COLUMNS
    X1        COST      2    COVER1    1    COVER2    1
    X2        COST      3    COVER3    1    COVER4    1
    X3        COST      1    COVER1    1    COVER3    1    COVER5    1
    X4        COST      4    COVER2    1    COVER4    1    COVER6    1
    X5        COST      5    COVER3    1    COVER5    1    COVER6    1

RHS
    RHS       COVER1    1
    RHS       COVER2    1
    RHS       COVER3    1
    RHS       COVER4    1
    RHS       COVER5    1
    RHS       COVER6    1

BOUNDS
 LO BND       X1        0
 LO BND       X2        0
 LO BND       X3        0
 LO BND       X4        0
 LO BND       X5        0
 UP BND       X1        1
 UP BND       X2        1
 UP BND       X3        1
 UP BND       X4        1
 UP BND       X5        1

ENDATA
