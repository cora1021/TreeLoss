load "graphsettings.gnuh"
set output "cache.tex"

set ylabel "cache miss rate\n(cache misses / cache accesses)"

set yrange [0:1]
set ytics autofreq 1 0.2

plot "cache.dat" using ($4/100):xtic(1) lc rgb "#663300" lt 1,\
     "cache.dat" using ($10/100):xtic(1) lc rgb "#cc9900" lt 1,\
     "cache.dat" using ($13/100):xtic(1) lc rgb "#0000ff" lt 1,\
     "cache.dat" using ($7/100):xtic(1) lc rgb "#7777ff" lt 1


#from cache.dat
#1000000         83.45               67.69               74.982                                                          80.22           72.40           78.218
#calhousing      39.49               27.41               41.396          22              42              6               37.63           27.92           24.613          22              42              6
