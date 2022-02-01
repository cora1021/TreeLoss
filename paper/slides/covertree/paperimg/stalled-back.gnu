load "graphsettings.gnuh"
set output "stalled-back.tex"

set ylabel "fraction of backend CPU cycles stalled"

set yrange [0:1]
set ytics autofreq 1 0.1

plot "cache.dat" using ($3/100):xtic(1) lc rgb "#ff0000" lt 1,\
     "cache.dat" using ($9/100):xtic(1) lc rgb "#00ff00" lt 1,\
     "cache.dat" using ($12/100):xtic(1) lc rgb "#0000ff" lt 1,\
     "cache.dat" using ($6/100):xtic(1) lc rgb "#7777ff" lt 1




