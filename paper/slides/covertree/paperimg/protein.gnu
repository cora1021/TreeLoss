load "settings-lineplot.gnuh"
set output "protein.tex"

set ylabel "total distance comparisons (millions)"
set xlabel "number of data points (thousands)"

#set ytics autofreq 1 0.1

set xrange [0:250]

#set key center top outside

plot "protein.dat" using ($1/1000):(($6+$7) / 1000000) lc rgb "#770000" lw 2 lt 1 with lines title "original cover tree (construction+query)", \
     "protein.dat" using ($1/1000):($6      / 1000000) lc rgb "#770000" lw 2 lt 2 with lines title "original cover tree (construction only)", \
     "protein.dat" using ($1/1000):(($2+$3) / 1000000) lc rgb "#007700" lw 4 lt 1 with lines title "simplified cover tree (construction+query)", \
     "protein.dat" using ($1/1000):($2      / 1000000) lc rgb "#007700" lw 4 lt 2 with lines title "simplified cover tree (construction only)", \
     "protein.dat" using ($1/1000):(($4+$5) / 1000000) lc rgb "#0000ff" lw 8 lt 1 with lines title "nearest ancestor cover tree (construction+query)", \
     "protein.dat" using ($1/1000):($4      / 1000000) lc rgb "#0000ff" lw 8 lt 2 with lines title "nearest ancestor cover tree (construction only)"


