load "graphsettings.gnuh"
set output "parallel.tex"

set ylabel "normalized total runtime\n(both \\emph\{construction\} and \\emph\{query\})\n  "

set logscale y 2
set yrange [0.03125:4]
set ytics ("$2^{+1}$" 2, "$2^{+0}$" 1, "$2^{-1}$" 0.5, "$2^{-2}$" 0.25, "$2^{-3}$" 0.125, "$2^{-4}$" 0.0625)
set grid ytics lt 0 lw 1 lc rgb "#bbbbbb"

#min(x,y)=(x<y)?x:y
min(x,y)=x

set label "\\tiny 1" at graph 0.13, graph 0.81
set label "\\tiny 1" at graph 0.155, graph 0.88
set label "\\tiny 1" at graph 0.18, graph 0.75
set label "\\tiny 2" at graph 0.20, graph 0.62
set label "\\tiny 4" at graph 0.225, graph 0.44
set label "\\tiny 8" at graph 0.247, graph 0.33
set label "\\tiny 16" at graph 0.267, graph 0.24

plot "parallel-total.dat" using ($20/$8):xtic(1) lc rgb "#663300" lt 1,\
     "parallel-total.dat" using ($16/$8):xtic(1) lc rgb "#cc9900" lt 1, \
     "parallel-total.dat" using ($8 /$8):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-total.dat" using ($9 /$8):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-total.dat" using ($10/$8):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-total.dat" using ($11/$8):xtic(1) lc rgb "#0000ff" lt 1, \
     "parallel-total.dat" using ($12/$8):xtic(1) lc rgb "#0000ff" lt 1
     #"parallel-total.dat" using ($13/$8):xtic(1) lc rgb "#0000ff" lt 1

#plot "parallel-total.dat" using ($20/min($2,$8)):xtic(1) lc rgb "#ff0000" lt 1,\
     #"parallel-total.dat" using ($16/min($2,$8)):xtic(1) lc rgb "#00ff00" lt 1, \
     #"parallel-total.dat" using (min($2,$8) /min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1, \
     #"parallel-total.dat" using (min($3,$9) /min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1, \
     #"parallel-total.dat" using (min($4,$10) /min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1, \
     #"parallel-total.dat" using (min($5,$11) /min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1, \
     #"parallel-total.dat" using (min($6,$12) /min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1, \
     #"parallel-total.dat" using (min($7,$13) /min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1

     #"parallel-total.dat" using ($20/min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1
     #"parallel-total.dat" using ($14/min($2,$8)):xtic(1) lc rgb "#00ff00" lt 1, \
     #"parallel-total.dat" using ($15/min($2,$8)):xtic(1) lc rgb "#00ff00" lt 1, \
     #"parallel-total.dat" using ($17/min($2,$8)):xtic(1) lc rgb "#00ff00" lt 1, \
     #"parallel-total.dat" using ($19/min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1, \
     #"parallel-total.dat" using ($21/min($2,$8)):xtic(1) lc rgb "#0000ff" lt 1


#covtype         31          16          10.5        8.6         7.5
#corel           83.75   44.58   23.86   18.23   14.13   11.89   43.29   27.3    19.15   17.48   18.24   20.03   9.76    13.12   74.98   75.56   61.36   61.18   20.00   22.5
#artificial4     9.55    5.25    4.44    2.15    1.95    1.55    10.25   5.86    3.15    2.28    1.90    2.11    8.45    6.64    13.5    13.49   23.79   23.79   5.42    6.31
#calhousing      2.8     2.06    1.39    1.57    1.31    1.36    2.84    2.35    1.35    1.11    1.09    1.17    0.25    0.2     0.42    0.41    0.49    0.49    0.31    0.2
#faces           4.54    2.34    1.4     0.88    0.88    0.77    3.98    2.27    1.41    0.88    0.85    0.75    0.69    0.98    1.5     1.5     1.26    1.25    0.6     0.48
