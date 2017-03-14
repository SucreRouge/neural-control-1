reset

set terminal pdf

cd "data"
set output "../figures/wcc.pdf"
set yrange [-3 to 3]
set ylabel "weight change (arbitrary units)"
set xlabel "error decay rate"
set key top right
plot "wcc0.5.txt" w l lw 2 lc 1 t "", "wcc1.0.txt" w l lw 2 lc 2 t "", "wcc2.0.txt" w l lw 2 lc 3 t "", \
     "wcc0.5.txt" u 1:3 w l lw 2 lc 1 dt "-" t "", "wcc1.0.txt" u 1:3 w l lw 2 lc 2 dt "-" t "", "wcc2.0.txt" u 1:3 w l lw 2 lc 3 dt "-" t "",\
     "" u 1:(0/0) w l lc 1 lw 2 t "r = 0.5", "" u 1:(0/0) w l lc 2 lw 2 t "r = 1.0", "" u 1:(0/0) w l lc 3 lw 2 t "r = 2.0",\
     0 w l lc black t ""

cd ".."
set output "/dev/null"
