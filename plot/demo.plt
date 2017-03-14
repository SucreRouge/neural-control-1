reset

set terminal pdf

set xlabel "time steps"
set ylabel "system state"

set output "figures/demo1.pdf"
plot "data/data_1.txt" u 0:1 w l lw 2 t "PID", "" u 0:7 w l lw 2 t "PIDNN", "" u 0:3 w l lc black dt "-" t ""

set output "figures/demo2.pdf"
set yrange [-1 to 2]
set key top left
plot "data/data_2.txt" u 0:1 w l lw 2 t "PID", "" u 0:7 w l lw 2 t "PIDNN", "" u 0:3 w l lc black dt "-" t ""

set output "/dev/null"
