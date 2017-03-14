reset

set terminal pdf

set output "figures/quiver.pdf"
set cbrange [1.15:1.6]
set xrange [0 to 10]
set yrange [0 to 10]
unset key
set xlabel "p"
set ylabel "d"
unset colorbox
scale = 20
min(x, y) = x > y ? y : x

set palette defined ( 0 "#6ab671", 0.05 "#c6f98a", 0.15 "#ffffcf", 0.3 "#fdde91", 0.8 "#d7696c")

plot "data/quiver.txt" u 1:2:(log(log($5))) w image, "" every 3:3 u 1:2:($3*scale):($4*scale) with vectors lc black
set output "/dev/null"