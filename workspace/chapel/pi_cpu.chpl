// pi with the midpoint rule: a forall loop with a + reduce intent on the CPU cores.
use Common, Time;

proc piCpu(n: int): real {
  const h = 1.0 / n;
  var sum = 0.0;
  forall i in 0..<n with (+ reduce sum) {
    const x = (i + 0.5) * h;
    sum += 4.0 / (1.0 + x * x);
  }
  return h * sum;
}

piCpu(1000);  // warm-up: starts the tasks
var watch: stopwatch;
watch.start();
const approx = piCpu(piSteps);
watch.stop();
exit(piReport("CPU (Chapel forall, " + here.maxTaskPar: string + " tasks)", approx, piSteps, watch.elapsed()));
