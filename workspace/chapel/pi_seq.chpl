// pi with the midpoint rule in a serial for loop on one CPU core, as reference for pi_cpu.chpl, which
// differs only in forall with a + reduce intent.
use Common, Time;

proc piSeq(n: int): real {
  const h = 1.0 / n;
  var sum = 0.0;
  for i in 0..<n {
    const x = (i + 0.5) * h;
    sum += 4.0 / (1.0 + x * x);
  }
  return h * sum;
}

var watch: stopwatch;
watch.start();
const approx = piSeq(piSteps);
watch.stop();
exit(piReport("CPU sequential (Chapel for)", approx, piSteps, watch.elapsed()));
