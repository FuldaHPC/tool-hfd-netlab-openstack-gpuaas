// pi with the midpoint rule on the GPU: inside "on here.gpus[0]" the forall loop becomes a GPU kernel
// (@assertOnGpu makes compilation fail if it would not). Each GPU thread sums one chunk of the steps
// into an array in GPU memory, and + reduce adds the chunk sums on the GPU.
use Common, Time;

config const chunks = 1 << 20;

proc piGpu(n: int): real {
  var approx = 0.0;
  on here.gpus[0] {
    const h = 1.0 / n, m = chunks, size = (n + m - 1) / m;
    var parts: [0..<m] real;
    @assertOnGpu
    forall c in 0..<m {
      var s = 0.0;
      for i in c * size..<min((c + 1) * size, n) {
        const x = (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
      }
      parts[c] = s;
    }
    approx = h * (+ reduce parts);
  }
  return approx;
}

if here.gpus.size == 0 {
  writeln("No GPU found");
  exit(1);
}
piGpu(1000);  // warm-up: initializes the GPU and loads the kernel
var watch: stopwatch;
watch.start();
const approx = piGpu(piSteps);
watch.stop();
exit(piReport("GPU (Chapel forall on here.gpus[0])", approx, piSteps, watch.elapsed()));
