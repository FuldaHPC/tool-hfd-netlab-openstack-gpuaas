// pi with the midpoint rule on all GPUs of the instance (pi_gpu.chpl uses one): a coforall starts one
// task per GPU (here.gpus), and inside "on gpu" the forall loop becomes a GPU kernel (@assertOnGpu makes
// compilation fail if it would not). Every GPU takes its share of the chunks; each GPU thread sums one
// chunk of the steps into an array in GPU memory, and + reduce adds the chunk sums on the GPU.
use Common, Time;

config const chunks = 1 << 20;

proc piGpu(n: int): real {
  const g = here.gpus.size;
  var sums: [0..<g] real;
  coforall (gpu, k) in zip(here.gpus, 0..) do on gpu {
    // Constants of the kernel are declared inside "on gpu", so that the kernel gets their values.
    const h = 1.0 / n, m = chunks, size = (n + m - 1) / m, mine = k * m / g ..< (k + 1) * m / g;
    var parts: [mine] real;
    @assertOnGpu
    forall c in mine {
      var s = 0.0;
      for i in c * size..<min((c + 1) * size, n) {
        const x = (i + 0.5) * h;
        s += 4.0 / (1.0 + x * x);
      }
      parts[c] = s;
    }
    sums[k] = + reduce parts;
  }
  return (+ reduce sums) / n;
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
const gpus = here.gpus.size;
exit(piReport("GPU (Chapel forall on " + gpus:string + " GPU" + (if gpus > 1 then "s" else "") + ")", approx, piSteps,
              watch.elapsed()));
