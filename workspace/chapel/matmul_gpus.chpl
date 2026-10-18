// Matrix product C = A B on all GPUs of the instance (matmul_gpu.chpl uses one): a coforall starts one
// task per GPU (here.gpus), and every GPU computes its block of rows of C. Arrays declared inside
// "on gpu" live in GPU memory, and assigning host arrays (or slices) to them copies them. The forall
// over all elements of the block becomes a GPU kernel with one thread per element (@assertOnGpu makes
// compilation fail if it would not). On one GPU the kernel takes about 8 % longer than matmul_gpu.chpl's
// because of the row offset. The time includes the copies.
use Common, Time;

proc matmulGpu(const ref a: [?D] real(32), const ref b: [D] real(32), ref c: [D] real(32)) {
  const g = here.gpus.size;
  coforall (gpu, k) in zip(here.gpus, 0..) do on gpu {
    // Constants of the kernel are declared inside "on gpu", so that the kernel gets their values.
    const n = D.dim(0).size, rows = k * n / g ..< (k + 1) * n / g;  // the rows of GPU k
    var aG, cG: [rows, 0..<n] real(32);
    var bG: [D] real(32);
    aG = a[rows, ..];
    bG = b;
    const first = rows.low, count = rows.size * n;
    @assertOnGpu
    forall ij in 0..<count {
      const i = first + ij / n, j = ij % n;
      var s: real(32) = 0;
      for l in 0..<n do s += aG[i, l] * bG[l, j];
      cG[i, j] = s;
    }
    c[rows, ..] = cG;
  }
}

if here.gpus.size == 0 {
  writeln("No GPU found");
  exit(1);
}
const D = {0..<matrixN, 0..<matrixN};
var a, b, c: [D] real(32);
matrixInit(a, b);
{  // warm-up: initializes the GPU and loads the kernel
  var x, y, z: [{0..<16, 0..<16}] real(32);
  matmulGpu(x, y, z);
}
var watch: stopwatch;
watch.start();
matmulGpu(a, b, c);
watch.stop();
const gpus = here.gpus.size;
exit(matrixReport("GPU (Chapel forall on " + gpus:string + " GPU" + (if gpus > 1 then "s" else "") + ")", a, b, c,
                  watch.elapsed()));
