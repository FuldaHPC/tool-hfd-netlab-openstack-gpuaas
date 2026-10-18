// Matrix product C = A B on the GPU: arrays declared inside "on here.gpus[0]" live in GPU memory, and
// assigning a host array to them copies it. The forall over all n*n elements of C becomes a GPU kernel
// with one thread per element (@assertOnGpu makes compilation fail if it would not); a forall over the
// 2D domain itself was 60 times slower with Chapel 2.10. The time includes the copies.
use Common, Time;

proc matmulGpu(const ref a: [?D] real(32), const ref b: [D] real(32), ref c: [D] real(32)) {
  on here.gpus[0] {
    const n = D.dim(0).size;
    var aG, bG, cG: [D] real(32);
    aG = a;
    bG = b;
    @assertOnGpu
    forall ij in 0..<n * n {
      const i = ij / n, j = ij % n;
      var s: real(32) = 0;
      for k in 0..<n do s += aG[i, k] * bG[k, j];
      cG[i, j] = s;
    }
    c = cG;
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
exit(matrixReport("GPU (Chapel forall on here.gpus[0])", a, b, c, watch.elapsed()));
