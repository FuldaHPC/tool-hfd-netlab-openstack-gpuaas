// Matrix product C = A B with a forall loop over the rows of C on the CPU cores. The loop order
// i-k-j lets the innermost loop run along rows of B and C.
use Common, Time;

proc matmul(const ref a: [?D] real(32), const ref b: [D] real(32), ref c: [D] real(32)) {
  const n = D.dim(0).size;
  forall i in 0..<n {
    for j in 0..<n do c[i, j] = 0;
    for k in 0..<n {
      const aik = a[i, k];
      for j in 0..<n do c[i, j] += aik * b[k, j];
    }
  }
}

const D = {0..<matrixN, 0..<matrixN};
var a, b, c: [D] real(32);
matrixInit(a, b);
{  // warm-up: starts the tasks
  var x, y, z: [{0..<16, 0..<16}] real(32);
  matmul(x, y, z);
}
var watch: stopwatch;
watch.start();
matmul(a, b, c);
watch.stop();
exit(matrixReport("CPU (Chapel forall, " + here.maxTaskPar: string + " tasks)", a, b, c, watch.elapsed()));
