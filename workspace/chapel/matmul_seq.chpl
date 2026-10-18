// Matrix product C = A B in serial for loops on one CPU core, as reference for matmul_cpu.chpl, which
// differs only in forall over the rows. The loop order i-k-j lets the innermost loop run along rows.
use Common, Time;

proc matmul(const ref a: [?D] real(32), const ref b: [D] real(32), ref c: [D] real(32)) {
  const n = D.dim(0).size;
  for i in 0..<n {
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
var watch: stopwatch;
watch.start();
matmul(a, b, c);
watch.stop();
exit(matrixReport("CPU sequential (Chapel for)", a, b, c, watch.elapsed()));
