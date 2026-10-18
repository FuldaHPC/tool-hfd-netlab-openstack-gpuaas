// Shared by the Chapel examples of this folder: problem sizes, test matrices and the result checks,
// so that the programs themselves only show the programming model.
module Common {
  use IO, Math;

  config const piSteps = 1_000_000_000,  // steps of the midpoint rule for pi
               matrixN = 2048;           // the matrices are matrixN x matrixN real(32)

  // pi is the integral of 4/(1+x^2) over [0,1]; the midpoint rule sums 4/(1+x^2) at x = (i+0.5)/n.
  // x = (i + 0.5) * h; sum += 4 / (1 + x * x): 3 adds, 2 multiplies, 1 division per step.
  // Rounding makes the error about 1e-13; one missing step would add at least 2e-9.
  param piFlopsPerStep = 6;
  proc piReport(title: string, approx: real, n: int, seconds: real): int {
    const error = abs(approx - Math.pi);
    writef("%s: pi = %.12dr (error %.1er) in %.3dr s (%.1dr GFLOP/s)\n", title, approx, error, seconds,
           piFlopsPerStep * n / seconds / 1e9);
    return if error < 1e-9 then 0 else 1;
  }

  // Small integers, so that every product and sum in C = A B is exact in single precision.
  proc matrixInit(ref a: [?D] real(32), ref b: [D] real(32)) {
    forall (i, j) in D {
      a[i, j] = ((i + 2 * j) % 8): real(32);
      b[i, j] = ((3 * i + j) % 8): real(32);
    }
  }

  // Checks C x == A (B x) for x = (1, 2, ..., n) in O(n^2); exact, as all values are integers below
  // 2^53. x has no zero, so that every element of C counts.
  proc matrixCorrect(const ref a: [?D] real(32), const ref b: [D] real(32), const ref c: [D] real(32)): bool {
    const n = D.dim(0).size;
    var bx: [0..<n] real, ok = true;
    forall k in 0..<n {
      var s = 0.0;
      for j in 0..<n do s += b[k, j]: real * (j + 1);
      bx[k] = s;
    }
    forall i in 0..<n with (&& reduce ok) {
      var cx = 0.0, abx = 0.0;
      for j in 0..<n {
        cx += c[i, j]: real * (j + 1);
        abx += a[i, j]: real * bx[j];
      }
      ok &&= cx == abx;
    }
    return ok;
  }

  proc matrixReport(title: string, const ref a: [?D] real(32), const ref b: [D] real(32),
                    const ref c: [D] real(32), seconds: real): int {
    const n = D.dim(0).size, ok = matrixCorrect(a, b, c);
    writef("%s: %ix%i matrix product %s in %.3dr s (%.1dr GFLOP/s)\n", title, n, n,
           if ok then "correct" else "WRONG", seconds, 2.0 * n * n * n / seconds / 1e9);
    return if ok then 0 else 1;
  }
}
