
// computing linear gradients for parameters w and b for y = wx + b linear reg

double compute_dw(double *xs, double *ys, double w, double b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += 2 * ((w * xs[i] + b) - ys[i]) * xs[i];
    }
    return sum / n;
}

double compute_db(double *xs, double *ys, double w, double b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += 2 * ((w * xs[i] + b) - ys[i]);
    }
    return sum / n;
}