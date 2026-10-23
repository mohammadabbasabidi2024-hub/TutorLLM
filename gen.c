/* TutorLLM dataset generator: solves OS problems exactly in C and emits
 * instruction-tuning records as JSONL. Ground truth comes from code, not an LLM.
 * Build: gcc -O2 -o gen gen.c     Run: ./gen 1200 42 > raw.jsonl
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXP 8
#define MAXR 4
#define OUT_SZ 4096

static int rnd(int lo, int hi) { return lo + rand() % (hi - lo + 1); }

/* ---------- 1. CPU scheduling: FCFS, SJF (non-preemptive), RR ---------- */
typedef struct { int at, bt, ct, wt, tat; } Proc;

static void sched_fcfs(Proc *p, int n) {
    int t = 0;
    for (int i = 0; i < n; i++) {           /* p sorted by arrival */
        if (t < p[i].at) t = p[i].at;
        t += p[i].bt; p[i].ct = t;
        p[i].tat = p[i].ct - p[i].at; p[i].wt = p[i].tat - p[i].bt;
    }
}
static void sched_sjf(Proc *p, int n) {
    int done[MAXP] = {0}, t = 0, cnt = 0;
    while (cnt < n) {
        int best = -1;
        for (int i = 0; i < n; i++)
            if (!done[i] && p[i].at <= t && (best < 0 || p[i].bt < p[best].bt)) best = i;
        if (best < 0) { t++; continue; }
        t += p[best].bt; p[best].ct = t; done[best] = 1; cnt++;
        p[best].tat = p[best].ct - p[best].at; p[best].wt = p[best].tat - p[best].bt;
    }
}
static void sched_rr(Proc *p, int n, int q) {
    int rem[MAXP], t = 0, cnt = 0, queue[512], qh = 0, qt = 0, inq[MAXP] = {0};
    for (int i = 0; i < n; i++) rem[i] = p[i].bt;
    for (int i = 0; i < n; i++) if (p[i].at <= 0) { queue[qt++] = i; inq[i] = 1; }
    while (cnt < n) {
        if (qh == qt) {                      /* CPU idle: jump to next arrival */
            int nx = -1;
            for (int i = 0; i < n; i++) if (!inq[i] && (nx < 0 || p[i].at < p[nx].at)) nx = i;
            t = p[nx].at; queue[qt++] = nx; inq[nx] = 1; continue;
        }
        int i = queue[qh++];
        int run = rem[i] < q ? rem[i] : q;
        t += run; rem[i] -= run;
        for (int j = 0; j < n; j++)          /* admit arrivals before re-queue */
            if (!inq[j] && p[j].at <= t) { queue[qt++] = j; inq[j] = 1; }
        if (rem[i] > 0) queue[qt++] = i;
        else { p[i].ct = t; p[i].tat = t - p[i].at; p[i].wt = p[i].tat - p[i].bt; cnt++; }
    }
}
static void emit_sched(void) {
    int n = rnd(3, 5), kind = rnd(0, 2), q = rnd(2, 4);
    Proc p[MAXP]; char prob[1024], sol[2048], tmp[128];
    for (int i = 0; i < n; i++) { p[i].at = (i == 0) ? 0 : rnd(0, 6); p[i].bt = rnd(2, 9); }
    for (int i = 0; i < n; i++)              /* stable sort by arrival */
        for (int j = i + 1; j < n; j++)
            if (p[j].at < p[i].at) { Proc t = p[i]; p[i] = p[j]; p[j] = t; }
    char qs[48] = ""; if (kind == 2) snprintf(qs, sizeof qs, " with time quantum %d", q);
    const char *name[] = {"FCFS", "SJF (non-preemptive)", "Round Robin"};
    if (kind == 0) sched_fcfs(p, n); else if (kind == 1) sched_sjf(p, n); else sched_rr(p, n, q);
    prob[0] = 0;
    for (int i = 0; i < n; i++) {
        snprintf(tmp, sizeof tmp, "P%d(arrival=%d, burst=%d)%s", i + 1, p[i].at, p[i].bt, i < n - 1 ? ", " : "");
        strcat(prob, tmp);
    }
    double sw = 0, st = 0; sol[0] = 0;
    for (int i = 0; i < n; i++) {
        snprintf(tmp, sizeof tmp, "P%d: CT=%d, TAT=CT-AT=%d, WT=TAT-BT=%d. ", i + 1, p[i].ct, p[i].tat, p[i].wt);
        strcat(sol, tmp); sw += p[i].wt; st += p[i].tat;
    }
    printf("{\"topic\":\"scheduling\",\"instruction\":\"Compute average waiting and turnaround time using %s%s for: %s\","
           "\"input\":\"\",\"output\":\"Step 1: order by arrival and simulate the CPU timeline. Step 2: for each process "
           "compute completion time (CT). %sAverage WT = %.2f, Average TAT = %.2f.\"}\n",
           name[kind], kind == 2 ? qs : "", prob, sol, sw / n, st / n);
    (void)q;
}

/* ---------- 2. Banker's algorithm ---------- */
static void emit_banker(void) {
    int n = rnd(3, 5), m = 3, max[MAXP][MAXR], alloc[MAXP][MAXR], need[MAXP][MAXR], avail[MAXR];
    for (int j = 0; j < m; j++) avail[j] = rnd(1, 5);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            max[i][j] = rnd(1, 8); alloc[i][j] = rnd(0, max[i][j]);
            need[i][j] = max[i][j] - alloc[i][j];
        }
    int work[MAXR], fin[MAXP] = {0}, seq[MAXP], sc = 0, progress = 1;
    memcpy(work, avail, sizeof work);
    while (progress) {
        progress = 0;
        for (int i = 0; i < n; i++) {
            if (fin[i]) continue;
            int ok = 1;
            for (int j = 0; j < m; j++) if (need[i][j] > work[j]) { ok = 0; break; }
            if (ok) { for (int j = 0; j < m; j++) work[j] += alloc[i][j]; fin[i] = 1; seq[sc++] = i; progress = 1; }
        }
    }
    char desc[1024] = "", tmp[128], ans[512] = "";
    for (int i = 0; i < n; i++) {
        snprintf(tmp, sizeof tmp, "P%d: Alloc=(%d,%d,%d) Max=(%d,%d,%d); ", i, alloc[i][0], alloc[i][1], alloc[i][2], max[i][0], max[i][1], max[i][2]);
        strcat(desc, tmp);
    }
    if (sc == n) {
        strcpy(ans, "SAFE. Safe sequence: ");
        for (int i = 0; i < n; i++) { snprintf(tmp, sizeof tmp, "P%d%s", seq[i], i < n - 1 ? " -> " : ""); strcat(ans, tmp); }
    } else strcpy(ans, "UNSAFE: no process can be guaranteed to finish, so deadlock is possible.");
    printf("{\"topic\":\"banker\",\"instruction\":\"Apply Banker's Algorithm. Available=(%d,%d,%d). %s Is the system safe?\","
           "\"input\":\"\",\"output\":\"Compute Need = Max - Allocation, then repeatedly pick a process whose Need <= Work "
           "and release its allocation into Work. Result: %s\"}\n", avail[0], avail[1], avail[2], desc, ans);
}

/* ---------- 3. Page replacement: FIFO, LRU, OPT ---------- */
static int faults(int *ref, int len, int fr, int algo) {
    int frame[8], age[8], cnt = 0, f = 0, clk = 0, ptr = 0;
    for (int i = 0; i < fr; i++) frame[i] = -1;
    for (int i = 0; i < len; i++) {
        int hit = -1; clk++;
        for (int j = 0; j < fr; j++) if (frame[j] == ref[i]) hit = j;
        if (hit >= 0) { age[hit] = clk; continue; }
        f++;
        if (cnt < fr) { frame[cnt] = ref[i]; age[cnt++] = clk; continue; }
        int v = 0;
        if (algo == 0) { v = ptr; ptr = (ptr + 1) % fr; }
        else if (algo == 1) { for (int j = 1; j < fr; j++) if (age[j] < age[v]) v = j; }
        else {
            int far = -1;
            for (int j = 0; j < fr; j++) {
                int nx = len + 1;
                for (int k = i + 1; k < len; k++) if (ref[k] == frame[j]) { nx = k; break; }
                if (nx > far) { far = nx; v = j; }
            }
        }
        frame[v] = ref[i]; age[v] = clk;
    }
    return f;
}
static void emit_paging(void) {
    int len = rnd(10, 16), fr = rnd(3, 4), ref[32], algo = rnd(0, 2);
    const char *nm[] = {"FIFO", "LRU", "Optimal"};
    char s[256] = "", t[16];
    for (int i = 0; i < len; i++) { ref[i] = rnd(0, 6); snprintf(t, sizeof t, "%d%s", ref[i], i < len - 1 ? "," : ""); strcat(s, t); }
    int f = faults(ref, len, fr, algo);
    printf("{\"topic\":\"paging\",\"instruction\":\"Count page faults using %s replacement with %d frames for reference string %s.\","
           "\"input\":\"\",\"output\":\"Simulate frame contents after every reference; a fault occurs whenever the page is "
           "absent. Total page faults = %d (hit ratio = %.2f).\"}\n", nm[algo], fr, s, f, 1.0 - (double)f / len);
}

/* ---------- 4. Producer-consumer / semaphore conceptual MCQ ---------- */
static void emit_mcq(void) {
    int cap = rnd(2, 8), prod = rnd(1, 4);
    printf("{\"topic\":\"sync\",\"instruction\":\"MCQ: A bounded buffer of size %d is guarded by semaphores empty, full and mutex. "
           "Initial value of 'empty' is? (A) 0 (B) 1 (C) %d (D) %d\",\"input\":\"\",\"output\":\"(C) %d. 'empty' counts free "
           "slots so it starts at the buffer size; 'full' starts at 0 and mutex at 1. With %d producer(s) the wait(empty) "
           "call blocks only when the buffer is full.\"}\n", cap, cap, cap + 1, cap, prod);
}

int main(int argc, char **argv) {
    int total = argc > 1 ? atoi(argv[1]) : 1000, seed = argc > 2 ? atoi(argv[2]) : 1;
    srand(seed);
    for (int i = 0; i < total; i++) {
        switch (rnd(0, 9)) {
            case 0: case 1: case 2: case 3: emit_sched(); break;
            case 4: case 5: case 6: emit_banker(); break;
            case 7: case 8: emit_paging(); break;
            default: emit_mcq();
        }
    }
    return 0;
}
