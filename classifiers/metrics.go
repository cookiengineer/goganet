package classifiers

import "sort"

// Confusion holds the binary confusion counts where positive means malicious.
type Confusion struct {
	TP, FP, TN, FN int
}

// Metrics summarizes binary classification quality.
type Metrics struct {
	Threshold        float32
	Accuracy         float32
	Precision        float32
	Recall           float32 // sensitivity, true positive rate
	Specificity      float32 // true negative rate (benign recall)
	F1               float32
	BalancedAccuracy float32
	AUC              float32
	N                int
	Positive         int
	Confusion        Confusion
}

// Evaluate computes binary metrics for probabilities at a fixed threshold.
func Evaluate(probs []float32, labels []int, threshold float32) Metrics {
	m := Metrics{Threshold: threshold, N: len(probs)}
	for i := range probs {
		pos := labels[i] == 1
		if pos {
			m.Positive++
		}
		pred := probs[i] >= threshold
		switch {
		case pos && pred:
			m.Confusion.TP++
		case !pos && pred:
			m.Confusion.FP++
		case !pos && !pred:
			m.Confusion.TN++
		default:
			m.Confusion.FN++
		}
	}
	c := m.Confusion
	m.Accuracy = ratio(c.TP+c.TN, m.N)
	m.Precision = ratio(c.TP, c.TP+c.FP)
	m.Recall = ratio(c.TP, c.TP+c.FN)
	m.Specificity = ratio(c.TN, c.TN+c.FP)
	m.BalancedAccuracy = (m.Recall + m.Specificity) / 2
	if m.Precision+m.Recall > 0 {
		m.F1 = 2 * m.Precision * m.Recall / (m.Precision + m.Recall)
	}
	m.AUC = AUC(probs, labels)
	return m
}

// BestThreshold returns the metrics at the threshold that maximises F1.
func BestThreshold(probs []float32, labels []int) Metrics {
	best := Evaluate(probs, labels, 0.5)
	for t := 0.05; t < 1.0; t += 0.05 {
		m := Evaluate(probs, labels, float32(t))
		if m.F1 > best.F1 {
			best = m
		}
	}
	return best
}

// AUC computes the area under the ROC curve using the rank (Mann-Whitney U)
// formulation, which is insensitive to the decision threshold.
func AUC(probs []float32, labels []int) float32 {
	n := len(probs)
	if n == 0 {
		return 0.5
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return probs[idx[a]] < probs[idx[b]] })

	ranks := make([]float64, n)
	for i := 0; i < n; {
		j := i
		for j+1 < n && probs[idx[j+1]] == probs[idx[i]] {
			j++
		}
		avg := float64(i+j)/2 + 1
		for k := i; k <= j; k++ {
			ranks[idx[k]] = avg
		}
		i = j + 1
	}

	var sumPos, nPos, nNeg float64
	for i := 0; i < n; i++ {
		if labels[i] == 1 {
			sumPos += ranks[i]
			nPos++
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return 0.5
	}
	return float32((sumPos - nPos*(nPos+1)/2) / (nPos * nNeg))
}

func ratio(num, den int) float32 {
	if den == 0 {
		return 0
	}
	return float32(num) / float32(den)
}
