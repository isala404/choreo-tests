package main

import (
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/google/uuid"
)

type JobStatus string

const (
	StatusPending   JobStatus = "pending"
	StatusRunning   JobStatus = "running"
	StatusCompleted JobStatus = "completed"
	StatusFailed    JobStatus = "failed"
)

// ProjectionRequest represents the input parameters for retirement planning
type ProjectionRequest struct {
	CurrentAge        int     `json:"current_age"`
	RetirementAge     int     `json:"retirement_age"`
	LifeExpectancy    int     `json:"life_expectancy"`
	CurrentSavings    float64 `json:"current_savings"`
	AnnualContribution float64 `json:"annual_contribution"`
	AnnualExpenses    float64 `json:"annual_expenses"`
	ExpectedReturn    float64 `json:"expected_return"`    // e.g., 0.07 for 7%
	ReturnVolatility  float64 `json:"return_volatility"`  // e.g., 0.15 for 15%
	InflationRate     float64 `json:"inflation_rate"`     // e.g., 0.03 for 3%
	Simulations       int     `json:"simulations"`        // default 10000
}

// ProjectionResult contains the Monte Carlo simulation results
type ProjectionResult struct {
	JobID             string    `json:"job_id"`
	Status            JobStatus `json:"status"`
	SuccessRate       float64   `json:"success_rate,omitempty"`
	Percentiles       *Percentiles `json:"percentiles,omitempty"`
	YearlyProjections []YearProjection `json:"yearly_projections,omitempty"`
	SimulationsRun    int       `json:"simulations_run,omitempty"`
	ComputeTimeMs     float64   `json:"compute_time_ms,omitempty"`
	Error             string    `json:"error,omitempty"`
	CreatedAt         time.Time `json:"created_at"`
	CompletedAt       *time.Time `json:"completed_at,omitempty"`
}

type Percentiles struct {
	P10 float64 `json:"p10"` // 10th percentile (pessimistic)
	P25 float64 `json:"p25"`
	P50 float64 `json:"p50"` // Median
	P75 float64 `json:"p75"`
	P90 float64 `json:"p90"` // 90th percentile (optimistic)
}

type YearProjection struct {
	Age    int     `json:"age"`
	Year   int     `json:"year"`
	P10    float64 `json:"p10"`
	P50    float64 `json:"p50"`
	P90    float64 `json:"p90"`
	Phase  string  `json:"phase"` // "accumulation" or "withdrawal"
}

type Job struct {
	ID        string
	Request   ProjectionRequest
	Result    *ProjectionResult
	Status    JobStatus
	CreatedAt time.Time
}

type JobStore struct {
	jobs map[string]*Job
	mu   sync.RWMutex
}

var store = &JobStore{jobs: make(map[string]*Job)}

func main() {
	http.HandleFunc("/", rootHandler)
	http.HandleFunc("/healthz", healthHandler)
	http.HandleFunc("/projection", projectionHandler)
	http.HandleFunc("/projection/", projectionDetailHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	log.Printf("retirement-api starting on port %s", port)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	hostname, _ := os.Hostname()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"service":     "Retirement Projection API",
		"description": "Monte Carlo simulation for retirement planning",
		"hostname":    hostname,
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func projectionHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(map[string]string{"error": "POST only"})
		return
	}

	var req ProjectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}

	// Validate and set defaults
	if req.CurrentAge <= 0 || req.RetirementAge <= req.CurrentAge {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "invalid age parameters"})
		return
	}
	if req.LifeExpectancy <= req.RetirementAge {
		req.LifeExpectancy = req.RetirementAge + 30
	}
	if req.Simulations <= 0 {
		req.Simulations = 10000
	}
	if req.Simulations > 50000 {
		req.Simulations = 50000
	}
	if req.ExpectedReturn == 0 {
		req.ExpectedReturn = 0.07
	}
	if req.ReturnVolatility == 0 {
		req.ReturnVolatility = 0.15
	}
	if req.InflationRate == 0 {
		req.InflationRate = 0.03
	}

	jobID := uuid.New().String()
	job := &Job{
		ID:        jobID,
		Request:   req,
		Status:    StatusPending,
		CreatedAt: time.Now(),
		Result: &ProjectionResult{
			JobID:     jobID,
			Status:    StatusPending,
			CreatedAt: time.Now(),
		},
	}

	store.mu.Lock()
	store.jobs[jobID] = job
	store.mu.Unlock()

	// Run simulation in background
	go runSimulation(job)

	log.Printf("[job:%s] created projection: age %d->%d, $%.0f savings, %d simulations",
		jobID[:8], req.CurrentAge, req.RetirementAge, req.CurrentSavings, req.Simulations)

	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(map[string]string{
		"job_id":  jobID,
		"status":  string(StatusPending),
		"message": "Projection started. Poll /projection/" + jobID + " for results.",
	})
}

func projectionDetailHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	jobID := r.URL.Path[len("/projection/"):]
	if jobID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "job_id required"})
		return
	}

	store.mu.RLock()
	job, ok := store.jobs[jobID]
	store.mu.RUnlock()

	if !ok {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "job not found"})
		return
	}

	if r.Method == http.MethodDelete {
		store.mu.Lock()
		delete(store.jobs, jobID)
		store.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
		return
	}

	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	store.mu.RLock()
	result := job.Result
	store.mu.RUnlock()

	if result.Status == StatusPending || result.Status == StatusRunning {
		w.WriteHeader(http.StatusAccepted)
	}
	json.NewEncoder(w).Encode(result)
}

func runSimulation(job *Job) {
	store.mu.Lock()
	job.Status = StatusRunning
	job.Result.Status = StatusRunning
	store.mu.Unlock()

	start := time.Now()
	req := job.Request

	yearsToRetirement := req.RetirementAge - req.CurrentAge
	yearsInRetirement := req.LifeExpectancy - req.RetirementAge
	totalYears := yearsToRetirement + yearsInRetirement

	// Store final portfolio values for each simulation
	finalValues := make([]float64, req.Simulations)

	// Store yearly values for percentile calculations
	yearlyValues := make([][]float64, totalYears)
	for i := range yearlyValues {
		yearlyValues[i] = make([]float64, req.Simulations)
	}

	successCount := 0
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Run Monte Carlo simulations
	for sim := 0; sim < req.Simulations; sim++ {
		portfolio := req.CurrentSavings
		contribution := req.AnnualContribution
		expenses := req.AnnualExpenses
		failed := false

		for year := 0; year < totalYears; year++ {
			// Generate random return using normal distribution
			annualReturn := rng.NormFloat64()*req.ReturnVolatility + req.ExpectedReturn

			if year < yearsToRetirement {
				// Accumulation phase: contribute and grow
				portfolio = portfolio * (1 + annualReturn) + contribution
				contribution *= (1 + req.InflationRate) // Increase contribution with inflation
			} else {
				// Withdrawal phase: withdraw expenses and grow remainder
				portfolio = portfolio * (1 + annualReturn) - expenses
				expenses *= (1 + req.InflationRate) // Increase expenses with inflation

				if portfolio < 0 {
					portfolio = 0
					failed = true
				}
			}

			yearlyValues[year][sim] = portfolio
		}

		finalValues[sim] = portfolio
		if !failed && portfolio > 0 {
			successCount++
		}
	}

	// Calculate percentiles for final values
	sortedFinal := make([]float64, len(finalValues))
	copy(sortedFinal, finalValues)
	quickSort(sortedFinal)

	percentiles := &Percentiles{
		P10: sortedFinal[int(float64(len(sortedFinal))*0.10)],
		P25: sortedFinal[int(float64(len(sortedFinal))*0.25)],
		P50: sortedFinal[int(float64(len(sortedFinal))*0.50)],
		P75: sortedFinal[int(float64(len(sortedFinal))*0.75)],
		P90: sortedFinal[int(float64(len(sortedFinal))*0.90)],
	}

	// Calculate yearly projections (sample every few years to keep response small)
	yearlyProjections := make([]YearProjection, 0)
	currentYear := time.Now().Year()

	for year := 0; year < totalYears; year++ {
		// Include first year, last year, retirement year, and every 5 years
		if year == 0 || year == totalYears-1 || year == yearsToRetirement || year%5 == 0 {
			sorted := make([]float64, len(yearlyValues[year]))
			copy(sorted, yearlyValues[year])
			quickSort(sorted)

			phase := "accumulation"
			if year >= yearsToRetirement {
				phase = "withdrawal"
			}

			yearlyProjections = append(yearlyProjections, YearProjection{
				Age:   req.CurrentAge + year,
				Year:  currentYear + year,
				P10:   math.Round(sorted[int(float64(len(sorted))*0.10)]),
				P50:   math.Round(sorted[int(float64(len(sorted))*0.50)]),
				P90:   math.Round(sorted[int(float64(len(sorted))*0.90)]),
				Phase: phase,
			})
		}
	}

	elapsed := time.Since(start)
	now := time.Now()

	store.mu.Lock()
	job.Status = StatusCompleted
	job.Result = &ProjectionResult{
		JobID:             job.ID,
		Status:            StatusCompleted,
		SuccessRate:       float64(successCount) / float64(req.Simulations) * 100,
		Percentiles:       percentiles,
		YearlyProjections: yearlyProjections,
		SimulationsRun:    req.Simulations,
		ComputeTimeMs:     float64(elapsed.Microseconds()) / 1000.0,
		CreatedAt:         job.CreatedAt,
		CompletedAt:       &now,
	}
	store.mu.Unlock()

	log.Printf("[job:%s] completed: %.1f%% success rate, %d simulations in %.2fms",
		job.ID[:8], float64(successCount)/float64(req.Simulations)*100, req.Simulations, float64(elapsed.Microseconds())/1000.0)
}

// quickSort for percentile calculations
func quickSort(arr []float64) {
	if len(arr) < 2 {
		return
	}
	left, right := 0, len(arr)-1
	pivot := len(arr) / 2

	arr[pivot], arr[right] = arr[right], arr[pivot]

	for i := range arr {
		if arr[i] < arr[right] {
			arr[left], arr[i] = arr[i], arr[left]
			left++
		}
	}

	arr[left], arr[right] = arr[right], arr[left]

	quickSort(arr[:left])
	quickSort(arr[left+1:])
}
