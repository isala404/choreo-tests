package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
)

type JobStatus string

const (
	StatusAccepted   JobStatus = "accepted"
	StatusProcessing JobStatus = "processing"
	StatusCompleted  JobStatus = "completed"
	StatusFailed     JobStatus = "failed"
)

type Job struct {
	ID           string    `json:"id"`
	Status       JobStatus `json:"status"`
	InputFormat  string    `json:"input_format"`
	OutputFormat string    `json:"output_format"`
	CreatedAt    time.Time `json:"created_at"`
	CompletedAt  *time.Time `json:"completed_at,omitempty"`
	Error        string    `json:"error,omitempty"`
	InputFile    string    `json:"-"`
	OutputFile   string    `json:"-"`
}

type JobStore struct {
	mu   sync.RWMutex
	jobs map[string]*Job
}

func NewJobStore() *JobStore {
	return &JobStore{
		jobs: make(map[string]*Job),
	}
}

func (s *JobStore) Create(job *Job) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[job.ID] = job
}

func (s *JobStore) Get(id string) (*Job, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	job, ok := s.jobs[id]
	return job, ok
}

func (s *JobStore) Update(job *Job) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.jobs[job.ID] = job
}

var store = NewJobStore()

type ConvertRequest struct {
	InputFormat  string `json:"input_format"`
	OutputFormat string `json:"output_format"`
}

type ConvertResponse struct {
	JobID   string    `json:"job_id"`
	Status  JobStatus `json:"status"`
	Message string    `json:"message"`
}

type StatusResponse struct {
	JobID       string     `json:"job_id"`
	Status      JobStatus  `json:"status"`
	CreatedAt   time.Time  `json:"created_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
	Error       string     `json:"error,omitempty"`
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

func main() {
	// Ensure temp directories exist
	os.MkdirAll("/tmp/pandoc/input", 0755)
	os.MkdirAll("/tmp/pandoc/output", 0755)

	http.HandleFunc("/", rootHandler)
	http.HandleFunc("/healthz", healthHandler)
	http.HandleFunc("/convert", convertHandler)
	http.HandleFunc("/status/", statusHandler)
	http.HandleFunc("/download/", downloadHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	log.Printf("Pandoc converter service starting on port %s", port)
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
		"message":  "Pandoc Document Converter Service",
		"hostname": hostname,
		"version":  "1.0.0",
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	// Check if pandoc is available
	cmd := exec.Command("pandoc", "--version")
	if err := cmd.Run(); err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"status": "unhealthy",
			"error":  "pandoc not available",
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	})
}

func convertHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "method_not_allowed",
			Message: "Only POST method is allowed",
		})
		return
	}

	// Parse multipart form (max 50MB)
	if err := r.ParseMultipartForm(50 << 20); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "invalid_request",
			Message: "Failed to parse multipart form: " + err.Error(),
		})
		return
	}

	// Get input and output formats
	inputFormat := r.FormValue("input_format")
	outputFormat := r.FormValue("output_format")

	if inputFormat == "" || outputFormat == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "missing_parameters",
			Message: "Both input_format and output_format are required",
		})
		return
	}

	// Get the uploaded file
	file, header, err := r.FormFile("file")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "missing_file",
			Message: "File upload is required",
		})
		return
	}
	defer file.Close()

	// Generate job ID
	jobID := uuid.New().String()

	// Save input file
	inputPath := filepath.Join("/tmp/pandoc/input", jobID+"_"+header.Filename)
	outputPath := filepath.Join("/tmp/pandoc/output", jobID+"."+outputFormat)

	inputFile, err := os.Create(inputPath)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "file_save_error",
			Message: "Failed to save uploaded file",
		})
		return
	}

	if _, err := io.Copy(inputFile, file); err != nil {
		inputFile.Close()
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "file_save_error",
			Message: "Failed to save uploaded file",
		})
		return
	}
	inputFile.Close()

	// Create job
	job := &Job{
		ID:           jobID,
		Status:       StatusAccepted,
		InputFormat:  inputFormat,
		OutputFormat: outputFormat,
		CreatedAt:    time.Now(),
		InputFile:    inputPath,
		OutputFile:   outputPath,
	}
	store.Create(job)

	// Start background conversion
	go processConversion(job)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(ConvertResponse{
		JobID:   jobID,
		Status:  StatusAccepted,
		Message: "Conversion job created. Poll /status/" + jobID + " for progress.",
	})
}

func processConversion(job *Job) {
	// Update status to processing
	job.Status = StatusProcessing
	store.Update(job)

	log.Printf("Starting conversion job %s: %s -> %s", job.ID, job.InputFormat, job.OutputFormat)

	// Run pandoc conversion
	cmd := exec.Command("pandoc",
		"-f", job.InputFormat,
		"-t", job.OutputFormat,
		"-o", job.OutputFile,
		job.InputFile,
	)

	output, err := cmd.CombinedOutput()
	now := time.Now()
	job.CompletedAt = &now

	if err != nil {
		job.Status = StatusFailed
		job.Error = fmt.Sprintf("Conversion failed: %v - %s", err, string(output))
		log.Printf("Job %s failed: %s", job.ID, job.Error)
	} else {
		job.Status = StatusCompleted
		log.Printf("Job %s completed successfully", job.ID)
	}

	store.Update(job)

	// Clean up input file after processing
	os.Remove(job.InputFile)
}

func statusHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "method_not_allowed",
			Message: "Only GET method is allowed",
		})
		return
	}

	// Extract job ID from path
	jobID := r.URL.Path[len("/status/"):]
	if jobID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "missing_job_id",
			Message: "Job ID is required",
		})
		return
	}

	job, ok := store.Get(jobID)
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "job_not_found",
			Message: "Job with ID " + jobID + " not found",
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(StatusResponse{
		JobID:       job.ID,
		Status:      job.Status,
		CreatedAt:   job.CreatedAt,
		CompletedAt: job.CompletedAt,
		Error:       job.Error,
	})
}

func downloadHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "method_not_allowed",
			Message: "Only GET method is allowed",
		})
		return
	}

	// Extract job ID from path
	jobID := r.URL.Path[len("/download/"):]
	if jobID == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "missing_job_id",
			Message: "Job ID is required",
		})
		return
	}

	job, ok := store.Get(jobID)
	if !ok {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "job_not_found",
			Message: "Job with ID " + jobID + " not found",
		})
		return
	}

	if job.Status != StatusCompleted {
		w.Header().Set("Content-Type", "application/json")
		if job.Status == StatusFailed {
			w.WriteHeader(http.StatusUnprocessableEntity)
			json.NewEncoder(w).Encode(ErrorResponse{
				Error:   "conversion_failed",
				Message: job.Error,
			})
		} else {
			w.WriteHeader(http.StatusAccepted)
			json.NewEncoder(w).Encode(StatusResponse{
				JobID:     job.ID,
				Status:    job.Status,
				CreatedAt: job.CreatedAt,
			})
		}
		return
	}

	// Check if output file exists
	if _, err := os.Stat(job.OutputFile); os.IsNotExist(err) {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "file_not_found",
			Message: "Output file not found",
		})
		return
	}

	// Serve the file
	outputFile, err := os.Open(job.OutputFile)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "file_read_error",
			Message: "Failed to read output file",
		})
		return
	}
	defer outputFile.Close()

	// Set content disposition header
	filename := fmt.Sprintf("converted.%s", job.OutputFormat)
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	w.Header().Set("Content-Type", "application/octet-stream")

	io.Copy(w, outputFile)
}
