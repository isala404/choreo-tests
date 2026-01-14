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
	"strings"
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
	ID           string     `json:"id"`
	Status       JobStatus  `json:"status"`
	InputFormat  string     `json:"input_format"`
	OutputFormat string     `json:"output_format"`
	Options      string     `json:"options,omitempty"`
	CreatedAt    time.Time  `json:"created_at"`
	CompletedAt  *time.Time `json:"completed_at,omitempty"`
	Error        string     `json:"error,omitempty"`
	InputFile    string     `json:"-"`
	OutputFile   string     `json:"-"`
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

type FormatsResponse struct {
	Video []string `json:"video"`
	Audio []string `json:"audio"`
	Image []string `json:"image"`
}

func main() {
	// Ensure temp directories exist
	os.MkdirAll("/tmp/ffmpeg/input", 0755)
	os.MkdirAll("/tmp/ffmpeg/output", 0755)

	http.HandleFunc("/", rootHandler)
	http.HandleFunc("/healthz", healthHandler)
	http.HandleFunc("/formats", formatsHandler)
	http.HandleFunc("/convert", convertHandler)
	http.HandleFunc("/status/", statusHandler)
	http.HandleFunc("/download/", downloadHandler)

	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	log.Printf("FFmpeg converter service starting on port %s", port)
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
		"message":  "FFmpeg Media Converter Service",
		"hostname": hostname,
		"version":  "1.0.0",
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	// Check if ffmpeg is available
	cmd := exec.Command("ffmpeg", "-version")
	if err := cmd.Run(); err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]string{
			"status": "unhealthy",
			"error":  "ffmpeg not available",
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "healthy",
	})
}

func formatsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.WriteHeader(http.StatusMethodNotAllowed)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "method_not_allowed",
			Message: "Only GET method is allowed",
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(FormatsResponse{
		Video: []string{"mp4", "avi", "mkv", "mov", "webm", "flv", "wmv", "mpeg", "3gp", "ogv"},
		Audio: []string{"mp3", "wav", "aac", "flac", "ogg", "wma", "m4a", "opus", "aiff"},
		Image: []string{"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"},
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

	// Parse multipart form (max 500MB for media files)
	if err := r.ParseMultipartForm(500 << 20); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "invalid_request",
			Message: "Failed to parse multipart form: " + err.Error(),
		})
		return
	}

	// Get output format (input format is auto-detected by ffmpeg)
	outputFormat := r.FormValue("output_format")
	if outputFormat == "" {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(ErrorResponse{
			Error:   "missing_parameters",
			Message: "output_format is required",
		})
		return
	}

	// Get optional ffmpeg options
	options := r.FormValue("options")

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

	// Detect input format from filename
	inputFormat := strings.TrimPrefix(filepath.Ext(header.Filename), ".")
	if inputFormat == "" {
		inputFormat = "unknown"
	}

	// Generate job ID
	jobID := uuid.New().String()

	// Save input file
	inputPath := filepath.Join("/tmp/ffmpeg/input", jobID+"_"+header.Filename)
	outputPath := filepath.Join("/tmp/ffmpeg/output", jobID+"."+outputFormat)

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
		Options:      options,
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

	// Build ffmpeg command
	args := []string{
		"-i", job.InputFile,
		"-y", // Overwrite output file if exists
	}

	// Add custom options if provided
	if job.Options != "" {
		// Split options by space (simple parsing)
		optionParts := strings.Fields(job.Options)
		args = append(args, optionParts...)
	}

	// Add output file
	args = append(args, job.OutputFile)

	// Run ffmpeg conversion
	cmd := exec.Command("ffmpeg", args...)

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

	// Set appropriate content type based on output format
	contentType := getContentType(job.OutputFormat)

	// Set content disposition header
	filename := fmt.Sprintf("converted.%s", job.OutputFormat)
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", filename))
	w.Header().Set("Content-Type", contentType)

	io.Copy(w, outputFile)
}

func getContentType(format string) string {
	contentTypes := map[string]string{
		// Video
		"mp4":  "video/mp4",
		"avi":  "video/x-msvideo",
		"mkv":  "video/x-matroska",
		"mov":  "video/quicktime",
		"webm": "video/webm",
		"flv":  "video/x-flv",
		"wmv":  "video/x-ms-wmv",
		"mpeg": "video/mpeg",
		"3gp":  "video/3gpp",
		"ogv":  "video/ogg",
		// Audio
		"mp3":  "audio/mpeg",
		"wav":  "audio/wav",
		"aac":  "audio/aac",
		"flac": "audio/flac",
		"ogg":  "audio/ogg",
		"wma":  "audio/x-ms-wma",
		"m4a":  "audio/mp4",
		"opus": "audio/opus",
		"aiff": "audio/aiff",
		// Image
		"png":  "image/png",
		"jpg":  "image/jpeg",
		"jpeg": "image/jpeg",
		"gif":  "image/gif",
		"bmp":  "image/bmp",
		"webp": "image/webp",
		"tiff": "image/tiff",
	}

	if ct, ok := contentTypes[format]; ok {
		return ct
	}
	return "application/octet-stream"
}
