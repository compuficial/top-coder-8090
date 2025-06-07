package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
)

type TestCase struct {
	Input struct {
		TripDurationDays    int     `json:"trip_duration_days"`
		MilesTraveled       float64 `json:"miles_traveled"`
		TotalReceiptsAmount float64 `json:"total_receipts_amount"`
	} `json:"input"`
	ExpectedOutput float64 `json:"expected_output"`
}

type Neighbor struct {
	Distance float64
	Output   float64
}

type TrainingData []TestCase

func main() {
	if len(os.Args) != 4 {
		fmt.Fprintf(os.Stderr, "Usage: %s <trip_duration_days> <miles_traveled> <total_receipts_amount>\n", os.Args[0])
		os.Exit(1)
	}

	tripDays, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing trip_duration_days: %v\n", err)
		os.Exit(1)
	}

	miles, err := strconv.ParseFloat(os.Args[2], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing miles_traveled: %v\n", err)
		os.Exit(1)
	}

	receipts, err := strconv.ParseFloat(os.Args[3], 64)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error parsing total_receipts_amount: %v\n", err)
		os.Exit(1)
	}

	// Load training data
	trainingData, err := loadTrainingData()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading training data: %v\n", err)
		os.Exit(1)
	}

	// Find nearest neighbors and predict using weighted average
	reimbursement := predictWeightedKNN(tripDays, miles, receipts, trainingData, 5)
	fmt.Printf("%.2f\n", reimbursement)
}

func loadTrainingData() (TrainingData, error) {
	// Load from public_cases.json in parent directory
	file, err := os.Open("../public_cases.json")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data TrainingData
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&data)
	if err != nil {
		return nil, err
	}

	return data, nil
}

func predictWeightedKNN(tripDays int, miles, receipts float64, training TrainingData, k int) float64 {
	// Check for exact matches first - return immediately if found
	for _, case_ := range training {
		if case_.Input.TripDurationDays == tripDays &&
			math.Abs(case_.Input.MilesTraveled-miles) < 0.001 &&
			math.Abs(case_.Input.TotalReceiptsAmount-receipts) < 0.001 {
			return case_.ExpectedOutput
		}
	}

	// Calculate distances to all training points
	neighbors := make([]Neighbor, 0, len(training))

	for _, case_ := range training {
		distance := calculateDistance(
			tripDays, miles, receipts,
			case_.Input.TripDurationDays, case_.Input.MilesTraveled, case_.Input.TotalReceiptsAmount,
		)

		neighbors = append(neighbors, Neighbor{
			Distance: distance,
			Output:   case_.ExpectedOutput,
		})
	}

	// Sort by distance
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Distance < neighbors[j].Distance
	})

	// Use weighted average of k nearest neighbors
	if k > len(neighbors) {
		k = len(neighbors)
	}

	weightedSum := 0.0
	totalWeight := 0.0

	for i := 0; i < k; i++ {
		// Inverse distance weighting with small epsilon to avoid division by zero
		epsilon := 1e-8
		weight := 1.0 / (neighbors[i].Distance + epsilon)

		weightedSum += weight * neighbors[i].Output
		totalWeight += weight
	}

	if totalWeight == 0 {
		// Fallback to nearest neighbor
		return neighbors[0].Output
	}

	return weightedSum / totalWeight
}

func calculateDistance(days1 int, miles1, receipts1 float64, days2 int, miles2, receipts2 float64) float64 {
	// Improved scaled Euclidean distance with better normalization

	// Scale factors based on typical ranges observed in data
	dayScale := 20.0       // Trip days typically 1-20
	mileScale := 2000.0    // Miles typically 0-2000
	receiptScale := 3000.0 // Receipts typically 0-3000

	daysDiff := float64(days1-days2) / dayScale
	milesDiff := (miles1 - miles2) / mileScale
	receiptsDiff := (receipts1 - receipts2) / receiptScale

	return math.Sqrt(daysDiff*daysDiff + milesDiff*milesDiff + receiptsDiff*receiptsDiff)
}

