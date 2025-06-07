#!/bin/bash

# Black Box Challenge - Go Implementation
# This script takes three parameters and outputs the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

cd top-coder-solution && go run main.go "$1" "$2" "$3"