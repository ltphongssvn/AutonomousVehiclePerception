#!/bin/bash
set -e
echo "🔍 Running per-file coverage check (minimum 70%)..."

FAILED=0

check_coverage() {
    local label="$1"
    shift
    local output
    output=$(PYTHONPATH=. python -m pytest "$@" --no-header -q 2>&1)
    
    # Parse coverage lines, check each file >= 70%
    echo "$output" | grep -E "^src/" | while IFS= read -r line; do
        file=$(echo "$line" | awk '{print $1}')
        cover=$(echo "$line" | awk '{print $4}' | tr -d '%')
        if [ -n "$cover" ] && [ "$cover" -lt 70 ]; then
            echo "❌ FAIL: $file at ${cover}% (need ≥70%)"
            exit 1
        fi
    done
    
    if [ $? -ne 0 ]; then
        FAILED=1
    fi
}

# Model + Data layer
check_coverage "Model+Data" tests/unit/model tests/unit/data --cov=src/model --cov=src/data --cov-report=term-missing

# FastAPI layer
check_coverage "FastAPI" tests/unit/fastapi --cov=src/fastapi_service --cov-report=term-missing

# Django layer
check_coverage "Django" tests/unit/django --cov=src/django_backend --cov-report=term-missing

if [ $FAILED -ne 0 ]; then
    echo ""
    echo "🚫 Push blocked: one or more files below 70% coverage."
    echo "   Fix coverage gaps before pushing."
    exit 1
fi

echo "✅ All files meet 70% minimum coverage threshold"
