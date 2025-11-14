// Insurance Recommendation Engine - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('recommendationForm');
    const submitBtn = document.getElementById('submitBtn');
    const loadingState = document.getElementById('loadingState');
    const resultsContainer = document.getElementById('resultsContainer');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Get form data
        const formData = {
            age: parseInt(document.getElementById('age').value),
            income: parseFloat(document.getElementById('income').value),
            sex: document.getElementById('sex').value,
            state: document.getElementById('state').value.toUpperCase(),
            job_class: document.getElementById('job_class').value
        };

        // Show loading state
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner inline-block mr-2"></span> Analyzing...';
        loadingState.classList.remove('hidden');
        resultsContainer.classList.add('hidden');

        try {
            // Call API
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                throw new Error('Failed to get recommendations');
            }

            const data = await response.json();

            // Display results
            displayResults(data);

            // Scroll to results
            resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            // Reset button
            submitBtn.disabled = false;
            submitBtn.textContent = 'Get Recommendations';
            loadingState.classList.add('hidden');
        }
    });
});

function displayResults(data) {
    const resultsContainer = document.getElementById('resultsContainer');
    const totalRecommended = document.getElementById('totalRecommended');
    const monthlyPremium = document.getElementById('monthlyPremium');
    const annualPremium = document.getElementById('annualPremium');
    const plansList = document.getElementById('plansList');

    // Update summary
    totalRecommended.textContent = data.summary.total_recommended;
    monthlyPremium.textContent = formatCurrency(data.summary.total_monthly_premium);
    annualPremium.textContent = formatCurrency(data.summary.total_annual_premium);

    // Clear previous plans
    plansList.innerHTML = '';

    // Display plans
    data.plans.forEach(plan => {
        const planCard = createPlanCard(plan);
        plansList.appendChild(planCard);
    });

    // Show results
    resultsContainer.classList.remove('hidden');
}

function createPlanCard(plan) {
    const card = document.createElement('div');

    // Determine colors based on recommendation
    let badgeColor, badgeText, badgeBg, badgeBorder, icon;

    if (plan.recommended) {
        if (plan.score >= 70) {
            badgeBg = '#e3faf3';
            badgeBorder = '#8deddb';
            badgeColor = '#045d42';
            badgeText = 'Highly Recommended';
            icon = '✓✓';
        } else if (plan.score >= 50) {
            badgeBg = '#e3faf3';
            badgeBorder = '#8deddb';
            badgeColor = '#045d42';
            badgeText = 'Recommended';
            icon = '✓';
        } else {
            badgeBg = '#eaf2ff';
            badgeBorder = '#5791f3';
            badgeColor = '#005dfe';
            badgeText = 'Consider';
            icon = '○';
        }
    } else {
        badgeBg = '#f8f9fa';
        badgeBorder = '#e9ecef';
        badgeColor = '#5d6b85';
        badgeText = 'Not Recommended';
        icon = '✗';
    }

    card.className = 'rounded-xl border p-4 transition-all duration-200 hover:shadow-md';
    card.style.borderColor = badgeBorder;
    card.style.backgroundColor = plan.recommended ? '#ffffff' : '#fafbfc';

    card.innerHTML = `
        <div class="flex items-start justify-between gap-4">
            <!-- Left: Icon and Plan Info -->
            <div class="flex-1 min-w-0">
                <div class="flex items-center gap-3 mb-2">
                    <span class="text-2xl flex-shrink-0">${icon}</span>
                    <div class="flex-1 min-w-0">
                        <h3 class="font-dm-sans font-semibold text-[16px] text-text-primary truncate">
                            ${plan.name}
                        </h3>
                        <p class="font-dm-sans text-[12px] text-text-tertiary">
                            ${plan.coverage_tier}
                        </p>
                    </div>
                </div>
            </div>

            <!-- Right: Score and Badge -->
            <div class="flex flex-col items-end gap-2 flex-shrink-0">
                <!-- Score -->
                <div class="text-right">
                    <div class="font-dm-sans font-bold text-[20px]" style="color: ${badgeColor}">
                        ${plan.score}%
                    </div>
                    <div class="font-dm-sans text-[10px] text-text-tertiary uppercase tracking-wide">
                        Confidence
                    </div>
                </div>

                <!-- Badge -->
                <span
                    class="px-3 py-1 rounded-[15px] text-[12px] font-medium whitespace-nowrap"
                    style="background-color: ${badgeBg}; border: 1px solid ${badgeBorder}; color: ${badgeColor}"
                >
                    ${badgeText}
                </span>

                <!-- Premium -->
                ${plan.premium ? `
                    <div class="font-dm-sans text-[14px] font-semibold text-text-secondary mt-1">
                        $${plan.premium.toFixed(2)}/mo
                    </div>
                ` : `
                    <div class="font-dm-sans text-[12px] text-text-tertiary mt-1">
                        Contact for pricing
                    </div>
                `}
            </div>
        </div>
    `;

    return card;
}

function formatCurrency(amount) {
    if (!amount || amount === 0) {
        return '$0.00';
    }
    return '$' + amount.toFixed(2);
}
