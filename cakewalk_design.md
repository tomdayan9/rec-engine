# Cakewalk Benefits Platform Design System Guidelines

This document outlines design principles and implementation guidelines for the Cakewalk Benefits Platform using shadcn/ui with Tailwind CSS. These guidelines ensure consistency, accessibility, and best practices throughout the UI development process.

## 🚨 MANDATORY COMPLIANCE

**All new elements, components, screens, and flows MUST follow these guidelines. No exceptions.**

Violations will be caught by automated linting rules and code review requirements.

## Core Design Principles

### 1. Typography System: Structured Font Hierarchy

#### Font Families
- **Space Grotesk**: H1 headings only (per memory ID: 4365734)
- **DM Sans**: All other text (H2-H6, body, labels, buttons)

#### Font Sizes & Usage
```scss
// H1 Page Titles - ONLY use Space Grotesk
font-size: 32px (text-[32px])
font-family: Space Grotesk
font-weight: 700 (bold)
color: #0a214a

// H2 Section Headers - ALWAYS use DM Sans
font-size: 20px (text-[20px]) 
font-family: DM Sans
font-weight: 600 (semibold)
color: #005dfe

// H3 Subsection Headers - ALWAYS use DM Sans
font-size: 16px (text-[16px])
font-family: DM Sans  
font-weight: 600 (semibold)
color: #424b5b

// Body Text - ALWAYS use DM Sans
font-size: 14px (text-[14px])
font-family: DM Sans
font-weight: 400-500 (normal/medium)
color: #424b5b

// Small Text/Labels - ALWAYS use DM Sans
font-size: 12px-13px
font-family: DM Sans
font-weight: 400 (normal)
color: #5d6b85
```

### 2. Color System (60/30/10 Rule)

#### Primary Brand Colors (10% - Use Sparingly)
```scss
--primary-blue: #005dfe        // Buttons, CTAs, links
--primary-blue-hover: #0051dc  // Hover states
--primary-blue-light: #5791f3  // Subtle accents
```

#### Text Colors (30% - Main Content)
```scss
--text-primary: #0a214a        // Headings, important text
--text-secondary: #424b5b      // Body text, labels
--text-tertiary: #5d6b85       // Supporting text, placeholders
```

#### Background Colors (60% - Main Surfaces)
```scss
--bg-primary: #ffffff          // Main backgrounds
--bg-secondary: #f8f9fa        // Subtle backgrounds  
--bg-accent: #f4f8ff          // Highlighted areas
--bg-blue-light: #eaf2ff      // Light blue backgrounds
```

#### Status Colors
```scss
--success: #15cb94             // Success states, confirmations
--success-light: #e3faf3       // Success backgrounds
--success-border: #8deddb      // Success borders

--warning: #bf6432             // Warnings, attention
--warning-light: #faede3       // Warning backgrounds  
--warning-border: #edc38d      // Warning borders

--error: #d44848               // Errors, destructive actions
--error-light: #ffeaea         // Error backgrounds
--error-border: #ffa1a1        // Error borders
```

### 3. Spacing System (8pt Grid) - MANDATORY

**All spacing MUST be divisible by 8 or 4. No exceptions.**

```scss
// Padding Values (ONLY these allowed)
p-2  = 8px     // Small elements
p-3  = 12px    // Form fields
p-4  = 16px    // Standard cards
p-6  = 24px    // Large cards, modals
p-8  = 32px    // Section spacing

// Margin Values (ONLY these allowed)
m-2  = 8px     // Small gaps
m-3  = 12px    // Standard gaps
m-4  = 16px    // Form spacing
m-6  = 24px    // Section gaps
m-8  = 32px    // Large section gaps

// Gap Values (ONLY these allowed)
gap-2 = 8px    // Tight spacing
gap-3 = 12px   // Form fields
gap-4 = 16px   // Standard spacing
gap-6 = 24px   // Generous spacing
gap-8 = 32px   // Section spacing
```

### 4. Border Radius System

```scss
rounded-lg   = 8px     // Standard cards, inputs, buttons
rounded-xl   = 12px    // Person cards, small components  
rounded-2xl  = 16px    // Policy cards, major components
rounded-3xl  = 24px    // Hero sections, large components

// Badge/Pill radius
rounded-[15px]         // Status badges, pills
```

### 5. Animation & Motion System

```scss
// Standard Transition Durations
--duration-fast: 200ms      // Hover states, small changes
--duration-normal: 300ms    // Standard transitions
--duration-slow: 500ms      // Modal animations, large changes
--duration-celebration: 1s  // Success animations

// Easing Functions
--ease-out-cubic: cubic-bezier(0.23, 1, 0.32, 1)
--ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55)
--ease-smooth: cubic-bezier(0.25, 0.46, 0.45, 0.94)

// Standard Transitions (Use these classes)
.transition-default: transition: all 300ms cubic-bezier(0.23, 1, 0.32, 1)
.hover-lift: transform: translateY(-2px) on hover
.hover-scale: transform: scale(1.02) on hover
```

## Component Architecture

### Button System (Per Memory ID: 4889980)

**ALL buttons MUST use design system variants:**

```jsx
// Primary Action Buttons (REQUIRED: variant="primary" size="lg")
<Button variant="primary" size="lg" className="w-full">
  Primary Action
</Button>

// Secondary Buttons
<Button variant="secondary" size="md">
  Secondary Action  
</Button>

// Tertiary Buttons (REQUIRED: with border)
<Button variant="tertiary" size="md" className="border">
  Tertiary Action
</Button>

// Button Sizes (ONLY these allowed)
size="sm"      // height: 32px, padding: 8px 12px
size="md"      // height: 40px, padding: 8px 16px  
size="lg"      // height: 48px, padding: 12px 24px (PREFERRED)
size="default" // height: 44px, padding: 10px 20px
```

### Card System

#### Standard Card (shadcn/ui)
```jsx
<Card className="rounded-lg border bg-card shadow-sm p-4">
  <CardHeader className="p-6">
    <CardTitle>Card Title</CardTitle>
  </CardHeader>
  <CardContent className="p-6 pt-0">
    Content
  </CardContent>
</Card>
```

#### Policy Cards (16px border radius)
```jsx
<div className="rounded-2xl border p-4 sm:p-6 shadow-sm">
  {/* Icon container */}
  <div className="bg-[#cbdeff] rounded-[10px] p-2">
    <Icon className="w-6 h-6" fill="#005dfe" />
  </div>
  {/* Content */}
</div>
```

### Modal System

**ALL modals MUST follow this structure:**

```jsx
<CakewalkModal open={isOpen} onOpenChange={handleClose}>
  <CakewalkModalContent className="max-w-[540px]">
    {/* REQUIRED: DM Sans for titles */}
    <CakewalkModalTitle className="font-dm-sans text-xl font-semibold mb-4">
      Modal Title
    </CakewalkModalTitle>
    
    {/* REQUIRED: Step indicators for multi-step flows */}
    <ModalStepIndicator steps={steps} currentStep={currentStep} />
    
    {/* REQUIRED: 16px step headers */}
    <h3 className="font-dm-sans font-semibold text-[16px] text-[#424b5b]">
      Step Header
    </h3>
  </CakewalkModalContent>
</CakewalkModal>
```

### Badge/Status System

```jsx
// Success Badge
<span className="bg-[#c5f8e9] border border-[#8deddb] text-[#045d42] px-2 py-1 rounded-[15px] text-sm font-medium">
  Active
</span>

// Plan Detail Pills (REQUIRED: 16px font)
<span className="bg-[#f8f9fa] border border-[#e9ecef] text-[#5d6b85] text-[16px] font-medium px-2 py-1 rounded">
  Basic LTD
</span>
```

## MANDATORY Tracking Attributes (Per Memory ID: 4374947)

**ALL interactive elements MUST include tracking attributes:**

```jsx
// Action Tracking Pattern
data-track="action-{section}-{element}-{action}-{type}"

// Examples:
<Button data-track="action-benefits-selection-confirm-button-click">
<Modal data-track="content-add-dependent-modal-display">
<Input data-track="form-personal-details-first-name-input-blur">
<Card data-tracking="benefits-wallet-view-policy-details">

// Form Tracking (REQUIRED for all inputs)
data-track="form-{section}-{field}-{event}"
data-track-focus="form-{section}-{field}-focus"
data-track-blur="form-{section}-{field}-blur"
```

## Layout & Responsive Guidelines

### Page Structure (MANDATORY)
```jsx
// Page Title (ALWAYS H1 with Space Grotesk)
<h1 className="font-space-grotesk font-bold text-[32px] leading-tight text-[#0a214a]">
  Page Title
</h1>

// Section Headers (ALWAYS H2 with DM Sans)
<h2 className="font-dm-sans font-semibold text-[20px] leading-tight text-[#005dfe]">
  Section Header
</h2>

// Subsection Headers (ALWAYS H3 with DM Sans)
<h3 className="font-dm-sans font-semibold text-[16px] leading-tight text-[#424b5b]">
  Subsection Header
</h3>
```

### Container Widths
```scss
// Page Container
max-width: 1200px
margin: 0 auto
padding: 0 16px (mobile) / 32px (desktop)

// Modal Container  
max-width: 540px

// Card Container
max-width: 816px
```

### Responsive Breakpoints (Mobile First)
```scss
sm:  640px   // Small tablets
md:  768px   // Tablets  
lg:  1024px  // Small desktops
xl:  1280px  // Large desktops
2xl: 1536px  // Extra large
```

## Accessibility Requirements (MANDATORY)

### Color Contrast
- **Normal text**: Minimum 4.5:1 ratio
- **Large text**: Minimum 3:1 ratio
- **Interactive elements**: Clear focus states

### Focus Management
```jsx
// REQUIRED focus ring for all interactive elements
focus-visible:ring-2 focus-visible:ring-[#005dfe] focus-visible:ring-offset-2
```

### Semantic HTML (REQUIRED)
- Proper heading hierarchy (H1 → H2 → H3)
- Form labels properly associated
- ARIA labels for interactive elements

## Error Handling & Loading States

### Form Validation
```jsx
// Error State (REQUIRED styling)
<Input 
  className={cn(
    "border-gray-300",
    errors.field && "border-red-500 focus:ring-red-500"
  )}
/>

// Error Message (REQUIRED for all form errors)
{errors.field && (
  <div className="text-sm text-red-600 mt-1">
    {errors.field}
  </div>
)}
```

### Loading States
```jsx
// Loading Button (REQUIRED for all async actions)
<Button loading={isSubmitting}>
  {isSubmitting ? <Spinner className="mr-2" /> : null}
  Submit
</Button>
```

## Icon System (Lucide React Only)

```jsx
// Icon Sizing Standards (ONLY these sizes)
icon-xs: 12px   // w-3 h-3 - Small inline icons
icon-sm: 16px   // w-4 h-4 - Form field icons
icon-md: 20px   // w-5 h-5 - Button icons
icon-lg: 24px   // w-6 h-6 - Card/product icons (PREFERRED)
icon-xl: 32px   // w-8 h-8 - Header icons

// Icon Colors (ONLY these colors)
text-[#005dfe]  // Primary actions
text-[#5d6b85]  // Secondary elements
text-[#15cb94]  // Success states
text-[#bf6432]  // Warning states
text-[#d44848]  // Error states
```

## Z-Index Scale (ONLY these values)

```scss
--z-dropdown: 1000
--z-sticky: 1020
--z-fixed: 1030
--z-modal-backdrop: 1040
--z-modal: 1050
--z-popover: 1060
--z-tooltip: 1070
--z-toast: 1080
```

## ENFORCEMENT RULES

### Pre-commit Checks (REQUIRED)
1. **Typography Validation**: All H1 must use Space Grotesk, H2+ must use DM Sans
2. **Spacing Validation**: All spacing values must be divisible by 8 or 4
3. **Color Validation**: Only design system colors allowed
4. **Tracking Validation**: All interactive elements must have tracking attributes
5. **Accessibility Validation**: Focus states and ARIA labels required

### Code Review Checklist (MANDATORY)

#### Typography ✅
- [ ] H1 uses Space Grotesk, 32px, bold, #0a214a
- [ ] H2 uses DM Sans, 20px, semibold, #005dfe
- [ ] H3 uses DM Sans, 16px, semibold, #424b5b
- [ ] Modal titles use DM Sans font
- [ ] Policy details use 16px font size
- [ ] No arbitrary font sizes outside the system

#### Colors ✅
- [ ] Primary blue (#005dfe) used only for CTAs (<10% of interface)
- [ ] Dark blue (#0a214a) used for headings
- [ ] Secondary gray (#424b5b) for body text
- [ ] Proper contrast ratios maintained (4.5:1 minimum)
- [ ] Status colors used consistently

#### Spacing & Layout ✅
- [ ] All spacing values divisible by 8 or 4
- [ ] No arbitrary margin/padding values
- [ ] Proper grid alignment maintained
- [ ] Responsive breakpoints used correctly

#### Components ✅
- [ ] Buttons use design system variants (primary/lg preferred)
- [ ] Cards use proper border radius (8px, 12px, 16px)
- [ ] Modals follow structure guidelines
- [ ] Icons from Lucide React only, proper sizing

#### Tracking & Accessibility ✅
- [ ] All interactive elements have tracking attributes
- [ ] Proper heading hierarchy maintained
- [ ] Focus states clearly visible
- [ ] Form validation follows patterns
- [ ] ARIA labels present where needed

#### Animation & Performance ✅
- [ ] Transitions use standard durations and easing
- [ ] Loading states implemented for async actions
- [ ] Error boundaries in place
- [ ] No memory leaks in useEffect

## VIOLATION CONSEQUENCES

❌ **Code that violates these guidelines will be:**
1. **Rejected in code review**
2. **Flagged by automated linting**
3. **Blocked from deployment**

✅ **All new code must pass:**
1. **Design system linting rules**
2. **Accessibility validation**
3. **Code review approval**
4. **Automated testing**

## EXCEPTIONS

**NO EXCEPTIONS ALLOWED** without explicit approval from:
1. Design system maintainer
2. Technical lead
3. Product owner

Any exceptions must be documented and include:
1. **Justification**: Why the exception is needed
2. **Impact**: What guidelines are being violated
3. **Mitigation**: How to minimize negative effects
4. **Timeline**: When the exception will be resolved

---

**Remember: Consistency is key to a professional, accessible, and maintainable application. These guidelines ensure every user has the best possible experience.**