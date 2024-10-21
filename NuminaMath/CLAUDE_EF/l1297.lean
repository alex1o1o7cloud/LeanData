import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_astroid_circle_intersection_radius_l1297_129761

-- Define the curve equation
def astroid_curve (x y : ℝ) : Prop := x^(2/3) + y^(2/3) = 1

-- Define the circle equation
def origin_circle (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Define the tangent condition
def is_tangent (x y : ℝ) : Prop := 
  ∃ (m : ℝ), ∀ (x' y' : ℝ), astroid_curve x' y' → (y' - y = m * (x' - x)) → (x' = x ∧ y' = y)

-- Define the square formation condition
def forms_squares (points : List (ℝ × ℝ)) : Prop :=
  points.length = 8 ∧ 
  ∃ (s1 s2 : List (ℝ × ℝ)), s1.length = 4 ∧ s2.length = 4 ∧
  (∀ (p : ℝ × ℝ), p ∈ points → p ∈ s1 ∨ p ∈ s2) ∧
  (∀ (p1 p2 : ℝ × ℝ), p1 ∈ s1 ∧ p2 ∈ s1 → ‖p1 - p2‖ = ‖p1 - p2‖) ∧
  (∀ (p1 p2 : ℝ × ℝ), p1 ∈ s2 ∧ p2 ∈ s2 → ‖p1 - p2‖ = ‖p1 - p2‖)

-- Main theorem
theorem astroid_circle_intersection_radius :
  ∃ (r : ℝ), r = Real.sqrt (2/5) ∧
  ∃ (points : List (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ points → astroid_curve p.1 p.2 ∧ origin_circle p.1 p.2 r) ∧
    forms_squares points ∧
    (∀ (p1 p2 : ℝ × ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 → 
      is_tangent ((p1.1 + p2.1)/2) ((p1.2 + p2.2)/2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_astroid_circle_intersection_radius_l1297_129761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_two_l1297_129718

-- Define the vectors a and b
variable (a b : ℝ × ℝ)

-- Define the angle between a and b
noncomputable def angle : ℝ := Real.pi / 6

-- Define the magnitudes of a and b
def mag_a : ℝ := 2
noncomputable def mag_b : ℝ := Real.sqrt 3

-- State the theorem
theorem dot_product_equals_two :
  (a.1 * (2 * b.1 - a.1) + a.2 * (2 * b.2 - a.2) = 2) ∧
  (a.1^2 + a.2^2 = mag_a^2) ∧
  (b.1^2 + b.2^2 = mag_b^2) ∧
  (a.1 * b.1 + a.2 * b.2 = mag_a * mag_b * Real.cos angle) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_equals_two_l1297_129718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equivalence_same_terminal_side_l1297_129768

noncomputable def α : ℝ := 1690 * (Real.pi / 180)

theorem angle_equivalence : 
  ∃ (k : ℤ) (β : ℝ), 
    α = 2 * k * Real.pi + β ∧ 
    k = 4 ∧ 
    β = 25 * Real.pi / 18 ∧ 
    0 ≤ β ∧ 
    β < 2 * Real.pi :=
by sorry

theorem same_terminal_side (θ : ℝ) 
  (h1 : ∃ (k : ℤ), θ = 2 * k * Real.pi + 25 * Real.pi / 18) 
  (h2 : -4 * Real.pi < θ ∧ θ < -2 * Real.pi) :
  θ = -47 * Real.pi / 18 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_equivalence_same_terminal_side_l1297_129768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1297_129763

/-- Helper function to represent points on an ellipse -/
def ellipse (f1 f2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (a : ℝ), a > 0 ∧ 
    Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) + 
    Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2) = 2 * a}

/-- An ellipse in the first quadrant tangent to both axes with foci at (3,4) and (d,4) has d = 55/6 -/
theorem ellipse_foci_distance (d : ℝ) : 
  let f1 : ℝ × ℝ := (3, 4)
  let f2 : ℝ × ℝ := (d, 4)
  let center : ℝ × ℝ := ((d + 3) / 2, 4)
  let tangent_point : ℝ × ℝ := ((d + 3) / 2, 0)
  -- Ellipse is in first quadrant
  3 < d ∧ 
  -- Ellipse is tangent to both axes
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x, 0) ∈ ellipse f1 f2 ∧ (0, y) ∈ ellipse f1 f2) ∧
  -- Definition of an ellipse (constant sum of distances from foci)
  (∀ (p : ℝ × ℝ), p ∈ ellipse f1 f2 ↔ 
    Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) + 
    Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2) = d + 3) →
  d = 55 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l1297_129763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_of_cosine_graph_l1297_129749

noncomputable def f (x : ℝ) := 2 * Real.cos (2 * x)

noncomputable def g (φ : ℝ) (x : ℝ) := 2 * Real.cos (2 * x - 2 * φ)

theorem translation_of_cosine_graph (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) :
  (∃ x₁ x₂ : ℝ, |f x₁ - g φ x₂| = 4 ∧ 
    (∀ y₁ y₂ : ℝ, |f y₁ - g φ y₂| = 4 → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
    |x₁ - x₂| = Real.pi / 6) →
  φ = Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_of_cosine_graph_l1297_129749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1297_129715

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * Real.cos t.C * (t.a * Real.cos t.B + t.b * Real.cos t.A) = t.c ∧
  t.c = Real.sqrt 7 ∧
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2

-- State the theorem
theorem triangle_perimeter (t : Triangle) 
  (h : triangle_conditions t) : 
  t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l1297_129715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_18_l1297_129773

def sequence_b : ℕ → ℚ
  | 0 => 2  -- We define b₁ as the 0th term for simplicity
  | 1 => 3  -- b₂ becomes the 1st term
  | n+2 => (1/2) * sequence_b (n+1) + (1/3) * sequence_b n

theorem sum_of_sequence_is_18 :
  ∑' n, sequence_b n = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_sequence_is_18_l1297_129773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_with_2009_distinct_l1297_129756

/-- Product of decimal digits of a natural number -/
def digit_product (n : ℕ) : ℕ := sorry

/-- Sequence defined by the recurrence relation -/
def seq (a k : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => seq a k n + k * digit_product (seq a k n)

/-- Number of distinct elements in the first n terms of the sequence -/
def distinct_count (a k n : ℕ) : ℕ := sorry

theorem exists_sequence_with_2009_distinct :
  ∃ a k : ℕ, a > 0 ∧ k > 0 ∧ distinct_count a k 2009 = 2009 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_with_2009_distinct_l1297_129756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_l1297_129711

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem triangle_circumcircle (t : Triangle) :
  ∃! (c : Circle), 
    distance c.center t.A = c.radius ∧
    distance c.center t.B = c.radius ∧
    distance c.center t.C = c.radius := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_l1297_129711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1297_129741

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := abs x * (x^2 - 3*t)

-- Define the function g
def g (t : ℝ) (x : ℝ) : ℝ := abs (f t x)

-- Define the function F
noncomputable def F (t : ℝ) : ℝ :=
  if t ≤ 1 then 8 - 6*t
  else if t < 4 then 2*t*Real.sqrt t
  else 6*t - 8

-- State the theorem
theorem max_value_of_g (t : ℝ) :
  ∃ (x : ℝ), x ∈ Set.Icc 0 2 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 2 → g t y ≤ g t x ∧ g t x = F t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1297_129741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_remaining_flight_time_l1297_129794

/-- Calculates the remaining flight time of a plane given its speed, headwind, fuel consumption rate, and remaining fuel. -/
noncomputable def remaining_flight_time (speed : ℝ) (headwind : ℝ) (fuel_consumption : ℝ) (remaining_fuel : ℝ) : ℝ :=
  (remaining_fuel / fuel_consumption) * 60

/-- Theorem stating that for a plane with given parameters, the remaining flight time is 40 minutes. -/
theorem plane_remaining_flight_time :
  let speed := (450 : ℝ)
  let headwind := (30 : ℝ)
  let fuel_consumption := (9.5 : ℝ)
  let remaining_fuel := (6.3333 : ℝ)
  remaining_flight_time speed headwind fuel_consumption remaining_fuel = 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_remaining_flight_time_l1297_129794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l1297_129757

theorem integer_solutions_count : ∃ (s : Finset ℤ), s.card = 4 ∧ ∀ x : ℤ, x ∈ s ↔ x - 1 < (x - 1)^2 ∧ (x - 1)^2 < 3*x + 7 := by
  let inequality (x : ℤ) : Prop := x - 1 < (x - 1)^2 ∧ (x - 1)^2 < 3*x + 7
  let solution_set : Set ℤ := {x | inequality x}
  -- We assume the existence of a finite set of solutions
  have h : ∃ (s : Finset ℤ), ∀ x : ℤ, x ∈ s ↔ x ∈ solution_set := sorry
  rcases h with ⟨s, hs⟩
  use s
  constructor
  · -- Prove that the cardinality of s is 4
    sorry
  · -- Prove that s contains exactly the solutions to the inequality
    exact hs


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_count_l1297_129757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1297_129777

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x^3 - 3*a*x + 2 else 2^(x+1) - a

-- State the theorem
theorem f_has_three_zeros (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧
    (∀ w : ℝ, f a w = 0 → w = x ∨ w = y ∨ w = z)) ↔
  1 < a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_three_zeros_l1297_129777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1297_129708

theorem sin_cos_difference (θ : Real) (h1 : Real.sin θ + Real.cos θ = 4/3) (h2 : 0 < θ) (h3 : θ < Real.pi/4) :
  Real.sin θ - Real.cos θ = -Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l1297_129708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_sufficient_not_necessary_for_g_increasing_l1297_129729

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a : ℝ) (x : ℝ) : ℝ := (2 - a) * x^3

-- Define what it means for a function to be decreasing or increasing on ℝ
def is_decreasing (h : ℝ → ℝ) : Prop := ∀ x y, x < y → h x > h y
def is_increasing (h : ℝ → ℝ) : Prop := ∀ x y, x < y → h x < h y

theorem f_decreasing_sufficient_not_necessary_for_g_increasing
  (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (is_decreasing (f a) → is_increasing (g a)) ∧
  ¬(is_increasing (g a) → is_decreasing (f a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_sufficient_not_necessary_for_g_increasing_l1297_129729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_2_l1297_129714

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^5 + b * Real.sin x + x^2

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : f a b (-2) = 3 → f a b 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_2_l1297_129714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1297_129780

noncomputable section

/-- The line equation y = 2x - 3 --/
def line_equation (x : ℝ) : ℝ := 2 * x - 3

/-- The point we're finding the closest point to --/
def given_point : ℝ × ℝ := (2, 5)

/-- The point on the line claimed to be closest to the given point --/
def closest_point : ℝ × ℝ := (18/5, 3)

/-- Theorem stating that the closest_point is on the line and is the closest to the given_point --/
theorem closest_point_on_line :
  (line_equation closest_point.1 = closest_point.2) ∧
  (∀ p : ℝ × ℝ, line_equation p.1 = p.2 →
    (closest_point.1 - given_point.1)^2 + (closest_point.2 - given_point.2)^2 ≤
    (p.1 - given_point.1)^2 + (p.2 - given_point.2)^2) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l1297_129780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_transformations_l1297_129782

-- Define the probability mass function and cumulative distribution function
noncomputable def P_ξ (ξ : ℝ → ℝ) (x : ℝ) : ℝ := sorry
noncomputable def F_ξ (ξ : ℝ → ℝ) (x : ℝ) : ℝ := sorry

theorem probability_transformations 
  (ξ : ℝ → ℝ) (a b x y : ℝ) 
  (h1 : a > 0) (h2 : y ≥ 0) :
  -- 1. PMF transformation
  P_ξ (λ t ↦ a * ξ t + b) x = P_ξ ξ ((x - b) / a) ∧
  -- 2. CDF transformation
  F_ξ (λ t ↦ a * ξ t + b) x = F_ξ ξ ((x - b) / a) ∧
  -- 3. CDF of ξ²
  F_ξ (λ t ↦ (ξ t)^2) y = F_ξ ξ (Real.sqrt y) - F_ξ ξ (-Real.sqrt y) + P_ξ ξ (-Real.sqrt y) ∧
  -- 4. CDF of ξ⁺
  F_ξ (λ t ↦ max (ξ t) 0) x = 
    if x < 0 then 0 else F_ξ ξ x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_transformations_l1297_129782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consequence_seeking_cause_method_l1297_129742

theorem consequence_seeking_cause_method (a b c : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a) ↔ ((a - b) * (a - c) > 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consequence_seeking_cause_method_l1297_129742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l1297_129755

noncomputable def work_rate (days : ℝ) : ℝ := 1 / days

theorem work_completion 
  (a_days : ℝ) 
  (b_days : ℝ) 
  (work_together_days : ℝ) 
  (work_completed : ℝ) :
  work_rate a_days + work_rate b_days = work_completed / work_together_days →
  b_days = 20 →
  work_together_days = 5 →
  work_completed = 7/12 →
  a_days = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_l1297_129755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_bounds_l1297_129797

noncomputable def number_set (x : ℝ) : Finset ℝ := {x, x+1, 4, 3, 6}

def is_median (s : Finset ℝ) (m : ℝ) : Prop :=
  2 * (s.filter (λ y => y ≤ m)).card ≥ s.card ∧
  2 * (s.filter (λ y => y ≥ m)).card ≥ s.card

theorem median_bounds (x : ℝ) :
  (∃ m, is_median (number_set x) m) ∧
  (∀ m, is_median (number_set x) m → 3 ≤ m ∧ m ≤ 6) ∧
  (∃ x₁ x₂, is_median (number_set x₁) 3 ∧ is_median (number_set x₂) 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_bounds_l1297_129797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_for_given_interest_difference_l1297_129791

/-- Given an interest rate and time period, calculates the difference between
    compound interest (compounded annually) and simple interest for a principal amount. -/
noncomputable def interestDifference (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1) - principal * rate * time / 100

theorem principal_for_given_interest_difference :
  ∀ P : ℝ, P > 0 →
    interestDifference P 20 2 = 360 →
    P = 9000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_for_given_interest_difference_l1297_129791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1297_129733

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) : ℝ := 
  (Real.arcsin (x/3))^2 - Real.pi * Real.arccos (x/3) + (Real.arccos (x/3))^2 + (Real.pi^2/8) * (x^2 - 4*x + 3)

-- State the theorem
theorem g_range :
  ∀ y ∈ Set.range g,
  (π^2/4 ≤ y ∧ y ≤ 33*π^2/8) ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = π^2/4 ∧
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, g x = 33*π^2/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l1297_129733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1297_129778

-- Define the loan amount
noncomputable def loan_amount : ℝ := 1200

-- Define the interest paid
noncomputable def interest_paid : ℝ := 192

-- Define the simple interest formula
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  principal * rate * time / 100

-- Theorem statement
theorem interest_rate_is_four_percent :
  ∃ (rate : ℝ), 
    simple_interest loan_amount rate rate = interest_paid ∧ 
    rate = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l1297_129778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_super_flippy_divisible_by_12_l1297_129717

def is_super_flippy (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ ({2, 4, 6, 8} : Set ℕ) ∧ b ∈ ({2, 4, 6, 8} : Set ℕ) ∧
  (n = a * 10000 + b * 1000 + a * 100 + b * 10 + a)

def is_five_digit (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

theorem unique_super_flippy_divisible_by_12 :
  ∃! n : ℕ, is_five_digit n ∧ is_super_flippy n ∧ n % 12 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_super_flippy_divisible_by_12_l1297_129717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_intersection_theorem_l1297_129752

-- Define a square with side length 1
def Unit_Square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define a convex polygon
def Convex_Polygon (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ S

-- Define the area of a set in ℝ²
noncomputable def Area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a line parallel to a side of the square
def Parallel_Line (y : ℝ) : Set (ℝ × ℝ) := {p | p.2 = y}

-- Define the intersection of a set with a line
def Intersection (S : Set (ℝ × ℝ)) (L : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := S ∩ L

-- Define the length of a segment
noncomputable def Segment_Length (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem polygon_intersection_theorem (M : Set (ℝ × ℝ)) 
  (h_convex : Convex_Polygon M) 
  (h_in_square : M ⊆ Unit_Square) 
  (h_area : Area M > 1/2) :
  ∃ y : ℝ, 0 ≤ y ∧ y ≤ 1 ∧ 
    Segment_Length (Intersection M (Parallel_Line y)) > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_intersection_theorem_l1297_129752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_before_two_digit_number_answer_is_correct_l1297_129774

def twoDigitNumber (t u : ℕ) : ℕ := 10 * t + u

def placeThreeBefore (t u : ℕ) : ℕ := 300 + 10 * t + u

theorem three_before_two_digit_number (t u : ℕ) :
  placeThreeBefore t u = 300 + 10 * t + u := by
  unfold placeThreeBefore
  rfl

#check three_before_two_digit_number

theorem answer_is_correct (t u : ℕ) :
  placeThreeBefore t u = 300 + 10 * t + u := by
  exact three_before_two_digit_number t u

#check answer_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_before_two_digit_number_answer_is_correct_l1297_129774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_sum_exists_l1297_129703

theorem power_of_two_sum_exists (M : Finset Nat) : 
  M ⊆ Finset.range 1999 → M.card = 1000 → 
  ∃ a b k, a ∈ M ∧ b ∈ M ∧ a + b = 2^k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_sum_exists_l1297_129703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_value_l1297_129784

-- Define the piecewise function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x^2 + b else 10 - 4 * x

-- State the theorem
theorem a_plus_b_value (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) →
  a + b = 221/64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_value_l1297_129784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_coloring_methods_l1297_129720

/-- Number of distinct coloring methods for a 2 × n grid with pre-colored endpoints -/
def coloringMethods (n : ℕ) : ℕ := 3^(n-2)

/-- Represents the actual number of distinct colorings for a 2 × n grid with the given constraints -/
def number_of_distinct_colorings_for_grid (n : ℕ) : ℕ :=
  sorry -- This function is left unimplemented for now

/-- Proof that coloringMethods gives the correct number of distinct coloring methods -/
theorem correct_coloring_methods (n : ℕ) (h : n ≥ 2) :
  coloringMethods n = number_of_distinct_colorings_for_grid n :=
by
  sorry -- The proof is skipped for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_coloring_methods_l1297_129720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_tangents_l1297_129753

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculates the distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Determines the number of common tangents between two circles -/
noncomputable def commonTangents (c1 c2 : Circle) : ℕ :=
  if distance c1.center c2.center == c1.radius + c2.radius then 3 else 0

/-- The main theorem stating that there are 3 common tangents between the given circles -/
theorem three_common_tangents :
  let o1 : Circle := { center := (2, -3), radius := 2 }
  let o2 : Circle := { center := (-1, 1), radius := 3 }
  commonTangents o1 o2 = 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_common_tangents_l1297_129753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_f_composite_l1297_129744

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x^6 else -2*x - 1

theorem coefficient_x_squared_in_f_composite (x : ℝ) (h : x ≤ -1) :
  ∃ (a b c d e : ℝ), f (f x) = 60*x^2 + a*x^5 + b*x^4 + c*x^3 + d*x + e :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_in_f_composite_l1297_129744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_two_power_plus_one_two_power_minus_one_l1297_129765

theorem gcd_two_power_plus_one_two_power_minus_one (m n : ℕ) (h : Odd n) :
  Nat.gcd (2^m + 1) (2^n - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_two_power_plus_one_two_power_minus_one_l1297_129765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_parity_l1297_129709

noncomputable def f (x : ℝ) : ℝ := Real.log (3 + x) + Real.log (3 - x)

theorem f_domain_and_parity :
  (∀ x : ℝ, f x ∈ Set.Ioo (-3) 3 ↔ x ∈ Set.Ioo (-3) 3) ∧
  (∀ x : ℝ, x ∈ Set.Ioo (-3) 3 → f (-x) = f x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_parity_l1297_129709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_4_l1297_129798

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp (x * Real.log 2) + Real.exp ((2 - x) * Real.log 2)

-- Theorem statement
theorem f_min_value_is_4 :
  ∃ (x₀ : ℝ), f x₀ = 4 ∧ ∀ (x : ℝ), f x ≥ 4 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_is_4_l1297_129798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_planes_l1297_129706

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A set of four points in 3D space -/
def FourPoints := Fin 4 → Point3D

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point3D) : Prop := sorry

/-- Predicate to check if four points form a plane -/
def formPlane (p q r s : Point3D) : Prop := sorry

/-- The number of planes formed by any three points from a set of four points -/
def numPlanes (points : FourPoints) : ℕ := sorry

/-- Theorem stating that given four non-collinear points, 
    the number of planes formed is either 4 or 1 -/
theorem four_points_planes (points : FourPoints) 
  (h : ∀ (i j k : Fin 4), i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  numPlanes points = 4 ∨ numPlanes points = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_points_planes_l1297_129706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_ten_l1297_129712

/-- The area of a triangle given its vertices -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Theorem stating the conditions for the area of triangle ABC to be 10 -/
theorem triangle_area_is_ten (k : ℝ) :
  triangleArea 7 7 3 2 0 k = 10 ↔ k = 9 ∨ k = -13/3 := by
  sorry

#check triangle_area_is_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_ten_l1297_129712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_solutions_of_equation_l1297_129767

theorem integral_solutions_of_equation :
  {(m, n) : ℤ × ℤ | (m^2 - n^2)^2 = 1 + 16*n} =
  {(1, 0), (-1, 0), (4, 3), (-4, 3), (4, 5), (-4, 5)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_solutions_of_equation_l1297_129767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1297_129710

-- Define the complex number z
variable (z : ℂ)

-- Define the equation |z - 2| = 3|z + 2|
def equation (z : ℂ) : Prop := Complex.abs (z - 2) = 3 * Complex.abs (z + 2)

-- Define the circle |z| = k
def circle_equation (k : ℝ) (z : ℂ) : Prop := Complex.abs z = k

-- Theorem statement
theorem unique_intersection :
  ∃! k : ℝ, ∃! z : ℂ, equation z ∧ circle_equation k z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_l1297_129710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_min_value_min_value_achieved_l1297_129779

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := x + a^2 * y + 1 = 0
def l₂ (a b : ℝ) (x y : ℝ) : Prop := (a^2 + 1) * x - b * y + 3 = 0

-- Theorem for the first part of the problem
theorem parallel_lines (a : ℝ) :
  (∃ b, b = -12 ∧ (∀ x y, l₁ a x y ↔ l₂ a b x y)) →
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by sorry

-- Theorem for the second part of the problem
theorem perpendicular_lines_min_value (a b : ℝ) :
  (a ≠ 0) →
  (∀ x y, (l₁ a x y ∧ l₂ a b x y) → (a^2 + 1) * (-a^2) = -1) →
  |a * b| ≥ 2 :=
by sorry

-- Theorem for the minimum value
theorem min_value_achieved (a : ℝ) :
  ∃ b, (a ≠ 0) ∧
       (∀ x y, (l₁ a x y ∧ l₂ a b x y) → (a^2 + 1) * (-a^2) = -1) ∧
       |a * b| = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_perpendicular_lines_min_value_min_value_achieved_l1297_129779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_equals_2017_l1297_129790

def f : ℕ → ℤ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1
  | 2 => 1
  | (n+3) => f (n+2) - f (n+1) + (n+3)

theorem f_2018_equals_2017 : f 2018 = 2017 := by
  sorry  -- We'll keep the sorry tactic for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2018_equals_2017_l1297_129790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l1297_129723

-- Define the angle α
noncomputable def α : ℝ := 44 * Real.pi / 180

-- Define the point P
noncomputable def P : ℝ × ℝ := (Real.sin (-50 * Real.pi / 180), Real.cos (10 * Real.pi / 180))

-- State the theorem
theorem angle_value :
  (0 < α) ∧ (α < Real.pi / 2) ∧  -- 0° < α < 90°
  (∃ (r : ℝ), r > 0 ∧ (r * Real.cos α = P.1) ∧ (r * Real.sin α = P.2)) →  -- P is on the terminal side of α
  α = 44 * Real.pi / 180 :=  -- α = 44°
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l1297_129723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_8_625_l1297_129721

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by slope and a point -/
structure Line where
  slope : ℝ
  point : Point2D

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (a b c : Point2D) : ℝ :=
  (1/2) * abs (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y))

theorem triangle_area_equals_8_625 
  (line1 line2 : Line)
  (line3_eq : ℝ → ℝ → Prop) :
  line1.point = ⟨3, 3⟩ →
  line2.point = ⟨3, 3⟩ →
  line1.slope = 1/3 →
  line2.slope = 3 →
  line3_eq = (fun x y ↦ x + y = 12) →
  ∃ (p1 p2 p3 : Point2D), 
    p1 ∈ {p : Point2D | p.y = line1.slope * (p.x - 3) + 3} ∧
    p1 ∈ {p : Point2D | line3_eq p.x p.y} ∧
    p2 ∈ {p : Point2D | p.y = line2.slope * (p.x - 3) + 3} ∧
    p2 ∈ {p : Point2D | line3_eq p.x p.y} ∧
    p3 = ⟨3, 3⟩ ∧
    triangleArea p1 p2 p3 = 8.625 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_equals_8_625_l1297_129721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocals_l1297_129789

open Real

theorem tan_sum_reciprocals (x y : ℝ) 
  (h1 : (sin x) / (cos y) + (sin y) / (cos x) = 2)
  (h2 : (cos x) / (sin y) + (cos y) / (sin x) = 5) :
  (tan x) / (tan y) + (tan y) / (tan x) = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocals_l1297_129789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_texas_migration_rate_l1297_129770

/-- The average number of people moving to Texas per hour, given the total number of people and days -/
def average_people_per_hour (total_people : ℕ) (days : ℕ) : ℕ :=
  (total_people + (days * 24) / 2) / (days * 24)

/-- Theorem stating that given 3000 people moving to Texas over 5 days, 
    the average number of people moving per hour, rounded to the nearest whole number, is 25 -/
theorem texas_migration_rate : average_people_per_hour 3000 5 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_texas_migration_rate_l1297_129770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_max_area_m_l1297_129740

-- Define the line l: x = my + 2
def line_l (m : ℝ) (y : ℝ) : ℝ := m * y + 2

-- Define the ellipse C: x²/a² + y² = 1
def ellipse_C (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

-- State that the line intersects x-axis at one of the foci
axiom line_intersects_focus (m : ℝ) : line_l m 0 = 2

-- State that a > 0
axiom a_positive : ∃ a : ℝ, a > 0

-- Theorem 1: The equation of ellipse C
theorem ellipse_equation : 
  ∃ a : ℝ, a > 0 ∧ ∀ x y : ℝ, ellipse_C a x y ↔ x^2 / 5 + y^2 = 1 :=
sorry

-- Define the area of triangle ABF1
noncomputable def triangle_area (m : ℝ) : ℝ := 
  4 * Real.sqrt 5 * Real.sqrt ((m^2 + 1) / ((m^2 + 5)^2))

-- Theorem 2: The value of m that maximizes the area of triangle ABF1
theorem max_area_m : 
  ∃ m : ℝ, ∀ k : ℝ, triangle_area m ≥ triangle_area k ∧ m^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_max_area_m_l1297_129740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_problem_l1297_129727

theorem sine_cosine_problem (α β : Real) (h_acute_α : 0 < α ∧ α < Real.pi/2) (h_acute_β : 0 < β ∧ β < Real.pi/2)
  (h_sin_α : Real.sin α = 2*Real.sqrt 5/5) (h_cos_π_β : Real.cos (Real.pi - β) = -Real.sqrt 2/10) :
  Real.sin (2*α) = 4/5 ∧ Real.cos (α - β) = 3*Real.sqrt 10/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_problem_l1297_129727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_circle_sums_l1297_129786

/-- Represents the arrangement of numbers on the circles -/
structure CircleArrangement where
  numbers : Fin 6 → Nat
  placed_at_intersections : ∀ n, numbers n ∈ Finset.range 7 \ {0}
  all_numbers_used : ∀ m ∈ Finset.range 7 \ {0}, ∃ n, numbers n = m
  six_is_placed : ∃ n, numbers n = 6

/-- The sum of numbers on each circle -/
def circle_sum (arr : CircleArrangement) (circle : Fin 3) : Nat :=
  sorry

theorem equal_circle_sums (arr : CircleArrangement) :
  (∀ c₁ c₂ : Fin 3, circle_sum arr c₁ = circle_sum arr c₂) →
  (∃ n, arr.numbers n = 1 ∧ ∃ m, arr.numbers m = 6 ∧ n ≠ m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_circle_sums_l1297_129786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l1297_129799

theorem trig_expression_simplification (x : ℝ) : 
  (Real.sin x + Real.sin (3 * x)) / (1 + Real.cos x + Real.cos (3 * x)) = 
  (4 * (Real.sin x - Real.sin x ^ 3)) / (1 - 2 * Real.cos x + 4 * Real.cos x ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_simplification_l1297_129799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elodie_rats_l1297_129704

/-- The number of rats Elodie has -/
def E : ℕ := sorry

/-- The number of rats Hunter has -/
def H : ℕ := sorry

/-- The number of rats Kenia has -/
def K : ℕ := sorry

/-- Elodie has 10 more rats than Hunter -/
axiom elodie_hunter_relation : E = H + 10

/-- Kenia has three times as many rats as Hunter and Elodie have together -/
axiom kenia_relation : K = 3 * (E + H)

/-- The total number of pets is 200 -/
axiom total_pets : E + H + K = 200

theorem elodie_rats : E = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elodie_rats_l1297_129704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_l1297_129731

/-- Represents the time it takes for all three leaks to empty a full cistern -/
noncomputable def emptyTime (x y z : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z)

/-- Theorem stating the conditions and the result to be proved -/
theorem cistern_empty_time (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_fill : 1/8 - (1/x + 1/y + 1/z) = 1/10) :
  emptyTime x y z = 40 := by
  sorry

#check cistern_empty_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_empty_time_l1297_129731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_omega_in_range_l1297_129716

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x)

theorem two_zeros_iff_omega_in_range (ω : ℝ) :
  (ω > 0) →
  (∃! (z₁ z₂ : ℝ), 0 ≤ z₁ ∧ z₁ < z₂ ∧ z₂ ≤ π/2 ∧ f ω z₁ = 0 ∧ f ω z₂ = 0 ∧
    ∀ (z : ℝ), 0 ≤ z ∧ z ≤ π/2 ∧ f ω z = 0 → z = z₁ ∨ z = z₂) ↔
  2 ≤ ω ∧ ω < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_iff_omega_in_range_l1297_129716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_minus_sectors_area_l1297_129754

/-- The area of the region inside a regular hexagon but outside circular sectors centered at its vertices -/
theorem hexagon_minus_sectors_area :
  let side_length : ℝ := 10
  let sector_radius : ℝ := 4
  let sector_angle : ℝ := 72
  let hexagon_area : ℝ := 6 * (Real.sqrt 3 / 4 * side_length ^ 2)
  let sector_area : ℝ := 6 * (sector_angle / 360 * Real.pi * sector_radius ^ 2)
  hexagon_area - sector_area = 150 * Real.sqrt 3 - 19.2 * Real.pi :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_minus_sectors_area_l1297_129754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1297_129707

def is_valid (n : ℕ) : Prop :=
  n % 2 ≠ 0 ∧ n % 3 ≠ 0 ∧ ∀ a b : ℕ, (2^a : ℤ) - (3^b : ℤ) ≠ n ∧ (3^b : ℤ) - (2^a : ℤ) ≠ n

theorem smallest_valid_n : 
  is_valid 35 ∧ ∀ m : ℕ, m < 35 → ¬(is_valid m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_n_l1297_129707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_fraction_equality_l1297_129795

theorem decimal_fraction_equality (b : ℕ) (hb : b > 0) : 
  (5 * b + 21 : ℚ) / (7 * b + 13 : ℚ) = 82 / 100 → b = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_fraction_equality_l1297_129795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_GOKU_l1297_129751

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is scalene -/
def is_scalene (t : Triangle) : Prop :=
  let AB := ((t.B.x - t.A.x)^2 + (t.B.y - t.A.y)^2).sqrt
  let AC := ((t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2).sqrt
  let BC := ((t.C.x - t.B.x)^2 + (t.C.y - t.B.y)^2).sqrt
  AB < AC ∧ AC < BC

/-- Represents the side mediator of a line segment -/
def side_mediator (A : Point) (B : Point) (P : Point) : Prop :=
  (P.x - A.x)^2 + (P.y - A.y)^2 = (P.x - B.x)^2 + (P.y - B.y)^2

/-- Checks if a point is on a line segment -/
def on_segment (A : Point) (B : Point) (P : Point) : Prop :=
  (P.x - A.x) * (B.x - A.x) + (P.y - A.y) * (B.y - A.y) ≥ 0 ∧
  (P.x - B.x) * (A.x - B.x) + (P.y - B.y) * (A.y - B.y) ≥ 0

/-- Checks if a point is on a line -/
def on_line (A : Point) (B : Point) (P : Point) : Prop :=
  (P.y - A.y) * (B.x - A.x) = (P.x - A.x) * (B.y - A.y)

/-- Checks if four points form a cyclic quadrilateral -/
noncomputable def is_cyclic (A : Point) (B : Point) (C : Point) (D : Point) : Prop :=
  let AB := ((B.x - A.x)^2 + (B.y - A.y)^2).sqrt
  let BC := ((C.x - B.x)^2 + (C.y - B.y)^2).sqrt
  let CD := ((D.x - C.x)^2 + (D.y - C.y)^2).sqrt
  let DA := ((A.x - D.x)^2 + (A.y - D.y)^2).sqrt
  let AC := ((C.x - A.x)^2 + (C.y - A.y)^2).sqrt
  let BD := ((D.x - B.x)^2 + (D.y - B.y)^2).sqrt
  AB * CD + BC * DA = AC * BD

theorem cyclic_quadrilateral_GOKU 
  (t : Triangle) 
  (K : Point) 
  (U : Point) 
  (O : Point) 
  (G : Point) 
  (h1 : is_scalene t)
  (h2 : side_mediator t.A t.B K)
  (h3 : on_segment t.B t.C K)
  (h4 : side_mediator t.A t.B U)
  (h5 : on_line t.A t.C U)
  (h6 : on_segment t.B t.C O)
  (h7 : on_line t.A t.C O)
  (h8 : on_line t.A t.B G)
  (h9 : on_line t.A t.C G) :
  is_cyclic G O K U := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_quadrilateral_GOKU_l1297_129751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l1297_129781

/-- Given a line with parametric equations x = 2 + 3t and y = 1 - t, 
    its slope is -1/3 -/
theorem line_slope : 
  ∀ t : ℝ, (1 - t - 1) / ((2 + 3 * t) - 2) = -1/3 := by
  intro t
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_l1297_129781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l1297_129732

/-- If the terminal side of angle α passes through point P(-5, 12), then cos α = -5/13 -/
theorem cos_alpha_for_point (α : ℝ) : 
  (∃ (r : ℝ), r * Real.cos α = -5 ∧ r * Real.sin α = 12) → Real.cos α = -5/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l1297_129732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_through_point_l1297_129735

/-- A proportional function passing through (m, 9) with negative slope has m = -3 -/
theorem proportional_function_through_point (m : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, f x = m * x) →  -- proportional function exists
  (∀ f : ℝ → ℝ, (∀ x, f x = m * x) → f m = 9) →  -- passes through (m, 9)
  (∀ f : ℝ → ℝ, (∀ x, f x = m * x) → ∀ x y, x < y → f x > f y) →  -- negative slope
  m = -3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_through_point_l1297_129735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_sqrt_6_l1297_129722

/-- The distance from a point to a line passing through the origin -/
noncomputable def distancePointToLine (point : ℝ × ℝ × ℝ) (lineDirection : ℝ × ℝ × ℝ) : ℝ :=
  let crossProduct := (
    (point.2.1 * lineDirection.2.2 - point.2.2 * lineDirection.2.1),
    (point.2.2 * lineDirection.1 - point.1 * lineDirection.2.2),
    (point.1 * lineDirection.2.1 - point.2.1 * lineDirection.1)
  )
  let crossProductNorm := Real.sqrt (crossProduct.1^2 + crossProduct.2.1^2 + crossProduct.2.2^2)
  let lineDirectionNorm := Real.sqrt (lineDirection.1^2 + lineDirection.2.1^2 + lineDirection.2.2^2)
  crossProductNorm / lineDirectionNorm

theorem distance_point_to_line_sqrt_6 :
  distancePointToLine (1, -1, 2) (0, 2, 1) = Real.sqrt 6 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_sqrt_6_l1297_129722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l1297_129734

theorem alpha_plus_beta_value (α β : ℝ) : 
  Real.sin (2 * α) = (Real.sqrt 5) / 5 →
  Real.sin (β - α) = (Real.sqrt 10) / 10 →
  α ∈ Set.Icc (π / 4) π →
  β ∈ Set.Icc π (3 * π / 2) →
  α + β = 7 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_plus_beta_value_l1297_129734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_adjacent_birch_is_6_143_l1297_129738

def total_trees : ℕ := 15
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := 9

def probability_no_adjacent_birch : ℚ :=
  (Nat.choose (non_birch_trees + 1) birch_trees) / (Nat.choose total_trees birch_trees)

theorem probability_no_adjacent_birch_is_6_143 :
  probability_no_adjacent_birch = 6 / 143 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_no_adjacent_birch_is_6_143_l1297_129738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_l1297_129724

/-- Given a line with equation 2x - y - 3 = 0, 
    the tangent of the angle it makes with the positive x-axis is 2. -/
theorem line_tangent (x y : ℝ) :
  (2 * x - y - 3 = 0) → Real.tan (Real.arctan ((y - 3) / (2 * x))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_l1297_129724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_g_properties_l1297_129758

noncomputable def f (x φ : ℝ) : ℝ := Real.cos (2 * x + φ)

noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + Real.pi/3)

theorem f_symmetry_and_g_properties 
  (h1 : ∀ x, f x (-Real.pi/6) = f (13*Real.pi/6 - x) (-Real.pi/6)) 
  (h2 : -Real.pi/2 < -Real.pi/6 ∧ -Real.pi/6 < Real.pi/2) :
  (∀ x, g (f · (-Real.pi/6)) x = -(g (f · (-Real.pi/6)) (-x))) ∧ 
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi/4 → g (f · (-Real.pi/6)) y < g (f · (-Real.pi/6)) x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_and_g_properties_l1297_129758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1297_129700

theorem divisibility_condition (n : ℕ) :
  (∃ k : ℕ, n = 1 + 6 * k ∨ n = 5 + 6 * k) ↔ 309 ∣ (20^n - 13^n - 7^n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l1297_129700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_of_inclination_l1297_129769

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2

-- Define the point of interest
noncomputable def point : ℝ × ℝ := (1, -3/2)

-- Define the slope at the point
noncomputable def tangent_slope : ℝ := 1

-- Theorem statement
theorem tangent_angle_of_inclination :
  let x := point.1
  let y := point.2
  f x = y ∧ 
  (deriv f) x = tangent_slope →
  Real.arctan tangent_slope = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_of_inclination_l1297_129769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_not_always_cylinder_l1297_129766

-- Define geometric shapes
structure Octahedron where

structure Tetrahedron where

structure Frustum where

structure Rectangle where

structure Cylinder where

-- Define properties
def has_faces (shape : Type) (n : ℕ) : Prop :=
  sorry

def can_be_cut_into_pyramids (shape : Type) (n : ℕ) : Prop :=
  sorry

def lateral_edges_intersect_at_point (shape : Type) : Prop :=
  sorry

def rotated_around_side_forms (shape1 : Type) (shape2 : Type) : Prop :=
  sorry

-- Axioms based on the given conditions
axiom octahedron_faces : has_faces Octahedron 10

axiom tetrahedron_division : can_be_cut_into_pyramids Tetrahedron 4

axiom frustum_edges : lateral_edges_intersect_at_point Frustum

-- Theorem to prove
theorem rectangle_rotation_not_always_cylinder : 
  ¬ (rotated_around_side_forms Rectangle Cylinder) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_rotation_not_always_cylinder_l1297_129766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l1297_129739

noncomputable section

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the circle Ω
def circle_Ω (x y D : ℝ) : Prop := x^2 + y^2 - D*x - 2 = 0

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem ellipse_and_circle_properties 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 6 / 3) 
  (D : ℝ) 
  (h4 : ∃ (x1 y1 x2 y2 x3 y3 : ℝ), 
    ellipse_C x1 y1 a b ∧ 
    ellipse_C x2 y2 a b ∧ 
    ellipse_C x3 y3 a b ∧ 
    circle_Ω x1 y1 D ∧ 
    circle_Ω x2 y2 D ∧ 
    circle_Ω x3 y3 D) :
  (∀ (x y : ℝ), ellipse_C x y a b ↔ x^2/6 + y^2/2 = 1) ∧
  (∃ (A : ℝ × ℝ), 
    A.1 = 7/3 ∧ A.2 = 0 ∧
    ∀ (k : ℝ) (P Q : ℝ × ℝ),
      k ≠ 0 →
      ellipse_C P.1 P.2 a b →
      ellipse_C Q.1 Q.2 a b →
      P.2 - Q.2 = k * (P.1 - Q.1) →
      ∃ (c : ℝ), 
        (P.1 - A.1)^2 + (P.2 - A.2)^2 + 
        (P.1 - A.1) * (Q.1 - P.1) + 
        (P.2 - A.2) * (Q.2 - P.2) = c) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_circle_properties_l1297_129739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DOB_l1297_129793

-- Define the points
def O : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (24, 0)
def D (k p : ℝ) : ℝ × ℝ := (0, k * p)

-- Define the constraints on k
axiom k_bounds (k : ℝ) : 0 < k ∧ k < 1

-- Define the area of a triangle given three points
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

-- Theorem statement
theorem area_of_triangle_DOB (k p : ℝ) :
  triangleArea O (D k p) B = 12 * k * p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_DOB_l1297_129793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_l1297_129719

theorem triangle_special_angle (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_condition : a^2 + c^2 - b^2 = Real.sqrt 3 * a * c) : 
  Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c)) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_special_angle_l1297_129719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l1297_129713

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a + 1/a)*x + 1

/-- The solution set of f(x) ≤ 0 for different values of a -/
noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a = 1/2 then {x | 1/2 ≤ x ∧ x ≤ 2}
  else if a = 1 then {1}
  else if a > 1 then {x | 1/a ≤ x ∧ x ≤ a}
  else {x | a ≤ x ∧ x ≤ 1/a}

theorem f_solution_set (a : ℝ) (ha : a > 0) :
  {x : ℝ | f a x ≤ 0} = solution_set a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_set_l1297_129713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1297_129764

-- Define the triangle ABC
variable (A B C : ℝ) 

-- Define the vectors m⃗ and n⃗
noncomputable def m (A : ℝ) : ℝ × ℝ := (Real.sin A, Real.cos A)
noncomputable def n : ℝ × ℝ := (1, -Real.sqrt 3)

-- State the theorem
theorem triangle_property 
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_dot_product : (m A).1 * n.1 + (m A).2 * n.2 = 0) 
  (h_tan_relation : ∃ m : ℝ, m / Real.tan A = 1 / Real.tan B + 1 / Real.tan C) :
  Real.tan A = Real.sqrt 3 ∧ 
  ∃ m : ℝ, m / Real.tan A = 1 / Real.tan B + 1 / Real.tan C ∧ 
    ∀ m' : ℝ, m' / Real.tan A = 1 / Real.tan B + 1 / Real.tan C → m ≤ m' ∧ m = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l1297_129764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_heat_sensation_with_grip_heat_sensation_consistent_with_thermodynamics_l1297_129760

noncomputable section

/-- Represents the heat flow rate according to Fourier's law of heat conduction -/
def heat_flow_rate (k : ℝ) (A : ℝ) (dT : ℝ) (dx : ℝ) : ℝ :=
  -k * A * (dT / dx)

/-- Represents the sensation of heat based on the heat flow rate -/
def heat_sensation (Q : ℝ) : ℝ := Q

/-- Represents the contact area between a hand and an object -/
def contact_area (grip_strength : ℝ) : ℝ :=
  grip_strength -- Simplified model where contact area increases with grip strength

theorem increased_heat_sensation_with_grip (k : ℝ) (dT : ℝ) (dx : ℝ) (grip1 grip2 : ℝ)
    (h_positive : k > 0)
    (h_temp_gradient : dT / dx > 0)
    (h_increased_grip : grip2 > grip1) :
    heat_sensation (heat_flow_rate k (contact_area grip2) dT dx) >
    heat_sensation (heat_flow_rate k (contact_area grip1) dT dx) := by
  sorry

theorem heat_sensation_consistent_with_thermodynamics 
    (Q1 Q2 : ℝ) (h_Q2_greater : Q2 > Q1) :
    heat_sensation Q2 > heat_sensation Q1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increased_heat_sensation_with_grip_heat_sensation_consistent_with_thermodynamics_l1297_129760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrays_count_l1297_129771

-- Define a 3x3 array
def Array3x3 : Type := Fin 3 → Fin 3 → Fin 9

-- Define the property of increasing order for a row
def IncreasingRow (a : Array3x3) (row : Fin 3) : Prop :=
  ∀ i j : Fin 3, i < j → a row i < a row j

-- Define the property of increasing order for the primary diagonal
def IncreasingDiagonal (a : Array3x3) : Prop :=
  ∀ i j : Fin 3, i < j → a i i < a j j

-- Define a valid array
def ValidArray (a : Array3x3) : Prop :=
  (∀ i j k : Fin 3, (i ≠ j ∨ i ≠ k ∨ j ≠ k) → a i j ≠ a i k) ∧
  (∀ row : Fin 3, IncreasingRow a row) ∧
  IncreasingDiagonal a

-- The main theorem
theorem valid_arrays_count :
  ∃ (s : Finset Array3x3), (∀ a ∈ s, ValidArray a) ∧ s.card = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrays_count_l1297_129771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_interval_l1297_129726

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 12*x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 12

-- Theorem statement
theorem not_monotonic_interval (k : ℝ) :
  (∃ x y, x ∈ Set.Ioo (k - 1) (k + 1) ∧ y ∈ Set.Ioo (k - 1) (k + 1) ∧ x < y ∧ f x > f y) →
  (-3 < k ∧ k < -1) ∨ (1 < k ∧ k < 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_monotonic_interval_l1297_129726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l1297_129743

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0
def circle_O2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the common chord
def common_chord (x y : ℝ) : Prop := circle_O1 x y ∧ circle_O2 x y

-- Theorem statement
theorem common_chord_length :
  ∃ (a b : ℝ), common_chord a b ∧
  (∀ (x y : ℝ), common_chord x y → (x - a)^2 + (y - b)^2 ≤ (4*Real.sqrt 5/5)^2) ∧
  (∃ (x y : ℝ), common_chord x y ∧ (x - a)^2 + (y - b)^2 = (4*Real.sqrt 5/5)^2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_l1297_129743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_legs_is_56_l1297_129728

/-- Represents the number of legs an octopus has -/
structure OctopusLegs where
  value : ℕ

/-- The number of legs a normal octopus has -/
def normalOctopusLegs : OctopusLegs := ⟨8⟩

/-- The initial number of octopuses in the tank -/
def initialOctopuses : ℕ := 5

/-- The number of octopuses removed from the tank -/
def removedOctopuses : ℕ := 2

/-- The number of new mutant octopuses added to the tank -/
def addedMutantOctopuses : ℕ := 3

/-- The number of legs of the first mutant octopus -/
def firstMutantLegs : OctopusLegs := ⟨10⟩

/-- The number of legs of the second mutant octopus -/
def secondMutantLegs : OctopusLegs := ⟨6⟩

/-- The number of legs of the third mutant octopus -/
def thirdMutantLegs : OctopusLegs := ⟨2 * normalOctopusLegs.value⟩

/-- Calculate the total number of octopus legs in the aquarium after the changes -/
def totalLegsAfterChanges : ℕ :=
  initialOctopuses * normalOctopusLegs.value -
  removedOctopuses * normalOctopusLegs.value +
  firstMutantLegs.value + secondMutantLegs.value + thirdMutantLegs.value

/-- Theorem stating that the total number of octopus legs after changes is 56 -/
theorem total_legs_is_56 : totalLegsAfterChanges = 56 := by
  rfl

#eval totalLegsAfterChanges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_legs_is_56_l1297_129728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_three_max_sum_of_a_and_b_max_sum_of_a_and_b_is_attained_l1297_129746

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle is S = (√3/4)(a² + b² - c²) -/
noncomputable def area (t : Triangle) : ℝ := (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)

theorem angle_C_is_pi_over_three (t : Triangle) (h : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)) : 
  t.C = π/3 := by sorry

theorem max_sum_of_a_and_b (t : Triangle) (h1 : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)) (h2 : t.c = Real.sqrt 3) :
  t.a + t.b ≤ 2 * Real.sqrt 3 := by sorry

theorem max_sum_of_a_and_b_is_attained (t : Triangle) (h1 : area t = (Real.sqrt 3 / 4) * (t.a^2 + t.b^2 - t.c^2)) (h2 : t.c = Real.sqrt 3) :
  ∃ (t' : Triangle), t'.c = Real.sqrt 3 ∧ t'.a + t'.b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_C_is_pi_over_three_max_sum_of_a_and_b_max_sum_of_a_and_b_is_attained_l1297_129746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_theorem_l1297_129775

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 3 = 0

/-- The line equation -/
def line_equation (x y a : ℝ) : Prop :=
  x - a*y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (1, 2)

/-- The distance formula from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A*x₀ + B*y₀ + C) / Real.sqrt (A^2 + B^2)

/-- The main theorem -/
theorem circle_line_distance_theorem (a : ℝ) :
  distance_point_to_line (circle_center.1) (circle_center.2) 1 (-a) 1 = 2 →
  a = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_theorem_l1297_129775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l1297_129745

/-- Represents a pyramid with a triangular base -/
structure Pyramid where
  H : ℝ  -- Height of the pyramid
  S_ABC : ℝ  -- Area of the base triangle

/-- The area of a cross-section parallel to the base of the pyramid -/
noncomputable def cross_section_area (p : Pyramid) (h : ℝ) : ℝ :=
  (p.S_ABC / p.H^2) * h^2

/-- Theorem: The area of a cross-section parallel to the base of a pyramid
    is proportional to the square of its distance from the apex -/
theorem cross_section_area_theorem (p : Pyramid) (h : ℝ) :
  cross_section_area p h = (p.S_ABC / p.H^2) * h^2 := by
  -- Unfold the definition of cross_section_area
  unfold cross_section_area
  -- The equality now follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_theorem_l1297_129745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l1297_129748

theorem derivative_at_two (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 3*x*(deriv f 1)) : 
  deriv f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_two_l1297_129748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1297_129705

noncomputable def f (x : ℝ) := x / (x^2 + 1)

theorem f_properties :
  let I := Set.Ioo (-1 : ℝ) 1
  (∀ x, x ∈ I → f (-x) = -f x) ∧
  (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x : ℝ, f (2*x - 1) + f x < 0 ↔ x ∈ Set.Ioo 0 (1/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1297_129705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_sum_reciprocals_transition_l1297_129792

open BigOperators

def sum_reciprocals (n : ℕ) : ℚ :=
  ∑ i in Finset.range (n + 1), 1 / (n + i : ℚ)

theorem sum_reciprocals_bound (n : ℕ) (h : n ≥ 2) :
  sum_reciprocals n < 1 := by
  sorry

theorem sum_reciprocals_transition (k : ℕ) (h : k ≥ 2) :
  sum_reciprocals (k + 1) - sum_reciprocals k =
    1 / (2 * k + 1 : ℚ) + 1 / (2 * k + 2 : ℚ) - 1 / (k : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_bound_sum_reciprocals_transition_l1297_129792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proctoring_arrangements_l1297_129701

/-- Represents the number of teachers and classes -/
def n : ℕ := 4

/-- The total number of possible arrangements without restrictions -/
def total_arrangements : ℕ := Nat.factorial n

/-- The number of arrangements where at least one teacher proctors their own class -/
def invalid_arrangements : ℕ := n.choose 1 * 2 + n.choose 2 + 1

/-- The number of valid arrangements where no teacher proctors their own class -/
def valid_arrangements : ℕ := total_arrangements - invalid_arrangements

theorem proctoring_arrangements :
  valid_arrangements = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proctoring_arrangements_l1297_129701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_8_9_equals_9_l1297_129737

/-- Represents a repeating decimal where the digit 9 repeats infinitely after the decimal point -/
noncomputable def repeating_decimal_8_9 : ℝ :=
  8 + (9/10) / (1 - 1/10)

/-- Theorem stating that the repeating decimal 8.999... is equal to 9 -/
theorem repeating_decimal_8_9_equals_9 : repeating_decimal_8_9 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_8_9_equals_9_l1297_129737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_f_gt_sin_g_cos_cos_gt_sin_sin_l1297_129772

-- Define the functions f and g
variable {f g : ℝ → ℝ}

-- State the first theorem
theorem cos_f_gt_sin_g (h : ∀ x, -π/2 < f x - g x ∧ f x - g x < π/2 ∧ -π/2 < f x + g x ∧ f x + g x < π/2) :
  ∀ x, Real.cos (f x) > Real.sin (g x) := by sorry

-- State the second theorem
theorem cos_cos_gt_sin_sin :
  ∀ x, Real.cos (Real.cos x) > Real.sin (Real.sin x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_f_gt_sin_g_cos_cos_gt_sin_sin_l1297_129772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_x_coordinates_l1297_129736

-- Define the lines l and m
noncomputable def line_l (x : ℝ) : ℝ := -5/3 * x + 5
noncomputable def line_m (x : ℝ) : ℝ := -2/7 * x + 2

-- Define the points where lines l and m intersect y = 20
noncomputable def x_l : ℝ := (20 - 5) / (5/3)
noncomputable def x_m : ℝ := (20 - 2) / (2/7)

-- Theorem statement
theorem difference_in_x_coordinates : 
  |x_l - x_m| = 54 := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_x_coordinates_l1297_129736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_one_intersection_l1297_129730

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + (h.b / h.a)^2)

/-- A parabola defined by y = x^2 + 1 -/
def parabola (x : ℝ) : ℝ := x^2 + 1

theorem hyperbola_eccentricity_with_one_intersection (h : Hyperbola) :
  (∃! x : ℝ, (x^2 / h.a^2) - ((parabola x)^2 / h.b^2) = 1) →
  eccentricity h = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_with_one_intersection_l1297_129730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_sum_of_extrema_l1297_129788

/-- The function y = a^x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

/-- The theorem stating the conditions and the result to be proved -/
theorem exponential_function_sum_of_extrema (a : ℝ) :
  a > 0 →
  a ≠ 1 →
  (∃ (ymax ymin : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ ymax) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = ymax) ∧
    (∀ x ∈ Set.Icc 1 2, f a x ≥ ymin) ∧
    (∃ x ∈ Set.Icc 1 2, f a x = ymin) ∧
    ymax + ymin = 6) →
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_sum_of_extrema_l1297_129788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l1297_129783

open Real MeasureTheory

theorem integral_sqrt_one_minus_x_squared_plus_x : 
  ∫ x in Set.Icc (-1) 1, (Real.sqrt (1 - x^2) + x) = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sqrt_one_minus_x_squared_plus_x_l1297_129783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strip_length_bounded_l1297_129702

/-- Represents the length of the paved part of the strip after n panels -/
noncomputable def strip_length (a₁ : ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - k^n) / (1 - k)

/-- Theorem stating that there exists an upper bound for the strip length -/
theorem strip_length_bounded (a₁ k : ℝ) (h₁ : a₁ ≠ 1) (h₂ : 0 < k) (h₃ : k < 1) :
  ∃ (M : ℝ), ∀ (n : ℕ), strip_length a₁ k n ≤ M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strip_length_bounded_l1297_129702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_two_over_a_l1297_129725

-- Define the functions f and g as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 - 4)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (x^2 + 4)

-- State the theorem
theorem function_sum_equals_two_over_a (a : ℝ) (h : 0 < a ∧ a < 1) :
  f (a + 1/a) + g (a - 1/a) = 2/a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_sum_equals_two_over_a_l1297_129725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_values_l1297_129776

/-- The number of distinct values obtained by parenthesizing 3^3^3^3 in all possible ways -/
def num_distinct_values : ℕ := 3

/-- The standard convention value of 3^(3^(3^3)) -/
noncomputable def standard_value : ℕ := 3^(3^27)

/-- A function to represent different parenthesizations -/
noncomputable def parenthesize (n : ℕ) (p : List (List ℕ)) : ℕ :=
  sorry  -- Implementation details omitted for brevity

theorem three_distinct_values :
  ∃ (a b : ℕ), a ≠ b ∧ a ≠ standard_value ∧ b ≠ standard_value ∧
  (∀ c : ℕ, c ≠ a ∧ c ≠ b ∧ c ≠ standard_value →
    c ∉ {x | ∃ p : List (List ℕ), x = parenthesize (3^3^3^3) p}) :=
by
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_distinct_values_l1297_129776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l1297_129759

theorem solution_difference : 
  ∃ r₁ r₂ : ℝ, 
    let f := fun r => (r^2 - 5*r - 12) / (r + 5) - (3*r + 10)
    f r₁ = 0 ∧ f r₂ = 0 ∧ 
    r₁ ≠ -5 ∧ r₂ ≠ -5 ∧
    |r₁ - r₂| = Real.sqrt 101 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l1297_129759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_l1297_129796

-- Define the segment CD and points R and S
variable (CD R S : ℝ)

-- Define the ratios for R and S
axiom R_ratio : R / (CD - R) = 3 / 5
axiom S_ratio : S / (CD - S) = 4 / 7

-- Define the length of RS
axiom RS_length : S - R = 5

-- Theorem to prove
theorem length_of_CD : CD = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_CD_l1297_129796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circles_to_cover_l1297_129762

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the cover relation
def covers (c1 c2 : Circle) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 ≤ c2.radius^2 →
    (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 ≤ c1.radius^2

-- Define the complete cover relation
def completely_covers (cs : List Circle) (c : Circle) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 ≤ c.radius^2 →
    ∃ c' ∈ cs, covers c' c

-- Main theorem
theorem min_circles_to_cover (K : Circle) (Ks : List Circle) 
  (h1 : Ks ≠ [])
  (h2 : K.radius = 2 * (Ks.head h1).radius)
  (h3 : ∀ K' ∈ Ks, K'.radius = (Ks.head h1).radius)
  (h4 : completely_covers Ks K) :
  Ks.length ≥ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circles_to_cover_l1297_129762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_19_satisfies_conditions_l1297_129785

def numbers : List Nat := [16, 18, 19, 21]

def is_less_than_20 (n : Nat) : Prop := n < 20

def is_not_multiple_of_2 (n : Nat) : Prop := n % 2 ≠ 0

theorem only_19_satisfies_conditions : 
  ∃! n, n ∈ numbers ∧ is_less_than_20 n ∧ is_not_multiple_of_2 n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_19_satisfies_conditions_l1297_129785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_distant_positions_l1297_129747

/-- A type representing a position of cars on n stations -/
def Position (n : ℕ) := Fin n → Fin n

/-- The distance between two positions -/
def distance {n : ℕ} (f g : Position n) : ℝ :=
  (Finset.univ.filter (λ i => f i ≠ g i)).card

/-- Main theorem statement -/
theorem existence_of_distant_positions :
  ∀ α : ℝ, α < 1 → 
  ∃ n : ℕ, ∃ positions : Finset (Position n),
    positions.card = 100^n ∧
    ∀ f g : Position n, f ∈ positions → g ∈ positions → f ≠ g → distance f g ≥ n * α := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_distant_positions_l1297_129747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_cube_dots_not_consecutive_l1297_129750

/-- Represents a die with specific dot distribution -/
structure Die where
  face1 : ℕ
  face2 : ℕ
  face3 : ℕ
  opposite_faces : face1 = 1 ∧ face2 = 2 ∧ face3 = 3

/-- Represents a 2x2x2 cube formed by 8 identical dice -/
def LargeCube := Fin 8 → Die

/-- Calculates the total number of dots on a face of the large cube -/
def faceDots (cube : LargeCube) (face : Fin 6) : ℕ :=
  sorry

/-- Checks if a list of 6 numbers are consecutive integers -/
def areConsecutiveIntegers (nums : List ℕ) : Prop :=
  nums.length = 6 ∧ ∃ n : ℕ, nums = [n, n+1, n+2, n+3, n+4, n+5]

/-- Theorem: It's impossible for the dots on the faces of the large cube to be consecutive integers -/
theorem large_cube_dots_not_consecutive (cube : LargeCube) :
  ¬ (areConsecutiveIntegers [faceDots cube 0, faceDots cube 1, faceDots cube 2,
                             faceDots cube 3, faceDots cube 4, faceDots cube 5]) :=
by
  sorry

#check large_cube_dots_not_consecutive

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_cube_dots_not_consecutive_l1297_129750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_l1297_129787

/-- Helper function to represent the area of an inscribed circle in a sector -/
noncomputable def area_of_inscribed_circle_in_sector (angle : ℝ) (radius : ℝ) : ℝ :=
  sorry

/-- The area of a circle inscribed in a circular sector with central angle 60° and radius r -/
theorem inscribed_circle_area (r : ℝ) (h : r > 0) : 
  ∃ A : ℝ, A = (r^2 * Real.pi) / 9 ∧ 
  A = area_of_inscribed_circle_in_sector 60 r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_area_l1297_129787
