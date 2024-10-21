import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_rate_theorem_scrap_rate_problem_l915_91537

theorem scrap_rate_theorem (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  1 - (1 - a) * (1 - b) = a + b - a * b := by
  ring

theorem scrap_rate_problem (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
  let total_scrap_rate := 1 - (1 - a) * (1 - b)
  let first_procedure_scrap_rate := a
  let second_procedure_scrap_rate := b
  total_scrap_rate = first_procedure_scrap_rate + second_procedure_scrap_rate - first_procedure_scrap_rate * second_procedure_scrap_rate := by
  simp [scrap_rate_theorem a b ha hb]

#check scrap_rate_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scrap_rate_theorem_scrap_rate_problem_l915_91537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l915_91526

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a-3)x with an even derivative,
    the tangent line to f(x) at the origin has the equation y = -3. -/
theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ) 
    (h_f : ∀ x, f x = x^3 + a*x^2 + (a-3)*x)
    (h_even : ∀ x, (deriv f) (-x) = (deriv f) x) :
    (fun y => y = -3) = (fun y => ∃ k, y = k * 0 + f 0 ∧ k = (deriv f) 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_origin_l915_91526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_pi_over_two_l915_91587

variable (a b : ℝ × ℝ)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude (length) of a 2D vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the angle between two 2D vectors
noncomputable def angle_between (v w : ℝ × ℝ) : ℝ :=
  Real.arccos ((dot_product v w) / (magnitude v * magnitude w))

-- State the theorem
theorem angle_is_pi_over_two (h1 : dot_product b (a + b) = 1) (h2 : magnitude b = 1) :
  angle_between a b = π / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_pi_over_two_l915_91587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l915_91577

/-- 
Given an initial investment of $400 that gains 25% in the first week 
and an additional 50% in the second week, prove that the final value is $750.
-/
theorem investment_growth (initial_investment : ℝ) 
  (first_week_gain : ℝ) (second_week_gain : ℝ) : 
  initial_investment = 400 ∧ 
  first_week_gain = 0.25 ∧ 
  second_week_gain = 0.50 → 
  initial_investment * (1 + first_week_gain) * (1 + second_week_gain) = 750 :=
by
  intro h
  have h1 : initial_investment = 400 := h.left
  have h2 : first_week_gain = 0.25 := h.right.left
  have h3 : second_week_gain = 0.50 := h.right.right
  
  -- Calculate the value after the first week
  let after_first_week := initial_investment * (1 + first_week_gain)
  
  -- Calculate the final value after the second week
  let final_value := after_first_week * (1 + second_week_gain)
  
  -- Prove that the final value is equal to 750
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l915_91577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_representatives_theorem_l915_91572

/-- Represents a seating arrangement of representatives around a table. -/
structure SeatingArrangement (n : ℕ) where
  seats : List ℕ
  valid : ∀ i j, i < seats.length → j < seats.length →
    i ≠ j → seats[i]! = seats[j]! → seats[(i+1) % seats.length]! ≠ seats[(j+1) % seats.length]!

/-- The maximum number of representatives that can be seated for n countries. -/
def maxRepresentatives (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the maximum number of representatives that can be seated. -/
theorem max_representatives_theorem (n : ℕ) (h : n > 1) :
  ∃ (arr : SeatingArrangement n), arr.seats.length = maxRepresentatives n ∧
  ∀ (arr' : SeatingArrangement n), arr'.seats.length ≤ maxRepresentatives n := by
  sorry

#check max_representatives_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_representatives_theorem_l915_91572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_derivative_at_two_l915_91514

-- Define the function f(x) = 1/x
noncomputable def f (x : ℝ) : ℝ := 1 / x

-- State the theorem
theorem limit_derivative_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (2 + Δx) - f 2) / Δx + 1/4| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_derivative_at_two_l915_91514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l915_91556

noncomputable def f (x : ℝ) : ℝ := -(2*x - 1)*(x + 3)

noncomputable def x₁ : ℝ := 1/2
noncomputable def x₂ : ℝ := -3

theorem parabola_intersection_distance : 
  x₁ - x₂ = 7/2 ∧ f x₁ = 0 ∧ f x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_distance_l915_91556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rolles_theorem_applicable_l915_91535

-- Define the function f(x) = 1 - ∛(x²)
noncomputable def f (x : ℝ) : ℝ := 1 - Real.rpow (x^2) (1/3)

-- Define the interval [-1, 1]
def interval : Set ℝ := Set.Icc (-1) 1

theorem not_rolles_theorem_applicable :
  ¬ (∀ x ∈ interval, ContinuousAt f x ∧ 
     f (-1) = f 1 ∧
     ∀ y ∈ Set.Ioo (-1) 1, DifferentiableAt ℝ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_rolles_theorem_applicable_l915_91535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_four_ninths_l915_91524

/-- The sum of the infinite series ∑(k/4^k) for k = 1 to ∞ -/
noncomputable def infinite_series_sum : ℝ := ∑' k, k / (4 ^ k)

/-- Theorem stating that the sum of the infinite series ∑(k/4^k) for k = 1 to ∞ is equal to 4/9 -/
theorem infinite_series_sum_equals_four_ninths : infinite_series_sum = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_equals_four_ninths_l915_91524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l915_91594

noncomputable def a : ℕ → ℝ
| 0 => 15
| n + 1 => (Real.sqrt ((a n)^2 + 1) - 1) / (a n)

theorem a_lower_bound : ∀ n : ℕ, a n > 3 / 2^n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l915_91594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hanas_stamp_collection_value_l915_91597

theorem hanas_stamp_collection_value 
  (sold_fraction : Rat) 
  (sold_value : ℕ) 
  (entire_collection_value : ℕ) 
  (h1 : sold_fraction = 4/7)
  (h2 : sold_value = 28)
  (h3 : entire_collection_value = 49)
  : entire_collection_value = (sold_value * 7) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hanas_stamp_collection_value_l915_91597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l915_91507

theorem orthogonal_vectors (y : ℝ) : 
  (2 * (-4) + (-6) * y + (-8) * (-2) = 0) ↔ y = 4/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l915_91507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l915_91581

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (2 : ℝ)^3 * (2 : ℝ)^x = 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_exponential_equation_l915_91581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_cone_height_ratio_l915_91579

/-- Represents a right cone with a circular base -/
structure RightCone where
  base_circumference : ℝ
  height : ℝ

/-- Calculates the volume of a right cone -/
noncomputable def volume (cone : RightCone) : ℝ :=
  (1/3) * (cone.base_circumference / (2 * Real.pi)) ^ 2 * Real.pi * cone.height

theorem shorter_cone_height_ratio (original : RightCone) (shorter : RightCone) :
  original.base_circumference = 24 * Real.pi →
  original.height = 40 →
  shorter.base_circumference = original.base_circumference →
  volume shorter = 432 * Real.pi →
  shorter.height / original.height = 9 / 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_cone_height_ratio_l915_91579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l915_91560

theorem trigonometric_equation_solution (n : ℕ) :
  ∀ x : ℝ, (1 / Real.sin x ^ (2 * n) + 1 / Real.cos x ^ (2 * n) = 2 ^ (n + 1)) ↔
  ∃ k : ℤ, x = (2 * k + 1 : ℝ) * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l915_91560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l915_91518

/-- Represents the number of friends each person invites to repost -/
def n : ℕ := sorry

/-- Represents the total number of participants after two rounds of spreading -/
def total_participants : ℕ := 1641

/-- The function that calculates the number of participants after two rounds -/
def participants_after_two_rounds (n : ℕ) : ℕ := 1 + n + n^2

/-- Theorem stating that the calculated number of participants equals the given total -/
theorem correct_equation : participants_after_two_rounds n = total_participants := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_l915_91518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l915_91588

-- Define the propositions
def proposition1 : Prop := ∀ m : ℝ, (∀ x : ℝ, x^2 + 2*x - m ≠ 0) → m ≤ 0

def proposition2 : Prop := ∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1

-- We'll replace Quadrilateral, DiagonalsEqual, and IsRectangle with placeholder definitions
def Quadrilateral := ℝ × ℝ × ℝ × ℝ
def DiagonalsEqual (q : Quadrilateral) : Prop := sorry
def IsRectangle (q : Quadrilateral) : Prop := sorry

def proposition3 : Prop := ∀ q : Quadrilateral, (DiagonalsEqual q) → (IsRectangle q)

def proposition4 : Prop := ∃ x : ℝ, x^2 + x + 3 ≤ 0

-- Define the theorem
theorem propositions_truth : proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l915_91588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_decreasing_l915_91519

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem function_monotonically_decreasing 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi / 2) 
  (h_f0 : f ω φ 0 = 1 / 2) 
  (h_period : ω = 2) :
  ∀ x ∈ Set.Icc (Real.pi / 6) ((2 : ℝ) * Real.pi / 3),
    ∀ y ∈ Set.Icc (Real.pi / 6) ((2 : ℝ) * Real.pi / 3),
      x < y → f ω φ x > f ω φ y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonically_decreasing_l915_91519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_powers_of_ten_in_ap_l915_91558

theorem infinitely_many_powers_of_ten_in_ap :
  ∃ f : ℕ → ℕ, StrictMono f ∧
  ∀ k : ℕ, ∃ n : ℕ, 10^(486 * f k) = 1 + 729 * n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_powers_of_ten_in_ap_l915_91558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_l915_91523

-- Define the piecewise function f
noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ :=
  if x > 0 then a * x + 3
  else if x = 0 then a * b
  else b * x^2 + c

-- State the theorem
theorem sum_of_constants (a b c : ℕ) :
  (f a b c 2 = 7) →
  (f a b c 0 = 6) →
  (f a b c (-1) = 8) →
  a + b + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_constants_l915_91523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patrick_pencils_l915_91574

/-- The number of pencils Patrick purchased -/
def num_pencils : ℕ := 80

/-- The selling price of one pencil -/
def S : ℝ := sorry

/-- The cost price of one pencil -/
def C : ℝ := 1.2 * S

/-- The total loss -/
def loss : ℝ := 16 * S

theorem patrick_pencils :
  (num_pencils : ℝ) * C - (num_pencils : ℝ) * S = loss := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_patrick_pencils_l915_91574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l915_91501

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 8 = 0

-- Define the right focus of the hyperbola
def right_focus : ℝ × ℝ := (3, 0)

-- Define the distance function from a point to a line
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

-- Theorem statement
theorem distance_right_focus_to_line :
  distance_point_to_line (right_focus.1) (right_focus.2) 1 2 (-8) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_right_focus_to_line_l915_91501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_bell_problem_l915_91580

/-- Represents the number of times Martin rings the big bell -/
noncomputable def B : ℝ := sorry

/-- Represents the number of times Martin rings the small bell -/
noncomputable def S : ℝ := (5/8) * Real.sqrt B

theorem martin_bell_problem :
  ∃ B : ℝ, B > 0 ∧ 
  64 * B^2 + 80 * B^(3/2) + 25 * B - 173056 = 0 ∧
  B + S = 52 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martin_bell_problem_l915_91580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_sales_problem_l915_91530

/-- Refrigerator sales problem -/
theorem refrigerator_sales_problem 
  (total_before : ℕ) 
  (total_after : ℕ) 
  (increase_type_I : ℚ) 
  (increase_type_II : ℚ) 
  (price_type_I : ℕ) 
  (price_type_II : ℕ) 
  (subsidy_rate : ℚ) 
  (h1 : total_before = 960) 
  (h2 : total_after = 1228) 
  (h3 : increase_type_I = 30 / 100) 
  (h4 : increase_type_II = 25 / 100) 
  (h5 : price_type_I = 2298) 
  (h6 : price_type_II = 1999) 
  (h7 : subsidy_rate = 13 / 100) :
  ∃ (x y : ℕ), 
    x + y = total_before ∧ 
    (1 + increase_type_I) * x + (1 + increase_type_II) * y = total_after ∧ 
    x = 560 ∧ 
    y = 400 ∧ 
    (((price_type_I * x * (1 + increase_type_I) + price_type_II * y * (1 + increase_type_II)) * subsidy_rate).floor : ℚ) = 350000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_sales_problem_l915_91530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_after_four_hours_l915_91548

def water_loss (hour : ℕ) : ℕ := 2^(hour - 1)

def water_added (hour : ℕ) : ℕ := 
  match hour with
  | 0 => 0
  | 1 => 0
  | 2 => 2
  | n + 3 => 2 * water_added (n + 2)

def water_in_tank (initial_water : ℕ) (hour : ℕ) : ℕ :=
  initial_water - (Finset.sum (Finset.range hour) (λ i => water_loss (i + 1))) + 
                  (Finset.sum (Finset.range hour) (λ i => water_added (i + 1)))

theorem water_after_four_hours (initial_water : ℕ) :
  initial_water = 80 → water_in_tank initial_water 4 = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_after_four_hours_l915_91548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_not_in_given_forms_l915_91570

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the slope of a line passing through two points -/
noncomputable def calculateSlope (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

/-- Calculates the y-intercept of a line given a point and slope -/
noncomputable def calculateIntercept (p : Point) (m : ℝ) : ℝ :=
  p.y - m * p.x

/-- Generates the equation of a line in the form Ax + By + C = 0 -/
def lineEquation (l : Line) : ℝ → ℝ → ℝ → Prop :=
  fun x y c => l.slope * x - y + l.intercept = c

/-- Checks if a given equation matches any of the provided forms -/
def matchesGivenForms (f : ℝ → ℝ → ℝ → Prop) (a T : ℝ) : Prop :=
  (∀ x y, f x y ((4 * T * x + 2 * a^2 * y + 4 * a * T) / (4 * T))) ∨
  (∀ x y, f x y ((4 * T * x - 2 * a^2 * y + 4 * a * T) / (4 * T))) ∨
  (∀ x y, f x y ((4 * T * x + 2 * a^2 * y - 4 * a * T) / (4 * T))) ∨
  (∀ x y, f x y ((4 * T * x - 2 * a^2 * y - 4 * a * T) / (4 * T)))

theorem line_equation_not_in_given_forms (a T : ℝ) (h1 : a ≠ 0) (h2 : T > 0) :
  ∃ (l : Line), 
    (l.slope = (T - a^2) / (2 * a^2)) ∧
    (l.intercept = (2 * T - 2 * a^2 + 2 * a^3) / (2 * a^2)) ∧
    (lineEquation l (-2 * a) a 0) ∧
    ¬(matchesGivenForms (lineEquation l) a T) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_not_in_given_forms_l915_91570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l915_91595

open Real

noncomputable def f (x : ℝ) : ℝ := sin (2*x) * cos (2*x + π/3) + 4*sqrt 3 * (sin x)^2 * (cos x)^2

theorem f_properties :
  -- The smallest positive period is π/2
  (∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q)) ∧
  -- The maximum value on [0, π/4] is (√3 + 2)/4
  (∃ (x : ℝ), x ∈ Set.Icc 0 (π/4) ∧ f x = (sqrt 3 + 2)/4 ∧ ∀ y ∈ Set.Icc 0 (π/4), f y ≤ (sqrt 3 + 2)/4) ∧
  -- The minimum value on [0, π/4] is 0
  (∃ (x : ℝ), x ∈ Set.Icc 0 (π/4) ∧ f x = 0 ∧ ∀ y ∈ Set.Icc 0 (π/4), f y ≥ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l915_91595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l915_91500

/-- Represents the distribution of scores in a mathematics test. -/
structure ScoreDistribution where
  score60 : ℝ
  score75 : ℝ
  score85 : ℝ
  score95 : ℝ
  sum_to_one : score60 + score75 + score85 + score95 = 1

/-- Calculates the mean score given a score distribution. -/
def mean_score (d : ScoreDistribution) : ℝ :=
  60 * d.score60 + 75 * d.score75 + 85 * d.score85 + 95 * d.score95

/-- Determines the median score given a score distribution. -/
noncomputable def median_score (d : ScoreDistribution) : ℝ :=
  if d.score60 + d.score75 > 0.5 then 75
  else if d.score60 + d.score75 + d.score85 > 0.5 then 85
  else 95

/-- The main theorem stating the difference between median and mean scores. -/
theorem median_mean_difference (d : ScoreDistribution) 
  (h1 : d.score60 = 0.2) 
  (h2 : d.score75 = 0.15) 
  (h3 : d.score85 = 0.4) : 
  median_score d - mean_score d = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_mean_difference_l915_91500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l915_91550

-- Define the arithmetic sequence a_n
noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

-- Define the sum of first n terms
noncomputable def sum_of_terms (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

-- Main theorem
theorem arithmetic_sequence_property (a₁ d : ℝ) :
  (∀ n : ℕ, n > 0 → ∃ k : ℝ, Real.sqrt (sum_of_terms a₁ d n + n) = k + (n - 1 : ℝ) * d) →
  (∀ n : ℕ, n > 0 → arithmetic_sequence a₁ d n = -1 ∨ arithmetic_sequence a₁ d n = (1/2 : ℝ) * n - 5/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l915_91550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_selection_theorem_l915_91540

/-- A function that returns the number of ways to choose 3 distinct numbers from 1 to 8 that sum to n -/
def validCombinations (n : ℕ) : ℕ := sorry

/-- The set of possible sums when choosing 3 distinct numbers from 1 to 8 -/
def validSums : Finset ℕ := Finset.filter (fun n => 3 ≤ n ∧ n ≤ 21) (Finset.range 22)

/-- The total number of valid ways to choose marbles -/
def totalWays : ℕ := validSums.sum validCombinations

theorem marble_selection_theorem :
  totalWays = (Finset.range 19).sum (fun i => validCombinations (i + 3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_selection_theorem_l915_91540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_even_remaining_l915_91554

theorem probability_at_least_one_even_remaining (S : Finset ℕ) : 
  S = {1, 2, 3, 4, 5} →
  (Finset.card S = 5) →
  (∃ (even : ℕ → Prop), ∀ n, even n ↔ n % 2 = 0) →
  (Nat.choose 5 3 - Nat.choose 3 2 : ℚ) / Nat.choose 5 3 = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_at_least_one_even_remaining_l915_91554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l915_91599

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sin θ)

noncomputable def line_l (α t : ℝ) : ℝ × ℝ := (t * Real.cos α, Real.sqrt 2 + t * Real.sin α)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_sum_distances (α : ℝ) :
  ∃ (θ₁ θ₂ t₁ t₂ : ℝ),
    curve_C θ₁ = line_l α t₁ ∧
    curve_C θ₂ = line_l α t₂ ∧
    θ₁ ≠ θ₂ →
    ∀ (θ₃ θ₄ t₃ t₄ : ℝ),
      curve_C θ₃ = line_l α t₃ →
      curve_C θ₄ = line_l α t₄ →
      distance (0, Real.sqrt 2) (line_l α t₁) + distance (0, Real.sqrt 2) (line_l α t₂) ≤
      distance (0, Real.sqrt 2) (line_l α t₃) + distance (0, Real.sqrt 2) (line_l α t₄) →
      distance (0, Real.sqrt 2) (line_l α t₁) + distance (0, Real.sqrt 2) (line_l α t₂) =
      4 * Real.sqrt 6 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l915_91599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_l915_91525

/-- Predicate to determine if a point is the orthocenter of a triangle -/
def is_orthocenter (A B C H : ℝ × ℝ) : Prop :=
  (H.1 - A.1) * (B.1 - C.1) + (H.2 - A.2) * (B.2 - C.2) = 0 ∧
  (H.1 - B.1) * (C.1 - A.1) + (H.2 - B.2) * (C.2 - A.2) = 0 ∧
  (H.1 - C.1) * (A.1 - B.1) + (H.2 - C.2) * (A.2 - B.2) = 0

/-- Given a triangle ABC with A = (-a, 0), B = (a, 0), and C = (k, b),
    where a > 0, b ≠ 0, and k is variable, the locus of the orthocenter
    H(x, y) satisfies the equation by = a² - x². -/
theorem orthocenter_locus (a b : ℝ) (h₁ : a > 0) (h₂ : b ≠ 0) :
  ∀ k x y : ℝ, is_orthocenter (-a, 0) (a, 0) (k, b) (x, y) →
    b * y = a^2 - x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_locus_l915_91525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l915_91598

/-- The function g as defined in the problem -/
noncomputable def g (x y z w : ℝ) : ℝ := x/(x+y) + y/(y+z) + z/(z+w) + w/(w+x)

/-- The theorem stating the range of g -/
theorem g_range (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w ≥ 0) :
  ∃ (a b : ℝ), a > 1.5 ∧ b < 2 ∧ g x y z w ∈ Set.Ioo a b := by
  sorry

/-- A helper lemma stating that g is always positive -/
lemma g_positive (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w ≥ 0) :
  g x y z w > 0 := by
  sorry

/-- A helper lemma stating that g is always less than 2 -/
lemma g_less_than_two (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w ≥ 0) :
  g x y z w < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l915_91598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_equation_l915_91502

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  origin : Point3D
  direction1 : Point3D
  direction2 : Point3D

/-- Defines the given parametric plane -/
def givenPlane : ParametricPlane :=
  { origin := { x := 1, y := 2, z := 3 }
  , direction1 := { x := 1, y := -1, z := -2 }
  , direction2 := { x := -1, y := 0, z := 2 } }

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The equation we want to prove -/
def targetEquation : PlaneEquation :=
  { A := 2, B := 0, C := 1, D := -5 }

/-- Checks if the coefficients satisfy the required conditions -/
def validCoefficients (eq : PlaneEquation) : Prop :=
  eq.A > 0 ∧ Int.gcd (Int.natAbs eq.A) (Int.gcd (Int.natAbs eq.B) (Int.gcd (Int.natAbs eq.C) (Int.natAbs eq.D))) = 1

/-- Main theorem: The given parametric plane equation is equivalent to the target equation -/
theorem parametric_to_equation : 
  ∀ (p : Point3D), 
    (∃ (s t : ℝ), 
      p.x = givenPlane.origin.x + s * givenPlane.direction1.x + t * givenPlane.direction2.x ∧
      p.y = givenPlane.origin.y + s * givenPlane.direction1.y + t * givenPlane.direction2.y ∧
      p.z = givenPlane.origin.z + s * givenPlane.direction1.z + t * givenPlane.direction2.z) ↔
    (targetEquation.A * p.x + targetEquation.B * p.y + targetEquation.C * p.z + targetEquation.D = 0) ∧
    validCoefficients targetEquation := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_equation_l915_91502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_equals_159_375_l915_91531

/-- Represents a trapezoid ABCD with parallel sides AB and CD -/
structure Trapezoid where
  ab : ℝ  -- Length of side AB
  cd : ℝ  -- Length of side CD
  h : ℝ   -- Altitude of the trapezoid

/-- Calculates the area of quadrilateral EFCD within trapezoid ABCD -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  let ef := (t.ab + t.cd) / 2
  let h_efcd := t.h / 2
  h_efcd * (ef + t.cd) / 2

/-- Theorem: The area of quadrilateral EFCD in the given trapezoid is 159.375 square units -/
theorem area_EFCD_equals_159_375 (t : Trapezoid) 
    (h_ab : t.ab = 10)
    (h_cd : t.cd = 25)
    (h_h : t.h = 15) : 
  area_EFCD t = 159.375 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_equals_159_375_l915_91531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l915_91590

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry
noncomputable def sequence_c : ℕ → ℝ := sorry
noncomputable def s : ℕ → ℝ := sorry
noncomputable def T : ℕ → ℝ := sorry

axiom s_def : ∀ n : ℕ, s n = 2 * sequence_a n - 2
axiom b_initial : sequence_b 1 = 1
axiom b_relation : ∀ n : ℕ, sequence_b (n + 1) = sequence_b n + 2
axiom c_def : ∀ n : ℕ, sequence_c n = sequence_a n * sequence_b n

theorem sequence_properties :
  (∀ n : ℕ, sequence_a n = 2^n) ∧
  (∀ n : ℕ, sequence_b n = 2*n - 1) ∧
  (∀ n : ℕ, T n = (2^n - 3) * 2^n + 7) ∧
  (∀ k : ℕ, k > 4 → T k ≥ 167) ∧ (T 4 < 167) :=
by
  sorry

#check sequence_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l915_91590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_points_l915_91563

/-- The distance between two points in polar coordinates -/
noncomputable def distance_polar (r₁ r₂ : ℝ) (θ₁ θ₂ : ℝ) : ℝ :=
  Real.sqrt (r₁^2 + r₂^2 - 2*r₁*r₂*(Real.cos (θ₁ - θ₂)))

/-- Theorem: The distance between points C(4, θ₃) and D(6, θ₄) in polar coordinates,
    where θ₃ - θ₄ = π/3, is equal to 2√7 -/
theorem distance_specific_points (θ₃ θ₄ : ℝ) (h : θ₃ - θ₄ = π/3) :
  distance_polar 4 6 θ₃ θ₄ = 2 * Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_specific_points_l915_91563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l915_91561

noncomputable section

/-- The radius of the larger circle -/
def R : ℝ := 2 * Real.sqrt 3

/-- The area of the larger circle -/
def A₁₂ : ℝ := Real.pi * R^2

/-- The area of the equilateral triangle with side length s -/
def A₃ (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The area of the smaller circle -/
def A₁ (s : ℝ) : ℝ := A₁₂ / 2 - A₃ s / 2

/-- The radius of the smaller circle -/
def r (s : ℝ) : ℝ := Real.sqrt (A₁ s / Real.pi)

/-- The theorem stating the radius of the smaller circle -/
theorem smaller_circle_radius (s : ℝ) : 
  r s = Real.sqrt (6 - (Real.sqrt 3 / 8) * s^2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_circle_radius_l915_91561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_propositions_l915_91569

-- Define the propositions
def proposition1 : Prop := ∀ (T : Type) (isIsosceles : T → Prop) (midpoint base : T → T) (legs : T → T × T) (dist : T → T → ℝ),
  ∀ t, isIsosceles t → dist (midpoint (base t)) (Prod.fst (legs t)) = dist (midpoint (base t)) (Prod.snd (legs t))

def proposition2 : Prop := ∀ (T : Type) (isIsosceles : T → Prop) (height median angleBisector : T → T),
  ∀ t, isIsosceles t → height t = median t ∧ median t = angleBisector t

def proposition3 : Prop := ∀ (T : Type) (isIsosceles : T → Prop) (baseAngle vertexAngle : T → ℝ) (isCongruent : T → T → Prop),
  ∀ t1 t2, isIsosceles t1 ∧ isIsosceles t2 ∧ baseAngle t1 = baseAngle t2 ∧ vertexAngle t1 = vertexAngle t2 → isCongruent t1 t2

def proposition4 : Prop := ∀ (T : Type) (isEquilateral : T → Prop) (angle : T → ℝ),
  ∀ t, angle t = 60 → isEquilateral t

def proposition5 : Prop := ∀ (T : Type) (isIsosceles : T → Prop) (axisOfSymmetry angleBisector : T → T),
  ∀ t, isIsosceles t → axisOfSymmetry t = angleBisector t

-- Helper function to count true propositions
def countTruePropositions (p1 p2 p3 p4 p5 : Bool) : Nat :=
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) + (if p4 then 1 else 0) + (if p5 then 1 else 0)

-- Theorem statement
theorem exactly_three_correct_propositions :
  ∃ (p1 p2 p3 p4 p5 : Bool),
    (p1 ↔ proposition1) ∧
    (p2 ↔ proposition2) ∧
    (p3 ↔ proposition3) ∧
    (p4 ↔ proposition4) ∧
    (p5 ↔ proposition5) ∧
    countTruePropositions p1 p2 p3 p4 p5 = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_correct_propositions_l915_91569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_general_formula_l915_91573

def a : ℕ → ℕ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | n + 1 => 2 * a n + 2^n

def b (n : ℕ) : ℚ := if n = 0 then 1 else (a n : ℚ) / 2^(n - 1)

theorem arithmetic_sequence_and_general_formula :
  (∀ n : ℕ, n ≥ 1 → b (n + 1) = b n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → a n = n * 2^(n - 1)) := by
  sorry

#eval a 5  -- Added to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_and_general_formula_l915_91573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_iff_in_open_interval_l915_91542

noncomputable section

/-- A piecewise function f(x) defined as:
    f(x) = (5-a)x-4a for x < 1
    f(x) = a^x for x ≥ 1
    where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 - a) * x - 4 * a else a^x

/-- The function f is increasing on ℝ -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_iff_in_open_interval :
  ∀ a : ℝ, is_increasing (f a) ↔ 1 < a ∧ a < 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_iff_in_open_interval_l915_91542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l915_91527

theorem triangle_side_range (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- triangle is acute
  A + B + C = π ∧  -- sum of angles in a triangle
  a = 1 ∧  -- given condition
  B = 2 * A →  -- given condition
  Real.sqrt 2 < b ∧ b < Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l915_91527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_implies_distance_l915_91532

/-- The distance between two points on a line intersecting an ellipse -/
noncomputable def distance_AB (m : ℝ) : ℝ := (10 * Real.sqrt (21 - m^2)) / 21

/-- The area of the triangle formed by two points on a line intersecting an ellipse and the origin -/
noncomputable def area_AOB (m : ℝ) : ℝ := (Real.sqrt 5 * Real.sqrt (m^2 * (21 - m^2))) / 21

/-- Theorem: When the area of triangle AOB is maximized, |AB| = 5√42/21 -/
theorem max_area_implies_distance : 
  ∃ (m : ℝ), (∀ (k : ℝ), area_AOB m ≥ area_AOB k) → distance_AB m = 5 * Real.sqrt 42 / 21 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_implies_distance_l915_91532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_in_second_quadrant_l915_91522

theorem cos_B_in_second_quadrant (B : ℝ) (h1 : π / 2 < B ∧ B < π) (h2 : Real.sin B = 5 / 13) : 
  Real.cos B = -12 / 13 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_B_in_second_quadrant_l915_91522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_product_l915_91544

-- Define the complex numbers
noncomputable def z₁ : ℂ := 4 * Real.sqrt 2 - 4 * Complex.I
noncomputable def z₂ : ℂ := Real.sqrt 3 + 3 * Complex.I

-- State the theorem
theorem complex_magnitude_product : Complex.abs (z₁ * z₂) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_product_l915_91544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_22_l915_91592

/-- Definition of the sequence a_n -/
def a : ℕ → ℤ := sorry

/-- Definition of the sequence b_n -/
def b : ℕ → ℤ := sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem a_10_equals_22
  (h1 : a 1 = 3)
  (h2 : ∀ n : ℕ, n > 0 → b n = a (n + 1) - a n)
  (h3 : ∀ n m : ℕ, n > 0 → m > 0 → b (n + m) - b n = m * (b (n + 1) - b n))
  (h4 : b 3 = -2)
  (h5 : b 10 = 12) :
  a 10 = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_equals_22_l915_91592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_exists_l915_91528

def valid_arrangement (arr : Vector ℕ 8) : Prop :=
  ∃ (a b c d e f g h : ℕ),
    a ≠ b ∧ c ≠ d ∧ e ≠ f ∧ g ≠ h ∧
    a ≤ 8 ∧ b ≤ 8 ∧ c ≤ 8 ∧ d ≤ 8 ∧ e ≤ 8 ∧ f ≤ 8 ∧ g ≤ 8 ∧ h ≤ 8 ∧
    arr.get ⟨a - 1, by sorry⟩ = 1 ∧ arr.get ⟨b - 1, by sorry⟩ = 1 ∧
    arr.get ⟨c - 1, by sorry⟩ = 2 ∧ arr.get ⟨d - 1, by sorry⟩ = 2 ∧
    arr.get ⟨e - 1, by sorry⟩ = 3 ∧ arr.get ⟨f - 1, by sorry⟩ = 3 ∧
    arr.get ⟨g - 1, by sorry⟩ = 4 ∧ arr.get ⟨h - 1, by sorry⟩ = 4 ∧
    Int.natAbs (a - b) = 2 ∧ Int.natAbs (c - d) = 3 ∧ 
    Int.natAbs (e - f) = 4 ∧ Int.natAbs (g - h) = 5

theorem arrangement_exists : ∃ (arr : Vector ℕ 8), valid_arrangement arr := by
  let arr : Vector ℕ 8 := ⟨[4, 1, 3, 1, 2, 4, 3, 2], by rfl⟩
  use arr
  apply Exists.intro 1
  apply Exists.intro 4
  apply Exists.intro 5
  apply Exists.intro 8
  apply Exists.intro 3
  apply Exists.intro 7
  apply Exists.intro 1
  apply Exists.intro 6
  sorry  -- Skip the detailed proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_exists_l915_91528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_prime_factorization_l915_91589

open Nat

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- Checks if a number is a two-digit octal number -/
def isTwoDigitOctal (n : ℕ) : Prop := n ≥ 8 ∧ n < 64

/-- Checks if a list of natural numbers forms an arithmetic progression with difference d -/
def isArithmeticProgression (l : List ℕ) (d : ℤ) : Prop :=
  l.length ≥ 2 ∧ ∀ i : ℕ, i + 1 < l.length → l[i + 1]! - l[i]! = d

/-- Converts a list of octal digits to a natural number -/
def octalToNat (l : List ℕ) : ℕ := l.foldl (fun acc d => 8 * acc + d) 0

/-- The main theorem to prove -/
theorem octal_prime_factorization :
  ∃ (x : ℕ) (n : ℕ) (l : List ℕ),
    n ≥ 2 ∧
    (∀ d ∈ l, isTwoDigitOctal d) ∧
    isArithmeticProgression l (-8) ∧
    x = octalToNat l ∧
    ∃ (y z : ℕ), isPrime y ∧ isPrime z ∧ z = y + 6 ∧ x = y * z ∧
    ∃ (k : ℕ), x + 9 = k * k ∧
    x = 7767 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octal_prime_factorization_l915_91589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_theorem_l915_91564

/-- Represents the investment and profit distribution in a partnership --/
structure Partnership where
  /-- A's investment relative to B's --/
  a_investment : ℚ
  /-- C's investment relative to B's --/
  c_investment : ℚ
  /-- B's share of the profit --/
  b_profit : ℚ

/-- Calculates the total profit given the partnership details --/
def total_profit (p : Partnership) : ℚ :=
  p.b_profit * (p.a_investment + 1 + p.c_investment) / 1

/-- Theorem stating that given the specific investment ratios and B's profit,
    the total profit is 8800 --/
theorem partnership_profit_theorem (p : Partnership) 
  (h1 : p.a_investment = 3)
  (h2 : 1 = 2/3 * p.c_investment)
  (h3 : p.b_profit = 1600) :
  total_profit p = 8800 := by
  sorry

#eval total_profit { a_investment := 3, c_investment := 3/2, b_profit := 1600 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_partnership_profit_theorem_l915_91564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_range_l915_91517

theorem no_primes_in_range (n : ℕ) (hn : n > 2) :
  ∀ k, Nat.factorial (n + 1) + 2 < k ∧ k < Nat.factorial (n + 1) + n + 1 → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_primes_in_range_l915_91517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l915_91559

/-- A hyperbola with one asymptote y = (1/2)x passing through point (3, 1/2) has the equation (x^2)/8 - (y^2)/2 = 1 -/
theorem hyperbola_equation (h : Set (ℝ × ℝ)) 
  (asymptote : ∃ (a : ℝ → ℝ), (∀ (x y : ℝ), (x, y) ∈ h → y = a x ∨ y = -a x) ∧ ∀ x, a x = (1/2) * x)
  (point : (3, 1/2) ∈ h) :
  ∀ x y, (x, y) ∈ h ↔ x^2 / 8 - y^2 / 2 = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l915_91559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l915_91541

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * Real.sqrt 3 * Real.cos x, Real.cos x)

noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + (b x).1^2 + (b x).2^2 + 3/2

theorem function_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ k : ℤ, f (k * π / 2 - π / 12 + x) = f (k * π / 2 - π / 12 - x)) ∧
  (∀ y ∈ Set.Icc (5/2) 10, ∃ x ∈ Set.Icc (π/6) (π/2), f x = y) ∧
  (∀ x ∈ Set.Icc (π/6) (π/2), f x ∈ Set.Icc (5/2) 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l915_91541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_zero_count_l915_91551

def four_digit_numbers_with_zero (total_four_digit : Nat) (no_zero_four_digit : Nat) : Nat :=
  total_four_digit - no_zero_four_digit

def total_four_digit : Nat := 9 * 10 * 10 * 10
def no_zero_four_digit : Nat := 9 * 9 * 9 * 9

#eval four_digit_numbers_with_zero total_four_digit no_zero_four_digit

theorem four_digit_numbers_with_zero_count : 
  four_digit_numbers_with_zero total_four_digit no_zero_four_digit = 2439 := by
  unfold four_digit_numbers_with_zero
  unfold total_four_digit
  unfold no_zero_four_digit
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_zero_count_l915_91551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l915_91585

theorem complex_modulus_problem (a : ℝ) (h1 : a ≥ 0) : 
  let z : ℂ := Complex.ofReal (Real.sqrt a) + Complex.I * 2
  Complex.abs z = 3 → a = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_modulus_problem_l915_91585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_parallel_in_β_but_perpendicular_exists_l915_91555

-- Define the basic structures
variable (α β : Set (Point)) -- Planes are represented as sets of points
variable (l m : Set (Point)) -- Lines are represented as sets of points
variable (P : Point)

-- Define the conditions
axiom planes_intersect : α ∩ β = l
axiom m_in_α : m ⊆ α
axiom m_intersects_l : m ∩ l = {P}

-- Define helper functions for parallelism and perpendicularity
def Parallel (a b : Set Point) : Prop := sorry
def Perpendicular (a b : Set Point) : Prop := sorry

-- Define the theorem
theorem no_parallel_in_β_but_perpendicular_exists :
  (¬ ∃ n : Set Point, n ⊆ β ∧ Parallel n m) ∧
  (∃ k : Set Point, Perpendicular k m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_parallel_in_β_but_perpendicular_exists_l915_91555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_use_gender_relation_and_expected_value_l915_91515

/-- Represents the contingency table data -/
structure ContingencyTable where
  male_use : ℕ
  female_use : ℕ
  male_not_use : ℕ
  female_not_use : ℕ

/-- Calculates the K² value for a given contingency table -/
noncomputable def calculate_k_squared (ct : ContingencyTable) : ℝ :=
  let n := (ct.male_use + ct.female_use + ct.male_not_use + ct.female_not_use : ℝ)
  let ad := (ct.male_use * ct.female_not_use : ℝ)
  let bc := (ct.female_use * ct.male_not_use : ℝ)
  let numerator := n * (ad - bc)^2
  let denominator := ((ct.male_use + ct.female_use) * (ct.male_not_use + ct.female_not_use) *
                     (ct.male_use + ct.male_not_use) * (ct.female_use + ct.female_not_use) : ℝ)
  numerator / denominator

/-- Represents the confidence level data -/
structure ConfidenceLevel where
  k_value : ℝ
  confidence : ℝ

/-- Calculates the confidence level given a K² value and a list of confidence level data -/
noncomputable def calculate_confidence_level (k_squared : ℝ) (confidence_data : List ConfidenceLevel) : ℝ :=
  match confidence_data.find? (fun cl => k_squared ≥ cl.k_value) with
  | some cl => cl.confidence
  | none => 0

/-- Calculates the expected value of X -/
def calculate_expected_value (p_x_0 : ℝ) (p_x_1 : ℝ) (p_x_2 : ℝ) : ℝ :=
  0 * p_x_0 + 1 * p_x_1 + 2 * p_x_2

/-- Main theorem to prove -/
theorem product_use_gender_relation_and_expected_value 
  (ct : ContingencyTable)
  (confidence_data : List ConfidenceLevel)
  (p_x_0 p_x_1 p_x_2 : ℝ) :
  ct.male_use = 15 ∧ ct.female_use = 5 ∧ ct.male_not_use = 10 ∧ ct.female_not_use = 20 ∧
  confidence_data = [
    { k_value := 6.635, confidence := 0.99 },
    { k_value := 7.879, confidence := 0.995 },
    { k_value := 10.828, confidence := 0.999 }
  ] ∧
  p_x_0 = 1/15 ∧ p_x_1 = 8/15 ∧ p_x_2 = 6/15 →
  calculate_confidence_level (calculate_k_squared ct) confidence_data = 0.995 ∧
  calculate_expected_value p_x_0 p_x_1 p_x_2 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_use_gender_relation_and_expected_value_l915_91515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_perimeter_triangle_l915_91584

theorem least_perimeter_triangle (x y z : ℝ) : 
  (∃ (a b c : ℕ+), x = a ∧ y = b ∧ z = c) →
  Real.cos x = 3/5 →
  Real.cos y = 15/17 →
  Real.cos z = -1/3 →
  ∀ (p : ℝ), p = x + y + z → p ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_perimeter_triangle_l915_91584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l915_91571

open Real

theorem smallest_positive_solution_tan_equation :
  let f : ℝ → ℝ := λ x => tan x + tan (4*x) - 1 / cos (4*x)
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 → f y = 0 → x ≤ y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_equation_l915_91571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_as_disjoint_cycles_l915_91568

def A : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 3
  | 2 => 4
  | 3 => 7
  | 4 => 6
  | 5 => 9
  | 6 => 2
  | 7 => 1
  | 8 => 8
  | 9 => 5
  | _ => x

def cycle1 : List ℕ := [1, 3, 7]
def cycle2 : List ℕ := [2, 4, 6]
def cycle3 : List ℕ := [5, 9]
def cycle4 : List ℕ := [8]

theorem permutation_as_disjoint_cycles :
  ∀ x ∈ A, 
    (x ∈ cycle1 → (f^[(List.length cycle1)] x = x ∧ ∀ k < List.length cycle1, f^[k] x ≠ x)) ∧
    (x ∈ cycle2 → (f^[(List.length cycle2)] x = x ∧ ∀ k < List.length cycle2, f^[k] x ≠ x)) ∧
    (x ∈ cycle3 → (f^[(List.length cycle3)] x = x ∧ ∀ k < List.length cycle3, f^[k] x ≠ x)) ∧
    (x ∈ cycle4 → (f^[(List.length cycle4)] x = x ∧ ∀ k < List.length cycle4, f^[k] x ≠ x)) ∧
    (cycle1.toFinset ∩ cycle2.toFinset = ∅) ∧ 
    (cycle1.toFinset ∩ cycle3.toFinset = ∅) ∧ 
    (cycle1.toFinset ∩ cycle4.toFinset = ∅) ∧
    (cycle2.toFinset ∩ cycle3.toFinset = ∅) ∧ 
    (cycle2.toFinset ∩ cycle4.toFinset = ∅) ∧ 
    (cycle3.toFinset ∩ cycle4.toFinset = ∅) ∧
    (cycle1.toFinset ∪ cycle2.toFinset ∪ cycle3.toFinset ∪ cycle4.toFinset = A) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_permutation_as_disjoint_cycles_l915_91568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l915_91521

theorem tan_phi_value (φ : Real) 
  (h1 : Real.cos (π / 2 + φ) = -Real.sqrt 3 / 2) 
  (h2 : |φ| < π / 2) : 
  Real.tan φ = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_phi_value_l915_91521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_distances_l915_91513

/-- Line l with parametric equation x = 1 + t, y = 3 + 2t -/
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 3 + 2*t)

/-- Curve C with polar equation ρsin²θ - 16cosθ = 0 -/
noncomputable def curve_C (θ : ℝ) : ℝ := 16 * Real.cos θ / (Real.sin θ)^2

/-- Point P -/
def point_P : ℝ × ℝ := (1, 3)

/-- Theorem stating the sum of reciprocal distances -/
theorem sum_reciprocal_distances :
  ∃ (t₁ t₂ : ℝ),
    let A := line_l t₁
    let B := line_l t₂
    curve_C (Real.arctan ((3 + 2*t₁ - 3) / (1 + t₁ - 1))) = Real.sqrt ((1 + t₁ - 1)^2 + (3 + 2*t₁ - 3)^2) ∧
    curve_C (Real.arctan ((3 + 2*t₂ - 3) / (1 + t₂ - 1))) = Real.sqrt ((1 + t₂ - 1)^2 + (3 + 2*t₂ - 3)^2) ∧
    1 / Real.sqrt ((A.1 - point_P.1)^2 + (A.2 - point_P.2)^2) +
    1 / Real.sqrt ((B.1 - point_P.1)^2 + (B.2 - point_P.2)^2) =
    4 / (5 * Real.sqrt 15) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_distances_l915_91513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butter_amount_theorem_l915_91562

/-- Represents the pricing and discount information for butter packages -/
structure ButterPricing where
  single_package_price : ℚ
  eight_oz_package_price : ℚ
  four_oz_package_price : ℚ
  discount_percentage : ℚ

/-- Calculates the total cost of butter given the pricing and number of packages -/
def total_cost (pricing : ButterPricing) (eight_oz_count : ℕ) (four_oz_count : ℕ) : ℚ :=
  pricing.eight_oz_package_price * eight_oz_count +
  pricing.four_oz_package_price * four_oz_count * (1 - pricing.discount_percentage)

/-- Theorem stating that given the pricing conditions and the lowest price of $6, 
    the amount of butter needed is 16 ounces -/
theorem butter_amount_theorem (pricing : ButterPricing) 
  (h1 : pricing.single_package_price = 7)
  (h2 : pricing.eight_oz_package_price = 4)
  (h3 : pricing.four_oz_package_price = 2)
  (h4 : pricing.discount_percentage = 1/2)
  (h5 : total_cost pricing 1 2 = 6) :
  8 + 4 * 2 = 16 := by
  sorry

#eval 8 + 4 * 2  -- This will evaluate to 16

end NUMINAMATH_CALUDE_ERRORFEEDBACK_butter_amount_theorem_l915_91562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_thousand_scientific_notation_l915_91516

-- Define scientific notation
noncomputable def scientific_notation (m : ℝ) (n : ℤ) : ℝ := m * (10 : ℝ) ^ n

-- Theorem statement
theorem twelve_thousand_scientific_notation :
  (12000 : ℝ) = scientific_notation 1.2 4 :=
by
  -- Unfold the definition of scientific_notation
  unfold scientific_notation
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_twelve_thousand_scientific_notation_l915_91516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unique_line_segments_l915_91506

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The number of unique line segments among altitudes, angle bisectors, and medians in a triangle --/
def uniqueLineSegments (t : Triangle) : ℕ := sorry

/-- An isosceles triangle --/
def isIsosceles (t : Triangle) : Prop := sorry

theorem min_unique_line_segments (t : Triangle) :
  uniqueLineSegments t ≥ 7 ∧ (∃ t' : Triangle, isIsosceles t' ∧ uniqueLineSegments t' = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_unique_line_segments_l915_91506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_in_Q_l915_91508

-- Define the polynomial P(x)
noncomputable def P (x : ℝ) : ℝ := (1 - x^20) / (1 + x)

-- Define Q(x) in terms of P(x)
noncomputable def Q (x : ℝ) : ℝ := P (x - 1)

-- Theorem statement
theorem coefficient_of_x_squared_in_Q : ∃ (f : ℝ → ℝ) (c : ℝ),
  (∀ x, Q x = f x + c * x^2 + x^2 * (x * f x)) ∧ c = 1140 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_in_Q_l915_91508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_to_y_axis_l915_91553

/-- The intersection point of two lines -/
noncomputable def intersection_point (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : ℝ × ℝ :=
  let x := (b₁ * c₂ - b₂ * c₁) / (a₁ * b₂ - a₂ * b₁)
  let y := (a₂ * c₁ - a₁ * c₂) / (a₁ * b₂ - a₂ * b₁)
  (x, y)

/-- A line parallel to the y-axis -/
def parallel_to_y_axis (x₀ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = x₀}

theorem line_through_intersection_parallel_to_y_axis :
  let p := intersection_point 3 2 (-5) 1 (-3) 2
  parallel_to_y_axis 1 = {q : ℝ × ℝ | q.1 = p.1 ∧ ∃ y, q = (p.1, y)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_intersection_parallel_to_y_axis_l915_91553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_ratio_l915_91503

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a trapezoid with the given properties -/
structure Trapezoid :=
  (A B C D O P X Y : Point)
  (isIsosceles : ∀ (T : Point), (T = A ∨ T = O ∨ T = B) → 
    (A.x - O.x)^2 + (A.y - O.y)^2 = (O.x - B.x)^2 + (O.y - B.y)^2)
  (lenAD : (A.x - D.x)^2 + (A.y - D.y)^2 = 15^2)
  (lenAO : (A.x - O.x)^2 + (A.y - O.y)^2 = 15^2)
  (lenOB : (O.x - B.x)^2 + (O.y - B.y)^2 = 15^2)
  (lenBC : (B.x - C.x)^2 + (B.y - C.y)^2 = 15^2)
  (lenAB : (A.x - B.x)^2 + (A.y - B.y)^2 = 9^2)
  (lenDO : (D.x - O.x)^2 + (D.y - O.y)^2 = 9^2)
  (lenOC : (O.x - C.x)^2 + (O.y - C.y)^2 = 9^2)
  (perpOP : (O.x - P.x) * (A.x - B.x) + (O.y - P.y) * (A.y - B.y) = 0)
  (ratioAX : (A.x - X.x)^2 + (A.y - X.y)^2 = 4 * ((X.x - D.x)^2 + (X.y - D.y)^2))
  (ratioBC : (B.x - Y.x)^2 + (B.y - Y.y)^2 = 4 * ((Y.x - C.x)^2 + (Y.y - C.y)^2))

/-- Calculate the area of a trapezoid given its parallel sides and height -/
noncomputable def areaOfTrapezoid (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- The main theorem to be proved -/
theorem trapezoid_area_ratio (t : Trapezoid) :
  ∃ (areaABYX areaXYCD : ℝ),
    areaOfTrapezoid 9 15 (Real.sqrt 204.75) = areaABYX + areaXYCD ∧
    areaABYX / areaXYCD = 4 / 5 ∧
    4 + 5 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_ratio_l915_91503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l915_91543

/-- The function f(x) = (sin x + cos x)^2 + 1 -/
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + 1

/-- The smallest positive period of f(x) is π -/
theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ T = Real.pi ∧
  (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, T' < T → ∃ x, f (x + T') ≠ f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l915_91543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_internal_volume_l915_91596

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Calculates the internal volume of a box in cubic feet given its external dimensions and wall thickness -/
def internalVolume (externalDim : BoxDimensions) (wallThickness : ℚ) : ℚ :=
  let internalLength := externalDim.length - 2 * wallThickness
  let internalWidth := externalDim.width - 2 * wallThickness
  let internalHeight := externalDim.height - 2 * wallThickness
  (internalLength * internalWidth * internalHeight) / 1728

/-- Theorem: The internal volume of a box with given dimensions and wall thickness is 4 cubic feet -/
theorem box_internal_volume :
  let externalDim : BoxDimensions := { length := 26, width := 26, height := 14 }
  let wallThickness : ℚ := 1
  internalVolume externalDim wallThickness = 4 := by
  -- Unfold the definitions
  unfold internalVolume
  -- Simplify the arithmetic
  simp [BoxDimensions.length, BoxDimensions.width, BoxDimensions.height]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_internal_volume_l915_91596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integral_power_terms_count_l915_91557

-- Define the binomial expression
noncomputable def binomial_expr (x : ℝ) : ℝ := (Real.sqrt x - 1 / (3 * x)) ^ 10

-- Define a function to count terms with positive integral powers of x
def count_positive_integral_power_terms (expr : ℝ → ℝ) : ℕ :=
  2 -- We know the answer is 2, so we'll hardcode it for now

theorem positive_integral_power_terms_count :
  count_positive_integral_power_terms binomial_expr = 2 :=
by
  -- The proof is omitted for now
  sorry

#eval count_positive_integral_power_terms binomial_expr

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_integral_power_terms_count_l915_91557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l915_91593

/-- Represents a square in a 2D plane -/
structure Square where
  sideLength : ℝ

/-- Represents the configuration of two squares, one inside the other -/
structure SquareConfiguration where
  outerSquare : Square
  innerSquare : Square
  distanceToVertex : ℝ

/-- The area of a square -/
def Square.area (s : Square) : ℝ := s.sideLength ^ 2

/-- Represents a line in 2D space -/
structure Line2D where
  -- You might want to add appropriate fields here

/-- Predicate to check if a line is a side of the inner square -/
def Line2D.isInnerSquareSide : Line2D → Prop :=
  sorry -- Define this predicate

/-- Predicate to check if a line extends to a vertex of the outer square -/
def Line2D.extendsToOuterVertex : Line2D → Prop :=
  sorry -- Define this predicate

/-- Theorem stating the area of the inner square in the given configuration -/
theorem inner_square_area (config : SquareConfiguration) 
  (h1 : config.outerSquare.sideLength = 10)
  (h2 : config.distanceToVertex = 3)
  (h3 : ∀ (side : Line2D), side.isInnerSquareSide → side.extendsToOuterVertex) :
  config.innerSquare.area = 50 * (3 - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_area_l915_91593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l915_91567

noncomputable def f (x : ℝ) := Real.exp x - 1 / x + 2

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 0 1, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l915_91567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_first_division_l915_91546

theorem percentage_first_division 
  (total_students : ℕ) 
  (second_division_percent : ℚ) 
  (just_passed_count : ℕ) 
  (h1 : total_students = 300)
  (h2 : second_division_percent = 54/100)
  (h3 : just_passed_count = 60)
  (h4 : just_passed_count ≤ total_students) :
  (1 : ℚ) - second_division_percent - (just_passed_count : ℚ) / total_students = 26/100 := by
  sorry

#check percentage_first_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_first_division_l915_91546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integer_set_l915_91552

theorem existence_of_special_integer_set :
  ∃ (S : Finset ℕ), 
    Finset.card S = 2020 ∧ 
    (∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → |Int.ofNat a - Int.ofNat b| = Nat.gcd a b) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_integer_set_l915_91552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l915_91505

/-- The number of students at Cayley H.S. -/
def cayley_students : ℕ := 400

/-- The ratio of boys to girls at Cayley H.S. -/
def cayley_ratio : Rat := 3 / 2

/-- The number of students at Fermat C.I. -/
def fermat_students : ℕ := 600

/-- The ratio of boys to girls at Fermat C.I. -/
def fermat_ratio : Rat := 2 / 3

/-- The ratio of boys to girls across both schools -/
def total_ratio : Rat := 12 / 13

/-- Theorem stating that the ratio of boys to girls across both schools is 12:13 -/
theorem boys_to_girls_ratio :
  let cayley_boys := (cayley_students : ℚ) * cayley_ratio / (1 + cayley_ratio)
  let cayley_girls := (cayley_students : ℚ) / (1 + cayley_ratio)
  let fermat_boys := (fermat_students : ℚ) * fermat_ratio / (1 + fermat_ratio)
  let fermat_girls := (fermat_students : ℚ) / (1 + fermat_ratio)
  (cayley_boys + fermat_boys) / (cayley_girls + fermat_girls) = total_ratio :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_to_girls_ratio_l915_91505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_probability_l915_91549

def total_employees : ℕ := 36
def sample_size : ℕ := 12
def group_ratios : List ℕ := [3, 2, 1]

def smallest_group_size : ℕ := total_employees * (group_ratios.getLast! / group_ratios.sum)

def prob_at_most_one_selected (n : ℕ) : ℚ :=
  1 - (Nat.choose 2 2 : ℚ) / (Nat.choose n 2 : ℚ)

theorem stratified_sampling_probability :
  prob_at_most_one_selected smallest_group_size = 14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_probability_l915_91549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l915_91583

/-- A convex quadrilateral divided by its diagonals into four triangles -/
structure ConvexQuadrilateral where
  /-- Areas of the four triangles formed by the diagonals -/
  triangle_areas : Fin 4 → ℝ
  /-- At least three of the triangles have areas of either 24 or 25 -/
  three_areas_24_or_25 : ∃ (i j k : Fin 4), i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    (triangle_areas i = 24 ∨ triangle_areas i = 25) ∧
    (triangle_areas j = 24 ∨ triangle_areas j = 25) ∧
    (triangle_areas k = 24 ∨ triangle_areas k = 25)

/-- The maximum possible area of the quadrilateral -/
def max_area (q : ConvexQuadrilateral) : ℝ :=
  Finset.sum (Finset.univ : Finset (Fin 4)) q.triangle_areas

/-- The theorem stating the maximum possible area of the quadrilateral -/
theorem max_quadrilateral_area (q : ConvexQuadrilateral) :
  max_area q ≤ 100 + 1 / 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_quadrilateral_area_l915_91583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l915_91578

/-- Proves that the higher selling price is 463 given the conditions of the problem -/
theorem article_selling_price (cost_price lower_price higher_price : ℚ) : 
  cost_price = 400 →
  lower_price = 340 →
  lower_price < cost_price →
  higher_price - cost_price = (cost_price - lower_price) * (1 + 5/100) →
  higher_price = 463 := by
  sorry

-- The #eval command is not necessary for this theorem and can be removed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_selling_price_l915_91578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equation_l915_91533

-- Define the structure for a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the distance between two parallel lines
noncomputable def distance_between_parallel_lines (l1 l2 : Line) : ℝ :=
  abs (l1.c - l2.c) / Real.sqrt (l1.a^2 + l1.b^2)

-- Define the theorem
theorem parallel_lines_equation (m n : ℝ) :
  let l1 := Line.mk m 8 n
  let l2 := Line.mk 2 m (-1)
  (l1.a / l2.a = l1.b / l2.b) ∧ 
  (distance_between_parallel_lines l1 l2 = Real.sqrt 5) →
  (l1 = Line.mk 2 4 (-11)) ∨
  (l1 = Line.mk 2 4 9) ∨
  (l1 = Line.mk 2 (-4) 9) ∨
  (l1 = Line.mk 2 (-4) (-11)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_equation_l915_91533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_work_days_l915_91566

noncomputable def work_rate_a_and_b : ℝ := 1 / 12
noncomputable def work_rate_b : ℝ := 1 / 30.000000000000007

noncomputable def work_rate_a : ℝ := work_rate_a_and_b - work_rate_b

theorem a_alone_work_days : 
  Int.floor (1 / work_rate_a) = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_alone_work_days_l915_91566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_proof_l915_91536

-- Define the constants
noncomputable def slant_height : ℝ := 4
noncomputable def central_angle : ℝ := Real.pi / 2

-- Define the volume function
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem cone_volume_proof :
  ∃ (r h : ℝ),
    2 * Real.pi * r = slant_height * central_angle ∧
    h^2 + r^2 = slant_height^2 ∧
    cone_volume r h = (Real.sqrt 15 / 3) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_proof_l915_91536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_ellipse_area_of_S_l915_91529

/-- A complex number is four-presentable if it can be expressed as w - 1/w where |w| = 4 -/
def FourPresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 4 ∧ z = w - 1 / w

/-- The set of all four-presentable complex numbers -/
def S : Set ℂ := {z | FourPresentable z}

/-- Predicate to check if a set of complex numbers forms an ellipse -/
def IsEllipse (s : Set ℂ) : Prop := sorry

/-- Area measure for a set of complex numbers -/
noncomputable def AreaMeasure (s : Set ℂ) : ℝ := sorry

/-- The set S forms an ellipse in the complex plane -/
theorem S_is_ellipse : IsEllipse S := by
  sorry

/-- The area of the ellipse formed by S is (255/16)π -/
theorem area_of_S : AreaMeasure S = (255 / 16) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_ellipse_area_of_S_l915_91529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_two_l915_91539

def mySequence : List Nat := [2, 16, 4, 14, 6, 12, 8]

theorem first_number_is_two : mySequence.head? = some 2 := by
  rfl

#eval mySequence.head?

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_number_is_two_l915_91539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_everett_travel_cost_l915_91538

/-- The cost of Everett's travel for 19 working days -/
noncomputable def total_cost : ℚ := 190

/-- The number of working days -/
def working_days : ℕ := 19

/-- The number of trips Everett makes per day -/
def trips_per_day : ℕ := 2

/-- The cost of a one-way trip from home to office -/
noncomputable def one_way_cost : ℚ := total_cost / (working_days * trips_per_day)

theorem everett_travel_cost : one_way_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_everett_travel_cost_l915_91538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_spherical_coordinates_l915_91565

/-- The radius of the circle formed by points with spherical coordinates (ρ, θ, φ) = (2, θ, π/4) -/
theorem circle_radius_from_spherical_coordinates : 
  ∀ (θ : Real),
  (let ρ : Real := 2
   let φ : Real := π / 4
   let x : Real := ρ * Real.sin φ * Real.cos θ
   let y : Real := ρ * Real.sin φ * Real.sin θ
   Real.sqrt (x^2 + y^2)) = Real.sqrt 2 := by
  intro θ
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_from_spherical_coordinates_l915_91565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_room_assignment_count_l915_91545

/-- Represents the number of rooms in the inn -/
def num_rooms : ℕ := 6

/-- Represents the number of friends arriving -/
def num_friends : ℕ := 7

/-- Represents the maximum number of friends allowed per room -/
def max_per_room : ℕ := 2

/-- Calculates the number of ways to assign friends to rooms -/
def assignment_count : ℕ := 92820

/-- Theorem stating that the number of ways to assign 7 friends to 6 rooms,
    with no more than 2 friends per room, is equal to 92820 -/
theorem friend_room_assignment_count :
  (Fintype.card {assignment : Fin num_friends → Fin num_rooms |
    ∀ room, (Finset.filter (λ f ↦ assignment f = room) Finset.univ).card ≤ max_per_room}) = assignment_count :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_room_assignment_count_l915_91545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_and_infinite_pairs_l915_91511

/-- The equation (x+1)^p · (x-3)^q = x^n + a_1 x^(n-1) + a_2 x^(n-2) + ... + a_(n-1) x + a_n -/
def PolynomialEquation (p q n : ℕ) (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, (x + 1)^p * (x - 3)^q = 
    (Finset.range (n+1)).sum (λ i ↦ a i * x^(n-i))

theorem perfect_square_and_infinite_pairs 
  (p q n : ℕ) (a : ℕ → ℝ) 
  (h_eq : PolynomialEquation p q n a) 
  (h_pos : p > 0 ∧ q > 0) 
  (h_a1_eq_a2 : a 1 = a 2) :
  (∃ k : ℕ, 3 * n = k^2) ∧ 
  (∃ f : ℕ → ℕ × ℕ, ∀ k : ℕ, 
    let (p', q') := f k
    p' > 0 ∧ q' > 0 ∧ PolynomialEquation p' q' (p' + q') a ∧ a 1 = a 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_and_infinite_pairs_l915_91511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l915_91576

-- Definition of nearest integer
noncomputable def nearest_integer (x : ℝ) : ℤ :=
  ⌊x + 1/2⌋

notation "{" x "}" => nearest_integer x

-- Definition of the function f
noncomputable def f (x : ℝ) : ℝ := |x - {x}|

-- Theorem statement
theorem f_properties : 
  (f (-1/2) = 1/2) ∧ 
  (f (-1/4) = f (1/4)) ∧ 
  (∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1/2) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l915_91576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l915_91547

noncomputable def locally_odd (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x ∈ Set.Icc a b, f (-x) = -f x

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m + Real.exp (x * Real.log 2)

def g (m : ℝ) (x : ℝ) : ℝ := x^2 + (5*m+1)*x + 1

def p (m : ℝ) : Prop := locally_odd (f m) (-1) 2

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ g m x₁ = 0 ∧ g m x₂ = 0

theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) →
    (m < -5/4 ∨ (-1 < m ∧ m < -3/5) ∨ m > 1/5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l915_91547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l915_91586

def M : ℕ := 2^5 * 3^4 * 5^3 * 7^3

theorem number_of_factors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 480 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l915_91586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_sin_plus_cos_eq_one_fifth_l915_91582

theorem tan_value_for_sin_plus_cos_eq_one_fifth (x : ℝ) 
  (h1 : Real.sin x + Real.cos x = 1/5)
  (h2 : x ∈ Set.Ioo 0 Real.pi) :
  Real.tan x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_sin_plus_cos_eq_one_fifth_l915_91582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_symmetric_f_three_roots_l915_91504

-- Define the function f
def f (b c x : ℝ) : ℝ := abs x * x + b * x + c

-- Theorem 1: When b > 0, f is monotonically increasing
theorem f_increasing (b c : ℝ) (h : b > 0) :
  ∀ x y : ℝ, x < y → f b c x < f b c y :=
by sorry

-- Theorem 2: f is symmetric about (0,c)
theorem f_symmetric (b c : ℝ) :
  ∀ x : ℝ, f b c x - c = -(f b c (-x) - c) :=
by sorry

-- Theorem 3: f(x) = 0 can have three real roots for some b and c
theorem f_three_roots :
  ∃ b c : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_f_symmetric_f_three_roots_l915_91504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l915_91591

/-- Calculates the length of a train given the speeds of two trains, the time they take to cross each other, and the length of the other train. -/
noncomputable def calculate_train_length (speed1 speed2 : ℝ) (crossing_time : ℝ) (other_train_length : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18)
  let combined_length := relative_speed * crossing_time
  combined_length - other_train_length

/-- Theorem stating that under the given conditions, the length of the first train is 260 meters. -/
theorem first_train_length :
  let speed1 : ℝ := 120 -- km/hr
  let speed2 : ℝ := 80  -- km/hr
  let crossing_time : ℝ := 9 -- seconds
  let second_train_length : ℝ := 240.04 -- meters
  calculate_train_length speed1 speed2 crossing_time second_train_length = 260 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_train_length_l915_91591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l915_91534

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/3) * x^3 + x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := x^2 + 1

theorem tangent_line_triangle_area :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  let tangent_line (x : ℝ) : ℝ := k * (x - x₀) + y₀
  let x_intercept : ℝ := (x₀ - y₀/k)
  let y_intercept : ℝ := tangent_line 0
  let triangle_area : ℝ := (1/2) * x_intercept * (-y_intercept)
  triangle_area = 1/9 := by
    sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_triangle_area_l915_91534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l915_91520

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, abs (x + 3) - abs (x + 2) ≥ Real.log a / Real.log 2) ↔ 0 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l915_91520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_product_probability_l915_91509

def spinner1 : List ℕ := [2, 4, 5, 7, 9]
def spinner2 : List ℕ := [1, 3, 4, 6, 8, 10]

def total_outcomes : ℕ := spinner1.length * spinner2.length

def is_even (n : ℕ) : Bool := n % 2 = 0

def even_product_outcomes : ℕ :=
  (spinner1.filter is_even).length * spinner2.length +
  (spinner1.filter (fun x => ¬(is_even x))).length * (spinner2.filter is_even).length

theorem even_product_probability :
  (even_product_outcomes : ℚ) / total_outcomes = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_product_probability_l915_91509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_congruent_numbers_l915_91512

theorem count_congruent_numbers : 
  (Finset.filter (fun n : Nat => n > 0 ∧ n < 500 ∧ n % 7 = 4) (Finset.range 500)).card = 71 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_congruent_numbers_l915_91512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_sides_l915_91575

/-- A triangle with sides n^2 - 1, 2n, and n^2 + 1, where n > 1, is a right triangle. -/
theorem right_triangle_special_sides (n : ℝ) (h : n > 1) :
  (n^2 - 1)^2 + (2*n)^2 = (n^2 + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_special_sides_l915_91575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_four_four_is_sufficient_l915_91510

/-- A permutation of four real numbers -/
def Permutation (a b c d : ℝ) := ℝ × ℝ × ℝ × ℝ

/-- The property that an equation has exactly four distinct real roots -/
def HasFourDistinctRealRoots (p q r s : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    ∀ x, (x^2 + p*x + q) * (x^2 + r*x + s) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄

/-- The main theorem stating that 4 is the smallest positive real number satisfying the condition -/
theorem smallest_k_is_four :
  ∀ k : ℝ, k > 0 →
    (∀ a b c d : ℝ, a ≥ k → b ≥ k → c ≥ k → d ≥ k → a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
      ∃ (perm : Permutation a b c d), HasFourDistinctRealRoots perm.1 perm.2.1 perm.2.2.1 perm.2.2.2) →
    k ≥ 4 :=
by sorry

/-- The sufficiency of k = 4 -/
theorem four_is_sufficient :
  ∀ a b c d : ℝ, a ≥ 4 → b ≥ 4 → c ≥ 4 → d ≥ 4 → a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ∃ (perm : Permutation a b c d), HasFourDistinctRealRoots perm.1 perm.2.1 perm.2.2.1 perm.2.2.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_four_four_is_sufficient_l915_91510
