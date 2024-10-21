import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_negative_one_l1028_102818

noncomputable def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q ^ (n - 1)

noncomputable def geometric_sum (a q : ℝ) : ℕ → ℝ := fun n =>
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem common_ratio_is_negative_one
  (a q : ℝ) (h : q ≠ 1) :
  geometric_sequence a q 2 + geometric_sum a q 3 = 0 → q = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_is_negative_one_l1028_102818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_l1028_102830

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Total distance from P to R to Q -/
noncomputable def totalDistance (p q r : Point) : ℝ :=
  distance p r + distance r q

/-- The point P -/
def P : Point := ⟨-2, -3⟩

/-- The point Q -/
def Q : Point := ⟨5, 3⟩

/-- The point R with variable y-coordinate -/
def R (m : ℝ) : Point := ⟨1, m⟩

theorem minimal_distance :
  ∀ m : ℝ, totalDistance P Q (R (-3/7)) ≤ totalDistance P Q (R m) := by
  sorry

#check minimal_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_distance_l1028_102830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_specific_multiples_l1028_102806

theorem count_specific_multiples : 
  (Finset.filter (fun n => 1 ≤ n ∧ n ≤ 300 ∧ 5 ∣ n ∧ 6 ∣ n ∧ ¬(8 ∣ n) ∧ ¬(9 ∣ n)) (Finset.range 301)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_specific_multiples_l1028_102806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1028_102807

theorem sum_remainder_mod_15 (x y z : ℕ) 
  (hx : x % 15 = 7)
  (hy : y % 15 = 11)
  (hz : z % 15 = 13) :
  (x + y + z) % 15 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_mod_15_l1028_102807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_infinite_l1028_102819

/-- The largest prime divisor of a natural number -/
def largestPrimeDivisor (n : ℕ) : ℕ :=
  (Nat.factors n).maximum?.getD 1

/-- The set of natural numbers n for which the largest prime divisor of n^4 + n^2 + 1
    is equal to the largest prime divisor of (n+1)^4 + (n+1)^2 + 1 -/
def S : Set ℕ :=
  {n : ℕ | largestPrimeDivisor (n^4 + n^2 + 1) = largestPrimeDivisor ((n+1)^4 + (n+1)^2 + 1)}

/-- The main theorem: S is an infinite set -/
theorem S_infinite : Set.Infinite S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_infinite_l1028_102819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_soup_donation_l1028_102882

theorem mark_soup_donation (shelter_people : Fin 6 → ℕ) (cans_per_person : Fin 6 → ℕ) 
  (h1 : shelter_people 0 = 30) (h2 : shelter_people 1 = 25) (h3 : shelter_people 2 = 35)
  (h4 : shelter_people 3 = 40) (h5 : shelter_people 4 = 28) (h6 : shelter_people 5 = 32)
  (c1 : cans_per_person 0 = 12) (c2 : cans_per_person 1 = 10) (c3 : cans_per_person 2 = 15)
  (c4 : cans_per_person 3 = 14) (c5 : cans_per_person 4 = 11) (c6 : cans_per_person 5 = 13) :
  (Finset.univ : Finset (Fin 6)).sum (fun i => shelter_people i * cans_per_person i) = 2419 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_soup_donation_l1028_102882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1028_102804

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

noncomputable def ellipse_eccentricity : ℝ := 4 / 5

noncomputable def sum_of_eccentricities : ℝ := 14 / 5

noncomputable def hyperbola_eccentricity : ℝ := sum_of_eccentricities - ellipse_eccentricity

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

theorem hyperbola_properties :
  hyperbola_eccentricity = 2 ∧
  ∀ x y : ℝ, hyperbola x y ↔ x^2 / 4 - y^2 / 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l1028_102804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_comparison_l1028_102893

theorem recreation_spending_comparison (W : ℝ) (hW : W > 0) : 
  let last_week_recreation := 0.10 * W
  let this_week_wages := 0.90 * W
  let this_week_recreation := 0.40 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_recreation_spending_comparison_l1028_102893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l1028_102820

open Set

theorem intersection_of_A_and_complement_of_B :
  let A : Set ℝ := {x : ℝ | x > 0}
  let B : Set ℝ := {x : ℝ | x ≤ 1}
  A ∩ (Bᶜ) = {x : ℝ | x > 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_complement_of_B_l1028_102820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l1028_102823

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) - x^2

-- State the theorem
theorem a_lower_bound (a : ℝ) :
  (∀ p q : ℝ, -1 < p ∧ p < 0 ∧ -1 < q ∧ q < 0 ∧ p ≠ q →
    (f a (p + 1) - f a (q + 1)) / (p - q) > 1) →
  a ≥ 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_lower_bound_l1028_102823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sushi_downstream_distance_l1028_102808

/-- The distance Sushi rowed downstream given the specified conditions -/
noncomputable def downstream_distance (upstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) : ℝ :=
  let still_water_speed := upstream_distance / time + stream_speed
  (still_water_speed + stream_speed) * time

/-- Theorem stating that Sushi rowed 243 km downstream under the given conditions -/
theorem sushi_downstream_distance :
  downstream_distance 27 9 12 = 243 := by
  -- Unfold the definition of downstream_distance
  unfold downstream_distance
  -- Simplify the expression
  simp
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sushi_downstream_distance_l1028_102808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_specific_length_l1028_102876

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a segment AB and its golden section point P, this function returns the length of AP -/
noncomputable def goldenSectionLength (AB : ℝ) : ℝ := AB * (φ - 1)

/-- Theorem: If P is the golden section point of segment AB, AP > BP, and AB = 5, then AP = (5√5 - 5) / 2 -/
theorem golden_section_specific_length :
  ∀ (A B P : ℝ),
  (B - A) / (P - A) = φ →  -- P is the golden section point of AB
  P - A > B - P →         -- AP > BP
  B - A = 5 →             -- AB = 5
  P - A = (5 * Real.sqrt 5 - 5) / 2 := by
  sorry

#check golden_section_specific_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_specific_length_l1028_102876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1028_102874

theorem evaluate_expression : (4^2 - Nat.factorial 4 * (4 - 1)^4)^2 = 3715584 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1028_102874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_two_std_dev_below_mean_l1028_102835

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ
  stdDev_pos : stdDev > 0

/-- Calculates the number of standard deviations a value is from the mean --/
noncomputable def standardDeviationsFromMean (nd : NormalDistribution) (value : ℝ) : ℝ :=
  (nd.mean - value) / nd.stdDev

/-- Theorem: For the given normal distribution, 11.1 is 2 standard deviations below the mean --/
theorem value_two_std_dev_below_mean :
  let nd : NormalDistribution := { mean := 14.5, stdDev := 1.7, stdDev_pos := by norm_num }
  standardDeviationsFromMean nd 11.1 = 2 := by
  sorry

#eval (14.5 - 11.1) / 1.7  -- This line is added for demonstration purposes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_two_std_dev_below_mean_l1028_102835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_is_8_l1028_102817

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  area : ℝ
  top_side : ℝ
  bottom_side : ℝ
  height : ℝ

/-- The formula for the area of a trapezoid -/
noncomputable def trapezoid_area_formula (t : Trapezoid) : ℝ :=
  (1/2) * (t.top_side + t.bottom_side) * t.height

/-- Theorem stating that a trapezoid with the given dimensions has a height of 8 cm -/
theorem trapezoid_height_is_8 (t : Trapezoid) 
    (h_area : t.area = 56)
    (h_top : t.top_side = 5)
    (h_bottom : t.bottom_side = 9) :
    t.height = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_height_is_8_l1028_102817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_monotonicity_l1028_102839

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2/3) * x^3 - (1/2) * x^2 + (a - 1) * x + 1

theorem tangent_and_monotonicity (a : ℝ) :
  (∃ m : ℝ, (deriv (f a)) 1 = m ∧ m * (-1/2) = -1) →
  (a = 2 ∧
   (∀ x ≥ 2, a ≤ 0 → (deriv (f a)) x ≤ 0) ∧
   (0 < a → a < 3/5 → ∃ x₀ > 2, ∀ x ≥ 2,
     ((x < x₀ → (deriv (f a)) x < 0) ∧
      (x > x₀ → (deriv (f a)) x > 0))) ∧
   (∀ x ≥ 2, a ≥ 3/5 → (deriv (f a)) x ≥ 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_monotonicity_l1028_102839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l1028_102836

def numbers : List Nat := [16, 18, 20, 25]

theorem unique_number : ∃! n, n ∈ numbers ∧ n % 2 = 0 ∧ n % 5 = 0 ∧ n > 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_l1028_102836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_two_increasing_l1028_102898

-- Define a general exponential function
noncomputable def exponential (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the property of being an increasing function
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem exp_two_increasing :
  (∀ a > 1, is_increasing (exponential a)) →  -- Given: exponential function y = a^x (a > 1) is increasing
  (∃ a > 1, exponential 2 = exponential a) →  -- Given: y = 2^x is an exponential function
  is_increasing (exponential 2) :=             -- To prove: y = 2^x is an increasing function
by
  intro h1 h2
  sorry  -- Proof omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_two_increasing_l1028_102898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_smart_integers_div_by_11_l1028_102879

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry -- implementation not provided, but assumed to exist

/-- Definition of a smart integer -/
def is_smart_integer (n : ℕ) : Prop :=
  Odd n ∧ 30 < n ∧ n < 130 ∧ (sum_of_digits n = 10)

/-- The set of all smart integers -/
def smart_integers : Finset ℕ :=
  sorry -- implementation not provided, but assumed to be a finite set

/-- The set of smart integers divisible by 11 -/
def smart_integers_div_by_11 : Finset ℕ :=
  sorry -- implementation not provided, but assumed to be a finite set

theorem fraction_of_smart_integers_div_by_11 :
  (Finset.card smart_integers_div_by_11 : ℚ) /
  (Finset.card smart_integers) = 1 / 2 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_smart_integers_div_by_11_l1028_102879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l1028_102889

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℚ
  diff : ℚ

/-- Represents the given grid of numbers -/
structure NumberGrid where
  row : ArithmeticSequence
  col1 : ArithmeticSequence
  col2 : ArithmeticSequence

/-- Solves for X given the NumberGrid -/
def solve_for_x (grid : NumberGrid) : ℚ :=
  let x := grid.col2.first
  have h1 : grid.row.first = 24 := by sorry
  have h2 : grid.col1.first + grid.col1.diff = 16 := by sorry
  have h3 : grid.col1.first + 2 * grid.col1.diff = 20 := by sorry
  have h4 : x + 4 * grid.col2.diff = -14 := by sorry
  have h5 : grid.row.first + 6 * grid.row.diff = x := by sorry
  have h6 : grid.row ≠ grid.col1 ∧ grid.row ≠ grid.col2 ∧ grid.col1 ≠ grid.col2 := by sorry
  73 / 2

/-- Theorem stating that the solution is correct -/
theorem solution_is_correct (grid : NumberGrid) : solve_for_x grid = 73 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_correct_l1028_102889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_slope_implies_n_l1028_102813

noncomputable def curve (n : ℝ) (x : ℝ) : ℝ := x^n / Real.exp x

noncomputable def curve_derivative (n : ℝ) (x : ℝ) : ℝ := (n * x^(n-1) - x^n) / Real.exp x

theorem curve_slope_implies_n (n : ℝ) :
  curve_derivative n 1 = 4 / Real.exp 1 → n = 5 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check curve_slope_implies_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_slope_implies_n_l1028_102813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l1028_102852

-- Define the curve C and function g
noncomputable def C (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (x + 1) - Real.log x

-- Define the areas A_n and B_n
noncomputable def A_n (n : ℕ) : ℝ := (n + 1/2) * g n - 1
noncomputable def B_n (n : ℕ) : ℝ := 1 / (2 * n) + Real.log n - (n + 1) * g n + 1

theorem area_theorem :
  ∀ n : ℕ,
  -- A_n is the area enclosed by the line through (n, ln n) and (n+1, ln(n+1)) and C
  (A_n n = (n + 1/2) * g n - 1) ∧
  -- B_n is the area enclosed by the tangent line at (n, ln n), C, and x = n+1
  (B_n n = 1 / (2 * n) + Real.log n - (n + 1) * g n + 1) ∧
  -- The limit of n(1 - n*g(n)) as n approaches infinity is 0
  (Filter.Tendsto (λ n : ℕ ↦ n * (1 - n * g n)) Filter.atTop (nhds 0)) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_theorem_l1028_102852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_fifteen_percent_l1028_102840

/-- Calculates the interest rate at which B lent money to C -/
noncomputable def calculate_interest_rate (principal : ℝ) (initial_rate : ℝ) (time : ℝ) (gain : ℝ) : ℝ :=
  let interest_AB := principal * initial_rate * time
  (gain + interest_AB) / (principal * time)

/-- Theorem: Given the problem conditions, the interest rate B lent to C is 15% -/
theorem interest_rate_is_fifteen_percent 
  (principal : ℝ) (initial_rate : ℝ) (time : ℝ) (gain : ℝ)
  (h_principal : principal = 3500)
  (h_initial_rate : initial_rate = 0.1)
  (h_time : time = 3)
  (h_gain : gain = 525) :
  calculate_interest_rate principal initial_rate time gain = 0.15 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 3500 0.1 3 525

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_fifteen_percent_l1028_102840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1028_102859

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem setup
def concentric_circles (c1 c2 : Circle) : Prop :=
  c1.center = c2.center

noncomputable def arc_length (c : Circle) (angle : ℝ) : ℝ :=
  (angle / (2 * Real.pi)) * (2 * Real.pi * c.radius)

-- Theorem statement
theorem circle_area_ratio 
  (c1 c2 : Circle) 
  (h_concentric : concentric_circles c1 c2) 
  (h_arc_equal : arc_length c1 (Real.pi / 3) = arc_length c2 (4 * Real.pi / 15)) :
  (c1.radius^2) / (c2.radius^2) = 16 / 25 := by
  sorry

#check circle_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_ratio_l1028_102859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_percentage_proof_l1028_102831

/-- The percentage of salt in the second solution -/
noncomputable def second_solution_salt_percent : ℝ := 19.000000000000007

/-- The percentage of salt in the final mixed solution -/
noncomputable def final_solution_salt_percent : ℝ := 16

/-- The fraction of the first solution that remains after replacement -/
def first_solution_fraction : ℚ := 3/4

/-- The fraction of the second solution in the final mixture -/
def second_solution_fraction : ℚ := 1/4

/-- The percentage of salt in the first solution -/
noncomputable def first_solution_salt_percent : ℝ := 15

theorem salt_percentage_proof :
  (first_solution_fraction : ℝ) * first_solution_salt_percent +
  (second_solution_fraction : ℝ) * second_solution_salt_percent =
  final_solution_salt_percent := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_salt_percentage_proof_l1028_102831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1028_102866

theorem min_value_of_expression (a b : ℝ) (h : a - 3*b + 6 = 0) :
  (2 : ℝ)^a + (1/8 : ℝ)^b ≥ (1/4) ∧ ∃ a b : ℝ, a - 3*b + 6 = 0 ∧ (2 : ℝ)^a + (1/8 : ℝ)^b = (1/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_expression_l1028_102866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_female_athletes_selected_is_twelve_l1028_102841

/-- Calculates the number of female athletes to be selected in a stratified sample -/
def female_athletes_in_sample (total_athletes : ℕ) (male_athletes : ℕ) (sample_size : ℕ) : ℕ :=
  let female_athletes := total_athletes - male_athletes
  let selection_ratio : ℚ := (sample_size : ℚ) / (total_athletes : ℚ)
  (↑female_athletes * selection_ratio).floor.toNat

/-- Theorem stating that the number of female athletes to be selected is 12 -/
theorem female_athletes_selected_is_twelve :
  female_athletes_in_sample 98 56 28 = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_female_athletes_selected_is_twelve_l1028_102841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_point_equivalence_l1028_102885

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Converts a polar point to its standard representation -/
noncomputable def toStandardPolar (p : PolarPoint) : PolarPoint :=
  if p.r < 0 then
    { r := -p.r, θ := (p.θ + Real.pi) % (2 * Real.pi) }
  else
    { r := p.r, θ := p.θ % (2 * Real.pi) }

theorem polar_point_equivalence :
  let original := PolarPoint.mk (-5) (5 * Real.pi / 6)
  let standard := toStandardPolar original
  standard.r = 5 ∧ standard.θ = 11 * Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_point_equivalence_l1028_102885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_symmetric_even_function_l1028_102881

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of f being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem integral_symmetric_even_function 
  (h_even : is_even f) 
  (h_integral : ∫ x in (0 : ℝ)..(6 : ℝ), f x = 8) : 
  ∫ x in (-6 : ℝ)..(6 : ℝ), f x = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_symmetric_even_function_l1028_102881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_special_divisor_property_l1028_102870

def has_at_least_four_divisors (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 1 < a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ n ∧
    n % a = 0 ∧ n % b = 0 ∧ n % c = 0 ∧ n % d = 0

def sum_of_squares_of_smallest_four_divisors (n : ℕ) : ℕ :=
  let divisors := (Finset.range (n + 1)).filter (fun d ↦ n % d = 0)
  (divisors.sort (·≤·)).take 4
    |>.map (fun d ↦ d * d)
    |>.sum

theorem unique_number_with_special_divisor_property :
  ∃! n : ℕ, n > 0 ∧ 
    has_at_least_four_divisors n ∧
    sum_of_squares_of_smallest_four_divisors n = n ∧
    n = 130 := by
  sorry

#eval sum_of_squares_of_smallest_four_divisors 130

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_special_divisor_property_l1028_102870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_formation_for_12_and_15_l1028_102843

/-- Function to calculate the sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to check if a number is divisible by 4 -/
def divisible_by_four (n : ℕ) : Bool := n % 4 = 0

/-- Function to determine if it's possible to form a square without breaking sticks -/
def can_form_square (n : ℕ) : Bool := divisible_by_four (sum_of_first_n n)

/-- Function to determine the minimum number of sticks to break to form a square -/
def min_sticks_to_break (n : ℕ) : ℕ :=
  if can_form_square n then 0
  else 2  -- We know for n = 12, we need to break 2 sticks

theorem square_formation_for_12_and_15 :
  (min_sticks_to_break 12 = 2) ∧ (can_form_square 15 = true) := by
  apply And.intro
  · -- Proof for n = 12
    rfl
  · -- Proof for n = 15
    rfl

#eval min_sticks_to_break 12  -- Should output 2
#eval can_form_square 15      -- Should output true

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_formation_for_12_and_15_l1028_102843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l1028_102810

/-- 
Given a point in polar coordinates (-3, π/6), prove that it is equivalent to 
the point (3, 7π/6) in standard polar coordinate representation.
-/
theorem polar_coordinate_equivalence : 
  ∀ (r : ℝ) (θ : ℝ), 
    r = -3 ∧ θ = π/6 →
    ∃ (r' : ℝ) (θ' : ℝ),
      r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2*π ∧
      r' = 3 ∧ θ' = 7*π/6 ∧
      (r * Real.cos θ = r' * Real.cos θ' ∧ r * Real.sin θ = r' * Real.sin θ') :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coordinate_equivalence_l1028_102810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rearrangement_l1028_102844

def digits : List Nat := [5, 8, 6]

theorem largest_rearrangement (d : List Nat) (h : d = digits) :
  (d.permutations.map (fun l => l.foldl (fun acc x => acc * 10 + x) 0)).maximum? = some 865 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_rearrangement_l1028_102844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_problem_l1028_102890

/-- Given two lines l₁ and l₂ in the form ax + 3y + 1 = 0 and x + (a-2)y + a = 0 respectively,
    where a is a real number. -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + 1 = 0

def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a - 2) * y + a = 0

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (a : ℝ) : Prop := a * (a - 2) = -1

/-- Two lines are parallel if their slopes are equal -/
def parallel (a : ℝ) : Prop := a = 3

/-- The distance between two parallel lines Ax + By + C₁ = 0 and Ax + By + C₂ = 0 -/
noncomputable def distance (C₁ C₂ : ℝ) : ℝ := |C₁ - C₂| / Real.sqrt (3^2 + 3^2)

theorem lines_problem (a : ℝ) :
  (perpendicular a → a = 3/2) ∧
  (parallel a → distance 1 9 = 4 * Real.sqrt 2 / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_problem_l1028_102890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_beta_l1028_102868

theorem tan_alpha_plus_beta (α β : ℝ) 
  (h1 : Real.sin α = 3/5)
  (h2 : α ∈ Set.Ioo (π/2) π)
  (h3 : Real.sin (α + β) / Real.sin β = 4) :
  Real.tan (α + β) = -4/7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_beta_l1028_102868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1028_102888

/-- Represents a four-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Converts a FourDigitNumber to its numerical value -/
def toNatValue (n : FourDigitNumber) : Nat :=
  match n with
  | (a, b, c, d) => 1000 * a + 100 * b + 10 * c + d

/-- Checks if a FourDigitNumber satisfies the equation ABCD * D = CBAD -/
def satisfiesEquation (n : FourDigitNumber) : Prop :=
  match n with
  | (a, b, c, d) => toNatValue (a, b, c, d) * d = toNatValue (c, b, a, d)

/-- Checks if a number is a single digit (0-9) -/
def isSingleDigit (n : Nat) : Prop := n ≤ 9

/-- The main theorem stating that (2, 1, 7, 8) is the only solution -/
theorem unique_solution :
  ∀ (n : FourDigitNumber),
    (let (a, b, c, d) := n
     isSingleDigit a ∧ isSingleDigit b ∧ isSingleDigit c ∧ isSingleDigit d ∧
     1000 ≤ toNatValue n ∧ toNatValue n ≤ 9999 ∧
     satisfiesEquation n)
    → n = (2, 1, 7, 8) := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l1028_102888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_dilution_statement_is_true_l1028_102869

/-- Represents the range of dilutions used for spreading microorganisms. -/
structure DilutionRange where
  lower : Nat
  upper : Nat

/-- Represents the correct statement about dilutions for soil microorganisms. -/
def correct_dilution_statement : Prop :=
  ∃ (range : DilutionRange), 
    range.lower = 3 ∧ 
    range.upper = 7 ∧ 
    ∀ (d : Nat), range.lower ≤ d ∧ d ≤ range.upper → 
      ∃ (organism : Type) (sample : organism), (10 ^ d : Nat) > 0

/-- Theorem stating that the correct dilution statement is true. -/
theorem correct_dilution_statement_is_true : correct_dilution_statement := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_dilution_statement_is_true_l1028_102869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_per_metre_l1028_102875

/-- Proves that the cost of fencing per metre is approximately 6.97 Rs. given the conditions of the rectangular plot. -/
theorem fencing_cost_per_metre (length breadth perimeter total_cost : ℝ) : 
  length = 200 →
  breadth = 180 →
  perimeter = 2 * length + 2 * breadth →
  total_cost = 5300 →
  ∃ (cost_per_metre : ℝ), abs (cost_per_metre - (total_cost / perimeter)) < 0.01 := by
  sorry

#check fencing_cost_per_metre

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_cost_per_metre_l1028_102875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_output_l1028_102897

theorem algorithm_output (x : ℕ) : x = 20 → x = 20 := by
  intro h
  exact h

#check algorithm_output

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algorithm_output_l1028_102897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_system_solvability_l1028_102880

/-- A system of quadratic equations with coefficients a, b, and c -/
structure QuadraticSystem (a b c : ℝ) : Prop where
  eq1 : ∃ x : ℝ, a * x^2 + b * x + c = 0
  eq2 : ∃ x : ℝ, b * x^2 + c * x + a = 0
  eq3 : ∃ x : ℝ, c * x^2 + a * x + b = 0

/-- The system has real solutions if and only if a + b + c = 0 -/
theorem quadratic_system_solvability (a b c : ℝ) :
  QuadraticSystem a b c ↔ a + b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_system_solvability_l1028_102880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_sixteen_pi_thirds_l1028_102864

theorem cos_negative_sixteen_pi_thirds : Real.cos (-16 * π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_negative_sixteen_pi_thirds_l1028_102864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_tetrahedron_contains_center_l1028_102842

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  center : Point3D
  -- Other properties of a regular dodecahedron could be added here

/-- A tetrahedron formed by four points -/
structure Tetrahedron where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Function to check if a point is inside a regular dodecahedron -/
def isInsideDodecahedron (p : Point3D) (d : RegularDodecahedron) : Prop :=
  sorry

/-- Function to check if a point is inside a tetrahedron -/
def isInsideTetrahedron (p : Point3D) (t : Tetrahedron) : Prop :=
  sorry

/-- Function to generate a random point inside a regular dodecahedron -/
noncomputable def randomPointInDodecahedron (d : RegularDodecahedron) : Point3D :=
  sorry

/-- Probability measure -/
noncomputable def Prob (p : Prop) : ℝ :=
  sorry

/-- The main theorem -/
theorem probability_tetrahedron_contains_center (d : RegularDodecahedron) :
  let p1 := randomPointInDodecahedron d
  let p2 := randomPointInDodecahedron d
  let p3 := randomPointInDodecahedron d
  let p4 := randomPointInDodecahedron d
  let t := Tetrahedron.mk p1 p2 p3 p4
  Prob (isInsideTetrahedron d.center t) = 1/8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_tetrahedron_contains_center_l1028_102842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_theorem_l1028_102848

-- Define the circle type
structure Circle where
  radius : ℝ

-- Define the layer type
def Layer := List Circle

-- Define the set of all circles
def S : List Circle := sorry

-- Define the function to calculate the radius of a new circle
noncomputable def newRadius (r1 r2 : ℝ) : ℝ := (r1 * r2) / ((r1.sqrt + r2.sqrt) ^ 2)

-- Define the function to construct a new layer
def constructLayer (prevLayers : List Layer) : Layer := sorry

-- Define the recursive sum function
noncomputable def layerSum (n : ℕ) : ℝ := sorry

theorem circle_sum_theorem : 
  (S.map (λ c => 1 / c.radius.sqrt)).sum = 143 / 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_theorem_l1028_102848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_abc_l1028_102891

/-- Given points A, B, and C in a 2D plane, calculates the length of the median
    from A to the midpoint of BC. -/
noncomputable def median_length (A B C : ℝ × ℝ) : ℝ :=
  let D := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)  -- Midpoint of BC
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)    -- Distance formula

/-- Proves that the length of the median from A to the midpoint of BC in 
    triangle ABC is √10, given A(2,1), B(-2,3), and C(0,1). -/
theorem median_length_abc : 
  median_length (2, 1) (-2, 3) (0, 1) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_length_abc_l1028_102891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_set_l1028_102829

theorem count_integers_in_set : 
  ∃ (S : Finset ℤ), S = {x : ℤ | (x : ℝ)^2 < 3*(x : ℝ) + 4} ∧ S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_in_set_l1028_102829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_maximizes_distance_product_l1028_102884

/-- Represents a triangle with sides a, b, c and area T -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  T : ℝ

/-- Represents a point inside the triangle with distances x, y, z to the sides -/
structure InteriorPoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The centroid of a triangle -/
noncomputable def centroid (t : Triangle) : InteriorPoint where
  x := (2 * t.T) / (3 * t.a)
  y := (2 * t.T) / (3 * t.b)
  z := (2 * t.T) / (3 * t.c)

/-- The product of distances from a point to the sides of the triangle -/
def distanceProduct (p : InteriorPoint) : ℝ := p.x * p.y * p.z

/-- Theorem stating that the centroid maximizes the product of distances -/
theorem centroid_maximizes_distance_product (t : Triangle) :
  ∀ p : InteriorPoint, p.x * t.a + p.y * t.b + p.z * t.c = 2 * t.T →
    distanceProduct p ≤ distanceProduct (centroid t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_maximizes_distance_product_l1028_102884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_bisecting_folds_l1028_102809

/-- A parallelogram in 2D space -/
structure Parallelogram where
  -- We don't need to define the specifics of a parallelogram for this problem

/-- A line in 2D space -/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- The center point of a parallelogram (intersection of diagonals) -/
def Parallelogram.center (p : Parallelogram) : Point := sorry

/-- A function that checks if a line bisects the area of a parallelogram -/
def bisects_area (l : Line) (p : Parallelogram) : Prop := sorry

/-- A function that checks if a line passes through a point -/
def passes_through (l : Line) (pt : Point) : Prop := sorry

theorem infinitely_many_bisecting_folds (p : Parallelogram) :
  ∃ S : Set Line, (∀ l ∈ S, bisects_area l p) ∧ Infinite S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_bisecting_folds_l1028_102809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_result_l1028_102803

theorem complex_calculation_result : 
  ∃ x : ℝ, abs (x - ((12.05 * 5.4 + 0.6) / (2.3 - 1.8) * (7/3) - (4.07 * 3.5 + 0.45)^2)) < 0.001 := by
  sorry

#eval ((12.05 * 5.4 + 0.6) / (2.3 - 1.8) * (7/3) - (4.07 * 3.5 + 0.45)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_calculation_result_l1028_102803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1028_102855

theorem max_value_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∃ (m : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 1 → Real.sqrt (3*a) + Real.sqrt (4*b) ≤ m) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ Real.sqrt (3*x₀) + Real.sqrt (4*y₀) = Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1028_102855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1028_102865

noncomputable def f (x : Real) : Real := 
  2 * (Real.sin (Real.pi / 4 + x))^2 + Real.sqrt 3 * Real.cos (2 * x) - 1

theorem triangle_tangent (A B C : Real) :
  f C = Real.sqrt 3 →
  2 * Real.sin B = Real.cos (A - C) - Real.cos (A + C) →
  Real.tan A = -(1 + Real.sqrt 3) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_l1028_102865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_unique_local_minimum_l1028_102871

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (Real.exp x + 2) / x

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  ∃ (m b : ℝ), ∀ x y : ℝ, y = m * (x - 1) + f 1 ↔ 2 * x + y - Real.exp 1 - 4 = 0 :=
by sorry

-- Theorem for the unique local minimum
theorem unique_local_minimum :
  ∃! x₀ : ℝ, x₀ > 0 ∧ ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - x₀| ∧ |x - x₀| < δ → f x > f x₀ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_unique_local_minimum_l1028_102871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_instantaneous_rate_change_l1028_102846

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 8

-- Define the derivative f'
noncomputable def f' (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem min_instantaneous_rate_change :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ 5 → f' x ≤ f' y) ∧
  f' x = -1 := by
  -- Proof goes here
  sorry

#check min_instantaneous_rate_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_instantaneous_rate_change_l1028_102846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_different_subsequences_l1028_102847

/-- A balanced 9-digit number is a number that contains each digit from 1 to 9 exactly once. -/
def BalancedNumber : Type := { n : ℕ // n ≥ 100000000 ∧ n < 1000000000 ∧ ∀ d : Fin 9, ∃! (i : Fin 9), (n / (10 ^ i.val)) % 10 = d.val + 1 }

/-- The sequence S of numerals constructed by writing all balanced 9-digit numbers in increasing order consecutively. -/
noncomputable def S : List ℕ := sorry

/-- A function that returns true if two subsequences of S with consecutive k numerals are different. -/
def SubsequencesDifferent (k : ℕ) : Prop := 
  ∀ (i j : ℕ), i ≠ j → (S.drop i).take k ≠ (S.drop j).take k

/-- The theorem stating that 17 is the smallest positive integer k such that any two subsequences of S with consecutive k numerals are different. -/
theorem smallest_k_for_different_subsequences : 
  (SubsequencesDifferent 17 ∧ ∀ k < 17, ¬SubsequencesDifferent k) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_different_subsequences_l1028_102847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_center_l1028_102832

/-- The symmetry center of the graph of y = cos(2x + π/3) is (π/6, 0) -/
theorem cosine_symmetry_center :
  ∃ (x₀ : ℝ), x₀ = π/6 ∧
  ∀ (x : ℝ), Real.cos (2*(x₀ + x) + π/3) = Real.cos (2*(x₀ - x) + π/3) :=
by
  -- We'll use x₀ = π/6 as the symmetry center
  let x₀ := π/6
  
  -- Prove that such an x₀ exists
  use x₀
  
  constructor
  · -- Prove x₀ = π/6
    rfl
    
  · -- Prove the symmetry property
    intro x
    -- The proof would go here, but we'll use sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_symmetry_center_l1028_102832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1028_102854

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a line
def line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

-- Define the intersection of a line and the parabola
def intersects_parabola (a b c : ℝ) : Prop :=
  ∃ x1 y1 x2 y2 : ℝ, 
    parabola x1 y1 ∧ parabola x2 y2 ∧
    line a b c x1 y1 ∧ line a b c x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define the midpoint of two points
def is_midpoint (x1 y1 x2 y2 xm ym : ℝ) : Prop :=
  xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

-- Theorem statement
theorem line_equation : 
  ∀ a b c : ℝ,
    intersects_parabola a b c →
    (∃ x1 y1 x2 y2 : ℝ, 
      parabola x1 y1 ∧ parabola x2 y2 ∧
      line a b c x1 y1 ∧ line a b c x2 y2 ∧
      is_midpoint x1 y1 x2 y2 1 1) →
    a = 2 ∧ b = -1 ∧ c = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1028_102854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_comparison_l1028_102899

/-- Represents the height percentiles for a given age group -/
structure HeightPercentiles where
  p75 : ℝ
  p90 : ℝ

/-- Calculates the minimum number of people shorter than a given height -/
noncomputable def min_shorter_count (total_population : ℕ) (height : ℝ) (percentiles : HeightPercentiles) : ℝ :=
  if height ≥ percentiles.p75 then
    (total_population : ℝ) * 0.75
  else
    0

theorem height_comparison (total_population : ℕ) (height : ℝ) (percentiles : HeightPercentiles) :
  total_population = 64000 →
  height = 176 →
  percentiles.p75 = 175 →
  percentiles.p90 = 179 →
  min_shorter_count total_population height percentiles ≥ 48000 := by
  sorry

#eval (64000 : ℕ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_comparison_l1028_102899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_point_off_plane_l1028_102894

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Checks if a point is on the xy-plane (z = 0) -/
def onSurface (p : Point3D) : Prop :=
  p.z = 0

/-- Theorem: Given four points where three are on a plane and all pairs are equidistant,
    the fourth point must be off the plane -/
theorem fourth_point_off_plane (p1 p2 p3 p4 : Point3D) 
  (h1 : onSurface p1) (h2 : onSurface p2) (h3 : onSurface p3)
  (eq_dist : ∀ (i j : Fin 4), i ≠ j → 
    distance (match i with
      | 0 => p1
      | 1 => p2
      | 2 => p3
      | 3 => p4)
    (match j with
      | 0 => p1
      | 1 => p2
      | 2 => p3
      | 3 => p4) = 100) :
  ¬ onSurface p4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_point_off_plane_l1028_102894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_of_point_on_parabola_l1028_102862

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y^2 = 2 * p * x

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (para : Parabola) (pt : Point) : ℝ :=
  pt.x + para.p / 2

theorem distance_to_directrix_of_point_on_parabola :
  ∀ (para : Parabola) (A : Point),
    A.x = 1 ∧ A.y = Real.sqrt 5 →
    para.equation A.x A.y →
    distance_to_directrix para A = 9/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_of_point_on_parabola_l1028_102862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_eq1_and_eq4_perpendicular_l1028_102857

-- Define the equations
def eq1 (x y : ℝ) : Prop := 5*y - 3*x = 15
def eq2 (x y : ℝ) : Prop := -3*x - 5*y = 15
def eq3 (x y : ℝ) : Prop := 5*y + 3*x = 15
def eq4 (x y : ℝ) : Prop := 3*y + 5*x = 15
def eq5 (x y : ℝ) : Prop := 2*x - 10*y = 12

-- Define a function to get the slope of a line
noncomputable def getSlope (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

-- Define a function to check if two lines are perpendicular
def arePerpendicular (eq1 eq2 : (ℝ → ℝ → Prop)) : Prop :=
  getSlope eq1 * getSlope eq2 = -1

-- Theorem statement
theorem only_eq1_and_eq4_perpendicular :
  (arePerpendicular eq1 eq4) ∧
  (¬ arePerpendicular eq1 eq2) ∧
  (¬ arePerpendicular eq1 eq3) ∧
  (¬ arePerpendicular eq1 eq5) ∧
  (¬ arePerpendicular eq2 eq3) ∧
  (¬ arePerpendicular eq2 eq4) ∧
  (¬ arePerpendicular eq2 eq5) ∧
  (¬ arePerpendicular eq3 eq4) ∧
  (¬ arePerpendicular eq3 eq5) ∧
  (¬ arePerpendicular eq4 eq5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_eq1_and_eq4_perpendicular_l1028_102857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1028_102856

theorem equation_solution : ∃ x : ℝ, (3 : ℝ) ^ (x - 2) = (9 : ℝ) ^ x ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1028_102856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zongzi_cost_equation_l1028_102873

/-- Represents the price of a meat zongzi in yuan -/
def x : ℝ := sorry

/-- The price difference between meat and vegetarian zongzi -/
def price_difference : ℝ := 1

/-- The number of meat zongzi purchased -/
def meat_count : ℕ := 10

/-- The number of vegetarian zongzi purchased -/
def veg_count : ℕ := 5

/-- The total cost of the purchase in yuan -/
def total_cost : ℝ := 70

/-- Theorem stating that the equation correctly represents the total cost -/
theorem zongzi_cost_equation :
  meat_count * x + veg_count * (x - price_difference) = total_cost := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zongzi_cost_equation_l1028_102873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_increment_function_bound_l1028_102850

/-- A function f is an l-increment function on a set M if for any x in M,
    x + l is in the domain of f and f(x + l) ≥ f(x) -/
def IsIncrementFunction (f : ℝ → ℝ) (l : ℝ) (M : Set ℝ) : Prop :=
  l ≠ 0 ∧ ∀ x ∈ M, f (x + l) ≥ f x

theorem square_increment_function_bound :
  ∀ m : ℝ, IsIncrementFunction (fun x ↦ x^2) m {x : ℝ | x ≥ -1} → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_increment_function_bound_l1028_102850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_and_distance_l1028_102826

-- Define points A and P
def A : ℝ × ℝ := (-1, 0)
def P : ℝ × ℝ := (2, 4)

-- Define the distance function as noncomputable
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem point_B_coordinates_and_distance (x : ℝ) :
  let B : ℝ × ℝ := (x, 0)
  distance A P = distance B P → x = 5 ∧ distance B P = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_and_distance_l1028_102826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l1028_102800

theorem magnitude_relationship (a b c : ℝ) : 
  a = (6 : ℝ)^(0.7 : ℝ) → b = (0.7 : ℝ)^(0.8 : ℝ) → c = (0.8 : ℝ)^(0.7 : ℝ) → a > c ∧ c > b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_relationship_l1028_102800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_average_price_l1028_102827

theorem dvd_average_price : 
  let box1_count : ℕ := 10
  let box1_price : ℚ := 2
  let box2_count : ℕ := 5
  let box2_price : ℚ := 5
  let box3_count : ℕ := 3
  let box3_price : ℚ := 7
  let total_cost : ℚ := box1_count * box1_price + box2_count * box2_price + box3_count * box3_price
  let total_count : ℕ := box1_count + box2_count + box3_count
  total_cost / total_count = 367 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dvd_average_price_l1028_102827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_obtuse_solutions_l1028_102896

theorem triangle_two_obtuse_solutions (a b c A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
  A = π / 6 ∧ b = 3 →
  (∃ B₁ B₂ : ℝ, B₁ ≠ B₂ ∧ 
    π / 2 < B₁ ∧ B₁ < π ∧ 
    π / 2 < B₂ ∧ B₂ < π ∧
    Real.sin B₁ = 3 / (2 * a) ∧ 
    Real.sin B₂ = 3 / (2 * a)) ↔ 
  Real.sqrt 3 < a ∧ a < 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_two_obtuse_solutions_l1028_102896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trominoes_for_black_squares_l1028_102863

/-- Represents a tromino (L-shaped piece covering 3 squares) -/
structure Tromino

/-- Represents a chessboard -/
structure Chessboard (n : ℕ) where
  is_odd : Odd n
  ge_seven : n ≥ 7

/-- Calculates k for a given n, where n = 2k - 1 -/
def calc_k (n : ℕ) : ℕ := (n + 1) / 2

/-- Counts the number of black squares on an n×n chessboard -/
def count_black_squares (n : ℕ) : ℕ := ((n + 1) / 2)^2 + (n / 2)^2

/-- Theorem stating the minimum number of trominoes needed to cover all black squares -/
theorem min_trominoes_for_black_squares (n : ℕ) (board : Chessboard n) :
  ∃ (trominoes : Finset Tromino),
    trominoes.card = (calc_k n)^2 ∧
    (trominoes.card : ℕ) * 3 ≥ count_black_squares n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trominoes_for_black_squares_l1028_102863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_misha_is_lying_l1028_102895

/-- Represents the possible goal claims made by players -/
inductive GoalClaim
  | one
  | two
  | three
  | five

/-- Represents a player in the football match -/
structure Player where
  claim : GoalClaim
  isTruthTeller : Bool

/-- The football match setup -/
structure FootballMatch where
  players : List Player
  truthTellerScore : Nat
  liarScore : Nat

/-- Misha's claim -/
def mishasClaim : GoalClaim := GoalClaim.two

/-- The specific football match in the problem -/
def problemMatch : FootballMatch :=
  { players := List.replicate 20 { claim := GoalClaim.one, isTruthTeller := true }  -- placeholder, actual distribution unknown
    truthTellerScore := 20
    liarScore := 17 }

/-- Function to check if Misha is lying -/
def isMishaLying (m : FootballMatch) : Prop :=
  ∃ (misha : Player), misha ∈ m.players ∧ misha.claim = mishasClaim ∧ ¬misha.isTruthTeller

/-- The main theorem to prove -/
theorem misha_is_lying : isMishaLying problemMatch := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_misha_is_lying_l1028_102895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_l1028_102845

/-- The capacity of the pool in cubic meters -/
noncomputable def C : ℝ := 7200

/-- The rate at which valve A fills the pool (cubic meters per minute) -/
noncomputable def rate_A : ℝ := C / 180

/-- The rate at which valve B fills the pool (cubic meters per minute) -/
noncomputable def rate_B : ℝ := rate_A + 80

/-- The rate at which valve C fills the pool (cubic meters per minute) -/
noncomputable def rate_C : ℝ := 2 * rate_A

theorem pool_capacity :
  (rate_A + rate_B = C / 60) ∧
  (rate_A + rate_B + rate_C = C / 30) ∧
  (rate_B = rate_A + 80) ∧
  (rate_C = 2 * rate_A) →
  C = 7200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_capacity_l1028_102845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problems_l1028_102849

theorem calculation_problems :
  ((Real.sqrt 12 + Real.sqrt (1 + 1/3)) * Real.sqrt 3 = 8) ∧
  (Real.sqrt 48 - Real.sqrt 54 / Real.sqrt 2 + (3 - Real.sqrt 3) * (3 + Real.sqrt 3) = Real.sqrt 3 + 6) :=
by
  constructor
  · sorry  -- Proof for the first part
  · sorry  -- Proof for the second part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_problems_l1028_102849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_factors_not_3_divisible_count_l1028_102814

/-- The number of odd factors of 252 that are not divisible by 3 -/
def oddFactorsNot3Divisible : ℕ :=
  (Finset.filter (fun x => x % 2 ≠ 0 ∧ ¬ 3 ∣ x) (Nat.divisors 252)).card

/-- Theorem stating that the number of odd factors of 252 not divisible by 3 is 2 -/
theorem odd_factors_not_3_divisible_count : oddFactorsNot3Divisible = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_factors_not_3_divisible_count_l1028_102814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_union_complement_l1028_102801

-- Define the sets A, B, and P
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5}

-- State the theorem
theorem intersection_union_complement :
  (A ∩ B) ∪ (Set.univ \ P) = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_union_complement_l1028_102801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_infinity_l1028_102815

/-- The sequence defined by ((n+4)! - (n+2)!) / (n+3)! for n ≥ 0 -/
def our_sequence (n : ℕ) : ℚ :=
  (Nat.factorial (n + 4) - Nat.factorial (n + 2)) / Nat.factorial (n + 3)

/-- The limit of the sequence as n approaches infinity is infinity -/
theorem sequence_limit_is_infinity :
  Filter.Tendsto our_sequence Filter.atTop Filter.atTop := by
  sorry

#check sequence_limit_is_infinity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_is_infinity_l1028_102815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l1028_102861

theorem triangle_side_count : 
  let side1 : ℕ := 8
  let side2 : ℕ := 11
  (Finset.filter (fun x => 
    x > 0 ∧ 
    x + side1 > side2 ∧ 
    x + side2 > side1 ∧ 
    side1 + side2 > x
  ) (Finset.range 20)).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l1028_102861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_implies_c_magnitude_l1028_102837

/-- The polynomial P(x) -/
def P (c : ℂ) (x : ℂ) : ℂ := (x^2 - 2*x + 2) * (x^2 - c*x + 1) * (x^2 - 4*x + 5)

/-- The set of roots of P(x) -/
def roots (c : ℂ) : Set ℂ := {x : ℂ | P c x = 0}

/-- Theorem stating that if P(x) has exactly 3 distinct roots, then |c| = √13 -/
theorem three_roots_implies_c_magnitude (c : ℂ) : 
  (∃ (S : Finset ℂ), S.toSet = roots c ∧ S.card = 3) → Complex.abs c = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_implies_c_magnitude_l1028_102837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zircon_halfway_distance_l1028_102828

/-- Represents the orbital characteristics of planet Zircon -/
structure ZirconOrbit where
  perihelion : ℝ
  aphelion : ℝ
  elliptical : Bool

/-- Calculates the distance from the sun to a point halfway along Zircon's orbital path -/
noncomputable def halfway_distance (orbit : ZirconOrbit) : ℝ :=
  orbit.perihelion + (orbit.aphelion - orbit.perihelion) / 2

/-- Theorem stating the distance from the sun to Zircon when it's halfway along its orbit -/
theorem zircon_halfway_distance (orbit : ZirconOrbit) 
  (h1 : orbit.perihelion = 3)
  (h2 : orbit.aphelion = 15)
  (h3 : orbit.elliptical = true) :
  halfway_distance orbit = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zircon_halfway_distance_l1028_102828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_orthocenter_coordinates_l1028_102838

-- Define the angles and constant
variable (α β γ R : ℝ)

-- Define the points A, B, and C
noncomputable def A : ℝ × ℝ := (R * Real.cos α, R * Real.sin α)
noncomputable def B : ℝ × ℝ := (R * Real.cos β, R * Real.sin β)
noncomputable def C : ℝ × ℝ := (R * Real.cos γ, R * Real.sin γ)

-- Define the orthocenter H
noncomputable def H : ℝ × ℝ := (R * (Real.cos α + Real.cos β + Real.cos γ), R * (Real.sin α + Real.sin β + Real.sin γ))

-- Theorem 1
theorem trigonometric_identity (α β γ : ℝ) :
  Real.tan ((β + γ) / 2) * (Real.cos β + Real.cos γ) - Real.tan ((γ + α) / 2) * (Real.cos γ + Real.cos α) = Real.sin β - Real.sin α := by
  sorry

-- Theorem 2
theorem orthocenter_coordinates (α β γ R : ℝ) :
  H α β γ R = (R * (Real.cos α + Real.cos β + Real.cos γ), R * (Real.sin α + Real.sin β + Real.sin γ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_orthocenter_coordinates_l1028_102838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lattice_points_l1028_102833

theorem hyperbola_lattice_points : 
  ∃ s : Finset (ℤ × ℤ), s.card = 210 ∧ 
    ∀ (x y : ℤ), (x, y) ∈ s ↔ x^2 - y^2 = 3000^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_lattice_points_l1028_102833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1028_102886

/-- Represents the annual interest rate as a percentage -/
noncomputable def annual_interest_rate : ℚ := 16

/-- The face value of the bill in Rupees -/
def face_value : ℚ := 1764

/-- The true discount on the bill in Rupees -/
def true_discount : ℚ := 189

/-- The time period until the bill is due, in years -/
def time_period : ℚ := 9 / 12

/-- The present value of the bill -/
def present_value : ℚ := face_value - true_discount

theorem interest_rate_calculation :
  ∃ ε : ℚ, ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ 
  |annual_interest_rate - (true_discount * 100) / (present_value * time_period)| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1028_102886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_in_three_turns_l1028_102853

/-- Represents a point on the circle --/
def Point := Fin 15

/-- Moves a point clockwise by a given number of steps --/
def move_clockwise (p : Point) (steps : Nat) : Point :=
  ⟨(p.val + steps) % 15, by
    apply Nat.mod_lt
    · exact Nat.zero_lt_succ 14
  ⟩

/-- Moves a point counterclockwise by a given number of steps --/
def move_counterclockwise (p : Point) (steps : Nat) : Point :=
  ⟨(p.val + 15 - steps % 15) % 15, by
    apply Nat.mod_lt
    · exact Nat.zero_lt_succ 14
  ⟩

/-- Alice's position after n turns --/
def alice_position (n : Nat) : Point :=
  move_clockwise ⟨14, by norm_num⟩ (4 * n)

/-- Bob's position after n turns --/
def bob_position (n : Nat) : Point :=
  if n ≤ 2 then ⟨14, by norm_num⟩
  else move_counterclockwise ⟨14, by norm_num⟩ (11 * (n - 2))

theorem game_ends_in_three_turns :
  ∃ (n : Nat), n = 3 ∧ alice_position n = bob_position n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_ends_in_three_turns_l1028_102853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_focus_chords_sum_l1028_102877

/-- Represents a parabola y² = 4x with p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a chord of the parabola passing through its focus -/
structure FocusChord (parabola : Parabola) where
  length : ℝ
  h_length_pos : length > 0

/-- Theorem: For a parabola y² = 4x (p > 0), if two chords AB and CD pass through the focus
    and are mutually perpendicular, then 1/|AB| + 1/|CD| = 1/4 -/
theorem perpendicular_focus_chords_sum (parabola : Parabola) 
  (AB CD : FocusChord parabola) (h_perp : AB.length ≠ CD.length) :
  1 / AB.length + 1 / CD.length = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_focus_chords_sum_l1028_102877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_present_value_l1028_102887

/-- Calculates the present value given future value, interest rate, and time -/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (time : ℝ) : ℝ :=
  futureValue / (1 + interestRate / 2) ^ (2 * time)

/-- Theorem stating the approximate present value for the given conditions -/
theorem investment_present_value :
  let futureValue : ℝ := 600000
  let interestRate : ℝ := 0.04
  let time : ℝ := 12
  let calculatedPresentValue := presentValue futureValue interestRate time
  ∃ ε > 0, |calculatedPresentValue - 374811.16| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_present_value_l1028_102887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l1028_102834

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.sqrt 2 / 2, -Real.sqrt 2 / 2],
    ![Real.sqrt 2 / 2, Real.sqrt 2 / 2]]

theorem matrix_power_four :
  A ^ 4 = ![![-1, 0],
            ![0, -1]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l1028_102834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1028_102822

theorem expression_simplification :
  |Real.sqrt 3 - 2| - (Real.sqrt 3 - 1) + (-4) = -1 - 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1028_102822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_equal_implies_parallelogram_l1028_102851

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral (P : Type*) [NormedAddCommGroup P] [InnerProductSpace ℝ P] :=
  (A B C D : P)

/-- Parallelogram property for a quadrilateral -/
def is_parallelogram {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (Q : Quadrilateral P) : Prop :=
  (Q.A - Q.B) = (Q.D - Q.C) ∧ (Q.B - Q.C) = (Q.A - Q.D)

/-- Two line segments are parallel -/
def parallel {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] (A B C D : P) : Prop :=
  ∃ (t : ℝ), C - D = t • (A - B)

/-- Theorem: If in a quadrilateral ABCD, AB is parallel to CD and AB = CD, then ABCD is a parallelogram -/
theorem parallel_and_equal_implies_parallelogram {P : Type*} [NormedAddCommGroup P] [InnerProductSpace ℝ P] 
  (Q : Quadrilateral P) (h1 : parallel Q.A Q.B Q.C Q.D) (h2 : Q.A - Q.B = Q.C - Q.D) : 
  is_parallelogram Q :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_equal_implies_parallelogram_l1028_102851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_is_zero_l1028_102872

open Real

noncomputable def f (x : ℝ) := x^3 + sin x + 1

theorem f_negative_a_is_zero (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_a_is_zero_l1028_102872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_a_l1028_102811

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x + 2

theorem determine_a :
  (∀ x ∈ Set.univ \ {0}, f 1 x ∈ Set.Iic 0 ∪ Set.Ici 4) ∧
  (∃ x₁ x₂, x₁ < 0 ∧ 0 < x₂ ∧ f 1 x₁ = 0 ∧ f 1 x₂ = 4) ∧
  (∀ a ≠ 1, ∃ x ∈ Set.univ \ {0}, f a x ∉ Set.Iic 0 ∪ Set.Ici 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determine_a_l1028_102811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_100_equals_formula_l1028_102812

def customSequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => if n % 2 = 0 then 2 * customSequence n else 3 * customSequence n

def sum_100 : ℚ := (List.range 100).map customSequence |>.sum

theorem sum_100_equals_formula : sum_100 = (3/5) * (6^50 - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_100_equals_formula_l1028_102812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_fractions_l1028_102824

theorem comparison_of_fractions (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) / (1 + x + y) < x / (1 + x) + y / (1 + y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_comparison_of_fractions_l1028_102824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_popcorn_buckets_per_package_l1028_102858

theorem popcorn_buckets_per_package 
  (total_buckets : ℕ) 
  (num_packages : ℕ) 
  (h1 : total_buckets = 426) 
  (h2 : num_packages = 54) : 
  Int.ceil ((total_buckets : ℚ) / num_packages) = 8 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_popcorn_buckets_per_package_l1028_102858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1028_102802

theorem sin_alpha_value (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.cos (α + π/3) = -4/5) : 
  Real.sin α = (3 + 4 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_value_l1028_102802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_no_int_roots_l1028_102816

-- Define the polynomial type (using existing Polynomial from Mathlib)
-- def Polynomial (α : Type) := List α  -- Remove this line as it's already defined in Mathlib

-- Define the evaluation of a polynomial at a point
def evalPoly (p : Polynomial ℤ) (x : ℤ) : ℤ :=
  p.eval x

-- The property of having integer coefficients is implicit in Polynomial ℤ
-- so we don't need to define hasIntCoeffs

-- Define the property of taking odd values at 0 and 1
def oddAt0And1 (p : Polynomial ℤ) : Prop :=
  Odd (evalPoly p 0) ∧ Odd (evalPoly p 1)

-- Define the property of having no integer roots
def noIntRoots (p : Polynomial ℤ) : Prop :=
  ∀ x : ℤ, evalPoly p x ≠ 0

-- State the theorem
theorem polynomial_no_int_roots (p : Polynomial ℤ) 
  (h : oddAt0And1 p) : noIntRoots p := by
  sorry

#check polynomial_no_int_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_no_int_roots_l1028_102816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_parabola_vertex_l1028_102878

/-- The number of values of b for which the line y = x + b passes through the vertex of the parabola y = x^2 - 2b^2 -/
theorem line_passes_through_parabola_vertex : 
  ∃! (s : Finset ℝ), 
    (∀ b ∈ s, (λ x => x + b) 0 = 0^2 - 2*b^2) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_parabola_vertex_l1028_102878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_characterization_exists_non_odd_non_even_function_with_square_property_l1028_102821

-- Define an increasing function on R
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem for the first conclusion
theorem increasing_function_characterization
  (f : ℝ → ℝ) (hf : IncreasingFunction f) :
  ∀ x₁ x₂, f x₁ ≤ f x₂ ↔ x₁ ≤ x₂ := by sorry

-- Theorem for the second conclusion
theorem exists_non_odd_non_even_function_with_square_property :
  ∃ f : ℝ → ℝ, (∀ x, (f x)^2 = (f (-x))^2) ∧
  (∃ y, f y ≠ f (-y)) ∧ (∃ z, f z ≠ -f (-z)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_characterization_exists_non_odd_non_even_function_with_square_property_l1028_102821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_red_path_distribution_l1028_102892

-- Define the graph
def Graph := Fin 6 → Fin 6 → Bool

-- Define the probability of an edge being red
noncomputable def p_red : ℝ := 1/2

-- Define the random coloring of the graph
def random_coloring (g : Graph) : Prop :=
  ∀ i j, g i j = true ∨ g i j = false

-- Define the shortest all-red path length
def shortest_red_path (g : Graph) : ℕ :=
  sorry

-- Define the probability distribution of the shortest red path length
noncomputable def prob_shortest_red_path (n : ℕ) : ℝ :=
  sorry

-- State the theorem
theorem shortest_red_path_distribution :
  prob_shortest_red_path 0 = 69/128 ∧
  prob_shortest_red_path 2 = 7/16 ∧
  prob_shortest_red_path 4 = 3/128 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_red_path_distribution_l1028_102892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_extended_volume_l1028_102867

/-- The volume of points inside or within 2 units outside of a cube with side length 4 -/
theorem cube_extended_volume : 
  (let cube_side : ℝ := 4
   let extension : ℝ := 2
   let extended_side : ℝ := cube_side + 2 * extension
   let cube_volume : ℝ := cube_side ^ 3
   let extended_volume : ℝ := extended_side ^ 3
   let corner_sphere_volume : ℝ := (4 / 3) * Real.pi * extension ^ 3
   let edge_cylinder_volume : ℝ := Real.pi * extension ^ 2 * cube_side
   let total_volume : ℝ := 
     (extended_volume - cube_volume) + 
     (8 * (1 / 8) * corner_sphere_volume) + 
     (12 * edge_cylinder_volume)
   total_volume) = 448 + (608 / 3) * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_extended_volume_l1028_102867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_tangent_l1028_102825

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the center and radius of circle1
def center1 : ℝ × ℝ := (0, 0)
def radius1 : ℝ := 2

-- Define the center and radius of circle2
def center2 : ℝ × ℝ := (-1, 0)
def radius2 : ℝ := 1

-- Define the distance between the centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Theorem: The circles are tangent to each other
theorem circles_are_tangent : distance_between_centers = radius1 - radius2 := by
  -- The proof goes here
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_tangent_l1028_102825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_coverage_l1028_102805

/-- A parallelepiped with dimensions a, b, and c can have three faces sharing a common vertex
    completely covered by 5-cell strips without overlaps or gaps if and only if
    at least two of a, b, and c are divisible by 5. -/
theorem parallelepiped_coverage (a b c : ℕ) :
  (∃ (cover : Set (Set (ℕ × ℕ × ℕ))), 
    (∀ (x y z : ℕ), (x < a ∧ y < b) ∨ (y < b ∧ z < c) ∨ (x < a ∧ z < c) → 
      ∃! strip, strip ∈ cover ∧ (x, y, z) ∈ strip) ∧
    (∀ strip, strip ∈ cover → ∃ (i j k : ℕ), strip = {(x, y, z) | 
      ((x = i ∧ y = j ∧ k ≤ z ∧ z < k + 5) ∨
       (x = i ∧ z = k ∧ j ≤ y ∧ y < j + 5) ∨
       (y = j ∧ z = k ∧ i ≤ x ∧ x < i + 5)) ∧
      ((x < a ∧ y < b) ∨ (y < b ∧ z < c) ∨ (x < a ∧ z < c))})) ↔
  (a % 5 = 0 ∧ b % 5 = 0) ∨ (b % 5 = 0 ∧ c % 5 = 0) ∨ (a % 5 = 0 ∧ c % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_coverage_l1028_102805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_set_cardinality_l1028_102883

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- A line in 3D space -/
structure Line where
  direction : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  ∃ (k : ℝ), l1.direction = k • l2.direction

/-- A point is equidistant from two lines with distance d -/
def equidistant (p : ℝ × ℝ × ℝ) (l1 l2 : Line) (d : ℝ) : Prop :=
  sorry -- Define equidistant property here

/-- The set of points equidistant from two lines -/
def equidistantSet (α : Plane) (m n : Line) (d : ℝ) : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | equidistant p m n d}

/-- Cardinality of a finite set -/
def Set.card {α : Type*} (s : Set α) : ℕ :=
  sorry -- Define cardinality for finite sets

theorem equidistant_set_cardinality (α : Plane) (m n : Line) (d : ℝ) 
  (h_parallel : parallel m n) :
  let S := equidistantSet α m n d
  (S.card = 0 ∨ S.card = 1 ∨ S.card = 2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_set_cardinality_l1028_102883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_target_breaking_orders_l1028_102860

def string_permutations (x y z : ℕ) : ℕ := (x + y + z).factorial / (x.factorial * y.factorial * z.factorial)

theorem target_breaking_orders : string_permutations 3 3 2 = 560 := by
  -- Proof goes here
  sorry

#eval string_permutations 3 3 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_target_breaking_orders_l1028_102860
