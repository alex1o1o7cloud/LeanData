import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_sum_l1191_119146

def total_population : ℕ := 40 + 10 + 30 + 20

def strata : List ℕ := [40, 10, 30, 20]

def sample_size : ℕ := 20

def stratified_sample (stratum : ℕ) : ℕ :=
  (stratum * sample_size) / total_population

theorem stratified_sampling_sum :
  stratified_sample (strata[1]) + stratified_sample (strata[3]) = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sum_l1191_119146


namespace NUMINAMATH_CALUDE_solve_complex_equation_l1191_119144

theorem solve_complex_equation :
  let z : ℂ := 10 + 180 * Complex.I
  let equation := fun (x : ℂ) ↦ 7 * x - z = 15000
  ∃ (x : ℂ), equation x ∧ x = 2144 + (2 / 7) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l1191_119144


namespace NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1191_119197

-- Problem 1
theorem simplify_fraction_1 (a : ℝ) (h : a ≠ -1) :
  (2 * a^2 - 3) / (a + 1) - (a^2 - 2) / (a + 1) = a - 1 := by
  sorry

-- Problem 2
theorem simplify_fraction_2 (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (x / (x^2 - 4)) / (x / (4 - 2*x)) = -2 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_1_simplify_fraction_2_l1191_119197


namespace NUMINAMATH_CALUDE_smallest_integer_y_l1191_119188

theorem smallest_integer_y : ∀ y : ℤ, (7 - 3*y < 22) → y ≥ -4 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_y_l1191_119188


namespace NUMINAMATH_CALUDE_product_of_three_primes_l1191_119110

theorem product_of_three_primes : 
  ∃ (p q r : ℕ), 
    989 * 1001 * 1007 + 320 = p * q * r ∧ 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p < q ∧ q < r ∧
    p = 991 ∧ q = 997 ∧ r = 1009 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_primes_l1191_119110


namespace NUMINAMATH_CALUDE_expression_value_l1191_119187

theorem expression_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + d^2 - a*d = b^2 + c^2 + b*c) (h2 : a^2 + b^2 = c^2 + d^2) :
  (a*b + c*d) / (a*d + b*c) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1191_119187


namespace NUMINAMATH_CALUDE_inverse_proportion_k_negative_l1191_119118

/-- Given two points A(-2, y₁) and B(5, y₂) on the graph of y = k/x (k ≠ 0),
    if y₁ > y₂, then k < 0. -/
theorem inverse_proportion_k_negative
  (k : ℝ) (y₁ y₂ : ℝ)
  (hk : k ≠ 0)
  (hA : y₁ = k / (-2))
  (hB : y₂ = k / 5)
  (hy : y₁ > y₂) :
  k < 0 :=
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_negative_l1191_119118


namespace NUMINAMATH_CALUDE_smallest_n_proof_n_is_minimal_l1191_119106

/-- Represents an answer sheet with 5 questions, each having 4 possible answers -/
def AnswerSheet := Fin 5 → Fin 4

/-- The total number of answer sheets -/
def totalSheets : ℕ := 2000

/-- Checks if two answer sheets have at most 3 matching answers -/
def atMostThreeMatches (sheet1 sheet2 : AnswerSheet) : Prop :=
  (Finset.filter (λ i => sheet1 i = sheet2 i) (Finset.univ : Finset (Fin 5))).card ≤ 3

/-- The smallest number n that satisfies the condition -/
def smallestN : ℕ := 25

theorem smallest_n_proof :
  ∀ (sheets : Finset AnswerSheet),
    sheets.card = totalSheets →
    ∀ (subset : Finset AnswerSheet),
      subset ⊆ sheets →
      subset.card = smallestN →
      ∃ (sheet1 sheet2 sheet3 sheet4 : AnswerSheet),
        sheet1 ∈ subset ∧ sheet2 ∈ subset ∧ sheet3 ∈ subset ∧ sheet4 ∈ subset ∧
        sheet1 ≠ sheet2 ∧ sheet1 ≠ sheet3 ∧ sheet1 ≠ sheet4 ∧
        sheet2 ≠ sheet3 ∧ sheet2 ≠ sheet4 ∧ sheet3 ≠ sheet4 ∧
        atMostThreeMatches sheet1 sheet2 ∧
        atMostThreeMatches sheet1 sheet3 ∧
        atMostThreeMatches sheet1 sheet4 ∧
        atMostThreeMatches sheet2 sheet3 ∧
        atMostThreeMatches sheet2 sheet4 ∧
        atMostThreeMatches sheet3 sheet4 :=
by
  sorry

theorem n_is_minimal :
  ∀ n : ℕ,
    n < smallestN →
    ∃ (sheets : Finset AnswerSheet),
      sheets.card = totalSheets ∧
      ∃ (subset : Finset AnswerSheet),
        subset ⊆ sheets ∧
        subset.card = n ∧
        ∀ (sheet1 sheet2 sheet3 sheet4 : AnswerSheet),
          sheet1 ∈ subset → sheet2 ∈ subset → sheet3 ∈ subset → sheet4 ∈ subset →
          sheet1 ≠ sheet2 → sheet1 ≠ sheet3 → sheet1 ≠ sheet4 →
          sheet2 ≠ sheet3 → sheet2 ≠ sheet4 → sheet3 ≠ sheet4 →
          ¬(atMostThreeMatches sheet1 sheet2 ∧
            atMostThreeMatches sheet1 sheet3 ∧
            atMostThreeMatches sheet1 sheet4 ∧
            atMostThreeMatches sheet2 sheet3 ∧
            atMostThreeMatches sheet2 sheet4 ∧
            atMostThreeMatches sheet3 sheet4) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_proof_n_is_minimal_l1191_119106


namespace NUMINAMATH_CALUDE_thirty_switches_connections_l1191_119108

/-- Given a network of switches where each switch is directly connected to
    exactly 4 other switches, calculate the number of unique connections. -/
def uniqueConnections (n : ℕ) : ℕ :=
  (n * 4) / 2

theorem thirty_switches_connections :
  uniqueConnections 30 = 60 := by
  sorry

end NUMINAMATH_CALUDE_thirty_switches_connections_l1191_119108


namespace NUMINAMATH_CALUDE_largest_fraction_addition_l1191_119168

def is_proper_fraction (n d : ℤ) : Prop := 0 < n ∧ n < d

def denominator_less_than_8 (d : ℤ) : Prop := 0 < d ∧ d < 8

theorem largest_fraction_addition :
  ∀ n d : ℤ,
    is_proper_fraction n d →
    denominator_less_than_8 d →
    is_proper_fraction (6 * n + d) (6 * d) →
    n * 7 ≤ 5 * d :=
by sorry

end NUMINAMATH_CALUDE_largest_fraction_addition_l1191_119168


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1191_119171

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {x | 3 - 2*x > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | x < 3/2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1191_119171


namespace NUMINAMATH_CALUDE_max_volume_difference_l1191_119177

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- The measured dimensions of the box -/
def measuredDimensions : BoxDimensions :=
  { length := 150, width := 150, height := 225 }

/-- The maximum error in each measurement -/
def maxError : ℝ := 1

/-- Theorem: The maximum possible difference between the actual capacity
    and the computed capacity of the box is 90726 cubic centimeters -/
theorem max_volume_difference :
  ∃ (actualDimensions : BoxDimensions),
    actualDimensions.length ≤ measuredDimensions.length + maxError ∧
    actualDimensions.length ≥ measuredDimensions.length - maxError ∧
    actualDimensions.width ≤ measuredDimensions.width + maxError ∧
    actualDimensions.width ≥ measuredDimensions.width - maxError ∧
    actualDimensions.height ≤ measuredDimensions.height + maxError ∧
    actualDimensions.height ≥ measuredDimensions.height - maxError ∧
    (boxVolume actualDimensions - boxVolume measuredDimensions) ≤ 90726 ∧
    ∀ (d : BoxDimensions),
      d.length ≤ measuredDimensions.length + maxError →
      d.length ≥ measuredDimensions.length - maxError →
      d.width ≤ measuredDimensions.width + maxError →
      d.width ≥ measuredDimensions.width - maxError →
      d.height ≤ measuredDimensions.height + maxError →
      d.height ≥ measuredDimensions.height - maxError →
      (boxVolume d - boxVolume measuredDimensions) ≤ 90726 :=
by sorry

end NUMINAMATH_CALUDE_max_volume_difference_l1191_119177


namespace NUMINAMATH_CALUDE_inverse_composition_theorem_l1191_119127

-- Define the functions f and g
variables (f g : ℝ → ℝ)

-- Define the condition f⁻¹ ∘ g = 3x - 2
def condition (f g : ℝ → ℝ) : Prop :=
  ∀ x, (f⁻¹ ∘ g) x = 3 * x - 2

-- Theorem statement
theorem inverse_composition_theorem (hfg : condition f g) :
  g⁻¹ (f (-10)) = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_theorem_l1191_119127


namespace NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l1191_119189

/-- Represents a tennis tournament with teams of women -/
structure WomensTennisTournament where
  total_teams : Nat
  women_per_team : Nat
  participating_teams : Nat

/-- Calculates the number of handshakes in the tournament -/
def calculate_handshakes (tournament : WomensTennisTournament) : Nat :=
  let total_women := tournament.participating_teams * tournament.women_per_team
  let handshakes_per_woman := (tournament.participating_teams - 1) * tournament.women_per_team
  (total_women * handshakes_per_woman) / 2

/-- Theorem stating the number of handshakes in the specific tournament scenario -/
theorem handshakes_in_specific_tournament :
  let tournament := WomensTennisTournament.mk 4 2 3
  calculate_handshakes tournament = 12 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_in_specific_tournament_l1191_119189


namespace NUMINAMATH_CALUDE_function_inequality_l1191_119134

/-- Given a differentiable function f: ℝ → ℝ, if f(x) + f''(x) < 0 for all x,
    then f(1) < f(0)/e < f(-1)/(e^2) -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (hf'' : Differentiable ℝ (deriv (deriv f)))
    (h : ∀ x, f x + (deriv (deriv f)) x < 0) :
    f 1 < f 0 / Real.exp 1 ∧ f 0 / Real.exp 1 < f (-1) / (Real.exp 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l1191_119134


namespace NUMINAMATH_CALUDE_visitor_growth_l1191_119166

/-- Represents the growth of visitors at a tourist attraction from January to March. -/
theorem visitor_growth (initial_visitors final_visitors : ℕ) (x : ℝ) :
  initial_visitors = 60000 →
  final_visitors = 150000 →
  (initial_visitors : ℝ) / 10000 * (1 + x)^2 = (final_visitors : ℝ) / 10000 →
  6 * (1 + x)^2 = 15 := by
  sorry

#check visitor_growth

end NUMINAMATH_CALUDE_visitor_growth_l1191_119166


namespace NUMINAMATH_CALUDE_negative_a_squared_cubed_div_negative_a_squared_l1191_119170

theorem negative_a_squared_cubed_div_negative_a_squared (a : ℝ) :
  (-a^2)^3 / (-a)^2 = -a^4 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_cubed_div_negative_a_squared_l1191_119170


namespace NUMINAMATH_CALUDE_system_solution_unique_l1191_119130

theorem system_solution_unique : 
  ∃! (x y : ℝ), (2 * x + y = 4) ∧ (x - y = -1) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l1191_119130


namespace NUMINAMATH_CALUDE_max_constant_C_l1191_119195

theorem max_constant_C : ∃ (C : ℝ), C = Real.sqrt 2 ∧
  (∀ (x y : ℝ), x^2 + y^2 + 1 ≥ C*(x + y)) ∧
  (∀ (x y : ℝ), x^2 + y^2 + x*y + 1 ≥ C*(x + y)) ∧
  (∀ (C' : ℝ), C' > C →
    (∃ (x y : ℝ), x^2 + y^2 + 1 < C'*(x + y) ∨ x^2 + y^2 + x*y + 1 < C'*(x + y))) := by
  sorry

end NUMINAMATH_CALUDE_max_constant_C_l1191_119195


namespace NUMINAMATH_CALUDE_min_draws_for_all_colors_l1191_119105

theorem min_draws_for_all_colors (white black yellow : ℕ) 
  (hw : white = 8) (hb : black = 9) (hy : yellow = 7) :
  (white + black + yellow - (white + black - 1)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_all_colors_l1191_119105


namespace NUMINAMATH_CALUDE_major_axis_length_l1191_119163

/-- Represents a right circular cylinder. -/
structure RightCircularCylinder where
  radius : ℝ

/-- Represents an ellipse formed by the intersection of a plane and a cylinder. -/
structure Ellipse where
  minorAxis : ℝ
  majorAxis : ℝ

/-- The ellipse formed by the intersection of a plane and a right circular cylinder. -/
def intersectionEllipse (c : RightCircularCylinder) : Ellipse where
  minorAxis := 2 * c.radius
  majorAxis := 2 * c.radius * 1.5

theorem major_axis_length 
  (c : RightCircularCylinder) 
  (h : c.radius = 1) :
  (intersectionEllipse c).majorAxis = 3 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_l1191_119163


namespace NUMINAMATH_CALUDE_less_likely_white_ball_l1191_119128

theorem less_likely_white_ball (red_balls white_balls : ℕ) 
  (h_red : red_balls = 8) (h_white : white_balls = 2) :
  (white_balls : ℚ) / (red_balls + white_balls) < (red_balls : ℚ) / (red_balls + white_balls) :=
by sorry

end NUMINAMATH_CALUDE_less_likely_white_ball_l1191_119128


namespace NUMINAMATH_CALUDE_factorization_equality_l1191_119125

theorem factorization_equality (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1191_119125


namespace NUMINAMATH_CALUDE_permutation_expressions_l1191_119164

open Nat

-- Define the permutation function A
def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

-- Theorem statement
theorem permutation_expressions (n : ℕ) : 
  (A (n + 1) n ≠ factorial n) ∧ 
  ((1 / (n + 1 : ℚ)) * A (n + 1) (n + 1) = factorial n) ∧
  (A n n = factorial n) ∧
  (n * A (n - 1) (n - 1) = factorial n) :=
sorry

end NUMINAMATH_CALUDE_permutation_expressions_l1191_119164


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l1191_119119

theorem quadratic_roots_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2*x₁ ∨ x = 2*x₂)) →
  a / c = 1 / 8 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_relation_l1191_119119


namespace NUMINAMATH_CALUDE_subset_proof_l1191_119154

def M : Set ℕ := {1}
def N : Set ℕ := {1, 2, 3}

theorem subset_proof : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_subset_proof_l1191_119154


namespace NUMINAMATH_CALUDE_point_P_in_second_quadrant_l1191_119173

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : CartesianPoint) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: Point P(-3, 2) is in the second quadrant -/
theorem point_P_in_second_quadrant :
  let P : CartesianPoint := ⟨-3, 2⟩
  is_in_second_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_point_P_in_second_quadrant_l1191_119173


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1191_119169

theorem geometric_sequence_common_ratio 
  (a : ℝ) : 
  let seq := λ (n : ℕ) => a + Real.log 3 / Real.log (2^(2^n))
  ∃ (q : ℝ), q = 1/3 ∧ ∀ (n : ℕ), seq (n+1) / seq n = q :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1191_119169


namespace NUMINAMATH_CALUDE_third_derivative_y_l1191_119147

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = 4 / (1 + x^2)^2 := by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l1191_119147


namespace NUMINAMATH_CALUDE_notched_circle_distance_l1191_119122

/-- Given a circle with radius √75 and a point B such that there exist points A and C on the circle
    where AB = 8, BC = 2, and angle ABC is a right angle, prove that the square of the distance
    from B to the center of the circle is 122. -/
theorem notched_circle_distance (O A B C : ℝ × ℝ) : 
  (∀ P : ℝ × ℝ, (P.1 - O.1)^2 + (P.2 - O.2)^2 = 75 → P = A ∨ P = C) →  -- A and C are on the circle
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64 →  -- AB = 8
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 →   -- BC = 2
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0 →  -- Angle ABC is right angle
  (B.1 - O.1)^2 + (B.2 - O.2)^2 = 122 := by
sorry


end NUMINAMATH_CALUDE_notched_circle_distance_l1191_119122


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1191_119158

theorem invalid_external_diagonals : ¬ ∃ (a b c : ℝ),
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (a^2 + b^2 = 5^2 ∧ b^2 + c^2 = 6^2 ∧ a^2 + c^2 = 8^2) :=
sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1191_119158


namespace NUMINAMATH_CALUDE_john_earnings_calculation_l1191_119142

/-- Calculates John's weekly earnings after fees and taxes --/
def johnWeeklyEarnings : ℝ :=
  let streamingHours : ℕ := 4
  let mondayRate : ℝ := 10
  let wednesdayRate : ℝ := 12
  let fridayRate : ℝ := 15
  let saturdayRate : ℝ := 20
  let platformFeeRate : ℝ := 0.20
  let taxRate : ℝ := 0.25

  let grossEarnings : ℝ := streamingHours * (mondayRate + wednesdayRate + fridayRate + saturdayRate)
  let platformFee : ℝ := grossEarnings * platformFeeRate
  let netEarningsBeforeTax : ℝ := grossEarnings - platformFee
  let tax : ℝ := netEarningsBeforeTax * taxRate
  netEarningsBeforeTax - tax

theorem john_earnings_calculation :
  johnWeeklyEarnings = 136.80 := by sorry

end NUMINAMATH_CALUDE_john_earnings_calculation_l1191_119142


namespace NUMINAMATH_CALUDE_slipper_cost_l1191_119178

theorem slipper_cost (total_items : ℕ) (slipper_count : ℕ) (lipstick_count : ℕ) (lipstick_price : ℚ) 
  (hair_color_count : ℕ) (hair_color_price : ℚ) (total_paid : ℚ) :
  total_items = slipper_count + lipstick_count + hair_color_count →
  total_items = 18 →
  slipper_count = 6 →
  lipstick_count = 4 →
  lipstick_price = 5/4 →
  hair_color_count = 8 →
  hair_color_price = 3 →
  total_paid = 44 →
  (total_paid - (lipstick_count * lipstick_price + hair_color_count * hair_color_price)) / slipper_count = 5/2 :=
by
  sorry

#check slipper_cost

end NUMINAMATH_CALUDE_slipper_cost_l1191_119178


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_symmetry_l1191_119167

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in row n of Pascal's triangle -/
def rowLength (n : ℕ) : ℕ := n + 1

theorem pascal_triangle_row20_symmetry :
  let n := 20
  let k := 5
  let row_length := rowLength n
  binomial n (k - 1) = binomial n (row_length - k) ∧
  binomial n (k - 1) = 4845 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_symmetry_l1191_119167


namespace NUMINAMATH_CALUDE_pentagon_area_l1191_119151

/-- Given integers p and q where 0 < q < p, and points P, Q, R, S, T defined by reflections,
    if the area of pentagon PQRST is 700, then 5pq - q² = 700 -/
theorem pentagon_area (p q : ℤ) (h1 : 0 < q) (h2 : q < p) 
  (h3 : (5 * p * q - q^2 : ℤ) = 700) : True := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_l1191_119151


namespace NUMINAMATH_CALUDE_root_in_interval_l1191_119194

noncomputable def f (x : ℝ) := 2^x - 3*x

theorem root_in_interval :
  ∃ (r : ℝ), r ∈ Set.Ioo 3 4 ∧ f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1191_119194


namespace NUMINAMATH_CALUDE_aladdin_journey_theorem_l1191_119191

-- Define the circle (equator)
def Equator : Real := 40000

-- Define Aladdin's path
def AladdinPath : Set ℝ → Prop :=
  λ path => ∀ x, x ∈ path → 0 ≤ x ∧ x < Equator

-- Define the property of covering every point on the equator
def CoversEquator (path : Set ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x < Equator → ∃ y ∈ path, y % Equator = x

-- Define the westward travel limit
def WestwardLimit : Real := 19000

-- Define the theorem
theorem aladdin_journey_theorem (path : Set ℝ) 
  (h_path : AladdinPath path)
  (h_covers : CoversEquator path)
  (h_westward : ∀ x ∈ path, x ≤ WestwardLimit) :
  ∃ x ∈ path, abs (x % Equator - x) ≥ Equator / 2 := by
sorry

end NUMINAMATH_CALUDE_aladdin_journey_theorem_l1191_119191


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_second_greater_first_l1191_119198

/-- A geometric sequence with positive first term -/
structure GeometricSequence where
  a : ℕ → ℝ
  first_positive : a 1 > 0
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n

/-- An increasing sequence -/
def IsIncreasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n + 1) > s n

theorem geometric_sequence_increasing_iff_second_greater_first (seq : GeometricSequence) :
  (seq.a 2 > seq.a 1) ↔ IsIncreasing seq.a :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_iff_second_greater_first_l1191_119198


namespace NUMINAMATH_CALUDE_inequality_necessary_not_sufficient_l1191_119120

/-- Predicate to check if the equation represents an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3

/-- The given inequality condition -/
def inequality_condition (m : ℝ) : Prop :=
  -3 < m ∧ m < 5

theorem inequality_necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → inequality_condition m) ∧
  ¬(∀ m : ℝ, inequality_condition m → is_ellipse m) :=
by sorry

end NUMINAMATH_CALUDE_inequality_necessary_not_sufficient_l1191_119120


namespace NUMINAMATH_CALUDE_no_two_obtuse_angles_l1191_119176

-- Define a triangle as a structure with three angles
structure Triangle where
  a : Real
  b : Real
  c : Real
  angle_sum : a + b + c = 180
  positive_angles : 0 < a ∧ 0 < b ∧ 0 < c

-- Theorem: A triangle cannot have two obtuse angles
theorem no_two_obtuse_angles (t : Triangle) : ¬(t.a > 90 ∧ t.b > 90) ∧ ¬(t.a > 90 ∧ t.c > 90) ∧ ¬(t.b > 90 ∧ t.c > 90) := by
  sorry

end NUMINAMATH_CALUDE_no_two_obtuse_angles_l1191_119176


namespace NUMINAMATH_CALUDE_excess_cans_l1191_119107

def initial_collection : ℕ := 30 + 43 + 55
def daily_collection_rate : ℕ := 8 + 11 + 15
def days : ℕ := 14
def goal : ℕ := 400

theorem excess_cans :
  initial_collection + daily_collection_rate * days - goal = 204 :=
by sorry

end NUMINAMATH_CALUDE_excess_cans_l1191_119107


namespace NUMINAMATH_CALUDE_dog_tail_length_l1191_119116

theorem dog_tail_length (body_length : ℝ) (head_length : ℝ) (tail_length : ℝ) 
  (overall_length : ℝ) (width : ℝ) (height : ℝ) :
  tail_length = body_length / 2 →
  head_length = body_length / 6 →
  height = 1.5 * width →
  overall_length = 30 →
  width = 12 →
  overall_length = body_length + head_length + tail_length →
  tail_length = 15 := by
  sorry

end NUMINAMATH_CALUDE_dog_tail_length_l1191_119116


namespace NUMINAMATH_CALUDE_min_value_a_l1191_119172

theorem min_value_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1191_119172


namespace NUMINAMATH_CALUDE_percentage_problem_l1191_119112

theorem percentage_problem (x : ℝ) : (0.15 * 0.30 * 0.50 * x = 90) → x = 4000 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1191_119112


namespace NUMINAMATH_CALUDE_product_congruence_l1191_119133

theorem product_congruence : 65 * 76 * 87 ≡ 5 [ZMOD 25] := by sorry

end NUMINAMATH_CALUDE_product_congruence_l1191_119133


namespace NUMINAMATH_CALUDE_chess_tournament_girls_l1191_119165

theorem chess_tournament_girls (n : ℕ) (x : ℕ) : 
  (n > 0) →  -- number of girls is positive
  (2 * n * x + 16 = (n + 2) * (n + 1)) →  -- total points equation
  (x > 0) →  -- each girl's score is positive
  (n = 7 ∨ n = 14) := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_girls_l1191_119165


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1191_119148

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arithmetic : is_arithmetic_sequence a d)
  (h_sum : a 1 + a 3 + a 5 = 15)
  (h_a4 : a 4 = 3) :
  d = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1191_119148


namespace NUMINAMATH_CALUDE_max_values_on_sphere_l1191_119162

theorem max_values_on_sphere (x y z : ℝ) :
  x^2 + y^2 + z^2 = 4 →
  (∃ (max_xz_yz : ℝ), ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 4 → x' * z' + y' * z' ≤ max_xz_yz ∧ max_xz_yz = 2 * Real.sqrt 2) ∧
  (x + y + z = 0 →
    ∃ (max_z : ℝ), ∀ (x' y' z' : ℝ), x'^2 + y'^2 + z'^2 = 4 ∧ x' + y' + z' = 0 → z' ≤ max_z ∧ max_z = (2 * Real.sqrt 6) / 3) := by
  sorry

end NUMINAMATH_CALUDE_max_values_on_sphere_l1191_119162


namespace NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l1191_119157

/-- An infinite sequence of positive integers where each element is strictly greater than the previous one. -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, a k < a (k + 1)

/-- The property that an element of the sequence can be expressed as a linear combination of two distinct earlier elements. -/
def CanBeExpressedAsLinearCombination (a : ℕ → ℕ) (m p q : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ p ≠ q ∧ a m = x * a p + y * a q

/-- The main theorem stating that infinitely many elements of the sequence can be expressed as linear combinations of two distinct earlier elements. -/
theorem infinitely_many_linear_combinations (a : ℕ → ℕ) 
    (h : StrictlyIncreasingSequence a) :
    ∀ N, ∃ m, m > N ∧ ∃ p q, CanBeExpressedAsLinearCombination a m p q := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_linear_combinations_l1191_119157


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1191_119132

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 7*a + 7 = 0) → (b^2 - 7*b + 7 = 0) → a^2 + b^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1191_119132


namespace NUMINAMATH_CALUDE_dice_probability_l1191_119126

theorem dice_probability : 
  let n : ℕ := 5  -- number of dice
  let s : ℕ := 12  -- number of sides on each die
  let p_one_digit : ℚ := 3 / 4  -- probability of rolling a one-digit number
  let p_two_digit : ℚ := 1 / 4  -- probability of rolling a two-digit number
  Nat.choose n (n / 2) * p_two_digit ^ (n / 2) * p_one_digit ^ (n - n / 2) = 135 / 512 :=
by sorry

end NUMINAMATH_CALUDE_dice_probability_l1191_119126


namespace NUMINAMATH_CALUDE_min_value_theorem_l1191_119180

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m > 0) (h3 : n > 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 2 → (1 / m + 2 / n) ≤ (1 / x + 2 / y) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2 * a + b = 2 ∧ 1 / a + 2 / b = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1191_119180


namespace NUMINAMATH_CALUDE_perpendicular_sum_difference_l1191_119199

/-- Given unit vectors a and b in the plane, prove that (a + b) is perpendicular to (a - b) -/
theorem perpendicular_sum_difference (a b : ℝ × ℝ) 
  (ha : a = (5/13, 12/13)) 
  (hb : b = (4/5, 3/5)) 
  (unit_a : a.1^2 + a.2^2 = 1) 
  (unit_b : b.1^2 + b.2^2 = 1) : 
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2) = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_sum_difference_l1191_119199


namespace NUMINAMATH_CALUDE_digit_150_is_zero_l1191_119115

/-- The decimal representation of 16/81 -/
def decimal_rep : ℚ := 16 / 81

/-- The repeating cycle in the decimal representation of 16/81 -/
def cycle : List ℕ := [1, 9, 7, 5, 3, 0, 8, 6, 4]

/-- The length of the repeating cycle -/
def cycle_length : ℕ := 9

/-- The position of the 150th digit within the cycle -/
def position_in_cycle : ℕ := 150 % cycle_length

/-- The 150th digit after the decimal point in the decimal representation of 16/81 -/
def digit_150 : ℕ := cycle[position_in_cycle]

theorem digit_150_is_zero : digit_150 = 0 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_zero_l1191_119115


namespace NUMINAMATH_CALUDE_blackboard_erasers_l1191_119143

theorem blackboard_erasers (erasers_per_class : ℕ) 
                            (broken_erasers : ℕ) 
                            (remaining_erasers : ℕ) : 
  erasers_per_class = 3 →
  broken_erasers = 12 →
  remaining_erasers = 60 →
  (((remaining_erasers + broken_erasers) / erasers_per_class) / 3) + 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_erasers_l1191_119143


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1191_119185

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 5

/-- The square is inscribed in the region bound by the parabola and x-axis -/
def is_inscribed_square (s : ℝ) : Prop :=
  ∃ (center : ℝ), 
    parabola (center - s) = 0 ∧
    parabola (center + s) = 0 ∧
    parabola (center + s) = 2*s

/-- The theorem stating the area of the inscribed square -/
theorem inscribed_square_area :
  ∃ (s : ℝ), is_inscribed_square s ∧ (2*s)^2 = 64 - 16*Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1191_119185


namespace NUMINAMATH_CALUDE_factors_of_2310_l1191_119109

theorem factors_of_2310 : Finset.card (Nat.divisors 2310) = 32 := by sorry

end NUMINAMATH_CALUDE_factors_of_2310_l1191_119109


namespace NUMINAMATH_CALUDE_inequality_proof_l1191_119114

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1191_119114


namespace NUMINAMATH_CALUDE_flight_time_theorem_l1191_119196

/-- Represents the flight scenario between two towns -/
structure FlightScenario where
  d : ℝ  -- distance between towns
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  time_against : ℝ  -- time for flight against wind
  time_diff : ℝ  -- difference in time between with-wind and still air flights

/-- The theorem stating the conditions and the result to be proved -/
theorem flight_time_theorem (scenario : FlightScenario) 
  (h1 : scenario.time_against = 84)
  (h2 : scenario.time_diff = 9)
  (h3 : scenario.d = scenario.time_against * (scenario.p - scenario.w))
  (h4 : scenario.d / (scenario.p + scenario.w) = scenario.d / scenario.p - scenario.time_diff) :
  scenario.d / (scenario.p + scenario.w) = 63 ∨ scenario.d / (scenario.p + scenario.w) = 12 := by
  sorry

end NUMINAMATH_CALUDE_flight_time_theorem_l1191_119196


namespace NUMINAMATH_CALUDE_apples_left_after_pies_l1191_119183

theorem apples_left_after_pies (initial_apples : ℕ) (difference : ℕ) (apples_left : ℕ) : 
  initial_apples = 46 → difference = 32 → apples_left = initial_apples - difference → apples_left = 14 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_after_pies_l1191_119183


namespace NUMINAMATH_CALUDE_min_value_parallel_vectors_l1191_119161

/-- Given parallel vectors a and b, prove the minimum value of 3/x + 2/y is 8 -/
theorem min_value_parallel_vectors (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, y - 1]
  (∃ (k : ℝ), b = k • a) →
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → 3 / x' + 2 / y' ≥ 8) ∧
  (∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_parallel_vectors_l1191_119161


namespace NUMINAMATH_CALUDE_y_investment_is_75000_l1191_119174

/-- Represents the investment and profit scenario of a business --/
structure BusinessScenario where
  x_investment : ℕ
  z_investment : ℕ
  z_join_time : ℕ
  z_profit_share : ℕ
  total_profit : ℕ
  total_duration : ℕ

/-- Calculates Y's investment given a business scenario --/
def calculate_y_investment (scenario : BusinessScenario) : ℕ :=
  sorry

/-- Theorem stating that Y's investment is 75000 for the given scenario --/
theorem y_investment_is_75000 (scenario : BusinessScenario) 
  (h1 : scenario.x_investment = 36000)
  (h2 : scenario.z_investment = 48000)
  (h3 : scenario.z_join_time = 4)
  (h4 : scenario.z_profit_share = 4064)
  (h5 : scenario.total_profit = 13970)
  (h6 : scenario.total_duration = 12) :
  calculate_y_investment scenario = 75000 :=
sorry

end NUMINAMATH_CALUDE_y_investment_is_75000_l1191_119174


namespace NUMINAMATH_CALUDE_set_operations_l1191_119135

/-- Given a universal set and two of its subsets, prove various set operations -/
theorem set_operations (U A B : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3})
  (hB : B = {2, 5}) :
  (U \ A = {2, 4, 5}) ∧
  (A ∩ B = ∅) ∧
  (A ∪ B = {1, 2, 3, 5}) ∧
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 4, 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1191_119135


namespace NUMINAMATH_CALUDE_base9_repeating_fraction_l1191_119149

/-- Represents a digit in base-9 system -/
def Base9Digit := Fin 9

/-- Converts a base-10 number to its base-9 representation -/
def toBase9 (n : ℚ) : List Base9Digit :=
  sorry

/-- Checks if a list of digits is repeating -/
def isRepeating (l : List Base9Digit) : Prop :=
  sorry

/-- The main theorem -/
theorem base9_repeating_fraction :
  ∃ (n d : ℕ) (l : List Base9Digit),
    n ≠ 0 ∧ d ≠ 0 ∧
    (n : ℚ) / d < 1 / 2 ∧
    isRepeating (toBase9 ((n : ℚ) / d)) ∧
    n = 13 ∧ d = 37 :=
  sorry

end NUMINAMATH_CALUDE_base9_repeating_fraction_l1191_119149


namespace NUMINAMATH_CALUDE_third_side_of_similar_altitude_triangle_l1191_119104

/-- A triangle with sides a, b, and c, where the triangle is similar to the triangle formed by its altitudes. -/
structure SimilarAltitudeTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  similar_to_altitude : a * b * c = 2 * (a^2 + b^2 + c^2)

/-- Theorem: In a triangle similar to its altitude triangle with two sides 9 and 4, the third side is 6. -/
theorem third_side_of_similar_altitude_triangle :
  ∀ (t : SimilarAltitudeTriangle), t.a = 9 → t.b = 4 → t.c = 6 := by
  sorry

#check third_side_of_similar_altitude_triangle

end NUMINAMATH_CALUDE_third_side_of_similar_altitude_triangle_l1191_119104


namespace NUMINAMATH_CALUDE_product_greater_than_sum_l1191_119155

theorem product_greater_than_sum (a b : ℝ) (ha : a > 2) (hb : b > 2) : a * b > a + b := by
  sorry

end NUMINAMATH_CALUDE_product_greater_than_sum_l1191_119155


namespace NUMINAMATH_CALUDE_train_length_is_500_l1191_119138

/-- The length of a train that passes a pole in 50 seconds and a 500 m long platform in 100 seconds -/
def train_length : ℝ := by sorry

/-- The time it takes for the train to pass a pole -/
def pole_passing_time : ℝ := 50

/-- The time it takes for the train to pass a platform -/
def platform_passing_time : ℝ := 100

/-- The length of the platform -/
def platform_length : ℝ := 500

theorem train_length_is_500 :
  train_length = 500 :=
by
  have h1 : train_length / pole_passing_time = (train_length + platform_length) / platform_passing_time :=
    by sorry
  sorry

#check train_length_is_500

end NUMINAMATH_CALUDE_train_length_is_500_l1191_119138


namespace NUMINAMATH_CALUDE_reach_11_from_1_l1191_119129

/-- Represents the set of operations allowed by the calculator -/
inductive CalcOp
  | mul3 : CalcOp  -- Multiply by 3
  | add3 : CalcOp  -- Add 3
  | div3 : CalcOp  -- Divide by 3 (when divisible)

/-- Applies a single calculator operation to a number -/
def applyOp (n : ℕ) (op : CalcOp) : ℕ :=
  match op with
  | CalcOp.mul3 => n * 3
  | CalcOp.add3 => n + 3
  | CalcOp.div3 => if n % 3 = 0 then n / 3 else n

/-- Checks if it's possible to reach the target number from the start number using the given operations -/
def canReach (start target : ℕ) : Prop :=
  ∃ (steps : List CalcOp), (steps.foldl applyOp start) = target

/-- Theorem stating that it's possible to reach 11 from 1 using the calculator operations -/
theorem reach_11_from_1 : canReach 1 11 := by
  sorry

end NUMINAMATH_CALUDE_reach_11_from_1_l1191_119129


namespace NUMINAMATH_CALUDE_equation_has_three_solutions_l1191_119190

/-- The number of distinct complex solutions to the equation (z^4 - 1) / (z^3 - 3z + 2) = 0 -/
def num_solutions : ℕ := 3

/-- The equation (z^4 - 1) / (z^3 - 3z + 2) = 0 -/
def equation (z : ℂ) : Prop :=
  (z^4 - 1) / (z^3 - 3*z + 2) = 0

theorem equation_has_three_solutions :
  ∃ (S : Finset ℂ), S.card = num_solutions ∧
    (∀ z ∈ S, equation z) ∧
    (∀ z : ℂ, equation z → z ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_three_solutions_l1191_119190


namespace NUMINAMATH_CALUDE_sum_in_base8_l1191_119175

/-- Converts a base-8 number represented as a list of digits to a natural number. -/
def base8ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits. -/
def natToBase8 (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: natToBase8 (n / 8)

theorem sum_in_base8 :
  let a := base8ToNat [3, 5, 6]  -- 653₈
  let b := base8ToNat [4, 7, 2]  -- 274₈
  let c := base8ToNat [7, 6, 1]  -- 167₈
  natToBase8 (a + b + c) = [6, 5, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base8_l1191_119175


namespace NUMINAMATH_CALUDE_isabella_sam_difference_l1191_119152

/-- The amount of money each person has -/
structure Money where
  isabella : ℕ
  sam : ℕ
  giselle : ℕ

/-- The conditions of the problem -/
def ProblemConditions (m : Money) : Prop :=
  m.isabella > m.sam ∧
  m.isabella = m.giselle + 15 ∧
  m.giselle = 120 ∧
  (m.isabella + m.sam + m.giselle) / 3 = 115

theorem isabella_sam_difference (m : Money) 
  (h : ProblemConditions m) : m.isabella - m.sam = 45 := by
  sorry

end NUMINAMATH_CALUDE_isabella_sam_difference_l1191_119152


namespace NUMINAMATH_CALUDE_counterexample_exists_l1191_119184

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ Nat.Prime (n + 2) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1191_119184


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l1191_119137

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the origin -/
def symmetricToOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The point symmetric to (2, -3) with respect to the origin is (-2, 3) -/
theorem symmetric_point_theorem :
  let P : Point := { x := 2, y := -3 }
  let P' : Point := symmetricToOrigin P
  P'.x = -2 ∧ P'.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l1191_119137


namespace NUMINAMATH_CALUDE_geometric_progression_middle_term_l1191_119121

theorem geometric_progression_middle_term :
  ∀ m : ℝ,
  (∃ r : ℝ, (m / (5 + 2 * Real.sqrt 6) = r) ∧ ((5 - 2 * Real.sqrt 6) / m = r)) →
  (m = 1 ∨ m = -1) :=
λ m h => by sorry

end NUMINAMATH_CALUDE_geometric_progression_middle_term_l1191_119121


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l1191_119139

/-- A decagon is a polygon with 10 vertices -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices : ℕ := 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def adjacent_vertices : ℕ := 2

/-- The probability of choosing two adjacent vertices when randomly selecting 2 distinct vertices from a decagon -/
theorem adjacent_vertices_probability (d : Decagon) : ℚ :=
  2 / 9

/-- Proof of the theorem -/
lemma adjacent_vertices_probability_proof (d : Decagon) : 
  adjacent_vertices_probability d = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l1191_119139


namespace NUMINAMATH_CALUDE_adjacent_product_sum_nonpositive_l1191_119131

theorem adjacent_product_sum_nonpositive (a b c d : ℝ) (h : a + b + c + d = 0) :
  a * b + b * c + c * d + d * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_product_sum_nonpositive_l1191_119131


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1191_119101

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_180_l1191_119101


namespace NUMINAMATH_CALUDE_cubic_equation_with_complex_root_l1191_119186

theorem cubic_equation_with_complex_root (k : ℝ) : 
  (∃ (z : ℂ), z^3 + 2*(k-1)*z^2 + 9*z + 5*(k-1) = 0 ∧ Complex.abs z = Real.sqrt 5) →
  (k = -1 ∨ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_with_complex_root_l1191_119186


namespace NUMINAMATH_CALUDE_unique_solution_power_of_two_l1191_119117

theorem unique_solution_power_of_two (a b m : ℕ) : 
  a > 0 → b > 0 → (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_power_of_two_l1191_119117


namespace NUMINAMATH_CALUDE_max_x_minus_y_l1191_119179

-- Define the condition function
def condition (x y : ℝ) : Prop := 3 * (x^2 + y^2) = x^2 + y

-- Define the objective function
def objective (x y : ℝ) : ℝ := x - y

-- Theorem statement
theorem max_x_minus_y :
  ∃ (max : ℝ), max = 1 / Real.sqrt 24 ∧
  ∀ (x y : ℝ), condition x y → objective x y ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l1191_119179


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l1191_119113

theorem fourth_root_simplification (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (2^7 * 3^3 : ℚ)^(1/4) = a * b^(1/4) → a + b = 218 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l1191_119113


namespace NUMINAMATH_CALUDE_special_triangle_ratio_constant_l1191_119141

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = c^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = a^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = b^2

-- Define the property AC^2 + BC^2 = 2 AB^2
def SpecialTriangleProperty (A B C : ℝ × ℝ) : Prop :=
  (C.1 - A.1)^2 + (C.2 - A.2)^2 + (C.1 - B.1)^2 + (C.2 - B.2)^2 = 
  2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define point M as the midpoint of AB
def Midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Define the angle equality ∠ACD = ∠BCD
def EqualAngles (A B C D : ℝ × ℝ) : Prop :=
  let v1 := (A.1 - C.1, A.2 - C.2)
  let v2 := (D.1 - C.1, D.2 - C.2)
  let v3 := (B.1 - C.1, B.2 - C.2)
  (v1.1 * v2.1 + v1.2 * v2.2)^2 / ((v1.1^2 + v1.2^2) * (v2.1^2 + v2.2^2)) =
  (v3.1 * v2.1 + v3.2 * v2.2)^2 / ((v3.1^2 + v3.2^2) * (v2.1^2 + v2.2^2))

-- Define D as the incenter of triangle CEM
def Incenter (D C E M : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = r^2 ∧
  (D.1 - E.1)^2 + (D.2 - E.2)^2 = r^2 ∧
  (D.1 - M.1)^2 + (D.2 - M.2)^2 = r^2

-- Main theorem
theorem special_triangle_ratio_constant 
  (A B C M D E : ℝ × ℝ) :
  Triangle A B C →
  SpecialTriangleProperty A B C →
  Midpoint M A B →
  EqualAngles A B C D →
  Incenter D C E M →
  (E.1 - M.1)^2 + (E.2 - M.2)^2 = 
  (1/9) * ((M.1 - C.1)^2 + (M.2 - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_ratio_constant_l1191_119141


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_abs_sum_inequality_equivalence_l1191_119124

-- Question 1
theorem abs_inequality_equivalence (x : ℝ) :
  |x - 2| < |x + 1| ↔ x > 1/2 := by sorry

-- Question 2
theorem abs_sum_inequality_equivalence (x : ℝ) :
  |2*x + 1| + |x - 2| > 4 ↔ x < -1 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_abs_sum_inequality_equivalence_l1191_119124


namespace NUMINAMATH_CALUDE_multiple_of_second_number_l1191_119182

theorem multiple_of_second_number (x y m : ℤ) : 
  y = m * x + 3 → 
  x + y = 27 → 
  y = 19 → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_second_number_l1191_119182


namespace NUMINAMATH_CALUDE_sqrt_three_square_form_l1191_119103

theorem sqrt_three_square_form (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = m^2 + 3 * n^2 ∧ b = 2 * m * n :=
by sorry

end NUMINAMATH_CALUDE_sqrt_three_square_form_l1191_119103


namespace NUMINAMATH_CALUDE_total_savings_l1191_119160

def holiday_savings (sam victory alex : ℕ) : Prop :=
  victory = sam - 200 ∧ alex = 2 * victory ∧ sam = 1200

theorem total_savings (sam victory alex : ℕ) 
  (h : holiday_savings sam victory alex) : 
  sam + victory + alex = 4200 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_l1191_119160


namespace NUMINAMATH_CALUDE_draw_balls_count_l1191_119193

/-- The number of ways to draw 4 balls from 20 balls numbered 1 through 20,
    where the sum of the first and last ball drawn is 21. -/
def draw_balls : ℕ :=
  let total_balls : ℕ := 20
  let balls_drawn : ℕ := 4
  let sum_first_last : ℕ := 21
  let valid_first_balls : ℕ := sum_first_last - 1
  let remaining_choices : ℕ := total_balls - 2
  valid_first_balls * remaining_choices * (remaining_choices - 1)

theorem draw_balls_count : draw_balls = 3060 := by
  sorry

end NUMINAMATH_CALUDE_draw_balls_count_l1191_119193


namespace NUMINAMATH_CALUDE_chicken_feathers_after_crossing_l1191_119100

/-- Represents the number of feathers a chicken has after crossing a road twice -/
def feathers_after_crossing (initial_feathers : ℕ) (cars_dodged : ℕ) : ℕ :=
  initial_feathers - 2 * cars_dodged

/-- Theorem stating the number of feathers remaining after the chicken's adventure -/
theorem chicken_feathers_after_crossing :
  feathers_after_crossing 5263 23 = 5217 := by
  sorry

#eval feathers_after_crossing 5263 23

end NUMINAMATH_CALUDE_chicken_feathers_after_crossing_l1191_119100


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l1191_119102

theorem intersection_x_coordinate : 
  let line1 : ℝ → ℝ := λ x => 3 * x + 5
  let line2 : ℝ → ℝ := λ x => 35 - 5 * x
  ∃ x : ℝ, x = 15 / 4 ∧ line1 x = line2 x :=
by sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l1191_119102


namespace NUMINAMATH_CALUDE_valid_license_plates_l1191_119140

/-- The number of valid English letters for the license plate. --/
def validLetters : Nat := 24

/-- The number of positions to choose from for placing the letters. --/
def positionsForLetters : Nat := 4

/-- The number of letter positions to fill. --/
def letterPositions : Nat := 2

/-- The number of digit positions to fill. --/
def digitPositions : Nat := 3

/-- The number of possible digits (0-9). --/
def possibleDigits : Nat := 10

/-- Calculates the number of ways to choose k items from n items. --/
def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of valid license plate combinations. --/
def totalCombinations : Nat :=
  choose positionsForLetters letterPositions * 
  validLetters ^ letterPositions * 
  possibleDigits ^ digitPositions

/-- Theorem stating that the total number of valid license plate combinations is 3,456,000. --/
theorem valid_license_plates : totalCombinations = 3456000 := by
  sorry

end NUMINAMATH_CALUDE_valid_license_plates_l1191_119140


namespace NUMINAMATH_CALUDE_one_square_covered_l1191_119159

/-- Represents a square on the checkerboard -/
structure Square where
  x : ℕ
  y : ℕ

/-- Represents the circular disc -/
structure Disc where
  center : Square
  diameter : ℝ

/-- Determines if a square is completely covered by the disc -/
def is_covered (s : Square) (d : Disc) : Prop :=
  (s.x - d.center.x)^2 + (s.y - d.center.y)^2 ≤ (d.diameter / 2)^2

/-- The checkerboard -/
def checkerboard : Set Square :=
  {s | s.x ≤ 8 ∧ s.y ≤ 8}

theorem one_square_covered (d : Disc) :
  d.diameter = Real.sqrt 2 →
  d.center ∈ checkerboard →
  ∃! s : Square, s ∈ checkerboard ∧ is_covered s d :=
sorry

end NUMINAMATH_CALUDE_one_square_covered_l1191_119159


namespace NUMINAMATH_CALUDE_cos_2x_satisfies_conditions_l1191_119181

theorem cos_2x_satisfies_conditions (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x)
  (f x = f (-x)) ∧ (f (x - π) = f x) := by sorry

end NUMINAMATH_CALUDE_cos_2x_satisfies_conditions_l1191_119181


namespace NUMINAMATH_CALUDE_expression_sum_l1191_119145

theorem expression_sum (d e : ℤ) (h : d ≠ 0) : 
  let original := (16 * d + 17 + 18 * d^2) + (4 * d + 3) + 2 * e
  ∃ (a b c : ℤ), 
    original = a * d + b + c * d^2 + d * e ∧ 
    a + b + c + e = 60 := by
  sorry

end NUMINAMATH_CALUDE_expression_sum_l1191_119145


namespace NUMINAMATH_CALUDE_abs_neg_three_l1191_119150

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_l1191_119150


namespace NUMINAMATH_CALUDE_temperature_difference_l1191_119136

/-- The temperature difference problem -/
theorem temperature_difference (T_NY T_M T_SD : ℝ) : 
  T_NY = 80 →
  T_M = T_NY + 10 →
  (T_NY + T_M + T_SD) / 3 = 95 →
  T_SD - T_M = 25 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l1191_119136


namespace NUMINAMATH_CALUDE_system_solution_l1191_119153

theorem system_solution :
  ∀ a b : ℚ,
  (a * Real.sqrt a + b * Real.sqrt b = 183 ∧
   a * Real.sqrt b + b * Real.sqrt a = 182) →
  ((a = 196/9 ∧ b = 169/9) ∨ (a = 169/9 ∧ b = 196/9)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1191_119153


namespace NUMINAMATH_CALUDE_square_division_theorem_l1191_119156

/-- Represents a cell in the square array -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents a rectangle in the square array -/
structure Rectangle where
  top_left : Cell
  bottom_right : Cell

/-- Represents the state of a cell (pink or not pink) -/
inductive CellState
  | Pink
  | NotPink

/-- Represents the square array -/
def SquareArray (n : Nat) := Fin n → Fin n → CellState

/-- Checks if a rectangle contains exactly one pink cell -/
def containsOnePinkCell (arr : SquareArray n) (rect : Rectangle) : Prop := sorry

/-- Checks if a list of rectangles forms a valid division of the square -/
def isValidDivision (n : Nat) (rectangles : List Rectangle) : Prop := sorry

/-- The main theorem -/
theorem square_division_theorem (n : Nat) (arr : SquareArray n) :
  (∃ (i j : Fin n), arr i j = CellState.Pink) →
  ∃ (rectangles : List Rectangle), 
    isValidDivision n rectangles ∧ 
    ∀ rect ∈ rectangles, containsOnePinkCell arr rect :=
sorry

end NUMINAMATH_CALUDE_square_division_theorem_l1191_119156


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l1191_119111

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (x : ℂ), a * x^2 + b * x + c = 0 ↔ x = 4 + 2*I ∨ x = 4 - 2*I) ∧
    (a * (4 + 2*I)^2 + b * (4 + 2*I) + c = 0) ∧
    (a = 3 ∧ b = -24 ∧ c = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l1191_119111


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_l1191_119192

-- Define the function representing the curve
def f (x : ℝ) := x^2 + 3*x

-- Define the derivative of the function
def f' (x : ℝ) := 2*x + 3

-- Theorem statement
theorem tangent_slope_at_point : 
  f 2 = 10 → f' 2 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_l1191_119192


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1191_119123

-- Define rational numbers
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define the statements p and q
def p (x : ℝ) : Prop := IsRational (x^2)
def q (x : ℝ) : Prop := IsRational x

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l1191_119123
