import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2830_283079

theorem equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (3 / (x - 1) - 2 / x = 0) ↔ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2830_283079


namespace NUMINAMATH_CALUDE_derivative_sin_minus_cos_at_pi_l2830_283052

open Real

theorem derivative_sin_minus_cos_at_pi :
  let f : ℝ → ℝ := fun x ↦ sin x - cos x
  deriv f π = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_minus_cos_at_pi_l2830_283052


namespace NUMINAMATH_CALUDE_M_equals_reals_l2830_283072

def M : Set ℂ := {z : ℂ | Complex.abs ((z - 1)^2) = Complex.abs (z - 1)^2}

theorem M_equals_reals : M = {z : ℂ | z.im = 0} := by sorry

end NUMINAMATH_CALUDE_M_equals_reals_l2830_283072


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2830_283045

theorem difference_of_squares_example : (23 + 12)^2 - (23 - 12)^2 = 1104 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2830_283045


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2830_283013

theorem concentric_circles_ratio (r R : ℝ) (h : r > 0) (H : R > r) :
  π * R^2 - π * r^2 = 4 * (π * r^2) → r / R = 1 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2830_283013


namespace NUMINAMATH_CALUDE_sqrt_gt_sufficient_not_necessary_for_exp_gt_l2830_283081

theorem sqrt_gt_sufficient_not_necessary_for_exp_gt (a b : ℝ) :
  (∀ a b : ℝ, Real.sqrt a > Real.sqrt b → Real.exp a > Real.exp b) ∧
  ¬(∀ a b : ℝ, Real.exp a > Real.exp b → Real.sqrt a > Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_gt_sufficient_not_necessary_for_exp_gt_l2830_283081


namespace NUMINAMATH_CALUDE_triangle_abc_problem_l2830_283008

theorem triangle_abc_problem (A B C : ℝ) (a b c : ℝ) :
  b * Real.sin A = 3 * c * Real.sin B →
  a = 3 →
  Real.cos B = 2/3 →
  b = Real.sqrt 6 ∧ 
  Real.sin (2*B - π/3) = (4*Real.sqrt 5 + Real.sqrt 3) / 18 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_problem_l2830_283008


namespace NUMINAMATH_CALUDE_exists_increasing_omega_sequence_l2830_283003

/-- The number of distinct prime factors of a natural number -/
def omega (n : ℕ) : ℕ := sorry

/-- For any k, there exists an n > k satisfying the omega inequality -/
theorem exists_increasing_omega_sequence (k : ℕ) :
  ∃ n : ℕ, n > k ∧ omega n < omega (n + 1) ∧ omega (n + 1) < omega (n + 2) :=
sorry

end NUMINAMATH_CALUDE_exists_increasing_omega_sequence_l2830_283003


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2830_283024

theorem initial_mean_calculation (n : ℕ) (incorrect_value correct_value : ℝ) (correct_mean : ℝ) :
  n = 30 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 ∧ 
  correct_mean = 151 →
  ∃ (initial_mean : ℝ),
    n * initial_mean + (correct_value - incorrect_value) = n * correct_mean ∧
    initial_mean = 150 :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2830_283024


namespace NUMINAMATH_CALUDE_factorization_and_difference_l2830_283099

theorem factorization_and_difference (y : ℤ) : ∃ (a b : ℤ), 
  (4 * y^2 - 3 * y - 28 = (4 * y + a) * (y + b)) ∧ (a - b = 11) := by
  sorry

end NUMINAMATH_CALUDE_factorization_and_difference_l2830_283099


namespace NUMINAMATH_CALUDE_sallys_initial_cards_l2830_283056

/-- Proves that Sally's initial number of cards was 27 given the problem conditions -/
theorem sallys_initial_cards : 
  ∀ x : ℕ, 
  (x + 41 + 20 = 88) → 
  x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sallys_initial_cards_l2830_283056


namespace NUMINAMATH_CALUDE_wand_price_l2830_283037

theorem wand_price (price : ℝ) (original : ℝ) : 
  price = 8 ∧ price = (1/8) * original → original = 64 := by
  sorry

end NUMINAMATH_CALUDE_wand_price_l2830_283037


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2830_283001

/-- Triangle ABC with vertices A(-2, 3), B(5, 3), and C(5, -2) is a right triangle with perimeter 12 + √74 -/
theorem triangle_abc_properties :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (5, 3)
  let C : ℝ × ℝ := (5, -2)
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  -- Triangle ABC is a right triangle with right angle at B
  AB^2 + BC^2 = AC^2 ∧
  -- The perimeter of triangle ABC is 12 + √74
  AB + BC + AC = 12 + Real.sqrt 74 :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2830_283001


namespace NUMINAMATH_CALUDE_kaleb_candy_count_l2830_283016

/-- The number of candies Kaleb can buy with his arcade tickets -/
def candies_kaleb_can_buy (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost

/-- Proof that Kaleb can buy 3 candies with his arcade tickets -/
theorem kaleb_candy_count : candies_kaleb_can_buy 8 7 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_candy_count_l2830_283016


namespace NUMINAMATH_CALUDE_female_democrats_count_l2830_283020

theorem female_democrats_count 
  (total : ℕ) 
  (female : ℕ) 
  (male : ℕ) 
  (h1 : total = 750)
  (h2 : female + male = total)
  (h3 : (female / 2 + male / 4 : ℚ) = total / 3) :
  female / 2 = 125 := by
sorry

end NUMINAMATH_CALUDE_female_democrats_count_l2830_283020


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2830_283032

/-- Given a geometric sequence {a_n} with a₁ = 3, if 4a₁, 2a₂, a₃ form an arithmetic sequence,
    then the common ratio of the geometric sequence is 2. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 1 = 3 →                     -- first term condition
  4 * a 1 - 2 * a 2 = 2 * a 2 - a 3 →  -- arithmetic sequence condition
  q = 2 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l2830_283032


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l2830_283042

def P : ℕ → ℝ → ℝ → ℝ → ℝ
  | 0, x, y, z => 1
  | m + 1, x, y, z => (x + z) * (y + z) * P m x y (z + 1) - z^2 * P m x y z

theorem polynomial_symmetry (m : ℕ) (x y z : ℝ) :
  P m x y z = P m y x z ∧
  P m x y z = P m x z y ∧
  P m x y z = P m y z x ∧
  P m x y z = P m z x y ∧
  P m x y z = P m z y x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l2830_283042


namespace NUMINAMATH_CALUDE_sequence_properties_l2830_283035

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def S (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S b n + b (n + 1)

def T (c : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => T c n + c (n + 1)

theorem sequence_properties (a b c : ℕ → ℝ) :
  arithmetic_sequence a
  ∧ a 5 = 14
  ∧ a 7 = 20
  ∧ b 1 = 2/3
  ∧ (∀ n : ℕ, n ≥ 2 → 3 * S b n = S b (n-1) + 2)
  ∧ (∀ n : ℕ, c n = a n * b n)
  →
  (∀ n : ℕ, a n = 3*n - 1)
  ∧ (∀ n : ℕ, b n = 2 * (1/3)^n)
  ∧ (∀ n : ℕ, n ≥ 1 → T c n < 7/2)
  ∧ (∀ m : ℝ, (∀ n : ℕ, n ≥ 1 → T c n < m) → m ≥ 7/2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2830_283035


namespace NUMINAMATH_CALUDE_no_rational_roots_l2830_283029

def f (x : ℚ) : ℚ := 3 * x^4 - 2 * x^3 - 8 * x^2 + 3 * x + 1

theorem no_rational_roots : ∀ (x : ℚ), f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l2830_283029


namespace NUMINAMATH_CALUDE_train_probability_is_half_l2830_283087

-- Define the time interval (in minutes)
def timeInterval : ℝ := 60

-- Define the waiting time of the train (in minutes)
def waitingTime : ℝ := 30

-- Define a function to calculate the probability
noncomputable def trainProbability : ℝ :=
  let triangleArea := (1 / 2) * waitingTime * waitingTime
  let trapezoidArea := (1 / 2) * (waitingTime + timeInterval) * (timeInterval - waitingTime)
  (triangleArea + trapezoidArea) / (timeInterval * timeInterval)

-- Theorem statement
theorem train_probability_is_half :
  trainProbability = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_train_probability_is_half_l2830_283087


namespace NUMINAMATH_CALUDE_equality_sum_l2830_283061

theorem equality_sum (M N : ℚ) : 
  (4 : ℚ) / 7 = M / 63 ∧ (4 : ℚ) / 7 = 84 / N → M + N = 183 := by
  sorry

end NUMINAMATH_CALUDE_equality_sum_l2830_283061


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l2830_283069

/-- Given two cylinders A and B with base areas S₁ and S₂, volumes V₁ and V₂,
    if S₁/S₂ = 9/4 and their lateral surface areas are equal, then V₁/V₂ = 3/2 -/
theorem cylinder_volume_ratio (S₁ S₂ V₁ V₂ R r H h : ℝ) 
    (h_base_ratio : S₁ / S₂ = 9 / 4)
    (h_S₁ : S₁ = π * R^2)
    (h_S₂ : S₂ = π * r^2)
    (h_V₁ : V₁ = S₁ * H)
    (h_V₂ : V₂ = S₂ * h)
    (h_lateral_area : 2 * π * R * H = 2 * π * r * h) : 
  V₁ / V₂ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l2830_283069


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l2830_283030

theorem complex_number_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (i / (1 + i) : ℂ) = a + b * I :=
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l2830_283030


namespace NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2830_283005

def isOddUnitsDigit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def isSingleDigit (d : ℕ) : Prop := d < 10

theorem smallest_non_odd_units_digit :
  ∀ d : ℕ, isSingleDigit d → (d < 0 ∨ isOddUnitsDigit d) → 0 ≤ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_odd_units_digit_l2830_283005


namespace NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l2830_283014

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 55 -/
theorem wickets_before_last_match_is_55 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 4)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 55 := by
  sorry

end NUMINAMATH_CALUDE_wickets_before_last_match_is_55_l2830_283014


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2830_283047

theorem smaller_number_proof (x y : ℝ) (h1 : x + y = 56) (h2 : y = x + 12) : x = 22 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2830_283047


namespace NUMINAMATH_CALUDE_water_added_to_container_l2830_283055

/-- Proves that the amount of water added to a container with a capacity of 40 liters,
    initially 40% full, to make it 3/4 full, is 14 liters. -/
theorem water_added_to_container (capacity : ℝ) (initial_percentage : ℝ) (final_fraction : ℝ) :
  capacity = 40 →
  initial_percentage = 0.4 →
  final_fraction = 3/4 →
  (final_fraction * capacity) - (initial_percentage * capacity) = 14 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_container_l2830_283055


namespace NUMINAMATH_CALUDE_cuboidal_block_dimension_l2830_283065

/-- Given a cuboidal block with dimensions x cm × 9 cm × 12 cm that can be cut into at least 24 equal cubes,
    prove that the length of the first dimension (x) must be 6 cm. -/
theorem cuboidal_block_dimension (x : ℕ) : 
  (∃ (n : ℕ), n ≥ 24 ∧ x * 9 * 12 = n * (gcd x (gcd 9 12))^3) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboidal_block_dimension_l2830_283065


namespace NUMINAMATH_CALUDE_logical_reasoning_classification_l2830_283009

-- Define the types of reasoning
inductive ReasoningType
  | Sphere
  | Triangle
  | Chair
  | Polygon

-- Define a predicate for logical reasoning
def is_logical (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Sphere => true   -- Analogy reasoning
  | ReasoningType.Triangle => true -- Inductive reasoning
  | ReasoningType.Chair => false   -- Not logical
  | ReasoningType.Polygon => true  -- Inductive reasoning

-- Theorem statement
theorem logical_reasoning_classification :
  (is_logical ReasoningType.Sphere) ∧
  (is_logical ReasoningType.Triangle) ∧
  (¬is_logical ReasoningType.Chair) ∧
  (is_logical ReasoningType.Polygon) :=
sorry

end NUMINAMATH_CALUDE_logical_reasoning_classification_l2830_283009


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_of_squares_l2830_283010

theorem arithmetic_geometric_mean_sum_of_squares 
  (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = 15) : 
  x^2 + y^2 = 1150 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_sum_of_squares_l2830_283010


namespace NUMINAMATH_CALUDE_distinct_roots_of_quadratic_l2830_283040

theorem distinct_roots_of_quadratic (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - a*x₁ - 2 = 0 ∧ x₂^2 - a*x₂ - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_distinct_roots_of_quadratic_l2830_283040


namespace NUMINAMATH_CALUDE_complex_number_problem_l2830_283098

theorem complex_number_problem (a : ℝ) :
  (((a^2 - 1) : ℂ) + (a + 1) * I).im ≠ 0 ∧ ((a^2 - 1) : ℂ).re = 0 →
  (a + I^2016) / (1 + I) = 1 - I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2830_283098


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l2830_283006

theorem polynomial_expansion_equality (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 1) - (x - 1)*(x^2 + 5*x - 72) + (4*x - 15)*(x - 1)*(x + 6) = 
  6*x^3 + 2*x^2 - 18*x + 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l2830_283006


namespace NUMINAMATH_CALUDE_complex_square_simplification_l2830_283012

theorem complex_square_simplification :
  let i : ℂ := Complex.I
  (5 - 3 * i)^2 = 16 - 30 * i := by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l2830_283012


namespace NUMINAMATH_CALUDE_exam_time_allocation_l2830_283000

theorem exam_time_allocation (total_time : ℝ) (total_questions : ℕ) (type_a_count : ℕ) :
  total_time = 180 ∧
  total_questions = 200 ∧
  type_a_count = 10 →
  let type_b_count : ℕ := total_questions - type_a_count
  let time_ratio : ℝ := 2
  let type_b_time : ℝ := total_time / (type_b_count + time_ratio * type_a_count)
  let type_a_time : ℝ := time_ratio * type_b_time
  type_a_count * type_a_time = 120 / 7 :=
by sorry

end NUMINAMATH_CALUDE_exam_time_allocation_l2830_283000


namespace NUMINAMATH_CALUDE_nine_div_repeating_third_eq_twentyseven_l2830_283094

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- Theorem stating that 9 divided by 0.3333... equals 27 --/
theorem nine_div_repeating_third_eq_twentyseven :
  9 / repeating_third = 27 := by sorry

end NUMINAMATH_CALUDE_nine_div_repeating_third_eq_twentyseven_l2830_283094


namespace NUMINAMATH_CALUDE_fraction_simplification_l2830_283034

theorem fraction_simplification : 
  (1 / 3 + 1 / 4) / (2 / 5 - 1 / 6) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2830_283034


namespace NUMINAMATH_CALUDE_manufacturing_expenses_calculation_l2830_283078

/-- Calculates the monthly manufacturing expenses for a textile firm. -/
def monthly_manufacturing_expenses (
  total_looms : ℕ)
  (aggregate_sales : ℕ)
  (establishment_charges : ℕ)
  (profit_decrease : ℕ) : ℕ :=
  let sales_per_loom := aggregate_sales / total_looms
  let expenses_per_loom := sales_per_loom - profit_decrease
  expenses_per_loom * total_looms

/-- Proves that the monthly manufacturing expenses are 150000 given the specified conditions. -/
theorem manufacturing_expenses_calculation :
  monthly_manufacturing_expenses 80 500000 75000 4375 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_expenses_calculation_l2830_283078


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2830_283064

theorem quadratic_equation_solution (m : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - (m + 3) * x₁ + m + 2 = 0) →
  (x₂^2 - (m + 3) * x₂ + m + 2 = 0) →
  (x₁ / (x₁ + 1) + x₂ / (x₂ + 1) = 13 / 10) →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2830_283064


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2830_283023

/-- Given a rectangular plot with area 363 sq m and breadth 11 m, 
    prove that the ratio of length to breadth is 3:1 -/
theorem rectangle_ratio : ∀ (length breadth : ℝ),
  breadth = 11 →
  length * breadth = 363 →
  length / breadth = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2830_283023


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2830_283060

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2830_283060


namespace NUMINAMATH_CALUDE_shaded_area_is_65_l2830_283046

/-- Represents a trapezoid with a line segment dividing it into two parts -/
structure DividedTrapezoid where
  total_area : ℝ
  dividing_segment_length : ℝ
  inner_segment_length : ℝ

/-- Calculates the area of the shaded region in the divided trapezoid -/
def shaded_area (t : DividedTrapezoid) : ℝ :=
  t.total_area - (t.dividing_segment_length * t.inner_segment_length)

/-- Theorem stating that for the given trapezoid, the shaded area is 65 -/
theorem shaded_area_is_65 (t : DividedTrapezoid) 
  (h1 : t.total_area = 117)
  (h2 : t.dividing_segment_length = 13)
  (h3 : t.inner_segment_length = 4) :
  shaded_area t = 65 := by
  sorry

#eval shaded_area { total_area := 117, dividing_segment_length := 13, inner_segment_length := 4 }

end NUMINAMATH_CALUDE_shaded_area_is_65_l2830_283046


namespace NUMINAMATH_CALUDE_least_three_digit_7_shifty_l2830_283017

def is_7_shifty (n : ℕ) : Prop := n % 7 > 2

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem least_three_digit_7_shifty : 
  (∀ m : ℕ, is_three_digit m → is_7_shifty m → 101 ≤ m) ∧ 
  is_three_digit 101 ∧ 
  is_7_shifty 101 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_7_shifty_l2830_283017


namespace NUMINAMATH_CALUDE_complex_circle_l2830_283080

-- Define a complex number
def z : ℂ := sorry

-- Define the condition |z - (-1 + i)| = 4
def condition (z : ℂ) : Prop := Complex.abs (z - (-1 + Complex.I)) = 4

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 16

-- Theorem statement
theorem complex_circle (z : ℂ) (h : condition z) :
  circle_equation z.re z.im := by sorry

end NUMINAMATH_CALUDE_complex_circle_l2830_283080


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2830_283058

noncomputable def f (x : ℝ) : ℝ := (4 * x^2 - x + 1) / (5 * (x - 1))

theorem functional_equation_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  x * f x + 2 * f ((x - 1) / (x + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2830_283058


namespace NUMINAMATH_CALUDE_nurses_who_quit_l2830_283057

theorem nurses_who_quit (initial_doctors initial_nurses doctors_quit total_remaining : ℕ) :
  initial_doctors = 11 →
  initial_nurses = 18 →
  doctors_quit = 5 →
  total_remaining = 22 →
  initial_doctors + initial_nurses - doctors_quit - total_remaining = 2 := by
  sorry

end NUMINAMATH_CALUDE_nurses_who_quit_l2830_283057


namespace NUMINAMATH_CALUDE_at_equals_rc_l2830_283039

-- Define the points
variable (A B C D M P R Q S T : Point)

-- Define the cyclic quadrilateral ABCD
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define M as midpoint of CD
def is_midpoint (M C D : Point) : Prop := sorry

-- Define P as intersection of diagonals AC and BD
def is_diagonal_intersection (P A B C D : Point) : Prop := sorry

-- Define circle through P touching CD at M and meeting AC at R and BD at Q
def circle_touches_and_meets (P M C D R Q : Point) : Prop := sorry

-- Define S on BD such that BS = DQ
def point_on_line_with_equal_distance (S B D Q : Point) : Prop := sorry

-- Define line through S parallel to AB meeting AC at T
def parallel_line_intersection (S T A B C : Point) : Prop := sorry

-- Theorem statement
theorem at_equals_rc 
  (h1 : is_cyclic_quadrilateral A B C D)
  (h2 : is_midpoint M C D)
  (h3 : is_diagonal_intersection P A B C D)
  (h4 : circle_touches_and_meets P M C D R Q)
  (h5 : point_on_line_with_equal_distance S B D Q)
  (h6 : parallel_line_intersection S T A B C) :
  AT = RC := by sorry

end NUMINAMATH_CALUDE_at_equals_rc_l2830_283039


namespace NUMINAMATH_CALUDE_solution_count_l2830_283096

/-- The number of distinct solutions to the system of equations:
    x = x^2 + y^2
    y = 3x^2y - y^3 -/
theorem solution_count : 
  (Set.ncard {p : ℝ × ℝ | let (x, y) := p; x = x^2 + y^2 ∧ y = 3*x^2*y - y^3} : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_l2830_283096


namespace NUMINAMATH_CALUDE_percentage_spent_on_hats_l2830_283028

def total_money : ℕ := 90
def scarf_count : ℕ := 18
def scarf_price : ℕ := 2
def hat_to_scarf_ratio : ℕ := 2

theorem percentage_spent_on_hats :
  let money_spent_on_scarves := scarf_count * scarf_price
  let money_spent_on_hats := total_money - money_spent_on_scarves
  let percentage_on_hats := (money_spent_on_hats : ℚ) / total_money * 100
  percentage_on_hats = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_spent_on_hats_l2830_283028


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l2830_283007

theorem dot_product_of_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a • b = -12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l2830_283007


namespace NUMINAMATH_CALUDE_candy_bar_profit_l2830_283038

/-- Calculates the profit from selling candy bars --/
theorem candy_bar_profit : 
  let total_bars : ℕ := 1500
  let purchase_price : ℚ := 3 / 4
  let sold_first : ℕ := 1200
  let price_first : ℚ := 2 / 3
  let sold_second : ℕ := 300
  let price_second : ℚ := 8 / 10
  let total_cost : ℚ := total_bars * purchase_price
  let revenue_first : ℚ := sold_first * price_first
  let revenue_second : ℚ := sold_second * price_second
  let total_revenue : ℚ := revenue_first + revenue_second
  let profit : ℚ := total_revenue - total_cost
  profit = -85 :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l2830_283038


namespace NUMINAMATH_CALUDE_chooseAndAssignTheorem_l2830_283093

-- Define the set of members
inductive Member : Type
| Alice : Member
| Bob : Member
| Carol : Member
| Dave : Member

-- Define the set of officer roles
inductive Role : Type
| President : Role
| Secretary : Role
| Treasurer : Role

-- Define a function to calculate the number of ways to choose and assign roles
def waysToChooseAndAssign : ℕ :=
  -- Number of ways to choose 3 out of 4 members
  (Nat.choose 4 3) *
  -- Number of ways to assign 3 roles to 3 chosen members
  (Nat.factorial 3)

-- Theorem statement
theorem chooseAndAssignTheorem : waysToChooseAndAssign = 24 := by
  sorry


end NUMINAMATH_CALUDE_chooseAndAssignTheorem_l2830_283093


namespace NUMINAMATH_CALUDE_interior_angle_sum_l2830_283088

theorem interior_angle_sum (n : ℕ) : 
  (180 * (n - 2) = 1800) → (180 * ((n + 4) - 2) = 2520) :=
by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_l2830_283088


namespace NUMINAMATH_CALUDE_student_sample_size_l2830_283022

/-- Represents the frequency distribution of student weights --/
structure WeightDistribution where
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ
  remaining : ℕ

/-- The total number of students in the sample --/
def total_students (w : WeightDistribution) : ℕ :=
  w.group1 + w.group2 + w.group3 + w.remaining

/-- The given conditions for the weight distribution --/
def weight_distribution_conditions (w : WeightDistribution) : Prop :=
  w.group1 + w.group2 + w.group3 > 0 ∧
  w.group2 = 12 ∧
  w.group2 = 2 * w.group1 ∧
  w.group3 = 3 * w.group1

theorem student_sample_size :
  ∃ w : WeightDistribution, weight_distribution_conditions w ∧ total_students w = 48 :=
sorry

end NUMINAMATH_CALUDE_student_sample_size_l2830_283022


namespace NUMINAMATH_CALUDE_reading_time_l2830_283090

theorem reading_time (total_pages : ℕ) (first_half_speed second_half_speed : ℕ) : 
  total_pages = 500 → 
  first_half_speed = 10 → 
  second_half_speed = 5 → 
  (total_pages / 2 / first_half_speed + total_pages / 2 / second_half_speed) = 75 := by
sorry

end NUMINAMATH_CALUDE_reading_time_l2830_283090


namespace NUMINAMATH_CALUDE_consecutive_even_sum_squares_l2830_283068

theorem consecutive_even_sum_squares (a b c d : ℕ) : 
  (∃ n : ℕ, a = 2*n ∧ b = 2*n + 2 ∧ c = 2*n + 4 ∧ d = 2*n + 6) →
  (a + b + c + d = 36) →
  (a^2 + b^2 + c^2 + d^2 = 344) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_squares_l2830_283068


namespace NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l2830_283026

/-- Represents the inventory of a bookstore -/
structure BookInventory where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Conditions for Joel's bookstore inventory -/
def joelsBookstore (inventory : BookInventory) : Prop :=
  inventory.historicalFiction = (30 * inventory.total) / 100 ∧
  inventory.historicalFictionNewReleases = (30 * inventory.historicalFiction) / 100 ∧
  inventory.otherNewReleases = (40 * (inventory.total - inventory.historicalFiction)) / 100

/-- Theorem: The fraction of all new releases that are historical fiction is 9/37 -/
theorem historical_fiction_new_releases_fraction 
  (inventory : BookInventory) (h : joelsBookstore inventory) :
  (inventory.historicalFictionNewReleases : ℚ) / 
  (inventory.historicalFictionNewReleases + inventory.otherNewReleases) = 9 / 37 := by
  sorry

end NUMINAMATH_CALUDE_historical_fiction_new_releases_fraction_l2830_283026


namespace NUMINAMATH_CALUDE_garden_area_unchanged_l2830_283083

/-- Represents a rectangular garden with given length and width -/
structure RectangularGarden where
  length : ℝ
  width : ℝ

/-- Represents a square garden with a given side length -/
structure SquareGarden where
  side : ℝ

/-- Calculates the area of a rectangular garden -/
def area_rectangular (g : RectangularGarden) : ℝ := g.length * g.width

/-- Calculates the perimeter of a rectangular garden -/
def perimeter_rectangular (g : RectangularGarden) : ℝ := 2 * (g.length + g.width)

/-- Calculates the area of a square garden -/
def area_square (g : SquareGarden) : ℝ := g.side * g.side

/-- Calculates the perimeter of a square garden -/
def perimeter_square (g : SquareGarden) : ℝ := 4 * g.side

theorem garden_area_unchanged 
  (rect : RectangularGarden) 
  (sq : SquareGarden) 
  (partition_length : ℝ) :
  rect.length = 60 →
  rect.width = 15 →
  partition_length = 30 →
  perimeter_rectangular rect = perimeter_square sq + partition_length →
  area_rectangular rect = area_square sq :=
by sorry

end NUMINAMATH_CALUDE_garden_area_unchanged_l2830_283083


namespace NUMINAMATH_CALUDE_garys_gold_cost_per_gram_l2830_283075

/-- Proves that Gary's gold costs $15 per gram given the conditions of the problem -/
theorem garys_gold_cost_per_gram (gary_grams : ℝ) (anna_grams : ℝ) (anna_cost_per_gram : ℝ) (total_cost : ℝ)
  (h1 : gary_grams = 30)
  (h2 : anna_grams = 50)
  (h3 : anna_cost_per_gram = 20)
  (h4 : total_cost = 1450)
  (h5 : gary_grams * x + anna_grams * anna_cost_per_gram = total_cost) :
  x = 15 := by
  sorry

#check garys_gold_cost_per_gram

end NUMINAMATH_CALUDE_garys_gold_cost_per_gram_l2830_283075


namespace NUMINAMATH_CALUDE_power_equality_l2830_283027

theorem power_equality (p : ℕ) : 81^6 = 3^p → p = 24 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2830_283027


namespace NUMINAMATH_CALUDE_part_one_part_two_l2830_283049

-- Define the propositions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x-3)/(x+2) < 0

-- Part 1
theorem part_one (x : ℝ) (h1 : p x 1) (h2 : q x) : 1 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x, ¬(q x) → ¬(p x a))
  (h_not_nec : ∃ x, q x ∧ ¬(p x a)) : 
  0 < a ∧ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2830_283049


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2830_283015

-- Define the line equation
def line_equation (m n x y : ℝ) : Prop := m * x + n * y + 2 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 3)^2 + (y + 1)^2 = 1

-- Define the chord length condition
def chord_length_condition (m n : ℝ) : Prop := 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_equation m n x₁ y₁ ∧ 
    line_equation m n x₂ y₂ ∧ 
    circle_equation x₁ y₁ ∧ 
    circle_equation x₂ y₂ ∧ 
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4

theorem minimum_value_theorem (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) 
  (h_chord : chord_length_condition m n) : 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → chord_length_condition m' n' → 1/m' + 3/n' ≥ 1/m + 3/n) → 
  1/m + 3/n = 6 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2830_283015


namespace NUMINAMATH_CALUDE_ratio_section_area_l2830_283066

/-- Regular quadrilateral prism -/
structure RegularQuadPrism where
  base : Real
  height : Real

/-- Cross-section passing through midpoints -/
def midpoint_section (p : RegularQuadPrism) : Real :=
  12

/-- Cross-section dividing axis in ratio 1:3 -/
def ratio_section (p : RegularQuadPrism) : Real :=
  9

/-- Theorem statement -/
theorem ratio_section_area (p : RegularQuadPrism) :
  midpoint_section p = 12 → ratio_section p = 9 := by
  sorry

end NUMINAMATH_CALUDE_ratio_section_area_l2830_283066


namespace NUMINAMATH_CALUDE_initial_amount_A_l2830_283018

theorem initial_amount_A (a b c : ℝ) : 
  b = 28 → 
  c = 20 → 
  (a - b - c) + 2 * (a - b - c) + 4 * (a - b - c) = 24 →
  (b + b) - (2 * (a - b - c) + 2 * c) + 2 * ((b + b) - (2 * (a - b - c) + 2 * c)) = 24 →
  (c + c) - (4 * (a - b - c) + 2 * ((b + b) - (2 * (a - b - c) + 2 * c))) = 24 →
  a = 54 := by
  sorry

#check initial_amount_A

end NUMINAMATH_CALUDE_initial_amount_A_l2830_283018


namespace NUMINAMATH_CALUDE_age_difference_l2830_283054

/-- Given three people A, B, and C, where C is 13 years younger than A,
    prove that the sum of ages of A and B is 13 years more than the sum of ages of B and C. -/
theorem age_difference (A B C : ℕ) (h : C = A - 13) :
  A + B - (B + C) = 13 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2830_283054


namespace NUMINAMATH_CALUDE_even_sum_difference_l2830_283043

def sum_even_range (a b : ℕ) : ℕ := 
  (b - a + 2) / 2 * (a + b) / 2

theorem even_sum_difference : 
  sum_even_range 62 110 - sum_even_range 22 70 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_even_sum_difference_l2830_283043


namespace NUMINAMATH_CALUDE_percentage_sum_l2830_283095

theorem percentage_sum : (0.15 * 25) + (0.12 * 45) = 9.15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_l2830_283095


namespace NUMINAMATH_CALUDE_triangle_vector_representation_l2830_283092

/-- Given a triangle ABC and a point P on line AB, prove that CP can be represented
    in terms of CA and CB under certain conditions. -/
theorem triangle_vector_representation (A B C P : EuclideanSpace ℝ (Fin 3))
    (a b : EuclideanSpace ℝ (Fin 3)) : 
    (C - A = a) →  -- CA = a
    (C - B = b) →  -- CB = b
    (∃ t : ℝ, P = (1 - t) • A + t • B) →  -- P is on line AB
    (A - P = 2 • (P - B)) →  -- AP = 2PB
    (C - P = (1/3) • a + (2/3) • b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_vector_representation_l2830_283092


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l2830_283051

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l2830_283051


namespace NUMINAMATH_CALUDE_movie_admission_problem_l2830_283071

theorem movie_admission_problem (total_admitted : ℕ) 
  (west_side_total : ℕ) (west_side_denied_percent : ℚ)
  (mountaintop_total : ℕ) (mountaintop_denied_percent : ℚ)
  (first_school_denied_percent : ℚ) :
  total_admitted = 148 →
  west_side_total = 90 →
  west_side_denied_percent = 70/100 →
  mountaintop_total = 50 →
  mountaintop_denied_percent = 1/2 →
  first_school_denied_percent = 20/100 →
  ∃ (first_school_total : ℕ),
    first_school_total = 120 ∧
    total_admitted = 
      (first_school_total * (1 - first_school_denied_percent)).floor +
      (west_side_total * (1 - west_side_denied_percent)).floor +
      (mountaintop_total * (1 - mountaintop_denied_percent)).floor :=
by sorry

end NUMINAMATH_CALUDE_movie_admission_problem_l2830_283071


namespace NUMINAMATH_CALUDE_quadratic_parabola_properties_l2830_283089

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given quadratic equation and parabola have two distinct real roots and specific form when intersecting x-axis symmetrically -/
theorem quadratic_parabola_properties (m : ℝ) :
  let q : QuadraticEquation := ⟨1, -2*m, m^2 - 4⟩
  let p : Parabola := ⟨1, -2*m, m^2 - 4⟩
  -- The quadratic equation has two distinct real roots
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ q.a * x₁^2 + q.b * x₁ + q.c = 0 ∧ q.a * x₂^2 + q.b * x₂ + q.c = 0 ∧
  -- When the parabola intersects x-axis symmetrically, it has the form y = x^2 - 4
  (∃ (x₁ x₂ : ℝ), x₁ < 0 ∧ x₂ > 0 ∧ x₁ = -x₂ ∧ 
   p.a * x₁^2 + p.b * x₁ + p.c = 0 ∧ p.a * x₂^2 + p.b * x₂ + p.c = 0) →
  p = ⟨1, 0, -4⟩ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_parabola_properties_l2830_283089


namespace NUMINAMATH_CALUDE_product_equals_900_l2830_283082

theorem product_equals_900 (a : ℝ) (h : (a + 25)^2 = 1000) : (a + 15) * (a + 35) = 900 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_900_l2830_283082


namespace NUMINAMATH_CALUDE_broken_bowls_l2830_283070

theorem broken_bowls (total_bowls : ℕ) (lost_bowls : ℕ) (fee : ℕ) (safe_payment : ℕ) (penalty : ℕ) (total_payment : ℕ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  fee = 100 →
  safe_payment = 3 →
  penalty = 4 →
  total_payment = 1825 →
  ∃ (broken_bowls : ℕ),
    fee + safe_payment * (total_bowls - lost_bowls - broken_bowls) - 
    (penalty * lost_bowls + penalty * broken_bowls) = total_payment ∧
    broken_bowls = 29 :=
sorry

end NUMINAMATH_CALUDE_broken_bowls_l2830_283070


namespace NUMINAMATH_CALUDE_sum_first_four_terms_l2830_283097

def arithmetic_sequence (a : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem sum_first_four_terms
  (a d : ℤ)
  (h5 : arithmetic_sequence a d 4 = 10)
  (h6 : arithmetic_sequence a d 5 = 14)
  (h7 : arithmetic_sequence a d 6 = 18) :
  (arithmetic_sequence a d 0) +
  (arithmetic_sequence a d 1) +
  (arithmetic_sequence a d 2) +
  (arithmetic_sequence a d 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_first_four_terms_l2830_283097


namespace NUMINAMATH_CALUDE_regular_ngon_diagonal_difference_l2830_283041

/-- The difference between the longest and shortest diagonals of a regular n-gon equals its side length if and only if n = 9 -/
theorem regular_ngon_diagonal_difference (n : ℕ) (h : n ≥ 3) :
  let R : ℝ := 1  -- Assume unit circle for simplicity
  let side_length := 2 * Real.sin (Real.pi / n)
  let shortest_diagonal := 2 * Real.sin (2 * Real.pi / n)
  let longest_diagonal := if n % 2 = 0 then 2 else 2 * Real.cos (Real.pi / (2 * n))
  longest_diagonal - shortest_diagonal = side_length ↔ n = 9 := by
sorry


end NUMINAMATH_CALUDE_regular_ngon_diagonal_difference_l2830_283041


namespace NUMINAMATH_CALUDE_sine_of_supplementary_angles_l2830_283074

theorem sine_of_supplementary_angles (α β : Real) :
  α + β = Real.pi → Real.sin α = Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sine_of_supplementary_angles_l2830_283074


namespace NUMINAMATH_CALUDE_ribbon_solution_l2830_283053

def ribbon_problem (total : ℝ) : Prop :=
  let remaining_after_first := total / 2
  let remaining_after_second := remaining_after_first * 2 / 3
  let remaining_after_third := remaining_after_second / 2
  remaining_after_third = 250

theorem ribbon_solution :
  ribbon_problem 1500 := by sorry

end NUMINAMATH_CALUDE_ribbon_solution_l2830_283053


namespace NUMINAMATH_CALUDE_largest_integer_m_l2830_283067

theorem largest_integer_m (m : ℤ) : (∀ k : ℤ, k > 6 → k^2 - 11*k + 28 ≥ 0) ∧ 6^2 - 11*6 + 28 < 0 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_m_l2830_283067


namespace NUMINAMATH_CALUDE_inequality_proof_l2830_283063

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y*z) / Real.sqrt (2*x^2*(y+z)) + 
  (y^2 + z*x) / Real.sqrt (2*y^2*(z+x)) + 
  (z^2 + x*y) / Real.sqrt (2*z^2*(x+y)) ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2830_283063


namespace NUMINAMATH_CALUDE_triangle_squares_area_l2830_283036

theorem triangle_squares_area (x : ℝ) : 
  let small_square_area := (3 * x)^2
  let large_square_area := (6 * x)^2
  let triangle_area := (1/2) * (3 * x) * (6 * x)
  small_square_area + large_square_area + triangle_area = 1000 →
  x = (10 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_squares_area_l2830_283036


namespace NUMINAMATH_CALUDE_n_equals_fourteen_l2830_283044

def first_seven_multiples_of_seven : List ℕ := [7, 14, 21, 28, 35, 42, 49]

def a : ℚ := (first_seven_multiples_of_seven.sum : ℚ) / 7

def first_three_multiples (n : ℕ) : List ℕ := [n, 2*n, 3*n]

def b (n : ℕ) : ℕ := (first_three_multiples n).nthLe 1 sorry

theorem n_equals_fourteen (n : ℕ) (h : a^2 - (b n : ℚ)^2 = 0) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_n_equals_fourteen_l2830_283044


namespace NUMINAMATH_CALUDE_train_platform_problem_l2830_283059

/-- Given a train and two platforms, calculate the length of the second platform -/
theorem train_platform_problem (train_length platform1_length : ℝ)
  (time1 time2 : ℝ) :
  train_length = 100 →
  platform1_length = 350 →
  time1 = 15 →
  time2 = 20 →
  (train_length + platform1_length) / time1 = (train_length + 500) / time2 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_problem_l2830_283059


namespace NUMINAMATH_CALUDE_function_properties_l2830_283033

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1

-- State the theorem
theorem function_properties (a : ℝ) (h : a > 0) :
  (∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, f a x ≥ m) ∧
  ((∀ x : ℝ, f a x > 0) → a > 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ → f a x₁ < f a x₂) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2830_283033


namespace NUMINAMATH_CALUDE_infinitely_many_even_floor_alpha_n_squared_l2830_283025

theorem infinitely_many_even_floor_alpha_n_squared (α : ℝ) (hα : α > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, Even ⌊α * n^2⌋ := by sorry

end NUMINAMATH_CALUDE_infinitely_many_even_floor_alpha_n_squared_l2830_283025


namespace NUMINAMATH_CALUDE_tickets_theorem_l2830_283002

/-- Calculates the total number of tickets Tate and Peyton have together -/
def totalTickets (tateInitial : ℕ) (tateAdditional : ℕ) : ℕ :=
  let tateFinal := tateInitial + tateAdditional
  let peyton := tateFinal / 2
  tateFinal + peyton

/-- Theorem stating that given the initial conditions, the total number of tickets is 51 -/
theorem tickets_theorem :
  totalTickets 32 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_tickets_theorem_l2830_283002


namespace NUMINAMATH_CALUDE_max_value_on_interval_max_value_at_one_l2830_283021

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- Define the property of f being monotonically decreasing on (-∞, 2]
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 2 → f a x ≥ f a y

-- Theorem 1: Maximum value of f(x) on [3, 5] is 2
theorem max_value_on_interval (a : ℝ) 
  (h : is_monotone_decreasing_on_interval a) : 
  (∀ x, x ∈ Set.Icc 3 5 → f a x ≤ 2) ∧ (∃ x, x ∈ Set.Icc 3 5 ∧ f a x = 2) :=
sorry

-- Theorem 2: Maximum value of f(1) is -6
theorem max_value_at_one (a : ℝ) 
  (h : is_monotone_decreasing_on_interval a) : 
  f a 1 ≤ -6 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_max_value_at_one_l2830_283021


namespace NUMINAMATH_CALUDE_solve_distance_problem_l2830_283076

def distance_problem (initial_speed : ℝ) (initial_time : ℝ) (speed_increase : ℝ) (additional_time : ℝ) : Prop :=
  let initial_distance := initial_speed * initial_time
  let new_speed := initial_speed * (1 + speed_increase)
  let additional_distance := new_speed * additional_time
  let total_distance := initial_distance + additional_distance
  total_distance = 13

theorem solve_distance_problem :
  distance_problem 2 2 0.5 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_distance_problem_l2830_283076


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l2830_283073

theorem quadratic_two_distinct_real_roots :
  ∃ x y : ℝ, x ≠ y ∧ (2 * x^2 + 3 * x - 4 = 0) ∧ (2 * y^2 + 3 * y - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_real_roots_l2830_283073


namespace NUMINAMATH_CALUDE_remainder_2_power_2015_mod_20_l2830_283085

theorem remainder_2_power_2015_mod_20 : ∃ (seq : Fin 4 → Nat),
  (∀ (n : Nat), (2^n : Nat) % 20 = seq (n % 4)) ∧
  (seq 0 = 4 ∧ seq 1 = 8 ∧ seq 2 = 16 ∧ seq 3 = 12) →
  (2^2015 : Nat) % 20 = 8 := by
sorry

end NUMINAMATH_CALUDE_remainder_2_power_2015_mod_20_l2830_283085


namespace NUMINAMATH_CALUDE_martha_apples_l2830_283050

/-- Given Martha's initial apples and the distribution to her friends, 
    prove the number of additional apples she needs to give away to be left with 4. -/
theorem martha_apples (initial_apples : ℕ) (jane_apples : ℕ) (james_extra : ℕ) :
  initial_apples = 20 →
  jane_apples = 5 →
  james_extra = 2 →
  initial_apples - jane_apples - (jane_apples + james_extra) - 4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_martha_apples_l2830_283050


namespace NUMINAMATH_CALUDE_max_value_f2019_l2830_283084

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the recursive function f_n
def f_n : ℕ → (ℝ → ℝ)
  | 0 => f
  | n + 1 => λ x => f (f_n n x)

-- State the theorem
theorem max_value_f2019 :
  ∀ x ∈ Set.Icc 1 2,
  f_n 2019 x ≤ 3^(2^2019) - 1 ∧
  ∃ y ∈ Set.Icc 1 2, f_n 2019 y = 3^(2^2019) - 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_f2019_l2830_283084


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l2830_283086

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = !![3, 4; -2, -2]) : 
  (A^2)⁻¹ = !![1, 4; -2, -4] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l2830_283086


namespace NUMINAMATH_CALUDE_brenda_bracelets_l2830_283031

theorem brenda_bracelets (total_stones : ℕ) (stones_per_bracelet : ℕ) (h1 : total_stones = 36) (h2 : stones_per_bracelet = 12) :
  total_stones / stones_per_bracelet = 3 := by
  sorry

end NUMINAMATH_CALUDE_brenda_bracelets_l2830_283031


namespace NUMINAMATH_CALUDE_sports_league_games_l2830_283091

/-- The number of games in a complete season for a sports league -/
def total_games (n : ℕ) (d : ℕ) (t : ℕ) (s : ℕ) (c : ℕ) : ℕ :=
  (n * (d - 1) * s + n * t * c) / 2

/-- Theorem: The total number of games in the given sports league is 296 -/
theorem sports_league_games :
  total_games 8 8 8 3 2 = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_league_games_l2830_283091


namespace NUMINAMATH_CALUDE_min_bounces_to_height_ball_bounce_problem_l2830_283011

def bounce_height (initial_height : ℝ) (bounce_ratio : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

theorem min_bounces_to_height (initial_height bounce_ratio target_height : ℝ) :
  ∃ (n : ℕ), 
    (∀ (k : ℕ), k < n → bounce_height initial_height bounce_ratio k ≥ target_height) ∧
    bounce_height initial_height bounce_ratio n < target_height :=
  sorry

theorem ball_bounce_problem :
  let initial_height := 243
  let bounce_ratio := 2/3
  let target_height := 30
  ∃ (n : ℕ), n = 6 ∧
    (∀ (k : ℕ), k < n → bounce_height initial_height bounce_ratio k ≥ target_height) ∧
    bounce_height initial_height bounce_ratio n < target_height :=
  sorry

end NUMINAMATH_CALUDE_min_bounces_to_height_ball_bounce_problem_l2830_283011


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2830_283062

theorem imaginary_part_of_complex_expression :
  let z : ℂ := (1 + I) / (1 - I) + (1 - I)^2
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2830_283062


namespace NUMINAMATH_CALUDE_least_valid_integer_l2830_283004

def is_valid (n : ℕ) : Prop :=
  ∃ (d : ℕ) (m : ℕ), 
    n = 10 * d + m ∧ 
    d ≠ 0 ∧ 
    d < 10 ∧ 
    19 * m = n

theorem least_valid_integer : 
  (is_valid 95) ∧ (∀ n : ℕ, n < 95 → ¬(is_valid n)) :=
sorry

end NUMINAMATH_CALUDE_least_valid_integer_l2830_283004


namespace NUMINAMATH_CALUDE_math_books_arrangement_l2830_283019

theorem math_books_arrangement (num_math_books num_english_books : ℕ) : 
  num_math_books = 2 → num_english_books = 2 → 
  (num_math_books.factorial * (num_math_books + num_english_books).factorial) = 12 := by
  sorry

end NUMINAMATH_CALUDE_math_books_arrangement_l2830_283019


namespace NUMINAMATH_CALUDE_prob_non_black_ball_l2830_283048

/-- Given a bag of balls where the odds of drawing a black ball are 5:3,
    the probability of drawing a non-black ball is 3/8 -/
theorem prob_non_black_ball (total : ℕ) (black : ℕ) (non_black : ℕ)
  (h_total : total = black + non_black)
  (h_odds : (black : ℚ) / non_black = 5 / 3) :
  (non_black : ℚ) / total = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_black_ball_l2830_283048


namespace NUMINAMATH_CALUDE_poster_difference_l2830_283077

/-- The number of posters Mario made -/
def mario_posters : ℕ := 18

/-- The total number of posters made by Mario and Samantha -/
def total_posters : ℕ := 51

/-- The number of posters Samantha made -/
def samantha_posters : ℕ := total_posters - mario_posters

/-- Samantha made more posters than Mario -/
axiom samantha_made_more : samantha_posters > mario_posters

theorem poster_difference : samantha_posters - mario_posters = 15 := by
  sorry

end NUMINAMATH_CALUDE_poster_difference_l2830_283077
