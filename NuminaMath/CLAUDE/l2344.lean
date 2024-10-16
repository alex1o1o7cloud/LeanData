import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2344_234406

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : a^2 = 16) 
  (h2 : b^3 = -27) 
  (h3 : |a - b| = a - b) : 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2344_234406


namespace NUMINAMATH_CALUDE_inequality_solution_range_inequality_equal_solution_sets_l2344_234477

-- Define the inequality
def inequality (m x : ℝ) : Prop := m * x - 3 > 2 * x + m

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ := {x | x < (m + 3) / (m - 2)}

-- Define the alternate inequality
def alt_inequality (x : ℝ) : Prop := 2 * x - 1 > 3 - x

theorem inequality_solution_range (m : ℝ) :
  (∀ x, inequality m x ↔ x ∈ solution_set m) → m < 2 := by sorry

theorem inequality_equal_solution_sets (m : ℝ) :
  (∀ x, inequality m x ↔ alt_inequality x) → m = 17 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_inequality_equal_solution_sets_l2344_234477


namespace NUMINAMATH_CALUDE_max_integers_greater_than_26_l2344_234459

theorem max_integers_greater_than_26 (a b c d e : ℤ) :
  a + b + c + d + e = 3 →
  ∃ (count : ℕ), count ≤ 4 ∧
    count = (if a > 26 then 1 else 0) +
            (if b > 26 then 1 else 0) +
            (if c > 26 then 1 else 0) +
            (if d > 26 then 1 else 0) +
            (if e > 26 then 1 else 0) ∧
    ∀ (other_count : ℕ),
      other_count = (if a > 26 then 1 else 0) +
                    (if b > 26 then 1 else 0) +
                    (if c > 26 then 1 else 0) +
                    (if d > 26 then 1 else 0) +
                    (if e > 26 then 1 else 0) →
      other_count ≤ count :=
by sorry

end NUMINAMATH_CALUDE_max_integers_greater_than_26_l2344_234459


namespace NUMINAMATH_CALUDE_uncertain_sum_l2344_234402

theorem uncertain_sum (a b c : ℤ) (h : |a - b|^19 + |c - a|^95 = 1) :
  ∃ (x : ℤ), (x = 1 ∨ x = 2) ∧ |c - a| + |a - b| + |b - a| = x :=
sorry

end NUMINAMATH_CALUDE_uncertain_sum_l2344_234402


namespace NUMINAMATH_CALUDE_rancher_animals_count_l2344_234450

theorem rancher_animals_count : ∀ (horses cows total : ℕ),
  cows = 5 * horses →
  cows = 140 →
  total = cows + horses →
  total = 168 :=
by
  sorry

end NUMINAMATH_CALUDE_rancher_animals_count_l2344_234450


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2344_234414

theorem sum_of_fractions : (10 + 20 + 30 + 40) / 10 + 10 / (10 + 20 + 30 + 40) = 10.1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2344_234414


namespace NUMINAMATH_CALUDE_intersection_M_N_l2344_234451

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}

def N : Set ℝ := {-3, -2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {-2, -1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2344_234451


namespace NUMINAMATH_CALUDE_f_has_unique_minimum_l2344_234411

open Real

-- Define the function f(x) = 2x - ln x
noncomputable def f (x : ℝ) : ℝ := 2 * x - log x

-- Theorem statement
theorem f_has_unique_minimum :
  ∃! (x : ℝ), x > 0 ∧ IsLocalMin f x ∧ f x = 1 + log 2 := by
  sorry

end NUMINAMATH_CALUDE_f_has_unique_minimum_l2344_234411


namespace NUMINAMATH_CALUDE_debt_installments_l2344_234417

theorem debt_installments (x : ℝ) : 
  (8 * x + 44 * (x + 65)) / 52 = 465 → x = 410 := by
  sorry

end NUMINAMATH_CALUDE_debt_installments_l2344_234417


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2344_234493

theorem complex_absolute_value (x y : ℝ) : 
  (Complex.I : ℂ) * (x + 3 * Complex.I) = y - Complex.I → 
  Complex.abs (x - y * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2344_234493


namespace NUMINAMATH_CALUDE_triangle_problem_l2344_234479

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  a = 2 →
  c = Real.sqrt 2 →
  Real.cos A = Real.sqrt 2 / 4 →
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  0 < a ∧ 0 < b ∧ 0 < c →
  a < b + c ∧ b < a + c ∧ c < a + b →
  -- Conclusions
  Real.sin C = Real.sqrt 7 / 4 ∧
  b = 1 ∧
  Real.cos (2 * A + π / 3) = (-3 + Real.sqrt 21) / 8 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2344_234479


namespace NUMINAMATH_CALUDE_heart_ratio_l2344_234476

/-- The heart operation defined as n ♥ m = n^2 * m^3 -/
def heart (n m : ℝ) : ℝ := n^2 * m^3

/-- Theorem stating that (3 ♥ 5) / (5 ♥ 3) = 5/3 -/
theorem heart_ratio : (heart 3 5) / (heart 5 3) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_l2344_234476


namespace NUMINAMATH_CALUDE_sound_travel_distance_l2344_234467

/-- The speed of sound in air at 20°C in meters per second -/
def speed_of_sound_at_20C : ℝ := 342

/-- The time of travel in seconds -/
def travel_time : ℝ := 5

/-- The distance traveled by sound in 5 seconds at 20°C -/
def distance_traveled : ℝ := speed_of_sound_at_20C * travel_time

theorem sound_travel_distance : distance_traveled = 1710 := by
  sorry

end NUMINAMATH_CALUDE_sound_travel_distance_l2344_234467


namespace NUMINAMATH_CALUDE_time_for_A_alone_l2344_234410

-- Define the work rates for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
def condition1 : Prop := 3 * (rA + rB) = 1
def condition2 : Prop := 6 * (rB + rC) = 1
def condition3 : Prop := (15/4) * (rA + rC) = 1

-- Theorem statement
theorem time_for_A_alone 
  (h1 : condition1 rA rB)
  (h2 : condition2 rB rC)
  (h3 : condition3 rA rC) :
  1 / rA = 60 / 13 :=
by sorry

end NUMINAMATH_CALUDE_time_for_A_alone_l2344_234410


namespace NUMINAMATH_CALUDE_gcd_factorial_8_9_l2344_234486

theorem gcd_factorial_8_9 : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_8_9_l2344_234486


namespace NUMINAMATH_CALUDE_sum_a_d_l2344_234452

theorem sum_a_d (a b c d : ℤ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_a_d_l2344_234452


namespace NUMINAMATH_CALUDE_max_x_minus_y_l2344_234498

theorem max_x_minus_y (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x * y + y * z + z * x = 1) : 
  x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l2344_234498


namespace NUMINAMATH_CALUDE_connie_marbles_proof_l2344_234462

/-- The number of marbles Connie had initially -/
def initial_marbles : ℕ := 2856

/-- The number of marbles Connie had after losing half -/
def marbles_after_loss : ℕ := initial_marbles / 2

/-- The number of marbles Connie had after giving away 2/3 of the remaining marbles -/
def final_marbles : ℕ := 476

theorem connie_marbles_proof : 
  initial_marbles = 2856 ∧ 
  marbles_after_loss = initial_marbles / 2 ∧
  final_marbles = marbles_after_loss / 3 ∧
  final_marbles = 476 := by sorry

end NUMINAMATH_CALUDE_connie_marbles_proof_l2344_234462


namespace NUMINAMATH_CALUDE_no_four_distinct_real_roots_l2344_234492

theorem no_four_distinct_real_roots (a b : ℝ) : 
  ¬ ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, x^4 - 4*x^3 + 6*x^2 + a*x + b = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄)) :=
sorry

end NUMINAMATH_CALUDE_no_four_distinct_real_roots_l2344_234492


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2344_234433

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a sampling process --/
structure SamplingProcess where
  interval : ℕ  -- Time interval between samples
  continuous : Bool  -- Whether the process is continuous

/-- Determines if a sampling process is systematic --/
def is_systematic (process : SamplingProcess) : Prop :=
  process.interval > 0 ∧ process.continuous

/-- Theorem: A sampling process with a fixed positive time interval 
    from a continuous process is systematic sampling --/
theorem factory_sampling_is_systematic 
  (process : SamplingProcess) 
  (h1 : process.interval = 10)  -- 10-minute interval
  (h2 : process.continuous = true)  -- Conveyor belt implies continuous process
  : is_systematic process ∧ 
    (λ method : SamplingMethod => 
      is_systematic process → method = SamplingMethod.Systematic) SamplingMethod.Systematic := by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2344_234433


namespace NUMINAMATH_CALUDE_distance_between_points_l2344_234449

/-- The distance between points (0, 8) and (6, 0) is 10. -/
theorem distance_between_points : Real.sqrt ((6 - 0)^2 + (0 - 8)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l2344_234449


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2344_234432

theorem factorial_sum_equality : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2344_234432


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l2344_234413

/-- Calculates the weekly earnings of a worker given their work schedule and hourly wage. -/
def weekly_earnings (hours_per_day_1 : ℕ) (days_1 : ℕ) (hours_per_day_2 : ℕ) (days_2 : ℕ) (hourly_wage : ℕ) : ℕ :=
  (hours_per_day_1 * days_1 + hours_per_day_2 * days_2) * hourly_wage

/-- Proves that Sheila's weekly earnings are $216 given her work schedule and hourly wage. -/
theorem sheila_weekly_earnings :
  weekly_earnings 8 3 6 2 6 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l2344_234413


namespace NUMINAMATH_CALUDE_complex_power_thousand_l2344_234495

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Main theorem: ((1 + i) / (1 - i)) ^ 1000 = 1 -/
theorem complex_power_thousand :
  ((1 + i) / (1 - i)) ^ 1000 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_complex_power_thousand_l2344_234495


namespace NUMINAMATH_CALUDE_probability_red_ball_specific_l2344_234480

/-- The probability of drawing a red ball from a bag with specified ball counts. -/
def probability_red_ball (red_count black_count white_count : ℕ) : ℚ :=
  red_count / (red_count + black_count + white_count)

/-- Theorem: The probability of drawing a red ball from a bag with 3 red balls,
    5 black balls, and 4 white balls is 1/4. -/
theorem probability_red_ball_specific : probability_red_ball 3 5 4 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_specific_l2344_234480


namespace NUMINAMATH_CALUDE_clara_quarters_problem_l2344_234453

theorem clara_quarters_problem : ∃! q : ℕ, 
  8 < q ∧ q < 80 ∧ 
  q % 3 = 1 ∧ 
  q % 4 = 1 ∧ 
  q % 5 = 1 ∧ 
  q = 61 := by
sorry

end NUMINAMATH_CALUDE_clara_quarters_problem_l2344_234453


namespace NUMINAMATH_CALUDE_min_max_problem_l2344_234408

theorem min_max_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (8 / x + 2 / y ≥ 18) ∧ (Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) ≤ 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_max_problem_l2344_234408


namespace NUMINAMATH_CALUDE_direct_proportion_condition_l2344_234426

/-- A function f(x) is a direct proportion function if there exists a non-zero constant k such that f(x) = k * x for all x -/
def IsDirectProportionFunction (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function defined by m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + (m^2 - 4)

theorem direct_proportion_condition (m : ℝ) :
  IsDirectProportionFunction (f m) ↔ m = -2 := by sorry

end NUMINAMATH_CALUDE_direct_proportion_condition_l2344_234426


namespace NUMINAMATH_CALUDE_matrix_not_invertible_iff_l2344_234435

def matrix (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2*y, 9],
    ![4 - 2*y, 5]]

theorem matrix_not_invertible_iff (y : ℝ) :
  ¬(IsUnit (matrix y).det) ↔ y = 9/7 := by
  sorry

end NUMINAMATH_CALUDE_matrix_not_invertible_iff_l2344_234435


namespace NUMINAMATH_CALUDE_percentage_problem_l2344_234441

theorem percentage_problem (x : ℝ) : x * 2 = 0.8 → x * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2344_234441


namespace NUMINAMATH_CALUDE_paving_company_calculation_l2344_234470

/-- Represents the properties of a street paved with cement -/
structure Street where
  length : Real
  width : Real
  thickness : Real
  cement_used : Real

/-- Calculates the volume of cement used for a street -/
def cement_volume (s : Street) : Real :=
  s.length * s.width * s.thickness

/-- Cement density in tons per cubic meter -/
def cement_density : Real := 1

theorem paving_company_calculation (lexi_street tess_street : Street) 
  (h1 : lexi_street.length = 200)
  (h2 : lexi_street.width = 10)
  (h3 : lexi_street.thickness = 0.1)
  (h4 : lexi_street.cement_used = 10)
  (h5 : tess_street.length = 100)
  (h6 : tess_street.thickness = 0.1)
  (h7 : tess_street.cement_used = 5.1) :
  tess_street.width = 0.51 ∧ lexi_street.cement_used + tess_street.cement_used = 15.1 := by
  sorry


end NUMINAMATH_CALUDE_paving_company_calculation_l2344_234470


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_l2344_234434

def product : ℕ := 11 * 101 * 111 * 110011

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_of_product :
  sum_of_digits product = 48 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_l2344_234434


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l2344_234437

/-- Given the total number of treats and the counts of chewing gums and candies,
    calculate the number of chocolate bars. -/
theorem chocolate_bars_count 
  (total_treats : ℕ) 
  (chewing_gums : ℕ) 
  (candies : ℕ) 
  (h1 : total_treats = 155) 
  (h2 : chewing_gums = 60) 
  (h3 : candies = 40) : 
  total_treats - (chewing_gums + candies) = 55 := by
  sorry

#eval 155 - (60 + 40)  -- Should output 55

end NUMINAMATH_CALUDE_chocolate_bars_count_l2344_234437


namespace NUMINAMATH_CALUDE_area_of_inscribed_square_l2344_234455

/-- A right triangle with an inscribed square -/
structure RightTriangleWithInscribedSquare where
  /-- Length of side AB -/
  ab : ℝ
  /-- Length of side CD -/
  cd : ℝ
  /-- Side length of the inscribed square BCFE -/
  x : ℝ
  /-- The inscribed square's side is perpendicular to both legs of the right triangle -/
  perpendicular : True
  /-- The inscribed square touches both legs of the right triangle -/
  touches_legs : True

/-- Theorem: Area of inscribed square in right triangle -/
theorem area_of_inscribed_square 
  (triangle : RightTriangleWithInscribedSquare) 
  (h1 : triangle.ab = 36)
  (h2 : triangle.cd = 64) :
  triangle.x^2 = 2304 := by
  sorry

end NUMINAMATH_CALUDE_area_of_inscribed_square_l2344_234455


namespace NUMINAMATH_CALUDE_max_sum_reciprocals_l2344_234409

theorem max_sum_reciprocals (k l m : ℕ+) (h : (k : ℝ)⁻¹ + (l : ℝ)⁻¹ + (m : ℝ)⁻¹ < 1) :
  ∃ (a b c : ℕ+), (a : ℝ)⁻¹ + (b : ℝ)⁻¹ + (c : ℝ)⁻¹ = 41/42 ∧
    ∀ (x y z : ℕ+), (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ < 1 →
      (x : ℝ)⁻¹ + (y : ℝ)⁻¹ + (z : ℝ)⁻¹ ≤ 41/42 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_reciprocals_l2344_234409


namespace NUMINAMATH_CALUDE_equivalent_expression_l2344_234448

theorem equivalent_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = (2 * x^3 + 2 * y^3) / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_equivalent_expression_l2344_234448


namespace NUMINAMATH_CALUDE_shirt_cost_is_9_l2344_234430

/-- The cost of one pair of jeans -/
def jeans_cost : ℝ := sorry

/-- The cost of one shirt -/
def shirt_cost : ℝ := sorry

/-- First condition: 3 pairs of jeans and 2 shirts cost $69 -/
axiom condition1 : 3 * jeans_cost + 2 * shirt_cost = 69

/-- Second condition: 2 pairs of jeans and 3 shirts cost $61 -/
axiom condition2 : 2 * jeans_cost + 3 * shirt_cost = 61

/-- Theorem: The cost of one shirt is $9 -/
theorem shirt_cost_is_9 : shirt_cost = 9 := by sorry

end NUMINAMATH_CALUDE_shirt_cost_is_9_l2344_234430


namespace NUMINAMATH_CALUDE_problem_1_l2344_234454

theorem problem_1 : 2 + (-5) - (-4) + |(-3)| = 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_l2344_234454


namespace NUMINAMATH_CALUDE_second_quadrant_trig_simplification_l2344_234423

theorem second_quadrant_trig_simplification (α : Real) 
  (h : π/2 < α ∧ α < π) : 
  (Real.sqrt (1 + 2 * Real.sin (5 * π - α) * Real.cos (α - π))) / 
  (Real.sin (α - 3 * π / 2) - Real.sqrt (1 - Real.sin (3 * π / 2 + α)^2)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_simplification_l2344_234423


namespace NUMINAMATH_CALUDE_rope_and_well_l2344_234483

theorem rope_and_well (x y : ℝ) (h : (1/4) * x = y + 3) : (1/5) * x = y + 2 := by
  sorry

end NUMINAMATH_CALUDE_rope_and_well_l2344_234483


namespace NUMINAMATH_CALUDE_two_from_five_permutation_l2344_234458

def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem two_from_five_permutation : 
  permutations 5 2 = 20 := by sorry

end NUMINAMATH_CALUDE_two_from_five_permutation_l2344_234458


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2344_234491

/-- Given a circle tangent to lines 3x + 4y = 24 and 3x + 4y = 0,
    with its center on the line x - 2y = 0,
    prove that the center (x, y) satisfies the given equations. -/
theorem circle_center_coordinates (x y : ℚ) : 
  (∃ (r : ℚ), r > 0 ∧ 
    (∀ (x' y' : ℚ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (3*x' + 4*y' = 24 ∨ 3*x' + 4*y' = 0))) →
  x - 2*y = 0 →
  3*x + 4*y = 12 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2344_234491


namespace NUMINAMATH_CALUDE_units_digit_of_sqrt_product_sum_last_three_digits_of_2012_cubed_l2344_234418

-- Define the function to calculate the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the function to calculate the sum of the last 3 digits
def sumLastThreeDigits (n : ℕ) : ℕ := (n % 1000) / 100 + ((n % 100) / 10) + (n % 10)

-- Theorem 1
theorem units_digit_of_sqrt_product (Q : ℤ) :
  ∃ X : ℕ, X^2 = (100 * 102 * 103 * 105 + (Q - 3)) ∧ unitsDigit X = 3 := by sorry

-- Theorem 2
theorem sum_last_three_digits_of_2012_cubed :
  sumLastThreeDigits (2012^3) = 17 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sqrt_product_sum_last_three_digits_of_2012_cubed_l2344_234418


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2344_234404

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 4*x - 2*m + 5 = 0 ↔ (x = x₁ ∨ x = x₂)) →
  (x₁ ≠ x₂) →
  (m ≥ 1/2) ∧ 
  (x₁ * x₂ + x₁ + x₂ = m^2 + 6 → m = 1) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2344_234404


namespace NUMINAMATH_CALUDE_ferry_travel_time_l2344_234443

/-- Represents the travel time of Ferry P in hours -/
def t : ℝ := 3

/-- Speed of Ferry P in km/h -/
def speed_p : ℝ := 6

/-- Speed of Ferry Q in km/h -/
def speed_q : ℝ := speed_p + 3

/-- Distance traveled by Ferry P in km -/
def distance_p : ℝ := speed_p * t

/-- Distance traveled by Ferry Q in km -/
def distance_q : ℝ := 2 * distance_p

/-- Travel time of Ferry Q in hours -/
def time_q : ℝ := t + 1

theorem ferry_travel_time :
  speed_q * time_q = distance_q ∧ t = 3 := by sorry

end NUMINAMATH_CALUDE_ferry_travel_time_l2344_234443


namespace NUMINAMATH_CALUDE_lines_are_parallel_l2344_234401

-- Define the slope and y-intercept of a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the two lines
def l1 : Line := { slope := 2, intercept := 1 }
def l2 : Line := { slope := 2, intercept := 5 }

-- Define parallel lines
def parallel (a b : Line) : Prop := a.slope = b.slope

-- Theorem statement
theorem lines_are_parallel : parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_lines_are_parallel_l2344_234401


namespace NUMINAMATH_CALUDE_points_four_units_from_negative_one_l2344_234436

theorem points_four_units_from_negative_one : 
  ∀ x : ℝ, abs (x - (-1)) = 4 ↔ x = 3 ∨ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_points_four_units_from_negative_one_l2344_234436


namespace NUMINAMATH_CALUDE_y_intercepts_of_curve_l2344_234429

/-- The y-intercepts of the curve 3x + 5y^2 = 25 are (0, √5) and (0, -√5) -/
theorem y_intercepts_of_curve (x y : ℝ) :
  3*x + 5*y^2 = 25 ∧ x = 0 ↔ y = Real.sqrt 5 ∨ y = -Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_y_intercepts_of_curve_l2344_234429


namespace NUMINAMATH_CALUDE_rectangular_field_fence_l2344_234403

theorem rectangular_field_fence (area : ℝ) (fence_length : ℝ) (uncovered_side : ℝ) :
  area = 680 →
  fence_length = 146 →
  uncovered_side * (fence_length - uncovered_side) / 2 = area →
  uncovered_side = 136 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fence_l2344_234403


namespace NUMINAMATH_CALUDE_parabola_point_property_l2344_234440

/-- Given a parabola y = a(x+3)^2 + c with two points (x₁, y₁) and (x₂, y₂),
    if |x₁+3| > |x₂+3|, then a(y₁-y₂) > 0 -/
theorem parabola_point_property (a c x₁ y₁ x₂ y₂ : ℝ) :
  y₁ = a * (x₁ + 3)^2 + c →
  y₂ = a * (x₂ + 3)^2 + c →
  |x₁ + 3| > |x₂ + 3| →
  a * (y₁ - y₂) > 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_property_l2344_234440


namespace NUMINAMATH_CALUDE_parabola_vertex_l2344_234457

/-- Given a quadratic function f(x) = -x^2 + cx + d whose inequality f(x) ≤ 0
    has the solution set (-∞, -4] ∪ [6, ∞), prove that its vertex is (5, 1) -/
theorem parabola_vertex (c d : ℝ) : 
  (∀ x, -x^2 + c*x + d ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) → 
  ∃ x y, x = 5 ∧ y = 1 ∧ ∀ t, -t^2 + c*t + d ≤ -(-t + x)^2 + y :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2344_234457


namespace NUMINAMATH_CALUDE_difference_of_squares_601_599_l2344_234416

theorem difference_of_squares_601_599 : 601^2 - 599^2 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_601_599_l2344_234416


namespace NUMINAMATH_CALUDE_baseball_gear_cost_l2344_234481

/-- Calculates the total cost of baseball gear including tax -/
def total_cost (birthday_money : ℚ) (glove_price : ℚ) (glove_discount : ℚ) 
  (baseball_price : ℚ) (bat_price : ℚ) (bat_discount : ℚ) (cleats_price : ℚ) 
  (cap_price : ℚ) (tax_rate : ℚ) : ℚ :=
  let discounted_glove := glove_price * (1 - glove_discount)
  let discounted_bat := bat_price * (1 - bat_discount)
  let subtotal := discounted_glove + baseball_price + discounted_bat + cleats_price + cap_price
  let total := subtotal * (1 + tax_rate)
  total

/-- Theorem stating the total cost of baseball gear -/
theorem baseball_gear_cost : 
  total_cost 120 35 0.2 15 50 0.1 30 10 0.07 = 136.96 := by
  sorry


end NUMINAMATH_CALUDE_baseball_gear_cost_l2344_234481


namespace NUMINAMATH_CALUDE_complex_roots_nature_l2344_234464

theorem complex_roots_nature (k : ℝ) (hk : k > 0) :
  ∃ (z₁ z₂ : ℂ), 
    (10 * z₁^2 + 5 * Complex.I * z₁ - k = 0) ∧
    (10 * z₂^2 + 5 * Complex.I * z₂ - k = 0) ∧
    (z₁.re ≠ 0 ∧ z₁.im ≠ 0) ∧
    (z₂.re ≠ 0 ∧ z₂.im ≠ 0) ∧
    (z₁ ≠ z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_nature_l2344_234464


namespace NUMINAMATH_CALUDE_max_sum_for_product_1386_l2344_234442

theorem max_sum_for_product_1386 :
  ∃ (A B C : ℕ+),
    A * B * C = 1386 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    ∀ (X Y Z : ℕ+),
      X * Y * Z = 1386 →
      X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →
      X + Y + Z ≤ A + B + C ∧
      A + B + C = 88 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_for_product_1386_l2344_234442


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2344_234407

theorem root_difference_implies_k_value (k : ℝ) : 
  (∀ x₁ x₂, x₁^2 + k*x₁ + 10 = 0 → x₂^2 - k*x₂ + 10 = 0 → x₂ = x₁ + 3) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2344_234407


namespace NUMINAMATH_CALUDE_rahul_mary_age_difference_l2344_234469

/-- 
Given:
- Mary's current age is 10 years
- In 20 years, Rahul will be twice as old as Mary

Prove that Rahul is currently 30 years older than Mary
-/
theorem rahul_mary_age_difference :
  ∀ (rahul_age mary_age : ℕ),
    mary_age = 10 →
    rahul_age + 20 = 2 * (mary_age + 20) →
    rahul_age - mary_age = 30 :=
by sorry

end NUMINAMATH_CALUDE_rahul_mary_age_difference_l2344_234469


namespace NUMINAMATH_CALUDE_prove_new_average_weight_l2344_234466

def average_weight_problem (num_boys num_girls : ℕ) 
                           (avg_weight_boys avg_weight_girls : ℚ)
                           (lightest_boy_weight lightest_girl_weight : ℚ) : Prop :=
  let total_weight_boys := num_boys * avg_weight_boys
  let total_weight_girls := num_girls * avg_weight_girls
  let remaining_weight_boys := total_weight_boys - lightest_boy_weight
  let remaining_weight_girls := total_weight_girls - lightest_girl_weight
  let total_remaining_weight := remaining_weight_boys + remaining_weight_girls
  let remaining_children := num_boys + num_girls - 2
  let new_average_weight := total_remaining_weight / remaining_children
  new_average_weight = 161.5

theorem prove_new_average_weight : 
  average_weight_problem 8 5 155 125 140 110 := by
  sorry

end NUMINAMATH_CALUDE_prove_new_average_weight_l2344_234466


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l2344_234427

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a - 5) + 3 * (b - 5) = 3 * S - 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l2344_234427


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2344_234487

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a12 (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_a4 : a 4 = 4) 
  (h_a8 : a 8 = -4) : 
  a 12 = -12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a12_l2344_234487


namespace NUMINAMATH_CALUDE_inequality_addition_l2344_234478

theorem inequality_addition (x y : ℝ) (h : x < y) : x + 6 < y + 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l2344_234478


namespace NUMINAMATH_CALUDE_article_selling_price_l2344_234420

theorem article_selling_price (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) : 
  cost_price = 10 →
  gain_percent = 150 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 25 := by
sorry

end NUMINAMATH_CALUDE_article_selling_price_l2344_234420


namespace NUMINAMATH_CALUDE_inequality_range_proof_l2344_234490

theorem inequality_range_proof : 
  {x : ℝ | ∀ t : ℝ, |t - 3| + |2*t + 1| ≥ |2*x - 1| + |x + 2|} = 
  {x : ℝ | -1/2 ≤ x ∧ x ≤ 5/6} := by sorry

end NUMINAMATH_CALUDE_inequality_range_proof_l2344_234490


namespace NUMINAMATH_CALUDE_tissue_purchase_cost_l2344_234460

/-- Calculate the total cost of tissues with discounts and tax --/
theorem tissue_purchase_cost
  (boxes : ℕ)
  (packs_per_box : ℕ)
  (tissues_per_pack : ℕ)
  (price_per_tissue : ℚ)
  (pack_discount : ℚ)
  (volume_discount : ℚ)
  (tax_rate : ℚ)
  (h_boxes : boxes = 25)
  (h_packs : packs_per_box = 18)
  (h_tissues : tissues_per_pack = 150)
  (h_price : price_per_tissue = 6 / 100)
  (h_pack_disc : pack_discount = 10 / 100)
  (h_vol_disc : volume_discount = 8 / 100)
  (h_tax : tax_rate = 5 / 100)
  : ∃ (total_cost : ℚ), total_cost = 3521.07 := by
  sorry

end NUMINAMATH_CALUDE_tissue_purchase_cost_l2344_234460


namespace NUMINAMATH_CALUDE_find_k_l2344_234482

theorem find_k : ∃ k : ℚ, (32 / k = 4) ∧ (k = 8) := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2344_234482


namespace NUMINAMATH_CALUDE_sandy_carrots_l2344_234400

theorem sandy_carrots (initial_carrots : ℕ) (taken_carrots : ℕ) :
  initial_carrots = 6 →
  taken_carrots = 3 →
  initial_carrots - taken_carrots = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_carrots_l2344_234400


namespace NUMINAMATH_CALUDE_four_men_absent_l2344_234499

/-- Represents the work scenario with contractors -/
structure WorkScenario where
  totalMen : ℕ
  plannedDays : ℕ
  actualDays : ℕ
  absentMen : ℕ

/-- The work scenario satisfies the given conditions -/
def validScenario (w : WorkScenario) : Prop :=
  w.totalMen = 10 ∧ w.plannedDays = 6 ∧ w.actualDays = 10 ∧
  w.totalMen * w.plannedDays = (w.totalMen - w.absentMen) * w.actualDays

/-- The theorem stating that 4 men were absent -/
theorem four_men_absent :
  ∃ (w : WorkScenario), validScenario w ∧ w.absentMen = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_men_absent_l2344_234499


namespace NUMINAMATH_CALUDE_equal_savings_l2344_234424

-- Define the total combined salary
def total_salary : ℝ := 6000

-- Define A's salary
def salary_A : ℝ := 4500

-- Define B's salary
def salary_B : ℝ := total_salary - salary_A

-- Define A's spending rate
def spending_rate_A : ℝ := 0.95

-- Define B's spending rate
def spending_rate_B : ℝ := 0.85

-- Define A's savings
def savings_A : ℝ := salary_A * (1 - spending_rate_A)

-- Define B's savings
def savings_B : ℝ := salary_B * (1 - spending_rate_B)

-- Theorem: A and B have the same savings
theorem equal_savings : savings_A = savings_B := by
  sorry

end NUMINAMATH_CALUDE_equal_savings_l2344_234424


namespace NUMINAMATH_CALUDE_range_of_a_l2344_234489

theorem range_of_a (a : ℝ) : 
  (∀ x, x^2 - 8*x - 20 > 0 → x^2 - 2*x + 1 - a^2 > 0) ∧ 
  (∃ x, x^2 - 2*x + 1 - a^2 > 0 ∧ x^2 - 8*x - 20 ≤ 0) ∧
  (a > 0) →
  3 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2344_234489


namespace NUMINAMATH_CALUDE_integer_terms_count_l2344_234446

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a₃ : ℝ  -- 3rd term
  a₁₈ : ℝ -- 18th term
  h₃ : a₃ = 14
  h₁₈ : a₁₈ = 23

/-- The number of integer terms in the first 2010 terms of the sequence -/
def integerTermCount (seq : ArithmeticSequence) : ℕ :=
  402

/-- Theorem stating the number of integer terms in the first 2010 terms -/
theorem integer_terms_count (seq : ArithmeticSequence) :
  integerTermCount seq = 402 := by
  sorry

end NUMINAMATH_CALUDE_integer_terms_count_l2344_234446


namespace NUMINAMATH_CALUDE_linear_increase_l2344_234419

theorem linear_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 6) :
  ∀ x, f (x + 12) - f x = 18 := by
  sorry

end NUMINAMATH_CALUDE_linear_increase_l2344_234419


namespace NUMINAMATH_CALUDE_no_base_for_perfect_square_l2344_234475

theorem no_base_for_perfect_square : ¬ ∃ (b : ℕ), ∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_perfect_square_l2344_234475


namespace NUMINAMATH_CALUDE_value_of_a_l2344_234445

theorem value_of_a (a : ℝ) : -3 ∈ ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l2344_234445


namespace NUMINAMATH_CALUDE_nine_hash_seven_l2344_234463

-- Define the # operation
noncomputable def hash (r s : ℝ) : ℝ :=
  sorry

-- State the conditions
axiom hash_zero (r : ℝ) : hash r 0 = r
axiom hash_comm (r s : ℝ) : hash r s = hash s r
axiom hash_succ (r s : ℝ) : hash (r + 1) s = hash r s + s + 1

-- State the theorem to be proved
theorem nine_hash_seven : hash 9 7 = 79 := by
  sorry

end NUMINAMATH_CALUDE_nine_hash_seven_l2344_234463


namespace NUMINAMATH_CALUDE_multiply_specific_numbers_l2344_234474

theorem multiply_specific_numbers : 469160 * 9999 = 4691183840 := by
  sorry

end NUMINAMATH_CALUDE_multiply_specific_numbers_l2344_234474


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2344_234496

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n.choose 2) - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2344_234496


namespace NUMINAMATH_CALUDE_auto_store_sales_time_l2344_234484

theorem auto_store_sales_time (total_cars : ℕ) (salespeople : ℕ) (cars_per_person : ℕ) :
  total_cars = 500 →
  salespeople = 10 →
  cars_per_person = 10 →
  (total_cars / (salespeople * cars_per_person) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_auto_store_sales_time_l2344_234484


namespace NUMINAMATH_CALUDE_sandy_initial_fish_count_l2344_234468

theorem sandy_initial_fish_count (initial_fish final_fish bought_fish : ℕ) 
  (h1 : final_fish = initial_fish + bought_fish)
  (h2 : final_fish = 32)
  (h3 : bought_fish = 6) : 
  initial_fish = 26 := by
sorry

end NUMINAMATH_CALUDE_sandy_initial_fish_count_l2344_234468


namespace NUMINAMATH_CALUDE_billy_points_billy_points_proof_l2344_234494

theorem billy_points : ℕ → Prop := fun b =>
  let friend_points : ℕ := 9
  let point_difference : ℕ := 2
  (b - friend_points = point_difference) → (b = 11)

-- The proof is omitted
theorem billy_points_proof : billy_points 11 := by sorry

end NUMINAMATH_CALUDE_billy_points_billy_points_proof_l2344_234494


namespace NUMINAMATH_CALUDE_symmetry_sum_l2344_234438

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetry_sum (m n : ℝ) : 
  symmetric_wrt_origin (m, 1) (-2, n) → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l2344_234438


namespace NUMINAMATH_CALUDE_solve_equation_l2344_234415

theorem solve_equation (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 8 → y = 40 / 3 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_l2344_234415


namespace NUMINAMATH_CALUDE_farmers_market_spending_l2344_234444

theorem farmers_market_spending (sandi_initial : ℕ) (gillian_total : ℕ) : 
  sandi_initial = 600 →
  gillian_total = 1050 →
  ∃ (multiple : ℕ), 
    gillian_total = (sandi_initial / 2) + (multiple * (sandi_initial / 2)) + 150 ∧
    multiple = 1 :=
by sorry

end NUMINAMATH_CALUDE_farmers_market_spending_l2344_234444


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2344_234465

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- The probability of an individual being selected given a population size, sample size, and sampling method -/
noncomputable def selectionProbability (N n : ℕ) (method : SamplingMethod) : ℝ :=
  sorry

theorem equal_selection_probability (N n : ℕ) (h1 : N > 0) (h2 : n > 0) (h3 : n ≤ N) :
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Systematic ∧
  selectionProbability N n SamplingMethod.SimpleRandom =
  selectionProbability N n SamplingMethod.Stratified :=
by
  sorry

end NUMINAMATH_CALUDE_equal_selection_probability_l2344_234465


namespace NUMINAMATH_CALUDE_expression_evaluation_l2344_234488

theorem expression_evaluation (a : ℚ) (h : a = -1/3) : 
  (2 - a) * (2 + a) - 2 * a * (a + 3) + 3 * a^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2344_234488


namespace NUMINAMATH_CALUDE_orange_groups_indeterminate_philips_collection_valid_philips_orange_groups_indeterminate_l2344_234405

/-- Represents a fruit collection with oranges and bananas -/
structure FruitCollection where
  oranges : ℕ
  bananas : ℕ
  banana_groups : ℕ
  bananas_per_group : ℕ

/-- Predicate to check if the banana distribution is valid -/
def valid_banana_distribution (fc : FruitCollection) : Prop :=
  fc.bananas = fc.banana_groups * fc.bananas_per_group

/-- Theorem stating that the number of orange groups cannot be determined -/
theorem orange_groups_indeterminate (fc : FruitCollection) 
  (h1 : fc.oranges > 0)
  (h2 : valid_banana_distribution fc) :
  ¬ ∃ (orange_groups : ℕ), orange_groups > 0 ∧ ∀ (oranges_per_group : ℕ), fc.oranges = orange_groups * oranges_per_group :=
by
  sorry

/-- Philip's fruit collection -/
def philips_collection : FruitCollection :=
  { oranges := 87
  , bananas := 290
  , banana_groups := 2
  , bananas_per_group := 145 }

/-- Proof that Philip's collection satisfies the conditions -/
theorem philips_collection_valid :
  valid_banana_distribution philips_collection :=
by
  sorry

/-- Application of the theorem to Philip's collection -/
theorem philips_orange_groups_indeterminate :
  ¬ ∃ (orange_groups : ℕ), orange_groups > 0 ∧ ∀ (oranges_per_group : ℕ), philips_collection.oranges = orange_groups * oranges_per_group :=
by
  apply orange_groups_indeterminate
  · simp [philips_collection]
  · exact philips_collection_valid

end NUMINAMATH_CALUDE_orange_groups_indeterminate_philips_collection_valid_philips_orange_groups_indeterminate_l2344_234405


namespace NUMINAMATH_CALUDE_different_grade_selections_l2344_234447

/-- The number of students in the first year -/
def first_year_students : ℕ := 4

/-- The number of students in the second year -/
def second_year_students : ℕ := 5

/-- The number of students in the third year -/
def third_year_students : ℕ := 4

/-- The total number of ways to select 2 students from different grades -/
def total_selections : ℕ := 56

theorem different_grade_selections :
  first_year_students * second_year_students +
  first_year_students * third_year_students +
  second_year_students * third_year_students = total_selections :=
by sorry

end NUMINAMATH_CALUDE_different_grade_selections_l2344_234447


namespace NUMINAMATH_CALUDE_square_sum_inequality_l2344_234497

theorem square_sum_inequality (a b : ℝ) : a^2 + b^2 ≥ 2*(a - b - 1) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l2344_234497


namespace NUMINAMATH_CALUDE_range_of_a_l2344_234431

theorem range_of_a (a : ℝ) : a > 5 → ∃ x : ℝ, x > -1 ∧ (x^2 + 3*x + 6) / (x + 1) < a := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2344_234431


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2344_234473

/-- The radius of a cylinder inscribed in a cone with specific dimensions -/
theorem inscribed_cylinder_radius (cylinder_height : ℝ) (cylinder_radius : ℝ) 
  (cone_diameter : ℝ) (cone_altitude : ℝ) : 
  cylinder_height = 2 * cylinder_radius →
  cone_diameter = 8 →
  cone_altitude = 10 →
  cylinder_radius = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l2344_234473


namespace NUMINAMATH_CALUDE_alik_collection_l2344_234428

theorem alik_collection (badges bracelets : ℕ) (n : ℚ) : 
  badges > bracelets →
  badges + n * bracelets = 100 →
  n * badges + bracelets = 101 →
  ((badges = 34 ∧ bracelets = 33) ∨ (badges = 66 ∧ bracelets = 33)) :=
by sorry

end NUMINAMATH_CALUDE_alik_collection_l2344_234428


namespace NUMINAMATH_CALUDE_det_special_matrix_l2344_234461

theorem det_special_matrix (a b : ℝ) : 
  Matrix.det ![![1, a, b], ![1, a+b, b], ![1, a, a+b]] = a * b := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l2344_234461


namespace NUMINAMATH_CALUDE_histogram_area_sum_is_one_l2344_234471

/-- Represents a histogram of sample frequency distribution -/
structure Histogram where
  rectangles : List ℝ
  -- Each element in the list represents the area of a small rectangle

/-- The sum of areas of all rectangles in a histogram equals 1 -/
theorem histogram_area_sum_is_one (h : Histogram) : 
  h.rectangles.sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_histogram_area_sum_is_one_l2344_234471


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2344_234485

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  q ≠ 1 →                          -- q ≠ 1 condition
  a 2 = 9 →                        -- a_2 = 9 condition
  a 3 + a 4 = 18 →                 -- a_3 + a_4 = 18 condition
  q = -2 :=                        -- conclusion: q = -2
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2344_234485


namespace NUMINAMATH_CALUDE_modulus_of_z_l2344_234421

theorem modulus_of_z (z : ℂ) (h : z * (-1 + 2*I) = 5*I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2344_234421


namespace NUMINAMATH_CALUDE_angle_A_value_max_area_l2344_234425

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def givenCondition (t : Triangle) : Prop :=
  (-t.b + Real.sqrt 2 * t.c) / Real.cos t.B = t.a / Real.cos t.A

-- Theorem 1: Prove that A = π/4
theorem angle_A_value (t : Triangle) (h : givenCondition t) : t.A = π / 4 := by
  sorry

-- Theorem 2: Prove the maximum area when a = 2
theorem max_area (t : Triangle) (h : givenCondition t) (ha : t.a = 2) :
  ∃ (S : ℝ), S = Real.sqrt 2 + 1 ∧ ∀ (S' : ℝ), S' ≤ S := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_max_area_l2344_234425


namespace NUMINAMATH_CALUDE_rectangle_length_l2344_234472

/-- Given a rectangle with width 6 inches and area 48 square inches, prove its length is 8 inches -/
theorem rectangle_length (width : ℝ) (area : ℝ) (h1 : width = 6) (h2 : area = 48) :
  area / width = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_l2344_234472


namespace NUMINAMATH_CALUDE_school_female_students_l2344_234439

theorem school_female_students 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (sample_difference : ℕ) : 
  total_students = 1600 → 
  sample_size = 200 → 
  sample_difference = 10 →
  (∃ (female_students : ℕ), 
    female_students = 760 ∧ 
    female_students + (total_students - female_students) = total_students ∧
    (female_students : ℚ) / (total_students - female_students) = 
      ((sample_size / 2 - sample_difference / 2) : ℚ) / (sample_size / 2 + sample_difference / 2)) :=
by sorry

end NUMINAMATH_CALUDE_school_female_students_l2344_234439


namespace NUMINAMATH_CALUDE_zero_subset_X_l2344_234456

-- Define the set X
def X : Set ℝ := {x | x > -4}

-- State the theorem
theorem zero_subset_X : {0} ⊆ X := by sorry

end NUMINAMATH_CALUDE_zero_subset_X_l2344_234456


namespace NUMINAMATH_CALUDE_milk_water_ratio_in_first_vessel_l2344_234422

-- Define the volumes of the vessels
def vessel1_volume : ℚ := 3
def vessel2_volume : ℚ := 5

-- Define the milk to water ratio in the second vessel
def vessel2_milk_ratio : ℚ := 6
def vessel2_water_ratio : ℚ := 4

-- Define the mixed ratio
def mixed_ratio : ℚ := 1

-- Define the unknown ratio for the first vessel
def vessel1_milk_ratio : ℚ := 1
def vessel1_water_ratio : ℚ := 2

theorem milk_water_ratio_in_first_vessel :
  (vessel1_milk_ratio / vessel1_water_ratio = 1 / 2) ∧
  (vessel1_milk_ratio * vessel1_volume + vessel2_milk_ratio * vessel2_volume) /
  (vessel1_water_ratio * vessel1_volume + vessel2_water_ratio * vessel2_volume) = mixed_ratio :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_in_first_vessel_l2344_234422


namespace NUMINAMATH_CALUDE_arrange_balls_theorem_l2344_234412

/-- The number of ways to arrange balls of different types in a row -/
def arrangeMultisetBalls (n₁ n₂ n₃ : ℕ) : ℕ :=
  Nat.factorial (n₁ + n₂ + n₃) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃)

/-- Theorem stating that arranging 5 basketballs, 3 volleyballs, and 2 footballs yields 2520 ways -/
theorem arrange_balls_theorem : arrangeMultisetBalls 5 3 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_arrange_balls_theorem_l2344_234412
