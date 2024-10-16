import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_at_one_l1521_152127

/-- The function f(x) = -x³ + 4x -/
def f (x : ℝ) : ℝ := -x^3 + 4*x

/-- The derivative of f(x) -/
def f_derivative (x : ℝ) : ℝ := -3*x^2 + 4

theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f_derivative x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = x + 2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l1521_152127


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_power_2017_l1521_152195

theorem imaginary_part_of_i_power_2017 : Complex.im (Complex.I ^ 2017) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_power_2017_l1521_152195


namespace NUMINAMATH_CALUDE_parabola_chord_perpendicular_bisector_l1521_152119

/-- The parabola y^2 = 8(x+2) with focus at (0, 0) -/
def parabola (x y : ℝ) : Prop := y^2 = 8*(x+2)

/-- The line y = x passing through (0, 0) -/
def line (x y : ℝ) : Prop := y = x

/-- The perpendicular bisector of a chord on the line y = x -/
def perp_bisector (x y : ℝ) : Prop := y = -x + 2*x

theorem parabola_chord_perpendicular_bisector :
  ∀ (x : ℝ),
  (∃ (y : ℝ), parabola x y ∧ line x y) →
  (∃ (P : ℝ × ℝ), P.1 = x ∧ P.2 = 0 ∧ perp_bisector P.1 P.2) →
  x = x := by sorry

end NUMINAMATH_CALUDE_parabola_chord_perpendicular_bisector_l1521_152119


namespace NUMINAMATH_CALUDE_solve_system_l1521_152129

theorem solve_system (s t : ℚ) 
  (eq1 : 7 * s + 8 * t = 150)
  (eq2 : s = 2 * t + 3) : 
  s = 162 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1521_152129


namespace NUMINAMATH_CALUDE_b_2017_eq_1_l1521_152175

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Sequence of Fibonacci numbers modulo 3 -/
def b (n : ℕ) : ℕ := fib n % 3

/-- The sequence b has period 8 -/
axiom b_period (n : ℕ) : b (n + 8) = b n

theorem b_2017_eq_1 : b 2017 = 1 := by sorry

end NUMINAMATH_CALUDE_b_2017_eq_1_l1521_152175


namespace NUMINAMATH_CALUDE_quadratic_factorization_l1521_152137

theorem quadratic_factorization (a b : ℤ) :
  (∀ y : ℝ, 2 * y^2 + 5 * y - 12 = (2 * y + a) * (y + b)) →
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l1521_152137


namespace NUMINAMATH_CALUDE_volume_is_eleven_sixths_l1521_152181

-- Define the region
def region (x y z : ℝ) : Prop :=
  (abs x + abs y + abs z ≤ 1) ∧ (abs x + abs y + abs (z - 2) ≤ 1)

-- Define the volume of the region
noncomputable def volume_of_region : ℝ := sorry

-- Theorem statement
theorem volume_is_eleven_sixths : volume_of_region = 11/6 := by sorry

end NUMINAMATH_CALUDE_volume_is_eleven_sixths_l1521_152181


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1521_152135

/-- Proves that for an angle α in the second quadrant, 
    if 2sin(2α) = cos(2α) - 1, then cos(α) = -√5/5 -/
theorem cos_alpha_value (α : Real) 
  (h1 : π/2 < α ∧ α < π) -- α is in the second quadrant
  (h2 : 2 * Real.sin (2 * α) = Real.cos (2 * α) - 1) -- given equation
  : Real.cos α = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1521_152135


namespace NUMINAMATH_CALUDE_total_peanuts_l1521_152174

def jose_peanuts : ℕ := 85
def kenya_peanuts : ℕ := jose_peanuts + 48
def marcos_peanuts : ℕ := kenya_peanuts + 37

theorem total_peanuts : jose_peanuts + kenya_peanuts + marcos_peanuts = 388 := by
  sorry

end NUMINAMATH_CALUDE_total_peanuts_l1521_152174


namespace NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l1521_152128

theorem smallest_value_of_sum_of_cubes (a b : ℂ) 
  (h1 : Complex.abs (a + b) = 2)
  (h2 : Complex.abs (a^2 + b^2) = 8) :
  20 ≤ Complex.abs (a^3 + b^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_value_of_sum_of_cubes_l1521_152128


namespace NUMINAMATH_CALUDE_four_students_three_lectures_l1521_152130

/-- The number of ways students can choose lectures -/
def lecture_choices (num_students : ℕ) (num_lectures : ℕ) : ℕ :=
  num_lectures ^ num_students

/-- Theorem: 4 students choosing from 3 lectures results in 81 different selections -/
theorem four_students_three_lectures : 
  lecture_choices 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_lectures_l1521_152130


namespace NUMINAMATH_CALUDE_lilly_fish_count_l1521_152154

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 8

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 18

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := total_fish - rosy_fish

theorem lilly_fish_count : lilly_fish = 10 := by
  sorry

end NUMINAMATH_CALUDE_lilly_fish_count_l1521_152154


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1521_152124

/-- Given a principal sum and a time period of 7 years, if the simple interest
    is one-fifth of the principal, prove that the annual interest rate is 20/7. -/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  (P * 7 * (20 / 7) / 100 = P / 5) → (20 / 7 : ℝ) = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1521_152124


namespace NUMINAMATH_CALUDE_johnnys_laps_per_minute_l1521_152133

/-- 
Given that Johnny ran 10 laps in 3.333 minutes, 
prove that the number of laps he runs per minute is equal to 10 divided by 3.333.
-/
theorem johnnys_laps_per_minute (total_laps : ℝ) (total_minutes : ℝ) 
  (h1 : total_laps = 10) 
  (h2 : total_minutes = 3.333) : 
  total_laps / total_minutes = 10 / 3.333 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_laps_per_minute_l1521_152133


namespace NUMINAMATH_CALUDE_monday_sales_calculation_l1521_152190

def total_stock : ℕ := 1200
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 665/1000

theorem monday_sales_calculation :
  ∃ (monday_sales : ℕ),
    monday_sales = total_stock - 
      (tuesday_sales + wednesday_sales + thursday_sales + friday_sales) - 
      (unsold_percentage * total_stock).num :=
by sorry

end NUMINAMATH_CALUDE_monday_sales_calculation_l1521_152190


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l1521_152182

/-- Given a geometric sequence {a_n} with common ratio q and a_2 = 8, prove that a_6 = 128 -/
theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 2 = 8) 
  (h3 : ∀ n : ℕ, a (n + 1) = q * a n) : a 6 = 128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l1521_152182


namespace NUMINAMATH_CALUDE_log_base_value_log_inequality_greater_than_one_log_inequality_less_than_one_l1521_152114

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem 1
theorem log_base_value (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  log a 8 = 3 → a = 2 := by sorry

-- Theorem 2
theorem log_inequality_greater_than_one (a : ℝ) (ha : a > 1) (x : ℝ) :
  log a x ≤ log a (2 - 3*x) ↔ (0 < x ∧ x ≤ 1/2) := by sorry

-- Theorem 3
theorem log_inequality_less_than_one (a : ℝ) (ha : 0 < a ∧ a < 1) (x : ℝ) :
  log a x ≤ log a (2 - 3*x) ↔ (1/2 ≤ x ∧ x < 2/3) := by sorry

end NUMINAMATH_CALUDE_log_base_value_log_inequality_greater_than_one_log_inequality_less_than_one_l1521_152114


namespace NUMINAMATH_CALUDE_even_function_property_l1521_152134

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def has_min_value (f : ℝ → ℝ) (m : ℝ) : Prop := ∀ x, m ≤ f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem even_function_property (f : ℝ → ℝ) :
  is_even f →
  is_increasing_on f 1 3 →
  has_min_value f 0 →
  is_decreasing_on f (-3) (-1) ∧ has_min_value f 0 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l1521_152134


namespace NUMINAMATH_CALUDE_min_value_problem_l1521_152172

theorem min_value_problem (y₁ y₂ y₃ : ℝ) 
  (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 75) : 
  y₁^2 + 2 * y₂^2 + 3 * y₃^2 ≥ 5625 / 29 ∧ 
  ∃ y₁' y₂' y₃', y₁'^2 + 2 * y₂'^2 + 3 * y₃'^2 = 5625 / 29 ∧ 
                 y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 
                 2 * y₁' + 3 * y₂' + 4 * y₃' = 75 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l1521_152172


namespace NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1521_152153

theorem orange_crates_pigeonhole (total_crates : ℕ) (min_oranges max_oranges : ℕ) :
  total_crates = 200 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ n : ℕ, n ≥ 7 ∧ 
    ∃ k : ℕ, min_oranges ≤ k ∧ k ≤ max_oranges ∧
      (∃ subset : Finset (Fin total_crates), subset.card = n ∧
        ∀ i ∈ subset, ∃ f : Fin total_crates → ℕ, 
          (∀ j, min_oranges ≤ f j ∧ f j ≤ max_oranges) ∧ f i = k) :=
by sorry

end NUMINAMATH_CALUDE_orange_crates_pigeonhole_l1521_152153


namespace NUMINAMATH_CALUDE_football_game_spectators_l1521_152131

theorem football_game_spectators (total_wristbands : ℕ) (wristbands_per_person : ℕ) 
  (h1 : total_wristbands = 290)
  (h2 : wristbands_per_person = 2)
  : total_wristbands / wristbands_per_person = 145 := by
  sorry

end NUMINAMATH_CALUDE_football_game_spectators_l1521_152131


namespace NUMINAMATH_CALUDE_existence_of_bounded_difference_l1521_152191

theorem existence_of_bounded_difference (n : ℕ) (x : Fin n → ℝ) 
  (h_n : n ≥ 3) 
  (h_pos : ∀ i, x i > 0) 
  (h_distinct : ∀ i j, i ≠ j → x i ≠ x j) : 
  ∃ i j, i ≠ j ∧ 
    0 < (x i - x j) / (1 + x i * x j) ∧ 
    (x i - x j) / (1 + x i * x j) < Real.tan (π / (2 * (n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_bounded_difference_l1521_152191


namespace NUMINAMATH_CALUDE_smallest_valid_integers_difference_l1521_152199

def is_valid_integer (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, 2 ≤ k ∧ k ≤ 12 → n % k = 1

theorem smallest_valid_integers_difference : 
  ∃ n₁ n₂ : ℕ, 
    is_valid_integer n₁ ∧
    is_valid_integer n₂ ∧
    n₁ < n₂ ∧
    (∀ m : ℕ, is_valid_integer m → m ≥ n₁) ∧
    (∀ m : ℕ, is_valid_integer m ∧ m ≠ n₁ → m ≥ n₂) ∧
    n₂ - n₁ = 27720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_integers_difference_l1521_152199


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l1521_152103

/-- Calculates the principal amount given the simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: If a loan at 12% annual simple interest generates Rs. 1500 interest in 10 years, 
    then the principal amount was Rs. 1250 -/
theorem loan_principal_calculation :
  let interest : ℚ := 1500
  let rate : ℚ := 12
  let time : ℚ := 10
  calculate_principal interest rate time = 1250 := by
sorry

end NUMINAMATH_CALUDE_loan_principal_calculation_l1521_152103


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1521_152125

theorem pure_imaginary_complex_number (m : ℝ) :
  let z : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)
  (z.re = 0 ∧ z.im ≠ 0) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1521_152125


namespace NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_l1521_152150

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, a₁ < a₃ if and only if aₙ < aₙ₊₁ for all n -/
theorem arithmetic_sequence_increasing_iff (a : ℕ → ℝ) :
  arithmetic_sequence a → (a 1 < a 3 ↔ ∀ n : ℕ, a n < a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_increasing_iff_l1521_152150


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1521_152101

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l1521_152101


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1521_152161

theorem min_value_trig_expression (x : Real) (h : 0 < x ∧ x < π / 2) :
  (8 / Real.sin x) + (1 / Real.cos x) ≥ 5 * Real.sqrt 5 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (8 / Real.sin y) + (1 / Real.cos y) = 5 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1521_152161


namespace NUMINAMATH_CALUDE_min_green_beads_l1521_152171

/-- Represents a necklace with red, blue, and green beads. -/
structure Necklace where
  total : Nat
  red : Nat
  blue : Nat
  green : Nat
  total_sum : red + blue + green = total
  red_between_blue : red ≥ blue
  green_between_red : green ≥ red

/-- The minimum number of green beads in a necklace of 80 beads
    satisfying the given conditions is 27. -/
theorem min_green_beads (n : Necklace) (h : n.total = 80) :
  n.green ≥ 27 := by sorry

end NUMINAMATH_CALUDE_min_green_beads_l1521_152171


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1521_152156

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∀ z : ℂ, z = Complex.mk (a^2 + 2*a - 3) (a^2 + a - 6) → z.re = 0) → 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1521_152156


namespace NUMINAMATH_CALUDE_ball_probabilities_l1521_152177

def initial_red_balls : ℕ := 5
def initial_black_balls : ℕ := 7
def additional_balls : ℕ := 6

def probability_red : ℚ := initial_red_balls / (initial_red_balls + initial_black_balls)
def probability_black : ℚ := initial_black_balls / (initial_red_balls + initial_black_balls)

def added_red_balls : ℕ := 4
def added_black_balls : ℕ := 2

theorem ball_probabilities :
  (probability_black > probability_red) ∧
  ((initial_red_balls + added_red_balls) / (initial_red_balls + initial_black_balls + additional_balls) =
   (initial_black_balls + added_black_balls) / (initial_red_balls + initial_black_balls + additional_balls)) :=
by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l1521_152177


namespace NUMINAMATH_CALUDE_unique_prime_perfect_square_l1521_152193

theorem unique_prime_perfect_square : 
  ∃! p : ℕ, Prime p ∧ ∃ k : ℕ, 5 * p * (2^(p+1) - 1) = k^2 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_square_l1521_152193


namespace NUMINAMATH_CALUDE_digits_of_powers_l1521_152145

/-- A number is even and not divisible by 10 -/
def IsEvenNotDivBy10 (n : ℕ) : Prop :=
  Even n ∧ ¬(10 ∣ n)

/-- The tens digit of a natural number -/
def TensDigit (n : ℕ) : ℕ :=
  (n / 10) % 10

/-- The hundreds digit of a natural number -/
def HundredsDigit (n : ℕ) : ℕ :=
  (n / 100) % 10

theorem digits_of_powers (N : ℕ) (h : IsEvenNotDivBy10 N) :
  TensDigit (N^20) = 7 ∧ HundredsDigit (N^200) = 3 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_powers_l1521_152145


namespace NUMINAMATH_CALUDE_translation_problem_l1521_152158

-- Define a translation of the complex plane
def translation (w : ℂ) : ℂ → ℂ := λ z ↦ z + w

-- Theorem statement
theorem translation_problem (w : ℂ) 
  (h : translation w (1 + 2*I) = 3 + 6*I) : 
  translation w (2 + 3*I) = 4 + 7*I := by
  sorry

end NUMINAMATH_CALUDE_translation_problem_l1521_152158


namespace NUMINAMATH_CALUDE_y_derivative_l1521_152112

noncomputable def y (x : ℝ) : ℝ := 
  Real.sqrt ((Real.tan x + Real.sqrt (2 * Real.tan x) + 1) / (Real.tan x - Real.sqrt (2 * Real.tan x) + 1))

theorem y_derivative (x : ℝ) : 
  deriv y x = 0 :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l1521_152112


namespace NUMINAMATH_CALUDE_equation_solution_l1521_152126

theorem equation_solution (a : ℝ) :
  (a ≠ 0 → ∃! x : ℝ, x ≠ 0 ∧ x ≠ a ∧ 3 * x^2 + 2 * a * x - a^2 = Real.log ((x - a) / (2 * x))) ∧
  (a = 0 → ¬∃ x : ℝ, x ≠ 0 ∧ 3 * x^2 = Real.log (1 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1521_152126


namespace NUMINAMATH_CALUDE_parallel_vectors_cos_relation_l1521_152142

/-- Given two parallel vectors a and b, prove that cos(π/2 + α) = -1/3 -/
theorem parallel_vectors_cos_relation (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ) 
  (h1 : a = (1/3, Real.tan α))
  (h2 : b = (Real.cos α, 1))
  (h3 : ∃ (k : ℝ), a = k • b) : 
  Real.cos (π/2 + α) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_cos_relation_l1521_152142


namespace NUMINAMATH_CALUDE_second_carpenter_proof_l1521_152188

/-- The time taken by the second carpenter to complete the job alone -/
def second_carpenter_time : ℚ :=
  10 / 3

theorem second_carpenter_proof (first_carpenter_time : ℚ) 
  (first_carpenter_initial_work : ℚ) (combined_work_time : ℚ) :
  first_carpenter_time = 5 →
  first_carpenter_initial_work = 1 →
  combined_work_time = 2 →
  second_carpenter_time = 10 / 3 :=
by
  sorry

#eval second_carpenter_time

end NUMINAMATH_CALUDE_second_carpenter_proof_l1521_152188


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1521_152149

theorem sum_of_reciprocal_equations (x y : ℚ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -8) : 
  x + y = -1/3 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_equations_l1521_152149


namespace NUMINAMATH_CALUDE_reflection_of_F_l1521_152170

/-- Reflects a point over the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

/-- The composition of reflecting over y-axis and then y = x -/
def double_reflection (p : ℝ × ℝ) : ℝ × ℝ := reflect_y_eq_x (reflect_y_axis p)

theorem reflection_of_F :
  double_reflection (5, 2) = (2, -5) := by sorry

end NUMINAMATH_CALUDE_reflection_of_F_l1521_152170


namespace NUMINAMATH_CALUDE_triangle_problem_l1521_152164

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The dot product of two vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ := sorry

theorem triangle_problem (t : Triangle) 
  (h_area : area t = 30)
  (h_cos : Real.cos t.A = 12/13) : 
  ∃ (ab ac : ℝ × ℝ), 
    dotProduct ab ac = 144 ∧ 
    (t.c - t.b = 1 → t.a = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l1521_152164


namespace NUMINAMATH_CALUDE_system_solution_l1521_152198

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1521_152198


namespace NUMINAMATH_CALUDE_consecutive_integer_averages_l1521_152187

theorem consecutive_integer_averages (c : ℤ) (d : ℚ) : 
  (c > 0) →
  (d = (7 * c + 21) / 7) →
  ((7 * d + 21) / 7 = c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integer_averages_l1521_152187


namespace NUMINAMATH_CALUDE_chips_sales_problem_l1521_152179

theorem chips_sales_problem (total_sales : ℕ) (first_week : ℕ) (second_week : ℕ) :
  total_sales = 100 →
  first_week = 15 →
  second_week = 3 * first_week →
  ∃ (third_fourth_week : ℕ),
    third_fourth_week * 2 = total_sales - (first_week + second_week) ∧
    third_fourth_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_chips_sales_problem_l1521_152179


namespace NUMINAMATH_CALUDE_max_dot_product_in_triangle_l1521_152165

theorem max_dot_product_in_triangle (A B C P : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let BAC := Real.arccos ((AB^2 + AC^2 - (B.1 - C.1)^2 - (B.2 - C.2)^2) / (2 * AB * AC))
  let AP := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := (B.1 - P.1, B.2 - P.2)
  let PC := (C.1 - P.1, C.2 - P.2)
  let dot_product := PB.1 * PC.1 + PB.2 * PC.2
  AB = 3 ∧ AC = 4 ∧ BAC = π/3 ∧ AP = 2 →
  ∃ P_max : ℝ × ℝ, dot_product ≤ 10 + 2 * Real.sqrt 37 ∧
            ∃ P_actual : ℝ × ℝ, dot_product = 10 + 2 * Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_max_dot_product_in_triangle_l1521_152165


namespace NUMINAMATH_CALUDE_specific_enclosed_area_l1521_152194

/-- The area enclosed by a curve composed of 9 congruent circular arcs, where the centers of the
    corresponding circles are among the vertices of a regular hexagon. -/
def enclosed_area (arc_length : ℝ) (hexagon_side : ℝ) : ℝ :=
  sorry

/-- The theorem stating that the area enclosed by the specific curve described in the problem
    is equal to (27√3)/2 + (1125π²)/96. -/
theorem specific_enclosed_area :
  enclosed_area (5 * π / 6) 3 = (27 * Real.sqrt 3) / 2 + (1125 * π^2) / 96 := by
  sorry

end NUMINAMATH_CALUDE_specific_enclosed_area_l1521_152194


namespace NUMINAMATH_CALUDE_total_cars_is_32_l1521_152148

/-- Given the number of cars owned by Cathy, calculate the total number of cars owned by Cathy, Carol, Susan, and Lindsey. -/
def totalCars (cathyCars : ℕ) : ℕ :=
  let carolCars := 2 * cathyCars
  let susanCars := carolCars - 2
  let lindseyCars := cathyCars + 4
  cathyCars + carolCars + susanCars + lindseyCars

/-- Theorem stating that given the conditions in the problem, the total number of cars is 32. -/
theorem total_cars_is_32 : totalCars 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cars_is_32_l1521_152148


namespace NUMINAMATH_CALUDE_least_common_multiple_345667_l1521_152122

theorem least_common_multiple_345667 :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (3 ∣ m) ∧ (4 ∣ m) ∧ (5 ∣ m) ∧ (6 ∣ m) ∧ (7 ∣ m) → n ≤ m) ∧
  (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (7 ∣ n) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_345667_l1521_152122


namespace NUMINAMATH_CALUDE_cannot_buy_without_change_l1521_152168

theorem cannot_buy_without_change (zloty_to_grosz : ℕ) (total_zloty : ℕ) (item_price_grosz : ℕ) :
  zloty_to_grosz = 1001 →
  total_zloty = 1986 →
  item_price_grosz = 1987 →
  ¬ (∃ n : ℕ, n * item_price_grosz = total_zloty * zloty_to_grosz) :=
by sorry

end NUMINAMATH_CALUDE_cannot_buy_without_change_l1521_152168


namespace NUMINAMATH_CALUDE_two_questions_sufficient_l1521_152140

/-- Represents a person who is either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents a position on a 2D plane -/
structure Position :=
  (x : ℝ)
  (y : ℝ)

/-- Represents the table with 10 people -/
structure Table :=
  (people : Fin 10 → Person)
  (positions : Fin 10 → Position)

/-- A function that simulates asking a question about the distance to the nearest liar -/
def askQuestion (t : Table) (travelerPos : Position) : Fin 10 → ℝ :=
  sorry

/-- The main theorem stating that 2 questions are sufficient to identify all liars -/
theorem two_questions_sufficient (t : Table) :
  ∃ (pos1 pos2 : Position),
    (∀ (p : Fin 10), t.people p = Person.Liar ↔
      ∃ (q : Fin 10), t.people q = Person.Liar ∧
        askQuestion t pos1 p ≠ askQuestion t pos1 q ∨
        askQuestion t pos2 p ≠ askQuestion t pos2 q) :=
sorry

end NUMINAMATH_CALUDE_two_questions_sufficient_l1521_152140


namespace NUMINAMATH_CALUDE_inequality_statements_truth_l1521_152143

theorem inequality_statements_truth :
  let statement1 := ∀ (a b c d : ℝ), a > b ∧ c > d → a - c > b - d
  let statement2 := ∀ (a b c d : ℝ), a > b ∧ b > 0 ∧ c > d ∧ d > 0 → a * c > b * d
  let statement3 := ∀ (a b : ℝ), a > b ∧ b > 0 → 3 * a > 3 * b
  let statement4 := ∀ (a b : ℝ), a > b ∧ b > 0 → 1 / (a^2) < 1 / (b^2)
  (¬statement1 ∧ statement2 ∧ statement3 ∧ statement4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_statements_truth_l1521_152143


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l1521_152139

/-- A line that does not pass through the second quadrant -/
def LineNotInSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a - 2) * y = (3 * a - 1) * x - 1 → (x ≤ 0 → y ≤ 0)

/-- The main theorem: characterization of a for which the line doesn't pass through the second quadrant -/
theorem line_not_in_second_quadrant_iff (a : ℝ) :
  LineNotInSecondQuadrant a ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_iff_l1521_152139


namespace NUMINAMATH_CALUDE_marbles_per_customer_l1521_152185

theorem marbles_per_customer 
  (initial_marbles : ℕ) 
  (num_customers : ℕ) 
  (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 400) 
  (h2 : num_customers = 20) 
  (h3 : remaining_marbles = 100) :
  (initial_marbles - remaining_marbles) / num_customers = 15 :=
by sorry

end NUMINAMATH_CALUDE_marbles_per_customer_l1521_152185


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l1521_152141

theorem thirty_percent_less_than_ninety (x : ℝ) : x + (1/4) * x = 90 - 0.3 * 90 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_ninety_l1521_152141


namespace NUMINAMATH_CALUDE_inequality_contradiction_l1521_152138

theorem inequality_contradiction (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬(a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l1521_152138


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1521_152100

theorem cubic_root_sum (a b c : ℝ) : 
  (45 * a^3 - 75 * a^2 + 33 * a - 2 = 0) →
  (45 * b^3 - 75 * b^2 + 33 * b - 2 = 0) →
  (45 * c^3 - 75 * c^2 + 33 * c - 2 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (0 < a ∧ a < 1) →
  (0 < b ∧ b < 1) →
  (0 < c ∧ c < 1) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1521_152100


namespace NUMINAMATH_CALUDE_fraction_mediant_l1521_152108

theorem fraction_mediant (r s u v : ℚ) (l m : ℕ+) 
  (h1 : 0 < r) (h2 : 0 < s) (h3 : 0 < u) (h4 : 0 < v) 
  (h5 : s * u - r * v = 1) : 
  (∀ x, r / u < x ∧ x < s / v → 
    ∃ l m : ℕ+, x = (l * r + m * s) / (l * u + m * v)) ∧
  (r / u < (l * r + m * s) / (l * u + m * v) ∧ 
   (l * r + m * s) / (l * u + m * v) < s / v) :=
sorry

end NUMINAMATH_CALUDE_fraction_mediant_l1521_152108


namespace NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1521_152102

/-- The number of rectangles in a row of length n -/
def rectangles_in_row (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of rectangles in an m x n grid -/
def total_rectangles (m n : ℕ) : ℕ :=
  m * rectangles_in_row n + n * rectangles_in_row m - m * n

theorem rectangles_in_5x4_grid :
  total_rectangles 5 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x4_grid_l1521_152102


namespace NUMINAMATH_CALUDE_second_number_is_three_l1521_152160

theorem second_number_is_three (x y : ℝ) 
  (sum_is_ten : x + y = 10) 
  (relation : 2 * x = 3 * y + 5) : 
  y = 3 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_three_l1521_152160


namespace NUMINAMATH_CALUDE_jellybean_probability_l1521_152115

/-- The probability of selecting exactly one red and two blue jellybeans from a bowl -/
theorem jellybean_probability :
  let total_jellybeans : ℕ := 15
  let red_jellybeans : ℕ := 5
  let blue_jellybeans : ℕ := 3
  let white_jellybeans : ℕ := 7
  let picked_jellybeans : ℕ := 3

  -- Ensure the total number of jellybeans is correct
  total_jellybeans = red_jellybeans + blue_jellybeans + white_jellybeans →

  -- Calculate the probability
  (Nat.choose red_jellybeans 1 * Nat.choose blue_jellybeans 2 : ℚ) /
  Nat.choose total_jellybeans picked_jellybeans = 3 / 91 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_probability_l1521_152115


namespace NUMINAMATH_CALUDE_eleven_operations_to_equal_l1521_152120

/-- The number of operations required to make two numbers equal --/
def operations_to_equal (a b : ℕ) (sub_a add_b : ℕ) : ℕ :=
  (a - b) / (sub_a + add_b)

/-- Theorem stating that it takes 11 operations to make the numbers equal --/
theorem eleven_operations_to_equal :
  operations_to_equal 365 24 19 12 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_operations_to_equal_l1521_152120


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1521_152118

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ), ∀ (x : ℚ), x ≠ 3 → x ≠ 5 →
    (4 * x) / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2 ∧
    A = 5 ∧ B = -5 ∧ C = -6 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1521_152118


namespace NUMINAMATH_CALUDE_exists_valid_arrangement_l1521_152159

/-- Represents a circle placement arrangement in a square -/
structure CircleArrangement where
  n : ℕ  -- Side length of the square
  num_circles : ℕ  -- Number of circles placed

/-- Checks if a circle arrangement is valid -/
def is_valid_arrangement (arr : CircleArrangement) : Prop :=
  arr.n ≥ 8 ∧ arr.num_circles > arr.n^2

/-- Theorem stating the existence of a valid circle arrangement -/
theorem exists_valid_arrangement : 
  ∃ (arr : CircleArrangement), is_valid_arrangement arr :=
sorry

end NUMINAMATH_CALUDE_exists_valid_arrangement_l1521_152159


namespace NUMINAMATH_CALUDE_book_collection_ratio_l1521_152166

theorem book_collection_ratio (first_week : ℕ) (total : ℕ) : 
  first_week = 9 → total = 99 → 
  (total - first_week) / first_week = 10 := by
sorry

end NUMINAMATH_CALUDE_book_collection_ratio_l1521_152166


namespace NUMINAMATH_CALUDE_red_balls_drawn_is_random_variable_l1521_152117

/-- A bag containing black and red balls -/
structure Bag where
  black : ℕ
  red : ℕ

/-- The result of drawing balls from the bag -/
structure DrawResult where
  total : ℕ
  red : ℕ

/-- A random variable is a function that assigns a real number to each outcome of a random experiment -/
def RandomVariable (α : Type) := α → ℝ

/-- The bag containing 2 black balls and 6 red balls -/
def bag : Bag := { black := 2, red := 6 }

/-- The number of balls drawn -/
def numDrawn : ℕ := 2

/-- The function that counts the number of red balls drawn -/
def countRedBalls : DrawResult → ℕ := fun r => r.red

/-- Statement: The number of red balls drawn is a random variable -/
theorem red_balls_drawn_is_random_variable :
  ∃ (rv : RandomVariable DrawResult), ∀ (result : DrawResult),
    result.total = numDrawn ∧ result.red ≤ bag.red →
      rv result = (countRedBalls result : ℝ) :=
sorry

end NUMINAMATH_CALUDE_red_balls_drawn_is_random_variable_l1521_152117


namespace NUMINAMATH_CALUDE_acute_angle_in_first_quadrant_l1521_152105

-- Define what an acute angle is
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Define what it means for an angle to be in the first quadrant
def in_first_quadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem stating that an acute angle is in the first quadrant
theorem acute_angle_in_first_quadrant (θ : Real) : 
  is_acute_angle θ → in_first_quadrant θ := by
  sorry


end NUMINAMATH_CALUDE_acute_angle_in_first_quadrant_l1521_152105


namespace NUMINAMATH_CALUDE_jim_lamp_purchase_jim_lamp_purchase_correct_l1521_152123

theorem jim_lamp_purchase (lamp_cost : ℕ) (bulb_cost_difference : ℕ) (num_bulbs : ℕ) (total_paid : ℕ) : ℕ :=
  let bulb_cost := lamp_cost - bulb_cost_difference
  let num_lamps := (total_paid - num_bulbs * bulb_cost) / lamp_cost
  num_lamps

#check jim_lamp_purchase 7 4 6 32

theorem jim_lamp_purchase_correct :
  jim_lamp_purchase 7 4 6 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_lamp_purchase_jim_lamp_purchase_correct_l1521_152123


namespace NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l1521_152107

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleC : ℝ

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a^2 - t.c^2 + t.b^2 = t.a * t.b

-- Theorem for part 1
theorem angle_C_measure (t : Triangle) (h : triangle_condition t) : 
  t.angleC = π / 3 := by sorry

-- Theorem for part 2
theorem triangle_area (t : Triangle) (h1 : triangle_condition t) (h2 : t.a = 3) (h3 : t.b = 3) :
  (1/2) * t.a * t.b * Real.sin t.angleC = 9 * Real.sqrt 3 / 4 := by sorry

end NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l1521_152107


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l1521_152167

/-- Calculates the total cost of concert tickets --/
def concertTicketCost (generalAdmissionPrice : ℚ) (vipPrice : ℚ) (premiumPrice : ℚ)
                      (generalAdmissionQuantity : ℕ) (vipQuantity : ℕ) (premiumQuantity : ℕ)
                      (generalAdmissionDiscount : ℚ) (vipDiscount : ℚ) : ℚ :=
  let generalAdmissionCost := generalAdmissionPrice * generalAdmissionQuantity * (1 - generalAdmissionDiscount)
  let vipCost := vipPrice * vipQuantity * (1 - vipDiscount)
  let premiumCost := premiumPrice * premiumQuantity
  generalAdmissionCost + vipCost + premiumCost

theorem concert_ticket_cost :
  concertTicketCost 6 10 15 6 2 1 (1/10) (3/20) = 644/10 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l1521_152167


namespace NUMINAMATH_CALUDE_log_sum_equality_l1521_152132

theorem log_sum_equality : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 * Real.log 2 / Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1521_152132


namespace NUMINAMATH_CALUDE_daniels_turtles_l1521_152147

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The total number of legs of all animals -/
def total_legs : ℕ := 72

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of dogs Daniel has -/
def num_dogs : ℕ := 5

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The number of legs each animal has -/
def legs_per_animal : ℕ := 4

theorem daniels_turtles :
  num_turtles * legs_per_animal + 
  num_horses * legs_per_animal + 
  num_dogs * legs_per_animal + 
  num_cats * legs_per_animal + 
  num_goats * legs_per_animal = total_legs :=
by sorry

end NUMINAMATH_CALUDE_daniels_turtles_l1521_152147


namespace NUMINAMATH_CALUDE_tuna_sales_problem_l1521_152183

/-- The number of packs of tuna fish sold per hour during the peak season -/
def peak_packs_per_hour : ℕ := 6

/-- The price of each tuna pack in dollars -/
def price_per_pack : ℕ := 60

/-- The number of hours fish are sold per day -/
def hours_per_day : ℕ := 15

/-- The additional revenue made during the high season compared to the low season, in dollars -/
def additional_revenue : ℕ := 1800

/-- The number of packs of tuna fish sold per hour during the low season -/
def low_season_packs : ℕ := 4

theorem tuna_sales_problem :
  peak_packs_per_hour * price_per_pack * hours_per_day =
  low_season_packs * price_per_pack * hours_per_day + additional_revenue := by
  sorry

end NUMINAMATH_CALUDE_tuna_sales_problem_l1521_152183


namespace NUMINAMATH_CALUDE_bus_speed_problem_l1521_152184

/-- The speed of Bus A in miles per hour -/
def speed_A : ℝ := 45

/-- The speed of Bus B in miles per hour -/
def speed_B : ℝ := 30

/-- The initial distance between Bus A and Bus B in miles -/
def initial_distance : ℝ := 150

/-- The time it takes for Bus A to overtake Bus B when both are driving west, in hours -/
def overtake_time : ℝ := 10

/-- The time it would take for the buses to meet if they drove towards each other, in hours -/
def meet_time : ℝ := 2

theorem bus_speed_problem :
  (speed_A - speed_B) * overtake_time = initial_distance ∧
  (speed_A + speed_B) * meet_time = initial_distance ∧
  speed_A = 45 := by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l1521_152184


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l1521_152106

theorem smallest_x_for_equation : 
  ∀ x : ℝ, x > 0 → (⌊x^2⌋ : ℤ) - x * (⌊x⌋ : ℤ) = 10 → x ≥ 131/11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l1521_152106


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1521_152169

theorem imaginary_part_of_z (z : ℂ) (m : ℝ) (h1 : z = 1 - m * I) (h2 : z^2 = -2 * I) :
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1521_152169


namespace NUMINAMATH_CALUDE_biased_coin_expected_value_l1521_152113

/-- The expected value of winnings for a biased coin flip -/
theorem biased_coin_expected_value :
  let p_head : ℚ := 1/4  -- Probability of getting a head
  let p_tail : ℚ := 3/4  -- Probability of getting a tail
  let win_head : ℚ := 4  -- Amount won for flipping a head
  let lose_tail : ℚ := 3 -- Amount lost for flipping a tail
  p_head * win_head - p_tail * lose_tail = -5/4 := by
sorry

end NUMINAMATH_CALUDE_biased_coin_expected_value_l1521_152113


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_sum_l1521_152163

theorem equilateral_triangle_area_sum : 
  let triangle1_side : ℝ := 2
  let triangle2_side : ℝ := 3
  let new_triangle_side : ℝ := Real.sqrt 13
  let area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2
  area new_triangle_side = area triangle1_side + area triangle2_side :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_sum_l1521_152163


namespace NUMINAMATH_CALUDE_quadratic_integer_values_iff_coefficients_integer_l1521_152189

theorem quadratic_integer_values_iff_coefficients_integer (a b c : ℚ) :
  (∀ x : ℤ, ∃ n : ℤ, a * x^2 + b * x + c = n) ↔
  (∃ k : ℤ, 2 * a = k) ∧ (∃ m : ℤ, a + b = m) ∧ (∃ p : ℤ, c = p) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_integer_values_iff_coefficients_integer_l1521_152189


namespace NUMINAMATH_CALUDE_min_value_theorem_l1521_152178

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (2 : ℝ) / (3 * a + b) + 1 / (a + 2 * b) = 4) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 : ℝ) / (3 * x + y) + 1 / (x + 2 * y) = 4 → 7 * a + 4 * b ≤ 7 * x + 4 * y) ∧
  (7 * a + 4 * b = 9 / 4) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1521_152178


namespace NUMINAMATH_CALUDE_train_speed_problem_l1521_152146

theorem train_speed_problem (length1 length2 : Real) (crossing_time : Real) (speed1 : Real) (speed2 : Real) :
  length1 = 150 ∧ 
  length2 = 160 ∧ 
  crossing_time = 11.159107271418288 ∧
  speed1 = 60 ∧
  (length1 + length2) / crossing_time = (speed1 * 1000 / 3600) + (speed2 * 1000 / 3600) →
  speed2 = 40 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1521_152146


namespace NUMINAMATH_CALUDE_apple_distribution_proof_l1521_152162

def total_apples : ℕ := 30
def num_people : ℕ := 3
def min_apples : ℕ := 3

def distribution_ways : ℕ := Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)

theorem apple_distribution_proof : distribution_ways = 253 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_proof_l1521_152162


namespace NUMINAMATH_CALUDE_workshop_average_age_l1521_152192

theorem workshop_average_age 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (num_elderly : ℕ) (avg_age_elderly : ℝ)
  (h1 : num_females = 8)
  (h2 : avg_age_females = 34)
  (h3 : num_males = 12)
  (h4 : avg_age_males = 32)
  (h5 : num_elderly = 5)
  (h6 : avg_age_elderly = 60) :
  let total_people := num_females + num_males + num_elderly
  let total_age := num_females * avg_age_females + num_males * avg_age_males + num_elderly * avg_age_elderly
  total_age / total_people = 38.24 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_age_l1521_152192


namespace NUMINAMATH_CALUDE_complex_power_eight_l1521_152109

theorem complex_power_eight :
  (3 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6))) ^ 8 =
  Complex.mk (-3280.5) (-3280.5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l1521_152109


namespace NUMINAMATH_CALUDE_infinitely_many_multiples_of_100_l1521_152116

theorem infinitely_many_multiples_of_100 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 100 ∣ (2^n + n^2) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_multiples_of_100_l1521_152116


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_neg_one_l1521_152157

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x

-- Theorem statement
theorem tangent_line_at_point_one_neg_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -x := by
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_neg_one_l1521_152157


namespace NUMINAMATH_CALUDE_roberto_outfits_l1521_152111

/-- The number of trousers Roberto has -/
def num_trousers : ℕ := 5

/-- The number of shirts Roberto has -/
def num_shirts : ℕ := 8

/-- The number of jackets Roberto has -/
def num_jackets : ℕ := 2

/-- An outfit consists of a pair of trousers, a shirt, and a jacket -/
def outfit := ℕ × ℕ × ℕ

/-- The total number of possible outfits -/
def total_outfits : ℕ := num_trousers * num_shirts * num_jackets

theorem roberto_outfits : total_outfits = 80 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1521_152111


namespace NUMINAMATH_CALUDE_ab_value_l1521_152180

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a^4 + b^4 = 31/16) : 
  a * b = Real.sqrt (33/32) := by
sorry

end NUMINAMATH_CALUDE_ab_value_l1521_152180


namespace NUMINAMATH_CALUDE_right_triangle_area_15_degree_l1521_152110

theorem right_triangle_area_15_degree (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_acute : Real.cos (15 * π / 180) = b / c) : a * b / 2 = c^2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_15_degree_l1521_152110


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1521_152151

theorem quadratic_transformation (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 2 * (x - 4)^2 + 8) →
  ∃ n k, ∀ x, 3 * a * x^2 + 3 * b * x + 3 * c = n * (x - 4)^2 + k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1521_152151


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l1521_152196

theorem smallest_sum_of_squares : ∃ (x y : ℕ), 
  x^2 - y^2 = 175 ∧ 
  x^2 ≥ 36 ∧ 
  y^2 ≥ 36 ∧ 
  x^2 + y^2 = 625 ∧ 
  (∀ (a b : ℕ), a^2 - b^2 = 175 → a^2 ≥ 36 → b^2 ≥ 36 → a^2 + b^2 ≥ 625) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l1521_152196


namespace NUMINAMATH_CALUDE_u_equivalence_l1521_152176

theorem u_equivalence (u : ℝ) : 
  u = 1 / (2 - Real.rpow 3 (1/3)) → 
  u = ((2 + Real.rpow 3 (1/3)) * (4 + Real.rpow 9 (1/3))) / 7 := by
sorry

end NUMINAMATH_CALUDE_u_equivalence_l1521_152176


namespace NUMINAMATH_CALUDE_max_three_digit_sum_l1521_152197

theorem max_three_digit_sum (A B C : Nat) : 
  A ≠ B ∧ B ≠ C ∧ C ≠ A → 
  A < 10 ∧ B < 10 ∧ C < 10 →
  110 * A + 10 * B + 3 * C ≤ 981 ∧ 
  (∃ A' B' C', A' ≠ B' ∧ B' ≠ C' ∧ C' ≠ A' ∧ 
               A' < 10 ∧ B' < 10 ∧ C' < 10 ∧ 
               110 * A' + 10 * B' + 3 * C' = 981) :=
by sorry

end NUMINAMATH_CALUDE_max_three_digit_sum_l1521_152197


namespace NUMINAMATH_CALUDE_min_tablets_extracted_l1521_152155

/-- The least number of tablets to extract to ensure at least two of each kind -/
def leastTablets (tabletA : ℕ) (tabletB : ℕ) : ℕ :=
  max (tabletB + 2) (tabletA + 2)

theorem min_tablets_extracted (tabletA tabletB : ℕ) 
  (hA : tabletA = 10) (hB : tabletB = 16) :
  leastTablets tabletA tabletB = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_tablets_extracted_l1521_152155


namespace NUMINAMATH_CALUDE_baseball_cards_problem_l1521_152121

theorem baseball_cards_problem (X : ℚ) : 3 * (X - (X + 1) / 2 - 1) = 18 ↔ X = 15 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_problem_l1521_152121


namespace NUMINAMATH_CALUDE_other_number_proof_l1521_152104

/-- Given two positive integers with specified HCF and LCM, prove that if one number is 36, the other is 176. -/
theorem other_number_proof (A B : ℕ+) : 
  Nat.gcd A B = 16 →
  Nat.lcm A B = 396 →
  A = 36 →
  B = 176 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1521_152104


namespace NUMINAMATH_CALUDE_students_playing_football_l1521_152173

theorem students_playing_football 
  (total : ℕ) 
  (cricket : ℕ) 
  (neither : ℕ) 
  (both : ℕ) 
  (h1 : total = 470) 
  (h2 : cricket = 175) 
  (h3 : neither = 50) 
  (h4 : both = 80) : 
  total - neither - cricket + both = 325 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_football_l1521_152173


namespace NUMINAMATH_CALUDE_milk_for_pizza_dough_l1521_152144

/-- Given a ratio of 50 mL of milk for every 250 mL of flour, 
    calculate the amount of milk needed for 1200 mL of flour. -/
theorem milk_for_pizza_dough (flour : ℝ) (milk : ℝ) : 
  flour = 1200 → 
  (milk / flour = 50 / 250) → 
  milk = 240 := by sorry

end NUMINAMATH_CALUDE_milk_for_pizza_dough_l1521_152144


namespace NUMINAMATH_CALUDE_parametric_equations_form_circle_parametric_equations_part_of_circle_l1521_152136

noncomputable def parametricCircle (θ : Real) : Real × Real :=
  (4 - Real.cos θ, 1 - Real.sin θ)

theorem parametric_equations_form_circle (θ : Real) 
  (h : 0 ≤ θ ∧ θ ≤ Real.pi / 2) : 
  let (x, y) := parametricCircle θ
  (x - 4)^2 + (y - 1)^2 = 1 := by
sorry

theorem parametric_equations_part_of_circle :
  ∃ (a b r : Real), 
    (∀ θ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
      let (x, y) := parametricCircle θ
      (x - a)^2 + (y - b)^2 = r^2) ∧
    (∃ θ₁ θ₂, 0 ≤ θ₁ ∧ θ₁ < θ₂ ∧ θ₂ ≤ Real.pi / 2 ∧ 
      parametricCircle θ₁ ≠ parametricCircle θ₂) := by
sorry

end NUMINAMATH_CALUDE_parametric_equations_form_circle_parametric_equations_part_of_circle_l1521_152136


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1521_152152

/-- An arithmetic sequence and its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ  -- The sequence
  d : ℚ       -- Common difference
  sum : ℕ+ → ℚ -- Sum function
  sum_def : ∀ n : ℕ+, sum n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The sequence b_n defined as S_n / n -/
def b (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  seq.sum n / n

/-- Main theorem about the properties of sequence b_n -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
    (h1 : seq.sum 7 = 7)
    (h2 : seq.sum 15 = 75) :
  (∀ n m : ℕ+, b seq (n + m) - b seq n = b seq (m + 1) - b seq 1) ∧
  (∀ n : ℕ+, b seq n = (n - 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1521_152152


namespace NUMINAMATH_CALUDE_lateral_surface_area_rotated_unit_square_l1521_152186

/-- The lateral surface area of a cylinder formed by rotating a square with area 1 around one of its sides. -/
theorem lateral_surface_area_rotated_unit_square : 
  ∀ (square_area : ℝ) (cylinder_height : ℝ) (cylinder_base_circumference : ℝ),
    square_area = 1 →
    cylinder_height = Real.sqrt square_area →
    cylinder_base_circumference = Real.sqrt square_area →
    cylinder_height * cylinder_base_circumference = 1 := by
  sorry

#check lateral_surface_area_rotated_unit_square

end NUMINAMATH_CALUDE_lateral_surface_area_rotated_unit_square_l1521_152186
