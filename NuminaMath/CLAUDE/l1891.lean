import Mathlib

namespace NUMINAMATH_CALUDE_min_value_M_l1891_189161

theorem min_value_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let M := (((a / (b + c)) ^ (1/4) : ℝ) + ((b / (c + a)) ^ (1/4) : ℝ) + ((c / (b + a)) ^ (1/4) : ℝ) +
            ((b + c) / a) ^ (1/2) + ((a + c) / b) ^ (1/2) + ((a + b) / c) ^ (1/2))
  M ≥ 3 * Real.sqrt 2 + (3 / 2) * (8 ^ (1/4) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_value_M_l1891_189161


namespace NUMINAMATH_CALUDE_reciprocal_roots_l1891_189142

theorem reciprocal_roots (p q : ℝ) (r₁ r₂ : ℂ) : 
  (r₁^2 + p*r₁ + q = 0 ∧ r₂^2 + p*r₂ + q = 0) → 
  ((1/r₁)^2 * q + (1/r₁) * p + 1 = 0 ∧ (1/r₂)^2 * q + (1/r₂) * p + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_reciprocal_roots_l1891_189142


namespace NUMINAMATH_CALUDE_import_tax_problem_l1891_189137

/-- The import tax problem -/
theorem import_tax_problem (tax_rate : ℝ) (tax_paid : ℝ) (total_value : ℝ) 
  (h1 : tax_rate = 0.07)
  (h2 : tax_paid = 112.70)
  (h3 : total_value = 2610) :
  ∃ (excess : ℝ), excess = 1000 ∧ tax_rate * (total_value - excess) = tax_paid :=
by sorry

end NUMINAMATH_CALUDE_import_tax_problem_l1891_189137


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1891_189116

-- Define arithmetic sequence
def is_arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

-- Define geometric sequence
def is_geometric_sequence (a b c d e : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem arithmetic_geometric_sequence_ratio :
  ∀ (x y a b c : ℝ),
  is_arithmetic_sequence 1 x y 4 →
  is_geometric_sequence (-2) a b c (-8) →
  (y - x) / b = -1/4 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1891_189116


namespace NUMINAMATH_CALUDE_excluded_angle_measure_l1891_189158

/-- Given a polygon where the sum of all but one interior angle is 1680°,
    prove that the measure of the excluded interior angle is 120°. -/
theorem excluded_angle_measure (n : ℕ) (h : n ≥ 3) :
  let sum_interior := (n - 2) * 180
  let sum_except_one := 1680
  sum_interior - sum_except_one = 120 := by
  sorry

end NUMINAMATH_CALUDE_excluded_angle_measure_l1891_189158


namespace NUMINAMATH_CALUDE_simplify_expression_l1891_189124

theorem simplify_expression (x y : ℝ) : 7*x + 9 - 2*x + 3*y = 5*x + 3*y + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1891_189124


namespace NUMINAMATH_CALUDE_correct_bonus_distribution_l1891_189170

/-- Represents the bonus distribution problem for a corporation --/
def BonusDistribution (total : ℕ) (a b c d e f : ℕ) : Prop :=
  -- Total bonus is $25,000
  total = 25000 ∧
  -- A receives twice the amount of B
  a = 2 * b ∧
  -- B and C receive the same amount
  b = c ∧
  -- D receives $1,500 less than A
  d = a - 1500 ∧
  -- E receives $2,000 more than C
  e = c + 2000 ∧
  -- F receives half of the total amount received by A and D combined
  f = (a + d) / 2 ∧
  -- The sum of all amounts equals the total bonus
  a + b + c + d + e + f = total

/-- Theorem stating the correct distribution of the bonus --/
theorem correct_bonus_distribution :
  BonusDistribution 25000 4950 2475 2475 3450 4475 4200 := by
  sorry

#check correct_bonus_distribution

end NUMINAMATH_CALUDE_correct_bonus_distribution_l1891_189170


namespace NUMINAMATH_CALUDE_system_solution_system_solution_zero_system_solution_one_l1891_189153

theorem system_solution :
  ∀ x y z : ℝ,
  (2 * y + x - x^2 - y^2 = 0 ∧
   z - x + y - y * (x + z) = 0 ∧
   -2 * y + z - y^2 - z^2 = 0) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

-- Alternatively, we can split it into two theorems for each solution

theorem system_solution_zero :
  2 * 0 + 0 - 0^2 - 0^2 = 0 ∧
  0 - 0 + 0 - 0 * (0 + 0) = 0 ∧
  -2 * 0 + 0 - 0^2 - 0^2 = 0 :=
by
  sorry

theorem system_solution_one :
  2 * 0 + 1 - 1^2 - 0^2 = 0 ∧
  1 - 1 + 0 - 0 * (1 + 1) = 0 ∧
  -2 * 0 + 1 - 0^2 - 1^2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_system_solution_zero_system_solution_one_l1891_189153


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l1891_189192

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  common_difference : Nat
  start : Nat

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, k < s.sample_size ∧ n = (s.start + k * s.common_difference) % s.total_students

/-- The main theorem to prove -/
theorem systematic_sample_theorem :
  ∃ (s : SystematicSample),
    s.total_students = 52 ∧
    s.sample_size = 4 ∧
    in_sample s 6 ∧
    in_sample s 32 ∧
    in_sample s 45 ∧
    in_sample s 19 :=
  sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l1891_189192


namespace NUMINAMATH_CALUDE_polynomial_divisibility_implies_coefficients_l1891_189160

theorem polynomial_divisibility_implies_coefficients
  (p q : ℤ)
  (h : ∀ x : ℝ, (x + 2) * (x - 1) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 4)) :
  p = -7 ∧ q = -12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_implies_coefficients_l1891_189160


namespace NUMINAMATH_CALUDE_limits_involving_x_and_n_l1891_189148

open Real

/-- For x > 0, prove the limits of two expressions involving n and x as n approaches infinity. -/
theorem limits_involving_x_and_n (x : ℝ) (h : x > 0) :
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |n * log (1 + x / n) - x| < ε) ∧
  (∀ ε > 0, ∃ N, ∀ n ≥ N, |(1 + x / n)^n - exp x| < ε) :=
sorry

end NUMINAMATH_CALUDE_limits_involving_x_and_n_l1891_189148


namespace NUMINAMATH_CALUDE_car_travel_time_l1891_189109

theorem car_travel_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) (t : ℝ) 
  (h1 : distance = 630)
  (h2 : new_speed = 70)
  (h3 : time_ratio = 3/2)
  (h4 : distance = (distance / t) * (time_ratio * t))
  (h5 : distance = new_speed * (time_ratio * t)) :
  t = 6 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l1891_189109


namespace NUMINAMATH_CALUDE_sum_even_divisors_140_l1891_189108

/-- Sum of even positive divisors of a natural number n -/
def sumEvenDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all even positive divisors of 140 is 288 -/
theorem sum_even_divisors_140 : sumEvenDivisors 140 = 288 := by sorry

end NUMINAMATH_CALUDE_sum_even_divisors_140_l1891_189108


namespace NUMINAMATH_CALUDE_cubic_value_given_quadratic_l1891_189126

theorem cubic_value_given_quadratic (x : ℝ) : 
  x^2 - 2*x - 1 = 0 → 3*x^3 - 10*x^2 + 5*x + 2027 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_cubic_value_given_quadratic_l1891_189126


namespace NUMINAMATH_CALUDE_simplify_expression_l1891_189154

theorem simplify_expression : 
  ((5^1010)^2 - (5^1008)^2) / ((5^1009)^2 - (5^1007)^2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1891_189154


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l1891_189121

theorem cubic_equation_solutions :
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3 + 3 * Complex.I * Real.sqrt 3) / 2
  let z₃ : ℂ := (3 - 3 * Complex.I * Real.sqrt 3) / 2
  (z₁^3 = -27 ∧ z₂^3 = -27 ∧ z₃^3 = -27) ∧
  ∀ z : ℂ, z^3 = -27 → (z = z₁ ∨ z = z₂ ∨ z = z₃) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l1891_189121


namespace NUMINAMATH_CALUDE_student_average_less_than_actual_average_l1891_189185

theorem student_average_less_than_actual_average (a b c : ℝ) (h : a < b ∧ b < c) :
  (a + (b + c) / 2) / 2 < (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_less_than_actual_average_l1891_189185


namespace NUMINAMATH_CALUDE_campers_hiking_morning_l1891_189112

theorem campers_hiking_morning (morning_rowers afternoon_rowers total_rowers : ℕ)
  (h1 : morning_rowers = 13)
  (h2 : afternoon_rowers = 21)
  (h3 : total_rowers = 34)
  (h4 : morning_rowers + afternoon_rowers = total_rowers) :
  total_rowers - (morning_rowers + afternoon_rowers) = 0 :=
by sorry

end NUMINAMATH_CALUDE_campers_hiking_morning_l1891_189112


namespace NUMINAMATH_CALUDE_ratio_a_to_c_l1891_189127

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 9) :
  a / c = 135 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_a_to_c_l1891_189127


namespace NUMINAMATH_CALUDE_max_tuesday_13ths_l1891_189136

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the months of the year -/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Returns the number of days in a given month -/
def daysInMonth (m : Month) : Nat :=
  match m with
  | .February => 28
  | .April | .June | .September | .November => 30
  | _ => 31

/-- Returns the day of the week for the 13th of a given month, 
    given the day of the week for January 13th -/
def dayOf13th (m : Month) (jan13 : DayOfWeek) : DayOfWeek :=
  sorry

/-- Counts the number of times the 13th falls on a Tuesday in a year -/
def countTuesday13ths (jan13 : DayOfWeek) : Nat :=
  sorry

theorem max_tuesday_13ths :
  ∃ (jan13 : DayOfWeek), countTuesday13ths jan13 = 3 ∧
  ∀ (d : DayOfWeek), countTuesday13ths d ≤ 3 :=
  sorry

end NUMINAMATH_CALUDE_max_tuesday_13ths_l1891_189136


namespace NUMINAMATH_CALUDE_remainder_theorem_l1891_189175

theorem remainder_theorem (n : ℤ) : (5 * n^2 + 7) - (3 * n - 2) ≡ 2 * n + 4 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1891_189175


namespace NUMINAMATH_CALUDE_even_sum_probability_l1891_189176

theorem even_sum_probability (p_even_1 p_odd_1 p_even_2 p_odd_2 : ℝ) :
  p_even_1 = 1/2 →
  p_odd_1 = 1/2 →
  p_even_2 = 1/5 →
  p_odd_2 = 4/5 →
  p_even_1 + p_odd_1 = 1 →
  p_even_2 + p_odd_2 = 1 →
  p_even_1 * p_even_2 + p_odd_1 * p_odd_2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_even_sum_probability_l1891_189176


namespace NUMINAMATH_CALUDE_birthday_age_proof_l1891_189151

theorem birthday_age_proof (A : ℤ) : A = 4 * (A - 10) - 5 ↔ A = 15 := by
  sorry

end NUMINAMATH_CALUDE_birthday_age_proof_l1891_189151


namespace NUMINAMATH_CALUDE_football_team_analysis_l1891_189119

structure FootballTeam where
  total_matches : Nat
  played_matches : Nat
  lost_matches : Nat
  points : Nat

def win_points : Nat := 3
def draw_points : Nat := 1
def loss_points : Nat := 0

def team : FootballTeam := {
  total_matches := 14,
  played_matches := 8,
  lost_matches := 1,
  points := 17
}

def wins_in_first_8 (t : FootballTeam) : Nat :=
  (t.points - (t.played_matches - t.lost_matches - 1)) / 2

def max_possible_points (t : FootballTeam) : Nat :=
  t.points + (t.total_matches - t.played_matches) * win_points

def min_wins_needed (t : FootballTeam) (target : Nat) : Nat :=
  ((target - t.points + 2) / win_points).min (t.total_matches - t.played_matches)

theorem football_team_analysis (t : FootballTeam) :
  wins_in_first_8 t = 5 ∧
  max_possible_points t = 35 ∧
  min_wins_needed t 29 = 3 := by
  sorry

end NUMINAMATH_CALUDE_football_team_analysis_l1891_189119


namespace NUMINAMATH_CALUDE_exists_prime_with_greater_remainder_l1891_189149

theorem exists_prime_with_greater_remainder
  (a b : ℕ+) (h : a < b) :
  ∃ p : ℕ, Nat.Prime p ∧ a % p > b % p :=
sorry

end NUMINAMATH_CALUDE_exists_prime_with_greater_remainder_l1891_189149


namespace NUMINAMATH_CALUDE_regression_line_mean_y_l1891_189101

theorem regression_line_mean_y (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ = 1) (h₂ : x₂ = 5) (h₃ : x₃ = 7) (h₄ : x₄ = 13) (h₅ : x₅ = 19)
  (regression_eq : ℝ → ℝ) (h_reg : ∀ x, regression_eq x = 1.5 * x + 45) : 
  let x_mean := (x₁ + x₂ + x₃ + x₄ + x₅) / 5
  regression_eq x_mean = 58.5 := by
sorry

end NUMINAMATH_CALUDE_regression_line_mean_y_l1891_189101


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1891_189163

theorem linear_equation_solution (m : ℝ) :
  (∃ k : ℝ, ∀ x, 3 * x^(m-1) + 2 = k * x + (-3)) →
  (∀ y, 3 * m * y + 2 * y = 3 + m ↔ y = 5/8) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1891_189163


namespace NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l1891_189122

theorem alcohol_quantity_in_mixture (initial_alcohol : ℝ) (initial_water : ℝ) :
  initial_alcohol / initial_water = 4 / 3 →
  initial_alcohol / (initial_water + 8) = 4 / 5 →
  initial_alcohol = 16 := by
sorry

end NUMINAMATH_CALUDE_alcohol_quantity_in_mixture_l1891_189122


namespace NUMINAMATH_CALUDE_derivative_at_two_l1891_189117

-- Define f as a real-valued function
variable (f : ℝ → ℝ)

-- Define the conditions
def tangent_coincide (f : ℝ → ℝ) : Prop :=
  ∃ (m : ℝ), (∀ x, x ≠ 0 → (f x) / x - 1 = m * (x - 2)) ∧
             (∀ x, f x = m * x)

-- Theorem statement
theorem derivative_at_two (f : ℝ → ℝ) 
  (h1 : tangent_coincide f) 
  (h2 : f 0 = 0) :
  deriv f 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l1891_189117


namespace NUMINAMATH_CALUDE_cans_collection_problem_l1891_189171

theorem cans_collection_problem (solomon_cans juwan_cans levi_cans : ℕ) : 
  solomon_cans = 66 →
  solomon_cans = 3 * juwan_cans →
  levi_cans = juwan_cans / 2 →
  solomon_cans + juwan_cans + levi_cans = 99 :=
by sorry

end NUMINAMATH_CALUDE_cans_collection_problem_l1891_189171


namespace NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l1891_189169

/-- An isosceles triangle with equal sides of length x and base of length y,
    where a median on one of the equal sides divides the perimeter into parts of 6 and 12 -/
structure IsoscelesTriangle where
  x : ℝ  -- Length of equal sides
  y : ℝ  -- Length of base
  perimeter_division : x + x/2 = 6 ∧ y/2 + y = 12

theorem isosceles_triangle_dimensions (t : IsoscelesTriangle) : t.x = 8 ∧ t.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_dimensions_l1891_189169


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_1968_l1891_189199

theorem sqrt_sum_eq_sqrt_1968 : 
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (x y : ℕ), (x, y) ∈ s ↔ Real.sqrt x + Real.sqrt y = Real.sqrt 1968) ∧ 
    s.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_sqrt_1968_l1891_189199


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisectors_l1891_189178

/-- Given a right triangle with legs 3 and 4, prove the lengths of its angle bisectors. -/
theorem right_triangle_angle_bisectors :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := (a^2 + b^2).sqrt
  let d : ℝ := (a * b * ((a + b + c) / (a + b - c))).sqrt
  let l_b : ℝ := (b * a * (1 - ((b - a)^2 / b^2))).sqrt
  let l_a : ℝ := (a * b * (1 - ((a - b)^2 / a^2))).sqrt
  (d = 12 * Real.sqrt 2 / 7) ∧
  (l_b = 3 * Real.sqrt 5 / 2) ∧
  (l_a = 4 * Real.sqrt 10 / 3) := by
sorry


end NUMINAMATH_CALUDE_right_triangle_angle_bisectors_l1891_189178


namespace NUMINAMATH_CALUDE_acute_triangle_sine_sum_l1891_189106

theorem acute_triangle_sine_sum (α β γ : Real) 
  (acute_triangle : 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi)
  (acute_angles : α < Real.pi/2 ∧ β < Real.pi/2 ∧ γ < Real.pi/2) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_sum_l1891_189106


namespace NUMINAMATH_CALUDE_intersection_M_N_l1891_189197

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1891_189197


namespace NUMINAMATH_CALUDE_unique_solution_l1891_189190

theorem unique_solution (x y : ℝ) : 
  x * (x + y)^2 = 9 ∧ x * (y^3 - x^3) = 7 → x = 1 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1891_189190


namespace NUMINAMATH_CALUDE_daily_increase_amount_l1891_189128

def fine_sequence (x : ℚ) : ℕ → ℚ
  | 0 => 0.05
  | n + 1 => min (fine_sequence x n + x) (2 * fine_sequence x n)

theorem daily_increase_amount :
  ∃ x : ℚ, x > 0 ∧ fine_sequence x 4 = 0.70 ∧ 
  ∀ n : ℕ, n > 0 → fine_sequence x n = fine_sequence x (n-1) + x :=
by sorry

end NUMINAMATH_CALUDE_daily_increase_amount_l1891_189128


namespace NUMINAMATH_CALUDE_donut_problem_l1891_189107

theorem donut_problem (initial_donuts : ℕ) (h1 : initial_donuts = 50) : 
  let after_bill_eats := initial_donuts - 2
  let after_secretary_takes := after_bill_eats - 4
  let stolen_by_coworkers := after_secretary_takes / 2
  initial_donuts - 2 - 4 - stolen_by_coworkers = 22 := by
  sorry

end NUMINAMATH_CALUDE_donut_problem_l1891_189107


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l1891_189141

theorem final_sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l1891_189141


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1891_189159

theorem smallest_common_multiple_of_6_and_15 (a : ℕ) :
  (a > 0 ∧ 6 ∣ a ∧ 15 ∣ a) → a ≥ 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_6_and_15_l1891_189159


namespace NUMINAMATH_CALUDE_smallest_N_value_l1891_189194

/-- Represents a point in the rectangular array. -/
structure Point where
  row : Fin 5
  col : ℕ

/-- The first numbering system (left to right, top to bottom). -/
def firstNumber (N : ℕ) (p : Point) : ℕ :=
  N * p.row.val + p.col

/-- The second numbering system (top to bottom, left to right). -/
def secondNumber (p : Point) : ℕ :=
  5 * (p.col - 1) + p.row.val + 1

/-- The main theorem stating the smallest possible value of N. -/
theorem smallest_N_value :
  ∃ (N : ℕ) (p₁ p₂ p₃ p₄ p₅ : Point),
    p₁.row = 0 ∧ p₂.row = 1 ∧ p₃.row = 2 ∧ p₄.row = 3 ∧ p₅.row = 4 ∧
    (∀ p : Point, p.col < N) ∧
    firstNumber N p₁ = secondNumber p₂ ∧
    firstNumber N p₂ = secondNumber p₁ ∧
    firstNumber N p₃ = secondNumber p₄ ∧
    firstNumber N p₄ = secondNumber p₅ ∧
    firstNumber N p₅ = secondNumber p₃ ∧
    (∀ N' : ℕ, N' < N →
      ¬∃ (q₁ q₂ q₃ q₄ q₅ : Point),
        q₁.row = 0 ∧ q₂.row = 1 ∧ q₃.row = 2 ∧ q₄.row = 3 ∧ q₅.row = 4 ∧
        (∀ q : Point, q.col < N') ∧
        firstNumber N' q₁ = secondNumber q₂ ∧
        firstNumber N' q₂ = secondNumber q₁ ∧
        firstNumber N' q₃ = secondNumber q₄ ∧
        firstNumber N' q₄ = secondNumber q₅ ∧
        firstNumber N' q₅ = secondNumber q₃) ∧
    N = 149 := by
  sorry

end NUMINAMATH_CALUDE_smallest_N_value_l1891_189194


namespace NUMINAMATH_CALUDE_largest_integer_for_binary_op_l1891_189120

def binary_op (n : ℤ) : ℤ := n - (n * 5)

theorem largest_integer_for_binary_op :
  ∃ m : ℤ, m = -19 ∧
  (∀ n : ℤ, n > 0 → binary_op n < m → n ≤ 5) ∧
  (∀ m' : ℤ, m' > m → ∃ n : ℤ, n > 0 ∧ n > 5 ∧ binary_op n < m') :=
sorry

end NUMINAMATH_CALUDE_largest_integer_for_binary_op_l1891_189120


namespace NUMINAMATH_CALUDE_quadratic_increasing_implies_a_range_l1891_189152

/-- A function f is increasing on an interval [a, +∞) if for all x, y in [a, +∞) with x < y, f(x) < f(y) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y → f x < f y

/-- The quadratic function f(x) = x^2 + (a-1)x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-1)*x + a

theorem quadratic_increasing_implies_a_range (a : ℝ) :
  IncreasingOnInterval (f a) 2 → a ∈ Set.Ici (-3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_implies_a_range_l1891_189152


namespace NUMINAMATH_CALUDE_power_of_two_l1891_189191

theorem power_of_two : (1 : ℕ) * 2^6 = 64 := by sorry

end NUMINAMATH_CALUDE_power_of_two_l1891_189191


namespace NUMINAMATH_CALUDE_consecutive_even_negative_integers_sum_l1891_189146

theorem consecutive_even_negative_integers_sum (n m : ℤ) : 
  n < 0 ∧ m < 0 ∧ 
  Even n ∧ Even m ∧ 
  m = n + 2 ∧ 
  n * m = 2496 → 
  n + m = -102 := by sorry

end NUMINAMATH_CALUDE_consecutive_even_negative_integers_sum_l1891_189146


namespace NUMINAMATH_CALUDE_dakota_medical_bill_l1891_189145

/-- Calculates the total medical bill for Dakota's hospital stay -/
def total_medical_bill (
  days : ℕ)
  (bed_charge_per_day : ℕ)
  (specialist_fee_per_hour : ℕ)
  (specialist_time_minutes : ℕ)
  (ambulance_ride_cost : ℕ)
  (iv_cost : ℕ)
  (surgery_duration_hours : ℕ)
  (surgeon_fee_per_hour : ℕ)
  (assistant_fee_per_hour : ℕ)
  (physical_therapy_fee_per_hour : ℕ)
  (physical_therapy_duration_hours : ℕ)
  (medication_a_times_per_day : ℕ)
  (medication_a_cost_per_pill : ℕ)
  (medication_b_duration_hours : ℕ)
  (medication_b_cost_per_hour : ℕ)
  (medication_c_times_per_day : ℕ)
  (medication_c_cost_per_injection : ℕ) : ℕ :=
  let bed_charges := days * bed_charge_per_day
  let specialist_fees := 2 * (specialist_fee_per_hour * specialist_time_minutes / 60) * days
  let iv_charges := days * iv_cost
  let surgery_costs := surgery_duration_hours * (surgeon_fee_per_hour + assistant_fee_per_hour)
  let physical_therapy_fees := physical_therapy_fee_per_hour * physical_therapy_duration_hours * days
  let medication_a_cost := medication_a_times_per_day * medication_a_cost_per_pill * days
  let medication_b_cost := medication_b_duration_hours * medication_b_cost_per_hour * days
  let medication_c_cost := medication_c_times_per_day * medication_c_cost_per_injection * days
  bed_charges + specialist_fees + ambulance_ride_cost + iv_charges + surgery_costs + 
  physical_therapy_fees + medication_a_cost + medication_b_cost + medication_c_cost

/-- Theorem stating that the total medical bill for Dakota's hospital stay is $11,635 -/
theorem dakota_medical_bill : 
  total_medical_bill 3 900 250 15 1800 200 2 1500 800 300 1 3 20 2 45 2 35 = 11635 := by
  sorry

end NUMINAMATH_CALUDE_dakota_medical_bill_l1891_189145


namespace NUMINAMATH_CALUDE_final_expression_l1891_189132

theorem final_expression (x : ℝ) : ((3 * x + 5) - 5 * x) / 3 = (-2 * x + 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_final_expression_l1891_189132


namespace NUMINAMATH_CALUDE_erica_money_l1891_189123

def total_money : ℕ := 91
def sam_money : ℕ := 38

theorem erica_money : total_money - sam_money = 53 := by
  sorry

end NUMINAMATH_CALUDE_erica_money_l1891_189123


namespace NUMINAMATH_CALUDE_min_value_theorem_l1891_189177

theorem min_value_theorem (x y z : ℝ) (h : 2 * x + 2 * y + z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1891_189177


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1891_189155

/-- Given vectors a and b, find k such that k*a - 2*b is perpendicular to a -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) : 
  a = (1, 1) → b = (2, -3) → 
  (k * a.1 - 2 * b.1, k * a.2 - 2 * b.2) • a = 0 → 
  k = -1 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1891_189155


namespace NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l1891_189165

theorem minimum_questionnaires_to_mail 
  (response_rate : ℝ) 
  (required_responses : ℕ) 
  (h1 : response_rate = 0.7) 
  (h2 : required_responses = 300) : 
  ℕ := by
  
  sorry

#check minimum_questionnaires_to_mail

end NUMINAMATH_CALUDE_minimum_questionnaires_to_mail_l1891_189165


namespace NUMINAMATH_CALUDE_square_difference_plus_two_cubed_l1891_189167

theorem square_difference_plus_two_cubed : (7^2 - 3^2 + 2)^3 = 74088 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_two_cubed_l1891_189167


namespace NUMINAMATH_CALUDE_equation_solution_l1891_189187

theorem equation_solution (a b : ℝ) (h : a * 3^2 - b * 3 = 6) : 2023 - 6 * a + 2 * b = 2019 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1891_189187


namespace NUMINAMATH_CALUDE_train_or_plane_prob_not_ship_prob_prob_half_combinations_prob_sum_one_l1891_189182

-- Define the probabilities of each transportation mode
def train_prob : ℝ := 0.3
def ship_prob : ℝ := 0.2
def car_prob : ℝ := 0.1
def plane_prob : ℝ := 0.4

-- Define the sum of all probabilities
def total_prob : ℝ := train_prob + ship_prob + car_prob + plane_prob

-- Theorem for the probability of taking either a train or a plane
theorem train_or_plane_prob : train_prob + plane_prob = 0.7 := by sorry

-- Theorem for the probability of not taking a ship
theorem not_ship_prob : 1 - ship_prob = 0.8 := by sorry

-- Theorem for the combinations with probability 0.5
theorem prob_half_combinations :
  (train_prob + ship_prob = 0.5 ∧ car_prob + plane_prob = 0.5) := by sorry

-- Ensure that the probabilities sum to 1
theorem prob_sum_one : total_prob = 1 := by sorry

end NUMINAMATH_CALUDE_train_or_plane_prob_not_ship_prob_prob_half_combinations_prob_sum_one_l1891_189182


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_complementary_l1891_189189

-- Define the sample space for a standard six-sided die
def DieOutcome : Type := Fin 6

-- Define the event "the number is odd"
def isOdd (n : DieOutcome) : Prop := n.val % 2 = 1

-- Define the event "the number is greater than 5"
def isGreaterThan5 (n : DieOutcome) : Prop := n.val = 6

-- Theorem stating that the events are mutually exclusive
theorem events_mutually_exclusive :
  ∀ (n : DieOutcome), ¬(isOdd n ∧ isGreaterThan5 n) :=
sorry

-- Theorem stating that the events are not complementary
theorem events_not_complementary :
  ¬(∀ (n : DieOutcome), isOdd n ↔ ¬isGreaterThan5 n) :=
sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_events_not_complementary_l1891_189189


namespace NUMINAMATH_CALUDE_perpendicular_slope_l1891_189144

theorem perpendicular_slope (x y : ℝ) :
  let given_line := {(x, y) | 5 * x - 2 * y = 10}
  let given_slope := 5 / 2
  let perpendicular_slope := -1 / given_slope
  perpendicular_slope = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l1891_189144


namespace NUMINAMATH_CALUDE_percentage_relationship_l1891_189135

theorem percentage_relationship (p t j w : ℝ) : 
  j = 0.75 * p → 
  j = 0.80 * t → 
  t = p * (1 - w / 100) → 
  w = 6.25 := by
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1891_189135


namespace NUMINAMATH_CALUDE_symmetric_point_l1891_189184

/-- The line of symmetry --/
def line_of_symmetry (x y : ℝ) : Prop := 5*x + 4*y + 21 = 0

/-- The original point P --/
def P : ℝ × ℝ := (4, 0)

/-- The symmetric point P' --/
def P' : ℝ × ℝ := (-6, -8)

/-- Theorem stating that P' is symmetric to P with respect to the line of symmetry --/
theorem symmetric_point : 
  let midpoint := ((P.1 + P'.1) / 2, (P.2 + P'.2) / 2)
  line_of_symmetry midpoint.1 midpoint.2 ∧ 
  (P'.2 - P.2) * 5 = -(P'.1 - P.1) * 4 :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_l1891_189184


namespace NUMINAMATH_CALUDE_f_at_four_equals_zero_l1891_189174

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_at_four_equals_zero : f 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_at_four_equals_zero_l1891_189174


namespace NUMINAMATH_CALUDE_problem_statement_l1891_189164

theorem problem_statement (a b : ℝ) 
  (h1 : a + 1 / (a + 1) = b + 1 / (b - 1) - 2)
  (h2 : a ≠ -1)
  (h3 : b ≠ 1)
  (h4 : a - b + 2 ≠ 0) :
  a * b - a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1891_189164


namespace NUMINAMATH_CALUDE_keychain_cost_decrease_l1891_189139

theorem keychain_cost_decrease (P : ℝ) : 
  P - P * 0.35 - (P - P * 0.50) = 15 ∧ P - P * 0.50 = 50 → 
  P - P * 0.35 = 65 := by
sorry

end NUMINAMATH_CALUDE_keychain_cost_decrease_l1891_189139


namespace NUMINAMATH_CALUDE_percent_relation_l1891_189156

/-- Given that x is p percent of y, prove that p = 100x / y -/
theorem percent_relation (x y p : ℝ) (h : x = (p / 100) * y) : p = 100 * x / y := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1891_189156


namespace NUMINAMATH_CALUDE_chocolate_bars_count_l1891_189147

/-- Represents the number of chocolate bars in the gigantic box -/
def chocolate_bars_in_gigantic_box : ℕ :=
  let small_box_bars : ℕ := 45
  let medium_box_small_boxes : ℕ := 10
  let large_box_medium_boxes : ℕ := 25
  let gigantic_box_large_boxes : ℕ := 50
  gigantic_box_large_boxes * large_box_medium_boxes * medium_box_small_boxes * small_box_bars

/-- Theorem stating that the number of chocolate bars in the gigantic box is 562,500 -/
theorem chocolate_bars_count : chocolate_bars_in_gigantic_box = 562500 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_count_l1891_189147


namespace NUMINAMATH_CALUDE_set_operation_equality_l1891_189134

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x > 0}

def B : Set ℝ := {x : ℝ | x ≤ -1}

theorem set_operation_equality : 
  (A ∩ (U \ B)) ∪ (B ∩ (U \ A)) = {x : ℝ | x > 0 ∨ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l1891_189134


namespace NUMINAMATH_CALUDE_monday_children_count_l1891_189150

/-- The number of children who went to the zoo on Monday -/
def monday_children : ℕ := sorry

/-- The number of adults who went to the zoo on Monday -/
def monday_adults : ℕ := 5

/-- The number of children who went to the zoo on Tuesday -/
def tuesday_children : ℕ := 4

/-- The number of adults who went to the zoo on Tuesday -/
def tuesday_adults : ℕ := 2

/-- The cost of a child ticket -/
def child_ticket_cost : ℕ := 3

/-- The cost of an adult ticket -/
def adult_ticket_cost : ℕ := 4

/-- The total revenue for both days -/
def total_revenue : ℕ := 61

theorem monday_children_count : 
  monday_children = 7 ∧
  monday_children * child_ticket_cost + 
  monday_adults * adult_ticket_cost +
  tuesday_children * child_ticket_cost +
  tuesday_adults * adult_ticket_cost = total_revenue :=
sorry

end NUMINAMATH_CALUDE_monday_children_count_l1891_189150


namespace NUMINAMATH_CALUDE_instrument_purchase_plan_l1891_189138

-- Define the cost prices of instruments A and B
def cost_A : ℕ := 400
def cost_B : ℕ := 300

-- Define the selling prices of instruments A and B
def sell_A : ℕ := 760
def sell_B : ℕ := 540

-- Define the function for the number of B given A
def num_B (a : ℕ) : ℕ := 3 * a + 10

-- Define the total cost function
def total_cost (a : ℕ) : ℕ := cost_A * a + cost_B * (num_B a)

-- Define the profit function
def profit (a : ℕ) : ℕ := (sell_A - cost_A) * a + (sell_B - cost_B) * (num_B a)

-- Theorem statement
theorem instrument_purchase_plan :
  (2 * cost_A + 3 * cost_B = 1700) ∧
  (3 * cost_A + cost_B = 1500) ∧
  (∀ a : ℕ, total_cost a ≤ 30000 → profit a ≥ 21600 → 
    (a = 18 ∧ num_B a = 64) ∨ 
    (a = 19 ∧ num_B a = 67) ∨ 
    (a = 20 ∧ num_B a = 70)) :=
by sorry

end NUMINAMATH_CALUDE_instrument_purchase_plan_l1891_189138


namespace NUMINAMATH_CALUDE_derivative_of_exponential_l1891_189186

variable (a : ℝ) (ha : a > 0)

theorem derivative_of_exponential (x : ℝ) : 
  deriv (fun x => a^x) x = a^x * Real.log a := by sorry

end NUMINAMATH_CALUDE_derivative_of_exponential_l1891_189186


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l1891_189104

theorem unique_solution_for_equation :
  ∃! (a b c d : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) ∧
  a = 2 ^ (1 / 672) ∧
  b = -2 * 2 ^ (1 / 672) ∧
  c = -4 ∧
  d = 4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l1891_189104


namespace NUMINAMATH_CALUDE_max_factors_of_power_l1891_189157

-- Define the type for positive integers from 1 to 15
def PositiveIntegerTo15 : Type := {x : ℕ // 1 ≤ x ∧ x ≤ 15}

-- Define the function to count factors
def countFactors (m : ℕ) : ℕ := sorry

-- Define the function to calculate b^n
def powerFunction (b n : PositiveIntegerTo15) : ℕ := sorry

-- Theorem statement
theorem max_factors_of_power :
  ∃ (b n : PositiveIntegerTo15),
    ∀ (b' n' : PositiveIntegerTo15),
      countFactors (powerFunction b n) ≥ countFactors (powerFunction b' n') ∧
      countFactors (powerFunction b n) = 496 :=
sorry

end NUMINAMATH_CALUDE_max_factors_of_power_l1891_189157


namespace NUMINAMATH_CALUDE_distance_scientific_notation_l1891_189110

/-- The distance from the Chinese space station to the apogee of the Earth in meters -/
def distance : ℝ := 347000

/-- The coefficient in the scientific notation representation -/
def coefficient : ℝ := 3.47

/-- The exponent in the scientific notation representation -/
def exponent : ℕ := 5

/-- Theorem stating that the distance is equal to its scientific notation representation -/
theorem distance_scientific_notation : distance = coefficient * (10 ^ exponent) := by
  sorry

end NUMINAMATH_CALUDE_distance_scientific_notation_l1891_189110


namespace NUMINAMATH_CALUDE_unicorns_games_played_l1891_189179

theorem unicorns_games_played (initial_games : ℕ) (initial_wins : ℕ) : 
  initial_wins = (initial_games * 45 / 100) →
  (initial_wins + 6) = ((initial_games + 8) * 1 / 2) →
  initial_games + 8 = 48 := by
sorry

end NUMINAMATH_CALUDE_unicorns_games_played_l1891_189179


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l1891_189173

/-- Given two points on a parabola that are symmetric with respect to a line, prove the value of m -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →  -- A is on the parabola
  y₂ = 2 * x₂^2 →  -- B is on the parabola
  (∃ (x₀ y₀ : ℝ), x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2 ∧ y₀ = x₀ + m) →  -- midpoint condition for symmetry
  x₁ * x₂ = -1/2 →  -- given condition
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l1891_189173


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l1891_189196

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^(n - 1)

theorem fourth_term_of_sequence (x : ℝ) :
  let a₁ := x
  let a₂ := 3 * x + 6
  let a₃ := 7 * x + 21
  (∃ r : ℝ, r ≠ 0 ∧ 
    geometric_sequence a₁ r 2 = a₂ ∧ 
    geometric_sequence a₁ r 3 = a₃) →
  geometric_sequence a₁ ((3 * x + 6) / x) 4 = 220.5 :=
by sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l1891_189196


namespace NUMINAMATH_CALUDE_star_calculation_l1891_189115

-- Define the ★ operation
def star (a b : ℚ) : ℚ := (a + b) / 3

-- Theorem statement
theorem star_calculation : star (star 7 15) 10 = 52 / 9 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1891_189115


namespace NUMINAMATH_CALUDE_whitney_money_left_l1891_189130

/-- The amount of money Whitney has left after her purchase at the school book fair --/
def money_left_over : ℕ :=
  let initial_money : ℕ := 2 * 20
  let poster_cost : ℕ := 5
  let notebook_cost : ℕ := 4
  let bookmark_cost : ℕ := 2
  let poster_quantity : ℕ := 2
  let notebook_quantity : ℕ := 3
  let bookmark_quantity : ℕ := 2
  let total_cost : ℕ := poster_cost * poster_quantity + 
                        notebook_cost * notebook_quantity + 
                        bookmark_cost * bookmark_quantity
  initial_money - total_cost

theorem whitney_money_left : money_left_over = 14 := by
  sorry

end NUMINAMATH_CALUDE_whitney_money_left_l1891_189130


namespace NUMINAMATH_CALUDE_shortest_distance_to_origin_l1891_189113

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

-- Define the left focus F
def left_focus : ℝ × ℝ := sorry

-- Define a point P on the right branch of the hyperbola
def point_P : ℝ × ℝ := sorry

-- Define point A satisfying the orthogonality condition
def point_A : ℝ × ℝ := sorry

-- State the theorem
theorem shortest_distance_to_origin :
  ∀ (A : ℝ × ℝ),
    (∃ (P : ℝ × ℝ), hyperbola P.1 P.2 ∧ 
      ((A.1 - P.1) * (A.1 - left_focus.1) + (A.2 - P.2) * (A.2 - left_focus.2) = 0)) →
    (∃ (d : ℝ), d = Real.sqrt 3 ∧ 
      ∀ (B : ℝ × ℝ), Real.sqrt (B.1^2 + B.2^2) ≥ d) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_to_origin_l1891_189113


namespace NUMINAMATH_CALUDE_polynomial_property_l1891_189168

-- Define the polynomial Q(x)
def Q (x d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the conditions
theorem polynomial_property (d e f : ℝ) :
  -- The y-intercept is 12
  Q 0 d e f = 12 →
  -- The product of zeros is equal to -f/3
  (∃ α β γ : ℝ, Q α d e f = 0 ∧ Q β d e f = 0 ∧ Q γ d e f = 0 ∧ α * β * γ = -f / 3) →
  -- The mean of zeros is equal to the product of zeros
  (∃ α β γ : ℝ, Q α d e f = 0 ∧ Q β d e f = 0 ∧ Q γ d e f = 0 ∧ (α + β + γ) / 3 = -f / 3) →
  -- The sum of coefficients is equal to the product of zeros
  3 + d + e + f = -f / 3 →
  -- Then e = -55
  e = -55 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l1891_189168


namespace NUMINAMATH_CALUDE_hypotenuse_product_square_l1891_189166

-- Define the triangles and their properties
def right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2

theorem hypotenuse_product_square (x y h₁ h₂ : ℝ) :
  right_triangle x (2*y) h₁ →  -- T1
  right_triangle x y h₂ →      -- T2
  x * (2*y) / 2 = 8 →          -- Area of T1
  x * y / 2 = 4 →              -- Area of T2
  (h₁ * h₂)^2 = 160 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_product_square_l1891_189166


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l1891_189183

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_proof (a b c : ℝ) :
  (∀ x, f a b c x = 0 ↔ x = -2 ∨ x = 1) →  -- Roots condition
  (∃ m, ∀ x, f a b c x ≤ m) →              -- Maximum value condition
  (∀ x, f a b c x = -x^2 - x + 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l1891_189183


namespace NUMINAMATH_CALUDE_expression_evaluation_l1891_189103

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (3 + Real.sqrt 3)⁻¹ + (Real.sqrt 3 - 3)⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1891_189103


namespace NUMINAMATH_CALUDE_lcm_210_396_l1891_189162

theorem lcm_210_396 : Nat.lcm 210 396 = 13860 := by
  sorry

end NUMINAMATH_CALUDE_lcm_210_396_l1891_189162


namespace NUMINAMATH_CALUDE_probability_of_black_ball_l1891_189129

theorem probability_of_black_ball 
  (p_red : ℝ) 
  (p_white : ℝ) 
  (h1 : p_red = 0.42) 
  (h2 : p_white = 0.28) 
  (h3 : 0 ≤ p_red ∧ p_red ≤ 1) 
  (h4 : 0 ≤ p_white ∧ p_white ≤ 1) : 
  1 - p_red - p_white = 0.30 := by
sorry

end NUMINAMATH_CALUDE_probability_of_black_ball_l1891_189129


namespace NUMINAMATH_CALUDE_jimmys_coffee_bean_weight_l1891_189118

/-- Proves the weight of Jimmy's coffee bean bags given the problem conditions -/
theorem jimmys_coffee_bean_weight 
  (suki_bags : ℝ) 
  (suki_weight_per_bag : ℝ) 
  (jimmy_bags : ℝ) 
  (container_weight : ℝ) 
  (num_containers : ℕ) 
  (h1 : suki_bags = 6.5)
  (h2 : suki_weight_per_bag = 22)
  (h3 : jimmy_bags = 4.5)
  (h4 : container_weight = 8)
  (h5 : num_containers = 28) :
  (↑num_containers * container_weight - suki_bags * suki_weight_per_bag) / jimmy_bags = 18 := by
  sorry

#check jimmys_coffee_bean_weight

end NUMINAMATH_CALUDE_jimmys_coffee_bean_weight_l1891_189118


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1891_189195

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    if S_n = 2 and S_{3n} = 18, then S_{4n} = 26 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = (k * (2 * (a 1) + (k - 1) * (a 2 - a 1))) / 2) →
  S n = 2 →
  S (3 * n) = 18 →
  S (4 * n) = 26 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1891_189195


namespace NUMINAMATH_CALUDE_meeting_arrangement_count_l1891_189198

/-- Represents the number of schools -/
def num_schools : ℕ := 3

/-- Represents the number of members per school -/
def members_per_school : ℕ := 6

/-- Represents the total number of members -/
def total_members : ℕ := num_schools * members_per_school

/-- Represents the number of representatives sent by the host school -/
def host_representatives : ℕ := 2

/-- Represents the number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The number of ways to arrange the meeting -/
def meeting_arrangements : ℕ := 1620

/-- Theorem stating the number of ways to arrange the meeting -/
theorem meeting_arrangement_count :
  num_schools * (Nat.choose members_per_school host_representatives *
    (Nat.choose members_per_school non_host_representatives)^(num_schools - 1)) = meeting_arrangements :=
by sorry

end NUMINAMATH_CALUDE_meeting_arrangement_count_l1891_189198


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l1891_189105

-- Define the universe set U
def U : Set ℕ := {x | 0 < x ∧ x ≤ 8}

-- Define set S
def S : Set ℕ := {1, 2, 4, 5}

-- Define set T
def T : Set ℕ := {3, 5, 7}

-- Theorem statement
theorem intersection_complement_equal : S ∩ (U \ T) = {1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l1891_189105


namespace NUMINAMATH_CALUDE_mass_of_man_in_boat_l1891_189125

/-- The mass of a man who causes a boat to sink by a certain amount in water. -/
def mass_of_man (boat_length boat_breadth boat_sinkage water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sinkage * water_density

/-- Theorem stating that the mass of the man is 60 kg given the specified conditions. -/
theorem mass_of_man_in_boat : 
  mass_of_man 3 2 0.01 1000 = 60 := by
  sorry

end NUMINAMATH_CALUDE_mass_of_man_in_boat_l1891_189125


namespace NUMINAMATH_CALUDE_smallest_number_after_removal_largest_number_after_removal_l1891_189181

-- Define the original number as a string
def originalNumber : String := "123456789101112...5657585960"

-- Define the number of digits to remove
def digitsToRemove : Nat := 100

-- Define the function to remove digits and get the smallest number
def smallestAfterRemoval (s : String) (n : Nat) : Nat :=
  sorry

-- Define the function to remove digits and get the largest number
def largestAfterRemoval (s : String) (n : Nat) : Nat :=
  sorry

-- Theorem for the smallest number
theorem smallest_number_after_removal :
  smallestAfterRemoval originalNumber digitsToRemove = 123450 :=
by sorry

-- Theorem for the largest number
theorem largest_number_after_removal :
  largestAfterRemoval originalNumber digitsToRemove = 56758596049 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_after_removal_largest_number_after_removal_l1891_189181


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l1891_189133

theorem square_rectangle_area_relation : 
  ∀ x : ℝ,
  let square_side : ℝ := x - 5
  let rect_length : ℝ := x - 4
  let rect_width : ℝ := x + 3
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_length * rect_width
  (rect_area = 3 * square_area) →
  (∃ y : ℝ, y ≠ x ∧ 
    let square_side' : ℝ := y - 5
    let rect_length' : ℝ := y - 4
    let rect_width' : ℝ := y + 3
    let square_area' : ℝ := square_side' ^ 2
    let rect_area' : ℝ := rect_length' * rect_width'
    (rect_area' = 3 * square_area')) →
  x + y = 33/2 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l1891_189133


namespace NUMINAMATH_CALUDE_max_value_3x_4y_l1891_189172

theorem max_value_3x_4y (x y : ℝ) (h : x^2 + y^2 = 14*x + 6*y + 6) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + b^2 = 14*a + 6*b + 6 → 3*a + 4*b ≤ max) ∧ max = 73 := by
  sorry

end NUMINAMATH_CALUDE_max_value_3x_4y_l1891_189172


namespace NUMINAMATH_CALUDE_distance_between_points_l1891_189131

/-- The distance between two points A and B given train travel conditions -/
theorem distance_between_points (v_pas v_freight : ℝ) (d : ℝ) : 
  (d / v_freight - d / v_pas = 3.2) →
  (v_pas * (d / v_freight) = d + 288) →
  (d / (v_freight + 10) - d / (v_pas + 10) = 2.4) →
  d = 360 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l1891_189131


namespace NUMINAMATH_CALUDE_distance_sum_difference_bound_l1891_189114

-- Define a convex dodecagon
def ConvexDodecagon : Type := Unit

-- Define a point inside the dodecagon
def Point : Type := Unit

-- Define the distance between two points
def distance (p q : Point) : ℝ := sorry

-- Define the vertices of the dodecagon
def vertices (d : ConvexDodecagon) : Finset Point := sorry

-- Define the sum of distances from a point to all vertices
def sum_distances (p : Point) (d : ConvexDodecagon) : ℝ :=
  (vertices d).sum (λ v => distance p v)

-- The main theorem
theorem distance_sum_difference_bound
  (d : ConvexDodecagon) (p q : Point)
  (h : distance p q = 10) :
  |sum_distances p d - sum_distances q d| < 100 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_difference_bound_l1891_189114


namespace NUMINAMATH_CALUDE_triangle_problem_l1891_189111

open Real

theorem triangle_problem (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle
  A + B + C = π ∧
  sin A / a = sin B / b ∧ sin A / a = sin C / c ∧  -- Law of sines
  sqrt 3 * c * cos A - a * cos C + b - 2 * c = 0 →
  A = π / 3 ∧ 
  sqrt 3 / 2 < cos B + cos C ∧ cos B + cos C ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1891_189111


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1891_189102

theorem polynomial_factorization (y : ℝ) :
  (16 * y^7 - 36 * y^5 + 8 * y) - (4 * y^7 - 12 * y^5 - 8 * y) = 8 * y * (3 * y^6 - 6 * y^4 + 4) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1891_189102


namespace NUMINAMATH_CALUDE_polynomial_root_relation_l1891_189100

/-- Two monic cubic polynomials with specified roots and a relation between them -/
theorem polynomial_root_relation (s : ℝ) (h j : ℝ → ℝ) : 
  (∀ x, h x = (x - (s + 2)) * (x - (s + 6)) * (x - c)) →
  (∀ x, j x = (x - (s + 4)) * (x - (s + 8)) * (x - d)) →
  (∀ x, 2 * (h x - j x) = s) →
  s = 64 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_relation_l1891_189100


namespace NUMINAMATH_CALUDE_regression_change_l1891_189140

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 7 - 3 * x

-- Theorem statement
theorem regression_change (x₁ x₂ : ℝ) (h : x₂ = x₁ + 2) :
  regression_equation x₁ - regression_equation x₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_regression_change_l1891_189140


namespace NUMINAMATH_CALUDE_square_sum_equality_l1891_189188

theorem square_sum_equality (x y : ℝ) 
  (h1 : x + 3 * y = 3) 
  (h2 : x * y = -3) : 
  x^2 + 9 * y^2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l1891_189188


namespace NUMINAMATH_CALUDE_remainder_theorem_l1891_189193

theorem remainder_theorem (P D Q R D' Q'' R'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  P % (D * D') = R := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1891_189193


namespace NUMINAMATH_CALUDE_frank_candy_weight_l1891_189143

/-- Frank's candy weight in pounds -/
def frank_candy : ℕ := 10

/-- Gwen's candy weight in pounds -/
def gwen_candy : ℕ := 7

/-- Total candy weight in pounds -/
def total_candy : ℕ := 17

theorem frank_candy_weight : 
  frank_candy + gwen_candy = total_candy :=
by sorry

end NUMINAMATH_CALUDE_frank_candy_weight_l1891_189143


namespace NUMINAMATH_CALUDE_car_rental_cost_l1891_189180

/-- Calculates the total cost of renting a car for three days with varying rates and mileage. -/
theorem car_rental_cost (base_rate_day1 base_rate_day2 base_rate_day3 : ℚ)
                        (per_mile_day1 per_mile_day2 per_mile_day3 : ℚ)
                        (miles_day1 miles_day2 miles_day3 : ℚ) :
  base_rate_day1 = 150 →
  base_rate_day2 = 100 →
  base_rate_day3 = 75 →
  per_mile_day1 = 0.5 →
  per_mile_day2 = 0.4 →
  per_mile_day3 = 0.3 →
  miles_day1 = 620 →
  miles_day2 = 744 →
  miles_day3 = 510 →
  base_rate_day1 + miles_day1 * per_mile_day1 +
  base_rate_day2 + miles_day2 * per_mile_day2 +
  base_rate_day3 + miles_day3 * per_mile_day3 = 1085.6 :=
by sorry

end NUMINAMATH_CALUDE_car_rental_cost_l1891_189180
