import Mathlib

namespace NUMINAMATH_CALUDE_det_necessary_not_sufficient_for_parallel_l4161_416161

/-- Determinant of a 2x2 matrix --/
def det (a₁ b₁ a₂ b₂ : ℝ) : ℝ := a₁ * b₂ - a₂ * b₁

/-- Two lines are parallel --/
def parallel (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ ≠ k * c₂

theorem det_necessary_not_sufficient_for_parallel
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) (h₁ : a₁^2 + b₁^2 ≠ 0) (h₂ : a₂^2 + b₂^2 ≠ 0) :
  (det a₁ b₁ a₂ b₂ = 0 → parallel a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(parallel a₁ b₁ c₁ a₂ b₂ c₂ → det a₁ b₁ a₂ b₂ = 0) :=
sorry

end NUMINAMATH_CALUDE_det_necessary_not_sufficient_for_parallel_l4161_416161


namespace NUMINAMATH_CALUDE_exam_score_proof_l4161_416104

/-- Proves that the average score of students who took the exam on the assigned day was 65% -/
theorem exam_score_proof (total_students : ℕ) (assigned_day_percentage : ℚ) 
  (makeup_score : ℚ) (class_average : ℚ) : 
  total_students = 100 →
  assigned_day_percentage = 70 / 100 →
  makeup_score = 95 / 100 →
  class_average = 74 / 100 →
  (assigned_day_percentage * total_students * assigned_day_score + 
   (1 - assigned_day_percentage) * total_students * makeup_score) / total_students = class_average →
  assigned_day_score = 65 / 100 :=
by
  sorry

#check exam_score_proof

end NUMINAMATH_CALUDE_exam_score_proof_l4161_416104


namespace NUMINAMATH_CALUDE_blocks_for_house_l4161_416133

/-- Given that Randy used 80 blocks in total for a tower and a house, 
    and 27 blocks for the tower, prove that he used 53 blocks for the house. -/
theorem blocks_for_house (total : ℕ) (tower : ℕ) (house : ℕ) 
    (h1 : total = 80) (h2 : tower = 27) (h3 : total = tower + house) : house = 53 := by
  sorry

end NUMINAMATH_CALUDE_blocks_for_house_l4161_416133


namespace NUMINAMATH_CALUDE_floor_times_x_eq_54_l4161_416157

theorem floor_times_x_eq_54 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 ∧ abs (x - 7.7143) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_eq_54_l4161_416157


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4161_416125

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 - 2*x - 3 > 0 ↔ x > 3 ∨ x < -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l4161_416125


namespace NUMINAMATH_CALUDE_cartesian_equation_chord_length_l4161_416101

-- Define the polar equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ)^2 = 4 * Real.cos θ

-- Define the parametric equations of line l
def line_equation (t x y : ℝ) : Prop :=
  x = 2 + (1/2) * t ∧ y = (Real.sqrt 3 / 2) * t

-- Theorem for the Cartesian equation of curve C
theorem cartesian_equation (x y : ℝ) :
  (∃ ρ θ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  y^2 = 4*x :=
sorry

-- Theorem for the length of chord AB
theorem chord_length (A B : ℝ × ℝ) :
  (∃ t₁ t₂, line_equation t₁ A.1 A.2 ∧ line_equation t₂ B.1 B.2 ∧
   A.2^2 = 4*A.1 ∧ B.2^2 = 4*B.1) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 7 / 3 :=
sorry

end NUMINAMATH_CALUDE_cartesian_equation_chord_length_l4161_416101


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l4161_416127

theorem fifteenth_student_age (total_students : ℕ) (avg_age : ℕ) 
  (group1_size : ℕ) (group1_avg : ℕ) (group2_size : ℕ) (group2_avg : ℕ) :
  total_students = 15 →
  avg_age = 15 →
  group1_size = 3 →
  group1_avg = 14 →
  group2_size = 11 →
  group2_avg = 16 →
  total_students * avg_age - (group1_size * group1_avg + group2_size * group2_avg) = 7 := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l4161_416127


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l4161_416176

theorem purely_imaginary_z (b : ℝ) :
  let z : ℂ := Complex.I * (1 + b * Complex.I) + 2 + 3 * b * Complex.I
  (z.re = 0) → z = 7 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l4161_416176


namespace NUMINAMATH_CALUDE_problem_statement_l4161_416183

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → b / a + 2 / b ≤ x / y + 2 / x) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 → a^2 + b^2 ≤ x^2 + y^2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l4161_416183


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l4161_416139

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  p^4 + 8*p^3 + 16*p^2 + 5*p + 2 = 0 →
  q^4 + 8*q^3 + 16*q^2 + 5*q + 2 = 0 →
  r^4 + 8*r^3 + 16*r^2 + 5*r + 2 = 0 →
  s^4 + 8*s^3 + 16*s^2 + 5*s + 2 = 0 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 8 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l4161_416139


namespace NUMINAMATH_CALUDE_m_div_125_eq_2_pow_124_l4161_416149

/-- The smallest positive integer that is a multiple of 125 and has exactly 125 positive integral divisors -/
def m : ℕ := sorry

/-- m is a multiple of 125 -/
axiom m_multiple_of_125 : 125 ∣ m

/-- m has exactly 125 positive integral divisors -/
axiom m_divisors_count : (Finset.filter (· ∣ m) (Finset.range (m + 1))).card = 125

/-- m is the smallest such positive integer -/
axiom m_is_smallest : ∀ k : ℕ, k < m → ¬(125 ∣ k ∧ (Finset.filter (· ∣ k) (Finset.range (k + 1))).card = 125)

/-- The main theorem to prove -/
theorem m_div_125_eq_2_pow_124 : m / 125 = 2^124 := by sorry

end NUMINAMATH_CALUDE_m_div_125_eq_2_pow_124_l4161_416149


namespace NUMINAMATH_CALUDE_bus_capacity_equality_l4161_416160

theorem bus_capacity_equality (x : ℕ) : 50 * x + 10 = 52 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_bus_capacity_equality_l4161_416160


namespace NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l4161_416166

def num_dice : ℕ := 4
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def valid_progressions : List (List ℕ) := [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]

def favorable_outcomes : ℕ := valid_progressions.length * (num_dice.factorial)

theorem dice_arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l4161_416166


namespace NUMINAMATH_CALUDE_largest_negative_integer_l4161_416150

theorem largest_negative_integer :
  ∀ n : ℤ, n < 0 → n ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_largest_negative_integer_l4161_416150


namespace NUMINAMATH_CALUDE_race_catchup_time_l4161_416129

/-- Proves that Nicky runs for 48 seconds before Cristina catches up to him in a 500-meter race --/
theorem race_catchup_time (race_distance : ℝ) (head_start : ℝ) (cristina_speed : ℝ) (nicky_speed : ℝ)
  (h1 : race_distance = 500)
  (h2 : head_start = 12)
  (h3 : cristina_speed = 5)
  (h4 : nicky_speed = 3) :
  let catchup_time := head_start + (head_start * nicky_speed) / (cristina_speed - nicky_speed)
  catchup_time = 48 := by
  sorry

end NUMINAMATH_CALUDE_race_catchup_time_l4161_416129


namespace NUMINAMATH_CALUDE_meal_pass_cost_sally_meal_pass_cost_l4161_416106

/-- Calculates the cost of a meal pass for Sally's trip to Sea World --/
theorem meal_pass_cost (savings : ℕ) (parking : ℕ) (entrance : ℕ) (distance : ℕ) (mpg : ℕ) 
  (gas_price : ℕ) (additional_savings : ℕ) : ℕ :=
  let round_trip := 2 * distance
  let gas_needed := round_trip / mpg
  let gas_cost := gas_needed * gas_price
  let known_costs := parking + entrance + gas_cost
  let remaining_costs := known_costs - savings
  additional_savings - remaining_costs

/-- The meal pass for Sally's trip to Sea World costs $25 --/
theorem sally_meal_pass_cost : 
  meal_pass_cost 28 10 55 165 30 3 95 = 25 := by
  sorry

end NUMINAMATH_CALUDE_meal_pass_cost_sally_meal_pass_cost_l4161_416106


namespace NUMINAMATH_CALUDE_no_real_roots_iff_range_m_range_when_necessary_not_sufficient_l4161_416186

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : Prop :=
  x^2 - 2*a*x + 2*a^2 - a - 6 = 0

-- Define the range of a for no real roots
def no_real_roots (a : ℝ) : Prop :=
  a < -2 ∨ a > 3

-- Define the necessary condition
def necessary_condition (a : ℝ) : Prop :=
  -2 ≤ a ∧ a ≤ 3

-- Define the condition q
def condition_q (m a : ℝ) : Prop :=
  m - 1 ≤ a ∧ a ≤ m + 3

-- Theorem 1: The equation has no real roots iff a is in the specified range
theorem no_real_roots_iff_range (a : ℝ) :
  (∀ x : ℝ, ¬(quadratic_equation a x)) ↔ no_real_roots a :=
sorry

-- Theorem 2: If the necessary condition is true but not sufficient for condition q,
-- then m is in the range [-1, 0]
theorem m_range_when_necessary_not_sufficient :
  (∀ a : ℝ, condition_q m a → necessary_condition a) ∧
  (∃ a : ℝ, necessary_condition a ∧ ¬(condition_q m a)) →
  -1 ≤ m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_range_m_range_when_necessary_not_sufficient_l4161_416186


namespace NUMINAMATH_CALUDE_louise_and_tom_ages_l4161_416153

/-- Given the age relationship between Louise and Tom, prove their current ages sum to 26 -/
theorem louise_and_tom_ages (L T : ℕ) 
  (h1 : L = T + 8) 
  (h2 : L + 4 = 3 * (T - 2)) : 
  L + T = 26 := by
  sorry

end NUMINAMATH_CALUDE_louise_and_tom_ages_l4161_416153


namespace NUMINAMATH_CALUDE_francie_remaining_money_l4161_416178

def initial_allowance : ℕ := 5
def initial_weeks : ℕ := 8
def raised_allowance : ℕ := 6
def raised_weeks : ℕ := 6
def cash_gift : ℕ := 20
def investment_amount : ℕ := 10
def investment_return_rate : ℚ := 5 / 100
def video_game_cost : ℕ := 35

def total_savings : ℚ :=
  (initial_allowance * initial_weeks +
   raised_allowance * raised_weeks +
   cash_gift : ℚ)

def total_with_investment : ℚ :=
  total_savings + investment_amount * investment_return_rate

def remaining_after_clothes : ℚ :=
  total_with_investment / 2

theorem francie_remaining_money :
  remaining_after_clothes - video_game_cost = 13.25 := by
  sorry

end NUMINAMATH_CALUDE_francie_remaining_money_l4161_416178


namespace NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l4161_416199

/-- The repeating cycle of the decimal representation of 1/13 -/
def cycle : List Nat := [0, 7, 6, 9, 2, 3]

/-- The length of the repeating cycle -/
def cycle_length : Nat := 6

/-- The position we're interested in -/
def target_position : Nat := 150

/-- Theorem: The 150th digit after the decimal point in the decimal 
    representation of 1/13 is 3 -/
theorem digit_150_of_one_thirteenth (h : cycle = [0, 7, 6, 9, 2, 3]) :
  cycle[target_position % cycle_length] = 3 := by
  sorry

end NUMINAMATH_CALUDE_digit_150_of_one_thirteenth_l4161_416199


namespace NUMINAMATH_CALUDE_equal_volumes_condition_l4161_416110

/-- Represents the side lengths of the square prisms to be removed from a cube --/
structure PrismSides where
  c : ℝ
  b : ℝ
  a : ℝ

/-- Calculates the volume of the remaining body after removing square prisms --/
def remainingVolume (sides : PrismSides) : ℝ :=
  1 - (sides.c^2 + (sides.b^2 - sides.c^2 * sides.b) + (sides.a^2 - sides.c^2 * sides.a - sides.b^2 * sides.a + sides.c^2 * sides.b))

/-- Theorem stating the conditions for equal volumes --/
theorem equal_volumes_condition (sides : PrismSides) : 
  sides.c = 1/2 ∧ 
  sides.b = (1 + Real.sqrt 17) / 8 ∧ 
  sides.a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64 ∧
  sides.c < sides.b ∧ sides.b < sides.a ∧ sides.a < 1 →
  remainingVolume sides = 1/4 :=
sorry

end NUMINAMATH_CALUDE_equal_volumes_condition_l4161_416110


namespace NUMINAMATH_CALUDE_son_age_l4161_416171

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_son_age_l4161_416171


namespace NUMINAMATH_CALUDE_bret_caught_12_frogs_l4161_416172

-- Define the number of frogs caught by each person
def alster_frogs : ℕ := 2
def quinn_frogs : ℕ := 2 * alster_frogs
def bret_frogs : ℕ := 3 * quinn_frogs

-- Theorem to prove
theorem bret_caught_12_frogs : bret_frogs = 12 := by
  sorry

end NUMINAMATH_CALUDE_bret_caught_12_frogs_l4161_416172


namespace NUMINAMATH_CALUDE_correct_seat_notation_l4161_416141

/-- Represents a cinema seat notation -/
def CinemaSeat := ℕ × ℕ

/-- Converts a row and seat number to cinema seat notation -/
def toSeatNotation (row : ℕ) (seat : ℕ) : CinemaSeat := (row, seat)

theorem correct_seat_notation :
  toSeatNotation 2 5 = (2, 5) := by sorry

end NUMINAMATH_CALUDE_correct_seat_notation_l4161_416141


namespace NUMINAMATH_CALUDE_vegetables_minus_fruits_l4161_416115

def cucumbers : ℕ := 6
def tomatoes : ℕ := 8
def apples : ℕ := 2
def bananas : ℕ := 4

def vegetables : ℕ := cucumbers + tomatoes
def fruits : ℕ := apples + bananas

theorem vegetables_minus_fruits : vegetables - fruits = 8 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_minus_fruits_l4161_416115


namespace NUMINAMATH_CALUDE_compute_expression_l4161_416123

theorem compute_expression : 75 * 1313 - 25 * 1313 + 50 * 1313 = 131300 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l4161_416123


namespace NUMINAMATH_CALUDE_floor_sum_inequality_l4161_416124

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := 
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := 
  x - floor x

-- State the theorem
theorem floor_sum_inequality (x y : ℝ) : 
  (floor x + floor y ≤ floor (x + y)) ∧ 
  (floor (x + y) ≤ floor x + floor y + 1) ∧ 
  (floor x + floor y = floor (x + y) ∨ floor (x + y) = floor x + floor y + 1) :=
sorry

end NUMINAMATH_CALUDE_floor_sum_inequality_l4161_416124


namespace NUMINAMATH_CALUDE_probability_multiple_5_or_7_l4161_416126

def is_multiple_of_5_or_7 (n : ℕ) : Bool :=
  n % 5 = 0 || n % 7 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_5_or_7 |>.length

theorem probability_multiple_5_or_7 :
  count_multiples 50 / 50 = 8 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_5_or_7_l4161_416126


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_1500_by_20_percent_l4161_416145

theorem increase_by_percentage (initial : ℕ) (percentage : ℚ) : 
  initial + (initial * percentage) = initial * (1 + percentage) := by sorry

theorem increase_1500_by_20_percent : 
  1500 + (1500 * (20 / 100)) = 1800 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_1500_by_20_percent_l4161_416145


namespace NUMINAMATH_CALUDE_fraction_equality_l4161_416188

theorem fraction_equality (w x y : ℚ) 
  (h1 : w / y = 3 / 4)
  (h2 : (x + y) / y = 13 / 4) :
  w / x = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l4161_416188


namespace NUMINAMATH_CALUDE_billion_difference_value_l4161_416170

/-- Arnaldo's definition of a billion -/
def arnaldo_billion : ℕ := 1000000 * 1000000

/-- Correct definition of a billion -/
def correct_billion : ℕ := 1000 * 1000000

/-- The difference between Arnaldo's definition and the correct definition -/
def billion_difference : ℕ := arnaldo_billion - correct_billion

theorem billion_difference_value : billion_difference = 999000000000 := by
  sorry

end NUMINAMATH_CALUDE_billion_difference_value_l4161_416170


namespace NUMINAMATH_CALUDE_total_time_is_ten_years_l4161_416184

/-- The total time taken to find two artifacts given the research and expedition time for the first artifact, and a multiplier for the second artifact. -/
def total_time_for_artifacts (research_time_1 : ℝ) (expedition_time_1 : ℝ) (multiplier : ℝ) : ℝ :=
  let time_1 := research_time_1 + expedition_time_1
  let time_2 := time_1 * multiplier
  time_1 + time_2

/-- Theorem stating that the total time to find both artifacts is 10 years -/
theorem total_time_is_ten_years :
  total_time_for_artifacts 0.5 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_total_time_is_ten_years_l4161_416184


namespace NUMINAMATH_CALUDE_find_m_l4161_416138

def A (m : ℕ) : Set ℕ := {1, 2, m}
def B : Set ℕ := {4, 7, 13}

def f (x : ℕ) : ℕ := 3 * x + 1

theorem find_m : ∃ m : ℕ, 
  (∀ x ∈ A m, f x ∈ B) ∧ 
  m = 4 := by sorry

end NUMINAMATH_CALUDE_find_m_l4161_416138


namespace NUMINAMATH_CALUDE_clara_age_l4161_416169

def anna_age : ℕ := 54
def years_ago : ℕ := 41

theorem clara_age : ℕ :=
  let anna_age_then := anna_age - years_ago
  let clara_age_then := 3 * anna_age_then
  clara_age_then + years_ago

#check clara_age

end NUMINAMATH_CALUDE_clara_age_l4161_416169


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l4161_416193

theorem least_number_divisible_by_five_primes : ℕ := by
  -- Define the property of being divisible by five different primes
  let divisible_by_five_primes (n : ℕ) :=
    ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ),
      Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
      p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
      p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
      p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
      p₄ ≠ p₅ ∧
      n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0

  -- State that 2310 is divisible by five different primes
  have h1 : divisible_by_five_primes 2310 := by sorry

  -- State that 2310 is the least such number
  have h2 : ∀ m : ℕ, m < 2310 → ¬(divisible_by_five_primes m) := by sorry

  -- Conclude that 2310 is the answer
  exact 2310

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l4161_416193


namespace NUMINAMATH_CALUDE_number_problem_l4161_416144

theorem number_problem (n : ℚ) : (1/2 : ℚ) * (3/5 : ℚ) * n = 36 → n = 120 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4161_416144


namespace NUMINAMATH_CALUDE_factor_of_100140001_l4161_416120

theorem factor_of_100140001 : ∃ (n : ℕ), 
  8000 < n ∧ 
  n < 9000 ∧ 
  100140001 % n = 0 :=
by
  use 8221
  sorry

end NUMINAMATH_CALUDE_factor_of_100140001_l4161_416120


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l4161_416152

/-- Given a hyperbola with equation x²/16 - y²/25 = 1, 
    the positive slope of its asymptotes is 5/4 -/
theorem hyperbola_asymptote_slope (x y : ℝ) :
  x^2 / 16 - y^2 / 25 = 1 → 
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l4161_416152


namespace NUMINAMATH_CALUDE_problem_statement_l4161_416117

theorem problem_statement (a b : ℝ) 
  (h1 : a^2 * (b^2 + 1) + b * (b + 2*a) = 40)
  (h2 : a * (b + 1) + b = 8) : 
  1/a^2 + 1/b^2 = 8 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l4161_416117


namespace NUMINAMATH_CALUDE_expected_value_r₃_l4161_416163

/-- The expected value of a single fair six-sided die roll -/
def single_die_ev : ℝ := 3.5

/-- The number of dice rolled in the first round -/
def first_round_dice : ℕ := 8

/-- The expected value of r₁ (the sum of first_round_dice fair dice rolls) -/
def r₁_ev : ℝ := first_round_dice * single_die_ev

/-- The expected value of r₂ (the sum of r₁_ev fair dice rolls) -/
def r₂_ev : ℝ := r₁_ev * single_die_ev

/-- The expected value of r₃ (the sum of r₂_ev fair dice rolls) -/
def r₃_ev : ℝ := r₂_ev * single_die_ev

theorem expected_value_r₃ : r₃_ev = 343 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_r₃_l4161_416163


namespace NUMINAMATH_CALUDE_quadratic_inequality_l4161_416121

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x)) :
  f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l4161_416121


namespace NUMINAMATH_CALUDE_sams_new_books_l4161_416191

theorem sams_new_books (adventure_books : ℕ) (mystery_books : ℕ) (used_books : ℕ) 
  (h1 : adventure_books = 13)
  (h2 : mystery_books = 17)
  (h3 : used_books = 15) : 
  adventure_books + mystery_books - used_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_sams_new_books_l4161_416191


namespace NUMINAMATH_CALUDE_unique_solution_rational_equation_l4161_416182

theorem unique_solution_rational_equation :
  ∃! x : ℝ, x ≠ -2 ∧ (x^2 + 2*x - 8)/(x + 2) = 3*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_rational_equation_l4161_416182


namespace NUMINAMATH_CALUDE_smallest_number_of_blocks_for_wall_l4161_416135

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  height : ℕ
  length : ℕ

/-- Represents the dimensions of a wall -/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a wall with given conditions -/
def calculateBlocksNeeded (wall : WallDimensions) (block : BlockDimensions) (longBlock : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the smallest number of blocks needed for the wall -/
theorem smallest_number_of_blocks_for_wall 
  (wall : WallDimensions)
  (block : BlockDimensions)
  (longBlock : BlockDimensions)
  (h1 : wall.length = 120)
  (h2 : wall.height = 8)
  (h3 : block.height = 1)
  (h4 : block.length = 1)
  (h5 : longBlock.height = 1)
  (h6 : longBlock.length = 3)
  : calculateBlocksNeeded wall block longBlock = 324 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_of_blocks_for_wall_l4161_416135


namespace NUMINAMATH_CALUDE_infinitely_many_divisible_by_2018_l4161_416159

def a (n : ℕ) : ℕ := 2 * 10^(n+2) + 18

theorem infinitely_many_divisible_by_2018 :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, 2018 ∣ a n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_divisible_by_2018_l4161_416159


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_3780_l4161_416187

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_3780 :
  let factorization := prime_factorization 3780
  (factorization = [(2, 2), (3, 3), (5, 1), (7, 2)]) →
  count_perfect_square_factors 3780 = 8 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_3780_l4161_416187


namespace NUMINAMATH_CALUDE_monthly_fee_is_two_l4161_416114

/-- Represents the monthly phone bill structure -/
structure PhoneBill where
  monthlyFee : ℝ
  perMinuteRate : ℝ
  minutesUsed : ℕ
  totalBill : ℝ

/-- Proves that the monthly fee is $2 given the specified conditions -/
theorem monthly_fee_is_two (bill : PhoneBill) 
    (h1 : bill.totalBill = bill.monthlyFee + bill.perMinuteRate * bill.minutesUsed)
    (h2 : bill.perMinuteRate = 0.12)
    (h3 : bill.totalBill = 23.36)
    (h4 : bill.minutesUsed = 178) :
    bill.monthlyFee = 2 := by
  sorry


end NUMINAMATH_CALUDE_monthly_fee_is_two_l4161_416114


namespace NUMINAMATH_CALUDE_rectangle_coverage_l4161_416100

/-- An L-shaped figure made of 4 unit squares -/
structure LShape :=
  (size : Nat)
  (h_size : size = 4)

/-- Represents a rectangle with dimensions m × n -/
structure Rectangle (m n : Nat) :=
  (width : Nat)
  (height : Nat)
  (h_width : width = m)
  (h_height : height = n)
  (h_positive : m > 1 ∧ n > 1)

/-- Predicate to check if a number is a multiple of 8 -/
def IsMultipleOf8 (n : Nat) : Prop := ∃ k, n = 8 * k

/-- Predicate to check if a rectangle can be covered by L-shaped figures -/
def CanBeCovered (r : Rectangle m n) (l : LShape) : Prop :=
  ∃ (arrangement : Nat), True  -- We don't define the specific arrangement here

/-- The main theorem -/
theorem rectangle_coverage (m n : Nat) (r : Rectangle m n) (l : LShape) :
  (CanBeCovered r l) ↔ (IsMultipleOf8 (m * n)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_coverage_l4161_416100


namespace NUMINAMATH_CALUDE_cubic_function_derivative_l4161_416155

theorem cubic_function_derivative (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ (3 * a * x^2) + (6 * x)
  f' (-1) = 4 → a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_derivative_l4161_416155


namespace NUMINAMATH_CALUDE_convention_handshakes_theorem_l4161_416194

/-- The number of handshakes in a convention with multiple companies -/
def convention_handshakes (num_companies : ℕ) (representatives_per_company : ℕ) : ℕ :=
  let total_people := num_companies * representatives_per_company
  let handshakes_per_person := total_people - representatives_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a convention with 3 companies, each having 5 representatives,
    where each person shakes hands only once with every person except those
    from their own company, the total number of handshakes is 75. -/
theorem convention_handshakes_theorem :
  convention_handshakes 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_theorem_l4161_416194


namespace NUMINAMATH_CALUDE_lcm_48_180_l4161_416128

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l4161_416128


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4161_416167

-- Problem 1
theorem problem_1 : Real.sqrt 8 - (1/2)⁻¹ + 4 * Real.sin (30 * π / 180) = 2 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_2 : (2^2 - 9) / (2^2 + 6*2 + 9) / (1 - 2 / (2 + 3)) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4161_416167


namespace NUMINAMATH_CALUDE_second_offer_more_advantageous_l4161_416156

/-- Represents the total cost of four items -/
def S : ℕ := 1000

/-- Represents the minimum cost of any item -/
def X : ℕ := 99

/-- Represents the prices of four items -/
structure Prices where
  s₁ : ℕ
  s₂ : ℕ
  s₃ : ℕ
  s₄ : ℕ
  sum_eq_S : s₁ + s₂ + s₃ + s₄ = S
  ordered : s₁ ≥ s₂ ∧ s₂ ≥ s₃ ∧ s₃ ≥ s₄
  min_price : s₄ ≥ X

/-- The maximum N for which the second offer is more advantageous -/
def maxN : ℕ := 504

theorem second_offer_more_advantageous (prices : Prices) :
  ∀ N : ℕ, N ≤ maxN →
  (0.2 * prices.s₁ + 0.8 * S : ℚ) < (S - prices.s₄ : ℚ) ∧
  ¬∃ M : ℕ, M > maxN ∧ (0.2 * prices.s₁ + 0.8 * S : ℚ) < (S - prices.s₄ : ℚ) :=
sorry

end NUMINAMATH_CALUDE_second_offer_more_advantageous_l4161_416156


namespace NUMINAMATH_CALUDE_other_person_age_is_six_l4161_416190

-- Define Noah's current age
def noah_current_age : ℕ := 22 - 10

-- Define the relationship between Noah's age and the other person's age
def other_person_age : ℕ := noah_current_age / 2

-- Theorem to prove
theorem other_person_age_is_six : other_person_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_other_person_age_is_six_l4161_416190


namespace NUMINAMATH_CALUDE_sally_dozens_of_eggs_l4161_416102

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Sally bought -/
def total_eggs : ℕ := 48

/-- The number of dozens of eggs Sally bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem sally_dozens_of_eggs : dozens_bought = 4 := by
  sorry

end NUMINAMATH_CALUDE_sally_dozens_of_eggs_l4161_416102


namespace NUMINAMATH_CALUDE_probability_one_black_one_white_l4161_416132

def total_balls : ℕ := 5
def black_balls : ℕ := 3
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_one_black_one_white :
  (black_balls.choose 1 * white_balls.choose 1 : ℚ) / total_balls.choose drawn_balls = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_black_one_white_l4161_416132


namespace NUMINAMATH_CALUDE_sunflower_seed_contest_l4161_416158

theorem sunflower_seed_contest (total_seeds : ℕ) (first_player : ℕ) (second_player : ℕ) 
  (h1 : total_seeds = 214)
  (h2 : first_player = 78)
  (h3 : second_player = 53)
  (h4 : total_seeds = first_player + second_player + (total_seeds - first_player - second_player))
  (h5 : total_seeds - first_player - second_player > second_player) :
  (total_seeds - first_player - second_player) - second_player = 30 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_seed_contest_l4161_416158


namespace NUMINAMATH_CALUDE_work_completion_time_l4161_416168

/-- Given that A can do a piece of work in 12 days and B is 20% more efficient than A,
    prove that B will complete the same work in 10 days. -/
theorem work_completion_time (work : ℝ) (a_time b_time : ℝ) : 
  work > 0 → 
  a_time = 12 → 
  b_time = work / ((work / a_time) * 1.2) → 
  b_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l4161_416168


namespace NUMINAMATH_CALUDE_quadratic_inequality_product_l4161_416175

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3), prove that ab = 6 -/
theorem quadratic_inequality_product (a b : ℝ) : 
  (∀ x, -1 < x ∧ x < 1/3 ↔ a * x^2 + b * x + 1 > 0) →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_product_l4161_416175


namespace NUMINAMATH_CALUDE_power_of_three_mod_nineteen_l4161_416143

theorem power_of_three_mod_nineteen : 3^17 % 19 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_nineteen_l4161_416143


namespace NUMINAMATH_CALUDE_garrett_roses_count_l4161_416116

/-- Mrs. Santiago's red roses -/
def santiago_roses : ℕ := 58

/-- The difference between Mrs. Santiago's and Mrs. Garrett's red roses -/
def rose_difference : ℕ := 34

/-- Mrs. Garrett's red roses -/
def garrett_roses : ℕ := santiago_roses - rose_difference

theorem garrett_roses_count : garrett_roses = 24 := by
  sorry

end NUMINAMATH_CALUDE_garrett_roses_count_l4161_416116


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4161_416108

-- Define the inequality
def inequality (x : ℝ) : Prop := abs ((2 - x) / x) > (x - 2) / x

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4161_416108


namespace NUMINAMATH_CALUDE_second_quadrant_fraction_negative_l4161_416111

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a point being in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem stating that for a point in the second quadrant, a/b < 0 -/
theorem second_quadrant_fraction_negative (p : Point) :
  is_in_second_quadrant p → p.x / p.y < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_second_quadrant_fraction_negative_l4161_416111


namespace NUMINAMATH_CALUDE_probability_no_red_square_l4161_416131

/-- Represents a coloring of a 4-by-4 grid -/
def Coloring := Fin 4 → Fin 4 → Bool

/-- Returns true if the coloring contains a 2-by-2 square of red squares -/
def has_red_square (c : Coloring) : Bool :=
  ∃ i j, i < 3 ∧ j < 3 ∧ 
    c i j ∧ c i (j+1) ∧ c (i+1) j ∧ c (i+1) (j+1)

/-- The probability of a square being red -/
def p_red : ℚ := 1/2

/-- The total number of possible colorings -/
def total_colorings : ℕ := 2^16

/-- The number of colorings without a 2-by-2 red square -/
def valid_colorings : ℕ := 40512

theorem probability_no_red_square :
  (valid_colorings : ℚ) / total_colorings = 315 / 512 :=
sorry

end NUMINAMATH_CALUDE_probability_no_red_square_l4161_416131


namespace NUMINAMATH_CALUDE_product_equals_four_l4161_416107

theorem product_equals_four (a b c : ℝ) 
  (h : ∀ (a b c : ℝ), a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) : 
  6 * 15 * 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_product_equals_four_l4161_416107


namespace NUMINAMATH_CALUDE_parallelogram_area_l4161_416181

theorem parallelogram_area (base height : ℝ) (h1 : base = 60) (h2 : height = 16) :
  base * height = 960 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l4161_416181


namespace NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l4161_416151

/-- For real numbers a, b, c forming an arithmetic progression,
    3(a² + b² + c²) = 6(a-b)² + (a+b+c)² -/
theorem arithmetic_progression_square_sum (a b c : ℝ) 
  (h : a + c = 2 * b) : 
  3 * (a^2 + b^2 + c^2) = 6 * (a - b)^2 + (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_square_sum_l4161_416151


namespace NUMINAMATH_CALUDE_taobao_villages_growth_l4161_416192

/-- 
Given an arithmetic sequence with first term 1311 and common difference 1000,
prove that the 8th term of this sequence is 8311.
-/
theorem taobao_villages_growth (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 1311 → d = 1000 → n = 8 →
  a₁ + (n - 1) * d = 8311 :=
by sorry

end NUMINAMATH_CALUDE_taobao_villages_growth_l4161_416192


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l4161_416185

theorem quadratic_roots_property (d e : ℝ) : 
  (2 * d^2 + 3 * d - 5 = 0) → 
  (2 * e^2 + 3 * e - 5 = 0) → 
  (d - 1) * (e - 1) = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l4161_416185


namespace NUMINAMATH_CALUDE_min_cubes_for_valid_config_l4161_416113

/-- Represents a cube with two opposite sides having protruding snaps and four sides with receptacle holes. -/
structure SpecialCube where
  snaps : Fin 2 → Bool
  holes : Fin 4 → Bool

/-- A configuration of special cubes. -/
def CubeConfiguration := List SpecialCube

/-- Checks if a configuration has no visible protruding snaps and only shows receptacle holes on visible surfaces. -/
def isValidConfiguration (config : CubeConfiguration) : Bool :=
  sorry

/-- The theorem stating that 6 is the minimum number of cubes required for a valid configuration. -/
theorem min_cubes_for_valid_config :
  ∃ (config : CubeConfiguration),
    config.length = 6 ∧ isValidConfiguration config ∧
    ∀ (smallerConfig : CubeConfiguration),
      smallerConfig.length < 6 → ¬isValidConfiguration smallerConfig :=
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_valid_config_l4161_416113


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4161_416109

theorem inequality_equivalence (θ : ℝ) (x : ℝ) :
  (|x + Real.cos θ ^ 2| ≤ Real.sin θ ^ 2) ↔ (-1 ≤ x ∧ x ≤ -Real.cos (2 * θ)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4161_416109


namespace NUMINAMATH_CALUDE_reggie_layups_l4161_416180

/-- Represents the score of a player in the basketball shooting contest -/
structure Score where
  layups : ℕ
  freeThrows : ℕ
  longShots : ℕ

/-- Calculates the total points for a given score -/
def totalPoints (s : Score) : ℕ :=
  s.layups + 2 * s.freeThrows + 3 * s.longShots

theorem reggie_layups : 
  ∀ (reggie_score : Score) (brother_score : Score),
    reggie_score.freeThrows = 2 →
    reggie_score.longShots = 1 →
    brother_score.layups = 0 →
    brother_score.freeThrows = 0 →
    brother_score.longShots = 4 →
    totalPoints reggie_score + 2 = totalPoints brother_score →
    reggie_score.layups = 3 := by
  sorry

#check reggie_layups

end NUMINAMATH_CALUDE_reggie_layups_l4161_416180


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l4161_416112

theorem algebraic_expression_value (x : ℝ) : 
  4 * x^2 - 2 * x + 3 = 11 → 2 * x^2 - x - 7 = -3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l4161_416112


namespace NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l4161_416122

theorem power_two_greater_than_sum_of_powers (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 2) (h2 : |x| < 1) : 
  (2 : ℝ) ^ n > (1 - x) ^ n + (1 + x) ^ n := by
  sorry

end NUMINAMATH_CALUDE_power_two_greater_than_sum_of_powers_l4161_416122


namespace NUMINAMATH_CALUDE_triangle_area_l4161_416189

/-- The area of a triangle with base 12 cm and height 7 cm is 42 square centimeters. -/
theorem triangle_area : 
  let base : ℝ := 12
  let height : ℝ := 7
  (1 / 2 : ℝ) * base * height = 42 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l4161_416189


namespace NUMINAMATH_CALUDE_mitchell_antonio_pencil_difference_l4161_416177

theorem mitchell_antonio_pencil_difference :
  ∀ (mitchell_pencils antonio_pencils : ℕ),
    mitchell_pencils = 30 →
    mitchell_pencils + antonio_pencils = 54 →
    mitchell_pencils > antonio_pencils →
    mitchell_pencils - antonio_pencils = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_mitchell_antonio_pencil_difference_l4161_416177


namespace NUMINAMATH_CALUDE_unusual_arithmetic_l4161_416146

/-- In a country with unusual arithmetic, given that 1/3 of 4 equals 6,
    if 1/6 of a number is 15, then that number is 405. -/
theorem unusual_arithmetic (country_multiplier : ℚ) : 
  (1/3 : ℚ) * 4 * country_multiplier = 6 →
  ∃ (x : ℚ), (1/6 : ℚ) * x * country_multiplier = 15 ∧ x * country_multiplier = 405 :=
by
  sorry


end NUMINAMATH_CALUDE_unusual_arithmetic_l4161_416146


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l4161_416165

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (donation : ℕ) (h_total_tickets : total_tickets = 200) 
  (h_total_revenue : total_revenue = 3200) (h_donation : donation = 200) :
  ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
    full_price + half_price = total_tickets ∧
    full_price * price + half_price * (price / 2) + donation = total_revenue ∧
    full_price * price = 1000 := by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l4161_416165


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l4161_416118

theorem closest_integer_to_cube_root_250 :
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (250 : ℝ)^(1/3)| ≤ |m - (250 : ℝ)^(1/3)| ∧ n = 6 :=
sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_250_l4161_416118


namespace NUMINAMATH_CALUDE_john_repair_results_l4161_416119

/-- Represents the repair job details for John --/
structure RepairJob where
  totalCars : ℕ
  standardRepairCars : ℕ
  standardRepairTime : ℕ
  longerRepairPercent : ℚ
  hourlyRate : ℚ

/-- Calculates the total repair time and money earned for a given repair job --/
def calculateRepairResults (job : RepairJob) : ℚ × ℚ :=
  let standardTime := job.standardRepairCars * job.standardRepairTime
  let longerRepairTime := job.standardRepairTime * (1 + job.longerRepairPercent)
  let longerRepairCars := job.totalCars - job.standardRepairCars
  let longerTime := longerRepairCars * longerRepairTime
  let totalMinutes := standardTime + longerTime
  let totalHours := totalMinutes / 60
  let moneyEarned := totalHours * job.hourlyRate
  (totalHours, moneyEarned)

/-- Theorem stating that for John's specific repair job, the total repair time is 11 hours and he earns $330 --/
theorem john_repair_results :
  let job : RepairJob := {
    totalCars := 10,
    standardRepairCars := 6,
    standardRepairTime := 50,
    longerRepairPercent := 4/5,
    hourlyRate := 30
  }
  calculateRepairResults job = (11, 330) := by sorry

end NUMINAMATH_CALUDE_john_repair_results_l4161_416119


namespace NUMINAMATH_CALUDE_opposite_gender_selections_l4161_416103

def society_size : ℕ := 24
def male_count : ℕ := 14
def female_count : ℕ := 10

theorem opposite_gender_selections :
  (male_count * female_count) + (female_count * male_count) = 280 := by
  sorry

end NUMINAMATH_CALUDE_opposite_gender_selections_l4161_416103


namespace NUMINAMATH_CALUDE_sally_score_l4161_416179

/-- Calculates the total score in a math competition given the number of correct, incorrect, and unanswered questions. -/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℚ :=
  (correct : ℚ) - 0.25 * (incorrect : ℚ)

/-- Theorem: Sally's total score in the math competition is 15 points. -/
theorem sally_score :
  let correct : ℕ := 17
  let incorrect : ℕ := 8
  let unanswered : ℕ := 5
  calculate_score correct incorrect unanswered = 15 := by
  sorry

end NUMINAMATH_CALUDE_sally_score_l4161_416179


namespace NUMINAMATH_CALUDE_x_plus_y_value_l4161_416136

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : |y| + x - y = 12) : 
  x + y = 18/5 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l4161_416136


namespace NUMINAMATH_CALUDE_max_candy_count_l4161_416130

/-- The number of candy pieces Frankie got -/
def frankies_candy : ℕ := 74

/-- The additional candy pieces Max got compared to Frankie -/
def extra_candy : ℕ := 18

/-- The number of candy pieces Max got -/
def maxs_candy : ℕ := frankies_candy + extra_candy

theorem max_candy_count : maxs_candy = 92 := by
  sorry

end NUMINAMATH_CALUDE_max_candy_count_l4161_416130


namespace NUMINAMATH_CALUDE_hot_dog_cost_l4161_416197

/-- The cost of a hot dog given the conditions of the concession stand problem -/
theorem hot_dog_cost (soda_cost : ℝ) (total_revenue : ℝ) (total_items : ℕ) (hot_dogs_sold : ℕ) :
  soda_cost = 0.50 →
  total_revenue = 78.50 →
  total_items = 87 →
  hot_dogs_sold = 35 →
  ∃ (hot_dog_cost : ℝ), 
    hot_dog_cost * hot_dogs_sold + soda_cost * (total_items - hot_dogs_sold) = total_revenue ∧
    hot_dog_cost = 1.50 := by
  sorry


end NUMINAMATH_CALUDE_hot_dog_cost_l4161_416197


namespace NUMINAMATH_CALUDE_estimated_probability_is_two_ninths_l4161_416173

/-- Represents the outcome of a single trial -/
inductive Outcome
| StopOnThird
| Other

/-- Represents the result of a random simulation -/
structure SimulationResult :=
  (trials : Nat)
  (stopsOnThird : Nat)

/-- Calculates the estimated probability -/
def estimateProbability (result : SimulationResult) : Rat :=
  result.stopsOnThird / result.trials

theorem estimated_probability_is_two_ninths 
  (result : SimulationResult)
  (h1 : result.trials = 18)
  (h2 : result.stopsOnThird = 4) :
  estimateProbability result = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_estimated_probability_is_two_ninths_l4161_416173


namespace NUMINAMATH_CALUDE_daves_ice_cubes_l4161_416174

theorem daves_ice_cubes (original : ℕ) (made : ℕ) (total : ℕ) : 
  made = 7 → total = 9 → original + made = total → original = 2 := by
sorry

end NUMINAMATH_CALUDE_daves_ice_cubes_l4161_416174


namespace NUMINAMATH_CALUDE_school_seminar_cost_l4161_416195

/-- Calculates the total amount spent by a school for a teacher seminar with discounts and food allowance. -/
def total_seminar_cost (regular_fee : ℝ) (discount_percent : ℝ) (num_teachers : ℕ) (food_allowance : ℝ) : ℝ :=
  let discounted_fee := regular_fee * (1 - discount_percent)
  let total_seminar_fees := discounted_fee * num_teachers
  let total_food_allowance := food_allowance * num_teachers
  total_seminar_fees + total_food_allowance

/-- Theorem stating the total cost for the school's teacher seminar -/
theorem school_seminar_cost :
  total_seminar_cost 150 0.05 10 10 = 1525 :=
by sorry

end NUMINAMATH_CALUDE_school_seminar_cost_l4161_416195


namespace NUMINAMATH_CALUDE_equal_quantities_solution_l4161_416162

theorem equal_quantities_solution (x y : ℝ) (h : y ≠ 0) :
  (((x + y = x - y ∧ x + y = x * y) ∨
    (x + y = x - y ∧ x + y = x / y) ∨
    (x + y = x * y ∧ x + y = x / y) ∨
    (x - y = x * y ∧ x - y = x / y)) →
   ((x = 1/2 ∧ y = -1) ∨ (x = -1/2 ∧ y = -1))) :=
by sorry

end NUMINAMATH_CALUDE_equal_quantities_solution_l4161_416162


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l4161_416198

theorem quadratic_real_solutions (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l4161_416198


namespace NUMINAMATH_CALUDE_dime_probability_l4161_416134

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
  | Coin.Quarter => 1200
  | Coin.Dime => 800
  | Coin.Penny => 500

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Quarter + coin_count Coin.Dime + coin_count Coin.Penny

/-- Theorem: The probability of randomly choosing a dime from the jar is 40/314 -/
theorem dime_probability : 
  (coin_count Coin.Dime : ℚ) / total_coins = 40 / 314 := by sorry

end NUMINAMATH_CALUDE_dime_probability_l4161_416134


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l4161_416148

theorem largest_prime_factor_of_expression :
  let n : ℤ := 25^2 + 35^3 - 10^5
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n.natAbs ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ n.natAbs → q ≤ p ∧ p = 113 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l4161_416148


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_l4161_416105

theorem alcohol_water_ratio (alcohol_volume water_volume : ℚ) : 
  alcohol_volume = 2/7 → water_volume = 3/7 → 
  alcohol_volume / water_volume = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_l4161_416105


namespace NUMINAMATH_CALUDE_ellipse_equation_l4161_416164

/-- An ellipse with given properties has the equation x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  let e := Real.sqrt 3 / 3
  let c := a * e
  let perimeter := 4 * Real.sqrt 3
  (c^2 = a^2 - b^2) →
  (perimeter = 2 * a + 2 * a) →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/3 + y^2/2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4161_416164


namespace NUMINAMATH_CALUDE_simplify_expression_l4161_416137

theorem simplify_expression (x : ℝ) : (2*x)^5 - (5*x)*(x^4) = 27*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4161_416137


namespace NUMINAMATH_CALUDE_product_of_complements_lower_bound_l4161_416142

theorem product_of_complements_lower_bound 
  (x₁ x₂ x₃ : ℝ) 
  (h₁ : 0 ≤ x₁) (h₂ : 0 ≤ x₂) (h₃ : 0 ≤ x₃) 
  (h₄ : x₁ + x₂ + x₃ ≤ 1/2) : 
  (1 - x₁) * (1 - x₂) * (1 - x₃) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_complements_lower_bound_l4161_416142


namespace NUMINAMATH_CALUDE_value_of_two_minus_c_l4161_416196

theorem value_of_two_minus_c (c d : ℝ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 3 + d = 8 + c) : 
  2 - c = -1 := by
  sorry

end NUMINAMATH_CALUDE_value_of_two_minus_c_l4161_416196


namespace NUMINAMATH_CALUDE_geometric_progression_m_existence_l4161_416154

theorem geometric_progression_m_existence (m : ℂ) : 
  ∃ r : ℂ, r ≠ 0 ∧ 
    r ≠ r^2 ∧ r ≠ r^3 ∧ r^2 ≠ r^3 ∧
    r / (1 - r^2) = m ∧ 
    r^2 / (1 - r^3) = m ∧ 
    r^3 / (1 - r) = m := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_m_existence_l4161_416154


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4161_416140

theorem rectangle_perimeter (L W : ℝ) (h : L * W - (L - 4) * (W - 4) = 168) :
  2 * (L + W) = 92 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4161_416140


namespace NUMINAMATH_CALUDE_flower_pot_profit_equation_l4161_416147

/-- Represents a flower pot system with variable number of plants and profit. -/
structure FlowerPot where
  initial_plants : ℕ
  initial_profit_per_plant : ℝ
  profit_decrease_per_plant : ℝ

/-- Calculates the total profit for a given number of additional plants. -/
def total_profit (fp : FlowerPot) (additional_plants : ℝ) : ℝ :=
  (fp.initial_plants + additional_plants) * 
  (fp.initial_profit_per_plant - additional_plants * fp.profit_decrease_per_plant)

/-- Theorem stating that the equation (x+3)(10-x)=40 correctly represents
    the total profit of 40 yuan for the given flower pot system. -/
theorem flower_pot_profit_equation (x : ℝ) : 
  let fp : FlowerPot := ⟨3, 10, 1⟩
  total_profit fp x = 40 ↔ (x + 3) * (10 - x) = 40 := by
  sorry

end NUMINAMATH_CALUDE_flower_pot_profit_equation_l4161_416147
