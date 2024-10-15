import Mathlib

namespace NUMINAMATH_CALUDE_additional_friends_average_weight_l3791_379164

theorem additional_friends_average_weight
  (initial_count : ℕ)
  (additional_count : ℕ)
  (average_increase : ℝ)
  (final_average : ℝ)
  (h1 : initial_count = 50)
  (h2 : additional_count = 40)
  (h3 : average_increase = 12)
  (h4 : final_average = 46) :
  let total_count := initial_count + additional_count
  let initial_average := final_average - average_increase
  let initial_total_weight := initial_count * initial_average
  let final_total_weight := total_count * final_average
  let additional_total_weight := final_total_weight - initial_total_weight
  let additional_average := additional_total_weight / additional_count
  additional_average = 61 := by
sorry

end NUMINAMATH_CALUDE_additional_friends_average_weight_l3791_379164


namespace NUMINAMATH_CALUDE_oil_leak_during_repairs_l3791_379150

theorem oil_leak_during_repairs 
  (total_leaked : ℕ) 
  (leaked_before_repairs : ℕ) 
  (h1 : total_leaked = 6206)
  (h2 : leaked_before_repairs = 2475) :
  total_leaked - leaked_before_repairs = 3731 := by
sorry

end NUMINAMATH_CALUDE_oil_leak_during_repairs_l3791_379150


namespace NUMINAMATH_CALUDE_square_root_decimal_shift_l3791_379116

theorem square_root_decimal_shift (x : ℝ) (hx : x > 0) :
  ∃ y : ℝ, y > 0 ∧ y^2 = x ∧ (100 * x).sqrt = 10 * y :=
by sorry

end NUMINAMATH_CALUDE_square_root_decimal_shift_l3791_379116


namespace NUMINAMATH_CALUDE_distance_to_gym_l3791_379172

theorem distance_to_gym (home_to_grocery : ℝ) (grocery_to_gym_speed : ℝ) 
  (time_difference : ℝ) :
  home_to_grocery = 200 →
  grocery_to_gym_speed = 2 →
  time_difference = 50 →
  grocery_to_gym_speed = 2 * (home_to_grocery / 200) →
  (200 / (home_to_grocery / 200)) - (200 / grocery_to_gym_speed) = time_difference →
  200 / grocery_to_gym_speed = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_gym_l3791_379172


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l3791_379169

/-- The slope of line l₁ -/
def m₁ : ℚ := 2/3

/-- The y-intercept of line l₁ in terms of a -/
def b₁ (a : ℚ) : ℚ := (1 - a) / 3

/-- The slope of line l₂ -/
def m₂ : ℚ := -1/2

/-- The y-intercept of line l₂ in terms of a -/
def b₂ (a : ℚ) : ℚ := a

/-- The x-coordinate of the intersection point of l₁ and l₂ -/
def x_intersect (a : ℚ) : ℚ := (b₂ a - b₁ a) / (m₁ - m₂)

/-- The y-coordinate of the intersection point of l₁ and l₂ -/
def y_intersect (a : ℚ) : ℚ := m₁ * x_intersect a + b₁ a

/-- The theorem stating the condition for the intersection point to be in the second quadrant -/
theorem intersection_in_second_quadrant (a : ℚ) :
  (x_intersect a > 0 ∧ y_intersect a > 0) ↔ a > 1/4 := by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l3791_379169


namespace NUMINAMATH_CALUDE_vector_operation_l3791_379142

/-- Given two 2D vectors a and b, prove that 3a - b equals (4, 2) -/
theorem vector_operation (a b : Fin 2 → ℝ) 
  (ha : a = ![1, 1]) 
  (hb : b = ![-1, 1]) : 
  (3 • a) - b = ![4, 2] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3791_379142


namespace NUMINAMATH_CALUDE_candy_problem_l3791_379121

theorem candy_problem (x : ℝ) : 
  let day1_remainder := x / 2 - 3
  let day2_remainder := day1_remainder * 3/4 - 5
  let day3_remainder := day2_remainder * 4/5
  day3_remainder = 9 → x = 136 := by sorry

end NUMINAMATH_CALUDE_candy_problem_l3791_379121


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3791_379194

theorem quadratic_equation_solution :
  {x : ℝ | x^2 = -2*x} = {0, -2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3791_379194


namespace NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l3791_379173

theorem least_positive_integer_with_congruences : ∃ b : ℕ+, 
  (b : ℤ) ≡ 2 [ZMOD 3] ∧ 
  (b : ℤ) ≡ 3 [ZMOD 4] ∧ 
  (b : ℤ) ≡ 4 [ZMOD 5] ∧ 
  (b : ℤ) ≡ 6 [ZMOD 7] ∧ 
  ∀ c : ℕ+, 
    ((c : ℤ) ≡ 2 [ZMOD 3] ∧ 
     (c : ℤ) ≡ 3 [ZMOD 4] ∧ 
     (c : ℤ) ≡ 4 [ZMOD 5] ∧ 
     (c : ℤ) ≡ 6 [ZMOD 7]) → 
    b ≤ c :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_congruences_l3791_379173


namespace NUMINAMATH_CALUDE_remainder_of_198_digits_mod_9_l3791_379103

/-- Represents the sequence of digits formed by concatenating consecutive natural numbers -/
def consecutiveDigitSequence (n : ℕ) : List ℕ :=
  sorry

/-- Computes the sum of digits in the sequence up to the nth digit -/
def sumOfDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that the sum of the first 198 digits in the sequence,
    when divided by 9, has a remainder of 6 -/
theorem remainder_of_198_digits_mod_9 :
  sumOfDigits 198 % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_198_digits_mod_9_l3791_379103


namespace NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3791_379190

-- Define the repeating decimal 0.overline{3}
def repeating_third : ℚ := 1/3

-- State the theorem
theorem divide_eight_by_repeating_third : 8 / repeating_third = 24 := by
  sorry

end NUMINAMATH_CALUDE_divide_eight_by_repeating_third_l3791_379190


namespace NUMINAMATH_CALUDE_first_knife_cost_is_five_l3791_379176

/-- The cost structure for knife sharpening -/
structure KnifeSharpening where
  first_knife_cost : ℝ
  next_three_cost : ℝ
  remaining_cost : ℝ
  total_knives : ℕ
  total_cost : ℝ

/-- The theorem stating the cost of sharpening the first knife -/
theorem first_knife_cost_is_five (ks : KnifeSharpening)
  (h1 : ks.next_three_cost = 4)
  (h2 : ks.remaining_cost = 3)
  (h3 : ks.total_knives = 9)
  (h4 : ks.total_cost = 32)
  (h5 : ks.total_cost = ks.first_knife_cost + 3 * ks.next_three_cost + 5 * ks.remaining_cost) :
  ks.first_knife_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_knife_cost_is_five_l3791_379176


namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l3791_379146

theorem quadratic_roots_max_value (a b u v : ℝ) : 
  (∀ x, x^2 - a*x + b = 0 ↔ x = u ∨ x = v) →
  (u + v = u^2 + v^2) →
  (u + v = u^4 + v^4) →
  (u + v = u^18 + v^18) →
  (∃ (M : ℝ), ∀ (a' b' u' v' : ℝ), 
    (∀ x, x^2 - a'*x + b' = 0 ↔ x = u' ∨ x = v') →
    (u' + v' = u'^2 + v'^2) →
    (u' + v' = u'^4 + v'^4) →
    (u' + v' = u'^18 + v'^18) →
    1/u'^20 + 1/v'^20 ≤ M) →
  1/u^20 + 1/v^20 = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l3791_379146


namespace NUMINAMATH_CALUDE_letter_digit_problem_l3791_379177

/-- Represents a mapping from letters to digits -/
def LetterDigitMap := Char → Nat

/-- Checks if a LetterDigitMap is valid according to the problem conditions -/
def is_valid_map (f : LetterDigitMap) : Prop :=
  (f 'E' ≠ f 'H') ∧ (f 'E' ≠ f 'M') ∧ (f 'E' ≠ f 'O') ∧ (f 'E' ≠ f 'P') ∧
  (f 'H' ≠ f 'M') ∧ (f 'H' ≠ f 'O') ∧ (f 'H' ≠ f 'P') ∧
  (f 'M' ≠ f 'O') ∧ (f 'M' ≠ f 'P') ∧
  (f 'O' ≠ f 'P') ∧
  (∀ c, c ∈ ['E', 'H', 'M', 'O', 'P'] → f c ∈ [1, 2, 3, 4, 6, 8, 9])

theorem letter_digit_problem (f : LetterDigitMap) 
  (h1 : is_valid_map f)
  (h2 : f 'E' * f 'H' = f 'M' * f 'O' * f 'P' * f 'O' * 3)
  (h3 : f 'E' + f 'H' = f 'M' + f 'O' + f 'P' + f 'O' + 3) :
  f 'E' * f 'H' + f 'M' * f 'O' * f 'P' * f 'O' * 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_letter_digit_problem_l3791_379177


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l3791_379101

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l3791_379101


namespace NUMINAMATH_CALUDE_unique_triple_gcd_sum_square_l3791_379182

theorem unique_triple_gcd_sum_square : 
  ∃! (m n l : ℕ), 
    m + n = (Nat.gcd m n)^2 ∧
    m + l = (Nat.gcd m l)^2 ∧
    n + l = (Nat.gcd n l)^2 ∧
    m = 2 ∧ n = 2 ∧ l = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_gcd_sum_square_l3791_379182


namespace NUMINAMATH_CALUDE_xinyu_taxi_fare_10km_l3791_379110

/-- Calculates the taxi fare in Xinyu city -/
def taxi_fare (distance : ℝ) : ℝ :=
  let base_fare := 5
  let mid_rate := 1.6
  let long_rate := 2.4
  let mid_distance := 6
  let long_distance := 2
  base_fare + mid_rate * mid_distance + long_rate * long_distance

/-- The total taxi fare for a 10 km journey in Xinyu city is 19.4 yuan -/
theorem xinyu_taxi_fare_10km : taxi_fare 10 = 19.4 := by
  sorry

end NUMINAMATH_CALUDE_xinyu_taxi_fare_10km_l3791_379110


namespace NUMINAMATH_CALUDE_solve_equation_l3791_379124

theorem solve_equation (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3791_379124


namespace NUMINAMATH_CALUDE_dog_to_rabbit_age_ratio_l3791_379129

/- Define the ages of the animals -/
def cat_age : ℕ := 8
def dog_age : ℕ := 12

/- Define the rabbit's age as half of the cat's age -/
def rabbit_age : ℕ := cat_age / 2

/- Define the ratio of the dog's age to the rabbit's age -/
def age_ratio : ℚ := dog_age / rabbit_age

/- Theorem statement -/
theorem dog_to_rabbit_age_ratio :
  age_ratio = 3 :=
sorry

end NUMINAMATH_CALUDE_dog_to_rabbit_age_ratio_l3791_379129


namespace NUMINAMATH_CALUDE_quadruplet_solution_l3791_379199

theorem quadruplet_solution (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_product : a * b * c * d = 1)
  (h_eq1 : a^2012 + 2012 * b = 2012 * c + d^2012)
  (h_eq2 : 2012 * a + b^2012 = c^2012 + 2012 * d) :
  ∃ t : ℝ, t > 0 ∧ a = t ∧ b = 1/t ∧ c = 1/t ∧ d = t :=
sorry

end NUMINAMATH_CALUDE_quadruplet_solution_l3791_379199


namespace NUMINAMATH_CALUDE_M_remainder_1000_l3791_379128

/-- The greatest integer multiple of 9 with no two digits being the same -/
def M : ℕ :=
  sorry

/-- M has no repeated digits -/
axiom M_distinct_digits : ∀ d₁ d₂, d₁ ≠ d₂ → (M / 10^d₁ % 10) ≠ (M / 10^d₂ % 10)

/-- M is divisible by 9 -/
axiom M_div_by_9 : M % 9 = 0

/-- M is the greatest such number -/
axiom M_greatest : ∀ n : ℕ, n % 9 = 0 → (∀ d₁ d₂, d₁ ≠ d₂ → (n / 10^d₁ % 10) ≠ (n / 10^d₂ % 10)) → n ≤ M

theorem M_remainder_1000 : M % 1000 = 810 := by
  sorry

end NUMINAMATH_CALUDE_M_remainder_1000_l3791_379128


namespace NUMINAMATH_CALUDE_total_working_days_l3791_379112

/-- Commute options for a person over a period of working days. -/
structure CommuteData where
  /-- Number of days driving car in the morning and riding bicycle in the afternoon -/
  car_morning_bike_afternoon : ℕ
  /-- Number of days riding bicycle in the morning and driving car in the afternoon -/
  bike_morning_car_afternoon : ℕ
  /-- Number of days using only bicycle both morning and afternoon -/
  bike_only : ℕ

/-- Theorem stating the total number of working days based on given commute data. -/
theorem total_working_days (data : CommuteData) : 
  data.car_morning_bike_afternoon + data.bike_morning_car_afternoon + data.bike_only = 23 :=
  by
  have morning_car : data.car_morning_bike_afternoon + data.bike_only = 12 := by sorry
  have afternoon_bike : data.bike_morning_car_afternoon + data.bike_only = 20 := by sorry
  have total_car : data.car_morning_bike_afternoon + data.bike_morning_car_afternoon = 14 := by sorry
  sorry

#check total_working_days

end NUMINAMATH_CALUDE_total_working_days_l3791_379112


namespace NUMINAMATH_CALUDE_compound_interest_problem_l3791_379180

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Theorem statement --/
theorem compound_interest_problem :
  let principal : ℝ := 500
  let rate : ℝ := 0.05
  let time : ℕ := 5
  let interest : ℝ := compound_interest principal rate time
  ∃ ε > 0, |interest - 138.14| < ε :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l3791_379180


namespace NUMINAMATH_CALUDE_teresa_age_at_birth_l3791_379108

/-- Calculates Teresa's age when Michiko was born given current ages and Morio's age at Michiko's birth -/
def teresaAgeAtBirth (teresaCurrentAge marioCurrentAge marioAgeAtBirth : ℕ) : ℕ :=
  marioAgeAtBirth - (marioCurrentAge - teresaCurrentAge)

theorem teresa_age_at_birth :
  teresaAgeAtBirth 59 71 38 = 26 := by
  sorry

end NUMINAMATH_CALUDE_teresa_age_at_birth_l3791_379108


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3791_379167

theorem divisibility_by_five (a b c d e f g : ℕ) 
  (h1 : (a + b + c + d + e + f) % 5 = 0)
  (h2 : (a + b + c + d + e + g) % 5 = 0)
  (h3 : (a + b + c + d + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + d + e + f + g) % 5 = 0)
  (h6 : (a + c + d + e + f + g) % 5 = 0)
  (h7 : (b + c + d + e + f + g) % 5 = 0) :
  (a % 5 = 0) ∧ (b % 5 = 0) ∧ (c % 5 = 0) ∧ (d % 5 = 0) ∧ 
  (e % 5 = 0) ∧ (f % 5 = 0) ∧ (g % 5 = 0) := by
  sorry

#check divisibility_by_five

end NUMINAMATH_CALUDE_divisibility_by_five_l3791_379167


namespace NUMINAMATH_CALUDE_parallelogram_vector_subtraction_l3791_379136

-- Define a parallelogram ABCD
structure Parallelogram (V : Type*) [AddCommGroup V] :=
  (A B C D : V)
  (parallelogram_condition : A - B = D - C)

-- Define the theorem
theorem parallelogram_vector_subtraction 
  {V : Type*} [AddCommGroup V] (ABCD : Parallelogram V) :
  ABCD.A - ABCD.C - (ABCD.B - ABCD.C) = ABCD.D - ABCD.C :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_vector_subtraction_l3791_379136


namespace NUMINAMATH_CALUDE_angle_difference_range_l3791_379165

theorem angle_difference_range (α β : Real) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π) :
  ∃ (x : Real), -3*π/2 < x ∧ x < 0 ∧ ∀ (y : Real), (-3*π/2 < y ∧ y < 0) → ∃ (α' β' : Real),
    -π/2 < α' ∧ α' < β' ∧ β' < π ∧ y = α' - β' :=
by sorry

end NUMINAMATH_CALUDE_angle_difference_range_l3791_379165


namespace NUMINAMATH_CALUDE_line_integral_equals_five_halves_l3791_379100

/-- Line segment from (0,0) to (4,3) -/
def L : Set (ℝ × ℝ) := {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (4*t, 3*t)}

/-- The function to be integrated -/
def f (p : ℝ × ℝ) : ℝ := p.1 - p.2

theorem line_integral_equals_five_halves :
  ∫ p in L, f p = 5/2 := by sorry

end NUMINAMATH_CALUDE_line_integral_equals_five_halves_l3791_379100


namespace NUMINAMATH_CALUDE_mary_jenny_red_marbles_equal_l3791_379139

/-- Represents the number of marbles collected by each person -/
structure MarbleCollection where
  red : ℕ
  blue : ℕ

/-- Given information about marble collections -/
def problem_setup (mary anie jenny : MarbleCollection) : Prop :=
  mary.blue = anie.blue / 2 ∧
  anie.red = mary.red + 20 ∧
  anie.blue = 2 * jenny.blue ∧
  jenny.red = 30 ∧
  jenny.blue = 25

/-- Theorem stating that Mary and Jenny collected the same number of red marbles -/
theorem mary_jenny_red_marbles_equal 
  (mary anie jenny : MarbleCollection) 
  (h : problem_setup mary anie jenny) : 
  mary.red = jenny.red := by
  sorry

end NUMINAMATH_CALUDE_mary_jenny_red_marbles_equal_l3791_379139


namespace NUMINAMATH_CALUDE_arrasta_um_min_moves_l3791_379192

/-- Represents the Arrasta Um game board -/
structure ArrastaUmBoard (n : ℕ) where
  size : n ≥ 2

/-- Represents a move in the Arrasta Um game -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Calculates the minimum number of moves required to complete the game -/
def minMoves (board : ArrastaUmBoard n) : ℕ :=
  6 * n - 8

/-- Theorem stating that the minimum number of moves to complete Arrasta Um on an n × n board is 6n - 8 -/
theorem arrasta_um_min_moves (n : ℕ) (board : ArrastaUmBoard n) :
  minMoves board = 6 * n - 8 :=
by sorry

end NUMINAMATH_CALUDE_arrasta_um_min_moves_l3791_379192


namespace NUMINAMATH_CALUDE_ascending_order_abc_l3791_379106

theorem ascending_order_abc (a b c : ℝ) : 
  a = (2 * Real.tan (70 * π / 180)) / (1 + Real.tan (70 * π / 180)^2) →
  b = Real.sqrt ((1 + Real.cos (109 * π / 180)) / 2) →
  c = (Real.sqrt 3 / 2) * Real.cos (81 * π / 180) + (1 / 2) * Real.sin (99 * π / 180) →
  b < c ∧ c < a := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l3791_379106


namespace NUMINAMATH_CALUDE_complete_square_formula_l3791_379161

theorem complete_square_formula (x : ℝ) : x^2 + 4*x + 4 = (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_formula_l3791_379161


namespace NUMINAMATH_CALUDE_special_collection_loans_l3791_379154

theorem special_collection_loans (initial_books final_books : ℕ) 
  (return_rate : ℚ) (loaned_books : ℕ) : 
  initial_books = 75 → 
  final_books = 57 → 
  return_rate = 7/10 →
  initial_books - final_books = (1 - return_rate) * loaned_books →
  loaned_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_collection_loans_l3791_379154


namespace NUMINAMATH_CALUDE_power_function_comparison_l3791_379193

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- State the theorem
theorem power_function_comparison
  (f : ℝ → ℝ)
  (h_power : isPowerFunction f)
  (h_condition : f 8 = 4) :
  f (Real.sqrt 2 / 2) > f (-Real.sqrt 3 / 3) := by
  sorry

end NUMINAMATH_CALUDE_power_function_comparison_l3791_379193


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3791_379198

def A : Set ℝ := {-1, 0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x > 0}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3791_379198


namespace NUMINAMATH_CALUDE_tangent_line_circle_product_l3791_379114

/-- Given a line ax + by - 3 = 0 tangent to the circle x^2 + y^2 + 4x - 1 = 0 at point P(-1, 2),
    the product ab equals 2. -/
theorem tangent_line_circle_product (a b : ℝ) : 
  (∀ x y, a * x + b * y - 3 = 0 → x^2 + y^2 + 4*x - 1 = 0 → (x + 1)^2 + (y - 2)^2 ≠ 0) →
  a * (-1) + b * 2 - 3 = 0 →
  (-1)^2 + 2^2 + 4*(-1) - 1 = 0 →
  a * b = 2 := by
sorry


end NUMINAMATH_CALUDE_tangent_line_circle_product_l3791_379114


namespace NUMINAMATH_CALUDE_parallelogram_area_l3791_379159

/-- The area of a parallelogram with base 48 cm and height 36 cm is 1728 square centimeters. -/
theorem parallelogram_area : 
  ∀ (base height area : ℝ), 
  base = 48 → 
  height = 36 → 
  area = base * height → 
  area = 1728 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l3791_379159


namespace NUMINAMATH_CALUDE_emily_widget_difference_l3791_379147

-- Define the variables
variable (t : ℝ)
variable (w : ℝ)

-- Define the conditions
def monday_production := w * t
def tuesday_production := (w + 6) * (t - 3)

-- Define the relationship between w and t
axiom w_eq_2t : w = 2 * t

-- State the theorem
theorem emily_widget_difference :
  monday_production - tuesday_production = 18 := by
  sorry

end NUMINAMATH_CALUDE_emily_widget_difference_l3791_379147


namespace NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l3791_379187

theorem tenth_power_sum_of_roots (u v : ℝ) : 
  u^2 - 2*u*Real.sqrt 3 + 1 = 0 ∧ 
  v^2 - 2*v*Real.sqrt 3 + 1 = 0 → 
  u^10 + v^10 = 93884 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_of_roots_l3791_379187


namespace NUMINAMATH_CALUDE_distinct_role_selection_l3791_379143

theorem distinct_role_selection (n : ℕ) (k : ℕ) : 
  n ≥ k → (n * (n - 1) * (n - 2) = (n.factorial) / ((n - k).factorial)) → 
  (8 * 7 * 6 = 336) :=
by sorry

end NUMINAMATH_CALUDE_distinct_role_selection_l3791_379143


namespace NUMINAMATH_CALUDE_divisible_by_three_after_rotation_l3791_379184

theorem divisible_by_three_after_rotation (n : ℕ) : 
  n = 857142 → 
  (n % 3 = 0) ∧ 
  ((285714 : ℕ) % 3 = 0) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_three_after_rotation_l3791_379184


namespace NUMINAMATH_CALUDE_product_of_sum_of_squares_l3791_379166

theorem product_of_sum_of_squares (p q r s : ℤ) :
  ∃ (x y : ℤ), (p^2 + q^2) * (r^2 + s^2) = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sum_of_squares_l3791_379166


namespace NUMINAMATH_CALUDE_percent_boys_in_class_l3791_379125

theorem percent_boys_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 49)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * total_students) * 100 = 42.86 := by
  sorry

end NUMINAMATH_CALUDE_percent_boys_in_class_l3791_379125


namespace NUMINAMATH_CALUDE_unique_n_with_prime_divisor_property_l3791_379137

theorem unique_n_with_prime_divisor_property : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (p : ℕ), Prime p ∧
    (∀ (q : ℕ), Prime q → q ∣ (n^2 + 3) → q ≤ p) ∧
    (∀ (q : ℕ), Prime q → q ∣ (n^4 + 6) → p ≤ q) ∧
    p ∣ (n^2 + 3) ∧ p ∣ (n^4 + 6)) ∧
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_n_with_prime_divisor_property_l3791_379137


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3791_379171

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 ∧ n ≥ 100 ∧ 17 ∣ n → n ≤ 986 := by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l3791_379171


namespace NUMINAMATH_CALUDE_efficiency_comparison_l3791_379189

theorem efficiency_comparison (p q : ℝ) (work : ℝ) 
  (h_p_time : p * 25 = work)
  (h_combined_time : (p + q) * 15 = work)
  (h_p_more_efficient : p > q) :
  (p - q) / q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_efficiency_comparison_l3791_379189


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3791_379162

/-- Given a parabola and a hyperbola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (x y : ℝ) :
  -- Parabola equation
  (∃ (x₀ y₀ : ℝ), y₀^2 = 8*x₀) →
  -- Hyperbola general form
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ x^2/a^2 - y^2/b^2 = 1) →
  -- Directrix of parabola (x = -2) passes through a focus of the hyperbola
  (∃ (x₁ y₁ : ℝ), x₁ = -2 ∧ x₁^2/a^2 - y₁^2/b^2 = 1) →
  -- Eccentricity of the hyperbola is 2
  (∃ (c : ℝ), c/a = 2 ∧ c^2 = a^2 + b^2) →
  -- Prove the equation of the hyperbola
  x^2 - y^2/3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3791_379162


namespace NUMINAMATH_CALUDE_no_unique_p_for_expected_value_l3791_379138

theorem no_unique_p_for_expected_value :
  ¬ ∃! p₀ : ℝ, 0 < p₀ ∧ p₀ < 1 ∧ 6 * p₀^2 - 5 * p₀^3 = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_no_unique_p_for_expected_value_l3791_379138


namespace NUMINAMATH_CALUDE_some_mythical_creatures_are_winged_animals_l3791_379156

-- Define the sets
variable (D : Type) -- Dragons
variable (M : Type) -- Mythical creatures
variable (W : Type) -- Winged animals

-- Define the relations
variable (isDragon : D → Prop)
variable (isMythical : M → Prop)
variable (isWinged : W → Prop)

-- Define the conditions
variable (h1 : ∀ d : D, ∃ m : M, isMythical m)
variable (h2 : ∃ w : W, ∃ d : D, isDragon d ∧ isWinged w)

-- Theorem to prove
theorem some_mythical_creatures_are_winged_animals :
  ∃ m : M, ∃ w : W, isMythical m ∧ isWinged w :=
sorry

end NUMINAMATH_CALUDE_some_mythical_creatures_are_winged_animals_l3791_379156


namespace NUMINAMATH_CALUDE_min_value_expression_l3791_379119

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 4*a + 2) * (b^2 + 4*b + 2) * (c^2 + 4*c + 2) / (a*b*c) ≥ 216 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3791_379119


namespace NUMINAMATH_CALUDE_complex_modulus_example_l3791_379144

theorem complex_modulus_example : Complex.abs (-5 - (8/3)*Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l3791_379144


namespace NUMINAMATH_CALUDE_wine_cost_proof_l3791_379133

/-- The current cost of a bottle of wine -/
def current_cost : ℝ := sorry

/-- The future cost of a bottle of wine after the price increase -/
def future_cost : ℝ := 1.25 * current_cost

/-- The increase in cost for five bottles -/
def total_increase : ℝ := 25

/-- The number of bottles -/
def num_bottles : ℕ := 5

theorem wine_cost_proof :
  (future_cost - current_cost) * num_bottles = total_increase ∧ current_cost = 20 := by sorry

end NUMINAMATH_CALUDE_wine_cost_proof_l3791_379133


namespace NUMINAMATH_CALUDE_arrangements_count_l3791_379191

/-- Represents the number of students -/
def total_students : ℕ := 6

/-- Represents the number of male students -/
def male_students : ℕ := 3

/-- Represents the number of female students -/
def female_students : ℕ := 3

/-- Represents that exactly two female students stand next to each other -/
def adjacent_female_students : ℕ := 2

/-- Calculates the number of arrangements satisfying the given conditions -/
def num_arrangements : ℕ := 288

/-- Theorem stating that the number of arrangements satisfying the given conditions is 288 -/
theorem arrangements_count :
  (total_students = male_students + female_students) →
  (male_students = 3) →
  (female_students = 3) →
  (adjacent_female_students = 2) →
  num_arrangements = 288 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l3791_379191


namespace NUMINAMATH_CALUDE_train_speed_l3791_379117

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 100)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 24) :
  (train_length + bridge_length) / crossing_time = 400 / 24 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l3791_379117


namespace NUMINAMATH_CALUDE_factory_earnings_l3791_379196

/-- Represents a factory with machines producing material --/
structure Factory where
  original_machines : ℕ
  original_hours : ℕ
  new_machines : ℕ
  new_hours : ℕ
  production_rate : ℕ
  price_per_kg : ℕ

/-- Calculates the daily earnings of the factory --/
def daily_earnings (f : Factory) : ℕ :=
  ((f.original_machines * f.original_hours + f.new_machines * f.new_hours) * f.production_rate) * f.price_per_kg

/-- Theorem stating that the factory's daily earnings are $8100 --/
theorem factory_earnings :
  ∃ (f : Factory), 
    f.original_machines = 3 ∧
    f.original_hours = 23 ∧
    f.new_machines = 1 ∧
    f.new_hours = 12 ∧
    f.production_rate = 2 ∧
    f.price_per_kg = 50 ∧
    daily_earnings f = 8100 := by
  sorry


end NUMINAMATH_CALUDE_factory_earnings_l3791_379196


namespace NUMINAMATH_CALUDE_f_divisible_by_13_l3791_379157

def f : ℕ → ℤ
  | 0 => 0
  | 1 => 0
  | (n + 2) => (4 * (n + 2) * f (n + 1) - 16 * (n + 1) * f n + n^2 * n^2) / n

theorem f_divisible_by_13 :
  13 ∣ f 1989 ∧ 13 ∣ f 1990 ∧ 13 ∣ f 1991 := by sorry

end NUMINAMATH_CALUDE_f_divisible_by_13_l3791_379157


namespace NUMINAMATH_CALUDE_coin_distribution_l3791_379120

def sum_of_integers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem coin_distribution (n : ℕ) (h : n = 20) :
  ∃ k : ℕ, sum_of_integers n = 3 * k ∧ ¬∃ m : ℕ, sum_of_integers n + 100 = 3 * m := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l3791_379120


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3791_379179

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i * i = -1) :
  (Complex.im ((1 - i^2023) / (a * i)) = 3) → a = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3791_379179


namespace NUMINAMATH_CALUDE_problem_solution_l3791_379107

/-- The problem setup and proof statements -/
theorem problem_solution :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (2, -2)
  let C : ℝ × ℝ := (4, 1)
  let D : ℝ × ℝ := (5, -4)
  let a : ℝ × ℝ := (1, -5)
  let b : ℝ × ℝ := (2, 3)
  let k : ℝ := -1/3
  -- Part 1
  (B.1 - A.1, B.2 - A.2) = (D.1 - C.1, D.2 - C.2) ∧
  -- Part 2
  ∃ (t : ℝ), t ≠ 0 ∧ (k * a.1 - b.1, k * a.2 - b.2) = (t * (a.1 + 3 * b.1), t * (a.2 + 3 * b.2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3791_379107


namespace NUMINAMATH_CALUDE_angle_BOK_formula_l3791_379109

/-- Represents a trihedral angle with vertex O and edges OA, OB, and OC -/
structure TrihedralAngle where
  α : ℝ  -- Angle BOC
  β : ℝ  -- Angle COA
  γ : ℝ  -- Angle AOB

/-- Represents a sphere inscribed in a trihedral angle -/
structure InscribedSphere (t : TrihedralAngle) where
  K : Point₃  -- Point where the sphere touches face BOC

/-- The angle BOK in a trihedral angle with an inscribed sphere -/
noncomputable def angleBOK (t : TrihedralAngle) (s : InscribedSphere t) : ℝ :=
  sorry

/-- Theorem stating that the angle BOK is equal to (α + γ - β) / 2 -/
theorem angle_BOK_formula (t : TrihedralAngle) (s : InscribedSphere t) :
  angleBOK t s = (t.α + t.γ - t.β) / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_BOK_formula_l3791_379109


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_l3791_379111

theorem monotonic_increasing_range (a : Real) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x > 0, Monotone (fun x => a^x + (1+a)^x)) →
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_l3791_379111


namespace NUMINAMATH_CALUDE_square_root_calculation_l3791_379155

theorem square_root_calculation : (Real.sqrt 2 + 1)^2 - Real.sqrt (9/2) = 3 + Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_calculation_l3791_379155


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l3791_379134

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (2 : ℚ) / 7 < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < (3 : ℚ) / 4 ∧ 
  ∀ (y : ℤ), ((2 : ℚ) / 7 < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < (3 : ℚ) / 4) → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l3791_379134


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3791_379197

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5) →
  a₃ + a₄ = -480 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3791_379197


namespace NUMINAMATH_CALUDE_mongolian_olympiad_inequality_l3791_379127

theorem mongolian_olympiad_inequality 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^4 + b^4 + c^4 + a^2/(b+c)^2 + b^2/(c+a)^2 + c^2/(a+b)^2 ≥ a*b + b*c + c*a :=
by sorry

end NUMINAMATH_CALUDE_mongolian_olympiad_inequality_l3791_379127


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l3791_379174

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  -- The side length of the equilateral triangle
  s : ℝ
  -- The coordinates of the special point P
  p : ℝ × ℝ
  -- Condition that the triangle is equilateral
  equilateral : s > 0
  -- Condition that P is inside the triangle
  p_inside : p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 < s
  -- Conditions for distances from P to vertices
  dist_ap : Real.sqrt ((0 - p.1)^2 + (0 - p.2)^2) = 1
  dist_bp : Real.sqrt ((s - p.1)^2 + (0 - p.2)^2) = Real.sqrt 3
  dist_cp : Real.sqrt ((s/2 - p.1)^2 + (s*Real.sqrt 3/2 - p.2)^2) = 2

theorem special_triangle_side_length (t : SpecialTriangle) : t.s = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_side_length_l3791_379174


namespace NUMINAMATH_CALUDE_champion_wins_39_l3791_379104

/-- Represents a basketball championship. -/
structure BasketballChampionship where
  n : ℕ                -- Number of teams
  totalPoints : ℕ      -- Total points of non-champion teams
  champPoints : ℕ      -- Points of the champion

/-- The number of matches won by the champion. -/
def championWins (championship : BasketballChampionship) : ℕ :=
  championship.champPoints - (championship.n - 1) * 2

/-- Theorem stating the number of matches won by the champion. -/
theorem champion_wins_39 (championship : BasketballChampionship) :
  championship.n = 27 ∧
  championship.totalPoints = 2015 ∧
  championship.champPoints = 3 * championship.n^2 - 3 * championship.n - championship.totalPoints →
  championWins championship = 39 := by
  sorry

#eval championWins { n := 27, totalPoints := 2015, champPoints := 91 }

end NUMINAMATH_CALUDE_champion_wins_39_l3791_379104


namespace NUMINAMATH_CALUDE_baseball_glove_price_l3791_379130

theorem baseball_glove_price :
  let cards_price : ℝ := 25
  let bat_price : ℝ := 10
  let cleats_price : ℝ := 10
  let total_sales : ℝ := 79
  let discount_rate : ℝ := 0.2
  let other_items_total : ℝ := cards_price + bat_price + 2 * cleats_price
  let glove_discounted_price : ℝ := total_sales - other_items_total
  let glove_original_price : ℝ := glove_discounted_price / (1 - discount_rate)
  glove_original_price = 42.5 := by
sorry

end NUMINAMATH_CALUDE_baseball_glove_price_l3791_379130


namespace NUMINAMATH_CALUDE_vector_scalar_addition_l3791_379195

theorem vector_scalar_addition (v₁ v₂ : Fin 3 → ℝ) (c : ℝ) :
  v₁ = ![2, -3, 4] →
  v₂ = ![-4, 7, -1] →
  c = 3 →
  c • v₁ + v₂ = ![2, -2, 11] := by sorry

end NUMINAMATH_CALUDE_vector_scalar_addition_l3791_379195


namespace NUMINAMATH_CALUDE_choose_one_from_each_set_l3791_379122

theorem choose_one_from_each_set : 
  ∀ (novels textbooks : ℕ), 
  novels = 5 → 
  textbooks = 6 → 
  novels * textbooks = 30 := by
sorry

end NUMINAMATH_CALUDE_choose_one_from_each_set_l3791_379122


namespace NUMINAMATH_CALUDE_berry_temperature_proof_l3791_379163

theorem berry_temperature_proof (temps : List Float) (avg : Float) : 
  temps = [99.1, 98.2, 98.7, 99.8, 99, 98.9] →
  avg = 99 →
  ∃ (wed_temp : Float), 
    wed_temp = 99.3 ∧ 
    (temps.sum + wed_temp) / 7 = avg :=
by sorry

end NUMINAMATH_CALUDE_berry_temperature_proof_l3791_379163


namespace NUMINAMATH_CALUDE_problem_statement_l3791_379149

theorem problem_statement (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 2)  -- absolute value of m is 2
  : (a + b) / m - m^2 + 2 * c * d = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3791_379149


namespace NUMINAMATH_CALUDE_chord_length_implies_a_values_l3791_379151

theorem chord_length_implies_a_values (a : ℝ) : 
  (∃ (x y : ℝ), (x - a)^2 + y^2 = 4 ∧ x - y = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ - a)^2 + y₁^2 = 4 ∧ x₁ - y₁ = 1 ∧
    (x₂ - a)^2 + y₂^2 = 4 ∧ x₂ - y₂ = 1 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 8) →
  a = -1 ∨ a = 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_values_l3791_379151


namespace NUMINAMATH_CALUDE_exponent_addition_l3791_379145

theorem exponent_addition (a : ℝ) (m n : ℕ) : a ^ m * a ^ n = a ^ (m + n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l3791_379145


namespace NUMINAMATH_CALUDE_two_self_inverse_matrices_l3791_379183

def is_self_inverse (a d : ℝ) : Prop :=
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; -8, d]
  M * M = 1

theorem two_self_inverse_matrices :
  ∃! (n : ℕ), ∃ (S : Finset (ℝ × ℝ)),
    S.card = n ∧
    (∀ (p : ℝ × ℝ), p ∈ S ↔ is_self_inverse p.1 p.2) :=
  sorry

end NUMINAMATH_CALUDE_two_self_inverse_matrices_l3791_379183


namespace NUMINAMATH_CALUDE_point_on_curve_iff_function_zero_l3791_379168

variable (f : ℝ × ℝ → ℝ)
variable (x₀ y₀ : ℝ)

theorem point_on_curve_iff_function_zero :
  f (x₀, y₀) = 0 ↔ (x₀, y₀) ∈ {p : ℝ × ℝ | f p = 0} := by sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_function_zero_l3791_379168


namespace NUMINAMATH_CALUDE_march_pancake_expense_l3791_379178

/-- Given the total expense on pancakes in March and the number of days,
    calculate the daily expense assuming equal consumption each day. -/
def daily_pancake_expense (total_expense : ℕ) (days : ℕ) : ℕ :=
  total_expense / days

/-- Theorem stating that the daily pancake expense in March is 11 dollars -/
theorem march_pancake_expense :
  daily_pancake_expense 341 31 = 11 := by
  sorry

end NUMINAMATH_CALUDE_march_pancake_expense_l3791_379178


namespace NUMINAMATH_CALUDE_composite_plus_four_prime_l3791_379175

/-- A number is composite if it has a factor between 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m : ℕ, 1 < m ∧ m < n ∧ n % m = 0

/-- A number is prime if it's greater than 1 and its only factors are 1 and itself -/
def IsPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m < n → n % m ≠ 0

theorem composite_plus_four_prime :
  ∃ n : ℕ, IsComposite n ∧ IsPrime (n + 4) :=
sorry

end NUMINAMATH_CALUDE_composite_plus_four_prime_l3791_379175


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l3791_379132

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point A
def point_A : ℝ × ℝ := (2, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (perpendicular_line point_A.1 point_A.2) ∧
  (∀ x y : ℝ, perpendicular_line x y → given_line x y →
    (y - point_A.2) = 2 * (x - point_A.1)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l3791_379132


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l3791_379123

/-- Given that x² varies inversely with y⁴, prove that x² = 4 when y = 4, given x = 8 when y = 2 -/
theorem inverse_variation_proof (x y : ℝ) (h1 : ∃ k : ℝ, ∀ x y, x^2 * y^4 = k) 
  (h2 : ∃ x₀ y₀ : ℝ, x₀ = 8 ∧ y₀ = 2 ∧ x₀^2 * y₀^4 = k) : 
  ∃ x₁ : ℝ, x₁^2 = 4 ∧ x₁^2 * 4^4 = k :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l3791_379123


namespace NUMINAMATH_CALUDE_technician_round_trip_completion_l3791_379185

theorem technician_round_trip_completion (D : ℝ) (h : D > 0) : 
  let total_distance : ℝ := 2 * D
  let completed_distance : ℝ := D + 0.3 * D
  (completed_distance / total_distance) * 100 = 65 := by sorry

end NUMINAMATH_CALUDE_technician_round_trip_completion_l3791_379185


namespace NUMINAMATH_CALUDE_sequence_consecutive_product_l3791_379181

/-- The nth term of the sequence, represented as n 1's followed by n 2's -/
def sequence_term (n : ℕ) : ℕ := 
  (10^n - 1) * (10^n + 2)

/-- The first factor of the product -/
def factor1 (n : ℕ) : ℕ := 
  (10^n - 1) / 3

/-- The second factor of the product -/
def factor2 (n : ℕ) : ℕ := 
  (10^n + 2) / 3

theorem sequence_consecutive_product (n : ℕ) : 
  sequence_term n = factor1 n * factor2 n ∧ factor2 n = factor1 n + 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_consecutive_product_l3791_379181


namespace NUMINAMATH_CALUDE_inequality_proof_l3791_379148

theorem inequality_proof (a b : ℝ) : (a^4 + a^2*b^2 + b^4) / 3 ≥ (a^3*b + b^3*a) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3791_379148


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l3791_379186

theorem complex_roots_quadratic (a b : ℝ) : 
  (Complex.mk a 3) ^ 2 - (Complex.mk 12 9) * (Complex.mk a 3) + (Complex.mk 15 65) = 0 ∧
  (Complex.mk b 6) ^ 2 - (Complex.mk 12 9) * (Complex.mk b 6) + (Complex.mk 15 65) = 0 →
  a = 7 / 3 ∧ b = 29 / 3 := by
sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l3791_379186


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3791_379135

/-- Given a geometric sequence {aₙ}, prove that if a₃ · a₄ = 5, then a₁ · a₂ · a₅ · a₆ = 5 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) 
  (h_prod : a 3 * a 4 = 5) : a 1 * a 2 * a 5 * a 6 = 5 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3791_379135


namespace NUMINAMATH_CALUDE_sum_congruence_mod_9_l3791_379105

theorem sum_congruence_mod_9 : 
  (1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888 + 999999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_9_l3791_379105


namespace NUMINAMATH_CALUDE_distance_between_bars_l3791_379102

/-- The distance between two bars given the walking times and speeds of two people --/
theorem distance_between_bars 
  (pierrot_extra_distance : ℝ) 
  (pierrot_time_after : ℝ) 
  (jeannot_time_after : ℝ) 
  (pierrot_speed_halved : ℝ → ℝ) 
  (jeannot_speed_halved : ℝ → ℝ) :
  ∃ (d : ℝ),
    pierrot_extra_distance = 200 ∧
    pierrot_time_after = 8 ∧
    jeannot_time_after = 18 ∧
    (∀ x, pierrot_speed_halved x = x / 2) ∧
    (∀ x, jeannot_speed_halved x = x / 2) ∧
    d > 0 ∧
    (d - pierrot_extra_distance) / (pierrot_speed_halved (d - pierrot_extra_distance) / pierrot_time_after) = 
      d / (jeannot_speed_halved d / jeannot_time_after) ∧
    2 * d - pierrot_extra_distance = 1000 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_bars_l3791_379102


namespace NUMINAMATH_CALUDE_special_day_price_l3791_379131

theorem special_day_price (original_price : ℝ) (first_discount_percent : ℝ) (second_discount_percent : ℝ) : 
  original_price = 240 →
  first_discount_percent = 40 →
  second_discount_percent = 25 →
  let first_discounted_price := original_price * (1 - first_discount_percent / 100)
  let special_day_price := first_discounted_price * (1 - second_discount_percent / 100)
  special_day_price = 108 := by
sorry

end NUMINAMATH_CALUDE_special_day_price_l3791_379131


namespace NUMINAMATH_CALUDE_cosine_inequality_solution_l3791_379115

theorem cosine_inequality_solution (y : Real) : 
  (y ∈ Set.Icc 0 (Real.pi / 2)) → 
  (∀ x ∈ Set.Icc 0 (2 * Real.pi), Real.cos (x + y) ≥ Real.cos x - Real.cos y) → 
  y = 0 :=
sorry

end NUMINAMATH_CALUDE_cosine_inequality_solution_l3791_379115


namespace NUMINAMATH_CALUDE_opposite_of_negative_eight_l3791_379152

theorem opposite_of_negative_eight :
  (∃ x : ℤ, -8 + x = 0) ∧ (∀ y : ℤ, -8 + y = 0 → y = 8) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_eight_l3791_379152


namespace NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l3791_379141

/-- A triangle with sides a, b, and c is either isosceles or right-angled if (a - b) * (a² + b² - c²) = 0 --/
theorem triangle_isosceles_or_right_angled (a b c : ℝ) (h : (a - b) * (a^2 + b^2 - c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_or_right_angled_l3791_379141


namespace NUMINAMATH_CALUDE_parallel_vectors_l3791_379188

/-- Given two 2D vectors a and b, find the value of k such that 
    (2a + b) is parallel to (1/2a + kb) -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (1, 2)) :
  ∃ k : ℝ, k = (1 : ℝ) / 4 ∧ 
  ∃ c : ℝ, c ≠ 0 ∧ c • (2 • a + b) = (1 / 2 : ℝ) • a + k • b :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_l3791_379188


namespace NUMINAMATH_CALUDE_bob_remaining_corn_l3791_379113

/-- Represents the amount of corn in bushels and ears -/
structure CornAmount where
  bushels : ℚ
  ears : ℕ

/-- Calculates the remaining corn after giving some away -/
def remaining_corn (initial : CornAmount) (given_away : List CornAmount) : ℕ :=
  sorry

/-- Theorem stating that Bob has 357 ears of corn left -/
theorem bob_remaining_corn :
  let initial := CornAmount.mk 50 0
  let given_away := [
    CornAmount.mk 8 0,    -- Terry
    CornAmount.mk 3 0,    -- Jerry
    CornAmount.mk 12 0,   -- Linda
    CornAmount.mk 0 21    -- Stacy
  ]
  let ears_per_bushel := 14
  remaining_corn initial given_away = 357 := by
  sorry

end NUMINAMATH_CALUDE_bob_remaining_corn_l3791_379113


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l3791_379140

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_size : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_size : Nat) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 4)
  (h4 : avg_age_group1 = 14)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  : ℝ := by
  sorry

#check fifteenth_student_age

end NUMINAMATH_CALUDE_fifteenth_student_age_l3791_379140


namespace NUMINAMATH_CALUDE_max_pairs_sum_l3791_379160

theorem max_pairs_sum (n : ℕ) (h : n = 2023) :
  ∃ (k : ℕ) (pairs : List (ℕ × ℕ)),
    k = 813 ∧
    pairs.length = k ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) ∧
    (∀ (p q : ℕ × ℕ), p ∈ pairs → q ∈ pairs → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs → p.1 + p.2 ≤ 2033) ∧
    (∀ (m : ℕ) (pairs' : List (ℕ × ℕ)),
      m > k →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 < p.2 ∧ p.1 ∈ Finset.range n ∧ p.2 ∈ Finset.range n) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 ≠ q.1 ∧ p.1 ≠ q.2 ∧ p.2 ≠ q.1 ∧ p.2 ≠ q.2) →
      (∀ (p q : ℕ × ℕ), p ∈ pairs' → q ∈ pairs' → p ≠ q → p.1 + p.2 ≠ q.1 + q.2) →
      (∀ (p : ℕ × ℕ), p ∈ pairs' → p.1 + p.2 ≤ 2033) →
      False) :=
by
  sorry

end NUMINAMATH_CALUDE_max_pairs_sum_l3791_379160


namespace NUMINAMATH_CALUDE_jennifer_remaining_money_l3791_379170

def initial_amount : ℚ := 150.75

def sandwich_fraction : ℚ := 3/10
def museum_fraction : ℚ := 1/4
def book_fraction : ℚ := 1/8
def coffee_percentage : ℚ := 2.5/100

def remaining_amount : ℚ := initial_amount - (
  initial_amount * sandwich_fraction +
  initial_amount * museum_fraction +
  initial_amount * book_fraction +
  initial_amount * coffee_percentage
)

theorem jennifer_remaining_money :
  remaining_amount = 45.225 := by sorry

end NUMINAMATH_CALUDE_jennifer_remaining_money_l3791_379170


namespace NUMINAMATH_CALUDE_faster_walking_speed_l3791_379126

/-- Given a person who walked 50 km at 10 km/hr, if they had walked at a faster speed
    that would allow them to cover an additional 20 km in the same time,
    prove that the faster speed would be 14 km/hr. -/
theorem faster_walking_speed (actual_distance : ℝ) (actual_speed : ℝ) (additional_distance : ℝ)
  (h1 : actual_distance = 50)
  (h2 : actual_speed = 10)
  (h3 : additional_distance = 20) :
  let time := actual_distance / actual_speed
  let total_distance := actual_distance + additional_distance
  let faster_speed := total_distance / time
  faster_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_faster_walking_speed_l3791_379126


namespace NUMINAMATH_CALUDE_coat_price_problem_l3791_379153

theorem coat_price_problem (price_reduction : ℝ) (percentage_reduction : ℝ) :
  price_reduction = 200 →
  percentage_reduction = 0.40 →
  ∃ original_price : ℝ, 
    original_price * percentage_reduction = price_reduction ∧
    original_price = 500 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_problem_l3791_379153


namespace NUMINAMATH_CALUDE_penguin_colony_ratio_l3791_379118

theorem penguin_colony_ratio :
  ∀ (initial_penguins end_first_year_penguins current_penguins : ℕ),
  end_first_year_penguins = 3 * initial_penguins →
  current_penguins = 3 * end_first_year_penguins + 129 →
  current_penguins = 1077 →
  end_first_year_penguins / initial_penguins = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_penguin_colony_ratio_l3791_379118


namespace NUMINAMATH_CALUDE_prob_different_suits_is_78_103_l3791_379158

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Represents two mixed standard 52-card decks -/
def MixedDecks := Deck × Deck

/-- The probability of picking two different cards of different suits from mixed decks -/
def prob_different_suits (decks : MixedDecks) : ℚ :=
  78 / 103

/-- Theorem stating the probability of picking two different cards of different suits -/
theorem prob_different_suits_is_78_103 (decks : MixedDecks) :
  prob_different_suits decks = 78 / 103 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_is_78_103_l3791_379158
