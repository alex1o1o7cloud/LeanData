import Mathlib

namespace NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l4073_407351

theorem sin_squared_minus_two_sin_range :
  ∀ x : ℝ, -1 ≤ Real.sin x ^ 2 - 2 * Real.sin x ∧ Real.sin x ^ 2 - 2 * Real.sin x ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_squared_minus_two_sin_range_l4073_407351


namespace NUMINAMATH_CALUDE_log_product_equals_four_l4073_407303

theorem log_product_equals_four : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_four_l4073_407303


namespace NUMINAMATH_CALUDE_digit_multiplication_puzzle_l4073_407322

def is_single_digit (n : ℕ) : Prop := n < 10

def is_five_digit_number (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def number_from_digits (a b c d e : ℕ) : ℕ := a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem digit_multiplication_puzzle :
  ∀ (a b c d e : ℕ),
    is_single_digit a ∧
    is_single_digit b ∧
    is_single_digit c ∧
    is_single_digit d ∧
    is_single_digit e ∧
    is_five_digit_number (number_from_digits a b c d e) ∧
    4 * (number_from_digits a b c d e) = number_from_digits e d c b a →
    a = 2 ∧ b = 1 ∧ c = 9 ∧ d = 7 ∧ e = 8 :=
by sorry

end NUMINAMATH_CALUDE_digit_multiplication_puzzle_l4073_407322


namespace NUMINAMATH_CALUDE_edwin_alvin_age_fraction_l4073_407374

/-- The fraction of Alvin's age that Edwin will be 20 more than in two years -/
def fraction_of_alvins_age : ℚ := 1 / 29

theorem edwin_alvin_age_fraction :
  let alvin_current_age : ℚ := (30.99999999 - 6) / 2
  let edwin_current_age : ℚ := alvin_current_age + 6
  let alvin_future_age : ℚ := alvin_current_age + 2
  let edwin_future_age : ℚ := edwin_current_age + 2
  edwin_future_age = fraction_of_alvins_age * alvin_future_age + 20 :=
by sorry

end NUMINAMATH_CALUDE_edwin_alvin_age_fraction_l4073_407374


namespace NUMINAMATH_CALUDE_women_married_fraction_l4073_407371

theorem women_married_fraction (total : ℕ) (women : ℕ) (married : ℕ) (men : ℕ) :
  women = (61 * total) / 100 →
  married = (60 * total) / 100 →
  men = total - women →
  (men - (men / 3)) * 3 = 2 * men →
  (married - (men / 3) : ℚ) / women = 47 / 61 :=
by
  sorry

end NUMINAMATH_CALUDE_women_married_fraction_l4073_407371


namespace NUMINAMATH_CALUDE_exponent_relationship_l4073_407344

theorem exponent_relationship (x y z a b : ℝ) 
  (h1 : 4^x = a) 
  (h2 : 2^y = b) 
  (h3 : 8^z = a * b) : 
  3 * z = 2 * x + y := by
sorry

end NUMINAMATH_CALUDE_exponent_relationship_l4073_407344


namespace NUMINAMATH_CALUDE_average_difference_l4073_407364

def average (a b : Int) : ℚ := (a + b) / 2

theorem average_difference : 
  average 500 1000 - average 100 500 = 450 := by sorry

end NUMINAMATH_CALUDE_average_difference_l4073_407364


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l4073_407304

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem monotonic_quadratic (a : ℝ) :
  monotonic_on (f a) 1 2 ↔ a ≤ 1 ∨ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l4073_407304


namespace NUMINAMATH_CALUDE_simplify_expression_l4073_407394

theorem simplify_expression (x : ℝ) : (3*x - 4)*(2*x + 10) - (x + 3)*(3*x - 2) = 3*x^2 + 15*x - 34 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4073_407394


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l4073_407335

/- Define the constants -/
def total_weight : ℕ := 1000
def round_weight : ℕ := 8
def square_weight : ℕ := 18
def round_price : ℕ := 160
def square_price : ℕ := 270

/- Part 1 -/
theorem part_one (a : ℕ) : 
  round_price * a + square_price * a = 8600 → a = 20 := by sorry

/- Part 2 -/
theorem part_two (x y : ℕ) :
  round_price * x + square_price * y = 16760 ∧
  round_weight * x + square_weight * y = total_weight →
  x = 44 ∧ y = 36 := by sorry

/- Part 3 -/
theorem part_three (m n b : ℕ) :
  b > 0 →
  round_price * m + square_price * n = 16760 ∧
  round_weight * (m + b) + square_weight * n = total_weight →
  (m + b = 80 ∧ n = 20) ∨ (m + b = 116 ∧ n = 4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l4073_407335


namespace NUMINAMATH_CALUDE_x_plus_y_equals_plus_minus_three_l4073_407309

theorem x_plus_y_equals_plus_minus_three (x y : ℝ) 
  (h1 : |x| = 1) 
  (h2 : |y| = 2) 
  (h3 : x * y > 0) : 
  x + y = 3 ∨ x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_plus_minus_three_l4073_407309


namespace NUMINAMATH_CALUDE_complex_sum_problem_l4073_407319

theorem complex_sum_problem (x y z w u v : ℝ) : 
  y = 2 ∧ 
  x = -z - u ∧ 
  (x + z + u) + (y + w + v) * I = 3 - 4 * I → 
  w + v = -6 := by sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l4073_407319


namespace NUMINAMATH_CALUDE_hidden_primes_sum_l4073_407354

/-- A card with two numbers -/
structure Card where
  visible : Nat
  hidden : Nat

/-- Predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop := sorry

/-- The sum of numbers on a card -/
def cardSum (c : Card) : Nat := c.visible + c.hidden

theorem hidden_primes_sum (c1 c2 c3 : Card) : 
  c1.visible = 17 →
  c2.visible = 26 →
  c3.visible = 41 →
  isPrime c1.hidden →
  isPrime c2.hidden →
  isPrime c3.hidden →
  cardSum c1 = cardSum c2 →
  cardSum c2 = cardSum c3 →
  c1.hidden + c2.hidden + c3.hidden = 198 := by
  sorry

end NUMINAMATH_CALUDE_hidden_primes_sum_l4073_407354


namespace NUMINAMATH_CALUDE_tornado_distance_l4073_407341

theorem tornado_distance (car_distance lawn_chair_distance birdhouse_distance : ℝ)
  (h1 : lawn_chair_distance = 2 * car_distance)
  (h2 : birdhouse_distance = 3 * lawn_chair_distance)
  (h3 : birdhouse_distance = 1200) :
  car_distance = 200 := by
sorry

end NUMINAMATH_CALUDE_tornado_distance_l4073_407341


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l4073_407369

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_triangle_interior_sum :
  (∀ k < 7, interior_sum k ≤ 50) ∧
  interior_sum 7 > 50 ∧
  interior_sum 7 = 62 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l4073_407369


namespace NUMINAMATH_CALUDE_min_value_inequality_l4073_407326

theorem min_value_inequality (x : ℝ) (h : x ≥ 4) : x + 4 / (x - 1) ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l4073_407326


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l4073_407317

/-- Given an ellipse and a hyperbola with specific foci, prove the product of their semi-axes -/
theorem ellipse_hyperbola_semi_axes_product (c d : ℝ) : 
  (∀ (x y : ℝ), x^2/c^2 + y^2/d^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2/c^2 - y^2/d^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |c * d| = Real.sqrt 868.5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_semi_axes_product_l4073_407317


namespace NUMINAMATH_CALUDE_truncated_cube_edges_l4073_407387

/-- Represents a cube with truncated corners -/
structure TruncatedCube where
  initialEdges : Nat
  vertices : Nat
  newEdgesPerVertex : Nat

/-- Calculates the total number of edges in a truncated cube -/
def totalEdges (c : TruncatedCube) : Nat :=
  c.initialEdges + c.vertices * c.newEdgesPerVertex

/-- Theorem stating that a cube with truncated corners has 36 edges -/
theorem truncated_cube_edges :
  ∀ (c : TruncatedCube),
  c.initialEdges = 12 ∧ c.vertices = 8 ∧ c.newEdgesPerVertex = 3 →
  totalEdges c = 36 := by
  sorry

end NUMINAMATH_CALUDE_truncated_cube_edges_l4073_407387


namespace NUMINAMATH_CALUDE_lowest_salary_grade_l4073_407306

/-- Represents the salary grade of an employee -/
def SalaryGrade := {s : ℝ // 1 ≤ s ∧ s ≤ 5}

/-- Calculates the hourly wage based on the salary grade -/
def hourlyWage (s : SalaryGrade) : ℝ :=
  7.50 + 0.25 * (s.val - 1)

/-- States that the difference in hourly wage between the highest and lowest salary grade is $1.25 -/
axiom wage_difference (s_min s_max : SalaryGrade) :
  s_min.val = 1 ∧ s_max.val = 5 →
  hourlyWage s_max - hourlyWage s_min = 1.25

theorem lowest_salary_grade :
  ∃ (s_min : SalaryGrade), s_min.val = 1 ∧
  ∀ (s : SalaryGrade), s_min.val ≤ s.val :=
by sorry

end NUMINAMATH_CALUDE_lowest_salary_grade_l4073_407306


namespace NUMINAMATH_CALUDE_problem_solution_l4073_407346

theorem problem_solution (x y z a b c : ℕ) : 
  x = 44 * 432 + 0 →
  x = 31 * y + z →
  x = a^3 * b^2 * c →
  Prime a ∧ Prime b ∧ Prime c →
  z = 5 ∧ a = 3 ∧ b = 4 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l4073_407346


namespace NUMINAMATH_CALUDE_fraction_depends_on_z_l4073_407349

theorem fraction_depends_on_z (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h : (4 * x + 2 * y) / (x - 4 * y) = 3) :
  ∃ z₁ z₂ : ℝ, z₁ ≠ z₂ ∧ 
    (x + 4 * y + z₁) / (4 * x - y - z₁) ≠ (x + 4 * y + z₂) / (4 * x - y - z₂) :=
by sorry

end NUMINAMATH_CALUDE_fraction_depends_on_z_l4073_407349


namespace NUMINAMATH_CALUDE_handshake_count_l4073_407372

theorem handshake_count (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_l4073_407372


namespace NUMINAMATH_CALUDE_elizabeth_study_time_l4073_407314

/-- The total study time for Elizabeth given her time spent on science and math tests -/
def total_study_time (science_time math_time : ℕ) : ℕ :=
  science_time + math_time

/-- Theorem stating that Elizabeth's total study time is 60 minutes -/
theorem elizabeth_study_time :
  total_study_time 25 35 = 60 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_study_time_l4073_407314


namespace NUMINAMATH_CALUDE_digits_of_4_20_5_28_3_10_l4073_407315

theorem digits_of_4_20_5_28_3_10 : 
  (fun n : ℕ => (Nat.log 10 n + 1 : ℕ)) (4^20 * 5^28 * 3^10) = 37 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_4_20_5_28_3_10_l4073_407315


namespace NUMINAMATH_CALUDE_odd_function_property_l4073_407333

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f 3 - f 2 = 1) :
  f (-2) - f (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l4073_407333


namespace NUMINAMATH_CALUDE_prime_minister_stays_l4073_407357

/-- Represents the message on a piece of paper -/
inductive Message
| stay
| leave

/-- Represents a piece of paper with a message -/
structure Paper :=
  (message : Message)

/-- The portfolio containing two papers -/
structure Portfolio :=
  (paper1 : Paper)
  (paper2 : Paper)

/-- The state of the game after the prime minister's action -/
structure GameState :=
  (destroyed : Paper)
  (revealed : Paper)

/-- The prime minister's strategy -/
def primeMinisterStrategy (portfolio : Portfolio) : GameState :=
  { destroyed := portfolio.paper1,
    revealed := portfolio.paper2 }

/-- The king's claim about the portfolio -/
def kingsClaim (p : Portfolio) : Prop :=
  (p.paper1.message = Message.stay ∧ p.paper2.message = Message.leave) ∨
  (p.paper1.message = Message.leave ∧ p.paper2.message = Message.stay)

/-- The actual content of the portfolio -/
def actualPortfolio : Portfolio :=
  { paper1 := { message := Message.leave },
    paper2 := { message := Message.leave } }

theorem prime_minister_stays :
  ∀ (state : GameState),
  state = primeMinisterStrategy actualPortfolio →
  state.revealed.message = Message.leave →
  ∃ (claim : Paper), claim.message = Message.stay ∧ 
    (claim = state.destroyed ∨ kingsClaim actualPortfolio = False) :=
by sorry

end NUMINAMATH_CALUDE_prime_minister_stays_l4073_407357


namespace NUMINAMATH_CALUDE_age_difference_l4073_407307

/-- Proves that A is 10 years older than B given the conditions in the problem -/
theorem age_difference (A B : ℕ) : 
  B = 70 →  -- B's present age is 70 years
  A + 20 = 2 * (B - 20) →  -- In 20 years, A will be twice as old as B was 20 years ago
  A - B = 10  -- A is 10 years older than B
  := by sorry

end NUMINAMATH_CALUDE_age_difference_l4073_407307


namespace NUMINAMATH_CALUDE_divisibility_condition_l4073_407336

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
  (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l4073_407336


namespace NUMINAMATH_CALUDE_linear_function_value_l4073_407331

/-- Given a linear function f(x) = ax + b, if f(3) = 7 and f(5) = -1, then f(0) = 19 -/
theorem linear_function_value (a b : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x + b)
    (h_3 : f 3 = 7)
    (h_5 : f 5 = -1) : 
  f 0 = 19 := by
sorry

end NUMINAMATH_CALUDE_linear_function_value_l4073_407331


namespace NUMINAMATH_CALUDE_sum_in_base5_l4073_407397

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of 201₅, 324₅, and 143₅ is equal to 1123₅ in base 5 --/
theorem sum_in_base5 :
  base10ToBase5 (base5ToBase10 201 + base5ToBase10 324 + base5ToBase10 143) = 1123 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base5_l4073_407397


namespace NUMINAMATH_CALUDE_pen_price_ratio_l4073_407329

theorem pen_price_ratio :
  ∀ (x y : ℕ) (b g : ℝ),
    x > 0 → y > 0 → b > 0 → g > 0 →
    (x + y) * g = 4 * (x * b + y * g) →
    (x + y) * b = (1 / 2) * (x * b + y * g) →
    g = 8 * b := by
  sorry

end NUMINAMATH_CALUDE_pen_price_ratio_l4073_407329


namespace NUMINAMATH_CALUDE_inverse_square_theorem_l4073_407384

/-- A function representing the inverse square relationship between x and y -/
def inverse_square_relation (k : ℝ) (x y : ℝ) : Prop :=
  x = k / (y ^ 2)

/-- Theorem stating the relationship between x and y -/
theorem inverse_square_theorem (k : ℝ) :
  (inverse_square_relation k 1 3) →
  (inverse_square_relation k 0.5625 4) :=
by sorry

end NUMINAMATH_CALUDE_inverse_square_theorem_l4073_407384


namespace NUMINAMATH_CALUDE_linear_function_composition_l4073_407363

-- Define a linear function
def IsLinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x + b

-- State the theorem
theorem linear_function_composition (f : ℝ → ℝ) :
  IsLinearFunction f → (∀ x, f (f x) = 4 * x + 8) →
  (∀ x, f x = 2 * x + 8 / 3) ∨ (∀ x, f x = -2 * x - 8) :=
by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l4073_407363


namespace NUMINAMATH_CALUDE_combined_storage_temperature_l4073_407373

-- Define the temperature ranges for each type of vegetable
def type_A_range : Set ℝ := {x | 3 ≤ x ∧ x ≤ 8}
def type_B_range : Set ℝ := {x | 5 ≤ x ∧ x ≤ 10}

-- Define the combined suitable temperature range
def combined_range : Set ℝ := type_A_range ∩ type_B_range

-- Theorem to prove
theorem combined_storage_temperature :
  combined_range = {x | 5 ≤ x ∧ x ≤ 8} := by
  sorry

end NUMINAMATH_CALUDE_combined_storage_temperature_l4073_407373


namespace NUMINAMATH_CALUDE_existence_of_z_l4073_407377

theorem existence_of_z (a p x y : ℕ) (hp : Prime p) (hx : x > 0) (hy : y > 0) (ha : a > 0)
  (hx41 : ∃ n : ℕ, x^41 = a + n*p) (hy49 : ∃ n : ℕ, y^49 = a + n*p) :
  ∃ (z : ℕ), z > 0 ∧ ∃ (n : ℕ), z^2009 = a + n*p :=
by sorry

end NUMINAMATH_CALUDE_existence_of_z_l4073_407377


namespace NUMINAMATH_CALUDE_train_journey_time_l4073_407356

theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (4 / 5 * usual_speed) * (usual_time + 1 / 2) = usual_speed * usual_time → 
  usual_time = 2 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l4073_407356


namespace NUMINAMATH_CALUDE_maximum_value_inequality_l4073_407325

theorem maximum_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ M : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → 
    a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(a*b^2 + b*c^2 + c*a^2 - 3*a*b*c)) →
  M ≤ 3 / Real.rpow 4 (1/3) :=
by sorry

end NUMINAMATH_CALUDE_maximum_value_inequality_l4073_407325


namespace NUMINAMATH_CALUDE_clock_hands_right_angles_l4073_407339

/-- Represents the number of times clock hands are at right angles in one hour -/
def right_angles_per_hour : ℕ := 2

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the number of days -/
def num_days : ℕ := 5

/-- Theorem: The hands of a clock are at right angles 240 times in 5 days -/
theorem clock_hands_right_angles :
  right_angles_per_hour * hours_per_day * num_days = 240 := by
  sorry

end NUMINAMATH_CALUDE_clock_hands_right_angles_l4073_407339


namespace NUMINAMATH_CALUDE_subset_implies_m_values_l4073_407311

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

-- State the theorem
theorem subset_implies_m_values (m : ℝ) : B m ⊆ A → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_values_l4073_407311


namespace NUMINAMATH_CALUDE_time_after_850_hours_l4073_407300

/-- Represents a time on a 12-hour clock -/
structure Time12Hour where
  hour : Nat
  minute : Nat
  period : Bool  -- false for AM, true for PM
  h_valid : hour ≥ 1 ∧ hour ≤ 12
  m_valid : minute ≥ 0 ∧ minute < 60

/-- Adds hours to a given time on a 12-hour clock -/
def addHours (t : Time12Hour) (h : Nat) : Time12Hour :=
  sorry

theorem time_after_850_hours : 
  let start_time := Time12Hour.mk 3 15 true (by norm_num) (by norm_num)
  let end_time := Time12Hour.mk 1 15 false (by norm_num) (by norm_num)
  addHours start_time 850 = end_time := by sorry

end NUMINAMATH_CALUDE_time_after_850_hours_l4073_407300


namespace NUMINAMATH_CALUDE_largest_common_value_l4073_407375

def arithmetic_progression_1 (n : ℕ) : ℕ := 4 + 5 * n
def arithmetic_progression_2 (n : ℕ) : ℕ := 5 + 8 * n

theorem largest_common_value :
  ∃ (k : ℕ), 
    (∃ (n m : ℕ), arithmetic_progression_1 n = arithmetic_progression_2 m ∧ arithmetic_progression_1 n = k) ∧
    k < 1000 ∧
    (∀ (l : ℕ), l < 1000 → 
      (∃ (p q : ℕ), arithmetic_progression_1 p = arithmetic_progression_2 q ∧ arithmetic_progression_1 p = l) →
      l ≤ k) ∧
    k = 989 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_value_l4073_407375


namespace NUMINAMATH_CALUDE_binomial_and_power_of_two_l4073_407388

theorem binomial_and_power_of_two : Nat.choose 8 3 = 56 ∧ 2^(Nat.choose 8 3) = 2^56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_and_power_of_two_l4073_407388


namespace NUMINAMATH_CALUDE_steve_blank_questions_l4073_407381

def total_questions : ℕ := 60
def word_problems : ℕ := 20
def add_sub_problems : ℕ := 25
def algebra_problems : ℕ := 10
def geometry_problems : ℕ := 5

def steve_word : ℕ := 15
def steve_add_sub : ℕ := 22
def steve_algebra : ℕ := 8
def steve_geometry : ℕ := 3

theorem steve_blank_questions :
  total_questions - (steve_word + steve_add_sub + steve_algebra + steve_geometry) = 12 :=
by sorry

end NUMINAMATH_CALUDE_steve_blank_questions_l4073_407381


namespace NUMINAMATH_CALUDE_total_savings_theorem_l4073_407337

def weekday_savings : ℝ := 24
def weekend_savings : ℝ := 30
def monthly_subscription : ℝ := 45
def annual_interest_rate : ℝ := 0.03
def weeks_in_year : ℕ := 52
def days_in_year : ℕ := 365

def total_savings : ℝ :=
  let weekday_count : ℕ := days_in_year - 2 * weeks_in_year
  let weekend_count : ℕ := 2 * weeks_in_year
  let total_savings_before_interest : ℝ :=
    weekday_count * weekday_savings + weekend_count * weekend_savings - 12 * monthly_subscription
  total_savings_before_interest * (1 + annual_interest_rate)

theorem total_savings_theorem :
  total_savings = 9109.32 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_theorem_l4073_407337


namespace NUMINAMATH_CALUDE_number_of_boys_in_school_l4073_407321

theorem number_of_boys_in_school :
  ∀ (x : ℕ),
  (x + (x * 900 / 100) = 900) →
  x = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_in_school_l4073_407321


namespace NUMINAMATH_CALUDE_speed_difference_l4073_407305

/-- The difference in average speeds between no traffic and heavy traffic conditions --/
theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ)
  (h1 : distance = 200)
  (h2 : time_heavy = 5)
  (h3 : time_no = 4) :
  distance / time_no - distance / time_heavy = 10 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l4073_407305


namespace NUMINAMATH_CALUDE_smallest_power_divisible_by_240_l4073_407383

theorem smallest_power_divisible_by_240 (n : ℕ) : 
  (∀ k : ℕ, k < n → ¬(240 ∣ 60^k)) ∧ (240 ∣ 60^n) → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_divisible_by_240_l4073_407383


namespace NUMINAMATH_CALUDE_power_sum_inequality_l4073_407376

theorem power_sum_inequality (k l m : ℕ) :
  2^(k+l) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l4073_407376


namespace NUMINAMATH_CALUDE_alan_shell_collection_l4073_407393

/-- Proves that Alan collected 48 shells given the conditions of the problem -/
theorem alan_shell_collection (laurie_shells : ℕ) (ben_ratio : ℚ) (alan_ratio : ℕ) : 
  laurie_shells = 36 → 
  ben_ratio = 1/3 → 
  alan_ratio = 4 → 
  (alan_ratio : ℚ) * ben_ratio * laurie_shells = 48 :=
by
  sorry

#check alan_shell_collection

end NUMINAMATH_CALUDE_alan_shell_collection_l4073_407393


namespace NUMINAMATH_CALUDE_equation_solution_l4073_407323

theorem equation_solution : ∃ c : ℚ, (c - 37) / 3 = (3 * c + 7) / 8 ∧ c = -317 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4073_407323


namespace NUMINAMATH_CALUDE_shaded_area_is_eight_l4073_407382

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a semicircle -/
structure Semicircle where
  center : Point
  radius : ℝ

/-- The geometric configuration -/
structure GeometricLayout where
  semicircle_ADB : Semicircle
  semicircle_BEC : Semicircle
  semicircle_DFE : Semicircle
  point_D : Point
  point_E : Point
  point_F : Point

/-- Conditions of the geometric layout -/
def validGeometricLayout (layout : GeometricLayout) : Prop :=
  layout.semicircle_ADB.radius = 2 ∧
  layout.semicircle_BEC.radius = 2 ∧
  layout.semicircle_DFE.radius = 1 ∧
  -- D is midpoint of ADB
  layout.point_D = { x := layout.semicircle_ADB.center.x, y := layout.semicircle_ADB.center.y + layout.semicircle_ADB.radius } ∧
  -- E is midpoint of BEC
  layout.point_E = { x := layout.semicircle_BEC.center.x, y := layout.semicircle_BEC.center.y + layout.semicircle_BEC.radius } ∧
  -- F is midpoint of DFE
  layout.point_F = { x := layout.semicircle_DFE.center.x, y := layout.semicircle_DFE.center.y + layout.semicircle_DFE.radius }

/-- Calculate the area of the shaded region -/
def shadedArea (layout : GeometricLayout) : ℝ :=
  -- Placeholder for the actual calculation
  8

/-- Theorem stating that the shaded area is 8 square units -/
theorem shaded_area_is_eight (layout : GeometricLayout) (h : validGeometricLayout layout) :
  shadedArea layout = 8 :=
by
  sorry


end NUMINAMATH_CALUDE_shaded_area_is_eight_l4073_407382


namespace NUMINAMATH_CALUDE_roberts_journey_distance_l4073_407348

/-- Represents the time in hours for each leg of Robert's journey -/
structure JourneyTimes where
  ab : ℝ
  bc : ℝ
  ca : ℝ

/-- Calculates the total distance of Robert's journey -/
def totalDistance (times : JourneyTimes) : ℝ :=
  let adjustedTime := times.ab + times.bc + times.ca - 1.5
  90 * adjustedTime

/-- Theorem stating that the total distance of Robert's journey is 1305 miles -/
theorem roberts_journey_distance (times : JourneyTimes) 
  (h1 : times.ab = 6)
  (h2 : times.bc = 5.5)
  (h3 : times.ca = 4.5) : 
  totalDistance times = 1305 := by
  sorry

#eval totalDistance { ab := 6, bc := 5.5, ca := 4.5 }

end NUMINAMATH_CALUDE_roberts_journey_distance_l4073_407348


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l4073_407360

/-- If the quadratic equation x^2 + x + m = 0 has two equal real roots,
    then m = 1/4 -/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + y + m = 0 → y = x) → 
  m = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l4073_407360


namespace NUMINAMATH_CALUDE_older_child_age_l4073_407389

def mother_charge : ℚ := 6.5
def child_charge_per_year : ℚ := 0.5
def total_bill : ℚ := 14.5
def num_children : ℕ := 4

def is_valid_age (triplet_age : ℕ) (older_age : ℕ) : Prop :=
  triplet_age > 0 ∧ 
  older_age > triplet_age ∧
  mother_charge + child_charge_per_year * (3 * triplet_age + older_age) = total_bill

theorem older_child_age :
  ∃ (triplet_age : ℕ) (older_age : ℕ), 
    is_valid_age triplet_age older_age ∧
    (older_age = 4 ∨ older_age = 7) ∧
    ¬∃ (other_age : ℕ), other_age ≠ 4 ∧ other_age ≠ 7 ∧ is_valid_age triplet_age other_age :=
by sorry

end NUMINAMATH_CALUDE_older_child_age_l4073_407389


namespace NUMINAMATH_CALUDE_interval_and_sum_l4073_407391

theorem interval_and_sum : 
  ∃ (m M : ℝ), 
    (∀ x : ℝ, x > 0 ∧ 2 * |x^2 - 9| ≤ 9 * |x| ↔ m ≤ x ∧ x ≤ M) ∧
    m = 3/2 ∧ 
    M = 6 ∧
    10 * m + M = 21 := by
  sorry

end NUMINAMATH_CALUDE_interval_and_sum_l4073_407391


namespace NUMINAMATH_CALUDE_division_remainder_proof_l4073_407398

theorem division_remainder_proof :
  ∀ (dividend quotient divisor remainder : ℕ),
    dividend = 144 →
    quotient = 13 →
    divisor = 11 →
    dividend = divisor * quotient + remainder →
    remainder = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l4073_407398


namespace NUMINAMATH_CALUDE_new_average_weight_l4073_407332

/-- Given 29 students with an average weight of 28 kg, after admitting a new student weighing 1 kg,
    the new average weight of all 30 students is 27.1 kg. -/
theorem new_average_weight (initial_count : ℕ) (initial_avg : ℝ) (new_student_weight : ℝ) :
  initial_count = 29 →
  initial_avg = 28 →
  new_student_weight = 1 →
  let total_weight := initial_count * initial_avg + new_student_weight
  let new_count := initial_count + 1
  (total_weight / new_count : ℝ) = 27.1 :=
by sorry

end NUMINAMATH_CALUDE_new_average_weight_l4073_407332


namespace NUMINAMATH_CALUDE_b_not_right_angle_sin_c_over_sin_a_range_l4073_407347

variable (A B C : Real)

-- Triangle ABC satisfies the given equation
axiom triangle_condition : 2 * Real.sin C * Real.sin (B - A) = 2 * Real.sin A * Real.sin C - (Real.sin B) ^ 2

-- Theorem 1: B cannot be a right angle
theorem b_not_right_angle : B ≠ Real.pi / 2 := by sorry

-- Theorem 2: Range of sin(C) / sin(A) for acute triangle
theorem sin_c_over_sin_a_range (acute_triangle : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = Real.pi) :
  1 / 3 < Real.sin C / Real.sin A ∧ Real.sin C / Real.sin A < 5 / 3 := by sorry

end NUMINAMATH_CALUDE_b_not_right_angle_sin_c_over_sin_a_range_l4073_407347


namespace NUMINAMATH_CALUDE_number_of_mappings_l4073_407313

/-- Given two finite sets A and B, where |A| = n and |B| = k, this function
    represents the number of order-preserving surjective mappings from A to B. -/
def orderPreservingSurjections (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The sets A and B -/
def A : Set ℝ := {a | ∃ i : Fin 60, a = i}
def B : Set ℝ := {b | ∃ i : Fin 25, b = i}

/-- The mapping f from A to B -/
def f : A → B := sorry

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- f preserves order -/
axiom f_order_preserving :
  ∀ (a₁ a₂ : A), (a₁ : ℝ) ≤ (a₂ : ℝ) → (f a₁ : ℝ) ≥ (f a₂ : ℝ)

/-- The main theorem: The number of such mappings is C₅₉²⁴ -/
theorem number_of_mappings :
  orderPreservingSurjections 60 25 = Nat.choose 59 24 := by sorry

end NUMINAMATH_CALUDE_number_of_mappings_l4073_407313


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l4073_407302

theorem quadratic_inequality_empty_solution (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 1 > 0) ↔ -2 < b ∧ b < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l4073_407302


namespace NUMINAMATH_CALUDE_roots_of_transformed_equation_l4073_407355

theorem roots_of_transformed_equation
  (p q : ℝ) (x₁ x₂ : ℝ)
  (h1 : x₁^2 + p*x₁ + q = 0)
  (h2 : x₂^2 + p*x₂ + q = 0)
  : (-x₁)^2 - p*(-x₁) + q = 0 ∧ (-x₂)^2 - p*(-x₂) + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_roots_of_transformed_equation_l4073_407355


namespace NUMINAMATH_CALUDE_probability_theorem_l4073_407392

/-- Represents a brother with a name of a certain length -/
structure Brother where
  name : String
  name_length : Nat

/-- Represents the problem setup -/
structure LetterCardProblem where
  adam : Brother
  brian : Brother
  total_letters : Nat
  (total_is_sum : total_letters = adam.name_length + brian.name_length)
  (total_is_twelve : total_letters = 12)

/-- The probability of selecting one letter from each brother's name -/
def probability_one_from_each (problem : LetterCardProblem) : Rat :=
  4 / 11

theorem probability_theorem (problem : LetterCardProblem) :
  probability_one_from_each problem = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l4073_407392


namespace NUMINAMATH_CALUDE_gardner_cupcakes_l4073_407345

theorem gardner_cupcakes (cookies brownies students treats_per_student : ℕ) 
  (h1 : cookies = 20)
  (h2 : brownies = 35)
  (h3 : students = 20)
  (h4 : treats_per_student = 4)
  (h5 : students * treats_per_student = cookies + brownies + (students * treats_per_student - cookies - brownies)) :
  students * treats_per_student - cookies - brownies = 25 := by
  sorry

end NUMINAMATH_CALUDE_gardner_cupcakes_l4073_407345


namespace NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l4073_407301

/-- Given two circles with radii 3 and 5, centered at (0, 0) and (12, 0) respectively,
    the x-coordinate of the point where a common tangent line intersects the x-axis is 9/2. -/
theorem tangent_intersection_x_coordinate :
  let circle1_radius : ℝ := 3
  let circle1_center : ℝ × ℝ := (0, 0)
  let circle2_radius : ℝ := 5
  let circle2_center : ℝ × ℝ := (12, 0)
  ∃ x : ℝ, x > 0 ∧ 
    (x / (12 - x) = circle1_radius / circle2_radius) ∧
    x = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_x_coordinate_l4073_407301


namespace NUMINAMATH_CALUDE_remaining_meat_l4073_407380

/-- Given an initial amount of meat and the amounts used for meatballs and spring rolls,
    prove that the remaining amount of meat is 12 kilograms. -/
theorem remaining_meat (initial_meat : ℝ) (meatball_fraction : ℝ) (spring_roll_meat : ℝ)
    (h1 : initial_meat = 20)
    (h2 : meatball_fraction = 1 / 4)
    (h3 : spring_roll_meat = 3) :
    initial_meat - (initial_meat * meatball_fraction) - spring_roll_meat = 12 :=
by sorry

end NUMINAMATH_CALUDE_remaining_meat_l4073_407380


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l4073_407334

theorem complex_parts_of_z (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := i * (-1 + 2*i)
  (z.re = -2) ∧ (z.im = -1) := by sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l4073_407334


namespace NUMINAMATH_CALUDE_mapping_A_to_B_l4073_407390

def A : Finset ℕ := {1, 2, 3, 4, 5}
def B : Finset ℕ := {0, 3, 8, 15, 24}

def f (x : ℕ) : ℕ := x^2 - 1

theorem mapping_A_to_B :
  ∀ x ∈ A, f x ∈ B :=
by sorry

end NUMINAMATH_CALUDE_mapping_A_to_B_l4073_407390


namespace NUMINAMATH_CALUDE_three_conical_planet_models_l4073_407366

/-- Represents a model of a conical planet --/
structure ConicalPlanetModel where
  /-- The type of coordinate lines in the model --/
  CoordinateLine : Type
  /-- Predicate for whether two coordinate lines intersect --/
  intersects : CoordinateLine → CoordinateLine → Prop
  /-- Predicate for whether a coordinate line self-intersects --/
  self_intersects : CoordinateLine → Prop
  /-- Predicate for whether the constant direction principle holds --/
  constant_direction : Prop

/-- Cylindrical projection model --/
def cylindrical_model : ConicalPlanetModel := sorry

/-- Traditional conical projection model --/
def conical_model : ConicalPlanetModel := sorry

/-- Hybrid model --/
def hybrid_model : ConicalPlanetModel := sorry

/-- Properties of the hybrid model --/
axiom hybrid_model_properties :
  ∀ (l1 l2 : hybrid_model.CoordinateLine),
    l1 ≠ l2 → (hybrid_model.intersects l1 l2 ∧ hybrid_model.intersects l2 l1) ∧
    hybrid_model.self_intersects l1 ∧
    hybrid_model.constant_direction

/-- Theorem stating the existence of three distinct conical planet models --/
theorem three_conical_planet_models :
  ∃ (m1 m2 m3 : ConicalPlanetModel),
    m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧
    (m1 = cylindrical_model ∨ m1 = conical_model ∨ m1 = hybrid_model) ∧
    (m2 = cylindrical_model ∨ m2 = conical_model ∨ m2 = hybrid_model) ∧
    (m3 = cylindrical_model ∨ m3 = conical_model ∨ m3 = hybrid_model) := by
  sorry

end NUMINAMATH_CALUDE_three_conical_planet_models_l4073_407366


namespace NUMINAMATH_CALUDE_ten_coin_flips_sequences_l4073_407327

/-- The number of distinct sequences when flipping a coin n times -/
def coin_flip_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coin_flip_sequences 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_ten_coin_flips_sequences_l4073_407327


namespace NUMINAMATH_CALUDE_horner_method_v3_l4073_407330

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 - 3x^3 + 2x^2 + 5x + 1 -/
def f : List ℝ := [4, 0, -3, 2, 5, 1]

theorem horner_method_v3 :
  let v3 := (horner (f.take 4) 3)
  v3 = 101 := by sorry

end NUMINAMATH_CALUDE_horner_method_v3_l4073_407330


namespace NUMINAMATH_CALUDE_unique_integers_square_sum_l4073_407350

theorem unique_integers_square_sum : ∃! (A B : ℕ), 
  A ≤ 9 ∧ B ≤ 9 ∧ (1001 * A + 110 * B)^2 = 57108249 ∧ 10 * A + B = 75 := by
  sorry

end NUMINAMATH_CALUDE_unique_integers_square_sum_l4073_407350


namespace NUMINAMATH_CALUDE_unique_solution_l4073_407320

theorem unique_solution : ∀ a b c : ℕ, 2^a + 9^b = 2 * 5^c + 5 ↔ a = 1 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4073_407320


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l4073_407353

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 4*x + 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - 5*x + 6 = 0
  let sol1 : Set ℝ := {2 + Real.sqrt 3, 2 - Real.sqrt 3}
  let sol2 : Set ℝ := {2, 3}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l4073_407353


namespace NUMINAMATH_CALUDE_elbertas_money_l4073_407359

/-- Given that Granny Smith has $45, Elberta has $4 more than Anjou, and Anjou has one-fourth as much as Granny Smith, prove that Elberta has $15.25. -/
theorem elbertas_money (granny_smith : ℝ) (elberta anjou : ℝ) 
  (h1 : granny_smith = 45)
  (h2 : elberta = anjou + 4)
  (h3 : anjou = granny_smith / 4) :
  elberta = 15.25 := by
  sorry

end NUMINAMATH_CALUDE_elbertas_money_l4073_407359


namespace NUMINAMATH_CALUDE_problem_solution_l4073_407338

theorem problem_solution (x : ℚ) : (5 * x - 8 = 15 * x + 4) → (3 * (x + 9) = 129 / 5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4073_407338


namespace NUMINAMATH_CALUDE_common_difference_is_half_l4073_407308

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_4_6 : a 4 + a 6 = 6
  sum_5 : (a 1 + a 2 + a 3 + a 4 + a 5 : ℚ) = 10

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℚ :=
  seq.a 2 - seq.a 1

/-- Theorem stating that the common difference is 1/2 -/
theorem common_difference_is_half (seq : ArithmeticSequence) :
  common_difference seq = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_half_l4073_407308


namespace NUMINAMATH_CALUDE_work_hours_ratio_l4073_407358

theorem work_hours_ratio (amber_hours : ℕ) (total_hours : ℕ) : 
  amber_hours = 12 →
  total_hours = 40 →
  ∃ (ella_hours : ℕ),
    (ella_hours + amber_hours + amber_hours / 3 = total_hours) ∧
    (ella_hours : ℚ) / amber_hours = 2 := by
  sorry

end NUMINAMATH_CALUDE_work_hours_ratio_l4073_407358


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l4073_407340

theorem geometric_sequence_third_term :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n > 0) →  -- Sequence of positive integers
    (∃ r : ℕ, ∀ n, a (n + 1) = a n * r) →  -- Geometric sequence
    a 1 = 5 →  -- First term is 5
    a 5 = 405 →  -- Fifth term is 405
    a 3 = 45 :=  -- Third term is 45
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l4073_407340


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l4073_407396

theorem ellipse_focus_distance (x y : ℝ) :
  x^2 / 25 + y^2 / 16 = 1 →
  ∃ (f1 f2 : ℝ × ℝ), 
    (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ 
      Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) = 3) →
    Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l4073_407396


namespace NUMINAMATH_CALUDE_dot_product_sum_equilateral_triangle_l4073_407361

-- Define the equilateral triangle
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  (dist A B = 1) ∧ (dist B C = 1) ∧ (dist C A = 1)

-- Define vectors a, b, c
def a (B C : ℝ × ℝ) : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def b (A C : ℝ × ℝ) : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def c (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_sum_equilateral_triangle (A B C : ℝ × ℝ) 
  (h : EquilateralTriangle A B C) : 
  dot_product (a B C) (b A C) + dot_product (b A C) (c A B) + dot_product (c A B) (a B C) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_sum_equilateral_triangle_l4073_407361


namespace NUMINAMATH_CALUDE_bakers_sales_l4073_407342

/-- Baker's cake and pastry problem -/
theorem bakers_sales (cakes_made pastries_made pastries_sold : ℕ) 
  (h1 : cakes_made = 157)
  (h2 : pastries_made = 169)
  (h3 : pastries_sold = 147)
  (h4 : ∃ cakes_sold : ℕ, cakes_sold = pastries_sold + 11) :
  ∃ cakes_sold : ℕ, cakes_sold = 158 := by
  sorry

end NUMINAMATH_CALUDE_bakers_sales_l4073_407342


namespace NUMINAMATH_CALUDE_larger_box_capacity_l4073_407386

/-- Represents a rectangular box with integer dimensions -/
structure Box where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box -/
def Box.volume (b : Box) : ℕ := b.length * b.width * b.height

/-- Represents the number of marbles a box can hold -/
def marbles_capacity (b : Box) (marbles : ℕ) : Prop :=
  b.volume = marbles

theorem larger_box_capacity 
  (kevin_box : Box)
  (kevin_marbles : ℕ)
  (laura_box : Box)
  (h1 : kevin_box.length = 3 ∧ kevin_box.width = 3 ∧ kevin_box.height = 8)
  (h2 : marbles_capacity kevin_box kevin_marbles)
  (h3 : kevin_marbles = 216)
  (h4 : laura_box.length = 3 * kevin_box.length ∧ 
        laura_box.width = 3 * kevin_box.width ∧ 
        laura_box.height = 3 * kevin_box.height) :
  marbles_capacity laura_box 5832 :=
sorry

end NUMINAMATH_CALUDE_larger_box_capacity_l4073_407386


namespace NUMINAMATH_CALUDE_suspension_ratio_l4073_407365

/-- The number of fingers and toes a typical person has -/
def typical_fingers_and_toes : ℕ := 20

/-- The number of days Kris is suspended for each bullying instance -/
def suspension_days_per_instance : ℕ := 3

/-- The number of bullying instances Kris is responsible for -/
def bullying_instances : ℕ := 20

/-- Kris's total suspension days -/
def total_suspension_days : ℕ := suspension_days_per_instance * bullying_instances

theorem suspension_ratio :
  total_suspension_days / typical_fingers_and_toes = 3 :=
by sorry

end NUMINAMATH_CALUDE_suspension_ratio_l4073_407365


namespace NUMINAMATH_CALUDE_train_length_l4073_407370

/-- The length of a train given its speed, the speed of a man running in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) 
  (h1 : train_speed = 50) 
  (h2 : man_speed = 4) 
  (h3 : passing_time = 8) : 
  (train_speed + man_speed) * passing_time * (1000 / 3600) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4073_407370


namespace NUMINAMATH_CALUDE_total_gold_stars_l4073_407310

def gold_stars_yesterday : ℕ := 4
def gold_stars_today : ℕ := 3

theorem total_gold_stars : gold_stars_yesterday + gold_stars_today = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_gold_stars_l4073_407310


namespace NUMINAMATH_CALUDE_prize_winning_probability_l4073_407316

def num_card_types : ℕ := 3
def num_bags : ℕ := 4

def winning_probability : ℚ :=
  1 - (num_card_types.choose 2 * 2^num_bags - num_card_types) / num_card_types^num_bags

theorem prize_winning_probability :
  winning_probability = 4/9 := by sorry

end NUMINAMATH_CALUDE_prize_winning_probability_l4073_407316


namespace NUMINAMATH_CALUDE_brick_length_proof_l4073_407312

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.length * d.width * d.height

theorem brick_length_proof (wall : Dimensions) (brick : Dimensions) (num_bricks : ℝ) :
  wall.length = 8 →
  wall.width = 6 →
  wall.height = 0.02 →
  brick.length = 0.11 →
  brick.width = 0.05 →
  brick.height = 0.06 →
  num_bricks = 2909.090909090909 →
  volume wall / volume brick = num_bricks →
  brick.length = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_proof_l4073_407312


namespace NUMINAMATH_CALUDE_inequality_proof_l4073_407379

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4073_407379


namespace NUMINAMATH_CALUDE_storybook_pages_l4073_407324

theorem storybook_pages : (10 + 5) / (1 - 1/5 * 2) = 25 := by
  sorry

end NUMINAMATH_CALUDE_storybook_pages_l4073_407324


namespace NUMINAMATH_CALUDE_angle_C_measure_l4073_407395

theorem angle_C_measure (A B C : ℝ) (h : A + B = 80) : A + B + C = 180 → C = 100 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_measure_l4073_407395


namespace NUMINAMATH_CALUDE_f_properties_l4073_407352

def f (x : ℝ) : ℝ := |x| + 1

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y → x < 0 → f y < f x) ∧
  (∀ x y : ℝ, x < y → 0 < x → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l4073_407352


namespace NUMINAMATH_CALUDE_power_subtraction_l4073_407385

theorem power_subtraction : (81 : ℝ) ^ (1/4) - (16 : ℝ) ^ (1/2) = -1 := by sorry

end NUMINAMATH_CALUDE_power_subtraction_l4073_407385


namespace NUMINAMATH_CALUDE_jacobStatementsDisproved_l4073_407399

-- Define the type for card sides
inductive CardSide
| Letter : Char → CardSide
| Number : Nat → CardSide

-- Define a card as a pair of sides
def Card := (CardSide × CardSide)

-- Define the properties of cards
def isVowel (c : Char) : Prop := c ∈ ['A', 'E', 'I', 'O', 'U']
def isEven (n : Nat) : Prop := n % 2 = 0
def isPrime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m > 1 → m < n → n % m ≠ 0)

-- Jacob's statements
def jacobStatement1 (card : Card) : Prop :=
  match card with
  | (CardSide.Letter c, CardSide.Number n) => isVowel c → isEven n
  | _ => True

def jacobStatement2 (card : Card) : Prop :=
  match card with
  | (CardSide.Number n, CardSide.Letter c) => isPrime n → isVowel c
  | _ => True

-- Define the set of cards
def cardSet : List Card := [
  (CardSide.Letter 'A', CardSide.Number 8),
  (CardSide.Letter 'R', CardSide.Number 5),
  (CardSide.Letter 'S', CardSide.Number 7),
  (CardSide.Number 1, CardSide.Letter 'R'),
  (CardSide.Number 8, CardSide.Letter 'S'),
  (CardSide.Number 5, CardSide.Letter 'A')
]

-- Theorem: There exist two cards that disprove at least one of Jacob's statements
theorem jacobStatementsDisproved : 
  ∃ (card1 card2 : Card), card1 ∈ cardSet ∧ card2 ∈ cardSet ∧ card1 ≠ card2 ∧
    (¬(jacobStatement1 card1) ∨ ¬(jacobStatement2 card1) ∨
     ¬(jacobStatement1 card2) ∨ ¬(jacobStatement2 card2)) :=
by sorry


end NUMINAMATH_CALUDE_jacobStatementsDisproved_l4073_407399


namespace NUMINAMATH_CALUDE_kelly_bought_five_more_paper_l4073_407368

/-- Calculates the number of additional pieces of construction paper Kelly bought --/
def additional_construction_paper (students : ℕ) (paper_per_student : ℕ) (glue_bottles : ℕ) (final_supplies : ℕ) : ℕ :=
  let initial_supplies := students * paper_per_student + glue_bottles
  let remaining_supplies := initial_supplies / 2
  final_supplies - remaining_supplies

/-- Proves that Kelly bought 5 additional pieces of construction paper --/
theorem kelly_bought_five_more_paper : 
  additional_construction_paper 8 3 6 20 = 5 := by
  sorry

#eval additional_construction_paper 8 3 6 20

end NUMINAMATH_CALUDE_kelly_bought_five_more_paper_l4073_407368


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l4073_407367

theorem triangle_angle_measure (P Q R : ℝ) (h1 : P = 2 * Q) (h2 : R = 5 * Q) (h3 : P + Q + R = 180) : P = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l4073_407367


namespace NUMINAMATH_CALUDE_expected_sixes_is_half_l4073_407343

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a standard die -/
def prob_not_six : ℚ := 1 - prob_six

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The expected number of 6's when rolling three standard dice -/
def expected_sixes : ℚ := 
  (0 : ℚ) * (prob_not_six ^ num_dice) +
  (1 : ℚ) * (num_dice.choose 1 * prob_six * prob_not_six^2) +
  (2 : ℚ) * (num_dice.choose 2 * prob_six^2 * prob_not_six) +
  (3 : ℚ) * (prob_six ^ num_dice)

theorem expected_sixes_is_half : expected_sixes = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_expected_sixes_is_half_l4073_407343


namespace NUMINAMATH_CALUDE_warden_citations_l4073_407362

/-- The total number of citations issued by a park warden -/
theorem warden_citations (littering : ℕ) (off_leash : ℕ) (parking : ℕ) 
  (h1 : littering = off_leash)
  (h2 : parking = 2 * littering)
  (h3 : littering = 4) : 
  littering + off_leash + parking = 16 := by
  sorry

end NUMINAMATH_CALUDE_warden_citations_l4073_407362


namespace NUMINAMATH_CALUDE_polynomial_equality_l4073_407328

/-- Given a polynomial M such that M + (5x^2 - 4x - 3) = -x^2 - 3x,
    prove that M = -6x^2 + x + 3 -/
theorem polynomial_equality (x : ℝ) (M : ℝ → ℝ) : 
  (M x + (5*x^2 - 4*x - 3) = -x^2 - 3*x) → 
  (M x = -6*x^2 + x + 3) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4073_407328


namespace NUMINAMATH_CALUDE_rectangle_to_hexagon_side_length_l4073_407318

theorem rectangle_to_hexagon_side_length :
  ∀ (rectangle_length rectangle_width : ℝ) (hexagon_side : ℝ),
    rectangle_length = 24 →
    rectangle_width = 8 →
    (3 * Real.sqrt 3 / 2) * hexagon_side^2 = rectangle_length * rectangle_width →
    hexagon_side = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_hexagon_side_length_l4073_407318


namespace NUMINAMATH_CALUDE_test_retake_count_l4073_407378

theorem test_retake_count (total : ℕ) (passed : ℕ) (retake : ℕ) : 
  total = 2500 → passed = 375 → retake = total - passed → retake = 2125 := by
  sorry

end NUMINAMATH_CALUDE_test_retake_count_l4073_407378
