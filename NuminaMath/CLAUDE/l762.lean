import Mathlib

namespace NUMINAMATH_CALUDE_rice_price_fall_l762_76211

theorem rice_price_fall (original_price : ℝ) (original_quantity : ℝ) : 
  original_price > 0 →
  original_quantity > 0 →
  let new_price := 0.8 * original_price
  let new_quantity := 50
  original_price * original_quantity = new_price * new_quantity →
  original_quantity = 40 := by
sorry

end NUMINAMATH_CALUDE_rice_price_fall_l762_76211


namespace NUMINAMATH_CALUDE_sum_of_arithmetic_series_l762_76281

/-- Sum of an arithmetic series with given parameters -/
theorem sum_of_arithmetic_series : 
  ∀ (a l d : ℤ) (n : ℕ+),
  a = -48 →
  d = 4 →
  l = 0 →
  a + (n - 1 : ℤ) * d = l →
  (n : ℤ) * (a + l) / 2 = -312 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_arithmetic_series_l762_76281


namespace NUMINAMATH_CALUDE_pentagonal_tiles_count_l762_76228

theorem pentagonal_tiles_count (total_tiles total_edges : ℕ) 
  (h1 : total_tiles = 30)
  (h2 : total_edges = 120) : 
  ∃ (triangular_tiles pentagonal_tiles : ℕ),
    triangular_tiles + pentagonal_tiles = total_tiles ∧
    3 * triangular_tiles + 5 * pentagonal_tiles = total_edges ∧
    pentagonal_tiles = 15 := by
  sorry

end NUMINAMATH_CALUDE_pentagonal_tiles_count_l762_76228


namespace NUMINAMATH_CALUDE_range_of_a_l762_76226

/-- An odd function f(x) = ax³ + bx² + cx + d satisfying certain conditions -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem stating the range of 'a' given the conditions -/
theorem range_of_a (a b c d : ℝ) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- f is odd
  (f a b c d 1 = 1) →  -- f(1) = 1
  (∀ x ∈ Set.Icc (-1) 1, |f a b c d x| ≤ 1) →  -- |f(x)| ≤ 1 for x ∈ [-1, 1]
  a ∈ Set.Icc (-1/2) 4 :=  -- a ∈ [-1/2, 4]
by sorry

end NUMINAMATH_CALUDE_range_of_a_l762_76226


namespace NUMINAMATH_CALUDE_weight_loss_program_result_l762_76203

/-- Calculates the final weight after a weight loss program -/
def final_weight (initial_weight : ℕ) (loss_rate1 : ℕ) (weeks1 : ℕ) (loss_rate2 : ℕ) (weeks2 : ℕ) : ℕ :=
  initial_weight - (loss_rate1 * weeks1 + loss_rate2 * weeks2)

/-- Theorem stating that the final weight after the given weight loss program is 222 pounds -/
theorem weight_loss_program_result :
  final_weight 250 3 4 2 8 = 222 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_program_result_l762_76203


namespace NUMINAMATH_CALUDE_gcd_equation_solution_l762_76266

theorem gcd_equation_solution (b d : ℕ) : 
  Nat.gcd b 175 = d → 
  176 * (b - 11 * d + 1) = 5 * d + 1 → 
  b = 385 := by
sorry

end NUMINAMATH_CALUDE_gcd_equation_solution_l762_76266


namespace NUMINAMATH_CALUDE_melissa_driving_hours_l762_76230

/-- Calculates the total driving hours in a year for a person who drives to town twice each month -/
def yearly_driving_hours (trips_per_month : ℕ) (hours_per_trip : ℕ) : ℕ :=
  trips_per_month * hours_per_trip * 12

theorem melissa_driving_hours :
  yearly_driving_hours 2 3 = 72 := by
sorry

end NUMINAMATH_CALUDE_melissa_driving_hours_l762_76230


namespace NUMINAMATH_CALUDE_evaluate_expression_l762_76224

theorem evaluate_expression (b : ℝ) : 
  let x := b + 9
  (x - b + 4) = 13 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l762_76224


namespace NUMINAMATH_CALUDE_hadassah_painting_time_l762_76295

/-- Represents the time taken to paint paintings and take breaks -/
def total_time (small_paint_rate : ℝ) (large_paint_rate : ℝ) (small_count : ℕ) (large_count : ℕ) (break_duration : ℝ) (paintings_per_break : ℕ) : ℝ :=
  let small_time := small_paint_rate * small_count
  let large_time := large_paint_rate * large_count
  let total_paintings := small_count + large_count
  let break_count := total_paintings / paintings_per_break
  let break_time := break_count * break_duration
  small_time + large_time + break_time

/-- Theorem stating the total time Hadassah takes to finish all paintings -/
theorem hadassah_painting_time : 
  let small_paint_rate := 6 / 12
  let large_paint_rate := 8 / 6
  let small_count := 15
  let large_count := 10
  let break_duration := 0.5
  let paintings_per_break := 3
  total_time small_paint_rate large_paint_rate small_count large_count break_duration paintings_per_break = 24.8 := by
  sorry

end NUMINAMATH_CALUDE_hadassah_painting_time_l762_76295


namespace NUMINAMATH_CALUDE_cookie_price_equality_l762_76210

/-- The radius of Art's circular cookies -/
def art_radius : ℝ := 2

/-- The side length of Roger's square cookies -/
def roger_side : ℝ := 4

/-- The number of cookies Art makes from one batch -/
def art_cookie_count : ℕ := 9

/-- The price of one of Art's cookies in cents -/
def art_cookie_price : ℕ := 50

/-- The price of one of Roger's cookies in cents -/
def roger_cookie_price : ℕ := 64

theorem cookie_price_equality :
  let art_total_area := art_cookie_count * Real.pi * art_radius^2
  let roger_cookie_area := roger_side^2
  let roger_cookie_count := art_total_area / roger_cookie_area
  art_cookie_count * art_cookie_price = ⌊roger_cookie_count⌋ * roger_cookie_price :=
sorry

end NUMINAMATH_CALUDE_cookie_price_equality_l762_76210


namespace NUMINAMATH_CALUDE_power_mod_eleven_l762_76273

theorem power_mod_eleven : (Nat.pow 3 101 + 5) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l762_76273


namespace NUMINAMATH_CALUDE_log_and_inverse_properties_l762_76269

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define the inverse function of log_a(x)
noncomputable def log_inverse (a : ℝ) (x : ℝ) : ℝ := a ^ x

-- Theorem statement
theorem log_and_inverse_properties (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  -- 1. Same monotonicity
  (∀ x y, x < y → log a x < log a y ↔ log_inverse a x < log_inverse a y) ∧
  -- 2. No intersection when a > 1
  (a > 1 → ∀ x, log a x ≠ log_inverse a x) ∧
  -- 3. Intersection point on y = x
  (∀ x, log a x = log_inverse a x → log a x = x) :=
by sorry

end NUMINAMATH_CALUDE_log_and_inverse_properties_l762_76269


namespace NUMINAMATH_CALUDE_greatest_integer_x_cube_less_than_15_l762_76237

theorem greatest_integer_x_cube_less_than_15 :
  ∃ (x : ℕ), x > 0 ∧ (x^6 / x^3 : ℚ) < 15 ∧ ∀ (y : ℕ), y > x → (y^6 / y^3 : ℚ) ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_x_cube_less_than_15_l762_76237


namespace NUMINAMATH_CALUDE_intersection_distance_l762_76258

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

-- Define the intersection points
def intersection1 : ℝ × ℝ := (1, 1)
def intersection2 : ℝ × ℝ := (0, 0)

-- State the theorem
theorem intersection_distance :
  curve1 (intersection1.1) (intersection1.2) ∧
  curve2 (intersection1.1) (intersection1.2) ∧
  curve1 (intersection2.1) (intersection2.2) ∧
  curve2 (intersection2.1) (intersection2.2) →
  Real.sqrt ((intersection1.1 - intersection2.1)^2 + (intersection1.2 - intersection2.2)^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l762_76258


namespace NUMINAMATH_CALUDE_solve_for_a_l762_76234

theorem solve_for_a (a b c d : ℝ) 
  (eq1 : a + b = d) 
  (eq2 : b + c = 6) 
  (eq3 : c + d = 7) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l762_76234


namespace NUMINAMATH_CALUDE_pauls_crayons_left_l762_76221

/-- Represents the number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 336

/-- Represents the initial number of crayons Paul got -/
def initial_crayons : ℕ := 601

/-- Represents the number of erasers Paul got -/
def erasers : ℕ := 406

theorem pauls_crayons_left :
  crayons_left = 336 ∧
  initial_crayons = 601 ∧
  erasers = 406 ∧
  erasers = crayons_left + 70 :=
by sorry

end NUMINAMATH_CALUDE_pauls_crayons_left_l762_76221


namespace NUMINAMATH_CALUDE_power_mod_eleven_l762_76284

theorem power_mod_eleven : 7^79 ≡ 6 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l762_76284


namespace NUMINAMATH_CALUDE_solve_equation_l762_76206

-- Define the ⊗ operation
def otimes (m n : ℝ) : ℝ := m^2 - 2*m*n

-- State the theorem
theorem solve_equation (x : ℝ) : 
  otimes (x + 1) (x - 2) = 5 → x = 0 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l762_76206


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l762_76252

/-- 
Given a hyperbola with equation x^2 + my^2 = 1, where m is a real number,
if the length of the imaginary axis is twice the length of the real axis,
then m = -1/4.
-/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, x^2 + m*y^2 = 1) →
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ b = 2*a ∧ 
    ∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →
  m = -1/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l762_76252


namespace NUMINAMATH_CALUDE_candy_bar_cost_is_131_l762_76225

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The number of quarters John uses -/
def num_quarters : ℕ := 4

/-- The number of dimes John uses -/
def num_dimes : ℕ := 3

/-- The number of nickels John uses -/
def num_nickels : ℕ := 1

/-- The amount of change John receives in cents -/
def change_received : ℕ := 4

/-- The cost of the candy bar in cents -/
def candy_bar_cost : ℕ := 
  num_quarters * quarter_value + 
  num_dimes * dime_value + 
  num_nickels * nickel_value - 
  change_received

theorem candy_bar_cost_is_131 : candy_bar_cost = 131 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_is_131_l762_76225


namespace NUMINAMATH_CALUDE_arrangement_count_correct_l762_76209

/-- The number of ways to arrange 4 students into 2 out of 6 classes, with 2 students in each chosen class -/
def arrangementCount : ℕ :=
  (Nat.choose 6 2 * Nat.factorial 2 * Nat.choose 4 2) / 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct :
  arrangementCount = (Nat.choose 6 2 * Nat.factorial 2 * Nat.choose 4 2) / 2 := by
  sorry

#eval arrangementCount

end NUMINAMATH_CALUDE_arrangement_count_correct_l762_76209


namespace NUMINAMATH_CALUDE_binomial_arithmetic_sequence_implies_seven_l762_76257

def factorial (r : ℕ) : ℕ := Nat.factorial r

def binomial_coefficient (j k : ℕ) : ℕ :=
  if k ≤ j then
    factorial j / (factorial k * factorial (j - k))
  else
    0

theorem binomial_arithmetic_sequence_implies_seven (n : ℕ) 
  (h1 : n > 3)
  (h2 : ∃ d : ℕ, binomial_coefficient n 2 - binomial_coefficient n 1 = d ∧
                 binomial_coefficient n 3 - binomial_coefficient n 2 = d) :
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_binomial_arithmetic_sequence_implies_seven_l762_76257


namespace NUMINAMATH_CALUDE_total_sheets_l762_76293

def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41

theorem total_sheets : sheets_in_desk + sheets_in_backpack = 91 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_l762_76293


namespace NUMINAMATH_CALUDE_exist_three_digits_for_infinite_square_representations_l762_76235

/-- A type representing a digit (0-9) -/
def Digit := Fin 10

/-- A function that checks if a digit is nonzero -/
def isNonzeroDigit (d : Digit) : Prop := d.val ≠ 0

/-- A function that checks if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that represents a natural number as a sequence of digits -/
def toDigitSequence (n : ℕ) : List Digit := sorry

/-- A function that checks if a list of digits contains only the given three digits -/
def containsOnlyGivenDigits (seq : List Digit) (d1 d2 d3 : Digit) : Prop :=
  ∀ d ∈ seq, d = d1 ∨ d = d2 ∨ d = d3

/-- The main theorem -/
theorem exist_three_digits_for_infinite_square_representations :
  ∃ (d1 d2 d3 : Digit),
    isNonzeroDigit d1 ∧ isNonzeroDigit d2 ∧ isNonzeroDigit d3 ∧
    ∀ n : ℕ, ∃ m : ℕ, 
      isPerfectSquare m ∧ 
      containsOnlyGivenDigits (toDigitSequence m) d1 d2 d3 := by
  sorry

end NUMINAMATH_CALUDE_exist_three_digits_for_infinite_square_representations_l762_76235


namespace NUMINAMATH_CALUDE_max_pq_plus_r_for_primes_l762_76239

theorem max_pq_plus_r_for_primes (p q r : ℕ) : 
  Prime p → Prime q → Prime r → 
  p * q + q * r + r * p = 2016 → 
  p * q + r ≤ 1008 := by
sorry

end NUMINAMATH_CALUDE_max_pq_plus_r_for_primes_l762_76239


namespace NUMINAMATH_CALUDE_absolute_value_and_trig_calculation_l762_76222

theorem absolute_value_and_trig_calculation : |(-3 : ℝ)| + 2⁻¹ - Real.cos (π / 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_trig_calculation_l762_76222


namespace NUMINAMATH_CALUDE_profit_maximized_at_150_l762_76287

/-- The profit function for a company based on the number of machines -/
def profit (x : ℝ) : ℝ := -25 * x^2 + 7500 * x

/-- Theorem stating that the profit is maximized when x = 150 -/
theorem profit_maximized_at_150 :
  ∃ (x_max : ℝ), x_max = 150 ∧ ∀ (x : ℝ), profit x ≤ profit x_max :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_150_l762_76287


namespace NUMINAMATH_CALUDE_remainder_problem_l762_76223

theorem remainder_problem (n : ℕ) (h1 : n % 13 = 11) (h2 : n = 349) : n % 17 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l762_76223


namespace NUMINAMATH_CALUDE_set_operations_and_inclusion_l762_76204

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a - 1}

theorem set_operations_and_inclusion (a : ℝ) :
  (a = 7/2 → M ∪ N a = {x | -2 ≤ x ∧ x ≤ 6} ∧
             (Set.univ \ M) ∩ N a = {x | 5 < x ∧ x ≤ 6}) ∧
  (M ⊇ N a ↔ a ∈ Set.Iic 3) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_inclusion_l762_76204


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l762_76288

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def A : Set Nat := {3, 4, 5, 6}
def B : Set Nat := {5, 6, 7, 8, 9}

theorem complement_intersection_theorem :
  (Aᶜ ∩ B) = {7, 8, 9} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l762_76288


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l762_76201

theorem volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l762_76201


namespace NUMINAMATH_CALUDE_quadratic_root_form_n_l762_76216

/-- Represents a quadratic equation ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Represents the form (m ± √n) / p for roots of a quadratic equation -/
structure RootForm where
  m : ℤ
  n : ℕ
  p : ℤ

/-- Check if the given RootForm satisfies the conditions for the quadratic equation -/
def isValidRootForm (eq : QuadraticEquation) (rf : RootForm) : Prop :=
  ∃ (x : ℚ), (eq.a * x^2 + eq.b * x + eq.c = 0) ∧
              (x = (rf.m + Real.sqrt rf.n) / rf.p ∨ x = (rf.m - Real.sqrt rf.n) / rf.p) ∧
              Nat.gcd (Nat.gcd rf.m.natAbs rf.n) rf.p.natAbs = 1

theorem quadratic_root_form_n (eq : QuadraticEquation) (rf : RootForm) :
  eq = QuadraticEquation.mk 3 (-7) 2 →
  isValidRootForm eq rf →
  rf.n = 25 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_form_n_l762_76216


namespace NUMINAMATH_CALUDE_transport_tax_calculation_l762_76294

def calculate_transport_tax (engine_power : ℕ) (tax_rate : ℕ) (ownership_months : ℕ) : ℕ :=
  (engine_power * tax_rate * ownership_months) / 12

theorem transport_tax_calculation :
  calculate_transport_tax 250 75 2 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_transport_tax_calculation_l762_76294


namespace NUMINAMATH_CALUDE_sum_of_smallest_solutions_l762_76214

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define our equation
def equation (x : ℝ) : Prop := x - floor x = 1 / (floor x : ℝ)

-- Define a function to check if a real number is a solution
def is_solution (x : ℝ) : Prop := equation x ∧ x > 0

-- State the theorem
theorem sum_of_smallest_solutions :
  ∃ (s1 s2 s3 : ℝ),
    is_solution s1 ∧ is_solution s2 ∧ is_solution s3 ∧
    (∀ (x : ℝ), is_solution x → x ≥ s1) ∧
    (∀ (x : ℝ), is_solution x ∧ x ≠ s1 → x ≥ s2) ∧
    (∀ (x : ℝ), is_solution x ∧ x ≠ s1 ∧ x ≠ s2 → x ≥ s3) ∧
    s1 + s2 + s3 = 10 + 1/12 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_smallest_solutions_l762_76214


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l762_76286

theorem student_congress_sample_size 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (selected_students : ℕ) 
  (h1 : num_classes = 40) 
  (h2 : students_per_class = 50) 
  (h3 : selected_students = 150) : 
  selected_students = 150 := by
sorry

end NUMINAMATH_CALUDE_student_congress_sample_size_l762_76286


namespace NUMINAMATH_CALUDE_inequalities_proof_l762_76242

theorem inequalities_proof (a b : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : a + b > 0) :
  a / b > -1 ∧ |a| < |b| := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l762_76242


namespace NUMINAMATH_CALUDE_number_of_candies_bought_l762_76275

/-- Given the cost of snacks and candies, the total number of items, and the total amount spent,
    prove that the number of candies bought is 3. -/
theorem number_of_candies_bought
  (snack_cost : ℕ)
  (candy_cost : ℕ)
  (total_items : ℕ)
  (total_spent : ℕ)
  (h1 : snack_cost = 300)
  (h2 : candy_cost = 500)
  (h3 : total_items = 8)
  (h4 : total_spent = 3000)
  : ∃ (num_candies : ℕ), num_candies = 3 ∧
    ∃ (num_snacks : ℕ),
      num_snacks + num_candies = total_items ∧
      num_snacks * snack_cost + num_candies * candy_cost = total_spent :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_candies_bought_l762_76275


namespace NUMINAMATH_CALUDE_prob_not_late_prob_late_and_miss_bus_l762_76262

-- Define the probability of Sam being late
def prob_late : ℚ := 5/9

-- Define the probability of Sam missing the bus if late
def prob_miss_bus_if_late : ℚ := 1/3

-- Theorem 1: Probability that Sam is not late
theorem prob_not_late : 1 - prob_late = 4/9 := by sorry

-- Theorem 2: Probability that Sam is late and misses the bus
theorem prob_late_and_miss_bus : prob_late * prob_miss_bus_if_late = 5/27 := by sorry

end NUMINAMATH_CALUDE_prob_not_late_prob_late_and_miss_bus_l762_76262


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_plus_1_l762_76238

theorem rationalize_denominator_sqrt3_plus_1 :
  (1 : ℝ) / (Real.sqrt 3 + 1) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt3_plus_1_l762_76238


namespace NUMINAMATH_CALUDE_least_even_integer_for_300p_perfect_square_l762_76205

theorem least_even_integer_for_300p_perfect_square :
  ∀ p : ℕ,
    p % 2 = 0 →
    (∃ n : ℕ, 300 * p = n^2) →
    p ≥ 18 :=
by sorry

end NUMINAMATH_CALUDE_least_even_integer_for_300p_perfect_square_l762_76205


namespace NUMINAMATH_CALUDE_range_when_proposition_false_l762_76264

theorem range_when_proposition_false (x : ℝ) :
  x^2 - 5*x + 4 ≤ 0 ↔ 1 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_when_proposition_false_l762_76264


namespace NUMINAMATH_CALUDE_functions_and_tangent_line_l762_76212

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x
def g (b c : ℝ) (x : ℝ) : ℝ := b * x^2 + c

-- Define the point P
def P : ℝ × ℝ := (2, 0)

-- State the theorem
theorem functions_and_tangent_line 
  (a b c : ℝ) 
  (h1 : f a P.1 = P.2) 
  (h2 : g b c P.1 = P.2) 
  (h3 : (deriv (f a)) P.1 = (deriv (g b c)) P.1) :
  (∃ (k : ℝ), 
    (∀ x, f a x = 2 * x^3 - 8 * x) ∧ 
    (∀ x, g b c x = 4 * x^2 - 16) ∧
    (∀ x y, k * x - y - k * P.1 + P.2 = 0 ↔ y = (deriv (f a)) P.1 * (x - P.1) + P.2)) :=
sorry

end

end NUMINAMATH_CALUDE_functions_and_tangent_line_l762_76212


namespace NUMINAMATH_CALUDE_product_sum_relation_l762_76249

theorem product_sum_relation (a b m : ℝ) : 
  a * b = m * (a + b) + 12 → b = 10 → b - a = 6 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l762_76249


namespace NUMINAMATH_CALUDE_cleaning_room_time_l762_76282

/-- Represents the time spent on various tasks in minutes -/
structure TaskTimes where
  total : ℕ
  laundry : ℕ
  bathroom : ℕ
  homework : ℕ

/-- Calculates the time spent cleaning the room given the other task times -/
def timeCleaningRoom (t : TaskTimes) : ℕ :=
  t.total - (t.laundry + t.bathroom + t.homework)

/-- Theorem stating that given the specific task times, the time spent cleaning the room is 35 minutes -/
theorem cleaning_room_time :
  let t : TaskTimes := {
    total := 120,
    laundry := 30,
    bathroom := 15,
    homework := 40
  }
  timeCleaningRoom t = 35 := by sorry

end NUMINAMATH_CALUDE_cleaning_room_time_l762_76282


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l762_76292

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given the initial conditions of their ages.
-/
theorem mans_age_twice_sons (man_age son_age : ℕ) (y : ℕ) : 
  man_age = son_age + 26 →
  son_age = 24 →
  man_age + y = 2 * (son_age + y) →
  y = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l762_76292


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l762_76247

theorem max_value_of_trig_function :
  ∃ (M : ℝ), M = Real.sqrt 5 / 2 ∧ 
  (∀ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x * Real.cos x - 1 ≤ M) ∧
  (∃ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x * Real.cos x - 1 = M) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l762_76247


namespace NUMINAMATH_CALUDE_fraction_zero_l762_76268

theorem fraction_zero (x : ℝ) (h : x ≠ -3) : (x^2 - 9) / (x + 3) = 0 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l762_76268


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l762_76267

theorem muffin_banana_price_ratio :
  ∀ (m b : ℝ), m > 0 → b > 0 →
  (5 * m + 4 * b) * 3 = 3 * m + 20 * b →
  m / b = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l762_76267


namespace NUMINAMATH_CALUDE_composite_probability_l762_76246

/-- The number of sides on a standard die -/
def standard_die_sides : ℕ := 6

/-- The number of sides on the special die -/
def special_die_sides : ℕ := 10

/-- The number of standard dice -/
def num_standard_dice : ℕ := 5

/-- The total number of dice -/
def total_dice : ℕ := num_standard_dice + 1

/-- The total number of possible outcomes -/
def total_outcomes : ℕ := standard_die_sides ^ num_standard_dice * special_die_sides

/-- The number of outcomes where the product is not composite -/
def non_composite_outcomes : ℕ := 25

/-- The probability of rolling a composite product -/
def prob_composite : ℚ := 1 - (non_composite_outcomes : ℚ) / total_outcomes

theorem composite_probability : prob_composite = 77735 / 77760 := by
  sorry

end NUMINAMATH_CALUDE_composite_probability_l762_76246


namespace NUMINAMATH_CALUDE_difference_of_squares_l762_76280

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l762_76280


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l762_76233

theorem fixed_point_on_line (m : ℝ) : 
  2 * (1/2 : ℝ) + m * ((1/2 : ℝ) - (1/2 : ℝ)) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l762_76233


namespace NUMINAMATH_CALUDE_point_transformation_l762_76255

def rotate90 (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflect_y_eq_x x₁ y₁
  (x₂ = 7 ∧ y₂ = -4) → d - c = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l762_76255


namespace NUMINAMATH_CALUDE_lower_limit_of_x_l762_76250

theorem lower_limit_of_x (n x y : ℤ) : 
  x > n → 
  x < 8 → 
  y > 8 → 
  y < 13 → 
  (∀ a b : ℤ, a > n ∧ a < 8 ∧ b > 8 ∧ b < 13 → b - a ≤ 7) → 
  (∃ a b : ℤ, a > n ∧ a < 8 ∧ b > 8 ∧ b < 13 ∧ b - a = 7) → 
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_lower_limit_of_x_l762_76250


namespace NUMINAMATH_CALUDE_sufficient_condition_for_square_inequality_l762_76285

theorem sufficient_condition_for_square_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) →
  (a ≥ 5 → (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0)) ∧
  ¬(∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_square_inequality_l762_76285


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l762_76299

/-- The initial capacity of the fuel tank in liters -/
def initial_capacity : ℝ := 3000

/-- The amount of fuel remaining on January 1, 2006 in liters -/
def remaining_jan1 : ℝ := 180

/-- The amount of fuel remaining on May 1, 2006 in liters -/
def remaining_may1 : ℝ := 1238

/-- The total volume of fuel used from November 1, 2005 to May 1, 2006 in liters -/
def total_fuel_used : ℝ := 4582

/-- Proof that the initial capacity of the fuel tank is 3000 liters -/
theorem fuel_tank_capacity : 
  initial_capacity = 
    (total_fuel_used + remaining_may1 + remaining_jan1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l762_76299


namespace NUMINAMATH_CALUDE_square_area_proof_l762_76245

theorem square_area_proof (side_length : ℝ) (h1 : side_length > 0) : 
  (3 * 4 * side_length - (2 * side_length + 2 * (3 * side_length)) = 28) → 
  side_length^2 = 49 := by
  sorry

#check square_area_proof

end NUMINAMATH_CALUDE_square_area_proof_l762_76245


namespace NUMINAMATH_CALUDE_range_of_m_l762_76243

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : 1/x + 4/y = 1) (h_solvable : ∃ m : ℝ, x + y/4 < m^2 - 3*m) :
  ∀ m : ℝ, (x + y/4 < m^2 - 3*m) → (m < -1 ∨ m > 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l762_76243


namespace NUMINAMATH_CALUDE_max_value_of_squared_differences_l762_76244

theorem max_value_of_squared_differences (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 = 10) : 
  (∃ (x : ℝ), (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ x) ∧ 
  (∀ (y : ℝ), (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ y → 40 ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_squared_differences_l762_76244


namespace NUMINAMATH_CALUDE_equation_solution_l762_76219

theorem equation_solution :
  let f (x : ℝ) := (x^2 - 11*x + 24)/(x-3) + (4*x^2 + 20*x - 32)/(2*x - 4)
  ∃ x₁ x₂ : ℝ, 
    x₁ = (-15 - Real.sqrt 417) / 4 ∧
    x₂ = (-15 + Real.sqrt 417) / 4 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l762_76219


namespace NUMINAMATH_CALUDE_function_max_min_sum_l762_76270

/-- Given a function f and a positive real number t, 
    this theorem states that if the sum of the maximum and minimum values of f is 4, 
    then t must equal 2. -/
theorem function_max_min_sum (t : ℝ) (h1 : t > 0) : 
  let f : ℝ → ℝ := λ x ↦ (t*x^2 + 2*x + t^2 + Real.sin x) / (x^2 + t)
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∀ x, f x ≥ N) ∧ (M + N = 4) → t = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_max_min_sum_l762_76270


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l762_76289

-- Define the points and quadrilateral
variable (A B C D P Q R S X Y : Point₂)

-- Define the cyclic quadrilateral property
def is_cyclic_quadrilateral (A B C D : Point₂) : Prop := sorry

-- Define the property of opposite sides not being parallel
def opposite_sides_not_parallel (A B C D : Point₂) : Prop := sorry

-- Define the interior point property
def is_interior_point (P : Point₂) (A B : Point₂) : Prop := sorry

-- Define angle equality
def angle_eq (A B C D E F : Point₂) : Prop := sorry

-- Define line intersection
def intersects_at (A B C D X : Point₂) : Prop := sorry

-- Define parallel or coincident lines
def parallel_or_coincide (A B C D : Point₂) : Prop := sorry

-- Theorem statement
theorem cyclic_quadrilateral_theorem 
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_not_parallel : opposite_sides_not_parallel A B C D)
  (h_P_interior : is_interior_point P A B)
  (h_Q_interior : is_interior_point Q B C)
  (h_R_interior : is_interior_point R C D)
  (h_S_interior : is_interior_point S D A)
  (h_angle1 : angle_eq P D A P C B)
  (h_angle2 : angle_eq Q A B Q D C)
  (h_angle3 : angle_eq R B C R A D)
  (h_angle4 : angle_eq S C D S B A)
  (h_intersect1 : intersects_at A Q B S X)
  (h_intersect2 : intersects_at D Q C S Y) :
  parallel_or_coincide P R X Y :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_theorem_l762_76289


namespace NUMINAMATH_CALUDE_percy_swims_52_hours_l762_76283

/-- Represents Percy's swimming schedule and calculates total swimming hours --/
def percy_swimming_hours : ℕ :=
  let weekday_hours := 2  -- 1 hour before school + 1 hour after school
  let weekdays_per_week := 5
  let weekend_hours := 3
  let weeks := 4
  let weekly_hours := weekday_hours * weekdays_per_week + weekend_hours
  weekly_hours * weeks

/-- Theorem stating that Percy swims 52 hours over 4 weeks --/
theorem percy_swims_52_hours : percy_swimming_hours = 52 := by
  sorry

end NUMINAMATH_CALUDE_percy_swims_52_hours_l762_76283


namespace NUMINAMATH_CALUDE_max_value_of_f_l762_76236

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 3 * Real.tan x) * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 3 ∧
  (∀ x, x ∈ Set.Icc 0 (Real.pi / 6) → f x ≤ M) ∧
  (∃ x, x ∈ Set.Icc 0 (Real.pi / 6) ∧ f x = M) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l762_76236


namespace NUMINAMATH_CALUDE_closed_polygonal_line_even_segments_l762_76208

/-- Represents a segment of the polygonal line -/
structure Segment where
  x : Int
  y : Int

/-- Represents a closed polygonal line on a grid -/
structure ClosedPolygonalLine where
  segments : List Segment
  is_closed : segments.length > 0
  same_length : ∀ s ∈ segments, s.x^2 + s.y^2 = 1
  on_grid : ∀ s ∈ segments, s.x = 0 ∨ s.y = 0

/-- The main theorem stating that the number of segments in a closed polygonal line is even -/
theorem closed_polygonal_line_even_segments (p : ClosedPolygonalLine) : 
  Even p.segments.length := by
  sorry

end NUMINAMATH_CALUDE_closed_polygonal_line_even_segments_l762_76208


namespace NUMINAMATH_CALUDE_geometric_relations_l762_76265

-- Define the types for lines and planes in space
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Notation
local infix:50 " ∥ " => parallel
local infix:50 " ⊥ " => perpendicular
local infix:50 " ∥ₚ " => parallel_plane
local infix:50 " ⊥ₚ " => perpendicular_plane
local infix:50 " ∥ₗₚ " => line_parallel_plane
local infix:50 " ⊥ₗₚ " => line_perpendicular_plane

-- Theorem statement
theorem geometric_relations (m n : Line) (α β : Plane) :
  (((m ⊥ₗₚ α) ∧ (n ⊥ₗₚ β) ∧ (α ⊥ₚ β)) → (m ⊥ n)) ∧
  (((m ⊥ₗₚ α) ∧ (n ∥ₗₚ β) ∧ (α ∥ₚ β)) → (m ⊥ n)) :=
sorry

end NUMINAMATH_CALUDE_geometric_relations_l762_76265


namespace NUMINAMATH_CALUDE_right_triangle_square_areas_l762_76260

theorem right_triangle_square_areas (P Q R : ℝ × ℝ) 
  (right_angle_Q : (Q.1 - P.1) * (R.1 - Q.1) + (Q.2 - P.2) * (R.2 - Q.2) = 0)
  (square_QR_area : (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = 144)
  (square_PR_area : (R.1 - P.1)^2 + (R.2 - P.2)^2 = 169) :
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_square_areas_l762_76260


namespace NUMINAMATH_CALUDE_circular_track_circumference_l762_76278

/-- The circumference of a circular track given two cyclists' speeds and meeting time -/
theorem circular_track_circumference (speed1 speed2 meeting_time : ℝ) 
  (h1 : speed1 = 7)
  (h2 : speed2 = 8)
  (h3 : meeting_time = 40)
  (h4 : speed1 > 0)
  (h5 : speed2 > 0)
  (h6 : meeting_time > 0) :
  speed1 * meeting_time + speed2 * meeting_time = 600 :=
by sorry

end NUMINAMATH_CALUDE_circular_track_circumference_l762_76278


namespace NUMINAMATH_CALUDE_increase_average_grades_l762_76229

theorem increase_average_grades (group_a_avg : ℝ) (group_b_avg : ℝ) 
  (group_a_size : ℕ) (group_b_size : ℕ) (student1_grade : ℝ) (student2_grade : ℝ) :
  group_a_avg = 44.2 →
  group_b_avg = 38.8 →
  group_a_size = 10 →
  group_b_size = 10 →
  student1_grade = 41 →
  student2_grade = 44 →
  let new_group_a_avg := (group_a_avg * group_a_size - student1_grade - student2_grade) / (group_a_size - 2)
  let new_group_b_avg := (group_b_avg * group_b_size + student1_grade + student2_grade) / (group_b_size + 2)
  new_group_a_avg > group_a_avg ∧ new_group_b_avg > group_b_avg := by
  sorry

end NUMINAMATH_CALUDE_increase_average_grades_l762_76229


namespace NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_l762_76200

theorem negation_of_universal_positive_cubic (x : ℝ) :
  (¬ ∀ x ≥ 0, x^3 + x > 0) ↔ (∃ x ≥ 0, x^3 + x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_positive_cubic_l762_76200


namespace NUMINAMATH_CALUDE_oranges_from_first_tree_l762_76272

/-- Represents the number of oranges picked from each tree -/
structure OrangesPicked where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The total number of oranges picked is the sum of oranges from all three trees -/
def total_oranges (op : OrangesPicked) : ℕ := op.first + op.second + op.third

/-- Theorem: Given the total oranges and the number from the second and third trees, 
    we can determine the number of oranges from the first tree -/
theorem oranges_from_first_tree (op : OrangesPicked) 
  (h1 : total_oranges op = 260)
  (h2 : op.second = 60)
  (h3 : op.third = 120) :
  op.first = 80 := by
  sorry

end NUMINAMATH_CALUDE_oranges_from_first_tree_l762_76272


namespace NUMINAMATH_CALUDE_raspberry_juice_volume_l762_76263

/-- Proves that the original volume of raspberry juice is 6 quarts -/
theorem raspberry_juice_volume : ∀ (original_volume : ℚ),
  (original_volume / 12 + 1 = 3) →
  (original_volume / 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_raspberry_juice_volume_l762_76263


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l762_76256

theorem adult_ticket_cost 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (concession_cost : ℚ) 
  (total_cost : ℚ) 
  (child_ticket_cost : ℚ) 
  (h1 : num_adults = 5) 
  (h2 : num_children = 2) 
  (h3 : concession_cost = 12) 
  (h4 : total_cost = 76) 
  (h5 : child_ticket_cost = 7) :
  ∃ (adult_ticket_cost : ℚ), 
    adult_ticket_cost = 10 ∧ 
    (num_adults : ℚ) * adult_ticket_cost + 
    (num_children : ℚ) * child_ticket_cost + 
    concession_cost = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l762_76256


namespace NUMINAMATH_CALUDE_m_less_than_n_l762_76207

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_d : d ≠ 0
  h_arith : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence, M and N are defined as follows -/
def M (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a n * seq.a (n + 3)

def N (seq : ArithmeticSequence) (n : ℕ) : ℝ := seq.a (n + 1) * seq.a (n + 2)

/-- For any arithmetic sequence with non-zero common difference, M < N -/
theorem m_less_than_n (seq : ArithmeticSequence) (n : ℕ) : M seq n < N seq n := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_n_l762_76207


namespace NUMINAMATH_CALUDE_quadratic_factorization_l762_76290

theorem quadratic_factorization (x : ℝ) : x^2 + 14*x + 49 = (x + 7)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l762_76290


namespace NUMINAMATH_CALUDE_speed_in_still_water_l762_76277

/-- The speed of a man in still water given his upstream and downstream speeds -/
theorem speed_in_still_water 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (h1 : upstream_speed = 60) 
  (h2 : downstream_speed = 90) : 
  (upstream_speed + downstream_speed) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l762_76277


namespace NUMINAMATH_CALUDE_floor_difference_equals_five_l762_76240

theorem floor_difference_equals_five (n : ℤ) : (⌊(n^2 : ℚ) / 4⌋ - ⌊(n : ℚ) / 2⌋^2 = 5) ↔ (n = 11) :=
sorry

end NUMINAMATH_CALUDE_floor_difference_equals_five_l762_76240


namespace NUMINAMATH_CALUDE_inequality_holds_l762_76227

theorem inequality_holds (r s : ℝ) : 
  r ≥ 0 → s > 0 → 
  (4 * (r * s^2 + r^2 * s + 4 * s^2 + 4 * r * s)) / (r + s) ≤ 3 * r^2 * s ↔ 
  r ≥ (2 + 2 * Real.sqrt 13) / 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_holds_l762_76227


namespace NUMINAMATH_CALUDE_v_1004_eq_3036_l762_76251

/-- Defines the nth term of the sequence -/
def v (n : ℕ) : ℕ := sorry

/-- The 1004th term of the sequence is 3036 -/
theorem v_1004_eq_3036 : v 1004 = 3036 := by sorry

end NUMINAMATH_CALUDE_v_1004_eq_3036_l762_76251


namespace NUMINAMATH_CALUDE_coffee_bean_price_proof_l762_76261

/-- The price of the first type of coffee bean -/
def first_bean_price : ℝ := 33

/-- The price of the second type of coffee bean -/
def second_bean_price : ℝ := 12

/-- The total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 100

/-- The selling price of the mixture per pound -/
def mixture_price_per_pound : ℝ := 11.25

/-- The weight of each type of bean used in the mixture -/
def each_bean_weight : ℝ := 25

theorem coffee_bean_price_proof : 
  first_bean_price * each_bean_weight + 
  second_bean_price * each_bean_weight = 
  total_mixture_weight * mixture_price_per_pound :=
by sorry

end NUMINAMATH_CALUDE_coffee_bean_price_proof_l762_76261


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l762_76218

theorem at_least_one_greater_than_point_seven (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (max a (max (b^2) (1 / (a^2 + b))) : ℝ) > 0.7 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l762_76218


namespace NUMINAMATH_CALUDE_root_square_plus_inverse_square_l762_76248

theorem root_square_plus_inverse_square (m : ℝ) : 
  m^2 - 2*m - 1 = 0 → m^2 + 1/m^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_square_plus_inverse_square_l762_76248


namespace NUMINAMATH_CALUDE_complex_number_existence_l762_76253

theorem complex_number_existence : ∃ z : ℂ, (z^2).re = 5 ∧ z.im ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_existence_l762_76253


namespace NUMINAMATH_CALUDE_pond_water_after_45_days_l762_76254

def water_amount (initial_amount : ℕ) (days : ℕ) : ℕ :=
  initial_amount - days + 2 * (days / 3)

theorem pond_water_after_45_days :
  water_amount 300 45 = 285 := by
  sorry

end NUMINAMATH_CALUDE_pond_water_after_45_days_l762_76254


namespace NUMINAMATH_CALUDE_ab_value_l762_76279

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B : Set ℝ := {x | ∃ a b : ℝ, x^2 - a*x + b = 0}

-- State the theorem
theorem ab_value (a b : ℝ) :
  A ∪ B = {2, 3, 5} →
  A ∩ B = {3} →
  B = {x | x^2 - a*x + b = 0} →
  a * b = 30 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l762_76279


namespace NUMINAMATH_CALUDE_equation_roots_existence_l762_76220

theorem equation_roots_existence :
  (∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0) ∧ 
  (∃ k : ℝ, ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0 ∧ 
    z^2 - 2*|z| - (2*k + 1)^2 = 0) ∧ 
  (¬ ∃ k : ℝ, ∃! x : ℝ, x^2 - 2*|x| - (2*k + 1)^2 = 0) ∧
  (¬ ∃ k : ℝ, ∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    w^2 - 2*|w| - (2*k + 1)^2 = 0 ∧ 
    x^2 - 2*|x| - (2*k + 1)^2 = 0 ∧ 
    y^2 - 2*|y| - (2*k + 1)^2 = 0 ∧ 
    z^2 - 2*|z| - (2*k + 1)^2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_existence_l762_76220


namespace NUMINAMATH_CALUDE_atMostOneHeads_atLeastTwoHeads_mutually_exclusive_l762_76213

/-- Represents the outcome of tossing 3 coins -/
inductive CoinToss
  | HHH
  | HHT
  | HTH
  | THH
  | HTT
  | THT
  | TTH
  | TTT

/-- The sample space of all possible outcomes when tossing 3 coins -/
def sampleSpace : Set CoinToss := {CoinToss.HHH, CoinToss.HHT, CoinToss.HTH, CoinToss.THH, CoinToss.HTT, CoinToss.THT, CoinToss.TTH, CoinToss.TTT}

/-- The event "At most one heads" -/
def atMostOneHeads : Set CoinToss := {CoinToss.HTT, CoinToss.THT, CoinToss.TTH, CoinToss.TTT}

/-- The event "At least two heads" -/
def atLeastTwoHeads : Set CoinToss := {CoinToss.HHH, CoinToss.HHT, CoinToss.HTH, CoinToss.THH}

/-- Theorem stating that "At most one heads" and "At least two heads" are mutually exclusive -/
theorem atMostOneHeads_atLeastTwoHeads_mutually_exclusive : 
  atMostOneHeads ∩ atLeastTwoHeads = ∅ := by sorry

end NUMINAMATH_CALUDE_atMostOneHeads_atLeastTwoHeads_mutually_exclusive_l762_76213


namespace NUMINAMATH_CALUDE_divisibility_by_five_l762_76215

theorem divisibility_by_five (B : ℕ) : 
  B < 10 → (947 * 10 + B) % 5 = 0 ↔ B = 0 ∨ B = 5 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l762_76215


namespace NUMINAMATH_CALUDE_cost_of_apple_and_watermelon_l762_76232

/-- Represents the price of fruits in yuan per kilogram -/
structure FruitPrices where
  apple : ℝ
  watermelon : ℝ
  orange : ℝ

/-- Represents a purchase of fruits -/
structure Purchase where
  apple : ℝ
  watermelon : ℝ
  orange : ℝ
  total : ℝ

def xiaoming_purchase : Purchase :=
  { apple := 1, watermelon := 4, orange := 2, total := 27.2 }

def xiaohui_purchase : Purchase :=
  { apple := 2, watermelon := 6, orange := 2, total := 32.4 }

theorem cost_of_apple_and_watermelon (prices : FruitPrices) :
  xiaoming_purchase.apple * prices.apple +
  xiaoming_purchase.watermelon * prices.watermelon +
  xiaoming_purchase.orange * prices.orange = xiaoming_purchase.total ∧
  xiaohui_purchase.apple * prices.apple +
  xiaohui_purchase.watermelon * prices.watermelon +
  xiaohui_purchase.orange * prices.orange = xiaohui_purchase.total →
  prices.apple + 2 * prices.watermelon = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_apple_and_watermelon_l762_76232


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l762_76298

def coin_distribution (x : ℕ) : ℕ := x * (x + 1) / 2

theorem pirate_treasure_distribution (x : ℕ) :
  (coin_distribution x = 5 * x) → (x + 5 * x = 54) :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l762_76298


namespace NUMINAMATH_CALUDE_binary_digits_difference_l762_76296

-- Define a function to calculate the number of digits in the binary representation of a number
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

-- State the theorem
theorem binary_digits_difference :
  binaryDigits 1500 - binaryDigits 300 = 2 := by
  sorry

end NUMINAMATH_CALUDE_binary_digits_difference_l762_76296


namespace NUMINAMATH_CALUDE_earthquake_damage_in_usd_l762_76274

/-- Converts Euros to US Dollars based on a given exchange rate -/
def euro_to_usd (euro_amount : ℝ) (exchange_rate : ℝ) : ℝ :=
  euro_amount * exchange_rate

/-- Theorem: The earthquake damage in USD is $75,000,000 -/
theorem earthquake_damage_in_usd :
  let damage_in_euros : ℝ := 50000000
  let exchange_rate : ℝ := 3/2 -- 2 Euros = 3 USD, so 1 Euro = 3/2 USD
  euro_to_usd damage_in_euros exchange_rate = 75000000 := by
  sorry

end NUMINAMATH_CALUDE_earthquake_damage_in_usd_l762_76274


namespace NUMINAMATH_CALUDE_power_mod_50_l762_76217

theorem power_mod_50 : 11^1501 % 50 = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_50_l762_76217


namespace NUMINAMATH_CALUDE_equation_solution_l762_76276

theorem equation_solution : 
  ∃ x : ℚ, (x - 1) / 2 - (2 - x) / 3 = 2 ∧ x = 19 / 5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l762_76276


namespace NUMINAMATH_CALUDE_three_digit_squares_divisible_by_12_l762_76297

theorem three_digit_squares_divisible_by_12 :
  (∃! (l : List Nat), l = (List.range 22).filter (fun n => 
    10 ≤ n ∧ n ≤ 31 ∧ (n^2 % 12 = 0)) ∧ l.length = 4) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_squares_divisible_by_12_l762_76297


namespace NUMINAMATH_CALUDE_bob_candy_count_l762_76202

/-- Calculates Bob's share of candies given a total amount and a ratio --/
def bobShare (total : ℕ) (samRatio : ℕ) (bobRatio : ℕ) : ℕ :=
  (total / (samRatio + bobRatio)) * bobRatio

/-- The total number of candies Bob received --/
def bobTotalCandies : ℕ :=
  bobShare 45 2 3 + bobShare 60 3 1 + (45 / 2)

theorem bob_candy_count : bobTotalCandies = 64 := by
  sorry

#eval bobTotalCandies

end NUMINAMATH_CALUDE_bob_candy_count_l762_76202


namespace NUMINAMATH_CALUDE_vector_operation_l762_76271

/-- Given vectors a and b in ℝ², prove that 2b - a equals the expected result. -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (0, -1)) :
  (2 : ℝ) • b - a = (-3, -4) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l762_76271


namespace NUMINAMATH_CALUDE_sum_of_perpendiculars_equals_height_l762_76291

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  /-- Side length of the equilateral triangle -/
  a : ℝ
  /-- Height of the equilateral triangle -/
  h : ℝ
  /-- Perpendicular distance from the point to side AB -/
  m₁ : ℝ
  /-- Perpendicular distance from the point to side BC -/
  m₂ : ℝ
  /-- Perpendicular distance from the point to side CA -/
  m₃ : ℝ
  /-- The point is inside the triangle -/
  point_inside : 0 < m₁ ∧ 0 < m₂ ∧ 0 < m₃
  /-- The triangle is equilateral -/
  equilateral : h = (Real.sqrt 3 / 2) * a
  /-- The height is positive -/
  height_positive : 0 < h

/-- 
The sum of perpendiculars from any point inside an equilateral triangle 
to its sides equals the triangle's height
-/
theorem sum_of_perpendiculars_equals_height (t : EquilateralTriangleWithPoint) : 
  t.m₁ + t.m₂ + t.m₃ = t.h := by
  sorry

end NUMINAMATH_CALUDE_sum_of_perpendiculars_equals_height_l762_76291


namespace NUMINAMATH_CALUDE_one_third_recipe_sugar_l762_76231

def original_recipe : ℚ := 7 + 3/4

theorem one_third_recipe_sugar (original : ℚ) (reduced : ℚ) : 
  original = 7 + 3/4 → reduced = original * (1/3) → reduced = 2 + 7/12 := by
  sorry

end NUMINAMATH_CALUDE_one_third_recipe_sugar_l762_76231


namespace NUMINAMATH_CALUDE_x_value_l762_76259

theorem x_value (x y : ℝ) (h : x / (x - 1) = (y^3 + 2*y^2 - 2) / (y^3 + 2*y^2 - 3)) :
  x = (y^3 + 2*y^2 - 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_x_value_l762_76259


namespace NUMINAMATH_CALUDE_initial_flower_plates_is_four_l762_76241

/-- Represents the initial number of flower pattern plates Jack has. -/
def initial_flower_plates : ℕ := sorry

/-- Represents the number of checked pattern plates Jack has. -/
def checked_plates : ℕ := 8

/-- Represents the number of polka dotted plates Jack buys. -/
def polka_dotted_plates : ℕ := 2 * checked_plates

/-- Represents the total number of plates Jack has after buying polka dotted plates and smashing one flower plate. -/
def total_plates : ℕ := 27

/-- Theorem stating that the initial number of flower pattern plates is 4. -/
theorem initial_flower_plates_is_four :
  initial_flower_plates = 4 :=
by
  have h1 : initial_flower_plates + checked_plates + polka_dotted_plates - 1 = total_plates := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_flower_plates_is_four_l762_76241
