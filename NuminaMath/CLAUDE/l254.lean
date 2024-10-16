import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_l254_25444

theorem simplify_expression (b : ℝ) :
  3 * b * (3 * b^2 - 2 * b + 1) + 2 * b^2 = 9 * b^3 - 4 * b^2 + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l254_25444


namespace NUMINAMATH_CALUDE_final_chicken_count_l254_25405

def chicken_count (initial : ℕ) (second_factor : ℕ) (second_subtract : ℕ) (dog_eat : ℕ) (final_factor : ℕ) (final_subtract : ℕ) : ℕ :=
  let after_second := initial + (second_factor * initial - second_subtract)
  let after_dog := after_second - dog_eat
  let final_addition := final_factor * (final_factor * after_dog - final_subtract)
  after_dog + final_addition

theorem final_chicken_count :
  chicken_count 12 3 8 2 2 10 = 246 := by
  sorry

end NUMINAMATH_CALUDE_final_chicken_count_l254_25405


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l254_25426

-- Define the sets M and N
def M : Set ℝ := {x | |x| ≤ 3}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x | x < -3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l254_25426


namespace NUMINAMATH_CALUDE_surface_area_order_l254_25423

/-- Represents the types of geometric solids -/
inductive Solid
  | Tetrahedron
  | Cube
  | Octahedron
  | Sphere
  | Cylinder
  | Cone

/-- Computes the surface area of a solid given its volume -/
noncomputable def surfaceArea (s : Solid) (v : ℝ) : ℝ :=
  match s with
  | Solid.Tetrahedron => (216 * Real.sqrt 3) ^ (1/3) * v ^ (2/3)
  | Solid.Cube => 6 * v ^ (2/3)
  | Solid.Octahedron => (108 * Real.sqrt 3) ^ (1/3) * v ^ (2/3)
  | Solid.Sphere => (36 * Real.pi) ^ (1/3) * v ^ (2/3)
  | Solid.Cylinder => (54 * Real.pi) ^ (1/3) * v ^ (2/3)
  | Solid.Cone => (81 * Real.pi) ^ (1/3) * v ^ (2/3)

/-- Theorem stating the order of surface areas for equal volume solids -/
theorem surface_area_order (v : ℝ) (h : v > 0) :
  surfaceArea Solid.Sphere v < surfaceArea Solid.Cylinder v ∧
  surfaceArea Solid.Cylinder v < surfaceArea Solid.Octahedron v ∧
  surfaceArea Solid.Octahedron v < surfaceArea Solid.Cube v ∧
  surfaceArea Solid.Cube v < surfaceArea Solid.Cone v ∧
  surfaceArea Solid.Cone v < surfaceArea Solid.Tetrahedron v :=
by
  sorry

end NUMINAMATH_CALUDE_surface_area_order_l254_25423


namespace NUMINAMATH_CALUDE_vessel_width_calculation_l254_25414

-- Define the given parameters
def cube_edge : ℝ := 15
def vessel_length : ℝ := 20
def water_rise : ℝ := 12.053571428571429

-- Define the theorem
theorem vessel_width_calculation (w : ℝ) :
  (cube_edge ^ 3 = vessel_length * w * water_rise) →
  w = 14 := by
  sorry

end NUMINAMATH_CALUDE_vessel_width_calculation_l254_25414


namespace NUMINAMATH_CALUDE_evaluate_expression_l254_25498

theorem evaluate_expression : 2001^3 - 2000 * 2001^2 - 2000^2 * 2001 + 2 * 2000^3 = 24008004001 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l254_25498


namespace NUMINAMATH_CALUDE_bob_walking_distance_l254_25443

/-- Proves that Bob walked 4 miles before meeting Yolanda given the specified conditions -/
theorem bob_walking_distance
  (total_distance : ℝ)
  (yolanda_rate : ℝ)
  (bob_rate : ℝ)
  (time_difference : ℝ)
  (h1 : total_distance = 10)
  (h2 : yolanda_rate = 3)
  (h3 : bob_rate = 4)
  (h4 : time_difference = 1)
  : ∃ t : ℝ, t > time_difference ∧ yolanda_rate * t + bob_rate * (t - time_difference) = total_distance ∧ bob_rate * (t - time_difference) = 4 :=
by sorry

end NUMINAMATH_CALUDE_bob_walking_distance_l254_25443


namespace NUMINAMATH_CALUDE_root_implies_m_value_l254_25460

theorem root_implies_m_value (m : ℝ) : 
  (1 : ℝ)^2 + m * (1 : ℝ) - 3 = 0 → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l254_25460


namespace NUMINAMATH_CALUDE_books_left_to_read_l254_25400

def total_books : ℕ := 89
def mcgregor_finished : ℕ := 34
def floyd_finished : ℕ := 32

theorem books_left_to_read :
  total_books - (mcgregor_finished + floyd_finished) = 23 := by
sorry

end NUMINAMATH_CALUDE_books_left_to_read_l254_25400


namespace NUMINAMATH_CALUDE_polynomial_perfect_square_l254_25458

/-- The polynomial (x-1)(x+3)(x-4)(x-8)+m is a perfect square if and only if m = 196 -/
theorem polynomial_perfect_square (x m : ℝ) : 
  ∃ y : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = y^2 ↔ m = 196 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_perfect_square_l254_25458


namespace NUMINAMATH_CALUDE_sqrt_15_factorial_simplification_l254_25472

theorem sqrt_15_factorial_simplification :
  ∃ (a b : ℕ+) (q : ℚ),
    (a:ℝ) * Real.sqrt b = Real.sqrt (Nat.factorial 15) ∧
    q * (Nat.factorial 15 : ℚ) = (a * b : ℚ) ∧
    q = 1 / 30240 := by sorry

end NUMINAMATH_CALUDE_sqrt_15_factorial_simplification_l254_25472


namespace NUMINAMATH_CALUDE_power_tower_comparison_l254_25485

theorem power_tower_comparison : 3^(3^(3^3)) > 2^(2^(2^(2^2))) := by
  sorry

end NUMINAMATH_CALUDE_power_tower_comparison_l254_25485


namespace NUMINAMATH_CALUDE_least_number_divisibility_l254_25404

theorem least_number_divisibility (n : ℕ) (h1 : (n + 6) % 24 = 0) (h2 : (n + 6) % 32 = 0)
  (h3 : (n + 6) % 36 = 0) (h4 : n + 6 = 858) :
  ∃ p : ℕ, Nat.Prime p ∧ p ≠ 2 ∧ p ≠ 3 ∧ n % p = 0 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisibility_l254_25404


namespace NUMINAMATH_CALUDE_xy_sum_problem_l254_25487

theorem xy_sum_problem (x y : ℝ) 
  (h1 : 1/x + 1/y = 5)
  (h2 : x*y + x + y = 7) :
  x^2*y + x*y^2 = 245/36 := by
sorry

end NUMINAMATH_CALUDE_xy_sum_problem_l254_25487


namespace NUMINAMATH_CALUDE_square_units_digit_nine_l254_25494

theorem square_units_digit_nine (n : ℕ) : n ≤ 9 → (n^2 % 10 = 9 ↔ n = 3 ∨ n = 7) := by
  sorry

end NUMINAMATH_CALUDE_square_units_digit_nine_l254_25494


namespace NUMINAMATH_CALUDE_max_bouquet_size_l254_25478

/-- Represents the cost of a yellow tulip in rubles -/
def yellow_cost : ℕ := 50

/-- Represents the cost of a red tulip in rubles -/
def red_cost : ℕ := 31

/-- Represents the maximum budget in rubles -/
def max_budget : ℕ := 600

/-- Represents a valid bouquet of tulips -/
structure Bouquet where
  yellow : ℕ
  red : ℕ
  odd_total : Odd (yellow + red)
  color_diff : (yellow = red + 1) ∨ (red = yellow + 1)
  within_budget : yellow * yellow_cost + red * red_cost ≤ max_budget

/-- The maximum number of tulips in a bouquet -/
def max_tulips : ℕ := 15

/-- Theorem stating that the maximum number of tulips in a valid bouquet is 15 -/
theorem max_bouquet_size :
  ∀ b : Bouquet, b.yellow + b.red ≤ max_tulips ∧
  ∃ b' : Bouquet, b'.yellow + b'.red = max_tulips :=
sorry

end NUMINAMATH_CALUDE_max_bouquet_size_l254_25478


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l254_25406

/-- The coefficient of x^n in the expansion of (x^2 + a/x)^m -/
def coeff (a : ℝ) (m n : ℕ) : ℝ := sorry

theorem binomial_expansion_coefficient (a : ℝ) :
  coeff a 5 7 = -15 → a = -3 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l254_25406


namespace NUMINAMATH_CALUDE_first_cube_weight_l254_25489

/-- Given two cubical blocks of the same metal, where the second cube's sides are twice as long
    as the first cube's and weighs 48 pounds, prove that the first cube weighs 6 pounds. -/
theorem first_cube_weight (s : ℝ) (weight_first : ℝ) (weight_second : ℝ) :
  s > 0 →
  weight_second = 48 →
  weight_second / weight_first = (2 * s)^3 / s^3 →
  weight_first = 6 :=
by sorry

end NUMINAMATH_CALUDE_first_cube_weight_l254_25489


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l254_25431

-- Define a function to convert base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

-- Define a function to convert decimal to base 4
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- Define the numbers in base 4 as lists of digits
def num1 : List Nat := [1, 3, 2]  -- 231₄
def num2 : List Nat := [1, 2]     -- 21₄
def num3 : List Nat := [3]        -- 3₄
def result : List Nat := [3, 3, 0, 2]  -- 2033₄

-- State the theorem
theorem base4_multiplication_division :
  decimalToBase4 ((base4ToDecimal num1 * base4ToDecimal num2) / base4ToDecimal num3) = result := by
  sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l254_25431


namespace NUMINAMATH_CALUDE_correct_last_digit_prob_l254_25451

/-- The number of possible digits for each position in the password -/
def num_digits : ℕ := 10

/-- The probability of guessing the correct digit on the first attempt -/
def first_attempt_prob : ℚ := 1 / num_digits

/-- The probability of guessing the correct digit on the second attempt, given the first attempt was incorrect -/
def second_attempt_prob : ℚ := 1 / (num_digits - 1)

/-- The probability of guessing the correct last digit within 2 attempts -/
def two_attempt_prob : ℚ := first_attempt_prob + (1 - first_attempt_prob) * second_attempt_prob

theorem correct_last_digit_prob :
  two_attempt_prob = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_last_digit_prob_l254_25451


namespace NUMINAMATH_CALUDE_decimal_representation_of_sqrt2_plus_sqrt3_power_1980_l254_25486

theorem decimal_representation_of_sqrt2_plus_sqrt3_power_1980 :
  let x := (Real.sqrt 2 + Real.sqrt 3) ^ 1980
  ∃ y : ℝ, 0 ≤ y ∧ y < 1 ∧ x = 7 + y ∧ y > 0.9 := by
  sorry

end NUMINAMATH_CALUDE_decimal_representation_of_sqrt2_plus_sqrt3_power_1980_l254_25486


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l254_25471

theorem algebraic_expression_value (a b : ℝ) (h : a - 3*b = -3) :
  5 - a + 3*b = 8 := by sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l254_25471


namespace NUMINAMATH_CALUDE_running_increase_per_week_l254_25459

theorem running_increase_per_week 
  (initial_capacity : ℝ) 
  (increase_percentage : ℝ) 
  (days : ℕ) 
  (h1 : initial_capacity = 100)
  (h2 : increase_percentage = 0.2)
  (h3 : days = 280) :
  let new_capacity := initial_capacity * (1 + increase_percentage)
  let weeks := days / 7
  (new_capacity - initial_capacity) / weeks = 3 := by sorry

end NUMINAMATH_CALUDE_running_increase_per_week_l254_25459


namespace NUMINAMATH_CALUDE_incorrect_addition_theorem_l254_25475

/-- Represents a 6-digit number as a list of digits -/
def SixDigitNumber := List Nat

/-- Checks if a number is a valid 6-digit number -/
def isValidSixDigitNumber (n : SixDigitNumber) : Prop :=
  n.length = 6 ∧ n.all (λ d => d < 10)

/-- Converts a 6-digit number to its integer value -/
def toInt (n : SixDigitNumber) : Nat :=
  n.foldl (λ acc d => acc * 10 + d) 0

/-- Replaces all occurrences of one digit with another in a number -/
def replaceDigit (n : SixDigitNumber) (d e : Nat) : SixDigitNumber :=
  n.map (λ x => if x = d then e else x)

theorem incorrect_addition_theorem :
  ∃ (A B : SixDigitNumber) (d e : Nat),
    isValidSixDigitNumber A ∧
    isValidSixDigitNumber B ∧
    d < 10 ∧
    e < 10 ∧
    toInt A + toInt B ≠ 1061835 ∧
    toInt (replaceDigit A d e) + toInt (replaceDigit B d e) = 1061835 ∧
    d + e = 1 :=
  sorry

end NUMINAMATH_CALUDE_incorrect_addition_theorem_l254_25475


namespace NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l254_25469

theorem sin_alpha_plus_pi_fourth (α : Real) :
  (Complex.mk (Real.sin α - 3/5) (Real.cos α - 4/5)).re = 0 →
  Real.sin (α + Real.pi/4) = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_pi_fourth_l254_25469


namespace NUMINAMATH_CALUDE_survey_result_l254_25455

theorem survey_result (X : ℝ) (total : ℕ) (h_total : total ≥ 100) : ℝ :=
  let liked_A := X
  let liked_both := 23
  let liked_neither := 23
  let liked_B := 100 - X
  sorry

end NUMINAMATH_CALUDE_survey_result_l254_25455


namespace NUMINAMATH_CALUDE_functional_equation_solution_l254_25477

theorem functional_equation_solution 
  (f g h : ℝ → ℝ) 
  (hf : Continuous f) 
  (hg : Continuous g) 
  (hh : Continuous h) 
  (h_eq : ∀ x y, f (x + y) = g x + h y) :
  ∃ a b c : ℝ, 
    (∀ x, f x = c * x + a + b) ∧
    (∀ x, g x = c * x + a) ∧
    (∀ x, h x = c * x + b) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l254_25477


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l254_25401

/-- Given a function f(x) = ax³ + bx + 1, prove that if f(a) = 8, then f(-a) = -6 -/
theorem function_value_at_negative_a (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + 1
  f a = 8 → f (-a) = -6 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l254_25401


namespace NUMINAMATH_CALUDE_product_of_squared_fractions_l254_25490

theorem product_of_squared_fractions : (1/3 * 9)^2 * (1/27 * 81)^2 * (1/243 * 729)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_product_of_squared_fractions_l254_25490


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l254_25441

/-- A regular polygon with an interior angle of 150° has 12 sides -/
theorem regular_polygon_with_150_degree_interior_angle_has_12_sides :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 150 →
    interior_angle = (n - 2) * 180 / n →
    n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_interior_angle_has_12_sides_l254_25441


namespace NUMINAMATH_CALUDE_cereal_spending_l254_25463

theorem cereal_spending (total : ℝ) (snap crackle pop : ℝ) : 
  total = 150 ∧ 
  snap = 2 * crackle ∧ 
  crackle = 3 * pop ∧ 
  total = snap + crackle + pop → 
  pop = 15 := by
  sorry

end NUMINAMATH_CALUDE_cereal_spending_l254_25463


namespace NUMINAMATH_CALUDE_katy_june_books_l254_25402

/-- The number of books Katy read in June -/
def june_books : ℕ := sorry

/-- The number of books Katy read in July -/
def july_books : ℕ := 2 * june_books

/-- The number of books Katy read in August -/
def august_books : ℕ := july_books - 3

/-- The total number of books Katy read during the summer -/
def total_books : ℕ := 37

theorem katy_june_books :
  june_books + july_books + august_books = total_books ∧ june_books = 8 := by sorry

end NUMINAMATH_CALUDE_katy_june_books_l254_25402


namespace NUMINAMATH_CALUDE_largest_four_digit_sum_20_l254_25439

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem largest_four_digit_sum_20 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 20 → n ≤ 9920 :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_sum_20_l254_25439


namespace NUMINAMATH_CALUDE_concert_revenue_l254_25448

def adult_price : ℕ := 26
def teenager_price : ℕ := 18
def children_price : ℕ := adult_price / 2

def num_adults : ℕ := 183
def num_teenagers : ℕ := 75
def num_children : ℕ := 28

def total_revenue : ℕ := 
  num_adults * adult_price + 
  num_teenagers * teenager_price + 
  num_children * children_price

theorem concert_revenue : total_revenue = 6472 := by
  sorry

end NUMINAMATH_CALUDE_concert_revenue_l254_25448


namespace NUMINAMATH_CALUDE_beth_twice_sister_age_l254_25462

/-- 
Given:
- Beth is currently 18 years old
- Beth's sister is currently 5 years old

Prove that the number of years until Beth is twice her sister's age is 8.
-/
theorem beth_twice_sister_age (beth_age : ℕ) (sister_age : ℕ) : 
  beth_age = 18 → sister_age = 5 → (beth_age + 8 = 2 * (sister_age + 8)) := by
  sorry

end NUMINAMATH_CALUDE_beth_twice_sister_age_l254_25462


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l254_25453

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a - b) →
  (10 * a + b) + (10 * b + a) = 33 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l254_25453


namespace NUMINAMATH_CALUDE_square_difference_l254_25468

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l254_25468


namespace NUMINAMATH_CALUDE_student_marks_average_l254_25496

theorem student_marks_average (P C M : ℕ) (h : P + C + M = P + 150) :
  (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_average_l254_25496


namespace NUMINAMATH_CALUDE_quadratic_sum_constrained_l254_25403

theorem quadratic_sum_constrained (p q r s : ℝ) 
  (sum_condition : p + q + r + s = 10) 
  (sum_squares_condition : p^2 + q^2 + r^2 + s^2 = 20) : 
  3 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) = 64 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_constrained_l254_25403


namespace NUMINAMATH_CALUDE_sum_equals_three_fourths_l254_25436

theorem sum_equals_three_fourths : 
  let original_sum := (1/3 : ℚ) + 1/6 + 1/9 + 1/12 + 1/15 + 1/18 + 1/21
  let removed_terms := (1/12 : ℚ) + 1/21
  original_sum - removed_terms = 3/4 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_three_fourths_l254_25436


namespace NUMINAMATH_CALUDE_two_correct_probability_l254_25457

/-- The number of houses and packages -/
def n : ℕ := 4

/-- The probability of exactly two packages being delivered to the correct houses -/
def prob_two_correct : ℚ := 1/4

/-- Theorem stating that the probability of exactly two packages being delivered 
    to the correct houses out of four is 1/4 -/
theorem two_correct_probability : 
  (n.choose 2 : ℚ) * (1/n) * (1/(n-1)) * (1/2) = prob_two_correct := by
  sorry

end NUMINAMATH_CALUDE_two_correct_probability_l254_25457


namespace NUMINAMATH_CALUDE_bobby_jump_difference_l254_25417

/-- The number of jumps Bobby can do per minute as a child -/
def child_jumps_per_minute : ℕ := 30

/-- The number of jumps Bobby can do per second as an adult -/
def adult_jumps_per_second : ℕ := 1

/-- The number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- The difference in jumps per minute between Bobby as an adult and as a child -/
theorem bobby_jump_difference : 
  adult_jumps_per_second * seconds_per_minute - child_jumps_per_minute = 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_jump_difference_l254_25417


namespace NUMINAMATH_CALUDE_numbers_satisfying_conditions_l254_25481

def satisfies_conditions (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 1000 ∧
  ∃ k : ℕ, n = 7 * k ∧
  ((∃ m : ℕ, n = 4 * m + 3) ∨ (∃ m : ℕ, n = 9 * m + 3))

theorem numbers_satisfying_conditions :
  {n : ℕ | satisfies_conditions n} = {147, 399, 651, 903} := by
  sorry

end NUMINAMATH_CALUDE_numbers_satisfying_conditions_l254_25481


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_18_div_5_l254_25434

/-- The equation has no solutions if and only if k = 18/5 -/
theorem no_solution_iff_k_eq_18_div_5 :
  let v1 : Fin 2 → ℝ := ![1, 3]
  let v2 : Fin 2 → ℝ := ![5, -9]
  let v3 : Fin 2 → ℝ := ![4, 0]
  let v4 : Fin 2 → ℝ := ![-2, k]
  (∀ t s : ℝ, v1 + t • v2 ≠ v3 + s • v4) ↔ k = 18/5 := by
sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_18_div_5_l254_25434


namespace NUMINAMATH_CALUDE_unique_number_with_divisor_sum_power_of_ten_l254_25442

theorem unique_number_with_divisor_sum_power_of_ten (N : ℕ) : 
  (∃ m : ℕ, m < N ∧ m ∣ N ∧ (∀ d : ℕ, d < N → d ∣ N → d ≤ m) ∧ 
   (∃ k : ℕ, N + m = 10^k)) → N = 75 := by
sorry

end NUMINAMATH_CALUDE_unique_number_with_divisor_sum_power_of_ten_l254_25442


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l254_25467

/-- A geometric sequence with given third and sixth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_3 : a 3 = 8)
  (h_6 : a 6 = 64) :
  ∃ (q : ℝ), (∀ (n : ℕ), a (n + 1) = a n * q) ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l254_25467


namespace NUMINAMATH_CALUDE_product_mod_seven_l254_25452

theorem product_mod_seven : (2009 * 2010 * 2011 * 2012) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l254_25452


namespace NUMINAMATH_CALUDE_spearman_correlation_approx_l254_25461

def scores_A : List ℝ := [95, 90, 86, 84, 75, 70, 62, 60, 57, 50]
def scores_B : List ℝ := [92, 93, 83, 80, 55, 60, 45, 72, 62, 70]

def spearman_rank_correlation (x y : List ℝ) : ℝ :=
  sorry

theorem spearman_correlation_approx :
  ∃ ε > 0, ε < 0.01 ∧ |spearman_rank_correlation scores_A scores_B - 0.64| < ε :=
by sorry

end NUMINAMATH_CALUDE_spearman_correlation_approx_l254_25461


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l254_25408

/-- Given that the binomial coefficients of the third and seventh terms
    in the expansion of (x+2)^n are equal, prove that n = 8 and
    the coefficient of the (k+1)th term is maximum when k = 5 or k = 6 -/
theorem binomial_expansion_properties (n : ℕ) :
  (Nat.choose n 2 = Nat.choose n 6) →
  (n = 8 ∧ 
   (∀ j : ℕ, j ≠ 5 ∧ j ≠ 6 → 
     Nat.choose 8 5 * 2^5 ≥ Nat.choose 8 j * 2^j ∧
     Nat.choose 8 6 * 2^6 ≥ Nat.choose 8 j * 2^j)) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l254_25408


namespace NUMINAMATH_CALUDE_problem_solution_l254_25418

theorem problem_solution :
  (∀ x : ℝ, x^2 - 5*x + 4 < 0 ∧ (x - 2)*(x - 5) < 0 ↔ 2 < x ∧ x < 4) ∧
  (∀ a : ℝ, a > 0 ∧ 
    (∀ x : ℝ, (x - 2)*(x - 5) < 0 → x^2 - 5*a*x + 4*a^2 < 0) ∧ 
    (∃ x : ℝ, x^2 - 5*a*x + 4*a^2 < 0 ∧ (x - 2)*(x - 5) ≥ 0) ↔
    5/4 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l254_25418


namespace NUMINAMATH_CALUDE_circle_equation_l254_25413

/-- The standard equation of a circle with center (2, -2) passing through the origin -/
theorem circle_equation : ∀ (x y : ℝ), 
  (x - 2)^2 + (y + 2)^2 = 8 ↔ 
  (x - 2)^2 + (y + 2)^2 = (2 - 0)^2 + (-2 - 0)^2 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_l254_25413


namespace NUMINAMATH_CALUDE_solve_system_for_q_l254_25422

theorem solve_system_for_q (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 20) 
  (eq2 : 6 * p + 5 * q = 29) : 
  q = -25 / 11 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_q_l254_25422


namespace NUMINAMATH_CALUDE_negative_sqrt_product_l254_25488

theorem negative_sqrt_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  -Real.sqrt a * Real.sqrt b = -Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_product_l254_25488


namespace NUMINAMATH_CALUDE_donut_distribution_l254_25415

/-- The number of ways to distribute n identical objects into k distinct boxes,
    with each box containing at least m objects. -/
def distributionWays (n k m : ℕ) : ℕ := sorry

/-- The theorem stating that there are 10 ways to distribute 10 donuts
    into 4 kinds with at least 2 of each kind. -/
theorem donut_distribution : distributionWays 10 4 2 = 10 := by sorry

end NUMINAMATH_CALUDE_donut_distribution_l254_25415


namespace NUMINAMATH_CALUDE_complex_number_problem_l254_25419

-- Define the complex number z
variable (z : ℂ)

-- Define the conditions
def condition1 : Prop := (z + 3 + 4 * Complex.I).im = 0
def condition2 : Prop := (z / (1 - 2 * Complex.I)).im = 0
def condition3 (m : ℝ) : Prop := 
  let w := (z - m * Complex.I)^2
  w.re < 0 ∧ w.im > 0

-- State the theorem
theorem complex_number_problem (h1 : condition1 z) (h2 : condition2 z) :
  z = 2 - 4 * Complex.I ∧ 
  ∃ m₀ : ℝ, ∀ m : ℝ, condition3 z m ↔ m < m₀ ∧ m₀ = -6 :=
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l254_25419


namespace NUMINAMATH_CALUDE_pr_qs_ratio_l254_25412

-- Define the points and distances
def P : ℝ := 0
def Q : ℝ := 3
def R : ℝ := 10
def S : ℝ := 18

-- State the theorem
theorem pr_qs_ratio :
  (R - P) / (S - Q) = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_pr_qs_ratio_l254_25412


namespace NUMINAMATH_CALUDE_min_value_is_three_l254_25425

/-- A quadratic function f(x) = ax² + bx + c where b > a and f(x) ≥ 0 for all x ∈ ℝ -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : b > a
  h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The minimum value of (a+b+c)/(b-a) for a QuadraticFunction is 3 -/
theorem min_value_is_three (f : QuadraticFunction) : 
  (∀ x : ℝ, (f.a + f.b + f.c) / (f.b - f.a) ≥ 3) ∧ 
  (∃ x : ℝ, (f.a + f.b + f.c) / (f.b - f.a) = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_three_l254_25425


namespace NUMINAMATH_CALUDE_counterexample_exists_l254_25410

theorem counterexample_exists : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a * b ≤ c ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l254_25410


namespace NUMINAMATH_CALUDE_cube_root_125_fourth_root_256_square_root_16_l254_25483

theorem cube_root_125_fourth_root_256_square_root_16 : 
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_fourth_root_256_square_root_16_l254_25483


namespace NUMINAMATH_CALUDE_pencil_count_l254_25432

theorem pencil_count (mitchell_pencils : ℕ) (difference : ℕ) : mitchell_pencils = 30 → difference = 6 →
  mitchell_pencils + (mitchell_pencils - difference) = 54 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l254_25432


namespace NUMINAMATH_CALUDE_election_percentage_l254_25495

theorem election_percentage (total_votes : ℝ) (candidate_votes : ℝ) 
  (h1 : candidate_votes > 0)
  (h2 : total_votes > candidate_votes)
  (h3 : candidate_votes + (1/3) * (total_votes - candidate_votes) = (1/2) * total_votes) :
  candidate_votes / total_votes = 1/4 := by
sorry

end NUMINAMATH_CALUDE_election_percentage_l254_25495


namespace NUMINAMATH_CALUDE_assignment_schemes_l254_25466

def number_of_roles : ℕ := 5
def number_of_members : ℕ := 5

def roles_for_A : ℕ := number_of_roles - 2
def roles_for_B : ℕ := 1
def remaining_members : ℕ := number_of_members - 2
def remaining_roles : ℕ := number_of_roles - 2

theorem assignment_schemes :
  (roles_for_B) * (roles_for_A) * (remaining_members.factorial) = 18 := by
  sorry

end NUMINAMATH_CALUDE_assignment_schemes_l254_25466


namespace NUMINAMATH_CALUDE_a_plus_b_equals_one_l254_25411

theorem a_plus_b_equals_one (a b : ℝ) (h : |a^3 - 27| + (b + 2)^2 = 0) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_one_l254_25411


namespace NUMINAMATH_CALUDE_myrtle_final_eggs_l254_25456

/-- Calculate the number of eggs Myrtle has after her trip --/
def myrtle_eggs (num_hens : ℕ) (eggs_per_hen_per_day : ℕ) (days_away : ℕ) 
                (eggs_taken_by_neighbor : ℕ) (eggs_dropped : ℕ) : ℕ :=
  num_hens * eggs_per_hen_per_day * days_away - eggs_taken_by_neighbor - eggs_dropped

/-- Theorem stating the number of eggs Myrtle has --/
theorem myrtle_final_eggs : 
  myrtle_eggs 3 3 7 12 5 = 46 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_final_eggs_l254_25456


namespace NUMINAMATH_CALUDE_remainder_3_100_mod_7_l254_25421

theorem remainder_3_100_mod_7 : 3^100 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_100_mod_7_l254_25421


namespace NUMINAMATH_CALUDE_taxi_distribution_eq_14_l254_25499

/-- The number of ways to distribute 4 people into 2 taxis with at least one person in each taxi -/
def taxi_distribution : ℕ :=
  2^4 - 2

/-- Theorem stating that the number of ways to distribute 4 people into 2 taxis
    with at least one person in each taxi is equal to 14 -/
theorem taxi_distribution_eq_14 : taxi_distribution = 14 := by
  sorry

end NUMINAMATH_CALUDE_taxi_distribution_eq_14_l254_25499


namespace NUMINAMATH_CALUDE_max_value_theorem_l254_25492

theorem max_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + 3 * b^2)) ≤ a^2 + 3 * b^2) ∧
  (∃ x : ℝ, 2 * (a - x) * (x + Real.sqrt (x^2 + 3 * b^2)) = a^2 + 3 * b^2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l254_25492


namespace NUMINAMATH_CALUDE_olympiad_sheet_distribution_l254_25482

theorem olympiad_sheet_distribution (n : ℕ) :
  let initial_total := 2 + 3 + 1 + 1
  let final_total := initial_total + 2 * n
  ¬ ∃ (k : ℕ), final_total = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_olympiad_sheet_distribution_l254_25482


namespace NUMINAMATH_CALUDE_train_distance_problem_l254_25409

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 16) (h2 : v2 = 21) 
  (h3 : v1 > 0) (h4 : v2 > 0) (h5 : d > 0) : 
  (∃ (t : ℝ), t > 0 ∧ v1 * t + v2 * t = v1 * t + d + v2 * t) → 
  v1 * t + v2 * t = 444 :=
sorry

end NUMINAMATH_CALUDE_train_distance_problem_l254_25409


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l254_25435

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (not_subset : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel
  (m n l : Line) (α : Plane)
  (distinct : m ≠ n ∧ m ≠ l ∧ n ≠ l)
  (perp_lm : perpendicular l m)
  (not_in_plane : not_subset m α)
  (perp_lα : perpendicular_plane l α) :
  parallel_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l254_25435


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l254_25449

theorem consecutive_numbers_product (n : ℕ) : 
  (n + (n + 1) = 11) → (n * (n + 1) * (n + 2) = 210) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l254_25449


namespace NUMINAMATH_CALUDE_negation_of_at_least_three_l254_25474

-- Define a proposition for "at least n"
def at_least (n : ℕ) : Prop := sorry

-- Define a proposition for "at most n"
def at_most (n : ℕ) : Prop := sorry

-- State the given condition
axiom negation_rule : ∀ n : ℕ, ¬(at_least n) ↔ at_most (n - 1)

-- State the theorem to be proved
theorem negation_of_at_least_three : ¬(at_least 3) ↔ at_most 2 := by sorry

end NUMINAMATH_CALUDE_negation_of_at_least_three_l254_25474


namespace NUMINAMATH_CALUDE_johns_earnings_l254_25424

/-- John's earnings over two weeks --/
theorem johns_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 20 →
  hours_week2 = 30 →
  extra_earnings = 102.75 →
  let hourly_wage := extra_earnings / (hours_week2 - hours_week1)
  let total_earnings := (hours_week1 + hours_week2) * hourly_wage
  total_earnings = 513.75 := by
  sorry

end NUMINAMATH_CALUDE_johns_earnings_l254_25424


namespace NUMINAMATH_CALUDE_bisecting_angle_tangent_l254_25480

/-- A triangle with side lengths 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13

/-- A line that bisects both the perimeter and area of the triangle -/
structure BisectingLine where
  x : ℝ
  y : ℝ

/-- The two bisecting lines of the triangle -/
def bisecting_lines (t : RightTriangle) : Prod BisectingLine BisectingLine :=
  ⟨⟨10, -5⟩, ⟨7.5, -7.5⟩⟩

/-- The acute angle between the two bisecting lines -/
def bisecting_angle (t : RightTriangle) : ℝ := sorry

theorem bisecting_angle_tangent (t : RightTriangle) :
  let lines := bisecting_lines t
  let φ := bisecting_angle t
  Real.tan φ = 
    let v1 := lines.1
    let v2 := lines.2
    let dot_product := v1.x * v2.x + v1.y * v2.y
    let mag1 := Real.sqrt (v1.x^2 + v1.y^2)
    let mag2 := Real.sqrt (v2.x^2 + v2.y^2)
    let cos_φ := dot_product / (mag1 * mag2)
    Real.sqrt (1 - cos_φ^2) / cos_φ := by sorry

end NUMINAMATH_CALUDE_bisecting_angle_tangent_l254_25480


namespace NUMINAMATH_CALUDE_correct_amount_returned_l254_25476

/-- Calculates the amount to be returned in rubles given the initial deposit in USD and the exchange rate. -/
def amount_to_be_returned (initial_deposit : ℝ) (exchange_rate : ℝ) : ℝ :=
  initial_deposit * exchange_rate

/-- Proves that the amount to be returned is 581,500 rubles given the initial deposit and exchange rate. -/
theorem correct_amount_returned (initial_deposit : ℝ) (exchange_rate : ℝ) 
  (h1 : initial_deposit = 10000)
  (h2 : exchange_rate = 58.15) :
  amount_to_be_returned initial_deposit exchange_rate = 581500 := by
  sorry

#eval amount_to_be_returned 10000 58.15

end NUMINAMATH_CALUDE_correct_amount_returned_l254_25476


namespace NUMINAMATH_CALUDE_initial_sony_games_l254_25428

/-- The number of Sony games Kelly gives away -/
def games_given_away : ℕ := 101

/-- The number of Sony games Kelly has left after giving away games -/
def games_left : ℕ := 31

/-- The initial number of Sony games Kelly has -/
def initial_games : ℕ := games_given_away + games_left

theorem initial_sony_games : initial_games = 132 := by sorry

end NUMINAMATH_CALUDE_initial_sony_games_l254_25428


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l254_25479

/-- 
Given that the third term of the expansion of (3x - 2/x)^n is a constant term,
prove that n = 8.
-/
theorem binomial_expansion_constant_term (n : ℕ) : 
  (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → 
    (Nat.choose n 2 * (3 * x - 2 / x)^(n - 2) * (-2 / x)^2 = c)) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l254_25479


namespace NUMINAMATH_CALUDE_total_distance_traveled_l254_25446

def speed : ℝ := 60
def driving_sessions : List ℝ := [4, 5, 3, 2]

theorem total_distance_traveled :
  (List.sum driving_sessions) * speed = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_traveled_l254_25446


namespace NUMINAMATH_CALUDE_point_outside_circle_l254_25454

theorem point_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  a^2 + b^2 > 1 := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l254_25454


namespace NUMINAMATH_CALUDE_max_uncovered_sections_specific_case_l254_25416

/-- Represents a corridor with carpet strips -/
structure CarpetedCorridor where
  corridorLength : ℕ
  numStrips : ℕ
  totalStripLength : ℕ

/-- Calculates the maximum number of uncovered sections in a carpeted corridor -/
def maxUncoveredSections (c : CarpetedCorridor) : ℕ :=
  sorry

/-- Theorem stating the maximum number of uncovered sections for the given problem -/
theorem max_uncovered_sections_specific_case :
  let c : CarpetedCorridor := {
    corridorLength := 100,
    numStrips := 20,
    totalStripLength := 1000
  }
  maxUncoveredSections c = 11 :=
by sorry

end NUMINAMATH_CALUDE_max_uncovered_sections_specific_case_l254_25416


namespace NUMINAMATH_CALUDE_interest_calculation_years_l254_25484

/-- Proves that the number of years for which the interest is calculated on the first part is 8 --/
theorem interest_calculation_years (total_sum : ℚ) (second_part : ℚ) 
  (interest_rate_first : ℚ) (interest_rate_second : ℚ) (time_second : ℚ) :
  total_sum = 2795 →
  second_part = 1720 →
  interest_rate_first = 3 / 100 →
  interest_rate_second = 5 / 100 →
  time_second = 3 →
  let first_part := total_sum - second_part
  let interest_second := second_part * interest_rate_second * time_second
  ∃ (time_first : ℚ), first_part * interest_rate_first * time_first = interest_second ∧ time_first = 8 :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_years_l254_25484


namespace NUMINAMATH_CALUDE_smallest_divisible_by_3_5_7_13_greater_than_1000_l254_25464

theorem smallest_divisible_by_3_5_7_13_greater_than_1000 : ∃ n : ℕ,
  n > 1000 ∧
  n % 3 = 0 ∧
  n % 5 = 0 ∧
  n % 7 = 0 ∧
  n % 13 = 0 ∧
  (∀ m : ℕ, m > 1000 ∧ m % 3 = 0 ∧ m % 5 = 0 ∧ m % 7 = 0 ∧ m % 13 = 0 → m ≥ n) ∧
  n = 1365 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_3_5_7_13_greater_than_1000_l254_25464


namespace NUMINAMATH_CALUDE_f_minimum_and_inequality_l254_25491

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_minimum_and_inequality :
  (∃ (x : ℝ), x > 0 ∧ f x = -1 / Real.exp 1) ∧
  (∀ (x : ℝ), x > 0 → Real.log x > 1 / Real.exp x - 2 / (Real.exp 1 * x)) := by
  sorry

end NUMINAMATH_CALUDE_f_minimum_and_inequality_l254_25491


namespace NUMINAMATH_CALUDE_sunlovers_happy_days_l254_25427

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sunlovers_happy_days_l254_25427


namespace NUMINAMATH_CALUDE_steve_final_marbles_l254_25440

/- Define the initial number of marbles for each person -/
def sam_initial : ℕ := 14
def steve_initial : ℕ := 7
def sally_initial : ℕ := 9

/- Define the number of marbles Sam gives away -/
def marbles_given : ℕ := 3

/- Define Sam's final number of marbles -/
def sam_final : ℕ := 8

/- Theorem to prove -/
theorem steve_final_marbles :
  /- Conditions -/
  (sam_initial = 2 * steve_initial) →
  (sally_initial = sam_initial - 5) →
  (sam_final = sam_initial - 2 * marbles_given) →
  /- Conclusion -/
  (steve_initial + marbles_given = 10) := by
sorry

end NUMINAMATH_CALUDE_steve_final_marbles_l254_25440


namespace NUMINAMATH_CALUDE_tempo_value_l254_25438

/-- The original value of an insured item given insurance rate, premium rate, and premium amount. -/
def original_value (insurance_rate : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) : ℚ :=
  (premium_amount * 100) / (insurance_rate * premium_rate)

/-- Theorem stating that given the specified conditions, the original value of the tempo is $14000. -/
theorem tempo_value : 
  let insurance_rate : ℚ := 5/7
  let premium_rate : ℚ := 3/100
  let premium_amount : ℚ := 300
  original_value insurance_rate premium_rate premium_amount = 14000 := by
sorry

end NUMINAMATH_CALUDE_tempo_value_l254_25438


namespace NUMINAMATH_CALUDE_profit_2004_l254_25407

/-- Represents the profit of a company over years -/
def CompanyProfit (initialProfit : ℝ) (growthRate : ℝ) (year : ℕ) : ℝ :=
  initialProfit * (1 + growthRate) ^ (year - 2002)

/-- Theorem stating the profit in 2004 given initial conditions -/
theorem profit_2004 (initialProfit growthRate : ℝ) :
  initialProfit = 10 →
  CompanyProfit initialProfit growthRate 2004 = 1000 * (1 + growthRate)^2 := by
  sorry

#check profit_2004

end NUMINAMATH_CALUDE_profit_2004_l254_25407


namespace NUMINAMATH_CALUDE_distance_to_axis_of_symmetry_l254_25445

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = 12 * x

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 2

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola_C A.1 A.2 ∧ parabola_C B.1 B.2 ∧
  line_l A.1 A.2 ∧ line_l B.1 B.2

-- Define the axis of symmetry
def axis_of_symmetry : ℝ := -3

-- Theorem statement
theorem distance_to_axis_of_symmetry (A B : ℝ × ℝ) :
  intersection_points A B →
  let midpoint_x := (A.1 + B.1) / 2
  |midpoint_x - axis_of_symmetry| = 11 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_axis_of_symmetry_l254_25445


namespace NUMINAMATH_CALUDE_stability_comparison_l254_25470

/-- Represents a student's math exam scores -/
structure StudentScores where
  mean : ℝ
  variance : ℝ
  exam_count : ℕ

/-- Defines the concept of stability for exam scores -/
def more_stable (a b : StudentScores) : Prop :=
  a.variance < b.variance

theorem stability_comparison 
  (student_A student_B : StudentScores)
  (h1 : student_A.mean = student_B.mean)
  (h2 : student_A.exam_count = student_B.exam_count)
  (h3 : student_A.exam_count = 5)
  (h4 : student_A.mean = 102)
  (h5 : student_A.variance = 38)
  (h6 : student_B.variance = 15) :
  more_stable student_B student_A :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l254_25470


namespace NUMINAMATH_CALUDE_solutions_count_l254_25437

def count_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_count (n : ℕ) : 
  (count_solutions 1 = 4) → 
  (count_solutions 2 = 8) → 
  (count_solutions 3 = 12) → 
  (count_solutions 20 = 80) :=
by sorry

end NUMINAMATH_CALUDE_solutions_count_l254_25437


namespace NUMINAMATH_CALUDE_union_cardinality_l254_25429

def A : Finset ℕ := {4, 5, 7, 9}
def B : Finset ℕ := {3, 4, 7, 8, 9}

theorem union_cardinality : (A ∪ B).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_l254_25429


namespace NUMINAMATH_CALUDE_cube_layer_removal_l254_25420

/-- Calculates the number of smaller cubes remaining inside a cube after removing layers to form a hollow cuboid --/
def remaining_cubes (original_size : Nat) (hollow_size : Nat) : Nat :=
  hollow_size^3 - (hollow_size - 2)^3

/-- Theorem stating that for a 12x12x12 cube with a 10x10x10 hollow cuboid, 488 smaller cubes remain --/
theorem cube_layer_removal :
  remaining_cubes 12 10 = 488 := by
  sorry

end NUMINAMATH_CALUDE_cube_layer_removal_l254_25420


namespace NUMINAMATH_CALUDE_remainder_problem_l254_25465

theorem remainder_problem (n : ℕ) (r₃ r₆ r₉ : ℕ) :
  r₃ < 3 ∧ r₆ < 6 ∧ r₉ < 9 →
  n % 3 = r₃ ∧ n % 6 = r₆ ∧ n % 9 = r₉ →
  r₃ + r₆ + r₉ = 15 →
  n % 18 = 17 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l254_25465


namespace NUMINAMATH_CALUDE_assistant_prof_charts_l254_25447

theorem assistant_prof_charts (associate_profs assistant_profs : ℕ) 
  (charts_per_assistant : ℕ) :
  associate_profs + assistant_profs = 7 →
  2 * associate_profs + assistant_profs = 10 →
  associate_profs + assistant_profs * charts_per_assistant = 11 →
  charts_per_assistant = 2 :=
by sorry

end NUMINAMATH_CALUDE_assistant_prof_charts_l254_25447


namespace NUMINAMATH_CALUDE_taehyung_walk_distance_l254_25493

/-- Proves that given Taehyung's step length of 0.45 meters, and moving 90 steps 13 times, the total distance walked is 526.5 meters. -/
theorem taehyung_walk_distance :
  let step_length : ℝ := 0.45
  let steps_per_set : ℕ := 90
  let num_sets : ℕ := 13
  step_length * (steps_per_set * num_sets : ℝ) = 526.5 := by sorry

end NUMINAMATH_CALUDE_taehyung_walk_distance_l254_25493


namespace NUMINAMATH_CALUDE_shooting_test_probability_l254_25430

/-- Represents the probability of hitting a single shot -/
def hit_prob : ℝ := 0.6

/-- Represents the probability of missing a single shot -/
def miss_prob : ℝ := 1 - hit_prob

/-- Calculates the probability of passing the shooting test -/
def pass_prob : ℝ := 
  hit_prob^3 + hit_prob^2 * miss_prob + miss_prob * hit_prob^2

theorem shooting_test_probability : pass_prob = 0.504 := by
  sorry

end NUMINAMATH_CALUDE_shooting_test_probability_l254_25430


namespace NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l254_25450

theorem three_digit_perfect_cube_divisible_by_16 :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^3 ∧ n % 16 = 0 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_perfect_cube_divisible_by_16_l254_25450


namespace NUMINAMATH_CALUDE_cube_inequality_iff_l254_25473

theorem cube_inequality_iff (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_inequality_iff_l254_25473


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l254_25497

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a : ℤ  -- First term
  d : ℤ  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a + seq.d * (n - 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a + seq.d * (n - 1)) / 2

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.nthTerm 7 = 4 ∧ seq.nthTerm 8 = 10 ∧ seq.nthTerm 9 = 16 →
  seq.sumFirstN 5 = -100 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l254_25497


namespace NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l254_25433

theorem triangle_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^2 + y^2 = z^2) : 
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) ≤ (2 + 3 * Real.sqrt 2) * x * y * z :=
sorry

theorem equality_condition (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x^2 + y^2 = z^2) : 
  x^2 * (y + z) + y^2 * (z + x) + z^2 * (x + y) = (2 + 3 * Real.sqrt 2) * x * y * z ↔ x = y :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_equality_condition_l254_25433
