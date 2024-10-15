import Mathlib

namespace NUMINAMATH_CALUDE_adelkas_numbers_l3561_356141

theorem adelkas_numbers : ∃ (a b : ℕ), 
  0 < a ∧ a < b ∧ b < 100 ∧
  (Nat.gcd a b) < a ∧ a < b ∧ b < (Nat.lcm a b) ∧ (Nat.lcm a b) < 100 ∧
  (Nat.lcm a b) / (Nat.gcd a b) = Nat.gcd (Nat.gcd a b) (Nat.gcd a (Nat.gcd b (Nat.lcm a b))) ∧
  a = 12 ∧ b = 18 := by
sorry

end NUMINAMATH_CALUDE_adelkas_numbers_l3561_356141


namespace NUMINAMATH_CALUDE_happy_boys_count_l3561_356165

theorem happy_boys_count (total_children : ℕ) (happy_children : ℕ) (sad_children : ℕ) 
  (neutral_children : ℕ) (total_boys : ℕ) (total_girls : ℕ) (sad_girls : ℕ) 
  (neutral_boys : ℕ) :
  total_children = 60 →
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  total_boys = 16 →
  total_girls = 44 →
  sad_girls = 4 →
  neutral_boys = 4 →
  ∃ (happy_boys : ℕ), happy_boys > 0 →
  happy_boys = 6 :=
by sorry

end NUMINAMATH_CALUDE_happy_boys_count_l3561_356165


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3561_356111

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_equation_solution (a b : V) (x y : ℝ) 
  (h_not_collinear : ¬ ∃ (k : ℝ), b = k • a) 
  (h_eq : (3 * x - 4 * y) • a + (2 * x - 3 * y) • b = 6 • a + 3 • b) :
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3561_356111


namespace NUMINAMATH_CALUDE_sum_xyz_equals_twenty_ninths_l3561_356132

theorem sum_xyz_equals_twenty_ninths 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + 4*y^2 + 9*z^2 = 4)
  (eq2 : 2*x + 4*y + 3*z = 6) :
  x + y + z = 20/9 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equals_twenty_ninths_l3561_356132


namespace NUMINAMATH_CALUDE_complex_sum_powers_of_i_l3561_356107

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_powers_of_i_l3561_356107


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3561_356128

theorem interest_rate_calculation (P t D : ℝ) (h1 : P = 500) (h2 : t = 2) (h3 : D = 20) : 
  ∃ r : ℝ, r = 20 ∧ 
    P * ((1 + r / 100) ^ t - 1) - P * r * t / 100 = D :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3561_356128


namespace NUMINAMATH_CALUDE_currency_denomination_proof_l3561_356137

theorem currency_denomination_proof :
  let press_F_rate : ℚ := 1000 / 60  -- bills per second
  let press_F_value : ℚ := 5 * press_F_rate  -- dollars per second
  let press_T_rate : ℚ := 200 / 60  -- bills per second
  let time : ℚ := 3  -- seconds
  let extra_value : ℚ := 50  -- dollars
  ∃ x : ℚ, 
    (time * press_F_value = time * (x * press_T_rate) + extra_value) ∧ 
    x = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_currency_denomination_proof_l3561_356137


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l3561_356161

theorem intersection_nonempty_implies_a_value (P Q : Set ℕ) (a : ℕ) :
  P = {0, a} →
  Q = {1, 2} →
  P ∩ Q ≠ ∅ →
  a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_a_value_l3561_356161


namespace NUMINAMATH_CALUDE_inheritance_problem_l3561_356150

theorem inheritance_problem (total_inheritance : ℕ) (additional_amount : ℕ) : 
  total_inheritance = 46800 →
  additional_amount = 1950 →
  ∃ (original_children : ℕ),
    original_children > 2 ∧
    (total_inheritance / original_children + additional_amount = total_inheritance / (original_children - 2)) ∧
    original_children = 8 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_problem_l3561_356150


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l3561_356136

/-- A pocket containing balls of two colors -/
structure Pocket where
  red : ℕ
  white : ℕ

/-- The possible outcomes when drawing two balls -/
inductive Outcome
  | TwoRed
  | OneRedOneWhite
  | TwoWhite

/-- Define the events -/
def ExactlyOneWhite (o : Outcome) : Prop :=
  o = Outcome.OneRedOneWhite

def ExactlyTwoWhite (o : Outcome) : Prop :=
  o = Outcome.TwoWhite

/-- The probability of an outcome given a pocket -/
def probability (p : Pocket) (o : Outcome) : ℚ :=
  match o with
  | Outcome.TwoRed => (p.red * (p.red - 1)) / ((p.red + p.white) * (p.red + p.white - 1))
  | Outcome.OneRedOneWhite => (2 * p.red * p.white) / ((p.red + p.white) * (p.red + p.white - 1))
  | Outcome.TwoWhite => (p.white * (p.white - 1)) / ((p.red + p.white) * (p.red + p.white - 1))

theorem mutually_exclusive_not_contradictory (p : Pocket) (h : p.red = 2 ∧ p.white = 2) :
  (∀ o : Outcome, ¬(ExactlyOneWhite o ∧ ExactlyTwoWhite o)) ∧ 
  (probability p Outcome.OneRedOneWhite + probability p Outcome.TwoWhite < 1) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l3561_356136


namespace NUMINAMATH_CALUDE_tangent_relation_l3561_356142

theorem tangent_relation (α β : Real) 
  (h : Real.tan (α - β) = Real.sin (2 * β) / (5 - Real.cos (2 * β))) :
  2 * Real.tan α = 3 * Real.tan β := by
  sorry

end NUMINAMATH_CALUDE_tangent_relation_l3561_356142


namespace NUMINAMATH_CALUDE_sin_beta_value_l3561_356116

theorem sin_beta_value (α β : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.cos (α - β) = -5/13)
  (h4 : Real.sin α = 4/5) :
  Real.sin β = -56/65 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l3561_356116


namespace NUMINAMATH_CALUDE_peach_difference_l3561_356167

/-- The number of peaches Steven has -/
def steven_peaches : ℕ := 14

/-- The number of peaches Jill has -/
def jill_peaches : ℕ := 5

/-- The number of peaches Jake has -/
def jake_peaches : ℕ := steven_peaches - 6

/-- Jake has more peaches than Jill -/
axiom jake_more_than_jill : jake_peaches > jill_peaches

theorem peach_difference : jake_peaches - jill_peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3561_356167


namespace NUMINAMATH_CALUDE_incorrect_division_result_l3561_356176

theorem incorrect_division_result (dividend : ℕ) :
  dividend / 36 = 32 →
  dividend / 48 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_division_result_l3561_356176


namespace NUMINAMATH_CALUDE_runner_journey_time_l3561_356101

/-- Represents the runner's journey --/
structure RunnerJourney where
  totalDistance : ℝ
  firstHalfSpeed : ℝ
  secondHalfSpeed : ℝ
  firstHalfTime : ℝ
  secondHalfTime : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem runner_journey_time (j : RunnerJourney) 
  (h1 : j.totalDistance = 40)
  (h2 : j.secondHalfSpeed = j.firstHalfSpeed / 2)
  (h3 : j.secondHalfTime = j.firstHalfTime + 5)
  (h4 : j.firstHalfTime = (j.totalDistance / 2) / j.firstHalfSpeed)
  (h5 : j.secondHalfTime = (j.totalDistance / 2) / j.secondHalfSpeed) :
  j.secondHalfTime = 10 := by
  sorry

end NUMINAMATH_CALUDE_runner_journey_time_l3561_356101


namespace NUMINAMATH_CALUDE_line_passes_through_point_min_length_AB_min_dot_product_l3561_356159

-- Define the line l: mx + y - 1 - 2m = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 - 2 * m = 0

-- Define the circle O: x^2 + y^2 = r^2
def circle_O (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2

-- Theorem 1: The line l passes through the point (2, 1) for all m
theorem line_passes_through_point (m : ℝ) : line_l m 2 1 := by sorry

-- Theorem 2: When r = 4, the minimum length of AB is 2√11
theorem min_length_AB (A B : ℝ × ℝ) 
  (hA : circle_O 4 A.1 A.2) (hB : circle_O 4 B.1 B.2) 
  (hl : ∃ m : ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2) :
  ∃ min_length : ℝ, min_length = 2 * Real.sqrt 11 ∧ 
  ∀ m : ℝ, line_l m A.1 A.2 → line_l m B.1 B.2 → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ min_length := by sorry

-- Theorem 3: When r = 4, the minimum value of OA · OB is -16
theorem min_dot_product (A B : ℝ × ℝ) 
  (hA : circle_O 4 A.1 A.2) (hB : circle_O 4 B.1 B.2) 
  (hl : ∃ m : ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2) :
  ∃ min_dot : ℝ, min_dot = -16 ∧ 
  ∀ m : ℝ, line_l m A.1 A.2 → line_l m B.1 B.2 → 
  A.1 * B.1 + A.2 * B.2 ≥ min_dot := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_min_length_AB_min_dot_product_l3561_356159


namespace NUMINAMATH_CALUDE_inequality_solution_l3561_356183

theorem inequality_solution (x : ℝ) (h : x ≠ 1) :
  1 / (x - 1) ≤ 1 ↔ x < 1 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3561_356183


namespace NUMINAMATH_CALUDE_add_three_preserves_inequality_l3561_356189

theorem add_three_preserves_inequality (a b : ℝ) : a > b → a + 3 > b + 3 := by
  sorry

end NUMINAMATH_CALUDE_add_three_preserves_inequality_l3561_356189


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l3561_356179

/-- A power function that passes through the point (3, 9) -/
def f (x : ℝ) : ℝ := x^2

/-- The point (3, 9) lies on the graph of f -/
axiom point_on_graph : f 3 = 9

/-- Theorem: The interval of monotonic increase for f is [0, +∞) -/
theorem monotonic_increase_interval :
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l3561_356179


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3561_356104

theorem quadratic_inequality_solution (x : ℝ) :
  -5 * x^2 + 7 * x + 2 > 0 ↔ 
  x > ((-7 - Real.sqrt 89) / -10) ∧ x < ((-7 + Real.sqrt 89) / -10) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3561_356104


namespace NUMINAMATH_CALUDE_a_10_equals_133_l3561_356140

/-- The number of subsets of {1,2,...,n} with at least two elements and 
    the absolute difference between any two elements greater than 1 -/
def a (n : ℕ) : ℕ :=
  if n ≤ 2 then 0
  else if n = 3 then 1
  else if n = 4 then 3
  else a (n-1) + a (n-2) + (n-2)

/-- The main theorem to prove -/
theorem a_10_equals_133 : a 10 = 133 := by
  sorry

end NUMINAMATH_CALUDE_a_10_equals_133_l3561_356140


namespace NUMINAMATH_CALUDE_airplane_hovering_time_l3561_356199

/-- Calculates the total hovering time for an airplane over two days -/
theorem airplane_hovering_time 
  (mountain_day1 : ℕ) 
  (central_day1 : ℕ) 
  (eastern_day1 : ℕ) 
  (additional_time : ℕ) 
  (h1 : mountain_day1 = 3)
  (h2 : central_day1 = 4)
  (h3 : eastern_day1 = 2)
  (h4 : additional_time = 2) :
  mountain_day1 + central_day1 + eastern_day1 + 
  (mountain_day1 + additional_time) + 
  (central_day1 + additional_time) + 
  (eastern_day1 + additional_time) = 24 := by
sorry


end NUMINAMATH_CALUDE_airplane_hovering_time_l3561_356199


namespace NUMINAMATH_CALUDE_circle_circumference_ratio_l3561_356124

theorem circle_circumference_ratio (r_large r_small : ℝ) (h : r_large / r_small = 3 / 2) :
  (2 * Real.pi * r_large) / (2 * Real.pi * r_small) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_ratio_l3561_356124


namespace NUMINAMATH_CALUDE_count_two_digit_numbers_unit_gte_tens_is_45_l3561_356175

/-- The count of two-digit numbers where the unit digit is not less than the tens digit -/
def count_two_digit_numbers_unit_gte_tens : ℕ := 45

/-- Proof that the count of two-digit numbers where the unit digit is not less than the tens digit is 45 -/
theorem count_two_digit_numbers_unit_gte_tens_is_45 :
  count_two_digit_numbers_unit_gte_tens = 45 := by
  sorry

end NUMINAMATH_CALUDE_count_two_digit_numbers_unit_gte_tens_is_45_l3561_356175


namespace NUMINAMATH_CALUDE_photo_arrangement_count_l3561_356152

/-- Represents the number of people in the arrangement -/
def total_people : ℕ := 6

/-- Represents the number of students in the arrangement -/
def num_students : ℕ := 4

/-- Represents the number of teachers in the arrangement -/
def num_teachers : ℕ := 2

/-- Represents the number of students that must stand together -/
def students_together : ℕ := 2

/-- Calculates the number of ways to arrange the people given the constraints -/
def arrangement_count : ℕ :=
  (num_teachers.factorial) *    -- Ways to arrange teachers in the middle
  2 *                           -- Ways to place students A and B (left or right of teachers)
  (students_together.factorial) * -- Ways to arrange A and B within their unit
  ((num_students - students_together).factorial) -- Ways to arrange remaining students

theorem photo_arrangement_count :
  arrangement_count = 8 := by sorry

end NUMINAMATH_CALUDE_photo_arrangement_count_l3561_356152


namespace NUMINAMATH_CALUDE_cards_not_in_box_l3561_356173

theorem cards_not_in_box (total_cards : ℕ) (cards_per_box : ℕ) (boxes_given : ℕ) (boxes_kept : ℕ) : 
  total_cards = 75 →
  cards_per_box = 10 →
  boxes_given = 2 →
  boxes_kept = 5 →
  total_cards - (cards_per_box * (boxes_given + boxes_kept)) = 5 := by
sorry

end NUMINAMATH_CALUDE_cards_not_in_box_l3561_356173


namespace NUMINAMATH_CALUDE_c_alone_time_l3561_356117

-- Define the rates of work for A, B, and C
variable (rA rB rC : ℝ)

-- Define the conditions
axiom ab_rate : rA + rB = 1/3
axiom bc_rate : rB + rC = 1/6
axiom ac_rate : rA + rC = 1/4

-- Define the theorem
theorem c_alone_time : 1 / rC = 24 := by
  sorry

end NUMINAMATH_CALUDE_c_alone_time_l3561_356117


namespace NUMINAMATH_CALUDE_K_bounds_l3561_356168

-- Define the variables and constraints
def K (x y z : ℝ) : ℝ := 5*x - 6*y + 7*z

def constraint1 (x y z : ℝ) : Prop := 4*x + y + 2*z = 4
def constraint2 (x y z : ℝ) : Prop := 3*x + 6*y - 2*z = 6

-- State the theorem
theorem K_bounds :
  ∀ x y z : ℝ,
    x ≥ 0 → y ≥ 0 → z ≥ 0 →
    constraint1 x y z →
    constraint2 x y z →
    -5 ≤ K x y z ∧ K x y z ≤ 7 :=
by
  sorry

end NUMINAMATH_CALUDE_K_bounds_l3561_356168


namespace NUMINAMATH_CALUDE_sum_of_first_89_l3561_356170

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: The sum of the first 89 natural numbers is 4005 -/
theorem sum_of_first_89 : sum_of_first_n 89 = 4005 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_89_l3561_356170


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l3561_356162

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 11 ∧ 
  (∀ m : ℕ, m < k → (128 : ℝ)^m ≤ 8^25 + 1000) ∧
  (128 : ℝ)^k > 8^25 + 1000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l3561_356162


namespace NUMINAMATH_CALUDE_rope_purchase_difference_l3561_356163

def inches_per_foot : ℕ := 12

def last_week_purchase : ℕ := 6

def this_week_purchase_inches : ℕ := 96

theorem rope_purchase_difference :
  last_week_purchase - (this_week_purchase_inches / inches_per_foot) = 2 :=
by sorry

end NUMINAMATH_CALUDE_rope_purchase_difference_l3561_356163


namespace NUMINAMATH_CALUDE_not_perfect_square_for_prime_l3561_356186

theorem not_perfect_square_for_prime (p : ℕ) (h_prime : Nat.Prime p) :
  ¬∃ (a : ℤ), a^2 = 7 * p + 3^p - 4 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_for_prime_l3561_356186


namespace NUMINAMATH_CALUDE_base_9_addition_multiplication_l3561_356193

/-- Converts a number from base 9 to base 10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 9^2 + tens * 9 + ones

/-- Converts a number from base 10 to base 9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  sorry -- Implementation not provided, as it's not required for the statement

theorem base_9_addition_multiplication :
  let a := base9ToBase10 436
  let b := base9ToBase10 782
  let c := base9ToBase10 204
  let d := base9ToBase10 12
  base10ToBase9 ((a + b + c) * d) = 18508 := by
  sorry

end NUMINAMATH_CALUDE_base_9_addition_multiplication_l3561_356193


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l3561_356122

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - k = 0 ∧ 
   ∀ y : ℝ, y^2 + 3*y - k = 0 → y = x) →
  k = -9/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l3561_356122


namespace NUMINAMATH_CALUDE_vasily_salary_higher_than_fedor_l3561_356158

/-- Represents the salary distribution for graduates --/
structure SalaryDistribution where
  high : ℝ  -- Proportion earning 60,000 rubles
  very_high : ℝ  -- Proportion earning 80,000 rubles
  low : ℝ  -- Proportion earning 25,000 rubles (not in field)
  medium : ℝ  -- Proportion earning 40,000 rubles

/-- Calculates the expected salary given a salary distribution --/
def expected_salary (dist : SalaryDistribution) : ℝ :=
  60000 * dist.high + 80000 * dist.very_high + 25000 * dist.low + 40000 * dist.medium

/-- Calculates Fedor's salary after a given number of years --/
def fedor_salary (years : ℕ) : ℝ :=
  25000 + 3000 * years

/-- Main theorem statement --/
theorem vasily_salary_higher_than_fedor :
  let total_students : ℝ := 300
  let successful_students : ℝ := 270
  let grad_prob : ℝ := successful_students / total_students
  let salary_dist : SalaryDistribution := {
    high := 1/5,
    very_high := 1/10,
    low := 1/20,
    medium := 1 - (1/5 + 1/10 + 1/20)
  }
  let vasily_expected_salary : ℝ := 
    grad_prob * expected_salary salary_dist + (1 - grad_prob) * 25000
  let fedor_final_salary : ℝ := fedor_salary 4
  vasily_expected_salary = 45025 ∧ 
  vasily_expected_salary - fedor_final_salary = 8025 := by
  sorry


end NUMINAMATH_CALUDE_vasily_salary_higher_than_fedor_l3561_356158


namespace NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l3561_356196

theorem modular_inverse_of_7_mod_31 : ∃ x : ℕ, x ≤ 30 ∧ (7 * x) % 31 = 1 :=
by
  use 9
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_7_mod_31_l3561_356196


namespace NUMINAMATH_CALUDE_ounces_per_can_l3561_356129

/-- Represents the number of ounces in a cup of chickpeas -/
def ounces_per_cup : ℕ := 6

/-- Represents the number of cups needed for one serving of hummus -/
def cups_per_serving : ℕ := 1

/-- Represents the number of servings Thomas wants to make -/
def total_servings : ℕ := 20

/-- Represents the number of cans Thomas needs to buy -/
def cans_needed : ℕ := 8

/-- Theorem stating the number of ounces in each can of chickpeas -/
theorem ounces_per_can : 
  (total_servings * cups_per_serving * ounces_per_cup) / cans_needed = 15 := by
  sorry

end NUMINAMATH_CALUDE_ounces_per_can_l3561_356129


namespace NUMINAMATH_CALUDE_jonah_calories_per_hour_l3561_356112

/-- The number of calories Jonah burns per hour while running -/
def calories_per_hour : ℝ := 30

/-- The number of hours Jonah actually ran -/
def actual_hours : ℝ := 2

/-- The hypothetical number of hours Jonah could have run -/
def hypothetical_hours : ℝ := 5

/-- The additional calories Jonah would have burned if he ran for the hypothetical hours -/
def additional_calories : ℝ := 90

theorem jonah_calories_per_hour :
  calories_per_hour * hypothetical_hours = 
  calories_per_hour * actual_hours + additional_calories :=
sorry

end NUMINAMATH_CALUDE_jonah_calories_per_hour_l3561_356112


namespace NUMINAMATH_CALUDE_cloth_cost_price_l3561_356106

/-- Given a trader selling cloth, calculate the cost price per metre. -/
theorem cloth_cost_price
  (total_metres : ℕ)
  (selling_price : ℕ)
  (profit_per_metre : ℕ)
  (h1 : total_metres = 30)
  (h2 : selling_price = 4500)
  (h3 : profit_per_metre = 10) :
  (selling_price - total_metres * profit_per_metre) / total_metres = 140 :=
sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l3561_356106


namespace NUMINAMATH_CALUDE_sine_function_vertical_shift_l3561_356174

/-- Given a sine function y = a * sin(b * x + c) + d that oscillates between 4 and -2,
    prove that the vertical shift d equals 1. -/
theorem sine_function_vertical_shift
  (a b c d : ℝ)
  (positive_constants : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (oscillation : ∀ x : ℝ, -2 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 4) :
  d = 1 := by
  sorry

end NUMINAMATH_CALUDE_sine_function_vertical_shift_l3561_356174


namespace NUMINAMATH_CALUDE_car_speed_proof_l3561_356144

/-- Proves that a car's speed is 48 km/h if it takes 15 seconds longer to travel 1 km compared to 60 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v - 1 / 60) * 3600 = 15 ↔ v = 48 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3561_356144


namespace NUMINAMATH_CALUDE_printers_finish_time_l3561_356102

-- Define the start time of the first printer
def printer1_start : Real := 9

-- Define the time when half the tasks are completed
def half_tasks_time : Real := 12.5

-- Define the start time of the second printer
def printer2_start : Real := 13

-- Define the time taken by the second printer to complete its set amount
def printer2_duration : Real := 2

-- Theorem to prove
theorem printers_finish_time :
  let printer1_duration := 2 * (half_tasks_time - printer1_start)
  let printer1_finish := printer1_start + printer1_duration
  let printer2_finish := printer2_start + printer2_duration
  max printer1_finish printer2_finish = 16 := by
  sorry

end NUMINAMATH_CALUDE_printers_finish_time_l3561_356102


namespace NUMINAMATH_CALUDE_bakers_pastries_l3561_356146

/-- Baker's pastry problem -/
theorem bakers_pastries 
  (initial_cakes : ℕ)
  (sold_cakes : ℕ)
  (sold_pastries : ℕ)
  (remaining_pastries : ℕ)
  (h1 : initial_cakes = 7)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45) :
  sold_pastries + remaining_pastries = 148 :=
sorry

end NUMINAMATH_CALUDE_bakers_pastries_l3561_356146


namespace NUMINAMATH_CALUDE_fraction_of_x_l3561_356166

theorem fraction_of_x (w x y f : ℝ) : 
  2/w + f*x = 2/y → 
  w*x = y → 
  (w + x)/2 = 0.5 → 
  f = 2/x - 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_x_l3561_356166


namespace NUMINAMATH_CALUDE_inequality_proof_l3561_356190

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (sum : a + b + c = Real.sqrt 2) :
  1 / Real.sqrt (1 + a^2) + 1 / Real.sqrt (1 + b^2) + 1 / Real.sqrt (1 + c^2) ≥ 2 + 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3561_356190


namespace NUMINAMATH_CALUDE_function_increasing_implies_omega_bound_l3561_356182

theorem function_increasing_implies_omega_bound 
  (ω : ℝ) 
  (h_pos : ω > 0)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/2) * Real.sin (ω * x / 2) * Real.cos (ω * x / 2))
  (h_increasing : StrictMonoOn f (Set.Icc (-π/3) (π/4))) :
  ω ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_function_increasing_implies_omega_bound_l3561_356182


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l3561_356184

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l3561_356184


namespace NUMINAMATH_CALUDE_condition_2_is_sufficient_for_condition_1_l3561_356139

-- Define the propositions
variable (A B C D : Prop)

-- Define the relationship between conditions
def condition_relationship (A B C D : Prop) : Prop :=
  (C < D → A > B)

-- Define sufficient condition
def is_sufficient_condition (P Q : Prop) : Prop :=
  P → Q

-- Theorem statement
theorem condition_2_is_sufficient_for_condition_1 
  (h : condition_relationship A B C D) :
  is_sufficient_condition (C < D) (A > B) :=
sorry

end NUMINAMATH_CALUDE_condition_2_is_sufficient_for_condition_1_l3561_356139


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3561_356145

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the general formula for the n-th term of the sequence -/
theorem arithmetic_sequence_formula (seq : ArithmeticSequence) 
  (h3 : (S seq 3)^2 = 9 * (S seq 2))
  (h4 : S seq 4 = 4 * (S seq 2)) :
  ∀ n : ℕ, seq.a n = (4 : ℚ) / 9 * (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3561_356145


namespace NUMINAMATH_CALUDE_power_equation_solution_l3561_356148

theorem power_equation_solution : ∃! x : ℤ, (3 : ℝ) ^ 7 * (3 : ℝ) ^ x = 81 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3561_356148


namespace NUMINAMATH_CALUDE_composite_8n_plus_3_l3561_356160

theorem composite_8n_plus_3 (n : ℕ) (x y : ℕ) 
  (h1 : 8 * n + 1 = x^2) 
  (h2 : 24 * n + 1 = y^2) 
  (h3 : n > 1) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 8 * n + 3 = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_8n_plus_3_l3561_356160


namespace NUMINAMATH_CALUDE_complex_fraction_power_l3561_356198

theorem complex_fraction_power (i : ℂ) : i * i = -1 → (((1 + i) / (1 - i)) ^ 2018 : ℂ) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_power_l3561_356198


namespace NUMINAMATH_CALUDE_ordered_pair_satisfies_equation_l3561_356108

theorem ordered_pair_satisfies_equation :
  let a : ℝ := 9
  let b : ℝ := -4
  (Real.sqrt (25 - 16 * Real.cos (π / 3)) = a - b * (1 / Real.cos (π / 3))) := by
  sorry

end NUMINAMATH_CALUDE_ordered_pair_satisfies_equation_l3561_356108


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3561_356147

theorem sqrt_equation_solution :
  ∃! (y : ℝ), y > 0 ∧ 3 * Real.sqrt (4 + y) + 3 * Real.sqrt (4 - y) = 6 * Real.sqrt 3 :=
by
  use 2 * Real.sqrt 3
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3561_356147


namespace NUMINAMATH_CALUDE_direct_proportion_problem_l3561_356115

theorem direct_proportion_problem (α β : ℝ) (k : ℝ) (h1 : α = k * β) (h2 : 6 = k * 18) (h3 : α = 15) : β = 45 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_problem_l3561_356115


namespace NUMINAMATH_CALUDE_lemonade_stand_problem_l3561_356134

/-- Represents the lemonade stand problem --/
theorem lemonade_stand_problem 
  (glasses_per_gallon : ℕ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ)
  (net_profit : ℚ)
  (h1 : glasses_per_gallon = 16)
  (h2 : gallons_made = 2)
  (h3 : price_per_glass = 1)
  (h4 : glasses_drunk = 5)
  (h5 : glasses_unsold = 6)
  (h6 : net_profit = 14) :
  (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass - net_profit = gallons_made * (7/2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_lemonade_stand_problem_l3561_356134


namespace NUMINAMATH_CALUDE_players_who_quit_l3561_356192

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives_after : ℕ) :
  initial_players = 16 →
  lives_per_player = 8 →
  total_lives_after = 72 →
  initial_players - (total_lives_after / lives_per_player) = 7 :=
by sorry

end NUMINAMATH_CALUDE_players_who_quit_l3561_356192


namespace NUMINAMATH_CALUDE_length_BC_is_sqrt_13_l3561_356118

/-- The cosine theorem for a triangle ABC -/
def cosine_theorem (a b c : ℝ) (A : ℝ) : Prop :=
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos A)

/-- Triangle ABC with given side lengths and angle -/
structure Triangle :=
  (AB : ℝ)
  (AC : ℝ)
  (angle_A : ℝ)
  (h_AB_pos : AB > 0)
  (h_AC_pos : AC > 0)
  (h_angle_A_pos : angle_A > 0)
  (h_angle_A_lt_pi : angle_A < π)

theorem length_BC_is_sqrt_13 (t : Triangle) 
  (h_AB : t.AB = 3)
  (h_AC : t.AC = 4)
  (h_angle_A : t.angle_A = π/3) :
  ∃ BC : ℝ, BC > 0 ∧ BC^2 = 13 ∧ cosine_theorem t.AB t.AC BC t.angle_A :=
sorry

end NUMINAMATH_CALUDE_length_BC_is_sqrt_13_l3561_356118


namespace NUMINAMATH_CALUDE_existence_of_suitable_set_l3561_356135

theorem existence_of_suitable_set (ε : Real) (h_ε : 0 < ε ∧ ε < 1) :
  ∃ N₀ : ℕ, ∀ N ≥ N₀, ∃ S : Finset ℕ,
    (S.card : ℝ) ≥ ε * N ∧
    (∀ x ∈ S, x ≤ N) ∧
    (∀ x ∈ S, Nat.gcd x (S.sum id) > 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_suitable_set_l3561_356135


namespace NUMINAMATH_CALUDE_prize_prices_and_min_cost_l3561_356138

/- Define the unit prices of prizes A and B -/
def price_A : ℝ := 20
def price_B : ℝ := 10

/- Define the total number of prizes and minimum number of prize A -/
def total_prizes : ℕ := 60
def min_prize_A : ℕ := 20

/- Define the cost function -/
def cost (m : ℕ) : ℝ := price_A * m + price_B * (total_prizes - m)

theorem prize_prices_and_min_cost :
  /- Condition 1: 1 A and 2 B cost $40 -/
  price_A + 2 * price_B = 40 ∧
  /- Condition 2: 2 A and 3 B cost $70 -/
  2 * price_A + 3 * price_B = 70 ∧
  /- The minimum cost occurs when m = min_prize_A -/
  (∀ m : ℕ, min_prize_A ≤ m → m ≤ total_prizes → cost min_prize_A ≤ cost m) ∧
  /- The minimum cost is $800 -/
  cost min_prize_A = 800 := by
  sorry

#check prize_prices_and_min_cost

end NUMINAMATH_CALUDE_prize_prices_and_min_cost_l3561_356138


namespace NUMINAMATH_CALUDE_valid_numbers_l3561_356121

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000 ∧ n ≤ 999999) ∧  -- six-digit number
  (∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 100000 + 2014 * 10 + b) ∧  -- formed by adding digits to 2014
  n % 36 = 0  -- divisible by 36

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {220140, 720144, 320148} :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_l3561_356121


namespace NUMINAMATH_CALUDE_lesser_number_l3561_356110

theorem lesser_number (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_l3561_356110


namespace NUMINAMATH_CALUDE_square_area_ratio_l3561_356191

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 48) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3561_356191


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3561_356126

def proposition (x : Real) : Prop := x ∈ Set.Icc 0 (2 * Real.pi) → |Real.sin x| ≤ 1

theorem negation_of_proposition :
  (¬ ∀ x, proposition x) ↔ (∃ x, x ∈ Set.Icc 0 (2 * Real.pi) ∧ |Real.sin x| > 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3561_356126


namespace NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l3561_356131

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

-- State the theorem
theorem third_term_of_specific_arithmetic_sequence 
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2) :
  a 3 = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_term_of_specific_arithmetic_sequence_l3561_356131


namespace NUMINAMATH_CALUDE_f_eight_eq_twelve_f_two_f_odd_l3561_356157

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f is not identically zero
axiom f_not_zero : ∃ x, f x ≠ 0

-- Define the functional equation
axiom f_eq (x y : ℝ) : f (x * y) = x * f y + y * f x

-- Theorem 1: f(8) = 12f(2)
theorem f_eight_eq_twelve_f_two : f 8 = 12 * f 2 := by sorry

-- Theorem 2: f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_eight_eq_twelve_f_two_f_odd_l3561_356157


namespace NUMINAMATH_CALUDE_circle_equation_l3561_356172

/-- The equation of a circle with center (1,1) passing through the origin (0,0) -/
theorem circle_equation : 
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), (x - 1)^2 + (y - 1)^2 = r^2 ∧ 0^2 + 0^2 = r^2) → 
  (x - 1)^2 + (y - 1)^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3561_356172


namespace NUMINAMATH_CALUDE_f_negative_a_l3561_356154

theorem f_negative_a (a : ℝ) (f : ℝ → ℝ) (h : f = λ x ↦ x^3 * Real.cos x + 1) (h_fa : f a = 11) :
  f (-a) = -9 := by
sorry

end NUMINAMATH_CALUDE_f_negative_a_l3561_356154


namespace NUMINAMATH_CALUDE_hiker_walking_problem_l3561_356130

/-- A hiker's walking problem over three days -/
theorem hiker_walking_problem 
  (day1_distance : ℝ) 
  (day1_speed : ℝ) 
  (day2_speed_increase : ℝ) 
  (day3_speed : ℝ) 
  (day3_time : ℝ) 
  (total_distance : ℝ) 
  (h1 : day1_distance = 18) 
  (h2 : day1_speed = 3) 
  (h3 : day2_speed_increase = 1) 
  (h4 : day3_speed = 5) 
  (h5 : day3_time = 6) 
  (h6 : total_distance = 68) :
  day1_distance / day1_speed - 
  (total_distance - day1_distance - day3_speed * day3_time) / (day1_speed + day2_speed_increase) = 1 := by
  sorry

end NUMINAMATH_CALUDE_hiker_walking_problem_l3561_356130


namespace NUMINAMATH_CALUDE_partial_fraction_coefficient_sum_l3561_356164

theorem partial_fraction_coefficient_sum :
  ∀ (A B C D E : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) →
  A + B + C + D + E = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_coefficient_sum_l3561_356164


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l3561_356114

noncomputable section

-- Define the function f
def f (e a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

-- Define the derivative of f
def f' (e a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_and_max_value 
  (e : ℝ) (h_e : e > 0) :
  ∀ a b : ℝ, 
  (a > 0) → 
  (f' e a b (-1) = 0) →
  (
    -- Part I
    (a = 1) → 
    (∃ m c : ℝ, m = 1 ∧ c = 1 ∧ 
      ∀ x : ℝ, f e a b x = m * x + c → x = 0
    ) ∧
    -- Part II
    (a > 1/5) → 
    (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f e a b x ≤ 4 * e) →
    (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f e a b x = 4 * e) →
    (a = (8 * e^2 - 3) / 5 ∧ b = (12 * e^2 - 2) / 5)
  ) := by sorry

end NUMINAMATH_CALUDE_tangent_line_and_max_value_l3561_356114


namespace NUMINAMATH_CALUDE_qualification_rate_example_l3561_356143

/-- Calculates the qualification rate given the total number of boxes and the number of qualified boxes -/
def qualification_rate (total : ℕ) (qualified : ℕ) : ℚ :=
  (qualified : ℚ) / (total : ℚ) * 100

/-- Theorem stating that given 50 total boxes and 38 qualified boxes, the qualification rate is 76% -/
theorem qualification_rate_example : qualification_rate 50 38 = 76 := by
  sorry

end NUMINAMATH_CALUDE_qualification_rate_example_l3561_356143


namespace NUMINAMATH_CALUDE_eleven_remainders_l3561_356100

theorem eleven_remainders (A : Fin 100 → ℕ) 
  (h_perm : Function.Bijective A) 
  (h_range : ∀ i : Fin 100, A i ∈ Finset.range 101 \ {0}) : 
  let B : Fin 100 → ℕ := λ i => (Finset.range i.succ).sum (λ j => A j)
  Finset.card (Finset.image (λ i => B i % 100) Finset.univ) ≥ 11 := by
sorry

end NUMINAMATH_CALUDE_eleven_remainders_l3561_356100


namespace NUMINAMATH_CALUDE_slower_walking_speed_l3561_356113

theorem slower_walking_speed (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 40 → delay = 10 → 
  (usual_time / (usual_time + delay)) = (4 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_slower_walking_speed_l3561_356113


namespace NUMINAMATH_CALUDE_response_change_difference_l3561_356123

/-- Represents the percentages of student responses --/
structure ResponsePercentages where
  yes : ℝ
  no : ℝ
  undecided : ℝ

/-- The problem statement --/
theorem response_change_difference
  (initial : ResponsePercentages)
  (final : ResponsePercentages)
  (h_initial_sum : initial.yes + initial.no + initial.undecided = 100)
  (h_final_sum : final.yes + final.no + final.undecided = 100)
  (h_initial_yes : initial.yes = 40)
  (h_initial_no : initial.no = 30)
  (h_initial_undecided : initial.undecided = 30)
  (h_final_yes : final.yes = 60)
  (h_final_no : final.no = 10)
  (h_final_undecided : final.undecided = 30) :
  ∃ (min_change max_change : ℝ),
    (∀ (change : ℝ), min_change ≤ change ∧ change ≤ max_change) ∧
    max_change - min_change = 20 :=
sorry

end NUMINAMATH_CALUDE_response_change_difference_l3561_356123


namespace NUMINAMATH_CALUDE_absolute_value_condition_l3561_356178

theorem absolute_value_condition (x : ℝ) : |x - 1| = 1 - x → x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_condition_l3561_356178


namespace NUMINAMATH_CALUDE_number_problem_l3561_356197

theorem number_problem (x : ℝ) : 
  3 - (1/4 * 2) - (1/3 * 3) - (1/7 * x) = 27 → 
  (10/100) * x = 17.85 := by
sorry

end NUMINAMATH_CALUDE_number_problem_l3561_356197


namespace NUMINAMATH_CALUDE_cone_height_l3561_356103

theorem cone_height (r l h : ℝ) : 
  r = 1 → l = 4 → l^2 = r^2 + h^2 → h = Real.sqrt 15 := by sorry

end NUMINAMATH_CALUDE_cone_height_l3561_356103


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3561_356171

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 18) : 
  max x (max (x + 1) (x + 2)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3561_356171


namespace NUMINAMATH_CALUDE_square_side_ratio_sum_l3561_356180

theorem square_side_ratio_sum (area_ratio : ℚ) : 
  area_ratio = 128 / 50 →
  ∃ (p q r : ℕ), 
    (p * Real.sqrt q : ℝ) / r = Real.sqrt (area_ratio) ∧
    p + q + r = 14 :=
by sorry

end NUMINAMATH_CALUDE_square_side_ratio_sum_l3561_356180


namespace NUMINAMATH_CALUDE_inscribed_angles_sum_l3561_356151

theorem inscribed_angles_sum (circle : Real) (x y : Real) : 
  (circle > 0) →  -- circle has positive circumference
  (x = (2 / 12) * circle) →  -- x subtends 2/12 of the circle
  (y = (4 / 12) * circle) →  -- y subtends 4/12 of the circle
  (∃ (central_x central_y : Real), 
    central_x = 2 * x ∧ 
    central_y = 2 * y ∧ 
    central_x + central_y = circle) →  -- inscribed angle theorem
  x + y = 90 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_angles_sum_l3561_356151


namespace NUMINAMATH_CALUDE_chicken_difference_l3561_356153

/-- The number of Rhode Island Reds Susie has -/
def susie_rir : ℕ := 11

/-- The number of Golden Comets Susie has -/
def susie_gc : ℕ := 6

/-- The number of Rhode Island Reds Britney has -/
def britney_rir : ℕ := 2 * susie_rir

/-- The number of Golden Comets Britney has -/
def britney_gc : ℕ := susie_gc / 2

/-- The total number of chickens Susie has -/
def susie_total : ℕ := susie_rir + susie_gc

/-- The total number of chickens Britney has -/
def britney_total : ℕ := britney_rir + britney_gc

theorem chicken_difference : britney_total - susie_total = 8 := by
  sorry

end NUMINAMATH_CALUDE_chicken_difference_l3561_356153


namespace NUMINAMATH_CALUDE_power_product_equality_l3561_356109

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3561_356109


namespace NUMINAMATH_CALUDE_quadratic_completed_square_l3561_356177

theorem quadratic_completed_square (b : ℝ) (m : ℝ) :
  (∀ x, x^2 + b*x + 1/6 = (x + m)^2 + 1/12) → b = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completed_square_l3561_356177


namespace NUMINAMATH_CALUDE_comics_after_reassembly_l3561_356187

/-- The number of comics in the box after reassembly -/
def total_comics (pages_per_comic : ℕ) (extra_pages : ℕ) (total_pages : ℕ) (untorn_comics : ℕ) : ℕ :=
  untorn_comics + (total_pages - extra_pages) / pages_per_comic

/-- Theorem stating the total number of comics after reassembly -/
theorem comics_after_reassembly :
  total_comics 47 3 3256 20 = 89 := by
  sorry

end NUMINAMATH_CALUDE_comics_after_reassembly_l3561_356187


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3561_356181

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum1 : a + b = 35)
  (sum2 : b + c = 42)
  (sum3 : c + a = 58) :
  a + b + c = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3561_356181


namespace NUMINAMATH_CALUDE_weight_difference_proof_l3561_356156

/-- Proves that the difference between the average weight of two departing students
    and Joe's weight is 7 kg, given the conditions of the problem. -/
theorem weight_difference_proof
  (initial_avg : ℝ)
  (joe_weight : ℝ)
  (new_avg : ℝ)
  (final_avg : ℝ)
  (h_initial_avg : initial_avg = 30)
  (h_joe_weight : joe_weight = 44)
  (h_new_avg : new_avg = initial_avg + 1)
  (h_final_avg : final_avg = initial_avg)
  : ∃ (n : ℕ) (departing_avg : ℝ),
    (n : ℝ) * initial_avg + joe_weight = (n + 1 : ℝ) * new_avg ∧
    (n + 1 : ℝ) * new_avg - departing_avg * 2 = (n - 1 : ℝ) * final_avg ∧
    joe_weight - departing_avg = 7 :=
by sorry

end NUMINAMATH_CALUDE_weight_difference_proof_l3561_356156


namespace NUMINAMATH_CALUDE_product_equals_square_l3561_356194

theorem product_equals_square : 
  250 * 9.996 * 3.996 * 500 = (4998 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l3561_356194


namespace NUMINAMATH_CALUDE_eggs_per_tray_calculation_l3561_356120

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 45

/-- The number of trays bought weekly -/
def trays_per_week : ℕ := 2

/-- The number of children -/
def num_children : ℕ := 2

/-- The number of eggs eaten by each child daily -/
def child_eggs_per_day : ℕ := 2

/-- The number of adults -/
def num_adults : ℕ := 2

/-- The number of eggs eaten by each adult daily -/
def adult_eggs_per_day : ℕ := 4

/-- The number of eggs left uneaten weekly -/
def uneaten_eggs_per_week : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem eggs_per_tray_calculation :
  eggs_per_tray * trays_per_week = 
    num_children * child_eggs_per_day * days_per_week +
    num_adults * adult_eggs_per_day * days_per_week +
    uneaten_eggs_per_week :=
by sorry

end NUMINAMATH_CALUDE_eggs_per_tray_calculation_l3561_356120


namespace NUMINAMATH_CALUDE_ice_cube_distribution_l3561_356155

theorem ice_cube_distribution (total_ice_cubes : ℕ) (ice_cubes_per_cup : ℕ) (h1 : total_ice_cubes = 30) (h2 : ice_cubes_per_cup = 5) :
  total_ice_cubes / ice_cubes_per_cup = 6 := by
  sorry

end NUMINAMATH_CALUDE_ice_cube_distribution_l3561_356155


namespace NUMINAMATH_CALUDE_age_ratio_proof_l3561_356188

/-- Given the ages of three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 17 →  -- total age is 17
  b = 6 →  -- b is 6 years old
  b = 2 * c  -- the ratio of b's age to c's age is 2:1
  := by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l3561_356188


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l3561_356185

theorem multiply_mixed_number : 8 * (12 + 2/5) = 99 + 1/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l3561_356185


namespace NUMINAMATH_CALUDE_binary_octal_equivalence_l3561_356119

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 8^i) 0

/-- The binary number 1001101₂ is equal to the octal number 115₈ -/
theorem binary_octal_equivalence : 
  binary_to_decimal [1, 0, 1, 1, 0, 0, 1] = octal_to_decimal [5, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_octal_equivalence_l3561_356119


namespace NUMINAMATH_CALUDE_zero_in_interval_l3561_356105

noncomputable def f (x : ℝ) : ℝ := 6 / x - Real.log x / Real.log 2

theorem zero_in_interval :
  ∃ c : ℝ, 2 < c ∧ c < 4 ∧ f c = 0 := by sorry

end NUMINAMATH_CALUDE_zero_in_interval_l3561_356105


namespace NUMINAMATH_CALUDE_equation_solution_l3561_356195

theorem equation_solution : ∃! x : ℝ, 4*x - 2*x + 1 - 3 = 0 :=
by
  use 1
  constructor
  · -- Prove that 1 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3561_356195


namespace NUMINAMATH_CALUDE_twentieth_term_of_combined_sequence_l3561_356125

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

def geometric_sequence (g₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := g₁ * r ^ (n - 1)

def combined_sequence (a₁ g₁ d r : ℝ) (n : ℕ) : ℝ :=
  arithmetic_sequence a₁ d n + geometric_sequence g₁ r n

theorem twentieth_term_of_combined_sequence :
  combined_sequence 3 2 4 2 20 = 1048655 := by sorry

end NUMINAMATH_CALUDE_twentieth_term_of_combined_sequence_l3561_356125


namespace NUMINAMATH_CALUDE_two_equal_roots_sum_l3561_356133

theorem two_equal_roots_sum (a : ℝ) (α β : ℝ) :
  (∃! (x : ℝ), x ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin x + 4 * Real.cos x = a) →
  (α ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin α + 4 * Real.cos α = a) →
  (β ∈ Set.Ioo 0 (2 * Real.pi) ∧ 3 * Real.sin β + 4 * Real.cos β = a) →
  (α + β = Real.pi - 2 * Real.arcsin (4/5) ∨ α + β = 3 * Real.pi - 2 * Real.arcsin (4/5)) :=
by sorry


end NUMINAMATH_CALUDE_two_equal_roots_sum_l3561_356133


namespace NUMINAMATH_CALUDE_area_max_opposite_angles_sum_pi_l3561_356127

/-- A quadrilateral with sides a, b, c, d and angles α, β, γ, δ. -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  angle_sum : α + β + γ + δ = 2 * Real.pi

/-- The area of a quadrilateral. -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Theorem: The area of a quadrilateral is maximized when the sum of its opposite angles is π (180°). -/
theorem area_max_opposite_angles_sum_pi (q : Quadrilateral) :
  ∀ q' : Quadrilateral, q'.a = q.a ∧ q'.b = q.b ∧ q'.c = q.c ∧ q'.d = q.d →
  area q' ≤ area q ↔ q.α + q.γ = Real.pi ∧ q.β + q.δ = Real.pi :=
sorry

end NUMINAMATH_CALUDE_area_max_opposite_angles_sum_pi_l3561_356127


namespace NUMINAMATH_CALUDE_parallel_vector_scalar_l3561_356169

/-- Given vectors a, b, and c in ℝ², prove that if a + kb is parallel to c, then k = 1/2 -/
theorem parallel_vector_scalar (a b c : ℝ × ℝ) (h : a = (2, -1) ∧ b = (1, 1) ∧ c = (-5, 1)) :
  (∃ k : ℝ, (a.1 + k * b.1, a.2 + k * b.2).1 * c.2 = (a.1 + k * b.1, a.2 + k * b.2).2 * c.1) →
  (∃ k : ℝ, k = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_parallel_vector_scalar_l3561_356169


namespace NUMINAMATH_CALUDE_fraction_equality_l3561_356149

theorem fraction_equality : (1 + 5) / (3 + 5) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3561_356149
