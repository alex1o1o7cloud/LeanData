import Mathlib

namespace NUMINAMATH_CALUDE_negation_at_most_two_solutions_l3300_330039

/-- Negation of "at most n" is "at least n+1" -/
axiom negation_at_most (n : ℕ) : ¬(∀ m : ℕ, m ≤ n) ↔ ∃ m : ℕ, m ≥ n + 1

/-- The negation of "there are at most two solutions" is equivalent to "there are at least three solutions" -/
theorem negation_at_most_two_solutions :
  ¬(∃ S : Set ℕ, (∀ n ∈ S, n ≤ 2)) ↔ ∃ S : Set ℕ, (∃ n ∈ S, n ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_negation_at_most_two_solutions_l3300_330039


namespace NUMINAMATH_CALUDE_store_sales_theorem_l3300_330067

/-- Represents the daily sales and profit calculations for a store. -/
structure StoreSales where
  initial_sales : ℕ
  initial_profit : ℕ
  sales_increase : ℕ
  min_profit : ℕ

/-- Calculates the new sales quantity after a price reduction. -/
def new_sales (s : StoreSales) (reduction : ℕ) : ℕ :=
  s.initial_sales + s.sales_increase * reduction

/-- Calculates the new profit per item after a price reduction. -/
def new_profit_per_item (s : StoreSales) (reduction : ℕ) : ℕ :=
  s.initial_profit - reduction

/-- Calculates the total daily profit after a price reduction. -/
def total_daily_profit (s : StoreSales) (reduction : ℕ) : ℕ :=
  (new_sales s reduction) * (new_profit_per_item s reduction)

/-- The main theorem stating the two parts of the problem. -/
theorem store_sales_theorem (s : StoreSales) 
    (h1 : s.initial_sales = 20)
    (h2 : s.initial_profit = 40)
    (h3 : s.sales_increase = 2)
    (h4 : s.min_profit = 25) : 
  (new_sales s 4 = 28) ∧ 
  (∃ (x : ℕ), x = 5 ∧ total_daily_profit s x = 1050 ∧ new_profit_per_item s x ≥ s.min_profit) := by
  sorry


end NUMINAMATH_CALUDE_store_sales_theorem_l3300_330067


namespace NUMINAMATH_CALUDE_solutions_to_equation_l3300_330098

def solution_set : Set ℂ := {
  (3 * Real.sqrt 2) / 2 + (3 * Real.sqrt 2) / 2 * Complex.I,
  -(3 * Real.sqrt 2) / 2 - (3 * Real.sqrt 2) / 2 * Complex.I,
  (3 * Real.sqrt 2) / 2 * Complex.I - (3 * Real.sqrt 2) / 2,
  -(3 * Real.sqrt 2) / 2 * Complex.I + (3 * Real.sqrt 2) / 2
}

theorem solutions_to_equation : 
  ∀ x : ℂ, x^4 + 81 = 0 ↔ x ∈ solution_set := by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_l3300_330098


namespace NUMINAMATH_CALUDE_blood_sample_count_l3300_330072

theorem blood_sample_count (total_cells : ℕ) (first_sample_cells : ℕ) : 
  total_cells = 7341 → first_sample_cells = 4221 → total_cells - first_sample_cells = 3120 := by
  sorry

end NUMINAMATH_CALUDE_blood_sample_count_l3300_330072


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3300_330062

-- Define the rhombus
structure Rhombus :=
  (side_length : ℝ)
  (diagonal_length : ℝ)

-- Define the conditions
def satisfies_equation (y : ℝ) : Prop :=
  y^2 - 7*y + 10 = 0

def is_valid_rhombus (r : Rhombus) : Prop :=
  r.diagonal_length = 6 ∧ satisfies_equation r.side_length

-- Theorem statement
theorem rhombus_perimeter (r : Rhombus) (h : is_valid_rhombus r) : 
  4 * r.side_length = 20 :=
sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3300_330062


namespace NUMINAMATH_CALUDE_triangle_side_value_l3300_330045

theorem triangle_side_value (m : ℝ) : m > 0 → 
  (3 + 4 > m ∧ 3 + m > 4 ∧ 4 + m > 3) →
  (m = 1 ∨ m = 5 ∨ m = 7 ∨ m = 9) →
  m = 5 := by sorry

end NUMINAMATH_CALUDE_triangle_side_value_l3300_330045


namespace NUMINAMATH_CALUDE_john_mary_difference_l3300_330092

/-- The number of chickens each person took -/
structure ChickenCount where
  ray : ℕ
  john : ℕ
  mary : ℕ

/-- The conditions of the chicken distribution problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.ray = 10 ∧
  c.john = c.ray + 11 ∧
  c.mary = c.ray + 6

/-- The theorem stating the difference between John's and Mary's chicken count -/
theorem john_mary_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.mary = 5 := by
  sorry

end NUMINAMATH_CALUDE_john_mary_difference_l3300_330092


namespace NUMINAMATH_CALUDE_fraction_equality_l3300_330042

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - 3 * b^2

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a + 2 * b - 2 * a * b^2

-- Theorem statement
theorem fraction_equality :
  (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3300_330042


namespace NUMINAMATH_CALUDE_zion_age_is_8_dad_age_relation_future_age_relation_l3300_330055

/-- Zion's current age in years -/
def zion_age : ℕ := 8

/-- Zion's dad's current age in years -/
def dad_age : ℕ := 4 * zion_age + 3

theorem zion_age_is_8 : zion_age = 8 := by sorry

theorem dad_age_relation : dad_age = 4 * zion_age + 3 := by sorry

theorem future_age_relation : dad_age + 10 = (zion_age + 10) + 27 := by sorry

end NUMINAMATH_CALUDE_zion_age_is_8_dad_age_relation_future_age_relation_l3300_330055


namespace NUMINAMATH_CALUDE_employee_pay_l3300_330044

theorem employee_pay (total : ℝ) (x y : ℝ) (h1 : total = 572) (h2 : x + y = total) (h3 : x = 1.2 * y) : y = 260 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l3300_330044


namespace NUMINAMATH_CALUDE_three_digit_numbers_count_l3300_330061

theorem three_digit_numbers_count : 
  let digits : Finset Nat := {1, 2, 3, 4, 5}
  (digits.card : Nat) ^ 3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_numbers_count_l3300_330061


namespace NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l3300_330007

def is_two_digit_prime (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem least_number_to_add_for_divisibility :
  ∃ (p : ℕ) (h : is_two_digit_prime p), ∀ (k : ℕ), k < 1 → ¬(∃ (q : ℕ), is_two_digit_prime q ∧ (54321 + k) % q = 0) :=
sorry

end NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l3300_330007


namespace NUMINAMATH_CALUDE_A_union_complement_B_eq_l3300_330064

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,4}

theorem A_union_complement_B_eq : A ∪ (U \ B) = {1,2,3,5} := by sorry

end NUMINAMATH_CALUDE_A_union_complement_B_eq_l3300_330064


namespace NUMINAMATH_CALUDE_fraction_value_l3300_330089

theorem fraction_value (m n : ℝ) (h : 1/m - 1/n = 6) : m * n / (m - n) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3300_330089


namespace NUMINAMATH_CALUDE_pi_is_max_l3300_330077

theorem pi_is_max : ∀ (π : ℝ), π > 0 → (1 / 2023 : ℝ) > 0 → -2 * π < 0 →
  max (max (max 0 π) (1 / 2023)) (-2 * π) = π :=
by sorry

end NUMINAMATH_CALUDE_pi_is_max_l3300_330077


namespace NUMINAMATH_CALUDE_correct_operation_l3300_330037

theorem correct_operation (a b : ℝ) : 5 * a * b - 3 * a * b = 2 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3300_330037


namespace NUMINAMATH_CALUDE_proposition_uses_or_l3300_330065

-- Define the equation
def equation (x : ℝ) : Prop := x^2 = 4

-- Define the solution set
def solution_set : Set ℝ := {2, -2}

-- Define the proposition
def proposition : Prop := ∀ x, equation x ↔ x ∈ solution_set

-- Theorem: The proposition uses the "or" conjunction
theorem proposition_uses_or : 
  (∀ x, equation x ↔ (x = 2 ∨ x = -2)) ↔ proposition := by sorry

end NUMINAMATH_CALUDE_proposition_uses_or_l3300_330065


namespace NUMINAMATH_CALUDE_b_remaining_work_days_l3300_330005

-- Define the work rates and time periods
def a_rate : ℚ := 1 / 4
def b_rate : ℚ := 1 / 14
def initial_work_days : ℕ := 2

-- Theorem statement
theorem b_remaining_work_days :
  let total_work : ℚ := 1
  let combined_rate : ℚ := a_rate + b_rate
  let work_done_together : ℚ := combined_rate * initial_work_days
  let remaining_work : ℚ := total_work - work_done_together
  (remaining_work / b_rate : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_b_remaining_work_days_l3300_330005


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3300_330050

theorem arithmetic_mean_problem (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 84 → a = 32 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3300_330050


namespace NUMINAMATH_CALUDE_isosceles_triangles_in_square_l3300_330030

theorem isosceles_triangles_in_square (s : ℝ) (h : s = 2) :
  let square_area := s^2
  let triangle_area := square_area / 4
  let half_base := s / 2
  let height := triangle_area / half_base
  let side_length := Real.sqrt (half_base^2 + height^2)
  side_length = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_in_square_l3300_330030


namespace NUMINAMATH_CALUDE_unbounded_digit_sum_ratio_l3300_330090

/-- Sum of digits of a natural number in base 10 -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For any positive real constant c, there exists a natural number n 
    such that the ratio of sum of digits of n to sum of digits of n^2 exceeds c -/
theorem unbounded_digit_sum_ratio :
  ∀ c : ℝ, c > 0 → ∃ n : ℕ, (sum_of_digits n : ℝ) / (sum_of_digits (n^2)) > c :=
sorry

end NUMINAMATH_CALUDE_unbounded_digit_sum_ratio_l3300_330090


namespace NUMINAMATH_CALUDE_common_root_values_l3300_330056

theorem common_root_values (a b c d k : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : a * k^3 + b * k^2 + c * k + d = 0)
  (h2 : b * k^3 + c * k^2 + d * k + a = 0) :
  k = 1 ∨ k = -1 ∨ k = Complex.I ∨ k = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_common_root_values_l3300_330056


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l3300_330001

-- Define the polynomial function g(x)
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

-- Theorem statement
theorem polynomial_value_theorem (p q r s : ℝ) :
  g p q r s (-1) = 4 → 6*p - 3*q + r - 2*s = -24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l3300_330001


namespace NUMINAMATH_CALUDE_calculation_proof_l3300_330031

theorem calculation_proof : 0.25^2005 * 4^2006 - 8^100 * 0.5^300 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3300_330031


namespace NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l3300_330084

theorem square_sum_equals_product_implies_zero (x y z : ℤ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_product_implies_zero_l3300_330084


namespace NUMINAMATH_CALUDE_g_of_three_equals_five_l3300_330033

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define g in terms of f
def g (x : ℝ) : ℝ := f (x - 2)

-- Theorem to prove
theorem g_of_three_equals_five : g 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_equals_five_l3300_330033


namespace NUMINAMATH_CALUDE_hypotenuse_product_squared_l3300_330087

-- Define the triangles
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the problem
def triangle_problem (T1 T2 : RightTriangle) : Prop :=
  -- Areas of the triangles
  T1.leg1 * T1.leg2 / 2 = 2 ∧
  T2.leg1 * T2.leg2 / 2 = 3 ∧
  -- Congruent sides
  (T1.leg1 = T2.leg1 ∨ T1.leg1 = T2.leg2) ∧
  (T1.leg2 = T2.leg1 ∨ T1.leg2 = T2.leg2) ∧
  -- Similar triangles
  T1.leg1 / T2.leg1 = T1.leg2 / T2.leg2

-- Theorem statement
theorem hypotenuse_product_squared (T1 T2 : RightTriangle) 
  (h : triangle_problem T1 T2) : 
  (T1.hypotenuse * T2.hypotenuse)^2 = 9216 / 25 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_product_squared_l3300_330087


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l3300_330074

/-- The sequence defined by a₀ = a₁ = a₂ = 1 and a_{n+2} = (a_n * a_{n+1} + 1) / a_{n-1} for n ≥ 1 -/
def a : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 1
| (n + 3) => (a n * a (n + 1) + 1) / a (n - 1)

/-- The property that needs to be satisfied by the triples (a, b, c) -/
def satisfies_equation (a b c : ℕ) : Prop :=
  1 / a + 1 / b + 1 / c + 1 / (a * b * c) = 12 / (a + b + c)

theorem infinitely_many_solutions :
  ∀ N : ℕ, ∃ a b c : ℕ, a > N ∧ b > N ∧ c > N ∧ satisfies_equation a b c :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l3300_330074


namespace NUMINAMATH_CALUDE_curve_W_properties_l3300_330027

-- Define the curve W
def W (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + 1)^2) * Real.sqrt (x^2 + (y - 1)^2) = 3

-- Theorem stating the properties of curve W
theorem curve_W_properties :
  -- 1. x = 0 is an axis of symmetry
  (∀ y : ℝ, W 0 y ↔ W 0 (-y)) ∧
  -- 2. (0, 2) and (0, -2) are points on W
  W 0 2 ∧ W 0 (-2) ∧
  -- 3. The range of y-coordinates is [-2, 2]
  (∀ x y : ℝ, W x y → -2 ≤ y ∧ y ≤ 2) ∧
  (∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → ∃ x : ℝ, W x y) :=
by sorry


end NUMINAMATH_CALUDE_curve_W_properties_l3300_330027


namespace NUMINAMATH_CALUDE_log_equation_implies_sum_l3300_330008

theorem log_equation_implies_sum (u v : ℝ) 
  (hu : u > 1) (hv : v > 1)
  (h : (Real.log u / Real.log 3)^3 + (Real.log v / Real.log 5)^3 + 6 = 
       6 * (Real.log u / Real.log 3) * (Real.log v / Real.log 5)) :
  u^Real.sqrt 3 + v^Real.sqrt 3 = 152 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_implies_sum_l3300_330008


namespace NUMINAMATH_CALUDE_taxi_problem_l3300_330097

/-- Represents the direction of travel -/
inductive Direction
| East
| West

/-- Represents a single trip -/
structure Trip where
  distance : ℝ
  direction : Direction

def trips : List Trip := [
  ⟨8, Direction.East⟩, ⟨6, Direction.West⟩, ⟨3, Direction.East⟩,
  ⟨7, Direction.West⟩, ⟨8, Direction.East⟩, ⟨4, Direction.East⟩,
  ⟨9, Direction.West⟩, ⟨4, Direction.West⟩, ⟨3, Direction.East⟩,
  ⟨3, Direction.East⟩
]

def totalTime : ℝ := 1.25 -- in hours

def startingFare : ℝ := 8
def additionalFarePerKm : ℝ := 2
def freeDistance : ℝ := 3

theorem taxi_problem (trips : List Trip) (totalTime : ℝ) 
    (startingFare additionalFarePerKm freeDistance : ℝ) :
  -- 1. Final position is 3 km east
  (trips.foldl (fun acc trip => 
    match trip.direction with
    | Direction.East => acc + trip.distance
    | Direction.West => acc - trip.distance
  ) 0 = 3) ∧
  -- 2. Average speed is 44 km/h
  ((trips.foldl (fun acc trip => acc + trip.distance) 0) / totalTime = 44) ∧
  -- 3. Total earnings are 130 yuan
  (trips.length * startingFare + 
   (trips.foldl (fun acc trip => acc + max (trip.distance - freeDistance) 0) 0) * additionalFarePerKm = 130) := by
  sorry

end NUMINAMATH_CALUDE_taxi_problem_l3300_330097


namespace NUMINAMATH_CALUDE_system_of_equations_range_l3300_330000

theorem system_of_equations_range (x y m : ℝ) : 
  x + 2*y = 1 + m →
  2*x + y = 3 →
  x + y > 0 →
  m > -4 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_range_l3300_330000


namespace NUMINAMATH_CALUDE_simplify_expression_l3300_330075

theorem simplify_expression (a : ℝ) : (a^2)^3 + 3*a^4*a^2 - a^8/a^2 = 3*a^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3300_330075


namespace NUMINAMATH_CALUDE_simplify_expression_l3300_330019

theorem simplify_expression (x : ℝ) : 125 * x - 57 * x = 68 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3300_330019


namespace NUMINAMATH_CALUDE_organize_toys_time_l3300_330024

/-- The time in minutes it takes to organize all toys given the following conditions:
  * There are 50 toys to organize
  * 4 toys are put into the box every 45 seconds
  * 3 toys are taken out immediately after each 45-second interval
-/
def organizeToys (totalToys : ℕ) (putIn : ℕ) (takeOut : ℕ) (cycleTime : ℚ) : ℚ :=
  let netIncrease : ℕ := putIn - takeOut
  let almostFullCycles : ℕ := (totalToys - putIn) / netIncrease
  let almostFullTime : ℚ := (almostFullCycles : ℚ) * cycleTime
  let finalCycleTime : ℚ := cycleTime
  (almostFullTime + finalCycleTime) / 60

theorem organize_toys_time :
  organizeToys 50 4 3 (45 / 60) = 35.25 := by
  sorry

end NUMINAMATH_CALUDE_organize_toys_time_l3300_330024


namespace NUMINAMATH_CALUDE_jerrys_age_l3300_330066

/-- Given that Mickey's age is 10 years more than 200% of Jerry's age,
    and Mickey is 22 years old, Jerry's age is 6 years. -/
theorem jerrys_age (mickey jerry : ℕ) 
  (h1 : mickey = 2 * jerry + 10)  -- Mickey's age relation to Jerry's
  (h2 : mickey = 22)              -- Mickey's age
  : jerry = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_age_l3300_330066


namespace NUMINAMATH_CALUDE_toms_common_cards_l3300_330034

theorem toms_common_cards (rare_count : ℕ) (uncommon_count : ℕ) (rare_cost : ℚ) (uncommon_cost : ℚ) (common_cost : ℚ) (total_cost : ℚ) :
  rare_count = 19 →
  uncommon_count = 11 →
  rare_cost = 1 →
  uncommon_cost = 1/2 →
  common_cost = 1/4 →
  total_cost = 32 →
  (total_cost - (rare_count * rare_cost + uncommon_count * uncommon_cost)) / common_cost = 30 := by
  sorry

#eval (32 : ℚ) - (19 * 1 + 11 * (1/2 : ℚ)) / (1/4 : ℚ)

end NUMINAMATH_CALUDE_toms_common_cards_l3300_330034


namespace NUMINAMATH_CALUDE_middle_integer_is_five_l3300_330025

/-- Given three consecutive one-digit, positive, odd integers where their sum is
    one-seventh of their product, the middle integer is 5. -/
theorem middle_integer_is_five : 
  ∀ n : ℕ, 
    (n > 0 ∧ n < 10) →  -- one-digit positive integer
    (n % 2 = 1) →  -- odd integer
    (∃ (a b : ℕ), a = n - 2 ∧ b = n + 2 ∧  -- consecutive odd integers
      a > 0 ∧ b < 10 ∧  -- all are one-digit positive
      a % 2 = 1 ∧ b % 2 = 1 ∧  -- all are odd
      (a + n + b) = (a * n * b) / 7) →  -- sum is one-seventh of product
    n = 5 :=
by sorry

end NUMINAMATH_CALUDE_middle_integer_is_five_l3300_330025


namespace NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l3300_330048

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a - 1}
def B : Set ℝ := {x | x ≤ 3 ∨ x > 5}

-- Theorem 1: When a = 4, A ∩ B = {6, 7}
theorem intersection_when_a_is_4 : A 4 ∩ B = {6, 7} := by sorry

-- Theorem 2: A ⊆ B if and only if a < 2 or a > 4
theorem subset_condition (a : ℝ) : A a ⊆ B ↔ a < 2 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_4_subset_condition_l3300_330048


namespace NUMINAMATH_CALUDE_sum_percentage_l3300_330088

theorem sum_percentage (A B : ℝ) : 
  (0.4 * A = 160) → 
  (160 = (2/3) * B) → 
  (0.6 * (A + B) = 384) := by
sorry

end NUMINAMATH_CALUDE_sum_percentage_l3300_330088


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l3300_330043

/-- The time required for bacteria growth under specific conditions -/
theorem bacteria_growth_time (initial_count : ℕ) (final_count : ℕ) (growth_factor : ℕ) (growth_time : ℕ) (total_time : ℕ) : 
  initial_count = 200 →
  final_count = 145800 →
  growth_factor = 3 →
  growth_time = 3 →
  (initial_count * growth_factor ^ (total_time / growth_time) = final_count) →
  total_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l3300_330043


namespace NUMINAMATH_CALUDE_books_left_to_read_l3300_330020

theorem books_left_to_read (total_books read_books : ℕ) : 
  total_books = 13 → read_books = 9 → total_books - read_books = 4 := by
  sorry

end NUMINAMATH_CALUDE_books_left_to_read_l3300_330020


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_l3300_330073

def initial_sequence : List ℕ := [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

def append_number (seq : List ℕ) (n : ℕ) : List ℕ :=
  seq ++ [n]

def to_single_number (seq : List ℕ) : ℕ :=
  seq.foldl (λ acc x => acc * 10^(Nat.digits 10 x).length + x) 0

def is_divisible_by_12 (n : ℕ) : Prop :=
  n % 12 = 0

theorem smallest_divisible_by_12 :
  ∃ N : ℕ, N ≥ 82 ∧
    is_divisible_by_12 (to_single_number (append_number initial_sequence N)) ∧
    ∀ k : ℕ, 82 ≤ k ∧ k < N →
      ¬is_divisible_by_12 (to_single_number (append_number initial_sequence k)) ∧
    N = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_l3300_330073


namespace NUMINAMATH_CALUDE_fraction_problem_l3300_330017

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (3 * n - 7 : ℚ) = 2 / 5 → n = 14 := by sorry

end NUMINAMATH_CALUDE_fraction_problem_l3300_330017


namespace NUMINAMATH_CALUDE_common_divisors_9180_10080_l3300_330070

theorem common_divisors_9180_10080 : 
  let a := 9180
  let b := 10080
  (a % 7 = 0) → 
  (b % 7 = 0) → 
  (Finset.filter (fun d => d ∣ a ∧ d ∣ b) (Finset.range (min a b + 1))).card = 36 := by
sorry

end NUMINAMATH_CALUDE_common_divisors_9180_10080_l3300_330070


namespace NUMINAMATH_CALUDE_cody_money_l3300_330014

def final_money (initial : ℕ) (birthday : ℕ) (game_cost : ℕ) : ℕ :=
  initial + birthday - game_cost

theorem cody_money : final_money 45 9 19 = 35 := by
  sorry

end NUMINAMATH_CALUDE_cody_money_l3300_330014


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3300_330086

theorem smallest_dual_base_representation :
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  (1 * a + 2 * 1 = 7) ∧
  (2 * b + 1 * 1 = 7) ∧
  (∀ (x y : ℕ), x > 2 → y > 2 → 1 * x + 2 * 1 = 2 * y + 1 * 1 → 1 * x + 2 * 1 ≥ 7) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3300_330086


namespace NUMINAMATH_CALUDE_arithmetic_to_harmonic_progression_l3300_330059

/-- Three non-zero real numbers form an arithmetic progression if and only if
    the difference between the second and first is equal to the difference between the third and second. -/
def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- Three non-zero real numbers form a harmonic progression if and only if
    the reciprocal of the middle term is the arithmetic mean of the reciprocals of the other two terms. -/
def is_harmonic_progression (a b c : ℝ) : Prop :=
  2 / b = 1 / a + 1 / c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

/-- If three non-zero real numbers form an arithmetic progression,
    then their reciprocals form a harmonic progression. -/
theorem arithmetic_to_harmonic_progression (a b c : ℝ) :
  is_arithmetic_progression a b c → is_harmonic_progression (1/a) (1/b) (1/c) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_to_harmonic_progression_l3300_330059


namespace NUMINAMATH_CALUDE_tank_capacity_l3300_330069

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  initial_volume : ℝ
  added_volume : ℝ
  final_volume : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.initial_volume = tank.capacity / 6)
  (h2 : tank.added_volume = 4)
  (h3 : tank.final_volume = tank.initial_volume + tank.added_volume)
  (h4 : tank.final_volume = tank.capacity / 5) :
  tank.capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l3300_330069


namespace NUMINAMATH_CALUDE_inverse_81_mod_101_l3300_330040

theorem inverse_81_mod_101 (h : (9 : ZMod 101)⁻¹ = 90) : (81 : ZMod 101)⁻¹ = 20 := by
  sorry

end NUMINAMATH_CALUDE_inverse_81_mod_101_l3300_330040


namespace NUMINAMATH_CALUDE_final_value_calculation_l3300_330094

theorem final_value_calculation : 
  let initial_number := 16
  let doubled := initial_number * 2
  let added_five := doubled + 5
  let final_value := added_five * 3
  final_value = 111 := by
sorry

end NUMINAMATH_CALUDE_final_value_calculation_l3300_330094


namespace NUMINAMATH_CALUDE_correct_calculation_l3300_330096

theorem correct_calculation (x : ℤ) (h : x + 238 = 637) : x - 382 = 17 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3300_330096


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3300_330063

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 2*x - 3) * (x^2 - 4*x + 4) < 0 ↔ -1 < x ∧ x < 3 ∧ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3300_330063


namespace NUMINAMATH_CALUDE_compare_base_6_and_base_2_l3300_330049

def base_6_to_decimal (n : ℕ) : ℕ := 
  (n / 10) * 6 + (n % 10)

def base_2_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 4 + ((n / 10) % 10) * 2 + (n % 10)

theorem compare_base_6_and_base_2 : 
  base_6_to_decimal 12 > base_2_to_decimal 101 := by
  sorry

end NUMINAMATH_CALUDE_compare_base_6_and_base_2_l3300_330049


namespace NUMINAMATH_CALUDE_unique_five_digit_divisible_by_72_l3300_330091

theorem unique_five_digit_divisible_by_72 :
  ∃! n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 72 = 0 ∧
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = a * 10000 + 6 * 1000 + 7 * 100 + 9 * 10 + b :=
by sorry

end NUMINAMATH_CALUDE_unique_five_digit_divisible_by_72_l3300_330091


namespace NUMINAMATH_CALUDE_kayla_apples_l3300_330038

theorem kayla_apples (total : ℕ) (kayla kylie : ℕ) : 
  total = 200 →
  total = kayla + kylie →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l3300_330038


namespace NUMINAMATH_CALUDE_closed_broken_line_length_lower_bound_l3300_330051

/-- A closed broken line on the surface of a unit cube -/
structure ClosedBrokenLine where
  /-- The line passes over the surface of the cube -/
  onSurface : Bool
  /-- The line has common points with all faces of the cube -/
  touchesAllFaces : Bool
  /-- The length of the line -/
  length : ℝ

/-- Theorem: The length of a closed broken line on a unit cube touching all faces is at least 3√2 -/
theorem closed_broken_line_length_lower_bound (line : ClosedBrokenLine) 
    (h1 : line.onSurface = true) 
    (h2 : line.touchesAllFaces = true) : 
  line.length ≥ 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_closed_broken_line_length_lower_bound_l3300_330051


namespace NUMINAMATH_CALUDE_petyas_friends_l3300_330057

/-- The number of stickers Petya gives to each friend in the first scenario -/
def stickers_per_friend_scenario1 : ℕ := 5

/-- The number of stickers Petya has left in the first scenario -/
def stickers_left_scenario1 : ℕ := 8

/-- The number of stickers Petya gives to each friend in the second scenario -/
def stickers_per_friend_scenario2 : ℕ := 6

/-- The number of additional stickers Petya needs in the second scenario -/
def additional_stickers_needed : ℕ := 11

/-- Petya's number of friends -/
def number_of_friends : ℕ := 19

theorem petyas_friends :
  (stickers_per_friend_scenario1 * number_of_friends + stickers_left_scenario1 =
   stickers_per_friend_scenario2 * number_of_friends - additional_stickers_needed) ∧
  (number_of_friends = 19) :=
by sorry

end NUMINAMATH_CALUDE_petyas_friends_l3300_330057


namespace NUMINAMATH_CALUDE_bathing_suit_combinations_total_combinations_l3300_330095

theorem bathing_suit_combinations : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ
  | men_styles, men_sizes, men_colors, women_styles, women_sizes, women_colors =>
    (men_styles * men_sizes * men_colors) + (women_styles * women_sizes * women_colors)

theorem total_combinations (men_styles men_sizes men_colors women_styles women_sizes women_colors : ℕ) :
  men_styles = 5 →
  men_sizes = 3 →
  men_colors = 4 →
  women_styles = 4 →
  women_sizes = 4 →
  women_colors = 5 →
  bathing_suit_combinations men_styles men_sizes men_colors women_styles women_sizes women_colors = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_bathing_suit_combinations_total_combinations_l3300_330095


namespace NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_l3300_330015

/-- Alphonse's jump function -/
def A (n : ℕ) : ℕ :=
  if n ≥ 8 then A (n - 8) + 1 else n

/-- Beryl's jump function -/
def B (n : ℕ) : ℕ :=
  if n ≥ 7 then B (n - 7) + 1 else n

/-- Theorem stating the existence of n > 200 where B(n) < A(n) -/
theorem exists_n_where_B_less_than_A :
  ∃ n : ℕ, n > 200 ∧ B n < A n :=
sorry

end NUMINAMATH_CALUDE_exists_n_where_B_less_than_A_l3300_330015


namespace NUMINAMATH_CALUDE_certain_number_solution_l3300_330058

theorem certain_number_solution : 
  ∃ x : ℝ, (0.02^2 + 0.52^2 + x^2) = 100 * (0.002^2 + 0.052^2 + 0.0035^2) ∧ x = 0.035 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_solution_l3300_330058


namespace NUMINAMATH_CALUDE_jays_family_female_guests_l3300_330076

theorem jays_family_female_guests 
  (total_guests : ℕ) 
  (female_percentage : ℚ) 
  (jays_family_percentage : ℚ) 
  (h1 : total_guests = 240)
  (h2 : female_percentage = 60 / 100)
  (h3 : jays_family_percentage = 50 / 100) :
  ↑total_guests * female_percentage * jays_family_percentage = 72 := by
  sorry

end NUMINAMATH_CALUDE_jays_family_female_guests_l3300_330076


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_with_difference_l3300_330009

theorem smallest_sum_of_squares_with_difference (x y : ℕ) : 
  x^2 - y^2 = 221 → 
  x^2 + y^2 ≥ 229 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_with_difference_l3300_330009


namespace NUMINAMATH_CALUDE_debate_team_girls_l3300_330052

/-- The number of girls on a debate team -/
def girls_on_team (total_students : ℕ) (boys : ℕ) : ℕ :=
  total_students - boys

theorem debate_team_girls :
  let total_students := 7 * 9
  let boys := 31
  girls_on_team total_students boys = 32 := by
  sorry

#check debate_team_girls

end NUMINAMATH_CALUDE_debate_team_girls_l3300_330052


namespace NUMINAMATH_CALUDE_original_triangle_area_l3300_330022

/-- Given a triangle whose dimensions are quintupled to form a new triangle with an area of 100 square feet,
    the area of the original triangle is 4 square feet. -/
theorem original_triangle_area (original : ℝ) (new : ℝ) : 
  new = 100 → 
  new = original * 25 → 
  original = 4 := by sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3300_330022


namespace NUMINAMATH_CALUDE_point_on_x_axis_point_neg_two_zero_on_x_axis_l3300_330006

/-- A point lies on the x-axis if and only if its y-coordinate is 0 -/
theorem point_on_x_axis (x y : ℝ) : 
  (x, y) ∈ {p : ℝ × ℝ | p.2 = 0} ↔ y = 0 := by sorry

/-- The point (-2, 0) lies on the x-axis -/
theorem point_neg_two_zero_on_x_axis : 
  (-2, 0) ∈ {p : ℝ × ℝ | p.2 = 0} := by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_point_neg_two_zero_on_x_axis_l3300_330006


namespace NUMINAMATH_CALUDE_max_ab_value_l3300_330053

noncomputable def g (x : ℝ) : ℝ := 2^x

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : g a * g b = 2) :
  ∀ (x y : ℝ), x > 0 → y > 0 → g x * g y = 2 → x * y ≤ a * b ∧ a * b = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l3300_330053


namespace NUMINAMATH_CALUDE_distinguishable_arrangements_l3300_330081

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 1
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem distinguishable_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 420 := by
  sorry

end NUMINAMATH_CALUDE_distinguishable_arrangements_l3300_330081


namespace NUMINAMATH_CALUDE_hexagonal_pyramid_impossible_equal_edge_and_slant_l3300_330003

structure RegularPyramid (n : ℕ) where
  baseEdgeLength : ℝ
  slantHeight : ℝ

/-- Theorem: In a regular hexagonal pyramid, it's impossible for the base edge length
    to be equal to the slant height. -/
theorem hexagonal_pyramid_impossible_equal_edge_and_slant :
  ¬∃ (p : RegularPyramid 6), p.baseEdgeLength = p.slantHeight :=
sorry

end NUMINAMATH_CALUDE_hexagonal_pyramid_impossible_equal_edge_and_slant_l3300_330003


namespace NUMINAMATH_CALUDE_team_discount_saving_l3300_330078

/-- Represents the prices for a brand's uniform items -/
structure BrandPrices where
  shirt : ℝ
  pants : ℝ
  socks : ℝ

/-- Represents the prices for customization -/
structure CustomizationPrices where
  name : ℝ
  number : ℝ

def teamSize : ℕ := 12

def brandA : BrandPrices := ⟨7.5, 15, 4.5⟩
def brandB : BrandPrices := ⟨10, 18, 6⟩

def discountedBrandA : BrandPrices := ⟨6.75, 13.5, 3.75⟩
def discountedBrandB : BrandPrices := ⟨9, 16.5, 5.5⟩

def customization : CustomizationPrices := ⟨5, 3⟩

def playersWithFullCustomization : ℕ := 11

theorem team_discount_saving :
  let regularCost := 
    teamSize * (brandA.shirt + customization.name + customization.number) +
    teamSize * brandB.pants +
    teamSize * brandA.socks
  let discountedCost := 
    playersWithFullCustomization * (discountedBrandA.shirt + customization.name + customization.number) +
    (discountedBrandA.shirt + customization.name) +
    teamSize * discountedBrandB.pants +
    teamSize * brandA.socks
  regularCost - discountedCost = 31 := by sorry

end NUMINAMATH_CALUDE_team_discount_saving_l3300_330078


namespace NUMINAMATH_CALUDE_bird_sanctuary_theorem_l3300_330023

def bird_sanctuary_problem (initial_storks initial_herons initial_sparrows : ℕ)
  (storks_left herons_left sparrows_arrived hummingbirds_arrived : ℕ) : ℤ :=
  let final_storks : ℕ := initial_storks - storks_left
  let final_herons : ℕ := initial_herons - herons_left
  let final_sparrows : ℕ := initial_sparrows + sparrows_arrived
  let final_hummingbirds : ℕ := hummingbirds_arrived
  let total_other_birds : ℕ := final_herons + final_sparrows + final_hummingbirds
  (final_storks : ℤ) - (total_other_birds : ℤ)

theorem bird_sanctuary_theorem :
  bird_sanctuary_problem 8 4 5 3 2 4 2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_bird_sanctuary_theorem_l3300_330023


namespace NUMINAMATH_CALUDE_birthday_candles_sharing_l3300_330029

/-- 
Given that Ambika has 4 birthday candles and Aniyah has 6 times as many,
this theorem proves that when they put their candles together and share them equally,
each will have 14 candles.
-/
theorem birthday_candles_sharing (ambika_candles : ℕ) (aniyah_multiplier : ℕ) :
  ambika_candles = 4 →
  aniyah_multiplier = 6 →
  let aniyah_candles := ambika_candles * aniyah_multiplier
  let total_candles := ambika_candles + aniyah_candles
  total_candles / 2 = 14 := by
  sorry


end NUMINAMATH_CALUDE_birthday_candles_sharing_l3300_330029


namespace NUMINAMATH_CALUDE_simplified_A_value_l3300_330011

theorem simplified_A_value (a : ℝ) : 
  let A := (a - 1) / (a + 2) * ((a^2 - 4) / (a^2 - 2*a + 1)) / (1 / (a - 1))
  (a^2 - a = 0) → A = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_A_value_l3300_330011


namespace NUMINAMATH_CALUDE_vector_addition_result_l3300_330060

theorem vector_addition_result :
  let a : ℝ × ℝ := (2, -1)
  let b : ℝ × ℝ := (-3, 4)
  2 • a + b = (1, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_addition_result_l3300_330060


namespace NUMINAMATH_CALUDE_runners_meeting_count_l3300_330082

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ
  direction : Int

/-- Calculates the number of meetings between two runners on a circular track -/
def calculateMeetings (runner1 runner2 : Runner) (trackLength : ℝ) (laps : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem runners_meeting_count :
  let runner1 : Runner := ⟨4, 1⟩
  let runner2 : Runner := ⟨10, -1⟩
  let trackLength : ℝ := 140
  let laps : ℕ := 28
  calculateMeetings runner1 runner2 trackLength laps = 77 := by sorry

end NUMINAMATH_CALUDE_runners_meeting_count_l3300_330082


namespace NUMINAMATH_CALUDE_system_solution_l3300_330035

theorem system_solution : ∃ (x y : ℝ), (7 * x - 3 * y = 2) ∧ (2 * x + y = 8) := by
  use 2, 4
  sorry

end NUMINAMATH_CALUDE_system_solution_l3300_330035


namespace NUMINAMATH_CALUDE_min_translation_value_l3300_330046

theorem min_translation_value (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (2 * x) + Real.cos (2 * x)) →
  (∀ x, g x = f (x - m)) →
  (m > 0) →
  (∀ x, g (-π/3) ≤ g x) →
  ∃ k : ℤ, m = k * π + π/24 ∧ 
  (∀ m' : ℝ, m' > 0 → (∀ x, g (-π/3) ≤ g x) → m' ≥ π/24) :=
sorry

end NUMINAMATH_CALUDE_min_translation_value_l3300_330046


namespace NUMINAMATH_CALUDE_candy_bar_difference_l3300_330004

/-- Given information about candy bars possessed by Lena, Kevin, and Nicole, 
    prove that Lena has 19.6 more candy bars than Nicole. -/
theorem candy_bar_difference (lena kevin nicole : ℝ) : 
  lena = 37.5 ∧ 
  lena + 9.5 = 5 * kevin ∧ 
  kevin = nicole - 8.5 → 
  lena - nicole = 19.6 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l3300_330004


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3300_330054

/-- A regular polygon with side length 8 units and exterior angle 90 degrees has a perimeter of 32 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) : 
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 90 ∧ 
  (360 : ℝ) / n = exterior_angle → 
  n * side_length = 32 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3300_330054


namespace NUMINAMATH_CALUDE_plane_flight_distance_l3300_330036

/-- Given a plane that flies with and against the wind, prove the distance flown against the wind -/
theorem plane_flight_distance 
  (distance_with_wind : ℝ) 
  (wind_speed : ℝ) 
  (plane_speed : ℝ) 
  (h1 : distance_with_wind = 420) 
  (h2 : wind_speed = 23) 
  (h3 : plane_speed = 253) : 
  (distance_with_wind * (plane_speed - wind_speed)) / (plane_speed + wind_speed) = 350 := by
  sorry

end NUMINAMATH_CALUDE_plane_flight_distance_l3300_330036


namespace NUMINAMATH_CALUDE_abc_inequality_l3300_330041

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : Real.sqrt a ^ 3 + Real.sqrt b ^ 3 + Real.sqrt c ^ 3 = 1) :
  a * b * c ≤ 1 / 9 ∧
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3300_330041


namespace NUMINAMATH_CALUDE_car_fuel_usage_l3300_330047

/-- Proves that a car traveling for 5 hours at 60 mph with a fuel efficiency of 1 gallon per 30 miles
    uses 5/6 of a 12-gallon tank. -/
theorem car_fuel_usage (speed : ℝ) (fuel_efficiency : ℝ) (tank_capacity : ℝ) (travel_time : ℝ) :
  speed = 60 →
  fuel_efficiency = 30 →
  tank_capacity = 12 →
  travel_time = 5 →
  (speed * travel_time / fuel_efficiency) / tank_capacity = 5 / 6 := by
  sorry


end NUMINAMATH_CALUDE_car_fuel_usage_l3300_330047


namespace NUMINAMATH_CALUDE_expression_evaluation_l3300_330013

theorem expression_evaluation : 121 + 2 * 11 * 4 + 16 + 7 = 232 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3300_330013


namespace NUMINAMATH_CALUDE_division_problem_l3300_330080

theorem division_problem (N : ℕ) (n : ℕ) (h1 : N > 0) :
  (∀ k : ℕ, k ≤ n → ∃ part : ℚ, part = N / (k * (k + 1))) →
  (N / (n * (n + 1)) = N / 400) →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3300_330080


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3300_330093

theorem fraction_to_decimal : (22 : ℚ) / 8 = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3300_330093


namespace NUMINAMATH_CALUDE_grocer_sales_problem_l3300_330068

theorem grocer_sales_problem (m1 m3 m4 m5 m6 avg : ℕ) (h1 : m1 = 4000) (h3 : m3 = 5689) (h4 : m4 = 7230) (h5 : m5 = 6000) (h6 : m6 = 12557) (havg : avg = 7000) :
  ∃ m2 : ℕ, m2 = 6524 ∧ (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg :=
sorry

end NUMINAMATH_CALUDE_grocer_sales_problem_l3300_330068


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3300_330079

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3300_330079


namespace NUMINAMATH_CALUDE_jons_toaster_cost_l3300_330028

/-- Calculates the total cost of a toaster purchase with given parameters. -/
def toaster_total_cost (msrp : ℝ) (standard_insurance_rate : ℝ) (premium_insurance_additional : ℝ) 
                       (tax_rate : ℝ) (recycling_fee : ℝ) : ℝ :=
  let standard_insurance := msrp * standard_insurance_rate
  let premium_insurance := standard_insurance + premium_insurance_additional
  let subtotal := msrp + premium_insurance
  let tax := subtotal * tax_rate
  subtotal + tax + recycling_fee

/-- Theorem stating that the total cost for Jon's toaster purchase is $69.50 -/
theorem jons_toaster_cost : 
  toaster_total_cost 30 0.2 7 0.5 5 = 69.5 := by
  sorry

end NUMINAMATH_CALUDE_jons_toaster_cost_l3300_330028


namespace NUMINAMATH_CALUDE_only_statement4_correct_l3300_330032

-- Define the structure of input/output statements
inductive Statement
| Input (vars : List String)
| InputAssign (var : String) (value : Nat)
| Print (expr : String)
| PrintMultiple (values : List Nat)

-- Define the rules for correct statements
def isCorrectInput (s : Statement) : Prop :=
  match s with
  | Statement.Input vars => vars.length > 0
  | Statement.InputAssign _ _ => false
  | _ => false

def isCorrectOutput (s : Statement) : Prop :=
  match s with
  | Statement.Print _ => false
  | Statement.PrintMultiple values => values.length > 0
  | _ => false

def isCorrect (s : Statement) : Prop :=
  isCorrectInput s ∨ isCorrectOutput s

-- Define the given statements
def statement1 : Statement := Statement.Input ["a", "b", "c"]
def statement2 : Statement := Statement.Print "a=1"
def statement3 : Statement := Statement.InputAssign "x" 2
def statement4 : Statement := Statement.PrintMultiple [20, 4]

-- Theorem to prove
theorem only_statement4_correct :
  ¬ isCorrect statement1 ∧
  ¬ isCorrect statement2 ∧
  ¬ isCorrect statement3 ∧
  isCorrect statement4 :=
sorry

end NUMINAMATH_CALUDE_only_statement4_correct_l3300_330032


namespace NUMINAMATH_CALUDE_odd_prob_greater_than_even_prob_l3300_330018

/-- Represents the number of beads in the bottle -/
def num_beads : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the total number of possible outcomes when pouring beads -/
def total_outcomes : ℕ := choose num_beads 1 + choose num_beads 2 + choose num_beads 3 + choose num_beads 4

/-- Calculates the number of outcomes resulting in an odd number of beads -/
def odd_outcomes : ℕ := choose num_beads 1 + choose num_beads 3

/-- Calculates the number of outcomes resulting in an even number of beads -/
def even_outcomes : ℕ := choose num_beads 2 + choose num_beads 4

/-- Theorem stating that the probability of pouring out an odd number of beads
    is greater than the probability of pouring out an even number of beads -/
theorem odd_prob_greater_than_even_prob :
  (odd_outcomes : ℚ) / total_outcomes > (even_outcomes : ℚ) / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_odd_prob_greater_than_even_prob_l3300_330018


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3300_330002

theorem complex_number_quadrant : ∃ (z : ℂ), z = 2 * Complex.I^3 / (1 - Complex.I) ∧ z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3300_330002


namespace NUMINAMATH_CALUDE_overtake_twice_implies_double_speed_l3300_330085

/-- Represents a runner with a constant speed -/
structure Runner where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents the race course -/
structure Course where
  distance_to_stadium : ℝ
  lap_length : ℝ
  total_laps : ℕ
  distance_to_stadium_pos : distance_to_stadium > 0
  lap_length_pos : lap_length > 0
  total_laps_pos : total_laps > 0

/-- Theorem: If a runner overtakes another runner twice in a race with three laps,
    then the faster runner's speed is at least twice the slower runner's speed -/
theorem overtake_twice_implies_double_speed
  (runner1 runner2 : Runner) (course : Course) :
  course.total_laps = 3 →
  (∃ (t1 t2 : ℝ), 0 < t1 ∧ t1 < t2 ∧
    runner1.speed * t1 = runner2.speed * t1 + course.lap_length ∧
    runner1.speed * t2 = runner2.speed * t2 + 2 * course.lap_length) →
  runner1.speed ≥ 2 * runner2.speed :=
by sorry

end NUMINAMATH_CALUDE_overtake_twice_implies_double_speed_l3300_330085


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3300_330016

theorem max_value_on_circle : 
  ∀ x y : ℝ, x^2 + y^2 - 6*x + 8 = 0 → x^2 + y^2 ≤ 16 ∧ ∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 6*x₀ + 8 = 0 ∧ x₀^2 + y₀^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3300_330016


namespace NUMINAMATH_CALUDE_abs_equation_solution_difference_l3300_330010

theorem abs_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15 ∧ |x₂ + 3| = 15) ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = 30 := by
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_difference_l3300_330010


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l3300_330071

def S : Set ℕ := {1, 2, 3, 4, 5}

def event1 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ (a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0)

def event2 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ ((a % 2 = 1 ∨ b % 2 = 1) ∧ (a % 2 = 1 ∧ b % 2 = 1))

def event3 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ ((a % 2 = 1 ∨ b % 2 = 1) ∧ (a % 2 = 0 ∧ b % 2 = 0))

def event4 (a b : ℕ) : Prop := (a ∈ S ∧ b ∈ S) ∧ (a % 2 = 1 ∨ b % 2 = 1) ∧ (a % 2 = 0 ∨ b % 2 = 0)

theorem mutually_exclusive_events :
  ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b →
  (¬(event1 a b ∧ event1 a b)) ∧
  (¬(event2 a b ∧ event2 a b)) ∧
  (¬(event3 a b ∧ event3 a b)) ∧
  (¬(event4 a b ∧ event4 a b)) ∧
  (∃ x y : ℕ, event1 x y ∧ event2 x y) ∧
  (∃ x y : ℕ, event1 x y ∧ event4 x y) ∧
  (∃ x y : ℕ, event2 x y ∧ event4 x y) ∧
  (∀ x y : ℕ, ¬(event3 x y ∧ event1 x y)) ∧
  (∀ x y : ℕ, ¬(event3 x y ∧ event2 x y)) ∧
  (∀ x y : ℕ, ¬(event3 x y ∧ event4 x y)) :=
by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l3300_330071


namespace NUMINAMATH_CALUDE_rachel_setup_time_l3300_330026

/-- Represents the time in hours for Rachel's speed painting process. -/
structure PaintingTime where
  setup : ℝ
  paintingPerVideo : ℝ
  cleanup : ℝ
  editAndPostPerVideo : ℝ
  totalPerVideo : ℝ
  batchSize : ℕ

/-- The setup time for Rachel's speed painting process is 1 hour. -/
theorem rachel_setup_time (t : PaintingTime) : t.setup = 1 :=
  by
  have h1 : t.paintingPerVideo = 1 := by sorry
  have h2 : t.cleanup = 1 := by sorry
  have h3 : t.editAndPostPerVideo = 1.5 := by sorry
  have h4 : t.totalPerVideo = 3 := by sorry
  have h5 : t.batchSize = 4 := by sorry
  
  have total_batch_time : t.setup + t.batchSize * (t.paintingPerVideo + t.editAndPostPerVideo) + t.cleanup = t.batchSize * t.totalPerVideo :=
    by sorry
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_rachel_setup_time_l3300_330026


namespace NUMINAMATH_CALUDE_convex_polygon_division_impossibility_l3300_330083

-- Define a polygon
def Polygon : Type := List (ℝ × ℝ)

-- Define a function to check if a polygon is convex
def isConvex (p : Polygon) : Prop := sorry

-- Define a function to check if a quadrilateral is non-convex
def isNonConvexQuadrilateral (q : Polygon) : Prop := sorry

-- Define a function to represent the division of a polygon into quadrilaterals
def divideIntoQuadrilaterals (p : Polygon) (qs : List Polygon) : Prop := sorry

theorem convex_polygon_division_impossibility (p : Polygon) (qs : List Polygon) :
  isConvex p → (∀ q ∈ qs, isNonConvexQuadrilateral q) → divideIntoQuadrilaterals p qs → False :=
by sorry

end NUMINAMATH_CALUDE_convex_polygon_division_impossibility_l3300_330083


namespace NUMINAMATH_CALUDE_no_roots_equation_l3300_330099

theorem no_roots_equation : ¬∃ (x : ℝ), x - 8 / (x - 4) = 4 - 8 / (x - 4) := by sorry

end NUMINAMATH_CALUDE_no_roots_equation_l3300_330099


namespace NUMINAMATH_CALUDE_parabola_intersection_range_l3300_330012

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus
def line (x y : ℝ) : Prop := y = x - 1

-- Define the circle E with AB as diameter
def circle_E (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = 16

-- Define the point D
def point_D (t : ℝ) : ℝ × ℝ := (-2, t)

theorem parabola_intersection_range (t : ℝ) :
  (∃ (A B P Q : ℝ × ℝ),
    parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
    line A.1 A.2 ∧ line B.1 B.2 ∧
    circle_E P.1 P.2 ∧ circle_E Q.1 Q.2 ∧
    (∃ (r : ℝ), (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 4*r^2 ∧
                ((point_D t).1 - P.1)^2 + ((point_D t).2 - P.2)^2 = r^2)) →
  2 - Real.sqrt 7 ≤ t ∧ t ≤ 2 + Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_range_l3300_330012


namespace NUMINAMATH_CALUDE_expression_evaluation_l3300_330021

theorem expression_evaluation : 
  (0.82 : Real)^3 - (0.1 : Real)^3 / (0.82 : Real)^2 + 0.082 + (0.1 : Real)^2 = 0.641881 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3300_330021
