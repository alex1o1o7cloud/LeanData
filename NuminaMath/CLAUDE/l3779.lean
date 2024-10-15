import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_cubes_minus_product_l3779_377913

theorem sum_of_cubes_minus_product (x y z : ℝ) 
  (h1 : x + y + z = 10) 
  (h2 : x*y + y*z + z*x = 20) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_minus_product_l3779_377913


namespace NUMINAMATH_CALUDE_fraction_without_finite_decimal_l3779_377947

def has_finite_decimal_expansion (n d : ℕ) : Prop :=
  ∃ k : ℕ, d * (10 ^ k) % n = 0

theorem fraction_without_finite_decimal : 
  has_finite_decimal_expansion 9 10 ∧ 
  has_finite_decimal_expansion 3 5 ∧ 
  ¬ has_finite_decimal_expansion 3 7 ∧ 
  has_finite_decimal_expansion 7 8 :=
sorry

end NUMINAMATH_CALUDE_fraction_without_finite_decimal_l3779_377947


namespace NUMINAMATH_CALUDE_expression_equality_l3779_377967

theorem expression_equality : 
  Real.sqrt (4/3) * Real.sqrt 15 + ((-8) ^ (1/3 : ℝ)) + (π - 3) ^ (0 : ℝ) = 2 * Real.sqrt 5 - 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l3779_377967


namespace NUMINAMATH_CALUDE_system_solution_l3779_377999

theorem system_solution (a b c : ℂ) : 
  (a^2 + a*b + c = 0 ∧ b^2 + b*c + a = 0 ∧ c^2 + c*a + b = 0) → 
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3779_377999


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3779_377920

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 9) / (Nat.factorial 4)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3779_377920


namespace NUMINAMATH_CALUDE_equal_sum_sequence_a8_l3779_377945

/-- An equal sum sequence is a sequence where the sum of each term and its next term is constant. --/
def EqualSumSequence (a : ℕ → ℝ) :=
  ∃ k : ℝ, ∀ n : ℕ, a n + a (n + 1) = k

/-- The common sum of an equal sum sequence. --/
def CommonSum (a : ℕ → ℝ) (k : ℝ) :=
  ∀ n : ℕ, a n + a (n + 1) = k

theorem equal_sum_sequence_a8 (a : ℕ → ℝ) (h1 : EqualSumSequence a) (h2 : a 1 = 2) (h3 : CommonSum a 5) :
  a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_a8_l3779_377945


namespace NUMINAMATH_CALUDE_cube_sum_geq_sqrt_product_square_sum_l3779_377988

theorem cube_sum_geq_sqrt_product_square_sum {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
  a^3 + b^3 ≥ Real.sqrt (a * b) * (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_geq_sqrt_product_square_sum_l3779_377988


namespace NUMINAMATH_CALUDE_height_of_cone_l3779_377974

/-- Theorem: Height of a cone with specific volume and vertex angle -/
theorem height_of_cone (V : ℝ) (angle : ℝ) (h : ℝ) :
  V = 16384 * Real.pi ∧ angle = 90 →
  h = (49152 : ℝ) ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_height_of_cone_l3779_377974


namespace NUMINAMATH_CALUDE_fib_1960_1988_gcd_l3779_377900

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The greatest common divisor of the 1960th and 1988th Fibonacci numbers is 317811 -/
theorem fib_1960_1988_gcd : Nat.gcd (fib 1988) (fib 1960) = 317811 := by
  sorry

end NUMINAMATH_CALUDE_fib_1960_1988_gcd_l3779_377900


namespace NUMINAMATH_CALUDE_probability_different_tens_digits_l3779_377962

/-- The number of integers in the range 10 to 79, inclusive. -/
def total_integers : ℕ := 70

/-- The number of different tens digits in the range 10 to 79. -/
def different_tens_digits : ℕ := 7

/-- The number of integers to be chosen. -/
def chosen_integers : ℕ := 6

/-- The number of integers for each tens digit. -/
def integers_per_tens : ℕ := 10

theorem probability_different_tens_digits :
  (different_tens_digits.choose chosen_integers * integers_per_tens ^ chosen_integers : ℚ) /
  (total_integers.choose chosen_integers) = 1750 / 2980131 := by sorry

end NUMINAMATH_CALUDE_probability_different_tens_digits_l3779_377962


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l3779_377930

theorem prime_squared_minus_one_divisible_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisible_by_24_l3779_377930


namespace NUMINAMATH_CALUDE_parabola_c_value_l3779_377984

/-- A parabola with equation y = ax^2 + bx + c, vertex at (3, -5), and passing through (0, -2) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 3
  vertex_y : ℝ := -5
  point_x : ℝ := 0
  point_y : ℝ := -2

/-- The c-value of the parabola is -2 -/
theorem parabola_c_value (p : Parabola) : p.c = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3779_377984


namespace NUMINAMATH_CALUDE_triangle_345_is_acute_l3779_377929

/-- A triangle with sides 3, 4, and 4.5 is acute. -/
theorem triangle_345_is_acute : 
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 4.5 → 
  (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_345_is_acute_l3779_377929


namespace NUMINAMATH_CALUDE_helen_gas_usage_l3779_377958

-- Define the number of months with 2 cuts and 4 cuts
def months_with_two_cuts : ℕ := 4
def months_with_four_cuts : ℕ := 4

-- Define the number of cuts per month for each category
def cuts_per_month_low : ℕ := 2
def cuts_per_month_high : ℕ := 4

-- Define the gas usage
def gas_per_fourth_cut : ℕ := 2
def cuts_per_gas_usage : ℕ := 4

-- Theorem statement
theorem helen_gas_usage :
  let total_cuts := months_with_two_cuts * cuts_per_month_low + months_with_four_cuts * cuts_per_month_high
  let gas_fill_ups := total_cuts / cuts_per_gas_usage
  gas_fill_ups * gas_per_fourth_cut = 12 := by
  sorry

end NUMINAMATH_CALUDE_helen_gas_usage_l3779_377958


namespace NUMINAMATH_CALUDE_john_roommates_l3779_377939

theorem john_roommates (bob_roommates : ℕ) (h1 : bob_roommates = 10) :
  let john_roommates := 2 * bob_roommates + 5
  john_roommates = 25 := by sorry

end NUMINAMATH_CALUDE_john_roommates_l3779_377939


namespace NUMINAMATH_CALUDE_sum_of_a_n_equals_1158_l3779_377904

def a (n : ℕ) : ℕ :=
  if n % 15 = 0 ∧ n % 10 = 0 then 12
  else if n % 10 = 0 ∧ n % 9 = 0 then 15
  else if n % 9 = 0 ∧ n % 15 = 0 then 10
  else 0

theorem sum_of_a_n_equals_1158 :
  (Finset.range 1499).sum a = 1158 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_n_equals_1158_l3779_377904


namespace NUMINAMATH_CALUDE_burger_sharing_l3779_377915

theorem burger_sharing (burger_length : ℝ) (brother_fraction : ℝ) (friend1_fraction : ℝ) (friend2_fraction : ℝ) :
  burger_length = 12 →
  brother_fraction = 1/3 →
  friend1_fraction = 1/4 →
  friend2_fraction = 1/2 →
  ∃ (brother_share friend1_share friend2_share valentina_share : ℝ),
    brother_share = burger_length * brother_fraction ∧
    friend1_share = (burger_length - brother_share) * friend1_fraction ∧
    friend2_share = (burger_length - brother_share - friend1_share) * friend2_fraction ∧
    valentina_share = burger_length - brother_share - friend1_share - friend2_share ∧
    brother_share = 4 ∧
    friend1_share = 2 ∧
    friend2_share = 3 ∧
    valentina_share = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_burger_sharing_l3779_377915


namespace NUMINAMATH_CALUDE_store_b_earns_more_l3779_377998

/-- Represents the total value of goods sold by each store in yuan -/
def total_sales : ℕ := 1000000

/-- Represents the discount rate offered by store A -/
def discount_rate : ℚ := 1/10

/-- Represents the cost of a lottery ticket in yuan -/
def ticket_cost : ℕ := 100

/-- Represents the number of tickets in a batch -/
def tickets_per_batch : ℕ := 10000

/-- Represents the prize structure for store B -/
structure PrizeStructure where
  first_prize : ℕ × ℕ  -- (number of prizes, value of each prize)
  second_prize : ℕ × ℕ
  third_prize : ℕ × ℕ
  fourth_prize : ℕ × ℕ
  fifth_prize : ℕ × ℕ

/-- The actual prize structure used by store B -/
def store_b_prizes : PrizeStructure := {
  first_prize := (5, 1000),
  second_prize := (10, 500),
  third_prize := (20, 200),
  fourth_prize := (40, 100),
  fifth_prize := (5000, 10)
}

/-- Calculates the total prize value for a given prize structure -/
def total_prize_value (ps : PrizeStructure) : ℕ :=
  ps.first_prize.1 * ps.first_prize.2 +
  ps.second_prize.1 * ps.second_prize.2 +
  ps.third_prize.1 * ps.third_prize.2 +
  ps.fourth_prize.1 * ps.fourth_prize.2 +
  ps.fifth_prize.1 * ps.fifth_prize.2

/-- Theorem stating that store B earns at least 32,000 yuan more than store A -/
theorem store_b_earns_more :
  ∃ (x : ℕ), x ≥ 32000 ∧
  (total_sales - (total_prize_value store_b_prizes) * (total_sales / (tickets_per_batch * ticket_cost))) =
  (total_sales * (1 - discount_rate)).floor + x :=
by sorry


end NUMINAMATH_CALUDE_store_b_earns_more_l3779_377998


namespace NUMINAMATH_CALUDE_average_decrease_l3779_377969

def initial_observations : ℕ := 6
def initial_average : ℚ := 12
def new_observation : ℕ := 5

theorem average_decrease :
  let initial_sum := initial_observations * initial_average
  let new_sum := initial_sum + new_observation
  let new_average := new_sum / (initial_observations + 1)
  initial_average - new_average = 1 := by
  sorry

end NUMINAMATH_CALUDE_average_decrease_l3779_377969


namespace NUMINAMATH_CALUDE_min_value_given_condition_l3779_377960

theorem min_value_given_condition (a b : ℝ) : 
  (|a - 2| + (b + 3)^2 = 0) → 
  (min (min (min (a + b) (a - b)) (b^a)) (a * b) = a * b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_given_condition_l3779_377960


namespace NUMINAMATH_CALUDE_intersection_S_T_l3779_377944

def S : Set ℝ := {x | (x - 3) / (x - 6) ≤ 0 ∧ x ≠ 6}
def T : Set ℝ := {2, 3, 4, 5, 6}

theorem intersection_S_T : S ∩ T = {3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_S_T_l3779_377944


namespace NUMINAMATH_CALUDE_city_population_problem_l3779_377982

theorem city_population_problem (population_b : ℕ) : 
  let population_a := (3 * population_b) / 5
  let population_c := 27500
  let total_population := population_a + population_b + population_c
  (population_c = (5 * population_b) / 4 + 4000) →
  (total_population % 250 = 0) →
  total_population = 57500 :=
by sorry

end NUMINAMATH_CALUDE_city_population_problem_l3779_377982


namespace NUMINAMATH_CALUDE_total_snake_owners_l3779_377950

/- Define the total number of pet owners -/
def total_pet_owners : ℕ := 120

/- Define the number of people owning specific combinations of pets -/
def only_dogs : ℕ := 25
def only_cats : ℕ := 18
def only_birds : ℕ := 12
def only_snakes : ℕ := 15
def only_hamsters : ℕ := 7
def cats_and_dogs : ℕ := 8
def dogs_and_birds : ℕ := 5
def cats_and_birds : ℕ := 6
def cats_and_snakes : ℕ := 7
def dogs_and_snakes : ℕ := 10
def dogs_and_hamsters : ℕ := 4
def cats_and_hamsters : ℕ := 3
def birds_and_hamsters : ℕ := 5
def birds_and_snakes : ℕ := 2
def snakes_and_hamsters : ℕ := 3
def cats_dogs_birds : ℕ := 3
def cats_dogs_snakes : ℕ := 4
def cats_snakes_hamsters : ℕ := 2
def all_pets : ℕ := 1

/- Theorem stating the total number of snake owners -/
theorem total_snake_owners : 
  only_snakes + cats_and_snakes + dogs_and_snakes + birds_and_snakes + 
  snakes_and_hamsters + cats_dogs_snakes + cats_snakes_hamsters + all_pets = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_total_snake_owners_l3779_377950


namespace NUMINAMATH_CALUDE_expenditure_problem_l3779_377923

/-- Proves that given the conditions of the expenditure problem, the number of days in the next part of the week is 4. -/
theorem expenditure_problem (first_part_days : ℕ) (second_part_days : ℕ) 
  (first_part_avg : ℚ) (second_part_avg : ℚ) (total_avg : ℚ) :
  first_part_days = 3 →
  first_part_avg = 350 →
  second_part_avg = 420 →
  total_avg = 390 →
  first_part_days + second_part_days = 7 →
  (first_part_days * first_part_avg + second_part_days * second_part_avg) / 7 = total_avg →
  second_part_days = 4 := by
sorry

end NUMINAMATH_CALUDE_expenditure_problem_l3779_377923


namespace NUMINAMATH_CALUDE_distance_between_points_l3779_377948

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, -1)
  let p2 : ℝ × ℝ := (7, 6)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 74 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_distance_between_points_l3779_377948


namespace NUMINAMATH_CALUDE_trigonometric_equalities_l3779_377963

theorem trigonometric_equalities
  (α β γ a b c : ℝ)
  (h_alpha : 0 < α ∧ α < π)
  (h_beta : 0 < β ∧ β < π)
  (h_gamma : 0 < γ ∧ γ < π)
  (h_a : a > 0)
  (h_b : b > 0)
  (h_c : c > 0)
  (h_b_eq : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (h_a_eq : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2)
  (h_identity : 1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0) :
  (Real.cos α + Real.cos β * Real.cos γ = Real.sin α * Real.sin β) ∧
  (Real.cos β + Real.cos α * Real.cos γ = Real.sin α * Real.sin γ) ∧
  (Real.cos γ + Real.cos α * Real.cos β = Real.sin β * Real.sin γ) ∧
  (a * Real.sin γ = c * Real.sin α) ∧
  (b * Real.sin γ = c * Real.sin β) ∧
  (c * Real.sin α = a * Real.sin γ) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equalities_l3779_377963


namespace NUMINAMATH_CALUDE_katie_pastries_left_l3779_377956

/-- Represents the number of pastries Katie had left after the bake sale -/
def pastries_left (cupcakes cookies sold : ℕ) : ℕ :=
  cupcakes + cookies - sold

/-- Proves that Katie had 8 pastries left after the bake sale -/
theorem katie_pastries_left : pastries_left 7 5 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_katie_pastries_left_l3779_377956


namespace NUMINAMATH_CALUDE_equal_commissions_l3779_377994

/-- The list price of an item that satisfies the given conditions -/
def list_price : ℝ := 40

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Charlie's selling price -/
def charlie_price (x : ℝ) : ℝ := x - 20

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := 0.15 * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := 0.25 * bob_price x

/-- Charlie's commission -/
def charlie_commission (x : ℝ) : ℝ := 0.20 * charlie_price x

theorem equal_commissions :
  alice_commission list_price = bob_commission list_price ∧
  bob_commission list_price = charlie_commission list_price := by
  sorry

end NUMINAMATH_CALUDE_equal_commissions_l3779_377994


namespace NUMINAMATH_CALUDE_travel_distance_ratio_l3779_377970

/-- Given a total distance traveled, with specified portions by plane and bus,
    calculate the ratio of train distance to bus distance. -/
theorem travel_distance_ratio
  (total_distance : ℝ)
  (plane_fraction : ℝ)
  (bus_distance : ℝ)
  (h1 : total_distance = 1800)
  (h2 : plane_fraction = 1 / 3)
  (h3 : bus_distance = 720)
  : (total_distance - plane_fraction * total_distance - bus_distance) / bus_distance = 2 / 3 := by
  sorry

#check travel_distance_ratio

end NUMINAMATH_CALUDE_travel_distance_ratio_l3779_377970


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3779_377916

theorem complex_fraction_simplification :
  ((-4 : ℂ) - 6*I) / (5 - 2*I) = -32/21 - 38/21*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3779_377916


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l3779_377942

/-- A linear function passing through the first quadrant -/
def passes_through_first_quadrant (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y > 0

/-- A linear function passing through the fourth quadrant -/
def passes_through_fourth_quadrant (k b : ℝ) : Prop :=
  ∃ x y : ℝ, y = k * x + b ∧ x > 0 ∧ y < 0

/-- Theorem stating that a linear function y = kx + b with kb < 0 passes through both
    the first and fourth quadrants -/
theorem linear_function_quadrants (k b : ℝ) (h : k * b < 0) :
  passes_through_first_quadrant k b ∧ passes_through_fourth_quadrant k b :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l3779_377942


namespace NUMINAMATH_CALUDE_diagonal_intersections_count_l3779_377914

/-- Represents a convex polygon with n sides where no two diagonals are parallel
    and no three diagonals intersect at the same point. -/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3
  no_parallel_diagonals : True
  no_triple_intersections : True

/-- The number of intersection points of diagonals outside a convex polygon. -/
def diagonal_intersections_outside (n : ℕ) (p : ConvexPolygon n) : ℚ :=
  (1 / 12 : ℚ) * n * (n - 3) * (n - 4) * (n - 5)

theorem diagonal_intersections_count (n : ℕ) (p : ConvexPolygon n) :
  diagonal_intersections_outside n p = (1 / 12 : ℚ) * n * (n - 3) * (n - 4) * (n - 5) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersections_count_l3779_377914


namespace NUMINAMATH_CALUDE_cos_squared_fifteen_degrees_l3779_377981

theorem cos_squared_fifteen_degrees :
  2 * (Real.cos (15 * π / 180))^2 - 1 = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_squared_fifteen_degrees_l3779_377981


namespace NUMINAMATH_CALUDE_M_union_N_eq_N_l3779_377924

def M : Set ℝ := {x | x^2 - 2*x ≤ 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

theorem M_union_N_eq_N : M ∪ N = N := by sorry

end NUMINAMATH_CALUDE_M_union_N_eq_N_l3779_377924


namespace NUMINAMATH_CALUDE_semicircle_problem_l3779_377937

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  (N * (π * r^2 / 2)) / ((π * (N * r)^2 / 2) - (N * (π * r^2 / 2))) = 1 / 18 → N = 19 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_problem_l3779_377937


namespace NUMINAMATH_CALUDE_watch_price_proof_l3779_377941

/-- Represents the original cost price of the watch in Rupees. -/
def original_price : ℝ := 1800

/-- The selling price after discounts and loss. -/
def selling_price (price : ℝ) : ℝ := price * (1 - 0.05) * (1 - 0.03) * (1 - 0.10)

/-- The selling price for an 8% gain with 12% tax. -/
def selling_price_with_gain_and_tax (price : ℝ) : ℝ := price * (1 + 0.08) + price * 0.12

theorem watch_price_proof :
  selling_price original_price = original_price * 0.90 ∧
  selling_price_with_gain_and_tax original_price = selling_price original_price + 540 :=
by sorry

end NUMINAMATH_CALUDE_watch_price_proof_l3779_377941


namespace NUMINAMATH_CALUDE_complex_product_theorem_l3779_377906

theorem complex_product_theorem :
  let i : ℂ := Complex.I
  let z₁ : ℂ := 1 - i
  let z₂ : ℂ := 2 + i
  z₁ * z₂ = 3 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l3779_377906


namespace NUMINAMATH_CALUDE_fraction_nonnegative_iff_l3779_377980

theorem fraction_nonnegative_iff (x : ℝ) :
  (x - 3) / (x^2 + 5*x + 11) ≥ 0 ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_fraction_nonnegative_iff_l3779_377980


namespace NUMINAMATH_CALUDE_product_inequality_l3779_377961

theorem product_inequality (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_condition : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (z + x) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l3779_377961


namespace NUMINAMATH_CALUDE_sum_of_areas_equals_100_l3779_377993

-- Define the circle radius
def circle_radius : ℝ := 5

-- Define the maximum rectangle inscribed in the circle
def max_rectangle_area (r : ℝ) : ℝ := 2 * r^2

-- Define the maximum parallelogram circumscribed around the circle
def max_parallelogram_area (r : ℝ) : ℝ := 4 * r^2

-- Theorem statement
theorem sum_of_areas_equals_100 :
  max_rectangle_area circle_radius + max_parallelogram_area circle_radius = 100 := by
  sorry

#eval max_rectangle_area circle_radius + max_parallelogram_area circle_radius

end NUMINAMATH_CALUDE_sum_of_areas_equals_100_l3779_377993


namespace NUMINAMATH_CALUDE_square_equals_self_implies_zero_or_one_l3779_377990

theorem square_equals_self_implies_zero_or_one (a : ℝ) : a^2 = a → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_self_implies_zero_or_one_l3779_377990


namespace NUMINAMATH_CALUDE_shoes_sold_l3779_377971

theorem shoes_sold (large medium small left : ℕ) 
  (h_large : large = 22)
  (h_medium : medium = 50)
  (h_small : small = 24)
  (h_left : left = 13) :
  large + medium + small - left = 83 := by
  sorry

end NUMINAMATH_CALUDE_shoes_sold_l3779_377971


namespace NUMINAMATH_CALUDE_playground_children_count_l3779_377972

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 44) 
  (h2 : girls = 53) : 
  boys + girls = 97 := by
  sorry

end NUMINAMATH_CALUDE_playground_children_count_l3779_377972


namespace NUMINAMATH_CALUDE_monogram_count_l3779_377931

/-- The number of letters in the alphabet before 'M' -/
def letters_before_m : Nat := 12

/-- The number of letters in the alphabet after 'M' -/
def letters_after_m : Nat := 13

/-- A monogram is valid if it satisfies the given conditions -/
def is_valid_monogram (f m l : Char) : Prop :=
  f < m ∧ m < l ∧ f ≠ m ∧ m ≠ l ∧ f ≠ l ∧ m = 'M'

/-- The total number of valid monograms -/
def total_valid_monograms : Nat := letters_before_m * letters_after_m

theorem monogram_count :
  total_valid_monograms = 156 :=
sorry

end NUMINAMATH_CALUDE_monogram_count_l3779_377931


namespace NUMINAMATH_CALUDE_rachels_budget_l3779_377973

/-- Rachel's budget for a beauty and modeling contest -/
theorem rachels_budget (sara_shoes : ℕ) (sara_dress : ℕ) : 
  sara_shoes = 50 → sara_dress = 200 → 2 * (sara_shoes + sara_dress) = 500 := by
  sorry

end NUMINAMATH_CALUDE_rachels_budget_l3779_377973


namespace NUMINAMATH_CALUDE_sum_of_base_8_digits_of_888_l3779_377997

def base_8_representation (n : ℕ) : List ℕ :=
  sorry

def sum_of_digits (digits : List ℕ) : ℕ :=
  sorry

theorem sum_of_base_8_digits_of_888 :
  sum_of_digits (base_8_representation 888) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_base_8_digits_of_888_l3779_377997


namespace NUMINAMATH_CALUDE_divisible_by_24_l3779_377992

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, (n^4 : ℤ) + 2*(n^3 : ℤ) + 11*(n^2 : ℤ) + 10*(n : ℤ) = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l3779_377992


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3779_377951

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + 3*b = 1) :
  (1/a + 1/b) ≥ 455/36 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3779_377951


namespace NUMINAMATH_CALUDE_max_k_inequality_l3779_377991

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ k : ℝ, k ≤ 6 → (2 * (a^2 + k*a*b + b^2)) / ((k+2)*(a+b)) ≥ Real.sqrt (a*b)) ∧
  (∀ ε > 0, ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ (2 * (a^2 + (6+ε)*a*b + b^2)) / ((6+ε+2)*(a+b)) < Real.sqrt (a*b)) :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l3779_377991


namespace NUMINAMATH_CALUDE_optimal_orange_purchase_l3779_377940

-- Define the pricing options
def price_option_1 : ℕ × ℕ := (4, 15)  -- 4 oranges for 15 cents
def price_option_2 : ℕ × ℕ := (7, 25)  -- 7 oranges for 25 cents

-- Define the number of oranges to purchase
def total_oranges : ℕ := 28

-- Theorem statement
theorem optimal_orange_purchase :
  ∃ (n m : ℕ),
    n * price_option_1.1 + m * price_option_2.1 = total_oranges ∧
    n * price_option_1.2 + m * price_option_2.2 = 100 ∧
    (n * price_option_1.2 + m * price_option_2.2) / total_oranges = 25 / 7 :=
sorry

end NUMINAMATH_CALUDE_optimal_orange_purchase_l3779_377940


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3779_377996

theorem arithmetic_geometric_mean_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  (a + b + c + d) / 4 ≥ (a * b * c * d) ^ (1/4) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3779_377996


namespace NUMINAMATH_CALUDE_stating_kevin_vanessa_age_multiple_l3779_377946

/-- Represents the age difference between Kevin and Vanessa -/
def age_difference : ℕ := 14

/-- Represents Kevin's initial age -/
def kevin_initial_age : ℕ := 16

/-- Represents Vanessa's initial age -/
def vanessa_initial_age : ℕ := 2

/-- 
Theorem stating that the first time Kevin's age becomes a multiple of Vanessa's age, 
Kevin will be 4.5 times older than Vanessa.
-/
theorem kevin_vanessa_age_multiple :
  ∃ (years : ℕ), 
    (kevin_initial_age + years) % (vanessa_initial_age + years) = 0 ∧
    (kevin_initial_age + years : ℚ) / (vanessa_initial_age + years : ℚ) = 4.5 ∧
    ∀ (y : ℕ), y < years → (kevin_initial_age + y) % (vanessa_initial_age + y) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_stating_kevin_vanessa_age_multiple_l3779_377946


namespace NUMINAMATH_CALUDE_trust_meteorologist_l3779_377952

-- Define the probability of a clear day
def prob_clear_day : ℝ := 0.74

-- Define the accuracy of a senator's forecast (as a variable)
variable (p : ℝ)

-- Define the accuracy of the meteorologist's forecast
def meteorologist_accuracy (p : ℝ) : ℝ := 1.5 * p

-- Define the event of both senators predicting a clear day and the meteorologist predicting rain
def forecast_event (p : ℝ) : ℝ := 
  (1 - meteorologist_accuracy p) * p * p * prob_clear_day + 
  meteorologist_accuracy p * (1 - p) * (1 - p) * (1 - prob_clear_day)

-- Theorem statement
theorem trust_meteorologist (p : ℝ) (h1 : 0 < p) (h2 : p < 1) : 
  meteorologist_accuracy p * (1 - p) * (1 - p) * (1 - prob_clear_day) / forecast_event p > 
  (1 - meteorologist_accuracy p) * p * p * prob_clear_day / forecast_event p :=
sorry

end NUMINAMATH_CALUDE_trust_meteorologist_l3779_377952


namespace NUMINAMATH_CALUDE_school_trip_students_l3779_377927

/-- The number of students in a school given the number of buses and seats per bus -/
def number_of_students (buses : ℕ) (seats_per_bus : ℕ) : ℕ :=
  buses * seats_per_bus

/-- Theorem stating that the number of students in the school is 111 -/
theorem school_trip_students :
  let buses : ℕ := 37
  let seats_per_bus : ℕ := 3
  number_of_students buses seats_per_bus = 111 := by
  sorry

#eval number_of_students 37 3

end NUMINAMATH_CALUDE_school_trip_students_l3779_377927


namespace NUMINAMATH_CALUDE_average_increase_l3779_377957

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := 1.6 * x + 2

-- Theorem statement
theorem average_increase (x : ℝ) : 
  linear_regression (x + 1) - linear_regression x = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_average_increase_l3779_377957


namespace NUMINAMATH_CALUDE_classroom_key_probability_is_two_sevenths_l3779_377932

/-- The probability of selecting a key that opens the classroom door -/
def classroom_key_probability (total_keys : ℕ) (classroom_keys : ℕ) : ℚ :=
  classroom_keys / total_keys

/-- Theorem: The probability of randomly selecting a key that can open the classroom door lock is 2/7 -/
theorem classroom_key_probability_is_two_sevenths :
  classroom_key_probability 7 2 = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_classroom_key_probability_is_two_sevenths_l3779_377932


namespace NUMINAMATH_CALUDE_yogurt_combinations_l3779_377933

theorem yogurt_combinations (flavors : Nat) (toppings : Nat) : 
  flavors = 5 → toppings = 7 → flavors * (toppings.choose 3) = 175 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_combinations_l3779_377933


namespace NUMINAMATH_CALUDE_special_function_value_at_one_l3779_377935

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y, and f(2) = 4 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ f 2 = 4

/-- Theorem: If f is a special function, then f(1) = 2 -/
theorem special_function_value_at_one (f : ℝ → ℝ) (h : special_function f) : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_at_one_l3779_377935


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3779_377919

theorem system_of_equations_solution : 
  let x : ℚ := -29/2
  let y : ℚ := -71/2
  (7 * x - 3 * y = 5) ∧ (y - 3 * x = 8) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3779_377919


namespace NUMINAMATH_CALUDE_ball_distribution_count_l3779_377954

theorem ball_distribution_count :
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 3
  let ways_per_ball : ℕ := n_boxes
  n_boxes ^ n_balls = 81 :=
by sorry

end NUMINAMATH_CALUDE_ball_distribution_count_l3779_377954


namespace NUMINAMATH_CALUDE_corn_acreage_l3779_377922

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) :
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l3779_377922


namespace NUMINAMATH_CALUDE_larger_number_ratio_l3779_377959

theorem larger_number_ratio (a b : ℕ+) (k : ℚ) (s : ℤ) 
  (h1 : (a : ℚ) / (b : ℚ) = k)
  (h2 : k < 1)
  (h3 : (a : ℤ) + (b : ℤ) = s) :
  max a b = |s| / (1 + k) :=
sorry

end NUMINAMATH_CALUDE_larger_number_ratio_l3779_377959


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3779_377901

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  6 * Real.sqrt (a * b) + 3 / a + 3 / b ≥ 12 :=
sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 6 * Real.sqrt (a * b) + 3 / a + 3 / b < 12 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l3779_377901


namespace NUMINAMATH_CALUDE_uncle_welly_roses_l3779_377986

/-- Proves that Uncle Welly planted 20 more roses yesterday compared to two days ago -/
theorem uncle_welly_roses : 
  ∀ (roses_two_days_ago roses_yesterday roses_today : ℕ),
  roses_two_days_ago = 50 →
  roses_today = 2 * roses_two_days_ago →
  roses_yesterday > roses_two_days_ago →
  roses_two_days_ago + roses_yesterday + roses_today = 220 →
  roses_yesterday - roses_two_days_ago = 20 := by
sorry


end NUMINAMATH_CALUDE_uncle_welly_roses_l3779_377986


namespace NUMINAMATH_CALUDE_modified_ohara_triple_27_8_l3779_377985

/-- Definition of a Modified O'Hara Triple -/
def is_modified_ohara_triple (a b x : ℕ+) : Prop :=
  (a.val : ℝ)^(1/3) - (b.val : ℝ)^(1/3) = x.val

/-- Theorem: If (27, 8, x) is a Modified O'Hara triple, then x = 1 -/
theorem modified_ohara_triple_27_8 (x : ℕ+) :
  is_modified_ohara_triple 27 8 x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_modified_ohara_triple_27_8_l3779_377985


namespace NUMINAMATH_CALUDE_area_of_awesome_points_l3779_377938

/-- A right triangle with sides 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  sides : a = 3 ∧ b = 4 ∧ c = 5

/-- A point is awesome if it's the center of a parallelogram with vertices on the triangle's boundary -/
def is_awesome (T : RightTriangle) (P : ℝ × ℝ) : Prop := sorry

/-- The set of awesome points -/
def awesome_points (T : RightTriangle) : Set (ℝ × ℝ) :=
  {P | is_awesome T P}

/-- The area of a set of points in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

/-- The main theorem: The area of awesome points in the 3-4-5 right triangle is 3/2 -/
theorem area_of_awesome_points (T : RightTriangle) :
  area (awesome_points T) = 3/2 := by sorry

end NUMINAMATH_CALUDE_area_of_awesome_points_l3779_377938


namespace NUMINAMATH_CALUDE_hex_numeric_count_2023_l3779_377943

/-- Converts a positive integer to its hexadecimal representation --/
def to_hex (n : ℕ+) : List (Fin 16) :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits (0-9) --/
def hex_only_numeric (l : List (Fin 16)) : Bool :=
  sorry

/-- Counts numbers up to n whose hexadecimal representation contains only numeric digits --/
def count_hex_numeric (n : ℕ+) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sum_digits (n : ℕ) : ℕ :=
  sorry

/-- Theorem statement --/
theorem hex_numeric_count_2023 :
  sum_digits (count_hex_numeric 2023) = 25 :=
sorry

end NUMINAMATH_CALUDE_hex_numeric_count_2023_l3779_377943


namespace NUMINAMATH_CALUDE_muffin_cost_savings_l3779_377912

/-- Represents the cost savings when choosing raspberries over blueberries for muffins -/
def cost_savings (num_batches : ℕ) (ounces_per_batch : ℕ) 
  (blueberry_price : ℚ) (blueberry_ounces : ℕ) 
  (raspberry_price : ℚ) (raspberry_ounces : ℕ) : ℚ :=
  let total_ounces := num_batches * ounces_per_batch
  let blueberry_cartons := (total_ounces + blueberry_ounces - 1) / blueberry_ounces
  let raspberry_cartons := (total_ounces + raspberry_ounces - 1) / raspberry_ounces
  blueberry_cartons * blueberry_price - raspberry_cartons * raspberry_price

/-- The cost savings when choosing raspberries over blueberries for 4 batches of muffins -/
theorem muffin_cost_savings : 
  cost_savings 4 12 (5 / 1) 6 (3 / 1) 8 = 22 := by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_savings_l3779_377912


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3779_377995

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis is the set of all points with y-coordinate equal to 0 -/
def x_axis : Set Point := {p : Point | p.y = 0}

/-- Theorem: If a point A has y-coordinate equal to 0, then A lies on the x-axis -/
theorem point_on_x_axis (A : Point) (h : A.y = 0) : A ∈ x_axis := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3779_377995


namespace NUMINAMATH_CALUDE_complex_function_from_real_part_l3779_377949

open Complex

/-- Given that u(x, y) = x^2 - y^2 + 2x is the real part of a differentiable complex function f(z),
    prove that f(z) = z^2 + 2z + c for some constant c. -/
theorem complex_function_from_real_part
  (f : ℂ → ℂ)
  (h_diff : Differentiable ℂ f)
  (h_real_part : ∀ z : ℂ, (f z).re = z.re^2 - z.im^2 + 2*z.re) :
  ∃ c : ℂ, ∀ z : ℂ, f z = z^2 + 2*z + c :=
sorry

end NUMINAMATH_CALUDE_complex_function_from_real_part_l3779_377949


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l3779_377905

theorem salary_increase_percentage
  (total_employees : ℕ)
  (travel_allowance_percentage : ℚ)
  (no_increase_count : ℕ)
  (h1 : total_employees = 480)
  (h2 : travel_allowance_percentage = 1/5)
  (h3 : no_increase_count = 336) :
  (total_employees - no_increase_count - (travel_allowance_percentage * total_employees)) / total_employees = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l3779_377905


namespace NUMINAMATH_CALUDE_prob_A_truth_l3779_377964

-- Define the probabilities
def prob_B_truth : ℝ := 0.60
def prob_both_truth : ℝ := 0.45

-- Theorem statement
theorem prob_A_truth :
  ∃ (prob_A : ℝ),
    prob_A * prob_B_truth = prob_both_truth ∧
    prob_A = 0.75 :=
by sorry

end NUMINAMATH_CALUDE_prob_A_truth_l3779_377964


namespace NUMINAMATH_CALUDE_complement_of_union_l3779_377936

def I : Finset Int := {-2, -1, 0, 1, 2, 3, 4, 5}
def A : Finset Int := {-1, 0, 1, 2, 3}
def B : Finset Int := {-2, 0, 2}

theorem complement_of_union :
  (I \ (A ∪ B)) = {4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l3779_377936


namespace NUMINAMATH_CALUDE_no_solution_exists_l3779_377908

def sumOfDigits (n : ℕ) : ℕ := sorry

theorem no_solution_exists : ¬∃ (x y : ℕ), sumOfDigits ((10^x)^y - 64) = 279 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3779_377908


namespace NUMINAMATH_CALUDE_quadratic_inequality_proof_l3779_377925

theorem quadratic_inequality_proof (x : ℝ) : 
  x^2 + 6*x + 8 ≥ -(x + 4)*(x + 6) ∧ 
  (x^2 + 6*x + 8 = -(x + 4)*(x + 6) ↔ x = -4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_proof_l3779_377925


namespace NUMINAMATH_CALUDE_ferris_wheel_capacity_l3779_377928

theorem ferris_wheel_capacity 
  (total_people : ℕ) 
  (total_seats : ℕ) 
  (h1 : total_people = 16) 
  (h2 : total_seats = 4) 
  : total_people / total_seats = 4 := by
  sorry

end NUMINAMATH_CALUDE_ferris_wheel_capacity_l3779_377928


namespace NUMINAMATH_CALUDE_arrangement_count_is_correct_l3779_377975

/-- The number of ways to arrange 8 balls in a row, with 5 red balls and 3 white balls,
    such that exactly three consecutive balls are painted red -/
def arrangementCount : ℕ := 24

/-- The total number of balls -/
def totalBalls : ℕ := 8

/-- The number of red balls -/
def redBalls : ℕ := 5

/-- The number of white balls -/
def whiteBalls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutiveRedBalls : ℕ := 3

theorem arrangement_count_is_correct :
  arrangementCount = 24 ∧
  totalBalls = 8 ∧
  redBalls = 5 ∧
  whiteBalls = 3 ∧
  consecutiveRedBalls = 3 ∧
  redBalls + whiteBalls = totalBalls ∧
  arrangementCount = (whiteBalls + 1) * (redBalls - consecutiveRedBalls + 1) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_is_correct_l3779_377975


namespace NUMINAMATH_CALUDE_find_n_value_l3779_377976

theorem find_n_value (n : ℕ) : (1/5 : ℝ)^n * (1/4 : ℝ)^18 = 1/(2*(10 : ℝ)^35) → n = 35 := by
  sorry

end NUMINAMATH_CALUDE_find_n_value_l3779_377976


namespace NUMINAMATH_CALUDE_power_division_equality_l3779_377979

theorem power_division_equality : (3 : ℕ)^12 / (27 : ℕ)^2 = 729 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l3779_377979


namespace NUMINAMATH_CALUDE_rachel_envelope_stuffing_l3779_377968

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing 
  (total_time : ℕ) 
  (total_envelopes : ℕ) 
  (first_hour : ℕ) 
  (second_hour : ℕ) 
  (h1 : total_time = 8)
  (h2 : total_envelopes = 1500)
  (h3 : first_hour = 135)
  (h4 : second_hour = 141) : 
  (total_envelopes - first_hour - second_hour) / (total_time - 2) = 204 := by
sorry

end NUMINAMATH_CALUDE_rachel_envelope_stuffing_l3779_377968


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3779_377987

theorem solution_set_equivalence :
  ∀ x : ℝ, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3779_377987


namespace NUMINAMATH_CALUDE_problem_solution_l3779_377955

theorem problem_solution (x y z : ℚ) : 
  x / (y + 1) = 4 / 5 → 
  3 * z = 2 * x + y → 
  y = 10 → 
  z = 46 / 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3779_377955


namespace NUMINAMATH_CALUDE_distance_z1z2_to_origin_l3779_377977

open Complex

theorem distance_z1z2_to_origin : 
  let z₁ : ℂ := I
  let z₂ : ℂ := 1 + I
  let z : ℂ := z₁ * z₂
  abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_distance_z1z2_to_origin_l3779_377977


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l3779_377917

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6 : ℚ) = 72 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l3779_377917


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3779_377909

theorem arithmetic_sequence_sum (a₁ a_n d : ℚ) (n : ℕ) (h1 : a₁ = 2/7) (h2 : a_n = 20/7) (h3 : d = 2/7) (h4 : n = 10) :
  (n : ℚ) / 2 * (a₁ + a_n) = 110/7 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3779_377909


namespace NUMINAMATH_CALUDE_custom_operation_equality_l3779_377934

/-- Custom operation $ for real numbers -/
def dollar (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating the equality for the given expression -/
theorem custom_operation_equality (x y : ℝ) :
  dollar (x^2 - y^2) (y^2 - x^2) = 4 * (x^4 - 2*x^2*y^2 + y^4) := by
  sorry

end NUMINAMATH_CALUDE_custom_operation_equality_l3779_377934


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3779_377911

-- Define the number of white and black balls
def num_white_balls : ℕ := 1
def num_black_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := num_white_balls + num_black_balls

-- Define the probability of drawing a white ball
def prob_white_ball : ℚ := num_white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball :
  prob_white_ball = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3779_377911


namespace NUMINAMATH_CALUDE_horner_method_v₃_l3779_377903

def f (x : ℝ) : ℝ := 7 * x^5 + 5 * x^4 + 3 * x^3 + x^2 + x + 2

def horner_v₀ : ℝ := 7
def horner_v₁ (x : ℝ) : ℝ := horner_v₀ * x + 5
def horner_v₂ (x : ℝ) : ℝ := horner_v₁ x * x + 3
def horner_v₃ (x : ℝ) : ℝ := horner_v₂ x * x + 1

theorem horner_method_v₃ : horner_v₃ 2 = 83 :=
sorry

end NUMINAMATH_CALUDE_horner_method_v₃_l3779_377903


namespace NUMINAMATH_CALUDE_complex_equality_sum_l3779_377965

theorem complex_equality_sum (a b : ℝ) : a - 3*I = 2 + b*I → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_sum_l3779_377965


namespace NUMINAMATH_CALUDE_range_of_independent_variable_l3779_377921

theorem range_of_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / Real.sqrt (2 - 3 * x)) ↔ x < 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_independent_variable_l3779_377921


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3779_377966

theorem largest_n_for_factorization : ∃ (n : ℤ),
  (∀ (m : ℤ), (∃ (a b c d : ℤ), 3 * X^2 + m * X + 108 = (a * X + b) * (c * X + d)) → m ≤ n) ∧
  (∃ (a b c d : ℤ), 3 * X^2 + n * X + 108 = (a * X + b) * (c * X + d)) ∧
  n = 325 :=
sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l3779_377966


namespace NUMINAMATH_CALUDE_gumball_probability_l3779_377983

theorem gumball_probability (orange green yellow : ℕ) 
  (h_orange : orange = 10)
  (h_green : green = 6)
  (h_yellow : yellow = 9) :
  let total := orange + green + yellow
  let p_first_orange := orange / total
  let p_second_not_orange := (green + yellow) / (total - 1)
  let p_third_orange := (orange - 1) / (total - 2)
  p_first_orange * p_second_not_orange * p_third_orange = 9 / 92 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l3779_377983


namespace NUMINAMATH_CALUDE_initial_cargo_calculation_l3779_377910

theorem initial_cargo_calculation (cargo_loaded : ℕ) (total_cargo : ℕ) 
  (h1 : cargo_loaded = 8723)
  (h2 : total_cargo = 14696) :
  total_cargo - cargo_loaded = 5973 := by
  sorry

end NUMINAMATH_CALUDE_initial_cargo_calculation_l3779_377910


namespace NUMINAMATH_CALUDE_difference_of_squares_l3779_377978

theorem difference_of_squares (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 6) : 
  x^2 - y^2 = 120 := by
sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3779_377978


namespace NUMINAMATH_CALUDE_subtraction_of_integers_l3779_377953

theorem subtraction_of_integers : -1 - 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_integers_l3779_377953


namespace NUMINAMATH_CALUDE_scout_hourly_rate_l3779_377902

/-- Represents Scout's weekend earnings --/
def weekend_earnings (hourly_rate : ℚ) : ℚ :=
  -- Saturday earnings
  (4 * hourly_rate + 5 * 5) +
  -- Sunday earnings
  (5 * hourly_rate + 8 * 5)

/-- Theorem stating that Scout's hourly rate is $10.00 --/
theorem scout_hourly_rate :
  ∃ (rate : ℚ), weekend_earnings rate = 155 ∧ rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_scout_hourly_rate_l3779_377902


namespace NUMINAMATH_CALUDE_sally_quarters_count_l3779_377918

/-- Given that Sally had 760 quarters initially and received 418 more quarters,
    prove that she now has 1178 quarters in total. -/
theorem sally_quarters_count (initial : ℕ) (additional : ℕ) (total : ℕ) 
    (h1 : initial = 760)
    (h2 : additional = 418)
    (h3 : total = initial + additional) :
  total = 1178 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_count_l3779_377918


namespace NUMINAMATH_CALUDE_distance_AB_is_600_l3779_377907

/-- The distance between city A and city B -/
def distance_AB : ℝ := 600

/-- The time taken by Eddy to travel from A to B -/
def time_Eddy : ℝ := 3

/-- The time taken by Freddy to travel from A to C -/
def time_Freddy : ℝ := 4

/-- The distance between city A and city C -/
def distance_AC : ℝ := 460

/-- The ratio of Eddy's average speed to Freddy's average speed -/
def speed_ratio : ℝ := 1.7391304347826086

theorem distance_AB_is_600 :
  distance_AB = (speed_ratio * distance_AC * time_Eddy) / time_Freddy :=
sorry

end NUMINAMATH_CALUDE_distance_AB_is_600_l3779_377907


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l3779_377926

theorem quadratic_roots_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 8 ∧ y^2 + 2*h*y = 8 ∧ x^2 + y^2 = 20) → 
  |h| = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_squares_l3779_377926


namespace NUMINAMATH_CALUDE_kwik_e_tax_revenue_l3779_377989

/-- Calculates the total revenue for Kwik-e-Tax Center given the prices and number of returns sold --/
def total_revenue (federal_price state_price quarterly_price : ℕ) 
                  (federal_sold state_sold quarterly_sold : ℕ) : ℕ :=
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold

/-- Theorem stating that the total revenue for the given scenario is $4400 --/
theorem kwik_e_tax_revenue : 
  total_revenue 50 30 80 60 20 10 = 4400 := by
sorry

end NUMINAMATH_CALUDE_kwik_e_tax_revenue_l3779_377989
