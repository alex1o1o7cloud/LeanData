import Mathlib

namespace NUMINAMATH_CALUDE_negative_division_equality_l1388_138846

theorem negative_division_equality : (-81) / (-9) = 9 := by
  sorry

end NUMINAMATH_CALUDE_negative_division_equality_l1388_138846


namespace NUMINAMATH_CALUDE_carmela_money_distribution_l1388_138889

/-- Proves that Carmela needs to give $1 to each cousin for equal distribution -/
theorem carmela_money_distribution (carmela_money : ℕ) (cousin_money : ℕ) (num_cousins : ℕ) :
  carmela_money = 7 →
  cousin_money = 2 →
  num_cousins = 4 →
  let total_money := carmela_money + num_cousins * cousin_money
  let num_people := num_cousins + 1
  let equal_share := total_money / num_people
  let carmela_gives := carmela_money - equal_share
  carmela_gives / num_cousins = 1 := by
  sorry

end NUMINAMATH_CALUDE_carmela_money_distribution_l1388_138889


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1388_138887

theorem solution_set_inequality (x : ℝ) : (x - 2) / x < 0 ↔ 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1388_138887


namespace NUMINAMATH_CALUDE_seating_arrangements_3_8_l1388_138827

/-- The number of distinct seating arrangements for 3 people in a row of 8 seats,
    with empty seats on both sides of each person. -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of seating arrangements
    for 3 people in 8 seats is 24. -/
theorem seating_arrangements_3_8 :
  seating_arrangements 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_3_8_l1388_138827


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l1388_138844

-- Define set A
def A : Set ℝ := {x | |x - 1| ≤ 1}

-- Define set B with parameter a
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

-- Theorem for part 1
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for part 2
theorem subset_condition (a : ℝ) :
  A ⊆ B a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_subset_condition_l1388_138844


namespace NUMINAMATH_CALUDE_grid_paths_6_5_l1388_138809

/-- The number of distinct paths on a grid from (0,0) to (m,n) using only right and up moves -/
def gridPaths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) n

theorem grid_paths_6_5 : gridPaths 6 5 = 462 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_6_5_l1388_138809


namespace NUMINAMATH_CALUDE_function_property_l1388_138854

theorem function_property (f : ℤ → ℝ) 
  (h1 : ∀ x y : ℤ, f x * f y = f (x + y) + f (x - y))
  (h2 : f 1 = 5/2) :
  ∀ x : ℤ, f x = (5/2) ^ x := by sorry

end NUMINAMATH_CALUDE_function_property_l1388_138854


namespace NUMINAMATH_CALUDE_leonards_age_l1388_138877

theorem leonards_age (leonard nina jerome : ℕ) 
  (h1 : leonard = nina - 4)
  (h2 : nina = jerome / 2)
  (h3 : leonard + nina + jerome = 36) :
  leonard = 6 := by
sorry

end NUMINAMATH_CALUDE_leonards_age_l1388_138877


namespace NUMINAMATH_CALUDE_safety_gear_to_test_tube_ratio_l1388_138842

def total_budget : ℚ := 325
def flask_cost : ℚ := 150
def remaining_budget : ℚ := 25

def test_tube_cost : ℚ := (2/3) * flask_cost

def total_spent : ℚ := total_budget - remaining_budget

def safety_gear_cost : ℚ := total_spent - flask_cost - test_tube_cost

theorem safety_gear_to_test_tube_ratio :
  safety_gear_cost / test_tube_cost = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_safety_gear_to_test_tube_ratio_l1388_138842


namespace NUMINAMATH_CALUDE_siblings_age_sum_l1388_138876

/-- The age difference between siblings -/
def age_gap : ℕ := 5

/-- The current age of the eldest sibling -/
def eldest_age_now : ℕ := 20

/-- The number of years into the future we're considering -/
def years_forward : ℕ := 10

/-- The total age of three siblings after a given number of years -/
def total_age_after (years : ℕ) : ℕ :=
  (eldest_age_now + years) + (eldest_age_now - age_gap + years) + (eldest_age_now - 2 * age_gap + years)

theorem siblings_age_sum :
  total_age_after years_forward = 75 :=
by sorry

end NUMINAMATH_CALUDE_siblings_age_sum_l1388_138876


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1388_138839

/-- The universal set U -/
def U : Set ℕ := {1, 2, 3, 4, 5}

/-- Set A -/
def A : Set ℕ := {1, 3, 4}

/-- Set B -/
def B : Set ℕ := {4, 5}

/-- Theorem stating that the intersection of A and the complement of B with respect to U is {1, 3} -/
theorem intersection_A_complement_B : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1388_138839


namespace NUMINAMATH_CALUDE_binomial_sum_equals_power_of_two_l1388_138812

theorem binomial_sum_equals_power_of_two : 
  3^2006 - Nat.choose 2006 1 * 3^2005 + Nat.choose 2006 2 * 3^2004 - Nat.choose 2006 3 * 3^2003 +
  Nat.choose 2006 4 * 3^2002 - Nat.choose 2006 5 * 3^2001 + 
  -- ... (omitting middle terms for brevity)
  Nat.choose 2006 2004 * 3^2 - Nat.choose 2006 2005 * 3 + 1 = 2^2006 :=
by sorry

end NUMINAMATH_CALUDE_binomial_sum_equals_power_of_two_l1388_138812


namespace NUMINAMATH_CALUDE_georgia_carnation_cost_l1388_138838

-- Define the cost of a single carnation
def single_carnation_cost : ℚ := 1/2

-- Define the cost of a dozen carnations
def dozen_carnation_cost : ℚ := 4

-- Define the number of teachers
def num_teachers : ℕ := 5

-- Define the number of friends
def num_friends : ℕ := 14

-- Theorem statement
theorem georgia_carnation_cost : 
  (num_teachers : ℚ) * dozen_carnation_cost + (num_friends : ℚ) * single_carnation_cost = 27 := by
  sorry

end NUMINAMATH_CALUDE_georgia_carnation_cost_l1388_138838


namespace NUMINAMATH_CALUDE_set_difference_N_M_l1388_138810

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 2, 3, 7}

theorem set_difference_N_M : N \ M = {7} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_N_M_l1388_138810


namespace NUMINAMATH_CALUDE_max_annual_profit_l1388_138865

/-- Represents the annual production quantity -/
def x : Type := { n : ℕ // n > 0 }

/-- Calculates the annual sales revenue in million yuan -/
def salesRevenue (x : x) : ℝ :=
  if x.val ≤ 20 then 33 * x.val - x.val^2 else 260

/-- Calculates the total annual investment in million yuan -/
def totalInvestment (x : x) : ℝ := 1 + 0.01 * x.val

/-- Calculates the annual profit in million yuan -/
def annualProfit (x : x) : ℝ := salesRevenue x - totalInvestment x

/-- Theorem stating the maximum annual profit and the production quantity that achieves it -/
theorem max_annual_profit :
  ∃ (x_max : x), 
    (∀ (x : x), annualProfit x ≤ annualProfit x_max) ∧
    (x_max.val = 16) ∧
    (annualProfit x_max = 156) := by sorry

end NUMINAMATH_CALUDE_max_annual_profit_l1388_138865


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1388_138869

theorem quadratic_minimum (x y : ℝ) : 
  y = x^2 + 16*x + 20 → (∀ z : ℝ, z = x^2 + 16*x + 20 → y ≤ z) → y = -44 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1388_138869


namespace NUMINAMATH_CALUDE_juan_stamp_cost_l1388_138848

/-- Represents the cost of stamps for a given country -/
structure StampCost where
  country : String
  cost : Float

/-- Represents the number of stamps for a country in a specific decade -/
structure StampCount where
  country : String
  decade : String
  count : Nat

def brazil_cost : StampCost := ⟨"Brazil", 0.07⟩
def peru_cost : StampCost := ⟨"Peru", 0.05⟩

def brazil_70s : StampCount := ⟨"Brazil", "70s", 12⟩
def brazil_80s : StampCount := ⟨"Brazil", "80s", 15⟩
def peru_70s : StampCount := ⟨"Peru", "70s", 6⟩
def peru_80s : StampCount := ⟨"Peru", "80s", 12⟩

def total_cost (costs : List StampCost) (counts : List StampCount) : Float :=
  sorry

theorem juan_stamp_cost :
  total_cost [brazil_cost, peru_cost] [brazil_70s, brazil_80s, peru_70s, peru_80s] = 2.79 :=
sorry

end NUMINAMATH_CALUDE_juan_stamp_cost_l1388_138848


namespace NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l1388_138888

theorem smallest_divisor_sum_of_squares (n : ℕ) : n ≥ 2 →
  (∃ (a b : ℕ), a > 1 ∧ a ∣ n ∧ b ∣ n ∧
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    n = a^2 + b^2) →
  n = 8 ∨ n = 20 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_sum_of_squares_l1388_138888


namespace NUMINAMATH_CALUDE_function_equation_l1388_138823

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_equation (h : ∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) : 
  f = fun x ↦ x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_equation_l1388_138823


namespace NUMINAMATH_CALUDE_reggies_book_cost_l1388_138862

theorem reggies_book_cost (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 48 →
  books_bought = 5 →
  amount_left = 38 →
  (initial_amount - amount_left) / books_bought = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_reggies_book_cost_l1388_138862


namespace NUMINAMATH_CALUDE_function_property_l1388_138804

/-- Given a function f(x) = ax^5 + bx^3 + 2 where f(2) = 7, prove that f(-2) = -3 -/
theorem function_property (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^5 + b * x^3 + 2
  (f 2 = 7) → (f (-2) = -3) := by
sorry

end NUMINAMATH_CALUDE_function_property_l1388_138804


namespace NUMINAMATH_CALUDE_off_road_vehicle_cost_l1388_138895

theorem off_road_vehicle_cost 
  (dirt_bike_cost : ℕ) 
  (dirt_bike_count : ℕ) 
  (off_road_count : ℕ) 
  (registration_fee : ℕ) 
  (total_cost : ℕ) 
  (h1 : dirt_bike_cost = 150)
  (h2 : dirt_bike_count = 3)
  (h3 : off_road_count = 4)
  (h4 : registration_fee = 25)
  (h5 : total_cost = 1825)
  (h6 : total_cost = dirt_bike_cost * dirt_bike_count + 
                     off_road_count * x + 
                     registration_fee * (dirt_bike_count + off_road_count)) :
  x = 300 := by
  sorry


end NUMINAMATH_CALUDE_off_road_vehicle_cost_l1388_138895


namespace NUMINAMATH_CALUDE_roller_coaster_capacity_l1388_138883

theorem roller_coaster_capacity 
  (total_cars : ℕ) 
  (total_capacity : ℕ) 
  (four_seater_cars : ℕ) 
  (four_seater_capacity : ℕ) 
  (h1 : total_cars = 15)
  (h2 : total_capacity = 72)
  (h3 : four_seater_cars = 9)
  (h4 : four_seater_capacity = 4) :
  (total_capacity - four_seater_cars * four_seater_capacity) / (total_cars - four_seater_cars) = 6 := by
sorry

end NUMINAMATH_CALUDE_roller_coaster_capacity_l1388_138883


namespace NUMINAMATH_CALUDE_sum_x_y_given_equations_l1388_138882

theorem sum_x_y_given_equations (x y : ℝ) 
  (eq1 : 2 * |x| + 3 * x + 3 * y = 30)
  (eq2 : 3 * x + 2 * |y| - 2 * y = 36) : 
  x + y = 8512 / 2513 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_given_equations_l1388_138882


namespace NUMINAMATH_CALUDE_number_of_female_democrats_l1388_138890

theorem number_of_female_democrats
  (total : ℕ)
  (h_total : total = 780)
  (female : ℕ)
  (male : ℕ)
  (h_sum : female + male = total)
  (female_democrats : ℕ)
  (male_democrats : ℕ)
  (h_female_dem : female_democrats = female / 2)
  (h_male_dem : male_democrats = male / 4)
  (h_total_dem : female_democrats + male_democrats = total / 3) :
  female_democrats = 130 := by
sorry

end NUMINAMATH_CALUDE_number_of_female_democrats_l1388_138890


namespace NUMINAMATH_CALUDE_max_value_when_t_2_t_value_when_max_2_l1388_138856

-- Define the function f(x, t)
def f (x t : ℝ) : ℝ := |2 * x - 1| - |t * x + 3|

-- Theorem 1: Maximum value of f(x) when t = 2 is 4
theorem max_value_when_t_2 :
  ∃ M : ℝ, M = 4 ∧ ∀ x : ℝ, f x 2 ≤ M :=
sorry

-- Theorem 2: When maximum value of f(x) is 2, t = 6
theorem t_value_when_max_2 :
  ∃ t : ℝ, t > 0 ∧ (∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, f x t ≤ M) → t = 6 :=
sorry

end NUMINAMATH_CALUDE_max_value_when_t_2_t_value_when_max_2_l1388_138856


namespace NUMINAMATH_CALUDE_min_value_abc_l1388_138833

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 27) :
  a^2 + 6*a*b + 9*b^2 + 3*c^2 ≥ 126 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 6*a₀*b₀ + 9*b₀^2 + 3*c₀^2 = 126 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l1388_138833


namespace NUMINAMATH_CALUDE_smallest_fraction_greater_than_four_ninths_l1388_138858

theorem smallest_fraction_greater_than_four_ninths :
  ∀ a b : ℕ,
    10 ≤ a ∧ a ≤ 99 →
    10 ≤ b ∧ b ≤ 99 →
    (4 : ℚ) / 9 < (a : ℚ) / b →
    (41 : ℚ) / 92 ≤ (a : ℚ) / b :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_greater_than_four_ninths_l1388_138858


namespace NUMINAMATH_CALUDE_coin_distribution_l1388_138835

theorem coin_distribution (x y : ℕ) : 
  x + y = 16 → 
  x^2 - y^2 = 16 * (x - y) → 
  x = 8 ∧ y = 8 := by
  sorry

end NUMINAMATH_CALUDE_coin_distribution_l1388_138835


namespace NUMINAMATH_CALUDE_value_of_a_l1388_138850

/-- Two circles centered at the origin with given properties -/
structure TwoCircles where
  -- Radius of the larger circle
  R : ℝ
  -- Radius of the smaller circle
  r : ℝ
  -- Point P on the larger circle
  P : ℝ × ℝ
  -- Point S on the smaller circle
  S : ℝ → ℝ × ℝ
  -- Distance between Q and R on x-axis
  QR_distance : ℝ
  -- Conditions
  center_origin : True
  P_on_larger : P.1^2 + P.2^2 = R^2
  S_on_smaller : ∀ a, (S a).1^2 + (S a).2^2 = r^2
  S_on_diagonal : ∀ a, (S a).1 = (S a).2
  QR_is_4 : QR_distance = 4
  R_eq_sqrt_104 : R = Real.sqrt 104
  r_eq_R_minus_4 : r = R - 4

/-- The theorem stating the value of a -/
theorem value_of_a (c : TwoCircles) : 
  ∃ a, c.S a = (a, a) ∧ a = Real.sqrt (60 - 4 * Real.sqrt 104) :=
sorry

end NUMINAMATH_CALUDE_value_of_a_l1388_138850


namespace NUMINAMATH_CALUDE_tau_phi_equality_characterization_l1388_138837

/-- Number of natural numbers dividing n -/
def tau (n : ℕ) : ℕ := sorry

/-- Number of natural numbers less than n that are relatively prime to n -/
def phi (n : ℕ) : ℕ := sorry

/-- Predicate for n having exactly two different prime divisors -/
def has_two_prime_divisors (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem tau_phi_equality_characterization (n : ℕ) :
  has_two_prime_divisors n ∧ tau (phi n) = phi (tau n) ↔
  ∃ k : ℕ, n = 3 * 2^(2^k - 1) :=
sorry

end NUMINAMATH_CALUDE_tau_phi_equality_characterization_l1388_138837


namespace NUMINAMATH_CALUDE_additional_amount_is_three_l1388_138808

/-- The minimum purchase amount required for free delivery -/
def min_purchase : ℝ := 18

/-- The cost of a quarter-pounder burger -/
def burger_cost : ℝ := 3.20

/-- The cost of large fries -/
def fries_cost : ℝ := 1.90

/-- The cost of a milkshake -/
def milkshake_cost : ℝ := 2.40

/-- The number of each item Danny ordered -/
def quantity : ℕ := 2

/-- The total cost of Danny's current order -/
def order_total : ℝ := quantity * burger_cost + quantity * fries_cost + quantity * milkshake_cost

/-- The additional amount needed for free delivery -/
def additional_amount : ℝ := min_purchase - order_total

theorem additional_amount_is_three :
  additional_amount = 3 :=
by sorry

end NUMINAMATH_CALUDE_additional_amount_is_three_l1388_138808


namespace NUMINAMATH_CALUDE_expression_result_l1388_138826

theorem expression_result : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end NUMINAMATH_CALUDE_expression_result_l1388_138826


namespace NUMINAMATH_CALUDE_sin_285_degrees_l1388_138852

theorem sin_285_degrees : 
  Real.sin (285 * π / 180) = -((Real.sqrt 6 + Real.sqrt 2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_sin_285_degrees_l1388_138852


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1388_138872

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l1388_138872


namespace NUMINAMATH_CALUDE_unique_rational_pair_l1388_138815

theorem unique_rational_pair : 
  ∀ (a b r s : ℚ), 
    a ≠ b → 
    r ≠ s → 
    (∀ (z : ℚ), (z - r) * (z - s) = (z - a*r) * (z - b*s)) → 
    ∃! (p : ℚ × ℚ), p.1 ≠ p.2 ∧ 
      ∀ (z : ℚ), (z - r) * (z - s) = (z - p.1*r) * (z - p.2*s) :=
by sorry

end NUMINAMATH_CALUDE_unique_rational_pair_l1388_138815


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l1388_138864

theorem largest_four_digit_divisible_by_five : ∃ n : ℕ, 
  (n ≤ 9999 ∧ n ≥ 1000) ∧ 
  n % 5 = 0 ∧
  ∀ m : ℕ, (m ≤ 9999 ∧ m ≥ 1000) → m % 5 = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_five_l1388_138864


namespace NUMINAMATH_CALUDE_cubic_parabola_collinearity_l1388_138831

/-- Represents a point on a cubic parabola -/
structure CubicPoint where
  x : ℝ
  y : ℝ

/-- Represents a cubic parabola y = x^3 + a₁x^2 + a₂x + a₃ -/
structure CubicParabola where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ

/-- Check if a point lies on the cubic parabola -/
def onCubicParabola (p : CubicPoint) (c : CubicParabola) : Prop :=
  p.y = p.x^3 + c.a₁ * p.x^2 + c.a₂ * p.x + c.a₃

/-- Check if three points are collinear -/
def areCollinear (p q r : CubicPoint) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Main theorem: Given a cubic parabola and three points on it with x-coordinates summing to -a₁, the points are collinear -/
theorem cubic_parabola_collinearity (c : CubicParabola) (p q r : CubicPoint)
    (h_p : onCubicParabola p c)
    (h_q : onCubicParabola q c)
    (h_r : onCubicParabola r c)
    (h_sum : p.x + q.x + r.x = -c.a₁) :
    areCollinear p q r := by
  sorry

end NUMINAMATH_CALUDE_cubic_parabola_collinearity_l1388_138831


namespace NUMINAMATH_CALUDE_equation_solution_l1388_138813

theorem equation_solution : ∃ x : ℝ, (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1388_138813


namespace NUMINAMATH_CALUDE_care_package_weight_l1388_138803

/-- Represents the weight of the care package contents -/
structure CarePackage where
  jellyBeans : ℝ
  brownies : ℝ
  gummyWorms : ℝ
  chocolateBars : ℝ
  popcorn : ℝ
  cookies : ℝ

/-- Calculates the total weight of the care package -/
def totalWeight (cp : CarePackage) : ℝ :=
  cp.jellyBeans + cp.brownies + cp.gummyWorms + cp.chocolateBars + cp.popcorn + cp.cookies

/-- The final weight of the care package after all modifications -/
def finalWeight (initialWeight : ℝ) : ℝ :=
  let weightAfterChocolate := initialWeight * 1.5
  let weightAfterPopcorn := weightAfterChocolate + 0.5
  let weightAfterCookies := weightAfterPopcorn * 2
  weightAfterCookies - 0.75

theorem care_package_weight :
  let initialPackage : CarePackage := {
    jellyBeans := 1.5,
    brownies := 0.5,
    gummyWorms := 2,
    chocolateBars := 0,
    popcorn := 0,
    cookies := 0
  }
  let initialWeight := totalWeight initialPackage
  finalWeight initialWeight = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_care_package_weight_l1388_138803


namespace NUMINAMATH_CALUDE_find_y_l1388_138851

theorem find_y (x : ℝ) (y : ℝ) (h1 : x^(3*y - 1) = 8) (h2 : x = 2) : y = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l1388_138851


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l1388_138811

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l1388_138811


namespace NUMINAMATH_CALUDE_prime_divides_binomial_coefficient_l1388_138857

theorem prime_divides_binomial_coefficient (p k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  p ∣ Nat.choose p k := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_binomial_coefficient_l1388_138857


namespace NUMINAMATH_CALUDE_hilt_garden_border_rocks_l1388_138896

/-- The number of rocks Mrs. Hilt needs to complete her garden border -/
def total_rocks_needed (rocks_on_hand : ℕ) (additional_rocks_needed : ℕ) : ℕ :=
  rocks_on_hand + additional_rocks_needed

/-- Theorem: Mrs. Hilt needs 125 rocks in total to complete her garden border -/
theorem hilt_garden_border_rocks : 
  total_rocks_needed 64 61 = 125 := by
  sorry

end NUMINAMATH_CALUDE_hilt_garden_border_rocks_l1388_138896


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l1388_138820

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  x₀ : ℝ
  y₀ : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The problem statement -/
theorem circle_and_tangent_line 
  (C : Circle) 
  (l : Line) :
  C.h = 2 ∧ 
  C.k = 3 ∧ 
  C.r = 1 ∧
  l.x₀ = 1 ∧ 
  l.y₀ = 0 →
  (∀ x y : ℝ, (x - C.h)^2 + (y - C.k)^2 = C.r^2) ∧
  ((l.a = 1 ∧ l.b = 0 ∧ l.c = -1) ∨
   (l.a = 4 ∧ l.b = -3 ∧ l.c = -4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l1388_138820


namespace NUMINAMATH_CALUDE_unique_angle_solution_l1388_138870

theorem unique_angle_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
    Real.tan ((150 - x) * π / 180) = 
      (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
      (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
    x = 110 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_solution_l1388_138870


namespace NUMINAMATH_CALUDE_allocation_five_to_three_l1388_138893

/-- The number of ways to allocate n identical objects to k distinct groups,
    with each group receiving at least one object -/
def allocations (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 6 ways to allocate 5 identical objects to 3 distinct groups,
    with each group receiving at least one object -/
theorem allocation_five_to_three :
  allocations 5 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_allocation_five_to_three_l1388_138893


namespace NUMINAMATH_CALUDE_chlorine_atomic_weight_l1388_138824

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 68

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := total_weight - hydrogen_weight - 2 * oxygen_weight

theorem chlorine_atomic_weight : chlorine_weight = 35 := by
  sorry

end NUMINAMATH_CALUDE_chlorine_atomic_weight_l1388_138824


namespace NUMINAMATH_CALUDE_equation_holds_iff_l1388_138899

theorem equation_holds_iff (a b c : ℝ) (ha : a ≠ 0) (hab : a + b ≠ 0) :
  (a + b + c) / a = (b + c) / (a + b) ↔ a = -(b + c) := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_iff_l1388_138899


namespace NUMINAMATH_CALUDE_quartic_sum_l1388_138840

theorem quartic_sum (f : ℝ → ℝ) :
  (∃ (a b c d : ℝ), ∀ x, f x = a*x^4 + b*x^3 + c*x^2 + d*x + (f 0)) →
  (f 1 = 10) →
  (f 2 = 20) →
  (f 3 = 30) →
  (f 10 + f (-6) = 8104) :=
by sorry

end NUMINAMATH_CALUDE_quartic_sum_l1388_138840


namespace NUMINAMATH_CALUDE_largest_prime_factors_difference_l1388_138832

theorem largest_prime_factors_difference (n : Nat) (h : n = 165033) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  (∀ (r : Nat), Nat.Prime r ∧ r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Nat.Prime r ∧ r ∣ n ∧ r ≠ p → r ≤ q) ∧
  p - q = 140 := by
  sorry

#eval 165033

end NUMINAMATH_CALUDE_largest_prime_factors_difference_l1388_138832


namespace NUMINAMATH_CALUDE_f_min_at_300_l1388_138819

/-- The quadratic expression we're minimizing -/
def f (x : ℝ) : ℝ := x^2 - 600*x + 369

/-- The theorem stating that f(x) takes its minimum value when x = 300 -/
theorem f_min_at_300 : 
  ∀ x : ℝ, f x ≥ f 300 := by sorry

end NUMINAMATH_CALUDE_f_min_at_300_l1388_138819


namespace NUMINAMATH_CALUDE_set_operations_l1388_138875

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x < 10}

-- Define the intervals for the results
def interval_3_10 : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def open_4_7 : Set ℝ := {x | 4 < x ∧ x < 7}
def union_4_7_7_10 : Set ℝ := {x | (4 < x ∧ x < 7) ∨ (7 ≤ x ∧ x < 10)}

-- State the theorem
theorem set_operations :
  (A ∪ B = interval_3_10) ∧
  (A ∩ B = open_4_7) ∧
  ((Set.univ \ A) ∩ B = union_4_7_7_10) := by sorry

end NUMINAMATH_CALUDE_set_operations_l1388_138875


namespace NUMINAMATH_CALUDE_bus_travel_time_l1388_138802

/-- Represents a time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference in hours between two times -/
def timeDifference (t1 t2 : TimeOfDay) : Nat :=
  -- Implementation details omitted
  sorry

/-- Theorem: The time difference between 12:30 PM and 9:30 AM is 3 hours -/
theorem bus_travel_time :
  let departure : TimeOfDay := ⟨9, 30, sorry⟩
  let arrival : TimeOfDay := ⟨12, 30, sorry⟩
  timeDifference departure arrival = 3 := by
  sorry

end NUMINAMATH_CALUDE_bus_travel_time_l1388_138802


namespace NUMINAMATH_CALUDE_circles_properties_l1388_138863

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 4 = 0

-- Define the common chord line
def common_chord (x y : ℝ) : Prop := x - 2*y + 4 = 0

-- Define the common tangent line
def common_tangent (x : ℝ) : Prop := x = -2

theorem circles_properties :
  (∀ x y : ℝ, circle1 x y ∧ circle2 x y → common_chord x y) ∧
  (∀ x y : ℝ, (circle1 x y ∧ common_tangent x) ∨ (circle2 x y ∧ common_tangent x) →
    ∃ (t : ℝ), (x + 2*t)^2 + (y + 2*t)^2 = 4 ∨ (x + 2*t)^2 + (y + 2*t)^2 + 2*(x + 2*t) - 4*(y + 2*t) + 4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_circles_properties_l1388_138863


namespace NUMINAMATH_CALUDE_number_problem_l1388_138828

theorem number_problem (x : ℝ) : 0.5 * x = 0.25 * x + 2 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1388_138828


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_diff_smallest_positive_integer_l1388_138836

theorem smallest_n_for_sqrt_diff (n : ℕ) : n ≥ 10001 ↔ Real.sqrt n - Real.sqrt (n - 1) < 0.005 := by
  sorry

theorem smallest_positive_integer : ∀ m : ℕ, m < 10001 → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.005 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_diff_smallest_positive_integer_l1388_138836


namespace NUMINAMATH_CALUDE_subtract_negative_one_three_l1388_138855

theorem subtract_negative_one_three : -1 - 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_one_three_l1388_138855


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1388_138805

/-- Greatest prime factor of a positive integer -/
def greatest_prime_factor (n : ℕ) : ℕ := sorry

/-- The theorem states that there exists exactly one positive integer n > 1
    satisfying both conditions simultaneously -/
theorem unique_n_satisfying_conditions : ∃! n : ℕ, n > 1 ∧ 
  (greatest_prime_factor n = n.sqrt) ∧ 
  (greatest_prime_factor (n + 72) = (n + 72).sqrt) := by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l1388_138805


namespace NUMINAMATH_CALUDE_postman_pete_miles_l1388_138830

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_steps * (p.resets + 1) + p.final_reading

/-- Converts steps to miles, rounded to the nearest mile --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  (steps + steps_per_mile / 2) / steps_per_mile

/-- Theorem stating the total miles walked by Postman Pete --/
theorem postman_pete_miles :
  let p : Pedometer := { max_steps := 100000, resets := 48, final_reading := 25000 }
  let steps_per_mile : ℕ := 1600
  steps_to_miles (total_steps p) steps_per_mile = 3016 := by
  sorry

end NUMINAMATH_CALUDE_postman_pete_miles_l1388_138830


namespace NUMINAMATH_CALUDE_a_1_value_l1388_138800

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y ^ 2 = x * z

theorem a_1_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_a_1_value_l1388_138800


namespace NUMINAMATH_CALUDE_cone_volume_l1388_138859

/-- Given a cone with lateral area 20π and angle between slant height and base arccos(4/5),
    prove that its volume is 16π. -/
theorem cone_volume (r l h : ℝ) (lateral_area : ℝ) (angle : ℝ) : 
  lateral_area = 20 * Real.pi →
  angle = Real.arccos (4/5) →
  r / l = 4 / 5 →
  lateral_area = Real.pi * r * l →
  h = Real.sqrt (l^2 - r^2) →
  (1/3) * Real.pi * r^2 * h = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l1388_138859


namespace NUMINAMATH_CALUDE_value_of_c_l1388_138807

theorem value_of_c (c : ℝ) : 
  4 * ((3.6 * 0.48 * c) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l1388_138807


namespace NUMINAMATH_CALUDE_unique_triple_product_sum_l1388_138818

theorem unique_triple_product_sum : 
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
    a * b = c ∧ b * c = a ∧ c * a = b ∧ a + b + c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_product_sum_l1388_138818


namespace NUMINAMATH_CALUDE_goats_minus_pigs_equals_35_l1388_138829

def farm_problem (goats chickens ducks pigs rabbits cows : ℕ) : Prop :=
  goats = 66 ∧
  chickens = 2 * goats - 10 ∧
  ducks = (goats + chickens) / 2 ∧
  pigs = ducks / 3 ∧
  rabbits = Int.sqrt (2 * ducks - pigs) ∧
  cows = (rabbits ^ pigs) / (Nat.factorial (goats / 2))

theorem goats_minus_pigs_equals_35 :
  ∀ goats chickens ducks pigs rabbits cows,
    farm_problem goats chickens ducks pigs rabbits cows →
    goats - pigs = 35 := by
  sorry

end NUMINAMATH_CALUDE_goats_minus_pigs_equals_35_l1388_138829


namespace NUMINAMATH_CALUDE_max_handshakes_60_men_l1388_138879

/-- The maximum number of handshakes for n people without cyclic handshakes -/
def max_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: For 60 men, the maximum number of handshakes without cyclic handshakes is 1770 -/
theorem max_handshakes_60_men :
  max_handshakes 60 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_60_men_l1388_138879


namespace NUMINAMATH_CALUDE_cafe_choices_l1388_138874

/-- The number of ways two people can choose different items from a set of n items -/
def differentChoices (n : ℕ) : ℕ := n * (n - 1)

/-- The number of menu items in the café -/
def menuItems : ℕ := 12

/-- Theorem: The number of ways Alex and Jamie can choose different dishes from a menu of 12 items is 132 -/
theorem cafe_choices : differentChoices menuItems = 132 := by
  sorry

end NUMINAMATH_CALUDE_cafe_choices_l1388_138874


namespace NUMINAMATH_CALUDE_f_at_2_l1388_138816

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem f_at_2 : f 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_f_at_2_l1388_138816


namespace NUMINAMATH_CALUDE_prob_B_wins_at_least_one_l1388_138847

/-- The probability of player A winning against player B in a single match. -/
def prob_A_win : ℝ := 0.5

/-- The probability of player B winning against player A in a single match. -/
def prob_B_win : ℝ := 0.3

/-- The probability of a tie between players A and B in a single match. -/
def prob_tie : ℝ := 0.2

/-- The number of matches played between A and B. -/
def num_matches : ℕ := 2

/-- Theorem: The probability of B winning at least one match against A in two independent matches. -/
theorem prob_B_wins_at_least_one (h1 : prob_A_win + prob_B_win + prob_tie = 1) :
  1 - (1 - prob_B_win) ^ num_matches = 0.51 := by
  sorry

end NUMINAMATH_CALUDE_prob_B_wins_at_least_one_l1388_138847


namespace NUMINAMATH_CALUDE_geometric_sequence_theorem_l1388_138845

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the arithmetic sequence condition
def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  2 * a 2 - a 1 = a 3 + 6 - 2 * a 2

-- State the theorem
theorem geometric_sequence_theorem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2 →
  arithmetic_condition a →
  ∀ n : ℕ, a n = 2^n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_theorem_l1388_138845


namespace NUMINAMATH_CALUDE_residue_of_negative_935_mod_24_l1388_138801

theorem residue_of_negative_935_mod_24 : 
  ∃ (r : ℤ), 0 ≤ r ∧ r < 24 ∧ -935 ≡ r [ZMOD 24] ∧ r = 1 :=
sorry

end NUMINAMATH_CALUDE_residue_of_negative_935_mod_24_l1388_138801


namespace NUMINAMATH_CALUDE_djibo_age_proof_l1388_138891

/-- Djibo's current age -/
def djibo_age : ℕ := 17

/-- Djibo's sister's current age -/
def sister_age : ℕ := 28

/-- Sum of Djibo's and his sister's ages 5 years ago -/
def sum_ages_5_years_ago : ℕ := 35

theorem djibo_age_proof :
  djibo_age = 17 ∧
  sister_age = 28 ∧
  (djibo_age - 5) + (sister_age - 5) = sum_ages_5_years_ago :=
by sorry

end NUMINAMATH_CALUDE_djibo_age_proof_l1388_138891


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1388_138886

/-- Given a distance of 200 miles and a speed of 25 miles per hour, the time taken is 8 hours. -/
theorem travel_time_calculation (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 200 ∧ speed = 25 → time = distance / speed → time = 8 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1388_138886


namespace NUMINAMATH_CALUDE_clock_chimes_in_day_l1388_138871

/-- Calculates the number of chimes a clock makes in a day -/
def clock_chimes : ℕ :=
  let hours_in_day : ℕ := 24
  let half_hours_in_day : ℕ := hours_in_day * 2
  let sum_of_hour_strikes : ℕ := (12 * (1 + 12)) / 2
  let total_hour_strikes : ℕ := sum_of_hour_strikes * 2
  let total_half_hour_strikes : ℕ := half_hours_in_day
  total_hour_strikes + total_half_hour_strikes

/-- Theorem stating that a clock striking hours (1 to 12) and half-hours in a 24-hour day will chime 204 times -/
theorem clock_chimes_in_day : clock_chimes = 204 := by
  sorry

end NUMINAMATH_CALUDE_clock_chimes_in_day_l1388_138871


namespace NUMINAMATH_CALUDE_m_range_l1388_138866

theorem m_range : 
  let m : ℝ := (-Real.sqrt 3 / 3) * (-2 * Real.sqrt 21)
  5 < m ∧ m < 6 := by
sorry

end NUMINAMATH_CALUDE_m_range_l1388_138866


namespace NUMINAMATH_CALUDE_F_neg_one_eq_zero_l1388_138841

noncomputable def F (x : ℝ) : ℝ := Real.sqrt (abs (x + 1)) + (9 / Real.pi) * Real.arctan (Real.sqrt (abs (x + 1)))

theorem F_neg_one_eq_zero : F (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_F_neg_one_eq_zero_l1388_138841


namespace NUMINAMATH_CALUDE_bucket_fill_time_l1388_138834

/-- Given that it takes 2 minutes to fill two-thirds of a bucket,
    prove that it takes 3 minutes to fill the entire bucket. -/
theorem bucket_fill_time :
  let partial_time : ℚ := 2
  let partial_fill : ℚ := 2/3
  let full_time : ℚ := 3
  (partial_fill * full_time = partial_time) → full_time = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l1388_138834


namespace NUMINAMATH_CALUDE_unique_point_equal_angles_l1388_138843

/-- The ellipse equation x²/4 + y² = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- The focus F = (2, 0) -/
def F : ℝ × ℝ := (2, 0)

/-- A chord AB passing through F -/
def is_chord_through_F (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), A = (2 + t * (B.1 - 2), t * B.2) ∧ 
             is_on_ellipse A.1 A.2 ∧ is_on_ellipse B.1 B.2

/-- Angles APF and BPF are equal -/
def equal_angles (P A B : ℝ × ℝ) : Prop :=
  (A.2 / (A.1 - P.1))^2 = (B.2 / (B.1 - P.1))^2

/-- The main theorem -/
theorem unique_point_equal_angles :
  ∃! (p : ℝ), p > 0 ∧ 
    (∀ (A B : ℝ × ℝ), is_chord_through_F A B → 
      equal_angles (p, 0) A B) ∧ 
    p = 2 := by sorry

end NUMINAMATH_CALUDE_unique_point_equal_angles_l1388_138843


namespace NUMINAMATH_CALUDE_g_of_3_equals_6_l1388_138814

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

theorem g_of_3_equals_6 : g 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_6_l1388_138814


namespace NUMINAMATH_CALUDE_smallest_common_multiple_10_15_gt_100_l1388_138873

theorem smallest_common_multiple_10_15_gt_100 : ∃ (n : ℕ), n > 100 ∧ n.lcm 10 = n ∧ n.lcm 15 = n ∧ ∀ (m : ℕ), m > 100 ∧ m.lcm 10 = m ∧ m.lcm 15 = m → n ≤ m :=
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_10_15_gt_100_l1388_138873


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l1388_138860

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime :
  (first_seven_primes.sum) % eighth_prime = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l1388_138860


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l1388_138897

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem ninth_term_of_arithmetic_sequence :
  ∀ (a₁ d : ℚ),
    a₁ = 2/3 →
    arithmetic_sequence a₁ d 17 = 3/2 →
    arithmetic_sequence a₁ d 9 = 13/12 :=
by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l1388_138897


namespace NUMINAMATH_CALUDE_irrational_equation_solution_l1388_138881

theorem irrational_equation_solution (a b : ℝ) : 
  Irrational a → (a * b + a - b = 1) → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_irrational_equation_solution_l1388_138881


namespace NUMINAMATH_CALUDE_soccer_league_female_fraction_l1388_138884

theorem soccer_league_female_fraction :
  -- Last year's male participants
  ∀ (last_year_males : ℕ),
  last_year_males = 30 →
  -- Total participation increase
  ∀ (total_increase_rate : ℚ),
  total_increase_rate = 108/100 →
  -- Male participation increase
  ∀ (male_increase_rate : ℚ),
  male_increase_rate = 110/100 →
  -- Female participation increase
  ∀ (female_increase_rate : ℚ),
  female_increase_rate = 115/100 →
  -- The fraction of female participants this year
  ∃ (female_fraction : ℚ),
  female_fraction = 10/43 ∧
  (∃ (last_year_females : ℕ),
    -- Total participants this year
    total_increase_rate * (last_year_males + last_year_females : ℚ) =
    -- Males this year + Females this year
    male_increase_rate * last_year_males + female_increase_rate * last_year_females ∧
    -- Female fraction calculation
    female_fraction = (female_increase_rate * last_year_females) /
      (male_increase_rate * last_year_males + female_increase_rate * last_year_females)) :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_league_female_fraction_l1388_138884


namespace NUMINAMATH_CALUDE_sequence_problem_l1388_138861

theorem sequence_problem (a b : ℝ) 
  (h1 : 0 < 2 ∧ 0 < a ∧ 0 < b ∧ 0 < 9)
  (h2 : a - 2 = b - a)  -- arithmetic sequence condition
  (h3 : a / 2 = b / a ∧ b / a = 9 / b)  -- geometric sequence condition
  : a = 4 ∧ b = 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l1388_138861


namespace NUMINAMATH_CALUDE_rubies_in_chest_l1388_138806

theorem rubies_in_chest (diamonds : ℕ) (difference : ℕ) (rubies : ℕ) : 
  diamonds = 421 → difference = 44 → diamonds = rubies + difference → rubies = 377 := by
  sorry

end NUMINAMATH_CALUDE_rubies_in_chest_l1388_138806


namespace NUMINAMATH_CALUDE_bridge_length_is_two_km_l1388_138885

/-- The length of a bridge crossed by a man -/
def bridge_length (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: The length of a bridge is 2 km when crossed by a man walking at 8 km/hr in 15 minutes -/
theorem bridge_length_is_two_km :
  let speed := 8 -- km/hr
  let time := 15 / 60 -- 15 minutes converted to hours
  bridge_length speed time = 2 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_is_two_km_l1388_138885


namespace NUMINAMATH_CALUDE_road_trip_distance_l1388_138892

/-- Road trip problem -/
theorem road_trip_distance (total_time hours_driving friend_distance jenna_speed friend_speed : ℝ)
  (h1 : total_time = 10)
  (h2 : hours_driving = total_time - 1)
  (h3 : friend_distance = 100)
  (h4 : jenna_speed = 50)
  (h5 : friend_speed = 20) :
  jenna_speed * (hours_driving - friend_distance / friend_speed) = 200 :=
by sorry

end NUMINAMATH_CALUDE_road_trip_distance_l1388_138892


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l1388_138853

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken to travel downstream. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 16 km/hr in still water, traveling in a stream
    with a speed of 4 km/hr for 3 hours, will travel 60 km downstream. -/
theorem boat_downstream_distance :
  distance_downstream 16 4 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l1388_138853


namespace NUMINAMATH_CALUDE_final_water_level_l1388_138898

/-- Represents the water level change in a reservoir over time -/
def waterLevelChange (initialLevel : Real) (riseRate : Real) (fallRate : Real) : Real :=
  let riseTime := 4  -- 8 a.m. to 12 p.m.
  let fallTime := 6  -- 12 p.m. to 6 p.m.
  initialLevel + riseTime * riseRate - fallTime * fallRate

/-- Theorem stating the final water level at 6 p.m. -/
theorem final_water_level (initialLevel : Real) (riseRate : Real) (fallRate : Real) :
  initialLevel = 45 ∧ riseRate = 0.6 ∧ fallRate = 0.3 →
  waterLevelChange initialLevel riseRate fallRate = 45.6 :=
by sorry

end NUMINAMATH_CALUDE_final_water_level_l1388_138898


namespace NUMINAMATH_CALUDE_ellipse_and_quadratic_conditions_l1388_138868

/-- Represents an ellipse equation with parameter a -/
def is_ellipse (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*a) + y^2 / (3*a - 6) = 1

/-- Checks if the ellipse has foci on the x-axis -/
def has_foci_on_x_axis (a : ℝ) : Prop :=
  2*a < 3*a - 6

/-- Represents the quadratic inequality with parameter a -/
def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 + (a + 4)*x + 16 > 0

/-- Checks if the solution set of the quadratic inequality is ℝ -/
def solution_set_is_reals (a : ℝ) : Prop :=
  ∀ x : ℝ, quadratic_inequality a x

/-- The main theorem stating the conditions for a -/
theorem ellipse_and_quadratic_conditions (a : ℝ) :
  (is_ellipse a ∧ has_foci_on_x_axis a ∧ solution_set_is_reals a) ↔ (2 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_quadratic_conditions_l1388_138868


namespace NUMINAMATH_CALUDE_intersection_implies_m_range_l1388_138849

/-- The set M defined by the equation 3x^2 + 4y^2 - 6mx + 3m^2 - 12 = 0 -/
def M (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1^2 + 4 * p.2^2 - 6 * m * p.1 + 3 * m^2 - 12 = 0}

/-- The set N defined by the equation 2y^2 - 12x + 9 = 0 -/
def N : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.2^2 - 12 * p.1 + 9 = 0}

/-- Theorem stating that if M and N have a non-empty intersection,
    then m is in the range [-5/4, 11/4] -/
theorem intersection_implies_m_range :
  ∀ m : ℝ, (M m ∩ N).Nonempty → -5/4 ≤ m ∧ m ≤ 11/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_range_l1388_138849


namespace NUMINAMATH_CALUDE_four_tangent_lines_l1388_138821

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if two circles are on the same side of a line -/
def sameSideOfLine (A B : Circle) (m : Line) : Prop := sorry

/-- Predicate to check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Function to reflect a line over another line -/
def reflectLine (l : Line) (m : Line) : Line := sorry

/-- The main theorem -/
theorem four_tangent_lines (A B : Circle) (m : Line) 
  (h : sameSideOfLine A B m) : 
  ∃ (l₁ l₂ l₃ l₄ : Line), 
    (isTangent l₁ A ∧ isTangent (reflectLine l₁ m) B) ∧
    (isTangent l₂ A ∧ isTangent (reflectLine l₂ m) B) ∧
    (isTangent l₃ A ∧ isTangent (reflectLine l₃ m) B) ∧
    (isTangent l₄ A ∧ isTangent (reflectLine l₄ m) B) ∧
    (l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₁ ≠ l₄ ∧ l₂ ≠ l₃ ∧ l₂ ≠ l₄ ∧ l₃ ≠ l₄) :=
by sorry

end NUMINAMATH_CALUDE_four_tangent_lines_l1388_138821


namespace NUMINAMATH_CALUDE_square_number_plus_minus_five_is_square_l1388_138878

theorem square_number_plus_minus_five_is_square : ∃ (n : ℕ), 
  (∃ (a : ℕ), n = a^2) ∧ 
  (∃ (b : ℕ), n + 5 = b^2) ∧ 
  (∃ (c : ℕ), n - 5 = c^2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_square_number_plus_minus_five_is_square_l1388_138878


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1388_138825

theorem cubic_root_sum (a b c : ℝ) : 
  a^3 - 15*a^2 + 22*a - 8 = 0 →
  b^3 - 15*b^2 + 22*b - 8 = 0 →
  c^3 - 15*c^2 + 22*c - 8 = 0 →
  (a / ((1/a) + b*c)) + (b / ((1/b) + c*a)) + (c / ((1/c) + a*b)) = 181/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1388_138825


namespace NUMINAMATH_CALUDE_boys_girls_relation_l1388_138867

/-- Represents the number of girls a boy dances with based on his position -/
def girls_danced_with (n : ℕ) : ℕ := 2 * n + 1

/-- 
Theorem: In a class where boys dance with girls following a specific pattern,
the number of boys is related to the number of girls by b = (g - 1) / 2.
-/
theorem boys_girls_relation (b g : ℕ) (h1 : b > 0) (h2 : g > 0) 
  (h3 : ∀ n, n ∈ Finset.range b → girls_danced_with n ≤ g) 
  (h4 : girls_danced_with b = g) : 
  b = (g - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_boys_girls_relation_l1388_138867


namespace NUMINAMATH_CALUDE_rectangle_side_length_l1388_138880

/-- Given two rectangles A and B, where A has sides a and b, and B has sides c and d,
    with the ratio of corresponding sides being 3/4, prove that when a = 3 and b = 6,
    the length of side d in Rectangle B is 8. -/
theorem rectangle_side_length (a b c d : ℝ) : 
  a = 3 → b = 6 → a / c = 3 / 4 → b / d = 3 / 4 → d = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l1388_138880


namespace NUMINAMATH_CALUDE_tims_coins_value_l1388_138822

/-- Represents the number of coins Tim has -/
def total_coins : ℕ := 18

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the number of dimes Tim has -/
def num_dimes : ℕ := 8

/-- Represents the number of quarters Tim has -/
def num_quarters : ℕ := 10

/-- Theorem stating the total value of Tim's coins -/
theorem tims_coins_value :
  (num_dimes * dime_value + num_quarters * quarter_value = 330) ∧
  (num_dimes + num_quarters = total_coins) ∧
  (num_dimes + 2 = num_quarters) :=
sorry

end NUMINAMATH_CALUDE_tims_coins_value_l1388_138822


namespace NUMINAMATH_CALUDE_best_approximation_l1388_138894

-- Define the function f(x) = x^2 - 3x - 4.6
def f (x : ℝ) : ℝ := x^2 - 3*x - 4.6

-- Define the table of values
def table : List (ℝ × ℝ) := [
  (-1.13, 4.67),
  (-1.12, 4.61),
  (-1.11, 4.56),
  (-1.10, 4.51),
  (-1.09, 4.46),
  (-1.08, 4.41),
  (-1.07, 4.35)
]

-- Define the given options
def options : List ℝ := [-1.073, -1.089, -1.117, -1.123]

-- Theorem statement
theorem best_approximation :
  ∃ (x : ℝ), x ∈ options ∧
  ∀ (y : ℝ), y ∈ options → |f x| ≤ |f y| ∧
  x = -1.117 := by
  sorry

end NUMINAMATH_CALUDE_best_approximation_l1388_138894


namespace NUMINAMATH_CALUDE_log_equality_l1388_138817

theorem log_equality (y : ℝ) : y = (Real.log 3 / Real.log 9) ^ (Real.log 9 / Real.log 3) → Real.log y / Real.log 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l1388_138817
