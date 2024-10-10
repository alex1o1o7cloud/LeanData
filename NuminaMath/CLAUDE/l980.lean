import Mathlib

namespace optimal_solution_l980_98051

-- Define the problem parameters
def days_together : ℝ := 12
def cost_A_per_day : ℝ := 40000
def cost_B_per_day : ℝ := 30000
def max_days : ℝ := 30 -- one month

-- Define the relationship between Team A and Team B's completion times
def team_B_multiplier : ℝ := 1.5

-- Define the function to calculate days needed for Team A
def days_A (x : ℝ) : Prop := (1 / x) + (1 / (team_B_multiplier * x)) = (1 / days_together)

-- Define the function to calculate days needed for Team B
def days_B (x : ℝ) : ℝ := team_B_multiplier * x

-- Define the cost function for Team A working alone
def cost_A (x : ℝ) : ℝ := cost_A_per_day * x

-- Define the cost function for Team B working alone
def cost_B (x : ℝ) : ℝ := cost_B_per_day * days_B x

-- Define the cost function for both teams working together
def cost_together : ℝ := (cost_A_per_day + cost_B_per_day) * days_together

-- Theorem: Team A working alone for 20 days is the optimal solution
theorem optimal_solution (x : ℝ) :
  days_A x →
  x ≤ max_days →
  days_B x ≤ max_days →
  cost_A x ≤ cost_B x ∧
  cost_A x ≤ cost_together ∧
  x = 20 ∧
  cost_A x = 800000 :=
sorry

end optimal_solution_l980_98051


namespace cube_root_simplification_l980_98005

theorem cube_root_simplification : 
  (80^3 + 100^3 + 120^3 : ℝ)^(1/3) = 20 * 405^(1/3) := by sorry

end cube_root_simplification_l980_98005


namespace root_sum_quotient_l980_98041

/-- Given a quadratic equation m(x^2 - 2x) + 3x + 4 = 0 with roots p and q,
    and m₁ and m₂ are values of m for which p/q + q/p = 2,
    prove that m₁/m₂ + m₂/m₁ = 178/9 -/
theorem root_sum_quotient (m₁ m₂ : ℝ) (p q : ℝ) :
  (m₁ * (p^2 - 2*p) + 3*p + 4 = 0) →
  (m₁ * (q^2 - 2*q) + 3*q + 4 = 0) →
  (m₂ * (p^2 - 2*p) + 3*p + 4 = 0) →
  (m₂ * (q^2 - 2*q) + 3*q + 4 = 0) →
  (p / q + q / p = 2) →
  m₁ / m₂ + m₂ / m₁ = 178 / 9 := by
  sorry

end root_sum_quotient_l980_98041


namespace solution_set_part1_solution_set_part2_l980_98083

-- Define the function y = mx^2 - mx - 1
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | y (1/2) x < 0} = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Part 2
theorem solution_set_part2 (m : ℝ) :
  {x : ℝ | y m x < (1 - m) * x - 1} =
    if m = 0 then
      {x : ℝ | x > 0}
    else if m > 0 then
      {x : ℝ | 0 < x ∧ x < 1/m}
    else
      {x : ℝ | x < 1/m ∨ x > 0} := by sorry

end solution_set_part1_solution_set_part2_l980_98083


namespace is_min_point_l980_98073

/-- The function representing the translated graph -/
def f (x : ℝ) : ℝ := |x - 4| - 3

/-- The minimum point of the translated graph -/
def min_point : ℝ × ℝ := (4, -3)

/-- Theorem stating that min_point is the minimum of the function f -/
theorem is_min_point :
  ∀ x : ℝ, f x ≥ f min_point.fst ∧ f min_point.fst = min_point.snd := by
  sorry

end is_min_point_l980_98073


namespace quadratic_discriminant_l980_98015

theorem quadratic_discriminant :
  let a : ℝ := 2
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := 1/2
  (b^2 - 4*a*c) = 2 + 4 * Real.sqrt 2 := by
  sorry

end quadratic_discriminant_l980_98015


namespace total_hotdogs_by_wednesday_l980_98018

def hotdog_sequence (n : ℕ) : ℕ := 10 + 2 * n

theorem total_hotdogs_by_wednesday :
  (hotdog_sequence 0) + (hotdog_sequence 1) + (hotdog_sequence 2) = 36 := by
  sorry

end total_hotdogs_by_wednesday_l980_98018


namespace specific_trapezoid_diagonal_l980_98064

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  leg : ℝ

/-- The diagonal length of an isosceles trapezoid -/
def diagonal_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The diagonal length of the specific isosceles trapezoid is 2√52 -/
theorem specific_trapezoid_diagonal :
  let t : IsoscelesTrapezoid := { base1 := 27, base2 := 11, leg := 12 }
  diagonal_length t = 2 * Real.sqrt 52 := by
  sorry

end specific_trapezoid_diagonal_l980_98064


namespace sine_bounds_l980_98085

theorem sine_bounds (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  (Real.sqrt 2 / 2) * x ≤ Real.sin x ∧ Real.sin x ≤ x := by
  sorry

end sine_bounds_l980_98085


namespace geometric_sequence_ninth_term_l980_98059

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- Given a geometric sequence where the 5th term is 80 and the 7th term is 320, the 9th term is 1280. -/
theorem geometric_sequence_ninth_term (a : ℕ → ℝ) 
    (h_geom : IsGeometricSequence a) 
    (h_5th : a 5 = 80) 
    (h_7th : a 7 = 320) : 
  a 9 = 1280 := by
  sorry


end geometric_sequence_ninth_term_l980_98059


namespace valid_selections_count_l980_98055

/-- The number of male athletes -/
def num_males : ℕ := 4

/-- The number of female athletes -/
def num_females : ℕ := 5

/-- The total number of athletes to be chosen -/
def num_chosen : ℕ := 3

/-- The function to calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of valid selections -/
def total_selections : ℕ := 
  choose num_males 1 * choose num_females 2 + 
  choose num_males 2 * choose num_females 1

theorem valid_selections_count : total_selections = 70 := by
  sorry

end valid_selections_count_l980_98055


namespace new_year_markup_l980_98030

theorem new_year_markup (initial_markup : ℝ) (february_discount : ℝ) (final_profit : ℝ) :
  initial_markup = 0.2 →
  february_discount = 0.06 →
  final_profit = 0.41 →
  ∃ (new_year_markup : ℝ),
    (1 - february_discount) * (1 + new_year_markup) * (1 + initial_markup) = 1 + final_profit ∧
    new_year_markup = 0.5 :=
by sorry

end new_year_markup_l980_98030


namespace number_of_tourists_l980_98008

theorem number_of_tourists (k : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) → 
  (∃ n : ℕ, n = 23 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) :=
by sorry

end number_of_tourists_l980_98008


namespace sixth_root_of_unity_product_l980_98078

theorem sixth_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = -6 := by
  sorry

end sixth_root_of_unity_product_l980_98078


namespace find_k_l980_98065

theorem find_k (k : ℚ) (h : 56 / k = 4) : k = 14 := by
  sorry

end find_k_l980_98065


namespace sam_coupons_l980_98032

/-- Calculates the number of coupons Sam used when buying tuna cans. -/
def calculate_coupons (num_cans : ℕ) (can_cost : ℕ) (coupon_value : ℕ) (paid : ℕ) (change : ℕ) : ℕ :=
  let total_spent := paid - change
  let total_cost := num_cans * can_cost
  let savings := total_cost - total_spent
  savings / coupon_value

/-- Proves that Sam had 5 coupons given the problem conditions. -/
theorem sam_coupons :
  calculate_coupons 9 175 25 2000 550 = 5 := by
  sorry

#eval calculate_coupons 9 175 25 2000 550

end sam_coupons_l980_98032


namespace solve_for_b_and_c_l980_98043

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 + a*x - 12 = 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c = 0}

-- State the theorem
theorem solve_for_b_and_c (a b c : ℝ) : 
  A a ≠ B b c →
  A a ∪ B b c = {-3, 4} →
  A a ∩ B b c = {-3} →
  b = 3 ∧ c = 9 := by
  sorry


end solve_for_b_and_c_l980_98043


namespace female_listeners_l980_98002

theorem female_listeners (total_listeners male_listeners : ℕ) 
  (h1 : total_listeners = 180) 
  (h2 : male_listeners = 80) : 
  total_listeners - male_listeners = 100 := by
  sorry

end female_listeners_l980_98002


namespace ana_win_probability_l980_98042

/-- Represents the probability of winning for a player in the coin flipping game -/
def winProbability (playerPosition : ℕ) : ℚ :=
  (1 / 2) ^ playerPosition / (1 - (1 / 2) ^ 4)

/-- The coin flipping game with four players -/
theorem ana_win_probability :
  winProbability 4 = 1 / 30 := by
  sorry

end ana_win_probability_l980_98042


namespace proposition_implication_l980_98035

theorem proposition_implication (P : ℕ+ → Prop) 
  (h1 : ∀ k : ℕ+, P k → P (k + 1))
  (h2 : ¬ P 7) : 
  ¬ P 6 := by
  sorry

end proposition_implication_l980_98035


namespace last_digit_base_5_l980_98075

theorem last_digit_base_5 (n : ℕ) (h : n = 119) : n % 5 = 4 := by
  sorry

end last_digit_base_5_l980_98075


namespace arithmetic_geometric_sequence_solution_l980_98068

theorem arithmetic_geometric_sequence_solution :
  ∀ a b c : ℝ,
  (b - a = c - b) →                      -- arithmetic sequence
  (a + b + c = 12) →                     -- sum is 12
  ((b + 2)^2 = (a + 2) * (c + 5)) →      -- geometric sequence condition
  ((a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2)) :=
by
  sorry

end arithmetic_geometric_sequence_solution_l980_98068


namespace ad_value_l980_98038

/-- Given two-digit numbers ab and cd, and that 1ab is a three-digit number -/
def two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def three_digit (n : ℕ) : Prop := n ≥ 100 ∧ n < 1000

/-- The theorem statement -/
theorem ad_value (a b c d : ℕ) 
  (h1 : two_digit (10 * a + b))
  (h2 : two_digit (10 * c + d))
  (h3 : three_digit (100 + 10 * a + b))
  (h4 : 10 * a + b = 10 * c + d + 24)
  (h5 : 100 + 10 * a + b = 100 * c + 10 * d + 1 + 15) :
  10 * a + d = 32 := by
sorry

end ad_value_l980_98038


namespace probability_concentric_circles_l980_98037

/-- The probability of a randomly chosen point from a circle with radius 3 
    lying within a concentric circle with radius 1 is 1/9. -/
theorem probability_concentric_circles : 
  let outer_radius : ℝ := 3
  let inner_radius : ℝ := 1
  let outer_area := π * outer_radius^2
  let inner_area := π * inner_radius^2
  (inner_area / outer_area : ℝ) = 1 / 9 := by
sorry

end probability_concentric_circles_l980_98037


namespace jason_pears_l980_98047

theorem jason_pears (total pears_keith pears_mike : ℕ) 
  (h_total : total = 105)
  (h_keith : pears_keith = 47)
  (h_mike : pears_mike = 12) :
  total - (pears_keith + pears_mike) = 46 := by
  sorry

end jason_pears_l980_98047


namespace circles_separated_l980_98053

-- Define the circles
def C₁ (x y : ℝ) : Prop := (x + 2)^2 + (y + 1)^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4

-- Define the centers of the circles
def center₁ : ℝ × ℝ := (-2, -1)
def center₂ : ℝ × ℝ := (2, 1)

-- Define the radius of the circles
def radius : ℝ := 2

-- Theorem: The circles C₁ and C₂ are separated
theorem circles_separated : 
  ∀ (x y : ℝ), (C₁ x y ∧ C₂ x y) → 
  (center₁.1 - center₂.1)^2 + (center₁.2 - center₂.2)^2 > (radius + radius)^2 :=
sorry

end circles_separated_l980_98053


namespace hiking_trip_solution_l980_98091

/-- Represents the hiking trip scenario -/
structure HikingTrip where
  men_count : ℕ
  women_count : ℕ
  total_weight : ℝ
  men_backpack_weight : ℝ
  women_backpack_weight : ℝ

/-- Checks if the hiking trip satisfies the given conditions -/
def is_valid_hiking_trip (trip : HikingTrip) : Prop :=
  trip.men_count = 2 ∧
  trip.women_count = 3 ∧
  trip.total_weight = 44 ∧
  trip.men_count * trip.men_backpack_weight + trip.women_count * trip.women_backpack_weight = trip.total_weight ∧
  trip.men_backpack_weight + trip.women_backpack_weight + trip.women_backpack_weight / 2 = 
    trip.women_backpack_weight + trip.men_backpack_weight / 2

theorem hiking_trip_solution (trip : HikingTrip) :
  is_valid_hiking_trip trip → trip.women_backpack_weight = 8 ∧ trip.men_backpack_weight = 10 := by
  sorry

end hiking_trip_solution_l980_98091


namespace jeans_pricing_l980_98013

theorem jeans_pricing (C : ℝ) (h1 : C > 0) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.96 * C
  (customer_price - retailer_price) / retailer_price = 0.40 := by sorry

end jeans_pricing_l980_98013


namespace equilateral_triangle_count_l980_98014

/-- Represents a point in a hexagonal lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a hexagonal lattice with a secondary layer -/
structure HexagonalLattice where
  inner : List LatticePoint
  outer : List LatticePoint

/-- Represents an equilateral triangle in the lattice -/
structure EquilateralTriangle where
  vertices : List LatticePoint
  sideLength : ℝ

/-- Function to count equilateral triangles in the lattice -/
def countEquilateralTriangles (lattice : HexagonalLattice) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles with side lengths 1 or √7 is 6 -/
theorem equilateral_triangle_count (lattice : HexagonalLattice) :
  countEquilateralTriangles lattice = 6 :=
sorry

end equilateral_triangle_count_l980_98014


namespace bridge_length_calculation_l980_98050

/-- The width of the river in inches -/
def river_width : ℕ := 487

/-- The additional length needed to cross the river in inches -/
def additional_length : ℕ := 192

/-- The current length of the bridge in inches -/
def bridge_length : ℕ := river_width - additional_length

theorem bridge_length_calculation :
  bridge_length = 295 := by sorry

end bridge_length_calculation_l980_98050


namespace runner_speed_l980_98080

/-- Calculates the speed of a runner overtaking a parade -/
theorem runner_speed (parade_length : ℝ) (parade_speed : ℝ) (runner_time : ℝ) :
  parade_length = 2 →
  parade_speed = 3 →
  runner_time = 0.222222222222 →
  parade_length / runner_time = 9 :=
by sorry

end runner_speed_l980_98080


namespace inequality_proof_l980_98016

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end inequality_proof_l980_98016


namespace certain_amount_problem_l980_98023

theorem certain_amount_problem : ∃ x : ℤ, 7 * 5 - 15 = 2 * 5 + x ∧ x = 10 := by
  sorry

end certain_amount_problem_l980_98023


namespace tower_has_four_levels_l980_98079

/-- Calculates the number of levels in a tower given the number of steps per level,
    blocks per step, and total blocks climbed. -/
def tower_levels (steps_per_level : ℕ) (blocks_per_step : ℕ) (total_blocks : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / steps_per_level

/-- Theorem stating that a tower with 8 steps per level, 3 blocks per step,
    and 96 total blocks climbed has 4 levels. -/
theorem tower_has_four_levels :
  tower_levels 8 3 96 = 4 := by
  sorry

end tower_has_four_levels_l980_98079


namespace cube_strictly_increasing_l980_98045

theorem cube_strictly_increasing (a b : ℝ) : a < b → a^3 < b^3 := by
  sorry

end cube_strictly_increasing_l980_98045


namespace cost_of_dozen_pens_l980_98021

/-- The cost of pens and pencils -/
def CostProblem (pen_cost : ℚ) (pencil_cost : ℚ) : Prop :=
  -- Condition 1: The cost of 3 pens and 5 pencils is Rs. 260
  3 * pen_cost + 5 * pencil_cost = 260 ∧
  -- Condition 2: The cost ratio of one pen to one pencil is 5:1
  pen_cost = 5 * pencil_cost

/-- The cost of one dozen pens is Rs. 780 -/
theorem cost_of_dozen_pens 
  (pen_cost : ℚ) (pencil_cost : ℚ) 
  (h : CostProblem pen_cost pencil_cost) : 
  12 * pen_cost = 780 := by
  sorry

end cost_of_dozen_pens_l980_98021


namespace right_triangle_sin_x_l980_98056

theorem right_triangle_sin_x (X Y Z : Real) (sinX cosX tanX : Real) :
  -- Right triangle XYZ with ∠Y = 90°
  (X^2 + Y^2 = Z^2) →
  -- 4sinX = 5cosX
  (4 * sinX = 5 * cosX) →
  -- tanX = XY/YZ
  (tanX = X / Y) →
  -- sinX = 5√41 / 41
  sinX = 5 * Real.sqrt 41 / 41 := by
  sorry

end right_triangle_sin_x_l980_98056


namespace height_radius_ratio_is_2pi_l980_98039

/-- A cylinder with a square lateral surface -/
structure SquareLateralCylinder where
  radius : ℝ
  height : ℝ
  lateral_surface_is_square : height = 2 * Real.pi * radius

/-- The ratio of height to radius for a cylinder with a square lateral surface is 2π -/
theorem height_radius_ratio_is_2pi (c : SquareLateralCylinder) :
  c.height / c.radius = 2 * Real.pi := by
  sorry

end height_radius_ratio_is_2pi_l980_98039


namespace series_sum_equals_one_l980_98070

/-- The sum of the series Σ(n=0 to ∞) of 3^n / (1 + 3^n + 3^(n+1) + 3^(2n+1)) is equal to 1 -/
theorem series_sum_equals_one :
  ∑' n : ℕ, (3 : ℝ) ^ n / (1 + 3 ^ n + 3 ^ (n + 1) + 3 ^ (2 * n + 1)) = 1 := by
  sorry

end series_sum_equals_one_l980_98070


namespace log_sum_squares_primes_l980_98087

theorem log_sum_squares_primes (a b : ℕ) (ha : Prime a) (hb : Prime b) 
  (hab : a ≠ b) (ha_gt_2 : a > 2) (hb_gt_2 : b > 2) :
  Real.log (a^2) / Real.log (a * b) + Real.log (b^2) / Real.log (a * b) = 2 := by
sorry

end log_sum_squares_primes_l980_98087


namespace product_inequality_l980_98027

theorem product_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) : 
  min (a * (1 - b)) (min (b * (1 - c)) (c * (1 - a))) ≤ 1/4 := by
  sorry

end product_inequality_l980_98027


namespace principal_correct_l980_98025

/-- Calculates the final amount after compound interest with varying rates and additional investments -/
def final_amount (principal : ℝ) (initial_rate : ℝ) (rate_increase : ℝ) (years : ℝ) (annual_investment : ℝ) : ℝ :=
  let first_year := principal * (1 + initial_rate) + annual_investment
  let second_year := first_year * (1 + (initial_rate + rate_increase)) + annual_investment
  second_year * (1 + (initial_rate + 2 * rate_increase) * (years - 2))

/-- The principal amount is correct if it results in the expected final amount -/
theorem principal_correct (principal : ℝ) : 
  abs (final_amount principal 0.07 0.02 2.4 200 - 1120) < 0.01 → 
  abs (principal - 556.25) < 0.01 := by
  sorry

#eval final_amount 556.25 0.07 0.02 2.4 200

end principal_correct_l980_98025


namespace function_properties_l980_98082

noncomputable section

variables (a b : ℝ) (x : ℝ)

def f (x : ℝ) := -a * x + b + a * x * Real.log x

theorem function_properties :
  a ≠ 0 →
  f e = 2 →
  (b = 2) ∧
  (a > 0 →
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂)) ∧
  (a < 0 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂)) :=
by sorry

end

end function_properties_l980_98082


namespace original_number_problem_l980_98062

theorem original_number_problem (x : ℝ) : x * 1.2 = 480 → x = 400 := by
  sorry

end original_number_problem_l980_98062


namespace berry_expense_l980_98044

-- Define the daily consumption of berries
def daily_consumption : ℚ := 1/2

-- Define the package size
def package_size : ℚ := 1

-- Define the cost per package
def cost_per_package : ℚ := 2

-- Define the number of days
def days : ℕ := 30

-- Theorem to prove
theorem berry_expense : 
  (days : ℚ) * cost_per_package * (daily_consumption / package_size) = 30 := by
  sorry

end berry_expense_l980_98044


namespace total_students_in_classes_l980_98093

theorem total_students_in_classes (class_a class_b : ℕ) : 
  (80 * class_a = 90 * (class_a - 8) + 20 * 8) →
  (70 * class_b = 85 * (class_b - 6) + 30 * 6) →
  class_a + class_b = 78 := by
  sorry

end total_students_in_classes_l980_98093


namespace tank_drainage_rate_l980_98095

/-- Prove that given the conditions of the tank filling problem, 
    the drainage rate of pipe C is 20 liters per minute. -/
theorem tank_drainage_rate 
  (tank_capacity : ℕ) 
  (fill_rate_A : ℕ) 
  (fill_rate_B : ℕ) 
  (total_time : ℕ) 
  (h1 : tank_capacity = 800)
  (h2 : fill_rate_A = 40)
  (h3 : fill_rate_B = 30)
  (h4 : total_time = 48)
  : ∃ (drain_rate_C : ℕ), 
    drain_rate_C = 20 ∧ 
    (total_time / 3) * (fill_rate_A + fill_rate_B - drain_rate_C) = tank_capacity :=
by sorry

end tank_drainage_rate_l980_98095


namespace multiplication_table_odd_fraction_l980_98010

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 13
  let total_entries : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_entries : ℕ := odd_numbers * odd_numbers
  (odd_entries : ℚ) / total_entries = 36 / 169 := by
sorry

end multiplication_table_odd_fraction_l980_98010


namespace parallel_postulate_l980_98012

-- Define a Point type
def Point : Type := ℝ × ℝ

-- Define a Line type
def Line : Type := Point → Point → Prop

-- Define a parallel relation between lines
def Parallel (l1 l2 : Line) : Prop := sorry

-- Define a point being on a line
def OnLine (p : Point) (l : Line) : Prop := sorry

-- State the theorem
theorem parallel_postulate (l : Line) (p : Point) : 
  ¬(OnLine p l) → ∃! (m : Line), Parallel m l ∧ OnLine p m := by sorry

end parallel_postulate_l980_98012


namespace fraction_meaningful_l980_98000

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (3 - x)) ↔ x ≠ 3 := by
  sorry

end fraction_meaningful_l980_98000


namespace wire_length_ratio_l980_98048

/-- The length of one piece of wire used in Bonnie's cube frame -/
def bonnie_wire_length : ℝ := 8

/-- The number of wire pieces used in Bonnie's cube frame -/
def bonnie_wire_count : ℕ := 12

/-- The length of one piece of wire used in Roark's unit cube frames -/
def roark_wire_length : ℝ := 2

/-- The volume of Bonnie's cube -/
def bonnie_cube_volume : ℝ := bonnie_wire_length ^ 3

/-- The volume of one of Roark's unit cubes -/
def roark_unit_cube_volume : ℝ := roark_wire_length ^ 3

/-- The number of wire pieces needed for one of Roark's unit cube frames -/
def roark_wire_count_per_cube : ℕ := 12

theorem wire_length_ratio :
  (bonnie_wire_count * bonnie_wire_length) / 
  (((bonnie_cube_volume / roark_unit_cube_volume) : ℝ) * 
   (roark_wire_count_per_cube : ℝ) * roark_wire_length) = 1 / 16 := by
  sorry

end wire_length_ratio_l980_98048


namespace decreasing_exponential_function_range_l980_98081

theorem decreasing_exponential_function_range :
  ∀ a : ℝ, a > 0 ∧ a ≠ 1 →
  (∀ x y : ℝ, x < y → a^x > a^y) →
  a ∈ Set.Ioo 0 1 :=
by sorry

end decreasing_exponential_function_range_l980_98081


namespace intersecting_lines_regions_l980_98089

/-- The number of regions created by n intersecting lines -/
def num_regions (n : ℕ) : ℕ := (n * n - n + 2) / 2 + 1

/-- Theorem stating that for any n ≥ 5, there exists a configuration of n intersecting lines
    that divides the plane into at least n regions -/
theorem intersecting_lines_regions (n : ℕ) (h : n ≥ 5) :
  num_regions n ≥ n :=
by sorry

end intersecting_lines_regions_l980_98089


namespace problem_solution_l980_98099

-- Define propositions P and Q
def P (x a : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := 2 * Real.sin x > 1 ∧ x^2 - x - 2 < 0

theorem problem_solution (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, a = 2 ∧ P x a ∧ Q x → π / 6 < x ∧ x < 2) ∧
  ((∀ x : ℝ, ¬(P x a) → ¬(Q x)) ∧ (∃ x : ℝ, Q x ∧ P x a) → 2 / 3 ≤ a) :=
by sorry

end problem_solution_l980_98099


namespace felipe_house_building_time_l980_98098

/-- Felipe and Emilio's house building problem -/
theorem felipe_house_building_time :
  ∀ (felipe_time emilio_time : ℝ),
  felipe_time + emilio_time = 7.5 →
  felipe_time = (1/2) * emilio_time →
  felipe_time * 12 = 30 := by
  sorry

end felipe_house_building_time_l980_98098


namespace exists_greater_term_l980_98007

/-- Two sequences of positive reals satisfying given recurrence relations -/
def SequencePair (x y : ℕ → ℝ) : Prop :=
  (∀ n, x n > 0 ∧ y n > 0) ∧
  (∀ n, x (n + 2) = x n + (x (n + 1))^2) ∧
  (∀ n, y (n + 2) = (y n)^2 + y (n + 1)) ∧
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

/-- There exists a k such that x_k > y_k -/
theorem exists_greater_term (x y : ℕ → ℝ) (h : SequencePair x y) :
  ∃ k, x k > y k := by
  sorry

end exists_greater_term_l980_98007


namespace expression_a_result_l980_98022

theorem expression_a_result : 
  (7 * (2 / 3) + 16 * (5 / 12)) = 34 / 3 := by sorry

end expression_a_result_l980_98022


namespace complex_multiplication_l980_98019

theorem complex_multiplication (i : ℂ) : i^2 = -1 → (1 - i)^2 * i = 2 := by
  sorry

end complex_multiplication_l980_98019


namespace two_red_marbles_probability_l980_98060

/-- The probability of drawing two red marbles without replacement from a bag containing 5 red marbles and 7 white marbles is 5/33. -/
theorem two_red_marbles_probability :
  let total_marbles : ℕ := 5 + 7
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let prob_first_red : ℚ := red_marbles / total_marbles
  let prob_second_red : ℚ := (red_marbles - 1) / (total_marbles - 1)
  prob_first_red * prob_second_red = 5 / 33 :=
by sorry

end two_red_marbles_probability_l980_98060


namespace ceiling_sqrt_count_l980_98036

theorem ceiling_sqrt_count : 
  (Finset.range 325 \ Finset.range 290).card = 35 := by sorry

#check ceiling_sqrt_count

end ceiling_sqrt_count_l980_98036


namespace photo_arrangements_count_l980_98092

/-- Represents the number of students -/
def total_students : ℕ := 7

/-- Represents the number of students on each side of the tallest student -/
def students_per_side : ℕ := 3

/-- The number of possible arrangements of students for the photo -/
def num_arrangements : ℕ := Nat.choose (total_students - 1) students_per_side

/-- Theorem stating that the number of arrangements is correct -/
theorem photo_arrangements_count :
  num_arrangements = 20 :=
sorry

end photo_arrangements_count_l980_98092


namespace fraction_modification_l980_98049

theorem fraction_modification (a b c d k : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) (h3 : d ≠ 0) (h4 : k ≠ 0) (h5 : k ≠ 1) :
  let x := (b * c - a * d) / (k * d - c)
  (a + k * x) / (b + x) = c / d :=
by sorry

end fraction_modification_l980_98049


namespace longest_side_length_l980_98040

/-- The polygonal region defined by the given system of inequalities -/
def PolygonalRegion : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 ≤ 5 ∧ 3 * p.1 + p.2 ≥ 3 ∧ p.1 ≥ 0 ∧ p.2 ≥ 0}

/-- The vertices of the polygonal region -/
def Vertices : Set (ℝ × ℝ) :=
  {(0, 3), (1, 0), (0, 5)}

/-- The squared distance between two points -/
def squaredDistance (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Theorem: The length of the longest side of the polygonal region is √26 -/
theorem longest_side_length :
  ∃ (p q : ℝ × ℝ), p ∈ Vertices ∧ q ∈ Vertices ∧
  ∀ (r s : ℝ × ℝ), r ∈ Vertices → s ∈ Vertices →
  squaredDistance p q ≥ squaredDistance r s ∧
  squaredDistance p q = 26 := by
  sorry

end longest_side_length_l980_98040


namespace sons_age_l980_98067

theorem sons_age (son_age father_age : ℕ) : 
  father_age = 7 * (son_age - 8) →
  father_age / 4 = 14 →
  son_age = 16 := by
sorry

end sons_age_l980_98067


namespace computer_profit_calculation_l980_98094

theorem computer_profit_calculation (C : ℝ) :
  (C + 0.4 * C = 2240) → (C + 0.6 * C = 2560) := by
  sorry

end computer_profit_calculation_l980_98094


namespace not_divisible_by_qplus1_l980_98020

theorem not_divisible_by_qplus1 (q : ℕ) (hodd : Odd q) (hq : q > 2) :
  ¬ (q + 1 ∣ (q + 1)^((q - 1)/2) + 2) := by
  sorry

end not_divisible_by_qplus1_l980_98020


namespace two_fifths_divided_by_three_l980_98028

theorem two_fifths_divided_by_three : (2 : ℚ) / 5 / 3 = 2 / 15 := by
  sorry

end two_fifths_divided_by_three_l980_98028


namespace extreme_point_value_bound_l980_98004

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := k * Real.exp x - x^2

-- Define the derivative of f
def f_deriv (k : ℝ) (x : ℝ) : ℝ := k * Real.exp x - 2 * x

-- Theorem statement
theorem extreme_point_value_bound 
  (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂) 
  (h2 : f_deriv k x₁ = 0) 
  (h3 : f_deriv k x₂ = 0) 
  (h4 : ∀ x, x₁ < x → x < x₂ → f_deriv k x ≠ 0) : 
  0 < f k x₁ ∧ f k x₁ < 1 := by
sorry

end

end extreme_point_value_bound_l980_98004


namespace curve_arc_length_l980_98066

noncomputable def arcLength (ρ : Real → Real) (φ₁ φ₂ : Real) : Real :=
  ∫ x in φ₁..φ₂, Real.sqrt (ρ x ^ 2 + (deriv ρ x) ^ 2)

theorem curve_arc_length :
  let ρ := fun φ => 3 * Real.exp (3 * φ / 4)
  let φ₁ := -π / 2
  let φ₂ := π / 2
  arcLength ρ φ₁ φ₂ = 10 * Real.sinh (3 * π / 8) := by
  sorry

end curve_arc_length_l980_98066


namespace square_root_sum_simplification_l980_98097

theorem square_root_sum_simplification :
  Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) - 3 = 12 := by
  sorry

end square_root_sum_simplification_l980_98097


namespace base4_1212_is_102_l980_98029

def base4_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

theorem base4_1212_is_102 : base4_to_decimal [2, 1, 2, 1] = 102 := by
  sorry

end base4_1212_is_102_l980_98029


namespace lauren_tuesday_earnings_l980_98046

/-- Calculates Lauren's earnings from her social media channel -/
def laurens_earnings (commercial_rate : ℚ) (subscription_rate : ℚ) (commercial_views : ℕ) (subscriptions : ℕ) : ℚ :=
  commercial_rate * commercial_views + subscription_rate * subscriptions

theorem lauren_tuesday_earnings :
  let commercial_rate : ℚ := 1/2
  let subscription_rate : ℚ := 1
  let commercial_views : ℕ := 100
  let subscriptions : ℕ := 27
  laurens_earnings commercial_rate subscription_rate commercial_views subscriptions = 77 := by
sorry

end lauren_tuesday_earnings_l980_98046


namespace anthony_total_pencils_l980_98006

/-- The total number of pencils Anthony has after receiving pencils from others -/
def total_pencils (initial : ℕ) (from_kathryn : ℕ) (from_greg : ℕ) (from_maria : ℕ) : ℕ :=
  initial + from_kathryn + from_greg + from_maria

/-- Theorem stating that Anthony's total pencils is 287 -/
theorem anthony_total_pencils :
  total_pencils 9 56 84 138 = 287 := by
  sorry

end anthony_total_pencils_l980_98006


namespace departure_representation_l980_98024

/-- Represents the change in grain quantity -/
inductive GrainChange
| Arrival (amount : ℕ)
| Departure (amount : ℕ)

/-- Records the change in grain quantity -/
def record (change : GrainChange) : ℤ :=
  match change with
  | GrainChange.Arrival amount => amount
  | GrainChange.Departure amount => -amount

/-- Theorem: If arrival of 30 tons is recorded as +30, then -30 represents departure of 30 tons -/
theorem departure_representation :
  (record (GrainChange.Arrival 30) = 30) →
  (record (GrainChange.Departure 30) = -30) :=
by
  sorry

end departure_representation_l980_98024


namespace train_speed_problem_l980_98057

def train_journey (x : ℝ) (v : ℝ) : Prop :=
  let first_part_distance := x
  let first_part_speed := 40
  let second_part_distance := 2 * x
  let second_part_speed := v
  let total_distance := 3 * x
  let average_speed := 24
  (first_part_distance / first_part_speed + second_part_distance / second_part_speed) * average_speed = total_distance

theorem train_speed_problem (x : ℝ) (hx : x > 0) :
  ∃ v : ℝ, train_journey x v ∧ v = 120 := by
  sorry

end train_speed_problem_l980_98057


namespace sqrt_15_minus_one_over_three_lt_one_l980_98017

theorem sqrt_15_minus_one_over_three_lt_one :
  (Real.sqrt 15 - 1) / 3 < 1 := by sorry

end sqrt_15_minus_one_over_three_lt_one_l980_98017


namespace mike_picked_52_peaches_l980_98090

/-- The number of peaches Mike initially had -/
def initial_peaches : ℕ := 34

/-- The number of peaches Mike has now -/
def current_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := current_peaches - initial_peaches

theorem mike_picked_52_peaches : picked_peaches = 52 := by
  sorry

end mike_picked_52_peaches_l980_98090


namespace ratio_sum_to_y_l980_98077

theorem ratio_sum_to_y (w x y : ℚ) 
  (h1 : w / x = 2 / 3) 
  (h2 : w / y = 6 / 15) : 
  (x + y) / y = 8 / 5 := by
sorry

end ratio_sum_to_y_l980_98077


namespace no_integer_solution_for_2006_l980_98003

theorem no_integer_solution_for_2006 : ¬∃ (x y : ℤ), x^2 - y^2 = 2006 := by
  sorry

end no_integer_solution_for_2006_l980_98003


namespace k_value_l980_98058

theorem k_value (k : ℝ) : (5 + k) * (5 - k) = 5^2 - 2^3 → k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2 := by
  sorry

end k_value_l980_98058


namespace negative_square_l980_98072

theorem negative_square : -3^2 = -9 := by
  sorry

end negative_square_l980_98072


namespace endpoint_coordinate_sum_l980_98001

/-- Given a line segment with one endpoint at (10, 2) and midpoint at (4, -6),
    the sum of the coordinates of the other endpoint is -16. -/
theorem endpoint_coordinate_sum :
  ∀ (x y : ℝ),
  (x + 10) / 2 = 4 →
  (y + 2) / 2 = -6 →
  x + y = -16 := by
sorry

end endpoint_coordinate_sum_l980_98001


namespace special_function_property_l980_98063

/-- A function satisfying the given property for all real numbers -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  ∀ c d : ℝ, d^2 * g c = c^2 * g d

theorem special_function_property (g : ℝ → ℝ) (h : SpecialFunction g) (h3 : g 3 ≠ 0) :
  (g 6 + g 2) / g 3 = 40/9 := by
  sorry

end special_function_property_l980_98063


namespace equality_of_expressions_l980_98061

theorem equality_of_expressions :
  (-2^7 = (-2)^7) ∧
  (-3^2 ≠ (-3)^2) ∧
  (-3 * 2^3 ≠ -3^2 * 2) ∧
  (-((-3)^2) ≠ -((-2)^3)) := by
  sorry

end equality_of_expressions_l980_98061


namespace parallel_transitivity_false_l980_98011

-- Define the necessary types
variable (Point Line Plane : Type)

-- Define the relations
variable (belongs_to : Point → Line → Prop)
variable (intersects : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_false :
  ¬(∀ (l m : Line) (α β : Plane),
    parallel_line_plane l α →
    parallel_line_plane m β →
    parallel_planes α β →
    parallel_lines l m) :=
sorry

end parallel_transitivity_false_l980_98011


namespace chess_tournament_participants_l980_98086

theorem chess_tournament_participants (n : ℕ) (m : ℕ) : 
  (2 : ℕ) + n = number_of_participants →
  8 = points_scored_by_7th_graders →
  m * n = points_scored_by_8th_graders →
  m * n + 8 = total_points_scored →
  (n + 2) * (n + 1) / 2 = total_games_played →
  total_points_scored = total_games_played →
  n = 7 :=
sorry

end chess_tournament_participants_l980_98086


namespace container_volume_ratio_l980_98026

theorem container_volume_ratio : 
  ∀ (v1 v2 : ℚ), v1 > 0 → v2 > 0 →
  (5 / 6 : ℚ) * v1 = (3 / 4 : ℚ) * v2 →
  v1 / v2 = (9 / 10 : ℚ) := by
sorry

end container_volume_ratio_l980_98026


namespace citric_acid_weight_l980_98033

/-- The molecular weight of Citric acid in g/mol -/
def citric_acid_molecular_weight : ℝ := 192.12

/-- Theorem stating that the molecular weight of Citric acid is 192.12 g/mol -/
theorem citric_acid_weight : citric_acid_molecular_weight = 192.12 := by sorry

end citric_acid_weight_l980_98033


namespace power_function_property_l980_98031

theorem power_function_property (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x > 0, f x = x ^ a) → f 2 = Real.sqrt 2 → f 4 = 2 := by
  sorry

end power_function_property_l980_98031


namespace largest_factor_and_smallest_multiple_of_18_l980_98084

theorem largest_factor_and_smallest_multiple_of_18 :
  (∃ n : ℕ, n ≤ 18 ∧ 18 % n = 0 ∧ ∀ m : ℕ, m ≤ 18 ∧ 18 % m = 0 → m ≤ n) ∧
  (∃ k : ℕ, 18 ∣ k ∧ ∀ j : ℕ, 18 ∣ j → k ≤ j) :=
by sorry

end largest_factor_and_smallest_multiple_of_18_l980_98084


namespace negation_of_fraction_inequality_l980_98071

theorem negation_of_fraction_inequality :
  (¬ ∀ x : ℝ, 1 / (x - 2) < 0) ↔ (∃ x : ℝ, 1 / (x - 2) > 0 ∨ x = 2) := by
  sorry

end negation_of_fraction_inequality_l980_98071


namespace f_max_min_difference_l980_98074

noncomputable def f (x : ℝ) : ℝ := x * |3 - x| - (x - 3) * |x|

theorem f_max_min_difference :
  ∃ (max min : ℝ), (∀ x, f x ≤ max) ∧ (∀ x, f x ≥ min) ∧ (max - min = 9/8) := by
  sorry

end f_max_min_difference_l980_98074


namespace sum_not_prime_l980_98054

theorem sum_not_prime (a b c x y z : ℕ+) (h1 : a * x * y = b * y * z) (h2 : b * y * z = c * z * x) :
  ∃ (k m : ℕ+), a + b + c + x + y + z = k * m ∧ k ≠ 1 ∧ m ≠ 1 :=
sorry

end sum_not_prime_l980_98054


namespace quadrilateral_is_rectangle_l980_98052

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral in the plane -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Distance squared between two points -/
def distanceSquared (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- A quadrilateral is a rectangle if its diagonals bisect each other -/
def isRectangle (quad : Quadrilateral) : Prop :=
  let midpointAC := Point.mk ((quad.A.x + quad.C.x) / 2) ((quad.A.y + quad.C.y) / 2)
  let midpointBD := Point.mk ((quad.B.x + quad.D.x) / 2) ((quad.B.y + quad.D.y) / 2)
  midpointAC = midpointBD

/-- Main theorem -/
theorem quadrilateral_is_rectangle (quad : Quadrilateral) :
  (∀ M N P : Point, ¬collinear M N P →
    distanceSquared M quad.A + distanceSquared M quad.C =
    distanceSquared M quad.B + distanceSquared M quad.D) →
  isRectangle quad :=
sorry

end quadrilateral_is_rectangle_l980_98052


namespace fraction_meaningful_l980_98009

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 3) / (x + 5)) ↔ x ≠ -5 := by sorry

end fraction_meaningful_l980_98009


namespace intersection_of_A_and_B_l980_98076

def A : Set ℝ := {-1, 1, 2, 3, 4}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l980_98076


namespace cube_surface_area_l980_98096

/-- The surface area of a cube with edge length 20 cm is 2400 cm². -/
theorem cube_surface_area : 
  let edge_length : ℝ := 20
  let surface_area : ℝ := 6 * edge_length * edge_length
  surface_area = 2400 :=
by
  sorry

end cube_surface_area_l980_98096


namespace blind_students_count_l980_98088

theorem blind_students_count (total : ℕ) (deaf_ratio : ℕ) : 
  total = 180 → deaf_ratio = 3 → 
  ∃ (blind : ℕ), blind = 45 ∧ total = blind + deaf_ratio * blind :=
by
  sorry

end blind_students_count_l980_98088


namespace brown_beads_count_l980_98069

theorem brown_beads_count (green red taken_out left_in : ℕ) : 
  green = 1 → 
  red = 3 → 
  taken_out = 2 → 
  left_in = 4 → 
  green + red + (taken_out + left_in - (green + red)) = taken_out + left_in → 
  taken_out + left_in - (green + red) = 2 :=
by sorry

end brown_beads_count_l980_98069


namespace quadratic_solution_sum_l980_98034

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℝ, x^2 - 6*x + 15 = 51 ↔ x = p ∨ x = q) →
  p ≥ q →
  3*p + 2*q = 15 + 3*Real.sqrt 5 := by
sorry

end quadratic_solution_sum_l980_98034
