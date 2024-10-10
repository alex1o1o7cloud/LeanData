import Mathlib

namespace set_inclusion_condition_l990_99006

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 8 < 0}

def B : Set ℝ := {x : ℝ | x^2 + 2*x - 3 > 0}

def C (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 3*a*x + 2*a^2 < 0}

theorem set_inclusion_condition (a : ℝ) : 
  C a ⊆ A ∩ B ↔ (1 ≤ a ∧ a ≤ 2) ∨ a = 0 := by sorry

end set_inclusion_condition_l990_99006


namespace two_player_three_point_probability_l990_99057

/-- The probability that at least one of two players makes both of their two three-point shots -/
theorem two_player_three_point_probability (p_a p_b : ℝ) 
  (h_a : p_a = 0.4) (h_b : p_b = 0.5) : 
  1 - (1 - p_a^2) * (1 - p_b^2) = 0.37 := by
  sorry

end two_player_three_point_probability_l990_99057


namespace apple_orange_cost_l990_99073

/-- The cost of oranges and apples in two scenarios -/
theorem apple_orange_cost (orange_cost apple_cost : ℝ) : 
  orange_cost = 29 →
  apple_cost = 29 →
  6 * orange_cost + 8 * apple_cost = 419 →
  5 * orange_cost + 7 * apple_cost = 488 →
  8 = ⌊(419 - 6 * orange_cost) / apple_cost⌋ := by
  sorry

end apple_orange_cost_l990_99073


namespace quadratic_roots_identities_l990_99002

theorem quadratic_roots_identities (x₁ x₂ S P : ℝ) 
  (hS : S = x₁ + x₂) 
  (hP : P = x₁ * x₂) : 
  (x₁^2 + x₂^2 = S^2 - 2*P) ∧ 
  (x₁^3 + x₂^3 = S^3 - 3*S*P) := by
  sorry

end quadratic_roots_identities_l990_99002


namespace scalene_triangle_with_double_angle_and_36_degrees_l990_99041

theorem scalene_triangle_with_double_angle_and_36_degrees :
  ∀ (x y z : ℝ),
  0 < x ∧ 0 < y ∧ 0 < z →  -- angles are positive
  x < y ∧ y < z →  -- scalene triangle condition
  x + y + z = 180 →  -- sum of angles in a triangle
  (x = 36 ∨ y = 36 ∨ z = 36) →  -- one angle is 36°
  (x = 2*y ∨ y = 2*x ∨ y = 2*z ∨ z = 2*x ∨ z = 2*y) →  -- one angle is double another
  ((x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126)) := by
  sorry

end scalene_triangle_with_double_angle_and_36_degrees_l990_99041


namespace f_of_g_of_3_l990_99029

/-- Given two functions f and g, prove that f(g(3)) = 97 -/
theorem f_of_g_of_3 (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 3 * x^2 - 2 * x + 1) 
  (hg : ∀ x, g x = x + 3) : 
  f (g 3) = 97 := by
  sorry

end f_of_g_of_3_l990_99029


namespace team_a_win_probability_l990_99018

/-- The probability of Team A winning a non-fifth set -/
def p_regular : ℚ := 2/3

/-- The probability of Team A winning the fifth set -/
def p_fifth : ℚ := 1/2

/-- The probability of Team A winning the match -/
def p_win : ℚ := 20/27

/-- Theorem stating that the probability of Team A winning the match is 20/27 -/
theorem team_a_win_probability : 
  p_win = p_regular^3 + 
          3 * p_regular^2 * (1 - p_regular) * p_regular + 
          6 * p_regular^2 * (1 - p_regular)^2 * p_fifth := by
  sorry

#check team_a_win_probability

end team_a_win_probability_l990_99018


namespace sues_trail_mix_composition_sues_dried_fruit_percentage_proof_l990_99076

/-- The percentage of dried fruit in Sue's trail mix -/
def sues_dried_fruit_percentage : ℝ := 70

theorem sues_trail_mix_composition :
  sues_dried_fruit_percentage = 70 :=
by
  -- Proof goes here
  sorry

/-- Sue's trail mix nuts percentage -/
def sues_nuts_percentage : ℝ := 30

/-- Jane's trail mix nuts percentage -/
def janes_nuts_percentage : ℝ := 60

/-- Jane's trail mix chocolate chips percentage -/
def janes_chocolate_percentage : ℝ := 40

/-- Combined mixture nuts percentage -/
def combined_nuts_percentage : ℝ := 45

/-- Combined mixture dried fruit percentage -/
def combined_dried_fruit_percentage : ℝ := 35

/-- Sue's trail mix consists of only nuts and dried fruit -/
axiom sues_mix_composition :
  sues_nuts_percentage + sues_dried_fruit_percentage = 100

/-- The combined mixture percentages are consistent with individual mixes -/
axiom combined_mix_consistency (s j : ℝ) :
  s > 0 ∧ j > 0 →
  sues_nuts_percentage * s + janes_nuts_percentage * j = combined_nuts_percentage * (s + j) ∧
  sues_dried_fruit_percentage * s = combined_dried_fruit_percentage * (s + j)

theorem sues_dried_fruit_percentage_proof :
  sues_dried_fruit_percentage = 70 :=
by
  -- Proof goes here
  sorry

end sues_trail_mix_composition_sues_dried_fruit_percentage_proof_l990_99076


namespace susan_tuesday_candies_l990_99005

/-- Represents the number of candies Susan bought on each day -/
structure CandyPurchases where
  tuesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Represents Susan's candy consumption and remaining candies -/
structure CandyConsumption where
  eaten : ℕ
  remaining : ℕ

/-- Calculates the total number of candies Susan had -/
def totalCandies (purchases : CandyPurchases) (consumption : CandyConsumption) : ℕ :=
  purchases.tuesday + purchases.thursday + purchases.friday

/-- Theorem: Susan bought 3 candies on Tuesday -/
theorem susan_tuesday_candies (purchases : CandyPurchases) (consumption : CandyConsumption) :
  purchases.thursday = 5 →
  purchases.friday = 2 →
  consumption.eaten = 6 →
  consumption.remaining = 4 →
  totalCandies purchases consumption = consumption.eaten + consumption.remaining →
  purchases.tuesday = 3 := by
  sorry

end susan_tuesday_candies_l990_99005


namespace reflection_across_y_axis_l990_99050

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-3, 4)

theorem reflection_across_y_axis :
  reflect_y P = (3, 4) := by
  sorry

end reflection_across_y_axis_l990_99050


namespace age_ratio_after_15_years_l990_99064

/-- Represents the ages of a father and his children -/
structure FamilyAges where
  fatherAge : ℕ
  childrenAgesSum : ℕ

/-- Theorem about the ratio of ages after 15 years -/
theorem age_ratio_after_15_years (family : FamilyAges) 
  (h1 : family.fatherAge = family.childrenAgesSum)
  (h2 : family.fatherAge = 75) :
  (family.childrenAgesSum + 5 * 15) / (family.fatherAge + 15) = 5 / 3 := by
  sorry

end age_ratio_after_15_years_l990_99064


namespace certain_number_proof_l990_99090

theorem certain_number_proof (N : ℝ) : 
  (1/2)^22 * N^11 = 1/(18^22) → N = 81 := by
sorry

end certain_number_proof_l990_99090


namespace number_to_add_for_divisibility_l990_99061

theorem number_to_add_for_divisibility (n m k : ℕ) (h1 : n = 956734) (h2 : m = 412) (h3 : k = 390) :
  (n + k) % m = 0 := by
  sorry

end number_to_add_for_divisibility_l990_99061


namespace davids_daughter_age_l990_99082

/-- David's current age -/
def david_age : ℕ := 40

/-- Number of years in the future when David's age will be twice his daughter's -/
def years_until_double : ℕ := 16

/-- David's daughter's current age -/
def daughter_age : ℕ := 12

/-- Theorem stating that David's daughter is 12 years old today -/
theorem davids_daughter_age :
  daughter_age = 12 ∧
  david_age + years_until_double = 2 * (daughter_age + years_until_double) :=
by sorry

end davids_daughter_age_l990_99082


namespace cubic_equation_root_sum_l990_99015

theorem cubic_equation_root_sum (p q : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + p * (Complex.I * Real.sqrt 2 + 2) + q = 0 → 
  p + q = 14 := by
  sorry

end cubic_equation_root_sum_l990_99015


namespace unit_digit_of_product_is_zero_l990_99016

/-- Get the unit digit of a natural number -/
def unitDigit (n : ℕ) : ℕ := n % 10

/-- The product of the given numbers -/
def productOfNumbers : ℕ := 785846 * 1086432 * 4582735 * 9783284 * 5167953 * 3821759 * 7594683

theorem unit_digit_of_product_is_zero :
  unitDigit productOfNumbers = 0 := by
  sorry

end unit_digit_of_product_is_zero_l990_99016


namespace maoming_population_scientific_notation_l990_99027

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The population of Maoming city in millions -/
def maoming_population : ℝ := 6.8

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem maoming_population_scientific_notation :
  to_scientific_notation maoming_population = ScientificNotation.mk 6.8 6 sorry := by
  sorry

end maoming_population_scientific_notation_l990_99027


namespace quadratic_and_optimization_l990_99021

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, f a x > 0 ↔ (x < 1 ∨ x > b)

-- Define the constraint equation
def constraint (x y : ℝ) : Prop :=
  (1 / (x + 1)) + (2 / (y + 1)) = 1

-- Define the objective function
def objective (x y : ℝ) : ℝ := 2 * x + y + 3

-- State the theorem
theorem quadratic_and_optimization :
  ∃ a b : ℝ,
    solution_set a b ∧
    (a = 1 ∧ b = 2) ∧
    (∀ x y : ℝ, x > 0 → y > 0 → constraint x y →
      objective x y ≥ 8 ∧
      ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ constraint x₀ y₀ ∧ objective x₀ y₀ = 8) :=
sorry

end quadratic_and_optimization_l990_99021


namespace train_speed_l990_99087

/-- The speed of a train given its length, time to pass a person, and the person's speed -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (person_speed : ℝ) :
  train_length = 125 →
  passing_time = 6 →
  person_speed = 5 →
  ∃ (train_speed : ℝ), (abs (train_speed - 70) < 0.5 ∧
    train_speed * 1000 / 3600 + person_speed * 1000 / 3600 = train_length / passing_time) :=
by
  sorry

end train_speed_l990_99087


namespace train_passing_time_l990_99042

/-- Proves the time it takes for a train to pass a stationary point given its speed and time to cross a platform of known length -/
theorem train_passing_time (train_speed_kmph : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) : 
  train_speed_kmph = 72 → 
  platform_length = 260 → 
  platform_crossing_time = 30 → 
  (platform_length + (train_speed_kmph * 1000 / 3600 * platform_crossing_time)) / (train_speed_kmph * 1000 / 3600) = 17 := by
  sorry

end train_passing_time_l990_99042


namespace inequality_proof_l990_99019

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
  sorry

end inequality_proof_l990_99019


namespace units_digit_of_3_pow_7_pow_6_l990_99024

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 3^(7^6) is 3 -/
theorem units_digit_of_3_pow_7_pow_6 : unitsDigit (3^(7^6)) = 3 := by
  sorry

end units_digit_of_3_pow_7_pow_6_l990_99024


namespace business_investment_problem_l990_99040

theorem business_investment_problem (y_investment : ℕ) (total_profit : ℕ) (x_profit_share : ℕ) (x_investment : ℕ) :
  y_investment = 15000 →
  total_profit = 1600 →
  x_profit_share = 400 →
  x_profit_share * y_investment = (total_profit - x_profit_share) * x_investment →
  x_investment = 5000 := by
sorry

end business_investment_problem_l990_99040


namespace remainder_71_cubed_73_fifth_mod_8_l990_99053

theorem remainder_71_cubed_73_fifth_mod_8 : (71^3 * 73^5) % 8 = 7 := by
  sorry

end remainder_71_cubed_73_fifth_mod_8_l990_99053


namespace sandy_has_24_red_balloons_l990_99077

/-- The number of red balloons Sandy has -/
def sandys_red_balloons (saras_red_balloons total_red_balloons : ℕ) : ℕ :=
  total_red_balloons - saras_red_balloons

/-- Theorem stating that Sandy has 24 red balloons -/
theorem sandy_has_24_red_balloons :
  sandys_red_balloons 31 55 = 24 := by
  sorry

end sandy_has_24_red_balloons_l990_99077


namespace quadratic_sum_l990_99007

/-- For the quadratic expression 4x^2 - 8x + 1, when expressed in the form a(x-h)^2 + k,
    the sum of a, h, and k equals 2. -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a h k : ℝ), (4 * x^2 - 8 * x + 1 = a * (x - h)^2 + k) ∧ (a + h + k = 2) :=
by sorry

end quadratic_sum_l990_99007


namespace cubic_integer_root_l990_99083

theorem cubic_integer_root (p q : ℤ) : 
  (∃ (x : ℝ), x^3 - p*x - q = 0 ∧ x = 4 - Real.sqrt 10) →
  ((-8 : ℝ)^3 - p*(-8) - q = 0) :=
sorry

end cubic_integer_root_l990_99083


namespace min_p_plus_q_min_p_plus_q_value_l990_99072

theorem min_p_plus_q (p q : ℕ+) (h : 90 * p = q^3) : 
  ∀ (p' q' : ℕ+), 90 * p' = q'^3 → p + q ≤ p' + q' :=
by sorry

theorem min_p_plus_q_value (p q : ℕ+) (h : 90 * p = q^3) 
  (h_min : ∀ (p' q' : ℕ+), 90 * p' = q'^3 → p + q ≤ p' + q') : 
  p + q = 330 :=
by sorry

end min_p_plus_q_min_p_plus_q_value_l990_99072


namespace geometric_sequence_properties_l990_99011

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_properties (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 4)
  (h_sum : a 1 + a 2 + a 3 = 14) :
  (∀ n : ℕ, a n = 2^n) ∧
  (∀ m n p : ℕ, m < n → n < p → a m + a p ≠ 2 * a n) :=
sorry

end geometric_sequence_properties_l990_99011


namespace sqrt_equality_implies_one_two_l990_99092

theorem sqrt_equality_implies_one_two :
  ∀ a b : ℕ+,
  a < b →
  (Real.sqrt (1 + Real.sqrt (18 + 8 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
  (a = 1 ∧ b = 2) :=
by
  sorry

end sqrt_equality_implies_one_two_l990_99092


namespace smallest_sum_of_squares_cube_l990_99030

theorem smallest_sum_of_squares_cube (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 →
  x ≠ y → y ≠ z → x ≠ z →
  x^2 + y^2 = z^3 →
  ∀ a b c : ℕ, a > 0 → b > 0 → c > 0 → 
    a ≠ b → b ≠ c → a ≠ c →
    a^2 + b^2 = c^3 →
    x + y + z ≤ a + b + c →
  x + y + z = 18 :=
sorry

end smallest_sum_of_squares_cube_l990_99030


namespace modulus_of_complex_fraction_l990_99091

noncomputable def i : ℂ := Complex.I

theorem modulus_of_complex_fraction :
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 :=
by sorry

end modulus_of_complex_fraction_l990_99091


namespace lisa_book_purchase_l990_99026

theorem lisa_book_purchase (total_volumes : ℕ) (standard_cost deluxe_cost total_cost : ℕ) 
  (h1 : total_volumes = 15)
  (h2 : standard_cost = 20)
  (h3 : deluxe_cost = 30)
  (h4 : total_cost = 390) :
  ∃ (deluxe_count : ℕ), 
    deluxe_count * deluxe_cost + (total_volumes - deluxe_count) * standard_cost = total_cost ∧
    deluxe_count = 9 := by
  sorry

end lisa_book_purchase_l990_99026


namespace latest_time_60_degrees_l990_99014

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 10*t + 40

-- State the theorem
theorem latest_time_60_degrees :
  ∃ t : ℝ, t ≤ 12 ∧ t ≥ 0 ∧ temperature t = 60 ∧
  ∀ s : ℝ, s > t ∧ s ≥ 0 → temperature s ≠ 60 :=
by sorry

end latest_time_60_degrees_l990_99014


namespace average_weight_problem_l990_99046

/-- Given three weights a, b, and c, prove that their average weights satisfy the given conditions and the average weight of b and c is 43. -/
theorem average_weight_problem (a b c : ℝ) : 
  (a + b + c) / 3 = 45 ∧ 
  (a + b) / 2 = 40 ∧ 
  b = 31 → 
  (b + c) / 2 = 43 := by
sorry

end average_weight_problem_l990_99046


namespace range_of_a_l990_99051

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Iic 6, StrictMonoOn (f a) (Set.Iic x)) →
  a ∈ Set.Iic (-5) :=
by sorry

end range_of_a_l990_99051


namespace root_ratio_implies_k_value_l990_99089

theorem root_ratio_implies_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y ∧
   x^2 + 10*x + k = 0 ∧ 
   y^2 + 10*y + k = 0 ∧
   x / y = 3) →
  k = 18.75 := by
sorry

end root_ratio_implies_k_value_l990_99089


namespace cube_side_length_l990_99009

theorem cube_side_length (C R T : ℝ) (h1 : C = 36.50) (h2 : R = 16) (h3 : T = 876) :
  ∃ L : ℝ, L = 8 ∧ T = (6 * L^2) * (C / R) :=
sorry

end cube_side_length_l990_99009


namespace sum_of_squares_l990_99094

theorem sum_of_squares (a b c : ℝ) : 
  a * b + b * c + a * c = 131 →
  a + b + c = 22 →
  a^2 + b^2 + c^2 = 222 := by
sorry

end sum_of_squares_l990_99094


namespace delta_u_zero_l990_99093

def u (n : ℕ) : ℤ := n^3 - n

def delta (k : ℕ) : (ℕ → ℤ) → (ℕ → ℤ) :=
  match k with
  | 0 => id
  | k+1 => fun f n => f (n+1) - f n

theorem delta_u_zero (k : ℕ) :
  (∀ n, delta k u n = 0) ↔ k ≥ 4 :=
sorry

end delta_u_zero_l990_99093


namespace marys_oranges_l990_99079

theorem marys_oranges :
  ∀ (oranges : ℕ),
    (14 + oranges + 6 - 3 = 26) →
    oranges = 9 :=
by
  sorry

end marys_oranges_l990_99079


namespace polynomial_root_behavior_l990_99013

def Q (x : ℝ) : ℝ := x^6 - 6*x^5 + 10*x^4 - x^3 - x + 12

theorem polynomial_root_behavior :
  (∀ x < 0, Q x ≠ 0) ∧ (∃ x > 0, Q x = 0) := by
  sorry

end polynomial_root_behavior_l990_99013


namespace expression_equals_one_l990_99022

theorem expression_equals_one (x : ℝ) 
  (h1 : x^4 + 2*x + 2 ≠ 0) 
  (h2 : x^4 - 2*x + 2 ≠ 0) : 
  ((((x+2)^3 * (x^3-2*x+2)^3) / (x^4+2*x+2)^3)^3 * 
   (((x-2)^3 * (x^3+2*x+2)^3) / (x^4-2*x+2)^3)^3) = 1 := by
  sorry

end expression_equals_one_l990_99022


namespace parallel_lines_a_values_l990_99055

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 : ℝ} : 
  (∃ b1 b2 : ℝ, ∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_a_values (a : ℝ) :
  (∃ x y : ℝ, y = a * x - 2 ∧ 3 * x - (a + 2) * y + 1 = 0) →
  (∀ x y : ℝ, y = a * x - 2 ↔ 3 * x - (a + 2) * y + 1 = 0) →
  a = 1 ∨ a = -3 := by
  sorry

end parallel_lines_a_values_l990_99055


namespace sin_thirty_degrees_l990_99049

/-- Sine of 30 degrees is 1/2 -/
theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_thirty_degrees_l990_99049


namespace central_angle_values_l990_99088

/-- A circular sector with perimeter p and area a -/
structure CircularSector where
  p : ℝ  -- perimeter
  a : ℝ  -- area
  h_p_pos : p > 0
  h_a_pos : a > 0

/-- The central angle (in radians) of a circular sector -/
def central_angle (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, r > 0 ∧ 2 * r + r * θ = s.p ∧ 1/2 * r^2 * θ = s.a}

/-- Theorem: For a circular sector with perimeter 6 and area 2, 
    the central angle is either 1 or 4 radians -/
theorem central_angle_values (s : CircularSector) 
  (h_p : s.p = 6) (h_a : s.a = 2) : 
  central_angle s = {1, 4} := by sorry

end central_angle_values_l990_99088


namespace shaded_fraction_is_one_fourth_l990_99054

/-- Represents a square board with shaded regions -/
structure Board :=
  (size : ℕ)
  (shaded_area : ℚ)

/-- Calculates the fraction of shaded area on the board -/
def shaded_fraction (b : Board) : ℚ :=
  b.shaded_area / (b.size * b.size : ℚ)

/-- Represents the specific board configuration described in the problem -/
def problem_board : Board :=
  { size := 4,
    shaded_area := 4 }

/-- Theorem stating that the shaded fraction of the problem board is 1/4 -/
theorem shaded_fraction_is_one_fourth :
  shaded_fraction problem_board = 1/4 := by
  sorry

#check shaded_fraction_is_one_fourth

end shaded_fraction_is_one_fourth_l990_99054


namespace quadratic_minimum_l990_99036

/-- The quadratic function f(c) = 3/4*c^2 - 6c + 4 is minimized when c = 4 -/
theorem quadratic_minimum : ∃ (c : ℝ), ∀ (x : ℝ), (3/4 : ℝ) * c^2 - 6*c + 4 ≤ (3/4 : ℝ) * x^2 - 6*x + 4 := by
  sorry

end quadratic_minimum_l990_99036


namespace event_probability_l990_99043

theorem event_probability (P_A_and_B P_A_or_B P_B : ℝ) 
  (h1 : P_A_and_B = 0.25)
  (h2 : P_A_or_B = 0.8)
  (h3 : P_B = 0.65) :
  ∃ P_A : ℝ, P_A = 0.4 ∧ P_A_or_B = P_A + P_B - P_A_and_B := by
  sorry

end event_probability_l990_99043


namespace isabel_math_pages_l990_99069

/-- The number of pages of math homework Isabel had -/
def math_pages : ℕ := sorry

/-- The number of pages of reading homework Isabel had -/
def reading_pages : ℕ := 4

/-- The number of problems per page -/
def problems_per_page : ℕ := 5

/-- The total number of problems Isabel had to complete -/
def total_problems : ℕ := 30

/-- Theorem stating that Isabel had 2 pages of math homework -/
theorem isabel_math_pages : math_pages = 2 := by
  sorry

end isabel_math_pages_l990_99069


namespace grass_eating_problem_l990_99086

/-- Amount of grass on one hectare initially -/
def initial_grass : ℝ := sorry

/-- Amount of grass that regrows on one hectare in one week -/
def grass_regrowth : ℝ := sorry

/-- Amount of grass one cow eats in one week -/
def cow_consumption : ℝ := sorry

/-- Number of cows that eat all grass on given hectares in given weeks -/
def cows_needed (hectares weeks : ℕ) : ℕ := sorry

theorem grass_eating_problem :
  (3 : ℕ) * cow_consumption * 2 = 2 * initial_grass + 4 * grass_regrowth ∧
  (2 : ℕ) * cow_consumption * 4 = 2 * initial_grass + 8 * grass_regrowth →
  cows_needed 6 6 = 5 := by sorry

end grass_eating_problem_l990_99086


namespace fourth_seat_is_19_l990_99063

/-- Represents a systematic sampling of students from a class. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Fin 3 → ℕ
  hclass_size : class_size = 52
  hsample_size : sample_size = 4
  hknown_seats : known_seats = ![6, 32, 45]

/-- The step size in systematic sampling -/
def step_size (s : SystematicSample) : ℕ := s.class_size / s.sample_size

/-- The first seat number in the systematic sample -/
def first_seat (s : SystematicSample) : ℕ := 19

/-- Theorem stating that the fourth seat in the systematic sample is 19 -/
theorem fourth_seat_is_19 (s : SystematicSample) : first_seat s = 19 := by
  sorry

end fourth_seat_is_19_l990_99063


namespace tea_store_theorem_l990_99071

/-- Represents the number of ways to buy items from a tea store. -/
def teaStoreCombinations (cups saucers spoons : ℕ) : ℕ :=
  let cupSaucer := cups * saucers
  let cupSpoon := cups * spoons
  let saucerSpoon := saucers * spoons
  let all := cups * saucers * spoons
  cups + saucers + spoons + cupSaucer + cupSpoon + saucerSpoon + all

/-- Theorem stating the total number of combinations for a specific tea store inventory. -/
theorem tea_store_theorem :
  teaStoreCombinations 5 3 4 = 119 := by
  sorry

end tea_store_theorem_l990_99071


namespace circle_intersection_radius_range_l990_99096

theorem circle_intersection_radius_range (r : ℝ) : 
  r > 0 ∧ 
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x + 3)^2 + (y - 4)^2 = 36) → 
  1 < r ∧ r < 11 := by
sorry

end circle_intersection_radius_range_l990_99096


namespace order_of_zeros_and_roots_l990_99058

def f (x m n : ℝ) : ℝ := 2 * (x - m) * (x - n) - 7

theorem order_of_zeros_and_roots (m n α β : ℝ) 
  (h1 : m < n) 
  (h2 : α < β) 
  (h3 : f α m n = 0)
  (h4 : f β m n = 0) :
  α < m ∧ m < n ∧ n < β := by sorry

end order_of_zeros_and_roots_l990_99058


namespace price_increase_l990_99032

theorem price_increase (x : ℝ) : 
  (1 + x / 100) * (1 + 30 / 100) = 1 + 62.5 / 100 → x = 25 := by
  sorry

end price_increase_l990_99032


namespace a_equals_plus_minus_two_l990_99025

-- Define the sets A and B
def A : Set ℝ := {0, 2}
def B (a : ℝ) : Set ℝ := {1, a^2}

-- Define the theorem
theorem a_equals_plus_minus_two (a : ℝ) : 
  A ∪ B a = {0, 1, 2, 4} → a = 2 ∨ a = -2 := by
  sorry

end a_equals_plus_minus_two_l990_99025


namespace pressure_volume_relation_l990_99033

-- Define the constants for the problem
def initial_pressure : ℝ := 8
def initial_volume : ℝ := 3
def final_volume : ℝ := 6

-- Define the theorem
theorem pressure_volume_relation :
  ∀ (p1 p2 v1 v2 : ℝ),
    p1 > 0 → p2 > 0 → v1 > 0 → v2 > 0 →
    p1 = initial_pressure →
    v1 = initial_volume →
    v2 = final_volume →
    (p1 * v1 = p2 * v2) →
    p2 = 4 := by
  sorry

end pressure_volume_relation_l990_99033


namespace at_least_one_red_not_basic_event_l990_99075

structure Ball := (color : String)

def bag : Multiset Ball := 
  2 • {Ball.mk "red"} + 2 • {Ball.mk "white"} + 2 • {Ball.mk "black"}

def is_basic_event (event : Set (Ball × Ball)) : Prop :=
  ∃ (b1 b2 : Ball), event = {(b1, b2)}

def at_least_one_red (pair : Ball × Ball) : Prop :=
  (pair.1.color = "red") ∨ (pair.2.color = "red")

theorem at_least_one_red_not_basic_event :
  ¬ (is_basic_event {pair | at_least_one_red pair}) :=
sorry

end at_least_one_red_not_basic_event_l990_99075


namespace inequality_proof_l990_99060

theorem inequality_proof (a d b c : ℝ) 
  (h1 : a ≥ 0) (h2 : d ≥ 0) (h3 : b > 0) (h4 : c > 0) (h5 : b + c ≥ a + d) : 
  (b / (c + d)) + (c / (b + a)) ≥ Real.sqrt 2 - 1/2 := by
  sorry

end inequality_proof_l990_99060


namespace random_event_last_third_probability_l990_99037

/-- The probability of a random event occurring in the last third of a given time interval is 1/3 -/
theorem random_event_last_third_probability (total_interval : ℝ) (h : total_interval > 0) :
  let last_third := total_interval / 3
  (last_third / total_interval) = 1 / 3 := by
sorry

end random_event_last_third_probability_l990_99037


namespace product_of_solutions_l990_99012

theorem product_of_solutions (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 1905) (h₂ : y₁^3 - 3*x₁^2*y₁ = 1910)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 1905) (h₄ : y₂^3 - 3*x₂^2*y₂ = 1910)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 1905) (h₆ : y₃^3 - 3*x₃^2*y₃ = 1910) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = -1/191 := by
  sorry

end product_of_solutions_l990_99012


namespace sufficient_not_necessary_condition_l990_99044

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + m

theorem sufficient_not_necessary_condition (m : ℝ) :
  (m < 1 → ∃ x, f m x = 0) ∧
  ¬(∀ m, (∃ x, f m x = 0) → m < 1) :=
sorry

end sufficient_not_necessary_condition_l990_99044


namespace spherical_coordinate_conversion_l990_99031

theorem spherical_coordinate_conversion (ρ θ φ : Real) :
  ρ = 5 ∧ θ = 5 * Real.pi / 7 ∧ φ = 11 * Real.pi / 6 →
  ∃ (ρ' θ' φ' : Real),
    ρ' > 0 ∧
    0 ≤ θ' ∧ θ' < 2 * Real.pi ∧
    0 ≤ φ' ∧ φ' ≤ Real.pi ∧
    ρ' = 5 ∧
    θ' = 12 * Real.pi / 7 ∧
    φ' = Real.pi / 6 :=
by sorry

end spherical_coordinate_conversion_l990_99031


namespace local_language_letters_l990_99001

theorem local_language_letters (n : ℕ) : 
  (n + n^2) - ((n - 1) + (n - 1)^2) = 139 → n = 69 := by
  sorry

end local_language_letters_l990_99001


namespace min_value_of_expression_l990_99000

theorem min_value_of_expression (x y : ℝ) : 
  x^2 + 4*x*y + 5*y^2 - 8*x - 4*y + x^3 ≥ -11.9 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀^2 + 4*x₀*y₀ + 5*y₀^2 - 8*x₀ - 4*y₀ + x₀^3 = -11.9 := by
  sorry

end min_value_of_expression_l990_99000


namespace hotel_payment_ratio_l990_99035

/-- Given a hotel with operations expenses and a loss, compute the ratio of total payments to operations cost -/
theorem hotel_payment_ratio (operations_cost loss : ℚ) 
  (h1 : operations_cost = 100)
  (h2 : loss = 25) :
  (operations_cost - loss) / operations_cost = 3 / 4 := by
  sorry

end hotel_payment_ratio_l990_99035


namespace teddy_cats_count_l990_99068

/-- Prove that Teddy has 8 cats given the conditions of the problem -/
theorem teddy_cats_count :
  -- Teddy's dogs
  let teddy_dogs : ℕ := 7
  -- Ben's dogs relative to Teddy's
  let ben_dogs : ℕ := teddy_dogs + 9
  -- Dave's dogs relative to Teddy's
  let dave_dogs : ℕ := teddy_dogs - 5
  -- Dave's cats relative to Teddy's
  let dave_cats (teddy_cats : ℕ) : ℕ := teddy_cats + 13
  -- Total pets
  let total_pets : ℕ := 54
  -- The number of Teddy's cats that satisfies all conditions
  ∃ (teddy_cats : ℕ),
    teddy_dogs + ben_dogs + dave_dogs + teddy_cats + dave_cats teddy_cats = total_pets ∧
    teddy_cats = 8 := by
  sorry

end teddy_cats_count_l990_99068


namespace function_range_l990_99099

theorem function_range (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = a + b * Real.cos x + c * Real.sin x) →
  f 0 = 1 →
  f (-π/4) = a →
  (∀ x ∈ Set.Icc 0 (π/2), |f x| ≤ Real.sqrt 2) →
  a ∈ Set.Icc 0 (4 + 2 * Real.sqrt 2) :=
sorry

end function_range_l990_99099


namespace quotient_problem_l990_99067

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
    (h1 : dividend = 166)
    (h2 : divisor = 20)
    (h3 : remainder = 6)
    (h4 : dividend = divisor * quotient + remainder) :
  quotient = 8 := by
  sorry

end quotient_problem_l990_99067


namespace yasmin_bank_account_l990_99098

/-- Yasmin's bank account problem -/
theorem yasmin_bank_account (deposit : ℕ) (new_balance : ℕ) (initial_balance : ℕ) : 
  deposit = 50 →
  4 * deposit = new_balance →
  initial_balance = new_balance - deposit →
  initial_balance = 150 := by
  sorry

end yasmin_bank_account_l990_99098


namespace class_outing_minimum_fee_l990_99003

/-- Calculates the minimum rental fee for a class outing --/
def minimum_rental_fee (total_students : ℕ) (small_boat_capacity : ℕ) (small_boat_cost : ℕ) 
  (large_boat_capacity : ℕ) (large_boat_cost : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum rental fee for the given conditions --/
theorem class_outing_minimum_fee : 
  minimum_rental_fee 48 3 16 5 24 = 232 := by
  sorry

end class_outing_minimum_fee_l990_99003


namespace expected_heads_is_75_l990_99097

/-- The number of coins -/
def num_coins : ℕ := 80

/-- The maximum number of flips per coin -/
def max_flips : ℕ := 4

/-- The probability of getting heads on a single flip of a fair coin -/
def p_heads : ℚ := 1/2

/-- The probability of getting heads at least once in up to four flips -/
def p_heads_in_four_flips : ℚ := 1 - (1 - p_heads)^max_flips

/-- The expected number of coins showing heads after all tosses -/
def expected_heads : ℚ := num_coins * p_heads_in_four_flips

theorem expected_heads_is_75 : expected_heads = 75 := by sorry

end expected_heads_is_75_l990_99097


namespace robins_hair_length_l990_99020

theorem robins_hair_length (initial_length cut_length growth_length final_length : ℕ) :
  cut_length = 11 →
  growth_length = 12 →
  final_length = 17 →
  final_length = initial_length - cut_length + growth_length →
  initial_length = 16 :=
by sorry

end robins_hair_length_l990_99020


namespace no_valid_reassignment_l990_99074

/-- Represents a seating arrangement in a classroom -/
structure Classroom :=
  (rows : Nat)
  (cols : Nat)
  (students : Nat)
  (center_empty : Bool)

/-- Checks if a reassignment is possible given the classroom setup -/
def reassignment_possible (c : Classroom) : Prop :=
  c.rows = 5 ∧ c.cols = 7 ∧ c.students = 34 ∧ c.center_empty = true →
  ∃ (new_arrangement : Fin c.students → Fin (c.rows * c.cols)),
    ∀ i : Fin c.students,
      let old_pos := i.val
      let new_pos := (new_arrangement i).val
      (new_pos ≠ old_pos) ∧
      ((new_pos = old_pos + 1 ∨ new_pos = old_pos - 1) ∨
       (new_pos = old_pos + c.cols ∨ new_pos = old_pos - c.cols))

theorem no_valid_reassignment (c : Classroom) :
  ¬(reassignment_possible c) :=
sorry

end no_valid_reassignment_l990_99074


namespace class_average_weight_l990_99078

theorem class_average_weight (n1 : ℕ) (n2 : ℕ) (w1 : ℝ) (w2 : ℝ) (h1 : n1 = 22) (h2 : n2 = 8) (h3 : w1 = 50.25) (h4 : w2 = 45.15) :
  (n1 * w1 + n2 * w2) / (n1 + n2) = 48.89 := by
  sorry

end class_average_weight_l990_99078


namespace sum_first_third_is_five_l990_99008

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  second_term : a 2 = 2
  inverse_sum : 1 / a 1 + 1 / a 3 = 5 / 4

/-- The sum of the first and third terms of the geometric sequence is 5 -/
theorem sum_first_third_is_five (seq : GeometricSequence) : seq.a 1 + seq.a 3 = 5 := by
  sorry

end sum_first_third_is_five_l990_99008


namespace profit_increase_l990_99034

theorem profit_increase (march_profit : ℝ) (h1 : march_profit > 0) : 
  let april_profit := 1.20 * march_profit
  let may_profit := 0.80 * april_profit
  let june_profit := 1.50 * may_profit
  (june_profit - march_profit) / march_profit * 100 = 44 := by
sorry

end profit_increase_l990_99034


namespace prob_three_odd_less_than_eighth_l990_99010

def total_integers : ℕ := 2020
def odd_integers : ℕ := total_integers / 2

theorem prob_three_odd_less_than_eighth :
  let p := (odd_integers : ℚ) / total_integers *
           ((odd_integers - 1) : ℚ) / (total_integers - 1) *
           ((odd_integers - 2) : ℚ) / (total_integers - 2)
  p < 1 / 8 := by sorry

end prob_three_odd_less_than_eighth_l990_99010


namespace quadratic_is_square_of_binomial_l990_99085

theorem quadratic_is_square_of_binomial :
  ∃ (r s : ℚ), (81/16 : ℚ) * x^2 + 18 * x + 16 = (r * x + s)^2 := by
  sorry

end quadratic_is_square_of_binomial_l990_99085


namespace xyz_sum_reciprocal_l990_99084

theorem xyz_sum_reciprocal (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_sum_x : x + 1 / z = 4)
  (h_sum_y : y + 1 / x = 30) :
  z + 1 / y = 36 / 119 := by
sorry

end xyz_sum_reciprocal_l990_99084


namespace min_value_of_expression_l990_99047

theorem min_value_of_expression (x : ℝ) :
  ∃ (min_val : ℝ), min_val = -6480.25 ∧
  ∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) ≥ min_val :=
sorry

end min_value_of_expression_l990_99047


namespace matrix_commute_l990_99070

theorem matrix_commute (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = !![5, 1; -2, 4]) : 
  D * C = !![5, 1; -2, 4] := by sorry

end matrix_commute_l990_99070


namespace five_zero_points_l990_99065

open Set
open Real

noncomputable def f (x : ℝ) := Real.sin (π * Real.cos x)

theorem five_zero_points :
  ∃! (s : Finset ℝ), s.card = 5 ∧ 
  (∀ x ∈ s, x ∈ Icc 0 (2 * π) ∧ f x = 0) ∧
  (∀ x ∈ Icc 0 (2 * π), f x = 0 → x ∈ s) := by
  sorry

end five_zero_points_l990_99065


namespace initial_mean_equals_correct_mean_l990_99059

/-- Proves that the initial mean is equal to the correct mean when one value is incorrectly copied --/
theorem initial_mean_equals_correct_mean (n : ℕ) (correct_value incorrect_value : ℝ) (correct_mean : ℝ) :
  n = 25 →
  correct_value = 165 →
  incorrect_value = 130 →
  correct_mean = 191.4 →
  (n * correct_mean - correct_value + incorrect_value) / n = correct_mean := by
  sorry

#check initial_mean_equals_correct_mean

end initial_mean_equals_correct_mean_l990_99059


namespace distance_AB_l990_99048

-- Define the points and distances
structure Points where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

-- Define the speeds
structure Speeds where
  vA : ℝ
  vB : ℝ

-- Define the problem conditions
structure Conditions where
  points : Points
  speeds : Speeds
  CD_distance : ℝ
  B_remaining_distance : ℝ
  speed_ratio : ℝ
  speed_reduction : ℝ

-- Theorem statement
theorem distance_AB (c : Conditions) : 
  c.CD_distance = 900 ∧ 
  c.B_remaining_distance = 720 ∧ 
  c.speed_ratio = 5/4 ∧ 
  c.speed_reduction = 4/5 →
  c.points.B - c.points.A = 5265 := by
  sorry


end distance_AB_l990_99048


namespace min_value_of_sum_of_roots_l990_99017

theorem min_value_of_sum_of_roots (x y : ℝ) :
  let z := Real.sqrt (x^2 + y^2 - 2*x - 2*y + 2) + Real.sqrt (x^2 + y^2 - 4*y + 4)
  z ≥ Real.sqrt 2 ∧
  (z = Real.sqrt 2 ↔ y = 2 - x ∧ 1 ≤ x ∧ x ≤ 2) := by
  sorry

end min_value_of_sum_of_roots_l990_99017


namespace third_group_data_points_l990_99095

/-- Given a sample divided into 5 groups with specific conditions, prove the number of data points in the third group --/
theorem third_group_data_points
  (total_groups : ℕ)
  (group_123_sum : ℕ)
  (group_345_sum : ℕ)
  (group_3_frequency : ℚ)
  (h1 : total_groups = 5)
  (h2 : group_123_sum = 160)
  (h3 : group_345_sum = 260)
  (h4 : group_3_frequency = 1/5) :
  ∃ (group_3 : ℕ), 
    group_3 = 70 ∧ 
    (group_3 : ℚ) / (group_123_sum + group_345_sum - group_3) = group_3_frequency :=
by sorry

end third_group_data_points_l990_99095


namespace cubic_sum_equals_zero_l990_99028

theorem cubic_sum_equals_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 - 2*(a + b + c) + 3 = 0 →
  a^3 + b^3 + c^3 - 3*a*b*c = 0 := by
  sorry

end cubic_sum_equals_zero_l990_99028


namespace robie_second_purchase_l990_99056

/-- The number of bags of chocolates Robie bought the second time -/
def second_purchase (initial : ℕ) (given_away : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_away)

/-- Theorem: Robie bought 3 bags of chocolates the second time -/
theorem robie_second_purchase :
  second_purchase 3 2 4 = 3 := by
  sorry

end robie_second_purchase_l990_99056


namespace conic_section_eccentricity_l990_99023

/-- Given a conic section with equation x²/m + y² = 1 and eccentricity √7, prove that m = -6 -/
theorem conic_section_eccentricity (m : ℝ) : 
  (∃ (x y : ℝ), x^2/m + y^2 = 1) →  -- Condition 1: Conic section equation
  (∃ (e : ℝ), e = Real.sqrt 7 ∧ e^2 = (1 - m)/1) →  -- Condition 2: Eccentricity
  m = -6 := by sorry

end conic_section_eccentricity_l990_99023


namespace min_value_abc_l990_99052

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 9*a + 4*b = a*b*c) : 
  a + b + c ≥ 10 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    9*a₀ + 4*b₀ = a₀*b₀*c₀ ∧ a₀ + b₀ + c₀ = 10 := by
  sorry

end min_value_abc_l990_99052


namespace heat_bulls_difference_l990_99080

/-- The number of games won by the Chicago Bulls -/
def bulls_games : ℕ := 70

/-- The total number of games won by both the Chicago Bulls and the Miami Heat -/
def total_games : ℕ := 145

/-- The number of games won by the Miami Heat -/
def heat_games : ℕ := total_games - bulls_games

/-- Theorem stating the difference in games won between the Miami Heat and the Chicago Bulls -/
theorem heat_bulls_difference : heat_games - bulls_games = 5 := by
  sorry

end heat_bulls_difference_l990_99080


namespace range_of_m_symmetrical_circle_equation_existence_of_m_for_circle_through_origin_l990_99081

-- Define the circle C and line l
def circle_C (x y m : ℝ) : Prop := x^2 + y^2 + x - 6*y + m = 0
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem 1: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x y : ℝ, circle_C x y m) → m < 37/4 :=
sorry

-- Theorem 2: Equation of symmetrical circle
theorem symmetrical_circle_equation :
  ∀ m : ℝ, (∃ x y : ℝ, circle_C x y m ∧ line_l x y) →
  (∀ x y : ℝ, x^2 + (y - 7/2)^2 = 1/8) :=
sorry

-- Theorem 3: Existence of m for circle through origin
theorem existence_of_m_for_circle_through_origin :
  ∃ m : ℝ, m = -3/2 ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    circle_C x₁ y₁ m ∧ circle_C x₂ y₂ m ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∃ a b r : ℝ, (x₁ - a)^2 + (y₁ - b)^2 = r^2 ∧
                  (x₂ - a)^2 + (y₂ - b)^2 = r^2 ∧
                  a^2 + b^2 = r^2)) :=
sorry

end range_of_m_symmetrical_circle_equation_existence_of_m_for_circle_through_origin_l990_99081


namespace planes_parallel_if_infinitely_many_parallel_lines_l990_99039

-- Define the concept of a plane in 3D space
variable (α β : Set (ℝ × ℝ × ℝ))

-- Define what it means for a line to be parallel to a plane
def LineParallelToPlane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for two planes to be parallel
def PlanesParallel (p1 p2 : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define the concept of infinitely many lines in a plane
def InfinitelyManyParallelLines (p1 p2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∃ (S : Set (Set (ℝ × ℝ × ℝ))), Infinite S ∧ (∀ l ∈ S, l ⊆ p1 ∧ LineParallelToPlane l p2)

-- State the theorem
theorem planes_parallel_if_infinitely_many_parallel_lines (α β : Set (ℝ × ℝ × ℝ)) :
  InfinitelyManyParallelLines α β → PlanesParallel α β := by sorry

end planes_parallel_if_infinitely_many_parallel_lines_l990_99039


namespace shortest_distance_between_circles_l990_99062

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : ∃ (d : ℝ),
  let circle1 := {(x, y) : ℝ × ℝ | x^2 - 6*x + y^2 + 10*y + 9 = 0}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + 8*x + y^2 - 2*y + 16 = 0}
  d = Real.sqrt 85 - 6 ∧
  ∀ (p1 : ℝ × ℝ) (p2 : ℝ × ℝ),
    p1 ∈ circle1 → p2 ∈ circle2 →
    d ≤ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) :=
by sorry

end shortest_distance_between_circles_l990_99062


namespace unicorn_journey_flowers_l990_99066

/-- The number of flowers that bloom when unicorns walk across a forest -/
def flowers_bloomed (num_unicorns : ℕ) (distance_km : ℕ) (step_length_m : ℕ) (flowers_per_step : ℕ) : ℕ :=
  num_unicorns * (distance_km * 1000 / step_length_m) * flowers_per_step

/-- Proof that 6 unicorns walking 9 km with 3m steps, each causing 4 flowers to bloom, results in 72000 flowers -/
theorem unicorn_journey_flowers : flowers_bloomed 6 9 3 4 = 72000 := by
  sorry

#eval flowers_bloomed 6 9 3 4

end unicorn_journey_flowers_l990_99066


namespace cyclist_distance_difference_l990_99045

/-- Represents a cyclist with a constant speed --/
structure Cyclist where
  speed : ℝ

/-- Calculates the distance traveled by a cyclist in a given time --/
def distance_traveled (cyclist : Cyclist) (time : ℝ) : ℝ :=
  cyclist.speed * time

/-- Theorem: The difference in distance traveled between two cyclists after 5 hours --/
theorem cyclist_distance_difference 
  (carlos dana : Cyclist)
  (h1 : carlos.speed = 0.9)
  (h2 : dana.speed = 0.72)
  : distance_traveled carlos 5 - distance_traveled dana 5 = 0.9 := by
  sorry

end cyclist_distance_difference_l990_99045


namespace five_digit_multiple_of_nine_l990_99038

theorem five_digit_multiple_of_nine : ∃ (n : ℕ), n = 45675 ∧ n % 9 = 0 := by
  sorry

end five_digit_multiple_of_nine_l990_99038


namespace nested_cube_root_simplification_l990_99004

theorem nested_cube_root_simplification (N : ℝ) (h : N > 1) :
  (4 * N * (8 * N * (12 * N)^(1/3))^(1/3))^(1/3) = 2 * 3^(1/3) * N^(13/27) := by
  sorry

end nested_cube_root_simplification_l990_99004
