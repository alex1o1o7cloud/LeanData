import Mathlib

namespace NUMINAMATH_GPT_calvin_gym_duration_l1831_183158

theorem calvin_gym_duration (initial_weight loss_per_month final_weight : ℕ) (h1 : initial_weight = 250)
    (h2 : loss_per_month = 8) (h3 : final_weight = 154) : 
    (initial_weight - final_weight) / loss_per_month = 12 :=
by 
  sorry

end NUMINAMATH_GPT_calvin_gym_duration_l1831_183158


namespace NUMINAMATH_GPT_soda_quantity_difference_l1831_183165

noncomputable def bottles_of_diet_soda := 19
noncomputable def bottles_of_regular_soda := 60
noncomputable def bottles_of_cherry_soda := 35
noncomputable def bottles_of_orange_soda := 45

theorem soda_quantity_difference : 
  (max bottles_of_regular_soda (max bottles_of_diet_soda 
    (max bottles_of_cherry_soda bottles_of_orange_soda)) 
  - min bottles_of_regular_soda (min bottles_of_diet_soda 
    (min bottles_of_cherry_soda bottles_of_orange_soda))) = 41 := 
by
  sorry

end NUMINAMATH_GPT_soda_quantity_difference_l1831_183165


namespace NUMINAMATH_GPT_linda_color_choices_l1831_183115

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def combination (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem linda_color_choices : combination 8 3 = 56 :=
  by sorry

end NUMINAMATH_GPT_linda_color_choices_l1831_183115


namespace NUMINAMATH_GPT_current_price_after_adjustment_l1831_183109

variable (x : ℝ) -- Define x, the original price per unit

theorem current_price_after_adjustment (x : ℝ) : (x + 10) * 0.75 = ((x + 10) * 0.75) :=
by
  sorry

end NUMINAMATH_GPT_current_price_after_adjustment_l1831_183109


namespace NUMINAMATH_GPT_X_investment_l1831_183172

theorem X_investment (P : ℝ) 
  (Y_investment : ℝ := 42000)
  (Z_investment : ℝ := 48000)
  (Z_joins_at : ℝ := 4)
  (total_profit : ℝ := 14300)
  (Z_share : ℝ := 4160) :
  (P * 12 / (P * 12 + Y_investment * 12 + Z_investment * (12 - Z_joins_at))) * total_profit = Z_share → P = 35700 :=
by
  sorry

end NUMINAMATH_GPT_X_investment_l1831_183172


namespace NUMINAMATH_GPT_hens_count_l1831_183184

theorem hens_count (H R : ℕ) (h₁ : H = 9 * R - 5) (h₂ : H + R = 75) : H = 67 :=
by {
  sorry
}

end NUMINAMATH_GPT_hens_count_l1831_183184


namespace NUMINAMATH_GPT_jane_purchased_pudding_l1831_183199

theorem jane_purchased_pudding (p : ℕ) 
  (ice_cream_cost_per_cone : ℕ := 5)
  (num_ice_cream_cones : ℕ := 15)
  (pudding_cost_per_cup : ℕ := 2)
  (cost_difference : ℕ := 65)
  (total_ice_cream_cost : ℕ := num_ice_cream_cones * ice_cream_cost_per_cone) 
  (total_pudding_cost : ℕ := p * pudding_cost_per_cup) :
  total_ice_cream_cost = total_pudding_cost + cost_difference → p = 5 :=
by
  sorry

end NUMINAMATH_GPT_jane_purchased_pudding_l1831_183199


namespace NUMINAMATH_GPT_pieces_to_cut_l1831_183107

-- Define the conditions
def rodLength : ℝ := 42.5  -- Length of the rod
def pieceLength : ℝ := 0.85  -- Length of each piece

-- Define the theorem that needs to be proven
theorem pieces_to_cut (h1 : rodLength = 42.5) (h2 : pieceLength = 0.85) : 
  (rodLength / pieceLength) = 50 := 
  by sorry

end NUMINAMATH_GPT_pieces_to_cut_l1831_183107


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1831_183156

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) * x + abs (x - 2) * (x - a)

theorem part1_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a < 0} = {x : ℝ | x < 1} :=
by
  sorry

theorem part2_solution (x : ℝ) (hx : x < 1) :
  {a : ℝ | f x a < 0} = {a : ℝ | 1 ≤ a} :=
by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1831_183156


namespace NUMINAMATH_GPT_geometric_sequence_sum_5_l1831_183108

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ i j : ℕ, ∃ r : ℝ, a (i + 1) = a i * r ∧ a (j + 1) = a j * r

theorem geometric_sequence_sum_5
  (a : ℕ → ℝ)
  (h : geometric_sequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 * a 6 + 2 * a 4 * a 5 + (a 5) ^ 2 = 25) :
  a 4 + a 5 = 5 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_5_l1831_183108


namespace NUMINAMATH_GPT_remainder_sum_mod_14_l1831_183194

theorem remainder_sum_mod_14 
  (a b c : ℕ) 
  (ha : a % 14 = 5) 
  (hb : b % 14 = 5) 
  (hc : c % 14 = 5) :
  (a + b + c) % 14 = 1 := 
by
  sorry

end NUMINAMATH_GPT_remainder_sum_mod_14_l1831_183194


namespace NUMINAMATH_GPT_sum_of_integers_ways_l1831_183178

theorem sum_of_integers_ways (n : ℕ) (h : n > 0) : 
  ∃ ways : ℕ, ways = 2^(n-1) := sorry

end NUMINAMATH_GPT_sum_of_integers_ways_l1831_183178


namespace NUMINAMATH_GPT_calc_nabla_l1831_183120

noncomputable def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calc_nabla : (op_nabla (op_nabla 2 3) 4) = 11 / 9 :=
by
  unfold op_nabla
  sorry

end NUMINAMATH_GPT_calc_nabla_l1831_183120


namespace NUMINAMATH_GPT_proposition_B_l1831_183127

-- Definitions of the conditions
def line (α : Type) := α
def plane (α : Type) := α
def is_within {α : Type} (a : line α) (p : plane α) : Prop := sorry
def is_perpendicular {α : Type} (a : line α) (p : plane α) : Prop := sorry
def planes_are_perpendicular {α : Type} (p₁ p₂ : plane α) : Prop := sorry
def is_prism (poly : Type) : Prop := sorry

-- Propositions
def p {α : Type} (a : line α) (α₁ α₂ : plane α) : Prop :=
  is_within a α₁ ∧ is_perpendicular a α₂ → planes_are_perpendicular α₁ α₂

def q (poly : Type) : Prop := 
  (∃ (face1 face2 : poly), face1 ≠ face2 ∧ sorry) ∧ sorry

-- Proposition B
theorem proposition_B {α : Type} (a : line α) (α₁ α₂ : plane α) (poly : Type) :
  (p a α₁ α₂) ∧ ¬(q poly) :=
by {
  -- Skipping proof
  sorry
}

end NUMINAMATH_GPT_proposition_B_l1831_183127


namespace NUMINAMATH_GPT_paint_fraction_second_week_l1831_183133

theorem paint_fraction_second_week
  (total_paint : ℕ)
  (first_week_fraction : ℚ)
  (total_used : ℕ)
  (paint_first_week : ℕ)
  (remaining_paint : ℕ)
  (paint_second_week : ℕ)
  (fraction_second_week : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/4 →
  total_used = 225 →
  paint_first_week = first_week_fraction * total_paint →
  remaining_paint = total_paint - paint_first_week →
  paint_second_week = total_used - paint_first_week →
  fraction_second_week = paint_second_week / remaining_paint →
  fraction_second_week = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_paint_fraction_second_week_l1831_183133


namespace NUMINAMATH_GPT_minimize_expression_l1831_183119

theorem minimize_expression (a : ℝ) : ∃ c : ℝ, 0 ≤ c ∧ c ≤ a ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → (x^2 + 3 * (a-x)^2) ≥ ((3*a/4)^2 + 3 * (a-3*a/4)^2)) :=
by
  sorry

end NUMINAMATH_GPT_minimize_expression_l1831_183119


namespace NUMINAMATH_GPT_find_number_l1831_183102

theorem find_number (x : ℝ) (h : 0.5 * x = 0.1667 * x + 10) : x = 30 :=
sorry

end NUMINAMATH_GPT_find_number_l1831_183102


namespace NUMINAMATH_GPT_parabola_num_xintercepts_l1831_183110

-- Defining the equation of the parabola
def parabola (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- The main theorem to state: the number of x-intercepts for the parabola is 2.
theorem parabola_num_xintercepts : ∃ (a b : ℝ), parabola a = 0 ∧ parabola b = 0 ∧ a ≠ b :=
by
  sorry

end NUMINAMATH_GPT_parabola_num_xintercepts_l1831_183110


namespace NUMINAMATH_GPT_incorrect_statement_l1831_183113

theorem incorrect_statement :
  ¬ (∀ (l1 l2 l3 : ℝ → ℝ → Prop), 
      (∀ (x y : ℝ), l3 x y → l1 x y) ∧ 
      (∀ (x y : ℝ), l3 x y → l2 x y) → 
      (∀ (x y : ℝ), l1 x y → l2 x y)) :=
by sorry

end NUMINAMATH_GPT_incorrect_statement_l1831_183113


namespace NUMINAMATH_GPT_vertex_of_parabola_l1831_183118

theorem vertex_of_parabola :
  (∃ (h k : ℤ), ∀ (x : ℝ), y = (x - h)^2 + k) → (h = 2 ∧ k = -3) := by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1831_183118


namespace NUMINAMATH_GPT_sum_distinct_x2_y2_z2_l1831_183138

/-
Given positive integers x, y, and z such that
x + y + z = 30 and gcd(x, y) + gcd(y, z) + gcd(z, x) = 10,
prove that the sum of all possible distinct values of x^2 + y^2 + z^2 is 404.
-/
theorem sum_distinct_x2_y2_z2 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 30) 
  (h_gcd : Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10) : 
  x^2 + y^2 + z^2 = 404 :=
sorry

end NUMINAMATH_GPT_sum_distinct_x2_y2_z2_l1831_183138


namespace NUMINAMATH_GPT_no_perfect_square_with_one_digit_appending_l1831_183159

def append_digit (n : Nat) (d : Fin 10) : Nat :=
  n * 10 + d.val

theorem no_perfect_square_with_one_digit_appending :
  ∀ n : Nat, (∃ k : Nat, k * k = n) → 
  (¬ (∃ d1 : Fin 10, ∃ k : Nat, k * k = append_digit n d1.val) ∧
   ¬ (∃ d2 : Fin 10, ∃ d3 : Fin 10, ∃ k : Nat, k * k = d2.val * 10 ^ (Nat.digits 10 n).length + n * 10 + d3.val)) :=
by sorry

end NUMINAMATH_GPT_no_perfect_square_with_one_digit_appending_l1831_183159


namespace NUMINAMATH_GPT_opinion_change_difference_l1831_183121

variables (initial_enjoy final_enjoy initial_not_enjoy final_not_enjoy : ℕ)
variables (n : ℕ) -- number of students in the class

-- Given conditions
def initial_conditions :=
  initial_enjoy = 40 * n / 100 ∧ initial_not_enjoy = 60 * n / 100

def final_conditions :=
  final_enjoy = 80 * n / 100 ∧ final_not_enjoy = 20 * n / 100

-- The theorem to prove
theorem opinion_change_difference :
  initial_conditions n initial_enjoy initial_not_enjoy →
  final_conditions n final_enjoy final_not_enjoy →
  (40 ≤ initial_enjoy + 20 ∧ 40 ≤ initial_not_enjoy + 20 ∧
  max_change = 60 ∧ min_change = 40 → max_change - min_change = 20) := 
  sorry

end NUMINAMATH_GPT_opinion_change_difference_l1831_183121


namespace NUMINAMATH_GPT_combined_share_b_d_l1831_183128

-- Definitions for the amounts shared between the children
def total_amount : ℝ := 15800
def share_a_plus_c : ℝ := 7022.222222222222

-- The goal is to prove that the combined share of B and D is 8777.777777777778
theorem combined_share_b_d :
  ∃ B D : ℝ, (B + D = total_amount - share_a_plus_c) :=
by
  sorry

end NUMINAMATH_GPT_combined_share_b_d_l1831_183128


namespace NUMINAMATH_GPT_maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l1831_183140

-- Defining productivity functions for male and female kittens
def male_productivity (K : ℝ) : ℝ := 80 - 4 * K
def female_productivity (K : ℝ) : ℝ := 16 - 0.25 * K

-- Condition (a): Maximizing number of mice caught by 2 kittens
theorem maximize_mice_two_kittens : 
  ∃ (male1 male2 : ℝ) (K_m1 K_m2 : ℝ), 
    (male1 = male_productivity K_m1) ∧ 
    (male2 = male_productivity K_m2) ∧
    (K_m1 = 0) ∧ (K_m2 = 0) ∧
    (male1 + male2 = 160) := 
sorry

-- Condition (b): Different versions of JPPF
theorem different_versions_JPPF : 
  ∃ (v1 v2 v3 : Unit), 
    (v1 ≠ v2) ∧ (v2 ≠ v3) ∧ (v1 ≠ v3) :=
sorry

-- Condition (c): Analytical form of JPPF for each combination
theorem JPPF_combinations :
  ∃ (M K1 K2 : ℝ),
    (M = 160 - 4 * K1 ∧ K1 ≤ 40) ∨
    (M = 32 - 0.5 * K2 ∧ K2 ≤ 64) ∨
    (M = 96 - 0.25 * K2 ∧ K2 ≤ 64) ∨
    (M = 336 - 4 * K2 ∧ 64 < K2 ∧ K2 ≤ 84) :=
sorry

-- Condition (d): Analytical form for 2 males and 1 female
theorem JPPF_two_males_one_female :
  ∃ (M K : ℝ), 
    (0 < K ∧ K ≤ 64 ∧ M = 176 - 0.25 * K) ∨
    (64 < K ∧ K ≤ 164 ∧ M = 416 - 4 * K) :=
sorry

end NUMINAMATH_GPT_maximize_mice_two_kittens_different_versions_JPPF_JPPF_combinations_JPPF_two_males_one_female_l1831_183140


namespace NUMINAMATH_GPT_general_term_formula_is_not_element_l1831_183101

theorem general_term_formula (a : ℕ → ℤ) (h1 : a 1 = 2) (h17 : a 17 = 66) :
  (∀ n, a n = 4 * n - 2) :=
by
  sorry

theorem is_not_element (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 2) :
  ¬ (∃ n : ℕ, a n = 88) :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_is_not_element_l1831_183101


namespace NUMINAMATH_GPT_algebra_expression_value_l1831_183154

theorem algebra_expression_value (a : ℝ) (h : a^2 - 4 * a - 6 = 0) : a^2 - 4 * a + 3 = 9 :=
by
  sorry

end NUMINAMATH_GPT_algebra_expression_value_l1831_183154


namespace NUMINAMATH_GPT_complement_A_A_inter_complement_B_l1831_183175

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem complement_A : compl A = {x | x ≤ 1 ∨ 4 ≤ x} :=
by sorry

theorem A_inter_complement_B : A ∩ compl B = {x | 3 < x ∧ x < 4} :=
by sorry

end NUMINAMATH_GPT_complement_A_A_inter_complement_B_l1831_183175


namespace NUMINAMATH_GPT_triangle_area_l1831_183167

-- Define the conditions and problem
def BC : ℝ := 10
def height_from_A : ℝ := 12
def AC : ℝ := 13

-- State the main theorem
theorem triangle_area (BC height_from_A AC : ℝ) (hBC : BC = 10) (hheight : height_from_A = 12) (hAC : AC = 13) : 
  (1/2 * BC * height_from_A) = 60 :=
by 
  -- Insert the proof
  sorry

end NUMINAMATH_GPT_triangle_area_l1831_183167


namespace NUMINAMATH_GPT_opposite_of_pi_is_neg_pi_l1831_183190

-- Definition that the opposite of a number x is -1 * x
def opposite (x : ℝ) : ℝ := -1 * x

-- Theorem stating that the opposite of π is -π
theorem opposite_of_pi_is_neg_pi : opposite π = -π := 
  sorry

end NUMINAMATH_GPT_opposite_of_pi_is_neg_pi_l1831_183190


namespace NUMINAMATH_GPT_cost_of_white_car_l1831_183188

variable (W : ℝ)
variable (red_cars white_cars : ℕ)
variable (rent_red rent_white : ℝ)
variable (rented_hours : ℝ)
variable (total_earnings : ℝ)

theorem cost_of_white_car 
  (h1 : red_cars = 3)
  (h2 : white_cars = 2) 
  (h3 : rent_red = 3)
  (h4 : rented_hours = 3)
  (h5 : total_earnings = 2340) :
  2 * W * (rented_hours * 60) + 3 * rent_red * (rented_hours * 60) = total_earnings → 
  W = 2 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_white_car_l1831_183188


namespace NUMINAMATH_GPT_Congcong_CO2_emissions_l1831_183134

-- Definitions based on conditions
def CO2_emissions (t: ℝ) : ℝ := t * 0.91 -- Condition 1: CO2 emissions calculation

def Congcong_water_usage : ℝ := 6 -- Condition 2: Congcong's water usage (6 tons)

-- Statement we want to prove
theorem Congcong_CO2_emissions : CO2_emissions Congcong_water_usage = 5.46 :=
by 
  sorry

end NUMINAMATH_GPT_Congcong_CO2_emissions_l1831_183134


namespace NUMINAMATH_GPT_problem_l1831_183169

noncomputable def p : Prop :=
  ∀ x : ℝ, (0 < x) → Real.exp x > 1 + x

def q (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) + 2 = -(f x + 2)) → ∀ x : ℝ, f (-x) = f x - 4

theorem problem (f : ℝ → ℝ) : p ∨ q f :=
  sorry

end NUMINAMATH_GPT_problem_l1831_183169


namespace NUMINAMATH_GPT_solve_for_x_l1831_183126

-- Step d: Lean 4 statement
theorem solve_for_x : 
  (∃ x : ℚ, (x + 7) / (x - 4) = (x - 5) / (x + 2)) → (∃ x : ℚ, x = 1 / 3) :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1831_183126


namespace NUMINAMATH_GPT_expected_value_of_white_balls_l1831_183125

-- Definitions for problem conditions
def totalBalls : ℕ := 6
def whiteBalls : ℕ := 2
def redBalls : ℕ := 4
def ballsDrawn : ℕ := 2

-- Probability calculations
def P_X_0 : ℚ := (Nat.choose 4 2) / (Nat.choose totalBalls ballsDrawn)
def P_X_1 : ℚ := ((Nat.choose whiteBalls 1) * (Nat.choose redBalls 1)) / (Nat.choose totalBalls ballsDrawn)
def P_X_2 : ℚ := (Nat.choose whiteBalls 2) / (Nat.choose totalBalls ballsDrawn)

-- Expected value calculation
def expectedValue : ℚ := (0 * P_X_0) + (1 * P_X_1) + (2 * P_X_2)

theorem expected_value_of_white_balls :
  expectedValue = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_white_balls_l1831_183125


namespace NUMINAMATH_GPT_arithmetic_mean_l1831_183193

variables (x y z : ℝ)

def condition1 : Prop := 1 / (x * y) = y / (z - x + 1)
def condition2 : Prop := 1 / (x * y) = 2 / (z + 1)

theorem arithmetic_mean (h1 : condition1 x y z) (h2 : condition2 x y z) : x = (z + y) / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_l1831_183193


namespace NUMINAMATH_GPT_simplify_expression_l1831_183166

theorem simplify_expression : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1831_183166


namespace NUMINAMATH_GPT_archers_in_golden_l1831_183162

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end NUMINAMATH_GPT_archers_in_golden_l1831_183162


namespace NUMINAMATH_GPT_find_other_number_l1831_183139

theorem find_other_number (a b : ℕ) (h₁ : Nat.lcm a b = 3780) (h₂ : Nat.gcd a b = 18) (h₃ : a = 180) : b = 378 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1831_183139


namespace NUMINAMATH_GPT_average_speed_round_trip_l1831_183123

theorem average_speed_round_trip (D : ℝ) (hD : D > 0) :
  let time_uphill := D / 5
  let time_downhill := D / 100
  let total_distance := 2 * D
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 200 / 21 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l1831_183123


namespace NUMINAMATH_GPT_part1_part2_l1831_183176

def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci (n+1) + fibonacci n

theorem part1 (m n : ℕ) (h : divides m n) : divides (fibonacci m) (fibonacci n) :=
sorry

theorem part2 (m n : ℕ) : Nat.gcd (fibonacci m) (fibonacci n) = fibonacci (Nat.gcd m n) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1831_183176


namespace NUMINAMATH_GPT_inequality_solution_set_range_of_a_l1831_183122

section
variable {x a : ℝ}

def f (x a : ℝ) := |2 * x - 5 * a| + |2 * x + 1|
def g (x : ℝ) := |x - 1| + 3

theorem inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} :=
sorry

theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ a = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 :=
sorry
end

end NUMINAMATH_GPT_inequality_solution_set_range_of_a_l1831_183122


namespace NUMINAMATH_GPT_range_of_a_l1831_183171

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < a + 2 → x ≤ 2) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1831_183171


namespace NUMINAMATH_GPT_probability_third_draw_first_class_expected_value_first_class_in_10_draws_l1831_183105

-- Define the problem with products
structure Products where
  total : ℕ
  first_class : ℕ
  second_class : ℕ

-- Given products configuration
def products : Products := { total := 5, first_class := 3, second_class := 2 }

-- Probability calculation without replacement
-- Define the event of drawing
def draw_without_replacement (p : Products) (draws : ℕ) (desired_event : ℕ -> Bool) : ℚ := 
  if draws = 3 ∧ desired_event 3 ∧ ¬ desired_event 1 ∧ ¬ desired_event 2 then
    (2 / 5) * ((1 : ℚ) / 4) * (3 / 3)
  else 
    0

-- Define desired_event for the specific problem
def desired_event (n : ℕ) : Bool := 
  match n with
  | 3 => true
  | _ => false

-- The first problem's proof statement
theorem probability_third_draw_first_class : draw_without_replacement products 3 desired_event = 1 / 10 := sorry

-- Expected value calculation with replacement
-- Binomial distribution to find expected value
def expected_value_with_replacement (p : Products) (draws : ℕ) : ℚ :=
  draws * (p.first_class / p.total)

-- The second problem's proof statement
theorem expected_value_first_class_in_10_draws : expected_value_with_replacement products 10 = 6 := sorry

end NUMINAMATH_GPT_probability_third_draw_first_class_expected_value_first_class_in_10_draws_l1831_183105


namespace NUMINAMATH_GPT_stripe_area_l1831_183150

-- Definitions based on conditions
def diameter : ℝ := 40
def stripe_width : ℝ := 4
def revolutions : ℝ := 3

-- The statement we want to prove
theorem stripe_area (π : ℝ) : 
  (revolutions * π * diameter * stripe_width) = 480 * π :=
by
  sorry

end NUMINAMATH_GPT_stripe_area_l1831_183150


namespace NUMINAMATH_GPT_find_letters_with_dot_but_no_straight_line_l1831_183130

-- Define the problem statement and conditions
def DL : ℕ := 16
def L : ℕ := 30
def Total_letters : ℕ := 50

-- Define the function that calculates the number of letters with a dot but no straight line
def letters_with_dot_but_no_straight_line (DL L Total_letters : ℕ) : ℕ := Total_letters - (L + DL)

-- State the theorem to be proved
theorem find_letters_with_dot_but_no_straight_line : letters_with_dot_but_no_straight_line DL L Total_letters = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_letters_with_dot_but_no_straight_line_l1831_183130


namespace NUMINAMATH_GPT_heidi_and_liam_paint_in_15_minutes_l1831_183170

-- Definitions
def Heidi_rate : ℚ := 1 / 60
def Liam_rate : ℚ := 1 / 90
def combined_rate : ℚ := Heidi_rate + Liam_rate
def painting_time : ℚ := 15

-- Theorem to Prove
theorem heidi_and_liam_paint_in_15_minutes : painting_time * combined_rate = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_heidi_and_liam_paint_in_15_minutes_l1831_183170


namespace NUMINAMATH_GPT_same_color_combination_sum_l1831_183183

theorem same_color_combination_sum (m n : ℕ) (coprime_mn : Nat.gcd m n = 1)
  (prob_together : ∀ (total_candies : ℕ), total_candies = 20 →
    let terry_red := Nat.choose 8 2;
    let total_cases := Nat.choose total_candies 2;
    let prob_terry_red := terry_red / total_cases;
    
    let mary_red_given_terry := Nat.choose 6 2;
    let reduced_total_cases := Nat.choose 18 2;
    let prob_mary_red_given_terry := mary_red_given_terry / reduced_total_cases;
    
    let both_red := prob_terry_red * prob_mary_red_given_terry;
    
    let terry_blue := Nat.choose 12 2;
    let prob_terry_blue := terry_blue / total_cases;
    
    let mary_blue_given_terry := Nat.choose 10 2;
    let prob_mary_blue_given_terry := mary_blue_given_terry / reduced_total_cases;
    
    let both_blue := prob_terry_blue * prob_mary_blue_given_terry;
    
    let mixed_red_blue := Nat.choose 8 1 * Nat.choose 12 1;
    let prob_mixed_red_blue := mixed_red_blue / total_cases;
    let both_mixed := prob_mixed_red_blue;
    
    let prob_same_combination := both_red + both_blue + both_mixed;
    
    prob_same_combination = m / n
  ) :
  m + n = 5714 :=
by
  sorry

end NUMINAMATH_GPT_same_color_combination_sum_l1831_183183


namespace NUMINAMATH_GPT_product_of_consecutive_integers_l1831_183174

theorem product_of_consecutive_integers (n : ℤ) :
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_l1831_183174


namespace NUMINAMATH_GPT_min_value_of_sum_l1831_183124

noncomputable def min_value_x_3y (x y : ℝ) : ℝ :=
  x + 3 * y

theorem min_value_of_sum (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0) 
  (cond : 1 / (x + 1) + 1 / (y + 1) = 1 / 2) :
  x + 3 * y ≥ 4 + 4 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_min_value_of_sum_l1831_183124


namespace NUMINAMATH_GPT_solve_for_x_l1831_183173

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h_eq : y = 1 / (3 * x^2 + 2 * x + 1)) : x = 0 ∨ x = -2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1831_183173


namespace NUMINAMATH_GPT_permutations_eq_factorial_l1831_183163

theorem permutations_eq_factorial (n : ℕ) : 
  (∃ Pn : ℕ, Pn = n!) := 
sorry

end NUMINAMATH_GPT_permutations_eq_factorial_l1831_183163


namespace NUMINAMATH_GPT_number_of_teams_in_league_l1831_183177

theorem number_of_teams_in_league (n : ℕ) :
  (6 * n * (n - 1)) / 2 = 396 ↔ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_teams_in_league_l1831_183177


namespace NUMINAMATH_GPT_new_lamp_height_is_correct_l1831_183189

-- Define the height of the old lamp
def old_lamp_height : ℝ := 1

-- Define the additional height of the new lamp
def additional_height : ℝ := 1.33

-- Proof statement
theorem new_lamp_height_is_correct :
  old_lamp_height + additional_height = 2.33 :=
sorry

end NUMINAMATH_GPT_new_lamp_height_is_correct_l1831_183189


namespace NUMINAMATH_GPT_distance_from_rachel_to_nicholas_l1831_183185

def distance (speed time : ℝ) := speed * time

theorem distance_from_rachel_to_nicholas :
  distance 2 5 = 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_distance_from_rachel_to_nicholas_l1831_183185


namespace NUMINAMATH_GPT_remainder_when_divided_by_10_l1831_183137

theorem remainder_when_divided_by_10 :
  (4219 * 2675 * 394082 * 5001) % 10 = 0 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_10_l1831_183137


namespace NUMINAMATH_GPT_physics_kit_prices_l1831_183131

theorem physics_kit_prices :
  ∃ (price_A price_B : ℝ), price_A = 180 ∧ price_B = 150 ∧
    price_A = 1.2 * price_B ∧
    9900 / price_A = 7500 / price_B + 5 :=
by
  use 180, 150
  sorry

end NUMINAMATH_GPT_physics_kit_prices_l1831_183131


namespace NUMINAMATH_GPT_minimum_value_am_bn_l1831_183111

-- Definitions and conditions
variables {a b m n : ℝ}
variables (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < m) (h₃ : 0 < n)
variables (h₄ : a + b = 1) (h₅ : m * n = 2)

-- Statement of the proof problem
theorem minimum_value_am_bn :
  ∃ c, (∀ a b m n : ℝ, 0 < a → 0 < b → 0 < m → 0 < n → a + b = 1 → m * n = 2 → (am * bn) * (bm * an) ≥ c) ∧ c = 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_am_bn_l1831_183111


namespace NUMINAMATH_GPT_length_CD_l1831_183195

-- Given data
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (AB BC : ℝ)

noncomputable def triangle_ABC : Prop :=
  AB = 5 ∧ BC = 7 ∧ ∃ (angle_ABC : ℝ), angle_ABC = 90

-- The target condition to prove
theorem length_CD {CD : ℝ} (h : triangle_ABC AB BC) : CD = 7 :=
by {
  -- proof would be here
  sorry
}

end NUMINAMATH_GPT_length_CD_l1831_183195


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1831_183149

-- (1) Prove (1 + sqrt 3) * (2 - sqrt 3) = -1 + sqrt 3
theorem problem1 : (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 :=
by sorry

-- (2) Prove (sqrt 36 * sqrt 12) / sqrt 3 = 12
theorem problem2 : (Real.sqrt 36 * Real.sqrt 12) / Real.sqrt 3 = 12 :=
by sorry

-- (3) Prove sqrt 18 - sqrt 8 + sqrt (1 / 8) = (5 * sqrt 2) / 4
theorem problem3 : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1 / 8) = (5 * Real.sqrt 2) / 4 :=
by sorry

-- (4) Prove (3 * sqrt 18 + (1 / 5) * sqrt 50 - 4 * sqrt (1 / 2)) / sqrt 32 = 2
theorem problem4 : (3 * Real.sqrt 18 + (1 / 5) * Real.sqrt 50 - 4 * Real.sqrt (1 / 2)) / Real.sqrt 32 = 2 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1831_183149


namespace NUMINAMATH_GPT_rectangle_integer_sides_noncongruent_count_l1831_183141

theorem rectangle_integer_sides_noncongruent_count (h w : ℕ) :
  (2 * (w + h) = 72 ∧ w ≠ h) ∨ ((w = h) ∧ 2 * (w + h) = 72) →
  (∃ (count : ℕ), count = 18) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_integer_sides_noncongruent_count_l1831_183141


namespace NUMINAMATH_GPT_find_m_l1831_183197

-- Given the condition
def condition (m : ℕ) := (1 / 5 : ℝ)^m * (1 / 4 : ℝ)^2 = 1 / (10 : ℝ)^4

-- Theorem to prove that m is 4 given the condition
theorem find_m (m : ℕ) (h : condition m) : m = 4 :=
sorry

end NUMINAMATH_GPT_find_m_l1831_183197


namespace NUMINAMATH_GPT_number_leaves_remainder_3_l1831_183187

theorem number_leaves_remainder_3 (n : ℕ) (h1 : 1680 % 9 = 0) (h2 : 1680 = n * 9) : 1680 % 1677 = 3 := by
  sorry

end NUMINAMATH_GPT_number_leaves_remainder_3_l1831_183187


namespace NUMINAMATH_GPT_find_cosine_of_angle_subtraction_l1831_183153

variable (α : ℝ)
variable (h : Real.sin ((Real.pi / 6) - α) = 1 / 3)

theorem find_cosine_of_angle_subtraction :
  Real.cos ((2 * Real.pi / 3) - α) = -1 / 3 :=
by
  exact sorry

end NUMINAMATH_GPT_find_cosine_of_angle_subtraction_l1831_183153


namespace NUMINAMATH_GPT_John_spending_l1831_183100

theorem John_spending
  (X : ℝ)
  (h1 : (1/2) * X + (1/3) * X + (1/10) * X + 10 = X) :
  X = 150 :=
by
  sorry

end NUMINAMATH_GPT_John_spending_l1831_183100


namespace NUMINAMATH_GPT_paths_via_checkpoint_l1831_183186

/-- Define the grid configuration -/
structure Point :=
  (x : ℕ) (y : ℕ)

/-- Calculate the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  n.choose k

/-- Define points A, B, C -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨5, 4⟩
def C : Point := ⟨3, 2⟩

/-- Calculate number of paths from A to C -/
def paths_A_to_C : ℕ :=
  binomial (3 + 2) 2

/-- Calculate number of paths from C to B -/
def paths_C_to_B : ℕ :=
  binomial (2 + 2) 2

/-- Calculate total number of paths from A to B via C -/
def total_paths_A_to_B_via_C : ℕ :=
  (paths_A_to_C * paths_C_to_B)

theorem paths_via_checkpoint :
  total_paths_A_to_B_via_C = 60 :=
by
  -- The proof is skipped as per the instruction
  sorry

end NUMINAMATH_GPT_paths_via_checkpoint_l1831_183186


namespace NUMINAMATH_GPT_max_isosceles_tris_2017_gon_l1831_183147

theorem max_isosceles_tris_2017_gon :
  ∀ (n : ℕ), n = 2017 →
  ∃ (t : ℕ), (∃ (d : ℕ), d = 2014 ∧ 2015 = (n - 2)) →
  t = 2010 :=
by
  sorry

end NUMINAMATH_GPT_max_isosceles_tris_2017_gon_l1831_183147


namespace NUMINAMATH_GPT_perpendicular_lines_l1831_183191

theorem perpendicular_lines {a : ℝ} :
  a*(a-1) + (1-a)*(2*a+3) = 0 → (a = 1 ∨ a = -3) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1831_183191


namespace NUMINAMATH_GPT_sixth_grade_percentage_combined_l1831_183145

def maplewood_percentages := [10, 20, 15, 15, 10, 15, 15]
def brookside_percentages := [16, 14, 13, 12, 12, 18, 15]

def maplewood_students := 150
def brookside_students := 180

def sixth_grade_maplewood := maplewood_students * (maplewood_percentages.get! 6) / 100
def sixth_grade_brookside := brookside_students * (brookside_percentages.get! 6) / 100

def total_students := maplewood_students + brookside_students
def total_sixth_graders := sixth_grade_maplewood + sixth_grade_brookside

def sixth_grade_percentage := total_sixth_graders / total_students * 100

theorem sixth_grade_percentage_combined : sixth_grade_percentage = 15 := by 
  sorry

end NUMINAMATH_GPT_sixth_grade_percentage_combined_l1831_183145


namespace NUMINAMATH_GPT_combined_mpg_rate_l1831_183116

-- Conditions of the problem
def ray_mpg : ℝ := 48
def tom_mpg : ℝ := 24
def ray_distance (s : ℝ) : ℝ := 2 * s
def tom_distance (s : ℝ) : ℝ := s

-- Theorem to prove the combined rate of miles per gallon
theorem combined_mpg_rate (s : ℝ) (h : s > 0) : 
  let total_distance := tom_distance s + ray_distance s
  let ray_gas_usage := ray_distance s / ray_mpg
  let tom_gas_usage := tom_distance s / tom_mpg
  let total_gas_usage := ray_gas_usage + tom_gas_usage
  total_distance / total_gas_usage = 36 := 
by
  sorry

end NUMINAMATH_GPT_combined_mpg_rate_l1831_183116


namespace NUMINAMATH_GPT_last_digit_x4_plus_inv_x4_l1831_183103

theorem last_digit_x4_plus_inv_x4 (x : ℝ) (h : x^2 - 13 * x + 1 = 0) : (x^4 + (1 / x)^4) % 10 = 7 := 
by
  sorry

end NUMINAMATH_GPT_last_digit_x4_plus_inv_x4_l1831_183103


namespace NUMINAMATH_GPT_cost_of_tissues_l1831_183136
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end NUMINAMATH_GPT_cost_of_tissues_l1831_183136


namespace NUMINAMATH_GPT_senya_mistakes_in_OCTAHEDRON_l1831_183112

noncomputable def mistakes_in_word (word : String) : Nat :=
  if word = "TETRAHEDRON" then 5
  else if word = "DODECAHEDRON" then 6
  else if word = "ICOSAHEDRON" then 7
  else if word = "OCTAHEDRON" then 5 
  else 0

theorem senya_mistakes_in_OCTAHEDRON : mistakes_in_word "OCTAHEDRON" = 5 := by
  sorry

end NUMINAMATH_GPT_senya_mistakes_in_OCTAHEDRON_l1831_183112


namespace NUMINAMATH_GPT_find_surface_area_of_sphere_l1831_183181

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 := a * b = 6
def condition2 := b * c = 2
def condition3 := a * c = 3
def vertices_on_sphere := true  -- Assuming vertices on tensor sphere condition for mathematical completion

theorem find_surface_area_of_sphere
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c)
  (h4 : vertices_on_sphere) :
  4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2)) / 2)^2 = 14 * Real.pi :=
  sorry

end NUMINAMATH_GPT_find_surface_area_of_sphere_l1831_183181


namespace NUMINAMATH_GPT_prove_product_of_b_l1831_183151

noncomputable def g (x b : ℝ) := b / (5 * x - 7)

noncomputable def g_inv (y b : ℝ) := (b + 7 * y) / (5 * y)

theorem prove_product_of_b (b1 b2 : ℝ) (h1 : g 3 b1 = g_inv (b1 + 2) b1) (h2 : g 3 b2 = g_inv (b2 + 2) b2) :
  b1 * b2 = -22.39 := by
  sorry

end NUMINAMATH_GPT_prove_product_of_b_l1831_183151


namespace NUMINAMATH_GPT_Jungkook_blue_balls_unchanged_l1831_183114

variable (initialRedBalls : ℕ) (initialBlueBalls : ℕ) (initialYellowBalls : ℕ)
variable (newYellowBallGifted: ℕ)

-- Define the initial conditions
def Jungkook_balls := initialRedBalls = 5 ∧ initialBlueBalls = 4 ∧ initialYellowBalls = 3 ∧ newYellowBallGifted = 1

-- State the theorem to prove
theorem Jungkook_blue_balls_unchanged (h : Jungkook_balls initRed initBlue initYellow newYellowGift): initialBlueBalls = 4 := 
by
sorry

end NUMINAMATH_GPT_Jungkook_blue_balls_unchanged_l1831_183114


namespace NUMINAMATH_GPT_probability_MAME_on_top_l1831_183152

theorem probability_MAME_on_top : 
  let num_sections := 8
  let favorable_outcome := 1
  (favorable_outcome : ℝ) / (num_sections : ℝ) = 1 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_probability_MAME_on_top_l1831_183152


namespace NUMINAMATH_GPT_function_C_is_even_l1831_183143

theorem function_C_is_even : ∀ x : ℝ, 2 * (-x)^2 - 1 = 2 * x^2 - 1 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_function_C_is_even_l1831_183143


namespace NUMINAMATH_GPT_integral_result_l1831_183104

theorem integral_result (b : ℝ) (h : ∫ x in e..b, (2 / x) = 6) : b = Real.exp 4 :=
sorry

end NUMINAMATH_GPT_integral_result_l1831_183104


namespace NUMINAMATH_GPT_a1964_eq_neg1_l1831_183117

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 1 ∧ a 3 = -1 ∧ ∀ n ≥ 4, a n = a (n-1) * a (n-3)

theorem a1964_eq_neg1 (a : ℕ → ℤ) (h : seq a) : a 1964 = -1 :=
  by sorry

end NUMINAMATH_GPT_a1964_eq_neg1_l1831_183117


namespace NUMINAMATH_GPT_circle_diameter_l1831_183155

theorem circle_diameter (A : ℝ) (π : ℝ) (r : ℝ) (d : ℝ) (h1 : A = 64 * π) (h2 : A = π * r^2) (h3 : d = 2 * r) :
  d = 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_diameter_l1831_183155


namespace NUMINAMATH_GPT_number_of_cuboids_painted_l1831_183192

/--
Suppose each cuboid has 6 outer faces and Amelia painted a total of 36 faces.
Prove that the number of cuboids Amelia painted is 6.
-/
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) 
  (h1 : total_faces = 36) (h2 : faces_per_cuboid = 6) :
  total_faces / faces_per_cuboid = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_number_of_cuboids_painted_l1831_183192


namespace NUMINAMATH_GPT_total_amount_shared_l1831_183148

theorem total_amount_shared (A B C : ℕ) (h1 : 3 * B = 5 * A) (h2 : B = 25) (h3 : 5 * C = 8 * B) : A + B + C = 80 := by
  sorry

end NUMINAMATH_GPT_total_amount_shared_l1831_183148


namespace NUMINAMATH_GPT_ham_block_cut_mass_distribution_l1831_183146

theorem ham_block_cut_mass_distribution
  (length width height : ℝ) (mass : ℝ)
  (parallelogram_side1 parallelogram_side2 : ℝ)
  (condition1 : length = 12) 
  (condition2 : width = 12) 
  (condition3 : height = 35)
  (condition4 : mass = 5)
  (condition5 : parallelogram_side1 = 15) 
  (condition6 : parallelogram_side2 = 20) :
  ∃ (mass_piece1 mass_piece2 : ℝ),
    mass_piece1 = 1.7857 ∧ mass_piece2 = 3.2143 :=
by
  sorry

end NUMINAMATH_GPT_ham_block_cut_mass_distribution_l1831_183146


namespace NUMINAMATH_GPT_fraction_inhabitable_l1831_183182

-- Define the constants based on the given conditions
def fraction_water : ℚ := 3 / 5
def fraction_inhabitable_land : ℚ := 3 / 4

-- Define the theorem to prove that the fraction of Earth's surface that is inhabitable is 3/10
theorem fraction_inhabitable (w h : ℚ) (hw : w = fraction_water) (hh : h = fraction_inhabitable_land) : 
  (1 - w) * h = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_inhabitable_l1831_183182


namespace NUMINAMATH_GPT_police_station_distance_l1831_183198

theorem police_station_distance (thief_speed police_speed: ℝ) (delay chase_time: ℝ) 
  (h_thief_speed: thief_speed = 20) 
  (h_police_speed: police_speed = 40) 
  (h_delay: delay = 1)
  (h_chase_time: chase_time = 4) : 
  ∃ D: ℝ, D = 60 :=
by
  sorry

end NUMINAMATH_GPT_police_station_distance_l1831_183198


namespace NUMINAMATH_GPT_derivative_at_0_l1831_183179

def f (x : ℝ) : ℝ := x + x^2

theorem derivative_at_0 : deriv f 0 = 1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_derivative_at_0_l1831_183179


namespace NUMINAMATH_GPT_Ahmed_total_distance_traveled_l1831_183168

/--
Ahmed stops one-quarter of the way to the store.
He continues for 12 km to reach the store.
Prove that the total distance Ahmed travels is 16 km.
-/
theorem Ahmed_total_distance_traveled
  (D : ℝ) (h1 : D > 0)  -- D is the total distance to the store, assumed to be positive
  (h_stop : D / 4 + 12 = D) : D = 16 := 
sorry

end NUMINAMATH_GPT_Ahmed_total_distance_traveled_l1831_183168


namespace NUMINAMATH_GPT_speed_of_freight_train_l1831_183161

-- Definitions based on the conditions
def distance := 390  -- The towns are 390 km apart
def express_speed := 80  -- The express train travels at 80 km per hr
def travel_time := 3  -- They pass one another 3 hr later

-- The freight train travels 30 km per hr slower than the express train
def freight_speed := express_speed - 30

-- The statement that we aim to prove:
theorem speed_of_freight_train : freight_speed = 50 := 
by 
  sorry

end NUMINAMATH_GPT_speed_of_freight_train_l1831_183161


namespace NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l1831_183129

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : 4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b :=
  sorry

-- Problem 2
theorem simplify_expression_2 (m : ℝ) : (3 * m / 2) - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 :=
  sorry

end NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l1831_183129


namespace NUMINAMATH_GPT_find_CP_A_l1831_183142

noncomputable def CP_A : Float := 173.41
def SP_B (CP_A : Float) : Float := 1.20 * CP_A
def SP_C (SP_B : Float) : Float := 1.25 * SP_B
def TC_C (SP_C : Float) : Float := 1.15 * SP_C
def SP_D1 (TC_C : Float) : Float := 1.30 * TC_C
def SP_D2 (SP_D1 : Float) : Float := 0.90 * SP_D1
def SP_D2_actual : Float := 350

theorem find_CP_A : 
  (SP_D2 (SP_D1 (TC_C (SP_C (SP_B CP_A))))) = SP_D2_actual → 
  CP_A = 173.41 := sorry

end NUMINAMATH_GPT_find_CP_A_l1831_183142


namespace NUMINAMATH_GPT_value_of_4_and_2_l1831_183164

noncomputable def custom_and (a b : ℕ) : ℕ :=
  ((a + b) * (a - b)) ^ 2

theorem value_of_4_and_2 : custom_and 4 2 = 144 :=
  sorry

end NUMINAMATH_GPT_value_of_4_and_2_l1831_183164


namespace NUMINAMATH_GPT_sum_divides_product_iff_l1831_183144

theorem sum_divides_product_iff (n : ℕ) : 
  (n*(n+1)/2) ∣ n! ↔ ∃ (a b : ℕ), 1 < a ∧ 1 < b ∧ a * b = n + 1 ∧ a ≤ n ∧ b ≤ n :=
sorry

end NUMINAMATH_GPT_sum_divides_product_iff_l1831_183144


namespace NUMINAMATH_GPT_angle_C_in_triangle_ABC_l1831_183160

noncomputable def find_angle_C (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : Prop :=
  C = Real.pi / 6

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : find_angle_C A B C h1 h2 h3 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_angle_C_in_triangle_ABC_l1831_183160


namespace NUMINAMATH_GPT_totalSolutions_l1831_183196

noncomputable def systemOfEquations (a b c d a1 b1 c1 d1 x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 = d ∧ a1 * x^2 + b1 * x * y + c1 * y^2 = d1

theorem totalSolutions 
  (a b c d a1 b1 c1 d1 : ℝ) 
  (h₀ : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)
  (h₁ : a1 ≠ 0 ∨ b1 ≠ 0 ∨ c1 ≠ 0) :
  ∃ x y : ℝ, systemOfEquations a b c d a1 b1 c1 d1 x y :=
sorry

end NUMINAMATH_GPT_totalSolutions_l1831_183196


namespace NUMINAMATH_GPT_correct_statement_l1831_183180

theorem correct_statement (a b : ℝ) (h_a : a ≥ 0) (h_b : b ≥ 0) : (a ≥ 0 ∧ b ≥ 0) :=
by
  exact ⟨h_a, h_b⟩

end NUMINAMATH_GPT_correct_statement_l1831_183180


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_property_l1831_183135

theorem smallest_n_for_divisibility_property (k : ℕ) : ∃ n : ℕ, n = k + 2 ∧ ∀ (S : Finset ℤ), 
  S.card = n → 
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ (a ≠ b ∧ (a + b) % (2 * k + 1) = 0 ∨ (a - b) % (2 * k + 1) = 0) :=
by
sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_property_l1831_183135


namespace NUMINAMATH_GPT_potions_needed_l1831_183157

-- Definitions
def galleons_to_knuts (galleons : Int) : Int := galleons * 17 * 23
def sickles_to_knuts (sickles : Int) : Int := sickles * 23

-- Conditions from the problem
def cost_of_owl_in_knuts : Int := galleons_to_knuts 2 + sickles_to_knuts 1 + 5
def knuts_per_potion : Int := 9

-- Prove the number of potions needed is 90
theorem potions_needed : cost_of_owl_in_knuts / knuts_per_potion = 90 := by
  sorry

end NUMINAMATH_GPT_potions_needed_l1831_183157


namespace NUMINAMATH_GPT_intersection_correct_l1831_183132

def setA : Set ℝ := { x | x - 1 ≤ 0 }
def setB : Set ℝ := { x | x^2 - 4 * x ≤ 0 }
def expected_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_correct : (setA ∩ setB) = expected_intersection :=
sorry

end NUMINAMATH_GPT_intersection_correct_l1831_183132


namespace NUMINAMATH_GPT_denote_loss_of_300_dollars_l1831_183106

-- Define the concept of financial transactions
def denote_gain (amount : Int) : Int := amount
def denote_loss (amount : Int) : Int := -amount

-- The condition given in the problem
def earn_500_dollars_is_500 := denote_gain 500 = 500

-- The assertion we need to prove
theorem denote_loss_of_300_dollars : denote_loss 300 = -300 := 
by 
  sorry

end NUMINAMATH_GPT_denote_loss_of_300_dollars_l1831_183106
