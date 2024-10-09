import Mathlib

namespace meadow_income_is_960000_l231_23135

theorem meadow_income_is_960000 :
  let boxes := 30
  let packs_per_box := 40
  let diapers_per_pack := 160
  let price_per_diaper := 5
  (boxes * packs_per_box * diapers_per_pack * price_per_diaper) = 960000 := 
by
  sorry

end meadow_income_is_960000_l231_23135


namespace ram_efficiency_eq_27_l231_23105

theorem ram_efficiency_eq_27 (R : ℕ) (h1 : ∀ Krish, 2 * (1 / (R : ℝ)) = 1 / Krish) 
  (h2 : ∀ s, 3 * (1 / (R : ℝ)) * s = 1 ↔ s = (9 : ℝ)) : R = 27 :=
sorry

end ram_efficiency_eq_27_l231_23105


namespace compare_rat_neg_l231_23108

-- Define the numbers
def a : ℚ := -3 / 4
def b : ℚ := -4 / 5

-- State the theorem
theorem compare_rat_neg : a > b :=
by
  -- Proof to be added here
  sorry

end compare_rat_neg_l231_23108


namespace y_works_in_40_days_l231_23101

theorem y_works_in_40_days :
  ∃ d, (d > 0) ∧ 
  (1/20 + 1/d = 3/40) ∧ 
  d = 40 :=
by
  use 40
  sorry

end y_works_in_40_days_l231_23101


namespace pencils_given_out_l231_23133

-- Defining the conditions
def num_children : ℕ := 4
def pencils_per_child : ℕ := 2

-- Formulating the problem statement, with the goal to prove the total number of pencils
theorem pencils_given_out : num_children * pencils_per_child = 8 := 
by 
  sorry

end pencils_given_out_l231_23133


namespace Claudia_solution_l231_23116

noncomputable def Claudia_coins : Prop :=
  ∃ (x y : ℕ), x + y = 12 ∧ 23 - x = 17 ∧ y = 6

theorem Claudia_solution : Claudia_coins :=
by
  existsi 6
  existsi 6
  sorry

end Claudia_solution_l231_23116


namespace mul_102_102_l231_23107

theorem mul_102_102 : 102 * 102 = 10404 := by
  sorry

end mul_102_102_l231_23107


namespace intersection_of_lines_l231_23127

theorem intersection_of_lines : 
  let x := (5 : ℚ) / 9
  let y := (5 : ℚ) / 3
  (y = 3 * x ∧ y - 5 = -6 * x) ↔ (x, y) = ((5 : ℚ) / 9, (5 : ℚ) / 3) := 
by 
  sorry

end intersection_of_lines_l231_23127


namespace max_xy_l231_23113

theorem max_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x / 3 + y / 4 = 1) : xy ≤ 3 :=
by {
  -- proof omitted
  sorry
}

end max_xy_l231_23113


namespace walnut_trees_l231_23102

theorem walnut_trees (logs_per_pine logs_per_maple logs_per_walnut pine_trees maple_trees total_logs walnut_trees : ℕ)
  (h1 : logs_per_pine = 80)
  (h2 : logs_per_maple = 60)
  (h3 : logs_per_walnut = 100)
  (h4 : pine_trees = 8)
  (h5 : maple_trees = 3)
  (h6 : total_logs = 1220)
  (h7 : total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut) :
  walnut_trees = 4 :=
by
  sorry

end walnut_trees_l231_23102


namespace prob_lfloor_XZ_YZ_product_eq_33_l231_23119

noncomputable def XZ_YZ_product : ℝ :=
  let AB := 15
  let BC := 14
  let CA := 13
  -- Definition of points and conditions
  -- Note: Specific geometric definitions and conditions need to be properly defined as per Lean's geometry library. This is a simplified placeholder.
  sorry

theorem prob_lfloor_XZ_YZ_product_eq_33 :
  (⌊XZ_YZ_product⌋ = 33) := sorry

end prob_lfloor_XZ_YZ_product_eq_33_l231_23119


namespace distance_from_y_axis_l231_23193

theorem distance_from_y_axis (x : ℝ) : abs x = 10 :=
by
  -- Define distances
  let d_x := 5
  let d_y := abs x
  -- Given condition
  have h : d_x = (1 / 2) * d_y := sorry
  -- Use the given condition to prove the required statement
  sorry

end distance_from_y_axis_l231_23193


namespace incorrect_proposition_C_l231_23174

theorem incorrect_proposition_C (a b c d : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a^4 + b^4 + c^4 + d^4 = 2 * (a^2 * b^2 + c^2 * d^2) → ¬ (a = b ∧ b = c ∧ c = d) := 
sorry

end incorrect_proposition_C_l231_23174


namespace shape_area_l231_23136

-- Define the conditions as Lean definitions
def side_length : ℝ := 3
def num_squares : ℕ := 4

-- Prove that the area of the shape is 36 cm² given the conditions
theorem shape_area : num_squares * (side_length * side_length) = 36 := by
    -- The proof is skipped with sorry
    sorry

end shape_area_l231_23136


namespace area_smaller_part_l231_23188

theorem area_smaller_part (A B : ℝ) (h₁ : A + B = 500) (h₂ : B - A = (A + B) / 10) : A = 225 :=
by sorry

end area_smaller_part_l231_23188


namespace fourth_hexagon_dots_l231_23166

   -- Define the number of dots in the first, second, and third hexagons
   def hexagon_dots (n : ℕ) : ℕ :=
     match n with
     | 1 => 1
     | 2 => 8
     | 3 => 22
     | 4 => 46
     | _ => 0

   -- State the theorem to be proved
   theorem fourth_hexagon_dots : hexagon_dots 4 = 46 :=
   by
     sorry
   
end fourth_hexagon_dots_l231_23166


namespace remainder_of_sum_mod_9_l231_23130

theorem remainder_of_sum_mod_9 :
  (9023 + 9024 + 9025 + 9026 + 9027) % 9 = 2 :=
by
  sorry

end remainder_of_sum_mod_9_l231_23130


namespace prime_divides_a_minus_3_l231_23152

theorem prime_divides_a_minus_3 (a p : ℤ) (hp : Prime p) (h1 : p ∣ 5 * a - 1) (h2 : p ∣ a - 10) : p ∣ a - 3 := by
  sorry

end prime_divides_a_minus_3_l231_23152


namespace triangle_is_right_angle_l231_23176

theorem triangle_is_right_angle (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) : 
  a^2 + b^2 = c^2 :=
by 
  sorry

end triangle_is_right_angle_l231_23176


namespace solution_l231_23191

def system (a b : ℝ) : Prop :=
  (2 * a + b = 3) ∧ (a - b = 1)

theorem solution (a b : ℝ) (h: system a b) : a + 2 * b = 2 :=
by
  cases h with
  | intro h1 h2 => sorry

end solution_l231_23191


namespace largest_integer_less_85_with_remainder_3_l231_23138

theorem largest_integer_less_85_with_remainder_3 (n : ℕ) : 
  n < 85 ∧ n % 9 = 3 → n ≤ 84 :=
by
  intro h
  sorry

end largest_integer_less_85_with_remainder_3_l231_23138


namespace no_integer_sided_triangle_with_odd_perimeter_1995_l231_23187

theorem no_integer_sided_triangle_with_odd_perimeter_1995 :
  ¬ ∃ (a b c : ℕ), (a + b + c = 1995) ∧ (∃ (h1 h2 h3 : ℕ), true) :=
by
  sorry

end no_integer_sided_triangle_with_odd_perimeter_1995_l231_23187


namespace smallest_sum_of_three_diff_numbers_l231_23150

theorem smallest_sum_of_three_diff_numbers : 
  ∀ (s : Set ℤ), s = {8, -7, 2, -4, 20} → ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c = -9) :=
by
  sorry

end smallest_sum_of_three_diff_numbers_l231_23150


namespace integral_sin_from_0_to_pi_div_2_l231_23131

theorem integral_sin_from_0_to_pi_div_2 :
  ∫ x in (0 : ℝ)..(Real.pi / 2), Real.sin x = 1 := by
  sorry

end integral_sin_from_0_to_pi_div_2_l231_23131


namespace find_height_of_cylinder_l231_23137

theorem find_height_of_cylinder (r SA : ℝ) (h : ℝ) (h_r : r = 3) (h_SA : SA = 30 * Real.pi) :
  SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h → h = 2 :=
by
  sorry

end find_height_of_cylinder_l231_23137


namespace find_principal_amount_l231_23156

theorem find_principal_amount (P r : ℝ) 
    (h1 : 815 - P = P * r * 3) 
    (h2 : 850 - P = P * r * 4) : 
    P = 710 :=
by
  -- proof steps will go here
  sorry

end find_principal_amount_l231_23156


namespace product_of_roots_l231_23157

theorem product_of_roots (x1 x2 : ℝ) (h : x1 * x2 = -2) :
  (∀ x : ℝ, x^2 + x - 2 = 0 → (x = x1 ∨ x = x2)) → x1 * x2 = -2 :=
by
  intros h_root
  exact h

end product_of_roots_l231_23157


namespace power_function_value_l231_23154

theorem power_function_value
  (α : ℝ)
  (h : 2^α = Real.sqrt 2) :
  (4 : ℝ) ^ α = 2 :=
by {
  sorry
}

end power_function_value_l231_23154


namespace extreme_values_x_axis_l231_23183

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := x * (a * x^2 + b * x + c)

theorem extreme_values_x_axis (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, f a b c x = x * (a * x^2 + b * x + c))
  (h3 : ∀ x, deriv (f a b c) x = 3 * a * x^2 + 2 * b * x + c)
  (h4 : deriv (f a b c) 1 = 0)
  (h5 : deriv (f a b c) (-1) = 0) :
  b = 0 :=
sorry

end extreme_values_x_axis_l231_23183


namespace rice_pounds_l231_23129

noncomputable def pounds_of_rice (r p : ℝ) : Prop :=
  r + p = 30 ∧ 1.10 * r + 0.55 * p = 23.50

theorem rice_pounds (r p : ℝ) (h : pounds_of_rice r p) : r = 12.7 :=
sorry

end rice_pounds_l231_23129


namespace malvina_correct_l231_23128
noncomputable def angle (x : ℝ) : Prop := 0 < x ∧ x < 180
noncomputable def malvina_identifies (x : ℝ) : Prop := x > 90

noncomputable def sum_of_values := (Real.sqrt 5 + Real.sqrt 2) / 2

theorem malvina_correct (x : ℝ) (h1 : angle x) (h2 : malvina_identifies x) :
  sum_of_values = (Real.sqrt 5 + Real.sqrt 2) / 2 :=
by sorry

end malvina_correct_l231_23128


namespace percentage_saved_on_hats_l231_23158

/-- Suppose the regular price of a hat is $60 and Maria buys four hats with progressive discounts: 
20% off the second hat, 40% off the third hat, and 50% off the fourth hat.
Prove that the percentage saved on the regular price for four hats is 27.5%. -/
theorem percentage_saved_on_hats :
  let regular_price := 60
  let discount_2 := 0.2 * regular_price
  let discount_3 := 0.4 * regular_price
  let discount_4 := 0.5 * regular_price
  let price_1 := regular_price
  let price_2 := regular_price - discount_2
  let price_3 := regular_price - discount_3
  let price_4 := regular_price - discount_4
  let total_regular := 4 * regular_price
  let total_discounted := price_1 + price_2 + price_3 + price_4
  let savings := total_regular - total_discounted
  let percentage_saved := (savings / total_regular) * 100
  percentage_saved = 27.5 :=
by
  sorry

end percentage_saved_on_hats_l231_23158


namespace angle_relation_l231_23147

-- Definitions for the triangle properties and angles.
variables {α : Type*} [LinearOrderedField α]
variables {A B C D E F : α}

-- Definitions stating the properties of the triangles.
def is_isosceles_triangle (a b c : α) : Prop :=
  a = b ∨ b = c ∨ c = a

def triangle_ABC_is_isosceles (AB AC : α) (ABC : α) : Prop :=
  is_isosceles_triangle AB AC ABC

def triangle_DEF_is_isosceles (DE DF : α) (DEF : α) : Prop :=
  is_isosceles_triangle DE DF DEF

-- Condition that gives the specific angle measure in triangle DEF.
def angle_DEF_is_100 (DEF : α) : Prop :=
  DEF = 100

-- The main theorem to prove.
theorem angle_relation (AB AC DE DF DEF a b c : α) :
  triangle_ABC_is_isosceles AB AC (AB + AC) →
  triangle_DEF_is_isosceles DE DF DEF →
  angle_DEF_is_100 DEF →
  a = c :=
by
  -- Assuming the conditions define the angles and state the relationship.
  sorry

end angle_relation_l231_23147


namespace one_point_shots_count_l231_23110

-- Define the given conditions
def three_point_shots : Nat := 15
def two_point_shots : Nat := 12
def total_points : Nat := 75
def points_per_three_shot : Nat := 3
def points_per_two_shot : Nat := 2

-- Define the total points contributed by three-point and two-point shots
def three_point_total : Nat := three_point_shots * points_per_three_shot
def two_point_total : Nat := two_point_shots * points_per_two_shot
def combined_point_total : Nat := three_point_total + two_point_total

-- Formulate the theorem to prove the number of one-point shots Tyson made
theorem one_point_shots_count : combined_point_total <= total_points →
  (total_points - combined_point_total = 6) :=
by 
  -- Skip the proof
  sorry

end one_point_shots_count_l231_23110


namespace total_birds_count_l231_23170

def cage1_parrots := 9
def cage1_finches := 4
def cage1_canaries := 7

def cage2_parrots := 5
def cage2_parakeets := 8
def cage2_finches := 10

def cage3_parakeets := 15
def cage3_finches := 7
def cage3_canaries := 3

def cage4_parrots := 10
def cage4_parakeets := 5
def cage4_finches := 12

def total_birds := cage1_parrots + cage1_finches + cage1_canaries +
                   cage2_parrots + cage2_parakeets + cage2_finches +
                   cage3_parakeets + cage3_finches + cage3_canaries +
                   cage4_parrots + cage4_parakeets + cage4_finches

theorem total_birds_count : total_birds = 95 :=
by
  -- Proof is omitted here.
  sorry

end total_birds_count_l231_23170


namespace picking_time_l231_23169

theorem picking_time (x : ℝ) 
  (h_wang : x * 8 - 0.25 = x * 7) : 
  x = 0.25 := 
by
  sorry

end picking_time_l231_23169


namespace john_allowance_calculation_l231_23121

theorem john_allowance_calculation (initial_money final_money game_cost allowance: ℕ) 
(h_initial: initial_money = 5) 
(h_game_cost: game_cost = 2) 
(h_final: final_money = 29) 
(h_allowance: final_money = initial_money - game_cost + allowance) : 
  allowance = 26 :=
by
  sorry

end john_allowance_calculation_l231_23121


namespace increase_in_rectangle_area_l231_23195

theorem increase_in_rectangle_area (L B : ℝ) :
  let L' := 1.11 * L
  let B' := 1.22 * B
  let original_area := L * B
  let new_area := L' * B'
  let area_increase := new_area - original_area
  let percentage_increase := (area_increase / original_area) * 100
  percentage_increase = 35.42 :=
by
  sorry

end increase_in_rectangle_area_l231_23195


namespace max_tickets_jane_can_buy_l231_23103

-- Define ticket prices and Jane's budget
def ticket_price := 15
def discounted_price := 12
def discount_threshold := 5
def jane_budget := 150

-- Prove that the maximum number of tickets Jane can buy is 11
theorem max_tickets_jane_can_buy : 
  ∃ (n : ℕ), n ≤ 11 ∧ (if n ≤ discount_threshold then ticket_price * n ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (n - discount_threshold)) ≤ jane_budget)
  ∧ ∀ m : ℕ, (if m ≤ 11 then (if m ≤ discount_threshold then ticket_price * m ≤ jane_budget else (ticket_price * discount_threshold + discounted_price * (m - discount_threshold)) ≤ jane_budget) else false)  → m ≤ 11 := 
by
  sorry

end max_tickets_jane_can_buy_l231_23103


namespace cost_price_watch_l231_23120

variable (cost_price : ℚ)

-- Conditions
def sold_at_loss (cost_price : ℚ) := 0.90 * cost_price
def sold_at_gain (cost_price : ℚ) := 1.03 * cost_price
def price_difference (cost_price : ℚ) := sold_at_gain cost_price - sold_at_loss cost_price = 140

-- Theorem
theorem cost_price_watch (h : price_difference cost_price) : cost_price = 1076.92 := by
  sorry

end cost_price_watch_l231_23120


namespace gcd_polynomial_997_l231_23123

theorem gcd_polynomial_997 (b : ℤ) (h : ∃ k : ℤ, b = 997 * k ∧ k % 2 = 1) :
  Int.gcd (3 * b ^ 2 + 17 * b + 31) (b + 7) = 1 := by
  sorry

end gcd_polynomial_997_l231_23123


namespace line_eq1_line_eq2_l231_23111

-- Define the line equations
def l1 (x y : ℝ) : Prop := 4 * x + y + 6 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 5 * y - 6 = 0

-- Theorem for when midpoint is at (0, 0)
theorem line_eq1 : ∀ x y : ℝ, (x + 6 * y = 0) ↔
  ∃ (a : ℝ), 
    l1 a (-(a / 6)) ∧
    l2 (-a) ((a / 6)) ∧
    (a + -a = 0) ∧ (-(a / 6) + a / 6 = 0) := 
by 
  sorry

-- Theorem for when midpoint is at (0, 1)
theorem line_eq2 : ∀ x y : ℝ, (x + 2 * y - 2 = 0) ↔
  ∃ (b : ℝ),
    l1 b (-b / 2 + 1) ∧
    l2 (-b) (1 - (-b / 2)) ∧
    (b + -b = 0) ∧ (-b / 2 + 1 + (1 - (-b / 2)) = 2) := 
by 
  sorry

end line_eq1_line_eq2_l231_23111


namespace unknown_number_is_six_l231_23112

theorem unknown_number_is_six (n : ℝ) (h : 12 * n^4 / 432 = 36) : n = 6 :=
by 
  -- This will be the placeholder for the proof
  sorry

end unknown_number_is_six_l231_23112


namespace max_value_of_linear_combination_l231_23165

theorem max_value_of_linear_combination
  (x y : ℝ)
  (h : x^2 + y^2 = 16 * x + 8 * y + 10) :
  ∃ z, z = 4.58 ∧ (∀ x y, (4 * x + 3 * y) ≤ z ∧ (x^2 + y^2 = 16 * x + 8 * y + 10) → (4 * x + 3 * y) ≤ 4.58) :=
by
  sorry

end max_value_of_linear_combination_l231_23165


namespace leq_sum_l231_23164

open BigOperators

theorem leq_sum (x : Fin 3 → ℝ) (hx_pos : ∀ i, 0 < x i) (hx_sum : ∑ i, x i = 1) :
  (∑ i, 1 / (1 + (x i)^2)) ≤ 27 / 10 :=
sorry

end leq_sum_l231_23164


namespace range_a_f_x_neg_l231_23146

noncomputable def f (a x : ℝ) : ℝ := x^2 + (2 * a - 1) * x - 3

theorem range_a_f_x_neg (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ f a x < 0) → a < 3 / 2 := sorry

end range_a_f_x_neg_l231_23146


namespace number_of_games_in_complete_season_l231_23151

-- Define the number of teams in each division
def teams_in_division_A : Nat := 6
def teams_in_division_B : Nat := 7
def teams_in_division_C : Nat := 5

-- Define the number of games each team must play within their division
def games_per_team_within_division (teams : Nat) : Nat :=
  (teams - 1) * 2

-- Calculate the total number of games within a division
def total_games_within_division (teams : Nat) : Nat :=
  (games_per_team_within_division teams * teams) / 2

-- Calculate cross-division games for a team in one division
def cross_division_games_per_team (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  (teams_other_div1 + teams_other_div2) * 2

-- Calculate total cross-division games from all teams in one division
def total_cross_division_games (teams_div : Nat) (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  cross_division_games_per_team teams_other_div1 teams_other_div2 * teams_div

-- Given conditions translated to definitions
def games_in_division_A : Nat := total_games_within_division teams_in_division_A
def games_in_division_B : Nat := total_games_within_division teams_in_division_B
def games_in_division_C : Nat := total_games_within_division teams_in_division_C

def cross_division_games_A : Nat := total_cross_division_games teams_in_division_A teams_in_division_B teams_in_division_C
def cross_division_games_B : Nat := total_cross_division_games teams_in_division_B teams_in_division_A teams_in_division_C
def cross_division_games_C : Nat := total_cross_division_games teams_in_division_C teams_in_division_A teams_in_division_B

-- Total cross-division games with each game counted twice
def total_cross_division_games_in_season : Nat :=
  (cross_division_games_A + cross_division_games_B + cross_division_games_C) / 2

-- Total number of games in the season
def total_games_in_season : Nat :=
  games_in_division_A + games_in_division_B + games_in_division_C + total_cross_division_games_in_season

-- The final proof statement
theorem number_of_games_in_complete_season : total_games_in_season = 306 :=
by
  -- This is the place where the proof would go if it were required.
  sorry

end number_of_games_in_complete_season_l231_23151


namespace constant_term_2x3_minus_1_over_sqrtx_pow_7_l231_23175

noncomputable def constant_term_in_expansion (n : ℕ) (x : ℝ) : ℝ :=
  (2 : ℝ) * (Nat.choose 7 6 : ℝ)

theorem constant_term_2x3_minus_1_over_sqrtx_pow_7 :
  constant_term_in_expansion 7 (2 : ℝ) = 14 :=
by
  -- proof is omitted
  sorry

end constant_term_2x3_minus_1_over_sqrtx_pow_7_l231_23175


namespace initial_garrison_men_l231_23140

theorem initial_garrison_men (M : ℕ) (h1 : 62 * M = 62 * M) 
  (h2 : M * 47 = (M + 2700) * 20) : M = 2000 := by
  sorry

end initial_garrison_men_l231_23140


namespace jellybeans_needed_l231_23168

theorem jellybeans_needed (n : ℕ) : (n ≥ 120 ∧ n % 15 = 14) → n = 134 :=
by sorry

end jellybeans_needed_l231_23168


namespace problem1_problem2_l231_23172

-- Problem 1: Prove the expression equals the calculated value
theorem problem1 : (-2:ℝ)^0 + (1 / Real.sqrt 2) - Real.sqrt 9 = (Real.sqrt 2) / 2 - 2 :=
by sorry

-- Problem 2: Prove the solution to the system of linear equations
theorem problem2 (x y : ℝ) (h1 : 2 * x - y = 3) (h2 : x + y = -2) :
  x = 1/3 ∧ y = -(7/3) :=
by sorry

end problem1_problem2_l231_23172


namespace original_number_of_boys_l231_23185

theorem original_number_of_boys (n : ℕ) (W : ℕ) 
  (h1 : W = n * 35) 
  (h2 : W + 40 = (n + 1) * 36) 
  : n = 4 :=
sorry

end original_number_of_boys_l231_23185


namespace ratio_turkeys_to_ducks_l231_23114

theorem ratio_turkeys_to_ducks (chickens ducks turkeys total_birds : ℕ)
  (h1 : chickens = 200)
  (h2 : ducks = 2 * chickens)
  (h3 : total_birds = 1800)
  (h4 : total_birds = chickens + ducks + turkeys) :
  (turkeys : ℚ) / ducks = 3 := by
sorry

end ratio_turkeys_to_ducks_l231_23114


namespace calculate_gallons_of_milk_l231_23122

-- Definitions of the given constants and conditions
def price_of_soup : Nat := 2
def price_of_bread : Nat := 5
def price_of_cereal : Nat := 3
def price_of_milk : Nat := 4
def total_amount_paid : Nat := 4 * 10

-- Calculation of total cost of non-milk items
def total_cost_non_milk : Nat :=
  (6 * price_of_soup) + (2 * price_of_bread) + (2 * price_of_cereal)

-- The function to calculate the remaining amount to be spent on milk
def remaining_amount : Nat := total_amount_paid - total_cost_non_milk

-- Statement to compute the number of gallons of milk
def gallons_of_milk (remaining : Nat) (price_per_gallon : Nat) : Nat :=
  remaining / price_per_gallon

-- Proof theorem statement (no implementation required, proof skipped)
theorem calculate_gallons_of_milk : 
  gallons_of_milk remaining_amount price_of_milk = 3 := 
by
  sorry

end calculate_gallons_of_milk_l231_23122


namespace math_problem_l231_23155

theorem math_problem:
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) * (5^32 + 7^32) * (5^64 + 7^64) = 7^128 - 5^128 :=
by
  sorry

end math_problem_l231_23155


namespace apple_cost_l231_23189

theorem apple_cost (x l q : ℝ) 
  (h1 : 10 * l = 3.62) 
  (h2 : x * l + (33 - x) * q = 11.67)
  (h3 : x * l + (36 - x) * q = 12.48) : 
  x = 30 :=
by
  sorry

end apple_cost_l231_23189


namespace almost_square_as_quotient_l231_23171

-- Defining what almost squares are
def isAlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

-- Statement of the theorem
theorem almost_square_as_quotient (n : ℕ) (hn : n > 0) :
  ∃ a b : ℕ, isAlmostSquare a ∧ isAlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end almost_square_as_quotient_l231_23171


namespace f_log_sum_l231_23143

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x ^ 2) - x) + 2

theorem f_log_sum (x : ℝ) : f (Real.log 5) + f (Real.log (1 / 5)) = 4 :=
by
  sorry

end f_log_sum_l231_23143


namespace area_of_plot_area_in_terms_of_P_l231_23104

-- Conditions and definitions.
variables (P : ℝ) (l w : ℝ)
noncomputable def perimeter := 2 * (l + w)
axiom h_perimeter : perimeter l w = 120
axiom h_equality : l = 2 * w

-- Proofs statements
theorem area_of_plot : l + w = 60 → l = 2 * w → (4 * w)^2 = 6400 := by
  sorry

theorem area_in_terms_of_P : (4 * (P / 6))^2 = (2 * P / 3)^2 → (2 * P / 3)^2 = 4 * P^2 / 9 := by
  sorry

end area_of_plot_area_in_terms_of_P_l231_23104


namespace digit_A_of_3AA1_divisible_by_9_l231_23160

theorem digit_A_of_3AA1_divisible_by_9 (A : ℕ) (h : (3 + A + A + 1) % 9 = 0) : A = 7 :=
sorry

end digit_A_of_3AA1_divisible_by_9_l231_23160


namespace third_term_of_geometric_sequence_l231_23198

theorem third_term_of_geometric_sequence
  (a₁ : ℕ) (a₄ : ℕ)
  (h1 : a₁ = 5)
  (h4 : a₄ = 320) :
  ∃ a₃ : ℕ, a₃ = 80 :=
by
  sorry

end third_term_of_geometric_sequence_l231_23198


namespace isosceles_triangle_l231_23145

theorem isosceles_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h : a + b = (Real.tan (C / 2)) * (a * Real.tan A + b * Real.tan B)) :
  A = B := 
sorry

end isosceles_triangle_l231_23145


namespace arithmetic_sqrt_of_4_l231_23186

theorem arithmetic_sqrt_of_4 : ∃ x : ℚ, x^2 = 4 ∧ x > 0 → x = 2 :=
by {
  sorry
}

end arithmetic_sqrt_of_4_l231_23186


namespace total_cost_of_shirts_l231_23115

theorem total_cost_of_shirts 
    (first_shirt_cost : ℤ)
    (second_shirt_cost : ℤ)
    (h1 : first_shirt_cost = 15)
    (h2 : first_shirt_cost = second_shirt_cost + 6) : 
    first_shirt_cost + second_shirt_cost = 24 := 
by
  sorry

end total_cost_of_shirts_l231_23115


namespace initial_pairs_l231_23197

variable (p1 p2 p3 p4 p_initial : ℕ)

def week1_pairs := 12
def week2_pairs := week1_pairs + 4
def week3_pairs := (week1_pairs + week2_pairs) / 2
def week4_pairs := week3_pairs - 3
def total_pairs := 57

theorem initial_pairs :
  let p1 := week1_pairs
  let p2 := week2_pairs
  let p3 := week3_pairs
  let p4 := week4_pairs
  p1 + p2 + p3 + p4 + p_initial = 57 → p_initial = 4 :=
by
  sorry

end initial_pairs_l231_23197


namespace problem_even_and_monotonically_increasing_l231_23192

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem problem_even_and_monotonically_increasing :
  is_even_function (fun x => Real.exp (|x|)) ∧ is_monotonically_increasing_on (fun x => Real.exp (|x|)) (Set.Ioo 0 1) :=
by
  sorry

end problem_even_and_monotonically_increasing_l231_23192


namespace chocolate_bars_count_l231_23181

theorem chocolate_bars_count (milk_chocolate dark_chocolate almond_chocolate white_chocolate : ℕ)
    (h_milk : milk_chocolate = 25)
    (h_almond : almond_chocolate = 25)
    (h_white : white_chocolate = 25)
    (h_percent : milk_chocolate = almond_chocolate ∧ almond_chocolate = white_chocolate ∧ white_chocolate = dark_chocolate) :
    dark_chocolate = 25 := by
  sorry

end chocolate_bars_count_l231_23181


namespace y_coordinate_in_fourth_quadrant_l231_23141
-- Importing the necessary libraries

-- Definition of the problem statement
theorem y_coordinate_in_fourth_quadrant (x y : ℝ) (h : x = 5 ∧ y < 0) : y < 0 :=
by 
  sorry

end y_coordinate_in_fourth_quadrant_l231_23141


namespace determine_c_l231_23179

-- Assume we have three integers a, b, and unique x, y, z such that
variables (a b c x y z : ℕ)

-- Define the conditions
def condition1 : Prop := a = Nat.lcm y z
def condition2 : Prop := b = Nat.lcm x z
def condition3 : Prop := c = Nat.lcm x y

-- Prove that Bob can determine c based on a and b
theorem determine_c (h1 : condition1 a y z) (h2 : condition2 b x z) (h3 : ∀ u v w : ℕ, (Nat.lcm u w = a ∧ Nat.lcm v w = b ∧ Nat.lcm u v = c) → (u = x ∧ v = y ∧ w = z) ) : ∃ c, condition3 c x y :=
by sorry

end determine_c_l231_23179


namespace basketball_children_l231_23106

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l231_23106


namespace sqrt_0_09_eq_0_3_l231_23194

theorem sqrt_0_09_eq_0_3 : Real.sqrt 0.09 = 0.3 := 
by 
  sorry

end sqrt_0_09_eq_0_3_l231_23194


namespace find_fraction_l231_23184

theorem find_fraction (x f : ℝ) (h₁ : x = 140) (h₂ : 0.65 * x = f * x - 21) : f = 0.8 :=
by
  sorry

end find_fraction_l231_23184


namespace stadium_fee_difference_l231_23199

theorem stadium_fee_difference :
  let capacity := 2000
  let entry_fee := 20
  let full_fees := capacity * entry_fee
  let three_quarters_fees := (capacity * 3 / 4) * entry_fee
  full_fees - three_quarters_fees = 10000 :=
by
  sorry

end stadium_fee_difference_l231_23199


namespace rectangle_volume_l231_23173

theorem rectangle_volume {a b c : ℕ} (h1 : a * b - c * a - b * c = 1) (h2 : c * a = b * c + 1) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a * b * c = 6 :=
sorry

end rectangle_volume_l231_23173


namespace largest_additional_plates_l231_23159

theorem largest_additional_plates
  (initial_first_set_size : ℕ)
  (initial_second_set_size : ℕ)
  (initial_third_set_size : ℕ)
  (new_letters : ℕ)
  (constraint : 1 ≤ initial_second_set_size + 1 ∧ 1 ≤ initial_third_set_size + 1)
  (initial_combinations : ℕ)
  (final_combinations1 : ℕ)
  (final_combinations2 : ℕ)
  (additional_combinations : ℕ) :
  initial_first_set_size = 5 →
  initial_second_set_size = 3 →
  initial_third_set_size = 4 →
  new_letters = 4 →
  initial_combinations = initial_first_set_size * initial_second_set_size * initial_third_set_size →
  final_combinations1 = initial_first_set_size * (initial_second_set_size + 2) * (initial_third_set_size + 2) →
  final_combinations2 = (initial_first_set_size + 1) * (initial_second_set_size + 2) * (initial_third_set_size + 1) →
  additional_combinations = max (final_combinations1 - initial_combinations) (final_combinations2 - initial_combinations) →
  additional_combinations = 90 :=
by sorry

end largest_additional_plates_l231_23159


namespace simple_interest_is_correct_l231_23134

def Principal : ℝ := 10000
def Rate : ℝ := 0.09
def Time : ℝ := 1

theorem simple_interest_is_correct :
  Principal * Rate * Time = 900 := by
  sorry

end simple_interest_is_correct_l231_23134


namespace number_of_teachers_at_Queen_Middle_School_l231_23149

-- Conditions
def num_students : ℕ := 1500
def classes_per_student : ℕ := 6
def classes_per_teacher : ℕ := 5
def students_per_class : ℕ := 25

-- Proof that the number of teachers is 72
theorem number_of_teachers_at_Queen_Middle_School :
  (num_students * classes_per_student) / students_per_class / classes_per_teacher = 72 :=
by sorry

end number_of_teachers_at_Queen_Middle_School_l231_23149


namespace characteristic_triangle_smallest_angle_l231_23163

theorem characteristic_triangle_smallest_angle 
  (α β : ℝ)
  (h1 : α = 2 * β)
  (h2 : α = 100)
  (h3 : β + α + γ = 180) : 
  min α (min β γ) = 30 := 
by 
  sorry

end characteristic_triangle_smallest_angle_l231_23163


namespace students_participated_in_function_l231_23153

theorem students_participated_in_function :
  ∀ (B G : ℕ),
  B + G = 800 →
  (3 / 4 : ℚ) * G = 150 →
  (2 / 3 : ℚ) * B + 150 = 550 :=
by
  intros B G h1 h2
  sorry

end students_participated_in_function_l231_23153


namespace no_function_exists_l231_23162

-- Main theorem statement
theorem no_function_exists : ¬ ∃ f : ℝ → ℝ, 
  (∀ x y : ℝ, 0 < x → 0 < y → (x + y) * f (2 * y * f x + f y) = x^3 * f (y * f x)) ∧ 
  (∀ z : ℝ, 0 < z → f z > 0) :=
sorry

end no_function_exists_l231_23162


namespace triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l231_23100

-- Given the conditions: two sides of one triangle are equal to two sides of another triangle.
-- And an angle opposite to one of these sides is equal to the angle opposite to the corresponding side.
variables {A B C D E F : Type}
variables {AB DE BC EF : ℝ} (h_AB_DE : AB = DE) (h_BC_EF : BC = EF)
variables {angle_A angle_D : ℝ} (h_angle_A_D : angle_A = angle_D)

-- Prove that the triangles may or may not be congruent
theorem triangles_may_or_may_not_be_congruent :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_be_congruent_or_not : Prop) :=
sorry

-- Prove that the triangles may have equal areas
theorem triangles_may_have_equal_areas :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_have_equal_areas : Prop) :=
sorry

end triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l231_23100


namespace shirts_washed_total_l231_23144

theorem shirts_washed_total (short_sleeve_shirts long_sleeve_shirts : Nat) (h1 : short_sleeve_shirts = 4) (h2 : long_sleeve_shirts = 5) : short_sleeve_shirts + long_sleeve_shirts = 9 := by
  sorry

end shirts_washed_total_l231_23144


namespace correct_value_division_l231_23109

theorem correct_value_division (x : ℕ) (h : 9 - x = 3) : 96 / x = 16 :=
by
  sorry

end correct_value_division_l231_23109


namespace find_a_l231_23180

-- Define the function f(x)
def f (a : ℚ) (x : ℚ) : ℚ := x^2 + (2 * a + 3) * x + (a^2 + 1)

-- State that the discriminant of f(x) is non-negative
def discriminant_nonnegative (a : ℚ) : Prop :=
  let Δ := (2 * a + 3)^2 - 4 * (a^2 + 1)
  Δ ≥ 0

-- Final statement expressing the final condition on a and the desired result |p| + |q|
theorem find_a (a : ℚ) (p q : ℤ) (h_relprime : Int.gcd p q = 1) (h_eq : a = -5 / 12) (h_abs : p * q = -5 * 12) :
  discriminant_nonnegative a →
  |p| + |q| = 17 :=
by sorry

end find_a_l231_23180


namespace min_odd_integers_l231_23142

-- Definitions of the conditions
variable (a b c d e f : ℤ)

-- The mathematical theorem statement
theorem min_odd_integers 
  (h1 : a + b = 30)
  (h2 : a + b + c + d = 50) 
  (h3 : a + b + c + d + e + f = 70)
  (h4 : e + f % 2 = 1) : 
  ∃ n, n ≥ 1 ∧ n = (if a % 2 = 1 then 1 else 0) + (if b % 2 = 1 then 1 else 0) + 
                    (if c % 2 = 1 then 1 else 0) + (if d % 2 = 1 then 1 else 0) + 
                    (if e % 2 = 1 then 1 else 0) + (if f % 2 = 1 then 1 else 0) :=
sorry

end min_odd_integers_l231_23142


namespace points_collinear_sum_l231_23196

theorem points_collinear_sum (x y : ℝ) :
  ∃ k : ℝ, (x - 1 = 3 * k ∧ 1 = k * (y - 2) ∧ -1 = 2 * k) → 
  x + y = -1 / 2 :=
by
  sorry

end points_collinear_sum_l231_23196


namespace equal_number_of_boys_and_girls_l231_23124

theorem equal_number_of_boys_and_girls
  (m d M D : ℕ)
  (h1 : (M / m) ≠ (D / d))
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : m = d :=
sorry

end equal_number_of_boys_and_girls_l231_23124


namespace probability_diamond_then_ace_l231_23167

theorem probability_diamond_then_ace :
  let total_cards := 104
  let diamonds := 26
  let aces := 8
  let remaining_cards_after_first_draw := total_cards - 1
  let ace_of_diamonds_prob := (2 : ℚ) / total_cards
  let any_ace_after_ace_of_diamonds := (7 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_ace_of_diamonds_then_any_ace := ace_of_diamonds_prob * any_ace_after_ace_of_diamonds
  let diamond_not_ace_prob := (24 : ℚ) / total_cards
  let any_ace_after_diamond_not_ace := (8 : ℚ) / remaining_cards_after_first_draw
  let combined_prob_diamond_not_ace_then_any_ace := diamond_not_ace_prob * any_ace_after_diamond_not_ace
  let total_prob := combined_prob_ace_of_diamonds_then_any_ace + combined_prob_diamond_not_ace_then_any_ace
  total_prob = (31 : ℚ) / 5308 :=
by
  sorry

end probability_diamond_then_ace_l231_23167


namespace min_major_axis_ellipse_l231_23139

theorem min_major_axis_ellipse (a b c : ℝ) (h1 : b * c = 1) (h2 : a^2 = b^2 + c^2) :
  2 * a ≥ 2 * Real.sqrt 2 :=
by {
  sorry
}

end min_major_axis_ellipse_l231_23139


namespace triangle_area_l231_23182

theorem triangle_area (B : Real) (AB AC : Real) 
  (hB : B = Real.pi / 6) 
  (hAB : AB = 2 * Real.sqrt 3)
  (hAC : AC = 2) : 
  let area := 1 / 2 * AB * AC * Real.sin B
  area = 2 * Real.sqrt 3 := by
  sorry

end triangle_area_l231_23182


namespace value_of_a_l231_23125

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, (x^2 - 4*a*x + 5*a^2 - 6*a = 0 → 
    ∃ x₁ x₂, x₁ + x₂ = 4*a ∧ x₁ * x₂ = 5*a^2 - 6*a ∧ |x₁ - x₂| = 6)) → a = 3 :=
by {
  sorry
}

end value_of_a_l231_23125


namespace hyperbola_slope_of_asymptote_positive_value_l231_23190

noncomputable def hyperbola_slope_of_asymptote (x y : ℝ) : ℝ :=
  if h : (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4)
  then (Real.sqrt 5) / 2
  else 0

-- Statement of the mathematically equivalent proof problem
theorem hyperbola_slope_of_asymptote_positive_value :
  ∃ x y : ℝ, (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) ∧
  hyperbola_slope_of_asymptote x y = (Real.sqrt 5) / 2 :=
by
  sorry

end hyperbola_slope_of_asymptote_positive_value_l231_23190


namespace cost_of_45_roses_l231_23126

theorem cost_of_45_roses (cost_15_roses : ℕ → ℝ) 
  (h1 : cost_15_roses 15 = 25) 
  (h2 : ∀ (n m : ℕ), cost_15_roses n / n = cost_15_roses m / m )
  (h3 : ∀ (n : ℕ), n > 30 → cost_15_roses n = (1 - 0.10) * cost_15_roses n) :
  cost_15_roses 45 = 67.5 :=
by
  sorry

end cost_of_45_roses_l231_23126


namespace find_x_in_average_l231_23117

theorem find_x_in_average (x : ℝ) :
  (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + x) / 9 = 207 → x = 217 :=
by
  intro h
  sorry

end find_x_in_average_l231_23117


namespace brother_paint_time_is_4_l231_23178

noncomputable def brother_paint_time (B : ℝ) : Prop :=
  (1 / 3) + (1 / B) = 1 / 1.714

theorem brother_paint_time_is_4 : ∃ B, brother_paint_time B ∧ abs (B - 4) < 0.001 :=
by {
  sorry -- Proof to be filled in later.
}

end brother_paint_time_is_4_l231_23178


namespace line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l231_23177

-- Define the set of people
def people : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : ℕ := 1 
def B : ℕ := 2
def C : ℕ := 3

-- Proof Problem 1: Prove that there are 1800 ways to line up 5 people out of 7 given A must be included.
theorem line_up_including_A : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 2: Prove that there are 1800 ways to line up 5 people out of 7 given A, B, and C are not all included.
theorem line_up_excluding_all_ABC : Finset ℕ → ℕ :=
by
  sorry

-- Proof Problem 3: Prove that there are 144 ways to line up 5 people out of 7 given A, B, and C are all included, A and B are adjacent, and C is not adjacent to A or B.
theorem line_up_adjacent_AB_not_adjacent_C : Finset ℕ → ℕ :=
by
  sorry

end line_up_including_A_line_up_excluding_all_ABC_line_up_adjacent_AB_not_adjacent_C_l231_23177


namespace train_speed_l231_23118

theorem train_speed (v : ℝ) :
  let speed_train1 := 80  -- speed of the first train in km/h
  let length_train1 := 150 / 1000 -- length of the first train in km
  let length_train2 := 100 / 1000 -- length of the second train in km
  let total_time := 5.999520038396928 / 3600 -- time in hours
  let total_length := length_train1 + length_train2 -- total length in km
  let relative_speed := total_length / total_time -- relative speed in km/h
  relative_speed = speed_train1 + v → v = 70 :=
by
  sorry

end train_speed_l231_23118


namespace Megan_bought_24_eggs_l231_23161

def eggs_problem : Prop :=
  ∃ (p c b : ℕ),
    b = 3 ∧
    c = 2 * b ∧
    p - c = 9 ∧
    p + c + b = 24

theorem Megan_bought_24_eggs : eggs_problem :=
  sorry

end Megan_bought_24_eggs_l231_23161


namespace correct_option_is_optionB_l231_23132

-- Definitions based on conditions
def optionA : ℝ := 0.37 * 1.5
def optionB : ℝ := 3.7 * 1.5
def optionC : ℝ := 0.37 * 1500
def original : ℝ := 0.37 * 15

-- Statement to prove that the correct answer (optionB) yields the same result as the original expression
theorem correct_option_is_optionB : optionB = original :=
sorry

end correct_option_is_optionB_l231_23132


namespace sequence_difference_l231_23148

-- Definition of sequences sums
def odd_sum (n : ℕ) : ℕ := (n * n)
def even_sum (n : ℕ) : ℕ := n * (n + 1)

-- Main property to prove
theorem sequence_difference :
  odd_sum 1013 - even_sum 1011 = 3047 :=
by
  -- Definitions and assertions here
  sorry

end sequence_difference_l231_23148
