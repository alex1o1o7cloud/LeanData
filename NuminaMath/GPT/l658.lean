import Mathlib

namespace italian_clock_hand_coincidence_l658_65820

theorem italian_clock_hand_coincidence :
  let hour_hand_rotation := 1 / 24
  let minute_hand_rotation := 1
  ∃ (t : ℕ), 0 ≤ t ∧ t < 24 ∧ (t * hour_hand_rotation) % 1 = (t * minute_hand_rotation) % 1
:= sorry

end italian_clock_hand_coincidence_l658_65820


namespace correct_graph_is_C_l658_65822

-- Define the years and corresponding remote work percentages
def percentages : List (ℕ × ℝ) := [
  (1960, 0.1),
  (1970, 0.15),
  (1980, 0.12),
  (1990, 0.25),
  (2000, 0.4)
]

-- Define the property of the graph trend
def isCorrectGraph (p : List (ℕ × ℝ)) : Prop :=
  p = [
    (1960, 0.1),
    (1970, 0.15),
    (1980, 0.12),
    (1990, 0.25),
    (2000, 0.4)
  ]

-- State the theorem
theorem correct_graph_is_C : isCorrectGraph percentages = True :=
  sorry

end correct_graph_is_C_l658_65822


namespace sector_angle_l658_65849

-- Define the conditions
def perimeter (r l : ℝ) : ℝ := 2 * r + l
def arc_length (α r : ℝ) : ℝ := α * r

-- Define the problem statement
theorem sector_angle (perimeter_eq : perimeter 1 l = 4) (arc_length_eq : arc_length α 1 = l) : α = 2 := 
by 
  -- remainder of the proof can be added here 
  sorry

end sector_angle_l658_65849


namespace opposite_of_2023_l658_65893

def opposite (x : ℤ) : ℤ := -x

theorem opposite_of_2023 : opposite 2023 = -2023 := 
by {
  -- The actual proof here would show that opposite(2023) = -2023
  sorry
}

end opposite_of_2023_l658_65893


namespace range_of_p_l658_65873

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - 10 * x

-- A = { x | f'(x) ≤ 0 }
def A : Set ℝ := { x | deriv f x ≤ 0 }

-- B = { x | p + 1 ≤ x ≤ 2p - 1 }
def B (p : ℝ) : Set ℝ := { x | p + 1 ≤ x ∧ x ≤ 2 * p - 1 }

-- Given that A ∪ B = A, prove the range of values for p is ≤ 3.
theorem range_of_p (p : ℝ) : (A ∪ B p = A) → p ≤ 3 := sorry

end range_of_p_l658_65873


namespace min_c_value_l658_65865

def y_eq_abs_sum (x a b c : ℝ) : ℝ := |x - a| + |x - b| + |x - c|
def y_eq_line (x : ℝ) : ℝ := -2 * x + 2023

theorem min_c_value (a b c : ℕ) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (order : a ≤ b ∧ b < c)
  (unique_sol : ∃! x : ℝ, y_eq_abs_sum x a b c = y_eq_line x) :
  c = 2022 := sorry

end min_c_value_l658_65865


namespace balls_in_boxes_l658_65884

theorem balls_in_boxes : 
  ∀ (n k : ℕ), n = 6 ∧ k = 3 ∧ ∀ i, i < k → 1 ≤ i → 
             ( ∃ ways : ℕ, ways = Nat.choose ((n - k) + k - 1) (k - 1) ∧ ways = 10 ) :=
by
  sorry

end balls_in_boxes_l658_65884


namespace simplify_fraction_l658_65801

theorem simplify_fraction (n : ℕ) : 
  (3 ^ (n + 3) - 3 * (3 ^ n)) / (3 * 3 ^ (n + 2)) = 8 / 9 :=
by sorry

end simplify_fraction_l658_65801


namespace pass_rate_eq_l658_65809

theorem pass_rate_eq (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : (1 - a) * (1 - b) = ab - a - b + 1 :=
by
  sorry

end pass_rate_eq_l658_65809


namespace mode_is_3_5_of_salaries_l658_65800

def salaries : List ℚ := [30, 14, 9, 6, 4, 3.5, 3]
def frequencies : List ℕ := [1, 2, 3, 4, 5, 6, 4]

noncomputable def mode_of_salaries (salaries : List ℚ) (frequencies : List ℕ) : ℚ :=
by
  sorry

theorem mode_is_3_5_of_salaries :
  mode_of_salaries salaries frequencies = 3.5 :=
by
  sorry

end mode_is_3_5_of_salaries_l658_65800


namespace coin_order_correct_l658_65815

-- Define the coins
inductive Coin
| A | B | C | D | E
deriving DecidableEq

open Coin

-- Define the conditions
def covers (x y : Coin) : Prop :=
  (x = A ∧ y = B) ∨
  (x = C ∧ (y = A ∨ y = D)) ∨
  (x = D ∧ y = B) ∨
  (y = E ∧ x = C)

-- Define the order of coins from top to bottom as a list
def coinOrder : List Coin := [C, E, A, D, B]

-- Prove that the order is correct
theorem coin_order_correct :
  ∀ c₁ c₂ : Coin, c₁ ≠ c₂ → List.indexOf c₁ coinOrder < List.indexOf c₂ coinOrder ↔ covers c₁ c₂ :=
by
  sorry

end coin_order_correct_l658_65815


namespace number_of_men_l658_65842

variable (W M : ℝ)
variable (N_women N_men : ℕ)

theorem number_of_men (h1 : M = 2 * W)
  (h2 : N_women * W * 30 = 21600) :
  (N_men * M * 20 = 14400) → N_men = N_women / 3 :=
by
  sorry

end number_of_men_l658_65842


namespace math_problem_l658_65898

open Real

-- Conditions extracted from the problem
def cond1 (a b : ℝ) : Prop := -|2 - a| + b = 5
def cond2 (a b : ℝ) : Prop := -|8 - a| + b = 3
def cond3 (c d : ℝ) : Prop := |2 - c| + d = 5
def cond4 (c d : ℝ) : Prop := |8 - c| + d = 3
def cond5 (a c : ℝ) : Prop := 2 < a ∧ a < 8
def cond6 (a c : ℝ) : Prop := 2 < c ∧ c < 8

-- Proof problem: Given the conditions, prove that a + c = 10
theorem math_problem (a b c d : ℝ) (h1 : cond1 a b) (h2 : cond2 a b) (h3 : cond3 c d) (h4 : cond4 c d)
  (h5 : cond5 a c) (h6 : cond6 a c) : a + c = 10 := 
by
  sorry

end math_problem_l658_65898


namespace find_n_l658_65857

open Classical

theorem find_n (n : ℕ) (h : (8 * Nat.choose n 3) = 8 * (2 * Nat.choose n 1)) : n = 5 := by
  sorry

end find_n_l658_65857


namespace rook_placements_5x5_l658_65860

/-- The number of ways to place five distinct rooks on a 
  5x5 chess board such that each column and row of the 
  board contains exactly one rook is 120. -/
theorem rook_placements_5x5 : 
  ∃! (f : Fin 5 → Fin 5), Function.Bijective f :=
by
  sorry

end rook_placements_5x5_l658_65860


namespace trader_profit_percentage_l658_65877

-- Definitions for the conditions
def trader_buys_weight (indicated_weight: ℝ) : ℝ :=
  1.10 * indicated_weight

def trader_claimed_weight_to_customer (actual_weight: ℝ) : ℝ :=
  1.30 * actual_weight

-- Main theorem statement
theorem trader_profit_percentage (indicated_weight: ℝ) (actual_weight: ℝ) (claimed_weight: ℝ) :
  trader_buys_weight 1000 = 1100 →
  trader_claimed_weight_to_customer actual_weight = claimed_weight →
  claimed_weight = 1000 →
  (1000 - actual_weight) / actual_weight * 100 = 30 :=
by
  intros h1 h2 h3
  sorry

end trader_profit_percentage_l658_65877


namespace inv_sum_mod_l658_65827

theorem inv_sum_mod 
  : (∃ (x y : ℤ), (3 * x ≡ 1 [ZMOD 25]) ∧ (3^2 * y ≡ 1 [ZMOD 25]) ∧ (x + y ≡ 6 [ZMOD 25])) :=
sorry

end inv_sum_mod_l658_65827


namespace rectangle_width_solution_l658_65889

noncomputable def solve_rectangle_width (W L w l : ℝ) :=
  L = 2 * W ∧ 3 * w = W ∧ 2 * l = L ∧ 6 * l * w = 5400

theorem rectangle_width_solution (W L w l : ℝ) :
  solve_rectangle_width W L w l → w = 10 * Real.sqrt 3 :=
by
  sorry

end rectangle_width_solution_l658_65889


namespace total_photos_newspaper_l658_65841

-- Define the conditions
def num_pages1 := 12
def photos_per_page1 := 2
def num_pages2 := 9
def photos_per_page2 := 3

-- Define the total number of photos
def total_photos := (num_pages1 * photos_per_page1) + (num_pages2 * photos_per_page2)

-- Prove that the total number of photos is 51
theorem total_photos_newspaper : total_photos = 51 := by
  -- We will fill the proof later
  sorry

end total_photos_newspaper_l658_65841


namespace projection_of_sum_on_vec_a_l658_65887

open Real

noncomputable def vector_projection (a b : ℝ) (angle : ℝ) : ℝ := 
  (cos angle) * (a * b) / a

theorem projection_of_sum_on_vec_a (a b : EuclideanSpace ℝ (Fin 3)) 
  (h₁ : ‖a‖ = 2) 
  (h₂ : ‖b‖ = 2) 
  (h₃ : inner a b = (2 * 2) * (cos (π / 3))):
  (inner (a + b) a) / ‖a‖ = 3 := 
by
  sorry

end projection_of_sum_on_vec_a_l658_65887


namespace quadratic_function_is_explicit_form_l658_65812

-- Conditions
variable {f : ℝ → ℝ}
variable (H1 : f (-1) = 0)
variable (H2 : ∀ x : ℝ, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2)

-- The quadratic function we aim to prove
def quadratic_function_form_proof (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = (1/4) * x^2 + (1/2) * x + (1/4)

-- Main theorem statement
theorem quadratic_function_is_explicit_form : quadratic_function_form_proof f :=
by
  -- Placeholder for the proof
  sorry

end quadratic_function_is_explicit_form_l658_65812


namespace decomposition_sum_of_cubes_l658_65858

theorem decomposition_sum_of_cubes 
  (a b c d e : ℤ) 
  (h : (512 : ℤ) * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 60 := 
sorry

end decomposition_sum_of_cubes_l658_65858


namespace passengers_in_each_car_l658_65803

theorem passengers_in_each_car (P : ℕ) (h1 : 20 * (P + 2) = 80) : P = 2 := 
by
  sorry

end passengers_in_each_car_l658_65803


namespace vasya_is_not_mistaken_l658_65897

theorem vasya_is_not_mistaken (X Y N A B : ℤ)
  (h_sum : X + Y = N)
  (h_tanya : A * X + B * Y ≡ 0 [ZMOD N]) :
  B * X + A * Y ≡ 0 [ZMOD N] :=
sorry

end vasya_is_not_mistaken_l658_65897


namespace max_value_fraction_l658_65867

theorem max_value_fraction (e a b : ℝ) (h : ∀ x : ℝ, (e - a) * Real.exp x + x + b + 1 ≤ 0) : 
  (b + 1) / a ≤ 1 / e :=
sorry

end max_value_fraction_l658_65867


namespace cost_per_gallon_l658_65802

theorem cost_per_gallon (weekly_spend : ℝ) (two_week_usage : ℝ) (weekly_spend_eq : weekly_spend = 36) (two_week_usage_eq : two_week_usage = 24) : 
  (2 * weekly_spend / two_week_usage) = 3 :=
by sorry

end cost_per_gallon_l658_65802


namespace firstDiscountIsTenPercent_l658_65861

def listPrice : ℝ := 70
def finalPrice : ℝ := 56.16
def secondDiscount : ℝ := 10.857142857142863

theorem firstDiscountIsTenPercent (x : ℝ) : 
    finalPrice = listPrice * (1 - x / 100) * (1 - secondDiscount / 100) ↔ x = 10 := 
by
  sorry

end firstDiscountIsTenPercent_l658_65861


namespace max_value_of_trig_expression_l658_65875

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end max_value_of_trig_expression_l658_65875


namespace rectangle_perimeter_l658_65818

theorem rectangle_perimeter (u v : ℝ) (π : ℝ) (major minor : ℝ) (area_rect area_ellipse : ℝ) 
  (inscribed : area_ellipse = 4032 * π ∧ area_rect = 4032 ∧ major = 2 * (u + v)) :
  2 * (u + v) = 128 := by
  -- Given: the area of the rectangle, the conditions of the inscribed ellipse, and the major axis constraint.
  sorry

end rectangle_perimeter_l658_65818


namespace correct_equation_l658_65876

-- Definitions of the conditions
def january_turnover (T : ℝ) : Prop := T = 36
def march_turnover (T : ℝ) : Prop := T = 48
def average_monthly_growth_rate (x : ℝ) : Prop := True

-- The goal to be proved
theorem correct_equation (T_jan T_mar : ℝ) (x : ℝ) 
  (h_jan : january_turnover T_jan) 
  (h_mar : march_turnover T_mar) 
  (h_growth : average_monthly_growth_rate x) : 
  36 * (1 + x)^2 = 48 :=
sorry

end correct_equation_l658_65876


namespace proof_problem_l658_65811

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 4 * x^2) - 2 * x) + 3

theorem proof_problem : f (Real.log 2) + f (Real.log (1 / 2)) = 6 := 
by 
  sorry

end proof_problem_l658_65811


namespace gage_needs_to_skate_l658_65848

noncomputable def gage_average_skating_time (d1 d2: ℕ) (t1 t2 t8: ℕ) : ℕ :=
  let total_time := (d1 * t1) + (d2 * t2) + t8
  (total_time / (d1 + d2 + 1))

theorem gage_needs_to_skate (t1 t2: ℕ) (d1 d2: ℕ) (avg: ℕ) 
  (t1_minutes: t1 = 80) (t2_minutes: t2 = 105) 
  (days1: d1 = 4) (days2: d2 = 3) (avg_goal: avg = 95) :
  gage_average_skating_time d1 d2 t1 t2 125 = avg :=
by
  sorry

end gage_needs_to_skate_l658_65848


namespace rebecca_soda_left_l658_65854

-- Definitions of the conditions
def total_bottles_purchased : ℕ := 3 * 6
def days_in_four_weeks : ℕ := 4 * 7
def total_half_bottles_drinks : ℕ := days_in_four_weeks
def total_whole_bottles_drinks : ℕ := total_half_bottles_drinks / 2

-- The final statement we aim to prove
theorem rebecca_soda_left : 
  total_bottles_purchased - total_whole_bottles_drinks = 4 := 
by
  -- proof is not required as per the guidelines
  sorry

end rebecca_soda_left_l658_65854


namespace watermelons_with_seeds_l658_65885

def ripe_watermelons : ℕ := 11
def unripe_watermelons : ℕ := 13
def seedless_watermelons : ℕ := 15
def total_watermelons := ripe_watermelons + unripe_watermelons

theorem watermelons_with_seeds :
  total_watermelons - seedless_watermelons = 9 :=
by
  sorry

end watermelons_with_seeds_l658_65885


namespace number_of_white_dogs_l658_65839

noncomputable def number_of_brown_dogs : ℕ := 20
noncomputable def number_of_black_dogs : ℕ := 15
noncomputable def total_number_of_dogs : ℕ := 45

theorem number_of_white_dogs : total_number_of_dogs - (number_of_brown_dogs + number_of_black_dogs) = 10 := by
  sorry

end number_of_white_dogs_l658_65839


namespace total_right_handed_players_l658_65810

theorem total_right_handed_players
  (total_players : ℕ)
  (total_throwers : ℕ)
  (left_handed_throwers_perc : ℕ)
  (right_handed_thrower_runs : ℕ)
  (left_handed_thrower_runs : ℕ)
  (total_runs : ℕ)
  (batsmen_to_allrounders_run_ratio : ℕ)
  (proportion_left_right_non_throwers : ℕ)
  (left_handed_non_thrower_runs : ℕ)
  (left_handed_batsmen_eq_allrounders : Prop)
  (left_handed_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (total_right_handed_thrower_runs : ℕ)
  (total_left_handed_thrower_runs : ℕ)
  (total_throwers_runs : ℕ)
  (total_non_thrower_runs : ℕ)
  (allrounder_runs : ℕ)
  (batsmen_runs : ℕ)
  (left_handed_batsmen : ℕ)
  (left_handed_allrounders : ℕ)
  (total_left_handed_non_throwers : ℕ)
  (right_handed_non_throwers : ℕ)
  (total_right_handed_players : ℕ) :
  total_players = 120 →
  total_throwers = 55 →
  left_handed_throwers_perc = 20 →
  right_handed_thrower_runs = 25 →
  left_handed_thrower_runs = 30 →
  total_runs = 3620 →
  batsmen_to_allrounders_run_ratio = 2 →
  proportion_left_right_non_throwers = 5 →
  left_handed_non_thrower_runs = 720 →
  left_handed_batsmen_eq_allrounders →
  left_handed_throwers = total_throwers * left_handed_throwers_perc / 100 →
  right_handed_throwers = total_throwers - left_handed_throwers →
  total_right_handed_thrower_runs = right_handed_throwers * right_handed_thrower_runs →
  total_left_handed_thrower_runs = left_handed_throwers * left_handed_thrower_runs →
  total_throwers_runs = total_right_handed_thrower_runs + total_left_handed_thrower_runs →
  total_non_thrower_runs = total_runs - total_throwers_runs →
  allrounder_runs = total_non_thrower_runs / (batsmen_to_allrounders_run_ratio + 1) →
  batsmen_runs = batsmen_to_allrounders_run_ratio * allrounder_runs →
  left_handed_batsmen = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  left_handed_allrounders = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  total_left_handed_non_throwers = left_handed_batsmen + left_handed_allrounders →
  right_handed_non_throwers = total_left_handed_non_throwers * proportion_left_right_non_throwers →
  total_right_handed_players = right_handed_throwers + right_handed_non_throwers →
  total_right_handed_players = 164 :=
by sorry

end total_right_handed_players_l658_65810


namespace number_of_dogs_l658_65835

theorem number_of_dogs
    (total_animals : ℕ)
    (dogs_ratio : ℕ) (bunnies_ratio : ℕ) (birds_ratio : ℕ)
    (h_total : total_animals = 816)
    (h_ratio : dogs_ratio = 3 ∧ bunnies_ratio = 9 ∧ birds_ratio = 11) :
    (total_animals / (dogs_ratio + bunnies_ratio + birds_ratio) * dogs_ratio = 105) :=
by
    sorry

end number_of_dogs_l658_65835


namespace total_liquid_consumption_l658_65825

-- Define the given conditions
def elijah_drink_pints : ℝ := 8.5
def emilio_drink_pints : ℝ := 9.5
def isabella_drink_liters : ℝ := 3
def xavier_drink_gallons : ℝ := 2
def pint_to_cups : ℝ := 2
def liter_to_cups : ℝ := 4.22675
def gallon_to_cups : ℝ := 16
def xavier_soda_fraction : ℝ := 0.60
def xavier_fruit_punch_fraction : ℝ := 0.40

-- Define the converted amounts
def elijah_cups := elijah_drink_pints * pint_to_cups
def emilio_cups := emilio_drink_pints * pint_to_cups
def isabella_cups := isabella_drink_liters * liter_to_cups
def xavier_total_cups := xavier_drink_gallons * gallon_to_cups
def xavier_soda_cups := xavier_soda_fraction * xavier_total_cups
def xavier_fruit_punch_cups := xavier_fruit_punch_fraction * xavier_total_cups

-- Total amount calculation
def total_cups := elijah_cups + emilio_cups + isabella_cups + xavier_soda_cups + xavier_fruit_punch_cups

-- Proof statement
theorem total_liquid_consumption : total_cups = 80.68025 := by
  sorry

end total_liquid_consumption_l658_65825


namespace smallest_piece_length_l658_65850

theorem smallest_piece_length (x : ℕ) :
  (9 - x) + (14 - x) ≤ (16 - x) → x ≥ 7 :=
by
  sorry

end smallest_piece_length_l658_65850


namespace reading_time_difference_l658_65836

theorem reading_time_difference
  (tristan_speed : ℕ := 120)
  (ella_speed : ℕ := 40)
  (book_pages : ℕ := 360) :
  let tristan_time := book_pages / tristan_speed
  let ella_time := book_pages / ella_speed
  let time_difference_hours := ella_time - tristan_time
  let time_difference_minutes := time_difference_hours * 60
  time_difference_minutes = 360 :=
by
  sorry

end reading_time_difference_l658_65836


namespace quadratic_inequality_solution_set_l658_65851

theorem quadratic_inequality_solution_set (a b c : ℝ) (h : a ≠ 0) :
  (∀ x : ℝ, a * x^2 + b * x + c < 0) ↔ (a < 0 ∧ (b^2 - 4 * a * c) < 0) :=
by sorry

end quadratic_inequality_solution_set_l658_65851


namespace find_y_intersection_of_tangents_l658_65863

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the tangent slope at a point on the parabola
def tangent_slope (x : ℝ) : ℝ := 2 * (x - 1)

-- Define the perpendicular condition for tangents at points A and B
def perpendicular_condition (a b : ℝ) : Prop := (a - 1) * (b - 1) = -1 / 4

-- Define the y-coordinate of the intersection point P of the tangents at A and B
def y_coordinate_of_intersection (a b : ℝ) : ℝ := a * b - a - b + 2

-- Theorem to be proved
theorem find_y_intersection_of_tangents (a b : ℝ) 
  (ha : parabola a = a ^ 2 - 2 * a - 3) 
  (hb : parabola b = b ^ 2 - 2 * b - 3) 
  (hp : perpendicular_condition a b) :
  y_coordinate_of_intersection a b = -1 / 4 :=
sorry

end find_y_intersection_of_tangents_l658_65863


namespace apples_minimum_count_l658_65852

theorem apples_minimum_count :
  ∃ n : ℕ, n ≡ 2 [MOD 3] ∧ n ≡ 2 [MOD 4] ∧ n ≡ 2 [MOD 5] ∧ n = 62 := by
sorry

end apples_minimum_count_l658_65852


namespace chocolate_bars_cost_l658_65882

variable (n : ℕ) (c : ℕ)

-- Jessica's purchase details
def gummy_bears_packs := 10
def gummy_bears_cost_per_pack := 2
def chocolate_chips_bags := 20
def chocolate_chips_cost_per_bag := 5

-- Calculated costs
def total_gummy_bears_cost := gummy_bears_packs * gummy_bears_cost_per_pack
def total_chocolate_chips_cost := chocolate_chips_bags * chocolate_chips_cost_per_bag

-- Total cost
def total_cost := 150

-- Remaining cost for chocolate bars
def remaining_cost_for_chocolate_bars := total_cost - (total_gummy_bears_cost + total_chocolate_chips_cost)

theorem chocolate_bars_cost (h : remaining_cost_for_chocolate_bars = n * c) : remaining_cost_for_chocolate_bars = 30 :=
by
  sorry

end chocolate_bars_cost_l658_65882


namespace exponential_fixed_point_l658_65847

theorem exponential_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (a^(4-4) + 5 = 6) :=
sorry

end exponential_fixed_point_l658_65847


namespace bananas_eaten_l658_65819

variable (initial_bananas : ℕ) (remaining_bananas : ℕ)

theorem bananas_eaten (initial_bananas remaining_bananas : ℕ) (h_initial : initial_bananas = 12) (h_remaining : remaining_bananas = 10) : initial_bananas - remaining_bananas = 2 := by
  -- Proof goes here
  sorry

end bananas_eaten_l658_65819


namespace triangle_area_l658_65806

-- Define the points P, Q, R and the conditions
structure Point :=
  (x : ℝ)
  (y : ℝ)

def PQR_right_triangle (P Q R : Point) : Prop := 
  (P.x - R.x)^2 + (P.y - R.y)^2 = 24^2 ∧  -- Length PR
  (Q.x - R.x)^2 + (Q.y - R.y)^2 = 73^2 ∧  -- Length RQ
  (P.x - Q.x)^2 + (P.y - Q.y)^2 = 75^2 ∧  -- Hypotenuse PQ
  (P.y = 3 * P.x + 4) ∧                   -- Median through P
  (Q.y = -Q.x + 5)                        -- Median through Q


noncomputable def area (P Q R : Point) : ℝ := 
  0.5 * abs (P.x * (Q.y - R.y) + Q.x * (R.y - P.y) + R.x * (P.y - Q.y))

theorem triangle_area (P Q R : Point) (h : PQR_right_triangle P Q R) : 
  area P Q R = 876 :=
sorry

end triangle_area_l658_65806


namespace cousins_room_distributions_l658_65828

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l658_65828


namespace calculate_r_when_n_is_3_l658_65856

theorem calculate_r_when_n_is_3 : 
  ∀ (r s n : ℕ), r = 4^s - s → s = 3^n + 2 → n = 3 → r = 4^29 - 29 :=
by 
  intros r s n h1 h2 h3
  sorry

end calculate_r_when_n_is_3_l658_65856


namespace arithmetic_geometric_sequence_k4_l658_65830

theorem arithmetic_geometric_sequence_k4 (a : ℕ → ℝ) (d : ℝ) (h_d_ne_zero : d ≠ 0)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_geo_seq : ∃ k : ℕ → ℕ, k 0 = 1 ∧ k 1 = 2 ∧ k 2 = 6 ∧ ∀ i, a (k i + 1) / a (k i) = a (k i + 2) / a (k i + 1)) :
  ∃ k4 : ℕ, k4 = 22 := 
by
  sorry

end arithmetic_geometric_sequence_k4_l658_65830


namespace alloy_gold_content_l658_65896

theorem alloy_gold_content (x : ℝ) (w : ℝ) (p0 p1 : ℝ) (h_w : w = 16)
  (h_p0 : p0 = 0.50) (h_p1 : p1 = 0.80) (h_alloy : x = 24) :
  (p0 * w + x) / (w + x) = p1 :=
by sorry

end alloy_gold_content_l658_65896


namespace inequality_not_always_true_l658_65834

theorem inequality_not_always_true
  (x y w : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x^2 > y^2) (hw : w ≠ 0) :
  ∃ w, w ≠ 0 ∧ x^2 * w ≤ y^2 * w :=
sorry

end inequality_not_always_true_l658_65834


namespace tan_alpha_minus_pi_over_4_l658_65881

variable (α β : ℝ)

-- Given conditions
axiom h1 : Real.tan (α + β) = 2 / 5
axiom h2 : Real.tan β = 1 / 3

-- The goal to prove
theorem tan_alpha_minus_pi_over_4: 
  Real.tan (α - π / 4) = -8 / 9 := by
  sorry

end tan_alpha_minus_pi_over_4_l658_65881


namespace households_with_dvd_player_l658_65832

noncomputable def numHouseholds : ℕ := 100
noncomputable def numWithCellPhone : ℕ := 90
noncomputable def numWithMP3Player : ℕ := 55
noncomputable def greatestWithAllThree : ℕ := 55 -- maximum x
noncomputable def differenceX_Y : ℕ := 25 -- x - y = 25

def numberOfDVDHouseholds : ℕ := 15

theorem households_with_dvd_player : ∀ (D : ℕ),
  D + 25 - D = 55 - 20 →
  D = numberOfDVDHouseholds :=
by
  intro D h
  sorry

end households_with_dvd_player_l658_65832


namespace product_of_roots_of_quadratic_l658_65816

   -- Definition of the quadratic equation used in the condition
   def quadratic (x : ℝ) : ℝ := x^2 - 2 * x - 8

   -- Problem statement: Prove that the product of the roots of the given quadratic equation is -8.
   theorem product_of_roots_of_quadratic : 
     (∀ x : ℝ, quadratic x = 0 → (x = 4 ∨ x = -2)) → (4 * -2 = -8) :=
   by
     sorry
   
end product_of_roots_of_quadratic_l658_65816


namespace find_f3_l658_65880

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_f3 
  (hf : is_odd f) 
  (hg : is_even g) 
  (h : ∀ x, f x + g x = 1 / (x - 1)) : 
  f 3 = 3 / 8 :=
by 
  sorry

end find_f3_l658_65880


namespace total_investment_is_10000_l658_65869

open Real

-- Definitions of conditions
def interest_rate_8 : Real := 0.08
def interest_rate_9 : Real := 0.09
def combined_interest : Real := 840
def investment_8 : Real := 6000
def total_interest (x : Real) : Real := (interest_rate_8 * investment_8 + interest_rate_9 * x)
def investment_9 : Real := 4000

-- Theorem stating the problem
theorem total_investment_is_10000 :
    (∀ x : Real,
        total_interest x = combined_interest → x = investment_9) →
    investment_8 + investment_9 = 10000 := 
by
    intros
    sorry

end total_investment_is_10000_l658_65869


namespace lights_on_bottom_layer_l658_65808

theorem lights_on_bottom_layer
  (a₁ : ℕ)
  (q : ℕ := 3)
  (S₅ : ℕ := 242)
  (n : ℕ := 5)
  (sum_formula : S₅ = (a₁ * (q^n - 1)) / (q - 1)) :
  (a₁ * q^(n-1) = 162) :=
by
  sorry

end lights_on_bottom_layer_l658_65808


namespace nickels_count_l658_65845

theorem nickels_count (original_nickels : ℕ) (additional_nickels : ℕ) 
                        (h₁ : original_nickels = 7) 
                        (h₂ : additional_nickels = 5) : 
    original_nickels + additional_nickels = 12 := 
by sorry

end nickels_count_l658_65845


namespace miki_sandcastle_height_correct_l658_65838

namespace SandcastleHeight

def sister_sandcastle_height := 0.5
def difference_in_height := 0.3333333333333333
def miki_sandcastle_height := sister_sandcastle_height + difference_in_height

theorem miki_sandcastle_height_correct : miki_sandcastle_height = 0.8333333333333333 := by
  unfold miki_sandcastle_height sister_sandcastle_height difference_in_height
  simp
  sorry

end SandcastleHeight

end miki_sandcastle_height_correct_l658_65838


namespace du_chin_fraction_of_sales_l658_65886

theorem du_chin_fraction_of_sales :
  let pies := 200
  let price_per_pie := 20
  let remaining_money := 1600
  let total_sales := pies * price_per_pie
  let used_for_ingredients := total_sales - remaining_money
  let fraction_used_for_ingredients := used_for_ingredients / total_sales
  fraction_used_for_ingredients = (3 / 5) := by
    sorry

end du_chin_fraction_of_sales_l658_65886


namespace route_comparison_l658_65859

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end route_comparison_l658_65859


namespace proof_probability_at_least_one_makes_both_shots_l658_65890

-- Define the shooting percentages for Player A and Player B
def shooting_percentage_A : ℝ := 0.4
def shooting_percentage_B : ℝ := 0.5

-- Define the probability that Player A makes both shots
def prob_A_makes_both_shots : ℝ := shooting_percentage_A * shooting_percentage_A

-- Define the probability that Player B makes both shots
def prob_B_makes_both_shots : ℝ := shooting_percentage_B * shooting_percentage_B

-- Define the probability that neither makes both shots
def prob_neither_makes_both_shots : ℝ := (1 - prob_A_makes_both_shots) * (1 - prob_B_makes_both_shots)

-- Define the probability that at least one of them makes both shots
def prob_at_least_one_makes_both_shots : ℝ := 1 - prob_neither_makes_both_shots

-- Prove that the probability that at least one of them makes both shots is 0.37
theorem proof_probability_at_least_one_makes_both_shots :
  prob_at_least_one_makes_both_shots = 0.37 :=
sorry

end proof_probability_at_least_one_makes_both_shots_l658_65890


namespace max_2ab_plus_2bc_sqrt2_l658_65833

theorem max_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_2ab_plus_2bc_sqrt2_l658_65833


namespace math_proof_problem_l658_65821

noncomputable def M : ℝ :=
  let x := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / (Real.sqrt (Real.sqrt 7 + 2))
  let y := Real.sqrt (5 - 2 * Real.sqrt 6)
  x - y

theorem math_proof_problem :
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 :=
by
  sorry

end math_proof_problem_l658_65821


namespace original_wire_length_l658_65866

theorem original_wire_length (S L : ℝ) (h1: S = 30) (h2: S = (3 / 5) * L) : S + L = 80 := by 
  sorry

end original_wire_length_l658_65866


namespace slices_needed_l658_65813

def number_of_sandwiches : ℕ := 5
def slices_per_sandwich : ℕ := 3
def total_slices_required (n : ℕ) (s : ℕ) : ℕ := n * s

theorem slices_needed : total_slices_required number_of_sandwiches slices_per_sandwich = 15 :=
by
  sorry

end slices_needed_l658_65813


namespace greening_task_equation_l658_65891

variable (x : ℝ)

theorem greening_task_equation (h1 : 600000 = 600 * 1000)
    (h2 : ∀ a b : ℝ, a * 1.25 = b -> b = a * (1 + 25 / 100)) :
  (60 * (1 + 25 / 100)) / x - 60 / x = 30 := by
  sorry

end greening_task_equation_l658_65891


namespace gcd_gx_x_l658_65894

def g (x : ℕ) : ℕ := (5 * x + 3) * (11 * x + 2) * (17 * x + 7) * (3 * x + 8)

theorem gcd_gx_x (x : ℕ) (h : 27720 ∣ x) : Nat.gcd (g x) x = 168 := by
  sorry

end gcd_gx_x_l658_65894


namespace unique_pair_l658_65804

theorem unique_pair (m n : ℕ) (h1 : m < n) (h2 : n ∣ m^2 + 1) (h3 : m ∣ n^2 + 1) : (m, n) = (1, 1) :=
sorry

end unique_pair_l658_65804


namespace projection_of_AB_on_AC_l658_65831

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (0, 3)
noncomputable def C : ℝ × ℝ := (3, 4)

noncomputable def vectorAB := (B.1 - A.1, B.2 - A.2)
noncomputable def vectorAC := (C.1 - A.1, C.2 - A.2)

noncomputable def dotProduct (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem projection_of_AB_on_AC :
  (dotProduct vectorAB vectorAC) / (magnitude vectorAC) = 2 :=
  sorry

end projection_of_AB_on_AC_l658_65831


namespace graph_does_not_pass_first_quadrant_l658_65879

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - 2

theorem graph_does_not_pass_first_quadrant :
  ¬ ∃ x > 0, f x > 0 := by
sorry

end graph_does_not_pass_first_quadrant_l658_65879


namespace top_four_cards_probability_l658_65899

def num_cards : ℕ := 52

def num_hearts : ℕ := 13

def num_diamonds : ℕ := 13

def num_clubs : ℕ := 13

def prob_first_heart := (num_hearts : ℚ) / num_cards
def prob_second_heart := (num_hearts - 1 : ℚ) / (num_cards - 1)
def prob_third_diamond := (num_diamonds : ℚ) / (num_cards - 2)
def prob_fourth_club := (num_clubs : ℚ) / (num_cards - 3)

def combined_prob :=
  prob_first_heart * prob_second_heart * prob_third_diamond * prob_fourth_club

theorem top_four_cards_probability :
  combined_prob = 39 / 63875 := by
  sorry

end top_four_cards_probability_l658_65899


namespace value_of_a_l658_65871

variable (a : ℤ)
def U : Set ℤ := {2, 4, 3 - a^2}
def P : Set ℤ := {2, a^2 + 2 - a}

theorem value_of_a (h : (U a) \ (P a) = {-1}) : a = 2 :=
sorry

end value_of_a_l658_65871


namespace original_average_age_l658_65817

theorem original_average_age (N : ℕ) (A : ℝ) (h1 : A = 50) (h2 : 12 * 32 + N * 50 = (N + 12) * (A - 4)) : A = 50 := by
  sorry 

end original_average_age_l658_65817


namespace sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l658_65855

variable (x : ℝ)

theorem sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) → (|x - 2| < 3) :=
by sorry

theorem not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3 :
        (|x - 2| < 3) → (0 < x ∧ x < 5) :=
by sorry

theorem sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3 :
        (0 < x ∧ x < 5) ↔ (|x - 2| < 3) → false :=
by sorry

end sufficient_but_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_not_necessary_0_lt_x_lt_5_implies_abs_x_minus_2_lt_3_sufficient_but_not_necessary_0_lt_x_lt_5_for_abs_x_minus_2_lt_3_l658_65855


namespace total_amount_spent_is_300_l658_65868

-- Definitions of conditions
def S : ℕ := 97
def H : ℕ := 2 * S + 9

-- The total amount spent
def total_spent : ℕ := S + H

-- Proof statement
theorem total_amount_spent_is_300 : total_spent = 300 :=
by
  sorry

end total_amount_spent_is_300_l658_65868


namespace total_kids_in_camp_l658_65807

-- Definitions from the conditions
variables (X : ℕ)
def kids_going_to_soccer_camp := X / 2
def kids_going_to_soccer_camp_morning := kids_going_to_soccer_camp / 4
def kids_going_to_soccer_camp_afternoon := kids_going_to_soccer_camp - kids_going_to_soccer_camp_morning

-- Given condition that 750 kids are going to soccer camp in the afternoon
axiom h : kids_going_to_soccer_camp_afternoon X = 750

-- The statement to prove that X = 2000
theorem total_kids_in_camp : X = 2000 :=
sorry

end total_kids_in_camp_l658_65807


namespace volume_of_bag_l658_65823

-- Define the dimensions of the cuboid
def width : ℕ := 9
def length : ℕ := 4
def height : ℕ := 7

-- Define the volume calculation function for a cuboid
def volume (l w h : ℕ) : ℕ :=
  l * w * h

-- Provide the theorem to prove the volume is 252 cubic centimeters
theorem volume_of_bag : volume length width height = 252 := by
  -- Since the proof is not requested, insert sorry to complete the statement.
  sorry

end volume_of_bag_l658_65823


namespace seated_ways_alice_between_bob_and_carol_l658_65824

-- Define the necessary entities and conditions for the problem.
def num_people : Nat := 7
def alice := "Alice"
def bob := "Bob"
def carol := "Carol"

-- The main theorem
theorem seated_ways_alice_between_bob_and_carol :
  ∃ (ways : Nat), ways = 48 := by
  sorry

end seated_ways_alice_between_bob_and_carol_l658_65824


namespace sin_double_angle_identity_l658_65872

theorem sin_double_angle_identity (alpha : ℝ) (h : Real.cos (Real.pi / 4 - alpha) = -4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l658_65872


namespace log_minus_one_has_one_zero_l658_65864

theorem log_minus_one_has_one_zero : ∃! x : ℝ, x > 0 ∧ (Real.log x - 1 = 0) :=
sorry

end log_minus_one_has_one_zero_l658_65864


namespace avg_weekly_income_500_l658_65878

theorem avg_weekly_income_500 :
  let base_salary := 350
  let income_past_5_weeks := [406, 413, 420, 436, 495]
  let commission_next_2_weeks_avg := 315
  let total_income_past_5_weeks := income_past_5_weeks.sum
  let total_base_salary_next_2_weeks := base_salary * 2
  let total_commission_next_2_weeks := commission_next_2_weeks_avg * 2
  let total_income := total_income_past_5_weeks + total_base_salary_next_2_weeks + total_commission_next_2_weeks
  let avg_weekly_income := total_income / 7
  avg_weekly_income = 500 := by
{
  sorry
}

end avg_weekly_income_500_l658_65878


namespace change_in_total_berries_l658_65888

theorem change_in_total_berries (B S : ℕ) (hB : B = 20) (hS : S + B = 50) : (S - B) = 10 := by
  sorry

end change_in_total_berries_l658_65888


namespace cost_price_of_one_ball_l658_65840

theorem cost_price_of_one_ball (x : ℝ) (h : 11 * x - 720 = 5 * x) : x = 120 :=
sorry

end cost_price_of_one_ball_l658_65840


namespace math_problem_l658_65837

theorem math_problem (x y : ℝ) :
  let A := x^3 + 3*x^2*y + y^3 - 3*x*y^2
  let B := x^2*y - x*y^2
  A - 3*B = x^3 + y^3 := by
  sorry

end math_problem_l658_65837


namespace intersection_M_N_l658_65844

def M (x : ℝ) : Prop := Real.log x / Real.log 2 ≥ 0
def N (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x | N x} = {x | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l658_65844


namespace price_adjustment_l658_65892

theorem price_adjustment (P : ℝ) (x : ℝ) (hx : P * (1 - (x / 100)^2) = 0.75 * P) : 
  x = 50 :=
by
  -- skipping the proof with sorry
  sorry

end price_adjustment_l658_65892


namespace adult_tickets_sold_l658_65829

theorem adult_tickets_sold (A C : ℕ) (h1 : A + C = 85) (h2 : 5 * A + 2 * C = 275) : A = 35 := by
  sorry

end adult_tickets_sold_l658_65829


namespace geometric_sequence_property_l658_65874

theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ)
  (H_geo : ∀ n, a (n + 1) = a n * q)
  (H_cond1 : a 5 * a 7 = 2)
  (H_cond2 : a 2 + a 10 = 3) :
  (a 12 / a 4 = 2) ∨ (a 12 / a 4 = 1/2) :=
sorry

end geometric_sequence_property_l658_65874


namespace heartsuit_example_l658_65853

def heartsuit (a b : ℤ) : ℤ := a * b^3 - 2 * b + 3

theorem heartsuit_example : heartsuit 2 3 = 51 :=
by
  sorry

end heartsuit_example_l658_65853


namespace faster_pipe_rate_l658_65870

-- Set up our variables and the condition
variable (F S : ℝ)
variable (n : ℕ)

-- Given conditions
axiom S_rate : S = 1 / 180
axiom combined_rate : F + S = 1 / 36
axiom faster_rate : F = n * S

-- Theorem to prove
theorem faster_pipe_rate : n = 4 := by
  sorry

end faster_pipe_rate_l658_65870


namespace problem_statement_l658_65846

variable (f : ℝ → ℝ) 

def prop1 (f : ℝ → ℝ) : Prop := ∃T > 0, T ≠ 3 / 2 ∧ ∀ x, f (x + T) = f x
def prop2 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 3 / 4) = f (-x + 3 / 4)
def prop3 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def prop4 (f : ℝ → ℝ) : Prop := Monotone f

theorem problem_statement (h₁ : ∀ x, f (x + 3 / 2) = -f x)
                          (h₂ : ∀ x, f (x - 3 / 4) = -f (-x - 3 / 4)) : 
                          (¬prop1 f) ∧ (prop2 f) ∧ (prop3 f) ∧ (¬prop4 f) :=
by
  sorry

end problem_statement_l658_65846


namespace expression_value_l658_65862

theorem expression_value
  (x y z : ℝ)
  (hx : x = -5 / 4)
  (hy : y = -3 / 2)
  (hz : z = Real.sqrt 2) :
  -2 * x ^ 3 - y ^ 2 + Real.sin z = 53 / 32 + Real.sin (Real.sqrt 2) :=
by
  rw [hx, hy, hz]
  sorry

end expression_value_l658_65862


namespace find_mini_cupcakes_l658_65895

-- Definitions of the conditions
def number_of_donut_holes := 12
def number_of_students := 13
def desserts_per_student := 2

-- Statement of the theorem to prove the number of mini-cupcakes is 14
theorem find_mini_cupcakes :
  let D := number_of_donut_holes
  let N := number_of_students
  let total_desserts := N * desserts_per_student
  let C := total_desserts - D
  C = 14 :=
by
  sorry

end find_mini_cupcakes_l658_65895


namespace find_initial_investment_l658_65843

-- Define the necessary parameters for the problem
variables (P r : ℝ)

-- Given conditions
def condition1 : Prop := P * (1 + r * 3) = 240
def condition2 : Prop := 150 * (1 + r * 6) = 210

-- The statement to be proved
theorem find_initial_investment (h1 : condition1 P r) (h2 : condition2 r) : P = 200 :=
sorry

end find_initial_investment_l658_65843


namespace watch_cost_l658_65883

theorem watch_cost (number_of_dimes : ℕ) (value_of_dime : ℝ) (h : number_of_dimes = 50) (hv : value_of_dime = 0.10) :
  number_of_dimes * value_of_dime = 5.00 :=
by
  sorry

end watch_cost_l658_65883


namespace sum_of_roots_of_quadratic_l658_65805

theorem sum_of_roots_of_quadratic :
  ∀ x1 x2 : ℝ, (∃ a b c, a = -1 ∧ b = 2 ∧ c = 4 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) → (x1 + x2 = 2) :=
by
  sorry

end sum_of_roots_of_quadratic_l658_65805


namespace find_x_l658_65814

theorem find_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end find_x_l658_65814


namespace explicit_form_of_f_l658_65826

noncomputable def f (x : ℝ) : ℝ := sorry

theorem explicit_form_of_f :
  (∀ x : ℝ, f x + f (x + 3) = 0) →
  (∀ x : ℝ, -1 < x ∧ x ≤ 1 → f x = 2 * x - 3) →
  (∀ x : ℝ, 2 < x ∧ x ≤ 4 → f x = -2 * x + 9) :=
by
  intros h1 h2
  sorry

end explicit_form_of_f_l658_65826
