import Mathlib

namespace intersection_one_point_l497_49786

open Set

def A (x y : ℝ) : Prop := x^2 - 3*x*y + 4*y^2 = 7 / 2
def B (k x y : ℝ) : Prop := k > 0 ∧ k*x + y = 2

theorem intersection_one_point (k : ℝ) (h : k > 0) :
  (∃ x y : ℝ, A x y ∧ B k x y) → (∀ x₁ y₁ x₂ y₂ : ℝ, (A x₁ y₁ ∧ B k x₁ y₁) ∧ (A x₂ y₂ ∧ B k x₂ y₂) → x₁ = x₂ ∧ y₁ = y₂) ↔ k = 1 / 4 :=
sorry

end intersection_one_point_l497_49786


namespace part1_l497_49770

variable (A B C : ℝ)
variable (a b c S : ℝ)
variable (h1 : a * (1 + Real.cos C) + c * (1 + Real.cos A) = (5 / 2) * b)
variable (h2 : a * Real.cos C + c * Real.cos A = b)

theorem part1 : 2 * (a + c) = 3 * b := 
sorry

end part1_l497_49770


namespace simplified_expression_term_count_l497_49775

def even_exponents_terms_count : ℕ :=
  let n := 2008
  let k := 1004
  Nat.choose (k + 2) 2

theorem simplified_expression_term_count :
  even_exponents_terms_count = 505815 :=
sorry

end simplified_expression_term_count_l497_49775


namespace unit_prices_purchase_plans_exchange_methods_l497_49741

theorem unit_prices (x r : ℝ) (hx : r = 2 * x) 
  (h_eq : (40/(2*r)) + 4 = 30/x) : 
  x = 2.5 ∧ r = 5 := sorry

theorem purchase_plans (x r : ℝ) (a b : ℕ)
  (hx : x = 2.5) (hr : r = 5) (h_eq : x * a + r * b = 200)
  (h_ge_20 : 20 ≤ a ∧ 20 ≤ b) (h_mult_10 : a % 10 = 0) :
  (a, b) = (20, 30) ∨ (a, b) = (30, 25) ∨ (a, b) = (40, 20) := sorry

theorem exchange_methods (a b t m : ℕ) 
  (hx : x = 2.5) (hr : r = 5) 
  (h_leq : 1 < m ∧ m < 10) 
  (h_eq : a + 2 * t = b + (m - t))
  (h_planA : (a = 20 ∧ b = 30) ∨ (a = 30 ∧ b = 25) ∨ (a = 40 ∧ b = 20)) :
  (m = 5 ∧ t = 5 ∧ b = 30) ∨
  (m = 8 ∧ t = 6 ∧ b = 25) ∨
  (m = 5 ∧ t = 0 ∧ b = 25) ∨
  (m = 8 ∧ t = 1 ∧ b = 20) := sorry

end unit_prices_purchase_plans_exchange_methods_l497_49741


namespace solution_to_diameter_area_problem_l497_49735

def diameter_area_problem : Prop :=
  let radius := 4
  let area_of_shaded_region := 16 + 8 * Real.pi
  -- Definitions derived directly from conditions
  let circle_radius := radius
  let diameter1_perpendicular_to_diameter2 := True
  -- Conclusively prove the area of the shaded region
  ∀ (PQ RS : ℝ) (h1 : PQ = 2 * circle_radius) (h2 : RS = 2 * circle_radius) (h3 : diameter1_perpendicular_to_diameter2),
  ∃ (area : ℝ), area = area_of_shaded_region

-- This is just the statement, the proof part is omitted.
theorem solution_to_diameter_area_problem : diameter_area_problem :=
  sorry

end solution_to_diameter_area_problem_l497_49735


namespace cookies_indeterminate_l497_49788

theorem cookies_indeterminate (bananas : ℕ) (boxes : ℕ) (bananas_per_box : ℕ) (cookies : ℕ)
  (h1 : bananas = 40)
  (h2 : boxes = 8)
  (h3 : bananas_per_box = 5)
  : ∃ c : ℕ, c = cookies :=
by sorry

end cookies_indeterminate_l497_49788


namespace gcd_20020_11011_l497_49734

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := 
by
  sorry

end gcd_20020_11011_l497_49734


namespace jen_total_birds_l497_49729

theorem jen_total_birds (C D G : ℕ) (h1 : D = 150) (h2 : D = 4 * C + 10) (h3 : G = (D + C) / 2) :
  D + C + G = 277 := sorry

end jen_total_birds_l497_49729


namespace kids_played_on_tuesday_l497_49776

-- Define the total number of kids Julia played with
def total_kids : ℕ := 18

-- Define the number of kids Julia played with on Monday
def monday_kids : ℕ := 4

-- Define the number of kids Julia played with on Tuesday
def tuesday_kids : ℕ := total_kids - monday_kids

-- The proof goal:
theorem kids_played_on_tuesday : tuesday_kids = 14 :=
by sorry

end kids_played_on_tuesday_l497_49776


namespace system_of_equations_property_l497_49728

theorem system_of_equations_property (a x y : ℝ)
  (h1 : x + y = 1 - a)
  (h2 : x - y = 3 * a + 5)
  (h3 : 0 < x)
  (h4 : 0 ≤ y) :
  (a = -5 / 3 → x = y) ∧ 
  (a = -2 → x + y = 5 + a) ∧ 
  (0 < x ∧ x ≤ 1 → 2 ≤ y ∧ y < 4) := 
by
  sorry

end system_of_equations_property_l497_49728


namespace weight_of_second_piece_of_wood_l497_49787

/--
Given: 
1) The density and thickness of the wood are uniform.
2) The first piece of wood is a square with a side length of 3 inches and a weight of 15 ounces.
3) The second piece of wood is a square with a side length of 6 inches.
Theorem: 
The weight of the second piece of wood is 60 ounces.
-/
theorem weight_of_second_piece_of_wood (s1 s2 w1 w2 : ℕ) (h1 : s1 = 3) (h2 : w1 = 15) (h3 : s2 = 6) :
  w2 = 60 :=
sorry

end weight_of_second_piece_of_wood_l497_49787


namespace claudia_total_earnings_l497_49793

def cost_per_beginner_class : Int := 15
def cost_per_advanced_class : Int := 20
def num_beginner_kids_saturday : Int := 20
def num_advanced_kids_saturday : Int := 10
def num_sibling_pairs : Int := 5
def sibling_discount : Int := 3

theorem claudia_total_earnings : 
  let beginner_earnings_saturday := num_beginner_kids_saturday * cost_per_beginner_class
  let advanced_earnings_saturday := num_advanced_kids_saturday * cost_per_advanced_class
  let total_earnings_saturday := beginner_earnings_saturday + advanced_earnings_saturday
  
  let num_beginner_kids_sunday := num_beginner_kids_saturday / 2
  let num_advanced_kids_sunday := num_advanced_kids_saturday / 2
  let beginner_earnings_sunday := num_beginner_kids_sunday * cost_per_beginner_class
  let advanced_earnings_sunday := num_advanced_kids_sunday * cost_per_advanced_class
  let total_earnings_sunday := beginner_earnings_sunday + advanced_earnings_sunday

  let total_earnings_no_discount := total_earnings_saturday + total_earnings_sunday

  let total_sibling_discount := num_sibling_pairs * 2 * sibling_discount
  
  let total_earnings := total_earnings_no_discount - total_sibling_discount
  total_earnings = 720 := 
by
  sorry

end claudia_total_earnings_l497_49793


namespace who_stole_the_pan_l497_49792

def Frog_statement := "Lackey-Lech stole the pan"
def LackeyLech_statement := "I did not steal any pan"
def KnaveOfHearts_statement := "I stole the pan"

axiom no_more_than_one_liar : ∀ (frog_is_lying : Prop) (lackey_lech_is_lying : Prop) (knave_of_hearts_is_lying : Prop), (frog_is_lying → ¬ lackey_lech_is_lying) ∧ (frog_is_lying → ¬ knave_of_hearts_is_lying) ∧ (lackey_lech_is_lying → ¬ knave_of_hearts_is_lying)

theorem who_stole_the_pan : KnaveOfHearts_statement = "I stole the pan" :=
sorry

end who_stole_the_pan_l497_49792


namespace independent_variable_range_l497_49799

/-- In the function y = 1 / (x - 2), the range of the independent variable x is all real numbers except 2. -/
theorem independent_variable_range (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end independent_variable_range_l497_49799


namespace determine_better_robber_l497_49761

def sum_of_odd_series (k : ℕ) : ℕ := k * k
def sum_of_even_series (k : ℕ) : ℕ := k * (k + 1)

def first_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then (k - 1) * (k - 1) + r else k * k

def second_robber_coins (n k : ℕ) (r : ℕ) : ℕ := 
  if r < 2 * k - 1 then k * (k + 1) else k * k - k + r

theorem determine_better_robber (n k r : ℕ) :
  if 2 * k * k - 2 * k < n ∧ n < 2 * k * k then
    first_robber_coins n k r > second_robber_coins n k r
  else if 2 * k * k < n ∧ n < 2 * k * k + 2 * k then
    second_robber_coins n k r > first_robber_coins n k r
  else 
    false :=
sorry

end determine_better_robber_l497_49761


namespace find_z_l497_49725

theorem find_z (x y z : ℝ) (h : 1 / (x + 1) + 1 / (y + 1) = 1 / z) :
  z = (x + 1) * (y + 1) / (x + y + 2) :=
sorry

end find_z_l497_49725


namespace solution_set_l497_49714

theorem solution_set (x y : ℝ) : 
  x^5 - 10 * x^3 * y^2 + 5 * x * y^4 = 0 ↔ 
  x = 0 
  ∨ y = x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = x / Real.sqrt (5 - 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 + 2 * Real.sqrt 5) 
  ∨ y = -x / Real.sqrt (5 - 2 * Real.sqrt 5) := 
by
  sorry

end solution_set_l497_49714


namespace fraction_meaningful_l497_49762

theorem fraction_meaningful (a : ℝ) : (a + 3 ≠ 0) ↔ (a ≠ -3) :=
by
  sorry

end fraction_meaningful_l497_49762


namespace goal_l497_49773

-- Definitions based on the conditions
def condition1 (x : ℕ) : Prop := 10 < x ∧ x < 17
def condition2 (x : ℕ) : Prop := 11 < x ∧ x < 18
def condition3 (x : ℕ) : Prop := x % 2 = 1

-- Definition for checking if exactly two conditions hold
def exactly_two_holds (h1 h2 h3 : Prop) : Prop :=
  (h1 ∧ h2 ∧ ¬h3) ∨ (h1 ∧ ¬h2 ∧ h3) ∨ (¬h1 ∧ h2 ∧ h3)

-- Main goal: find x where exactly two out of three conditions hold
theorem goal (x : ℕ) : exactly_two_holds (condition1 x) (condition2 x) (condition3 x) ↔ 
  x = 11 ∨ x = 12 ∨ x = 14 ∨ x = 16 ∨ x = 17 :=
sorry

end goal_l497_49773


namespace anna_has_4_twenty_cent_coins_l497_49707

theorem anna_has_4_twenty_cent_coins (x y : ℕ) (h1 : x + y = 15) (h2 : 59 - 3 * x = 24) : y = 4 :=
by {
  -- evidence based on the established conditions would be derived here
  sorry
}

end anna_has_4_twenty_cent_coins_l497_49707


namespace jasper_sold_31_drinks_l497_49756

def chips := 27
def hot_dogs := chips - 8
def drinks := hot_dogs + 12

theorem jasper_sold_31_drinks : drinks = 31 := by
  sorry

end jasper_sold_31_drinks_l497_49756


namespace keaton_annual_profit_l497_49763

theorem keaton_annual_profit :
  let orange_harvests_per_year := 12 / 2
  let apple_harvests_per_year := 12 / 3
  let peach_harvests_per_year := 12 / 4
  let blackberry_harvests_per_year := 12 / 6

  let orange_profit_per_harvest := 50 - 20
  let apple_profit_per_harvest := 30 - 15
  let peach_profit_per_harvest := 45 - 25
  let blackberry_profit_per_harvest := 70 - 30

  let total_orange_profit := orange_harvests_per_year * orange_profit_per_harvest
  let total_apple_profit := apple_harvests_per_year * apple_profit_per_harvest
  let total_peach_profit := peach_harvests_per_year * peach_profit_per_harvest
  let total_blackberry_profit := blackberry_harvests_per_year * blackberry_profit_per_harvest

  let total_annual_profit := total_orange_profit + total_apple_profit + total_peach_profit + total_blackberry_profit

  total_annual_profit = 380
:= by
  sorry

end keaton_annual_profit_l497_49763


namespace opposite_of_seven_l497_49795

theorem opposite_of_seven : ∃ x : ℤ, 7 + x = 0 ∧ x = -7 :=
by
  sorry

end opposite_of_seven_l497_49795


namespace solve_for_y_l497_49789

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3)) : y = 1000 := 
by
  sorry

end solve_for_y_l497_49789


namespace southton_capsule_depth_l497_49784

theorem southton_capsule_depth :
  ∃ S : ℕ, 4 * S + 12 = 48 ∧ S = 9 :=
by
  sorry

end southton_capsule_depth_l497_49784


namespace find_same_color_integers_l497_49759

variable (Color : Type) (red blue green yellow : Color)

theorem find_same_color_integers
  (color : ℤ → Color)
  (m n : ℤ)
  (hm : Odd m)
  (hn : Odd n)
  (h_not_zero : m + n ≠ 0) :
  ∃ a b : ℤ, color a = color b ∧ (a - b = m ∨ a - b = n ∨ a - b = m + n ∨ a - b = m - n) :=
sorry

end find_same_color_integers_l497_49759


namespace solve_toenail_problem_l497_49779

def toenail_problem (b_toenails r_toenails_already r_toenails_more : ℕ) : Prop :=
  (b_toenails = 20) ∧
  (r_toenails_already = 40) ∧
  (r_toenails_more = 20) →
  (r_toenails_already + r_toenails_more = 60)

theorem solve_toenail_problem : toenail_problem 20 40 20 :=
by {
  sorry
}

end solve_toenail_problem_l497_49779


namespace initial_eggs_proof_l497_49768

noncomputable def initial_eggs (total_cost : ℝ) (price_per_egg : ℝ) (leftover_eggs : ℝ) : ℝ :=
  let eggs_sold := total_cost / price_per_egg
  eggs_sold + leftover_eggs

theorem initial_eggs_proof : initial_eggs 5 0.20 5 = 30 := by
  sorry

end initial_eggs_proof_l497_49768


namespace intersection_points_lie_on_circle_l497_49704

variables (u x y : ℝ)

theorem intersection_points_lie_on_circle :
  (∃ u : ℝ, 3 * u - 4 * y + 2 = 0 ∧ 2 * x - 3 * u * y - 4 = 0) →
  ∃ r : ℝ, (x^2 + y^2 = r^2) :=
by 
  sorry

end intersection_points_lie_on_circle_l497_49704


namespace problem_statement_l497_49719

-- Define function f(x) given parameter m
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + m * x + 3

-- Define even function condition
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Define the monotonic decreasing interval condition
def is_monotonically_decreasing (f : ℝ → ℝ) (I : Set ℝ) :=
 ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f x ≥ f y

theorem problem_statement :
  (∀ x : ℝ, f m x = f m (-x)) → is_monotonically_decreasing (f 0) {x | 0 < x} :=
by 
  sorry

end problem_statement_l497_49719


namespace parking_methods_count_l497_49754

theorem parking_methods_count : 
  ∃ (n : ℕ), n = 72 ∧ (∃ (spaces cars slots remainingSlots : ℕ), 
  spaces = 7 ∧ cars = 3 ∧ slots = 1 ∧ remainingSlots = 4 ∧
  ∃ (perm_ways slot_ways : ℕ), perm_ways = 6 ∧ slot_ways = 12 ∧ n = perm_ways * slot_ways) :=
  by
    sorry

end parking_methods_count_l497_49754


namespace bus_capacity_total_kids_l497_49767

-- Definitions based on conditions
def total_rows : ℕ := 25
def lower_deck_rows : ℕ := 15
def upper_deck_rows : ℕ := 10
def lower_deck_capacity_per_row : ℕ := 5
def upper_deck_capacity_per_row : ℕ := 3
def staff_members : ℕ := 4

-- Theorem statement
theorem bus_capacity_total_kids : 
  (lower_deck_rows * lower_deck_capacity_per_row) + 
  (upper_deck_rows * upper_deck_capacity_per_row) - staff_members = 101 := 
by
  sorry

end bus_capacity_total_kids_l497_49767


namespace good_numbers_correct_l497_49723

noncomputable def good_numbers (n : ℕ) : ℝ :=
  1 / 2 * (8^n + 10^n) - 1

theorem good_numbers_correct (n : ℕ) : good_numbers n = 
  1 / 2 * (8^n + 10^n) - 1 := 
sorry

end good_numbers_correct_l497_49723


namespace karting_routes_10_min_l497_49753

-- Define the recursive function for M_{n, A}
def num_routes : ℕ → ℕ
| 0 => 1   -- Starting point at A for 0 minutes (0 routes)
| 1 => 0   -- Impossible to end at A in just 1 move
| 2 => 1   -- Only one way to go A -> B -> A in 2 minutes
| n + 1 =>
  if n = 1 then 0 -- Additional base case for n=2 as defined
  else if n = 2 then 1
  else num_routes (n - 1) + num_routes (n - 2)

theorem karting_routes_10_min : num_routes 10 = 34 := by
  -- Proof steps go here
  sorry

end karting_routes_10_min_l497_49753


namespace sum_of_series_is_correct_l497_49708

noncomputable def geometric_series_sum_5_terms : ℚ :=
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  a * (1 - r^n) / (1 - r)

theorem sum_of_series_is_correct :
  geometric_series_sum_5_terms = 1023 / 3072 := by
  sorry

end sum_of_series_is_correct_l497_49708


namespace art_collection_total_cost_l497_49700

theorem art_collection_total_cost 
  (price_first_three : ℕ)
  (price_fourth : ℕ)
  (total_first_three : price_first_three * 3 = 45000)
  (price_fourth_cond : price_fourth = price_first_three + (price_first_three / 2)) :
  3 * price_first_three + price_fourth = 67500 :=
by
  sorry

end art_collection_total_cost_l497_49700


namespace minimize_J_l497_49737

noncomputable def H (p q : ℝ) : ℝ := -3 * p * q + 4 * p * (1 - q) + 2 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ := max (H p 0) (H p 1)

theorem minimize_J : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p ≤ J p') ∧ p = 1 / 2 :=
by
  sorry

end minimize_J_l497_49737


namespace work_completion_time_for_A_l497_49747

-- Define the conditions
def B_completion_time : ℕ := 30
def joint_work_days : ℕ := 4
def work_left_fraction : ℚ := 2 / 3

-- Define the required proof statement
theorem work_completion_time_for_A (x : ℚ) : 
  (4 * (1 / x + 1 / B_completion_time) = 1 / 3) → x = 20 := 
by
  sorry

end work_completion_time_for_A_l497_49747


namespace greatest_perimeter_of_triangle_l497_49752

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l497_49752


namespace weight_of_new_person_l497_49749

-- Define the problem conditions
variables (W : ℝ) -- Weight of the new person
variable (initial_weight : ℝ := 65) -- Weight of the person being replaced
variable (increase_in_avg : ℝ := 4) -- Increase in average weight
variable (num_persons : ℕ := 8) -- Number of persons

-- Define the total increase in weight due to the new person
def total_increase : ℝ := num_persons * increase_in_avg

-- The Lean statement to prove
theorem weight_of_new_person (W : ℝ) (h : total_increase = W - initial_weight) : W = 97 := sorry

end weight_of_new_person_l497_49749


namespace total_apples_bought_l497_49709

def apples_bought_by_Junhyeok := 7 * 16
def apples_bought_by_Jihyun := 6 * 25

theorem total_apples_bought : apples_bought_by_Junhyeok + apples_bought_by_Jihyun = 262 := by
  sorry

end total_apples_bought_l497_49709


namespace minimum_value_of_u_l497_49736

noncomputable def minimum_value_lemma (x y : ℝ) (hx : Real.sin x + Real.sin y = 1 / 3) : Prop :=
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m

theorem minimum_value_of_u
  (x y : ℝ)
  (hx : Real.sin x + Real.sin y = 1 / 3) :
  ∃ m : ℝ, m = -1/9 ∧ ∀ z, z = Real.sin x + Real.cos x ^ 2 → z ≥ m :=
sorry

end minimum_value_of_u_l497_49736


namespace calculate_expression_l497_49751

theorem calculate_expression : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end calculate_expression_l497_49751


namespace botanical_garden_correct_path_length_l497_49758

noncomputable def correct_path_length_on_ground
  (inch_length_on_map : ℝ)
  (inch_per_error_segment : ℝ)
  (conversion_rate : ℝ) : ℝ :=
  (inch_length_on_map * conversion_rate) - (inch_per_error_segment * conversion_rate)

theorem botanical_garden_correct_path_length :
  correct_path_length_on_ground 6.5 0.75 1200 = 6900 := 
by
  sorry

end botanical_garden_correct_path_length_l497_49758


namespace area_ratio_of_quadrilateral_ADGJ_to_decagon_l497_49785

noncomputable def ratio_of_areas (k : ℝ) : ℝ :=
  (2 * k^2 * Real.sin (72 * Real.pi / 180)) / (5 * Real.sqrt (5 + 2 * Real.sqrt 5))

theorem area_ratio_of_quadrilateral_ADGJ_to_decagon
  (k : ℝ) :
  ∃ (n m : ℝ), m / n = ratio_of_areas k :=
  sorry

end area_ratio_of_quadrilateral_ADGJ_to_decagon_l497_49785


namespace last_ball_probability_l497_49766

theorem last_ball_probability (w b : ℕ) (H : w > 0 ∨ b > 0) :
  (w % 2 = 1 → ∃ p : ℝ, p = 1 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) ∧ 
  (w % 2 = 0 → ∃ p : ℝ, p = 0 ∧ (∃ n, (∀ (k : ℕ), k < n → (sorry))) ) :=
by sorry

end last_ball_probability_l497_49766


namespace complement_of_P_l497_49777

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | x^2 < 2}

theorem complement_of_P :
  (U \ P) = {2} :=
by
  sorry

end complement_of_P_l497_49777


namespace total_carrots_grown_l497_49717

theorem total_carrots_grown
  (Sandy_carrots : ℕ) (Sam_carrots : ℕ) (Sophie_carrots : ℕ) (Sara_carrots : ℕ)
  (h1 : Sandy_carrots = 6)
  (h2 : Sam_carrots = 3)
  (h3 : Sophie_carrots = 2 * Sam_carrots)
  (h4 : Sara_carrots = (Sandy_carrots + Sam_carrots + Sophie_carrots) - 5) :
  Sandy_carrots + Sam_carrots + Sophie_carrots + Sara_carrots = 25 :=
by sorry

end total_carrots_grown_l497_49717


namespace tip_percentage_calculation_l497_49796

theorem tip_percentage_calculation :
  let a := 8
  let r := 20
  let w := 3
  let n_w := 2
  let d := 6
  let t := 38
  let discount := 0.5
  let full_cost_without_tip := a + r + (w * n_w) + d
  let discounted_meal_cost := a + (r - (r * discount)) + (w * n_w) + d
  let tip_amount := t - discounted_meal_cost
  let tip_percentage := (tip_amount / full_cost_without_tip) * 100
  tip_percentage = 20 :=
by
  sorry

end tip_percentage_calculation_l497_49796


namespace non_zero_real_value_l497_49740

theorem non_zero_real_value (y : ℝ) (hy : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 :=
sorry

end non_zero_real_value_l497_49740


namespace min_perimeter_l497_49743

theorem min_perimeter (a b : ℕ) (h1 : b = 3 * a) (h2 : 3 * a + 8 * a = 11) (h3 : 2 * a + 12 * a = 14)
  : 2 * (15 + 11) = 52 := 
sorry

end min_perimeter_l497_49743


namespace complex_imaginary_condition_l497_49720

theorem complex_imaginary_condition (m : ℝ) : (∀ m : ℝ, (m^2 - 3*m - 4 = 0) → (m^2 - 5*m - 6) ≠ 0) ↔ (m ≠ -1 ∧ m ≠ 6) :=
by
  sorry

end complex_imaginary_condition_l497_49720


namespace pyramid_volume_l497_49712

noncomputable def volume_of_pyramid 
  (ABCD : Type) 
  (rectangle : ABCD) 
  (DM_perpendicular : Prop) 
  (MA MC MB : ℕ) 
  (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) : ℝ :=
  80 * Real.sqrt 6

theorem pyramid_volume (ABCD : Type) 
    (rectangle : ABCD) 
    (DM_perpendicular : Prop) 
    (MA MC MB DM : ℕ)
    (lengths : MA = 11 ∧ MC = 13 ∧ MB = 15 ∧ DM = 5) 
  : volume_of_pyramid ABCD rectangle DM_perpendicular MA MC MB lengths = 80 * Real.sqrt 6 :=
  by {
    sorry
  }

end pyramid_volume_l497_49712


namespace vertical_bisecting_line_of_circles_l497_49730

theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x + 6 * y + 2 = 0 ∨ x^2 + y^2 + 4 * x - 2 * y - 4 = 0) →
  (4 * x + 3 * y + 5 = 0) :=
sorry

end vertical_bisecting_line_of_circles_l497_49730


namespace fraction_zero_iff_numerator_zero_l497_49703

variable (x : ℝ)

def numerator (x : ℝ) : ℝ := x - 5
def denominator (x : ℝ) : ℝ := 6 * x + 12

theorem fraction_zero_iff_numerator_zero (h_denominator_nonzero : denominator 5 ≠ 0) : 
  numerator x / denominator x = 0 ↔ x = 5 :=
by sorry

end fraction_zero_iff_numerator_zero_l497_49703


namespace rectangular_floor_problem_possibilities_l497_49732

theorem rectangular_floor_problem_possibilities :
  ∃ (s : Finset (ℕ × ℕ)), 
    (∀ (p : ℕ × ℕ), p ∈ s → p.2 > p.1 ∧ p.2 % 3 = 0 ∧ (p.1 - 6) * (p.2 - 6) = 36) 
    ∧ s.card = 2 := 
sorry

end rectangular_floor_problem_possibilities_l497_49732


namespace tree_difference_l497_49705

-- Given constants
def Hassans_apple_trees : Nat := 1
def Hassans_orange_trees : Nat := 2

def Ahmeds_orange_trees : Nat := 8
def Ahmeds_apple_trees : Nat := 4 * Hassans_apple_trees

-- Total trees computations
def Ahmeds_total_trees : Nat := Ahmeds_apple_trees + Ahmeds_orange_trees
def Hassans_total_trees : Nat := Hassans_apple_trees + Hassans_orange_trees

-- Theorem to prove the difference in total trees
theorem tree_difference : Ahmeds_total_trees - Hassans_total_trees = 9 := by
  sorry

end tree_difference_l497_49705


namespace complement_intersection_l497_49715

def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {0, 2}

theorem complement_intersection :
  (U \ A) ∩ B = {0} :=
  by
    sorry

end complement_intersection_l497_49715


namespace convert_polar_to_rectangular_example_l497_49702

noncomputable def polar_to_rectangular (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem convert_polar_to_rectangular_example :
  polar_to_rectangular 6 (5 * Real.pi / 2) = (0, 6) := by
  sorry

end convert_polar_to_rectangular_example_l497_49702


namespace intersection_of_A_and_complement_B_l497_49783

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x < 3}
def complement_B : Set ℝ := {x | x ≥ 3}

theorem intersection_of_A_and_complement_B : A ∩ complement_B = {3, 4, 5} :=
by
  sorry

end intersection_of_A_and_complement_B_l497_49783


namespace no_point_satisfies_both_systems_l497_49711

theorem no_point_satisfies_both_systems (x y : ℝ) :
  (y < 3 ∧ x - y < 3 ∧ x + y < 4) ∧
  ((y - 3) * (x - y - 3) ≥ 0 ∧ (y - 3) * (x + y - 4) ≤ 0 ∧ (x - y - 3) * (x + y - 4) ≤ 0)
  → false :=
sorry

end no_point_satisfies_both_systems_l497_49711


namespace find_given_number_l497_49713

theorem find_given_number (x : ℕ) : 10 * x + 2 = 3 * (x + 200000) → x = 85714 :=
by
  sorry

end find_given_number_l497_49713


namespace license_plate_combinations_l497_49706

def num_choices_two_repeat_letters : ℕ :=
  (Nat.choose 26 2) * (Nat.choose 4 2) * (5 * 4)

theorem license_plate_combinations : num_choices_two_repeat_letters = 39000 := by
  sorry

end license_plate_combinations_l497_49706


namespace find_particular_number_l497_49701

theorem find_particular_number (x : ℤ) (h : ((x / 23) - 67) * 2 = 102) : x = 2714 := 
by 
  sorry

end find_particular_number_l497_49701


namespace find_x_l497_49760

theorem find_x (x : ℕ) (h : 1 + 2 + 3 + 4 + 5 + x = 21 + 22 + 23 + 24 + 25) : x = 100 :=
by {
  sorry
}

end find_x_l497_49760


namespace tom_age_l497_49774

theorem tom_age (S T : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 := by
  sorry

end tom_age_l497_49774


namespace equation_solution_l497_49724

theorem equation_solution (x : ℝ) (h : x + 1/x = 2.5) : x^2 + 1/x^2 = 4.25 := 
by sorry

end equation_solution_l497_49724


namespace fixed_monthly_fee_l497_49726

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + 20 * y = 15.20) 
  (h2 : x + 40 * y = 25.20) : 
  x = 5.20 := 
sorry

end fixed_monthly_fee_l497_49726


namespace max_value_expression_l497_49780

theorem max_value_expression (a b c d : ℝ) 
  (h1 : -11.5 ≤ a ∧ a ≤ 11.5)
  (h2 : -11.5 ≤ b ∧ b ≤ 11.5)
  (h3 : -11.5 ≤ c ∧ c ≤ 11.5)
  (h4 : -11.5 ≤ d ∧ d ≤ 11.5):
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a ≤ 552 :=
by
  sorry

end max_value_expression_l497_49780


namespace trig_order_l497_49791

theorem trig_order (θ : ℝ) (h1 : -Real.pi / 8 < θ) (h2 : θ < 0) : Real.tan θ < Real.sin θ ∧ Real.sin θ < Real.cos θ := 
sorry

end trig_order_l497_49791


namespace abs_neg_one_eq_one_l497_49771

theorem abs_neg_one_eq_one : abs (-1 : ℚ) = 1 := 
by
  sorry

end abs_neg_one_eq_one_l497_49771


namespace number_of_grandchildren_l497_49739

-- Definitions based on the conditions
def cards_per_grandkid := 2
def money_per_card := 80
def total_money_given_away := 480

-- Calculation of money each grandkid receives per year
def money_per_grandkid := cards_per_grandkid * money_per_card

-- The theorem we want to prove
theorem number_of_grandchildren :
  (total_money_given_away / money_per_grandkid) = 3 :=
by
  -- Placeholder for the proof
  sorry 

end number_of_grandchildren_l497_49739


namespace smallest_possible_sum_l497_49710

theorem smallest_possible_sum (E F G H : ℕ) (h1 : F > 0) (h2 : E + F + G = 3 * F) (h3 : F * G = 4 * F * F / 3) :
  E = 6 ∧ F = 9 ∧ G = 12 ∧ H = 16 ∧ E + F + G + H = 43 :=
by 
  sorry

end smallest_possible_sum_l497_49710


namespace lunch_cost_before_tip_l497_49755

theorem lunch_cost_before_tip (C : ℝ) (h : C + 0.20 * C = 60.24) : C = 50.20 :=
sorry

end lunch_cost_before_tip_l497_49755


namespace remainder_3_pow_19_mod_10_l497_49769

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 :=
by
  sorry

end remainder_3_pow_19_mod_10_l497_49769


namespace sqrt_ax3_eq_negx_sqrt_ax_l497_49721

variable (a x : ℝ)
variable (ha : a < 0) (hx : x < 0)

theorem sqrt_ax3_eq_negx_sqrt_ax : Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) := by
  sorry

end sqrt_ax3_eq_negx_sqrt_ax_l497_49721


namespace calc_expression_l497_49744

theorem calc_expression : (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 :=
  sorry

end calc_expression_l497_49744


namespace find_abc_l497_49733

open Real

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc (a b c : ℝ)
  (h₁ : a - b = 3)
  (h₂ : a^2 + b^2 = 39)
  (h₃ : a + b + c = 10) :
  abc_value a b c = -150 + 15 * Real.sqrt 69 :=
by
  sorry

end find_abc_l497_49733


namespace evaluate_expression_l497_49750

theorem evaluate_expression : 6 / (-1 / 2 + 1 / 3) = -36 := 
by
  sorry

end evaluate_expression_l497_49750


namespace student_A_recruit_as_pilot_exactly_one_student_pass_l497_49716

noncomputable def student_A_recruit_prob : ℝ :=
  1 * 0.5 * 0.6 * 1

theorem student_A_recruit_as_pilot :
  student_A_recruit_prob = 0.3 :=
by
  sorry

noncomputable def one_student_pass_reinspection : ℝ :=
  0.5 * (1 - 0.6) * (1 - 0.75) +
  (1 - 0.5) * 0.6 * (1 - 0.75) +
  (1 - 0.5) * (1 - 0.6) * 0.75

theorem exactly_one_student_pass :
  one_student_pass_reinspection = 0.275 :=
by
  sorry

end student_A_recruit_as_pilot_exactly_one_student_pass_l497_49716


namespace total_payment_correct_l497_49794

theorem total_payment_correct 
  (bob_bill : ℝ) 
  (kate_bill : ℝ) 
  (bob_discount_rate : ℝ) 
  (kate_discount_rate : ℝ) 
  (bob_discount : ℝ := bob_bill * bob_discount_rate / 100) 
  (kate_discount : ℝ := kate_bill * kate_discount_rate / 100) 
  (bob_final_payment : ℝ := bob_bill - bob_discount) 
  (kate_final_payment : ℝ := kate_bill - kate_discount) : 
  (bob_bill = 30) → 
  (kate_bill = 25) → 
  (bob_discount_rate = 5) → 
  (kate_discount_rate = 2) → 
  (bob_final_payment + kate_final_payment = 53) :=
by
  intros
  sorry

end total_payment_correct_l497_49794


namespace twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l497_49781

theorem twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number :
  ∃ n : ℝ, (80 - 0.25 * 80) = (5 / 4) * n ∧ n = 48 := 
by
  sorry

end twenty_five_percent_less_than_eighty_equals_one_fourth_more_than_what_number_l497_49781


namespace time_after_midnight_1453_minutes_l497_49757

def minutes_to_time (minutes : Nat) : Nat × Nat :=
  let hours := minutes / 60
  let remaining_minutes := minutes % 60
  (hours, remaining_minutes)

def time_of_day (hours : Nat) : Nat × Nat :=
  let days := hours / 24
  let remaining_hours := hours % 24
  (days, remaining_hours)

theorem time_after_midnight_1453_minutes : 
  let midnight := (0, 0) -- Midnight as a tuple of hours and minutes
  let total_minutes := 1453
  let (total_hours, minutes) := minutes_to_time total_minutes
  let (days, hours) := time_of_day total_hours
  days = 1 ∧ hours = 0 ∧ minutes = 13
  := by
    let midnight := (0, 0)
    let total_minutes := 1453
    let (total_hours, minutes) := minutes_to_time total_minutes
    let (days, hours) := time_of_day total_hours
    sorry

end time_after_midnight_1453_minutes_l497_49757


namespace solve_for_y_l497_49778

theorem solve_for_y (y : ℝ) (h : 7 - y = 12) : y = -5 := sorry

end solve_for_y_l497_49778


namespace largest_class_students_l497_49765

theorem largest_class_students (n1 n2 n3 n4 n5 : ℕ) (h1 : n1 = x) (h2 : n2 = x - 2) (h3 : n3 = x - 4) (h4 : n4 = x - 6) (h5 : n5 = x - 8) (h_sum : n1 + n2 + n3 + n4 + n5 = 140) : x = 32 :=
by {
  sorry
}

end largest_class_students_l497_49765


namespace common_ratio_geometric_progression_l497_49727

theorem common_ratio_geometric_progression {x y z r : ℝ} (h_diff1 : x ≠ y) (h_diff2 : y ≠ z) (h_diff3 : z ≠ x)
  (hx_nonzero : x ≠ 0) (hy_nonzero : y ≠ 0) (hz_nonzero : z ≠ 0)
  (h_gm_progression : ∃ r : ℝ, x * (y - z) = x * (y - z) * r ∧ z * (x - y) = (y * (z - x)) * r) : r^2 + r + 1 = 0 :=
sorry

end common_ratio_geometric_progression_l497_49727


namespace least_faces_triangular_pyramid_l497_49748

def triangular_prism_faces : ℕ := 5
def quadrangular_prism_faces : ℕ := 6
def triangular_pyramid_faces : ℕ := 4
def quadrangular_pyramid_faces : ℕ := 5
def truncated_quadrangular_pyramid_faces : ℕ := 5 -- assuming the minimum possible value

theorem least_faces_triangular_pyramid :
  triangular_pyramid_faces < triangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_prism_faces ∧
  triangular_pyramid_faces < quadrangular_pyramid_faces ∧
  triangular_pyramid_faces ≤ truncated_quadrangular_pyramid_faces :=
by
  sorry

end least_faces_triangular_pyramid_l497_49748


namespace min_value_f_l497_49782

-- Define the function f(x)
def f (x : ℝ) : ℝ := (15 - x) * (13 - x) * (15 + x) * (13 + x) + 200 * x^2

-- State the theorem to be proved
theorem min_value_f : ∃ (x : ℝ), (∀ y : ℝ, f y ≥ 33) ∧ f x = 33 := by
  sorry

end min_value_f_l497_49782


namespace evaluate_expression_l497_49746

theorem evaluate_expression (a : ℕ) (h : a = 3) : a^2 * a^5 = 2187 :=
by sorry

end evaluate_expression_l497_49746


namespace total_customers_l497_49745

-- Define the initial number of customers
def initial_customers : ℕ := 14

-- Define the number of customers that left
def customers_left : ℕ := 3

-- Define the number of new customers gained
def new_customers : ℕ := 39

-- Prove that the total number of customers is 50
theorem total_customers : initial_customers - customers_left + new_customers = 50 := 
by
  sorry

end total_customers_l497_49745


namespace sin_alpha_plus_7pi_over_12_l497_49764

theorem sin_alpha_plus_7pi_over_12 (α : Real) 
  (h1 : Real.cos (α + π / 12) = 1 / 5) : 
  Real.sin (α + 7 * π / 12) = 1 / 5 :=
by
  sorry

end sin_alpha_plus_7pi_over_12_l497_49764


namespace symmetric_points_sum_l497_49722

theorem symmetric_points_sum (n m : ℤ) 
  (h₁ : (3 : ℤ) = m)
  (h₂ : n = (-5 : ℤ)) : 
  m + n = (-2 : ℤ) := 
by 
  sorry

end symmetric_points_sum_l497_49722


namespace revenue_increase_l497_49731

theorem revenue_increase (n : ℕ) (C P : ℝ) 
  (h1 : n * P = 1.20 * C) : 
  (0.95 * n * P) = 1.14 * C :=
by
  sorry

end revenue_increase_l497_49731


namespace remainder_is_4_l497_49718

-- Definitions based on the given conditions
def dividend := 132
def divisor := 16
def quotient := 8

-- The theorem we aim to prove, stating the remainder
theorem remainder_is_4 : dividend = divisor * quotient + 4 := sorry

end remainder_is_4_l497_49718


namespace alex_age_thrice_ben_in_n_years_l497_49797

-- Definitions based on the problem's conditions
def Ben_current_age := 4
def Alex_current_age := Ben_current_age + 30

-- The main problem defined as a theorem to be proven
theorem alex_age_thrice_ben_in_n_years :
  ∃ n : ℕ, Alex_current_age + n = 3 * (Ben_current_age + n) ∧ n = 11 :=
by
  sorry

end alex_age_thrice_ben_in_n_years_l497_49797


namespace volume_of_box_l497_49742

noncomputable def volume_expression (y : ℝ) : ℝ :=
  (15 - 2 * y) * (12 - 2 * y) * y

theorem volume_of_box (y : ℝ) :
  volume_expression y = 4 * y^3 - 54 * y^2 + 180 * y :=
by
  sorry

end volume_of_box_l497_49742


namespace find_function_l497_49738

theorem find_function (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (x + y) = f x + f y - 2023) →
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 :=
by
  intros h
  sorry

end find_function_l497_49738


namespace cylinder_properties_l497_49772

theorem cylinder_properties (h r : ℝ) (h_eq : h = 15) (r_eq : r = 5) :
  let total_surface_area := 2 * Real.pi * r^2 + 2 * Real.pi * r * h
  let volume := Real.pi * r^2 * h
  total_surface_area = 200 * Real.pi ∧ volume = 375 * Real.pi :=
by
  sorry

end cylinder_properties_l497_49772


namespace coords_of_a_in_m_n_l497_49798

variable {R : Type} [Field R]

def coords_in_basis (a : R × R) (p q : R × R) (c1 c2 : R) : Prop :=
  a = c1 • p + c2 • q

theorem coords_of_a_in_m_n
  (a p q m n : R × R)
  (hp : p = (1, -1)) (hq : q = (2, 1)) (hm : m = (-1, 1)) (hn : n = (1, 2))
  (coords_pq : coords_in_basis a p q (-2) 2) :
  coords_in_basis a m n 0 2 :=
by
  sorry

end coords_of_a_in_m_n_l497_49798


namespace smallest_x_for_cubic_l497_49790

theorem smallest_x_for_cubic (x N : ℕ) (h1 : 1260 * x = N^3) : x = 7350 :=
sorry

end smallest_x_for_cubic_l497_49790
