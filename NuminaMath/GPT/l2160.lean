import Mathlib

namespace surface_area_of_cube_is_correct_l2160_216044

noncomputable def edge_length (a : ℝ) : ℝ := 5 * a

noncomputable def surface_area_of_cube (a : ℝ) : ℝ :=
  let edge := edge_length a
  6 * edge * edge

theorem surface_area_of_cube_is_correct (a : ℝ) :
  surface_area_of_cube a = 150 * a ^ 2 := by
  sorry

end surface_area_of_cube_is_correct_l2160_216044


namespace integer_pairs_satisfying_equation_l2160_216001

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x * (x + 1) * (x + 7) * (x + 8) = y^2 →
    (x = 1 ∧ y = 12) ∨ (x = 1 ∧ y = -12) ∨ 
    (x = -9 ∧ y = 12) ∨ (x = -9 ∧ y = -12) ∨ 
    (x = -4 ∧ y = 12) ∨ (x = -4 ∧ y = -12) ∨ 
    (x = 0 ∧ y = 0) ∨ (x = -8 ∧ y = 0) ∨ 
    (x = -1 ∧ y = 0) ∨ (x = -7 ∧ y = 0) :=
by sorry

end integer_pairs_satisfying_equation_l2160_216001


namespace how_many_cakes_each_friend_ate_l2160_216080

-- Definitions pertaining to the problem conditions
def crackers : ℕ := 29
def cakes : ℕ := 30
def friends : ℕ := 2

-- The main theorem statement we aim to prove
theorem how_many_cakes_each_friend_ate 
  (h1 : crackers = 29)
  (h2 : cakes = 30)
  (h3 : friends = 2) : 
  (cakes / friends = 15) :=
by
  sorry

end how_many_cakes_each_friend_ate_l2160_216080


namespace leggings_needed_l2160_216085

theorem leggings_needed (dogs : ℕ) (cats : ℕ) (dogs_legs : ℕ) (cats_legs : ℕ) (pair_of_leggings : ℕ) 
                        (hd : dogs = 4) (hc : cats = 3) (hl1 : dogs_legs = 4) (hl2 : cats_legs = 4) (lp : pair_of_leggings = 2)
                        : (dogs * dogs_legs + cats * cats_legs) / pair_of_leggings = 14 :=
by
  sorry

end leggings_needed_l2160_216085


namespace sequence_solution_l2160_216090

theorem sequence_solution
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a1 : a 1 = 10)
  (h_b1 : b 1 = 10)
  (h_recur_a : ∀ n : ℕ, a (n + 1) = 1 / (a n * b n))
  (h_recur_b : ∀ n : ℕ, b (n + 1) = (a n)^4 * b n) :
  (∀ n : ℕ, n > 0 → a n = 10^((2 - 3 * n) * (-1 : ℝ)^n) ∧ b n = 10^((6 * n - 7) * (-1 : ℝ)^n)) :=
by
  sorry

end sequence_solution_l2160_216090


namespace paul_has_five_dogs_l2160_216000

theorem paul_has_five_dogs
  (w1 w2 w3 w4 w5 : ℕ)
  (food_per_10_pounds : ℕ)
  (total_food_required : ℕ)
  (h1 : w1 = 20)
  (h2 : w2 = 40)
  (h3 : w3 = 10)
  (h4 : w4 = 30)
  (h5 : w5 = 50)
  (h6 : food_per_10_pounds = 1)
  (h7 : total_food_required = 15) :
  (w1 / 10 * food_per_10_pounds) +
  (w2 / 10 * food_per_10_pounds) +
  (w3 / 10 * food_per_10_pounds) +
  (w4 / 10 * food_per_10_pounds) +
  (w5 / 10 * food_per_10_pounds) = total_food_required → 
  5 = 5 :=
by
  intros
  sorry

end paul_has_five_dogs_l2160_216000


namespace necessary_and_sufficient_condition_l2160_216047

theorem necessary_and_sufficient_condition (a b : ℝ) (h1 : a > b) (h2 : 1 / a > 1 / b) : a > 0 ∧ 0 > b :=
by
  sorry

end necessary_and_sufficient_condition_l2160_216047


namespace fraction_collectors_edition_is_correct_l2160_216072

-- Let's define the necessary conditions
variable (DinaDolls IvyDolls CollectorsEditionDolls : ℕ)
variable (FractionCollectorsEdition : ℚ)

-- Given conditions
axiom DinaHas60Dolls : DinaDolls = 60
axiom DinaHasTwiceAsManyDollsAsIvy : DinaDolls = 2 * IvyDolls
axiom IvyHas20CollectorsEditionDolls : CollectorsEditionDolls = 20

-- The statement to prove
theorem fraction_collectors_edition_is_correct :
  FractionCollectorsEdition = (CollectorsEditionDolls : ℚ) / (IvyDolls : ℚ) ∧
  DinaDolls = 60 →
  DinaDolls = 2 * IvyDolls →
  CollectorsEditionDolls = 20 →
  FractionCollectorsEdition = 2 / 3 := 
by
  sorry

end fraction_collectors_edition_is_correct_l2160_216072


namespace bowling_ball_weight_l2160_216045

-- Definitions based on given conditions
variable (k b : ℕ)

-- Condition 1: one kayak weighs 35 pounds
def kayak_weight : Prop := k = 35

-- Condition 2: four kayaks weigh the same as five bowling balls
def balance_equation : Prop := 4 * k = 5 * b

-- Goal: prove the weight of one bowling ball is 28 pounds
theorem bowling_ball_weight (hk : kayak_weight k) (hb : balance_equation k b) : b = 28 :=
by
  sorry

end bowling_ball_weight_l2160_216045


namespace exists_x_abs_ge_one_fourth_l2160_216016

theorem exists_x_abs_ge_one_fourth :
  ∀ (a b c : ℝ), ∃ x : ℝ, |x| ≤ 1 ∧ |x^3 + a * x^2 + b * x + c| ≥ 1 / 4 :=
by sorry

end exists_x_abs_ge_one_fourth_l2160_216016


namespace linear_eq_value_abs_sum_l2160_216042

theorem linear_eq_value_abs_sum (a m : ℤ)
  (h1: m^2 - 9 = 0)
  (h2: m ≠ 3)
  (h3: |a| ≤ 3) : 
  |a + m| + |a - m| = 6 :=
by
  sorry

end linear_eq_value_abs_sum_l2160_216042


namespace taimour_time_to_paint_alone_l2160_216087

theorem taimour_time_to_paint_alone (T : ℝ) (h1 : Jamshid_time = T / 2)
  (h2 : (1 / T + 1 / (T / 2)) = 1 / 3) : T = 9 :=
sorry

end taimour_time_to_paint_alone_l2160_216087


namespace correct_mark_l2160_216051

theorem correct_mark (x : ℕ) (h1 : 73 - x = 10) : x = 63 :=
by
  sorry

end correct_mark_l2160_216051


namespace remainder_2_pow_305_mod_9_l2160_216062

theorem remainder_2_pow_305_mod_9 :
  2^305 % 9 = 5 :=
by sorry

end remainder_2_pow_305_mod_9_l2160_216062


namespace cannot_make_it_in_time_l2160_216060

theorem cannot_make_it_in_time (time_available : ℕ) (distance_to_station : ℕ) (v1 : ℕ) :
  time_available = 2 ∧ distance_to_station = 2 ∧ v1 = 30 → 
  ¬ ∃ v2, (time_available - (distance_to_station / v1)) * v2 ≥ 1 :=
by
  sorry

end cannot_make_it_in_time_l2160_216060


namespace bacteria_after_7_hours_l2160_216093

noncomputable def bacteria_growth (initial : ℝ) (t : ℝ) (k : ℝ) : ℝ := initial * (10 * (Real.exp (k * t)))

noncomputable def solve_bacteria_problem : ℝ :=
let doubling_time := 1 / 60 -- In hours, since 60 minutes is 1 hour
-- Given that it doubles in 1 hour, we expect the growth to be such that y = initial * (2) in 1 hour.
let k := Real.log 2 -- Since when t = 1, we have 10 * e^(k * 1) = 2 * 10
bacteria_growth 10 7 k

theorem bacteria_after_7_hours :
  solve_bacteria_problem = 1280 :=
by
  sorry

end bacteria_after_7_hours_l2160_216093


namespace second_polygon_sides_l2160_216076

-- Conditions as definitions
def perimeter_first_polygon (s : ℕ) := 50 * (3 * s)
def perimeter_second_polygon (N s : ℕ) := N * s
def same_perimeter (s N : ℕ) := perimeter_first_polygon s = perimeter_second_polygon N s

-- Theorem statement
theorem second_polygon_sides (s N : ℕ) :
  same_perimeter s N → N = 150 :=
by
  sorry

end second_polygon_sides_l2160_216076


namespace impossible_grid_arrangement_l2160_216052

theorem impossible_grid_arrangement :
  ¬ ∃ (f : Fin 25 → Fin 41 → ℤ),
    (∀ i j, abs (f i j - f (i + 1) j) ≤ 16 ∧ abs (f i j - f i (j + 1)) ≤ 16 ∧
            f i j ≠ f (i + 1) j ∧ f i j ≠ f i (j + 1)) := 
sorry

end impossible_grid_arrangement_l2160_216052


namespace proof_problem_l2160_216018

noncomputable def f (x a : ℝ) : ℝ := (1 + x^2) * Real.exp x - a
noncomputable def f' (x a : ℝ) : ℝ := (1 + 2 * x + x^2) * Real.exp x
noncomputable def k_OP (a : ℝ) : ℝ := a - 2 / Real.exp 1
noncomputable def g (m : ℝ) : ℝ := Real.exp m - (m + 1)

theorem proof_problem (a m : ℝ) (h₁ : a > 0) (h₂ : f' (-1) a = 0) (h₃ : f' m a = k_OP a) 
  : m + 1 ≤ 3 * a - 2 / Real.exp 1 := by
  sorry

end proof_problem_l2160_216018


namespace largest_digit_not_in_odd_units_digits_l2160_216020

-- Defining the sets of digits
def odd_units_digits : Set ℕ := {1, 3, 5, 7, 9}
def even_units_digits : Set ℕ := {0, 2, 4, 6, 8}

-- Statement to prove
theorem largest_digit_not_in_odd_units_digits : 
  ∀ n ∈ even_units_digits, n ≤ 8 ∧ (∀ d ∈ odd_units_digits, d < n) → n = 8 :=
by
  sorry

end largest_digit_not_in_odd_units_digits_l2160_216020


namespace find_x2_y2_l2160_216068

variable (x y : ℝ)

-- Given conditions
def average_commute_time (x y : ℝ) := (x + y + 10 + 11 + 9) / 5 = 10
def variance_commute_time (x y : ℝ) := ( (x - 10) ^ 2 + (y - 10) ^ 2 + (10 - 10) ^ 2 + (11 - 10) ^ 2 + (9 - 10) ^ 2 ) / 5 = 2

-- The theorem to prove
theorem find_x2_y2 (hx_avg : average_commute_time x y) (hx_var : variance_commute_time x y) : 
  x^2 + y^2 = 208 :=
sorry

end find_x2_y2_l2160_216068


namespace cost_small_and_large_puzzle_l2160_216058

-- Define the cost of a large puzzle L and the cost equation for large and small puzzles
def cost_large_puzzle : ℤ := 15

def cost_equation (S : ℤ) : Prop := cost_large_puzzle + 3 * S = 39

-- Theorem to prove the total cost of a small puzzle and a large puzzle together
theorem cost_small_and_large_puzzle : ∃ S : ℤ, cost_equation S ∧ (S + cost_large_puzzle = 23) :=
by
  sorry

end cost_small_and_large_puzzle_l2160_216058


namespace set_intersection_eq_l2160_216027

def setA : Set ℝ := { x | x^2 - 3 * x - 4 > 0 }
def setB : Set ℝ := { x | -2 < x ∧ x < 5 }
def setC : Set ℝ := { x | (-2 < x ∧ x < -1) ∨ (4 < x ∧ x < 5) }

theorem set_intersection_eq : setA ∩ setB = setC := by
  sorry

end set_intersection_eq_l2160_216027


namespace common_factor_polynomials_l2160_216089

-- Define the two polynomials
def poly1 (x y z : ℝ) := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
def poly2 (x y z : ℝ) := 6 * x^4 * y * z^2

-- Define the common factor
def common_factor (x y z : ℝ) := 3 * x^2 * y * z

-- The statement to prove that the common factor of poly1 and poly2 is 3 * x^2 * y * z
theorem common_factor_polynomials (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (poly1 x y z) = (f x y z) * (common_factor x y z) ∧
                          (poly2 x y z) = (f x y z) * (common_factor x y z) :=
sorry

end common_factor_polynomials_l2160_216089


namespace DM_eq_r_plus_R_l2160_216014

noncomputable def radius_incircle (A B D : ℝ) (s K : ℝ) : ℝ := K / s

noncomputable def radius_excircle (A C D : ℝ) (s' K' : ℝ) (AD : ℝ) : ℝ := K' / (s' - AD)

theorem DM_eq_r_plus_R 
  (A B C D M : ℝ)
  (h1 : A ≠ B)
  (h2 : B ≠ C)
  (h3 : A ≠ C)
  (h4 : D = (B + C) / 2)
  (h5 : M = (B + C) / 2)
  (r : ℝ)
  (h6 : r = radius_incircle A B D ((A + B + D) / 2) (abs ((A - B) * (A - D) / 2)))
  (R : ℝ)
  (h7 : R = radius_excircle A C D ((A + C + D) / 2) (abs ((A - C) * (A - D) / 2)) (abs (A - D))) :
  dist D M =r + R :=
by sorry

end DM_eq_r_plus_R_l2160_216014


namespace james_final_sticker_count_l2160_216081

-- Define the conditions
def initial_stickers := 478
def gift_stickers := 182
def given_away_stickers := 276

-- Define the correct answer
def final_stickers := 384

-- State the theorem
theorem james_final_sticker_count :
  initial_stickers + gift_stickers - given_away_stickers = final_stickers :=
by
  sorry

end james_final_sticker_count_l2160_216081


namespace grocery_store_price_l2160_216013

-- Definitions based on the conditions
def bulk_price_per_case : ℝ := 12.00
def bulk_cans_per_case : ℝ := 48.0
def grocery_cans_per_pack : ℝ := 12.0
def additional_cost_per_can : ℝ := 0.25

-- The proof statement
theorem grocery_store_price : 
  (bulk_price_per_case / bulk_cans_per_case + additional_cost_per_can) * grocery_cans_per_pack = 6.00 :=
by
  sorry

end grocery_store_price_l2160_216013


namespace isosceles_trapezoid_with_inscribed_circle_area_is_20_l2160_216009

def isosceles_trapezoid_area (a b c1 c2 h : ℕ) : ℕ :=
  (a + b) * h / 2

theorem isosceles_trapezoid_with_inscribed_circle_area_is_20
  (a b c h : ℕ)
  (ha : a = 2)
  (hb : b = 8)
  (hc : a + b = 2 * c)
  (hh : h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2) :
  isosceles_trapezoid_area a b c c h = 20 := 
by {
  sorry
}

end isosceles_trapezoid_with_inscribed_circle_area_is_20_l2160_216009


namespace entree_cost_l2160_216086

/-- 
Prove that if the total cost is 23 and the entree costs 5 more than the dessert, 
then the cost of the entree is 14.
-/
theorem entree_cost (D : ℝ) (H1 : D + (D + 5) = 23) : D + 5 = 14 :=
by
  -- note: no proof required as per instructions
  sorry

end entree_cost_l2160_216086


namespace gwen_spending_l2160_216097

theorem gwen_spending : 
    ∀ (initial_amount spent remaining : ℕ), 
    initial_amount = 7 → remaining = 5 → initial_amount - remaining = 2 :=
by
    sorry

end gwen_spending_l2160_216097


namespace initial_birds_count_l2160_216098

theorem initial_birds_count (B : ℕ) :
  ∃ B, B + 4 = 5 + 2 → B = 3 :=
by
  sorry

end initial_birds_count_l2160_216098


namespace average_annual_growth_rate_l2160_216022

theorem average_annual_growth_rate (x : ℝ) (h1 : 6.4 * (1 + x)^2 = 8.1) : x = 0.125 :=
by
  -- proof goes here
  sorry

end average_annual_growth_rate_l2160_216022


namespace weighted_average_remaining_two_l2160_216088

theorem weighted_average_remaining_two (avg_10 : ℝ) (avg_2 : ℝ) (avg_3 : ℝ) (avg_3_next : ℝ) :
  avg_10 = 4.25 ∧ avg_2 = 3.4 ∧ avg_3 = 3.85 ∧ avg_3_next = 4.7 →
  (42.5 - (2 * 3.4 + 3 * 3.85 + 3 * 4.7)) / 2 = 5.025 :=
by
  intros
  sorry

end weighted_average_remaining_two_l2160_216088


namespace solve_x_l2160_216039

noncomputable def solveEquation (a b c d : ℝ) (x : ℝ) : Prop :=
  x = 3 * a * b + 33 * b^2 + 333 * c^3 + 3.33 * (Real.sin d)^4

theorem solve_x :
  solveEquation 2 (-1) 0.5 (Real.pi / 6) 68.833125 :=
by
  sorry

end solve_x_l2160_216039


namespace no_such_function_exists_l2160_216023

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x - f y) = 1 + x - y := by
  sorry

end no_such_function_exists_l2160_216023


namespace difference_between_max_and_min_coins_l2160_216053

theorem difference_between_max_and_min_coins (n : ℕ) : 
  (∃ x y : ℕ, x * 10 + y * 25 = 45 ∧ x + y = n) →
  (∃ p q : ℕ, p * 10 + q * 25 = 45 ∧ p + q = n) →
  (n = 2) :=
by
  sorry

end difference_between_max_and_min_coins_l2160_216053


namespace tax_is_one_l2160_216099

-- Define costs
def cost_eggs : ℕ := 3
def cost_pancakes : ℕ := 2
def cost_cocoa : ℕ := 2

-- Initial order
def initial_eggs := 1
def initial_pancakes := 1
def initial_mugs_of_cocoa := 2

-- Additional order by Ben
def additional_pancakes := 1
def additional_mugs_of_cocoa := 1

-- Calculate costs
def initial_cost : ℕ := initial_eggs * cost_eggs + initial_pancakes * cost_pancakes + initial_mugs_of_cocoa * cost_cocoa
def additional_cost : ℕ := additional_pancakes * cost_pancakes + additional_mugs_of_cocoa * cost_cocoa
def total_cost_before_tax : ℕ := initial_cost + additional_cost

-- Payment and change
def total_paid : ℕ := 15
def change : ℕ := 1
def actual_payment : ℕ := total_paid - change

-- Calculate tax
def tax : ℕ := actual_payment - total_cost_before_tax

-- Prove that the tax is $1
theorem tax_is_one : tax = 1 :=
by
  sorry

end tax_is_one_l2160_216099


namespace find_f_ln_log_52_l2160_216021

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

axiom given_condition (a : ℝ) : f a (Real.log (Real.log 5 / Real.log 2)) = 5

theorem find_f_ln_log_52 (a : ℝ) : f a (Real.log (Real.log 2 / Real.log 5)) = 3 :=
by
  -- The details of the proof are omitted
  sorry

end find_f_ln_log_52_l2160_216021


namespace barbata_interest_rate_l2160_216010

theorem barbata_interest_rate (r : ℝ) : 
  let initial_investment := 2800
  let additional_investment := 1400
  let total_investment := initial_investment + additional_investment
  let annual_income := 0.06 * total_investment
  let additional_interest_rate := 0.08
  let income_from_initial := initial_investment * r
  let income_from_additional := additional_investment * additional_interest_rate
  income_from_initial + income_from_additional = annual_income → 
  r = 0.05 :=
by
  intros
  sorry

end barbata_interest_rate_l2160_216010


namespace valid_duty_schedules_l2160_216035

noncomputable def validSchedules : ℕ := 
  let A_schedule := Nat.choose 7 4  -- \binom{7}{4} for A
  let B_schedule := Nat.choose 4 4  -- \binom{4}{4} for B
  let C_schedule := Nat.choose 6 3  -- \binom{6}{3} for C
  let D_schedule := Nat.choose 5 5  -- \binom{5}{5} for D
  A_schedule * B_schedule * C_schedule * D_schedule

theorem valid_duty_schedules : validSchedules = 700 := by
  -- proof steps will go here
  sorry

end valid_duty_schedules_l2160_216035


namespace solve_sum_of_digits_eq_2018_l2160_216077

def s (n : ℕ) : ℕ := (Nat.digits 10 n).sum

theorem solve_sum_of_digits_eq_2018 : ∃ n : ℕ, n + s n = 2018 := by
  sorry

end solve_sum_of_digits_eq_2018_l2160_216077


namespace rabbit_speed_l2160_216055

theorem rabbit_speed (s : ℕ) (h : (s * 2 + 4) * 2 = 188) : s = 45 :=
sorry

end rabbit_speed_l2160_216055


namespace cost_per_pound_of_mixed_candy_l2160_216024

def w1 := 10
def p1 := 8
def w2 := 20
def p2 := 5

theorem cost_per_pound_of_mixed_candy : 
    (w1 * p1 + w2 * p2) / (w1 + w2) = 6 := by
  sorry

end cost_per_pound_of_mixed_candy_l2160_216024


namespace integer_solutions_of_inequality_count_l2160_216056

theorem integer_solutions_of_inequality_count :
  let a := -2 - Real.sqrt 6
  let b := -2 + Real.sqrt 6
  ∃ n, n = 5 ∧ ∀ x : ℤ, x < a ∨ b < x ↔ (4 * x^2 + 16 * x + 15 ≤ 23) → n = 5 :=
by sorry

end integer_solutions_of_inequality_count_l2160_216056


namespace average_of_roots_l2160_216041

theorem average_of_roots (a b: ℝ) (h : a ≠ 0) (hr : ∃ x1 x2: ℝ, a * x1 ^ 2 - 3 * a * x1 + b = 0 ∧ a * x2 ^ 2 - 3 * a * x2 + b = 0 ∧ x1 ≠ x2):
  (∃ r1 r2: ℝ, a * r1 ^ 2 - 3 * a * r1 + b = 0 ∧ a * r2 ^ 2 - 3 * a * r2 + b = 0 ∧ r1 ≠ r2) →
  ((r1 + r2) / 2 = 3 / 2) :=
by
  sorry

end average_of_roots_l2160_216041


namespace emily_widgets_production_l2160_216057

variable (w t : ℕ) (work_hours_monday work_hours_tuesday production_monday production_tuesday : ℕ)

theorem emily_widgets_production :
  (w = 2 * t) → 
  (work_hours_monday = t) →
  (work_hours_tuesday = t - 3) →
  (production_monday = w * work_hours_monday) → 
  (production_tuesday = (w + 6) * work_hours_tuesday) →
  (production_monday - production_tuesday) = 18 :=
by
  intros hw hwm hwmt hpm hpt
  sorry

end emily_widgets_production_l2160_216057


namespace geometric_sequence_sum_l2160_216019

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement to prove
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : is_geometric_sequence a q)
  (h2 : a 1 + a 2 = 40)
  (h3 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l2160_216019


namespace minimum_value_expression_l2160_216048

theorem minimum_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ z, (z = a^2 + b^2 + 1 / a^2 + 2 * b / a) ∧ z ≥ 2 :=
sorry

end minimum_value_expression_l2160_216048


namespace length_of_each_section_25_l2160_216084

theorem length_of_each_section_25 (x : ℝ) 
  (h1 : ∃ x, x > 0)
  (h2 : 1000 / x = 15 / (1 / 2 * 3 / 4))
  : x = 25 := 
  sorry

end length_of_each_section_25_l2160_216084


namespace range_of_a_l2160_216083

theorem range_of_a (a x : ℝ) (h_p : a - 4 < x ∧ x < a + 4) (h_q : (x - 2) * (x - 3) > 0) :
  a ≤ -2 ∨ a ≥ 7 :=
sorry

end range_of_a_l2160_216083


namespace fifth_roll_six_probability_l2160_216082
noncomputable def probability_fifth_roll_six : ℚ := sorry

theorem fifth_roll_six_probability :
  let fair_die_prob : ℚ := (1/6)^4
  let biased_die_6_prob : ℚ := (2/3)^3 * (1/15)
  let biased_die_3_prob : ℚ := (1/10)^3 * (1/2)
  let total_prob := (1/3) * fair_die_prob + (1/3) * biased_die_6_prob + (1/3) * biased_die_3_prob
  let normalized_biased_6_prob := (1/3) * biased_die_6_prob / total_prob
  let prob_of_fifth_six := normalized_biased_6_prob * (2/3)
  probability_fifth_roll_six = prob_of_fifth_six :=
sorry

end fifth_roll_six_probability_l2160_216082


namespace simplify_expression_l2160_216064

-- Define constants
variables (z : ℝ)

-- Define the problem and its solution
theorem simplify_expression :
  (5 - 2 * z) - (4 + 5 * z) = 1 - 7 * z := 
sorry

end simplify_expression_l2160_216064


namespace scalene_triangles_count_l2160_216061

/-- Proving existence of exactly 3 scalene triangles with integer side lengths and perimeter < 13. -/
theorem scalene_triangles_count : 
  ∃ (triangles : Finset (ℕ × ℕ × ℕ)), 
    triangles.card = 3 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ triangles → a < b ∧ b < c ∧ a + b + c < 13 :=
sorry

end scalene_triangles_count_l2160_216061


namespace mary_income_percentage_l2160_216032

-- Declare noncomputable as necessary
noncomputable def calculate_percentage_more
    (J : ℝ) -- Juan's income
    (T : ℝ) (M : ℝ)
    (hT : T = 0.70 * J) -- Tim's income is 30% less than Juan's income
    (hM : M = 1.12 * J) -- Mary's income is 112% of Juan's income
    : ℝ :=
  ((M - T) / T) * 100

theorem mary_income_percentage
    (J T M : ℝ)
    (hT : T = 0.70 * J)
    (hM : M = 1.12 * J) :
    calculate_percentage_more J T M hT hM = 60 :=
by sorry

end mary_income_percentage_l2160_216032


namespace ticket_cost_l2160_216046

-- Conditions
def seats : ℕ := 400
def capacity_percentage : ℝ := 0.8
def performances : ℕ := 3
def total_revenue : ℝ := 28800

-- Question: Prove that the cost of each ticket is $30
theorem ticket_cost : (total_revenue / (seats * capacity_percentage * performances)) = 30 := 
by
  sorry

end ticket_cost_l2160_216046


namespace fraction_zero_solution_l2160_216071

theorem fraction_zero_solution (x : ℝ) (h1 : |x| - 3 = 0) (h2 : x + 3 ≠ 0) : x = 3 := 
sorry

end fraction_zero_solution_l2160_216071


namespace first_equation_value_l2160_216073

theorem first_equation_value (x y : ℝ) (V : ℝ) 
  (h1 : x + |x| + y = V) 
  (h2 : x + |y| - y = 6) 
  (h3 : x + y = 12) : 
  V = 18 := 
by
  sorry

end first_equation_value_l2160_216073


namespace reflected_ray_equation_l2160_216075

theorem reflected_ray_equation (x y : ℝ) (incident_ray : y = 2 * x + 1) (reflecting_line : y = x) :
  x - 2 * y - 1 = 0 :=
sorry

end reflected_ray_equation_l2160_216075


namespace johns_final_push_time_l2160_216091

-- Definitions and assumptions
def john_speed : ℝ := 4.2
def steve_speed : ℝ := 3.8
def initial_gap : ℝ := 15
def final_gap : ℝ := 2

theorem johns_final_push_time :
  ∃ t : ℝ, john_speed * t = steve_speed * t + initial_gap + final_gap ∧ t = 42.5 :=
by
  sorry

end johns_final_push_time_l2160_216091


namespace LittleJohnnyAnnualIncome_l2160_216079

theorem LittleJohnnyAnnualIncome :
  ∀ (total_amount bank_amount bond_amount : ℝ) 
    (bank_interest bond_interest annual_income : ℝ),
    total_amount = 10000 →
    bank_amount = 6000 →
    bond_amount = 4000 →
    bank_interest = 0.05 →
    bond_interest = 0.09 →
    annual_income = bank_amount * bank_interest + bond_amount * bond_interest →
    annual_income = 660 :=
by
  intros total_amount bank_amount bond_amount bank_interest bond_interest annual_income 
  intros h_total_amount h_bank_amount h_bond_amount h_bank_interest h_bond_interest h_annual_income
  -- Proof is not required
  sorry

end LittleJohnnyAnnualIncome_l2160_216079


namespace pure_imaginary_iff_a_eq_2_l2160_216026

theorem pure_imaginary_iff_a_eq_2 (a : ℝ) : (∃ k : ℝ, (∃ x : ℝ, (2-a) / 2 = x ∧ x = 0) ∧ (2+a)/2 = k ∧ k ≠ 0) ↔ a = 2 :=
by
  sorry

end pure_imaginary_iff_a_eq_2_l2160_216026


namespace badminton_costs_l2160_216005

variables (x : ℕ) (h : x > 16)

-- Define costs at Store A and Store B
def cost_A : ℕ := 1760 + 40 * x
def cost_B : ℕ := 1920 + 32 * x

-- Lean statement to prove the costs
theorem badminton_costs : 
  cost_A x = 1760 + 40 * x ∧ cost_B x = 1920 + 32 * x :=
by {
  -- This proof is expected but not required for the task
  sorry
}

end badminton_costs_l2160_216005


namespace initial_observations_l2160_216008

theorem initial_observations {n : ℕ} (S : ℕ) (new_observation : ℕ) 
  (h1 : S = 15 * n) (h2 : new_observation = 14 - n)
  (h3 : (S + new_observation) / (n + 1) = 14) : n = 6 :=
sorry

end initial_observations_l2160_216008


namespace compute_expression_l2160_216059

theorem compute_expression (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2017)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2016)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2017)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2016)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2017)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2016) :
  (2 - x1 / y1) * (2 - x2 / y2) * (2 - x3 / y3) = 26219 / 2016 := 
by
  sorry

end compute_expression_l2160_216059


namespace sum_of_unit_fractions_l2160_216037

theorem sum_of_unit_fractions : (1 / 2) + (1 / 3) + (1 / 7) + (1 / 42) = 1 := 
by 
  sorry

end sum_of_unit_fractions_l2160_216037


namespace sequence_general_term_l2160_216054

theorem sequence_general_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n + 1) : a n = n * n :=
by
  sorry

end sequence_general_term_l2160_216054


namespace find_t_l2160_216036

-- Define the utility function
def utility (r j : ℕ) : ℕ := r * j

-- Define the Wednesday and Thursday utilities
def utility_wednesday (t : ℕ) : ℕ := utility (t + 1) (7 - t)
def utility_thursday (t : ℕ) : ℕ := utility (3 - t) (t + 4)

theorem find_t : (utility_wednesday t = utility_thursday t) → t = 5 / 8 :=
by
  sorry

end find_t_l2160_216036


namespace polynomial_remainder_l2160_216065

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^3 + 2*x + 3

-- Define the divisor q(x)
def q (x : ℝ) : ℝ := x + 2

-- The theorem asserting the remainder when p(x) is divided by q(x)
theorem polynomial_remainder : (p (-2)) = -9 :=
by
  sorry

end polynomial_remainder_l2160_216065


namespace vasya_wins_l2160_216067

-- Define the grid size and initial setup
def grid_size : ℕ := 13
def initial_stones : ℕ := 2023

-- Define a condition that checks if a move can put a stone on the 13th cell
def can_win (position : ℕ) : Prop :=
  position = grid_size

-- Define the game logic for Petya and Vasya
def next_position (pos : ℕ) (move : ℕ) : ℕ :=
  pos + move

-- Ensure a win by always ensuring the next move does not leave Petya on positions 4, 7, 10, 13
def winning_strategy_for_vasya (current_pos : ℕ) (move : ℕ) : Prop :=
  (next_position current_pos move) ≠ 4 ∧
  (next_position current_pos move) ≠ 7 ∧
  (next_position current_pos move) ≠ 10 ∧
  (next_position current_pos move) ≠ 13

theorem vasya_wins : ∃ strategy : ℕ → ℕ → Prop,
  ∀ current_pos move, winning_strategy_for_vasya current_pos move → can_win (next_position current_pos move) :=
by
  sorry -- To be provided

end vasya_wins_l2160_216067


namespace grid_X_value_l2160_216050

theorem grid_X_value :
  ∃ X, (∃ b d1 d2 d3 d4, 
    b = 16 ∧
    d1 = (25 - 20) ∧
    d2 = (16 - 15) / 3 ∧
    d3 = (d1 * 5) / 4 ∧
    d4 = d1 - d3 ∧
    (-12 - d4 * 4) = -30 ∧ 
    X = d4 ∧
    X = 10.5) :=
sorry

end grid_X_value_l2160_216050


namespace sum_of_roots_l2160_216038

theorem sum_of_roots (α β : ℝ)
  (hα : α^3 - 3*α^2 + 5*α - 4 = 0)
  (hβ : β^3 - 3*β^2 + 5*β - 2 = 0) :
  α + β = 2 :=
sorry

end sum_of_roots_l2160_216038


namespace number_of_perfect_square_factors_of_360_l2160_216066

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

def number_of_perfect_square_factors (n : ℕ) : ℕ :=
  if n = 360 then 4 else 0

theorem number_of_perfect_square_factors_of_360 :
  number_of_perfect_square_factors 360 = 4 := 
by {
  -- Sorry is used here as a placeholder for the proof steps.
  sorry
}

end number_of_perfect_square_factors_of_360_l2160_216066


namespace right_pyramid_volume_l2160_216025

noncomputable def volume_of_right_pyramid (base_area lateral_face_area total_surface_area : ℝ) : ℝ := 
  let height := (10 : ℝ) / 3
  (1 / 3) * base_area * height

theorem right_pyramid_volume (total_surface_area base_area lateral_face_area : ℝ)
  (h0 : total_surface_area = 300)
  (h1 : base_area + 3 * lateral_face_area = total_surface_area)
  (h2 : lateral_face_area = base_area / 3) 
  : volume_of_right_pyramid base_area lateral_face_area total_surface_area = 500 / 3 := 
by
  sorry

end right_pyramid_volume_l2160_216025


namespace Kim_min_score_for_target_l2160_216078

noncomputable def Kim_exam_scores : List ℚ := [86, 82, 89]

theorem Kim_min_score_for_target :
  ∃ x : ℚ, ↑((Kim_exam_scores.sum + x) / (Kim_exam_scores.length + 1) ≥ (Kim_exam_scores.sum / Kim_exam_scores.length) + 2)
  ∧ x = 94 := sorry

end Kim_min_score_for_target_l2160_216078


namespace min_x9_minus_x1_l2160_216007

theorem min_x9_minus_x1
  (x : Fin 9 → ℕ)
  (h_pos : ∀ i, x i > 0)
  (h_sorted : ∀ i j, i < j → x i < x j)
  (h_sum : (Finset.univ.sum x) = 220) :
    ∃ x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℕ,
    x1 < x2 ∧ x2 < x3 ∧ x3 < x4 ∧ x4 < x5 ∧ x5 < x6 ∧ x6 < x7 ∧ x7 < x8 ∧ x8 < x9 ∧
    (x1 + x2 + x3 + x4 + x5 = 110) ∧
    x1 = x 0 ∧ x2 = x 1 ∧ x3 = x 2 ∧ x4 = x 3 ∧ x5 = x 4 ∧ x6 = x 5 ∧ x7 = x 6 ∧ x8 = x 7 ∧ x9 = x 8
    ∧ (x9 - x1 = 9) :=
sorry

end min_x9_minus_x1_l2160_216007


namespace problem_a_problem_b_l2160_216011

-- Define given points and lines
variables (A B P Q R L T K S : Type) 
variables (l : A) -- line through A
variables (a : A) -- line through A perpendicular to l
variables (b : B) -- line through B perpendicular to l
variables (PQ_intersects_a : Q) (PR_intersects_b : R)
variables (line_through_A_perp_BQ : L) (line_through_B_perp_AR : K)
variables (intersects_BQ_at_L : L) (intersects_BR_at_T : T)
variables (intersects_AR_at_K : K) (intersects_AQ_at_S : S)

-- Define collinearity properties
def collinear (X Y Z : Type) : Prop := sorry

-- Formalize the mathematical proofs as Lean theorems
theorem problem_a : collinear P T S :=
sorry

theorem problem_b : collinear P K L :=
sorry

end problem_a_problem_b_l2160_216011


namespace condition_suff_not_necess_l2160_216012

theorem condition_suff_not_necess (x : ℝ) (h : |x - (1 / 2)| < 1 / 2) : x^3 < 1 :=
by
  have h1 : 0 < x := sorry
  have h2 : x < 1 := sorry
  sorry

end condition_suff_not_necess_l2160_216012


namespace number_of_correct_conclusions_l2160_216003

theorem number_of_correct_conclusions : 
    (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
    (∀ x : ℝ, (x ≠ 0 → x - Real.sin x ≠ 0)) ∧
    (∀ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) ∧
    (¬ (∀ x : ℝ, x - Real.log x > 0))
    → 3 = 3 :=
by
  sorry

end number_of_correct_conclusions_l2160_216003


namespace unique_solution_set_l2160_216043

theorem unique_solution_set :
  {a : ℝ | ∃! x : ℝ, (x^2 - 4) / (x + a) = 1} = { -17 / 4, -2, 2 } :=
by sorry

end unique_solution_set_l2160_216043


namespace solve_equation_l2160_216049

theorem solve_equation (x : ℝ) (h : x ≠ 3) : 
  -x^2 = (3*x - 3) / (x - 3) → x = 1 :=
by
  intro h1
  sorry

end solve_equation_l2160_216049


namespace total_cost_of_pencils_and_erasers_l2160_216034

theorem total_cost_of_pencils_and_erasers 
  (pencil_cost : ℕ)
  (eraser_cost : ℕ)
  (pencils_bought : ℕ)
  (erasers_bought : ℕ)
  (total_cost_dollars : ℝ)
  (cents_to_dollars : ℝ)
  (hc : pencil_cost = 2)
  (he : eraser_cost = 5)
  (hp : pencils_bought = 500)
  (he2 : erasers_bought = 250)
  (cents_to_dollars_def : cents_to_dollars = 100)
  (total_cost_calc : total_cost_dollars = 
    ((pencils_bought * pencil_cost + erasers_bought * eraser_cost : ℕ) : ℝ) / cents_to_dollars) 
  : total_cost_dollars = 22.50 :=
sorry

end total_cost_of_pencils_and_erasers_l2160_216034


namespace triangle_inequality_power_sum_l2160_216069

theorem triangle_inequality_power_sum
  (a b c : ℝ) (n : ℕ)
  (h_a_bc : a + b + c = 1)
  (h_a_b_c : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c)
  (h_a_triangl : a + b > c)
  (h_b_triangl : b + c > a)
  (h_c_triangl : c + a > b)
  (h_n : n > 1) :
  (a^n + b^n)^(1/n : ℝ) + (b^n + c^n)^(1/n : ℝ) + (c^n + a^n)^(1/n : ℝ) < 1 + (2^(1/n : ℝ)) / 2 :=
by
  sorry

end triangle_inequality_power_sum_l2160_216069


namespace jens_son_age_l2160_216063

theorem jens_son_age
  (J : ℕ)
  (S : ℕ)
  (h1 : J = 41)
  (h2 : J = 3 * S - 7) :
  S = 16 :=
by
  sorry

end jens_son_age_l2160_216063


namespace find_a9_l2160_216015

variable (a : ℕ → ℤ)

-- Condition 1: The sequence is arithmetic
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

-- Condition 2: Given a_4 = 5
def a4_value (a : ℕ → ℤ) : Prop :=
  a 4 = 5

-- Condition 3: Given a_5 = 4
def a5_value (a : ℕ → ℤ) : Prop :=
  a 5 = 4

-- Problem: Prove a_9 = 0
theorem find_a9 (h1 : arithmetic_sequence a) (h2 : a4_value a) (h3 : a5_value a) : a 9 = 0 := 
sorry

end find_a9_l2160_216015


namespace sum_of_three_numbers_l2160_216074

theorem sum_of_three_numbers :
  ∃ A B C : ℕ, 
    (100 ≤ A ∧ A < 1000) ∧  -- A is a three-digit number
    (10 ≤ B ∧ B < 100) ∧     -- B is a two-digit number
    (10 ≤ C ∧ C < 100) ∧     -- C is a two-digit number
    (A + (if (B / 10 = 7 ∨ B % 10 = 7) then B else 0) + 
       (if (C / 10 = 7 ∨ C % 10 = 7) then C else 0) = 208) ∧
    (if (B / 10 = 3 ∨ B % 10 = 3) then B else 0) + 
    (if (C / 10 = 3 ∨ C % 10 = 3) then C else 0) = 76 ∧
    A + B + C = 247 :=
by
  sorry

end sum_of_three_numbers_l2160_216074


namespace g_value_at_neg3_l2160_216095

noncomputable def g : ℚ → ℚ := sorry

theorem g_value_at_neg3 (h : ∀ x : ℚ, x ≠ 0 → 4 * g (1 / x) + 3 * g x / x = 2 * x^2) : 
  g (-3) = 98 / 13 := 
sorry

end g_value_at_neg3_l2160_216095


namespace find_baseball_deck_price_l2160_216096

variables (numberOfBasketballPacks : ℕ) (pricePerBasketballPack : ℝ) (numberOfBaseballDecks : ℕ)
           (totalMoney : ℝ) (changeReceived : ℝ) (totalSpent : ℝ) (spentOnBasketball : ℝ) (baseballDeckPrice : ℝ)

noncomputable def problem_conditions : Prop :=
  numberOfBasketballPacks = 2 ∧
  pricePerBasketballPack = 3 ∧
  numberOfBaseballDecks = 5 ∧
  totalMoney = 50 ∧
  changeReceived = 24 ∧
  totalSpent = totalMoney - changeReceived ∧
  spentOnBasketball = numberOfBasketballPacks * pricePerBasketballPack ∧
  totalSpent = spentOnBasketball + (numberOfBaseballDecks * baseballDeckPrice)

theorem find_baseball_deck_price (h : problem_conditions numberOfBasketballPacks pricePerBasketballPack numberOfBaseballDecks totalMoney changeReceived totalSpent spentOnBasketball baseballDeckPrice) :
  baseballDeckPrice = 4 :=
sorry

end find_baseball_deck_price_l2160_216096


namespace range_of_a_exists_x_l2160_216017

theorem range_of_a_exists_x (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x - x ^ 2 ≥ a) ↔ a ≤ 1 := 
sorry

end range_of_a_exists_x_l2160_216017


namespace combined_instruments_l2160_216031

def charlie_flutes := 1
def charlie_horns := 2
def charlie_harps := 1

def carli_flutes := 2 * charlie_flutes
def carli_horns := charlie_horns / 2
def carli_harps := 0

def charlie_total := charlie_flutes + charlie_horns + charlie_harps
def carli_total := carli_flutes + carli_horns + carli_harps
def combined_total := charlie_total + carli_total

theorem combined_instruments :
  combined_total = 7 :=
by
  sorry

end combined_instruments_l2160_216031


namespace triangle_problem_l2160_216033

noncomputable def a : ℝ := 2 * Real.sqrt 3
noncomputable def B : ℝ := 45
noncomputable def S : ℝ := 3 + Real.sqrt 3

noncomputable def c : ℝ := Real.sqrt 2 + Real.sqrt 6
noncomputable def C : ℝ := 75

theorem triangle_problem
  (a_val : a = 2 * Real.sqrt 3)
  (B_val : B = 45)
  (S_val : S = 3 + Real.sqrt 3) :
  c = Real.sqrt 2 + Real.sqrt 6 ∧ C = 75 :=
by
  sorry

end triangle_problem_l2160_216033


namespace Christopher_joggers_eq_80_l2160_216028

variable (T A C : ℕ)

axiom Tyson_joggers : T > 0                  -- Tyson bought a positive number of joggers.

axiom Alexander_condition : A = T + 22        -- Alexander bought 22 more joggers than Tyson.

axiom Christopher_condition : C = 20 * T      -- Christopher bought twenty times as many joggers as Tyson.

axiom Christopher_Alexander : C = A + 54     -- Christopher bought 54 more joggers than Alexander.

theorem Christopher_joggers_eq_80 : C = 80 := 
by
  sorry

end Christopher_joggers_eq_80_l2160_216028


namespace paint_cost_of_cube_l2160_216004

theorem paint_cost_of_cube (side_length cost_per_kg coverage_per_kg : ℝ) (h₀ : side_length = 10) 
(h₁ : cost_per_kg = 60) (h₂ : coverage_per_kg = 20) : 
(cost_per_kg * (6 * (side_length^2) / coverage_per_kg) = 1800) :=
by
  sorry

end paint_cost_of_cube_l2160_216004


namespace sum_modulo_remainder_l2160_216040

theorem sum_modulo_remainder :
  ((82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17) = 12 :=
by
  sorry

end sum_modulo_remainder_l2160_216040


namespace sara_lunch_total_l2160_216002

theorem sara_lunch_total :
  let hotdog := 5.36
  let salad := 5.10
  hotdog + salad = 10.46 :=
by
  let hotdog := 5.36
  let salad := 5.10
  sorry

end sara_lunch_total_l2160_216002


namespace area_of_rhombus_l2160_216029

variable (a b θ : ℝ)
variable (h_a : 0 < a) (h_b : 0 < b)

theorem area_of_rhombus (h : true) : (2 * a) * (2 * b) / 2 = 2 * a * b := by
  sorry

end area_of_rhombus_l2160_216029


namespace arithmetic_geometric_sequence_l2160_216006

theorem arithmetic_geometric_sequence {a b c x y : ℝ} (h₁: a ≠ b) (h₂: b ≠ c) (h₃: a ≠ c)
  (h₄ : 2 * b = a + c) (h₅ : x^2 = a * b) (h₆ : y^2 = b * c) :
  (x^2 + y^2 = 2 * b^2) ∧ (x^2 * y^2 ≠ b^4) :=
by {
  sorry
}

end arithmetic_geometric_sequence_l2160_216006


namespace convert_base_5_to_base_10_l2160_216030

theorem convert_base_5_to_base_10 :
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  a3 + a2 + a1 + a0 = 302 := by
  let a3 := 2 * 5^3
  let a2 := 2 * 5^2
  let a1 := 0 * 5^1
  let a0 := 2 * 5^0
  show a3 + a2 + a1 + a0 = 302
  sorry

end convert_base_5_to_base_10_l2160_216030


namespace shorter_piece_length_l2160_216094

theorem shorter_piece_length (total_len : ℝ) (h1 : total_len = 60)
                            (short_len long_len : ℝ) (h2 : long_len = (1 / 2) * short_len)
                            (h3 : short_len + long_len = total_len) :
  short_len = 40 := 
  sorry

end shorter_piece_length_l2160_216094


namespace zero_is_smallest_natural_number_l2160_216070

theorem zero_is_smallest_natural_number : ∀ n : ℕ, 0 ≤ n :=
by
  intro n
  exact Nat.zero_le n

#check zero_is_smallest_natural_number  -- confirming the theorem check

end zero_is_smallest_natural_number_l2160_216070


namespace verify_sum_of_fourth_powers_l2160_216092

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_fourth_powers (n : ℕ) : ℕ :=
  ((n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30)

noncomputable def square_of_sum (n : ℕ) : ℕ :=
  (n * (n + 1) / 2)^2

theorem verify_sum_of_fourth_powers (n : ℕ) :
  5 * sum_of_fourth_powers n = (4 * n + 2) * square_of_sum n - sum_of_squares n := 
  sorry

end verify_sum_of_fourth_powers_l2160_216092
