import Mathlib

namespace manolo_makes_45_masks_in_four_hours_l132_132959

noncomputable def face_masks_in_four_hour_shift : ℕ :=
  let first_hour_rate := 4
  let subsequent_hour_rate := 6
  let first_hour_face_masks := 60 / first_hour_rate
  let subsequent_hours_face_masks_per_hour := 60 / subsequent_hour_rate
  let total_face_masks :=
    first_hour_face_masks + subsequent_hours_face_masks_per_hour * (4 - 1)
  total_face_masks

theorem manolo_makes_45_masks_in_four_hours :
  face_masks_in_four_hour_shift = 45 :=
 by sorry

end manolo_makes_45_masks_in_four_hours_l132_132959


namespace min_value_of_2a_plus_3b_l132_132445

theorem min_value_of_2a_plus_3b
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_perpendicular : (x - (2 * b - 3) * y + 6 = 0) ∧ (2 * b * x + a * y - 5 = 0)) :
  2 * a + 3 * b = 25 / 2 :=
sorry

end min_value_of_2a_plus_3b_l132_132445


namespace pencils_per_friend_l132_132801

theorem pencils_per_friend (total_pencils num_friends : ℕ) (h_total : total_pencils = 24) (h_friends : num_friends = 3) : total_pencils / num_friends = 8 :=
by
  -- Proof would go here
  sorry

end pencils_per_friend_l132_132801


namespace vincent_books_cost_l132_132493

theorem vincent_books_cost :
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  cost_per_book = 16 :=
by
  let num_animals := 10
  let num_outer_space := 1
  let num_trains := 3
  let total_books := num_animals + num_outer_space + num_trains
  let total_spent := 224
  let cost_per_book := total_spent / total_books
  show cost_per_book = 16
  sorry

end vincent_books_cost_l132_132493


namespace not_necessarily_divisible_by_28_l132_132783

theorem not_necessarily_divisible_by_28 (k : ℤ) (h : 7 ∣ (k * (k + 1) * (k + 2))) : ¬ (28 ∣ (k * (k + 1) * (k + 2))) :=
sorry

end not_necessarily_divisible_by_28_l132_132783


namespace food_waste_in_scientific_notation_l132_132709

-- Given condition that 1 billion equals 10^9
def billion : ℕ := 10 ^ 9

-- Problem statement: expressing 530 billion kilograms in scientific notation
theorem food_waste_in_scientific_notation :
  (530 * billion : ℝ) = 5.3 * 10^10 := 
  sorry

end food_waste_in_scientific_notation_l132_132709


namespace passing_marks_l132_132926

variable (T P : ℝ)

theorem passing_marks :
  (0.35 * T = P - 40) →
  (0.60 * T = P + 25) →
  P = 131 :=
by
  intro h1 h2
  -- Proof steps should follow here.
  sorry

end passing_marks_l132_132926


namespace cost_of_largest_pot_is_2_52_l132_132319

/-
Mark bought a set of 6 flower pots of different sizes at a total pre-tax cost.
Each pot cost 0.4 more than the next one below it in size.
The total cost, including a sales tax of 7.5%, was $9.80.
Prove that the cost of the largest pot before sales tax was $2.52.
-/

def cost_smallest_pot (x : ℝ) : Prop :=
  let total_cost := x + (x + 0.4) + (x + 0.8) + (x + 1.2) + (x + 1.6) + (x + 2.0)
  let pre_tax_cost := total_cost / 1.075
  let pre_tax_total_cost := (9.80 / 1.075)
  (total_cost = 6 * x + 6 ∧ total_cost = pre_tax_total_cost) →
  (x + 2.0 = 2.52)

theorem cost_of_largest_pot_is_2_52 :
  ∃ x : ℝ, cost_smallest_pot x :=
sorry

end cost_of_largest_pot_is_2_52_l132_132319


namespace function_ordering_l132_132876

-- Definitions for the function and conditions
variable (f : ℝ → ℝ)

-- Assuming properties of the function
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodicity : ∀ x, f (x + 4) = -f x
axiom increasing_on : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 2 → f x < f y

-- Main theorem statement
theorem function_ordering : f (-25) < f 80 ∧ f 80 < f 11 :=
by 
  sorry

end function_ordering_l132_132876


namespace train_speed_proof_l132_132381

noncomputable def train_speed_kmh (length_train : ℝ) (time_crossing : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := length_train / time_crossing
  let train_speed_ms := relative_speed - man_speed_ms
  train_speed_ms * (3600 / 1000)

theorem train_speed_proof :
  train_speed_kmh 150 8 7 = 60.5 :=
by
  sorry

end train_speed_proof_l132_132381


namespace number_of_ways_is_64_l132_132995

-- Definition of the problem conditions
def ways_to_sign_up (students groups : ℕ) : ℕ :=
  groups ^ students

-- Theorem statement asserting that for 3 students and 4 groups, the number of ways is 64
theorem number_of_ways_is_64 : ways_to_sign_up 3 4 = 64 :=
by sorry

end number_of_ways_is_64_l132_132995


namespace janet_earnings_eur_l132_132078

noncomputable def usd_to_eur (usd : ℚ) : ℚ :=
  usd * 0.85

def janet_earnings_usd : ℚ :=
  (130 * 0.25) + (90 * 0.30) + (30 * 0.40)

theorem janet_earnings_eur : usd_to_eur janet_earnings_usd = 60.78 :=
  by
    sorry

end janet_earnings_eur_l132_132078


namespace distance_between_x_intercepts_l132_132738

theorem distance_between_x_intercepts :
  let slope1 := 4
  let slope2 := -2
  let point := (8, 20)
  let line1 (x : ℝ) := slope1 * (x - point.1) + point.2
  let line2 (x : ℝ) := slope2 * (x - point.1) + point.2
  let x_intercept1 := (0 - point.2) / slope1 + point.1
  let x_intercept2 := (0 - point.2) / slope2 + point.1
  abs (x_intercept1 - x_intercept2) = 15 := sorry

end distance_between_x_intercepts_l132_132738


namespace granddaughter_fraction_l132_132207

noncomputable def betty_age : ℕ := 60
def fraction_younger (p : ℕ) : ℕ := (p * 40) / 100
noncomputable def daughter_age : ℕ := betty_age - fraction_younger betty_age
def granddaughter_age : ℕ := 12
def fraction (a b : ℕ) : ℚ := a / b

theorem granddaughter_fraction :
  fraction granddaughter_age daughter_age = 1 / 3 := 
by
  sorry

end granddaughter_fraction_l132_132207


namespace roots_square_difference_l132_132457

theorem roots_square_difference (a b : ℚ)
  (ha : 6 * a^2 + 13 * a - 28 = 0)
  (hb : 6 * b^2 + 13 * b - 28 = 0) : (a - b)^2 = 841 / 36 :=
sorry

end roots_square_difference_l132_132457


namespace pipe_fills_cistern_l132_132551

theorem pipe_fills_cistern (t : ℕ) (h : t = 5) : 11 * t = 55 :=
by
  sorry

end pipe_fills_cistern_l132_132551


namespace abe_age_is_22_l132_132188

-- Define the conditions of the problem
def abe_age_condition (A : ℕ) : Prop := A + (A - 7) = 37

-- State the theorem
theorem abe_age_is_22 : ∃ A : ℕ, abe_age_condition A ∧ A = 22 :=
by
  sorry

end abe_age_is_22_l132_132188


namespace always_in_range_l132_132934

noncomputable def g (x k : ℝ) : ℝ := x^2 + 2 * k * x + 1

theorem always_in_range (k : ℝ) : 
  ∃ x : ℝ, g x k = 3 :=
by
  sorry

end always_in_range_l132_132934


namespace small_bottles_needed_l132_132622

noncomputable def small_bottle_capacity := 40 -- in milliliters
noncomputable def large_bottle_capacity := 540 -- in milliliters
noncomputable def worst_case_small_bottle_capacity := 38 -- in milliliters

theorem small_bottles_needed :
  let n_bottles := Int.ceil (large_bottle_capacity / worst_case_small_bottle_capacity : ℚ)
  n_bottles = 15 :=
by
  sorry

end small_bottles_needed_l132_132622


namespace inequality_solution_l132_132335

theorem inequality_solution (x : ℝ) : 
  (x + 1) * (2 - x) < 0 ↔ x < -1 ∨ x > 2 := 
sorry

end inequality_solution_l132_132335


namespace largest_non_zero_ending_factor_decreasing_number_l132_132621

theorem largest_non_zero_ending_factor_decreasing_number :
  ∃ n: ℕ, n = 180625 ∧ (n % 10 ≠ 0) ∧ (∃ m: ℕ, m < n ∧ (n % m = 0) ∧ (n / 10 ≤ m ∧ m * 10 > 0)) :=
by {
  sorry
}

end largest_non_zero_ending_factor_decreasing_number_l132_132621


namespace monotonic_when_a_is_neg1_find_extreme_points_l132_132349

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/3) * x ^ 3 - (1/2) * (a^2 + a + 2) * x ^ 2 + a^2 * (a + 2) * x

theorem monotonic_when_a_is_neg1 :
  ∀ x : ℝ, f x (-1) ≤ f x (-1) :=
sorry

theorem find_extreme_points (a : ℝ) :
  if h : a = -1 ∨ a = 2 then
    True  -- The function is monotonically increasing, no extreme points
  else if h : a < -1 ∨ a > 2 then
    ∃ x_max x_min : ℝ, x_max = a + 2 ∧ x_min = a^2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) 
  else
    ∃ x_max x_min : ℝ, x_max = a^2 ∧ x_min = a + 2 ∧ (f x_max a ≥ f x a ∧ f x_min a ≤ f x a) :=
sorry

end monotonic_when_a_is_neg1_find_extreme_points_l132_132349


namespace total_spent_on_toys_and_clothes_l132_132727

def cost_toy_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_toy_trucks : ℝ := 5.86
def cost_pants : ℝ := 14.55
def cost_shirt : ℝ := 7.43
def cost_hat : ℝ := 12.50

theorem total_spent_on_toys_and_clothes :
  (cost_toy_cars + cost_skateboard + cost_toy_trucks) + (cost_pants + cost_shirt + cost_hat) = 60.10 :=
by
  sorry

end total_spent_on_toys_and_clothes_l132_132727


namespace min_value_2a_plus_b_l132_132139

theorem min_value_2a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + b = a^2 + a * b) :
  2 * a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end min_value_2a_plus_b_l132_132139


namespace sequence_general_formula_l132_132002

theorem sequence_general_formula (a : ℕ → ℚ) 
  (h1 : a 1 = 1 / 2) 
  (h_rec : ∀ n : ℕ, a (n + 2) = 3 * a (n + 1) / (a (n + 1) + 3)) 
  (n : ℕ) : 
  a (n + 1) = 3 / (n + 6) :=
by
  sorry

end sequence_general_formula_l132_132002


namespace john_spending_l132_132824

open Nat Real

noncomputable def cost_of_silver (silver_ounce: Real) (silver_price: Real) : Real :=
  silver_ounce * silver_price

noncomputable def quantity_of_gold (silver_ounce: Real): Real :=
  2 * silver_ounce

noncomputable def cost_per_ounce_gold (silver_price: Real) (multiplier: Real): Real :=
  silver_price * multiplier

noncomputable def cost_of_gold (gold_ounce: Real) (gold_price: Real) : Real :=
  gold_ounce * gold_price

noncomputable def total_cost (cost_silver: Real) (cost_gold: Real): Real :=
  cost_silver + cost_gold

theorem john_spending :
  let silver_ounce := 1.5
  let silver_price := 20
  let gold_multiplier := 50
  let cost_silver := cost_of_silver silver_ounce silver_price
  let gold_ounce := quantity_of_gold silver_ounce
  let gold_price := cost_per_ounce_gold silver_price gold_multiplier
  let cost_gold := cost_of_gold gold_ounce gold_price
  let total := total_cost cost_silver cost_gold
  total = 3030 :=
by
  sorry

end john_spending_l132_132824


namespace journey_speed_second_half_l132_132347

theorem journey_speed_second_half (total_time : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (v : ℝ) : 
  total_time = 10 ∧ first_half_speed = 21 ∧ total_distance = 224 →
  v = 24 :=
by
  intro h
  sorry

end journey_speed_second_half_l132_132347


namespace proposition_statementC_l132_132989

-- Definitions of each statement
def statementA := "Draw a parallel line to line AB"
def statementB := "Take a point C on segment AB"
def statementC := "The complement of equal angles are equal"
def statementD := "Is the perpendicular segment the shortest?"

-- Proving that among the statements A, B, C, and D, statement C is the proposition
theorem proposition_statementC : 
  (statementC = "The complement of equal angles are equal") :=
by
  -- We assume it directly from the equivalence given in the problem statement
  sorry

end proposition_statementC_l132_132989


namespace p_at_zero_l132_132288

-- Define the quartic monic polynomial
noncomputable def p (x : ℝ) : ℝ := sorry

-- Conditions
axiom p_monic : true -- p is a monic polynomial, we represent it by an axiom here for simplicity
axiom p_neg2 : p (-2) = -4
axiom p_1 : p (1) = -1
axiom p_3 : p (3) = -9
axiom p_5 : p (5) = -25

-- The theorem to be proven
theorem p_at_zero : p 0 = -30 := by
  sorry

end p_at_zero_l132_132288


namespace polynomial_mult_of_6_l132_132079

theorem polynomial_mult_of_6 (P : Polynomial ℤ)
  (h1 : 6 ∣ P.eval 2)
  (h2 : 6 ∣ P.eval 3) : 6 ∣ P.eval 5 := 
sorry

end polynomial_mult_of_6_l132_132079


namespace max_product_production_l132_132223

theorem max_product_production (C_mats A_mats C_ship A_ship B_mats B_ship : ℝ)
  (cost_A cost_B ship_A ship_B : ℝ) (prod_A prod_B max_cost_mats max_cost_ship prod_max : ℝ)
  (h_prod_A : prod_A = 90)
  (h_cost_A : cost_A = 1000)
  (h_ship_A : ship_A = 500)
  (h_prod_B : prod_B = 100)
  (h_cost_B : cost_B = 1500)
  (h_ship_B : ship_B = 400)
  (h_max_cost_mats : max_cost_mats = 6000)
  (h_max_cost_ship : max_cost_ship = 2000)
  (h_prod_max : prod_max = 440)
  (H_C_mats : C_mats = cost_A * A_mats + cost_B * B_mats)
  (H_C_ship : C_ship = ship_A * A_ship + ship_B * B_ship)
  (H_A_mats_ship : A_mats = A_ship)
  (H_B_mats_ship : B_mats = B_ship)
  (H_C_mats_le : C_mats ≤ max_cost_mats)
  (H_C_ship_le : C_ship ≤ max_cost_ship) :
  prod_A * A_mats + prod_B * B_mats ≤ prod_max :=
by {
  sorry
}

end max_product_production_l132_132223


namespace evaluate_expression_l132_132953

theorem evaluate_expression (b : ℕ) (hb : b = 2) : (b^3 * b^4) - b^2 = 124 :=
by
  -- leave the proof empty with a placeholder
  sorry

end evaluate_expression_l132_132953


namespace pyramid_surface_area_l132_132048

noncomputable def total_surface_area : Real :=
  let ab := 14
  let bc := 8
  let pf := 15
  let base_area := ab * bc
  let fm := ab / 2
  let pm_ab := Real.sqrt (pf^2 + fm^2)
  let pm_bc := Real.sqrt (pf^2 + (bc / 2)^2)
  base_area + 2 * (ab / 2 * pm_ab) + 2 * (bc / 2 * pm_bc)

theorem pyramid_surface_area :
  total_surface_area = 112 + 14 * Real.sqrt 274 + 8 * Real.sqrt 241 := by
  sorry

end pyramid_surface_area_l132_132048


namespace survey_steps_correct_l132_132974

theorem survey_steps_correct :
  ∀ steps : (ℕ → ℕ), (steps 1 = 2) → (steps 2 = 4) → (steps 3 = 3) → (steps 4 = 1) → True :=
by
  intros steps h1 h2 h3 h4
  exact sorry

end survey_steps_correct_l132_132974


namespace geometric_sequence_a3_l132_132983

theorem geometric_sequence_a3 (a : ℕ → ℝ) (q : ℝ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 4 = 8)
  (h3 : ∀ k : ℕ, a (k + 1) = a k * q) : a 3 = 4 :=
sorry

end geometric_sequence_a3_l132_132983


namespace white_balls_count_l132_132957

theorem white_balls_count (W B R : ℕ) (h1 : B = W + 14) (h2 : R = 3 * (B - W)) (h3 : W + B + R = 1000) : W = 472 :=
sorry

end white_balls_count_l132_132957


namespace count_total_coins_l132_132282

theorem count_total_coins (quarters nickels : Nat) (h₁ : quarters = 4) (h₂ : nickels = 8) : quarters + nickels = 12 :=
by sorry

end count_total_coins_l132_132282


namespace magician_weeks_worked_l132_132237

theorem magician_weeks_worked
  (hourly_rate : ℕ)
  (hours_per_day : ℕ)
  (total_payment : ℕ)
  (days_per_week : ℕ)
  (h1 : hourly_rate = 60)
  (h2 : hours_per_day = 3)
  (h3 : total_payment = 2520)
  (h4 : days_per_week = 7) :
  total_payment / (hourly_rate * hours_per_day * days_per_week) = 2 := 
by
  -- sorry to skip the proof
  sorry

end magician_weeks_worked_l132_132237


namespace sum_of_variables_l132_132504

theorem sum_of_variables (a b c d : ℤ)
  (h1 : a - b + 2 * c = 7)
  (h2 : b - c + d = 8)
  (h3 : c - d + a = 5)
  (h4 : d - a + b = 4) : a + b + c + d = 20 :=
by
  sorry

end sum_of_variables_l132_132504


namespace dogs_daily_food_total_l132_132264

theorem dogs_daily_food_total :
  let first_dog_food := 0.125
  let second_dog_food := 0.25
  let third_dog_food := 0.375
  let fourth_dog_food := 0.5
  first_dog_food + second_dog_food + third_dog_food + fourth_dog_food = 1.25 :=
by
  sorry

end dogs_daily_food_total_l132_132264


namespace complement_union_eq_l132_132538

-- Defining the universal set and subsets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}

-- Goal: Prove the complement of M ∪ N with respect to U is {5}
theorem complement_union_eq {U M N : Set ℕ} (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2}) (hN : N = {3, 4}) : 
  U \ (M ∪ N) = {5} :=
by
  sorry

end complement_union_eq_l132_132538


namespace fraction_checked_by_worker_y_l132_132190

-- Definitions of conditions given in the problem
variable (P Px Py : ℝ)
variable (h1 : Px + Py = P)
variable (h2 : 0.005 * Px = defective_x)
variable (h3 : 0.008 * Py = defective_y)
variable (defective_x defective_y : ℝ)
variable (total_defective : ℝ)
variable (h4 : defective_x + defective_y = total_defective)
variable (h5 : total_defective = 0.0065 * P)

-- The fraction of products checked by worker y
theorem fraction_checked_by_worker_y (h : Px + Py = P) (h2 : 0.005 * Px = 0.0065 * P) (h3 : 0.008 * Py = 0.0065 * P) :
  Py / P = 1 / 2 := 
  sorry

end fraction_checked_by_worker_y_l132_132190


namespace original_savings_calculation_l132_132705

theorem original_savings_calculation (S : ℝ) (F : ℝ) (T : ℝ) 
  (h1 : 0.8 * F = (3 / 4) * S)
  (h2 : 1.1 * T = 150)
  (h3 : (1 / 4) * S = T) :
  S = 545.44 :=
by
  sorry

end original_savings_calculation_l132_132705


namespace sum_of_first_5_terms_is_55_l132_132898

variable (a : ℕ → ℝ) -- the arithmetic sequence
variable (d : ℝ) -- the common difference
variable (a_2 : a 2 = 7)
variable (a_4 : a 4 = 15)
noncomputable def sum_of_first_5_terms : ℝ := (5 * (a 2 + a 4)) / 2

theorem sum_of_first_5_terms_is_55 :
  sum_of_first_5_terms a = 55 :=
by
  sorry

end sum_of_first_5_terms_is_55_l132_132898


namespace prod_of_three_consec_ints_l132_132698

theorem prod_of_three_consec_ints (a : ℤ) (h : a + (a + 1) + (a + 2) = 27) :
  a * (a + 1) * (a + 2) = 720 :=
by
  sorry

end prod_of_three_consec_ints_l132_132698


namespace altitude_of_dolphin_l132_132617

theorem altitude_of_dolphin (h_submarine : altitude_submarine = -50) (h_dolphin : distance_above_submarine = 10) : altitude_dolphin = -40 :=
by
  -- Altitude of the dolphin is the altitude of the submarine plus the distance above it
  have h_dolphin_altitude : altitude_dolphin = altitude_submarine + distance_above_submarine := sorry
  -- Substitute the values
  rw [h_submarine, h_dolphin] at h_dolphin_altitude
  -- Simplify the expression
  exact h_dolphin_altitude

end altitude_of_dolphin_l132_132617


namespace num_men_in_second_group_l132_132956

def total_work_hours_week (men: ℕ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℕ :=
  men * hours_per_day * days_per_week

def earnings_per_man_hour (total_earnings: ℕ) (total_work_hours: ℕ) : ℚ :=
  total_earnings / total_work_hours

def required_man_hours (total_earnings: ℕ) (earnings_per_hour: ℚ) : ℚ :=
  total_earnings / earnings_per_hour

def number_of_men (total_man_hours: ℚ) (hours_per_day: ℕ) (days_per_week: ℕ) : ℚ :=
  total_man_hours / (hours_per_day * days_per_week)

theorem num_men_in_second_group :
  let hours_per_day_1 := 10
  let hours_per_day_2 := 6
  let days_per_week := 7
  let men_1 := 4
  let earnings_1 := 1000
  let earnings_2 := 1350
  let work_hours_1 := total_work_hours_week men_1 hours_per_day_1 days_per_week
  let rate_1 := earnings_per_man_hour earnings_1 work_hours_1
  let work_hours_2 := required_man_hours earnings_2 rate_1
  number_of_men work_hours_2 hours_per_day_2 days_per_week = 9 := by
  sorry

end num_men_in_second_group_l132_132956


namespace sequence_general_term_l132_132404

theorem sequence_general_term {a : ℕ → ℚ} 
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n ≥ 2, a n = 3 * a (n - 1) / (a (n - 1) + 3)) : 
  ∀ n, a n = 3 / (n + 2) :=
by
  sorry

end sequence_general_term_l132_132404


namespace find_socks_cost_l132_132505

variable (S : ℝ)
variable (socks_cost : ℝ := 9.5)
variable (shoe_cost : ℝ := 92)
variable (jack_has : ℝ := 40)
variable (needs_more : ℝ := 71)
variable (total_funds : ℝ := jack_has + needs_more)

theorem find_socks_cost (h : 2 * S + shoe_cost = total_funds) : S = socks_cost :=
by 
  sorry

end find_socks_cost_l132_132505


namespace smallest_natural_number_condition_l132_132650

theorem smallest_natural_number_condition (N : ℕ) : 
  (∀ k : ℕ, (10^6 - 1) * k = (10^54 - 1) / 9 → k < N) →
  N = 111112 :=
by
  sorry

end smallest_natural_number_condition_l132_132650


namespace foci_of_ellipse_l132_132853

def ellipse_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ (y = 12 ∨ y = -12))

theorem foci_of_ellipse :
  ∀ (x y : ℝ), (x^2)/25 + (y^2)/169 = 1 → ellipse_focus x y :=
by
  intros x y h
  sorry

end foci_of_ellipse_l132_132853


namespace solve_for_x_l132_132329

theorem solve_for_x {x : ℤ} (h : 3 * x + 7 = -2) : x = -3 := 
by
  sorry

end solve_for_x_l132_132329


namespace minimum_value_l132_132365

open Real

theorem minimum_value (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^2 / (y - 2) + y^2 / (x - 2)) ≥ 12 :=
sorry

end minimum_value_l132_132365


namespace older_brother_catches_younger_brother_l132_132065

theorem older_brother_catches_younger_brother
  (y_time_reach_school o_time_reach_school : ℕ) 
  (delay : ℕ) 
  (catchup_time : ℕ) 
  (h1 : y_time_reach_school = 25) 
  (h2 : o_time_reach_school = 15) 
  (h3 : delay = 8) 
  (h4 : catchup_time = 17):
  catchup_time = delay + ((8 * y_time_reach_school) / (o_time_reach_school - y_time_reach_school) * (y_time_reach_school / 25)) :=
by
  sorry

end older_brother_catches_younger_brother_l132_132065


namespace proportion_solution_l132_132142

theorem proportion_solution (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) : 
  x = 17.5 * c / (4 * b) := 
sorry

end proportion_solution_l132_132142


namespace teachers_photos_l132_132773

theorem teachers_photos (n : ℕ) (ht : n = 5) : 6 * 7 = 42 :=
by
  sorry

end teachers_photos_l132_132773


namespace seating_arrangement_l132_132874

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 6 * y = 57) : x = 1 :=
sorry

end seating_arrangement_l132_132874


namespace no_seven_sum_possible_l132_132847

theorem no_seven_sum_possible :
  let outcomes := [-1, -3, -5, 2, 4, 6]
  ∀ (a b : Int), a ∈ outcomes → b ∈ outcomes → a + b ≠ 7 :=
by
  sorry

end no_seven_sum_possible_l132_132847


namespace center_of_circle_eq_minus_two_four_l132_132790

theorem center_of_circle_eq_minus_two_four : 
  ∀ (x y : ℝ), x^2 + 4 * x + y^2 - 8 * y + 16 = 0 → (x, y) = (-2, 4) :=
by {
  sorry
}

end center_of_circle_eq_minus_two_four_l132_132790


namespace total_hours_uploaded_l132_132960

def hours_June_1_to_10 : ℝ := 5 * 2 * 10
def hours_June_11_to_20 : ℝ := 10 * 1 * 10
def hours_June_21_to_25 : ℝ := 7 * 3 * 5
def hours_June_26_to_30 : ℝ := 15 * 0.5 * 5

def total_video_hours : ℝ :=
  hours_June_1_to_10 + hours_June_11_to_20 + hours_June_21_to_25 + hours_June_26_to_30

theorem total_hours_uploaded :
  total_video_hours = 342.5 :=
by
  sorry

end total_hours_uploaded_l132_132960


namespace greatest_divisor_of_arithmetic_sum_l132_132887

theorem greatest_divisor_of_arithmetic_sum (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ k : ℕ, k = 6 ∧ ∀ a d : ℕ, 12 * a + 66 * d % k = 0 :=
by sorry

end greatest_divisor_of_arithmetic_sum_l132_132887


namespace find_excluded_digit_l132_132350

theorem find_excluded_digit (a b : ℕ) (d : ℕ) (h : a * b = 1024) (ha : a % 10 ≠ d) (hb : b % 10 ≠ d) : 
  ∃ r : ℕ, d = r ∧ r < 10 :=
by 
  sorry

end find_excluded_digit_l132_132350


namespace product_consecutive_natural_not_equal_even_l132_132478

theorem product_consecutive_natural_not_equal_even (n m : ℕ) (h : m % 2 = 0 ∧ m > 0) : n * (n + 1) ≠ m * (m + 2) :=
sorry

end product_consecutive_natural_not_equal_even_l132_132478


namespace profit_maximization_problem_l132_132937

-- Step 1: Define the data points and linear function
def data_points : List (ℝ × ℝ) := [(65, 70), (70, 60), (75, 50), (80, 40)]

-- Step 2: Define the linear function between y and x
def linear_function (k b x : ℝ) : ℝ := k * x + b

-- Step 3: Define cost and profit function
def cost_per_kg : ℝ := 60
def profit_function (y x : ℝ) : ℝ := y * (x - cost_per_kg)

-- Step 4: The main problem statement
theorem profit_maximization_problem :
  ∃ (k b : ℝ), 
  (∀ (x₁ x₂ : ℝ), (x₁, y₁) ∈ data_points ∧ (x₂, y₂) ∈ data_points → linear_function k b x₁ = y₁ ∧ linear_function k b x₂ = y₂) ∧
  ∃ (x : ℝ), profit_function (linear_function k b x) x = 600 ∧
  ∀ x : ℝ, -2 * x^2 + 320 * x - 12000 ≤ -2 * 80^2 + 320 * 80 - 12000
  :=
sorry

end profit_maximization_problem_l132_132937


namespace necessary_but_not_sufficient_condition_l132_132192

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  {x | 1 / x ≤ 1} ⊆ {x | Real.log x ≥ 0} ∧ 
  ¬ ({x | Real.log x ≥ 0} ⊆ {x | 1 / x ≤ 1}) :=
by
  sorry

end necessary_but_not_sufficient_condition_l132_132192


namespace tank_cost_correct_l132_132086

noncomputable def tankPlasteringCost (l w d cost_per_m2 : ℝ) : ℝ :=
  let long_walls_area := 2 * (l * d)
  let short_walls_area := 2 * (w * d)
  let bottom_area := l * w
  let total_area := long_walls_area + short_walls_area + bottom_area
  total_area * cost_per_m2

theorem tank_cost_correct :
  tankPlasteringCost 25 12 6 0.75 = 558 := by
  sorry

end tank_cost_correct_l132_132086


namespace greatest_possible_average_speed_l132_132866

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem greatest_possible_average_speed :
  ∀ (o₁ o₂ : ℕ) (v_max t : ℝ), 
  is_palindrome o₁ → 
  is_palindrome o₂ → 
  o₁ = 12321 → 
  t = 2 ∧ v_max = 65 → 
  (∃ d, d = o₂ - o₁ ∧ d / t <= v_max) → 
  d / t = v_max :=
sorry

end greatest_possible_average_speed_l132_132866


namespace domain_of_sqrt_quadratic_l132_132179

open Set

def domain_of_f : Set ℝ := {x : ℝ | 2*x - x^2 ≥ 0}

theorem domain_of_sqrt_quadratic :
  domain_of_f = Icc 0 2 :=
by
  sorry

end domain_of_sqrt_quadratic_l132_132179


namespace find_number_l132_132771

theorem find_number 
    (x : ℝ)
    (h1 : 3 < x) 
    (h2 : x < 8) 
    (h3 : 6 < x) 
    (h4 : x < 10) : 
    x = 7 :=
sorry

end find_number_l132_132771


namespace find_min_n_l132_132744

theorem find_min_n (k : ℕ) : ∃ n, 
  (∀ (m : ℕ), (k = 2 * m → n = 100 * (m + 1)) ∨ (k = 2 * m + 1 → n = 100 * (m + 1) + 1)) ∧
  (∀ n', (∀ (m : ℕ), (k = 2 * m → n' ≥ 100 * (m + 1)) ∨ (k = 2 * m + 1 → n' ≥ 100 * (m + 1) + 1)) → n' ≥ n) :=
by {
  sorry
}

end find_min_n_l132_132744


namespace rearrange_cards_l132_132587

theorem rearrange_cards :
  (∀ (arrangement : List ℕ), arrangement = [3, 1, 2, 4, 5, 6] ∨ arrangement = [1, 2, 4, 5, 6, 3] →
  (∀ card, card ∈ arrangement → List.erase arrangement card = [1, 2, 4, 5, 6] ∨
                                        List.erase arrangement card = [3, 1, 2, 4, 5]) →
  List.length arrangement = 6) →
  (∃ n, n = 10) :=
by
  sorry

end rearrange_cards_l132_132587


namespace min_colors_needed_for_boxes_l132_132949

noncomputable def min_colors_needed : Nat := 23

theorem min_colors_needed_for_boxes :
  ∀ (boxes : Fin 8 → Fin 6 → Nat), 
  (∀ i, ∀ j : Fin 6, boxes i j < min_colors_needed) → 
  (∀ i, (Function.Injective (boxes i))) → 
  (∀ c1 c2, c1 ≠ c2 → (∃! b, ∃ p1 p2, (p1 ≠ p2 ∧ boxes b p1 = c1 ∧ boxes b p2 = c2))) → 
  min_colors_needed = 23 := 
by sorry

end min_colors_needed_for_boxes_l132_132949


namespace total_transaction_loss_l132_132815

-- Define the cost and selling prices given the conditions
def cost_price_house (h : ℝ) := (7 / 10) * h = 15000
def cost_price_store (s : ℝ) := (5 / 4) * s = 15000

-- Define the loss calculation for the transaction
def transaction_loss : Prop :=
  ∃ (h s : ℝ),
    (7 / 10) * h = 15000 ∧
    (5 / 4) * s = 15000 ∧
    h + s - 2 * 15000 = 3428.57

-- The theorem stating the transaction resulted in a loss of $3428.57
theorem total_transaction_loss : transaction_loss :=
by
  sorry

end total_transaction_loss_l132_132815


namespace percentage_forgot_homework_l132_132820

def total_students_group_A : ℕ := 30
def total_students_group_B : ℕ := 50
def forget_percentage_A : ℝ := 0.20
def forget_percentage_B : ℝ := 0.12

theorem percentage_forgot_homework :
  let num_students_forgot_A := forget_percentage_A * total_students_group_A
  let num_students_forgot_B := forget_percentage_B * total_students_group_B
  let total_students_forgot := num_students_forgot_A + num_students_forgot_B
  let total_students := total_students_group_A + total_students_group_B
  let percentage_forgot := (total_students_forgot / total_students) * 100
  percentage_forgot = 15 := sorry

end percentage_forgot_homework_l132_132820


namespace inequality_solution_l132_132740

theorem inequality_solution :
  { x : ℝ | (x^3 - 4 * x) / (x^2 - 1) > 0 } = { x : ℝ | x < -2 ∨ (0 < x ∧ x < 1) ∨ 2 < x } :=
by
  sorry

end inequality_solution_l132_132740


namespace jeff_pencils_initial_l132_132133

def jeff_initial_pencils (J : ℝ) := J
def jeff_remaining_pencils (J : ℝ) := 0.70 * J
def vicki_initial_pencils (J : ℝ) := 2 * J
def vicki_remaining_pencils (J : ℝ) := 0.25 * vicki_initial_pencils J
def remaining_pencils (J : ℝ) := jeff_remaining_pencils J + vicki_remaining_pencils J

theorem jeff_pencils_initial (J : ℝ) (h : remaining_pencils J = 360) : J = 300 :=
by
  sorry

end jeff_pencils_initial_l132_132133


namespace positive_integer_satisfies_condition_l132_132431

theorem positive_integer_satisfies_condition : 
  ∃ n : ℕ, (12 * n = n^2 + 36) ∧ n = 6 :=
by
  sorry

end positive_integer_satisfies_condition_l132_132431


namespace red_pairs_count_l132_132843

theorem red_pairs_count (blue_shirts red_shirts total_pairs blue_blue_pairs : ℕ)
  (h1 : blue_shirts = 63) 
  (h2 : red_shirts = 81) 
  (h3 : total_pairs = 72) 
  (h4 : blue_blue_pairs = 21)
  : (red_shirts - (blue_shirts - blue_blue_pairs * 2)) / 2 = 30 :=
by
  sorry

end red_pairs_count_l132_132843


namespace missing_angle_correct_l132_132466

theorem missing_angle_correct (n : ℕ) (h1 : n ≥ 3) (angles_sum : ℕ) (h2 : angles_sum = 2017) 
    (sum_interior_angles : ℕ) (h3 : sum_interior_angles = 180 * (n - 2)) :
    (sum_interior_angles - angles_sum) = 143 :=
by
  sorry

end missing_angle_correct_l132_132466


namespace teams_in_league_l132_132330

def number_of_teams (n : ℕ) := n * (n - 1) / 2

theorem teams_in_league : ∃ n : ℕ, number_of_teams n = 36 ∧ n = 9 := by
  sorry

end teams_in_league_l132_132330


namespace fraction_blue_after_doubling_l132_132702

theorem fraction_blue_after_doubling (x : ℕ) (h1 : ∃ x, (2 : ℚ) / 3 * x + (1 : ℚ) / 3 * x = x) :
  ((2 * (2 / 3 * x)) / ((2 / 3 * x) + (1 / 3 * x))) = (4 / 5) := by
  sorry

end fraction_blue_after_doubling_l132_132702


namespace range_of_a_l132_132968

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  ∃ x y : ℝ, (x > 0 ∧ y > 0 ∨ x < 0 ∧ y < 0) ∧ y + (a - 1) * x + 2 * a - 1 = 0

def valid_a (a : ℝ) : Prop :=
  (p a ∨ q a) ∧ ¬(p a ∧ q a)

theorem range_of_a (a : ℝ) :
  valid_a a →
  (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
sorry

end range_of_a_l132_132968


namespace find_u_plus_v_l132_132895

-- Conditions: 3u - 4v = 17 and 5u - 2v = 1.
-- Question: Find the value of u + v.

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 4 * v = 17) (h2 : 5 * u - 2 * v = 1) : u + v = -8 :=
by
  sorry

end find_u_plus_v_l132_132895


namespace range_of_m_l132_132900

theorem range_of_m (m : ℝ) :
  (3 * 1 - 2 + m) * (3 * 1 - 1 + m) < 0 →
  -2 < m ∧ m < -1 :=
by
  intro h
  sorry

end range_of_m_l132_132900


namespace solve_equation_l132_132308

theorem solve_equation (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2/3) :=
by
  sorry

end solve_equation_l132_132308


namespace sequence_problem_l132_132799

theorem sequence_problem (a : ℕ → ℝ) (pos_terms : ∀ n, a n > 0)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, (a n + 1) * a (n + 2) = 1)
  (h2 : a 2 = a 6) :
  a 11 + a 12 = (11 / 18) + ((Real.sqrt 5 - 1) / 2) := by
  sorry

end sequence_problem_l132_132799


namespace hyperbola_range_k_l132_132105

noncomputable def hyperbola_equation (x y k : ℝ) : Prop :=
    (x^2) / (|k|-2) + (y^2) / (5-k) = 1

theorem hyperbola_range_k (k : ℝ) :
    (∃ x y, hyperbola_equation x y k) → (k > 5 ∨ (-2 < k ∧ k < 2)) :=
by 
    sorry

end hyperbola_range_k_l132_132105


namespace ounces_per_bowl_l132_132595

theorem ounces_per_bowl (oz_per_gallon : ℕ) (gallons : ℕ) (bowls_per_minute : ℕ) (minutes : ℕ) (total_ounces : ℕ) (total_bowls : ℕ) (oz_per_bowl : ℕ) : 
  oz_per_gallon = 128 → 
  gallons = 6 →
  bowls_per_minute = 5 →
  minutes = 15 →
  total_ounces = oz_per_gallon * gallons →
  total_bowls = bowls_per_minute * minutes →
  oz_per_bowl = total_ounces / total_bowls →
  round (oz_per_bowl : ℚ) = 10 :=
by
  sorry

end ounces_per_bowl_l132_132595


namespace intersect_sets_example_l132_132120

open Set

theorem intersect_sets_example : 
  let A := {x : ℝ | -1 < x ∧ x ≤ 3}
  let B := {x : ℝ | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4}
  A ∩ B = {x : ℝ | x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3} :=
by
  sorry

end intersect_sets_example_l132_132120


namespace chocolate_cost_is_75_l132_132306

def candy_bar_cost : ℕ := 25
def juice_pack_cost : ℕ := 50
def num_quarters : ℕ := 11
def total_cost_in_cents : ℕ := num_quarters * candy_bar_cost
def num_candy_bars : ℕ := 3
def num_pieces_of_chocolate : ℕ := 2

def chocolate_cost_in_cents (x : ℕ) : Prop :=
  (num_candy_bars * candy_bar_cost) + (num_pieces_of_chocolate * x) + juice_pack_cost = total_cost_in_cents

theorem chocolate_cost_is_75 : chocolate_cost_in_cents 75 :=
  sorry

end chocolate_cost_is_75_l132_132306


namespace P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l132_132633

def is_subtract_set (T : Set ℕ) (i : ℕ) := T ⊆ Set.univ ∧ T ≠ {1} ∧ (∀ {x y : ℕ}, x ∈ Set.univ → y ∈ Set.univ → x + y ∈ T → x * y - i ∈ T)

theorem P_is_subtract_0_set : is_subtract_set {1, 2} 0 := sorry

theorem P_is_not_subtract_1_set : ¬ is_subtract_set {1, 2} 1 := sorry

theorem no_subtract_2_set_exists : ¬∃ T : Set ℕ, is_subtract_set T 2 := sorry

theorem all_subtract_1_sets : ∀ T : Set ℕ, is_subtract_set T 1 ↔ T = {1, 3} ∨ T = {1, 3, 5} := sorry

end P_is_subtract_0_set_P_is_not_subtract_1_set_no_subtract_2_set_exists_all_subtract_1_sets_l132_132633


namespace correct_option_D_l132_132516

theorem correct_option_D (x y : ℝ) : (x - y) ^ 2 = (y - x) ^ 2 := by
  sorry

end correct_option_D_l132_132516


namespace cost_of_watch_l132_132734

variable (saved amount_needed total_cost : ℕ)

-- Conditions
def connie_saved : Prop := saved = 39
def connie_needs : Prop := amount_needed = 16

-- Theorem to prove
theorem cost_of_watch : connie_saved saved → connie_needs amount_needed → total_cost = 55 :=
by
  sorry

end cost_of_watch_l132_132734


namespace training_cost_per_month_correct_l132_132130

-- Define the conditions
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_duration : ℕ := 3
def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2 : ℕ := (45000 / 100) -- 1% of salary2 which is 450
def net_gain_diff : ℕ := 850

-- Define the monthly training cost for the first applicant
def monthly_training_cost : ℕ := 1786667 / 100

-- Prove that the monthly training cost for the first applicant is correct
theorem training_cost_per_month_correct :
  (revenue1 - (salary1 + 3 * monthly_training_cost) = revenue2 - (salary2 + bonus2) + net_gain_diff) :=
by
  sorry

end training_cost_per_month_correct_l132_132130


namespace gamma_max_two_day_success_ratio_l132_132176

theorem gamma_max_two_day_success_ratio :
  ∃ (e g f h : ℕ), 0 < e ∧ 0 < g ∧
  e + g = 335 ∧ 
  e < f ∧ g < h ∧ 
  f + h = 600 ∧ 
  (e : ℚ) / f < (180 : ℚ) / 360 ∧ 
  (g : ℚ) / h < (150 : ℚ) / 240 ∧ 
  (e + g) / 600 = 67 / 120 :=
by
  sorry

end gamma_max_two_day_success_ratio_l132_132176


namespace four_consecutive_product_divisible_by_12_l132_132290

theorem four_consecutive_product_divisible_by_12 (n : ℕ) (hn : 0 < n) :
  ∃ k, (n * (n + 1) * (n + 2) * (n + 3)) = 12 * k := 
sorry

end four_consecutive_product_divisible_by_12_l132_132290


namespace Ian_hourly_wage_l132_132560

variable (hours_worked : ℕ)
variable (money_left : ℕ)
variable (hourly_wage : ℕ)

theorem Ian_hourly_wage :
  hours_worked = 8 ∧
  money_left = 72 ∧
  hourly_wage = 18 →
  2 * money_left = hours_worked * hourly_wage :=
by
  intros
  sorry

end Ian_hourly_wage_l132_132560


namespace parabola_transformation_correct_l132_132256

-- Definitions and conditions
def original_parabola (x : ℝ) : ℝ := 2 * x^2

def transformed_parabola (x : ℝ) : ℝ := 2 * (x + 3)^2 - 4

-- Theorem to prove that the above definition is correct
theorem parabola_transformation_correct : 
  ∀ x : ℝ, transformed_parabola x = 2 * (x + 3)^2 - 4 :=
by
  intros x
  rfl -- This uses the definition of 'transformed_parabola' directly

end parabola_transformation_correct_l132_132256


namespace joey_return_speed_l132_132798

theorem joey_return_speed
    (h1: 1 = (2 : ℝ) / u)
    (h2: (4 : ℝ) / (1 + t) = 3)
    (h3: u = 2)
    (h4: t = 1 / 3) :
    (2 : ℝ) / t = 6 :=
by
  sorry

end joey_return_speed_l132_132798


namespace vectors_parallel_x_value_l132_132839

theorem vectors_parallel_x_value :
  ∀ (x : ℝ), (∀ a b : ℝ × ℝ, a = (2, 1) → b = (4, x+1) → (a.1 / b.1 = a.2 / b.2)) → x = 1 :=
by
  intros x h
  sorry

end vectors_parallel_x_value_l132_132839


namespace box_dimensions_l132_132456

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) : 
  a = 5 ∧ b = 8 ∧ c = 12 := 
by
  sorry

end box_dimensions_l132_132456


namespace union_complement_inter_l132_132535

noncomputable def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | -1 ≤ x ∧ x < 5 }

def C_U_M : Set ℝ := U \ M
def M_inter_N : Set ℝ := { x | x ≥ 2 ∧ x < 5 }

theorem union_complement_inter (C_U_M M_inter_N : Set ℝ) :
  C_U_M ∪ M_inter_N = { x | x < 5 } :=
by
  sorry

end union_complement_inter_l132_132535


namespace victoria_more_scoops_l132_132275

theorem victoria_more_scoops (Oli_scoops : ℕ) (Victoria_scoops : ℕ) 
  (hOli : Oli_scoops = 4) (hVictoria : Victoria_scoops = 2 * Oli_scoops) : 
  (Victoria_scoops - Oli_scoops) = 4 :=
by
  sorry

end victoria_more_scoops_l132_132275


namespace second_part_of_sum_l132_132837

-- Defining the problem conditions
variables (x : ℚ)
def sum_parts := (2 * x) + (1/2 * x) + (1/4 * x)

theorem second_part_of_sum :
  sum_parts x = 104 →
  (1/2 * x) = 208 / 11 :=
by
  intro h
  sorry

end second_part_of_sum_l132_132837


namespace fraction_sum_l132_132958

theorem fraction_sum : (3 / 8) + (9 / 12) = 9 / 8 :=
by
  sorry

end fraction_sum_l132_132958


namespace simplification_correct_l132_132304

noncomputable def given_equation (x : ℚ) : Prop := 
  x / (2 * x - 1) - 3 = 2 / (1 - 2 * x)

theorem simplification_correct (x : ℚ) (h : given_equation x) : 
  x - 3 * (2 * x - 1) = -2 :=
sorry

end simplification_correct_l132_132304


namespace unique_scalar_matrix_l132_132755

theorem unique_scalar_matrix (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, Matrix.mulVec N v = 5 • v) → 
  N = !![5, 0, 0; 0, 5, 0; 0, 0, 5] :=
by
  intro hv
  sorry -- Proof omitted as per instructions

end unique_scalar_matrix_l132_132755


namespace amoeba_population_after_5_days_l132_132472

theorem amoeba_population_after_5_days 
  (initial : ℕ)
  (split_factor : ℕ)
  (days : ℕ)
  (h_initial : initial = 2)
  (h_split : split_factor = 3)
  (h_days : days = 5) :
  (initial * split_factor ^ days) = 486 :=
by sorry

end amoeba_population_after_5_days_l132_132472


namespace quadratic_inequality_solution_l132_132548

theorem quadratic_inequality_solution (a b c : ℝ) (h : a < 0) 
  (h_sol : ∀ x, ax^2 + bx + c > 0 ↔ x > -2 ∧ x < 1) :
  ∀ x, ax^2 + (a + b) * x + c - a < 0 ↔ x < -3 ∨ x > 1 := 
sorry

end quadratic_inequality_solution_l132_132548


namespace odd_function_value_l132_132167

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 + x else -(x^2 + x)

theorem odd_function_value : f (-3) = -12 :=
by
  -- proof goes here
  sorry

end odd_function_value_l132_132167


namespace fill_tank_time_l132_132170

variable (A_rate := 1/3)
variable (B_rate := 1/4)
variable (C_rate := -1/4)

def combined_rate := A_rate + B_rate + C_rate

theorem fill_tank_time (hA : A_rate = 1/3) (hB : B_rate = 1/4) (hC : C_rate = -1/4) : (1 / combined_rate) = 3 := by
  sorry

end fill_tank_time_l132_132170


namespace basketball_team_lineup_l132_132056

-- Define the problem conditions
def total_players : ℕ := 12
def twins : ℕ := 2
def lineup_size : ℕ := 5
def remaining_players : ℕ := total_players - twins
def positions_to_fill : ℕ := lineup_size - twins

-- Define the combination function as provided in the standard libraries
def combination (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem translating to the proof problem
theorem basketball_team_lineup : combination remaining_players positions_to_fill = 120 := 
sorry

end basketball_team_lineup_l132_132056


namespace commutative_otimes_l132_132648

def otimes (a b : ℝ) : ℝ := a * b + a + b

theorem commutative_otimes (a b : ℝ) : otimes a b = otimes b a :=
by
  /- The proof will go here, but we omit it and use sorry. -/
  sorry

end commutative_otimes_l132_132648


namespace car_average_speed_l132_132145

-- Definitions based on conditions
def distance_first_hour : ℤ := 100
def distance_second_hour : ℤ := 60
def time_first_hour : ℤ := 1
def time_second_hour : ℤ := 1

-- Total distance and time calculations
def total_distance : ℤ := distance_first_hour + distance_second_hour
def total_time : ℤ := time_first_hour + time_second_hour

-- The average speed of the car
def average_speed : ℤ := total_distance / total_time

-- Proof statement
theorem car_average_speed : average_speed = 80 := by
  sorry

end car_average_speed_l132_132145


namespace quadratic_equation_must_be_minus_2_l132_132316

-- Define the main problem statement
theorem quadratic_equation_must_be_minus_2 (m : ℝ) :
  (∀ x : ℝ, (m - 2) * x ^ |m| - 3 * x - 7 = 0) →
  (∀ (h : |m| = 2), m - 2 ≠ 0) →
  m = -2 :=
sorry

end quadratic_equation_must_be_minus_2_l132_132316


namespace silver_excess_in_third_chest_l132_132720

theorem silver_excess_in_third_chest :
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ),
    x1 + x2 + x3 = 40 →
    y1 + y2 + y3 = 40 →
    x1 = y1 + 7 →
    y2 = x2 - 15 →
    y3 = x3 + 22 :=
by
  intros x1 y1 x2 y2 x3 y3 h1 h2 h3 h4
  sorry

end silver_excess_in_third_chest_l132_132720


namespace average_weight_of_remaining_boys_l132_132833

theorem average_weight_of_remaining_boys :
  ∀ (total_boys remaining_boys_num : ℕ)
    (avg_weight_22 remaining_boys_avg_weight total_class_avg_weight : ℚ),
    total_boys = 30 →
    remaining_boys_num = total_boys - 22 →
    avg_weight_22 = 50.25 →
    total_class_avg_weight = 48.89 →
    (remaining_boys_num : ℚ) * remaining_boys_avg_weight =
    total_boys * total_class_avg_weight - 22 * avg_weight_22 →
    remaining_boys_avg_weight = 45.15 :=
by
  intros total_boys remaining_boys_num avg_weight_22 remaining_boys_avg_weight total_class_avg_weight
         h_total_boys h_remaining_boys_num h_avg_weight_22 h_total_class_avg_weight h_equation
  sorry

end average_weight_of_remaining_boys_l132_132833


namespace quadratic_form_rewrite_l132_132569

theorem quadratic_form_rewrite (x : ℝ) : 2 * x ^ 2 + 7 = 4 * x → 2 * x ^ 2 - 4 * x + 7 = 0 :=
by
    intro h
    linarith

end quadratic_form_rewrite_l132_132569


namespace gcd_five_pentagonal_and_n_plus_one_l132_132913

-- Definition of the nth pentagonal number
def pentagonal_number (n : ℕ) : ℕ :=
  (n * (3 * n - 1)) / 2

-- Proof statement
theorem gcd_five_pentagonal_and_n_plus_one (n : ℕ) (h : 0 < n) : 
  Nat.gcd (5 * pentagonal_number n) (n + 1) = 1 :=
sorry

end gcd_five_pentagonal_and_n_plus_one_l132_132913


namespace vertex_of_parabola_l132_132589

-- Definition of the parabola
def parabola (x : ℝ) : ℝ := -2 * (x - 3)^2 - 2

-- The theorem stating the vertex of the parabola
theorem vertex_of_parabola : ∃ h k : ℝ, (h, k) = (2, -5) :=
by
  sorry

end vertex_of_parabola_l132_132589


namespace cost_of_swim_trunks_is_14_l132_132194

noncomputable def cost_of_swim_trunks : Real :=
  let flat_rate_shipping := 5.00
  let shipping_rate := 0.20
  let price_shirt := 12.00
  let price_socks := 5.00
  let price_shorts := 15.00
  let cost_known_items := 3 * price_shirt + price_socks + 2 * price_shorts
  let total_bill := 102.00
  let x := (total_bill - 0.20 * cost_known_items - cost_known_items) / 1.20
  x

theorem cost_of_swim_trunks_is_14 : cost_of_swim_trunks = 14 := by
  -- sorry is used to skip the proof
  sorry

end cost_of_swim_trunks_is_14_l132_132194


namespace find_4_digit_number_l132_132945

theorem find_4_digit_number (a b c d : ℕ) 
(h1 : 1000 * a + 100 * b + 10 * c + d = 1000 * d + 100 * c + 10 * b + a - 7182)
(h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
(h6 : 0 ≤ c) (h7 : c ≤ 9) (h8 : 1 ≤ d) (h9 : d ≤ 9) : 
1000 * a + 100 * b + 10 * c + d = 1909 :=
sorry

end find_4_digit_number_l132_132945


namespace polynomial_value_l132_132451

theorem polynomial_value :
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  5 * a + 3 * b + 2 * c + d = 25 :=
by
  let a := -4
  let b := 23
  let c := -17
  let d := 10
  sorry

end polynomial_value_l132_132451


namespace power_of_product_l132_132691

variable (x y : ℝ)

theorem power_of_product (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 :=
  sorry

end power_of_product_l132_132691


namespace expected_number_of_shots_l132_132100

def probability_hit : ℝ := 0.8
def probability_miss := 1 - probability_hit
def max_shots : ℕ := 3

theorem expected_number_of_shots : ∃ ξ : ℝ, ξ = 1.24 := by
  sorry

end expected_number_of_shots_l132_132100


namespace factorization_of_expression_l132_132227

theorem factorization_of_expression (x y : ℝ) : x^2 - x * y = x * (x - y) := 
by
  sorry

end factorization_of_expression_l132_132227


namespace no_solutions_abs_eq_3x_plus_6_l132_132832

theorem no_solutions_abs_eq_3x_plus_6 : ¬ ∃ x : ℝ, |x| = 3 * (|x| + 2) :=
by {
  sorry
}

end no_solutions_abs_eq_3x_plus_6_l132_132832


namespace A_days_to_complete_work_alone_l132_132781

theorem A_days_to_complete_work_alone (x : ℝ) (h1 : 0 < x) (h2 : 0 < 18) (h3 : 1/x + 1/18 = 1/6) : x = 9 :=
by
  sorry

end A_days_to_complete_work_alone_l132_132781


namespace total_seeds_l132_132986

-- Definitions and conditions
def Bom_seeds : ℕ := 300
def Gwi_seeds : ℕ := Bom_seeds + 40
def Yeon_seeds : ℕ := 3 * Gwi_seeds
def Eun_seeds : ℕ := 2 * Gwi_seeds

-- Theorem statement
theorem total_seeds : Bom_seeds + Gwi_seeds + Yeon_seeds + Eun_seeds = 2340 :=
by
  -- Skipping the proof steps with sorry
  sorry

end total_seeds_l132_132986


namespace isosceles_right_triangle_quotient_l132_132543

theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
sorry

end isosceles_right_triangle_quotient_l132_132543


namespace problem1_problem2_l132_132652

-- Define conditions for Problem 1
def problem1_cond (x : ℝ) : Prop :=
  x ≠ 0 ∧ 2 * x ≠ 1

-- Statement for Problem 1
theorem problem1 (x : ℝ) (h : problem1_cond x) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by
  sorry

-- Define conditions for Problem 2
def problem2_cond (x : ℝ) : Prop :=
  x ≠ 2 

-- Statement for Problem 2
theorem problem2 (x : ℝ) (h : problem2_cond x) :
  ((x - 3) / (x - 2) + 1 = 3 / (2 - x)) ↔ x = 1 := by
  sorry

end problem1_problem2_l132_132652


namespace num_valid_colorings_l132_132611

namespace ColoringGrid

-- Definition of the grid and the constraint.
-- It's easier to represent with simply 9 nodes and adjacent constraints, however,
-- we will declare the conditions and result as discussed.

def Grid := Fin 3 × Fin 3
def Colors := Fin 2

-- Define adjacency relationship
def adjacent (a b : Grid) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Condition stating no two adjacent squares can share the same color
def valid_coloring (f : Grid → Colors) : Prop :=
  ∀ a b : Grid, adjacent a b → f a ≠ f b

-- The main theorem stating the number of valid colorings
theorem num_valid_colorings : ∃ (n : ℕ), n = 2 ∧ ∀ (f : Grid → Colors), valid_coloring f → n = 2 :=
by sorry

end ColoringGrid

end num_valid_colorings_l132_132611


namespace sequence_term_101_l132_132315

theorem sequence_term_101 :
  ∃ a : ℕ → ℚ, a 1 = 2 ∧ (∀ n : ℕ, 2 * a (n+1) - 2 * a n = 1) ∧ a 101 = 52 :=
by
  sorry

end sequence_term_101_l132_132315


namespace base_2_representation_of_123_l132_132128

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end base_2_representation_of_123_l132_132128


namespace arccos_one_over_sqrt_two_l132_132022

theorem arccos_one_over_sqrt_two :
  Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_l132_132022


namespace work_done_by_b_l132_132889

theorem work_done_by_b (x : ℝ) (h1 : (1/6) + (1/13) = (1/x)) : x = 78/7 :=
  sorry

end work_done_by_b_l132_132889


namespace complex_coordinates_l132_132813

theorem complex_coordinates : (⟨(-1:ℝ), (-1:ℝ)⟩ : ℂ) = (⟨0,1⟩ : ℂ) * (⟨-2,0⟩ : ℂ) / (⟨1,1⟩ : ℂ) :=
by
  sorry

end complex_coordinates_l132_132813


namespace certain_fraction_is_half_l132_132072

theorem certain_fraction_is_half (n : ℕ) (fraction : ℚ) (h : (37 + 1/2) / fraction = 75) : fraction = 1/2 :=
by
    sorry

end certain_fraction_is_half_l132_132072


namespace cups_of_flour_already_put_in_correct_l132_132636

-- Let F be the number of cups of flour Mary has already put in
def cups_of_flour_already_put_in (F : ℕ) : Prop :=
  let total_flour_needed := 12
  let cups_of_salt := 7
  let additional_flour_needed := cups_of_salt + 3
  F = total_flour_needed - additional_flour_needed

-- Theorem stating that F = 2
theorem cups_of_flour_already_put_in_correct (F : ℕ) : cups_of_flour_already_put_in F → F = 2 :=
by
  intro h
  sorry

end cups_of_flour_already_put_in_correct_l132_132636


namespace equal_areas_of_shapes_l132_132697

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def semicircle_area (r : ℝ) : ℝ :=
  (Real.pi * r^2) / 2

noncomputable def sector_area (theta : ℝ) (r : ℝ) : ℝ :=
  (theta / (2 * Real.pi)) * Real.pi * r^2

noncomputable def shape1_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * semicircle_area (s / 4) - 6 * sector_area (Real.pi / 3) (s / 4)

noncomputable def shape2_area (s : ℝ) : ℝ :=
  hexagon_area s + 6 * sector_area (2 * Real.pi / 3) (s / 4) - 3 * semicircle_area (s / 4)

theorem equal_areas_of_shapes (s : ℝ) : shape1_area s = shape2_area s :=
by {
  sorry
}

end equal_areas_of_shapes_l132_132697


namespace number_of_zeros_f_l132_132552

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^2 - x - 1

-- The theorem statement that proves the function has exactly two zeros
theorem number_of_zeros_f : ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ f r1 = 0 ∧ f r2 = 0 :=
by
  sorry

end number_of_zeros_f_l132_132552


namespace seeds_per_watermelon_l132_132752

theorem seeds_per_watermelon (total_seeds : ℕ) (num_watermelons : ℕ) (h : total_seeds = 400 ∧ num_watermelons = 4) : total_seeds / num_watermelons = 100 :=
by
  sorry

end seeds_per_watermelon_l132_132752


namespace average_of_a_and_b_l132_132725

theorem average_of_a_and_b (a b c M : ℝ)
  (h1 : (a + b) / 2 = M)
  (h2 : (b + c) / 2 = 180)
  (h3 : a - c = 200) : 
  M = 280 :=
sorry

end average_of_a_and_b_l132_132725


namespace johnny_closed_days_l132_132107

theorem johnny_closed_days :
  let dishes_per_day := 40
  let pounds_per_dish := 1.5
  let price_per_pound := 8
  let weekly_expenditure := 1920
  let daily_pounds := dishes_per_day * pounds_per_dish
  let daily_cost := daily_pounds * price_per_pound
  let days_open := weekly_expenditure / daily_cost
  let days_in_week := 7
  let days_closed := days_in_week - days_open
  days_closed = 3 :=
by
  sorry

end johnny_closed_days_l132_132107


namespace problem1_problem2_l132_132384

open Real

-- Proof problem for the first expression
theorem problem1 : 
  (-2^2 * (1 / 4) + 4 / (4/9) + (-1) ^ 2023 = 7) :=
by 
  sorry

-- Proof problem for the second expression
theorem problem2 : 
  (-1 ^ 4 + abs (2 - (-3)^2) + (1/2) / (-3/2) = 17/3) :=
by 
  sorry

end problem1_problem2_l132_132384


namespace longer_diagonal_of_rhombus_l132_132555

theorem longer_diagonal_of_rhombus {a b d1 : ℕ} (h1 : a = b) (h2 : a = 65) (h3 : d1 = 60) : 
  ∃ d2, (d2^2) = (2 * (a^2) - (d1^2)) ∧ d2 = 110 :=
by
  sorry

end longer_diagonal_of_rhombus_l132_132555


namespace probability_of_winning_l132_132912

variable (P_A P_B P_C P_M_given_A P_M_given_B P_M_given_C : ℝ)

theorem probability_of_winning :
  P_A = 0.6 →
  P_B = 0.3 →
  P_C = 0.1 →
  P_M_given_A = 0.1 →
  P_M_given_B = 0.2 →
  P_M_given_C = 0.3 →
  (P_A * P_M_given_A + P_B * P_M_given_B + P_C * P_M_given_C) = 0.15 :=
by sorry

end probability_of_winning_l132_132912


namespace find_insect_stickers_l132_132135

noncomputable def flower_stickers : ℝ := 15
noncomputable def animal_stickers : ℝ := 2 * flower_stickers - 3.5
noncomputable def space_stickers : ℝ := 1.5 * flower_stickers + 5.5
noncomputable def total_stickers : ℝ := 70
noncomputable def insect_stickers : ℝ := total_stickers - (animal_stickers + space_stickers)

theorem find_insect_stickers : insect_stickers = 15.5 := by
  sorry

end find_insect_stickers_l132_132135


namespace solve_for_x_l132_132862

-- Definitions and conditions from a) directly 
def f (x : ℝ) : ℝ := 64 * (2 * x - 1) ^ 3

-- Lean 4 statement to prove the problem
theorem solve_for_x (x : ℝ) : f x = 27 → x = 7 / 8 :=
by
  intro h
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l132_132862


namespace gcd_pow_of_subtraction_l132_132976

noncomputable def m : ℕ := 2^2100 - 1
noncomputable def n : ℕ := 2^1950 - 1

theorem gcd_pow_of_subtraction : Nat.gcd m n = 2^150 - 1 :=
by
  -- To be proven
  sorry

end gcd_pow_of_subtraction_l132_132976


namespace infinite_solutions_b_value_l132_132447

-- Given condition for the equation to hold
def equation_condition (x b : ℤ) : Prop :=
  4 * (3 * x - b) = 3 * (4 * x + 16)

-- The statement we need to prove: b = -12
theorem infinite_solutions_b_value :
  (∀ x : ℤ, equation_condition x b) → b = -12 :=
sorry

end infinite_solutions_b_value_l132_132447


namespace sum_of_primes_less_than_20_is_77_l132_132706

def is_prime (n : ℕ) : Prop := Nat.Prime n

def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 : ℕ :=
  primes_less_than_20.foldl (· + ·) 0

theorem sum_of_primes_less_than_20_is_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_is_77_l132_132706


namespace calc_result_l132_132224

noncomputable def expMul := (-0.25)^11 * (-4)^12

theorem calc_result : expMul = -4 := 
by
  -- Sorry is used here to skip the proof as instructed.
  sorry

end calc_result_l132_132224


namespace anya_more_erasers_l132_132936

theorem anya_more_erasers (anya_erasers andrea_erasers : ℕ)
  (h1 : anya_erasers = 4 * andrea_erasers)
  (h2 : andrea_erasers = 4) :
  anya_erasers - andrea_erasers = 12 := by
  sorry

end anya_more_erasers_l132_132936


namespace roots_are_simplified_sqrt_form_l132_132806

theorem roots_are_simplified_sqrt_form : 
  ∃ m p n : ℕ, gcd m p = 1 ∧ gcd p n = 1 ∧ gcd m n = 1 ∧
    (∀ x : ℝ, (3 * x^2 - 8 * x + 1 = 0) ↔ 
    (x = (m : ℝ) + (Real.sqrt n)/(p : ℝ) ∨ x = (m : ℝ) - (Real.sqrt n)/(p : ℝ))) ∧
    n = 13 :=
by
  sorry

end roots_are_simplified_sqrt_form_l132_132806


namespace frozen_yogurt_price_l132_132351

variable (F G S : ℝ) -- Define the variables F, G, S as real numbers

-- Define the conditions given in the problem
variable (h1 : 5 * F + 2 * G + 5 * S = 55)
variable (h2 : S = 5)
variable (h3 : G = 1 / 2 * F)

-- State the proof goal
theorem frozen_yogurt_price : F = 5 :=
by
  sorry

end frozen_yogurt_price_l132_132351


namespace investment_value_l132_132082

noncomputable def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r)^n

theorem investment_value :
  ∀ (P : ℕ) (r : ℚ) (n : ℕ),
  P = 8000 →
  r = 0.05 →
  n = 3 →
  compound_interest P r n = 9250 := by
    intros P r n hP hr hn
    unfold compound_interest
    -- calculation steps would be here
    sorry

end investment_value_l132_132082


namespace Mason_fathers_age_indeterminate_l132_132295

theorem Mason_fathers_age_indeterminate
  (Mason_age : ℕ) (Sydney_age Mason_father_age D : ℕ)
  (hM : Mason_age = 20)
  (hS_M : Mason_age = Sydney_age / 3)
  (hS_F : Mason_father_age - D = Sydney_age) :
  ¬ ∃ F, Mason_father_age = F :=
by {
  sorry
}

end Mason_fathers_age_indeterminate_l132_132295


namespace a_plus_b_eq_neg7_l132_132153

theorem a_plus_b_eq_neg7 (a b : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * x - 3 > 0) ∨ (x^2 + a * x + b ≤ 0)) ∧
  (∀ x : ℝ, (3 < x ∧ x ≤ 4) → ((x^2 - 2 * x - 3 > 0) ∧ (x^2 + a * x + b ≤ 0))) →
  a + b = -7 :=
by
  sorry

end a_plus_b_eq_neg7_l132_132153


namespace remainder_of_a55_l132_132083

def concatenate_integers (n : ℕ) : ℕ :=
  -- Function to concatenate integers from 1 to n into a single number.
  -- This is a placeholder, actual implementation may vary.
  sorry

theorem remainder_of_a55 (n : ℕ) (hn : n = 55) :
  concatenate_integers n % 55 = 0 := by
  -- Proof is omitted, provided as a guideline.
  sorry

end remainder_of_a55_l132_132083


namespace ratio_of_largest_to_smallest_root_in_geometric_progression_l132_132750

theorem ratio_of_largest_to_smallest_root_in_geometric_progression 
    (a b c d : ℝ) (r s t : ℝ) 
    (h_poly : 81 * r^3 - 243 * r^2 + 216 * r - 64 = 0)
    (h_geo_prog : r > 0 ∧ s > 0 ∧ t > 0 ∧ ∃ (k : ℝ),  k > 0 ∧ s = r * k ∧ t = s * k) :
    ∃ (k : ℝ), k = r^2 ∧ s = r * k ∧ t = s * k := 
sorry

end ratio_of_largest_to_smallest_root_in_geometric_progression_l132_132750


namespace Tammy_second_day_speed_l132_132397

variable (v t : ℝ)

/-- This statement represents Tammy's climbing situation -/
theorem Tammy_second_day_speed:
  (t + (t - 2) = 14) ∧
  (v * t + (v + 0.5) * (t - 2) = 52) →
  (v + 0.5 = 4) :=
by
  sorry

end Tammy_second_day_speed_l132_132397


namespace derek_age_calculation_l132_132274

theorem derek_age_calculation 
  (bob_age : ℕ)
  (evan_age : ℕ)
  (derek_age : ℕ) 
  (h1 : bob_age = 60)
  (h2 : evan_age = (2 * bob_age) / 3)
  (h3 : derek_age = evan_age - 10) : 
  derek_age = 30 :=
by
  -- The proof is to be filled in
  sorry

end derek_age_calculation_l132_132274


namespace poly_expansion_sum_l132_132747

theorem poly_expansion_sum (A B C D E : ℤ) (x : ℤ):
  (x + 3) * (4 * x^3 - 2 * x^2 + 3 * x - 1) = A * x^4 + B * x^3 + C * x^2 + D * x + E → 
  A + B + C + D + E = 16 :=
by
  sorry

end poly_expansion_sum_l132_132747


namespace inequalities_not_equivalent_l132_132039

theorem inequalities_not_equivalent (x : ℝ) (h1 : x ≠ 1) :
  (x + 3 - (1 / (x - 1)) > -x + 2 - (1 / (x - 1))) ↔ (x + 3 > -x + 2) → False :=
by
  sorry

end inequalities_not_equivalent_l132_132039


namespace repeating_pattern_sum_23_l132_132879

def repeating_pattern_sum (n : ℕ) : ℤ :=
  let pattern := [4, -3, 2, -1, 0]
  let block_sum := List.sum pattern
  let complete_blocks := n / pattern.length
  let remainder := n % pattern.length
  complete_blocks * block_sum + List.sum (pattern.take remainder)

theorem repeating_pattern_sum_23 : repeating_pattern_sum 23 = 11 := 
  sorry

end repeating_pattern_sum_23_l132_132879


namespace correct_operation_B_l132_132363

variable (a : ℝ)

theorem correct_operation_B :
  2 * a^2 * a^4 = 2 * a^6 :=
by sorry

end correct_operation_B_l132_132363


namespace question1_question2_l132_132378

-- Define required symbols and parameters
variables {x : ℝ} {b c : ℝ}

-- Statement 1: Proving b + c given the conditions on the inequality
theorem question1 (h : ∀ x, -1 < x ∧ x < 3 → 5*x^2 - b*x + c < 0) : b + c = -25 := sorry

-- Statement 2: Proving the solution set for the given inequality
theorem question2 (h : ∀ x, (2 * x - 5) / (x + 4) ≥ 0 → (x ≥ 5 / 2 ∨ x < -4)) : 
  {x | (2 * x - 5) / (x + 4) ≥ 0} = {x | x ≥ 5/2 ∨ x < -4} := sorry

end question1_question2_l132_132378


namespace part_a_part_b_part_c_l132_132584

-- Part (a)
theorem part_a (x y : ℕ) (h : (2 * x + 11 * y) = 3 * x + 4 * y) : x = 7 * y := by
  sorry

-- Part (b)
theorem part_b (u v : ℚ) : ∃ (x y : ℚ), (x + y) / 2 = (u.num * v.den + v.num * u.den) / (2 * u.den * v.den) := by
  sorry

-- Part (c)
theorem part_c (u v : ℚ) (h : u < v) : ∀ (m : ℚ), (m.num = u.num + v.num) ∧ (m.den = u.den + v.den) → u < m ∧ m < v := by
  sorry

end part_a_part_b_part_c_l132_132584


namespace triangle_is_isosceles_right_l132_132251

theorem triangle_is_isosceles_right (a b S : ℝ) (h : S = (1/4) * (a^2 + b^2)) :
  ∃ C : ℝ, C = 90 ∧ a = b :=
by
  sorry

end triangle_is_isosceles_right_l132_132251


namespace dining_bill_split_l132_132590

theorem dining_bill_split (original_bill : ℝ) (num_people : ℕ) (tip_percent : ℝ) (total_bill_with_tip : ℝ) (amount_per_person : ℝ)
  (h1 : original_bill = 139.00)
  (h2 : num_people = 3)
  (h3 : tip_percent = 0.10)
  (h4 : total_bill_with_tip = original_bill + (tip_percent * original_bill))
  (h5 : amount_per_person = total_bill_with_tip / num_people) :
  amount_per_person = 50.97 :=
by 
  sorry

end dining_bill_split_l132_132590


namespace mia_socks_problem_l132_132707

theorem mia_socks_problem (x y z w : ℕ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) (hw : 1 ≤ w)
  (h1 : x + y + z + w = 16) (h2 : x + 2*y + 3*z + 4*w = 36) : x = 3 :=
sorry

end mia_socks_problem_l132_132707


namespace keith_total_cost_correct_l132_132486

noncomputable def total_cost_keith_purchases : Real :=
  let discount_toy := 6.51
  let price_toy := discount_toy / 0.90
  let pet_food := 5.79
  let cage_price := 12.51
  let tax_rate := 0.08
  let cage_tax := cage_price * tax_rate
  let price_with_tax := cage_price + cage_tax
  let water_bottle := 4.99
  let bedding := 7.65
  let discovered_money := 1.0
  let total_cost := discount_toy + pet_food + price_with_tax + water_bottle + bedding
  total_cost - discovered_money

theorem keith_total_cost_correct :
  total_cost_keith_purchases = 37.454 :=
by
  sorry -- Proof of the theorem will go here

end keith_total_cost_correct_l132_132486


namespace percentage_relation_l132_132036

theorem percentage_relation (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 12) : y = 60 := by
  sorry

end percentage_relation_l132_132036


namespace smallest_base_b_l132_132289

theorem smallest_base_b (b : ℕ) (n : ℕ) : b > 3 ∧ 3 * b + 4 = n ^ 2 → b = 4 := 
by
  sorry

end smallest_base_b_l132_132289


namespace sum_of_digits_power_of_9_gt_9_l132_132137

def sum_of_digits (n : ℕ) : ℕ :=
  -- function to calculate the sum of digits of n 
  sorry

theorem sum_of_digits_power_of_9_gt_9 (n : ℕ) (h : n ≥ 3) : sum_of_digits (9^n) > 9 :=
  sorry

end sum_of_digits_power_of_9_gt_9_l132_132137


namespace inequality_does_not_hold_l132_132618

theorem inequality_does_not_hold (a b : ℝ) (h₁ : a < b) (h₂ : b < 0) :
  ¬ (1 / (a - 1) < 1 / b) :=
by
  sorry

end inequality_does_not_hold_l132_132618


namespace proof_abc_div_def_l132_132169

def abc_div_def (a b c d e f : ℚ) : Prop := 
  a / b = 1 / 3 ∧ b / c = 2 ∧ c / d = 1 / 2 ∧ d / e = 3 ∧ e / f = 1 / 8 → (a * b * c) / (d * e * f) = 1 / 16

theorem proof_abc_div_def (a b c d e f : ℚ) :
  abc_div_def a b c d e f :=
by 
  sorry

end proof_abc_div_def_l132_132169


namespace lcm_of_6_8_10_l132_132352

theorem lcm_of_6_8_10 : Nat.lcm (Nat.lcm 6 8) 10 = 120 := 
  by sorry

end lcm_of_6_8_10_l132_132352


namespace students_failed_l132_132674

theorem students_failed (Q : ℕ) (x : ℕ) (h1 : 4 * Q < 56) (h2 : x = Nat.lcm 3 (Nat.lcm 7 2)) (h3 : x < 56) :
  let R := x - (x / 3 + x / 7 + x / 2) 
  R = 1 := 
by
  sorry

end students_failed_l132_132674


namespace brendan_match_ratio_l132_132767

noncomputable def brendanMatches (totalMatches firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound : ℕ) :=
  matchesWonFirstTwoRounds = firstRound + secondRound ∧
  matchesWonFirstTwoRounds = 12 ∧
  totalMatches = matchesWonTotal ∧
  matchesWonTotal = 14 ∧
  firstRound = 6 ∧
  secondRound = 6 ∧
  matchesInLastRound = 4

theorem brendan_match_ratio :
  ∃ ratio: ℕ × ℕ,
    let firstRound := 6
    let secondRound := 6
    let matchesInLastRound := 4
    let matchesWonFirstTwoRounds := firstRound + secondRound
    let matchesWonTotal := 14
    let matchesWonLastRound := matchesWonTotal - matchesWonFirstTwoRounds
    let ratio := (matchesWonLastRound, matchesInLastRound)
    brendanMatches matchesWonTotal firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound ∧
    ratio = (1, 2) :=
by
  sorry

end brendan_match_ratio_l132_132767


namespace intersection_points_in_decagon_l132_132932

-- Define the number of sides for a regular decagon
def n : ℕ := 10

-- The formula to calculate the number of ways to choose 4 vertices from n vertices
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The statement that needs to be proven
theorem intersection_points_in_decagon : choose 10 4 = 210 := by
  sorry

end intersection_points_in_decagon_l132_132932


namespace log_expression_simplifies_to_zero_l132_132525

theorem log_expression_simplifies_to_zero : 
  (1/2 : ℝ) * (Real.log 4) + Real.log 5 - Real.exp (0 * Real.log (Real.pi + 1)) = 0 := 
by
  sorry

end log_expression_simplifies_to_zero_l132_132525


namespace f_identity_l132_132729

def f (x : ℝ) : ℝ := (2 * x + 1)^5 - 5 * (2 * x + 1)^4 + 10 * (2 * x + 1)^3 - 10 * (2 * x + 1)^2 + 5 * (2 * x + 1) - 1

theorem f_identity (x : ℝ) : f x = 32 * x^5 :=
by
  -- the proof is omitted
  sorry

end f_identity_l132_132729


namespace son_age_next_year_l132_132607

-- Definitions based on the given conditions
def my_current_age : ℕ := 35
def son_current_age : ℕ := my_current_age / 5

-- Theorem statement to prove the answer
theorem son_age_next_year : son_current_age + 1 = 8 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end son_age_next_year_l132_132607


namespace Quentin_chickens_l132_132671

variable (C S Q : ℕ)

theorem Quentin_chickens (h1 : C = 37)
    (h2 : S = 3 * C - 4)
    (h3 : Q + S + C = 383) :
    (Q = 2 * S + 32) :=
by
  sorry

end Quentin_chickens_l132_132671


namespace arithmetic_expr_eval_l132_132967

/-- A proof that the arithmetic expression (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) evaluates to -13122. -/
theorem arithmetic_expr_eval : (80 + (5 * 12) / (180 / 3)) ^ 2 * (7 - (3^2)) = -13122 :=
by
  sorry

end arithmetic_expr_eval_l132_132967


namespace max_f_value_l132_132225

open Real

noncomputable def problem (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) : Prop :=
  x1 * x2 * x3 = ((12 - x1) * (12 - x2) * (12 - x3))^2

theorem max_f_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) (h : problem x1 x2 x3 h1 h2 h3) : 
  x1 * x2 * x3 ≤ 729 :=
sorry

end max_f_value_l132_132225


namespace circles_intersect_range_l132_132160

def circle1_radius := 3
def circle2_radius := 5

theorem circles_intersect_range : 2 < d ∧ d < 8 :=
by
  let r1 := circle1_radius
  let r2 := circle2_radius
  have h1 : d > r2 - r1 := sorry
  have h2 : d < r2 + r1 := sorry
  exact ⟨h1, h2⟩

end circles_intersect_range_l132_132160


namespace paul_can_buy_toys_l132_132111

-- Definitions of the given conditions
def initial_dollars : ℕ := 3
def allowance : ℕ := 7
def toy_cost : ℕ := 5

-- Required proof statement
theorem paul_can_buy_toys : (initial_dollars + allowance) / toy_cost = 2 := by
  sorry

end paul_can_buy_toys_l132_132111


namespace men_in_group_l132_132389

theorem men_in_group (A : ℝ) (n : ℕ) (h : n > 0) 
  (inc_avg : ↑n * A + 2 * 32 - (21 + 23) = ↑n * (A + 1)) : n = 20 :=
sorry

end men_in_group_l132_132389


namespace line_intersects_x_axis_l132_132122

theorem line_intersects_x_axis (x y : ℝ) (h : 5 * y - 6 * x = 15) (hy : y = 0) : x = -2.5 ∧ y = 0 := 
by
  sorry

end line_intersects_x_axis_l132_132122


namespace quadratic_solution_l132_132071

-- Definitions come from the conditions of the problem
def satisfies_equation (y : ℝ) : Prop := 6 * y^2 + 2 = 4 * y + 12

-- Statement of the proof
theorem quadratic_solution (y : ℝ) (hy : satisfies_equation y) : (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := 
sorry

end quadratic_solution_l132_132071


namespace find_positive_integers_l132_132403

noncomputable def positive_integer_solutions_ineq (x : ℕ) : Prop :=
  x > 0 ∧ (x : ℝ) < 4

theorem find_positive_integers (x : ℕ) : 
  (x > 0 ∧ (↑x - 3)/3 < 7 - 5*(↑x)/3) ↔ positive_integer_solutions_ineq x :=
by
  sorry

end find_positive_integers_l132_132403


namespace exists_m_divisible_by_2k_l132_132480

theorem exists_m_divisible_by_2k {k : ℕ} (h_k : 0 < k) {a : ℤ} (h_a : a % 8 = 3) :
  ∃ m : ℕ, 0 < m ∧ 2^k ∣ (a^m + a + 2) :=
sorry

end exists_m_divisible_by_2k_l132_132480


namespace Anil_profit_in_rupees_l132_132506

def cost_scooter (C : ℝ) : Prop := 0.10 * C = 500
def profit (C P : ℝ) : Prop := P = 0.20 * C

theorem Anil_profit_in_rupees (C P : ℝ) (h1 : cost_scooter C) (h2 : profit C P) : P = 1000 :=
by
  sorry

end Anil_profit_in_rupees_l132_132506


namespace original_price_l132_132977

theorem original_price (x : ℝ) (h1 : x > 0) (h2 : 1.12 * x - x = 270) : x = 2250 :=
by
  sorry

end original_price_l132_132977


namespace train_speed_l132_132809

noncomputable def train_length : ℝ := 2500
noncomputable def time_to_cross_pole : ℝ := 35

noncomputable def speed_in_kmph (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed :
  speed_in_kmph train_length time_to_cross_pole = 257.14 := by
  sorry

end train_speed_l132_132809


namespace balance_scale_with_blue_balls_l132_132703

variables (G Y W B : ℝ)

-- Conditions
def green_to_blue := 4 * G = 8 * B
def yellow_to_blue := 3 * Y = 8 * B
def white_to_blue := 5 * B = 3 * W

-- Proof problem statement
theorem balance_scale_with_blue_balls (h1 : green_to_blue G B) (h2 : yellow_to_blue Y B) (h3 : white_to_blue W B) : 
  3 * G + 3 * Y + 3 * W = 19 * B :=
by sorry

end balance_scale_with_blue_balls_l132_132703


namespace main_theorem_l132_132930

variable (x : ℝ)

-- Define proposition p
def p : Prop := ∃ x0 : ℝ, x0^2 < x0

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- Main proof problem
theorem main_theorem : p ∧ q := 
by {
  sorry
}

end main_theorem_l132_132930


namespace logarithm_identity_l132_132939

theorem logarithm_identity (k x : ℝ) (hk : 0 < k ∧ k ≠ 1) (hx : 0 < x) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 3 → x = 343 :=
by
  intro h
  sorry

end logarithm_identity_l132_132939


namespace probability_single_trial_l132_132757

open Real

theorem probability_single_trial :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (1 - p)^4 = 16 / 81 ∧ p = 1 / 3 :=
by
  -- The proof steps have been skipped.
  sorry

end probability_single_trial_l132_132757


namespace student_count_l132_132081

theorem student_count 
( M S N : ℕ ) 
(h1 : N - M = 10) 
(h2 : N - S = 15) 
(h3 : N - (M + S - 7) = 2) : 
N = 34 :=
by
  sorry

end student_count_l132_132081


namespace number_20_l132_132138

def Jo (n : ℕ) : ℕ :=
  1 + 5 * (n - 1)

def Blair (n : ℕ) : ℕ :=
  3 + 5 * (n - 1)

def number_at_turn (k : ℕ) : ℕ :=
  if k % 2 = 1 then Jo ((k + 1) / 2) else Blair (k / 2)

theorem number_20 : number_at_turn 20 = 48 :=
by
  sorry

end number_20_l132_132138


namespace union_of_A_B_l132_132301

def A : Set ℝ := { x | x^2 - x - 2 ≤ 0 }

def B : Set ℝ := { x | 1 < x ∧ x ≤ 3 }

theorem union_of_A_B : A ∪ B = { x | -1 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end union_of_A_B_l132_132301


namespace sum_of_powers_of_i_l132_132527

open Complex

def i := Complex.I

theorem sum_of_powers_of_i : (i + i^2 + i^3 + i^4) = 0 := 
by
  sorry

end sum_of_powers_of_i_l132_132527


namespace simplify_evaluate_l132_132438

noncomputable def a := (1 / 2) + Real.sqrt (1 / 2)

theorem simplify_evaluate (a : ℝ) (h : a = (1 / 2) + Real.sqrt (1 / 2)) :
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 :=
by sorry

end simplify_evaluate_l132_132438


namespace daragh_sisters_count_l132_132044

theorem daragh_sisters_count (initial_bears : ℕ) (favorite_bears : ℕ) (eden_initial_bears : ℕ) (eden_total_bears : ℕ) 
    (remaining_bears := initial_bears - favorite_bears)
    (eden_received_bears := eden_total_bears - eden_initial_bears)
    (bears_per_sister := eden_received_bears) :
    initial_bears = 20 → favorite_bears = 8 → eden_initial_bears = 10 → eden_total_bears = 14 → 
    remaining_bears / bears_per_sister = 3 := 
by
  sorry

end daragh_sisters_count_l132_132044


namespace negation_of_proposition_true_l132_132753

theorem negation_of_proposition_true :
  (¬ (∀ x: ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ (∃ x: ℝ, x^2 ≥ 1 ∧ (x ≤ -1 ∨ x ≥ 1)) :=
by
  sorry

end negation_of_proposition_true_l132_132753


namespace relationship_between_a_and_b_l132_132029

variable {a b : ℝ} (n : ℕ)

theorem relationship_between_a_and_b (h₁ : a^n = a + 1) (h₂ : b^(2 * n) = b + 3 * a)
  (h₃ : 2 ≤ n) (h₄ : 1 < a) (h₅ : 1 < b) : a > b ∧ b > 1 :=
by
  sorry

end relationship_between_a_and_b_l132_132029


namespace arc_length_l132_132181

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 2) (h_α : α = π / 7) : (α * r) = 2 * π / 7 := by
  sorry

end arc_length_l132_132181


namespace find_z_when_w_15_l132_132951

-- Define a direct variation relationship
def varies_directly (z w : ℕ) (k : ℕ) : Prop :=
  z = k * w

-- Using the given conditions and to prove the statement
theorem find_z_when_w_15 :
  ∃ k, (varies_directly 10 5 k) → (varies_directly 30 15 k) :=
by
  sorry

end find_z_when_w_15_l132_132951


namespace rain_total_duration_l132_132701

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end rain_total_duration_l132_132701


namespace central_angle_double_score_l132_132284

theorem central_angle_double_score 
  (prob: ℚ)
  (total_angle: ℚ)
  (num_regions: ℚ)
  (eq_regions: ℚ → Prop)
  (double_score_prob: prob = 1/8)
  (total_angle_eq: total_angle = 360)
  (num_regions_eq: num_regions = 6) 
  : ∃ x: ℚ, (prob = x / total_angle) → x = 45 :=
by
  sorry

end central_angle_double_score_l132_132284


namespace factor_polynomial_l132_132639

theorem factor_polynomial (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := 
by sorry

end factor_polynomial_l132_132639


namespace valid_P_values_l132_132722

/-- 
Construct a 3x3 grid of distinct natural numbers where the product of the numbers 
in each row and each column is equal. Verify the valid values of P among the given set.
-/
theorem valid_P_values (P : ℕ) :
  (∃ (a b c d e f g h i : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ 
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ 
    g ≠ h ∧ g ≠ i ∧ 
    h ≠ i ∧ 
    a * b * c = P ∧ 
    d * e * f = P ∧ 
    g * h * i = P ∧ 
    a * d * g = P ∧ 
    b * e * h = P ∧ 
    c * f * i = P ∧ 
    P = (Nat.sqrt ((1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9)) )) ↔ P = 1998 ∨ P = 2000 :=
sorry

end valid_P_values_l132_132722


namespace children_selection_l132_132762

-- Conditions and definitions
def comb (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Proof problem statement
theorem children_selection : ∃ r : ℕ, comb 10 r = 210 ∧ r = 4 :=
by
  sorry

end children_selection_l132_132762


namespace group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l132_132485

-- Question 1
theorem group_photo_arrangements {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ arrangements : ℕ, arrangements = 14400 := 
sorry

-- Question 2
theorem grouping_methods {N : ℕ} (hN : N = 8) :
  ∃ methods : ℕ, methods = 2520 := 
sorry

-- Question 3
theorem selection_methods_with_at_least_one_male {M F : ℕ} (hM : M = 3) (hF : F = 5) :
  ∃ methods : ℕ, methods = 1560 := 
sorry

end group_photo_arrangements_grouping_methods_selection_methods_with_at_least_one_male_l132_132485


namespace initial_amount_100000_l132_132013

noncomputable def compound_interest_amount (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value (P CI : ℝ) : ℝ :=
  P + CI

theorem initial_amount_100000
  (CI : ℝ) (P : ℝ) (r : ℝ) (n t : ℕ) 
  (h1 : CI = 8243.216)
  (h2 : r = 0.04)
  (h3 : n = 2)
  (h4 : t = 2)
  (h5 : future_value P CI = compound_interest_amount P r n t) :
  P = 100000 :=
by
  sorry

end initial_amount_100000_l132_132013


namespace trigonometric_identity_proof_l132_132888

theorem trigonometric_identity_proof (alpha : Real)
(h1 : Real.tan (alpha + π / 4) = 1 / 2)
(h2 : -π / 2 < alpha ∧ alpha < 0) :
  (2 * Real.sin alpha ^ 2 + Real.sin (2 * alpha)) / Real.cos (alpha - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end trigonometric_identity_proof_l132_132888


namespace total_peaches_l132_132908

theorem total_peaches (x : ℕ) (P : ℕ) 
(h1 : P = 6 * x + 57)
(h2 : 6 * x + 57 = 9 * x - 51) : 
  P = 273 :=
by
  sorry

end total_peaches_l132_132908


namespace sin_tan_identity_of_cos_eq_tan_identity_l132_132796

open Real

variable (α : ℝ)
variable (hα : α ∈ Ioo 0 π)   -- α is in the interval (0, π)
variable (hcos : cos (2 * α) = 2 * cos (α + π / 4))

theorem sin_tan_identity_of_cos_eq_tan_identity : 
  sin (2 * α) = 1 ∧ tan α = 1 :=
by
  sorry

end sin_tan_identity_of_cos_eq_tan_identity_l132_132796


namespace no_fixed_points_l132_132341

def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 1

theorem no_fixed_points (a : ℝ) :
  (∀ x : ℝ, f x a ≠ x) ↔ (-1/2 < a ∧ a < 3/2) := by
    sorry

end no_fixed_points_l132_132341


namespace ratio_of_boys_to_girls_l132_132095

theorem ratio_of_boys_to_girls (boys : ℕ) (students : ℕ) (h1 : boys = 42) (h2 : students = 48) : (boys : ℚ) / (students - boys : ℚ) = 7 / 1 := 
by
  sorry

end ratio_of_boys_to_girls_l132_132095


namespace inequality_holds_for_all_x_l132_132261

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x)) ↔ -2 < m ∧ m ≤ 2 := 
by
  sorry

end inequality_holds_for_all_x_l132_132261


namespace initial_ducks_l132_132712

theorem initial_ducks (D : ℕ) (h1 : D + 20 = 33) : D = 13 :=
by sorry

end initial_ducks_l132_132712


namespace total_banana_produce_correct_l132_132439

-- Defining the conditions as variables and constants
def B_nearby : ℕ := 9000
def B_Jakies : ℕ := 10 * B_nearby
def T : ℕ := B_nearby + B_Jakies

-- Theorem statement
theorem total_banana_produce_correct : T = 99000 := by
  sorry  -- Proof placeholder

end total_banana_produce_correct_l132_132439


namespace g_s_difference_l132_132229

def g (n : ℤ) : ℤ := n^3 + 3 * n^2 + 3 * n + 1

theorem g_s_difference (s : ℤ) : g s - g (s - 2) = 6 * s^2 + 2 := by
  sorry

end g_s_difference_l132_132229


namespace arithmetic_progressions_count_l132_132210

theorem arithmetic_progressions_count (d : ℕ) (h_d : d = 2) (S : ℕ) (h_S : S = 200) : 
  ∃ n : ℕ, n = 6 := sorry

end arithmetic_progressions_count_l132_132210


namespace point_of_tangency_l132_132461

def parabola1 (x y : ℝ) : Prop := y = x^2 + 15*x + 32
def parabola2 (x y : ℝ) : Prop := x = y^2 + 49*y + 593

theorem point_of_tangency :
  parabola1 (-7) (-24) ∧ parabola2 (-7) (-24) := by
  sorry

end point_of_tangency_l132_132461


namespace max_students_can_participate_l132_132598

theorem max_students_can_participate (max_funds rent cost_per_student : ℕ) (h_max_funds : max_funds = 800) (h_rent : rent = 300) (h_cost_per_student : cost_per_student = 15) :
  ∃ x : ℕ, x ≤ (max_funds - rent) / cost_per_student ∧ x = 33 :=
by
  sorry

end max_students_can_participate_l132_132598


namespace price_per_gallon_in_NC_l132_132825

variable (P : ℝ)
variable (price_nc := P) -- price per gallon in North Carolina
variable (price_va := P + 1) -- price per gallon in Virginia
variable (gallons_nc := 10) -- gallons bought in North Carolina
variable (gallons_va := 10) -- gallons bought in Virginia
variable (total_cost := 50) -- total amount spent on gas

theorem price_per_gallon_in_NC :
  (gallons_nc * price_nc) + (gallons_va * price_va) = total_cost → price_nc = 2 :=
by
  sorry

end price_per_gallon_in_NC_l132_132825


namespace investment_worth_l132_132679

noncomputable def initial_investment (total_earning : ℤ) : ℤ := total_earning / 2

noncomputable def current_worth (initial_investment total_earning : ℤ) : ℤ :=
  initial_investment + total_earning

theorem investment_worth (monthly_earning : ℤ) (months : ℤ) (earnings : ℤ)
  (h1 : monthly_earning * months = earnings)
  (h2 : earnings = 2 * initial_investment earnings) :
  current_worth (initial_investment earnings) earnings = 90 := 
by
  -- We proceed to show the current worth is $90
  -- Proof will be constructed here
  sorry
  
end investment_worth_l132_132679


namespace percentage_cut_second_week_l132_132124

noncomputable def calculate_final_weight (initial_weight : ℝ) (percentage1 : ℝ) (percentage2 : ℝ) (percentage3 : ℝ) : ℝ :=
  let weight_after_first_week := (1 - percentage1 / 100) * initial_weight
  let weight_after_second_week := (1 - percentage2 / 100) * weight_after_first_week
  let final_weight := (1 - percentage3 / 100) * weight_after_second_week
  final_weight

theorem percentage_cut_second_week : 
  ∀ (initial_weight : ℝ) (final_weight : ℝ), (initial_weight = 250) → (final_weight = 105) →
    (calculate_final_weight initial_weight 30 x 25 = final_weight) → 
    x = 20 := 
by 
  intros initial_weight final_weight h1 h2 h3
  sorry

end percentage_cut_second_week_l132_132124


namespace projection_of_b_onto_a_l132_132823

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := a.1 * a.1 + a.2 * a.2
  let scalar := dot_product / magnitude_squared
  (scalar * a.1, scalar * a.2)

theorem projection_of_b_onto_a :
  vector_projection (2, -1) (6, 2) = (4, -2) :=
by
  simp [vector_projection]
  sorry

end projection_of_b_onto_a_l132_132823


namespace expression_simplification_l132_132141

variable (x : ℝ)

-- Define the expression as given in the problem
def Expr : ℝ := (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 3)

-- Lean statement to verify that the expression simplifies to the given polynomial
theorem expression_simplification : Expr x = 6 * x^3 - 16 * x^2 + 43 * x - 70 := by
  sorry

end expression_simplification_l132_132141


namespace simplify_fractions_l132_132970

theorem simplify_fractions :
  (36 / 51) * (35 / 24) * (68 / 49) = (20 / 7) :=
by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 51 = 3 * 17 := by norm_num
  have h3 : 35 = 5 * 7 := by norm_num
  have h4 : 24 = 2^3 * 3 := by norm_num
  have h5 : 68 = 2^2 * 17 := by norm_num
  have h6 : 49 = 7^2 := by norm_num
  sorry

end simplify_fractions_l132_132970


namespace randy_blocks_left_l132_132303

-- Formalize the conditions
def initial_blocks : ℕ := 78
def blocks_used_first_tower : ℕ := 19
def blocks_used_second_tower : ℕ := 25

-- Formalize the result for verification
def blocks_left : ℕ := initial_blocks - blocks_used_first_tower - blocks_used_second_tower

-- State the theorem to be proven
theorem randy_blocks_left :
  blocks_left = 34 :=
by
  -- Not providing the proof as per instructions
  sorry

end randy_blocks_left_l132_132303


namespace amount_c_l132_132296

theorem amount_c (a b c d : ℝ) :
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  a + b + c + d = 750 →
  c = 225 :=
by 
  intros h1 h2 h3 h4 h5
  -- Proof omitted.
  sorry

end amount_c_l132_132296


namespace problem_c_d_sum_l132_132868

theorem problem_c_d_sum (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (C / (x - 3) + D * (x - 2) = (5 * x ^ 2 - 8 * x - 6) / (x - 3))) : C + D = 20 :=
sorry

end problem_c_d_sum_l132_132868


namespace mortar_shell_hits_the_ground_at_50_seconds_l132_132374

noncomputable def mortar_shell_firing_equation (x : ℝ) : ℝ :=
  - (1 / 5) * x^2 + 10 * x

theorem mortar_shell_hits_the_ground_at_50_seconds : 
  ∃ x : ℝ, mortar_shell_firing_equation x = 0 ∧ x = 50 :=
by
  sorry

end mortar_shell_hits_the_ground_at_50_seconds_l132_132374


namespace a_eq_one_sufficient_not_necessary_P_subset_M_iff_l132_132737

open Set

-- Define sets P and M based on conditions
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem a_eq_one_sufficient_not_necessary (a : ℝ) : (a = 1) → (P ⊆ M a) := 
by
  sorry

theorem P_subset_M_iff (a : ℝ) : (P ⊆ M a) ↔ (a < 2) :=
by
  sorry

end a_eq_one_sufficient_not_necessary_P_subset_M_iff_l132_132737


namespace min_x_squared_plus_y_squared_l132_132509

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  x^2 + y^2 ≥ 50 :=
by
  sorry

end min_x_squared_plus_y_squared_l132_132509


namespace Dvaneft_percentage_bounds_l132_132038

noncomputable def percentageDvaneftShares (x y z : ℤ) (n m : ℕ) : ℚ :=
  n * 100 / (2 * (n + m))

theorem Dvaneft_percentage_bounds
  (x y z : ℤ) (n m : ℕ)
  (h1 : 4 * x * n = y * m)
  (h2 : x * n + y * m = z * (n + m))
  (h3 : 16 ≤ y - x)
  (h4 : y - x ≤ 20)
  (h5 : 42 ≤ z)
  (h6 : z ≤ 60) :
  12.5 ≤ percentageDvaneftShares x y z n m ∧ percentageDvaneftShares x y z n m ≤ 15 := by
  sorry

end Dvaneft_percentage_bounds_l132_132038


namespace four_digit_integer_l132_132103

theorem four_digit_integer (a b c d : ℕ) (h1 : a + b + c + d = 18)
  (h2 : b + c = 11) (h3 : a - d = 1) (h4 : 11 ∣ (1000 * a + 100 * b + 10 * c + d)) :
  1000 * a + 100 * b + 10 * c + d = 4653 :=
by sorry

end four_digit_integer_l132_132103


namespace tile_ratio_l132_132521

-- Definitions corresponding to the conditions in the problem
def orig_grid_size : ℕ := 6
def orig_black_tiles : ℕ := 12
def orig_white_tiles : ℕ := 24
def border_size : ℕ := 1

-- The combined problem statement
theorem tile_ratio (orig_grid_size orig_black_tiles orig_white_tiles border_size : ℕ) :
  let new_grid_size := orig_grid_size + 2 * border_size
  let new_tiles := new_grid_size^2
  let added_tiles := new_tiles - orig_grid_size^2
  let total_white_tiles := orig_white_tiles + added_tiles
  let black_to_white_ratio := orig_black_tiles / total_white_tiles
  black_to_white_ratio = (3 : ℕ) / 13 :=
by {
  sorry
}

end tile_ratio_l132_132521


namespace ageOfX_l132_132364

def threeYearsAgo (x y : ℕ) := x - 3 = 2 * (y - 3)
def sevenYearsHence (x y : ℕ) := (x + 7) + (y + 7) = 83

theorem ageOfX (x y : ℕ) (h1 : threeYearsAgo x y) (h2 : sevenYearsHence x y) : x = 45 := by
  sorry

end ageOfX_l132_132364


namespace problem_l132_132980

-- Conditions
variables (x y : ℚ)
def condition1 := 3 * x + 5 = 12
def condition2 := 10 * y - 2 = 5

-- Theorem to prove
theorem problem (h1 : condition1 x) (h2 : condition2 y) : x + y = 91 / 30 := sorry

end problem_l132_132980


namespace probability_coin_covers_black_region_l132_132226

open Real

noncomputable def coin_cover_black_region_probability : ℝ :=
  let side_length_square := 10
  let triangle_leg := 3
  let diamond_side_length := 3 * sqrt 2
  let smaller_square_side := 1
  let coin_diameter := 1
  -- The derived probability calculation
  (32 + 9 * sqrt 2 + π) / 81

theorem probability_coin_covers_black_region :
  coin_cover_black_region_probability = (32 + 9 * sqrt 2 + π) / 81 :=
by
  -- Proof goes here
  sorry

end probability_coin_covers_black_region_l132_132226


namespace quadratic_fixed_points_l132_132547

noncomputable def quadratic_function (a x : ℝ) : ℝ :=
  a * x^2 + (3 * a - 1) * x - (10 * a + 3)

theorem quadratic_fixed_points (a : ℝ) (h : a ≠ 0) :
  quadratic_function a 2 = -5 ∧ quadratic_function a (-5) = 2 :=
by sorry

end quadratic_fixed_points_l132_132547


namespace avg_of_nine_numbers_l132_132154

theorem avg_of_nine_numbers (average : ℝ) (sum : ℝ) (h : average = (sum / 9)) (h_avg : average = 5.3) : sum = 47.7 := by
  sorry

end avg_of_nine_numbers_l132_132154


namespace d_is_distance_function_l132_132298

noncomputable def d (x y : ℝ) : ℝ := |x - y| / (Real.sqrt (1 + x^2) * Real.sqrt (1 + y^2))

theorem d_is_distance_function : 
  (∀ x, d x x = 0) ∧ 
  (∀ x y, d x y = d y x) ∧ 
  (∀ x y z, d x y + d y z ≥ d x z) :=
by
  sorry

end d_is_distance_function_l132_132298


namespace binom_computation_l132_132864

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l132_132864


namespace polynomial_perfect_square_value_of_k_l132_132861

noncomputable def is_perfect_square (p : Polynomial ℝ) : Prop :=
  ∃ (q : Polynomial ℝ), p = q^2

theorem polynomial_perfect_square_value_of_k {k : ℝ} :
  is_perfect_square (Polynomial.X^2 - Polynomial.C k * Polynomial.X + Polynomial.C 25) ↔ (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_value_of_k_l132_132861


namespace cindy_correct_answer_l132_132110

theorem cindy_correct_answer (x : ℝ) (h₀ : (x - 12) / 4 = 32) : (x - 7) / 5 = 27 :=
by
  sorry

end cindy_correct_answer_l132_132110


namespace problem1_problem2_l132_132032

variable {a b : ℝ}

theorem problem1
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : (a + b) * (a^5 + b^5) ≥ 4 := sorry

theorem problem2
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : a + b ≤ 2 := sorry

end problem1_problem2_l132_132032


namespace probability_of_one_each_color_is_two_fifths_l132_132449

/-- Definition for marbles bag containing 2 red, 2 blue, and 2 green marbles -/
structure MarblesBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  total : ℕ := red + blue + green

/-- Initial setup for the problem -/
def initialBag : MarblesBag := { red := 2, blue := 2, green := 2 }

/-- Represents the outcome of selecting marbles without replacement -/
def selectMarbles (bag : MarblesBag) (count : ℕ) : ℕ :=
  Nat.choose bag.total count

/-- The number of ways to select one marble of each color -/
def selectOneOfEachColor (bag : MarblesBag) : ℕ :=
  bag.red * bag.blue * bag.green

/-- Calculate the probability of selecting one marble of each color -/
def probabilityOneOfEachColor (bag : MarblesBag) (selectCount : ℕ) : ℚ :=
  selectOneOfEachColor bag / selectMarbles bag selectCount

/-- Theorem stating the answer to the probability problem -/
theorem probability_of_one_each_color_is_two_fifths (bag : MarblesBag) :
  probabilityOneOfEachColor bag 3 = 2 / 5 := by
  sorry

end probability_of_one_each_color_is_two_fifths_l132_132449


namespace frisbee_sales_l132_132050

/-- A sporting goods store sold some frisbees, with $3 and $4 price points.
The total receipts from frisbee sales were $204. The fewest number of $4 frisbees that could have been sold is 24.
Prove the total number of frisbees sold is 60. -/
theorem frisbee_sales (x y : ℕ) (h1 : 3 * x + 4 * y = 204) (h2 : 24 ≤ y) : x + y = 60 :=
by {
  -- Proof skipped
  sorry
}

end frisbee_sales_l132_132050


namespace find_y_l132_132719

variables (y : ℝ)

def rectangle_vertices (A B C D : (ℝ × ℝ)) : Prop :=
  (A = (-2, y)) ∧ (B = (10, y)) ∧ (C = (-2, 1)) ∧ (D = (10, 1))

def rectangle_area (length height : ℝ) : Prop :=
  length * height = 108

def positive_value (x : ℝ) : Prop :=
  0 < x

theorem find_y (A B C D : (ℝ × ℝ)) (hV : rectangle_vertices y A B C D) (hA : rectangle_area 12 (y - 1)) (hP : positive_value y) :
  y = 10 :=
sorry

end find_y_l132_132719


namespace lcm_nuts_bolts_l132_132369

theorem lcm_nuts_bolts : Nat.lcm 13 8 = 104 := 
sorry

end lcm_nuts_bolts_l132_132369


namespace express_w_l132_132412

theorem express_w (w a b c : ℝ) (x y z : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ w ∧ b ≠ w ∧ c ≠ w)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  w = - (a * b * c) / (a * b + b * c + c * a) :=
sorry

end express_w_l132_132412


namespace quadratic_to_vertex_form_l132_132549

theorem quadratic_to_vertex_form :
  ∀ (x : ℝ), (x^2 - 2*x + 3 = (x-1)^2 + 2) :=
by intro x; sorry

end quadratic_to_vertex_form_l132_132549


namespace arrangements_of_masters_and_apprentices_l132_132991

theorem arrangements_of_masters_and_apprentices : 
  ∃ n : ℕ, n = 48 ∧ 
     let pairs := 3 
     let ways_to_arrange_pairs := pairs.factorial 
     let ways_to_arrange_within_pairs := 2 ^ pairs 
     ways_to_arrange_pairs * ways_to_arrange_within_pairs = n := 
sorry

end arrangements_of_masters_and_apprentices_l132_132991


namespace range_of_a1_l132_132828

theorem range_of_a1 (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 1 / (2 - a n)) (h2 : ∀ n, a (n + 1) > a n) :
  a 1 < 1 :=
sorry

end range_of_a1_l132_132828


namespace complex_eq_l132_132777

theorem complex_eq : ∀ (z : ℂ), (i * z = i + z) → (z = (1 - i) / 2) :=
by
  intros z h
  sorry

end complex_eq_l132_132777


namespace probability_of_same_color_pairs_left_right_l132_132097

-- Define the counts of different pairs
def total_pairs := 15
def black_pairs := 8
def red_pairs := 4
def white_pairs := 3

-- Define the total number of shoes
def total_shoes := 30

-- Define the total ways to choose any 2 shoes out of total_shoes
def total_ways := Nat.choose total_shoes 2

-- Define the ways to choose one left and one right for each color
def black_ways := black_pairs * black_pairs
def red_ways := red_pairs * red_pairs
def white_ways := white_pairs * white_pairs

-- Define the total favorable outcomes for same color pairs
def total_favorable := black_ways + red_ways + white_ways

-- Define the probability
def probability := (total_favorable, total_ways)

-- Statement to prove
theorem probability_of_same_color_pairs_left_right :
  probability = (89, 435) :=
by
  sorry

end probability_of_same_color_pairs_left_right_l132_132097


namespace total_salary_correct_l132_132353

-- Define the daily salaries
def owner_salary : ℕ := 20
def manager_salary : ℕ := 15
def cashier_salary : ℕ := 10
def clerk_salary : ℕ := 5
def bagger_salary : ℕ := 3

-- Define the number of employees
def num_owners : ℕ := 1
def num_managers : ℕ := 3
def num_cashiers : ℕ := 5
def num_clerks : ℕ := 7
def num_baggers : ℕ := 9

-- Define the total salary calculation
def total_daily_salary : ℕ :=
  (num_owners * owner_salary) +
  (num_managers * manager_salary) +
  (num_cashiers * cashier_salary) +
  (num_clerks * clerk_salary) +
  (num_baggers * bagger_salary)

-- The theorem we need to prove
theorem total_salary_correct :
  total_daily_salary = 177 :=
by
  -- Proof can be filled in later
  sorry

end total_salary_correct_l132_132353


namespace confidence_95_implies_K2_gt_3_841_l132_132950

-- Conditions
def confidence_no_relationship (K2 : ℝ) : Prop := K2 ≤ 3.841
def confidence_related_95 (K2 : ℝ) : Prop := K2 > 3.841
def confidence_related_99 (K2 : ℝ) : Prop := K2 > 6.635

theorem confidence_95_implies_K2_gt_3_841 (K2 : ℝ) :
  confidence_related_95 K2 ↔ K2 > 3.841 :=
by sorry

end confidence_95_implies_K2_gt_3_841_l132_132950


namespace proof_problem_l132_132972

def diamondsuit (a b : ℝ) : ℝ := a^3 * b - a * b^3

theorem proof_problem :
  { (x, y) : ℝ × ℝ | diamondsuit x y = diamondsuit y x } =
  { (x, y) | x = 0 } ∪ { (x, y) | y = 0 } ∪ { (x, y) | x = y } ∪ { (x, y) | x = -y } :=
by
  sorry

end proof_problem_l132_132972


namespace coin_value_permutations_l132_132947

theorem coin_value_permutations : 
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540 := by
  let digits := [1, 2, 2, 4, 4, 5, 9]
  let odd_digits := [1, 5, 9]
  let permutations (l : List ℕ) := Nat.factorial (l.length) / (l.filter (· = 2)).length.factorial / (l.filter (· = 4)).length.factorial
  show 3 * permutations (digits.erase 1 ++ digits.erase 5 ++ digits.erase 9) = 540
  
  -- Steps for the proof can be filled in
  -- sorry in place to indicate incomplete proof steps
  sorry

end coin_value_permutations_l132_132947


namespace flour_maximum_weight_l132_132430

/-- Given that the bag of flour is marked with 25kg + 50g, prove that the maximum weight of the flour is 25.05kg. -/
theorem flour_maximum_weight :
  let weight_kg := 25
  let weight_g := 50
  (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 :=
by 
  -- provide definitions
  let weight_kg := 25
  let weight_g := 50
  have : (weight_kg + (weight_g / 1000 : ℝ)) = 25.05 := sorry
  exact this

end flour_maximum_weight_l132_132430


namespace prove_k_eq_5_l132_132074

variable (a b k : ℕ)

theorem prove_k_eq_5 (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (a^2 - 1 - b^2) / (a * b - 1) = k) : k = 5 :=
sorry

end prove_k_eq_5_l132_132074


namespace max_ways_to_ascend_descend_l132_132612

theorem max_ways_to_ascend_descend :
  let east_paths := 2
  let west_paths := 1
  let south_paths := 3
  let north_paths := 4

  let descend_from_east := west_paths + south_paths + north_paths
  let descend_from_west := east_paths + south_paths + north_paths
  let descend_from_south := east_paths + west_paths + north_paths
  let descend_from_north := east_paths + west_paths + south_paths

  let ways_from_east := east_paths * descend_from_east
  let ways_from_west := west_paths * descend_from_west
  let ways_from_south := south_paths * descend_from_south
  let ways_from_north := north_paths * descend_from_north

  max ways_from_east (max ways_from_west (max ways_from_south ways_from_north)) = 24 := 
by
  -- Insert the proof here
  sorry

end max_ways_to_ascend_descend_l132_132612


namespace range_of_n_l132_132010

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B (n : ℝ) : Set ℝ := {x | n-1 < x ∧ x < n+1}

-- Define the condition A ∩ B ≠ ∅
def A_inter_B_nonempty (n : ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B n

-- Prove the range of n for which A ∩ B ≠ ∅ is (-2, 2)
theorem range_of_n : ∀ n, A_inter_B_nonempty n ↔ (-2 < n ∧ n < 2) := by
  sorry

end range_of_n_l132_132010


namespace leftover_stickers_l132_132570

-- Definitions for each person's stickers
def ninaStickers : ℕ := 53
def oliverStickers : ℕ := 68
def pattyStickers : ℕ := 29

-- The number of stickers in a package
def packageSize : ℕ := 18

-- The total number of stickers
def totalStickers : ℕ := ninaStickers + oliverStickers + pattyStickers

-- Proof that the number of leftover stickers is 6 when all stickers are divided into packages of 18
theorem leftover_stickers : totalStickers % packageSize = 6 := by
  sorry

end leftover_stickers_l132_132570


namespace length_of_second_train_is_approximately_159_98_l132_132921

noncomputable def length_of_second_train : ℝ :=
  let length_first_train := 110 -- meters
  let speed_first_train := 60 -- km/hr
  let speed_second_train := 40 -- km/hr
  let time_to_cross := 9.719222462203025 -- seconds
  let km_per_hr_to_m_per_s := 5 / 18 -- conversion factor from km/hr to m/s
  let relative_speed := (speed_first_train + speed_second_train) * km_per_hr_to_m_per_s -- relative speed in m/s
  let total_distance := relative_speed * time_to_cross -- total distance covered
  total_distance - length_first_train -- length of the second train

theorem length_of_second_train_is_approximately_159_98 :
  abs (length_of_second_train - 159.98) < 0.01 := 
by
  sorry -- Placeholder for the actual proof

end length_of_second_train_is_approximately_159_98_l132_132921


namespace arithmetic_mean_pq_l132_132821

variable (p q r : ℝ)

-- Definitions from conditions
def condition1 := (p + q) / 2 = 10
def condition2 := (q + r) / 2 = 26
def condition3 := r - p = 32

-- Theorem statement
theorem arithmetic_mean_pq : condition1 p q → condition2 q r → condition3 p r → (p + q) / 2 = 10 :=
by
  intros h1 h2 h3
  exact h1

end arithmetic_mean_pq_l132_132821


namespace evaluate_g_at_neg2_l132_132536

def g (x : ℝ) : ℝ := 3 * x^4 - 20 * x^3 + 35 * x^2 - 28 * x - 84

theorem evaluate_g_at_neg2 : g (-2) = 320 := by
  sorry

end evaluate_g_at_neg2_l132_132536


namespace solve_y_l132_132196

theorem solve_y (y : ℤ) (h : 7 - y = 10) : y = -3 := by
  sorry

end solve_y_l132_132196


namespace length_of_AB_l132_132602

theorem length_of_AB :
  let ellipse := {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 = 1}
  let focus := (Real.sqrt 3, 0)
  let line := {p : ℝ × ℝ | p.2 = p.1 - Real.sqrt 3}
  ∃ A B : ℝ × ℝ, A ∈ ellipse ∧ B ∈ ellipse ∧ A ∈ line ∧ B ∈ line ∧
  (dist A B = 8 / 5) :=
by
  sorry

end length_of_AB_l132_132602


namespace floor_sqrt_80_eq_8_l132_132730

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end floor_sqrt_80_eq_8_l132_132730


namespace range_of_x_l132_132046

open Real

def p (x : ℝ) : Prop := log (x^2 - 2 * x - 2) ≥ 0
def q (x : ℝ) : Prop := 0 < x ∧ x < 4
def not_p (x : ℝ) : Prop := -1 < x ∧ x < 3
def not_q (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 4

theorem range_of_x (x : ℝ) :
  (¬ p x ∧ ¬ q x ∧ (p x ∨ q x)) →
  x ≤ -1 ∨ (0 < x ∧ x < 3) ∨ x ≥ 4 :=
sorry

end range_of_x_l132_132046


namespace find_a5_l132_132338

-- Define the problem conditions within Lean
variable {a : ℕ → ℝ}
variable {q : ℝ}

-- Conditions of the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q
def positive_terms (a : ℕ → ℝ) := ∀ n, 0 < a n
def condition1 (a : ℕ → ℝ) := a 1 * a 3 = 4
def condition2 (a : ℕ → ℝ) := a 7 * a 9 = 25

-- Proposition to prove
theorem find_a5 :
  geometric_sequence a q →
  positive_terms a →
  condition1 a →
  condition2 a →
  a 5 = Real.sqrt 10 :=
by
  sorry

end find_a5_l132_132338


namespace quadratic_root_relationship_l132_132346

noncomputable def roots_of_quadratic (a b c: ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : Prop :=
  b / c = 27

theorem quadratic_root_relationship (a b c : ℚ) (h_nonzero: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_root_relation: ∀ (s₁ s₂ : ℚ), s₁ + s₂ = -c ∧ s₁ * s₂ = a → (3 * s₁) + (3 * s₂) = -a ∧ (3 * s₁) * (3 * s₂) = b) : 
  roots_of_quadratic a b c h_nonzero h_root_relation := 
by 
  sorry

end quadratic_root_relationship_l132_132346


namespace fields_fertilized_in_25_days_l132_132213

-- Definitions from conditions
def fertilizer_per_horse_per_day : ℕ := 5
def number_of_horses : ℕ := 80
def fertilizer_needed_per_acre : ℕ := 400
def number_of_acres : ℕ := 20
def acres_fertilized_per_day : ℕ := 4

-- Total fertilizer produced per day
def total_fertilizer_per_day : ℕ := fertilizer_per_horse_per_day * number_of_horses

-- Total fertilizer needed
def total_fertilizer_needed : ℕ := fertilizer_needed_per_acre * number_of_acres

-- Days to collect enough fertilizer
def days_to_collect_fertilizer : ℕ := total_fertilizer_needed / total_fertilizer_per_day

-- Days to spread fertilizer
def days_to_spread_fertilizer : ℕ := number_of_acres / acres_fertilized_per_day

-- Calculate the total time until all fields are fertilized
def total_days : ℕ := days_to_collect_fertilizer + days_to_spread_fertilizer

-- Theorem statement
theorem fields_fertilized_in_25_days : total_days = 25 :=
by
  sorry

end fields_fertilized_in_25_days_l132_132213


namespace parabola_focus_l132_132599

theorem parabola_focus (x y : ℝ) : (y = x^2 / 8) → (y = x^2 / 8) ∧ (∃ p, p = (0, 2)) :=
by
  sorry

end parabola_focus_l132_132599


namespace avg_of_first_three_groups_prob_of_inspection_l132_132476
  
-- Define the given frequency distribution as constants
def freq_40_50 : ℝ := 0.04
def freq_50_60 : ℝ := 0.06
def freq_60_70 : ℝ := 0.22
def freq_70_80 : ℝ := 0.28
def freq_80_90 : ℝ := 0.22
def freq_90_100 : ℝ := 0.18

-- Calculate the midpoint values for the first three groups
def mid_40_50 : ℝ := 45
def mid_50_60 : ℝ := 55
def mid_60_70 : ℝ := 65

-- Define the probabilities interpreted from the distributions
def prob_poor : ℝ := freq_40_50 + freq_50_60
def prob_avg : ℝ := freq_60_70 + freq_70_80
def prob_good : ℝ := freq_80_90 + freq_90_100

-- Define the main theorem for the average score of the first three groups
theorem avg_of_first_three_groups :
  (mid_40_50 * freq_40_50 + mid_50_60 * freq_50_60 + mid_60_70 * freq_60_70) /
  (freq_40_50 + freq_50_60 + freq_60_70) = 60.625 := 
by { sorry }

-- Define the theorem for the probability of inspection
theorem prob_of_inspection :
  1 - (3 * (prob_good * prob_avg * prob_avg) + 3 * (prob_avg * prob_avg * prob_good) + (prob_good * prob_good * prob_good)) = 0.396 :=
by { sorry }

end avg_of_first_three_groups_prob_of_inspection_l132_132476


namespace compare_sums_l132_132498

open Classical

-- Define the necessary sequences and their properties
variable {α : Type*} [LinearOrderedField α]

-- Arithmetic Sequence {a_n}
noncomputable def arith_seq (a_1 d : α) : ℕ → α
| 0     => a_1
| (n+1) => (arith_seq a_1 d n) + d

-- Geometric Sequence {b_n}
noncomputable def geom_seq (b_1 q : α) : ℕ → α
| 0     => b_1
| (n+1) => (geom_seq b_1 q n) * q

-- Sum of the first n terms of an arithmetic sequence
noncomputable def arith_sum (a_1 d : α) (n : ℕ) : α :=
(n + 1) * (a_1 + arith_seq a_1 d n) / 2

-- Sum of the first n terms of a geometric sequence
noncomputable def geom_sum (b_1 q : α) (n : ℕ) : α :=
if q = 1 then (n + 1) * b_1
else b_1 * (1 - q^(n + 1)) / (1 - q)

theorem compare_sums
  (a_1 b_1 : α) (d q : α)
  (hd : d ≠ 0) (hq : q > 0) (hq1 : q ≠ 1)
  (h_eq1 : a_1 = b_1)
  (h_eq2 : arith_seq a_1 d 1011 = geom_seq b_1 q 1011) :
  arith_sum a_1 d 2022 < geom_sum b_1 q 2022 :=
sorry

end compare_sums_l132_132498


namespace quadratic_has_real_roots_iff_l132_132855

theorem quadratic_has_real_roots_iff (k : ℝ) (hk : k ≠ 0) :
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ k ≤ 1 / 4 :=
by
  sorry

end quadratic_has_real_roots_iff_l132_132855


namespace average_after_17th_inning_l132_132202

theorem average_after_17th_inning (A : ℝ) (total_runs_16th_inning : ℝ) 
  (average_before_17th : A * 16 = total_runs_16th_inning) 
  (increased_average_by_3 : (total_runs_16th_inning + 83) / 17 = A + 3) :
  (A + 3) = 35 := 
sorry

end average_after_17th_inning_l132_132202


namespace new_class_mean_l132_132969

theorem new_class_mean {X Y : ℕ} {mean_a mean_b : ℚ}
  (hx : X = 30) (hy : Y = 6) 
  (hmean_a : mean_a = 72) (hmean_b : mean_b = 78) :
  (X * mean_a + Y * mean_b) / (X + Y) = 73 := 
by 
  sorry

end new_class_mean_l132_132969


namespace initial_card_distribution_l132_132172

variables {A B C D : ℕ}

theorem initial_card_distribution 
  (total_cards : A + B + C + D = 32)
  (alfred_final : ∀ c, c = A → ((c / 2) + (c / 2)) + B + C + D = 8)
  (bruno_final : ∀ c, c = B → ((c / 2) + (c / 2)) + A + C + D = 8)
  (christof_final : ∀ c, c = C → ((c / 2) + (c / 2)) + A + B + D = 8)
  : A = 7 ∧ B = 7 ∧ C = 10 ∧ D = 8 :=
by sorry

end initial_card_distribution_l132_132172


namespace scalene_triangle_angle_obtuse_l132_132655

theorem scalene_triangle_angle_obtuse (a b c : ℝ) 
  (h_scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_longest : a > b ∧ a > c)
  (h_obtuse_angle : a^2 > b^2 + c^2) : 
  ∃ A : ℝ, A = (Real.pi / 2) ∧ (b^2 + c^2 - a^2) / (2 * b * c) < 0 := 
sorry

end scalene_triangle_angle_obtuse_l132_132655


namespace intersection_A_B_l132_132693

def A : Set ℝ := { x | |x - 1| < 2 }
def B : Set ℝ := { x | Real.log x / Real.log 2 ≤ 1 }

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 2} := 
sorry

end intersection_A_B_l132_132693


namespace min_value_reciprocal_sum_l132_132420

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) : 
  (1/m + 1/n) = 4 :=
by
  sorry

end min_value_reciprocal_sum_l132_132420


namespace sampled_students_within_interval_l132_132500

/-- Define the conditions for the student's problem --/
def student_count : ℕ := 1221
def sampled_students : ℕ := 37
def sampling_interval : ℕ := student_count / sampled_students
def interval_lower_bound : ℕ := 496
def interval_upper_bound : ℕ := 825
def interval_range : ℕ := interval_upper_bound - interval_lower_bound + 1

/-- State the goal within the above conditions --/
theorem sampled_students_within_interval :
  interval_range / sampling_interval = 10 :=
sorry

end sampled_students_within_interval_l132_132500


namespace cistern_fill_time_l132_132448

/--
  A cistern can be filled by tap A in 4 hours,
  emptied by tap B in 6 hours,
  and filled by tap C in 3 hours.
  If all the taps are opened simultaneously,
  then the cistern will be filled in exactly 2.4 hours.
-/
theorem cistern_fill_time :
  let rate_A := 1 / 4
  let rate_B := -1 / 6
  let rate_C := 1 / 3
  let combined_rate := rate_A + rate_B + rate_C
  let fill_time := 1 / combined_rate
  fill_time = 2.4 := by
  sorry

end cistern_fill_time_l132_132448


namespace speed_of_stream_l132_132501

theorem speed_of_stream (b s : ℝ) (h1 : 75 = 5 * (b + s)) (h2 : 45 = 5 * (b - s)) : s = 3 :=
by
  have eq1 : b + s = 15 := by linarith [h1]
  have eq2 : b - s = 9 := by linarith [h2]
  have b_val : b = 12 := by linarith [eq1, eq2]
  linarith 

end speed_of_stream_l132_132501


namespace rectangle_symmetry_l132_132140

-- Define basic geometric terms and the notion of symmetry
structure Rectangle where
  length : ℝ
  width : ℝ
  (length_pos : 0 < length)
  (width_pos : 0 < width)

def is_axes_of_symmetry (r : Rectangle) (n : ℕ) : Prop :=
  -- A hypothetical function that determines whether a rectangle r has n axes of symmetry
  sorry

theorem rectangle_symmetry (r : Rectangle) : is_axes_of_symmetry r 2 := 
  -- This theorem states that a rectangle has exactly 2 axes of symmetry
  sorry

end rectangle_symmetry_l132_132140


namespace alpha_minus_beta_l132_132417

theorem alpha_minus_beta {α β : ℝ} (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
    (h_cos_alpha : Real.cos α = 2 * Real.sqrt 5 / 5) 
    (h_cos_beta : Real.cos β = Real.sqrt 10 / 10) : 
    α - β = -π / 4 := 
sorry

end alpha_minus_beta_l132_132417


namespace lcm_of_fractions_l132_132400

-- Definitions based on the problem's conditions
def numerators : List ℕ := [7, 8, 3, 5, 13, 15, 22, 27]
def denominators : List ℕ := [10, 9, 8, 12, 14, 100, 45, 35]

-- LCM and GCD functions for lists of natural numbers
def list_lcm (l : List ℕ) : ℕ := l.foldr lcm 1
def list_gcd (l : List ℕ) : ℕ := l.foldr gcd 0

-- Main proposition
theorem lcm_of_fractions : list_lcm numerators / list_gcd denominators = 13860 :=
by {
  -- to be proven
  sorry
}

end lcm_of_fractions_l132_132400


namespace value_range_for_positive_roots_l132_132690

theorem value_range_for_positive_roots (a : ℝ) :
  (∀ x : ℝ, x > 0 → a * |x| + |x + a| = 0) ↔ (-1 < a ∧ a < 0) :=
by
  sorry

end value_range_for_positive_roots_l132_132690


namespace probability_spinner_lands_in_shaded_region_l132_132067

theorem probability_spinner_lands_in_shaded_region :
  let total_regions := 4
  let shaded_regions := 3
  (shaded_regions: ℝ) / total_regions = 3 / 4 :=
by
  let total_regions := 4
  let shaded_regions := 3
  sorry

end probability_spinner_lands_in_shaded_region_l132_132067


namespace rectangle_semi_perimeter_l132_132271

variables (BC AC AM x y : ℝ)

theorem rectangle_semi_perimeter (hBC : BC = 5) (hAC : AC = 12) (hAM : AM = x)
  (hMN_AC : ∀ (MN : ℝ), MN = 5 / 12 * AM)
  (hNP_BC : ∀ (NP : ℝ), NP = AC - AM)
  (hy_def : y = (5 / 12 * x) + (12 - x)) :
  y = (144 - 7 * x) / 12 :=
sorry

end rectangle_semi_perimeter_l132_132271


namespace set_intersection_example_l132_132175

theorem set_intersection_example :
  let A := { y | ∃ x, y = Real.log x / Real.log 2 ∧ x ≥ 3 }
  let B := { x | x^2 - 4 * x + 3 = 0 }
  A ∩ B = {3} :=
by
  sorry

end set_intersection_example_l132_132175


namespace copy_pages_l132_132343

theorem copy_pages (total_cents : ℕ) (cost_per_page : ℕ) (h1 : total_cents = 1500) (h2 : cost_per_page = 5) : 
  (total_cents / cost_per_page = 300) :=
sorry

end copy_pages_l132_132343


namespace box_dimensions_l132_132676

theorem box_dimensions (x : ℝ) (bow_length_top bow_length_side : ℝ)
  (h1 : bow_length_top = 156 - 6 * x)
  (h2 : bow_length_side = 178 - 7 * x)
  (h_eq : bow_length_top = bow_length_side) :
  x = 22 :=
by sorry

end box_dimensions_l132_132676


namespace quadratic_inequality_always_positive_l132_132605

theorem quadratic_inequality_always_positive (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
by sorry

end quadratic_inequality_always_positive_l132_132605


namespace not_unique_y_20_paise_l132_132108

theorem not_unique_y_20_paise (x y z w : ℕ) : 
  x + y + z + w = 750 → 10 * x + 20 * y + 50 * z + 100 * w = 27500 → ∃ (y₁ y₂ : ℕ), y₁ ≠ y₂ :=
by 
  intro h1 h2
  -- Without additional constraints on x, y, z, w,
  -- suppose that there are at least two different solutions satisfying both equations,
  -- demonstrating the non-uniqueness of y.
  sorry

end not_unique_y_20_paise_l132_132108


namespace zero_point_exists_between_2_and_3_l132_132020

noncomputable def f (x : ℝ) := 2^(x-1) + x - 5

theorem zero_point_exists_between_2_and_3 :
  ∃ x₀ ∈ Set.Ioo (2 : ℝ) 3, f x₀ = 0 :=
sorry

end zero_point_exists_between_2_and_3_l132_132020


namespace hamburgers_total_l132_132143

theorem hamburgers_total (initial_hamburgers : ℝ) (additional_hamburgers : ℝ) (h₁ : initial_hamburgers = 9.0) (h₂ : additional_hamburgers = 3.0) : initial_hamburgers + additional_hamburgers = 12.0 :=
by
  rw [h₁, h₂]
  norm_num

end hamburgers_total_l132_132143


namespace train_speed_l132_132051

theorem train_speed (L V : ℝ) (h1 : L = V * 10) (h2 : L + 500 = V * 35) : V = 20 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end train_speed_l132_132051


namespace fraction_identity_l132_132733

theorem fraction_identity (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : ((m + n) : ℚ) / n = 10 / 7 := 
sorry

end fraction_identity_l132_132733


namespace wind_speed_l132_132529

theorem wind_speed (w : ℝ) (h : 420 / (253 + w) = 350 / (253 - w)) : w = 23 :=
by
  sorry

end wind_speed_l132_132529


namespace find_number_of_students_l132_132842

theorem find_number_of_students (N T : ℕ) 
  (avg_mark_all : T = 80 * N) 
  (avg_mark_exclude : (T - 150) / (N - 5) = 90) : 
  N = 30 := by
  sorry

end find_number_of_students_l132_132842


namespace negation_of_proposition_l132_132398

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*x > 0) ↔ (∀ x : ℝ, x^2 + 2*x ≤ 0) :=
sorry

end negation_of_proposition_l132_132398


namespace inappropriate_survey_method_l132_132646

/-
Parameters:
- A: Using a sampling survey method to understand the water-saving awareness of middle school students in the city (appropriate).
- B: Investigating the capital city to understand the environmental pollution situation of the entire province (inappropriate due to lack of representativeness).
- C: Investigating the audience's evaluation of a movie by surveying those seated in odd-numbered seats (appropriate).
- D: Using a census method to understand the compliance rate of pilots' vision (appropriate).
-/

theorem inappropriate_survey_method (A B C D : Prop) 
  (hA : A = true)
  (hB : B = false)  -- This condition defines B as inappropriate
  (hC : C = true)
  (hD : D = true) : B = false :=
sorry

end inappropriate_survey_method_l132_132646


namespace ways_to_change_12_dollars_into_nickels_and_quarters_l132_132718

theorem ways_to_change_12_dollars_into_nickels_and_quarters :
  ∃ n q : ℕ, 5 * n + 25 * q = 1200 ∧ n > 0 ∧ q > 0 ∧ ∀ q', (q' ≥ 1 ∧ q' ≤ 47) ↔ (n = 240 - 5 * q') :=
by
  sorry

end ways_to_change_12_dollars_into_nickels_and_quarters_l132_132718


namespace exists_x_y_not_divisible_by_3_l132_132572

theorem exists_x_y_not_divisible_by_3 (k : ℕ) (h_pos : 0 < k) :
  ∃ x y : ℤ, (x^2 + 2 * y^2 = 3^k) ∧ (x % 3 ≠ 0) ∧ (y % 3 ≠ 0) := 
sorry

end exists_x_y_not_divisible_by_3_l132_132572


namespace percentage_increase_bears_with_assistant_l132_132212

theorem percentage_increase_bears_with_assistant
  (B H : ℝ)
  (h_positive_hours : H > 0)
  (h_positive_bears : B > 0)
  (hours_with_assistant : ℝ := 0.90 * H)
  (rate_increase : ℝ := 2 * B / H) :
  ((rate_increase * hours_with_assistant) - B) / B * 100 = 80 := by
  -- This is the statement for the given problem.
  sorry

end percentage_increase_bears_with_assistant_l132_132212


namespace abs_neg_eq_iff_nonpos_l132_132446

theorem abs_neg_eq_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 :=
by sorry

end abs_neg_eq_iff_nonpos_l132_132446


namespace quadratic_one_solution_l132_132490

theorem quadratic_one_solution (p : ℝ) : (3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0) 
  → ((-6) ^ 2 - 4 * 3 * p = 0) 
  → p = 3 :=
by
  intro h1 h2
  have h1' : 3 * (1 : ℝ) ^ 2 - 6 * (1 : ℝ) + p = 0 := h1
  have h2' : (-6) ^ 2 - 4 * 3 * p = 0 := h2
  sorry

end quadratic_one_solution_l132_132490


namespace parallel_lines_direction_vector_l132_132545

theorem parallel_lines_direction_vector (k : ℝ) :
  (∃ c : ℝ, (5, -3) = (c * -2, c * k)) ↔ k = 6 / 5 :=
by sorry

end parallel_lines_direction_vector_l132_132545


namespace total_grapes_is_157_l132_132458

def number_of_grapes_in_robs_bowl : ℕ := 25

def number_of_grapes_in_allies_bowl : ℕ :=
  number_of_grapes_in_robs_bowl + 5

def number_of_grapes_in_allyns_bowl : ℕ :=
  2 * number_of_grapes_in_allies_bowl - 2

def number_of_grapes_in_sams_bowl : ℕ :=
  (number_of_grapes_in_allies_bowl + number_of_grapes_in_allyns_bowl) / 2

def total_number_of_grapes : ℕ :=
  number_of_grapes_in_robs_bowl +
  number_of_grapes_in_allies_bowl +
  number_of_grapes_in_allyns_bowl +
  number_of_grapes_in_sams_bowl

theorem total_grapes_is_157 : total_number_of_grapes = 157 :=
  sorry

end total_grapes_is_157_l132_132458


namespace dog_bones_initial_count_l132_132925

theorem dog_bones_initial_count (buried : ℝ) (final : ℝ) : buried = 367.5 → final = -860 → (buried + (final + 367.5) + 860) = 367.5 :=
by
  intros h1 h2
  sorry

end dog_bones_initial_count_l132_132925


namespace fraction_of_arith_geo_seq_l132_132779

theorem fraction_of_arith_geo_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_seq_arith : ∀ n, a (n+1) = a n + d)
  (h_seq_geo : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
by
  sorry

end fraction_of_arith_geo_seq_l132_132779


namespace evaluate_expression_l132_132615

theorem evaluate_expression : (723 * 723) - (722 * 724) = 1 :=
by
  sorry

end evaluate_expression_l132_132615


namespace nancy_crayons_l132_132831

theorem nancy_crayons (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat) 
  (h1 : packs = 41) (h2 : crayons_per_pack = 15) (h3 : total_crayons = packs * crayons_per_pack) : 
  total_crayons = 615 := by
  sorry

end nancy_crayons_l132_132831


namespace six_digit_divisibility_by_37_l132_132254

theorem six_digit_divisibility_by_37 (a b c d e f : ℕ) (H : (100 * a + 10 * b + c + 100 * d + 10 * e + f) % 37 = 0) : 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) % 37 = 0 := 
sorry

end six_digit_divisibility_by_37_l132_132254


namespace jackson_holidays_l132_132484

theorem jackson_holidays (holidays_per_month : ℕ) (months_per_year : ℕ) (total_holidays : ℕ) :
  holidays_per_month = 3 → months_per_year = 12 → total_holidays = holidays_per_month * months_per_year →
  total_holidays = 36 :=
by
  intros
  sorry

end jackson_holidays_l132_132484


namespace remainder_when_summed_divided_by_15_l132_132807

theorem remainder_when_summed_divided_by_15 (k j : ℤ) (x y : ℤ)
  (hx : x = 60 * k + 47)
  (hy : y = 45 * j + 26) :
  (x + y) % 15 = 13 := 
sorry

end remainder_when_summed_divided_by_15_l132_132807


namespace problem_solution_l132_132641

def f (x m : ℝ) : ℝ :=
  3 * x ^ 2 + m * (m - 6) * x + 5

theorem problem_solution (m n : ℝ) :
  (f 1 m > 0) ∧ (∀ x : ℝ, -1 < x ∧ x < 4 → f x m < n) ↔ (m = 3 ∧ n = 17) :=
by sorry

end problem_solution_l132_132641


namespace option_D_min_value_is_2_l132_132998

noncomputable def funcD (x : ℝ) : ℝ :=
  (x^2 + 2) / Real.sqrt (x^2 + 1)

theorem option_D_min_value_is_2 :
  ∃ x : ℝ, funcD x = 2 :=
sorry

end option_D_min_value_is_2_l132_132998


namespace prisoners_can_be_freed_l132_132494

-- Condition: We have 100 prisoners and 100 drawers.
def prisoners : Nat := 100
def drawers : Nat := 100

-- Predicate to represent the strategy
def successful_strategy (strategy: (Fin prisoners) → (Fin drawers) → Bool) : Bool :=
  -- We use a hypothetical strategy function to model this
  (true) -- Placeholder for the actual strategy computation

-- Statement: Prove that there exists a strategy where all prisoners finding their names has a probability greater than 30%.
theorem prisoners_can_be_freed :
  ∃ strategy: (Fin prisoners) → (Fin drawers) → Bool, 
    (successful_strategy strategy) ∧ (0.3118 > 0.3) :=
sorry

end prisoners_can_be_freed_l132_132494


namespace IvanPetrovich_daily_lessons_and_charity_l132_132165

def IvanPetrovichConditions (L k : ℕ) : Prop :=
  24 = 8 + 3*L + k ∧
  3000 * L * 21 + 14000 = 70000 + (7000 * k / 3)

theorem IvanPetrovich_daily_lessons_and_charity
  (L k : ℕ) (h : IvanPetrovichConditions L k) :
  L = 2 ∧ 7000 * k / 3 = 70000 := 
by
  sorry

end IvanPetrovich_daily_lessons_and_charity_l132_132165


namespace prime_power_divides_binomial_l132_132334

theorem prime_power_divides_binomial {p n k α : ℕ} (hp : Nat.Prime p) 
  (h : p^α ∣ Nat.choose n k) : p^α ≤ n := 
sorry

end prime_power_divides_binomial_l132_132334


namespace optimal_solution_for_z_is_1_1_l132_132197

def x := 1
def y := 1
def z (x y : ℝ) := 2 * x + y

theorem optimal_solution_for_z_is_1_1 :
  ∀ (x y : ℝ), z x y ≥ z 1 1 := 
by
  simp [z]
  sorry

end optimal_solution_for_z_is_1_1_l132_132197


namespace varphi_le_one_varphi_l132_132794

noncomputable def f (a x : ℝ) := -a * Real.log x

-- Definition of the minimum value function φ for a > 0
noncomputable def varphi (a : ℝ) := -a * Real.log a

theorem varphi_le_one (a : ℝ) (h : 0 < a) : varphi a ≤ 1 := 
by sorry

theorem varphi'_le (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
    (1 - Real.log a) ≤ (1 - Real.log b) := 
by sorry

end varphi_le_one_varphi_l132_132794


namespace factor_expression_l132_132558

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) :=
by
  sorry

end factor_expression_l132_132558


namespace feet_of_pipe_per_bolt_l132_132151

-- Definition of the initial conditions
def total_pipe_length := 40 -- total feet of pipe
def washers_per_bolt := 2
def initial_washers := 20
def remaining_washers := 4

-- The proof statement
theorem feet_of_pipe_per_bolt :
  ∀ (total_pipe_length washers_per_bolt initial_washers remaining_washers : ℕ),
  initial_washers - remaining_washers = 16 → -- 16 washers used
  16 / washers_per_bolt = 8 → -- 8 bolts used
  total_pipe_length / 8 = 5 :=
by
  intros
  sorry

end feet_of_pipe_per_bolt_l132_132151


namespace puppy_weight_l132_132533

theorem puppy_weight (a b c : ℕ) 
  (h1 : a + b + c = 24) 
  (h2 : a + c = 2 * b) 
  (h3 : a + b = c) : 
  a = 4 :=
sorry

end puppy_weight_l132_132533


namespace sum_of_abc_is_40_l132_132469

theorem sum_of_abc_is_40 (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h1 : a * b + c = 55) (h2 : b * c + a = 55) (h3 : c * a + b = 55) :
    a + b + c = 40 :=
by
  sorry

end sum_of_abc_is_40_l132_132469


namespace problem_statement_l132_132583

variable {a : ℕ+ → ℝ} 

theorem problem_statement (h : ∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) :
  (∀ n : ℕ+, a (n + 1) < a n) ∧ -- Sequence is decreasing (original proposition)
  (∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n)) ∧ -- Inverse
  ((∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n) → (∀ n : ℕ+, a (n + 1) < a n)) ∧ -- Converse
  ((∀ (a : ℕ+ → ℝ), (∀ n : ℕ+, a (n + 1) < a n) → (∀ n : ℕ+, (a n + a (n + 1)) / 2 < a n))) -- Contrapositive
:= by
  sorry

end problem_statement_l132_132583


namespace red_paint_cans_l132_132061

theorem red_paint_cans (total_cans : ℕ) (ratio_red_blue : ℕ) (ratio_blue : ℕ) (h_ratio : ratio_red_blue = 4) (h_blue : ratio_blue = 1) (h_total_cans : total_cans = 50) : 
  (total_cans * ratio_red_blue) / (ratio_red_blue + ratio_blue) = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end red_paint_cans_l132_132061


namespace hyperbola_eccentricity_l132_132526

-- Definition of the parabola C1: y^2 = 2px with p > 0.
def parabola (p : ℝ) (p_pos : 0 < p) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Definition of the hyperbola C2: x^2 / a^2 - y^2 / b^2 = 1 with a > 0 and b > 0.
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

-- Definition of having a common focus F at (p / 2, 0).
def common_focus (p a b c : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) : Prop := 
  c = p / 2 ∧ c^2 = a^2 + b^2

-- Definition for points A and B on parabola C1 and point M on hyperbola C2.
def points_A_B_M (c a b : ℝ) (x1 y1 x2 y2 yM : ℝ) : Prop := 
  x1 = c ∧ y1 = 2 * c ∧ x2 = c ∧ y2 = -2 * c ∧ yM = b^2 / a

-- Condition for OM, OA, and OB relation and mn = 1/8.
def OM_OA_OB_relation (m n : ℝ) : Prop := 
  m * n = 1 / 8

-- Theorem statement: Given the conditions, the eccentricity of hyperbola C2 is √6 + √2 / 2.
theorem hyperbola_eccentricity (p a b c m n : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) :
  parabola p p_pos c (2 * c) → 
  hyperbola a b a_pos b_pos c (b^2 / a) → 
  common_focus p a b c p_pos a_pos b_pos →
  points_A_B_M c a b c (2 * c) c (-2 * c) (b^2 / a) →
  OM_OA_OB_relation m n → 
  m * n = 1 / 8 →
  ∃ e : ℝ, e = (Real.sqrt 6 + Real.sqrt 2) / 2 :=
sorry

end hyperbola_eccentricity_l132_132526


namespace dasha_rectangle_problem_l132_132910

variables (a b c : ℕ)

theorem dasha_rectangle_problem
  (h1 : a > 0) 
  (h2 : a * (b + c) + a * (b - a) + a^2 + a * (c - a) = 43) 
  : (a = 1 ∧ b + c = 22) ∨ (a = 43 ∧ b + c = 2) :=
by
  sorry

end dasha_rectangle_problem_l132_132910


namespace find_x_l132_132601

theorem find_x (x y : ℕ) (h1 : y = 30) (h2 : x / y = 5 / 2) : x = 75 := by
  sorry

end find_x_l132_132601


namespace cookies_total_is_60_l132_132248

def Mona_cookies : ℕ := 20
def Jasmine_cookies : ℕ := Mona_cookies - 5
def Rachel_cookies : ℕ := Jasmine_cookies + 10
def Total_cookies : ℕ := Mona_cookies + Jasmine_cookies + Rachel_cookies

theorem cookies_total_is_60 : Total_cookies = 60 := by
  sorry

end cookies_total_is_60_l132_132248


namespace adult_ticket_cost_given_conditions_l132_132528

variables (C A S : ℕ)

def cost_relationships : Prop :=
  A = C + 10 ∧ S = A - 5 ∧ (5 * C + 2 * A + 2 * S + (S - 3) = 212)

theorem adult_ticket_cost_given_conditions :
  cost_relationships C A S → A = 28 :=
by
  intros h
  have h1 : A = C + 10 := h.left
  have h2 : S = A - 5 := h.right.left
  have h3 : (5 * C + 2 * A + 2 * S + (S - 3) = 212) := h.right.right
  sorry

end adult_ticket_cost_given_conditions_l132_132528


namespace max_value_of_f_l132_132094

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f (Real.exp 1) = 1 / Real.exp 1 := by
  sorry

end max_value_of_f_l132_132094


namespace price_reductions_l132_132760

theorem price_reductions (a : ℝ) : 18400 * (1 - a / 100)^2 = 16000 :=
sorry

end price_reductions_l132_132760


namespace perfect_squares_m_l132_132909

theorem perfect_squares_m (m : ℕ) (hm_pos : m > 0) (hm_min4_square : ∃ a : ℕ, m - 4 = a^2) (hm_plus5_square : ∃ b : ℕ, m + 5 = b^2) : m = 20 ∨ m = 4 :=
by
  sorry

end perfect_squares_m_l132_132909


namespace regular_pentagonal_pyramid_angle_l132_132035

noncomputable def angle_between_slant_height_and_non_intersecting_edge (base_edge_slant_height : ℝ) : ℝ :=
  -- Assuming the base edge and slant height are given as input and equal
  if base_edge_slant_height > 0 then 36 else 0

theorem regular_pentagonal_pyramid_angle
  (base_edge_slant_height : ℝ)
  (h : base_edge_slant_height > 0) :
  angle_between_slant_height_and_non_intersecting_edge base_edge_slant_height = 36 :=
by
  -- omitted proof steps
  sorry

end regular_pentagonal_pyramid_angle_l132_132035


namespace initial_cabinets_l132_132688

theorem initial_cabinets (C : ℤ) (h1 : 26 = C + 6 * C + 5) : C = 3 := 
by 
  sorry

end initial_cabinets_l132_132688


namespace dice_probability_sum_three_l132_132851

theorem dice_probability_sum_three (total_outcomes : ℕ := 36) (favorable_outcomes : ℕ := 2) :
  favorable_outcomes / total_outcomes = 1 / 18 :=
by
  sorry

end dice_probability_sum_three_l132_132851


namespace sequence_a_n_derived_conditions_derived_sequence_is_even_l132_132649

-- Statement of the first problem
theorem sequence_a_n_derived_conditions (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : b 1 = 5 ∧ b 2 = -2 ∧ b 3 = 7 ∧ b 4 = 2):
  a 1 = 2 ∧ a 2 = 1 ∧ a 3 = 4 ∧ a 4 = 5 :=
sorry

-- Statement of the second problem
theorem derived_sequence_is_even (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : c 1 = b n)
  (h4 : ∀ k, 2 ≤ k ∧ k ≤ n → c k = b (k - 1) + b k - c (k - 1)):
  ∀ i, 1 ≤ i ∧ i ≤ n → c i = a i :=
sorry

end sequence_a_n_derived_conditions_derived_sequence_is_even_l132_132649


namespace students_disliked_menu_l132_132867

theorem students_disliked_menu (total_students liked_students : ℕ) (h1 : total_students = 400) (h2 : liked_students = 235) : total_students - liked_students = 165 :=
by 
  sorry

end students_disliked_menu_l132_132867


namespace grain_milling_l132_132994

theorem grain_milling (A : ℚ) (h1 : 0.9 * A = 100) : A = 111 + 1 / 9 :=
by
  sorry

end grain_milling_l132_132994


namespace max_value_of_expression_l132_132382

theorem max_value_of_expression (x y z : ℤ) 
  (h1 : x * y + x + y = 20) 
  (h2 : y * z + y + z = 6) 
  (h3 : x * z + x + z = 2) : 
  x^2 + y^2 + z^2 ≤ 84 :=
sorry

end max_value_of_expression_l132_132382


namespace solution_set_of_inequality_l132_132858

theorem solution_set_of_inequality (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := 
by
  sorry

end solution_set_of_inequality_l132_132858


namespace set_intersection_l132_132467

   -- Define set A
   def A : Set ℝ := {x : ℝ | (x - 3) / (x + 1) ≥ 0 }
   
   -- Define set B
   def B : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}

   -- Define the relative complement of A in the real numbers
   def complement_R (A : Set ℝ) : Set ℝ := {x : ℝ | ¬ (A x)}

   -- The main statement that needs to be proven
   theorem set_intersection :
     (complement_R A) ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by
     sorry
   
end set_intersection_l132_132467


namespace div_power_sub_one_l132_132568

theorem div_power_sub_one : 11 * 31 * 61 ∣ 20^15 - 1 := 
by
  sorry

end div_power_sub_one_l132_132568


namespace youngest_child_age_l132_132246

theorem youngest_child_age (x : ℕ) (h1 : Prime x)
  (h2 : Prime (x + 2))
  (h3 : Prime (x + 6))
  (h4 : Prime (x + 8))
  (h5 : Prime (x + 12))
  (h6 : Prime (x + 14)) :
  x = 5 := 
sorry

end youngest_child_age_l132_132246


namespace gcd_102_238_l132_132683

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l132_132683


namespace work_ratio_l132_132524

theorem work_ratio (M B : ℝ) 
  (h1 : 5 * (12 * M + 16 * B) = 1)
  (h2 : 4 * (13 * M + 24 * B) = 1) : 
  M / B = 2 := 
  sorry

end work_ratio_l132_132524


namespace condition_on_a_and_b_l132_132099

variable (x a b : ℝ)

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem condition_on_a_and_b
  (h1 : a > 0)
  (h2 : b > 0) :
  (∀ x : ℝ, |f x + 3| < a ↔ |x - 1| < b) ↔ (b^2 + 2*b + 3 ≤ a) :=
sorry

end condition_on_a_and_b_l132_132099


namespace store_profit_is_33_percent_l132_132850

noncomputable def store_profit (C : ℝ) : ℝ :=
  let initial_markup := 1.20 * C
  let new_year_markup := initial_markup + 0.25 * initial_markup
  let february_discount := new_year_markup * 0.92
  let shipping_cost := C * 1.05
  (february_discount - shipping_cost)

theorem store_profit_is_33_percent (C : ℝ) : store_profit C = 0.33 * C :=
by
  sorry

end store_profit_is_33_percent_l132_132850


namespace apple_and_cherry_pies_total_l132_132492

-- Given conditions state that:
def apple_pies : ℕ := 6
def cherry_pies : ℕ := 5

-- We aim to prove that the total number of apple and cherry pies is 11.
theorem apple_and_cherry_pies_total : apple_pies + cherry_pies = 11 := by
  sorry

end apple_and_cherry_pies_total_l132_132492


namespace find_distance_to_school_l132_132361

variable (v d : ℝ)
variable (h_rush_hour : d = v * (1 / 2))
variable (h_no_traffic : d = (v + 20) * (1 / 4))

theorem find_distance_to_school (h_rush_hour : d = v * (1 / 2)) (h_no_traffic : d = (v + 20) * (1 / 4)) : d = 10 := by
  sorry

end find_distance_to_school_l132_132361


namespace sally_initial_cards_l132_132368

variable (initial_cards : ℕ)

-- Define the conditions
def cards_given := 41
def cards_lost := 20
def cards_now := 48

-- Define the proof problem
theorem sally_initial_cards :
  initial_cards + cards_given - cards_lost = cards_now → initial_cards = 27 :=
by
  intro h
  sorry

end sally_initial_cards_l132_132368


namespace total_weight_of_8_bags_total_sales_amount_of_qualified_products_l132_132546

-- Definitions
def deviations : List ℤ := [-6, -3, -2, 0, 1, 4, 5, -1]
def standard_weight_per_bag : ℤ := 450
def threshold : ℤ := 4
def price_per_bag : ℤ := 3

-- Part 1: Total weight of the 8 bags of laundry detergent
theorem total_weight_of_8_bags : 
  8 * standard_weight_per_bag + deviations.sum = 3598 := 
by
  sorry

-- Part 2: Total sales amount of qualified products
theorem total_sales_amount_of_qualified_products : 
  price_per_bag * (deviations.filter (fun x => abs x ≤ threshold)).length = 18 := 
by
  sorry

end total_weight_of_8_bags_total_sales_amount_of_qualified_products_l132_132546


namespace table_seating_problem_l132_132340

theorem table_seating_problem 
  (n : ℕ) 
  (label : ℕ → ℕ) 
  (h1 : label 31 = 31) 
  (h2 : label (31 - 17 + n) = 14) 
  (h3 : label (31 + 16) = 7) 
  : n = 41 :=
sorry

end table_seating_problem_l132_132340


namespace compute_K_l132_132582

theorem compute_K (P Q T N K : ℕ) (x y z : ℕ) 
  (hP : P * x + Q * y = z) 
  (hT : T * x + N * y = z)
  (hK : K * x = z)
  (h_unique : P > 0 ∧ Q > 0 ∧ T > 0 ∧ N > 0 ∧ K > 0) :
  K = (P * K - T * Q) / (N - Q) :=
by sorry

end compute_K_l132_132582


namespace bees_on_20th_day_l132_132854

-- Define the conditions
def initial_bees : ℕ := 1

def companions_per_bee : ℕ := 4

-- Define the total number of bees on day n
def total_bees (n : ℕ) : ℕ :=
  (initial_bees + companions_per_bee) ^ n

-- Statement to prove
theorem bees_on_20th_day : total_bees 20 = 5^20 :=
by
  -- The proof is omitted
  sorry

end bees_on_20th_day_l132_132854


namespace probability_of_neighboring_points_l132_132201

theorem probability_of_neighboring_points (n : ℕ) (h : n ≥ 3) : 
  (2 / (n - 1) : ℝ) = (n / (n * (n - 1) / 2) : ℝ) :=
by sorry

end probability_of_neighboring_points_l132_132201


namespace original_price_l132_132626

theorem original_price (a b x : ℝ) (h : (x - a) * 0.60 = b) : x = (5 / 3 * b) + a :=
  sorry

end original_price_l132_132626


namespace evaluate_expression_l132_132168

theorem evaluate_expression : 5^2 + 15 / 3 - (3 * 2)^2 = -6 := 
by
  sorry

end evaluate_expression_l132_132168


namespace jemma_total_grasshoppers_l132_132070

def number_of_grasshoppers_on_plant : Nat := 7
def number_of_dozen_baby_grasshoppers : Nat := 2
def number_in_a_dozen : Nat := 12

theorem jemma_total_grasshoppers :
  number_of_grasshoppers_on_plant + number_of_dozen_baby_grasshoppers * number_in_a_dozen = 31 := by
  sorry

end jemma_total_grasshoppers_l132_132070


namespace point_A_in_fourth_quadrant_l132_132409

-- Defining the coordinates of point A
def x_A : ℝ := 2
def y_A : ℝ := -3

-- Defining the property of the quadrant
def in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Proposition stating point A is in the fourth quadrant
theorem point_A_in_fourth_quadrant : in_fourth_quadrant x_A y_A :=
by
  sorry

end point_A_in_fourth_quadrant_l132_132409


namespace dot_product_u_v_l132_132885

def u : ℝ × ℝ × ℝ × ℝ := (4, -3, 5, -2)
def v : ℝ × ℝ × ℝ × ℝ := (-6, 1, 2, 3)

theorem dot_product_u_v : (4 * -6 + -3 * 1 + 5 * 2 + -2 * 3) = -23 := by
  sorry

end dot_product_u_v_l132_132885


namespace ex_sq_sum_l132_132357

theorem ex_sq_sum (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -1) : x^2 + y^2 = 11 :=
by
  sorry

end ex_sq_sum_l132_132357


namespace melanie_total_dimes_l132_132670

-- Definitions based on the problem conditions
def initial_dimes : ℕ := 7
def dad_dimes : ℕ := 8
def mom_dimes : ℕ := 4

def total_dimes : ℕ := initial_dimes + dad_dimes + mom_dimes

-- Proof statement based on the correct answer
theorem melanie_total_dimes : total_dimes = 19 := by 
  -- Proof here is omitted as per instructions
  sorry

end melanie_total_dimes_l132_132670


namespace arrange_order_l132_132531

noncomputable def a : Real := Real.sqrt 3
noncomputable def b : Real := Real.log 2 / Real.log 3
noncomputable def c : Real := Real.cos 2

theorem arrange_order : c < b ∧ b < a :=
by
  sorry

end arrange_order_l132_132531


namespace c_rent_share_l132_132460

-- Definitions based on conditions
def a_oxen := 10
def a_months := 7
def b_oxen := 12
def b_months := 5
def c_oxen := 15
def c_months := 3
def total_rent := 105

-- Calculate the shares in ox-months
def share_a := a_oxen * a_months
def share_b := b_oxen * b_months
def share_c := c_oxen * c_months

-- Calculate the total ox-months
def total_ox_months := share_a + share_b + share_c

-- Calculate the rent per ox-month
def rent_per_ox_month := total_rent / total_ox_months

-- Calculate the amount C should pay
def amount_c_should_pay := share_c * rent_per_ox_month

-- Prove the statement
theorem c_rent_share : amount_c_should_pay = 27 := by
  sorry

end c_rent_share_l132_132460


namespace alice_preferred_numbers_l132_132836

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

def is_not_multiple_of_3 (n : ℕ) : Prop :=
  ¬ (n % 3 = 0)

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def alice_pref_num (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧ is_multiple_of_7 n ∧ is_not_multiple_of_3 n ∧ is_prime (digit_sum n)

theorem alice_preferred_numbers :
  ∀ n, alice_pref_num n ↔ n = 119 ∨ n = 133 ∨ n = 140 := 
sorry

end alice_preferred_numbers_l132_132836


namespace rectangle_area_l132_132943

theorem rectangle_area (length : ℝ) (width_dm : ℝ) (width_m : ℝ) (h1 : length = 8) (h2 : width_dm = 50) (h3 : width_m = width_dm / 10) : 
  (length * width_m = 40) :=
by {
  sorry
}

end rectangle_area_l132_132943


namespace total_people_correct_current_people_correct_l132_132132

-- Define the conditions as constants
def morning_people : ℕ := 473
def noon_left : ℕ := 179
def afternoon_people : ℕ := 268

-- Define the total number of people
def total_people : ℕ := morning_people + afternoon_people

-- Define the current number of people in the amusement park
def current_people : ℕ := morning_people - noon_left + afternoon_people

-- Theorem proofs
theorem total_people_correct : total_people = 741 := by sorry
theorem current_people_correct : current_people = 562 := by sorry

end total_people_correct_current_people_correct_l132_132132


namespace correct_calculation_l132_132091

theorem correct_calculation (a : ℝ) :
  (¬ (a^2 + a^2 = a^4)) ∧ (¬ (a^2 * a^3 = a^6)) ∧ (¬ ((a + 1)^2 = a^2 + 1)) ∧ ((-a^2)^2 = a^4) :=
by
  sorry

end correct_calculation_l132_132091


namespace power_of_a_point_l132_132014

noncomputable def PA : ℝ := 4
noncomputable def PB : ℝ := 14 + 2 * Real.sqrt 13
noncomputable def PT : ℝ := PB - 8
noncomputable def AB : ℝ := PB - PA

theorem power_of_a_point (PA PB PT : ℝ) (h1 : PA = 4) (h2 : PB = 14 + 2 * Real.sqrt 13) (h3 : PT = PB - 8) : 
  PA * PB = PT * PT :=
by
  rw [h1, h2, h3]
  sorry

end power_of_a_point_l132_132014


namespace ralph_has_18_fewer_pictures_l132_132148

/-- Ralph has 58 pictures of wild animals. Derrick has 76 pictures of wild animals.
    Prove that Ralph has 18 fewer pictures of wild animals compared to Derrick. -/
theorem ralph_has_18_fewer_pictures :
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  76 - 58 = 18 :=
by
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  show 76 - 58 = 18
  sorry

end ralph_has_18_fewer_pictures_l132_132148


namespace image_of_element_2_l132_132235

-- Define the mapping f and conditions
def f (x : ℕ) : ℕ := 2 * x + 1

-- Define the element and its image using f
def element_in_set_A : ℕ := 2
def image_in_set_B : ℕ := f element_in_set_A

-- The theorem to prove
theorem image_of_element_2 : image_in_set_B = 5 :=
by
  -- This is where the proof would go, but we omit it with sorry
  sorry

end image_of_element_2_l132_132235


namespace range_of_a_l132_132627

theorem range_of_a (a : ℝ) : 
  (∀ (x : ℝ), |x + 3| - |x - 1| ≤ a ^ 2 - 3 * a) ↔ a ≤ -1 ∨ a ≥ 4 :=
by
  sorry

end range_of_a_l132_132627


namespace smallest_positive_period_f_max_min_f_on_interval_l132_132109

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem smallest_positive_period_f : 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ Real.pi) :=
sorry

theorem max_min_f_on_interval :
  let a := Real.pi / 4
  let b := 2 * Real.pi / 3
  ∃ M m, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M ∧ f x ≥ m) ∧ (M = 2) ∧ (m = -1) :=
sorry

end smallest_positive_period_f_max_min_f_on_interval_l132_132109


namespace regular_octahedron_has_4_pairs_l132_132883

noncomputable def regular_octahedron_parallel_edges : ℕ :=
  4

theorem regular_octahedron_has_4_pairs
  (h : true) : regular_octahedron_parallel_edges = 4 :=
by
  sorry

end regular_octahedron_has_4_pairs_l132_132883


namespace negation_of_at_most_one_odd_l132_132314

variable (a b c : ℕ)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def at_most_one_odd (a b c : ℕ) : Prop :=
  (is_odd a ∧ ¬is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ is_odd b ∧ ¬is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ is_odd c) ∨
  (¬is_odd a ∧ ¬is_odd b ∧ ¬is_odd c)

theorem negation_of_at_most_one_odd :
  ¬ at_most_one_odd a b c ↔
  ∃ x y, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ is_odd x ∧ is_odd y :=
sorry

end negation_of_at_most_one_odd_l132_132314


namespace evaluate_expression_l132_132077

theorem evaluate_expression (x : ℝ) (h1 : x^4 + 2 * x + 2 ≠ 0)
    (h2 : x^4 - 2 * x + 2 ≠ 0) :
    ( ( ( (x + 2) ^ 3 * (x^3 - 2 * x + 2) ^ 3 ) / ( ( x^4 + 2 * x + 2) ) ^ 3 ) ^ 3 * 
      ( ( (x - 2) ^ 3 * ( x^3 + 2 * x + 2 ) ^ 3 ) / ( ( x^4 - 2 * x + 2 ) ) ^ 3 ) ^ 3 ) = 1 :=
by
  sorry

end evaluate_expression_l132_132077


namespace sum_of_faces_of_rectangular_prism_l132_132479

/-- Six positive integers are written on the faces of a rectangular prism.
Each vertex is labeled with the product of the three numbers on the faces adjacent to that vertex.
If the sum of the numbers on the eight vertices is equal to 720, 
prove that the sum of the numbers written on the faces is equal to 27. -/
theorem sum_of_faces_of_rectangular_prism (a b c d e f : ℕ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
(h_vertex_sum : a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 720) :
  (a + d) + (b + e) + (c + f) = 27 :=
by
  sorry

end sum_of_faces_of_rectangular_prism_l132_132479


namespace amount_of_flour_per_new_bread_roll_l132_132433

theorem amount_of_flour_per_new_bread_roll :
  (24 * (1 / 8) = 3) → (16 * f = 3) → (f = 3 / 16) :=
by
  intro h1 h2
  sorry

end amount_of_flour_per_new_bread_roll_l132_132433


namespace cyclic_sum_inequality_l132_132076

theorem cyclic_sum_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a^2 + b^2 + c^2 + (a + b + c)^2 ≤ 4) :
  (ab + 1) / (a + b)^2 + (bc + 1) / (b + c)^2 + (ca + 1) / (c + a)^2 ≥ 3 :=
by
  -- TODO: Provide proof here
  sorry

end cyclic_sum_inequality_l132_132076


namespace domain_transform_l132_132905

-- Definitions based on conditions
def domain_f_x_plus_1 : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def domain_f_id : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def domain_f_2x_minus_1 : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5/2 }

-- The theorem to prove the mathematically equivalent problem
theorem domain_transform :
  (∀ x, (x + 1) ∈ domain_f_x_plus_1) →
  (∀ y, y ∈ domain_f_2x_minus_1 ↔ 2 * y - 1 ∈ domain_f_id) :=
by
  sorry

end domain_transform_l132_132905


namespace find_M_l132_132006

theorem find_M : 
  let S := (981 + 983 + 985 + 987 + 989 + 991 + 993 + 995 + 997 + 999)
  let Target := 5100 - M
  S = Target → M = 4800 :=
by
  sorry

end find_M_l132_132006


namespace money_lent_to_C_l132_132778

theorem money_lent_to_C (X : ℝ) (interest_rate : ℝ) (P_b : ℝ) (T_b : ℝ) (T_c : ℝ) (total_interest : ℝ) :
  interest_rate = 0.09 →
  P_b = 5000 →
  T_b = 2 →
  T_c = 4 →
  total_interest = 1980 →
  (P_b * interest_rate * T_b + X * interest_rate * T_c = total_interest) →
  X = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end money_lent_to_C_l132_132778


namespace check_error_difference_l132_132024

-- Let us define x and y as two-digit natural numbers
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

theorem check_error_difference
    (x y : ℕ)
    (hx : isTwoDigit x)
    (hy : isTwoDigit y)
    (hxy : x > y)
    (h_difference : (100 * y + x) - (100 * x + y) = 2187)
    : x - y = 22 :=
by
  sorry

end check_error_difference_l132_132024


namespace total_ages_l132_132658

-- Definitions of the conditions
variables (A B : ℕ) (x : ℕ)

-- Condition 1: 10 years ago, A was half of B in age.
def condition1 : Prop := A - 10 = 1/2 * (B - 10)

-- Condition 2: The ratio of their present ages is 3:4.
def condition2 : Prop := A = 3 * x ∧ B = 4 * x

-- Main theorem to prove
theorem total_ages (A B : ℕ) (x : ℕ) (h1 : condition1 A B) (h2 : condition2 A B x) : A + B = 35 := 
by
  sorry

end total_ages_l132_132658


namespace alice_safe_paths_l132_132393

/-
Define the coordinate system and conditions.
-/

def total_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

def paths_through_dangerous_area : ℕ :=
  (total_paths 2 2) * (total_paths 2 1)

def safe_paths : ℕ :=
  total_paths 4 3 - paths_through_dangerous_area

theorem alice_safe_paths : safe_paths = 17 := by
  sorry

end alice_safe_paths_l132_132393


namespace compute_expression_l132_132423

theorem compute_expression : (23 + 15)^2 - (23 - 15)^2 = 1380 := by
  sorry

end compute_expression_l132_132423


namespace volume_decreases_by_sixteen_point_sixty_seven_percent_l132_132577

variable {P V k : ℝ}

-- Stating the conditions
def inverse_proportionality (P V k : ℝ) : Prop :=
  P * V = k

def increased_pressure (P : ℝ) : ℝ :=
  1.2 * P

-- Theorem statement to prove the volume decrease percentage
theorem volume_decreases_by_sixteen_point_sixty_seven_percent (P V k : ℝ)
  (h1 : inverse_proportionality P V k)
  (h2 : P' = increased_pressure P) :
  V' = V / 1.2 ∧ (100 * (V - V') / V) = 16.67 :=
by
  sorry

end volume_decreases_by_sixteen_point_sixty_seven_percent_l132_132577


namespace mrs_hilt_money_left_l132_132567

theorem mrs_hilt_money_left (initial_money : ℕ) (cost_of_pencil : ℕ) (money_left : ℕ) (h1 : initial_money = 15) (h2 : cost_of_pencil = 11) : money_left = 4 :=
by
  sorry

end mrs_hilt_money_left_l132_132567


namespace mulch_price_per_pound_l132_132183

noncomputable def price_per_pound (total_cost : ℝ) (total_tons : ℝ) (pounds_per_ton : ℝ) : ℝ :=
  total_cost / (total_tons * pounds_per_ton)

theorem mulch_price_per_pound :
  price_per_pound 15000 3 2000 = 2.5 :=
by
  sorry

end mulch_price_per_pound_l132_132183


namespace quadratic_solution_exists_for_any_a_b_l132_132845

theorem quadratic_solution_exists_for_any_a_b (a b : ℝ) : 
  ∃ x : ℝ, (a^6 - b^6)*x^2 + 2*(a^5 - b^5)*x + (a^4 - b^4) = 0 := 
by
  -- The proof would go here
  sorry

end quadratic_solution_exists_for_any_a_b_l132_132845


namespace monica_sees_121_individual_students_l132_132608

def students_count : ℕ :=
  let class1 := 20
  let class2 := 25
  let class3 := 25
  let class4 := class1 / 2
  let class5 := 28
  let class6 := 28
  let total_spots := class1 + class2 + class3 + class4 + class5 + class6
  let overlap12 := 5
  let overlap45 := 3
  let overlap36 := 7
  total_spots - overlap12 - overlap45 - overlap36

theorem monica_sees_121_individual_students : students_count = 121 := by
  sorry

end monica_sees_121_individual_students_l132_132608


namespace sequence_solution_l132_132104

theorem sequence_solution (a : ℕ → ℤ) :
  a 0 = -1 →
  a 1 = 1 →
  (∀ n ≥ 2, a n = 2 * a (n - 1) + 3 * a (n - 2) + 3^n) →
  ∀ n, a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by
  -- Detailed proof steps will go here.
  sorry

end sequence_solution_l132_132104


namespace cuboid_count_l132_132001

def length_small (m : ℕ) : ℕ := 6
def width_small (m : ℕ) : ℕ := 4
def height_small (m : ℕ) : ℕ := 3

def length_large (m : ℕ): ℕ := 18
def width_large (m : ℕ) : ℕ := 15
def height_large (m : ℕ) : ℕ := 2

def volume (l : ℕ) (w : ℕ) (h : ℕ) : ℕ := l * w * h

def n_small_cuboids (v_large v_small : ℕ) : ℕ := v_large / v_small

theorem cuboid_count : 
  n_small_cuboids (volume (length_large 1) (width_large 1) (height_large 1)) (volume (length_small 1) (width_small 1) (height_small 1)) = 7 :=
by
  sorry

end cuboid_count_l132_132001


namespace trigonometric_identity_l132_132383

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 2 * Real.sin θ + Real.sin θ * Real.cos θ = 2 := by
  sorry

end trigonometric_identity_l132_132383


namespace evaluate_expression_l132_132672

theorem evaluate_expression : 6 + 4 / 2 = 8 :=
by
  sorry

end evaluate_expression_l132_132672


namespace sqrt_inequality_l132_132859

theorem sqrt_inequality : (Real.sqrt 3 + Real.sqrt 7) < 2 * Real.sqrt 5 := 
  sorry

end sqrt_inequality_l132_132859


namespace annual_increase_rate_l132_132899

theorem annual_increase_rate (PV FV : ℝ) (n : ℕ) (r : ℝ) :
  PV = 32000 ∧ FV = 40500 ∧ n = 2 ∧ FV = PV * (1 + r)^2 → r = 0.125 :=
by
  sorry

end annual_increase_rate_l132_132899


namespace second_runner_stop_time_l132_132098

-- Definitions provided by the conditions
def pace_first := 8 -- pace of the first runner in minutes per mile
def pace_second := 7 -- pace of the second runner in minutes per mile
def time_elapsed := 56 -- time elapsed in minutes before the second runner stops
def distance_first := time_elapsed / pace_first -- distance covered by the first runner in miles
def distance_second := time_elapsed / pace_second -- distance covered by the second runner in miles
def distance_gap := distance_second - distance_first -- gap between the runners in miles

-- Statement of the proof problem
theorem second_runner_stop_time :
  8 = distance_gap * pace_first :=
by
sorry

end second_runner_stop_time_l132_132098


namespace find_largest_x_l132_132817

theorem find_largest_x : 
  ∃ x : ℝ, (4 * x ^ 3 - 17 * x ^ 2 + x + 10 = 0) ∧ 
           (∀ y : ℝ, 4 * y ^ 3 - 17 * y ^ 2 + y + 10 = 0 → y ≤ x) ∧ 
           x = (25 + Real.sqrt 545) / 8 :=
sorry

end find_largest_x_l132_132817


namespace fraction_zero_l132_132325

theorem fraction_zero (x : ℝ) (h : x ≠ 3) : (2 * x^2 - 6 * x) / (x - 3) = 0 ↔ x = 0 := 
by
  sorry

end fraction_zero_l132_132325


namespace find_tip_percentage_l132_132164

def original_bill : ℝ := 139.00
def per_person_share : ℝ := 30.58
def number_of_people : ℕ := 5

theorem find_tip_percentage (original_bill : ℝ) (per_person_share : ℝ) (number_of_people : ℕ) 
  (total_paid : ℝ := per_person_share * number_of_people) 
  (tip_amount : ℝ := total_paid - original_bill) : 
  (tip_amount / original_bill) * 100 = 10 :=
by
  sorry

end find_tip_percentage_l132_132164


namespace find_f_neg_two_l132_132856

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (h1 : ∀ a b : ℝ, f (a + b) = f a * f b)
variable (h2 : ∀ x : ℝ, f x > 0)
variable (h3 : f 1 = 1 / 2)

-- State the theorem to prove that f(-2) = 4
theorem find_f_neg_two : f (-2) = 4 :=
by
  sorry

end find_f_neg_two_l132_132856


namespace seth_initial_boxes_l132_132189

-- Definitions based on conditions:
def remaining_boxes_after_giving_half (initial_boxes : ℕ) : ℕ :=
  let boxes_after_giving_to_mother := initial_boxes - 1
  let remaining_boxes := boxes_after_giving_to_mother / 2
  remaining_boxes

-- Main problem statement to prove.
theorem seth_initial_boxes (initial_boxes : ℕ) (remaining_boxes : ℕ) :
  remaining_boxes_after_giving_half initial_boxes = remaining_boxes ->
  remaining_boxes = 4 ->
  initial_boxes = 9 := 
by
  intros h1 h2
  sorry

end seth_initial_boxes_l132_132189


namespace least_positive_integer_remainder_l132_132354

theorem least_positive_integer_remainder :
  ∃ n : ℕ, (n > 0) ∧ (n % 5 = 1) ∧ (n % 4 = 2) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 5 = 1) ∧ (m % 4 = 2) → n ≤ m) :=
sorry

end least_positive_integer_remainder_l132_132354


namespace move_point_right_3_units_l132_132126

theorem move_point_right_3_units (x y : ℤ) (hx : x = 2) (hy : y = -1) :
  (x + 3, y) = (5, -1) :=
by
  sorry

end move_point_right_3_units_l132_132126


namespace parallelLines_perpendicularLines_l132_132780

-- Problem A: Parallel lines
theorem parallelLines (a : ℝ) : 
  (∀x y : ℝ, y = -x + 2 * a → y = (a^2 - 2) * x + 2 → -1 = a^2 - 2) → 
  a = -1 := 
sorry

-- Problem B: Perpendicular lines
theorem perpendicularLines (a : ℝ) : 
  (∀x y : ℝ, y = (2 * a - 1) * x + 3 → y = 4 * x - 3 → (2 * a - 1) * 4 = -1) →
  a = 3 / 8 := 
sorry

end parallelLines_perpendicularLines_l132_132780


namespace min_value_y_minus_one_over_x_l132_132009

variable {x y : ℝ}

-- Condition 1: x is the median of the dataset
def is_median (x : ℝ) : Prop := 3 ≤ x ∧ x ≤ 5

-- Condition 2: The average of the dataset is 1
def average_is_one (x y : ℝ) : Prop := 1 + 2 + x^2 - y = 4

-- The statement to be proved
theorem min_value_y_minus_one_over_x :
  ∀ (x y : ℝ), is_median x → average_is_one x y → y = x^2 - 1 → (y - 1/x) ≥ 23/3 :=
by 
  -- This is a placeholder for the actual proof
  sorry

end min_value_y_minus_one_over_x_l132_132009


namespace find_x_l132_132214

theorem find_x (x : ℝ) (h : (3 * x) / 4 = 24) : x = 32 :=
by
  sorry

end find_x_l132_132214


namespace remainder_correct_l132_132680

open Polynomial

noncomputable def polynomial_remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem remainder_correct : polynomial_remainder (X^6 - 2*X^5 + X^4 - X^2 - 2*X + 1)
                                                  ((X^2 - 1)*(X - 2)*(X + 2))
                                                = 2*X^3 - 9*X^2 + 3*X + 2 :=
by
  sorry

end remainder_correct_l132_132680


namespace train_length_l132_132482

theorem train_length (speed_kmph : ℝ) (time_s : ℝ) (length_m : ℝ) :
  speed_kmph = 60 →
  time_s = 3 →
  length_m = 50.01 :=
by
  sorry

end train_length_l132_132482


namespace prob_of_2_digit_in_frac_1_over_7_l132_132250

noncomputable def prob (n : ℕ) : ℚ := (3/2)^(n-1) / (3/2 - 1)

theorem prob_of_2_digit_in_frac_1_over_7 :
  let infinite_series_sum := ∑' n : ℕ, (2/3)^(6 * n + 3)
  ∑' (n : ℕ), prob (6 * n + 3) = 108 / 665 :=
by
  sorry

end prob_of_2_digit_in_frac_1_over_7_l132_132250


namespace exists_subset_no_three_ap_l132_132818

-- Define the set S_n
def S (n : ℕ) : Finset ℕ := (Finset.range ((3^n + 1) / 2 + 1)).image (λ i => i + 1)

-- Define the property of no three elements forming an arithmetic progression
def no_three_form_ap (M : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a < b → b < c → 2 * b ≠ a + c

-- Define the theorem statement
theorem exists_subset_no_three_ap (n : ℕ) :
  ∃ M : Finset ℕ, M ⊆ S n ∧ M.card = 2^n ∧ no_three_form_ap M :=
sorry

end exists_subset_no_three_ap_l132_132818


namespace sum_of_coefficients_l132_132208

noncomputable def problem_expr (d : ℝ) := (16 * d + 15 + 18 * d^2 + 3 * d^3) + (4 * d + 2 + d^2 + 2 * d^3)
noncomputable def simplified_expr (d : ℝ) := 5 * d^3 + 19 * d^2 + 20 * d + 17

theorem sum_of_coefficients (d : ℝ) (h : d ≠ 0) : 
  problem_expr d = simplified_expr d ∧ (5 + 19 + 20 + 17 = 61) := 
by
  sorry

end sum_of_coefficients_l132_132208


namespace dot_product_example_l132_132948

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_example 
  (ha : a = (-1, 1)) 
  (hb : b = (3, -2)) : dot_product a b = -5 := by
  sorry

end dot_product_example_l132_132948


namespace range_of_a_l132_132640

noncomputable def f (x : ℝ) := x * Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≥ -x^2 + a*x - 6) → a ≤ 5 + Real.log 2 :=
by
  sorry

end range_of_a_l132_132640


namespace part_I_part_II_l132_132604

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x - x * Real.log x

theorem part_I (a : ℝ) :
  (∀ x > 0, 0 ≤ a * Real.exp x - (1 + Real.log x)) ↔ a ≥ 1 / Real.exp 1 :=
sorry

theorem part_II (a : ℝ) (h : a ≥ 2 / Real.exp 2) (x : ℝ) (hx : x > 0) :
  f a x > 0 :=
sorry

end part_I_part_II_l132_132604


namespace count_of_valid_four_digit_numbers_l132_132571

def is_four_digit_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9

def digits_sum_to_twelve (a b c d : ℕ) : Prop :=
  a + b + c + d = 12

def divisible_by_eleven (a b c d : ℕ) : Prop :=
  (a + c - (b + d)) % 11 = 0

theorem count_of_valid_four_digit_numbers : ∃ n : ℕ, n = 20 ∧
  (∀ a b c d : ℕ, is_four_digit_number a b c d →
  digits_sum_to_twelve a b c d →
  divisible_by_eleven a b c d →
  true) :=
sorry

end count_of_valid_four_digit_numbers_l132_132571


namespace inequality_proof_l132_132386

theorem inequality_proof (a b : ℝ) (h1 : a < 1) (h2 : b < 1) (h3 : a + b ≥ 1/3) : 
  (1 - a) * (1 - b) ≤ 25/36 :=
by
  sorry

end inequality_proof_l132_132386


namespace find_number_l132_132812

noncomputable def percentage_of (p : ℝ) (n : ℝ) := p / 100 * n

noncomputable def fraction_of (f : ℝ) (n : ℝ) := f * n

theorem find_number :
  ∃ x : ℝ, percentage_of 40 60 = fraction_of (4/5) x + 4 ∧ x = 25 :=
by
  sorry

end find_number_l132_132812


namespace proof_expr_28_times_35_1003_l132_132834

theorem proof_expr_28_times_35_1003 :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 :=
by
  sorry

end proof_expr_28_times_35_1003_l132_132834


namespace joan_total_socks_l132_132268

theorem joan_total_socks (n : ℕ) (h1 : n / 3 = 60) : n = 180 :=
by
  -- Proof goes here
  sorry

end joan_total_socks_l132_132268


namespace simplify_expression_l132_132685

theorem simplify_expression : 5 * (18 / -9) * (24 / 36) = -(20 / 3) :=
by
  sorry

end simplify_expression_l132_132685


namespace angles_MAB_NAC_l132_132682

/-- Given equal chords AB and AC, and a tangent MAN, with arc BC's measure (excluding point A) being 200 degrees,
prove that the angles MAB and NAC are either 40 degrees or 140 degrees. -/
theorem angles_MAB_NAC (AB AC : ℝ) (tangent_MAN : Prop)
    (arc_BC_measure : ∀ A : ℝ , A = 200) : 
    ∃ θ : ℝ, (θ = 40 ∨ θ = 140) :=
by
  sorry

end angles_MAB_NAC_l132_132682


namespace percent_of_a_is_4b_l132_132530

variable (a b : ℝ)
variable (h : a = 1.2 * b)

theorem percent_of_a_is_4b :
  (4 * b) = (10 / 3 * 100 * a) / 100 :=
by sorry

end percent_of_a_is_4b_l132_132530


namespace discount_per_coupon_l132_132758

-- Definitions and conditions from the problem
def num_cans : ℕ := 9
def cost_per_can : ℕ := 175 -- in cents
def num_coupons : ℕ := 5
def total_payment : ℕ := 2000 -- $20 in cents
def change_received : ℕ := 550 -- $5.50 in cents
def amount_paid := total_payment - change_received

-- Mathematical proof problem
theorem discount_per_coupon :
  let total_cost_without_coupons := num_cans * cost_per_can 
  let total_discount := total_cost_without_coupons - amount_paid
  let discount_per_coupon := total_discount / num_coupons
  discount_per_coupon = 25 :=
by
  sorry

end discount_per_coupon_l132_132758


namespace Y_3_2_eq_1_l132_132852

def Y (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem Y_3_2_eq_1 : Y 3 2 = 1 := by
  sorry

end Y_3_2_eq_1_l132_132852


namespace max_subjects_per_teacher_l132_132471

theorem max_subjects_per_teacher
  (math_teachers : ℕ := 7)
  (physics_teachers : ℕ := 6)
  (chemistry_teachers : ℕ := 5)
  (min_teachers_required : ℕ := 6)
  (total_subjects : ℕ := 18) :
  ∀ (x : ℕ), x ≥ 3 ↔ 6 * x ≥ total_subjects := by
  sorry

end max_subjects_per_teacher_l132_132471


namespace polynomial_unique_f_g_l132_132749

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_unique_f_g :
  (∀ x : ℝ, (x^2 + x + 1) * f (x^2 - x + 1) = (x^2 - x + 1) * g (x^2 + x + 1)) →
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x ∧ g x = k * x) :=
sorry

end polynomial_unique_f_g_l132_132749


namespace plane_equation_l132_132080

noncomputable def equation_of_plane (x y z : ℝ) :=
  3 * x + 2 * z - 1

theorem plane_equation :
  ∀ (x y z : ℝ), 
    (∃ (p : ℝ × ℝ × ℝ), p = (1, 2, -1) ∧ 
                         (∃ (n : ℝ × ℝ × ℝ), n = (3, 0, 2) ∧ 
                                              equation_of_plane x y z = 0)) :=
by
  -- The statement setup is done. The proof is not included as per instructions.
  sorry

end plane_equation_l132_132080


namespace misha_is_older_l132_132588

-- Definitions for the conditions
def tanya_age_19_months_ago : ℕ := 16
def months_ago_for_tanya : ℕ := 19
def misha_age_in_16_months : ℕ := 19
def months_ahead_for_misha : ℕ := 16

-- Convert months to years and residual months
def months_to_years_months (m : ℕ) : ℕ × ℕ := (m / 12, m % 12)

-- Computation for Tanya's current age
def tanya_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ago_for_tanya
  (tanya_age_19_months_ago + years, months)

-- Computation for Misha's current age
def misha_age_now : ℕ × ℕ :=
  let (years, months) := months_to_years_months months_ahead_for_misha
  (misha_age_in_16_months - years, months)

-- Proof statement
theorem misha_is_older : misha_age_now > tanya_age_now := by
  sorry

end misha_is_older_l132_132588


namespace unique_intersection_of_line_and_parabola_l132_132422

theorem unique_intersection_of_line_and_parabola :
  ∃! k : ℚ, ∀ y : ℝ, -3 * y^2 - 4 * y + 7 = k → k = 25 / 3 :=
by
  sorry

end unique_intersection_of_line_and_parabola_l132_132422


namespace min_overlap_percentage_l132_132884

theorem min_overlap_percentage (A B : ℝ) (hA : A = 0.9) (hB : B = 0.8) : ∃ x, x = 0.7 := 
by sorry

end min_overlap_percentage_l132_132884


namespace complement_of_A_in_I_l132_132258

def I : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 6, 7}
def C_I_A : Set ℕ := {1, 3, 5}

theorem complement_of_A_in_I :
  (I \ A) = C_I_A := by
  sorry

end complement_of_A_in_I_l132_132258


namespace min_buses_needed_l132_132443

theorem min_buses_needed (students : ℕ) (cap1 cap2 : ℕ) (h_students : students = 530) (h_cap1 : cap1 = 40) (h_cap2 : cap2 = 45) :
  min (Nat.ceil (students / cap1)) (Nat.ceil (students / cap2)) = 12 :=
  sorry

end min_buses_needed_l132_132443


namespace percentage_discount_l132_132049

theorem percentage_discount (discounted_price original_price : ℝ) (h1 : discounted_price = 560) (h2 : original_price = 700) :
  (original_price - discounted_price) / original_price * 100 = 20 :=
by
  simp [h1, h2]
  sorry

end percentage_discount_l132_132049


namespace sine_of_angle_from_point_l132_132040

theorem sine_of_angle_from_point (x y : ℤ) (r : ℝ) (h : r = Real.sqrt ((x : ℝ)^2 + (y : ℝ)^2)) (hx : x = -12) (hy : y = 5) :
  Real.sin (Real.arctan (y / x)) = y / r := 
by
  sorry

end sine_of_angle_from_point_l132_132040


namespace find_number_l132_132630

theorem find_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 14) : 0.40 * N = 168 :=
sorry

end find_number_l132_132630


namespace royWeight_l132_132371

-- Define the problem conditions
def johnWeight : ℕ := 81
def johnHeavierBy : ℕ := 77

-- Define the main proof problem
theorem royWeight : (johnWeight - johnHeavierBy) = 4 := by
  sorry

end royWeight_l132_132371


namespace equal_spacing_between_paintings_l132_132240

/--
Given:
- The width of each painting is 30 centimeters.
- The total width of the wall in the exhibition hall is 320 centimeters.
- There are six pieces of artwork.
Prove that: The distance between the end of the wall and the artwork, and between the artworks, is 20 centimeters.
-/
theorem equal_spacing_between_paintings :
  let width_painting := 30 -- in centimeters
  let total_wall_width := 320 -- in centimeters
  let num_paintings := 6
  let total_paintings_width := num_paintings * width_painting
  let remaining_space := total_wall_width - total_paintings_width
  let num_spaces := num_paintings + 1
  let space_between := remaining_space / num_spaces
  space_between = 20 := sorry

end equal_spacing_between_paintings_l132_132240


namespace complex_number_solution_l132_132016

theorem complex_number_solution (z : ℂ) (i : ℂ) (h_i : i^2 = -1) 
  (h : -i * z = (3 + 2 * i) * (1 - i)) : z = 1 + 5 * i :=
by
  sorry

end complex_number_solution_l132_132016


namespace sum_of_solutions_l132_132117

theorem sum_of_solutions (x : ℝ) : (∃ x₁ x₂ : ℝ, (x - 4)^2 = 16 ∧ x = x₁ ∨ x = x₂ ∧ x₁ + x₂ = 8) :=
by sorry

end sum_of_solutions_l132_132117


namespace quotient_of_integers_l132_132292

theorem quotient_of_integers
  (a b : ℤ)
  (h : 1996 * a + b / 96 = a + b) :
  b / a = 2016 ∨ a / b = 2016 := 
sorry

end quotient_of_integers_l132_132292


namespace molecular_weight_correct_l132_132746

noncomputable def molecular_weight_compound : ℝ :=
  (3 * 12.01) + (6 * 1.008) + (1 * 16.00)

theorem molecular_weight_correct :
  molecular_weight_compound = 58.078 := by
  sorry

end molecular_weight_correct_l132_132746


namespace number_of_sides_of_polygon_l132_132997

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 540) : n = 5 :=
by
  sorry

end number_of_sides_of_polygon_l132_132997


namespace log_ratio_l132_132782

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : log_base 4 a = log_base 6 b)
  (h4 : log_base 6 b = log_base 9 (a + b)) :
  b / a = (1 + Real.sqrt 5) / 2 := sorry

end log_ratio_l132_132782


namespace program_output_is_1023_l132_132841

-- Definition placeholder for program output.
def program_output : ℕ := 1023

-- Theorem stating the program's output.
theorem program_output_is_1023 : program_output = 1023 := 
by 
  -- Proof details are omitted.
  sorry

end program_output_is_1023_l132_132841


namespace quadratic_inequality_solutions_l132_132654

theorem quadratic_inequality_solutions {k : ℝ} (h1 : 0 < k) (h2 : k < 16) :
  ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l132_132654


namespace trapezium_area_correct_l132_132377

-- Define the lengths of the parallel sides and the distance between them
def a := 24  -- length of the first parallel side in cm
def b := 14  -- length of the second parallel side in cm
def h := 18  -- distance between the parallel sides in cm

-- Define the area calculation function for the trapezium
def trapezium_area (a b h : ℕ) : ℕ :=
  1 / 2 * (a + b) * h

-- The theorem to prove that the area of the given trapezium is 342 square centimeters
theorem trapezium_area_correct : trapezium_area a b h = 342 :=
  sorry

end trapezium_area_correct_l132_132377


namespace rainfall_ratio_l132_132735

noncomputable def total_rainfall := 35
noncomputable def rainfall_second_week := 21

theorem rainfall_ratio 
  (R1 R2 : ℝ)
  (hR2 : R2 = rainfall_second_week)
  (hTotal : R1 + R2 = total_rainfall) :
  R2 / R1 = 3 / 2 := 
by 
  sorry

end rainfall_ratio_l132_132735


namespace area_of_right_square_l132_132119

theorem area_of_right_square (side_length_left : ℕ) (side_length_left_eq : side_length_left = 10) : ∃ area_right, area_right = 68 := 
by
  sorry

end area_of_right_square_l132_132119


namespace find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l132_132603

open Real

-- Given conditions
def line_passes_through (x1 y1 x2 y2 : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l x1 y1 ∧ l x2 y2

def circle_tangent_to_x_axis (center_x center_y : ℝ) (r : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  C center_x center_y ∧ center_y = r

-- We want to prove:
-- 1. The equation of line l is x - 2y = 0
theorem find_line_equation_through_two_points:
  ∃ l : ℝ → ℝ → Prop, line_passes_through 2 1 6 3 l ∧ (∀ x y, l x y ↔ x - 2 * y = 0) :=
  sorry

-- 2. The equation of circle C is (x - 2)^2 + (y - 1)^2 = 1
theorem find_circle_equation_tangent_to_x_axis:
  ∃ C : ℝ → ℝ → Prop, circle_tangent_to_x_axis 2 1 1 C ∧ (∀ x y, C x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :=
  sorry

end find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l132_132603


namespace exponent_division_l132_132623

theorem exponent_division (a : ℕ) (m n : ℕ) (h1 : 19 = a) (h2 : 11 = m) (h3 : 8 = n) : a^(m - n) = 6859 := by
  sorry

end exponent_division_l132_132623


namespace vacant_seats_l132_132177

theorem vacant_seats (total_seats filled_percentage : ℕ) (h_filled_percentage : filled_percentage = 62) (h_total_seats : total_seats = 600) : 
  (total_seats - total_seats * filled_percentage / 100) = 228 :=
by
  sorry

end vacant_seats_l132_132177


namespace sum_of_four_consecutive_even_integers_l132_132802

theorem sum_of_four_consecutive_even_integers (x : ℕ) (hx : x > 4) :
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → (x - 4) + (x - 2) + x + (x + 2) = 28 := by
{
  sorry
}

end sum_of_four_consecutive_even_integers_l132_132802


namespace compare_abc_l132_132645

noncomputable def a : ℝ := 1 / (1 + Real.exp 2)
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := Real.log ((1 + Real.exp 2) / (Real.exp 2))

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l132_132645


namespace min_value_of_a_is_five_l132_132406

-- Given: a, b, c in table satisfying the conditions
-- We are to prove that the minimum value of a is 5.
theorem min_value_of_a_is_five
  {a b c: ℤ} (h_pos: 0 < a) (hx_distinct: 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 ∧ 
                               a*x₁^2 + b*x₁ + c = 0 ∧ 
                               a*x₂^2 + b*x₂ + c = 0) (hb_neg: b < 0) 
                               (h_disc_pos: (b^2 - 4*a*c) > 0) : a = 5 :=
sorry

end min_value_of_a_is_five_l132_132406


namespace problem_statement_l132_132624

def S : Set Nat := {x | x ∈ Finset.range 13 \ Finset.range 1}

def n : Nat :=
  4^12 - 3 * 3^12 + 3 * 2^12

theorem problem_statement : n % 1000 = 181 :=
by
  sorry

end problem_statement_l132_132624


namespace find_k_if_equal_roots_l132_132253

theorem find_k_if_equal_roots (a b k : ℚ) 
  (h1 : 2 * a + b = -4) 
  (h2 : 2 * a * b + a^2 = -60) 
  (h3 : -2 * a^2 * b = k)
  (h4 : a ≠ b)
  (h5 : k > 0) :
  k = 6400 / 27 :=
by {
  sorry
}

end find_k_if_equal_roots_l132_132253


namespace fraction_square_eq_decimal_l132_132892

theorem fraction_square_eq_decimal :
  ∃ (x : ℚ), x^2 = 0.04000000000000001 ∧ x = 1 / 5 :=
by
  sorry

end fraction_square_eq_decimal_l132_132892


namespace sum_of_sides_l132_132470

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (cosB cosC : ℝ)
variable (sinB : ℝ)
variable (area : ℝ)

-- Given conditions
axiom h1 : b = 2
axiom h2 : b * cosC + c * cosB = 3 * a * cosB
axiom h3 : area = 3 * Real.sqrt 2 / 2
axiom h4 : sinB = Real.sqrt (1 - cosB ^ 2)

-- Prove the desired result
theorem sum_of_sides (A B C a b c cosB cosC sinB : ℝ) (area : ℝ)
  (h1 : b = 2)
  (h2 : b * cosC + c * cosB = 3 * a * cosB)
  (h3 : area = 3 * Real.sqrt 2 / 2)
  (h4 : sinB = Real.sqrt (1 - cosB ^ 2)) :
  a + c = 4 := 
sorry

end sum_of_sides_l132_132470


namespace andrena_has_more_dolls_than_debelyn_l132_132520

-- Define the initial number of dolls
def initial_dolls_Debelyn : ℕ := 20
def initial_dolls_Christel : ℕ := 24

-- Define the number of dolls given to Andrena
def dolls_given_by_Debelyn : ℕ := 2
def dolls_given_by_Christel : ℕ := 5

-- Define the condition that Andrena has 2 more dolls than Christel after receiving the dolls
def andrena_more_than_christel : ℕ := 2

-- Define the dolls count after gift exchange
def dolls_Debelyn_after : ℕ := initial_dolls_Debelyn - dolls_given_by_Debelyn
def dolls_Christel_after : ℕ := initial_dolls_Christel - dolls_given_by_Christel
def dolls_Andrena_after : ℕ := dolls_Christel_after + andrena_more_than_christel

-- Define the proof problem
theorem andrena_has_more_dolls_than_debelyn : dolls_Andrena_after - dolls_Debelyn_after = 3 := by
  sorry

end andrena_has_more_dolls_than_debelyn_l132_132520


namespace divisor_greater_than_8_l132_132695

-- Define the condition that remainder is 8
def remainder_is_8 (n m : ℕ) : Prop :=
  n % m = 8

-- Theorem: If n divided by m has remainder 8, then m must be greater than 8
theorem divisor_greater_than_8 (m : ℕ) (hm : m ≤ 8) : ¬ exists n, remainder_is_8 n m :=
by
  sorry

end divisor_greater_than_8_l132_132695


namespace hexagons_cover_65_percent_l132_132299

noncomputable def hexagon_percent_coverage
    (a : ℝ)
    (square_area : ℝ := a^2) 
    (hexagon_area : ℝ := (3 * Real.sqrt 3 / 8 * a^2))
    (tile_pattern : ℝ := 3): Prop :=
    hexagon_area / square_area * tile_pattern = (65 / 100)

theorem hexagons_cover_65_percent (a : ℝ) : hexagon_percent_coverage a :=
by
    sorry

end hexagons_cover_65_percent_l132_132299


namespace inequality_proof_l132_132205

variables {a b c : ℝ}

theorem inequality_proof (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b))) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l132_132205


namespace cos_Z_value_l132_132647

-- The conditions given in the problem
def sin_X := 4 / 5
def cos_Y := 3 / 5

-- The theorem we want to prove
theorem cos_Z_value (sin_X : ℝ) (cos_Y : ℝ) (hX : sin_X = 4/5) (hY : cos_Y = 3/5) : 
  ∃ cos_Z : ℝ, cos_Z = 7 / 25 :=
by
  -- Attach all conditions and solve
  sorry

end cos_Z_value_l132_132647


namespace Sarahs_score_l132_132675

theorem Sarahs_score (x g : ℕ) (h1 : g = x - 50) (h2 : (x + g) / 2 = 110) : x = 135 := by 
  sorry

end Sarahs_score_l132_132675


namespace merchant_discount_percentage_l132_132191

theorem merchant_discount_percentage
  (CP MP SP : ℝ)
  (h1 : MP = CP + 0.40 * CP)
  (h2 : SP = CP + 0.26 * CP)
  : ((MP - SP) / MP) * 100 = 10 := by
  sorry

end merchant_discount_percentage_l132_132191


namespace ratio_of_x_intercepts_l132_132797

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end ratio_of_x_intercepts_l132_132797


namespace socks_selection_l132_132519

theorem socks_selection :
  ∀ (R Y G B O : ℕ), 
    R = 80 → Y = 70 → G = 50 → B = 60 → O = 40 →
    (∃ k, k = 38 ∧ ∀ (N : ℕ → ℕ), (N R + N Y + N G + N B + N O ≥ k)
          → (exists (pairs : ℕ), pairs ≥ 15 ∧ pairs = (N R / 2) + (N Y / 2) + (N G / 2) + (N B / 2) + (N O / 2) )) :=
by
  sorry

end socks_selection_l132_132519


namespace max_download_speed_l132_132157

def download_speed (size_GB : ℕ) (time_hours : ℕ) : ℚ :=
  let size_MB := size_GB * 1024
  let time_seconds := time_hours * 60 * 60
  size_MB / time_seconds

theorem max_download_speed (h₁ : size_GB = 360) (h₂ : time_hours = 2) :
  download_speed size_GB time_hours = 51.2 :=
by
  sorry

end max_download_speed_l132_132157


namespace f_of_g_of_3_l132_132239

def f (x : ℝ) : ℝ := 4 * x - 5
def g (x : ℝ) : ℝ := (x + 2)^2
theorem f_of_g_of_3 : f (g 3) = 95 := by
  sorry

end f_of_g_of_3_l132_132239


namespace sum_op_two_triangles_l132_132475

def op (a b c : ℕ) : ℕ := 2 * a - b + c

theorem sum_op_two_triangles : op 3 7 5 + op 6 2 8 = 22 := by
  sorry

end sum_op_two_triangles_l132_132475


namespace problem_1_simplified_problem_2_simplified_l132_132664

noncomputable def problem_1 : ℝ :=
  2 * Real.sqrt 18 - Real.sqrt 50 + (1/2) * Real.sqrt 32

theorem problem_1_simplified : problem_1 = 3 * Real.sqrt 2 :=
  sorry

noncomputable def problem_2 : ℝ :=
  (Real.sqrt 5 + Real.sqrt 6) * (Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - 1)^2

theorem problem_2_simplified : problem_2 = -7 + 2 * Real.sqrt 5 :=
  sorry

end problem_1_simplified_problem_2_simplified_l132_132664


namespace arithmetic_sequence_fifth_term_l132_132860

theorem arithmetic_sequence_fifth_term (x : ℝ) (a₂ : ℝ := x) (a₃ : ℝ := 3) 
    (a₁ : ℝ := -1) (h₁ : a₂ = a₁ + (1*(x + 1))) (h₂ : a₃ = a₁ + 2*(x + 1)) : 
    a₁ + 4*(a₃ - a₂ + 1) = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l132_132860


namespace max_k_divides_expression_l132_132594

theorem max_k_divides_expression : ∃ k, (∀ n : ℕ, n > 0 → 2^k ∣ (3^(2*n + 3) + 40*n - 27)) ∧ k = 6 :=
sorry

end max_k_divides_expression_l132_132594


namespace ellipse_range_k_l132_132637

theorem ellipse_range_k (k : ℝ) (h1 : 3 + k > 0) (h2 : 2 - k > 0) (h3 : k ≠ -1 / 2) :
  k ∈ Set.Ioo (-3 : ℝ) (-1 / 2) ∪ Set.Ioo (-1 / 2) (2 : ℝ) :=
sorry

end ellipse_range_k_l132_132637


namespace select_3_products_select_exactly_1_defective_select_at_least_1_defective_l132_132402

noncomputable def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

namespace ProductInspection

def total_products : Nat := 100
def qualified_products : Nat := 98
def defective_products : Nat := 2

-- Proof Problem 1
theorem select_3_products (h : combination total_products 3 = 161700) : True := by
  trivial

-- Proof Problem 2
theorem select_exactly_1_defective (h : combination defective_products 1 * combination qualified_products 2 = 9506) : True := by
  trivial

-- Proof Problem 3
theorem select_at_least_1_defective (h : combination total_products 3 - combination qualified_products 3 = 9604) : True := by
  trivial

end ProductInspection

end select_3_products_select_exactly_1_defective_select_at_least_1_defective_l132_132402


namespace cost_price_eq_l132_132360

variables (x : ℝ)

def f (x : ℝ) : ℝ := x * (1 + 0.30)
def g (x : ℝ) : ℝ := f x * 0.80

theorem cost_price_eq (h : g x = 2080) : x * (1 + 0.30) * 0.80 = 2080 :=
by sorry

end cost_price_eq_l132_132360


namespace luke_bus_time_l132_132259

theorem luke_bus_time
  (L : ℕ)   -- Luke's bus time to work in minutes
  (P : ℕ)   -- Paula's bus time to work in minutes
  (B : ℕ)   -- Luke's bike time home in minutes
  (h1 : P = 3 * L / 5) -- Paula's bus time is \( \frac{3}{5} \) of Luke's bus time
  (h2 : B = 5 * L)     -- Luke's bike time is 5 times his bus time
  (h3 : L + P + B + P = 504) -- Total travel time is 504 minutes
  : L = 70 := 
sorry

end luke_bus_time_l132_132259


namespace tiles_difference_between_tenth_and_eleventh_square_l132_132811

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

-- Define the area of the nth square
def area (n : ℕ) : ℕ :=
  (side_length n) ^ 2

-- The math proof statement
theorem tiles_difference_between_tenth_and_eleventh_square : area 11 - area 10 = 88 :=
by 
  -- Proof goes here, but we use sorry to skip it for now
  sorry

end tiles_difference_between_tenth_and_eleventh_square_l132_132811


namespace slope_of_regression_line_l132_132553

variable (h : ℝ)
variable (t1 T1 t2 T2 t3 T3 : ℝ)

-- Given conditions.
axiom t2_is_equally_spaced : t2 = t1 + h
axiom t3_is_equally_spaced : t3 = t1 + 2 * h

theorem slope_of_regression_line :
  t2 = t1 + h →
  t3 = t1 + 2 * h →
  (T3 - T1) / (t3 - t1) = (T3 - T1) / ((t1 + 2 * h) - t1) := 
by
  sorry

end slope_of_regression_line_l132_132553


namespace domain_f_l132_132510

def domain (f : ℝ → ℝ) : Set ℝ := {x | ∃ y, f x = y}

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x-1) / Real.log 2) + 1

theorem domain_f : domain f = {x | 1 < x} :=
by {
  sorry
}

end domain_f_l132_132510


namespace packages_delivered_by_third_butcher_l132_132380

theorem packages_delivered_by_third_butcher 
  (x y z : ℕ) 
  (h1 : x = 10) 
  (h2 : y = 7) 
  (h3 : 4 * x + 4 * y + 4 * z = 100) : 
  z = 8 :=
by { sorry }

end packages_delivered_by_third_butcher_l132_132380


namespace factorized_sum_is_33_l132_132610

theorem factorized_sum_is_33 (p q r : ℤ)
  (h1 : ∀ x : ℤ, x^2 + 21 * x + 110 = (x + p) * (x + q))
  (h2 : ∀ x : ℤ, x^2 - 23 * x + 132 = (x - q) * (x - r)) : 
  p + q + r = 33 := by
  sorry

end factorized_sum_is_33_l132_132610


namespace example_problem_l132_132816

def diamond (a b : ℕ) : ℕ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem example_problem : diamond 3 2 = 125 := by
  sorry

end example_problem_l132_132816


namespace system_sol_l132_132745

theorem system_sol {x y : ℝ} (h1 : x + 2 * y = -1) (h2 : 2 * x + y = 3) : x - y = 4 := by
  sorry

end system_sol_l132_132745


namespace total_meters_examined_l132_132424

theorem total_meters_examined (total_meters : ℝ) (h : 0.10 * total_meters = 12) :
  total_meters = 120 :=
sorry

end total_meters_examined_l132_132424


namespace cube_less_than_triple_l132_132902

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l132_132902


namespace total_selling_price_l132_132481

theorem total_selling_price (cost_per_meter profit_per_meter : ℕ) (total_meters : ℕ) :
  cost_per_meter = 90 → 
  profit_per_meter = 15 → 
  total_meters = 85 → 
  (cost_per_meter + profit_per_meter) * total_meters = 8925 :=
by
  intros
  sorry

end total_selling_price_l132_132481


namespace count_multiples_5_or_10_l132_132751

theorem count_multiples_5_or_10 (n : ℕ) (hn : n = 999) : 
  ∃ k : ℕ, k = 199 ∧ (∀ i : ℕ, i < 1000 → (i % 5 = 0 ∨ i % 10 = 0) → i = k) := 
by {
  sorry
}

end count_multiples_5_or_10_l132_132751


namespace sum_q_p_is_minus_12_l132_132387

noncomputable def p (x : ℝ) : ℝ := x^2 - 3 * x + 2

noncomputable def q (x : ℝ) : ℝ := -x^2

theorem sum_q_p_is_minus_12 :
  (q (p 0) + q (p 1) + q (p 2) + q (p 3) + q (p 4)) = -12 :=
by
  sorry

end sum_q_p_is_minus_12_l132_132387


namespace sphere_tangent_radius_l132_132933

variables (a b : ℝ) (h : b > a)

noncomputable def radius (a b : ℝ) : ℝ := a * (b - a) / Real.sqrt (b^2 - a^2)

theorem sphere_tangent_radius (a b : ℝ) (h : b > a) : 
  radius a b = a * (b - a) / Real.sqrt (b^2 - a^2) :=
by sorry

end sphere_tangent_radius_l132_132933


namespace inequality_holds_l132_132281

variable {a b c : ℝ}

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
sorry

end inequality_holds_l132_132281


namespace h_value_at_3_l132_132756

noncomputable def f (x : ℝ) : ℝ := 3 * x + 4
noncomputable def g (x : ℝ) : ℝ := (Real.sqrt (f x) - 3) ^ 2
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_value_at_3 : h 3 = 70 - 18 * Real.sqrt 13 := 
by
  -- Proof goes here
  sorry

end h_value_at_3_l132_132756


namespace find_b_in_geometric_sequence_l132_132629

theorem find_b_in_geometric_sequence 
  (a b c : ℝ) 
  (q : ℝ) 
  (h1 : -1 * q^4 = -9) 
  (h2 : a = -1 * q) 
  (h3 : b = a * q) 
  (h4 : c = b * q) 
  (h5 : -9 = c * q) : 
  b = -3 :=
by
  sorry

end find_b_in_geometric_sequence_l132_132629


namespace complete_square_variant_l132_132090

theorem complete_square_variant (x : ℝ) :
    3 * x^2 + 4 * x + 1 = 0 → (x + 2 / 3) ^ 2 = 1 / 9 :=
by
  intro h
  sorry

end complete_square_variant_l132_132090


namespace dice_sum_probability_l132_132410

-- Define a noncomputable function to calculate the number of ways to get a sum of 15
noncomputable def dice_sum_ways (dices : ℕ) (sides : ℕ) (target_sum : ℕ) : ℕ :=
  sorry

-- Define the Lean 4 statement
theorem dice_sum_probability :
  dice_sum_ways 5 6 15 = 2002 :=
sorry

end dice_sum_probability_l132_132410


namespace range_of_first_person_l132_132206

variable (R1 R2 R3 : ℕ)
variable (min_range : ℕ)
variable (condition1 : min_range = 25)
variable (condition2 : R2 = 25)
variable (condition3 : R3 = 30)
variable (condition4 : min_range ≤ R1 ∧ min_range ≤ R2 ∧ min_range ≤ R3)

theorem range_of_first_person : R1 = 25 :=
by
  sorry

end range_of_first_person_l132_132206


namespace symmetry_axis_g_l132_132609

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x - Real.pi / 3) - Real.pi / 3)

theorem symmetry_axis_g :
  ∃ k : ℤ, (x = k * Real.pi / 2 + Real.pi / 4) := sorry

end symmetry_axis_g_l132_132609


namespace prob_not_same_city_is_056_l132_132871

def probability_not_same_city (P_A_cityA P_B_cityA : ℝ) : ℝ :=
  let P_A_cityB := 1 - P_A_cityA
  let P_B_cityB := 1 - P_B_cityA
  (P_A_cityA * P_B_cityB) + (P_A_cityB * P_B_cityA)

theorem prob_not_same_city_is_056 :
  probability_not_same_city 0.6 0.2 = 0.56 :=
by
  sorry

end prob_not_same_city_is_056_l132_132871


namespace amount_needed_is_72_l132_132269

-- Define the given conditions
def original_price : ℝ := 90
def discount_rate : ℝ := 20

-- The goal is to prove that the amount of money needed after the discount is $72
theorem amount_needed_is_72 (P : ℝ) (D : ℝ) (hP : P = original_price) (hD : D = discount_rate) : P - (D / 100 * P) = 72 := 
by sorry

end amount_needed_is_72_l132_132269


namespace flight_duration_sum_l132_132924

theorem flight_duration_sum 
  (departure_time : ℕ×ℕ) (arrival_time : ℕ×ℕ) (delay : ℕ)
  (h m : ℕ)
  (h0 : 0 < m ∧ m < 60)
  (h1 : departure_time = (9, 20))
  (h2 : arrival_time = (13, 45)) -- using 13 for 1 PM, 24-hour format
  (h3 : delay = 25)
  (h4 : ((arrival_time.1 * 60 + arrival_time.2) - (departure_time.1 * 60 + departure_time.2) + delay) = h * 60 + m) :
  h + m = 29 :=
by {
  -- Proof is skipped
  sorry
}

end flight_duration_sum_l132_132924


namespace garden_perimeter_l132_132215

theorem garden_perimeter (width_garden length_playground width_playground : ℕ) 
  (h1 : width_garden = 12) 
  (h2 : length_playground = 16) 
  (h3 : width_playground = 12) 
  (area_playground : ℕ)
  (h4 : area_playground = length_playground * width_playground) 
  (area_garden : ℕ) 
  (h5 : area_garden = area_playground) 
  (length_garden : ℕ) 
  (h6 : area_garden = length_garden * width_garden) :
  2 * length_garden + 2 * width_garden = 56 := by
  sorry

end garden_perimeter_l132_132215


namespace find_a3_plus_a9_l132_132236

variable (a : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

-- Conditions stating sequence is arithmetic and a₁ + a₆ + a₁₁ = 3
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def a_1_6_11_sum (a : ℕ → ℝ) : Prop :=
  a 1 + a 6 + a 11 = 3

theorem find_a3_plus_a9 
  (h_arith : is_arithmetic_sequence a d)
  (h_sum : a_1_6_11_sum a) : 
  a 3 + a 9 = 2 := 
sorry

end find_a3_plus_a9_l132_132236


namespace cost_price_of_article_l132_132741

theorem cost_price_of_article (C : ℝ) (h1 : 86 - C = C - 42) : C = 64 :=
by
  sorry

end cost_price_of_article_l132_132741


namespace infer_correct_l132_132257

theorem infer_correct (a b c : ℝ) (h1: c < b) (h2: b < a) (h3: a + b + c = 0) :
  (c * b^2 ≤ ab^2) ∧ (ab > ac) :=
by
  sorry

end infer_correct_l132_132257


namespace jason_quarters_l132_132405

def quarters_original := 49
def quarters_added := 25
def quarters_total := 74

theorem jason_quarters : quarters_original + quarters_added = quarters_total :=
by
  sorry

end jason_quarters_l132_132405


namespace number_with_at_least_two_zeros_l132_132441

-- A 6-digit number can have for its leftmost digit anything from 1 to 9 inclusive,
-- and for each of its next five digits anything from 0 through 9 inclusive.
def total_6_digit_numbers : ℕ := 9 * 10^5

-- A 6-digit number with no zeros consists solely of digits from 1 to 9
def no_zero : ℕ := 9^6

-- A 6-digit number with exactly one zero
def exactly_one_zero : ℕ := 5 * 9^5

-- The number of 6-digit numbers with less than two zeros is the sum of no_zero and exactly_one_zero
def less_than_two_zeros : ℕ := no_zero + exactly_one_zero

-- The number of 6-digit numbers with at least two zeros is the difference between total_6_digit_numbers and less_than_two_zeros
def at_least_two_zeros : ℕ := total_6_digit_numbers - less_than_two_zeros

-- The theorem that states the number of 6-digit numbers with at least two zeros is 73,314
theorem number_with_at_least_two_zeros : at_least_two_zeros = 73314 := 
by
  sorry

end number_with_at_least_two_zeros_l132_132441


namespace value_of_a_l132_132242

theorem value_of_a (a : ℕ) (h1 : a * 9^3 = 3 * 15^5) (h2 : a = 5^5) : a = 3125 := by
  sorry

end value_of_a_l132_132242


namespace largest_divisor_of_5_consecutive_integers_l132_132775

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end largest_divisor_of_5_consecutive_integers_l132_132775


namespace Oliver_monster_club_cards_l132_132118

theorem Oliver_monster_club_cards (BG AB MC : ℕ) 
  (h1 : BG = 48) 
  (h2 : BG = 3 * AB) 
  (h3 : MC = 2 * AB) : 
  MC = 32 :=
by sorry

end Oliver_monster_club_cards_l132_132118


namespace find_p_plus_q_l132_132245

noncomputable def probability_only_one (factor : ℕ → Prop) : ℚ := 0.08 -- Condition 1
noncomputable def probability_exaclty_two (factor1 factor2 : ℕ → Prop) : ℚ := 0.12 -- Condition 2
noncomputable def probability_all_three_given_two (factor1 factor2 factor3 : ℕ → Prop) : ℚ := 1 / 4 -- Condition 3
def women_without_D_has_no_risk_factors (total_women women_with_D women_with_all_factors women_without_D_no_risk_factors : ℕ) : ℚ :=
  women_without_D_no_risk_factors / (total_women - women_with_D)

theorem find_p_plus_q : ∃ (p q : ℕ), (women_without_D_has_no_risk_factors 100 (8 + 2 * 12 + 4) 4 28 = p / q) ∧ (Nat.gcd p q = 1) ∧ p + q = 23 :=
by
  sorry

end find_p_plus_q_l132_132245


namespace positive_number_property_l132_132499

-- Define the problem conditions and the goal
theorem positive_number_property (y : ℝ) (hy : y > 0) (h : y^2 / 100 = 9) : y = 30 := by
  sorry

end positive_number_property_l132_132499


namespace equal_cost_at_150_miles_l132_132171

def cost_Safety (m : ℝ) := 41.95 + 0.29 * m
def cost_City (m : ℝ) := 38.95 + 0.31 * m
def cost_Metro (m : ℝ) := 44.95 + 0.27 * m

theorem equal_cost_at_150_miles (m : ℝ) :
  cost_Safety m = cost_City m ∧ cost_Safety m = cost_Metro m → m = 150 :=
by
  sorry

end equal_cost_at_150_miles_l132_132171


namespace greene_family_admission_cost_l132_132317

theorem greene_family_admission_cost (x : ℝ) (h1 : ∀ y : ℝ, y = x - 13) (h2 : ∀ z : ℝ, z = x + (x - 13)) :
  x = 45 :=
by
  sorry

end greene_family_admission_cost_l132_132317


namespace evaluate_expression_l132_132068

def x : ℝ := 2
def y : ℝ := 4

theorem evaluate_expression : y * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l132_132068


namespace solve_for_q_l132_132407

theorem solve_for_q : 
  let n : ℤ := 63
  let m : ℤ := 14
  ∀ (q : ℤ),
  (7 : ℤ) / 9 = n / 81 ∧
  (7 : ℤ) / 9 = (m + n) / 99 ∧
  (7 : ℤ) / 9 = (q - m) / 135 → 
  q = 119 :=
by
  sorry

end solve_for_q_l132_132407


namespace cost_of_downloading_360_songs_in_2005_is_144_dollars_l132_132385

theorem cost_of_downloading_360_songs_in_2005_is_144_dollars :
  (∀ (c_2004 c_2005 : ℕ), (∀ c : ℕ, c_2005 = c ∧ c_2004 = c + 32) →
  200 * c_2004 = 360 * c_2005 → 360 * c_2005 / 100 = 144) :=
  by sorry

end cost_of_downloading_360_songs_in_2005_is_144_dollars_l132_132385


namespace energy_conservation_l132_132692

-- Define the conditions
variables (m : ℝ) (v_train v_ball : ℝ)
-- The speed of the train and the ball, converted to m/s
variables (v := 60 * 1000 / 3600) -- 60 km/h in m/s
variables (E_initial : ℝ := 0.5 * m * (v ^ 2))

-- Kinetic energy of the ball when thrown in the same direction
variables (E_same_direction : ℝ := 0.5 * m * (2 * v)^2)

-- Kinetic energy of the ball when thrown in the opposite direction
variables (E_opposite_direction : ℝ := 0.5 * m * (0)^2)

-- Prove energy conservation
theorem energy_conservation : 
  (E_same_direction - E_initial) + (E_opposite_direction - E_initial) = 0 :=
sorry

end energy_conservation_l132_132692


namespace initial_pencils_correct_l132_132544

variable (pencils_taken remaining_pencils initial_pencils : ℕ)

def initial_number_of_pencils (pencils_taken remaining_pencils : ℕ) : ℕ :=
  pencils_taken + remaining_pencils

theorem initial_pencils_correct (h₁ : pencils_taken = 22) (h₂ : remaining_pencils = 12) :
  initial_number_of_pencils pencils_taken remaining_pencils = 34 := by
  rw [h₁, h₂]
  rfl

end initial_pencils_correct_l132_132544


namespace frogs_need_new_pond_l132_132375

theorem frogs_need_new_pond
  (num_frogs : ℕ) 
  (num_tadpoles : ℕ) 
  (num_survivor_tadpoles : ℕ) 
  (pond_capacity : ℕ) 
  (hc1 : num_frogs = 5)
  (hc2 : num_tadpoles = 3 * num_frogs)
  (hc3 : num_survivor_tadpoles = (2 * num_tadpoles) / 3)
  (hc4 : pond_capacity = 8):
  ((num_frogs + num_survivor_tadpoles) - pond_capacity) = 7 :=
by sorry

end frogs_need_new_pond_l132_132375


namespace t_shaped_grid_sum_l132_132089

open Finset

theorem t_shaped_grid_sum :
  ∃ (a b c d e : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧
    (d ≠ e) ∧
    a + b + c = 20 ∧
    d + e = 7 ∧
    (a + b + c + d + e + b) = 33 :=
sorry

end t_shaped_grid_sum_l132_132089


namespace incorrect_transformation_l132_132962

theorem incorrect_transformation (a b : ℕ) (h : a ≠ 0 ∧ b ≠ 0) (h_eq : a / 2 = b / 3) :
  (∃ k : ℕ, 2 * a = 3 * b → false) ∧ 
  (a / b = 2 / 3) ∧ 
  (b / a = 3 / 2) ∧
  (3 * a = 2 * b) :=
by
  sorry

end incorrect_transformation_l132_132962


namespace trig_expression_simplification_l132_132653

theorem trig_expression_simplification (α : Real) :
  Real.cos (3/2 * Real.pi + 4 * α)
  + Real.sin (3 * Real.pi - 8 * α)
  - Real.sin (4 * Real.pi - 12 * α)
  = 4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) :=
sorry

end trig_expression_simplification_l132_132653


namespace value_of_a_for_perfect_square_trinomial_l132_132681

theorem value_of_a_for_perfect_square_trinomial (a : ℝ) (x y : ℝ) :
  (∃ b : ℝ, (x + b * y) ^ 2 = x^2 + a * x * y + y^2) ↔ (a = 2 ∨ a = -2) :=
by
  sorry

end value_of_a_for_perfect_square_trinomial_l132_132681


namespace vector_transitivity_l132_132840

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem vector_transitivity (h1 : a = b) (h2 : b = c) : a = c :=
by {
  sorry
}

end vector_transitivity_l132_132840


namespace gcd_2835_9150_l132_132249

theorem gcd_2835_9150 : Nat.gcd 2835 9150 = 15 := by
  sorry

end gcd_2835_9150_l132_132249


namespace dodecahedron_edge_coloring_l132_132789

-- Define the properties of the dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)          -- 12 pentagonal faces
  (edges : Fin 30)         -- 30 edges
  (vertices : Fin 20)      -- 20 vertices
  (edge_faces : Fin 30 → Fin 2) -- Each edge contributes to two faces

-- Prove the number of valid edge colorations such that each face has an even number of red edges
theorem dodecahedron_edge_coloring : 
    (∃ num_colorings : ℕ, num_colorings = 2^11) :=
sorry

end dodecahedron_edge_coloring_l132_132789


namespace more_newborn_elephants_than_baby_hippos_l132_132561

-- Define the given conditions
def initial_elephants := 20
def initial_hippos := 35
def female_frac := 5 / 7
def births_per_female_hippo := 5
def total_animals_after_birth := 315

-- Calculate the required values
def female_hippos := female_frac * initial_hippos
def baby_hippos := female_hippos * births_per_female_hippo
def total_animals_before_birth := initial_elephants + initial_hippos
def total_newborns := total_animals_after_birth - total_animals_before_birth
def newborn_elephants := total_newborns - baby_hippos

-- Define the proof statement
theorem more_newborn_elephants_than_baby_hippos :
  (newborn_elephants - baby_hippos) = 10 :=
by
  sorry

end more_newborn_elephants_than_baby_hippos_l132_132561


namespace triple_solution_exists_and_unique_l132_132114

theorem triple_solution_exists_and_unique:
  ∀ (x y z : ℝ), (1 + x^4 ≤ 2 * (y - z) ^ 2) ∧ (1 + y^4 ≤ 2 * (z - x) ^ 2) ∧ (1 + z^4 ≤ 2 * (x - y) ^ 2)
  → (x = 1 ∧ y = 0 ∧ z = -1) :=
by
  sorry

end triple_solution_exists_and_unique_l132_132114


namespace find_x_range_l132_132158

theorem find_x_range {x : ℝ} : 
  (∀ (m : ℝ), abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 ) →
  ( ( -1 + Real.sqrt 7 ) / 2 < x ∧ x < ( 1 + Real.sqrt 3 ) / 2 ) :=
by
  intros h
  sorry

end find_x_range_l132_132158


namespace minimize_y_l132_132184

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, x = (3 * a + b) / 4 ∧ 
  ∀ y : ℝ, (3 * (y - a) ^ 2 + (y - b) ^ 2) ≥ (3 * ((3 * a + b) / 4 - a) ^ 2 + ((3 * a + b) / 4 - b) ^ 2) :=
sorry

end minimize_y_l132_132184


namespace base5_addition_l132_132345

theorem base5_addition : 
  (14 : ℕ) + (132 : ℕ) = (101 : ℕ) :=
by {
  sorry
}

end base5_addition_l132_132345


namespace crayons_lost_or_given_away_total_l132_132774

def initial_crayons_box1 := 479
def initial_crayons_box2 := 352
def initial_crayons_box3 := 621

def remaining_crayons_box1 := 134
def remaining_crayons_box2 := 221
def remaining_crayons_box3 := 487

def crayons_lost_or_given_away_box1 := initial_crayons_box1 - remaining_crayons_box1
def crayons_lost_or_given_away_box2 := initial_crayons_box2 - remaining_crayons_box2
def crayons_lost_or_given_away_box3 := initial_crayons_box3 - remaining_crayons_box3

def total_crayons_lost_or_given_away := crayons_lost_or_given_away_box1 + crayons_lost_or_given_away_box2 + crayons_lost_or_given_away_box3

theorem crayons_lost_or_given_away_total : total_crayons_lost_or_given_away = 610 :=
by
  -- Proof should go here
  sorry

end crayons_lost_or_given_away_total_l132_132774


namespace number_of_sixes_l132_132566

theorem number_of_sixes
  (total_runs : ℕ)
  (boundaries : ℕ)
  (percent_runs_by_running : ℚ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (runs_by_running : ℚ)
  (runs_by_boundaries : ℕ)
  (runs_by_sixes : ℕ)
  (number_of_sixes : ℕ)
  (h1 : total_runs = 120)
  (h2 : boundaries = 6)
  (h3 : percent_runs_by_running = 0.6)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6)
  (h6 : runs_by_running = percent_runs_by_running * total_runs)
  (h7 : runs_by_boundaries = boundaries * runs_per_boundary)
  (h8 : runs_by_sixes = total_runs - (runs_by_running + runs_by_boundaries))
  (h9 : number_of_sixes = runs_by_sixes / runs_per_six)
  : number_of_sixes = 4 :=
by
  sorry

end number_of_sixes_l132_132566


namespace solve_for_x_l132_132704

theorem solve_for_x : 
  (∀ x : ℝ, x ≠ -2 → (x^2 - x - 2) / (x + 2) = x - 1 ↔ x = 0) := 
by 
  sorry

end solve_for_x_l132_132704


namespace prime_solution_l132_132491

theorem prime_solution (p q : ℕ) (hp : Prime p) (hq : Prime q) (h : q^5 - 2 * p^2 = 1) : p + q = 14 :=
sorry

end prime_solution_l132_132491


namespace rope_total_length_is_54m_l132_132614

noncomputable def totalRopeLength : ℝ :=
  let horizontalDistance : ℝ := 16
  let heightAB : ℝ := 18
  let heightCD : ℝ := 30
  let ropeBC := Real.sqrt (horizontalDistance^2 + (heightCD - heightAB)^2)
  let ropeAC := Real.sqrt (horizontalDistance^2 + heightCD^2)
  ropeBC + ropeAC

theorem rope_total_length_is_54m : totalRopeLength = 54 := sorry

end rope_total_length_is_54m_l132_132614


namespace determine_M_l132_132662

noncomputable def M : Set ℤ :=
  {a | ∃ k : ℕ, k > 0 ∧ 6 = k * (5 - a)}

theorem determine_M : M = {-1, 2, 3, 4} :=
  sorry

end determine_M_l132_132662


namespace abs_le_and_interval_iff_l132_132689

variable (x : ℝ)

theorem abs_le_and_interval_iff :
  (|x - 2| ≤ 5) ↔ (-3 ≤ x ∧ x ≤ 7) :=
by
  sorry

end abs_le_and_interval_iff_l132_132689


namespace semicircle_radius_in_trapezoid_l132_132321

theorem semicircle_radius_in_trapezoid 
  (AB CD : ℝ) (AD BC : ℝ) (r : ℝ)
  (h1 : AB = 27) 
  (h2 : CD = 45) 
  (h3 : AD = 13) 
  (h4 : BC = 15) 
  (h5 : r = 13.5) :
  r = 13.5 :=
by
  sorry  -- Detailed proof steps will go here

end semicircle_radius_in_trapezoid_l132_132321


namespace frequency_of_second_group_l132_132232

theorem frequency_of_second_group (total_capacity : ℕ) (freq_percentage : ℝ)
    (h_capacity : total_capacity = 80)
    (h_percentage : freq_percentage = 0.15) :
    total_capacity * freq_percentage = 12 :=
by
  sorry

end frequency_of_second_group_l132_132232


namespace solve_positive_integer_l132_132030

theorem solve_positive_integer (n : ℕ) (h : ∀ m : ℕ, m > 0 → n^m ≥ m^n) : n = 3 :=
sorry

end solve_positive_integer_l132_132030


namespace planes_parallel_if_any_line_parallel_l132_132262

-- Definitions for Lean statements:
variable (P1 P2 : Set Point)
variable (line : Set Point)

-- Conditions
def is_parallel_to_plane (line : Set Point) (plane : Set Point) : Prop := sorry

def is_parallel_plane (plane1 plane2 : Set Point) : Prop := sorry

-- Lean statement to be proved:
theorem planes_parallel_if_any_line_parallel (h : ∀ line, 
  line ⊆ P1 → is_parallel_to_plane line P2) : is_parallel_plane P1 P2 := sorry

end planes_parallel_if_any_line_parallel_l132_132262


namespace find_a7_a8_l132_132199

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (g : geometric_sequence a q)

def sum_1_2 : ℝ := a 1 + a 2
def sum_3_4 : ℝ := a 3 + a 4

theorem find_a7_a8
  (h1 : sum_1_2 = 30)
  (h2 : sum_3_4 = 60)
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 7 + a 8 = (a 1 + a 2) * (q ^ 6) := 
sorry

end find_a7_a8_l132_132199


namespace boxwoods_shaped_into_spheres_l132_132193

theorem boxwoods_shaped_into_spheres :
  ∀ (total_boxwoods : ℕ) (cost_trimming : ℕ) (cost_shaping : ℕ) (total_charge : ℕ) (x : ℕ),
    total_boxwoods = 30 →
    cost_trimming = 5 →
    cost_shaping = 15 →
    total_charge = 210 →
    30 * 5 + x * 15 = 210 →
    x = 4 :=
by
  intros total_boxwoods cost_trimming cost_shaping total_charge x
  rintro rfl rfl rfl rfl h
  sorry

end boxwoods_shaped_into_spheres_l132_132193


namespace mary_change_in_dollars_l132_132846

theorem mary_change_in_dollars :
  let cost_berries_euros := 7.94
  let cost_peaches_dollars := 6.83
  let exchange_rate := 1.2
  let money_handed_euros := 20
  let money_handed_dollars := 10
  let cost_berries_dollars := cost_berries_euros * exchange_rate
  let total_cost_dollars := cost_berries_dollars + cost_peaches_dollars
  let total_handed_dollars := (money_handed_euros * exchange_rate) + money_handed_dollars
  total_handed_dollars - total_cost_dollars = 17.642 :=
by
  intros
  sorry

end mary_change_in_dollars_l132_132846


namespace vasya_numbers_l132_132075

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l132_132075


namespace equation_has_two_solutions_l132_132857

theorem equation_has_two_solutions : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ∀ x : ℝ, ¬ ( |x - 1| = |x - 2| + |x - 3| ) ↔ (x ≠ x₁ ∧ x ≠ x₂) :=
sorry

end equation_has_two_solutions_l132_132857


namespace exists_five_digit_number_with_property_l132_132045

theorem exists_five_digit_number_with_property :
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ (n^2 % 100000) = n := 
sorry

end exists_five_digit_number_with_property_l132_132045


namespace midpoint_plane_distance_l132_132087

noncomputable def midpoint_distance (A B : ℝ) (dA dB : ℝ) : ℝ :=
  (dA + dB) / 2

theorem midpoint_plane_distance (A B : ℝ) (dA dB : ℝ) (hA : dA = 1) (hB : dB = 3) :
  midpoint_distance A B dA dB = 1 ∨ midpoint_distance A B dA dB = 2 :=
by
  sorry

end midpoint_plane_distance_l132_132087


namespace simplification_of_expression_l132_132677

variable {a b : ℚ}

theorem simplification_of_expression (h1a : a ≠ 0) (h1b : b ≠ 0) (h2 : 3 * a - b / 3 ≠ 0) :
  (3 * a - b / 3)⁻¹ * ( (3 * a)⁻¹ - (b / 3)⁻¹ ) = -(a * b)⁻¹ := 
sorry

end simplification_of_expression_l132_132677


namespace original_ratio_l132_132711

theorem original_ratio (x y : ℤ) (h1 : y = 24) (h2 : (x + 6) / y = 1 / 2) : x / y = 1 / 4 := by
  sorry

end original_ratio_l132_132711


namespace volume_of_pyramid_l132_132766

theorem volume_of_pyramid 
  (QR RS : ℝ) (PT : ℝ) 
  (hQR_pos : 0 < QR) (hRS_pos : 0 < RS) (hPT_pos : 0 < PT)
  (perp1 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * QR) * (x * y) = 0)
  (perp2 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * RS) * (x * y) = 0) :
  QR = 10 -> RS = 5 -> PT = 9 -> 
  (1/3) * QR * RS * PT = 150 :=
by
  sorry

end volume_of_pyramid_l132_132766


namespace sum_three_times_integers_15_to_25_l132_132161

noncomputable def sumArithmeticSequence (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem sum_three_times_integers_15_to_25 :
  let a := 15
  let d := 1
  let n := 25 - 15 + 1
  3 * sumArithmeticSequence a d n = 660 := by
  -- This part can be filled in with the actual proof
  sorry

end sum_three_times_integers_15_to_25_l132_132161


namespace find_pairs_l132_132266

def is_solution_pair (m n : ℕ) : Prop :=
  Nat.lcm m n = 3 * m + 2 * n + 1

theorem find_pairs :
  { pairs : List (ℕ × ℕ) // ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n } :=
by
  let pairs := [(3,10), (4,9)]
  have key : ∀ (m n : ℕ), (m, n) ∈ pairs ↔ is_solution_pair m n := sorry
  exact ⟨pairs, key⟩

end find_pairs_l132_132266


namespace mass_of_compound_l132_132178

-- Constants as per the conditions
def molecular_weight : ℕ := 444           -- The molecular weight in g/mol.
def number_of_moles : ℕ := 6             -- The number of moles.

-- Defining the main theorem we want to prove.
theorem mass_of_compound : (number_of_moles * molecular_weight) = 2664 := by 
  sorry

end mass_of_compound_l132_132178


namespace distance_run_l132_132238

theorem distance_run (D : ℝ) (A_time : ℝ) (B_time : ℝ) (A_beats_B : ℝ) : 
  A_time = 90 ∧ B_time = 180 ∧ A_beats_B = 2250 → D = 2250 :=
by
  sorry

end distance_run_l132_132238


namespace original_population_l132_132228

variable (n : ℝ)

theorem original_population
  (h1 : n + 1500 - 0.15 * (n + 1500) = n - 45) :
  n = 8800 :=
sorry

end original_population_l132_132228


namespace a_greater_than_b_for_n_ge_2_l132_132669

theorem a_greater_than_b_for_n_ge_2 
  (n : ℕ) 
  (hn : n ≥ 2) 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a^n = a + 1) 
  (h2 : b^(2 * n) = b + 3 * a) : 
  a > b := 
  sorry

end a_greater_than_b_for_n_ge_2_l132_132669


namespace simplify_fraction_l132_132901

theorem simplify_fraction (b : ℝ) (h : b ≠ 1) : 
  (b - 1) / (b + b / (b - 1)) = (b - 1) ^ 2 / b ^ 2 := 
by {
  sorry
}

end simplify_fraction_l132_132901


namespace range_of_t_l132_132396

noncomputable def a_n (n : ℕ) (t : ℝ) : ℝ := -n + t
noncomputable def b_n (n : ℕ) : ℝ := 3^(n-3)
noncomputable def c_n (n : ℕ) (t : ℝ) : ℝ := 
  let a := a_n n t 
  let b := b_n n
  (a + b) / 2 + (|a - b|) / 2

theorem range_of_t (t : ℝ) (h : ∀ n : ℕ, n > 0 → c_n n t ≥ c_n 3 t) : 10/3 < t ∧ t < 5 :=
    sorry

end range_of_t_l132_132396


namespace marbles_shared_equally_l132_132219

def initial_marbles_Wolfgang : ℕ := 16
def additional_fraction_Ludo : ℚ := 1/4
def fraction_Michael : ℚ := 2/3

theorem marbles_shared_equally :
  let marbles_Wolfgang := initial_marbles_Wolfgang
  let additional_marbles_Ludo := additional_fraction_Ludo * initial_marbles_Wolfgang
  let marbles_Ludo := initial_marbles_Wolfgang + additional_marbles_Ludo
  let marbles_Wolfgang_Ludo := marbles_Wolfgang + marbles_Ludo
  let marbles_Michael := fraction_Michael * marbles_Wolfgang_Ludo
  let total_marbles := marbles_Wolfgang + marbles_Ludo + marbles_Michael
  let marbles_each := total_marbles / 3
  marbles_each = 20 :=
by
  sorry

end marbles_shared_equally_l132_132219


namespace true_propositions_l132_132748

open Set

theorem true_propositions (M N : Set ℕ) (a b m : ℕ) (h1 : M ⊆ N) 
  (h2 : a > b) (h3 : b > 0) (h4 : m > 0) (p : ∀ x : ℝ, x > 0) :
  (M ⊆ M ∪ N) ∧ ((b + m) / (a + m) > b / a) ∧ 
  ¬(∀ (a b c : ℝ), a = b ↔ a * c ^ 2 = b * c ^ 2) ∧ 
  ¬(∃ x₀ : ℝ, x₀ ≤ 0) := sorry

end true_propositions_l132_132748


namespace prove_a3_l132_132515

variable (a1 a2 a3 a4 : ℕ)
variable (q : ℕ)

-- Definition of the geometric sequence
def geom_seq (n : ℕ) : ℕ :=
  a1 * q^(n-1)

-- Given conditions
def cond1 := geom_seq 4 = 8
def cond2 := (geom_seq 2 + geom_seq 3) / (geom_seq 1 + geom_seq 2) = 2

-- Proving the required condition
theorem prove_a3 : cond1 ∧ cond2 → geom_seq 3 = 4 :=
by
sorry

end prove_a3_l132_132515


namespace rhombus_perimeter_l132_132069

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * (Nat.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 68 :=
by
  sorry

end rhombus_perimeter_l132_132069


namespace eel_count_l132_132606

theorem eel_count 
  (x y z : ℕ)
  (h1 : y + z = 12)
  (h2 : x + z = 14)
  (h3 : x + y = 16) : 
  x + y + z = 21 := 
by 
  sorry

end eel_count_l132_132606


namespace find_m_l132_132058

variables {a1 a2 b1 b2 c1 c2 : ℝ} {m : ℝ}
def vectorA := (3, -2 * m)
def vectorB := (m - 1, 2)
def vectorC := (-2, 1)
def vectorAC := (5, -2 * m - 1)

theorem find_m (h : (5 * (m - 1) + (-2 * m - 1) * 2) = 0) : 
  m = 7 := 
  sorry

end find_m_l132_132058


namespace exists_polynomial_f_divides_f_x2_sub_1_l132_132631

open Polynomial

theorem exists_polynomial_f_divides_f_x2_sub_1 (n : ℕ) :
    ∃ f : Polynomial ℝ, degree f = n ∧ f ∣ (f.comp (X ^ 2 - 1)) :=
by {
  sorry
}

end exists_polynomial_f_divides_f_x2_sub_1_l132_132631


namespace sum_possible_values_A_B_l132_132776

theorem sum_possible_values_A_B : 
  ∀ (A B : ℕ), 
  (0 ≤ A ∧ A ≤ 9) ∧ 
  (0 ≤ B ∧ B ≤ 9) ∧ 
  ∃ k : ℕ, 28 + A + B = 9 * k 
  → (A + B = 8 ∨ A + B = 17) 
  → A + B = 25 :=
by
  sorry

end sum_possible_values_A_B_l132_132776


namespace shopkeeper_milk_sold_l132_132370

theorem shopkeeper_milk_sold :
  let morning_packets := 150
  let morning_250 := 60
  let morning_300 := 40
  let morning_350 := morning_packets - morning_250 - morning_300
  
  let evening_packets := 100
  let evening_400 := evening_packets * 50 / 100
  let evening_500 := evening_packets * 25 / 100
  let evening_450 := evening_packets * 25 / 100

  let morning_milk := morning_250 * 250 + morning_300 * 300 + morning_350 * 350
  let evening_milk := evening_400 * 400 + evening_500 * 500 + evening_450 * 450
  let total_milk := morning_milk + evening_milk

  let remaining_milk := 42000
  let sold_milk := total_milk - remaining_milk

  let ounces_per_mil := 1 / 30
  let sold_milk_ounces := sold_milk * ounces_per_mil

  sold_milk_ounces = 1541.67 := by sorry

end shopkeeper_milk_sold_l132_132370


namespace height_at_15_inches_l132_132613

-- Define the conditions
def parabolic_eq (a x : ℝ) : ℝ := a * x^2 + 24
noncomputable def a : ℝ := -2 / 75
def x : ℝ := 15
def expected_y : ℝ := 18

-- Lean 4 statement
theorem height_at_15_inches :
  parabolic_eq a x = expected_y :=
by
  sorry

end height_at_15_inches_l132_132613


namespace second_bill_late_fee_l132_132411

def first_bill_amount : ℕ := 200
def first_bill_interest_rate : ℝ := 0.10
def first_bill_months : ℕ := 2
def second_bill_amount : ℕ := 130
def second_bill_months : ℕ := 6
def third_bill_first_month_fee : ℕ := 40
def third_bill_second_month_fee : ℕ := 80
def total_amount_owed : ℕ := 1234

theorem second_bill_late_fee (x : ℕ) 
(h : first_bill_amount * (first_bill_interest_rate * first_bill_months) + first_bill_amount + third_bill_first_month_fee + third_bill_second_month_fee + second_bill_amount + second_bill_months * x = total_amount_owed) : x = 124 :=
sorry

end second_bill_late_fee_l132_132411


namespace zero_interval_of_f_l132_132503

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_interval_of_f :
    ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  sorry

end zero_interval_of_f_l132_132503


namespace bacteria_initial_count_l132_132634

theorem bacteria_initial_count (n : ℕ) :
  (∀ t : ℕ, t % 30 = 0 → n * 2^(t / 30) = 262144 → t = 240) → n = 1024 :=
by sorry

end bacteria_initial_count_l132_132634


namespace non_congruent_rectangles_count_l132_132916

theorem non_congruent_rectangles_count :
  let grid_width := 6
  let grid_height := 4
  let axis_aligned_rectangles := (grid_width.choose 2) * (grid_height.choose 2)
  let squares_1x1 := (grid_width - 1) * (grid_height - 1)
  let squares_2x2 := (grid_width - 2) * (grid_height - 2)
  let non_congruent_rectangles := axis_aligned_rectangles - (squares_1x1 + squares_2x2)
  non_congruent_rectangles = 67 := 
by {
  sorry
}

end non_congruent_rectangles_count_l132_132916


namespace increasing_sequence_range_l132_132619

theorem increasing_sequence_range (a : ℝ) (f : ℝ → ℝ) (a_n : ℕ+ → ℝ) :
  (∀ n : ℕ+, a_n n = f n) →
  (∀ n m : ℕ+, n < m → a_n n < a_n m) →
  (∀ x : ℝ, f x = if  x ≤ 7 then (3 - a) * x - 3 else a ^ (x - 6) ) →
  2 < a ∧ a < 3 :=
by
  sorry

end increasing_sequence_range_l132_132619


namespace smallest_degree_measure_for_WYZ_l132_132625

def angle_XYZ : ℝ := 130
def angle_XYW : ℝ := 100
def angle_WYZ : ℝ := angle_XYZ - angle_XYW

theorem smallest_degree_measure_for_WYZ : angle_WYZ = 30 :=
by
  sorry

end smallest_degree_measure_for_WYZ_l132_132625


namespace flu_infection_equation_l132_132134

theorem flu_infection_equation (x : ℝ) :
  (1 + x)^2 = 144 :=
sorry

end flu_infection_equation_l132_132134


namespace one_third_greater_than_333_l132_132616

theorem one_third_greater_than_333 :
  (1 : ℝ) / 3 > (333 : ℝ) / 1000 - 1 / 3000 :=
sorry

end one_third_greater_than_333_l132_132616


namespace vertex_parabola_is_parabola_l132_132092

variables {a c : ℝ} (h_a : 0 < a) (h_c : 0 < c)

theorem vertex_parabola_is_parabola :
  ∀ (x y : ℝ), (∃ b : ℝ, x = -b / (2 * a) ∧ y = a * (-b / (2 * a)) ^ 2 + b * (-b / (2 * a)) + c) ↔ y = -a * x ^ 2 + c :=
by sorry

end vertex_parabola_is_parabola_l132_132092


namespace max_value_of_f_l132_132835

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : ∃ x : ℝ, f x = Real.sqrt 10 :=
sorry

end max_value_of_f_l132_132835


namespace half_angle_second_quadrant_l132_132907

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
    ∃ j : ℤ, (j * π + π / 4 < α / 2 ∧ α / 2 < j * π + π / 2) ∨ (j * π + 5 * π / 4 < α / 2 ∧ α / 2 < (j + 1) * π / 2) :=
sorry

end half_angle_second_quadrant_l132_132907


namespace min_trig_expression_l132_132003

theorem min_trig_expression (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (h_sum : A + B + C = Real.pi) : 
  ∃ (x : ℝ), (x = 16 - 8 * Real.sqrt 2) ∧ (∀ A B C, 0 < A → 0 < B → 0 < C → A + B + C = Real.pi → 
    (1 / (Real.sin A)^2 + 1 / (Real.sin B)^2 + 4 / (1 + Real.sin C)) ≥ x) := 
sorry

end min_trig_expression_l132_132003


namespace Sum_a2_a3_a7_l132_132975

-- Definitions from the conditions
variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers
variable {S : ℕ → ℝ} -- Define the sum of the first n terms as a function from natural numbers to real numbers

-- Given conditions
axiom Sn_formula : ∀ n : ℕ, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))
axiom S7_eq_42 : S 7 = 42

theorem Sum_a2_a3_a7 :
  a 2 + a 3 + a 7 = 18 :=
sorry

end Sum_a2_a3_a7_l132_132975


namespace shaded_area_of_rotated_semicircle_l132_132724

noncomputable def area_of_shaded_region (R : ℝ) (α : ℝ) : ℝ :=
  (1 / 2) * (2 * R) ^ 2 * (α / (2 * Real.pi))

theorem shaded_area_of_rotated_semicircle (R : ℝ) (α : ℝ) (h : α = Real.pi / 9) :
  area_of_shaded_region R α = 2 * Real.pi * R ^ 2 / 9 :=
by
  sorry

end shaded_area_of_rotated_semicircle_l132_132724


namespace solution_set_ineq_l132_132121

noncomputable
def f (x : ℝ) : ℝ := sorry
noncomputable
def g (x : ℝ) : ℝ := sorry

axiom h_f_odd : ∀ x : ℝ, f (-x) = -f x
axiom h_g_even : ∀ x : ℝ, g (-x) = g x
axiom h_deriv_pos : ∀ x : ℝ, x < 0 → deriv f x * g x + f x * deriv g x > 0
axiom h_g_neg_three_zero : g (-3) = 0

theorem solution_set_ineq : { x : ℝ | f x * g x < 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | 0 < x ∧ x < 3 } := 
by sorry

end solution_set_ineq_l132_132121


namespace compute_expression_l132_132731

theorem compute_expression : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end compute_expression_l132_132731


namespace mary_number_l132_132173

-- Definitions for conditions
def has_factor_150 (m : ℕ) : Prop := 150 ∣ m
def is_multiple_of_45 (m : ℕ) : Prop := 45 ∣ m
def in_range (m : ℕ) : Prop := 1000 < m ∧ m < 3000

-- Theorem stating that Mary's number is one of {1350, 1800, 2250, 2700} given the conditions
theorem mary_number 
  (m : ℕ) 
  (h1 : has_factor_150 m)
  (h2 : is_multiple_of_45 m)
  (h3 : in_range m) :
  m = 1350 ∨ m = 1800 ∨ m = 2250 ∨ m = 2700 :=
sorry

end mary_number_l132_132173


namespace bus_stop_time_l132_132559

theorem bus_stop_time (speed_without_stoppages speed_with_stoppages : ℝ) (h1: speed_without_stoppages = 48) (h2: speed_with_stoppages = 24) :
  ∃ (minutes_stopped_per_hour : ℝ), minutes_stopped_per_hour = 30 :=
by
  sorry

end bus_stop_time_l132_132559


namespace wall_clock_time_at_car_5PM_l132_132278

-- Define the initial known conditions
def initial_time : ℕ := 7 -- 7:00 AM
def wall_time_at_10AM : ℕ := 10 -- 10:00 AM
def car_time_at_10AM : ℕ := 11 -- 11:00 AM
def car_time_at_5PM : ℕ := 17 -- 5:00 PM = 17:00 in 24-hour format

-- Define the calculations for the rate of the car clock
def rate_of_car_clock : ℚ := (car_time_at_10AM - initial_time : ℚ) / (wall_time_at_10AM - initial_time : ℚ) -- rate = 4/3

-- Prove the actual time according to the wall clock when the car clock shows 5:00 PM
theorem wall_clock_time_at_car_5PM :
  let elapsed_real_time := (car_time_at_5PM - car_time_at_10AM) * (3 : ℚ) / (4 : ℚ)
  let actual_time := wall_time_at_10AM + elapsed_real_time
  (actual_time : ℚ) = 15 + (15 / 60 : ℚ) := -- 3:15 PM as 15.25 in 24-hour time
by
  sorry

end wall_clock_time_at_car_5PM_l132_132278


namespace math_problem_l132_132312

theorem math_problem (x y : ℝ) (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 :=
sorry

end math_problem_l132_132312


namespace max_difference_second_largest_second_smallest_l132_132597

theorem max_difference_second_largest_second_smallest :
  ∀ (a b c d e f g h : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 ∧
  a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧ f < g ∧ g < h ∧
  a + b + c = 27 ∧
  a + b + c + d + e + f + g + h = 152 ∧
  f + g + h = 87 →
  g - b = 26 :=
by
  intros;
  sorry

end max_difference_second_largest_second_smallest_l132_132597


namespace inequality_3var_l132_132019

theorem inequality_3var (x y z : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y) (h₃ : 0 ≤ z) (h₄ : x * y + y * z + z * x = 1) : 
    1 / (x + y) + 1 / (y + z) + 1 / (z + x) ≥ 5 / 2 :=
sorry

end inequality_3var_l132_132019


namespace min_balls_to_guarantee_18_l132_132464

noncomputable def min_balls_needed {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) : ℕ :=
  95

theorem min_balls_to_guarantee_18 {red green yellow blue white black : ℕ}
    (h_red : red = 30) 
    (h_green : green = 23) 
    (h_yellow : yellow = 21) 
    (h_blue : blue = 17) 
    (h_white : white = 14) 
    (h_black : black = 12) :
  min_balls_needed h_red h_green h_yellow h_blue h_white h_black = 95 :=
  by
  -- Placeholder for the actual proof
  sorry

end min_balls_to_guarantee_18_l132_132464


namespace cosine_inequality_l132_132687

theorem cosine_inequality
  (x y z : ℝ)
  (hx : 0 < x ∧ x < π / 2)
  (hy : 0 < y ∧ y < π / 2)
  (hz : 0 < z ∧ z < π / 2) :
  (x * Real.cos x + y * Real.cos y + z * Real.cos z) / (x + y + z) ≤
  (Real.cos x + Real.cos y + Real.cos z) / 3 := sorry

end cosine_inequality_l132_132687


namespace probability_of_continuous_stripe_pattern_l132_132203

def tetrahedron_stripes := 
  let faces := 4
  let configurations_per_face := 2
  2 ^ faces

def continuous_stripe_probability := 
  let total_configurations := tetrahedron_stripes
  1 / total_configurations * 4 -- Since final favorable outcomes calculation is already given and inferred to be 1/4.
  -- or any other logic that follows here based on problem description but this matches problem's derivation

theorem probability_of_continuous_stripe_pattern : continuous_stripe_probability = 1 / 4 := by
  sorry

end probability_of_continuous_stripe_pattern_l132_132203


namespace smallest_odd_digit_number_gt_1000_mult_5_l132_132127

def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def valid_number (n : ℕ) : Prop :=
  n > 1000 ∧ (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  is_odd_digit d1 ∧ is_odd_digit d2 ∧ is_odd_digit d3 ∧ is_odd_digit d4 ∧ 
  d4 = 5)

theorem smallest_odd_digit_number_gt_1000_mult_5 : ∃ n : ℕ, valid_number n ∧ 
  ∀ m : ℕ, valid_number m → m ≥ n := 
by
  use 1115
  simp [valid_number, is_odd_digit]
  sorry

end smallest_odd_digit_number_gt_1000_mult_5_l132_132127


namespace determine_moles_Al2O3_formed_l132_132373

noncomputable def initial_moles_Al : ℝ := 10
noncomputable def initial_moles_Fe2O3 : ℝ := 6
noncomputable def balanced_eq (moles_Al moles_Fe2O3 moles_Al2O3 moles_Fe : ℝ) : Prop :=
  2 * moles_Al + moles_Fe2O3 = moles_Al2O3 + 2 * moles_Fe

theorem determine_moles_Al2O3_formed :
  ∃ moles_Al2O3 : ℝ, balanced_eq 10 6 moles_Al2O3 (moles_Al2O3 * 2) ∧ moles_Al2O3 = 5 := 
  by 
  sorry

end determine_moles_Al2O3_formed_l132_132373


namespace confectioner_customers_l132_132041

theorem confectioner_customers (x : ℕ) (h : 0 < x) :
  (49 * (392 / x - 6) = 392) → x = 28 :=
by
sorry

end confectioner_customers_l132_132041


namespace shorten_to_sixth_power_l132_132919

theorem shorten_to_sixth_power (x n m p q r : ℕ) (h1 : x > 1000000)
  (h2 : x / 10 = n^2)
  (h3 : n^2 / 10 = m^3)
  (h4 : m^3 / 10 = p^4)
  (h5 : p^4 / 10 = q^5) :
  q^5 / 10 = r^6 :=
sorry

end shorten_to_sixth_power_l132_132919


namespace xyz_sum_fraction_l132_132052

theorem xyz_sum_fraction (a1 a2 a3 b1 b2 b3 c1 c2 c3 a b c : ℤ) 
  (h1 : a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1) = 9)
  (h2 : a * (b2 * c3 - b3 * c2) - a2 * (b * c3 - b3 * c) + a3 * (b * c2 - b2 * c) = 17)
  (h3 : a1 * (b * c3 - b3 * c) - a * (b1 * c3 - b3 * c1) + a3 * (b1 * c - b * c1) = -8)
  (h4 : a1 * (b2 * c - b * c2) - a2 * (b1 * c - b * c1) + a * (b1 * c2 - b2 * c1) = 7)
  (eq1 : a1 * x + a2 * y + a3 * z = a)
  (eq2 : b1 * x + b2 * y + b3 * z = b)
  (eq3 : c1 * x + c2 * y + c3 * z = c)
  : x + y + z = 16 / 9 := 
sorry

end xyz_sum_fraction_l132_132052


namespace ratio_of_width_to_length_l132_132512

variable {w: ℕ}

theorem ratio_of_width_to_length (w: ℕ) (h1: 2*w + 2*10 = 30) (h2: w = 5) :
  ∃ (x y : ℕ), x = 1 ∧ y = 2 ∧ x.gcd y = 1 ∧ w / 10 = 1 / 2 := 
by
  sorry

end ratio_of_width_to_length_l132_132512


namespace find_angle_D_l132_132366

theorem find_angle_D (A B C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = 2 * D)
  (h3 : B = C + 40) : D = 70 := by
  sorry

end find_angle_D_l132_132366


namespace irreducible_fraction_l132_132764

theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end irreducible_fraction_l132_132764


namespace animal_lifespan_probability_l132_132743

theorem animal_lifespan_probability
    (P_B : ℝ) (hP_B : P_B = 0.8)
    (P_A : ℝ) (hP_A : P_A = 0.4)
    : (P_A / P_B = 0.5) :=
by
    sorry

end animal_lifespan_probability_l132_132743


namespace min_max_value_expression_l132_132982

theorem min_max_value_expression
  (x1 x2 x3 : ℝ) 
  (hx : x1 + x2 + x3 = 1)
  (hx1 : 0 ≤ x1)
  (hx2 : 0 ≤ x2)
  (hx3 : 0 ≤ x3) :
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5) = 1 := 
sorry

end min_max_value_expression_l132_132982


namespace total_students_l132_132495

theorem total_students (x : ℕ) (h1 : (x + 6) / (2*x + 6) = 2 / 3) : 2 * x + 6 = 18 :=
sorry

end total_students_l132_132495


namespace regression_line_passes_through_center_l132_132379

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 1.5 * x - 15

-- Define the condition of the sample center point
def sample_center (x_bar y_bar : ℝ) : Prop :=
  y_bar = regression_eq x_bar

-- The proof goal
theorem regression_line_passes_through_center (x_bar y_bar : ℝ) (h : sample_center x_bar y_bar) :
  y_bar = 1.5 * x_bar - 15 :=
by
  -- Using the given condition as hypothesis
  exact h

end regression_line_passes_through_center_l132_132379


namespace suraj_average_increase_l132_132324

namespace SurajAverage

theorem suraj_average_increase (A : ℕ) (h : (16 * A + 112) / 17 = A + 6) : (A + 6) = 16 :=
  by
  sorry

end SurajAverage

end suraj_average_increase_l132_132324


namespace diff_of_squares_div_l132_132918

-- Definitions from the conditions
def a : ℕ := 125
def b : ℕ := 105

-- The main statement to be proved
theorem diff_of_squares_div {a b : ℕ} (h1 : a = 125) (h2 : b = 105) : (a^2 - b^2) / 20 = 230 := by
  sorry

end diff_of_squares_div_l132_132918


namespace value_of_a_l132_132488

theorem value_of_a (a : ℝ) (h : (a, 0) ∈ {p : ℝ × ℝ | p.2 = p.1 + 8}) : a = -8 :=
sorry

end value_of_a_l132_132488


namespace geometric_sequence_common_ratio_l132_132462

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : ∀ n, 0 < a n)
  (h3 : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2) : q = 3 := by
  sorry

end geometric_sequence_common_ratio_l132_132462


namespace sampling_method_is_systematic_l132_132575

def conveyor_belt_sampling (interval: ℕ) (product_picking: ℕ → ℕ) : Prop :=
  ∀ (n: ℕ), product_picking n = n * interval

theorem sampling_method_is_systematic
  (interval: ℕ)
  (product_picking: ℕ → ℕ)
  (h: conveyor_belt_sampling interval product_picking) :
  interval = 30 → product_picking = systematic_sampling := 
sorry

end sampling_method_is_systematic_l132_132575


namespace passengers_on_board_l132_132870

/-- 
Given the fractions of passengers from different continents and remaining 42 passengers,
show that the total number of passengers P is 240.
-/
theorem passengers_on_board :
  ∃ P : ℕ,
    (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) ∧ P = 240 :=
by
  let P := 240
  have h : (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) := sorry
  exact ⟨P, h, rfl⟩

end passengers_on_board_l132_132870


namespace parabola_vertex_sum_l132_132277

theorem parabola_vertex_sum (p q r : ℝ) 
  (h1 : ∃ (a b c : ℝ), ∀ (x : ℝ), a * x ^ 2 + b * x + c = y)
  (h2 : ∃ (vertex_x vertex_y : ℝ), vertex_x = 3 ∧ vertex_y = -1)
  (h3 : ∀ (x : ℝ), y = p * x ^ 2 + q * x + r)
  (h4 : y = p * (0 - 3) ^ 2 + r - 1)
  (h5 : y = 8)
  : p + q + r = 3 := 
by
  sorry

end parabola_vertex_sum_l132_132277


namespace union_of_sets_l132_132297

def A : Set Int := {-1, 2, 3, 5}
def B : Set Int := {2, 4, 5}

theorem union_of_sets :
  A ∪ B = {-1, 2, 3, 4, 5} := by
  sorry

end union_of_sets_l132_132297


namespace area_of_quadrilateral_AXYD_l132_132996

open Real

noncomputable def area_quadrilateral_AXYD: ℝ :=
  let A := (0, 0)
  let B := (20, 0)
  let C := (20, 12)
  let D := (0, 12)
  let Z := (20, 30)
  let E := (6, 6)
  let X := (2.5, 0)
  let Y := (9.5, 12)
  let base1 := (B.1 - X.1)  -- Length from B to X
  let base2 := (Y.1 - A.1)  -- Length from D to Y
  let height := (C.2 - A.2) -- Height common for both bases
  (base1 + base2) * height / 2

theorem area_of_quadrilateral_AXYD : area_quadrilateral_AXYD = 72 :=
by
  sorry

end area_of_quadrilateral_AXYD_l132_132996


namespace angle_conversion_l132_132459

theorem angle_conversion :
  (12 * (Real.pi / 180)) = (Real.pi / 15) := by
  sorry

end angle_conversion_l132_132459


namespace sharks_at_other_beach_is_12_l132_132700

-- Define the conditions
def cape_may_sharks := 32
def sharks_other_beach (S : ℕ) := 2 * S + 8

-- Statement to prove
theorem sharks_at_other_beach_is_12 (S : ℕ) (h : cape_may_sharks = sharks_other_beach S) : S = 12 :=
by
  -- Sorry statement to skip the proof part
  sorry

end sharks_at_other_beach_is_12_l132_132700


namespace part1_part2_l132_132765

-- Definitions for Part (1)
def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }
def B (m : ℝ) : Set ℝ := { x | -1 ≤ x ∧ x ≤ 3 }

-- Part (1) Statement
theorem part1 (m : ℝ) (hm : m = 2) : A ∩ ((compl B m)) = {x | (-2 ≤ x ∧ x < -1) ∨ (3 < x ∧ x ≤ 4)} := 
by
  sorry

-- Definitions for Part (2)
def B_interval (m : ℝ) : Set ℝ := { x | (1 - m) ≤ x ∧ x ≤ (1 + m) }

-- Part (2) Statement
theorem part2 (m : ℝ) (h : ∀ x, (x ∈ A → x ∈ B_interval m)) : 0 < m ∧ m < 3 := 
by
  sorry

end part1_part2_l132_132765


namespace Freddie_ratio_l132_132497

noncomputable def Veronica_distance : ℕ := 1000

noncomputable def Freddie_distance (F : ℕ) : Prop :=
  1000 + 12000 = 5 * F - 2000

theorem Freddie_ratio (F : ℕ) (h : Freddie_distance F) :
  F / Veronica_distance = 3 := by
  sorry

end Freddie_ratio_l132_132497


namespace exist_functions_fg_neq_f1f1_g1g1_l132_132580

-- Part (a)
theorem exist_functions_fg :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, (f ∘ g) x = (g ∘ f) x) ∧ 
    (∀ x, (f ∘ f) x = (g ∘ g) x) ∧ 
    (∀ x, f x ≠ g x) := 
sorry

-- Part (b)
theorem neq_f1f1_g1g1 
  (f1 g1 : ℝ → ℝ)
  (H_comm : ∀ x, (f1 ∘ g1) x = (g1 ∘ f1) x)
  (H_neq: ∀ x, f1 x ≠ g1 x) :
  ∀ x, (f1 ∘ f1) x ≠ (g1 ∘ g1) x :=
sorry

end exist_functions_fg_neq_f1f1_g1g1_l132_132580


namespace ratio_eq_one_l132_132848

theorem ratio_eq_one {a b : ℝ} (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0 ∧ b ≠ 0) : (a^2 / 5) / (b^3 / 4) = 1 :=
by
  sorry

end ratio_eq_one_l132_132848


namespace intersection_of_A_and_B_l132_132211

open Set

def A : Set ℝ := { x | 3 * x + 2 > 0 }
def B : Set ℝ := { x | (x + 1) * (x - 3) > 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | x > 3 } :=
by 
  sorry

end intersection_of_A_and_B_l132_132211


namespace two_zeros_range_l132_132101

noncomputable def f (x k : ℝ) : ℝ := x * Real.exp x - k

theorem two_zeros_range (k : ℝ) : -1 / Real.exp 1 < k ∧ k < 0 → ∃! x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 k = 0 ∧ f x2 k = 0 :=
by
  sorry

end two_zeros_range_l132_132101


namespace inequality_pos_distinct_l132_132054

theorem inequality_pos_distinct (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    (a + b + c) * (1/a + 1/b + 1/c) > 9 := by
  sorry

end inequality_pos_distinct_l132_132054


namespace find_value_in_box_l132_132523

theorem find_value_in_box (x : ℕ) :
  10 * 20 * 30 * 40 * 50 = 100 * 2 * 300 * 4 * x ↔ x = 50 := by
  sorry

end find_value_in_box_l132_132523


namespace beetles_eaten_per_day_l132_132684
-- Import the Mathlib library

-- Declare the conditions as constants
def bird_eats_beetles_per_day : Nat := 12
def snake_eats_birds_per_day : Nat := 3
def jaguar_eats_snakes_per_day : Nat := 5
def number_of_jaguars : Nat := 6

-- Define the theorem and provide the expected proof
theorem beetles_eaten_per_day :
  12 * (3 * (5 * 6)) = 1080 := by
  sorry

end beetles_eaten_per_day_l132_132684


namespace evaluate_fraction_sum_l132_132241

theorem evaluate_fraction_sum (a b c : ℝ) (h : a ≠ 40) (h_a : b ≠ 75) (h_b : c ≠ 85)
  (h_cond : (a / (40 - a)) + (b / (75 - b)) + (c / (85 - c)) = 8) :
  (8 / (40 - a)) + (15 / (75 - b)) + (17 / (85 - c)) = 40 := 
sorry

end evaluate_fraction_sum_l132_132241


namespace set_intersection_l132_132955

theorem set_intersection (A B : Set ℝ) 
  (hA : A = { x : ℝ | 0 < x ∧ x < 5 }) 
  (hB : B = { x : ℝ | -1 ≤ x ∧ x < 4 }) : 
  (A ∩ B) = { x : ℝ | 0 < x ∧ x < 4 } :=
by
  sorry

end set_intersection_l132_132955


namespace relatively_prime_m_n_l132_132230

noncomputable def probability_of_distinct_real_solutions : ℝ :=
  let b := (1 : ℝ)
  if 1 ≤ b ∧ b ≤ 25 then 1 else 0

theorem relatively_prime_m_n : ∃ m n : ℕ, 
  Nat.gcd m n = 1 ∧ 
  (1 : ℝ) = (m : ℝ) / (n : ℝ) ∧ m + n = 2 := 
by
  sorry

end relatively_prime_m_n_l132_132230


namespace number_of_students_l132_132279

-- Defining the parameters and conditions
def passing_score : ℕ := 65
def average_score_whole_class : ℕ := 66
def average_score_passed : ℕ := 71
def average_score_failed : ℕ := 56
def increased_score : ℕ := 5
def post_increase_average_passed : ℕ := 75
def post_increase_average_failed : ℕ := 59
def num_students_lb : ℕ := 15 
def num_students_ub : ℕ := 30

-- Lean statement to prove the number of students in the class
theorem number_of_students (x y n : ℕ) 
  (h1 : average_score_passed * x + average_score_failed * y = average_score_whole_class * (x + y))
  (h2 : (average_score_whole_class + increased_score) * (x + y) = post_increase_average_passed * (x + n) + post_increase_average_failed * (y - n))
  (h3 : num_students_lb < x + y ∧ x + y < num_students_ub)
  (h4 : x = 2 * y)
  (h5 : y = 4 * n) : x + y = 24 :=
sorry

end number_of_students_l132_132279


namespace cube_surface_area_l132_132085

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) : 6 * (edge_length * edge_length) = 2400 := by
  -- We state our theorem and assumptions here
  sorry

end cube_surface_area_l132_132085


namespace no_real_solutions_l132_132573

theorem no_real_solutions :
  ∀ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) ≠ 1 / 8) :=
by
  intro x
  sorry

end no_real_solutions_l132_132573


namespace quiz_answer_key_count_l132_132150

theorem quiz_answer_key_count :
  let true_false_possibilities := 6  -- Combinations for 3 T/F questions where not all are same
  let multiple_choice_possibilities := 4^3  -- 4 choices for each of 3 multiple-choice questions
  true_false_possibilities * multiple_choice_possibilities = 384 := by
  sorry

end quiz_answer_key_count_l132_132150


namespace rectangle_perimeter_l132_132659

noncomputable def perimeter_rectangle (x y : ℝ) : ℝ := 2 * (x + y)

theorem rectangle_perimeter
  (x y a b : ℝ)
  (H1 : x * y = 2006)
  (H2 : x + y = 2 * a)
  (H3 : x^2 + y^2 = 4 * (a^2 - b^2))
  (b_val : b = Real.sqrt 1003)
  (a_val : a = 2 * Real.sqrt 1003) :
  perimeter_rectangle x y = 8 * Real.sqrt 1003 := by
  sorry

end rectangle_perimeter_l132_132659


namespace positive_integer_perfect_square_l132_132563

theorem positive_integer_perfect_square (n : ℕ) (h1: n > 0) (h2 : ∃ k : ℕ, n^2 - 19 * n - 99 = k^2) : n = 199 :=
sorry

end positive_integer_perfect_square_l132_132563


namespace game_cost_l132_132483

theorem game_cost
    (total_earnings : ℕ)
    (expenses : ℕ)
    (games_bought : ℕ)
    (remaining_money := total_earnings - expenses)
    (cost_per_game := remaining_money / games_bought)
    (h1 : total_earnings = 104)
    (h2 : expenses = 41)
    (h3 : games_bought = 7) :
    cost_per_game = 9 := by
  sorry

end game_cost_l132_132483


namespace correct_calculation_l132_132620

variable (a : ℝ)

theorem correct_calculation : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_l132_132620


namespace A_E_not_third_l132_132532

-- Define the runners and their respective positions.
inductive Runner
| A : Runner
| B : Runner
| C : Runner
| D : Runner
| E : Runner
open Runner

variable (position : Runner → Nat)

-- Conditions
axiom A_beats_B : position A < position B
axiom C_beats_D : position C < position D
axiom B_beats_E : position B < position E
axiom D_after_A_before_B : position A < position D ∧ position D < position B

-- Prove that A and E cannot be in third place.
theorem A_E_not_third : position A ≠ 3 ∧ position E ≠ 3 :=
sorry

end A_E_not_third_l132_132532


namespace solve_for_b_l132_132231

theorem solve_for_b (b : ℝ) (hb : b + ⌈b⌉ = 17.8) : b = 8.8 := 
by sorry

end solve_for_b_l132_132231


namespace textbook_weight_difference_l132_132252

variable (chemWeight : ℝ) (geomWeight : ℝ)

def chem_weight := chemWeight = 7.12
def geom_weight := geomWeight = 0.62

theorem textbook_weight_difference : chemWeight - geomWeight = 6.50 :=
by
  sorry

end textbook_weight_difference_l132_132252


namespace number_of_integers_with_6_or_7_as_digit_in_base9_l132_132894

/-- 
  There are 729 smallest positive integers written in base 9.
  We want to determine how many of these integers use the digits 6 or 7 (or both) at least once.
-/
theorem number_of_integers_with_6_or_7_as_digit_in_base9 : 
  ∃ n : ℕ, n = 729 ∧ ∃ m : ℕ, m = n - 7^3 := sorry

end number_of_integers_with_6_or_7_as_digit_in_base9_l132_132894


namespace f_2017_plus_f_2016_l132_132838

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_even_shift : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom f_at_neg1 : f (-1) = -1

theorem f_2017_plus_f_2016 : f 2017 + f 2016 = 1 :=
by
  sorry

end f_2017_plus_f_2016_l132_132838


namespace greatest_possible_q_minus_r_l132_132963

theorem greatest_possible_q_minus_r :
  ∃ (q r : ℕ), 945 = 21 * q + r ∧ 0 ≤ r ∧ r < 21 ∧ q - r = 45 :=
by
  sorry

end greatest_possible_q_minus_r_l132_132963


namespace descent_time_l132_132286

-- Definitions based on conditions
def time_to_top : ℝ := 4
def avg_speed_up : ℝ := 2.625
def avg_speed_total : ℝ := 3.5
def distance_to_top : ℝ := avg_speed_up * time_to_top -- 10.5 km
def total_distance : ℝ := 2 * distance_to_top       -- 21 km

-- Theorem statement: the time to descend (t_down) should be 2 hours
theorem descent_time (t_down : ℝ) : 
  avg_speed_total * (time_to_top + t_down) = total_distance →
  t_down = 2 := 
by 
  -- skip the proof
  sorry

end descent_time_l132_132286


namespace exists_five_distinct_natural_numbers_product_eq_1000_l132_132129

theorem exists_five_distinct_natural_numbers_product_eq_1000 :
  ∃ (a b c d e : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 1000 := sorry

end exists_five_distinct_natural_numbers_product_eq_1000_l132_132129


namespace marcel_potatoes_eq_l132_132322

-- Define the given conditions
def marcel_corn := 10
def dale_corn := marcel_corn / 2
def dale_potatoes := 8
def total_vegetables := 27

-- Define the fact that they bought 27 vegetables in total
def total_corn := marcel_corn + dale_corn
def total_potatoes := total_vegetables - total_corn

-- State the theorem
theorem marcel_potatoes_eq :
  (total_potatoes - dale_potatoes) = 4 :=
by
  -- Lean proof would go here
  sorry

end marcel_potatoes_eq_l132_132322


namespace kite_perimeter_l132_132437

-- Given the kite's diagonals, shorter sides, and longer sides
def diagonals : ℕ × ℕ := (12, 30)
def shorter_sides : ℕ := 10
def longer_sides : ℕ := 15

-- Problem statement: Prove that the perimeter is 50 inches
theorem kite_perimeter (diag1 diag2 short_len long_len : ℕ) 
                       (h_diag : diag1 = 12 ∧ diag2 = 30)
                       (h_short : short_len = 10)
                       (h_long : long_len = 15) : 
                       2 * short_len + 2 * long_len = 50 :=
by
  -- We provide no proof, only the statement
  sorry

end kite_perimeter_l132_132437


namespace maximize_ab_l132_132180

theorem maximize_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ab + a + b = 1) : 
  ab ≤ 3 - 2 * Real.sqrt 2 :=
sorry

end maximize_ab_l132_132180


namespace triangle_angle_A_l132_132162

theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) (hC : C = Real.pi / 6) (hCos : c = 2 * a * Real.cos B) : A = (5 * Real.pi) / 12 :=
  sorry

end triangle_angle_A_l132_132162


namespace total_playtime_l132_132465

-- Conditions
def lena_playtime_hours : ℝ := 3.5
def minutes_per_hour : ℝ := 60
def lena_playtime_minutes : ℝ := lena_playtime_hours * minutes_per_hour
def brother_playtime_extra_minutes : ℝ := 17
def brother_playtime_minutes : ℝ := lena_playtime_minutes + brother_playtime_extra_minutes

-- Proof problem
theorem total_playtime : lena_playtime_minutes + brother_playtime_minutes = 437 := by
  sorry

end total_playtime_l132_132465


namespace min_value_of_expression_l132_132043

-- Define the conditions in the problem
def conditions (m n : ℝ) : Prop :=
  (2 * m + n = 2) ∧ (m > 0) ∧ (n > 0)

-- Define the problem statement
theorem min_value_of_expression (m n : ℝ) (h : conditions m n) : 
  (∀ m n, conditions m n → (1 / m + 2 / n) ≥ 4) :=
by 
  sorry

end min_value_of_expression_l132_132043


namespace common_root_l132_132556

variable (m x : ℝ)
variable (h₁ : m * x - 1000 = 1021)
variable (h₂ : 1021 * x = m - 1000 * x)

theorem common_root (hx : m * x - 1000 = 1021 ∧ 1021 * x = m - 1000 * x) : m = 2021 ∨ m = -2021 := sorry

end common_root_l132_132556


namespace tickets_spent_on_beanie_l132_132643

variable (initial_tickets won_tickets tickets_left tickets_spent: ℕ)

theorem tickets_spent_on_beanie
  (h1 : initial_tickets = 49)
  (h2 : won_tickets = 6)
  (h3 : tickets_left = 30)
  (h4 : tickets_spent = initial_tickets + won_tickets - tickets_left) :
  tickets_spent = 25 :=
by
  sorry

end tickets_spent_on_beanie_l132_132643


namespace cost_price_per_meter_l132_132313

def selling_price_for_85_meters : ℝ := 8925
def profit_per_meter : ℝ := 25
def number_of_meters : ℝ := 85

theorem cost_price_per_meter : (selling_price_for_85_meters - profit_per_meter * number_of_meters) / number_of_meters = 80 := by
  sorry

end cost_price_per_meter_l132_132313


namespace ratio_is_five_to_three_l132_132455

variable (g b : ℕ)

def girls_more_than_boys : Prop := g - b = 6
def total_pupils : Prop := g + b = 24
def ratio_girls_to_boys : ℚ := g / b

theorem ratio_is_five_to_three (h1 : girls_more_than_boys g b) (h2 : total_pupils g b) : ratio_girls_to_boys g b = 5 / 3 := by
  sorry

end ratio_is_five_to_three_l132_132455


namespace train_speed_equivalent_l132_132517

def length_train1 : ℝ := 180
def length_train2 : ℝ := 160
def speed_train1 : ℝ := 60 
def crossing_time_sec : ℝ := 12.239020878329734

noncomputable def speed_train2 (length1 length2 speed1 time : ℝ) : ℝ :=
  let total_length_km := (length1 + length2) / 1000
  let time_hr := time / 3600
  let relative_speed := total_length_km / time_hr
  relative_speed - speed1

theorem train_speed_equivalent :
  speed_train2 length_train1 length_train2 speed_train1 crossing_time_sec = 40 :=
by
  simp [length_train1, length_train2, speed_train1, crossing_time_sec, speed_train2]
  sorry

end train_speed_equivalent_l132_132517


namespace minimum_value_of_expression_l132_132320

theorem minimum_value_of_expression (p q r s t u : ℝ) 
  (hpqrsu_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ 0 < t ∧ 0 < u) 
  (sum_eq : p + q + r + s + t + u = 8) : 
  98 ≤ (2 / p + 4 / q + 9 / r + 16 / s + 25 / t + 36 / u) :=
sorry

end minimum_value_of_expression_l132_132320


namespace probability_of_passing_test_l132_132023

theorem probability_of_passing_test (p : ℝ) (h : p + p * (1 - p) + p * (1 - p)^2 = 0.784) : p = 0.4 :=
sorry

end probability_of_passing_test_l132_132023


namespace nes_sale_price_l132_132827

noncomputable def price_of_nes
    (snes_value : ℝ)
    (tradein_rate : ℝ)
    (cash_given : ℝ)
    (change_received : ℝ)
    (game_value : ℝ) : ℝ :=
  let tradein_credit := snes_value * tradein_rate
  let additional_cost := cash_given - change_received
  let total_cost := tradein_credit + additional_cost
  let nes_price := total_cost - game_value
  nes_price

theorem nes_sale_price 
  (snes_value : ℝ)
  (tradein_rate : ℝ)
  (cash_given : ℝ)
  (change_received : ℝ)
  (game_value : ℝ) :
  snes_value = 150 → tradein_rate = 0.80 → cash_given = 80 → change_received = 10 → game_value = 30 →
  price_of_nes snes_value tradein_rate cash_given change_received game_value = 160 := by
  intros
  sorry

end nes_sale_price_l132_132827


namespace ratio_john_to_jenna_l132_132021

theorem ratio_john_to_jenna (J : ℕ) 
  (h1 : 100 - J - 40 = 35) : 
  J = 25 ∧ (J / 100 = 1 / 4) := 
by
  sorry

end ratio_john_to_jenna_l132_132021


namespace car_speed_decrease_l132_132421

theorem car_speed_decrease (d : ℝ) (speed_first : ℝ) (distance_fifth : ℝ) (time_interval : ℝ) :
  speed_first = 45 ∧ distance_fifth = 4.4 ∧ time_interval = 8 / 60 ∧ speed_first - 4 * d = distance_fifth / time_interval -> d = 3 :=
by
  intros h
  obtain ⟨_, _, _, h_eq⟩ := h
  sorry

end car_speed_decrease_l132_132421


namespace y_value_on_line_l132_132367

theorem y_value_on_line (x y : ℝ) (k : ℝ → ℝ)
  (h1 : k 0 = 0)
  (h2 : ∀ x, k x = (1/5) * x)
  (hx1 : k x = 1)
  (hx2 : k 5 = y) :
  y = 1 :=
sorry

end y_value_on_line_l132_132367


namespace which_is_system_lin_eq_l132_132331

def option_A : Prop := ∀ (x : ℝ), x - 1 = 2 * x
def option_B : Prop := ∀ (x y : ℝ), x - 1/y = 1
def option_C : Prop := ∀ (x z : ℝ), x + z = 3
def option_D : Prop := ∀ (x y z : ℝ), x - y + z = 1

theorem which_is_system_lin_eq (hA : option_A) (hB : option_B) (hC : option_C) (hD : option_D) :
    (∀ (x z : ℝ), x + z = 3) :=
by
  sorry

end which_is_system_lin_eq_l132_132331


namespace dice_probability_sum_12_l132_132442

open Nat

/-- Probability that the sum of three six-faced dice rolls equals 12 is 10 / 216 --/
theorem dice_probability_sum_12 : 
  let outcomes := 6^3
  let favorable := 10
  (favorable : ℚ) / outcomes = 10 / 216 := 
by
  let outcomes := 6^3
  let favorable := 10
  sorry

end dice_probability_sum_12_l132_132442


namespace find_x_given_total_area_l132_132804

theorem find_x_given_total_area :
  ∃ x : ℝ, (16 * x^2 + 36 * x^2 + 6 * x^2 + 3 * x^2 = 1100) ∧ (x = Real.sqrt (1100 / 61)) :=
sorry

end find_x_given_total_area_l132_132804


namespace probability_defective_units_l132_132106

theorem probability_defective_units (X : ℝ) (hX : X > 0) :
  let defectA := (14 / 2000) * (0.40 * X)
  let defectB := (9 / 1500) * (0.35 * X)
  let defectC := (7 / 1000) * (0.25 * X)
  let total_defects := defectA + defectB + defectC
  let total_units := X
  let probability := total_defects / total_units
  probability = 0.00665 :=
by
  sorry

end probability_defective_units_l132_132106


namespace pet_store_initial_house_cats_l132_132163

theorem pet_store_initial_house_cats
    (H : ℕ)
    (h1 : 13 + H - 10 = 8) :
    H = 5 :=
by
  sorry

end pet_store_initial_house_cats_l132_132163


namespace new_persons_joined_l132_132761

theorem new_persons_joined :
  ∀ (A : ℝ) (N : ℕ) (avg_new : ℝ) (avg_combined : ℝ), 
  N = 15 → avg_new = 15 → avg_combined = 15.5 → 1 = (N * avg_combined + N * avg_new - 232.5) / (avg_combined - avg_new) := by
  intros A N avg_new avg_combined
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end new_persons_joined_l132_132761


namespace y_intercept_of_parallel_line_l132_132159

-- Define the conditions for the problem
def line_parallel (m1 m2 : ℝ) : Prop := 
  m1 = m2

def point_on_line (m : ℝ) (b x1 y1 : ℝ) : Prop := 
  y1 = m * x1 + b

-- Define the main problem statement
theorem y_intercept_of_parallel_line (m b1 b2 x1 y1 : ℝ) 
  (h1 : line_parallel m 3) 
  (h2 : point_on_line m b1 x1 y1) 
  (h3 : x1 = 1) 
  (h4 : y1 = 2) 
  : b1 = -1 :=
sorry

end y_intercept_of_parallel_line_l132_132159


namespace plant_lamp_arrangement_count_l132_132882

theorem plant_lamp_arrangement_count :
  let basil_plants := 2
  let aloe_plants := 2
  let white_lamps := 3
  let red_lamps := 3
  (∀ plant, plant = basil_plants ∨ plant = aloe_plants)
  ∧ (∀ lamp, lamp = white_lamps ∨ lamp = red_lamps)
  → (∀ plant, ∃ lamp, plant → lamp)
  → ∃ count, count = 50 := 
by
  sorry

end plant_lamp_arrangement_count_l132_132882


namespace trapezoid_EFBA_area_l132_132721

theorem trapezoid_EFBA_area {a : ℚ} (AE BF : ℚ) (area_ABCD : ℚ) (column_areas : List ℚ)
  (h_grid : column_areas = [a, 2 * a, 4 * a, 8 * a])
  (h_total_area : 3 * (a + 2 * a + 4 * a + 8 * a) = 48)
  (h_AE : AE = 2)
  (h_BF : BF = 4) :
  let AFGB_area := 15 * a
  let triangle_EF_area := 7 * a
  let total_trapezoid_area := AFGB_area + (triangle_EF_area / 2)
  total_trapezoid_area = 352 / 15 :=
by
  sorry

end trapezoid_EFBA_area_l132_132721


namespace condition_sufficiency_not_necessity_l132_132115

variable {x y : ℝ}

theorem condition_sufficiency_not_necessity (hx : x ≥ 0) (hy : y ≥ 0) :
  (xy > 0 → |x + y| = |x| + |y|) ∧ (|x + y| = |x| + |y| → xy ≥ 0) :=
sorry

end condition_sufficiency_not_necessity_l132_132115


namespace units_digit_of_modifiedLucas_L20_eq_d_l132_132327

def modifiedLucas : ℕ → ℕ
| 0 => 3
| 1 => 2
| n + 2 => 2 * modifiedLucas (n + 1) + modifiedLucas n

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_modifiedLucas_L20_eq_d :
  ∃ d, units_digit (modifiedLucas (modifiedLucas 20)) = d :=
by
  sorry

end units_digit_of_modifiedLucas_L20_eq_d_l132_132327


namespace Jana_taller_than_Kelly_l132_132929

-- Definitions and given conditions
def Jess_height := 72
def Jana_height := 74
def Kelly_height := Jess_height - 3

-- Proof statement
theorem Jana_taller_than_Kelly : Jana_height - Kelly_height = 5 := by
  sorry

end Jana_taller_than_Kelly_l132_132929


namespace Amanda_second_day_tickets_l132_132954

/-- Amanda's ticket sales problem set up -/
def Amanda_total_tickets := 80
def Amanda_first_day_tickets := 5 * 4
def Amanda_third_day_tickets := 28

theorem Amanda_second_day_tickets :
  ∃ (tickets_sold_second_day : ℕ), tickets_sold_second_day = 32 :=
by
  let first_day := Amanda_first_day_tickets
  let third_day := Amanda_third_day_tickets
  let needed_before_third := Amanda_total_tickets - third_day
  let second_day := needed_before_third - first_day
  use second_day
  sorry

end Amanda_second_day_tickets_l132_132954


namespace largest_sample_number_l132_132060

theorem largest_sample_number (n : ℕ) (start interval total : ℕ) (h1 : start = 7) (h2 : interval = 25) (h3 : total = 500) (h4 : n = total / interval) : 
(start + interval * (n - 1) = 482) :=
sorry

end largest_sample_number_l132_132060


namespace tom_jerry_coffee_total_same_amount_total_coffee_l132_132280

noncomputable def total_coffee_drunk (x : ℚ) : ℚ := 
  let jerry_coffee := 1.25 * x
  let tom_drinks := (2/3) * x
  let jerry_drinks := (2/3) * jerry_coffee
  let jerry_remainder := (5/12) * x
  let jerry_gives_tom := (5/48) * x + 3
  tom_drinks + jerry_gives_tom

theorem tom_jerry_coffee_total (x : ℚ) : total_coffee_drunk x = jerry_drinks + (1.25 * x - jerry_gives_tom) := sorry

theorem same_amount_total_coffee (x : ℚ) 
  (h : total_coffee_drunk x = (5/4) * x - ((5/48) * x + 3)) : 
  (1.25 * x + x = 36) :=
by sorry

end tom_jerry_coffee_total_same_amount_total_coffee_l132_132280


namespace cost_of_each_scoop_l132_132333

theorem cost_of_each_scoop (x : ℝ) 
  (pierre_scoops : ℝ := 3)
  (mom_scoops : ℝ := 4)
  (total_bill : ℝ := 14) 
  (h : 7 * x = total_bill) :
  x = 2 :=
by 
  sorry

end cost_of_each_scoop_l132_132333


namespace smallest_number_first_digit_is_9_l132_132800

def sum_of_digits (n : Nat) : Nat :=
  (n.digits 10).sum

def first_digit (n : Nat) : Nat :=
  n.digits 10 |>.headD 0

theorem smallest_number_first_digit_is_9 :
  ∃ N : Nat, sum_of_digits N = 2020 ∧ ∀ M : Nat, (sum_of_digits M = 2020 → N ≤ M) ∧ first_digit N = 9 :=
by
  sorry

end smallest_number_first_digit_is_9_l132_132800


namespace stateA_selection_percentage_l132_132540

theorem stateA_selection_percentage :
  ∀ (P : ℕ), (∀ (n : ℕ), n = 8000) → (7 * 8000 / 100 = P * 8000 / 100 + 80) → P = 6 := by
  -- The proof steps go here
  sorry

end stateA_selection_percentage_l132_132540


namespace sandwich_cost_l132_132596

theorem sandwich_cost (total_cost soda_cost sandwich_count soda_count : ℝ) :
  total_cost = 8.38 → soda_cost = 0.87 → sandwich_count = 2 → soda_count = 4 → 
  (∀ S, sandwich_count * S + soda_count * soda_cost = total_cost → S = 2.45) :=
by
  intros h_total h_soda h_sandwich_count h_soda_count S h_eqn
  sorry

end sandwich_cost_l132_132596


namespace acme_vowel_soup_l132_132428

-- Define the set of vowels
def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

-- Define the number of each vowel
def num_vowels (v : Char) : ℕ := 5

-- Define a function to count the number of five-letter words
def count_five_letter_words : ℕ :=
  (vowels.card) ^ 5

-- Theorem to be proven
theorem acme_vowel_soup :
  count_five_letter_words = 3125 :=
by
  -- Proof omitted
  sorry

end acme_vowel_soup_l132_132428


namespace find_monthly_income_l132_132474

-- Define the percentages spent on various categories
def household_items_percentage : ℝ := 0.35
def clothing_percentage : ℝ := 0.18
def medicines_percentage : ℝ := 0.06
def entertainment_percentage : ℝ := 0.11
def transportation_percentage : ℝ := 0.12
def mutual_fund_percentage : ℝ := 0.05
def taxes_percentage : ℝ := 0.07

-- Define the savings amount
def savings_amount : ℝ := 12500

-- Total spent percentage
def total_spent_percentage := household_items_percentage + clothing_percentage + medicines_percentage + entertainment_percentage + transportation_percentage + mutual_fund_percentage + taxes_percentage

-- Percentage saved
def savings_percentage := 1 - total_spent_percentage

-- Prove that Ajay's monthly income is Rs. 208,333.33
theorem find_monthly_income (I : ℝ) (h : I * savings_percentage = savings_amount) : I = 208333.33 := by
  sorry

end find_monthly_income_l132_132474


namespace cosine_squared_is_half_l132_132332

def sides_of_triangle (p q r : ℝ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q > r ∧ q + r > p ∧ r + p > q

noncomputable def cosine_squared (p q r : ℝ) : ℝ :=
  ((p^2 + q^2 - r^2) / (2 * p * q))^2

theorem cosine_squared_is_half (p q r : ℝ) (h : sides_of_triangle p q r) 
  (h_eq : p^4 + q^4 + r^4 = 2 * r^2 * (p^2 + q^2)) : cosine_squared p q r = 1 / 2 :=
by
  sorry

end cosine_squared_is_half_l132_132332


namespace max_candy_remainder_l132_132358

theorem max_candy_remainder (x : ℕ) : x % 11 < 11 ∧ (∀ r : ℕ, r < 11 → x % 11 ≤ r) → x % 11 = 10 := 
sorry

end max_candy_remainder_l132_132358


namespace sequence_inequality_l132_132453

-- Define the problem
theorem sequence_inequality (a : ℕ → ℕ) (h0 : ∀ n, 0 < a n) (h1 : a 1 > a 0) (h2 : ∀ n ≥ 2, a n = 3 * a (n-1) - 2 * a (n-2)) : a 100 > 2^99 :=
by
  sorry

end sequence_inequality_l132_132453


namespace days_to_shovel_l132_132026

-- Defining conditions as formal statements
def original_task_time := 10
def original_task_people := 10
def original_task_weight := 10000
def new_task_weight := 40000
def new_task_people := 5

-- Definition of rate in terms of weight, people and time
def rate_per_person (total_weight : ℕ) (total_people : ℕ) (total_time : ℕ) : ℕ :=
  total_weight / total_people / total_time

-- Theorem statement to prove
theorem days_to_shovel (t : ℕ) :
  (rate_per_person original_task_weight original_task_people original_task_time) * new_task_people * t = new_task_weight := sorry

end days_to_shovel_l132_132026


namespace greatest_value_of_x_l132_132830

theorem greatest_value_of_x (x : ℝ) : 
  (∃ (M : ℝ), (∀ y : ℝ, (y ^ 2 - 14 * y + 45 <= 0) → y <= M) ∧ (M ^ 2 - 14 * M + 45 <= 0)) ↔ M = 9 :=
by
  sorry

end greatest_value_of_x_l132_132830


namespace selection_schemes_l132_132768

theorem selection_schemes (boys girls : ℕ) (hb : boys = 4) (hg : girls = 2) :
  (boys * girls = 8) :=
by
  -- Proof goes here
  intros
  sorry

end selection_schemes_l132_132768


namespace number_of_int_pairs_l132_132355

theorem number_of_int_pairs (x y : ℤ) (h : x^2 + 2 * y^2 < 25) : 
  ∃ S : Finset (ℤ × ℤ), S.card = 55 ∧ ∀ (a : ℤ × ℤ), a ∈ S ↔ a.1^2 + 2 * a.2^2 < 25 :=
sorry

end number_of_int_pairs_l132_132355


namespace sequence_term_position_l132_132965

theorem sequence_term_position (n : ℕ) (h : 2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) : n = 7 :=
sorry

end sequence_term_position_l132_132965


namespace apprentice_time_l132_132785

theorem apprentice_time
  (x y : ℝ)
  (h1 : 7 * x + 4 * y = 5 / 9)
  (h2 : 11 * x + 8 * y = 17 / 18)
  (hy : y > 0) :
  1 / y = 24 :=
by
  sorry

end apprentice_time_l132_132785


namespace tank_depth_is_six_l132_132434

-- Definitions derived from the conditions
def tank_length : ℝ := 25
def tank_width : ℝ := 12
def plastering_cost_per_sq_meter : ℝ := 0.45
def total_cost : ℝ := 334.8

-- Compute the surface area to be plastered
def surface_area (d : ℝ) : ℝ := (tank_length * tank_width) + 2 * (tank_length * d) + 2 * (tank_width * d)

-- Equation relating the plastering cost to the surface area
def cost_equation (d : ℝ) : ℝ := plastering_cost_per_sq_meter * (surface_area d)

-- The mathematical result we need to prove
theorem tank_depth_is_six : ∃ d : ℝ, cost_equation d = total_cost ∧ d = 6 := by
  sorry

end tank_depth_is_six_l132_132434


namespace geom_prog_min_third_term_l132_132418

theorem geom_prog_min_third_term :
  ∃ (d : ℝ), (-4 + 10 * Real.sqrt 6 = d ∨ -4 - 10 * Real.sqrt 6 = d) ∧
  (∀ x, x = 37 + 2 * d → x ≤ 29 - 20 * Real.sqrt 6) := 
sorry

end geom_prog_min_third_term_l132_132418


namespace matrix_expression_l132_132392

variable {F : Type} [Field F] {n : Type} [Fintype n] [DecidableEq n]
variable (B : Matrix n n F)

-- Suppose B is invertible
variable [Invertible B]

-- Condition given in the problem
theorem matrix_expression (h : (B - 3 • (1 : Matrix n n F)) * (B - 5 • (1 : Matrix n n F)) = 0) :
  B + 10 • (B⁻¹) = 10 • (B⁻¹) + (32 / 3 : F) • (1 : Matrix n n F) :=
sorry

end matrix_expression_l132_132392


namespace minimum_value_y_l132_132665

theorem minimum_value_y (x y : ℕ) (h1 : x + y = 64) (h2 : 3 * x + 4 * y = 200) : y = 8 :=
by
  sorry

end minimum_value_y_l132_132665


namespace squirrel_pine_cones_l132_132795

theorem squirrel_pine_cones (x y : ℕ) (hx : 26 - 10 + 9 + (x + 14)/2 = x/2) (hy : y + 5 - 18 + 9 + (x + 14)/2 = x/2) :
  x = 86 := sorry

end squirrel_pine_cones_l132_132795


namespace problem_statement_l132_132987

def p (x : ℝ) : ℝ := x^2 - x + 1

theorem problem_statement (α : ℝ) (h : p (p (p (p α))) = 0) :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 :=
by
  sorry

end problem_statement_l132_132987


namespace discount_percentage_for_two_pairs_of_jeans_l132_132489

theorem discount_percentage_for_two_pairs_of_jeans
  (price_per_pair : ℕ := 40)
  (price_for_three_pairs : ℕ := 112)
  (discount : ℕ := 8)
  (original_price_for_two_pairs : ℕ := price_per_pair * 2)
  (discount_percentage : ℕ := (discount * 100) / original_price_for_two_pairs) :
  discount_percentage = 10 := 
by
  sorry

end discount_percentage_for_two_pairs_of_jeans_l132_132489


namespace find_m_for_even_function_l132_132057

def f (x : ℝ) (m : ℝ) := x^2 + (m - 1) * x + 3

theorem find_m_for_even_function : ∃ m : ℝ, (∀ x : ℝ, f (-x) m = f x m) ∧ m = 1 :=
sorry

end find_m_for_even_function_l132_132057


namespace curved_surface_area_of_sphere_l132_132708

theorem curved_surface_area_of_sphere (r : ℝ) (h : r = 4) : 4 * π * r^2 = 64 * π :=
by
  rw [h, sq]
  norm_num
  sorry

end curved_surface_area_of_sphere_l132_132708


namespace flood_monitoring_technology_l132_132000

def geographicInformationTechnologies : Type := String

def RemoteSensing : geographicInformationTechnologies := "Remote Sensing"
def GlobalPositioningSystem : geographicInformationTechnologies := "Global Positioning System"
def GeographicInformationSystem : geographicInformationTechnologies := "Geographic Information System"
def DigitalEarth : geographicInformationTechnologies := "Digital Earth"

def effectiveFloodMonitoring (tech1 tech2 : geographicInformationTechnologies) : Prop :=
  (tech1 = RemoteSensing ∧ tech2 = GeographicInformationSystem) ∨ 
  (tech1 = GeographicInformationSystem ∧ tech2 = RemoteSensing)

theorem flood_monitoring_technology :
  effectiveFloodMonitoring RemoteSensing GeographicInformationSystem :=
by
  sorry

end flood_monitoring_technology_l132_132000


namespace visible_sides_probability_l132_132018

theorem visible_sides_probability
  (r : ℝ)
  (side_length : ℝ := 4)
  (probability : ℝ := 3 / 4) :
  r = 8 * Real.sqrt 3 / 3 :=
sorry

end visible_sides_probability_l132_132018


namespace compare_f_values_l132_132715

noncomputable def f (x : ℝ) : ℝ := Real.sin x - x

theorem compare_f_values : 
  f (-π / 4) > f 1 ∧ f 1 > f (π / 3) := 
sorry

end compare_f_values_l132_132715


namespace size_ratio_l132_132592

variable {A B C : ℝ} -- Declaring that A, B, and C are real numbers (their sizes)
variable (h1 : A = 3 * B) -- A is three times the size of B
variable (h2 : B = (1 / 2) * C) -- B is half the size of C

theorem size_ratio (h1 : A = 3 * B) (h2 : B = (1 / 2) * C) : A / C = 1.5 :=
by
  sorry -- Proof goes here, to be completed

end size_ratio_l132_132592


namespace solution_for_x_l132_132938

theorem solution_for_x (x : ℝ) : 
  (∀ (y : ℝ), 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3 / 2 :=
by 
  -- Proof should go here
  sorry

end solution_for_x_l132_132938


namespace lucy_withdrawal_l132_132217

-- Given conditions
def initial_balance : ℕ := 65
def deposit : ℕ := 15
def final_balance : ℕ := 76

-- Define balance before withdrawal
def balance_before_withdrawal := initial_balance + deposit

-- Theorem to prove
theorem lucy_withdrawal : balance_before_withdrawal - final_balance = 4 :=
by sorry

end lucy_withdrawal_l132_132217


namespace intersection_correct_l132_132576

def A : Set ℝ := { x : ℝ | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x : ℝ | 2 < x ∧ x < 4 }

theorem intersection_correct : A ∩ B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_correct_l132_132576


namespace exists_real_ge_3_l132_132247

-- Definition of the existential proposition
theorem exists_real_ge_3 : ∃ x : ℝ, x ≥ 3 :=
sorry

end exists_real_ge_3_l132_132247


namespace discount_percentage_l132_132005

theorem discount_percentage (cp mp pm : ℤ) (x : ℤ) 
    (Hcp : cp = 160) 
    (Hmp : mp = 240) 
    (Hpm : pm = 20) 
    (Hcondition : mp * (100 - x) = cp * (100 + pm)) : 
  x = 20 := 
  sorry

end discount_percentage_l132_132005


namespace redistribute_marbles_l132_132717

theorem redistribute_marbles :
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  (d + m + p + v) / n = 15 :=
by
  let d := 14
  let m := 20
  let p := 19
  let v := 7
  let n := 4
  sorry

end redistribute_marbles_l132_132717


namespace expand_polynomial_l132_132565

theorem expand_polynomial (x : ℂ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := 
sorry

end expand_polynomial_l132_132565


namespace find_y_l132_132042

theorem find_y : 
  let mean1 := (7 + 9 + 14 + 23) / 4
  let mean2 := (18 + y) / 2
  mean1 = mean2 → y = 8.5 :=
by
  let y := 8.5
  sorry

end find_y_l132_132042


namespace students_study_both_l132_132376

-- Define variables and conditions
variable (total_students G B G_and_B : ℕ)
variable (G_percent B_percent : ℝ)
variable (total_students_eq : total_students = 300)
variable (G_percent_eq : G_percent = 0.8)
variable (B_percent_eq : B_percent = 0.5)
variable (G_eq : G = G_percent * total_students)
variable (B_eq : B = B_percent * total_students)
variable (students_eq : total_students = G + B - G_and_B)

-- Theorem statement
theorem students_study_both :
  G_and_B = 90 :=
by
  sorry

end students_study_both_l132_132376


namespace evaluate_expression_to_zero_l132_132539

-- Assuming 'm' is an integer with specific constraints and providing a proof that the expression evaluates to 0 when m = -1
theorem evaluate_expression_to_zero (m : ℤ) (h1 : -2 ≤ m) (h2 : m ≤ 2) (h3 : m ≠ 0) (h4 : m ≠ 1) (h5 : m ≠ 2) (h6 : m ≠ -2) : 
  (m = -1) → ((m / (m - 2) - 4 / (m ^ 2 - 2 * m)) / (m + 2) / (m ^ 2 - m)) = 0 := 
by
  intro hm_eq_neg1
  sorry

end evaluate_expression_to_zero_l132_132539


namespace system_of_equations_solution_l132_132063

theorem system_of_equations_solution (x y z : ℝ) (hx : x = Real.exp (Real.log y))
(hy : y = Real.exp (Real.log z)) (hz : z = Real.exp (Real.log x)) : x = y ∧ y = z ∧ z = x ∧ x = Real.exp 1 :=
by
  sorry

end system_of_equations_solution_l132_132063


namespace factorize_expr_l132_132904

theorem factorize_expr (x y : ℝ) : x^3 - 4 * x * y^2 = x * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end factorize_expr_l132_132904


namespace evaluate_expression_l132_132265

theorem evaluate_expression : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end evaluate_expression_l132_132265


namespace profit_per_meter_correct_l132_132786

noncomputable def total_selling_price := 6788
noncomputable def num_meters := 78
noncomputable def cost_price_per_meter := 58.02564102564102
noncomputable def total_cost_price := 4526 -- rounded total
noncomputable def total_profit := 2262 -- calculated total profit
noncomputable def profit_per_meter := 29

theorem profit_per_meter_correct :
  (total_selling_price - total_cost_price) / num_meters = profit_per_meter :=
by
  sorry

end profit_per_meter_correct_l132_132786


namespace sum_remainder_l132_132195

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 14) (h2 : b % 30 = 5) (h3 : c % 30 = 18) : 
  (a + b + c) % 30 = 7 :=
by
  sorry

end sum_remainder_l132_132195


namespace rectangle_area_error_l132_132276

theorem rectangle_area_error
  (L W : ℝ)
  (measured_length : ℝ := 1.15 * L)
  (measured_width : ℝ := 1.20 * W)
  (true_area : ℝ := L * W)
  (measured_area : ℝ := measured_length * measured_width)
  (percentage_error : ℝ := ((measured_area - true_area) / true_area) * 100) :
  percentage_error = 38 :=
by
  sorry

end rectangle_area_error_l132_132276


namespace percentage_conversion_l132_132007

-- Define the condition
def decimal_fraction : ℝ := 0.05

-- Define the target percentage
def percentage : ℝ := 5

-- State the theorem
theorem percentage_conversion (df : ℝ) (p : ℝ) (h1 : df = 0.05) (h2 : p = 5) : df * 100 = p :=
by
  rw [h1, h2]
  sorry

end percentage_conversion_l132_132007


namespace roots_reciprocal_l132_132302

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 - 1 = 0) (h2 : x2^2 - 3 * x2 - 1 = 0) 
                         (h_sum : x1 + x2 = 3) (h_prod : x1 * x2 = -1) :
  (1 / x1) + (1 / x2) = -3 :=
by
  sorry

end roots_reciprocal_l132_132302


namespace John_lost_3_ebook_readers_l132_132372

-- Definitions based on the conditions
def A : Nat := 50  -- Anna bought 50 eBook readers
def J : Nat := A - 15  -- John bought 15 less than Anna
def total : Nat := 82  -- Total eBook readers now

-- The number of eBook readers John has after the loss:
def J_after_loss : Nat := total - A

-- The number of eBook readers John lost:
def John_loss : Nat := J - J_after_loss

theorem John_lost_3_ebook_readers : John_loss = 3 :=
by
  sorry

end John_lost_3_ebook_readers_l132_132372


namespace defective_units_l132_132961

-- Conditions given in the problem
variable (D : ℝ) (h1 : 0.05 * D = 0.35)

-- The percent of the units produced that are defective is 7%
theorem defective_units (h1 : 0.05 * D = 0.35) : D = 7 := sorry

end defective_units_l132_132961


namespace woman_away_time_l132_132399

noncomputable def angle_hour_hand (n : ℝ) : ℝ := 150 + n / 2
noncomputable def angle_minute_hand (n : ℝ) : ℝ := 6 * n

theorem woman_away_time : 
  (∀ n : ℝ, abs (angle_hour_hand n - angle_minute_hand n) = 120) → 
  abs ((540 / 11 : ℝ) - (60 / 11 : ℝ)) = 43.636 :=
by sorry

end woman_away_time_l132_132399


namespace lasagna_pieces_l132_132787

theorem lasagna_pieces (m a k r l : ℕ → ℝ)
  (hm : m 1 = 1)                -- Manny's consumption
  (ha : a 0 = 0)                -- Aaron's consumption
  (hk : ∀ n, k n = 2 * (m 1))   -- Kai's consumption
  (hr : ∀ n, r n = (1 / 2) * (m 1)) -- Raphael's consumption
  (hl : ∀ n, l n = 2 + (r n))   -- Lisa's consumption
  : m 1 + a 0 + k 1 + r 1 + l 1 = 6 :=
by
  -- Proof goes here
  sorry

end lasagna_pieces_l132_132787


namespace max_notebooks_l132_132759

-- Definitions based on the conditions
def joshMoney : ℕ := 1050
def notebookCost : ℕ := 75

-- Statement to prove
theorem max_notebooks (x : ℕ) : notebookCost * x ≤ joshMoney → x ≤ 14 := by
  -- Placeholder for the proof
  sorry

end max_notebooks_l132_132759


namespace number_of_pencils_purchased_l132_132973

variable {total_pens : ℕ} (total_cost : ℝ) (avg_price_pencil avg_price_pen : ℝ)

theorem number_of_pencils_purchased 
  (h1 : total_pens = 30)
  (h2 : total_cost = 570)
  (h3 : avg_price_pencil = 2.00)
  (h4 : avg_price_pen = 14)
  : 
  ∃ P : ℕ, P = 75 :=
by
  sorry

end number_of_pencils_purchased_l132_132973


namespace rectangles_in_grid_l132_132985

-- Define a function that calculates the number of rectangles formed
def number_of_rectangles (n m : ℕ) : ℕ :=
  ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4

-- Prove that the number_of_rectangles function correctly calculates the number of rectangles given n and m 
theorem rectangles_in_grid (n m : ℕ) :
  number_of_rectangles n m = ((m + 2) * (m + 1) * (n + 2) * (n + 1)) / 4 := 
by
  sorry

end rectangles_in_grid_l132_132985


namespace greg_spent_on_shirt_l132_132300

-- Define the conditions in Lean
variables (S H : ℤ)
axiom condition1 : H = 2 * S + 9
axiom condition2 : S + H = 300

-- State the theorem to prove
theorem greg_spent_on_shirt : S = 97 :=
by
  sorry

end greg_spent_on_shirt_l132_132300


namespace rectangle_area_l132_132328

theorem rectangle_area (w L : ℝ) (h1 : L = w^2) (h2 : L + w = 25) : 
  L * w = (Real.sqrt 101 - 1)^3 / 8 := 
sorry

end rectangle_area_l132_132328


namespace max_value_sqrt_sum_l132_132635

open Real

noncomputable def max_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) : ℝ :=
  sqrt (3 * x + 1) + sqrt (3 * y + 1) + sqrt (3 * z + 1)

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h_sum : x + y + z = 7) :
  max_sqrt_sum x y z h1 h2 h3 h_sum ≤ 3 * sqrt 8 :=
sorry

end max_value_sqrt_sum_l132_132635


namespace opposite_of_x_abs_of_x_recip_of_x_l132_132736

noncomputable def x : ℝ := 1 - Real.sqrt 2

theorem opposite_of_x : -x = Real.sqrt 2 - 1 := 
by sorry

theorem abs_of_x : |x| = Real.sqrt 2 - 1 :=
by sorry

theorem recip_of_x : 1/x = -1 - Real.sqrt 2 :=
by sorry

end opposite_of_x_abs_of_x_recip_of_x_l132_132736


namespace difference_between_x_and_y_l132_132944

theorem difference_between_x_and_y (x y : ℕ) (h₁ : 3 ^ x * 4 ^ y = 59049) (h₂ : x = 10) : x - y = 10 := by
  sorry

end difference_between_x_and_y_l132_132944


namespace tetrahedron_condition_proof_l132_132574

/-- Define the conditions for the necessary and sufficient condition for each k -/
def tetrahedron_condition (a : ℝ) (k : ℕ) : Prop :=
  match k with
  | 1 => a < Real.sqrt 3
  | 2 => Real.sqrt (2 - Real.sqrt 3) < a ∧ a < Real.sqrt (2 + Real.sqrt 3)
  | 3 => a < Real.sqrt 3
  | 4 => a > Real.sqrt (2 - Real.sqrt 3)
  | 5 => a > 1 / Real.sqrt 3
  | _ => False -- not applicable for other values of k

/-- Prove that the condition is valid for given a and k -/
theorem tetrahedron_condition_proof (a : ℝ) (k : ℕ) : tetrahedron_condition a k := 
  by
  sorry

end tetrahedron_condition_proof_l132_132574


namespace part1_part2_l132_132198

def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

theorem part1 (x : ℝ) : f x (-1) ≤ 0 ↔ x ≤ -1/3 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2_l132_132198


namespace chocolates_exceeding_200_l132_132829

-- Define the initial amount of chocolates
def initial_chocolates : ℕ := 3

-- Define the function that computes the amount of chocolates on the nth day
def chocolates_on_day (n : ℕ) : ℕ := initial_chocolates * 3 ^ (n - 1)

-- Define the proof problem
theorem chocolates_exceeding_200 : ∃ (n : ℕ), chocolates_on_day n > 200 :=
by
  -- Proof required here
  sorry

end chocolates_exceeding_200_l132_132829


namespace prob_exactly_one_hits_prob_at_least_one_hits_l132_132305

noncomputable def prob_A_hits : ℝ := 1 / 2
noncomputable def prob_B_hits : ℝ := 1 / 3
noncomputable def prob_A_misses : ℝ := 1 - prob_A_hits
noncomputable def prob_B_misses : ℝ := 1 - prob_B_hits

theorem prob_exactly_one_hits :
  (prob_A_hits * prob_B_misses) + (prob_A_misses * prob_B_hits) = 1 / 2 :=
by sorry

theorem prob_at_least_one_hits :
  1 - (prob_A_misses * prob_B_misses) = 2 / 3 :=
by sorry

end prob_exactly_one_hits_prob_at_least_one_hits_l132_132305


namespace marissas_sunflower_height_in_meters_l132_132413

-- Define the conversion factors
def inches_per_foot : ℝ := 12
def cm_per_inch : ℝ := 2.54
def cm_per_meter : ℝ := 100

-- Define the given data
def sister_height_feet : ℝ := 4.15
def additional_height_cm : ℝ := 37
def height_difference_inches : ℝ := 63

-- Calculate the height of Marissa's sunflower in meters
theorem marissas_sunflower_height_in_meters :
  let sister_height_inches := sister_height_feet * inches_per_foot
  let sister_height_cm := sister_height_inches * cm_per_inch
  let total_sister_height_cm := sister_height_cm + additional_height_cm
  let height_difference_cm := height_difference_inches * cm_per_inch
  let marissas_sunflower_height_cm := total_sister_height_cm + height_difference_cm
  let marissas_sunflower_height_m := marissas_sunflower_height_cm / cm_per_meter
  marissas_sunflower_height_m = 3.23512 :=
by
  sorry

end marissas_sunflower_height_in_meters_l132_132413


namespace correct_histogram_height_representation_l132_132008

   def isCorrectHeightRepresentation (heightRep : String) : Prop :=
     heightRep = "ratio of the frequency of individuals in that group within the sample to the class interval"

   theorem correct_histogram_height_representation :
     isCorrectHeightRepresentation "ratio of the frequency of individuals in that group within the sample to the class interval" :=
   by 
     sorry
   
end correct_histogram_height_representation_l132_132008


namespace beyonce_total_songs_l132_132660

theorem beyonce_total_songs :
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  total_songs = 140 := by
  let singles := 12
  let albums := 4
  let songs_per_album := 18 + 14
  let total_album_songs := albums * songs_per_album
  let total_songs := total_album_songs + singles
  sorry

end beyonce_total_songs_l132_132660


namespace parabola_c_value_l132_132473

theorem parabola_c_value (b c : ℝ)
  (h1 : 3 = 2^2 + b * 2 + c)
  (h2 : 6 = 5^2 + b * 5 + c) :
  c = -13 :=
by
  -- Proof would follow here
  sorry

end parabola_c_value_l132_132473


namespace shopkeeper_loss_percentage_l132_132112

theorem shopkeeper_loss_percentage {cp sp : ℝ} (h1 : cp = 100) (h2 : sp = cp * 1.1) (h_loss : sp * 0.33 = cp * (1 - x / 100)) :
  x = 70 :=
by
  sorry

end shopkeeper_loss_percentage_l132_132112


namespace coordinates_of_B_l132_132726

theorem coordinates_of_B (a : ℝ) (h : a - 2 = 0) : (a + 2, a - 1) = (4, 1) :=
by
  sorry

end coordinates_of_B_l132_132726


namespace initial_fund_is_890_l132_132586

-- Given Conditions
def initial_fund (n : ℕ) : ℝ := 60 * n - 10
def bonus_given (n : ℕ) : ℝ := 50 * n
def remaining_fund (initial : ℝ) (bonus : ℝ) : ℝ := initial - bonus

-- Proof problem: Prove that the initial amount equals $890 under the given constraints
theorem initial_fund_is_890 :
  ∃ n : ℕ, 
    initial_fund n = 890 ∧ 
    initial_fund n - bonus_given n = 140 :=
by
  sorry

end initial_fund_is_890_l132_132586


namespace circle_to_ellipse_scaling_l132_132356

theorem circle_to_ellipse_scaling :
  ∀ (x' y' : ℝ), (4 * x')^2 + y'^2 = 16 → x'^2 / 16 + y'^2 / 4 = 1 :=
by
  intro x' y'
  intro h
  sorry

end circle_to_ellipse_scaling_l132_132356


namespace handshakes_count_l132_132391

def num_teams : ℕ := 4
def players_per_team : ℕ := 2
def total_players : ℕ := num_teams * players_per_team
def shakeable_players (total : ℕ) : ℕ := total * (total - players_per_team) / 2

theorem handshakes_count :
  shakeable_players total_players = 24 :=
by
  sorry

end handshakes_count_l132_132391


namespace part1_part2_l132_132993

noncomputable def f (a x : ℝ) : ℝ := a * x - a * Real.log x - Real.exp x / x

theorem part1 (a : ℝ) :
  (∀ x > 0, f a x < 0) → a < Real.exp 1 :=
sorry

theorem part2 (a : ℝ) (x1 x2 x3 : ℝ) :
  (∀ x, f a x = 0 → x = x1 ∨ x = x2 ∨ x = x3) ∧
  f a x1 + f a x2 + f a x3 ≤ 3 * Real.exp 2 - Real.exp 1 →
  Real.exp 1 < a ∧ a ≤ Real.exp 2 :=
sorry

end part1_part2_l132_132993


namespace tank_width_problem_l132_132984

noncomputable def tank_width (cost_per_sq_meter : ℚ) (total_cost : ℚ) (length depth : ℚ) : ℚ :=
  let total_cost_in_paise := total_cost * 100
  let total_area := total_cost_in_paise / cost_per_sq_meter
  let w := (total_area - (2 * length * depth) - (2 * depth * 6)) / (length + 2 * depth)
  w

theorem tank_width_problem :
  tank_width 55 409.20 25 6 = 12 := 
by 
  sorry

end tank_width_problem_l132_132984


namespace expression_equals_41_l132_132244

theorem expression_equals_41 (x : ℝ) (h : 3*x^2 + 9*x + 5 ≠ 0) : 
  (3*x^2 + 9*x + 15) / (3*x^2 + 9*x + 5) = 41 :=
by
  sorry

end expression_equals_41_l132_132244


namespace total_pastries_l132_132763

-- Defining the initial conditions
def Grace_pastries : ℕ := 30
def Calvin_pastries : ℕ := Grace_pastries - 5
def Phoebe_pastries : ℕ := Grace_pastries - 5
def Frank_pastries : ℕ := Calvin_pastries - 8

-- The theorem we want to prove
theorem total_pastries : 
  Calvin_pastries + Phoebe_pastries + Frank_pastries + Grace_pastries = 97 := by
  sorry

end total_pastries_l132_132763


namespace range_of_m_l132_132791

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + 2 ≥ m

def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → -(7 - 3*m)^x > -(7 - 3*m)^y

theorem range_of_m (m : ℝ) :
  (proposition_p m ∧ ¬ proposition_q m) ∨ (¬ proposition_p m ∧ proposition_q m) ↔ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l132_132791


namespace smallest_solution_is_39_over_8_l132_132477

noncomputable def smallest_solution (x : ℝ) : Prop :=
  (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x ≠ 0) ∧ (x ≠ 3)

theorem smallest_solution_is_39_over_8 : ∃ x > 0, smallest_solution x ∧ x = 39 / 8 :=
by
  sorry

end smallest_solution_is_39_over_8_l132_132477


namespace probability_not_snowing_l132_132222

variable (P_snowing : ℚ)
variable (h : P_snowing = 2/5)

theorem probability_not_snowing (P_not_snowing : ℚ) : 
  P_not_snowing = 3 / 5 :=
by 
  -- sorry to skip the proof
  sorry

end probability_not_snowing_l132_132222


namespace dvd_packs_l132_132416

theorem dvd_packs (cost_per_pack : ℕ) (discount_per_pack : ℕ) (money_available : ℕ) 
  (h_cost : cost_per_pack = 107) 
  (h_discount : discount_per_pack = 106) 
  (h_money : money_available = 93) : 
  (money_available / (cost_per_pack - discount_per_pack)) = 93 := 
by 
  -- Implementation of the proof goes here
  sorry

end dvd_packs_l132_132416


namespace solve_for_m_l132_132401

def A := {x : ℝ | x^2 + 3*x - 10 ≤ 0}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem solve_for_m (m : ℝ) (h : B m ⊆ A) : m < 2 :=
by
  sorry

end solve_for_m_l132_132401


namespace proof_value_g_expression_l132_132336

noncomputable def g : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom g_invertible : ∀ x, g (g_inv x) = x ∧ g_inv (g x) = x
axiom g_table : ∀ x, (x = 1 → g x = 4) ∧ (x = 2 → g x = 5) ∧ (x = 3 → g x = 7) ∧ (x = 4 → g x = 9) ∧ (x = 5 → g x = 10)

theorem proof_value_g_expression :
  g (g 2) + g (g_inv 9) + g_inv (g_inv 7) = 21 :=
by
  sorry

end proof_value_g_expression_l132_132336


namespace g_at_2_eq_9_l132_132591

def g (x : ℝ) : ℝ := x^2 + 3 * x - 1

theorem g_at_2_eq_9 : g 2 = 9 := by
  sorry

end g_at_2_eq_9_l132_132591


namespace find_alpha_l132_132971

-- Define the problem in Lean terms
variable (x y α : ℝ)

-- Conditions
def condition1 : Prop := 3 + α + y = 4 + α + x
def condition2 : Prop := 1 + x + 3 + 3 + α + y + 4 + 1 = 2 * (4 + α + x)

-- The theorem to prove
theorem find_alpha (h1 : condition1 x y α) (h2 : condition2 x y α) : α = 5 := 
  sorry

end find_alpha_l132_132971


namespace problem_part1_problem_part2_problem_part3_l132_132714

noncomputable def f (x : ℝ) : ℝ := 2^x + 2^(-x)

theorem problem_part1 : f 1 = 5 / 2 ∧ f 2 = 17 / 4 := 
by
  sorry

theorem problem_part2 : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

theorem problem_part3 : ∀ x1 x2 : ℝ, x1 < x2 → x1 < 0 → x2 < 0 → f x1 > f x2 :=
by
  sorry

end problem_part1_problem_part2_problem_part3_l132_132714


namespace determine_a_l132_132027

theorem determine_a (a : ℕ) : 
  (2 * 10^10 + a ) % 11 = 0 ∧ 0 ≤ a ∧ a < 11 → a = 9 :=
by
  sorry

end determine_a_l132_132027


namespace arithmetic_seq_a12_l132_132946

theorem arithmetic_seq_a12 (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 4 = 1)
  (h2 : a 7 + a 9 = 16)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 12 = 15 :=
by sorry

end arithmetic_seq_a12_l132_132946


namespace evaluate_K_l132_132966

theorem evaluate_K : ∃ K : ℕ, 32^2 * 4^4 = 2^K ∧ K = 18 := by
  use 18
  sorry

end evaluate_K_l132_132966


namespace find_C_work_rate_l132_132218

-- Conditions
def A_work_rate := 1 / 4
def B_work_rate := 1 / 6

-- Combined work rate of A and B
def AB_work_rate := A_work_rate + B_work_rate

-- Total work rate when C is assisting, completing in 2 days
def total_work_rate_of_ABC := 1 / 2

theorem find_C_work_rate : ∃ c : ℕ, (AB_work_rate + 1 / c = total_work_rate_of_ABC) ∧ c = 12 :=
by
  -- To complete the proof, we solve the equation for c
  sorry

end find_C_work_rate_l132_132218


namespace circle_center_sum_is_one_l132_132273

def circle_center_sum (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 + 4 * x - 6 * y = 3) → ((h = -2) ∧ (k = 3))

theorem circle_center_sum_is_one :
  ∀ h k : ℝ, circle_center_sum h k → h + k = 1 :=
by
  intros h k hc
  sorry

end circle_center_sum_is_one_l132_132273


namespace Gwen_avg_speed_trip_l132_132452

theorem Gwen_avg_speed_trip : 
  ∀ (d1 d2 s1 s2 t1 t2 : ℝ), 
  d1 = 40 → d2 = 40 → s1 = 15 → s2 = 30 →
  d1 / s1 = t1 → d2 / s2 = t2 →
  (d1 + d2) / (t1 + t2) = 20 :=
by 
  intros d1 d2 s1 s2 t1 t2 hd1 hd2 hs1 hs2 ht1 ht2
  sorry

end Gwen_avg_speed_trip_l132_132452


namespace basketball_games_count_l132_132638

noncomputable def tokens_per_game : ℕ := 3
noncomputable def total_tokens : ℕ := 18
noncomputable def air_hockey_games : ℕ := 2
noncomputable def air_hockey_tokens := air_hockey_games * tokens_per_game
noncomputable def remaining_tokens := total_tokens - air_hockey_tokens

theorem basketball_games_count :
  (remaining_tokens / tokens_per_game) = 4 := by
  sorry

end basketball_games_count_l132_132638


namespace find_ab_l132_132927

variables {a b : ℝ}

theorem find_ab
  (h : ∀ x : ℝ, 0 ≤ x → 0 ≤ x^4 - x^3 + a * x + b ∧ x^4 - x^3 + a * x + b ≤ (x^2 - 1)^2) :
  a * b = -1 :=
sorry

end find_ab_l132_132927


namespace a_range_l132_132311

noncomputable def f (a x : ℝ) : ℝ := x^3 + a*x^2 - 2*x + 5

noncomputable def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 2

theorem a_range (a : ℝ) :
  (∃ x y : ℝ, (1/3 < x ∧ x < 1/2) ∧ (1/3 < y ∧ y < 1/2) ∧ f' a x = 0 ∧ f' a y = 0) ↔
  a ∈ Set.Ioo (5/4) (5/2) :=
by
  sorry

end a_range_l132_132311


namespace parametric_to_standard_l132_132534

theorem parametric_to_standard (theta : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : x = 1 + 2 * Real.cos theta)
  (h2 : y = -2 + 2 * Real.sin theta) :
  (x - 1)^2 + (y + 2)^2 = 4 :=
sorry

end parametric_to_standard_l132_132534


namespace value_of_a_minus_b_l132_132844

theorem value_of_a_minus_b (a b : ℤ) (h1 : 2020 * a + 2024 * b = 2040) (h2 : 2022 * a + 2026 * b = 2044) :
  a - b = 1002 :=
sorry

end value_of_a_minus_b_l132_132844


namespace terminating_decimal_representation_l132_132713

-- Definitions derived from conditions
def given_fraction : ℚ := 53 / (2^2 * 5^3)

-- The theorem we aim to state that expresses the question and correct answer
theorem terminating_decimal_representation : given_fraction = 0.106 :=
by
  sorry  -- proof goes here

end terminating_decimal_representation_l132_132713


namespace g_ge_one_l132_132723

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + Real.log x + 4

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 1) - Real.log x

theorem g_ge_one (x : ℝ) (h : 0 < x) : g x ≥ 1 :=
sorry

end g_ge_one_l132_132723


namespace distance_P_to_AB_l132_132136

def point_P_condition (P : ℝ) : Prop :=
  P > 0 ∧ P < 1

def parallel_line_property (P : ℝ) (h : ℝ) : Prop :=
  h = 1 - P / 1

theorem distance_P_to_AB (P h : ℝ) (area_total : ℝ) (area_smaller : ℝ) :
  point_P_condition P →
  parallel_line_property P h →
  (area_smaller / area_total) = 1 / 3 →
  h = 2 / 3 :=
by
  intro hP hp hratio
  sorry

end distance_P_to_AB_l132_132136


namespace no_pairs_for_arithmetic_progression_l132_132600

-- Define the problem in Lean
theorem no_pairs_for_arithmetic_progression :
  ¬ ∃ (a b : ℝ), (2 * a = 5 + b) ∧ (2 * b = a * (1 + b)) :=
sorry

end no_pairs_for_arithmetic_progression_l132_132600


namespace solve_inequality_l132_132025

theorem solve_inequality (a : ℝ) : 
  {x : ℝ | x^2 - (a + 2) * x + 2 * a > 0} = 
  (if a > 2 then {x | x < 2 ∨ x > a}
   else if a = 2 then {x | x ≠ 2}
   else {x | x < a ∨ x > 2}) :=
sorry

end solve_inequality_l132_132025


namespace sum_of_first_five_terms_is_31_l132_132425

variable (a : ℕ → ℝ) (q : ℝ)

-- The geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Condition 1: a_2 * a_3 = 2 * a_1
def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 3 = 2 * a 1

-- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5/4
def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 2 * a 7) / 2 = 5 / 4

-- Sum of the first 5 terms of the geometric sequence
def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

-- The theorem to prove
theorem sum_of_first_five_terms_is_31 (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) 
  (hc1 : condition1 a q) 
  (hc2 : condition2 a q) : 
  S_5 a = 31 := by
  sorry

end sum_of_first_five_terms_is_31_l132_132425


namespace find_m_l132_132578

theorem find_m (a b c d : ℕ) (m : ℕ) (a_n b_n c_n d_n: ℕ → ℕ)
  (ha : ∀ n, a_n n = a * n + b)
  (hb : ∀ n, b_n n = c * n + d)
  (hc : ∀ n, c_n n = a_n n * b_n n)
  (hd : ∀ n, d_n n = c_n (n + 1) - c_n n)
  (ha1b1 : m = a_n 1 * b_n 1)
  (hca2b2 : a_n 2 * b_n 2 = 4)
  (hca3b3 : a_n 3 * b_n 3 = 8)
  (hca4b4 : a_n 4 * b_n 4 = 16) :
  m = 4 := 
by sorry

end find_m_l132_132578


namespace inequality_proof_l132_132906

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c :=
sorry

end inequality_proof_l132_132906


namespace shirt_price_l132_132066

theorem shirt_price (S : ℝ) (h : (5 * S + 5 * 3) / 2 = 10) : S = 1 :=
by
  sorry

end shirt_price_l132_132066


namespace multiplication_expression_l132_132166

theorem multiplication_expression : 45 * 27 + 18 * 45 = 2025 := by
  sorry

end multiplication_expression_l132_132166


namespace speed_of_second_train_l132_132557

theorem speed_of_second_train
  (distance : ℝ)
  (speed_fast : ℝ)
  (time_difference : ℝ)
  (v : ℝ)
  (h_distance : distance = 425.80645161290323)
  (h_speed_fast : speed_fast = 75)
  (h_time_difference : time_difference = 4)
  (h_v : v = distance / (distance / speed_fast + time_difference)) :
  v = 44 := 
sorry

end speed_of_second_train_l132_132557


namespace line_perpendicular_to_plane_l132_132537

open Classical

-- Define the context of lines and planes.
variables {Line : Type} {Plane : Type}

-- Define the perpendicular and parallel relations.
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry

-- Declare the distinct lines and non-overlapping planes.
variable {m n : Line}
variable {α β : Plane}

-- State the theorem.
theorem line_perpendicular_to_plane (h1 : parallel m n) (h2 : perpendicular n β) : perpendicular m β :=
sorry

end line_perpendicular_to_plane_l132_132537


namespace functional_equation_solution_l132_132174

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, y^2 * f x + x^2 * f y + x * y = x * y * f (x + y) + x^2 + y^2) →
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x + 1 :=
by
  sorry

end functional_equation_solution_l132_132174


namespace each_nap_duration_l132_132216

-- Definitions based on the problem conditions
def BillProjectDurationInDays : ℕ := 4
def HoursPerDay : ℕ := 24
def TotalProjectHours : ℕ := BillProjectDurationInDays * HoursPerDay
def WorkHours : ℕ := 54
def NapsTaken : ℕ := 6

-- Calculate the time spent on naps and the duration of each nap
def NapHoursTotal : ℕ := TotalProjectHours - WorkHours
def DurationEachNap : ℕ := NapHoursTotal / NapsTaken

-- The theorem stating the expected answer
theorem each_nap_duration :
  DurationEachNap = 7 := by
  sorry

end each_nap_duration_l132_132216


namespace principal_amount_l132_132628

theorem principal_amount
(SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
(h₀ : SI = 800)
(h₁ : R = 0.08)
(h₂ : T = 1)
(h₃ : SI = P * R * T) : P = 10000 :=
by
  sorry

end principal_amount_l132_132628


namespace find_x_when_y4_l132_132435

theorem find_x_when_y4 
  (k : ℝ) 
  (h_var : ∀ y : ℝ, ∃ x : ℝ, x = k * y^2)
  (h_initial : ∃ x : ℝ, x = 6 ∧ 1 = k) :
  ∃ x : ℝ, x = 96 :=
by 
  sorry

end find_x_when_y4_l132_132435


namespace b_plus_d_over_a_l132_132221

theorem b_plus_d_over_a (a b c d e : ℝ) (h : a ≠ 0) 
  (root1 : a * (5:ℝ)^4 + b * (5:ℝ)^3 + c * (5:ℝ)^2 + d * (5:ℝ) + e = 0)
  (root2 : a * (-3:ℝ)^4 + b * (-3:ℝ)^3 + c * (-3:ℝ)^2 + d * (-3:ℝ) + e = 0)
  (root3 : a * (2:ℝ)^4 + b * (2:ℝ)^3 + c * (2:ℝ)^2 + d * (2:ℝ) + e = 0) :
  (b + d) / a = - (12496 / 3173) :=
sorry

end b_plus_d_over_a_l132_132221


namespace trigonometric_identity_l132_132047

theorem trigonometric_identity (m : ℝ) (h : m < 0) :
  2 * (3 / -5) + 4 / -5 = -2 / 5 :=
by
  sorry

end trigonometric_identity_l132_132047


namespace total_arms_collected_l132_132872

-- Define the conditions as parameters
def arms_of_starfish := 7 * 5
def arms_of_seastar := 14

-- Define the theorem to prove total arms
theorem total_arms_collected : arms_of_starfish + arms_of_seastar = 49 := by
  sorry

end total_arms_collected_l132_132872


namespace distinct_divisors_in_set_l132_132667

theorem distinct_divisors_in_set (p : ℕ) (hp : Nat.Prime p) (hp5 : 5 < p) :
  ∃ (x y : ℕ), x ∈ {p - n^2 | n : ℕ} ∧ y ∈ {p - n^2 | n : ℕ} ∧ x ≠ y ∧ x ≠ 1 ∧ x ∣ y :=
by
  sorry

end distinct_divisors_in_set_l132_132667


namespace arithmetic_sum_nine_l132_132788

noncomputable def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem arithmetic_sum_nine (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 4 = 9)
  (h3 : a 6 = 11) : arithmetic_sequence_sum a 9 = 90 :=
by
  sorry

end arithmetic_sum_nine_l132_132788


namespace max_total_toads_l132_132513

variable (x y : Nat)
variable (frogs total_frogs : Nat)
variable (total_toads : Nat)

def pond1_frogs := 3 * x
def pond1_toads := 4 * x
def pond2_frogs := 5 * y
def pond2_toads := 6 * y

def all_frogs := pond1_frogs x + pond2_frogs y
def all_toads := pond1_toads x + pond2_toads y

theorem max_total_toads (h_frogs : all_frogs x y = 36) : all_toads x y = 46 := 
sorry

end max_total_toads_l132_132513


namespace tangent_line_at_one_l132_132408

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + Real.log x

theorem tangent_line_at_one (a : ℝ)
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < δ → |(f a x - f a 1) / (x - 1) - 3| < ε) :
  ∃ m b, m = 3 ∧ b = -2 ∧ (∀ x y, y = f a x → m * x = y + b) := sorry

end tangent_line_at_one_l132_132408


namespace initial_oranges_l132_132710

theorem initial_oranges (X : ℕ) : 
  (X - 9 + 38 = 60) → X = 31 :=
sorry

end initial_oranges_l132_132710


namespace FriedChickenDinner_orders_count_l132_132826

-- Defining the number of pieces of chicken used by each type of order
def piecesChickenPasta := 2
def piecesBarbecueChicken := 3
def piecesFriedChickenDinner := 8

-- Defining the number of orders for Chicken Pasta and Barbecue Chicken
def numChickenPastaOrders := 6
def numBarbecueChickenOrders := 3

-- Defining the total pieces of chicken needed for all orders
def totalPiecesOfChickenNeeded := 37

-- Defining the number of pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPasta : Nat := piecesChickenPasta * numChickenPastaOrders
def piecesNeededBarbecueChicken : Nat := piecesBarbecueChicken * numBarbecueChickenOrders

-- Defining the total pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPastaAndBarbecue : Nat := piecesNeededChickenPasta + piecesNeededBarbecueChicken

-- Calculating the pieces of chicken needed for Fried Chicken Dinner orders
def piecesNeededFriedChickenDinner : Nat := totalPiecesOfChickenNeeded - piecesNeededChickenPastaAndBarbecue

-- Defining the number of Fried Chicken Dinner orders
def numFriedChickenDinnerOrders : Nat := piecesNeededFriedChickenDinner / piecesFriedChickenDinner

-- Proving Victor has 2 Fried Chicken Dinner orders
theorem FriedChickenDinner_orders_count : numFriedChickenDinnerOrders = 2 := by
  unfold numFriedChickenDinnerOrders
  unfold piecesNeededFriedChickenDinner
  unfold piecesNeededChickenPastaAndBarbecue
  unfold piecesNeededBarbecueChicken
  unfold piecesNeededChickenPasta
  unfold totalPiecesOfChickenNeeded
  unfold numBarbecueChickenOrders
  unfold piecesBarbecueChicken
  unfold numChickenPastaOrders
  unfold piecesChickenPasta
  sorry

end FriedChickenDinner_orders_count_l132_132826


namespace find_p_l132_132644

theorem find_p (a b p : ℝ) (h1: a ≠ 0) (h2: b ≠ 0) 
  (h3: a^2 - 4 * b = 0) 
  (h4: a + b = 5 * p) 
  (h5: a * b = 2 * p^3) : p = 3 := 
sorry

end find_p_l132_132644


namespace average_of_first_two_is_1_point_1_l132_132267

theorem average_of_first_two_is_1_point_1
  (a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.5)
  (h2 : (a1 + a2) / 2 = x)
  (h3 : (a3 + a4) / 2 = 1.4)
  (h4 : (a5 + a6) / 2 = 5) :
  x = 1.1 := 
sorry

end average_of_first_two_is_1_point_1_l132_132267


namespace largest_five_digit_integer_l132_132156

/-- The product of the digits of the integer 98752 is (7 * 6 * 5 * 4 * 3 * 2 * 1), and
    98752 is the largest five-digit integer with this property. -/
theorem largest_five_digit_integer :
  (∃ (n : ℕ), n = 98752 ∧ (∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1 * 10^4 + d2 * 10^3 + d3 * 10^2 + d4 * 10 + d5 ∧
    (d1 * d2 * d3 * d4 * d5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) ∧
    (∀ (m : ℕ), m ≠ 98752 → m < 100000 ∧ (∃ (e1 e2 e3 e4 e5 : ℕ),
    m = e1 * 10^4 + e2 * 10^3 + e3 * 10^2 + e4 * 10 + e5 →
    (e1 * e2 * e3 * e4 * e5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) → m < 98752)))) :=
  sorry

end largest_five_digit_integer_l132_132156


namespace algebraic_expression_value_l132_132272

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 2 * x + 7 = 6) : 4 * x^2 + 8 * x - 5 = -9 :=
by
  sorry

end algebraic_expression_value_l132_132272


namespace parabola_constant_term_l132_132772

theorem parabola_constant_term
  (a b c : ℝ)
  (h1 : ∀ x, (-2 * (x - 1)^2 + 3) = a * x^2 + b * x + c ) :
  c = 2 :=
sorry

end parabola_constant_term_l132_132772


namespace max_distinct_integer_solutions_le_2_l132_132053

def f (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem max_distinct_integer_solutions_le_2 
  (a b c : ℝ) (h₀ : a > 100) :
  ∀ (x : ℤ), |f a b c (x : ℝ)| ≤ 50 → 
  ∃ (x₁ x₂ : ℤ), x = x₁ ∨ x = x₂ :=
by
  sorry

end max_distinct_integer_solutions_le_2_l132_132053


namespace find_nearest_integer_x_minus_y_l132_132881

variable (x y : ℝ)

theorem find_nearest_integer_x_minus_y
  (h1 : abs x + y = 5)
  (h2 : abs x * y - x^3 = 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0) :
  |x - y| = 5 := sorry

end find_nearest_integer_x_minus_y_l132_132881


namespace transformed_inequality_solution_l132_132318

variable {a b c d : ℝ}

theorem transformed_inequality_solution (H : ∀ x : ℝ, ((-1 < x ∧ x < -1/3) ∨ (1/2 < x ∧ x < 1)) → 
  (b / (x + a) + (x + d) / (x + c) < 0)) :
  ∀ x : ℝ, ((1 < x ∧ x < 3) ∨ (-2 < x ∧ x < -1)) ↔ (bx / (ax - 1) + (dx - 1) / (cx - 1) < 0) :=
sorry

end transformed_inequality_solution_l132_132318


namespace trapezoid_area_l132_132323

theorem trapezoid_area 
  (diagonals_perpendicular : ∀ A B C D : ℝ, (A ≠ B → C ≠ D → A * C + B * D = 0)) 
  (diagonal_length : ∀ B D : ℝ, B ≠ D → (B - D) = 17) 
  (height_of_trapezoid : ∀ (height : ℝ), height = 15) : 
  ∃ (area : ℝ), area = 4335 / 16 := 
sorry

end trapezoid_area_l132_132323


namespace range_of_a_l132_132992

theorem range_of_a (a : ℝ) : (∀ x : ℝ, abs (2 * x + 2) - abs (2 * x - 2) ≤ a) ↔ 4 ≤ a :=
sorry

end range_of_a_l132_132992


namespace butter_remaining_correct_l132_132502

-- Definitions of the conditions
def cupsOfBakingMix : ℕ := 6
def butterPerCup : ℕ := 2
def substituteRatio : ℕ := 1
def coconutOilUsed : ℕ := 8

-- Calculation based on the conditions
def butterNeeded : ℕ := butterPerCup * cupsOfBakingMix
def butterReplaced : ℕ := coconutOilUsed * substituteRatio
def butterRemaining : ℕ := butterNeeded - butterReplaced

-- The theorem to prove the chef has 4 ounces of butter remaining
theorem butter_remaining_correct : butterRemaining = 4 := 
by
  -- Note: We insert 'sorry' since the proof itself is not required.
  sorry

end butter_remaining_correct_l132_132502


namespace value_of_f_1985_l132_132805

def f : ℝ → ℝ := sorry -- Assuming the existence of f, let ℝ be the type of real numbers

-- Given condition as a hypothesis
axiom functional_eq (x y : ℝ) : f (x + y) = f (x^2) + f (2 * y)

-- The main theorem we want to prove
theorem value_of_f_1985 : f 1985 = 0 :=
by
  sorry

end value_of_f_1985_l132_132805


namespace carpenter_additional_logs_needed_l132_132287

theorem carpenter_additional_logs_needed 
  (total_woodblocks_needed : ℕ) 
  (logs_available : ℕ) 
  (woodblocks_per_log : ℕ) 
  (additional_logs_needed : ℕ)
  (h1 : total_woodblocks_needed = 80)
  (h2 : logs_available = 8)
  (h3 : woodblocks_per_log = 5)
  (h4 : additional_logs_needed = 8) : 
  (total_woodblocks_needed - (logs_available * woodblocks_per_log)) / woodblocks_per_log = additional_logs_needed :=
by
  sorry

end carpenter_additional_logs_needed_l132_132287


namespace fraction_of_sum_l132_132770

theorem fraction_of_sum (S n : ℝ) (h1 : n = S / 6) : n / (S + n) = 1 / 7 :=
by sorry

end fraction_of_sum_l132_132770


namespace average_rainfall_per_hour_in_June_1882_l132_132233

open Real

theorem average_rainfall_per_hour_in_June_1882 
  (total_rainfall : ℝ) (days_in_June : ℕ) (hours_per_day : ℕ)
  (H1 : total_rainfall = 450) (H2 : days_in_June = 30) (H3 : hours_per_day = 24) :
  total_rainfall / (days_in_June * hours_per_day) = 5 / 8 :=
by
  sorry

end average_rainfall_per_hour_in_June_1882_l132_132233


namespace tangent_expression_l132_132661

open Real

theorem tangent_expression
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (geom_seq : ∀ n m, a (n + m) = a n * a m) 
  (arith_seq : ∀ n, b (n + 1) = b n + (b 2 - b 1))
  (cond1 : a 1 * a 6 * a 11 = -3 * sqrt 3)
  (cond2 : b 1 + b 6 + b 11 = 7 * pi) :
  tan ( (b 3 + b 9) / (1 - a 4 * a 8) ) = -sqrt 3 :=
sorry

end tangent_expression_l132_132661


namespace point_on_graph_l132_132283

noncomputable def f (x : ℝ) : ℝ := abs (x^3 + 1) + abs (x^3 - 1)

theorem point_on_graph (a : ℝ) : ∃ (x y : ℝ), (x = a) ∧ (y = f (-a)) ∧ (y = f x) :=
by 
  sorry

end point_on_graph_l132_132283


namespace friend_selling_price_l132_132454

-- Define the conditions
def CP : ℝ := 51136.36
def loss_percent : ℝ := 0.12
def gain_percent : ℝ := 0.20

-- Define the selling prices SP1 and SP2
def SP1 := CP * (1 - loss_percent)
def SP2 := SP1 * (1 + gain_percent)

-- State the theorem
theorem friend_selling_price : SP2 = 54000 := 
by sorry

end friend_selling_price_l132_132454


namespace inequality_proof_l132_132055

theorem inequality_proof (x y : ℝ) (h1 : x^2 ≥ y) (h2 : y^2 ≥ x) : 
  (x / (y^2 + 1) + y / (x^2 + 1) ≤ 1) :=
sorry

end inequality_proof_l132_132055


namespace problem1_solution_problem2_solution_l132_132914

theorem problem1_solution (x : ℝ) :
  (2 < |2 * x - 5| ∧ |2 * x - 5| ≤ 7) → ((-1 ≤ x ∧ x < 3 / 2) ∨ (7 / 2 < x ∧ x ≤ 6)) := by
  sorry

theorem problem2_solution (x : ℝ) :
  (1 / (x - 1) > x + 1) → (x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)) := by
  sorry

end problem1_solution_problem2_solution_l132_132914


namespace isosceles_triangle_y_value_l132_132037

theorem isosceles_triangle_y_value :
  ∃ y : ℝ, (y = 1 + Real.sqrt 51 ∨ y = 1 - Real.sqrt 51) ∧ 
  (Real.sqrt ((y - 1)^2 + (4 - (-3))^2) = 10) :=
by sorry

end isosceles_triangle_y_value_l132_132037


namespace dodecahedron_has_150_interior_diagonals_l132_132814

def dodecahedron_diagonals (vertices : ℕ) (adjacent : ℕ) : ℕ :=
  let total := vertices * (vertices - adjacent - 1) / 2
  total

theorem dodecahedron_has_150_interior_diagonals :
  dodecahedron_diagonals 20 4 = 150 :=
by
  sorry

end dodecahedron_has_150_interior_diagonals_l132_132814


namespace find_k_l132_132810

def green_balls : ℕ := 7

noncomputable def probability_green (k : ℕ) : ℚ := green_balls / (green_balls + k)
noncomputable def probability_purple (k : ℕ) : ℚ := k / (green_balls + k)

noncomputable def winning_for_green : ℤ := 3
noncomputable def losing_for_purple : ℤ := -1

noncomputable def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * (winning_for_green : ℚ) + (probability_purple k) * (losing_for_purple : ℚ)

theorem find_k (k : ℕ) (h : expected_value k = 1) : k = 7 :=
  sorry

end find_k_l132_132810


namespace quadratic_complete_square_l132_132062

theorem quadratic_complete_square (x m n : ℝ) 
  (h : 9 * x^2 - 36 * x - 81 = 0) :
  (x + m)^2 = n ∧ m + n = 11 :=
sorry

end quadratic_complete_square_l132_132062


namespace expression_evaluation_l132_132291

open Rat

theorem expression_evaluation :
  ∀ (a b c : ℚ),
  c = b - 4 →
  b = a + 4 →
  a = 3 →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 10) / (c + 7) = 117 / 40 :=
by
  intros a b c hc hb ha h1 h2 h3
  simp [hc, hb, ha]
  have h1 : 3 + 1 ≠ 0 := by norm_num
  have h2 : 7 - 3 ≠ 0 := by norm_num
  have h3 : 3 + 7 ≠ 0 := by norm_num
  -- Placeholder for the simplified expression computation
  sorry

end expression_evaluation_l132_132291


namespace legs_in_room_l132_132436

def total_legs_in_room (tables4 : Nat) (sofa : Nat) (chairs4 : Nat) (tables3 : Nat) (table1 : Nat) (rocking_chair2 : Nat) : Nat :=
  (tables4 * 4) + (sofa * 4) + (chairs4 * 4) + (tables3 * 3) + (table1 * 1) + (rocking_chair2 * 2)

theorem legs_in_room :
  total_legs_in_room 4 1 2 3 1 1 = 40 :=
by
  -- Skipping proof steps
  sorry

end legs_in_room_l132_132436


namespace gcd_765432_654321_l132_132463

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 9 := by
  sorry

end gcd_765432_654321_l132_132463


namespace find_k_l132_132394

theorem find_k (k : ℝ) : (1 - 1.5 * k = (k - 2.5) / 3) → k = 1 :=
by
  intro h
  sorry

end find_k_l132_132394


namespace sin_90_eq_one_l132_132696

-- Definition of the rotation by 90 degrees counterclockwise
def rotate90 (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, p.1)

-- Definition of the sine function for a 90 degree angle
def sin90 : ℝ :=
  let initial_point := (1, 0)
  let rotated_point := rotate90 initial_point
  rotated_point.2

-- Theorem to be proven: sin90 should be equal to 1
theorem sin_90_eq_one : sin90 = 1 :=
by
  sorry

end sin_90_eq_one_l132_132696


namespace part_a_part_b_l132_132988

theorem part_a (x : ℝ) (n : ℕ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) (hn_pos : 0 < n) :
  Real.log x < n * (x ^ (1 / n) - 1) ∧ n * (x ^ (1 / n) - 1) < (x ^ (1 / n)) * Real.log x := sorry

theorem part_b (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
  (Real.log x) = (Real.log x) := sorry

end part_a_part_b_l132_132988


namespace smallest_divisor_after_323_l132_132012

-- Let n be an even 4-digit number such that 323 is a divisor of n.
def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisor (d n : ℕ) : Prop :=
  n % d = 0

theorem smallest_divisor_after_323 (n : ℕ) (h1 : is_even n) (h2 : is_4_digit n) (h3 : is_divisor 323 n) : ∃ k, k > 323 ∧ is_divisor k n ∧ k = 340 :=
by
  sorry

end smallest_divisor_after_323_l132_132012


namespace line_intersects_circle_l132_132822

theorem line_intersects_circle (a : ℝ) :
  ∃ (x y : ℝ), (y = a * x + 1) ∧ ((x - 1) ^ 2 + y ^ 2 = 4) :=
by
  sorry

end line_intersects_circle_l132_132822


namespace largest_divisor_power_of_ten_l132_132419

theorem largest_divisor_power_of_ten (N : ℕ) (m : ℕ) (k : ℕ) 
  (h1 : m ∣ N)
  (h2 : m < N)
  (h3 : N + m = 10^k) : N = 75 := sorry

end largest_divisor_power_of_ten_l132_132419


namespace ratio_expenditure_l132_132651

variable (I : ℝ) -- Assume the income in the first year is I.

-- Conditions
def savings_first_year := 0.25 * I
def expenditure_first_year := 0.75 * I
def income_second_year := 1.25 * I
def savings_second_year := 2 * savings_first_year
def expenditure_second_year := income_second_year - savings_second_year
def total_expenditure_two_years := expenditure_first_year + expenditure_second_year

-- Statement to be proved
theorem ratio_expenditure 
  (savings_first_year : ℝ := 0.25 * I)
  (expenditure_first_year : ℝ := 0.75 * I)
  (income_second_year : ℝ := 1.25 * I)
  (savings_second_year : ℝ := 2 * savings_first_year)
  (expenditure_second_year : ℝ := income_second_year - savings_second_year)
  (total_expenditure_two_years : ℝ := expenditure_first_year + expenditure_second_year) :
  (total_expenditure_two_years / expenditure_first_year) = 2 := by
    sorry

end ratio_expenditure_l132_132651


namespace polynomial_coeff_sum_abs_l132_132339

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) :
    (2 * x - 1)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 242 := by 
  sorry

end polynomial_coeff_sum_abs_l132_132339


namespace same_terminal_side_l132_132922

theorem same_terminal_side : 
  let θ1 := 23 * Real.pi / 3
  let θ2 := 5 * Real.pi / 3
  (∃ k : ℤ, θ1 - 2 * k * Real.pi = θ2) :=
sorry

end same_terminal_side_l132_132922


namespace bowls_initially_bought_l132_132784

theorem bowls_initially_bought 
  (x : ℕ) 
  (cost_per_bowl : ℕ := 13) 
  (revenue_per_bowl : ℕ := 17)
  (sold_bowls : ℕ := 108)
  (profit_percentage : ℝ := 23.88663967611336) 
  (approx_x : ℝ := 139) :
  (23.88663967611336 / 100) * (cost_per_bowl : ℝ) * (x : ℝ) = 
    (sold_bowls * revenue_per_bowl) - (sold_bowls * cost_per_bowl) → 
  abs ((x : ℝ) - approx_x) < 0.5 :=
by
  sorry

end bowls_initially_bought_l132_132784


namespace intersection_points_of_lines_l132_132450

theorem intersection_points_of_lines :
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ x + 3 * y = 3 ∧ x = 10 / 11 ∧ y = 13 / 11) ∧
  (∃ (x y : ℚ), 2 * y - 3 * x = 4 ∧ 5 * x - 3 * y = 6 ∧ x = 24 ∧ y = 38) :=
by
  sorry

end intersection_points_of_lines_l132_132450


namespace number_of_children_l132_132656
-- Import the entirety of the Mathlib library

-- Define the conditions and the theorem to be proven
theorem number_of_children (C n : ℕ) 
  (h1 : C = 8 * n + 4) 
  (h2 : C = 11 * (n - 1)) : 
  n = 5 :=
by sorry

end number_of_children_l132_132656


namespace proof_problem_l132_132511

noncomputable def p (a : ℝ) : Prop :=
∀ x : ℝ, x^2 + a * x + a^2 ≥ 0

noncomputable def q : Prop :=
∃ x₀ : ℕ, 0 < x₀ ∧ 2 * x₀^2 - 1 ≤ 0

theorem proof_problem (a : ℝ) (hp : p a) (hq : q) : p a ∨ q :=
by
  sorry

end proof_problem_l132_132511


namespace parametric_line_eq_l132_132123

-- Define the parameterized functions for x and y 
def parametric_x (t : ℝ) : ℝ := 3 * t + 7
def parametric_y (t : ℝ) : ℝ := 5 * t - 8

-- Define the equation of the line (here it's a relation that relates x and y)
def line_equation (x y : ℝ) : Prop := 
  y = (5 / 3) * x - (59 / 3)

theorem parametric_line_eq : 
  ∃ t : ℝ, line_equation (parametric_x t) (parametric_y t) := 
by
  -- Proof goes here
  sorry

end parametric_line_eq_l132_132123


namespace average_letters_per_day_l132_132564

theorem average_letters_per_day 
  (letters_tuesday : ℕ)
  (letters_wednesday : ℕ)
  (days : ℕ := 2) 
  (letters_total : ℕ := letters_tuesday + letters_wednesday) :
  letters_tuesday = 7 → letters_wednesday = 3 → letters_total / days = 5 :=
by
  -- The proof is omitted
  sorry

end average_letters_per_day_l132_132564


namespace Marley_fruit_count_l132_132185

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l132_132185


namespace remainder_of_sum_division_l132_132113

theorem remainder_of_sum_division (f y : ℤ) (a b : ℤ) (h_f : f = 5 * a + 3) (h_y : y = 5 * b + 4) :  
  (f + y) % 5 = 2 :=
by
  sorry

end remainder_of_sum_division_l132_132113


namespace find_value_l132_132004

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic : ∀ x : ℝ, f (x + Real.pi) = f x
axiom value_at_neg_pi_third : f (-Real.pi / 3) = 1 / 2

theorem find_value : f (2017 * Real.pi / 3) = 1 / 2 :=
by
  sorry

end find_value_l132_132004


namespace boys_down_slide_l132_132487

theorem boys_down_slide (boys_1 boys_2 : ℕ) (h : boys_1 = 22) (h' : boys_2 = 13) : boys_1 + boys_2 = 35 := by
  sorry

end boys_down_slide_l132_132487


namespace quadratic_real_roots_implies_k_range_l132_132952

theorem quadratic_real_roots_implies_k_range (k : ℝ) 
  (h : ∃ x : ℝ, k * x^2 + 2 * x - 1 = 0)
  (hk : k ≠ 0) : k ≥ -1 ∧ k ≠ 0 :=
sorry

end quadratic_real_roots_implies_k_range_l132_132952


namespace length_of_faster_train_l132_132878

-- Definitions for the given conditions
def speed_faster_train_kmh : ℝ := 50
def speed_slower_train_kmh : ℝ := 32
def time_seconds : ℝ := 15

theorem length_of_faster_train : 
  let speed_relative_kmh := speed_faster_train_kmh - speed_slower_train_kmh
  let speed_relative_mps := speed_relative_kmh * (1000 / 3600)
  let length_faster_train := speed_relative_mps * time_seconds
  length_faster_train = 75 := 
by 
  sorry 

end length_of_faster_train_l132_132878


namespace probability_of_getting_specific_clothing_combination_l132_132084

def total_articles := 21

def ways_to_choose_4_articles : ℕ := Nat.choose total_articles 4

def ways_to_choose_2_shirts_from_6 : ℕ := Nat.choose 6 2

def ways_to_choose_1_pair_of_shorts_from_7 : ℕ := Nat.choose 7 1

def ways_to_choose_1_pair_of_socks_from_8 : ℕ := Nat.choose 8 1

def favorable_outcomes := 
  ways_to_choose_2_shirts_from_6 * 
  ways_to_choose_1_pair_of_shorts_from_7 * 
  ways_to_choose_1_pair_of_socks_from_8

def probability := (favorable_outcomes : ℚ) / (ways_to_choose_4_articles : ℚ)

theorem probability_of_getting_specific_clothing_combination : 
  probability = 56 / 399 := by
  sorry

end probability_of_getting_specific_clothing_combination_l132_132084


namespace garden_ratio_2_l132_132432

theorem garden_ratio_2 :
  ∃ (P C k R : ℤ), 
      P = 237 ∧ 
      C = P - 60 ∧ 
      P + C + k = 768 ∧ 
      R = k / C ∧ 
      R = 2 := 
by
  sorry

end garden_ratio_2_l132_132432


namespace joe_lowest_score_dropped_l132_132508

theorem joe_lowest_score_dropped (A B C D : ℕ) 
  (h1 : A + B + C + D = 160)
  (h2 : A + B + C = 135) 
  (h3 : D ≤ A ∧ D ≤ B ∧ D ≤ C) :
  D = 25 :=
sorry

end joe_lowest_score_dropped_l132_132508


namespace percentage_increase_in_weight_l132_132668

theorem percentage_increase_in_weight :
  ∀ (num_plates : ℕ) (weight_per_plate lowered_weight : ℝ),
    num_plates = 10 →
    weight_per_plate = 30 →
    lowered_weight = 360 →
    ((lowered_weight - num_plates * weight_per_plate) / (num_plates * weight_per_plate)) * 100 = 20 :=
by
  intros num_plates weight_per_plate lowered_weight h_num_plates h_weight_per_plate h_lowered_weight
  sorry

end percentage_increase_in_weight_l132_132668


namespace circle_radius_three_points_on_line_l132_132362

theorem circle_radius_three_points_on_line :
  ∀ R : ℝ,
  (∀ x y : ℝ, (x - 1)^2 + (y + 1)^2 = R^2 → (4 * x + 3 * y = 11) → (dist (x, y) (1, -1) = 1)) →
  R = 3
:= sorry

end circle_radius_three_points_on_line_l132_132362


namespace grandpa_uncle_ratio_l132_132220

def initial_collection := 150
def dad_gift := 10
def mum_gift := dad_gift + 5
def auntie_gift := 6
def uncle_gift := auntie_gift - 1
def final_collection := 196
def total_cars_needed := final_collection - initial_collection
def other_gifts := dad_gift + mum_gift + auntie_gift + uncle_gift
def grandpa_gift := total_cars_needed - other_gifts

theorem grandpa_uncle_ratio : grandpa_gift = 2 * uncle_gift := by
  sorry

end grandpa_uncle_ratio_l132_132220


namespace number_of_men_in_second_group_l132_132155

theorem number_of_men_in_second_group 
  (work : ℕ)
  (days_first_group days_second_group : ℕ)
  (men_first_group men_second_group : ℕ)
  (h1 : work = men_first_group * days_first_group)
  (h2 : work = men_second_group * days_second_group)
  (h3 : men_first_group = 20)
  (h4 : days_first_group = 30)
  (h5 : days_second_group = 24) :
  men_second_group = 25 :=
by
  sorry

end number_of_men_in_second_group_l132_132155


namespace range_of_y_eq_4_sin_squared_x_minus_2_l132_132941

theorem range_of_y_eq_4_sin_squared_x_minus_2 : 
  (∀ x : ℝ, y = 4 * (Real.sin x)^2 - 2) → 
  (∃ a b : ℝ, ∀ x : ℝ, y ∈ Set.Icc a b ∧ a = -2 ∧ b = 2) :=
sorry

end range_of_y_eq_4_sin_squared_x_minus_2_l132_132941


namespace ratio_of_sides_l132_132911

theorem ratio_of_sides (s r : ℝ) (h : s^2 = 2 * r^2 * Real.sqrt 2) : r / s = 1 / Real.sqrt (2 * Real.sqrt 2) := 
by
  sorry

end ratio_of_sides_l132_132911


namespace distance_yolkino_palkino_l132_132260

theorem distance_yolkino_palkino (d_1 d_2 : ℕ) (h : ∀ k : ℕ, d_1 + d_2 = 13) : 
  ∀ k : ℕ, d_1 + d_2 = 13 → (d_1 + d_2 = 13) :=
by
  sorry

end distance_yolkino_palkino_l132_132260


namespace loss_percent_l132_132294

theorem loss_percent (CP SP : ℝ) (h_CP : CP = 600) (h_SP : SP = 550) :
  ((CP - SP) / CP) * 100 = 8.33 := by
  sorry

end loss_percent_l132_132294


namespace cost_when_q_is_2_l132_132149

-- Defining the cost function
def cost (q : ℕ) : ℕ := q^3 + q - 1

-- Theorem to prove the cost when q = 2
theorem cost_when_q_is_2 : cost 2 = 9 :=
by
  -- placeholder for the proof
  sorry

end cost_when_q_is_2_l132_132149


namespace solution_mixture_l132_132875

/-
  Let X be a solution that is 10% alcohol by volume.
  Let Y be a solution that is 30% alcohol by volume.
  We define the final solution to be 22% alcohol by volume.
  We need to prove that the amount of solution Y that needs
  to be added to 300 milliliters of solution X to achieve this 
  concentration is 450 milliliters.
-/

theorem solution_mixture (y : ℝ) : 
  (0.10 * 300) + (0.30 * y) = 0.22 * (300 + y) → 
  y = 450 :=
by {
  sorry
}

end solution_mixture_l132_132875


namespace find_number_l132_132808

theorem find_number (x : ℝ) (h₁ : 0.40 * x = 130 + 190) : x = 800 :=
sorry

end find_number_l132_132808


namespace find_u_plus_v_l132_132673

theorem find_u_plus_v (u v : ℚ) (h1: 5 * u - 3 * v = 26) (h2: 3 * u + 5 * v = -19) :
  u + v = -101 / 34 :=
sorry

end find_u_plus_v_l132_132673


namespace determine_n_l132_132769

theorem determine_n (n : ℕ) : (2 : ℕ)^n = 2 * 4^2 * 16^3 ↔ n = 17 := 
by
  sorry

end determine_n_l132_132769


namespace sufficient_but_not_necessary_condition_l132_132978

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l132_132978


namespace cone_volume_half_sector_rolled_l132_132200

theorem cone_volume_half_sector_rolled {r slant_height h V : ℝ}
  (radius_given : r = 3)
  (height_calculated : h = 3 * Real.sqrt 3)
  (slant_height_given : slant_height = 6)
  (arc_length : 2 * Real.pi * r = 6 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * (r^2) * h) :
  V = 9 * Real.pi * Real.sqrt 3 :=
by {
  sorry
}

end cone_volume_half_sector_rolled_l132_132200


namespace simplified_fraction_of_num_l132_132940

def num : ℚ := 368 / 100

theorem simplified_fraction_of_num : num = 92 / 25 := by
  sorry

end simplified_fraction_of_num_l132_132940


namespace weight_of_new_student_l132_132414

-- Define some constants for the problem
def avg_weight_29_students : ℝ := 28
def number_of_students_29 : ℕ := 29
def new_avg_weight_30_students : ℝ := 27.5
def number_of_students_30 : ℕ := 30

-- Calculate total weights
def total_weight_29_students : ℝ := avg_weight_29_students * number_of_students_29
def new_total_weight_30_students : ℝ := new_avg_weight_30_students * number_of_students_30

-- The proposition we need to prove
theorem weight_of_new_student :
  new_total_weight_30_students - total_weight_29_students = 13 := by
  -- Placeholder for the actual proof
  sorry

end weight_of_new_student_l132_132414


namespace solution_correct_l132_132059

variable (a b c d : ℝ)

theorem solution_correct (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end solution_correct_l132_132059


namespace evaluate_expression_l132_132440

theorem evaluate_expression : ∀ (a b c d : ℤ), 
  a = 3 →
  b = a + 3 →
  c = b - 8 →
  d = a + 5 →
  (a + 2 ≠ 0) →
  (b - 4 ≠ 0) →
  (c + 5 ≠ 0) →
  (d - 3 ≠ 0) →
  ((a + 3) * (b - 2) * (c + 9) * (d + 1) = 1512 * (a + 2) * (b - 4) * (c + 5) * (d - 3)) :=
by
  intros a b c d ha hb hc hd ha2 hb4 hc5 hd3
  sorry

end evaluate_expression_l132_132440


namespace larger_integer_value_l132_132928

theorem larger_integer_value (a b : ℕ) (h1 : a * b = 189) (h2 : a / gcd a b = 7 ∧ b / gcd a b = 3 ∨ a / gcd a b = 3 ∧ b / gcd a b = 7) : max a b = 21 :=
by
  sorry

end larger_integer_value_l132_132928


namespace mean_of_two_remaining_numbers_l132_132728

theorem mean_of_two_remaining_numbers (a b c: ℝ) (h1: (a + b + c + 100) / 4 = 90) (h2: a = 70) : (b + c) / 2 = 95 := by
  sorry

end mean_of_two_remaining_numbers_l132_132728


namespace largest_of_given_numbers_l132_132942

theorem largest_of_given_numbers :
  ∀ (a b c d e : ℝ), a = 0.998 → b = 0.9899 → c = 0.99 → d = 0.981 → e = 0.995 →
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  intros a b c d e Ha Hb Hc Hd He
  rw [Ha, Hb, Hc, Hd, He]
  exact ⟨ by norm_num, by norm_num, by norm_num, by norm_num ⟩

end largest_of_given_numbers_l132_132942


namespace articleWords_l132_132326

-- Define the number of words per page for larger and smaller types
def wordsLargerType : Nat := 1800
def wordsSmallerType : Nat := 2400

-- Define the total number of pages and the number of pages in smaller type
def totalPages : Nat := 21
def smallerTypePages : Nat := 17

-- The number of pages in larger type
def largerTypePages : Nat := totalPages - smallerTypePages

-- Calculate the total number of words in the article
def totalWords : Nat := (largerTypePages * wordsLargerType) + (smallerTypePages * wordsSmallerType)

-- Prove that the total number of words in the article is 48,000
theorem articleWords : totalWords = 48000 := 
by
  sorry

end articleWords_l132_132326


namespace sum_of_a_for_quadratic_has_one_solution_l132_132390

noncomputable def discriminant (a : ℝ) : ℝ := (a + 12)^2 - 4 * 3 * 16

theorem sum_of_a_for_quadratic_has_one_solution : 
  (∀ a : ℝ, discriminant a = 0) → 
  (-12 + 8 * Real.sqrt 3) + (-12 - 8 * Real.sqrt 3) = -24 :=
by
  intros h
  simp [discriminant] at h
  sorry

end sum_of_a_for_quadratic_has_one_solution_l132_132390


namespace calculate_expression_l132_132632

theorem calculate_expression : 
  (10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5) :=
by
  -- Proof goes here
  sorry

end calculate_expression_l132_132632


namespace rectangle_ratio_l132_132146

theorem rectangle_ratio (s x y : ℝ) 
  (h_outer_area : (2 * s) ^ 2 = 4 * s ^ 2)
  (h_inner_sides : s + 2 * y = 2 * s)
  (h_outer_sides : x + y = 2 * s) :
  x / y = 3 :=
by
  sorry

end rectangle_ratio_l132_132146


namespace problem_l132_132348

theorem problem {x y n : ℝ} 
  (h1 : 2 * x + y = 4) 
  (h2 : (x + y) / 3 = 1) 
  (h3 : x + 2 * y = n) : n = 5 := 
sorry

end problem_l132_132348


namespace division_result_l132_132204

def n : ℕ := 16^1024

theorem division_result : n / 8 = 2^4093 :=
by sorry

end division_result_l132_132204


namespace minimum_value_of_expression_l132_132732

theorem minimum_value_of_expression (x y : ℝ) : 
    ∃ (x y : ℝ), (2 * x * y - 1) ^ 2 + (x - y) ^ 2 = 0 :=
by
  sorry

end minimum_value_of_expression_l132_132732


namespace minimum_value_f_x_l132_132979

theorem minimum_value_f_x (x : ℝ) (h : 1 < x) : 
  x + (1 / (x - 1)) ≥ 3 :=
sorry

end minimum_value_f_x_l132_132979


namespace solve_for_x_l132_132891

theorem solve_for_x {x : ℝ} (h : -3 * x - 10 = 4 * x + 5) : x = -15 / 7 :=
  sorry

end solve_for_x_l132_132891


namespace price_of_pants_l132_132666

theorem price_of_pants (S P H : ℝ) (h1 : 0.8 * S + P + H = 340) (h2 : S = (3 / 4) * P) (h3 : H = P + 10) : P = 91.67 :=
by sorry

end price_of_pants_l132_132666


namespace cliff_rock_collection_l132_132427

theorem cliff_rock_collection (S I : ℕ) 
  (h1 : I = S / 2) 
  (h2 : 2 * I / 3 = 40) : S + I = 180 := by
  sorry

end cliff_rock_collection_l132_132427


namespace red_light_cherries_cost_price_min_value_m_profit_l132_132309

-- Define the constants and cost conditions
def cost_price_red_light_cherries (x : ℝ) (y : ℝ) : Prop :=
  (6000 / (2 * x) - 100 = 1000 / x)

-- Define sales conditions and profit requirement
def min_value_m (m : ℝ) (profit : ℝ) : Prop :=
  (20 * 3 * m + 20 * (20 - 0.5 * m) + (28 - 20) * (50 - 3 * m - 20) >= profit)

-- Define the main proof goal statements
theorem red_light_cherries_cost_price :
  ∃ x, cost_price_red_light_cherries x 6000 ∧ 20 = x :=
sorry

theorem min_value_m_profit :
  ∃ m, min_value_m m 770 ∧ m >= 5 :=
sorry

end red_light_cherries_cost_price_min_value_m_profit_l132_132309


namespace set_diff_N_M_l132_132686

universe u

def set_difference {α : Type u} (A B : Set α) : Set α :=
  { x | x ∈ A ∧ x ∉ B }

def M : Set ℕ := { 1, 2, 3, 4, 5 }
def N : Set ℕ := { 1, 2, 3, 7 }

theorem set_diff_N_M : set_difference N M = { 7 } :=
  by
    sorry

end set_diff_N_M_l132_132686


namespace sum_of_squares_of_roots_l132_132581

theorem sum_of_squares_of_roots (x₁ x₂ : ℚ) (h : 6 * x₁^2 - 9 * x₁ + 5 = 0 ∧ 6 * x₂^2 - 9 * x₂ + 5 = 0 ∧ x₁ ≠ x₂) : x₁^2 + x₂^2 = 7 / 12 :=
by
  -- Since we are only required to write the statement, we leave the proof as sorry
  sorry

end sum_of_squares_of_roots_l132_132581


namespace smallest_prime_l132_132873

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ , m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n 

theorem smallest_prime :
  ∃ n : ℕ, n = 29 ∧ 
  n >= 10 ∧ n < 100 ∧
  is_prime n ∧
  ((n / 10) = 3) ∧ 
  is_composite (n % 10 * 10 + n / 10) ∧
  (n % 10 * 10 + n / 10) % 5 = 0 :=
by {
  sorry
}

end smallest_prime_l132_132873


namespace tens_digit_less_than_5_probability_l132_132093

theorem tens_digit_less_than_5_probability 
  (n : ℕ) 
  (hn : 10000 ≤ n ∧ n ≤ 99999)
  (h_even : ∃ k, n % 10 = 2 * k ∧ k < 5) :
  (∃ p, 0 ≤ p ∧ p ≤ 1 ∧ p = 1 / 2) :=
by
  sorry

end tens_digit_less_than_5_probability_l132_132093


namespace find_AC_l132_132739

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem find_AC :
  let AB := (2, 3)
  let BC := (1, -4)
  vector_add AB BC = (3, -1) :=
by 
  sorry

end find_AC_l132_132739


namespace inequality_proof_l132_132542

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
sorry

end inequality_proof_l132_132542


namespace walking_speed_l132_132468

theorem walking_speed (W : ℝ) : (1 / (1 / W + 1 / 8)) * 6 = 2.25 * (12 / 2) -> W = 4 :=
by
  intro h
  sorry

end walking_speed_l132_132468


namespace factorize_x_cubic_l132_132699

-- Define the function and the condition
def factorize (x : ℝ) : Prop := x^3 - 9 * x = x * (x + 3) * (x - 3)

-- Prove the factorization property
theorem factorize_x_cubic (x : ℝ) : factorize x :=
by
  sorry

end factorize_x_cubic_l132_132699


namespace quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l132_132187

-- Case 1
theorem quadratic_function_expression 
  (a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = 3) : 
  by {exact (a = -2 ∧ b = 3)} := sorry

theorem quadratic_function_range 
  (x : ℝ) 
  (h : -1 ≤ x ∧ x ≤ 2) : 
  (-3 ≤ -2*x^2 + 3*x + 2 ∧ -2*x^2 + 3*x + 2 ≤ 25/8) := sorry

-- Case 2
theorem quadratic_function_m_range 
  (m a b : ℝ) 
  (h₁ : 4 * a + 2 * b + 2 = 0) 
  (h₂ : a + b + 2 = m) 
  (h₃ : a > 0) : 
  m < 1 := sorry

end quadratic_function_expression_quadratic_function_range_quadratic_function_m_range_l132_132187


namespace abc_product_l132_132429

theorem abc_product (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 13) (h2 : b * c = 52) (h3 : c * a = 4) : a * b * c = 52 := 
  sorry

end abc_product_l132_132429


namespace veronica_max_area_l132_132593

noncomputable def max_area_garden : ℝ :=
  let l := 105
  let w := 420 - 2 * l
  l * w

theorem veronica_max_area : ∃ (A : ℝ), max_area_garden = 22050 :=
by
  use 22050
  show max_area_garden = 22050
  sorry

end veronica_max_area_l132_132593


namespace David_marks_in_Chemistry_l132_132507

theorem David_marks_in_Chemistry (e m p b avg c : ℕ) 
  (h1 : e = 91) 
  (h2 : m = 65) 
  (h3 : p = 82) 
  (h4 : b = 85) 
  (h5 : avg = 78) 
  (h6 : avg * 5 = e + m + p + b + c) :
  c = 67 := 
sorry

end David_marks_in_Chemistry_l132_132507


namespace students_without_A_l132_132285

theorem students_without_A (total_students : ℕ) (students_english : ℕ) 
  (students_math : ℕ) (students_both : ℕ) (students_only_math : ℕ) :
  total_students = 30 → students_english = 6 → students_math = 15 → 
  students_both = 3 → students_only_math = 1 →
  (total_students - (students_math - students_only_math + 
                     students_english - students_both + 
                     students_both) = 12) :=
by sorry

end students_without_A_l132_132285


namespace equal_roots_iff_k_eq_one_l132_132263

theorem equal_roots_iff_k_eq_one (k : ℝ) : (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + 2 = 0 → ∀ y : ℝ, 2 * k * y^2 + 4 * k * y + 2 = 0 → x = y) ↔ k = 1 := sorry

end equal_roots_iff_k_eq_one_l132_132263


namespace fixed_rate_calculation_l132_132270

theorem fixed_rate_calculation (f n : ℕ) (h1 : f + 4 * n = 220) (h2 : f + 7 * n = 370) : f = 20 :=
by
  sorry

end fixed_rate_calculation_l132_132270


namespace part1_inequality_part2_range_l132_132234

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 1)

-- Part 1: Prove that f(x) ≥ f(0) for all x
theorem part1_inequality : ∀ x : ℝ, f x ≥ f 0 :=
sorry

-- Part 2: Prove that the range of a satisfying 2f(x) ≥ f(a+1) for all x is -4.5 ≤ a ≤ 1.5
theorem part2_range (a : ℝ) (h : ∀ x : ℝ, 2 * f x ≥ f (a + 1)) : -4.5 ≤ a ∧ a ≤ 1.5 :=
sorry

end part1_inequality_part2_range_l132_132234


namespace patient_treatment_volume_l132_132255

noncomputable def total_treatment_volume : ℝ :=
  let drop_rate1 := 15     -- drops per minute for the first drip
  let ml_rate1 := 6 / 120  -- milliliters per drop for the first drip
  let drop_rate2 := 25     -- drops per minute for the second drip
  let ml_rate2 := 7.5 / 90 -- milliliters per drop for the second drip
  let total_time := 4 * 60 -- total minutes including breaks
  let break_time := 4 * 10 -- total break time in minutes
  let actual_time := total_time - break_time -- actual running time in minutes
  let total_drops1 := actual_time * drop_rate1
  let total_drops2 := actual_time * drop_rate2
  let volume1 := total_drops1 * ml_rate1
  let volume2 := total_drops2 * ml_rate2
  volume1 + volume2 -- total volume from both drips

theorem patient_treatment_volume : total_treatment_volume = 566.67 :=
  by
    -- Place the necessary calculation steps as assumptions or directly as one-liner
    sorry

end patient_treatment_volume_l132_132255


namespace joe_used_225_gallons_l132_132803

def initial_paint : ℕ := 360

def paint_first_week (initial : ℕ) : ℕ := initial / 4

def remaining_paint_after_first_week (initial : ℕ) : ℕ :=
  initial - paint_first_week initial

def paint_second_week (remaining : ℕ) : ℕ := remaining / 2

def total_paint_used (initial : ℕ) : ℕ :=
  paint_first_week initial + paint_second_week (remaining_paint_after_first_week initial)

theorem joe_used_225_gallons :
  total_paint_used initial_paint = 225 :=
by
  sorry

end joe_used_225_gallons_l132_132803


namespace negation_proposition_l132_132793

theorem negation_proposition {x : ℝ} (h : ∀ x > 0, Real.sin x > 0) : ∃ x > 0, Real.sin x ≤ 0 :=
sorry

end negation_proposition_l132_132793


namespace min_additional_games_l132_132716

-- Definitions of parameters
def initial_total_games : ℕ := 5
def initial_falcon_wins : ℕ := 2
def win_percentage_threshold : ℚ := 91 / 100

-- Theorem stating the minimum value for N
theorem min_additional_games (N : ℕ) : (initial_falcon_wins + N : ℚ) / (initial_total_games + N : ℚ) ≥ win_percentage_threshold → N ≥ 29 :=
by
  sorry

end min_additional_games_l132_132716


namespace parabola_through_point_l132_132116

-- Define the parabola equation property
def parabola (a x : ℝ) : ℝ := x^2 + (a+1) * x + a

-- Introduce the main problem statement
theorem parabola_through_point (a m : ℝ) (h : parabola a (-1) = m) : m = 0 :=
by
  sorry

end parabola_through_point_l132_132116


namespace rohan_food_percentage_l132_132182

noncomputable def rohan_salary : ℝ := 7500
noncomputable def rohan_savings : ℝ := 1500
noncomputable def house_rent_percentage : ℝ := 0.20
noncomputable def entertainment_percentage : ℝ := 0.10
noncomputable def conveyance_percentage : ℝ := 0.10
noncomputable def total_spent : ℝ := rohan_salary - rohan_savings
noncomputable def known_percentage : ℝ := house_rent_percentage + entertainment_percentage + conveyance_percentage

theorem rohan_food_percentage (F : ℝ) :
  total_spent = rohan_salary * (1 - known_percentage - F) →
  F = 0.20 :=
sorry

end rohan_food_percentage_l132_132182


namespace geom_sequence_next_term_l132_132344

def geom_seq (a r : ℕ → ℤ) (i : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * r i

theorem geom_sequence_next_term (y : ℤ) (a : ℕ → ℤ) (r : ℕ → ℤ) (n : ℕ) : 
  geom_seq a r 0 →
  a 0 = 3 →
  a 1 = 9 * y^2 →
  a 2 = 27 * y^4 →
  a 3 = 81 * y^6 →
  r 0 = 3 * y^2 →
  a 4 = 243 * y^8 :=
by
  intro h_seq h1 h2 h3 h4 hr
  sorry

end geom_sequence_next_term_l132_132344


namespace min_value_l132_132518

theorem min_value (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 3 * y + 3 * x * y = 6) : 2 * x + 3 * y ≥ 4 :=
sorry

end min_value_l132_132518


namespace positive_difference_largest_prime_factors_l132_132792

theorem positive_difference_largest_prime_factors :
  let p1 := 139
  let p2 := 29
  p1 - p2 = 110 := sorry

end positive_difference_largest_prime_factors_l132_132792


namespace range_of_a_l132_132920

noncomputable def y (a x : ℝ) : ℝ := a * Real.exp x + 3 * x
noncomputable def y_prime (a x : ℝ) : ℝ := a * Real.exp x + 3

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, a * Real.exp x + 3 = 0 ∧ a * Real.exp x + 3 * x < 0) → a < -3 :=
by
  sorry

end range_of_a_l132_132920


namespace completing_square_solution_l132_132585

theorem completing_square_solution (x : ℝ) : x^2 + 4 * x - 1 = 0 → (x + 2)^2 = 5 :=
by
  sorry

end completing_square_solution_l132_132585


namespace circumscribed_sphere_surface_area_l132_132990

theorem circumscribed_sphere_surface_area 
    (x y z : ℝ) 
    (h1 : x * y = Real.sqrt 6) 
    (h2 : y * z = Real.sqrt 2) 
    (h3 : z * x = Real.sqrt 3) : 
    4 * Real.pi * ((Real.sqrt (x^2 + y^2 + z^2)) / 2)^2 = 6 * Real.pi := 
by
  sorry

end circumscribed_sphere_surface_area_l132_132990


namespace minimum_problems_45_l132_132678

-- Define the types for problems and their corresponding points
structure Problem :=
(points : ℕ)

def isValidScore (s : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s

def minimumProblems (s : ℕ) (min_problems : ℕ) : Prop :=
  ∃ x y z : ℕ, 3 * x + 8 * y + 10 * z = s ∧ x + y + z = min_problems

-- Main statement
theorem minimum_problems_45 : minimumProblems 45 6 :=
by 
  sorry

end minimum_problems_45_l132_132678


namespace find_m_l132_132031

-- Define the arithmetic sequence and its properties
variable {α : Type*} [OrderedRing α]
variable (a : Nat → α) (S : Nat → α) (m : ℕ)

-- The conditions from the problem
variable (is_arithmetic_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
variable (sum_of_terms : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)
variable (m_gt_one : m > 1)
variable (condition1 : a (m - 1) + a (m + 1) - a m ^ 2 - 1 = 0)
variable (condition2 : S (2 * m - 1) = 39)

-- Prove that m = 20
theorem find_m : m = 20 :=
sorry

end find_m_l132_132031


namespace Cid_charges_5_for_car_wash_l132_132293

theorem Cid_charges_5_for_car_wash (x : ℝ) :
  5 * 20 + 10 * 30 + 15 * x = 475 → x = 5 :=
by
  intro h
  sorry

end Cid_charges_5_for_car_wash_l132_132293


namespace union_of_A_B_l132_132562

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l132_132562


namespace total_nails_used_l132_132015

-- Given definitions from the conditions
def square_side_length : ℕ := 36
def nails_per_side : ℕ := 40
def sides_of_square : ℕ := 4
def corners_of_square : ℕ := 4

-- Statement of the problem proof
theorem total_nails_used : nails_per_side * sides_of_square - corners_of_square = 156 := by
  sorry

end total_nails_used_l132_132015


namespace quadratic_roots_and_expression_value_l132_132897

theorem quadratic_roots_and_expression_value :
  let a := 3 + Real.sqrt 21
  let b := 3 - Real.sqrt 21
  (a ≥ b) →
  (∃ x : ℝ, x^2 - 6 * x + 11 = 23) →
  3 * a + 2 * b = 15 + Real.sqrt 21 :=
by
  intros a b h1 h2
  sorry

end quadratic_roots_and_expression_value_l132_132897


namespace solve_system_and_compute_l132_132999

-- Given system of equations
variables {x y : ℝ}
variables (h1 : 2 * x + y = 4) (h2 : x + 2 * y = 5)

-- Statement to prove
theorem solve_system_and_compute :
  (x - y = -1) ∧ (x + y = 3) ∧ ((1/3 * (x^2 - y^2)) * (x^2 - 2*x*y + y^2) = -1) :=
by
  sorry

end solve_system_and_compute_l132_132999


namespace find_digits_for_divisibility_l132_132981

theorem find_digits_for_divisibility (d1 d2 : ℕ) (h1 : d1 < 10) (h2 : d2 < 10) :
  (32 * 10^7 + d1 * 10^6 + 35717 * 10 + d2) % 72 = 0 →
  d1 = 2 ∧ d2 = 6 :=
by
  sorry

end find_digits_for_divisibility_l132_132981


namespace parabola_focus_coincides_hyperbola_focus_l132_132865

theorem parabola_focus_coincides_hyperbola_focus (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2 * p * x -> (3,0) = (3,0)) → 
  (∀ x y : ℝ, x^2 / 6 - y^2 / 3 = 1 -> x = 3) → 
  p = 6 :=
by
  sorry

end parabola_focus_coincides_hyperbola_focus_l132_132865


namespace proof_m_range_l132_132337

variable {x m : ℝ}

def A (m : ℝ) : Set ℝ := {x | x^2 + x + m + 2 = 0}
def B : Set ℝ := {x | x > 0}

theorem proof_m_range (h : A m ∩ B = ∅) : m ≤ -2 := 
sorry

end proof_m_range_l132_132337


namespace toms_age_l132_132935

theorem toms_age (T S : ℕ) (h1 : T = 2 * S - 1) (h2 : T + S = 14) : T = 9 :=
sorry

end toms_age_l132_132935


namespace square_of_binomial_conditions_l132_132915

variable (x a b m : ℝ)

theorem square_of_binomial_conditions :
  ∃ u v : ℝ, (x + a) * (x - a) = u^2 - v^2 ∧
             ∃ e f : ℝ, (-x - b) * (x - b) = - (e^2 - f^2) ∧
             ∃ g h : ℝ, (b + m) * (m - b) = g^2 - h^2 ∧
             ¬ ∃ p q : ℝ, (a + b) * (-a - b) = p^2 - q^2 :=
by
  sorry

end square_of_binomial_conditions_l132_132915


namespace number_of_yellow_marbles_l132_132426

theorem number_of_yellow_marbles 
  (total_marbles : ℕ) 
  (red_marbles : ℕ) 
  (blue_marbles : ℕ) 
  (yellow_marbles : ℕ)
  (h1 : total_marbles = 85) 
  (h2 : red_marbles = 14) 
  (h3 : blue_marbles = 3 * red_marbles) 
  (h4 : yellow_marbles = total_marbles - (red_marbles + blue_marbles)) :
  yellow_marbles = 29 :=
  sorry

end number_of_yellow_marbles_l132_132426


namespace bruno_coconuts_per_trip_is_8_l132_132033

-- Definitions related to the problem conditions
def total_coconuts : ℕ := 144
def barbie_coconuts_per_trip : ℕ := 4
def trips : ℕ := 12
def bruno_coconuts_per_trip : ℕ := total_coconuts - (barbie_coconuts_per_trip * trips)

-- The main theorem stating the question and the answer
theorem bruno_coconuts_per_trip_is_8 : bruno_coconuts_per_trip / trips = 8 :=
by
  sorry

end bruno_coconuts_per_trip_is_8_l132_132033


namespace a_share_in_gain_l132_132125

noncomputable def investment_share (x: ℝ) (total_gain: ℝ): ℝ := 
  let a_interest := x * 0.1
  let b_interest := (2 * x) * (7 / 100) * (1.5)
  let c_interest := (3 * x) * (10 / 100) * (1.33)
  let total_interest := a_interest + b_interest + c_interest
  a_interest

theorem a_share_in_gain (total_gain: ℝ) (a_share: ℝ) (x: ℝ)
  (hx: 0.709 * x = total_gain):
  investment_share x total_gain = a_share :=
sorry

end a_share_in_gain_l132_132125


namespace least_6_digit_number_sum_of_digits_l132_132869

-- Definitions based on conditions
def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def leaves_remainder2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Problem statement
theorem least_6_digit_number_sum_of_digits :
  ∃ n : ℕ, is_6_digit n ∧ leaves_remainder2 n 4 ∧ leaves_remainder2 n 610 ∧ leaves_remainder2 n 15 ∧ sum_of_digits n = 17 :=
sorry

end least_6_digit_number_sum_of_digits_l132_132869


namespace coconut_grove_l132_132694

theorem coconut_grove (x N : ℕ) (h1 : (x + 4) * 60 + x * N + (x - 4) * 180 = 3 * x * 100) (hx : x = 8) : N = 120 := 
by
  subst hx
  sorry

end coconut_grove_l132_132694


namespace price_of_turban_correct_l132_132550

noncomputable def initial_yearly_salary : ℝ := 90
noncomputable def initial_monthly_salary : ℝ := initial_yearly_salary / 12
noncomputable def raise : ℝ := 0.05 * initial_monthly_salary

noncomputable def first_3_months_salary : ℝ := 3 * initial_monthly_salary
noncomputable def second_3_months_salary : ℝ := 3 * (initial_monthly_salary + raise)
noncomputable def third_3_months_salary : ℝ := 3 * (initial_monthly_salary + 2 * raise)

noncomputable def total_cash_salary : ℝ := first_3_months_salary + second_3_months_salary + third_3_months_salary
noncomputable def actual_cash_received : ℝ := 80
noncomputable def price_of_turban : ℝ := actual_cash_received - total_cash_salary

theorem price_of_turban_correct : price_of_turban = 9.125 :=
by
  sorry

end price_of_turban_correct_l132_132550


namespace total_eyes_insects_l132_132742

-- Defining the conditions given in the problem
def numSpiders : Nat := 3
def numAnts : Nat := 50
def eyesPerSpider : Nat := 8
def eyesPerAnt : Nat := 2

-- Statement to prove: the total number of eyes among Nina's pet insects is 124
theorem total_eyes_insects : (numSpiders * eyesPerSpider + numAnts * eyesPerAnt) = 124 := by
  sorry

end total_eyes_insects_l132_132742


namespace smallest_a_l132_132359

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem smallest_a (a : ℕ) (h1 : is_factor 112 (a * 43 * 62 * 1311)) (h2 : is_factor 33 (a * 43 * 62 * 1311)) : a = 1848 :=
by
  sorry

end smallest_a_l132_132359


namespace simplify_expression_l132_132657

theorem simplify_expression : (225 / 10125) * 45 = 1 := by
  sorry

end simplify_expression_l132_132657


namespace distance_from_circumcenter_to_orthocenter_l132_132088

variables {A B C A1 H O : Type}

-- Condition Definitions
variable (acute_triangle : Prop)
variable (is_altitude : Prop)
variable (is_orthocenter : Prop)
variable (AH_dist : ℝ := 3)
variable (A1H_dist : ℝ := 2)
variable (circum_radius : ℝ := 4)

-- Prove the distance from O to H
theorem distance_from_circumcenter_to_orthocenter
  (h1 : acute_triangle)
  (h2 : is_altitude)
  (h3 : is_orthocenter)
  (h4 : AH_dist = 3)
  (h5 : A1H_dist = 2)
  (h6 : circum_radius = 4) : 
  ∃ (d : ℝ), d = 2 := 
sorry

end distance_from_circumcenter_to_orthocenter_l132_132088


namespace find_x_l132_132554

variable (n : ℝ) (x : ℝ)

theorem find_x (h1 : n = 15.0) (h2 : 3 * n - x = 40) : x = 5.0 :=
by
  sorry

end find_x_l132_132554


namespace square_pyramid_properties_l132_132034

-- Definitions for the square pyramid with a square base
def square_pyramid_faces : Nat := 4 + 1
def square_pyramid_edges : Nat := 4 + 4
def square_pyramid_vertices : Nat := 4 + 1

-- Definition for the number of diagonals in a square
def diagonals_in_square_base (n : Nat) : Nat := n * (n - 3) / 2

-- Theorem statement
theorem square_pyramid_properties :
  (square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18) ∧ (diagonals_in_square_base 4 = 2) :=
by
  sorry

end square_pyramid_properties_l132_132034


namespace smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l132_132917

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999
def divisible_by_35 (n : ℕ) : Prop := n % 35 = 0

theorem smallest_and_second_smallest_four_digit_numbers_divisible_by_35 :
  ∃ a b : ℕ, 
    is_four_digit a ∧ 
    is_four_digit b ∧ 
    divisible_by_35 a ∧ 
    divisible_by_35 b ∧ 
    a < b ∧ 
    ∀ c : ℕ, is_four_digit c → divisible_by_35 c → a ≤ c → (c = a ∨ c = b) :=
by
  sorry

end smallest_and_second_smallest_four_digit_numbers_divisible_by_35_l132_132917


namespace one_cow_eating_one_bag_in_12_days_l132_132579

def average_days_to_eat_one_bag (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) : ℕ :=
  total_days / (total_bags / number_of_cows)

theorem one_cow_eating_one_bag_in_12_days (total_bags : ℕ) (total_days : ℕ) (number_of_cows : ℕ) (h_total_bags : total_bags = 50) (h_total_days : total_days = 20) (h_number_of_cows : number_of_cows = 30) : 
  average_days_to_eat_one_bag total_bags total_days number_of_cows = 12 := by
  sorry

end one_cow_eating_one_bag_in_12_days_l132_132579


namespace Alan_shells_l132_132310

theorem Alan_shells (l b a : ℕ) (h1 : l = 36) (h2 : b = l / 3) (h3 : a = 4 * b) : a = 48 :=
by
sorry

end Alan_shells_l132_132310


namespace apples_per_bucket_l132_132923

theorem apples_per_bucket (total_apples buckets : ℕ) (h1 : total_apples = 56) (h2 : buckets = 7) : 
  (total_apples / buckets) = 8 :=
by
  sorry

end apples_per_bucket_l132_132923


namespace repeating_decimal_as_fraction_l132_132931

def repeating_decimal := 567 / 999

theorem repeating_decimal_as_fraction : repeating_decimal = 21 / 37 := by
  sorry

end repeating_decimal_as_fraction_l132_132931


namespace factor_polynomial_l132_132152

theorem factor_polynomial (x : ℝ) : 54*x^3 - 135*x^5 = 27*x^3*(2 - 5*x^2) := 
by
  sorry

end factor_polynomial_l132_132152


namespace final_price_for_tiffany_l132_132863

noncomputable def calculate_final_price (n : ℕ) (c : ℝ) (d : ℝ) (s : ℝ) : ℝ :=
  let total_cost := n * c
  let discount := d * total_cost
  let discounted_price := total_cost - discount
  let sales_tax := s * discounted_price
  let final_price := discounted_price + sales_tax
  final_price

theorem final_price_for_tiffany :
  calculate_final_price 9 4.50 0.20 0.07 = 34.67 :=
by
  sorry

end final_price_for_tiffany_l132_132863


namespace maximum_sum_each_side_equals_22_l132_132017

theorem maximum_sum_each_side_equals_22 (A B C D : ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ 10)
  → (∀ S, S = A ∨ S = B ∨ S = C ∨ S = D ∧ A + B + C + D = 33)
  → (A + B + C + D + 55) / 4 = 22 :=
by
  sorry

end maximum_sum_each_side_equals_22_l132_132017


namespace cos_squared_alpha_plus_pi_over_4_l132_132964

theorem cos_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 :=
by
  sorry

end cos_squared_alpha_plus_pi_over_4_l132_132964


namespace extra_apples_l132_132444

-- Defining the given conditions
def redApples : Nat := 60
def greenApples : Nat := 34
def studentsWantFruit : Nat := 7

-- Defining the theorem to prove the number of extra apples
theorem extra_apples : redApples + greenApples - studentsWantFruit = 87 := by
  sorry

end extra_apples_l132_132444


namespace derivative_of_f_l132_132342

def f (x : ℝ) : ℝ := 2 * x + 3

theorem derivative_of_f :
  ∀ x : ℝ, (deriv f x) = 2 :=
by 
  sorry

end derivative_of_f_l132_132342


namespace area_of_rhombus_l132_132102

-- Defining conditions for the problem
def d1 : ℝ := 40   -- Length of the first diagonal in meters
def d2 : ℝ := 30   -- Length of the second diagonal in meters

-- Calculating the area of the rhombus
noncomputable def area : ℝ := (d1 * d2) / 2

-- Statement of the theorem
theorem area_of_rhombus : area = 600 := by
  sorry

end area_of_rhombus_l132_132102


namespace total_nickels_l132_132395

-- Definition of the number of original nickels Mary had
def original_nickels := 7

-- Definition of the number of nickels her dad gave her
def added_nickels := 5

-- Prove that the total number of nickels Mary has now is 12
theorem total_nickels : original_nickels + added_nickels = 12 := by
  sorry

end total_nickels_l132_132395


namespace exp_log_pb_eq_log_ba_l132_132388

noncomputable def log_b (b a : ℝ) := Real.log a / Real.log b

theorem exp_log_pb_eq_log_ba (a b p : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : p = log_b b (log_b b a) / log_b b a) :
  a^p = log_b b a :=
by
  sorry

end exp_log_pb_eq_log_ba_l132_132388


namespace correctness_statement_l132_132642

-- Given points A, B, C are on the specific parabola
variable (a c x1 x2 x3 y1 y2 y3 : ℝ)
variable (ha : a < 0) -- a < 0 since the parabola opens upwards
variable (hA : y1 = - (a / 4) * x1^2 + a * x1 + c)
variable (hB : y2 = a + c) -- B is the vertex
variable (hC : y3 = - (a / 4) * x3^2 + a * x3 + c)
variable (hOrder : y1 > y3 ∧ y3 ≥ y2)

theorem correctness_statement : abs (x1 - x2) > abs (x3 - x2) :=
sorry

end correctness_statement_l132_132642


namespace find_b_l132_132541

theorem find_b (a b c : ℚ) (h : (3 * x^2 - 4 * x + 2) * (a * x^2 + b * x + c) = 9 * x^4 - 10 * x^3 + 5 * x^2 - 8 * x + 4)
  (ha : a = 3) : b = 2 / 3 :=
by
  sorry

end find_b_l132_132541


namespace stratified_sampling_first_level_l132_132896

-- Definitions from the conditions
def num_senior_teachers : ℕ := 90
def num_first_level_teachers : ℕ := 120
def num_second_level_teachers : ℕ := 170
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_second_level_teachers
def sample_size : ℕ := 38

-- Definition of the stratified sampling result
def num_first_level_selected : ℕ := (num_first_level_teachers * sample_size) / total_teachers

-- The statement to be proven
theorem stratified_sampling_first_level : num_first_level_selected = 12 :=
by
  sorry

end stratified_sampling_first_level_l132_132896


namespace incorrect_multiplicative_inverse_product_l132_132028

theorem incorrect_multiplicative_inverse_product:
  ∃ (a b : ℝ), a + b = 0 ∧ a * b ≠ 1 :=
by
  sorry

end incorrect_multiplicative_inverse_product_l132_132028


namespace next_term_in_geom_sequence_l132_132209

   /- Define the given geometric sequence as a function in Lean -/

   def geom_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ n

   theorem next_term_in_geom_sequence (x : ℤ) (n : ℕ) 
     (h₁ : geom_sequence 3 (-3*x) 0 = 3)
     (h₂ : geom_sequence 3 (-3*x) 1 = -9*x)
     (h₃ : geom_sequence 3 (-3*x) 2 = 27*(x^2))
     (h₄ : geom_sequence 3 (-3*x) 3 = -81*(x^3)) :
     geom_sequence 3 (-3*x) 4 = 243*(x^4) := 
   sorry
   
end next_term_in_geom_sequence_l132_132209


namespace Maria_students_l132_132073

variable (M J : ℕ)

def conditions : Prop :=
  (M = 4 * J) ∧ (M + J = 2500)

theorem Maria_students : conditions M J → M = 2000 :=
by
  intro h
  sorry

end Maria_students_l132_132073


namespace avg_hamburgers_per_day_l132_132186

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 49) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 7 :=
by {
  sorry
}

end avg_hamburgers_per_day_l132_132186


namespace circumcircle_area_l132_132147

theorem circumcircle_area (a b c A B C : ℝ) (h : a * Real.cos B + b * Real.cos A = 4 * Real.sin C) :
    π * (2 : ℝ) ^ 2 = 4 * π :=
by
  sorry

end circumcircle_area_l132_132147


namespace fraction_simplification_l132_132849

theorem fraction_simplification (b : ℝ) (hb : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b^2 = 10 / 81 :=
by
  rw [hb]
  sorry

end fraction_simplification_l132_132849


namespace rita_swimming_months_l132_132011

theorem rita_swimming_months
    (total_required_hours : ℕ := 1500)
    (backstroke_hours : ℕ := 50)
    (breaststroke_hours : ℕ := 9)
    (butterfly_hours : ℕ := 121)
    (monthly_hours : ℕ := 220) :
    (total_required_hours - (backstroke_hours + breaststroke_hours + butterfly_hours)) / monthly_hours = 6 := 
by
    -- Proof is omitted
    sorry

end rita_swimming_months_l132_132011


namespace minimize_expression_10_l132_132514

theorem minimize_expression_10 (n : ℕ) (h : 0 < n) : 
  (∃ m : ℕ, 0 < m ∧ (∀ k : ℕ, 0 < k → (n = k) → (n = 10))) :=
by
  sorry

end minimize_expression_10_l132_132514


namespace lengths_of_angle_bisectors_areas_of_triangles_l132_132890

-- Given conditions
variables (x y : ℝ) (S1 S2 : ℝ)
variables (hx1 : x + y = 15) (hx2 : x / y = 3 / 2)
variables (hS1 : S1 / S2 = 9 / 4) (hS2 : S1 - S2 = 6)

-- Prove the lengths of the angle bisectors
theorem lengths_of_angle_bisectors :
  x = 9 ∧ y = 6 :=
by sorry

-- Prove the areas of the triangles
theorem areas_of_triangles :
  S1 = 54 / 5 ∧ S2 = 24 / 5 :=
by sorry

end lengths_of_angle_bisectors_areas_of_triangles_l132_132890


namespace xiaoming_problem_l132_132886

theorem xiaoming_problem (a x : ℝ) 
  (h1 : 20.18 * a - 20.18 = x)
  (h2 : x = 2270.25) : 
  a = 113.5 := 
by 
  sorry

end xiaoming_problem_l132_132886


namespace initial_chocolate_amount_l132_132903

-- Define the problem conditions

def initial_dough (d : ℕ) := d = 36
def left_over_chocolate (lo_choc : ℕ) := lo_choc = 4
def chocolate_percentage (p : ℚ) := p = 0.20
def total_weight (d : ℕ) (c_choc : ℕ) := d + c_choc - 4
def chocolate_used (c_choc : ℕ) (lo_choc : ℕ) := c_choc - lo_choc

-- The main proof goal
theorem initial_chocolate_amount (d : ℕ) (lo_choc : ℕ) (p : ℚ) (C : ℕ) :
  initial_dough d → left_over_chocolate lo_choc → chocolate_percentage p →
  p * (total_weight d C) = chocolate_used C lo_choc → C = 13 :=
by
  intros hd hlc hp h
  sorry

end initial_chocolate_amount_l132_132903


namespace find_x_l132_132064

theorem find_x (x y : ℝ) (h : y ≠ -5 * x) : (x - 5) / (5 * x + y) = 0 → x = 5 := by
  sorry

end find_x_l132_132064


namespace candy_count_l132_132243

theorem candy_count (initial_candy : ℕ) (eaten_candy : ℕ) (received_candy : ℕ) (final_candy : ℕ) :
  initial_candy = 33 → eaten_candy = 17 → received_candy = 19 → final_candy = 35 :=
by
  intros h_initial h_eaten h_received
  sorry

end candy_count_l132_132243


namespace find_a_l132_132415

noncomputable def f (t : ℝ) (a : ℝ) : ℝ := (1 / (Real.cos t)) + (a / (1 - (Real.cos t)))

theorem find_a (t : ℝ) (a : ℝ) (h1 : 0 < t) (h2 : t < (Real.pi / 2)) (h3 : 0 < a) (h4 : ∀ t, 0 < t ∧ t < (Real.pi / 2) → f t a = 16) :
  a = 9 :=
sorry

end find_a_l132_132415


namespace max_profit_l132_132754

noncomputable def profit_A (x : ℕ) : ℝ := -↑x^2 + 21 * ↑x
noncomputable def profit_B (x : ℕ) : ℝ := 2 * ↑x
noncomputable def total_profit (x : ℕ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit : 
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 120 := sorry

end max_profit_l132_132754


namespace arithmetic_result_l132_132307

theorem arithmetic_result :
  1325 + (572 / 52) - 225 + (2^3) = 1119 :=
by
  sorry

end arithmetic_result_l132_132307


namespace polynomial_abc_value_l132_132819

theorem polynomial_abc_value (a b c : ℝ) (h : a * (x^2) + b * x + c = (x - 1) * (x - 2)) : a * b * c = -6 :=
by
  sorry

end polynomial_abc_value_l132_132819


namespace value_is_twenty_l132_132893

theorem value_is_twenty (n : ℕ) (h : n = 16) : 32 - 12 = 20 :=
by {
  -- Simplification of the proof process
  sorry
}

end value_is_twenty_l132_132893


namespace h_at_3_l132_132144

noncomputable def f (x : ℝ) := 3 * x + 4
noncomputable def g (x : ℝ) := Real.sqrt (f x) - 3
noncomputable def h (x : ℝ) := g (f x)

theorem h_at_3 : h 3 = Real.sqrt 43 - 3 := by
  sorry

end h_at_3_l132_132144


namespace sum_first_two_integers_l132_132496

/-- Prove that the sum of the first two integers n > 1 such that 3^n is divisible by n 
and 3^n - 1 is divisible by n - 1 is equal to 30. -/
theorem sum_first_two_integers (n : ℕ) (h1 : n > 1) (h2 : 3 ^ n % n = 0) (h3 : (3 ^ n - 1) % (n - 1) = 0) : 
  n = 3 ∨ n = 27 → n + 3 + 27 = 30 :=
sorry

end sum_first_two_integers_l132_132496


namespace floor_width_is_120_l132_132663

def tile_length := 25 -- cm
def tile_width := 16 -- cm
def floor_length := 180 -- cm
def max_tiles := 54

theorem floor_width_is_120 :
  ∃ (W : ℝ), W = 120 ∧ (floor_length / tile_width) * W = max_tiles * (tile_length * tile_width) := 
sorry

end floor_width_is_120_l132_132663


namespace problem1_solution_set_problem2_range_of_a_l132_132096

section Problem1

def f1 (x : ℝ) : ℝ := |x - 4| + |x - 2|

theorem problem1_solution_set (a : ℝ) (h : a = 2) :
  { x : ℝ | f1 x > 10 } = { x : ℝ | x > 8 ∨ x < -2 } := sorry

end Problem1


section Problem2

def f2 (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem problem2_range_of_a (f_geq : ∀ x : ℝ, f2 x a ≥ 1) :
  a ≥ 5 ∨ a ≤ 3 := sorry

end Problem2

end problem1_solution_set_problem2_range_of_a_l132_132096


namespace probability_no_defective_pencils_l132_132880

-- Definitions based on conditions
def total_pencils : ℕ := 11
def defective_pencils : ℕ := 2
def selected_pencils : ℕ := 3

-- Helper function to compute combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The proof statement
theorem probability_no_defective_pencils :
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination (total_pencils - defective_pencils) selected_pencils
  total_ways ≠ 0 → 
  (non_defective_ways / total_ways : ℚ) = 28 / 55 := 
by
  sorry

end probability_no_defective_pencils_l132_132880


namespace find_original_price_l132_132522

-- Definitions based on the conditions
def original_price_increased (x : ℝ) : ℝ := 1.25 * x
def loan_payment (total_cost : ℝ) : ℝ := 0.75 * total_cost
def own_funds (total_cost : ℝ) : ℝ := 0.25 * total_cost

-- Condition values
def new_home_cost : ℝ := 500000
def loan_amount := loan_payment new_home_cost
def funds_paid := own_funds new_home_cost

-- Proof statement
theorem find_original_price : 
  ∃ x : ℝ, original_price_increased x = funds_paid ↔ x = 100000 :=
by
  -- Placeholder for actual proof
  sorry

end find_original_price_l132_132522


namespace Anne_cleaning_time_l132_132877

theorem Anne_cleaning_time (B A C : ℚ) 
  (h1 : B + A + C = 1 / 6) 
  (h2 : B + 2 * A + 3 * C = 1 / 2)
  (h3 : B + A = 1 / 4)
  (h4 : B + C = 1 / 3) : 
  A = 1 / 6 := 
sorry

end Anne_cleaning_time_l132_132877


namespace car_speed_first_hour_l132_132131

theorem car_speed_first_hour (x : ℕ) :
  (x + 60) / 2 = 75 → x = 90 :=
by
  -- To complete the proof in Lean, we would need to solve the equation,
  -- reversing the steps provided in the solution. 
  -- But as per instructions, we don't need the proof, hence we put sorry.
  sorry

end car_speed_first_hour_l132_132131
