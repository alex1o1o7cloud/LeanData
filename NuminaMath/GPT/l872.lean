import Mathlib

namespace glucose_solution_volume_l872_87253

theorem glucose_solution_volume
  (h1 : 6.75 / 45 = 15 / x) :
  x = 100 :=
by
  sorry

end glucose_solution_volume_l872_87253


namespace number_of_members_l872_87263

theorem number_of_members (n : ℕ) (h : n^2 = 9801) : n = 99 :=
sorry

end number_of_members_l872_87263


namespace trig_identity_l872_87232

theorem trig_identity :
  (2 * Real.sin (46 * Real.pi / 180) - Real.sqrt 3 * Real.cos (74 * Real.pi / 180)) / Real.cos (16 * Real.pi / 180) = 1 :=
by
  sorry

end trig_identity_l872_87232


namespace sum_of_possible_values_of_G_F_l872_87217

theorem sum_of_possible_values_of_G_F (G F : ℕ) (hG : 0 ≤ G ∧ G ≤ 9) (hF : 0 ≤ F ∧ F ≤ 9)
  (hdiv : (G + 2 + 4 + 3 + F + 1 + 6) % 9 = 0) : G + F = 2 ∨ G + F = 11 → 2 + 11 = 13 :=
by { sorry }

end sum_of_possible_values_of_G_F_l872_87217


namespace one_third_of_flour_l872_87266

-- Definition of the problem conditions
def initial_flour : ℚ := 5 + 2 / 3
def portion : ℚ := 1 / 3

-- Definition of the theorem to prove
theorem one_third_of_flour : portion * initial_flour = 1 + 8 / 9 :=
by {
  -- Placeholder proof
  sorry
}

end one_third_of_flour_l872_87266


namespace complex_power_identity_l872_87243

theorem complex_power_identity (z : ℂ) (i : ℂ) 
  (h1 : z = (1 + i) / Real.sqrt 2) 
  (h2 : z^2 = i) : 
  z^100 = -1 := 
  sorry

end complex_power_identity_l872_87243


namespace regular_price_of_shrimp_l872_87240

theorem regular_price_of_shrimp 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (quarter_pound_price : ℝ) 
  (full_pound_price : ℝ) 
  (price_relation : quarter_pound_price = discounted_price * (1 - discount_rate) / 4) 
  (discounted_value : quarter_pound_price = 2) 
  (given_discount_rate : discount_rate = 0.6) 
  (given_discounted_price : discounted_price = full_pound_price) 
  : full_pound_price = 20 :=
by {
  sorry
}

end regular_price_of_shrimp_l872_87240


namespace train_speed_l872_87270

theorem train_speed (length : ℕ) (cross_time : ℕ) (speed : ℝ)
    (h1 : length = 250)
    (h2 : cross_time = 3)
    (h3 : speed = (length / cross_time : ℝ) * 3.6) :
    speed = 300 := 
sorry

end train_speed_l872_87270


namespace relationship_between_D_and_A_l872_87271

variables (A B C D : Prop)

def sufficient_not_necessary (P Q : Prop) : Prop := (P → Q) ∧ ¬ (Q → P)
def necessary_not_sufficient (P Q : Prop) : Prop := (Q → P) ∧ ¬ (P → Q)
def necessary_and_sufficient (P Q : Prop) : Prop := (P ↔ Q)

-- Conditions
axiom h1 : sufficient_not_necessary A B
axiom h2 : necessary_not_sufficient C B
axiom h3 : necessary_and_sufficient D C

-- Proof Goal
theorem relationship_between_D_and_A : necessary_not_sufficient D A :=
by
  sorry

end relationship_between_D_and_A_l872_87271


namespace find_x_of_parallel_vectors_l872_87227

theorem find_x_of_parallel_vectors
  (x : ℝ)
  (p : ℝ × ℝ := (2, -3))
  (q : ℝ × ℝ := (x, 6))
  (h : ∃ k : ℝ, q = k • p) :
  x = -4 :=
sorry

end find_x_of_parallel_vectors_l872_87227


namespace greater_expected_area_l872_87237

/-- Let X be a random variable representing a single roll of a die, which can take integer values from 1 through 6. -/
def X : Type := { x : ℕ // 1 ≤ x ∧ x ≤ 6 }

/-- Define independent random variables A and B representing the outcomes of Asya’s die rolls, which can take integer values from 1 through 6 with equal probability. -/
noncomputable def A : Type := { a : ℕ // 1 ≤ a ∧ a ≤ 6 }
noncomputable def B : Type := { b : ℕ // 1 ≤ b ∧ b ≤ 6 }

/-- The expected value of a random variable taking integer values from 1 through 6. 
    E[X] = (1 + 2 + 3 + 4 + 5 + 6) / 6 = 3.5, and E[X^2] = (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) / 6 = 15.1667 -/
noncomputable def expected_X_squared : ℝ := 91 / 6

/-- The expected value of the product of two independent random variables each taking integer values from 1 through 6. 
    E[A * B] = E[A] * E[B] = 3.5 * 3.5 = 12.25 -/
noncomputable def expected_A_times_B : ℝ := 12.25

/-- Prove that the expected area of Vasya's square is greater than Asya's rectangle.
    i.e., E[X^2] > E[A * B] -/
theorem greater_expected_area : expected_X_squared > expected_A_times_B :=
sorry

end greater_expected_area_l872_87237


namespace remainder_21_l872_87293

theorem remainder_21 (y : ℤ) (k : ℤ) (h : y = 288 * k + 45) : y % 24 = 21 := 
  sorry

end remainder_21_l872_87293


namespace total_tiles_l872_87236

theorem total_tiles (s : ℕ) (h1 : true) (h2 : true) (h3 : true) (h4 : true) (h5 : 4 * s - 4 = 100): s * s = 676 :=
by
  sorry

end total_tiles_l872_87236


namespace find_theta_l872_87229

def rectangle : Type := sorry
def angle (α : ℝ) : Prop := 0 ≤ α ∧ α < 180

-- Given conditions in the problem
variables {α β γ δ θ : ℝ}

axiom angle_10 : angle 10
axiom angle_14 : angle 14
axiom angle_33 : angle 33
axiom angle_26 : angle 26

axiom zig_zag_angles (a b c d e f : ℝ) :
  a = 26 ∧ f = 10 ∧
  26 + b = 33 ∧ b = 7 ∧
  e + 10 = 14 ∧ e = 4 ∧
  c = b ∧ d = e ∧
  θ = c + d

theorem find_theta : θ = 11 :=
sorry

end find_theta_l872_87229


namespace measure_of_angle_A_l872_87275

theorem measure_of_angle_A
    (A B : ℝ)
    (h1 : A + B = 90)
    (h2 : A = 3 * B) :
    A = 67.5 :=
by
  sorry

end measure_of_angle_A_l872_87275


namespace calculation_correct_l872_87257

-- Defining the initial values
def a : ℕ := 20 ^ 10
def b : ℕ := 20 ^ 9
def c : ℕ := 10 ^ 6
def d : ℕ := 2 ^ 12

-- The expression we need to prove
theorem calculation_correct : ((a / b) ^ 3 * c) / d = 1953125 :=
by
  sorry

end calculation_correct_l872_87257


namespace cards_per_deck_l872_87273

theorem cards_per_deck (decks : ℕ) (cards_per_layer : ℕ) (layers : ℕ) 
  (h_decks : decks = 16) 
  (h_cards_per_layer : cards_per_layer = 26) 
  (h_layers : layers = 32) 
  (total_cards_used : ℕ := cards_per_layer * layers) 
  (number_of_cards_per_deck : ℕ := total_cards_used / decks) :
  number_of_cards_per_deck = 52 :=
by 
  sorry

end cards_per_deck_l872_87273


namespace none_of_these_l872_87259

def y_values_match (f : ℕ → ℕ) : Prop :=
  f 0 = 200 ∧ f 1 = 140 ∧ f 2 = 80 ∧ f 3 = 20 ∧ f 4 = 0

theorem none_of_these :
  ¬ (∃ f : ℕ → ℕ, 
    (∀ x, f x = 200 - 15 * x ∨ 
    f x = 200 - 20 * x + 5 * x^2 ∨ 
    f x = 200 - 30 * x + 10 * x^2 ∨ 
    f x = 150 - 50 * x) ∧ 
    y_values_match f) :=
by sorry

end none_of_these_l872_87259


namespace labor_arrangement_count_l872_87246

theorem labor_arrangement_count (volunteers : ℕ) (choose_one_day : ℕ) (days : ℕ) 
    (h_volunteers : volunteers = 7) 
    (h_choose_one_day : choose_one_day = 3) 
    (h_days : days = 2) : 
    (Nat.choose volunteers choose_one_day) * (Nat.choose (volunteers - choose_one_day) choose_one_day) = 140 := 
by
  sorry

end labor_arrangement_count_l872_87246


namespace positive_inequality_l872_87261

open Real

/-- Given positive real numbers x, y, z such that xyz ≥ 1, prove that
    (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + x^2 + z^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0.
-/
theorem positive_inequality (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h : x * y * z ≥ 1) : 
  (x^5 - x^2) / (x^5 + y^2 + z^2) + 
  (y^5 - y^2) / (y^5 + x^2 + z^2) + 
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 :=
by
  sorry

end positive_inequality_l872_87261


namespace problem1_problem2_problem3_l872_87280

-- Problem 1
theorem problem1 : 13 + (-7) - (-9) + 5 * (-2) = 5 :=
by 
  sorry

-- Problem 2
theorem problem2 : abs (-7 / 2) * (12 / 7) / (4 / 3) / (3 ^ 2) = 1 / 2 :=
by 
  sorry

-- Problem 3
theorem problem3 : -1^4 - (1 / 6) * (2 - (-3)^2) = 1 / 6 :=
by 
  sorry

end problem1_problem2_problem3_l872_87280


namespace find_a_l872_87213

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb 2 (x^2 + a)

theorem find_a (a : ℝ) : f 3 a = 1 → a = -7 :=
by
  intro h
  unfold f at h
  sorry

end find_a_l872_87213


namespace complement_is_correct_l872_87286

def U : Set ℝ := {x | x^2 ≤ 4}
def A : Set ℝ := {x | abs (x + 1) ≤ 1}
def complement_U_A : Set ℝ := U \ A

theorem complement_is_correct :
  complement_U_A = {x | 0 < x ∧ x ≤ 2} := by
  sorry

end complement_is_correct_l872_87286


namespace jimin_notebooks_proof_l872_87200

variable (m f o n : ℕ)

theorem jimin_notebooks_proof (hm : m = 7) (hf : f = 14) (ho : o = 33) (hn : n = o + m + f) :
  n - o = 21 := by
  sorry

end jimin_notebooks_proof_l872_87200


namespace investment_amount_l872_87269

theorem investment_amount (x y : ℝ) (hx : x ≤ 11000) (hy : 0.07 * x + 0.12 * y ≥ 2450) : x + y = 25000 := 
sorry

end investment_amount_l872_87269


namespace grace_clyde_ratio_l872_87250

theorem grace_clyde_ratio (C G : ℕ) (h1 : G = C + 35) (h2 : G = 40) : G / C = 8 :=
by sorry

end grace_clyde_ratio_l872_87250


namespace sandy_initial_amount_l872_87298

theorem sandy_initial_amount 
  (cost_shirt : ℝ) (cost_jacket : ℝ) (found_money : ℝ)
  (h1 : cost_shirt = 12.14) (h2 : cost_jacket = 9.28) (h3 : found_money = 7.43) : 
  (cost_shirt + cost_jacket + found_money = 28.85) :=
by
  rw [h1, h2, h3]
  norm_num

end sandy_initial_amount_l872_87298


namespace joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l872_87230

noncomputable def distance_joseph : ℝ := 48 * 2.5 + 60 * 1.5
noncomputable def distance_kyle : ℝ := 70 * 2 + 63 * 2.5
noncomputable def distance_emily : ℝ := 65 * 3

theorem joseph_vs_kyle : distance_joseph - distance_kyle = -87.5 := by
  unfold distance_joseph
  unfold distance_kyle
  sorry

theorem emily_vs_joseph : distance_emily - distance_joseph = -15 := by
  unfold distance_emily
  unfold distance_joseph
  sorry

theorem emily_vs_kyle : distance_emily - distance_kyle = -102.5 := by
  unfold distance_emily
  unfold distance_kyle
  sorry

end joseph_vs_kyle_emily_vs_joseph_emily_vs_kyle_l872_87230


namespace inequality_part_1_inequality_part_2_l872_87287

theorem inequality_part_1 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≥ 1 := by
sorry

theorem inequality_part_2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 / (b + c)) + (b^2 / (a + c)) + (c^2 / (a + b)) ≥ 1 / 2 := by
sorry

end inequality_part_1_inequality_part_2_l872_87287


namespace sphere_surface_area_l872_87216

noncomputable def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2

theorem sphere_surface_area (r_circle r_distance : ℝ) :
  (Real.pi * r_circle^2 = 16 * Real.pi) →
  (r_distance = 3) →
  (surface_area_of_sphere (Real.sqrt (r_distance^2 + r_circle^2)) = 100 * Real.pi) := by
sorry

end sphere_surface_area_l872_87216


namespace car_capacities_rental_plans_l872_87245

-- Define the capacities for part 1
def capacity_A : ℕ := 3
def capacity_B : ℕ := 4

theorem car_capacities (x y : ℕ) (h₁ : 2 * x + y = 10) (h₂ : x + 2 * y = 11) : 
  x = capacity_A ∧ y = capacity_B := by
  sorry

-- Define the valid rental plans for part 2
def valid_rental_plan (a b : ℕ) : Prop :=
  3 * a + 4 * b = 31

theorem rental_plans (a b : ℕ) (h : valid_rental_plan a b) : 
  (a = 1 ∧ b = 7) ∨ (a = 5 ∧ b = 4) ∨ (a = 9 ∧ b = 1) := by
  sorry

end car_capacities_rental_plans_l872_87245


namespace remaining_bottles_after_2_days_l872_87242

-- Definitions based on the conditions:
def initial_bottles : ℕ := 24
def fraction_first_day : ℚ := 1 / 3
def fraction_second_day : ℚ := 1 / 2

-- Theorem statement proving the remaining number of bottles after 2 days
theorem remaining_bottles_after_2_days : 
    (initial_bottles - initial_bottles * fraction_first_day) - 
    ((initial_bottles - initial_bottles * fraction_first_day) * fraction_second_day) = 8 := 
by 
    -- Skipping the proof
    sorry

end remaining_bottles_after_2_days_l872_87242


namespace y_intercept_of_line_l872_87294

theorem y_intercept_of_line : 
  ∃ y : ℝ, ∀ x : ℝ, (3 * x - 4 * y = 12) ∧ x = 0 → y = -3 := by
  -- proof skipped
  sorry

end y_intercept_of_line_l872_87294


namespace ms_warren_walking_time_l872_87210

/-- 
Ms. Warren ran at 6 mph for 20 minutes. After the run, 
she walked at 2 mph for a certain amount of time. 
She ran and walked a total of 3 miles.
-/
def time_spent_walking (running_speed walking_speed : ℕ) (running_time_minutes : ℕ) (total_distance : ℕ) : ℕ := 
  let running_time_hours := running_time_minutes / 60;
  let distance_ran := running_speed * running_time_hours;
  let distance_walked := total_distance - distance_ran;
  let time_walked_hours := distance_walked / walking_speed;
  time_walked_hours * 60

theorem ms_warren_walking_time :
  time_spent_walking 6 2 20 3 = 30 :=
by
  sorry

end ms_warren_walking_time_l872_87210


namespace practice_minutes_l872_87297

def month_total_days : ℕ := (2 * 6) + (2 * 7)

def piano_daily_minutes : ℕ := 25

def violin_daily_minutes := piano_daily_minutes * 3

def flute_daily_minutes := violin_daily_minutes / 2

theorem practice_minutes (piano_total : ℕ) (violin_total : ℕ) (flute_total : ℕ) :
  (26 * piano_daily_minutes = 650) ∧ 
  (20 * violin_daily_minutes = 1500) ∧ 
  (16 * flute_daily_minutes = 600) := by
  sorry

end practice_minutes_l872_87297


namespace triangle_inequality_l872_87278

theorem triangle_inequality (a b c : ℝ) (h : a^2 = b^2 + c^2) : 
  (b - c)^2 * (a^2 + 4 * b * c)^2 ≤ 2 * a^6 :=
by
  sorry

end triangle_inequality_l872_87278


namespace fraction_of_tomato_plants_in_second_garden_l872_87251

theorem fraction_of_tomato_plants_in_second_garden 
    (total_plants_first_garden : ℕ := 20)
    (percent_tomato_first_garden : ℚ := 10 / 100)
    (total_plants_second_garden : ℕ := 15)
    (percent_total_tomato_plants : ℚ := 20 / 100) :
    (15 : ℚ) * (1 / 3) = 5 :=
by
  sorry

end fraction_of_tomato_plants_in_second_garden_l872_87251


namespace scientific_notation_86560_l872_87289

theorem scientific_notation_86560 : ∃ a n, (86560 : ℝ) = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8.656 ∧ n = 4 :=
by {
  sorry
}

end scientific_notation_86560_l872_87289


namespace john_spends_6_dollars_l872_87267

-- Let treats_per_day, cost_per_treat, and days_in_month be defined by the conditions of the problem.
def treats_per_day : ℕ := 2
def cost_per_treat : ℝ := 0.1
def days_in_month : ℕ := 30

-- The total expenditure should be defined as the number of treats multiplied by their cost.
def total_number_of_treats := treats_per_day * days_in_month
def total_expenditure := total_number_of_treats * cost_per_treat

-- The statement to be proven: John spends $6 on the treats.
theorem john_spends_6_dollars :
  total_expenditure = 6 :=
sorry

end john_spends_6_dollars_l872_87267


namespace find_value_of_a_l872_87204

variable (a b : ℝ)

def varies_inversely (a : ℝ) (b_minus_one_sq : ℝ) : ℝ :=
  a * b_minus_one_sq

theorem find_value_of_a 
  (h₁ : ∀ b : ℝ, varies_inversely a ((b - 1) ^ 2) = 64)
  (h₂ : b = 5) : a = 4 :=
by sorry

end find_value_of_a_l872_87204


namespace diamonds_in_G_10_l872_87296

-- Define the sequence rule for diamonds in Gn
def diamonds_in_G (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

-- The main theorem to prove that the number of diamonds in G₁₀ is 218
theorem diamonds_in_G_10 : diamonds_in_G 10 = 218 := by
  sorry

end diamonds_in_G_10_l872_87296


namespace smallest_solution_l872_87264

noncomputable def equation (x : ℝ) : Prop :=
  (1 / (x - 1)) + (1 / (x - 5)) = 4 / (x - 4)

theorem smallest_solution : 
  ∃ x : ℝ, equation x ∧ x ≠ 1 ∧ x ≠ 5 ∧ x ≠ 4 ∧ x = (5 - Real.sqrt 33) / 2 :=
by
  sorry

end smallest_solution_l872_87264


namespace reciprocal_sum_fractions_l872_87282

theorem reciprocal_sum_fractions:
  let a := (3: ℚ) / 4
  let b := (5: ℚ) / 6
  let c := (1: ℚ) / 2
  (a + b + c)⁻¹ = 12 / 25 :=
by
  sorry

end reciprocal_sum_fractions_l872_87282


namespace tangents_and_fraction_l872_87224

theorem tangents_and_fraction
  (α β : ℝ)
  (tan_diff : Real.tan (α - β) = 2)
  (tan_beta : Real.tan β = 4) :
  (7 * Real.sin α - Real.cos α) / (7 * Real.sin α + Real.cos α) = 7 / 5 :=
sorry

end tangents_and_fraction_l872_87224


namespace distance_between_trees_l872_87235

theorem distance_between_trees (trees : ℕ) (total_length : ℝ) (n : trees = 26) (l : total_length = 500) :
  ∃ d : ℝ, d = total_length / (trees - 1) ∧ d = 20 :=
by
  sorry

end distance_between_trees_l872_87235


namespace rational_sum_zero_cube_nonzero_fifth_power_zero_l872_87285

theorem rational_sum_zero_cube_nonzero_fifth_power_zero
  (a b c : ℚ) 
  (h_sum : a + b + c = 0)
  (h_cube_nonzero : a^3 + b^3 + c^3 ≠ 0) 
  : a^5 + b^5 + c^5 = 0 :=
sorry

end rational_sum_zero_cube_nonzero_fifth_power_zero_l872_87285


namespace baking_powder_now_l872_87265

def baking_powder_yesterday : ℝ := 0.4
def baking_powder_used : ℝ := 0.1

theorem baking_powder_now : 
  baking_powder_yesterday - baking_powder_used = 0.3 :=
by
  sorry

end baking_powder_now_l872_87265


namespace buddy_met_boy_students_l872_87218

theorem buddy_met_boy_students (total_students : ℕ) (girl_students : ℕ) (boy_students : ℕ) (h1 : total_students = 123) (h2 : girl_students = 57) : boy_students = 66 :=
by
  sorry

end buddy_met_boy_students_l872_87218


namespace parabola_line_no_intersection_l872_87225

theorem parabola_line_no_intersection (x y : ℝ) (h : y^2 < 4 * x) :
  ¬ ∃ (x' y' : ℝ), y' = y ∧ y'^2 = 4 * x' ∧ 2 * x' = x + x :=
by sorry

end parabola_line_no_intersection_l872_87225


namespace hyperbola_asymptote_slopes_l872_87223

theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), 2 * (y^2 / 16) - 2 * (x^2 / 9) = 1 → (∃ m : ℝ, y = m * x ∨ y = -m * x) ∧ m = (Real.sqrt 80) / 3 :=
by
  sorry

end hyperbola_asymptote_slopes_l872_87223


namespace value_of_expression_l872_87299

theorem value_of_expression 
  (triangle square : ℝ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 :=
by
  sorry

end value_of_expression_l872_87299


namespace arithmetic_geometric_sequence_l872_87222

theorem arithmetic_geometric_sequence
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (S : ℕ → ℝ)
  (f : ℕ → ℝ)
  (h₁ : a 1 = 3)
  (h₂ : b 1 = 1)
  (h₃ : b 2 * S 2 = 64)
  (h₄ : b 3 * S 3 = 960)
  : (∀ n, a n = 2 * n + 1) ∧ (∀ n, b n = 8^(n - 1)) ∧ 
    (∀ n, f n = (a n - 1) / (S n + 100)) ∧ 
    (∃ n, f n = 1 / 11 ∧ n = 10) := 
sorry

end arithmetic_geometric_sequence_l872_87222


namespace num_integers_between_cubed_values_l872_87258

theorem num_integers_between_cubed_values : 
  let a : ℝ := 10.5
  let b : ℝ := 10.7
  let c1 := a^3
  let c2 := b^3
  let first_integer := Int.ceil c1
  let last_integer := Int.floor c2
  first_integer ≤ last_integer → 
  last_integer - first_integer + 1 = 67 :=
by
  sorry

end num_integers_between_cubed_values_l872_87258


namespace base_of_second_term_l872_87274

theorem base_of_second_term (h : ℕ) (a b c : ℕ) (H1 : h > 0) 
  (H2 : 225 ∣ h) (H3 : 216 ∣ h) 
  (H4 : h = (2^a) * (some_number^b) * (5^c)) 
  (H5 : a + b + c = 8) : some_number = 3 :=
by
  sorry

end base_of_second_term_l872_87274


namespace investment_duration_l872_87283

theorem investment_duration 
  (P SI R : ℕ) (T : ℕ) 
  (hP : P = 800) 
  (hSI : SI = 128) 
  (hR : R = 4) 
  (h : SI = P * R * T / 100) 
  : T = 4 :=
by 
  rw [hP, hSI, hR] at h
  sorry

end investment_duration_l872_87283


namespace reservoir_full_percentage_after_storm_l872_87260

theorem reservoir_full_percentage_after_storm 
  (original_contents water_added : ℤ) 
  (percentage_full_before_storm: ℚ) 
  (total_capacity new_contents : ℚ) 
  (H1 : original_contents = 220 * 10^9) 
  (H2 : water_added = 110 * 10^9) 
  (H3 : percentage_full_before_storm = 0.40)
  (H4 : total_capacity = original_contents / percentage_full_before_storm)
  (H5 : new_contents = original_contents + water_added) :
  (new_contents / total_capacity) = 0.60 := 
by 
  sorry

end reservoir_full_percentage_after_storm_l872_87260


namespace age_of_15th_person_l872_87226

variable (avg_age_20 : ℕ) (avg_age_5 : ℕ) (avg_age_9 : ℕ) (A : ℕ)
variable (num_20 : ℕ) (num_5 : ℕ) (num_9 : ℕ)

theorem age_of_15th_person (h1 : avg_age_20 = 15) (h2 : avg_age_5 = 14) (h3 : avg_age_9 = 16)
  (h4 : num_20 = 20) (h5 : num_5 = 5) (h6 : num_9 = 9) :
  (num_20 * avg_age_20) = (num_5 * avg_age_5) + (num_9 * avg_age_9) + A → A = 86 :=
by
  sorry

end age_of_15th_person_l872_87226


namespace part1_part2_l872_87212

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 
  Real.log x + 0.5 * m * x^2 - 2 

def perpendicular_slope_condition (m : ℝ) : Prop := 
  let k := (1 / 1 + m)
  k = -1 / 2

def inequality_condition (m : ℝ) : Prop := 
  ∀ x > 0, 
  Real.log x - 0.5 * m * x^2 + (1 - m) * x + 1 ≤ 0

theorem part1 : perpendicular_slope_condition (-3/2) :=
  sorry

theorem part2 : ∃ m : ℤ, m ≥ 2 ∧ inequality_condition m :=
  sorry

end part1_part2_l872_87212


namespace largest_s_for_angle_ratio_l872_87214

theorem largest_s_for_angle_ratio (r s : ℕ) (hr : r ≥ 3) (hs : s ≥ 3) (h_angle_ratio : (130 * (r - 2)) * s = (131 * (s - 2)) * r) :
  s ≤ 260 :=
by 
  sorry

end largest_s_for_angle_ratio_l872_87214


namespace eccentricity_of_ellipse_l872_87207

open Real

noncomputable def eccentricity_min (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) : ℝ :=
  if h : m = 2 then (sqrt 6)/3 else 0

theorem eccentricity_of_ellipse (m : ℝ) (h₁ : m > 0) (h₂ : m ≥ 2) :
    eccentricity_min m h₁ h₂ = (sqrt 6)/3 := by
  sorry

end eccentricity_of_ellipse_l872_87207


namespace min_sum_abc_l872_87201

theorem min_sum_abc (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1716) :
  a + b + c = 31 :=
sorry

end min_sum_abc_l872_87201


namespace find_m_for_given_slope_l872_87262

theorem find_m_for_given_slope (m : ℝ) :
  (∃ (P Q : ℝ × ℝ),
    P = (-2, m) ∧ Q = (m, 4) ∧
    (Q.2 - P.2) / (Q.1 - P.1) = 1) → m = 1 :=
by
  sorry

end find_m_for_given_slope_l872_87262


namespace conference_games_l872_87248

/-- 
Two divisions of 8 teams each, where each team plays 21 games within its division 
and 8 games against the teams of the other division. 
Prove total number of scheduled conference games is 232.
-/
theorem conference_games (div_teams : ℕ) (intra_div_games : ℕ) (inter_div_games : ℕ) (total_teams : ℕ) :
  div_teams = 8 →
  intra_div_games = 21 →
  inter_div_games = 8 →
  total_teams = 2 * div_teams →
  (total_teams * (intra_div_games + inter_div_games)) / 2 = 232 :=
by
  intros
  sorry


end conference_games_l872_87248


namespace units_digit_diff_l872_87256

theorem units_digit_diff (p : ℕ) (hp : p > 0) (even_p : p % 2 = 0) (units_p1_7 : (p + 1) % 10 = 7) : (p^3 % 10) = (p^2 % 10) :=
by
  sorry

end units_digit_diff_l872_87256


namespace alpha_beta_sum_equal_two_l872_87209

theorem alpha_beta_sum_equal_two (α β : ℝ) 
  (hα : α^3 - 3*α^2 + 5*α - 17 = 0) 
  (hβ : β^3 - 3*β^2 + 5*β + 11 = 0) : 
  α + β = 2 := 
sorry

end alpha_beta_sum_equal_two_l872_87209


namespace second_machine_equation_l872_87228

-- Let p1_rate and p2_rate be the rates of printing for machine 1 and 2 respectively.
-- Let x be the unknown time for the second machine to print 500 envelopes.

theorem second_machine_equation (x : ℝ) :
    (500 / 8) + (500 / x) = (500 / 2) :=
  sorry

end second_machine_equation_l872_87228


namespace cost_of_letter_is_0_37_l872_87239

-- Definitions based on the conditions
def total_cost : ℝ := 4.49
def package_cost : ℝ := 0.88
def num_letters : ℕ := 5
def num_packages : ℕ := 3
def letter_cost (L : ℝ) : ℝ := 5 * L
def package_total_cost : ℝ := num_packages * package_cost

-- Theorem that encapsulates the mathematical proof problem
theorem cost_of_letter_is_0_37 (L : ℝ) (h : letter_cost L + package_total_cost = total_cost) : L = 0.37 :=
by sorry

end cost_of_letter_is_0_37_l872_87239


namespace find_n_l872_87284

theorem find_n (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 10) (h3 : n % 11 = 99999 % 11) : n = 9 :=
sorry

end find_n_l872_87284


namespace half_guests_want_two_burgers_l872_87291

theorem half_guests_want_two_burgers 
  (total_guests : ℕ) (half_guests : ℕ)
  (time_per_side : ℕ) (time_per_burger : ℕ)
  (grill_capacity : ℕ) (total_time : ℕ)
  (guests_one_burger : ℕ) (total_burgers : ℕ) : 
  total_guests = 30 →
  time_per_side = 4 →
  time_per_burger = 8 →
  grill_capacity = 5 →
  total_time = 72 →
  guests_one_burger = 15 →
  total_burgers = 45 →
  half_guests * 2 = total_burgers - guests_one_burger →
  half_guests = 15 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end half_guests_want_two_burgers_l872_87291


namespace distance_walked_by_man_l872_87220

theorem distance_walked_by_man (x t : ℝ) (h1 : d = (x + 0.5) * (4 / 5) * t) (h2 : d = (x - 0.5) * (t + 2.5)) : d = 15 :=
by
  sorry

end distance_walked_by_man_l872_87220


namespace league_games_count_l872_87254

theorem league_games_count :
  let num_divisions := 2
  let teams_per_division := 9
  let intra_division_games (teams_per_div : ℕ) := (teams_per_div * (teams_per_div - 1) / 2) * 3
  let inter_division_games (teams_per_div : ℕ) (num_div : ℕ) := teams_per_div * teams_per_div * 2
  intra_division_games teams_per_division * num_divisions + inter_division_games teams_per_division num_divisions = 378 :=
by
  sorry

end league_games_count_l872_87254


namespace other_root_of_quadratic_l872_87292

theorem other_root_of_quadratic (m : ℝ) (x2 : ℝ) : (x^2 + m * x + 6 = 0) → (x + 2) * (x + x2) = 0 → x2 = -3 :=
by
  sorry

end other_root_of_quadratic_l872_87292


namespace parabola_directrix_eq_neg2_l872_87288

-- Definitions based on conditions
def ellipse_focus (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ x = 2 ∧ y = 0

def parabola_directrix (p x y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ ∃ x, x = -p / 2

theorem parabola_directrix_eq_neg2 (p : ℝ) (hp : p > 0) :
  (∀ (x y : ℝ), ellipse_focus 9 5 x y → parabola_directrix p x y) →
  (∃ x y : ℝ, parabola_directrix p x y → x = -2) :=
sorry

end parabola_directrix_eq_neg2_l872_87288


namespace maximum_rectangle_area_l872_87211

theorem maximum_rectangle_area (P : ℝ) (hP : P = 36) :
  ∃ (A : ℝ), A = (P / 4) * (P / 4) :=
by
  use 81
  sorry

end maximum_rectangle_area_l872_87211


namespace quadrilateral_is_parallelogram_l872_87238

theorem quadrilateral_is_parallelogram
  (A B C D : Type)
  (angle_DAB angle_ABC angle_BAD angle_DCB : ℝ)
  (h1 : angle_DAB = 135)
  (h2 : angle_ABC = 45)
  (h3 : angle_BAD = 45)
  (h4 : angle_DCB = 45) :
  (A B C D : Type) → Prop :=
by
  -- Definitions and conditions are given.
  sorry

end quadrilateral_is_parallelogram_l872_87238


namespace men_build_wall_l872_87215

theorem men_build_wall (k : ℕ) (h1 : 20 * 6 = k) : ∃ d : ℝ, (30 * d = k) ∧ d = 4.0 := by
  sorry

end men_build_wall_l872_87215


namespace find_y_l872_87272

theorem find_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hrem : x % y = 11.52) (hdiv : x / y = 96.12) : y = 96 := 
sorry

end find_y_l872_87272


namespace axis_of_symmetry_l872_87241

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (4 - x)) : ∀ y, f 2 = y ↔ f 2 = y := 
by
  sorry

end axis_of_symmetry_l872_87241


namespace number_of_rectangles_with_one_gray_cell_l872_87219

theorem number_of_rectangles_with_one_gray_cell 
    (num_gray_cells : Nat) 
    (num_blue_cells : Nat) 
    (num_red_cells : Nat) 
    (blue_rectangles_per_cell : Nat) 
    (red_rectangles_per_cell : Nat)
    (total_gray_cells_calc : num_gray_cells = 2 * 20)
    (num_gray_cells_definition : num_gray_cells = num_blue_cells + num_red_cells)
    (blue_rect_cond : blue_rectangles_per_cell = 4)
    (red_rect_cond : red_rectangles_per_cell = 8)
    (num_blue_cells_calc : num_blue_cells = 36)
    (num_red_cells_calc : num_red_cells = 4)
  : num_blue_cells * blue_rectangles_per_cell + num_red_cells * red_rectangles_per_cell = 176 := 
  by
  sorry

end number_of_rectangles_with_one_gray_cell_l872_87219


namespace students_per_group_l872_87295

theorem students_per_group (total_students not_picked groups : ℕ) 
    (h1 : total_students = 64) 
    (h2 : not_picked = 36) 
    (h3 : groups = 4) : (total_students - not_picked) / groups = 7 :=
by
  sorry

end students_per_group_l872_87295


namespace inequality_solution_l872_87205

-- Define the variable x as a real number
variable (x : ℝ)

-- Define the given condition that x is positive
def is_positive (x : ℝ) := x > 0

-- Define the condition that x satisfies the inequality sqrt(9x) < 3x^2
def satisfies_inequality (x : ℝ) := Real.sqrt (9 * x) < 3 * x^2

-- The statement we need to prove
theorem inequality_solution (x : ℝ) (h : is_positive x) : satisfies_inequality x ↔ x > 1 :=
sorry

end inequality_solution_l872_87205


namespace find_a_l872_87233

noncomputable def f (a x : ℝ) := (x - 1)^2 + a * x + Real.cos x

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → 
  a = 2 :=
by
  sorry

end find_a_l872_87233


namespace quadratic_at_most_two_roots_l872_87268

theorem quadratic_at_most_two_roots (a b c x1 x2 x3 : ℝ) (ha : a ≠ 0) 
(h1 : a * x1^2 + b * x1 + c = 0)
(h2 : a * x2^2 + b * x2 + c = 0)
(h3 : a * x3^2 + b * x3 + c = 0)
(h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : 
false :=
sorry

end quadratic_at_most_two_roots_l872_87268


namespace number_of_children_l872_87221

namespace CurtisFamily

variables {m x : ℕ} {xy : ℕ}

/-- Given conditions for Curtis family average ages. -/
def family_average_age (m x xy : ℕ) : Prop := (m + 50 + xy) / (2 + x) = 25

def mother_children_average_age (m x xy : ℕ) : Prop := (m + xy) / (1 + x) = 20

/-- The number of children in Curtis family is 4, given the average age conditions. -/
theorem number_of_children (m xy : ℕ) (h1 : family_average_age m 4 xy) (h2 : mother_children_average_age m 4 xy) : x = 4 :=
by
  sorry

end CurtisFamily

end number_of_children_l872_87221


namespace num_positive_int_values_l872_87290

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l872_87290


namespace members_of_groups_l872_87247

variable {x y : ℕ}

theorem members_of_groups (h1 : x = y + 10) (h2 : x - 1 = 2 * (y + 1)) :
  x = 17 ∧ y = 7 :=
by
  sorry

end members_of_groups_l872_87247


namespace expand_polynomial_l872_87234

theorem expand_polynomial (N : ℕ) :
  (∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a + b + c + d + 1)^N = 715) ↔ N = 13 := by
  sorry -- Replace with the actual proof when ready

end expand_polynomial_l872_87234


namespace number_of_whole_numbers_between_sqrts_l872_87249

noncomputable def count_whole_numbers_between_sqrts : ℕ :=
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let min_int := Int.ceil lower_bound
  let max_int := Int.floor upper_bound
  Int.natAbs (max_int - min_int + 1)

theorem number_of_whole_numbers_between_sqrts :
  count_whole_numbers_between_sqrts = 7 :=
by
  sorry

end number_of_whole_numbers_between_sqrts_l872_87249


namespace volume_ratio_l872_87244

theorem volume_ratio (V1 V2 M1 M2 : ℝ)
  (h1 : M1 / (V1 - M1) = 1 / 2)
  (h2 : M2 / (V2 - M2) = 3 / 2)
  (h3 : (M1 + M2) / (V1 - M1 + V2 - M2) = 1) :
  V1 / V2 = 9 / 5 :=
by
  sorry

end volume_ratio_l872_87244


namespace difference_between_numbers_l872_87255

theorem difference_between_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 190) : x - y = 19 :=
by sorry

end difference_between_numbers_l872_87255


namespace boys_in_fifth_grade_l872_87279

theorem boys_in_fifth_grade (T S : ℕ) (percent_boys_soccer : ℝ) (girls_not_playing_soccer : ℕ) 
    (hT : T = 420) (hS : S = 250) (h_percent : percent_boys_soccer = 0.86) 
    (h_girls_not_playing_soccer : girls_not_playing_soccer = 65) : 
    ∃ B : ℕ, B = 320 :=
by
  -- We don't need to provide the proof details here
  sorry

end boys_in_fifth_grade_l872_87279


namespace find_x_angle_l872_87208

-- Define the conditions
def angles_around_point (a b c d : ℝ) : Prop :=
  a + b + c + d = 360

-- The given problem implies:
-- 120 + x + x + 2x = 360
-- We need to find x such that the above equation holds.
theorem find_x_angle :
  angles_around_point 120 x x (2 * x) → x = 60 :=
by
  sorry

end find_x_angle_l872_87208


namespace lara_yesterday_more_than_sarah_l872_87277

variable (yesterdaySarah todaySarah todayLara : ℕ)
variable (cansDifference : ℕ)

axiom yesterdaySarah_eq : yesterdaySarah = 50
axiom todaySarah_eq : todaySarah = 40
axiom todayLara_eq : todayLara = 70
axiom cansDifference_eq : cansDifference = 20

theorem lara_yesterday_more_than_sarah :
  let totalCansYesterday := yesterdaySarah + todaySarah + cansDifference
  let laraYesterday := totalCansYesterday - yesterdaySarah
  laraYesterday - yesterdaySarah = 30 :=
by
  sorry

end lara_yesterday_more_than_sarah_l872_87277


namespace solve_inequality_l872_87202

namespace InequalityProof

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem solve_inequality (x : ℝ) : cube_root x + 3 / (cube_root x + 4) ≤ 0 ↔ x ∈ Set.Icc (-27 : ℝ) (-1 : ℝ) :=
by
  have y_eq := cube_root x
  sorry

end InequalityProof

end solve_inequality_l872_87202


namespace r_and_s_earns_per_day_l872_87206

variable (P Q R S : Real)

-- Conditions as given in the problem
axiom cond1 : P + Q + R + S = 2380 / 9
axiom cond2 : P + R = 600 / 5
axiom cond3 : Q + S = 800 / 6
axiom cond4 : Q + R = 910 / 7
axiom cond5 : P = 150 / 3

theorem r_and_s_earns_per_day : R + S = 143.33 := by
  sorry

end r_and_s_earns_per_day_l872_87206


namespace rectangle_perimeter_l872_87276

-- Definitions and assumptions
variables (outer_square_area inner_square_area : ℝ) (rectangles_identical : Prop)

-- Given conditions
def outer_square_area_condition : Prop := outer_square_area = 9
def inner_square_area_condition : Prop := inner_square_area = 1
def rectangles_identical_condition : Prop := rectangles_identical

-- The main theorem to prove
theorem rectangle_perimeter (h_outer : outer_square_area_condition outer_square_area)
                            (h_inner : inner_square_area_condition inner_square_area)
                            (h_rectangles : rectangles_identical_condition rectangles_identical) :
  ∃ perimeter : ℝ, perimeter = 6 :=
by
  sorry

end rectangle_perimeter_l872_87276


namespace arbitrary_large_sum_of_digits_l872_87231

noncomputable def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem arbitrary_large_sum_of_digits (a : Nat) (h1 : 2 ≤ a) (h2 : ¬ (2 ∣ a)) (h3 : ¬ (5 ∣ a)) :
  ∃ m : Nat, sum_of_digits (a^m) > m :=
by
  sorry

end arbitrary_large_sum_of_digits_l872_87231


namespace line_circle_no_intersection_l872_87281

theorem line_circle_no_intersection : 
  ¬ ∃ (x y : ℝ), 3 * x + 4 * y = 12 ∧ x^2 + y^2 = 4 := by
  sorry

end line_circle_no_intersection_l872_87281


namespace sum_of_intercepts_modulo_13_l872_87203

theorem sum_of_intercepts_modulo_13 :
  ∃ (x0 y0 : ℤ), 0 ≤ x0 ∧ x0 < 13 ∧ 0 ≤ y0 ∧ y0 < 13 ∧
    (4 * x0 ≡ 1 [ZMOD 13]) ∧ (3 * y0 ≡ 12 [ZMOD 13]) ∧ (x0 + y0 = 14) := 
sorry

end sum_of_intercepts_modulo_13_l872_87203


namespace combined_age_of_siblings_l872_87252

-- We are given Aaron's age
def aaronAge : ℕ := 15

-- Henry's sister's age is three times Aaron's age
def henrysSisterAge : ℕ := 3 * aaronAge

-- Henry's age is four times his sister's age
def henryAge : ℕ := 4 * henrysSisterAge

-- The combined age of the siblings
def combinedAge : ℕ := aaronAge + henrysSisterAge + henryAge

theorem combined_age_of_siblings : combinedAge = 240 := by
  sorry

end combined_age_of_siblings_l872_87252
