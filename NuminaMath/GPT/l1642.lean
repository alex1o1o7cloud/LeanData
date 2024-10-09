import Mathlib

namespace Jack_goal_l1642_164285

-- Define the amounts Jack made from brownies and lemon squares
def brownies (n : ℕ) (price : ℕ) : ℕ := n * price
def lemonSquares (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the amount Jack needs to make from cookies
def cookies (n : ℕ) (price : ℕ) : ℕ := n * price

-- Define the total goal for Jack
def totalGoal (browniesCount : ℕ) (browniesPrice : ℕ) 
              (lemonSquaresCount : ℕ) (lemonSquaresPrice : ℕ) 
              (cookiesCount : ℕ) (cookiesPrice: ℕ) : ℕ :=
  brownies browniesCount browniesPrice + lemonSquares lemonSquaresCount lemonSquaresPrice + cookies cookiesCount cookiesPrice

theorem Jack_goal : totalGoal 4 3 5 2 7 4 = 50 :=
by
  -- Adding up the different components of the total earnings
  let totalFromBrownies := brownies 4 3
  let totalFromLemonSquares := lemonSquares 5 2
  let totalFromCookies := cookies 7 4
  -- Summing up the amounts
  have step1 : totalFromBrownies = 12 := rfl
  have step2 : totalFromLemonSquares = 10 := rfl
  have step3 : totalFromCookies = 28 := rfl
  have step4 : totalGoal 4 3 5 2 7 4 = totalFromBrownies + totalFromLemonSquares + totalFromCookies := rfl
  have step5 : totalFromBrownies + totalFromLemonSquares + totalFromCookies = 12 + 10 + 28 := by rw [step1, step2, step3]
  have step6 : 12 + 10 + 28 = 50 := by norm_num
  exact step4 ▸ (step5 ▸ step6)

end Jack_goal_l1642_164285


namespace equation1_solution_equation2_solution_equation3_solution_l1642_164298

theorem equation1_solution :
  ∀ x : ℝ, x^2 + 4 * x = 0 ↔ x = 0 ∨ x = -4 :=
by
  sorry

theorem equation2_solution :
  ∀ x : ℝ, 2 * (x - 1) + x * (x - 1) = 0 ↔ x = 1 ∨ x = -2 :=
by
  sorry

theorem equation3_solution :
  ∀ x : ℝ, 3 * x^2 - 2 * x - 4 = 0 ↔ x = (1 + Real.sqrt 13) / 3 ∨ x = (1 - Real.sqrt 13) / 3 :=
by
  sorry

end equation1_solution_equation2_solution_equation3_solution_l1642_164298


namespace radius_of_circle_l1642_164261

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = 2 * Real.pi * r^2) : r = 3 :=
by
  sorry

end radius_of_circle_l1642_164261


namespace train_crosses_pole_in_15_seconds_l1642_164203

theorem train_crosses_pole_in_15_seconds
    (train_speed : ℝ) (train_length_meters : ℝ) (time_seconds : ℝ) : 
    train_speed = 300 →
    train_length_meters = 1250 →
    time_seconds = 15 :=
by
  sorry

end train_crosses_pole_in_15_seconds_l1642_164203


namespace calculate_f_5_5_l1642_164209

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic_condition (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f (x + 2) = -1 / f x
axiom defined_segment (x : ℝ) (h₂ : 2 ≤ x ∧ x ≤ 3) : f x = x

theorem calculate_f_5_5 : f 5.5 = 2.5 := sorry

end calculate_f_5_5_l1642_164209


namespace orthogonal_trajectory_eqn_l1642_164229

theorem orthogonal_trajectory_eqn (a C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 2 * a * x) → 
  (∃ C : ℝ, ∀ x y : ℝ, x^2 + y^2 = C * y) :=
sorry

end orthogonal_trajectory_eqn_l1642_164229


namespace hyperbola_a_solution_l1642_164234

noncomputable def hyperbola_a_value (a : ℝ) : Prop :=
  (a > 0) ∧ (∀ x y : ℝ, (x^2 / a^2) - (y^2 / 2) = 1) ∧ (∃ e : ℝ, e = 2)

theorem hyperbola_a_solution : ∃ a : ℝ, hyperbola_a_value a ∧ a = (Real.sqrt 6) / 3 :=
  by
    sorry

end hyperbola_a_solution_l1642_164234


namespace altitude_triangle_eq_2w_l1642_164293

theorem altitude_triangle_eq_2w (l w h : ℕ) (h₀ : w ≠ 0) (h₁ : l ≠ 0)
    (h_area_rect : l * w = (1 / 2) * l * h) : h = 2 * w :=
by
  -- Consider h₀ (w is not zero) and h₁ (l is not zero)
  -- We need to prove h = 2w given l * w = (1 / 2) * l * h
  sorry

end altitude_triangle_eq_2w_l1642_164293


namespace surface_area_increase_l1642_164290

noncomputable def percent_increase_surface_area (s p : ℝ) : ℝ :=
  let new_edge_length := s * (1 + p / 100)
  let new_surface_area := 6 * (new_edge_length)^2
  let original_surface_area := 6 * s^2
  let percent_increase := (new_surface_area / original_surface_area - 1) * 100
  percent_increase

theorem surface_area_increase (s p : ℝ) :
  percent_increase_surface_area s p = 2 * p + p^2 / 100 :=
by
  sorry

end surface_area_increase_l1642_164290


namespace debby_vacation_pictures_l1642_164230

theorem debby_vacation_pictures :
  let zoo_initial := 150
  let aquarium_initial := 210
  let museum_initial := 90
  let amusement_park_initial := 120
  let zoo_deleted := (25 * zoo_initial) / 100  -- 25% of zoo pictures deleted
  let aquarium_deleted := (15 * aquarium_initial) / 100  -- 15% of aquarium pictures deleted
  let museum_added := 30  -- 30 additional pictures at the museum
  let amusement_park_deleted := 20  -- 20 pictures deleted at the amusement park
  let zoo_kept := zoo_initial - zoo_deleted
  let aquarium_kept := aquarium_initial - aquarium_deleted
  let museum_kept := museum_initial + museum_added
  let amusement_park_kept := amusement_park_initial - amusement_park_deleted
  let total_pictures := zoo_kept + aquarium_kept + museum_kept + amusement_park_kept
  total_pictures = 512 :=
by
  sorry

end debby_vacation_pictures_l1642_164230


namespace gcd_divisibility_and_scaling_l1642_164263

theorem gcd_divisibility_and_scaling (a b n : ℕ) (c : ℕ) (h₁ : a ≠ 0) (h₂ : c > 0) (d : ℕ := Nat.gcd a b) :
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧ Nat.gcd (a * c) (b * c) = c * d :=
by 
  sorry

end gcd_divisibility_and_scaling_l1642_164263


namespace div_poly_l1642_164297

theorem div_poly (m n p : ℕ) : 
  (X^2 + X + 1) ∣ (X^(3*m) + X^(3*n + 1) + X^(3*p + 2)) := 
sorry

end div_poly_l1642_164297


namespace basketball_lineups_l1642_164204

noncomputable def num_starting_lineups (total_players : ℕ) (fixed_players : ℕ) (chosen_players : ℕ) : ℕ :=
  Nat.choose (total_players - fixed_players) (chosen_players - fixed_players)

theorem basketball_lineups :
  num_starting_lineups 15 2 6 = 715 := by
  sorry

end basketball_lineups_l1642_164204


namespace count_valid_tuples_l1642_164276

variable {b_0 b_1 b_2 b_3 : ℕ}

theorem count_valid_tuples : 
  (∃ b_0 b_1 b_2 b_3 : ℕ, 
    0 ≤ b_0 ∧ b_0 ≤ 99 ∧ 
    0 ≤ b_1 ∧ b_1 ≤ 99 ∧ 
    0 ≤ b_2 ∧ b_2 ≤ 99 ∧ 
    0 ≤ b_3 ∧ b_3 ≤ 99 ∧ 
    5040 = b_3 * 10^3 + b_2 * 10^2 + b_1 * 10 + b_0) ∧ 
    ∃ (M : ℕ), 
    M = 504 :=
sorry

end count_valid_tuples_l1642_164276


namespace connie_remaining_marbles_l1642_164259

def initial_marbles : ℕ := 73
def marbles_given : ℕ := 70

theorem connie_remaining_marbles : initial_marbles - marbles_given = 3 := by
  sorry

end connie_remaining_marbles_l1642_164259


namespace solution_set_inequality_l1642_164237

noncomputable def f : ℝ → ℝ := sorry
noncomputable def derivative_f : ℝ → ℝ := sorry -- f' is the derivative of f

-- Conditions
axiom f_domain {x : ℝ} (h1 : 0 < x) : f x ≠ 0
axiom derivative_condition {x : ℝ} (h1 : 0 < x) : f x + x * derivative_f x > 0
axiom initial_value : f 1 = 2

-- Proof that the solution set of the inequality f(x) < 2/x is (0, 1)
theorem solution_set_inequality : ∀ x : ℝ, 0 < x ∧ x < 1 → f x < 2 / x := sorry

end solution_set_inequality_l1642_164237


namespace distinct_real_numbers_proof_l1642_164251

variables {a b c : ℝ}

theorem distinct_real_numbers_proof (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
  (h₄ : (a / (b - c) + b / (c - a) + c / (a - b)) = -1) :
  (a^3 / (b - c)^2) + (b^3 / (c - a)^2) + (c^3 / (a - b)^2) = 0 :=
sorry

end distinct_real_numbers_proof_l1642_164251


namespace option_C_correct_l1642_164292

theorem option_C_correct (a b : ℝ) (h : a + b = 1) : a^2 + b^2 ≥ 1 / 2 :=
sorry

end option_C_correct_l1642_164292


namespace range_of_k_l1642_164273

def tensor (a b : ℝ) : ℝ := a * b + a + b^2

theorem range_of_k (k : ℝ) : (∀ x : ℝ, tensor k x > 0) ↔ (0 < k ∧ k < 4) :=
by
  sorry

end range_of_k_l1642_164273


namespace arithmetic_sequence_sum_l1642_164248

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (d : ℕ) (h1 : a 1 + a 13 = 10) 
  (h2 : ∀ n m : ℕ, a (n + 1) = a n + d) : a 3 + a 5 + a 7 + a 9 + a 11 = 25 :=
  sorry

end arithmetic_sequence_sum_l1642_164248


namespace trigonometric_identity_l1642_164226

open Real

theorem trigonometric_identity (α β : ℝ) (h : 2 * cos (2 * α + β) - 3 * cos β = 0) :
  tan α * tan (α + β) = -1 / 5 := 
by {
  sorry
}

end trigonometric_identity_l1642_164226


namespace volume_third_bottle_is_250_milliliters_l1642_164228

-- Define the volumes of the bottles in milliliters
def volume_first_bottle : ℕ := 2 * 1000                        -- 2000 milliliters
def volume_second_bottle : ℕ := 750                            -- 750 milliliters
def total_volume : ℕ := 3 * 1000                               -- 3000 milliliters
def volume_third_bottle : ℕ := total_volume - (volume_first_bottle + volume_second_bottle)

-- The theorem stating the volume of the third bottle
theorem volume_third_bottle_is_250_milliliters :
  volume_third_bottle = 250 :=
by
  sorry

end volume_third_bottle_is_250_milliliters_l1642_164228


namespace weight_of_mixture_is_correct_l1642_164271

noncomputable def weight_mixture_kg (weight_per_liter_a weight_per_liter_b ratio_a ratio_b total_volume_liters : ℕ) : ℝ :=
  let volume_a := (ratio_a * total_volume_liters) / (ratio_a + ratio_b)
  let volume_b := (ratio_b * total_volume_liters) / (ratio_a + ratio_b)
  let weight_a := (volume_a * weight_per_liter_a) 
  let weight_b := (volume_b * weight_per_liter_b) 
  (weight_a + weight_b) / 1000

theorem weight_of_mixture_is_correct :
  weight_mixture_kg 900 700 3 2 4 = 3.280 := 
sorry

end weight_of_mixture_is_correct_l1642_164271


namespace probability_check_l1642_164240

def total_students : ℕ := 12

def total_clubs : ℕ := 3

def equiprobable_clubs := ∀ s : Fin total_students, ∃ c : Fin total_clubs, true

noncomputable def probability_diff_students : ℝ := 1 - (34650 / (total_clubs ^ total_students))

theorem probability_check :
  equiprobable_clubs →
  probability_diff_students = 0.935 := 
by
  intros
  sorry

end probability_check_l1642_164240


namespace find_uv_l1642_164255

def mat_eqn (u v : ℝ) : Prop :=
  (3 + 8 * u = -3 * v) ∧ (-1 - 6 * u = 1 + 4 * v)

theorem find_uv : ∃ (u v : ℝ), mat_eqn u v ∧ u = -6/7 ∧ v = 5/7 := 
by
  sorry

end find_uv_l1642_164255


namespace woman_work_rate_l1642_164208

theorem woman_work_rate :
  let M := 1/6
  let B := 1/9
  let combined_rate := 1/3
  ∃ W : ℚ, M + B + W = combined_rate ∧ 1 / W = 18 := 
by
  sorry

end woman_work_rate_l1642_164208


namespace number_of_boys_l1642_164246

theorem number_of_boys (B G : ℕ) 
    (h1 : B + G = 345) 
    (h2 : G = B + 69) : B = 138 :=
by
  sorry

end number_of_boys_l1642_164246


namespace two_A_minus_B_l1642_164265

theorem two_A_minus_B (A B : ℝ) 
  (h1 : Real.tan (A - B - Real.pi) = 1 / 2) 
  (h2 : Real.tan (3 * Real.pi - B) = 1 / 7) : 
  2 * A - B = -3 * Real.pi / 4 :=
sorry

end two_A_minus_B_l1642_164265


namespace count_congruent_to_4_mod_7_l1642_164223

theorem count_congruent_to_4_mod_7 : 
  ∃ (n : ℕ), 
  n = 71 ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 70 → ∃ m : ℕ, m = 4 + 7 * k ∧ m < 500 := 
by
  sorry

end count_congruent_to_4_mod_7_l1642_164223


namespace max_min_value_of_product_l1642_164266

theorem max_min_value_of_product (x y : ℝ) (h : x ^ 2 + y ^ 2 = 1) :
  (1 + x * y) * (1 - x * y) ≤ 1 ∧ (1 + x * y) * (1 - x * y) ≥ 3 / 4 :=
by sorry

end max_min_value_of_product_l1642_164266


namespace frank_bakes_for_5_days_l1642_164257

variable (d : ℕ) -- The number of days Frank bakes cookies

def cookies_baked_per_day : ℕ := 2 * 12
def cookies_eaten_per_day : ℕ := 1

-- Total cookies baked over d days minus the cookies Frank eats each day
def cookies_remaining_before_ted (d : ℕ) : ℕ :=
  d * (cookies_baked_per_day - cookies_eaten_per_day)

-- Ted eats 4 cookies on the last day, so we add that back to get total before Ted ate
def total_cookies_before_ted (d : ℕ) : ℕ :=
  cookies_remaining_before_ted d + 4

-- After Ted's visit, there are 134 cookies left
axiom ted_leaves_134_cookies : total_cookies_before_ted d = 138

-- Prove that Frank bakes cookies for 5 days
theorem frank_bakes_for_5_days : d = 5 := by
  sorry

end frank_bakes_for_5_days_l1642_164257


namespace license_plates_count_l1642_164258

theorem license_plates_count :
  (20 * 6 * 20 * 10 * 26 = 624000) :=
by
  sorry

end license_plates_count_l1642_164258


namespace uniquePlantsTotal_l1642_164274

-- Define the number of plants in each bed
def numPlantsInA : ℕ := 600
def numPlantsInB : ℕ := 500
def numPlantsInC : ℕ := 400

-- Define the number of shared plants between beds
def sharedPlantsAB : ℕ := 60
def sharedPlantsAC : ℕ := 120
def sharedPlantsBC : ℕ := 80
def sharedPlantsABC : ℕ := 30

-- Prove that the total number of unique plants in the garden is 1270
theorem uniquePlantsTotal : 
  numPlantsInA + numPlantsInB + numPlantsInC 
  - sharedPlantsAB - sharedPlantsAC - sharedPlantsBC 
  + sharedPlantsABC = 1270 := 
by sorry

end uniquePlantsTotal_l1642_164274


namespace neg_one_third_squared_l1642_164225

theorem neg_one_third_squared :
  (-(1/3))^2 = 1/9 :=
sorry

end neg_one_third_squared_l1642_164225


namespace gcf_180_240_300_l1642_164260

theorem gcf_180_240_300 : Nat.gcd (Nat.gcd 180 240) 300 = 60 := sorry

end gcf_180_240_300_l1642_164260


namespace assistant_professors_charts_l1642_164241

theorem assistant_professors_charts (A B C : ℕ) (h1 : 2 * A + B = 10) (h2 : A + B * C = 11) (h3 : A + B = 7) : C = 2 :=
by
  sorry

end assistant_professors_charts_l1642_164241


namespace compare_squares_l1642_164279

theorem compare_squares (a b : ℝ) : a^2 + b^2 ≥ ab + a + b - 1 :=
by
  sorry

end compare_squares_l1642_164279


namespace total_weight_of_compound_l1642_164275

variable (molecular_weight : ℕ) (moles : ℕ)

theorem total_weight_of_compound (h1 : molecular_weight = 72) (h2 : moles = 4) :
  moles * molecular_weight = 288 :=
by
  sorry

end total_weight_of_compound_l1642_164275


namespace distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l1642_164220

-- Definitions based on conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def odd_digits : Finset ℕ := {1, 3, 5}

-- Problem 1: Number of distinct three-digit numbers
theorem distinct_three_digit_numbers : (digits.erase 0).card * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 100 := by
  sorry

-- Problem 2: Number of distinct three-digit odd numbers
theorem distinct_three_digit_odd_numbers : (odd_digits.card) * (digits.erase 0).card.pred * (digits.erase 0).card.pred.pred = 48 := by
  sorry

end distinct_three_digit_numbers_distinct_three_digit_odd_numbers_l1642_164220


namespace root_equation_val_l1642_164242

theorem root_equation_val (a : ℝ) (h : a^2 - 2 * a - 5 = 0) : 2 * a^2 - 4 * a = 10 :=
by 
  sorry

end root_equation_val_l1642_164242


namespace ball_rebound_percentage_l1642_164254

theorem ball_rebound_percentage (P : ℝ) 
  (h₁ : 100 + 2 * 100 * P + 2 * 100 * P^2 = 250) : P = 0.5 := 
by 
  sorry

end ball_rebound_percentage_l1642_164254


namespace no_integer_solution_l1642_164206

open Polynomial

theorem no_integer_solution (P : Polynomial ℤ) (a b c d : ℤ)
  (h₁ : P.eval a = 2016) (h₂ : P.eval b = 2016) (h₃ : P.eval c = 2016) 
  (h₄ : P.eval d = 2016) (dist : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : 
  ¬ ∃ x : ℤ, P.eval x = 2019 :=
sorry

end no_integer_solution_l1642_164206


namespace points_needed_for_office_l1642_164214

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25

def jerry_interruptions : ℕ := 2
def jerry_insults : ℕ := 4
def jerry_throwings : ℕ := 2

def jerry_total_points (interrupt_points insult_points throw_points : ℕ) 
                       (interruptions insults throwings : ℕ) : ℕ :=
  (interrupt_points * interruptions) +
  (insult_points * insults) +
  (throw_points * throwings)

theorem points_needed_for_office : 
  jerry_total_points points_for_interrupting points_for_insulting points_for_throwing 
                     (jerry_interruptions) 
                     (jerry_insults) 
                     (jerry_throwings) = 100 := 
  sorry

end points_needed_for_office_l1642_164214


namespace isosceles_right_triangle_hypotenuse_l1642_164215

theorem isosceles_right_triangle_hypotenuse (a : ℝ) (h : ℝ) (hyp : a = 30 ∧ h^2 = a^2 + a^2) : h = 30 * Real.sqrt 2 :=
sorry

end isosceles_right_triangle_hypotenuse_l1642_164215


namespace find_a_perpendicular_lines_l1642_164224

variable (a : ℝ)

theorem find_a_perpendicular_lines :
  (∃ a : ℝ, ∀ x y : ℝ, (a * x - y + 2 * a = 0) ∧ ((2 * a - 1) * x + a * y + a = 0) → a = 0 ∨ a = 1) := 
sorry

end find_a_perpendicular_lines_l1642_164224


namespace calculate_expression_l1642_164278

noncomputable def expr : ℚ := (5 - 2 * (3 - 6 : ℚ)⁻¹ ^ 2)⁻¹

theorem calculate_expression :
  expr = (9 / 43 : ℚ) := by
  sorry

end calculate_expression_l1642_164278


namespace scientific_notation_of_570_million_l1642_164211

theorem scientific_notation_of_570_million :
  570000000 = 5.7 * 10^8 := sorry

end scientific_notation_of_570_million_l1642_164211


namespace set_intersection_l1642_164205

open Set

def U : Set ℤ := univ
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {-1, 1}
def C_U_B : Set ℤ := U \ B

theorem set_intersection :
  A ∩ C_U_B = {2} := 
by
  sorry

end set_intersection_l1642_164205


namespace prism_ratio_l1642_164222

theorem prism_ratio (a b c d : ℝ) (h_d : d = 60) (h_c : c = 104) (h_b : b = 78 * Real.pi) (h_a : a = (4 * Real.pi) / 3) :
  b * c / (a * d) = 8112 / 240 := 
by 
  sorry

end prism_ratio_l1642_164222


namespace range_of_a_l1642_164207

open Real 

noncomputable def trigonometric_inequality (θ a : ℝ) : Prop :=
  sin (2 * θ) - (2 * sqrt 2 + sqrt 2 * a) * sin (θ + π / 4) - 2 * sqrt 2 / cos (θ - π / 4) > -3 - 2 * a

theorem range_of_a (a : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π / 2 → trigonometric_inequality θ a) ↔ (a > 3) :=
sorry

end range_of_a_l1642_164207


namespace range_of_x_l1642_164249

theorem range_of_x (x y : ℝ) (h : x - 6 * Real.sqrt y - 4 * Real.sqrt (x - y) + 12 = 0) : 
  12 ≤ x := 
sorry

end range_of_x_l1642_164249


namespace nat_divisibility_l1642_164291

theorem nat_divisibility {n : ℕ} : (n + 1 ∣ n^2 + 1) ↔ (n = 0 ∨ n = 1) := 
sorry

end nat_divisibility_l1642_164291


namespace curler_ratio_l1642_164280

theorem curler_ratio
  (total_curlers : ℕ)
  (pink_curlers : ℕ)
  (blue_curlers : ℕ)
  (green_curlers : ℕ)
  (h1 : total_curlers = 16)
  (h2 : blue_curlers = 2 * pink_curlers)
  (h3 : green_curlers = 4) :
  pink_curlers / total_curlers = 1 / 4 := by
  sorry

end curler_ratio_l1642_164280


namespace trip_total_charge_l1642_164253

noncomputable def initial_fee : ℝ := 2.25
noncomputable def additional_charge_per_increment : ℝ := 0.25
noncomputable def increment_length : ℝ := 2 / 5
noncomputable def trip_length : ℝ := 3.6

theorem trip_total_charge :
  initial_fee + (trip_length / increment_length) * additional_charge_per_increment = 4.50 :=
by
  sorry

end trip_total_charge_l1642_164253


namespace negation_exists_lt_zero_l1642_164232

variable {f : ℝ → ℝ}

theorem negation_exists_lt_zero :
  ¬ (∃ x : ℝ, f x < 0) → ∀ x : ℝ, 0 ≤ f x := by
  sorry

end negation_exists_lt_zero_l1642_164232


namespace positive_integer_solutions_x_plus_2y_eq_5_l1642_164238

theorem positive_integer_solutions_x_plus_2y_eq_5 :
  ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ (x + 2 * y = 5) ∧ ((x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 1)) :=
by
  sorry

end positive_integer_solutions_x_plus_2y_eq_5_l1642_164238


namespace factorization_problem_l1642_164270

theorem factorization_problem (a b c x : ℝ) :
  ¬(2 * a^2 - b^2 = (a + b) * (a - b) + a^2) ∧
  ¬(2 * a * (b + c) = 2 * a * b + 2 * a * c) ∧
  (x^3 - 2 * x^2 + x = x * (x - 1)^2) ∧
  ¬ (x^2 + x = x^2 * (1 + 1 / x)) :=
by
  sorry

end factorization_problem_l1642_164270


namespace sphere_volume_l1642_164218

theorem sphere_volume (S : ℝ) (r : ℝ) (V : ℝ) (h₁ : S = 256 * Real.pi) (h₂ : S = 4 * Real.pi * r^2) : V = 2048 / 3 * Real.pi :=
by
  sorry

end sphere_volume_l1642_164218


namespace reconstruct_points_l1642_164227

noncomputable def symmetric (x y : ℝ) := 2 * y - x

theorem reconstruct_points (A' B' C' D' B C D : ℝ) :
  (∃ (A B C D : ℝ),
     B = (A + A') / 2 ∧  -- B is the midpoint of line segment AA'
     C = (B + B') / 2 ∧  -- C is the midpoint of line segment BB'
     D = (C + C') / 2 ∧  -- D is the midpoint of line segment CC'
     A = (D + D') / 2)   -- A is the midpoint of line segment DD'
  ↔ (∃ (A : ℝ), A = symmetric D D') → True := sorry

end reconstruct_points_l1642_164227


namespace remainder_product_modulo_17_l1642_164287

theorem remainder_product_modulo_17 :
  (1234 % 17) = 5 ∧ (1235 % 17) = 6 ∧ (1236 % 17) = 7 ∧ (1237 % 17) = 8 ∧ (1238 % 17) = 9 →
  ((1234 * 1235 * 1236 * 1237 * 1238) % 17) = 9 :=
by
  sorry

end remainder_product_modulo_17_l1642_164287


namespace trig_identity_l1642_164221

theorem trig_identity :
  2 * Real.sin (Real.pi / 6) - Real.cos (Real.pi / 4)^2 + Real.cos (Real.pi / 3) = 1 :=
by
  sorry

end trig_identity_l1642_164221


namespace max_area_of_sector_l1642_164212

theorem max_area_of_sector (α R C : Real) (hC : C > 0) (h : C = 2 * R + α * R) : 
  ∃ S_max : Real, S_max = (C^2) / 16 :=
by
  sorry

end max_area_of_sector_l1642_164212


namespace regular_polygon_sides_l1642_164269

theorem regular_polygon_sides (n : ℕ) (h : ∀ (θ : ℝ), θ = 36 → θ = 360 / n) : n = 10 := by
  sorry

end regular_polygon_sides_l1642_164269


namespace arithmetic_sequence_a8_l1642_164219

def sum_arithmetic_sequence_first_n_terms (a d : ℕ) (n : ℕ): ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_a8 
  (a d : ℕ) 
  (h : sum_arithmetic_sequence_first_n_terms a d 15 = 45) : 
  a + 7 * d = 3 := 
by
  sorry

end arithmetic_sequence_a8_l1642_164219


namespace simplify_and_evaluate_l1642_164243

theorem simplify_and_evaluate (m : ℤ) (h : m = -2) :
  let expr := (m / (m^2 - 9)) / (1 + (3 / (m - 3)))
  expr = 1 :=
by
  sorry

end simplify_and_evaluate_l1642_164243


namespace bouncy_ball_pack_count_l1642_164272

theorem bouncy_ball_pack_count
  (x : ℤ)  -- Let x be the number of bouncy balls in each pack
  (r : ℤ := 7 * x)  -- Total number of red bouncy balls
  (y : ℤ := 6 * x)  -- Total number of yellow bouncy balls
  (h : r = y + 18)  -- Condition: 7x = 6x + 18
  : x = 18 := sorry

end bouncy_ball_pack_count_l1642_164272


namespace domain_g_l1642_164235

noncomputable def g (x : ℝ) := Real.tan (Real.arccos (x ^ 3))

theorem domain_g :
  {x : ℝ | ∃ y, g x = y} = {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)} :=
by
  sorry

end domain_g_l1642_164235


namespace triangle_identity_proof_l1642_164252

variables (r r_a r_b r_c R S p : ℝ)
-- assume necessary properties for valid triangle (not explicitly given in problem but implied)
-- nonnegativity, relations between inradius, exradii and circumradius, etc.

theorem triangle_identity_proof
  (h_r_pos : 0 < r)
  (h_ra_pos : 0 < r_a)
  (h_rb_pos : 0 < r_b)
  (h_rc_pos : 0 < r_c)
  (h_R_pos : 0 < R)
  (h_S_pos : 0 < S)
  (h_p_pos : 0 < p)
  (h_area : S = r * p) :
  (1 / r^3) - (1 / r_a^3) - (1 / r_b^3) - (1 / r_c^3) = (12 * R) / (S^2) :=
sorry

end triangle_identity_proof_l1642_164252


namespace cubic_equation_roots_l1642_164289

theorem cubic_equation_roots (a b c d r s t : ℝ) (h_eq : a ≠ 0) 
(ht1 : a * r^3 + b * r^2 + c * r + d = 0)
(ht2 : a * s^3 + b * s^2 + c * s + d = 0)
(ht3 : a * t^3 + b * t^2 + c * t + d = 0)
(h1 : r * s = 3) 
(h2 : r * t = 3) 
(h3 : s * t = 3) : 
c = 3 * a := 
sorry

end cubic_equation_roots_l1642_164289


namespace new_average_mark_l1642_164236

theorem new_average_mark (average_mark : ℕ) (average_excluded : ℕ) (total_students : ℕ) (excluded_students: ℕ)
    (h1 : average_mark = 90)
    (h2 : average_excluded = 45)
    (h3 : total_students = 20)
    (h4 : excluded_students = 2) :
  ((total_students * average_mark - excluded_students * average_excluded) / (total_students - excluded_students)) = 95 := by
  sorry

end new_average_mark_l1642_164236


namespace square_length_QP_l1642_164201

theorem square_length_QP (r1 r2 dist : ℝ) (h_r1 : r1 = 10) (h_r2 : r2 = 7) (h_dist : dist = 15)
  (x : ℝ) (h_equal_chords: QP = PR) :
  x ^ 2 = 65 :=
sorry

end square_length_QP_l1642_164201


namespace number_of_girls_and_boys_l1642_164286

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) := g = 4 * (g + b) / 7 ∧ b = 3 * (g + b) / 7
def total_students (g b : ℕ) := g + b = 56

-- The main proof statement
theorem number_of_girls_and_boys (g b : ℕ) 
  (h_ratio : ratio_girls_to_boys g b)
  (h_total : total_students g b) : 
  g = 32 ∧ b = 24 :=
by {
  sorry
}

end number_of_girls_and_boys_l1642_164286


namespace bees_count_l1642_164295

theorem bees_count (x : ℕ) (h1 : (1/5 : ℚ) * x + (1/3 : ℚ) * x + 
    3 * ((1/3 : ℚ) * x - (1/5 : ℚ) * x) + 1 = x) : x = 15 := 
sorry

end bees_count_l1642_164295


namespace exists_j_half_for_all_j_l1642_164250

def is_j_half (n j : ℕ) : Prop := 
  ∃ (q : ℕ), n = (2 * j + 1) * q + j

theorem exists_j_half_for_all_j (k : ℕ) : 
  ∃ n : ℕ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ k → is_j_half n j :=
by
  sorry

end exists_j_half_for_all_j_l1642_164250


namespace check_conditions_l1642_164267

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) := a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℤ) (n : ℕ) := n * a + (n * (n - 1) / 2) * d

theorem check_conditions {a d : ℤ}
  (S6 S7 S5 : ℤ)
  (h1 : S6 = sum_of_first_n_terms a d 6)
  (h2 : S7 = sum_of_first_n_terms a d 7)
  (h3 : S5 = sum_of_first_n_terms a d 5)
  (h : S6 > S7 ∧ S7 > S5) :
  d < 0 ∧
  sum_of_first_n_terms a d 11 > 0 ∧
  sum_of_first_n_terms a d 13 < 0 ∧
  sum_of_first_n_terms a d 9 > sum_of_first_n_terms a d 3 := 
sorry

end check_conditions_l1642_164267


namespace sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l1642_164247

-- Definitions based on conditions
def sum_arithmetic (n : ℕ) : ℕ := n * (n + 1) / 2
def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Theorem statements based on the correct answers
theorem sum_first_twelve_multiples_17 : 
  17 * sum_arithmetic 12 = 1326 := 
by
  sorry

theorem sum_squares_first_twelve_multiples_17 : 
  17^2 * sum_squares 12 = 187850 :=
by
  sorry

end sum_first_twelve_multiples_17_sum_squares_first_twelve_multiples_17_l1642_164247


namespace total_enemies_l1642_164294

theorem total_enemies (n : ℕ) : (n - 3) * 9 = 72 → n = 11 :=
by
  sorry

end total_enemies_l1642_164294


namespace jennifer_initial_oranges_l1642_164262

theorem jennifer_initial_oranges (O : ℕ) : 
  ∀ (pears apples remaining_fruits : ℕ),
    pears = 10 →
    apples = 2 * pears →
    remaining_fruits = pears - 2 + apples - 2 + O - 2 →
    remaining_fruits = 44 →
    O = 20 :=
by
  intros pears apples remaining_fruits h1 h2 h3 h4
  sorry

end jennifer_initial_oranges_l1642_164262


namespace desiredCircleEquation_l1642_164231

-- Definition of the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Definition of the given line
def givenLine (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- The required proof problem statement
theorem desiredCircleEquation :
  (∀ P Q : ℝ × ℝ, givenCircle P.1 P.2 ∧ givenLine P.1 P.2 → givenCircle Q.1 Q.2 ∧ givenLine Q.1 Q.2 →
  (P ≠ Q) → 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0)) :=
by
  -- Proof omitted
  sorry

end desiredCircleEquation_l1642_164231


namespace find_a_l1642_164284

def f (x : ℝ) : ℝ := 5 * x - 6
def g (x : ℝ) : ℝ := 2 * x + 1

theorem find_a : ∃ a : ℝ, f a + g a = 0 ∧ a = 5 / 7 :=
by
  sorry

end find_a_l1642_164284


namespace remainder_product_mod_5_l1642_164282

theorem remainder_product_mod_5 : (1657 * 2024 * 1953 * 1865) % 5 = 0 := by
  sorry

end remainder_product_mod_5_l1642_164282


namespace prove_true_statement_l1642_164217

-- Definitions based on conditions in the problem
def A_statement := ∀ x : ℝ, x = 2 → (x - 2) * (x - 1) = 0

-- Equivalent proof problem in Lean 4
theorem prove_true_statement : A_statement :=
by
  sorry

end prove_true_statement_l1642_164217


namespace parabola_standard_equation_l1642_164268

theorem parabola_standard_equation (x y : ℝ) :
  (3 * x - 4 * y - 12 = 0) →
  (y^2 = 16 * x ∨ x^2 = -12 * y) :=
sorry

end parabola_standard_equation_l1642_164268


namespace ryan_learning_schedule_l1642_164213

theorem ryan_learning_schedule
  (E1 E2 E3 S1 S2 S3 : ℕ)
  (hE1 : E1 = 7) (hE2 : E2 = 6) (hE3 : E3 = 8)
  (hS1 : S1 = 4) (hS2 : S2 = 5) (hS3 : S3 = 3):
  (E1 + E2 + E3) - (S1 + S2 + S3) = 9 :=
by
  sorry

end ryan_learning_schedule_l1642_164213


namespace percentage_reduction_in_oil_price_l1642_164256

theorem percentage_reduction_in_oil_price (R : ℝ) (P : ℝ) (hR : R = 48) (h_quantity : (800/R) - (800/P) = 5) : 
    ((P - R) / P) * 100 = 30 := 
    sorry

end percentage_reduction_in_oil_price_l1642_164256


namespace ned_mowed_in_summer_l1642_164277

def mowed_in_summer (total_mows spring_mows summer_mows : ℕ) : Prop :=
  total_mows = spring_mows + summer_mows

theorem ned_mowed_in_summer :
  ∀ (total_mows spring_mows summer_mows : ℕ),
  total_mows = 11 →
  spring_mows = 6 →
  mowed_in_summer total_mows spring_mows summer_mows →
  summer_mows = 5 :=
by
  intros total_mows spring_mows summer_mows h_total h_spring h_mowed
  sorry

end ned_mowed_in_summer_l1642_164277


namespace max_xy_l1642_164281

variable {x y : ℝ}

theorem max_xy (h1 : 0 < x) (h2 : 0 < y) (h3 : 3 * x + 8 * y = 48) : x * y ≤ 24 :=
sorry

end max_xy_l1642_164281


namespace problem_statement_l1642_164200

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

-- Define the complement of B in U
def C_U_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- State the theorem
theorem problem_statement : (A ∩ C_U_B) = {1, 2} :=
by {
  -- Proof is omitted
  sorry
}

end problem_statement_l1642_164200


namespace distance_apart_after_two_hours_l1642_164245

theorem distance_apart_after_two_hours :
  (Jay_walk_rate : ℝ) = 1 / 20 →
  (Paul_jog_rate : ℝ) = 3 / 40 →
  (time_duration : ℝ) = 2 * 60 →
  (distance_apart : ℝ) = 15 :=
by
  sorry

end distance_apart_after_two_hours_l1642_164245


namespace stratified_sampling_medium_stores_l1642_164244

noncomputable def total_stores := 300
noncomputable def large_stores := 30
noncomputable def medium_stores := 75
noncomputable def small_stores := 195
noncomputable def sample_size := 20

theorem stratified_sampling_medium_stores : 
  (medium_stores : ℕ) * (sample_size : ℕ) / (total_stores : ℕ) = 5 :=
by
  sorry

end stratified_sampling_medium_stores_l1642_164244


namespace range_of_m_l1642_164202

variable {x m : ℝ}

def absolute_value_inequality (x m : ℝ) : Prop := |x + 1| - |x - 2| > m

theorem range_of_m : (∀ x : ℝ, absolute_value_inequality x m) ↔ m < -3 :=
by
  sorry

end range_of_m_l1642_164202


namespace helicopter_rental_cost_l1642_164288

theorem helicopter_rental_cost :
  let hours_per_day := 2
  let days := 3
  let rate_first_day := 85
  let rate_second_day := 75
  let rate_third_day := 65
  let total_cost_before_discount := hours_per_day * rate_first_day + hours_per_day * rate_second_day + hours_per_day * rate_third_day
  let discount := 0.05
  let discounted_amount := total_cost_before_discount * discount
  let total_cost_after_discount := total_cost_before_discount - discounted_amount
  total_cost_after_discount = 427.50 :=
by
  sorry

end helicopter_rental_cost_l1642_164288


namespace ratio_of_boat_to_stream_l1642_164233

theorem ratio_of_boat_to_stream (B S : ℝ) (h : ∀ D : ℝ, D / (B - S) = 2 * (D / (B + S))) :
  B / S = 3 :=
by 
  sorry

end ratio_of_boat_to_stream_l1642_164233


namespace average_retail_price_l1642_164239

theorem average_retail_price 
  (products : Fin 20 → ℝ)
  (h1 : ∀ i, 400 ≤ products i) 
  (h2 : ∃ s : Finset (Fin 20), s.card = 10 ∧ ∀ i ∈ s, products i < 1000)
  (h3 : ∃ i, products i = 11000): 
  (Finset.univ.sum products) / 20 = 1200 := 
by
  sorry

end average_retail_price_l1642_164239


namespace min_value_expression_l1642_164296

theorem min_value_expression (a b c : ℝ) (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0) (h2 : a < b) :
  ∃ x : ℝ, x = 1 ∧ x = (3 * a - 2 * b + c) / (b - a) := 
  sorry

end min_value_expression_l1642_164296


namespace pentagon_AEDCB_area_l1642_164299

-- Definitions based on the given conditions
def rectangle_ABCD (AB BC : ℕ) : Prop :=
AB = 12 ∧ BC = 10

def triangle_ADE (AE ED : ℕ) : Prop :=
AE = 9 ∧ ED = 6 ∧ AE * ED ≠ 0 ∧ (AE^2 + ED^2 = (AE^2 + ED^2))

def area_of_rectangle (AB BC : ℕ) : ℕ :=
AB * BC

def area_of_triangle (AE ED : ℕ) : ℕ :=
(AE * ED) / 2

-- The theorem to be proved
theorem pentagon_AEDCB_area (AB BC AE ED : ℕ) (h_rect : rectangle_ABCD AB BC) (h_tri : triangle_ADE AE ED) :
  area_of_rectangle AB BC - area_of_triangle AE ED = 93 :=
sorry

end pentagon_AEDCB_area_l1642_164299


namespace total_boys_in_camp_l1642_164210

theorem total_boys_in_camp (T : ℝ) (h : 0.70 * (0.20 * T) = 28) : T = 200 := 
by
  sorry

end total_boys_in_camp_l1642_164210


namespace complement_set_l1642_164216

open Set

theorem complement_set (U M : Set ℕ) (hU : U = {1, 2, 3, 4, 5, 6}) (hM : M = {1, 2, 4}) :
  compl M ∩ U = {3, 5, 6} := 
by
  rw [compl, hU, hM]
  sorry

end complement_set_l1642_164216


namespace collinear_points_sum_l1642_164283

-- Points in 3-dimensional space.
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition of collinearity for three points
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ k : ℝ,
    k ≠ 0 ∧
    (p2.x - p1.x) * k = (p3.x - p1.x) ∧
    (p2.y - p1.y) * k = (p3.y - p1.y) ∧
    (p2.z - p1.z) * k = (p3.z - p1.z)

-- Main statement
theorem collinear_points_sum {a b : ℝ} :
  collinear (Point3D.mk 2 a b) (Point3D.mk a 3 b) (Point3D.mk a b 4) → a + b = 6 :=
by
  sorry

end collinear_points_sum_l1642_164283


namespace given_condition_implies_result_l1642_164264

theorem given_condition_implies_result (a : ℝ) (h : a ^ 2 + 2 * a = 1) : 2 * a ^ 2 + 4 * a + 1 = 3 :=
sorry

end given_condition_implies_result_l1642_164264
