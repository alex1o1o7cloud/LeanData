import Mathlib

namespace f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l1343_134338

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem f_monotonically_increasing_intervals:
  ∀ (k : ℤ), ∀ x y, (-Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ y ∧ y ≤ (k * Real.pi + Real.pi / 3) → f x ≤ f y :=
sorry

theorem f_max_min_in_range:
  ∀ x, (-Real.pi / 12) ≤ x ∧ x ≤ (5 * Real.pi / 12) → 
  (f x ≤ 2 ∧ f x ≥ -Real.sqrt 3) :=
sorry

theorem f_max_at_pi_over_3:
  f (Real.pi / 3) = 2 :=
sorry

theorem f_min_at_neg_pi_over_12:
  f (-Real.pi / 12) = -Real.sqrt 3 :=
sorry

end f_monotonically_increasing_intervals_f_max_min_in_range_f_max_at_pi_over_3_f_min_at_neg_pi_over_12_l1343_134338


namespace fgh_supermarkets_l1343_134366

theorem fgh_supermarkets (U C : ℕ) 
  (h1 : U + C = 70) 
  (h2 : U = C + 14) : U = 42 :=
by
  sorry

end fgh_supermarkets_l1343_134366


namespace same_color_eye_proportion_l1343_134399

theorem same_color_eye_proportion :
  ∀ (a b c d e f : ℝ),
  a + b + c = 0.30 →
  a + d + e = 0.40 →
  b + d + f = 0.50 →
  a + b + c + d + e + f = 1 →
  c + e + f = 0.80 :=
by
  intros a b c d e f h1 h2 h3 h4
  sorry

end same_color_eye_proportion_l1343_134399


namespace units_digit_of_result_l1343_134346

theorem units_digit_of_result (a b c : ℕ) (h1 : a = c + 3) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  let result := original - reversed
  result % 10 = 7 :=
by
  sorry

end units_digit_of_result_l1343_134346


namespace pencils_total_l1343_134372

def pencils_remaining (Jeff_pencils_initial : ℕ) (Jeff_donation_percent : ℕ) 
                      (Vicki_factor : ℕ) (Vicki_donation_fraction_num : ℕ) 
                      (Vicki_donation_fraction_den : ℕ) : ℕ :=
  let Jeff_donated := Jeff_pencils_initial * Jeff_donation_percent / 100
  let Jeff_remaining := Jeff_pencils_initial - Jeff_donated
  let Vicki_pencils_initial := Vicki_factor * Jeff_pencils_initial
  let Vicki_donated := Vicki_pencils_initial * Vicki_donation_fraction_num / Vicki_donation_fraction_den
  let Vicki_remaining := Vicki_pencils_initial - Vicki_donated
  Jeff_remaining + Vicki_remaining

theorem pencils_total :
  pencils_remaining 300 30 2 3 4 = 360 :=
by
  -- The proof should be inserted here
  sorry

end pencils_total_l1343_134372


namespace proof_y_pow_x_equal_1_by_9_l1343_134320

theorem proof_y_pow_x_equal_1_by_9 
  (x y : ℝ)
  (h : (x - 2)^2 + abs (y + 1/3) = 0) :
  y^x = 1/9 := by
  sorry

end proof_y_pow_x_equal_1_by_9_l1343_134320


namespace total_fish_l1343_134318

theorem total_fish (fish_lilly fish_rosy : ℕ) (hl : fish_lilly = 10) (hr : fish_rosy = 14) :
  fish_lilly + fish_rosy = 24 := 
by 
  sorry

end total_fish_l1343_134318


namespace blue_face_area_greater_than_red_face_area_l1343_134381

theorem blue_face_area_greater_than_red_face_area :
  let original_cube_side := 13
  let total_red_area := 6 * original_cube_side^2
  let num_mini_cubes := original_cube_side^3
  let total_faces_mini_cubes := 6 * num_mini_cubes
  let total_blue_area := total_faces_mini_cubes - total_red_area
  (total_blue_area / total_red_area) = 12 :=
by
  sorry

end blue_face_area_greater_than_red_face_area_l1343_134381


namespace solve_equations_l1343_134391

theorem solve_equations :
  (∀ x : ℝ, x^2 - 2 * x - 15 = 0 ↔ x = 5 ∨ x = -3) ∧
  (∀ x : ℝ, 2 * x^2 + 3 * x - 1 = 0 ↔ x = (-3 + Real.sqrt 17) / 4 ∨ x = (-3 - Real.sqrt 17) / 4) :=
by
  sorry

end solve_equations_l1343_134391


namespace round_trip_ticket_percentage_l1343_134317

theorem round_trip_ticket_percentage (P R : ℝ) 
  (h1 : 0.20 * P = 0.50 * R) : R = 0.40 * P :=
by
  sorry

end round_trip_ticket_percentage_l1343_134317


namespace intersection_M_N_l1343_134361

open Set

variable (x y : ℝ)

theorem intersection_M_N :
  let M := {x | x < 1}
  let N := {y | ∃ x, x < 1 ∧ y = 1 - 2 * x}
  M ∩ N = ∅ := sorry

end intersection_M_N_l1343_134361


namespace remainder_8_pow_310_mod_9_l1343_134342

theorem remainder_8_pow_310_mod_9 : (8 ^ 310) % 9 = 8 := 
by
  sorry

end remainder_8_pow_310_mod_9_l1343_134342


namespace find_a_l1343_134364

-- Definitions for the hyperbola and its eccentricity
def hyperbola_eq (a : ℝ) : Prop := a > 0 ∧ ∃ b : ℝ, b^2 = 3 ∧ ∃ e : ℝ, e = 2 ∧ 
  e = Real.sqrt (1 + b^2 / a^2)

-- The main theorem stating the value of 'a' given the conditions
theorem find_a (a : ℝ) (h : hyperbola_eq a) : a = 1 := 
by {
  sorry
}

end find_a_l1343_134364


namespace max_triangle_area_l1343_134358

-- Definitions for the conditions
def Point := (ℝ × ℝ)

def point_A : Point := (0, 0)
def point_B : Point := (17, 0)
def point_C : Point := (23, 0)

def slope_ell_A : ℝ := 2
def slope_ell_C : ℝ := -2

axiom rotating_clockwise_with_same_angular_velocity (A B C : Point) : Prop

-- Question transcribed as proving a statement about the maximum area
theorem max_triangle_area (A B C : Point)
  (hA : A = point_A)
  (hB : B = point_B)
  (hC : C = point_C)
  (h_slopeA : ∀ p: Point, slope_ell_A = 2)
  (h_slopeC : ∀ p: Point, slope_ell_C = -2)
  (h_rotation : rotating_clockwise_with_same_angular_velocity A B C) :
  ∃ area_max : ℝ, area_max = 264.5 :=
sorry

end max_triangle_area_l1343_134358


namespace p_sufficient_not_necessary_for_q_l1343_134331

noncomputable def p (x : ℝ) : Prop := |x - 3| < 1
noncomputable def q (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem p_sufficient_not_necessary_for_q (x : ℝ) :
  (p x → q x) ∧ (¬ (q x → p x)) := by
  sorry

end p_sufficient_not_necessary_for_q_l1343_134331


namespace ned_trips_l1343_134398

theorem ned_trips : 
  ∀ (carry_capacity : ℕ) (table1 : ℕ) (table2 : ℕ) (table3 : ℕ) (table4 : ℕ),
  carry_capacity = 5 →
  table1 = 7 →
  table2 = 10 →
  table3 = 12 →
  table4 = 3 →
  (table1 + table2 + table3 + table4 + carry_capacity - 1) / carry_capacity = 8 :=
by
  intro carry_capacity table1 table2 table3 table4
  intro h1 h2 h3 h4 h5
  sorry

end ned_trips_l1343_134398


namespace share_ratio_l1343_134339

theorem share_ratio (A B C : ℝ) (x : ℝ) (h1 : A + B + C = 500) (h2 : A = 200) (h3 : A = x * (B + C)) (h4 : B = (6/9) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end share_ratio_l1343_134339


namespace smallest_k_l1343_134377

theorem smallest_k (a b c : ℤ) (k : ℤ) (h1 : a < b) (h2 : b < c) 
  (h3 : 2 * b = a + c) (h4 : (k * c) ^ 2 = a * b) (h5 : k > 1) : 
  c > 0 → k = 2 := 
sorry

end smallest_k_l1343_134377


namespace erick_total_revenue_l1343_134337

def lemon_price_increase := 4
def grape_price_increase := lemon_price_increase / 2
def original_lemon_price := 8
def original_grape_price := 7
def lemons_sold := 80
def grapes_sold := 140

def new_lemon_price := original_lemon_price + lemon_price_increase -- $12 per lemon
def new_grape_price := original_grape_price + grape_price_increase -- $9 per grape

def revenue_from_lemons := lemons_sold * new_lemon_price -- $960
def revenue_from_grapes := grapes_sold * new_grape_price -- $1260

def total_revenue := revenue_from_lemons + revenue_from_grapes

theorem erick_total_revenue : total_revenue = 2220 := by
  -- Skipping proof with sorry
  sorry

end erick_total_revenue_l1343_134337


namespace dogs_eat_times_per_day_l1343_134352

theorem dogs_eat_times_per_day (dogs : ℕ) (food_per_dog_per_meal : ℚ) (total_food : ℚ) 
                                (food_left : ℚ) (days : ℕ) 
                                (dogs_eat_times_per_day : ℚ)
                                (h_dogs : dogs = 3)
                                (h_food_per_dog_per_meal : food_per_dog_per_meal = 1 / 2)
                                (h_total_food : total_food = 30)
                                (h_food_left : food_left = 9)
                                (h_days : days = 7) :
                                dogs_eat_times_per_day = 2 :=
by
  -- Proof goes here
  sorry

end dogs_eat_times_per_day_l1343_134352


namespace total_week_cost_proof_l1343_134379

-- Defining variables for costs and consumption
def cost_brand_a_biscuit : ℝ := 0.25
def cost_brand_b_biscuit : ℝ := 0.35
def cost_small_rawhide : ℝ := 1
def cost_large_rawhide : ℝ := 1.50

def odd_days_biscuits_brand_a : ℕ := 3
def odd_days_biscuits_brand_b : ℕ := 2
def odd_days_small_rawhide : ℕ := 1
def odd_days_large_rawhide : ℕ := 1

def even_days_biscuits_brand_a : ℕ := 4
def even_days_small_rawhide : ℕ := 2

def odd_day_cost : ℝ :=
  odd_days_biscuits_brand_a * cost_brand_a_biscuit +
  odd_days_biscuits_brand_b * cost_brand_b_biscuit +
  odd_days_small_rawhide * cost_small_rawhide +
  odd_days_large_rawhide * cost_large_rawhide

def even_day_cost : ℝ :=
  even_days_biscuits_brand_a * cost_brand_a_biscuit +
  even_days_small_rawhide * cost_small_rawhide

def total_cost_per_week : ℝ :=
  4 * odd_day_cost + 3 * even_day_cost

theorem total_week_cost_proof :
  total_cost_per_week = 24.80 :=
  by
    unfold total_cost_per_week
    unfold odd_day_cost
    unfold even_day_cost
    norm_num
    sorry

end total_week_cost_proof_l1343_134379


namespace calculate_y_l1343_134386

theorem calculate_y (w x y : ℝ) (h1 : (7 / w) + (7 / x) = 7 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : y = 0.25 :=
by
  sorry

end calculate_y_l1343_134386


namespace unique_solution_k_l1343_134370

theorem unique_solution_k (k : ℕ) (f : ℕ → ℕ) :
  (∀ n : ℕ, (Nat.iterate f n n) = n + k) → k = 0 :=
by
  sorry

end unique_solution_k_l1343_134370


namespace find_a2_plus_b2_l1343_134356

theorem find_a2_plus_b2 (a b : ℝ) (h1 : a * b = -1) (h2 : a - b = 2) : a^2 + b^2 = 2 := 
by
  sorry

end find_a2_plus_b2_l1343_134356


namespace acute_triangle_orthocenter_l1343_134314

variables (A B C H : Point) (a b c h_a h_b h_c : Real)

def acute_triangle (α β γ : Point) : Prop := 
-- Definition that ensures triangle αβγ is acute
sorry

def orthocenter (α β γ ω : Point) : Prop := 
-- Definition that ω is the orthocenter of triangle αβγ 
sorry

def sides_of_triangle (α β γ : Point) : (Real × Real × Real) := 
-- Function that returns the side lengths of triangle αβγ as (a, b, c)
sorry

def altitudes_of_triangle (α β γ θ : Point) : (Real × Real × Real) := 
-- Function that returns the altitudes of triangle αβγ with orthocenter θ as (h_a, h_b, h_c)
sorry

theorem acute_triangle_orthocenter 
  (A B C H : Point)
  (a b c h_a h_b h_c : Real)
  (ht : acute_triangle A B C)
  (orth : orthocenter A B C H)
  (sides : sides_of_triangle A B C = (a, b, c))
  (alts : altitudes_of_triangle A B C H = (h_a, h_b, h_c)) :
  AH * h_a + BH * h_b + CH * h_c = (a^2 + b^2 + c^2) / 2 :=
by sorry


end acute_triangle_orthocenter_l1343_134314


namespace probability_of_four_of_a_kind_is_correct_l1343_134322

noncomputable def probability_four_of_a_kind: ℚ :=
  let total_ways := Nat.choose 52 5
  let successful_ways := 13 * 1 * 12 * 4
  (successful_ways: ℚ) / (total_ways: ℚ)

theorem probability_of_four_of_a_kind_is_correct :
  probability_four_of_a_kind = 13 / 54145 := 
by
  -- sorry is used because we are only writing the statement, no proof required
  sorry

end probability_of_four_of_a_kind_is_correct_l1343_134322


namespace jack_should_leave_300_in_till_l1343_134334

-- Defining the amounts of each type of bill
def num_100_bills := 2
def num_50_bills := 1
def num_20_bills := 5
def num_10_bills := 3
def num_5_bills := 7
def num_1_bills := 27

-- The amount he needs to hand in
def amount_to_hand_in := 142

-- Calculating the total amount in notes
def total_in_notes := 
  (num_100_bills * 100) + 
  (num_50_bills * 50) + 
  (num_20_bills * 20) + 
  (num_10_bills * 10) + 
  (num_5_bills * 5) + 
  (num_1_bills * 1)

-- Calculating the amount to leave in the till
def amount_to_leave := total_in_notes - amount_to_hand_in

-- Proof statement
theorem jack_should_leave_300_in_till :
  amount_to_leave = 300 :=
by sorry

end jack_should_leave_300_in_till_l1343_134334


namespace great_wall_scientific_notation_l1343_134343

theorem great_wall_scientific_notation : 
  (21200000 : ℝ) = 2.12 * 10^7 :=
by
  sorry

end great_wall_scientific_notation_l1343_134343


namespace marble_problem_l1343_134325

theorem marble_problem {r b : ℕ} 
  (h1 : 9 * r - b = 27) 
  (h2 : 3 * r - b = 3) : r + b = 13 := 
by
  sorry

end marble_problem_l1343_134325


namespace problem1_l1343_134374

theorem problem1 (a b c : ℝ) (h : a * c + b * c + c^2 < 0) : b^2 > 4 * a * c := sorry

end problem1_l1343_134374


namespace perfect_cube_divisor_count_l1343_134300

noncomputable def num_perfect_cube_divisors : Nat :=
  let a_choices := Nat.succ (38 / 3)
  let b_choices := Nat.succ (17 / 3)
  let c_choices := Nat.succ (7 / 3)
  let d_choices := Nat.succ (4 / 3)
  a_choices * b_choices * c_choices * d_choices

theorem perfect_cube_divisor_count :
  num_perfect_cube_divisors = 468 :=
by
  sorry

end perfect_cube_divisor_count_l1343_134300


namespace find_point_on_x_axis_l1343_134306

theorem find_point_on_x_axis (a : ℝ) (h : abs (3 * a + 6) = 30) : (a = -12) ∨ (a = 8) :=
sorry

end find_point_on_x_axis_l1343_134306


namespace no_triangles_with_geometric_progression_angles_l1343_134333

theorem no_triangles_with_geometric_progression_angles :
  ¬ ∃ (a r : ℕ), a ≥ 10 ∧ (a + a * r + a * r^2 = 180) ∧ (a ≠ a * r) ∧ (a ≠ a * r^2) ∧ (a * r ≠ a * r^2) :=
sorry

end no_triangles_with_geometric_progression_angles_l1343_134333


namespace alcohol_percentage_second_vessel_l1343_134394

theorem alcohol_percentage_second_vessel:
  ∃ x : ℝ, 
  let alcohol_in_first := 0.25 * 2
  let alcohol_in_second := 0.01 * x * 6
  let total_alcohol := 0.29 * 8
  alcohol_in_first + alcohol_in_second = total_alcohol → 
  x = 30.333333333333332 :=
by
  sorry

end alcohol_percentage_second_vessel_l1343_134394


namespace house_number_units_digit_is_five_l1343_134349

/-- Define the house number as a two-digit number -/
def is_two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

/-- Define the properties for the statements -/
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_power_of_prime (n : ℕ) : Prop := ∃ p : ℕ, Nat.Prime p ∧ p ^ Nat.log p n = n
def is_divisible_by_five (n : ℕ) : Prop := n % 5 = 0
def has_digit_seven (n : ℕ) : Prop := (n / 10 = 7 ∨ n % 10 = 7)

/-- The theorem stating that the units digit of the house number is 5 -/
theorem house_number_units_digit_is_five (n : ℕ) 
  (h1 : is_two_digit_number n)
  (h2 : (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n) ∨ 
        (¬is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (¬is_prime n ∧ is_power_of_prime n ∧ ¬is_divisible_by_five n ∧ has_digit_seven n) ∨ 
        (is_prime n ∧ ¬is_power_of_prime n ∧ is_divisible_by_five n ∧ ¬has_digit_seven n))
  : n % 10 = 5 := 
sorry

end house_number_units_digit_is_five_l1343_134349


namespace differences_l1343_134315

def seq (n : ℕ) : ℕ := n^2 + 1

def first_diff (n : ℕ) : ℕ := (seq (n + 1)) - (seq n)

def second_diff (n : ℕ) : ℕ := (first_diff (n + 1)) - (first_diff n)

def third_diff (n : ℕ) : ℕ := (second_diff (n + 1)) - (second_diff n)

theorem differences (n : ℕ) : first_diff n = 2 * n + 1 ∧ 
                             second_diff n = 2 ∧ 
                             third_diff n = 0 := by 
  sorry

end differences_l1343_134315


namespace diff_sum_even_odd_l1343_134368

theorem diff_sum_even_odd (n : ℕ) (hn : n = 1500) :
  let sum_odd := n * (2 * n - 1)
  let sum_even := n * (2 * n + 1)
  sum_even - sum_odd = 1500 :=
by
  sorry

end diff_sum_even_odd_l1343_134368


namespace constant_term_expansion_l1343_134303

theorem constant_term_expansion (x : ℝ) (n : ℕ) (h : (x + 2 + 1/x)^n = 20) : n = 3 :=
by
sorry

end constant_term_expansion_l1343_134303


namespace general_formula_correct_sequence_T_max_term_l1343_134369

open Classical

noncomputable def geometric_sequence_term (n : ℕ) : ℝ :=
  if h : n > 0 then (-1)^(n-1) * (3 / 2^n)
  else 0

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ :=
  if h : n > 0 then 1 - (-1 / 2)^n
  else 0

noncomputable def sequence_T (n : ℕ) : ℝ :=
  geometric_sequence_sum n + 1 / geometric_sequence_sum n

theorem general_formula_correct :
  ∀ n : ℕ, n > 0 → geometric_sequence_term n = (-1)^(n-1) * (3 / 2^n) :=
sorry

theorem sequence_T_max_term :
  ∀ n : ℕ, n > 0 → sequence_T n ≤ sequence_T 1 ∧ sequence_T 1 = 13 / 6 :=
sorry

end general_formula_correct_sequence_T_max_term_l1343_134369


namespace total_gift_amount_l1343_134365

-- Definitions based on conditions
def workers_per_block := 200
def number_of_blocks := 15
def worth_of_each_gift := 2

-- The statement we need to prove
theorem total_gift_amount : workers_per_block * number_of_blocks * worth_of_each_gift = 6000 := by
  sorry

end total_gift_amount_l1343_134365


namespace tan_theta_solution_l1343_134327

theorem tan_theta_solution (θ : ℝ) (h : 2 * Real.sin θ = 1 + Real.cos θ) :
  Real.tan θ = 0 ∨ Real.tan θ = 4 / 3 :=
sorry

end tan_theta_solution_l1343_134327


namespace PQ_sum_l1343_134328

theorem PQ_sum (P Q : ℕ) (h1 : 5 / 7 = P / 63) (h2 : 5 / 7 = 70 / Q) : P + Q = 143 :=
by
  sorry

end PQ_sum_l1343_134328


namespace other_coin_denomination_l1343_134396

theorem other_coin_denomination :
  ∀ (total_coins : ℕ) (value_rs : ℕ) (paise_per_rs : ℕ) (num_20_paise_coins : ℕ) (total_value_paise : ℕ),
  total_coins = 324 →
  value_rs = 71 →
  paise_per_rs = 100 →
  num_20_paise_coins = 200 →
  total_value_paise = value_rs * paise_per_rs →
  (∃ (denom_other_coin : ℕ),
    total_value_paise - num_20_paise_coins * 20 = (total_coins - num_20_paise_coins) * denom_other_coin
    → denom_other_coin = 25) :=
by
  sorry

end other_coin_denomination_l1343_134396


namespace amount_of_c_l1343_134313

theorem amount_of_c (A B C : ℕ) (h1 : A + B + C = 350) (h2 : A + C = 200) (h3 : B + C = 350) : C = 200 :=
sorry

end amount_of_c_l1343_134313


namespace trains_crossing_time_l1343_134312

-- Definitions based on given conditions
noncomputable def length_A : ℝ := 2500
noncomputable def time_A : ℝ := 50
noncomputable def length_B : ℝ := 3500
noncomputable def speed_factor : ℝ := 1.2

-- Speed computations
noncomputable def speed_A : ℝ := length_A / time_A
noncomputable def speed_B : ℝ := speed_A * speed_factor

-- Relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := speed_A + speed_B

-- Total distance covered when crossing each other
noncomputable def total_distance : ℝ := length_A + length_B

-- Time taken to cross each other
noncomputable def time_to_cross : ℝ := total_distance / relative_speed

-- Proof statement: Time taken is approximately 54.55 seconds
theorem trains_crossing_time :
  |time_to_cross - 54.55| < 0.01 := by
  sorry

end trains_crossing_time_l1343_134312


namespace circle_line_intersect_property_l1343_134371
open Real

theorem circle_line_intersect_property :
  let ρ := fun θ : ℝ => 4 * sqrt 2 * sin (3 * π / 4 - θ)
  let cartesian_eq := fun x y : ℝ => (x - 2) ^ 2 + (y - 2) ^ 2 = 8
  let slope := sqrt 3
  let line_param := fun t : ℝ => (1/2 * t, 2 + sqrt 3 / 2 * t)
  let t_roots := {t | ∃ t1 t2 : ℝ, t1 + t2 = 2 ∧ t1 * t2 = -4 ∧ (t = t1 ∨ t = t2)}
  
  (∀ t ∈ t_roots, 
    let (x, y) := line_param t
    cartesian_eq x y)
  → abs ((1 : ℝ) / abs 1 - (1 : ℝ) / abs 2) = 1 / 2 :=
by
  intro ρ cartesian_eq slope line_param t_roots h
  sorry

end circle_line_intersect_property_l1343_134371


namespace largest_decimal_of_4bit_binary_l1343_134359

-- Define the maximum 4-bit binary number and its interpretation in base 10
def max_4bit_binary_value : ℕ := 2^4 - 1

-- The theorem to prove the statement
theorem largest_decimal_of_4bit_binary : max_4bit_binary_value = 15 :=
by
  -- Lean tactics or explicitly writing out the solution steps can be used here.
  -- Skipping proof as instructed.
  sorry

end largest_decimal_of_4bit_binary_l1343_134359


namespace sally_pokemon_cards_count_l1343_134373

-- Defining the initial conditions
def initial_cards : ℕ := 27
def cards_given_by_dan : ℕ := 41
def cards_bought_by_sally : ℕ := 20

-- Statement of the problem to be proved
theorem sally_pokemon_cards_count :
  initial_cards + cards_given_by_dan + cards_bought_by_sally = 88 := by
  sorry

end sally_pokemon_cards_count_l1343_134373


namespace sequence_12th_term_l1343_134301

theorem sequence_12th_term (C : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 3) (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = C / a n) (h4 : C = 12) : a 12 = 4 :=
sorry

end sequence_12th_term_l1343_134301


namespace johns_cloth_cost_per_metre_l1343_134397

noncomputable def calculate_cost_per_metre (total_cost : ℝ) (total_metres : ℝ) : ℝ :=
  total_cost / total_metres

def johns_cloth_purchasing_data : Prop :=
  calculate_cost_per_metre 444 9.25 = 48

theorem johns_cloth_cost_per_metre : johns_cloth_purchasing_data :=
  sorry

end johns_cloth_cost_per_metre_l1343_134397


namespace count_solutions_sin_equation_l1343_134395

theorem count_solutions_sin_equation : 
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 3 * (Real.sin x)^4 - 7 * (Real.sin x)^3 + 5 * (Real.sin x)^2 - Real.sin x = 0) ∧ S.card = 4 :=
by
  sorry

end count_solutions_sin_equation_l1343_134395


namespace third_class_males_eq_nineteen_l1343_134355

def first_class_males : ℕ := 17
def first_class_females : ℕ := 13
def second_class_males : ℕ := 14
def second_class_females : ℕ := 18
def third_class_females : ℕ := 17
def students_unable_to_partner : ℕ := 2
def total_males_from_first_two_classes : ℕ := first_class_males + second_class_males
def total_females_from_first_two_classes : ℕ := first_class_females + second_class_females
def total_females : ℕ := total_females_from_first_two_classes + third_class_females

theorem third_class_males_eq_nineteen (M : ℕ) : 
  total_males_from_first_two_classes + M - (total_females + students_unable_to_partner) = 0 → M = 19 :=
by
  sorry

end third_class_males_eq_nineteen_l1343_134355


namespace area_ratio_of_squares_l1343_134385

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 1 / 2 * (4 * b)) : (b^2 / a^2) = 4 :=
by
  -- Proof goes here
  sorry

end area_ratio_of_squares_l1343_134385


namespace find_f_x_l1343_134344

def f (x : ℝ) : ℝ := x^2 - 5*x + 6

theorem find_f_x (x : ℝ) : (f (x+1)) = x^2 - 3*x + 2 :=
by
  sorry

end find_f_x_l1343_134344


namespace find_a_for_positive_root_l1343_134353

theorem find_a_for_positive_root (h : ∃ x > 0, (1 - x) / (x - 2) = a / (2 - x) - 2) : a = 1 :=
sorry

end find_a_for_positive_root_l1343_134353


namespace expand_polynomial_l1343_134305

theorem expand_polynomial (x : ℝ) : (x - 2) * (x + 2) * (x^2 + 4 * x + 4) = x^4 + 4 * x^3 - 16 * x - 16 := 
by
  sorry

end expand_polynomial_l1343_134305


namespace percentage_cut_is_50_l1343_134350

-- Conditions
def yearly_subscription_cost : ℝ := 940.0
def reduction_amount : ℝ := 470.0

-- Assertion to be proved
theorem percentage_cut_is_50 :
  (reduction_amount / yearly_subscription_cost) * 100 = 50 :=
by
  sorry

end percentage_cut_is_50_l1343_134350


namespace race_distance_l1343_134357

theorem race_distance
  (x y z d : ℝ) 
  (h1 : d / x = (d - 25) / y)
  (h2 : d / y = (d - 15) / z)
  (h3 : d / x = (d - 35) / z) : 
  d = 75 := 
sorry

end race_distance_l1343_134357


namespace food_cost_max_l1343_134390

theorem food_cost_max (x : ℝ) (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_total : ℝ) (food_cost_max : ℝ) :
  total_cost = x * (1 + tax_rate + tip_rate) →
  tax_rate = 0.07 →
  tip_rate = 0.15 →
  max_total = 50 →
  total_cost ≤ max_total →
  food_cost_max = 50 / 1.22 →
  x ≤ food_cost_max :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end food_cost_max_l1343_134390


namespace circumcircle_excircle_distance_squared_l1343_134336

variable (R r_A d_A : ℝ)

theorem circumcircle_excircle_distance_squared 
  (h : R ≥ 0)
  (h1 : r_A ≥ 0)
  (h2 : d_A^2 = R^2 + 2 * R * r_A) : d_A^2 = R^2 + 2 * R * r_A := 
by
  sorry

end circumcircle_excircle_distance_squared_l1343_134336


namespace solve_equation_l1343_134367

noncomputable def fourthRoot (x : ℝ) := Real.sqrt (Real.sqrt x)

theorem solve_equation (x : ℝ) (hx : x ≥ 0) :
  fourthRoot x = 18 / (9 - fourthRoot x) ↔ x = 81 ∨ x = 1296 :=
by
  sorry

end solve_equation_l1343_134367


namespace smallest_square_length_proof_l1343_134332

-- Define square side length required properties
noncomputable def smallest_square_side_length (rect_w rect_h min_side : ℝ) : ℝ :=
  if h : min_side^2 % (rect_w * rect_h) = 0 then min_side 
  else if h : (min_side + 1)^2 % (rect_w * rect_h) = 0 then min_side + 1
  else if h : (min_side + 2)^2 % (rect_w * rect_h) = 0 then min_side + 2
  else if h : (min_side + 3)^2 % (rect_w * rect_h) = 0 then min_side + 3
  else if h : (min_side + 4)^2 % (rect_w * rect_h) = 0 then min_side + 4
  else if h : (min_side + 5)^2 % (rect_w * rect_h) = 0 then min_side + 5
  else if h : (min_side + 6)^2 % (rect_w * rect_h) = 0 then min_side + 6
  else if h : (min_side + 7)^2 % (rect_w * rect_h) = 0 then min_side + 7
  else if h : (min_side + 8)^2 % (rect_w * rect_h) = 0 then min_side + 8
  else if h : (min_side + 9)^2 % (rect_w * rect_h) = 0 then min_side + 9
  else min_side + 2 -- ensuring it can't be less than min_side

-- State the theorem
theorem smallest_square_length_proof : smallest_square_side_length 2 3 10 = 12 :=
by 
  unfold smallest_square_side_length
  norm_num
  sorry

end smallest_square_length_proof_l1343_134332


namespace direction_vector_arithmetic_sequence_l1343_134384

theorem direction_vector_arithmetic_sequence (a_n : ℕ → ℤ) (S_n : ℕ → ℤ) 
    (n : ℕ) 
    (S2_eq_10 : S_n 2 = 10) 
    (S5_eq_55 : S_n 5 = 55)
    (arith_seq_sum : ∀ n, S_n n = (n * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))) / 2): 
    (a_n (n + 2) - a_n n) / (n + 2 - n) = 4 :=
by
  sorry

end direction_vector_arithmetic_sequence_l1343_134384


namespace prob_of_three_digit_divisible_by_3_l1343_134323

/-- Define the exponents and the given condition --/
def a : ℕ := 5
def b : ℕ := 2
def c : ℕ := 3
def d : ℕ := 1

def condition : Prop := (2^a) * (3^b) * (5^c) * (7^d) = 252000

/-- The probability that a randomly chosen three-digit number formed by any 3 of a, b, c, d 
    is divisible by 3 and less than 250 is 1/4 --/
theorem prob_of_three_digit_divisible_by_3 :
  condition →
  ((sorry : ℝ) = 1/4) := sorry

end prob_of_three_digit_divisible_by_3_l1343_134323


namespace incircle_hexagon_area_ratio_l1343_134389

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def radius_incircle (s : ℝ) : ℝ :=
  (s * Real.sqrt 3) / 2

noncomputable def area_incircle (r : ℝ) : ℝ :=
  Real.pi * r^2

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let A_hexagon := area_hexagon s
  let r := radius_incircle s
  let A_incircle := area_incircle r
  A_incircle / A_hexagon

theorem incircle_hexagon_area_ratio (s : ℝ) (h : s = 1) :
  area_ratio s = (Real.pi * Real.sqrt 3) / 6 :=
by
  sorry

end incircle_hexagon_area_ratio_l1343_134389


namespace simplify_fraction_l1343_134345

theorem simplify_fraction :
  ( (2^1010)^2 - (2^1008)^2 ) / ( (2^1009)^2 - (2^1007)^2 ) = 4 :=
by
  sorry

end simplify_fraction_l1343_134345


namespace rodney_lifting_capacity_l1343_134351

theorem rodney_lifting_capacity 
  (R O N : ℕ)
  (h1 : R + O + N = 239)
  (h2 : R = 2 * O)
  (h3 : O = 4 * N - 7) : 
  R = 146 := 
by
  sorry

end rodney_lifting_capacity_l1343_134351


namespace initial_numbers_conditions_l1343_134360

theorem initial_numbers_conditions (a b c : ℤ)
    (h : ∀ (x y z : ℤ), (x, y, z) = (17, 1967, 1983) → 
      x = y + z - 1 ∨ y = x + z - 1 ∨ z = x + y - 1) :
  (a = 2 ∧ b = 2 ∧ c = 2) → false ∧ 
  (a = 3 ∧ b = 3 ∧ c = 3) → true := 
sorry

end initial_numbers_conditions_l1343_134360


namespace faye_candy_count_l1343_134375

theorem faye_candy_count :
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  initial_candy - candy_ate + candy_given = 62 :=
by
  let initial_candy := 47
  let candy_ate := 25
  let candy_given := 40
  sorry

end faye_candy_count_l1343_134375


namespace find_white_balls_l1343_134304

-- Define a structure to hold the probabilities and total balls
structure BallProperties where
  totalBalls : Nat
  probRed : Real
  probBlack : Real

-- Given data as conditions
def givenData : BallProperties := 
  { totalBalls := 50, probRed := 0.15, probBlack := 0.45 }

-- The statement to prove the number of white balls
theorem find_white_balls (data : BallProperties) : 
  data.totalBalls = 50 →
  data.probRed = 0.15 →
  data.probBlack = 0.45 →
  ∃ whiteBalls : Nat, whiteBalls = 20 :=
by
  sorry

end find_white_balls_l1343_134304


namespace cylinder_side_surface_area_l1343_134387

-- Define the given conditions
def base_circumference : ℝ := 4
def height_of_cylinder : ℝ := 4

-- Define the relation we need to prove
theorem cylinder_side_surface_area : 
  base_circumference * height_of_cylinder = 16 := 
by
  sorry

end cylinder_side_surface_area_l1343_134387


namespace min_value_frac_sum_l1343_134310

open Real

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  3 ≤ (1 / (x + y)) + ((x + y) / z) := 
sorry

end min_value_frac_sum_l1343_134310


namespace polynomial_base5_representation_l1343_134335

-- Define the polynomials P and Q
def P(x : ℕ) : ℕ := 3 * 5^6 + 0 * 5^5 + 0 * 5^4 + 1 * 5^3 + 2 * 5^2 + 4 * 5 + 1
def Q(x : ℕ) : ℕ := 4 * 5^2 + 3 * 5 + 2

-- Define the representation of these polynomials in base-5
def base5_P : ℕ := 3001241
def base5_Q : ℕ := 432

-- Define the expected interpretation of the base-5 representation in decimal
def decimal_P : ℕ := P 0
def decimal_Q : ℕ := Q 0

-- The proof statement
theorem polynomial_base5_representation :
  decimal_P = base5_P ∧ decimal_Q = base5_Q :=
sorry

end polynomial_base5_representation_l1343_134335


namespace monotonic_interval_a_l1343_134388

theorem monotonic_interval_a (a : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → (2 * x - 2 * a) * (2 * 2 - 2 * a) ≥ 0 ∧ (2 * x - 2 * a) * (2 * 3 - 2 * a) ≥ 0) →
  a ≤ 2 ∨ a ≥ 3 := sorry

end monotonic_interval_a_l1343_134388


namespace lowest_possible_sale_price_is_30_percent_l1343_134308

noncomputable def list_price : ℝ := 80
noncomputable def discount_factor : ℝ := 0.5
noncomputable def additional_discount : ℝ := 0.2
noncomputable def lowest_price := (list_price - discount_factor * list_price) - additional_discount * list_price
noncomputable def percent_of_list_price := (lowest_price / list_price) * 100

theorem lowest_possible_sale_price_is_30_percent :
  percent_of_list_price = 30 := by
  sorry

end lowest_possible_sale_price_is_30_percent_l1343_134308


namespace max_participants_won_at_least_three_matches_l1343_134311

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end max_participants_won_at_least_three_matches_l1343_134311


namespace reduction_in_jury_running_time_l1343_134363

def week1_miles : ℕ := 2
def week2_miles : ℕ := 2 * week1_miles + 3
def week3_miles : ℕ := (9 * week2_miles) / 7
def week4_miles : ℕ := 4

theorem reduction_in_jury_running_time : week3_miles - week4_miles = 5 :=
by
  -- sorry specifies the proof is skipped
  sorry

end reduction_in_jury_running_time_l1343_134363


namespace number_of_persons_l1343_134376

theorem number_of_persons (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
sorry

end number_of_persons_l1343_134376


namespace speed_of_man_in_still_water_l1343_134382

def upstream_speed := 34 -- in kmph
def downstream_speed := 48 -- in kmph

def speed_in_still_water := (upstream_speed + downstream_speed) / 2

theorem speed_of_man_in_still_water :
  speed_in_still_water = 41 := by
  sorry

end speed_of_man_in_still_water_l1343_134382


namespace correct_quotient_l1343_134383

def original_number : ℕ :=
  8 * 156 + 2

theorem correct_quotient :
  (8 * 156 + 2) / 5 = 250 :=
sorry

end correct_quotient_l1343_134383


namespace bananas_in_each_box_l1343_134324

theorem bananas_in_each_box 
    (bananas : ℕ) (boxes : ℕ) 
    (h_bananas : bananas = 40) 
    (h_boxes : boxes = 10) : 
    bananas / boxes = 4 := by
  sorry

end bananas_in_each_box_l1343_134324


namespace abc_relationship_l1343_134392

variable (x y : ℝ)

def parabola (x : ℝ) : ℝ :=
  x^2 + x + 2

def a := parabola 2
def b := parabola (-1)
def c := parabola 3

theorem abc_relationship : c > a ∧ a > b := by
  sorry

end abc_relationship_l1343_134392


namespace abc_solution_l1343_134348

theorem abc_solution (a b c : ℕ) (h1 : a + b = c - 1) (h2 : a^3 + b^3 = c^2 - 1) : 
  (a = 2 ∧ b = 3 ∧ c = 6) ∨ (a = 3 ∧ b = 2 ∧ c = 6) :=
sorry

end abc_solution_l1343_134348


namespace lunch_break_duration_l1343_134362

theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (∃ (p a : ℝ),
      (6 - L) * (p + a) = 0.4 ∧
      (4 - L) * a = 0.15 ∧
      (10 - L) * p = 0.45) ∧
    291 = L * 60 := 
by
  sorry

end lunch_break_duration_l1343_134362


namespace breadth_of_hall_l1343_134309

theorem breadth_of_hall (length_hall : ℝ) (stone_length_dm : ℝ) (stone_breadth_dm : ℝ)
    (num_stones : ℕ) (area_stone_m2 : ℝ) (total_area_m2 : ℝ) (breadth_hall : ℝ):
    length_hall = 36 → 
    stone_length_dm = 8 → 
    stone_breadth_dm = 5 → 
    num_stones = 1350 → 
    area_stone_m2 = (stone_length_dm * stone_breadth_dm) / 100 → 
    total_area_m2 = num_stones * area_stone_m2 → 
    breadth_hall = total_area_m2 / length_hall → 
    breadth_hall = 15 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4] at *
  simp [h5, h6, h7]
  sorry

end breadth_of_hall_l1343_134309


namespace mike_reaches_office_time_l1343_134393

-- Define the given conditions
def dave_steps_per_minute : ℕ := 80
def dave_step_length_cm : ℕ := 85
def dave_time_min : ℕ := 20

def mike_steps_per_minute : ℕ := 95
def mike_step_length_cm : ℕ := 70

-- Define Dave's walking speed
def dave_speed_cm_per_min : ℕ := dave_steps_per_minute * dave_step_length_cm

-- Define the total distance to the office
def distance_to_office_cm : ℕ := dave_speed_cm_per_min * dave_time_min

-- Define Mike's walking speed
def mike_speed_cm_per_min : ℕ := mike_steps_per_minute * mike_step_length_cm

-- Define the time it takes Mike to walk to the office
noncomputable def mike_time_to_office_min : ℚ := distance_to_office_cm / mike_speed_cm_per_min

-- State the theorem to prove
theorem mike_reaches_office_time :
  mike_time_to_office_min = 20.45 :=
sorry

end mike_reaches_office_time_l1343_134393


namespace solution_set_for_log_inequality_l1343_134302

noncomputable def log_base_0_1 (x: ℝ) : ℝ := Real.log x / Real.log 0.1

theorem solution_set_for_log_inequality :
  ∀ x : ℝ, (0 < x) → 
  log_base_0_1 (2^x - 1) < 0 ↔ x > 1 :=
by
  sorry

end solution_set_for_log_inequality_l1343_134302


namespace average_weight_increase_l1343_134316

theorem average_weight_increase (A : ℝ) (X : ℝ) (h : (8 * A - 65 + 93) / 8 = A + X) :
  X = 3.5 :=
sorry

end average_weight_increase_l1343_134316


namespace problem1_problem2_l1343_134347

-- Problem 1
theorem problem1 (a : ℝ) (h : a = Real.sqrt 3 - 1) : (a^2 + a) * (a + 1) / a = 3 := 
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h : a = 1 / 2) : (a + 1) / (a^2 - 1) - (a + 1) / (1 - a) = -5 := 
sorry

end problem1_problem2_l1343_134347


namespace consecutive_integer_sets_sum_27_l1343_134307

theorem consecutive_integer_sets_sum_27 :
  ∃! s : Set (List ℕ), ∀ l ∈ s, 
  (∃ n a, n ≥ 3 ∧ l = List.range n ++ [a] ∧ (List.sum l) = 27)
:=
sorry

end consecutive_integer_sets_sum_27_l1343_134307


namespace amount_spent_on_machinery_l1343_134326

-- Define the given conditions
def raw_materials_spent : ℤ := 80000
def total_amount : ℤ := 137500
def cash_spent : ℤ := (20 * total_amount) / 100

-- The goal is to prove the amount spent on machinery
theorem amount_spent_on_machinery : 
  ∃ M : ℤ, raw_materials_spent + M + cash_spent = total_amount ∧ M = 30000 := by
  sorry

end amount_spent_on_machinery_l1343_134326


namespace common_difference_in_arithmetic_sequence_l1343_134378

theorem common_difference_in_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 3)
  (h2 : a 5 = 12) :
  d = 3 :=
by
  sorry

end common_difference_in_arithmetic_sequence_l1343_134378


namespace simple_interest_correct_l1343_134340

def principal : ℝ := 400
def rate : ℝ := 0.20
def time : ℝ := 2

def simple_interest (P R T : ℝ) : ℝ := P * R * T

theorem simple_interest_correct :
  simple_interest principal rate time = 160 :=
by
  sorry

end simple_interest_correct_l1343_134340


namespace initial_percentage_is_30_l1343_134319

def percentage_alcohol (P : ℝ) : Prop :=
  let initial_alcohol := (P / 100) * 50
  let mixed_solution_volume := 50 + 30
  let final_percentage_alcohol := 18.75
  let final_alcohol := (final_percentage_alcohol / 100) * mixed_solution_volume
  initial_alcohol = final_alcohol

theorem initial_percentage_is_30 :
  percentage_alcohol 30 :=
by
  unfold percentage_alcohol
  sorry

end initial_percentage_is_30_l1343_134319


namespace find_y_of_set_with_mean_l1343_134321

theorem find_y_of_set_with_mean (y : ℝ) (h : ((8 + 15 + 20 + 6 + y) / 5 = 12)) : y = 11 := 
by 
    sorry

end find_y_of_set_with_mean_l1343_134321


namespace find_income_l1343_134341

def income_and_savings (x : ℕ) : ℕ := 10 * x
def expenditure (x : ℕ) : ℕ := 4 * x
def savings (x : ℕ) : ℕ := income_and_savings x - expenditure x

theorem find_income (savings_eq : 6 * 1900 = 11400) : income_and_savings 1900 = 19000 :=
by
  sorry

end find_income_l1343_134341


namespace incorrect_option_B_l1343_134354

noncomputable def Sn : ℕ → ℝ := sorry
-- S_n is the sum of the first n terms of the arithmetic sequence

axiom S5_S6 : Sn 5 < Sn 6
axiom S6_eq_S_gt_S8 : Sn 6 = Sn 7 ∧ Sn 7 > Sn 8

theorem incorrect_option_B : ¬ (Sn 9 < Sn 5) := sorry

end incorrect_option_B_l1343_134354


namespace smallest_multiple_of_2019_of_form_abcabcabc_l1343_134380

def is_digit (n : ℕ) : Prop := n < 10

theorem smallest_multiple_of_2019_of_form_abcabcabc
    (a b c : ℕ)
    (h_a : is_digit a)
    (h_b : is_digit b)
    (h_c : is_digit c)
    (k : ℕ)
    (form : Nat)
    (rep: ℕ) : 
  (form = (a * 100 + b * 10 + c) * rep) →
  (∃ n : ℕ, form = 2019 * n) →
  form >= 673673673 :=
sorry

end smallest_multiple_of_2019_of_form_abcabcabc_l1343_134380


namespace second_digging_breadth_l1343_134329

theorem second_digging_breadth :
  ∀ (A B depth1 length1 breadth1 depth2 length2 : ℕ),
  (A / B) = 1 → -- Assuming equal number of days and people
  depth1 = 100 → length1 = 25 → breadth1 = 30 → 
  depth2 = 75 → length2 = 20 → 
  (A = depth1 * length1 * breadth1) → 
  (B = depth2 * length2 * x) →
  x = 50 :=
by sorry

end second_digging_breadth_l1343_134329


namespace original_length_in_meters_l1343_134330

-- Conditions
def erased_length : ℝ := 10 -- 10 cm
def remaining_length : ℝ := 90 -- 90 cm

-- Question: What is the original length of the line in meters?
theorem original_length_in_meters : (remaining_length + erased_length) / 100 = 1 := 
by 
  -- The proof is omitted
  sorry

end original_length_in_meters_l1343_134330
