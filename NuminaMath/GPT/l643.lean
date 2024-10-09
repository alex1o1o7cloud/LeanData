import Mathlib

namespace correct_expression_l643_64365

theorem correct_expression (a b c : ℝ) : a - b + c = a - (b - c) :=
by
  sorry

end correct_expression_l643_64365


namespace find_three_numbers_l643_64329

theorem find_three_numbers (x y z : ℝ)
  (h1 : x - y = (1 / 3) * z)
  (h2 : y - z = (1 / 3) * x)
  (h3 : z - 10 = (1 / 3) * y) :
  x = 45 ∧ y = 37.5 ∧ z = 22.5 :=
by
  sorry

end find_three_numbers_l643_64329


namespace NoahClosetsFit_l643_64317

-- Declare the conditions as Lean variables and proofs
variable (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
variable (H1 : AliClosetCapacity = 200)
variable (H2 : NoahClosetsRatio = 1 / 4)
variable (H3 : NoahClosetsCount = 2)

-- Define the total number of jeans both of Noah's closets can fit
noncomputable def NoahTotalJeans : ℕ := (AliClosetCapacity * NoahClosetsRatio) * NoahClosetsCount

-- Theorem to prove
theorem NoahClosetsFit (AliClosetCapacity : ℕ) (NoahClosetsRatio : ℕ) (NoahClosetsCount : ℕ)
  (H1 : AliClosetCapacity = 200) 
  (H2 : NoahClosetsRatio = 1 / 4) 
  (H3 : NoahClosetsCount = 2) 
  : NoahTotalJeans AliClosetCapacity NoahClosetsRatio NoahClosetsCount = 100 := 
  by 
    sorry

end NoahClosetsFit_l643_64317


namespace quadratic_square_binomial_l643_64390

theorem quadratic_square_binomial (a r s : ℚ) (h1 : a = r^2) (h2 : 2 * r * s = 26) (h3 : s^2 = 9) :
  a = 169/9 := sorry

end quadratic_square_binomial_l643_64390


namespace linear_decreasing_sequence_l643_64336

theorem linear_decreasing_sequence 
  (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_func1 : y1 = -3 * x1 + 1)
  (h_func2 : y2 = -3 * x2 + 1)
  (h_func3 : y3 = -3 * x3 + 1)
  (hx_seq : x1 < x2 ∧ x2 < x3)
  : y3 < y2 ∧ y2 < y1 := 
sorry

end linear_decreasing_sequence_l643_64336


namespace temperature_on_last_day_l643_64383

noncomputable def last_day_temperature (T1 T2 T3 T4 T5 T6 T7 : ℕ) (mean : ℕ) : ℕ :=
  8 * mean - (T1 + T2 + T3 + T4 + T5 + T6 + T7)

theorem temperature_on_last_day 
  (T1 T2 T3 T4 T5 T6 T7 mean x : ℕ)
  (hT1 : T1 = 82) (hT2 : T2 = 80) (hT3 : T3 = 84) 
  (hT4 : T4 = 86) (hT5 : T5 = 88) (hT6 : T6 = 90) 
  (hT7 : T7 = 88) (hmean : mean = 86) 
  (hx : x = last_day_temperature T1 T2 T3 T4 T5 T6 T7 mean) :
  x = 90 := by
  sorry

end temperature_on_last_day_l643_64383


namespace g_6_eq_1_l643_64386

variable (f : ℝ → ℝ)

noncomputable def g (x : ℝ) := f x + 1 - x

theorem g_6_eq_1 
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1) :
  g f 6 = 1 :=
by
  sorry

end g_6_eq_1_l643_64386


namespace intervals_of_monotonicity_and_min_value_l643_64371

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 2

theorem intervals_of_monotonicity_and_min_value : 
  (∀ x, (x < -1 → f x < f (x + 0.0001)) ∧ (x > -1 ∧ x < 3 → f x > f (x + 0.0001)) ∧ (x > 3 → f x < f (x + 0.0001))) ∧
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≥ f 2) :=
by
  sorry

end intervals_of_monotonicity_and_min_value_l643_64371


namespace minimum_value_of_y_l643_64323

-- Define the function y
noncomputable def y (x : ℝ) := 2 + 4 * x + 1 / x

-- Prove that the minimum value is 6 for x > 0
theorem minimum_value_of_y : ∃ (x : ℝ), x > 0 ∧ (∀ (y : ℝ), (2 + 4 * x + 1 / x) ≤ y) ∧ (2 + 4 * x + 1 / x) = 6 := 
sorry

end minimum_value_of_y_l643_64323


namespace range_of_a_inequality_solution_set_l643_64387

noncomputable def quadratic_condition_holds (a : ℝ) : Prop :=
∀ (x : ℝ), x^2 - 2 * a * x + a > 0

theorem range_of_a (a : ℝ) (h : quadratic_condition_holds a) : 0 < a ∧ a < 1 := sorry

theorem inequality_solution_set (a x : ℝ) (h1 : 0 < a) (h2 : a < 1) : (a^(x^2 - 3) < a^(2 * x) ∧ a^(2 * x) < 1) ↔ x > 3 := sorry

end range_of_a_inequality_solution_set_l643_64387


namespace pradeep_maximum_marks_l643_64328

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.20 * M = 185) : M = 925 :=
by
  sorry

end pradeep_maximum_marks_l643_64328


namespace smallest_N_div_a3_possible_values_of_a3_l643_64395

-- Problem (a)
theorem smallest_N_div_a3 (a : Fin 10 → Nat) (h : StrictMono a) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) / (a 2) = 8 :=
sorry

-- Problem (b)
theorem possible_values_of_a3 (a : Nat) (h_a3_range : 1 ≤ a ∧ a ≤ 1000) :
  a = 315 ∨ a = 630 ∨ a = 945 :=
sorry

end smallest_N_div_a3_possible_values_of_a3_l643_64395


namespace fewer_sevens_l643_64341

def seven_representation (n : ℕ) : ℕ :=
  (7 * (10^n - 1)) / 9

theorem fewer_sevens (n : ℕ) :
  ∃ m, m < n ∧ 
    (∃ expr : ℕ → ℕ, (∀ i < n, expr i = 7) ∧ seven_representation n = expr m) :=
sorry

end fewer_sevens_l643_64341


namespace difference_of_squares_example_l643_64368

theorem difference_of_squares_example : 204^2 - 202^2 = 812 := by
  sorry

end difference_of_squares_example_l643_64368


namespace solve_floor_equation_l643_64375

theorem solve_floor_equation (x : ℝ) (hx : (∃ (y : ℤ), (x^3 - 40 * (y : ℝ) - 78 = 0) ∧ (y : ℝ) ≤ x ∧ x < (y + 1 : ℝ))) :
  x = -5.45 ∨ x = -4.96 ∨ x = -1.26 ∨ x = 6.83 ∨ x = 7.10 :=
by sorry

end solve_floor_equation_l643_64375


namespace greatest_possible_value_x_y_l643_64347

noncomputable def max_x_y : ℕ :=
  let s1 := 150
  let s2 := 210
  let s3 := 270
  let s4 := 330
  (3 * (s3 + s4) - (s1 + s2 + s3 + s4))

theorem greatest_possible_value_x_y :
  max_x_y = 840 := by
  sorry

end greatest_possible_value_x_y_l643_64347


namespace quadratic_roots_bc_minus_two_l643_64372

theorem quadratic_roots_bc_minus_two (b c : ℝ) 
  (h1 : 1 + -2 = -b) 
  (h2 : 1 * -2 = c) : b * c = -2 :=
by 
  sorry

end quadratic_roots_bc_minus_two_l643_64372


namespace geometric_sequence_ratio_l643_64304

noncomputable def geometric_sequence_pos (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a n > 0 ∧ a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h : geometric_sequence_pos a q) (h_q : q^2 = 4) :
  (a 2 + a 3) / (a 3 + a 4) = 1 / 2 :=
sorry

end geometric_sequence_ratio_l643_64304


namespace initial_mixture_volume_is_165_l643_64394

noncomputable def initial_volume_of_mixture (initial_milk_volume initial_water_volume water_added final_milk_water_ratio : ℕ) : ℕ :=
  if (initial_milk_volume + initial_water_volume) = 5 * (initial_milk_volume / 3) &&
     initial_water_volume = 2 * (initial_milk_volume / 3) &&
     water_added = 66 &&
     final_milk_water_ratio = 3 / 4 then
    5 * (initial_milk_volume / 3)
  else
    0

theorem initial_mixture_volume_is_165 :
  ∀ initial_milk_volume initial_water_volume water_added final_milk_water_ratio,
    initial_volume_of_mixture initial_milk_volume initial_water_volume water_added final_milk_water_ratio = 165 :=
by
  intros
  sorry

end initial_mixture_volume_is_165_l643_64394


namespace locus_of_point_P_l643_64315

theorem locus_of_point_P (P : ℝ × ℝ) (M N : ℝ × ℝ)
  (hxM : M = (-2, 0))
  (hxN : N = (2, 0))
  (hxPM : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPM)
  (hxPN : P.fst ^ 2 + (P.snd - 0) ^ 2 = xPN)
  : P.fst ^ 2 + P.snd ^ 2 = 4 ∧ P.fst ≠ 2 ∧ P.fst ≠ -2 :=
by
  -- proof omitted
  sorry

end locus_of_point_P_l643_64315


namespace rational_solution_unique_l643_64367

theorem rational_solution_unique
  (n : ℕ) (x y : ℚ)
  (hn : Odd n)
  (hx_eqn : x ^ n + 2 * y = y ^ n + 2 * x) :
  x = y :=
sorry

end rational_solution_unique_l643_64367


namespace pies_in_each_row_l643_64348

theorem pies_in_each_row (pecan_pies apple_pies rows : Nat) (hpecan : pecan_pies = 16) (happle : apple_pies = 14) (hrows : rows = 30) :
  (pecan_pies + apple_pies) / rows = 1 :=
by
  sorry

end pies_in_each_row_l643_64348


namespace iron_wire_left_l643_64384

-- Given conditions as variables
variable (initial_usage : ℚ) (additional_usage : ℚ)

-- Conditions as hypotheses
def conditions := initial_usage = 2 / 9 ∧ additional_usage = 3 / 9

-- The goal to prove
theorem iron_wire_left (h : conditions initial_usage additional_usage):
  1 - initial_usage - additional_usage = 4 / 9 :=
by
  -- Insert proof here
  sorry

end iron_wire_left_l643_64384


namespace cody_steps_l643_64305

theorem cody_steps (S steps_week1 steps_week2 steps_week3 steps_week4 total_steps_4weeks : ℕ) 
  (h1 : steps_week1 = 7 * S) 
  (h2 : steps_week2 = 7 * (S + 1000)) 
  (h3 : steps_week3 = 7 * (S + 2000)) 
  (h4 : steps_week4 = 7 * (S + 3000)) 
  (h5 : total_steps_4weeks = steps_week1 + steps_week2 + steps_week3 + steps_week4) 
  (h6 : total_steps_4weeks = 70000) : 
  S = 1000 := 
    sorry

end cody_steps_l643_64305


namespace find_n_l643_64351

theorem find_n (n : ℕ) (h : 7^(2*n) = (1/7)^(n-12)) : n = 4 :=
sorry

end find_n_l643_64351


namespace oliver_final_money_l643_64357

-- Define the initial conditions as variables and constants
def initial_amount : Nat := 9
def savings : Nat := 5
def earnings : Nat := 6
def spent_frisbee : Nat := 4
def spent_puzzle : Nat := 3
def spent_stickers : Nat := 2
def movie_ticket_price : Nat := 10
def movie_ticket_discount : Nat := 20 -- 20%
def snack_price : Nat := 3
def snack_discount : Nat := 1
def birthday_gift : Nat := 8

-- Define the final amount of money Oliver has left based on the problem statement
def final_amount : Nat :=
  let total_money := initial_amount + savings + earnings
  let total_spent := spent_frisbee + spent_puzzle + spent_stickers
  let remaining_after_spending := total_money - total_spent
  let discounted_movie_ticket := movie_ticket_price * (100 - movie_ticket_discount) / 100
  let discounted_snack := snack_price - snack_discount
  let total_spent_after_discounts := discounted_movie_ticket + discounted_snack
  let remaining_after_discounts := remaining_after_spending - total_spent_after_discounts
  remaining_after_discounts + birthday_gift

-- Lean theorem statement to prove that Oliver ends up with $9
theorem oliver_final_money : final_amount = 9 := by
  sorry

end oliver_final_money_l643_64357


namespace race_head_start_l643_64324

-- This statement defines the problem in Lean 4
theorem race_head_start (Va Vb L H : ℝ) 
(h₀ : Va = 51 / 44 * Vb) 
(h₁ : L / Va = (L - H) / Vb) : 
H = 7 / 51 * L := 
sorry

end race_head_start_l643_64324


namespace proof_inequality_l643_64369

theorem proof_inequality (n : ℕ) (a b : ℝ) (c : ℝ) (h_n : 1 ≤ n) (h_a : 1 ≤ a) (h_b : 1 ≤ b) (h_c : 0 < c) : 
  ((ab + c)^n - c) / ((b + c)^n - c) ≤ a^n :=
sorry

end proof_inequality_l643_64369


namespace odd_function_characterization_l643_64385

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.log (abs (a + 1 / (1 - x))) + b

theorem odd_function_characterization :
  (∀ x : ℝ, f (-a) (-b) (-x) = f a b x) →
  a = -1/2 ∧ b = Real.log 2 :=
by
  sorry

end odd_function_characterization_l643_64385


namespace chantel_final_bracelets_count_l643_64344

def bracelets_made_in_first_5_days : ℕ := 5 * 2

def bracelets_after_giving_away_at_school : ℕ := bracelets_made_in_first_5_days - 3

def bracelets_made_in_next_4_days : ℕ := 4 * 3

def total_bracelets_before_soccer_giveaway : ℕ := bracelets_after_giving_away_at_school + bracelets_made_in_next_4_days

def bracelets_after_giving_away_at_soccer : ℕ := total_bracelets_before_soccer_giveaway - 6

theorem chantel_final_bracelets_count : bracelets_after_giving_away_at_soccer = 13 :=
sorry

end chantel_final_bracelets_count_l643_64344


namespace boat_b_takes_less_time_l643_64353

theorem boat_b_takes_less_time (A_speed_still : ℝ) (B_speed_still : ℝ)
  (A_current : ℝ) (B_current : ℝ) (distance_downstream : ℝ)
  (A_speed_downstream : A_speed_still + A_current = 26)
  (B_speed_downstream : B_speed_still + B_current = 28)
  (A_time : A_speed_still + A_current = 26 → distance_downstream / (A_speed_still + A_current) = 4.6154)
  (B_time : B_speed_still + B_current = 28 → distance_downstream / (B_speed_still + B_current) = 4.2857) :
  distance_downstream / (B_speed_still + B_current) < distance_downstream / (A_speed_still + A_current) :=
by sorry

end boat_b_takes_less_time_l643_64353


namespace seven_pow_eight_mod_100_l643_64355

theorem seven_pow_eight_mod_100 :
  (7 ^ 8) % 100 = 1 := 
by {
  -- here can be the steps of the proof, but for now we use sorry
  sorry
}

end seven_pow_eight_mod_100_l643_64355


namespace distance_is_12_l643_64308

def distance_to_Mount_Overlook (D : ℝ) : Prop :=
  let T1 := D / 4
  let T2 := D / 6
  T1 + T2 = 5

theorem distance_is_12 : ∃ D : ℝ, distance_to_Mount_Overlook D ∧ D = 12 :=
by
  use 12
  rw [distance_to_Mount_Overlook]
  sorry

end distance_is_12_l643_64308


namespace find_m_range_l643_64389

noncomputable def f (x m : ℝ) : ℝ := x * abs (x - m) + 2 * x - 3

theorem find_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ m ≤ f x₂ m)
    ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end find_m_range_l643_64389


namespace find_a_find_A_l643_64376

-- Part (I)
theorem find_a (b c : ℝ) (A : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = 5 * Real.pi / 6) :
  ∃ a : ℝ, a = 2 * Real.sqrt 7 :=
by {
  sorry
}

-- Part (II)
theorem find_A (b c : ℝ) (C : ℝ) (hb : b = 2) (hc : c = 2 * Real.sqrt 3) (hC : C = Real.pi / 2 + A) :
  ∃ A : ℝ, A = Real.pi / 6 :=
by {
  sorry
}

end find_a_find_A_l643_64376


namespace university_admission_l643_64316

def students_ratio (x y z : ℕ) : Prop :=
  x * 5 = y * 2 ∧ y * 3 = z * 5

def third_tier_students : ℕ := 1500

theorem university_admission :
  ∀ x y z : ℕ, students_ratio x y z → z = third_tier_students → y - x = 1500 :=
by
  intros x y z hratio hthird
  sorry

end university_admission_l643_64316


namespace fraction_ratio_l643_64397

theorem fraction_ratio (x y a b : ℝ) (h1 : 4 * x - 3 * y = a) (h2 : 6 * y - 8 * x = b) (h3 : b ≠ 0) : a / b = -1 / 2 :=
by
  sorry

end fraction_ratio_l643_64397


namespace main_theorem_l643_64332

theorem main_theorem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * c * a) + c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end main_theorem_l643_64332


namespace no_valid_a_l643_64339

theorem no_valid_a : ¬ ∃ (a : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 
  ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ 2 * x₁^2 + (3 * a + 1) * x₁ + a^2 = 0 ∧ 2 * x₂^2 + (3 * a + 1) * x₂ + a^2 = 0 :=
by {
  sorry
}

end no_valid_a_l643_64339


namespace janet_earnings_per_hour_l643_64380

theorem janet_earnings_per_hour :
  let text_posts := 150
  let image_posts := 80
  let video_posts := 20
  let rate_text := 0.25
  let rate_image := 0.30
  let rate_video := 0.40
  text_posts * rate_text + image_posts * rate_image + video_posts * rate_video = 69.50 :=
by
  sorry

end janet_earnings_per_hour_l643_64380


namespace solution_set_A_solution_set_B_subset_A_l643_64319

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem solution_set_A :
  {x : ℝ | f x > 6} = {x : ℝ | x < -1 ∨ x > 2} :=
sorry

theorem solution_set_B_subset_A {a : ℝ} :
  (∀ x, f x > |a-1| → x < -1 ∨ x > 2) → a ≤ -5 ∨ a ≥ 7 :=
sorry

end solution_set_A_solution_set_B_subset_A_l643_64319


namespace coffee_last_days_l643_64374

theorem coffee_last_days (coffee_weight : ℕ) (cups_per_lb : ℕ) (angie_daily : ℕ) (bob_daily : ℕ) (carol_daily : ℕ) 
  (angie_coffee_weight : coffee_weight = 3) (cups_brewing_rate : cups_per_lb = 40)
  (angie_consumption : angie_daily = 3) (bob_consumption : bob_daily = 2) (carol_consumption : carol_daily = 4) : 
  ((coffee_weight * cups_per_lb) / (angie_daily + bob_daily + carol_daily) = 13) := by
  sorry

end coffee_last_days_l643_64374


namespace strictly_decreasing_interval_l643_64300

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem strictly_decreasing_interval :
  ∀ x, (0 < x) ∧ (x < 2) → (deriv f x < 0) := by
sorry

end strictly_decreasing_interval_l643_64300


namespace range_of_m_l643_64314

variable (m : ℝ)
def p := ∀ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 → x^2 - 2*x - 4*m^2 + 8*m - 2 ≥ 0
def q := ∃ x : ℝ, x ∈ Set.Icc (1 : ℝ) 2 ∧ Real.log (x^2 - m*x + 1) / Real.log (1/2) < -1

theorem range_of_m (hp : p m) (hq : q m) (hl : (p m) ∨ (q m)) (hf : ¬ ((p m) ∧ (q m))) :
  m < 1/2 ∨ m = 3/2 := sorry

end range_of_m_l643_64314


namespace geom_seq_product_l643_64392

def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  2 * a 3 - (a 8) ^ 2 + 2 * a 13 = 0

def geometric_seq (b : ℕ → ℤ) (a8 : ℤ) : Prop :=
  b 8 = a8

theorem geom_seq_product (a b : ℕ → ℤ) (a8 : ℤ) 
  (h1 : arithmetic_seq a)
  (h2 : geometric_seq b a8)
  (h3 : a8 = 4)
: b 4 * b 12 = 16 := sorry

end geom_seq_product_l643_64392


namespace johns_age_l643_64309

theorem johns_age (d j : ℕ) 
  (h1 : j = d - 30) 
  (h2 : j + d = 80) : 
  j = 25 :=
by
  sorry

end johns_age_l643_64309


namespace apartments_decrease_l643_64379

theorem apartments_decrease (p_initial e_initial p e q : ℕ) (h1: p_initial = 5) (h2: e_initial = 2) (h3: q = 1)
    (first_mod: p = p_initial - 2) (e_first_mod: e = e_initial + 3) (q_eq: q = 1)
    (second_mod: p = p - 2) (e_second_mod: e = e + 3) :
    p_initial * e_initial * q > p * e * q := by
  sorry

end apartments_decrease_l643_64379


namespace example_problem_l643_64325

variables (a b : ℕ)

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem example_problem (hcf_ab : HCF 385 180 = 30) (a_def: a = 385) (b_def: b = 180) :
  LCM 385 180 = 2310 := 
by
  sorry

end example_problem_l643_64325


namespace cara_cats_correct_l643_64321

def martha_cats_rats : ℕ := 3
def martha_cats_birds : ℕ := 7
def martha_cats_animals : ℕ := martha_cats_rats + martha_cats_birds

def cara_cats_animals : ℕ := 5 * martha_cats_animals - 3

theorem cara_cats_correct : cara_cats_animals = 47 :=
by
  -- Proof omitted
  -- Here's where the actual calculation steps would go, but we'll just use sorry for now.
  sorry

end cara_cats_correct_l643_64321


namespace discriminant_of_quadratic_polynomial_l643_64345

theorem discriminant_of_quadratic_polynomial :
  let a := 5
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ) 
  let Δ := b^2 - 4 * a * c
  Δ = (576/25 : ℚ) :=
by
  sorry

end discriminant_of_quadratic_polynomial_l643_64345


namespace total_amount_l643_64327

-- Declare the variables
variables (A B C : ℕ)

-- Introduce the conditions as hypotheses
theorem total_amount (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B = 290) : 
  A + B + C = 980 := 
by {
  sorry
}

end total_amount_l643_64327


namespace pyramid_distance_to_larger_cross_section_l643_64312

theorem pyramid_distance_to_larger_cross_section
  (A1 A2 : ℝ) (d : ℝ)
  (h : ℝ)
  (hA1 : A1 = 256 * Real.sqrt 2)
  (hA2 : A2 = 576 * Real.sqrt 2)
  (hd : d = 12)
  (h_ratio : (Real.sqrt (A1 / A2)) = 2 / 3) :
  h = 36 := 
  sorry

end pyramid_distance_to_larger_cross_section_l643_64312


namespace base_representation_l643_64360

theorem base_representation (b : ℕ) (h₁ : b^2 ≤ 125) (h₂ : 125 < b^3) :
  (∀ b, b = 12 → 125 % b % 2 = 1) → b = 12 := 
by
  sorry

end base_representation_l643_64360


namespace arrangement_of_digits_11250_l643_64398

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then
    1
  else
    n * factorial (n - 1)

def number_of_arrangements (digits : List ℕ) : ℕ :=
  let number_ends_in_0 := factorial 4 / factorial 2
  let number_ends_in_5 := 3 * (factorial 3 / factorial 2)
  number_ends_in_0 + number_ends_in_5

theorem arrangement_of_digits_11250 :
  number_of_arrangements [1, 1, 2, 5, 0] = 21 :=
by
  sorry

end arrangement_of_digits_11250_l643_64398


namespace log_sum_l643_64388

theorem log_sum : Real.logb 2 1 + Real.logb 3 9 = 2 := by
  sorry

end log_sum_l643_64388


namespace remaining_kibble_l643_64363

def starting_kibble : ℕ := 12
def mary_kibble_morning : ℕ := 1
def mary_kibble_evening : ℕ := 1
def frank_kibble_afternoon : ℕ := 1
def frank_kibble_late_evening : ℕ := 2 * frank_kibble_afternoon

theorem remaining_kibble : starting_kibble - (mary_kibble_morning + mary_kibble_evening + frank_kibble_afternoon + frank_kibble_late_evening) = 7 := by
  sorry

end remaining_kibble_l643_64363


namespace ababab_divisible_by_7_l643_64330

theorem ababab_divisible_by_7 (a b : ℕ) (ha : a < 10) (hb : b < 10) : (101010 * a + 10101 * b) % 7 = 0 :=
by sorry

end ababab_divisible_by_7_l643_64330


namespace div_expression_l643_64335

theorem div_expression : (124 : ℝ) / (8 + 14 * 3) = 2.48 := by
  sorry

end div_expression_l643_64335


namespace six_digit_number_l643_64343

theorem six_digit_number : ∃ x : ℕ, 100000 ≤ x ∧ x < 1000000 ∧ 3 * x = (x - 300000) * 10 + 3 ∧ x = 428571 :=
by
sorry

end six_digit_number_l643_64343


namespace find_initial_workers_l643_64340

-- Define the initial number of workers.
def initial_workers (W : ℕ) (A : ℕ) : Prop :=
  -- Condition 1: W workers can complete work A in 25 days.
  ( W * 25 = A )  ∧
  -- Condition 2: (W + 10) workers can complete work A in 15 days.
  ( (W + 10) * 15 = A )

-- The theorem states that given the conditions, the initial number of workers is 15.
theorem find_initial_workers {W A : ℕ} (h : initial_workers W A) : W = 15 :=
  sorry

end find_initial_workers_l643_64340


namespace minimizing_reciprocal_sum_l643_64301

theorem minimizing_reciprocal_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 30) :
  a = 10 ∧ b = 5 :=
by
  sorry

end minimizing_reciprocal_sum_l643_64301


namespace initial_marbles_l643_64381

theorem initial_marbles (M : ℕ) :
  (M - 5) % 3 = 0 ∧ M = 65 :=
by
  sorry

end initial_marbles_l643_64381


namespace area_of_sector_l643_64346

theorem area_of_sector (r l : ℝ) (h1 : l + 2 * r = 12) (h2 : l / r = 2) : (1 / 2) * l * r = 9 :=
by
  sorry

end area_of_sector_l643_64346


namespace geometric_sequence_a7_l643_64322

-- Define the geometric sequence
def geometic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Conditions
def a1 (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def a2a4 (a : ℕ → ℝ) : Prop :=
  a 2 * a 4 = 16

-- The statement to prove
theorem geometric_sequence_a7 (a : ℕ → ℝ) (h1 : a1 a) (h2 : a2a4 a) (gs : geometic_sequence a) :
  a 7 = 64 :=
by
  sorry

end geometric_sequence_a7_l643_64322


namespace total_notebooks_eq_216_l643_64349

theorem total_notebooks_eq_216 (n : ℕ) 
  (h1 : total_notebooks = n^2 + 20)
  (h2 : total_notebooks = (n + 1)^2 - 9) : 
  total_notebooks = 216 := 
by 
  sorry

end total_notebooks_eq_216_l643_64349


namespace area_of_sector_l643_64373

theorem area_of_sector (l : ℝ) (α : ℝ) (r : ℝ) (S : ℝ) 
  (h1 : l = 3)
  (h2 : α = 1)
  (h3 : l = α * r) : 
  S = 9 / 2 :=
by
  sorry

end area_of_sector_l643_64373


namespace find_amount_l643_64377

theorem find_amount (N : ℝ) (hN : N = 24) (A : ℝ) (hA : A = 0.6667 * N - 0.25 * N) : A = 10.0008 :=
by
  rw [hN] at hA
  sorry

end find_amount_l643_64377


namespace bike_average_speed_l643_64320

theorem bike_average_speed (distance time : ℕ)
    (h1 : distance = 48)
    (h2 : time = 6) :
    distance / time = 8 := 
  by
    sorry

end bike_average_speed_l643_64320


namespace domain_of_inverse_l643_64310

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x - 1) + 1

theorem domain_of_inverse :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y = f x) → (y ∈ Set.Icc (3/2) 3) :=
by
  sorry

end domain_of_inverse_l643_64310


namespace larger_triangle_perimeter_l643_64337

def is_similar (a b c : ℕ) (x y z : ℕ) : Prop :=
  x * c = z * a ∧
  x * c = z * b ∧
  y * c = z * a ∧
  y * c = z * c ∧
  a ≠ b ∧ c ≠ b

def is_isosceles (a b c : ℕ) : Prop :=
  a = b ∧ a ≠ c

theorem larger_triangle_perimeter (a b c x y z : ℕ) 
  (h1 : is_isosceles a b c) 
  (h2 : is_similar a b c x y z) 
  (h3 : c = 12) 
  (h4 : z = 36)
  (h5 : a = 7) 
  (h6 : b = 7) : 
  x + y + z = 78 :=
sorry

end larger_triangle_perimeter_l643_64337


namespace calculate_profit_l643_64306

def additional_cost (purchase_cost : ℕ) : ℕ := (purchase_cost * 20) / 100

def total_feeding_cost (purchase_cost : ℕ) : ℕ := purchase_cost + additional_cost purchase_cost

def total_cost (purchase_cost : ℕ) (feeding_cost : ℕ) : ℕ := purchase_cost + feeding_cost

def selling_price_per_cow (weight : ℕ) (price_per_pound : ℕ) : ℕ := weight * price_per_pound

def total_revenue (price_per_cow : ℕ) (number_of_cows : ℕ) : ℕ := price_per_cow * number_of_cows

def profit (revenue : ℕ) (total_cost : ℕ) : ℕ := revenue - total_cost

def purchase_cost : ℕ := 40000
def number_of_cows : ℕ := 100
def weight_per_cow : ℕ := 1000
def price_per_pound : ℕ := 2

-- The theorem to prove
theorem calculate_profit : 
  profit (total_revenue (selling_price_per_cow weight_per_cow price_per_pound) number_of_cows) 
         (total_cost purchase_cost (total_feeding_cost purchase_cost)) = 112000 := by
  sorry

end calculate_profit_l643_64306


namespace combined_solid_volume_l643_64326

open Real

noncomputable def volume_truncated_cone (R r h : ℝ) :=
  (1 / 3) * π * h * (R^2 + R * r + r^2)

noncomputable def volume_cylinder (r h : ℝ): ℝ :=
  π * r^2 * h

theorem combined_solid_volume :
  let R := 10
  let r := 3
  let h_cone := 8
  let h_cyl := 10
  volume_truncated_cone R r h_cone + volume_cylinder r h_cyl = (1382 * π) / 3 :=
  by
  sorry

end combined_solid_volume_l643_64326


namespace total_earnings_l643_64333

-- Define the constants and conditions.
def regular_hourly_rate : ℕ := 5
def overtime_hourly_rate : ℕ := 6
def regular_hours_per_week : ℕ := 40
def first_week_hours : ℕ := 44
def second_week_hours : ℕ := 48

-- Define the proof problem in Lean 4.
theorem total_earnings : (regular_hours_per_week * 2 * regular_hourly_rate + 
                         ((first_week_hours - regular_hours_per_week) + 
                          (second_week_hours - regular_hours_per_week)) * overtime_hourly_rate) = 472 := 
by 
  exact sorry -- Detailed proof steps would go here.

end total_earnings_l643_64333


namespace maximum_value_of_k_l643_64342

theorem maximum_value_of_k (x y k : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < k)
    (h4 : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) : k ≤ 1.5 :=
by
  sorry

end maximum_value_of_k_l643_64342


namespace trapezoid_height_l643_64338

theorem trapezoid_height (A : ℝ) (d1 d2 : ℝ) (h : ℝ) :
  A = 2 ∧ d1 + d2 = 4 → h = Real.sqrt 2 :=
by
  sorry

end trapezoid_height_l643_64338


namespace Aarti_work_days_l643_64318

theorem Aarti_work_days (x : ℕ) : (3 * x = 24) → x = 8 := by
  intro h
  linarith

end Aarti_work_days_l643_64318


namespace calculate_expr_l643_64350

theorem calculate_expr :
  ( (5 / 12: ℝ) ^ 2022) * (-2.4) ^ 2023 = - (12 / 5: ℝ) := 
by 
  sorry

end calculate_expr_l643_64350


namespace solution_set_nonempty_implies_a_range_l643_64366

theorem solution_set_nonempty_implies_a_range (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end solution_set_nonempty_implies_a_range_l643_64366


namespace find_function_l643_64356

theorem find_function (f : ℕ → ℕ) (h : ∀ m n, f (m + f n) = f (f m) + f n) :
  ∃ d, d > 0 ∧ (∀ m, ∃ k, f m = k * d) :=
sorry

end find_function_l643_64356


namespace discount_savings_l643_64361

theorem discount_savings (initial_price discounted_price : ℝ)
  (h_initial : initial_price = 475)
  (h_discounted : discounted_price = 199) :
  initial_price - discounted_price = 276 :=
by
  rw [h_initial, h_discounted]
  sorry

end discount_savings_l643_64361


namespace negative_integers_abs_le_4_l643_64362

theorem negative_integers_abs_le_4 (x : Int) (h1 : x < 0) (h2 : abs x ≤ 4) : 
  x = -1 ∨ x = -2 ∨ x = -3 ∨ x = -4 :=
by
  sorry

end negative_integers_abs_le_4_l643_64362


namespace find_first_train_length_l643_64313

theorem find_first_train_length
  (length_second_train : ℝ)
  (initial_distance : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (time_minutes : ℝ) :
  length_second_train = 200 →
  initial_distance = 100 →
  speed_first_train_kmph = 54 →
  speed_second_train_kmph = 72 →
  time_minutes = 2.856914303998537 →
  ∃ (L : ℝ), L = 5699.52 :=
by
  sorry

end find_first_train_length_l643_64313


namespace cheryl_used_total_amount_l643_64354

theorem cheryl_used_total_amount :
  let bought_A := (5 / 8 : ℚ)
  let bought_B := (2 / 9 : ℚ)
  let bought_C := (2 / 5 : ℚ)
  let leftover_A := (1 / 12 : ℚ)
  let leftover_B := (5 / 36 : ℚ)
  let leftover_C := (1 / 10 : ℚ)
  let used_A := bought_A - leftover_A
  let used_B := bought_B - leftover_B
  let used_C := bought_C - leftover_C
  used_A + used_B + used_C = 37 / 40 :=
by 
  sorry

end cheryl_used_total_amount_l643_64354


namespace negation_of_universal_proposition_l643_64396
open Classical

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 > 0) → ∃ x : ℝ, ¬(x^2 > 0) :=
by
  intro h
  have := (not_forall.mp h)
  exact this

end negation_of_universal_proposition_l643_64396


namespace sqrt_range_l643_64378

theorem sqrt_range (a : ℝ) : 2 * a - 1 ≥ 0 ↔ a ≥ 1 / 2 :=
by sorry

end sqrt_range_l643_64378


namespace least_value_a2000_l643_64311

theorem least_value_a2000 (a : ℕ → ℕ)
  (h1 : ∀ m n, (m ∣ n) → (m < n) → (a m ∣ a n))
  (h2 : ∀ m n, (m ∣ n) → (m < n) → (a m < a n)) :
  a 2000 >= 128 :=
sorry

end least_value_a2000_l643_64311


namespace determine_constant_l643_64331

theorem determine_constant (c : ℝ) :
  (∃ d : ℝ, 9 * x^2 - 24 * x + c = (3 * x + d)^2) ↔ c = 16 :=
by
  sorry

end determine_constant_l643_64331


namespace area_of_region_l643_64393

-- Define the equation as a predicate
def region (x y : ℝ) : Prop := x^2 + y^2 + 6*x = 2*y + 10

-- The proof statement
theorem area_of_region : (∃ (x y : ℝ), region x y) → ∃ A : ℝ, A = 20 * Real.pi :=
by 
  sorry

end area_of_region_l643_64393


namespace gain_percent_correct_l643_64358

theorem gain_percent_correct (C S : ℝ) (h : 50 * C = 28 * S) : 
  ( (S - C) / C ) * 100 = 1100 / 14 :=
by
  sorry

end gain_percent_correct_l643_64358


namespace perimeter_of_plot_l643_64334

theorem perimeter_of_plot
  (width : ℝ) 
  (cost_per_meter : ℝ)
  (total_cost : ℝ)
  (h1 : cost_per_meter = 6.5)
  (h2 : total_cost = 1170)
  (h3 : total_cost = (2 * (width + (width + 10))) * cost_per_meter) 
  :
  (2 * ((width + 10) + width)) = 180 :=
by
  sorry

end perimeter_of_plot_l643_64334


namespace teddy_bears_ordered_l643_64364

theorem teddy_bears_ordered (days : ℕ) (T : ℕ)
  (h1 : 20 * days + 100 = T)
  (h2 : 23 * days - 20 = T) :
  T = 900 ∧ days = 40 := 
by 
  sorry

end teddy_bears_ordered_l643_64364


namespace range_of_a_l643_64302

noncomputable def f (x a : ℝ) := 2^(2*x) - a * 2^x + 4

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a ≤ 4 :=
by
  sorry

end range_of_a_l643_64302


namespace magnitude_of_c_is_correct_l643_64370

noncomputable def a : ℝ × ℝ := (2, 4)
noncomputable def b : ℝ × ℝ := (-1, 2)
noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2
noncomputable def c : ℝ × ℝ := (a.1 - (dot_product a b) * b.1, a.2 - (dot_product a b) * b.2)

noncomputable def magnitude (u : ℝ × ℝ) : ℝ :=
  Real.sqrt ((u.1 ^ 2) + (u.2 ^ 2))

theorem magnitude_of_c_is_correct :
  magnitude c = 8 * Real.sqrt 2 :=
by
  sorry

end magnitude_of_c_is_correct_l643_64370


namespace farmer_children_l643_64382

noncomputable def numberOfChildren 
  (totalLeft : ℕ)
  (eachChildCollected : ℕ)
  (eatenApples : ℕ)
  (soldApples : ℕ) : ℕ :=
  let totalApplesEaten := eatenApples * 2
  let initialCollection := eachChildCollected * (totalLeft + totalApplesEaten + soldApples) / eachChildCollected
  initialCollection / eachChildCollected

theorem farmer_children (totalLeft : ℕ) (eachChildCollected : ℕ) (eatenApples : ℕ) (soldApples : ℕ) : 
  totalLeft = 60 → eachChildCollected = 15 → eatenApples = 4 → soldApples = 7 → 
  numberOfChildren totalLeft eachChildCollected eatenApples soldApples = 5 := 
by
  intro h_totalLeft h_eachChildCollected h_eatenApples h_soldApples
  unfold numberOfChildren
  simp
  sorry

end farmer_children_l643_64382


namespace sequence_root_formula_l643_64359

theorem sequence_root_formula {a : ℕ → ℝ} 
    (h1 : ∀ n, (a (n + 1))^2 = (a n)^2 + 4)
    (h2 : a 1 = 1)
    (h3 : ∀ n, a n > 0) :
    ∀ n, a n = Real.sqrt (4 * n - 3) := 
sorry

end sequence_root_formula_l643_64359


namespace squared_diagonal_inequality_l643_64391

theorem squared_diagonal_inequality 
  (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) :
  let AB := (x1 - x2)^2 + (y1 - y2)^2
  let BC := (x2 - x3)^2 + (y2 - y3)^2
  let CD := (x3 - x4)^2 + (y3 - y4)^2
  let DA := (x1 - x4)^2 + (y1 - y4)^2
  let AC := (x1 - x3)^2 + (y1 - y3)^2
  let BD := (x2 - x4)^2 + (y2 - y4)^2
  AC + BD ≤ AB + BC + CD + DA := 
by
  sorry

end squared_diagonal_inequality_l643_64391


namespace find_base_l643_64399

theorem find_base (r : ℕ) (h1 : 5 * r^2 + 3 * r + 4 + 3 * r^2 + 6 * r + 6 = r^3) : r = 10 :=
by
  sorry

end find_base_l643_64399


namespace fraction_sum_l643_64303

theorem fraction_sum :
  (7 : ℚ) / 12 + (3 : ℚ) / 8 = 23 / 24 :=
by
  -- Proof is omitted
  sorry

end fraction_sum_l643_64303


namespace number_of_scoops_l643_64307

/-- Pierre gets 3 scoops of ice cream given the conditions described -/
theorem number_of_scoops (P : ℕ) (cost_per_scoop total_bill : ℝ) (mom_scoops : ℕ)
  (h1 : cost_per_scoop = 2) 
  (h2 : mom_scoops = 4) 
  (h3 : total_bill = 14) 
  (h4 : cost_per_scoop * P + cost_per_scoop * mom_scoops = total_bill) :
  P = 3 :=
by
  sorry

end number_of_scoops_l643_64307


namespace smallest_c_ineq_l643_64352

noncomputable def smallest_c {d : ℕ → ℕ} (h_d : ∀ n > 0, d n ≤ d n + 1) := Real.sqrt 3

theorem smallest_c_ineq (d : ℕ → ℕ) (h_d : ∀ n > 0, (d n) ≤ d n + 1) :
  ∀ n : ℕ, n > 0 → d n ≤ smallest_c h_d * (Real.sqrt n) :=
sorry

end smallest_c_ineq_l643_64352
