import Mathlib

namespace proof_problem_l463_46399

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := 2 * x + 4

theorem proof_problem : (f (g 3))^2 - (g (f 3))^2 = 28 := by
  sorry

end proof_problem_l463_46399


namespace compound_interest_principal_l463_46300

theorem compound_interest_principal (CI t : ℝ) (r n : ℝ) (P : ℝ) : CI = 630 ∧ t = 2 ∧ r = 0.10 ∧ n = 1 → P = 3000 :=
by
  -- Proof to be provided
  sorry

end compound_interest_principal_l463_46300


namespace no_member_of_T_is_divisible_by_4_or_5_l463_46365

def sum_of_squares_of_four_consecutive_integers (n : ℤ) : ℤ :=
  (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2

theorem no_member_of_T_is_divisible_by_4_or_5 :
  ∀ (n : ℤ), ¬ (∃ (T : ℤ), T = sum_of_squares_of_four_consecutive_integers n ∧ (T % 4 = 0 ∨ T % 5 = 0)) :=
by
  sorry

end no_member_of_T_is_divisible_by_4_or_5_l463_46365


namespace books_not_sold_l463_46304

-- Definitions capturing the conditions
variable (B : ℕ)
variable (books_price : ℝ := 3.50)
variable (total_received : ℝ := 252)

-- Lean statement to capture the proof problem
theorem books_not_sold (h : (2 / 3 : ℝ) * B * books_price = total_received) :
  B / 3 = 36 :=
by
  sorry

end books_not_sold_l463_46304


namespace trajectory_of_point_P_l463_46301

open Real

theorem trajectory_of_point_P (a : ℝ) (ha : a > 0) :
  (∀ x y : ℝ, (a = 1 → x = 0) ∧ 
    (a ≠ 1 → (x - (a^2 + 1) / (a^2 - 1))^2 + y^2 = 4 * a^2 / (a^2 - 1)^2)) := 
by 
  sorry

end trajectory_of_point_P_l463_46301


namespace lcm_gcd_48_180_l463_46383

theorem lcm_gcd_48_180 :
  Nat.lcm 48 180 = 720 ∧ Nat.gcd 48 180 = 12 :=
by
  sorry

end lcm_gcd_48_180_l463_46383


namespace transaction_loss_l463_46382

theorem transaction_loss 
  (sell_price_house sell_price_store : ℝ)
  (cost_price_house cost_price_store : ℝ)
  (house_loss_percent store_gain_percent : ℝ)
  (house_loss_eq : sell_price_house = (4/5) * cost_price_house)
  (store_gain_eq : sell_price_store = (6/5) * cost_price_store)
  (sell_prices_eq : sell_price_house = 12000 ∧ sell_price_store = 12000)
  (house_loss_percent_eq : house_loss_percent = 0.20)
  (store_gain_percent_eq : store_gain_percent = 0.20) :
  cost_price_house + cost_price_store - (sell_price_house + sell_price_store) = 1000 :=
by
  sorry

end transaction_loss_l463_46382


namespace range_of_k_l463_46385

noncomputable def point_satisfies_curve (a k : ℝ) : Prop :=
(-a)^2 - a * (-a) + 2 * a + k = 0

theorem range_of_k (a k : ℝ) (h : point_satisfies_curve a k) : k ≤ 1 / 2 :=
by
  sorry

end range_of_k_l463_46385


namespace probability_at_least_one_black_ball_l463_46363

def total_balls : ℕ := 10
def red_balls : ℕ := 6
def black_balls : ℕ := 4
def selected_balls : ℕ := 4

theorem probability_at_least_one_black_ball :
  (∃ (p : ℚ), p = 13 / 14 ∧ 
  (number_of_ways_to_choose4_balls_has_at_least_1_black / number_of_ways_to_choose4_balls) = p) :=
by
  sorry

end probability_at_least_one_black_ball_l463_46363


namespace max_product_of_three_numbers_l463_46366

theorem max_product_of_three_numbers (n : ℕ) (h_n_pos : 0 < n) :
  ∃ a b c : ℕ, (a + b + c = 3 * n + 1) ∧ (∀ a' b' c' : ℕ,
        (a' + b' + c' = 3 * n + 1) →
        a' * b' * c' ≤ a * b * c) ∧
    (a * b * c = n^3 + n^2) :=
by
  sorry

end max_product_of_three_numbers_l463_46366


namespace no_integer_a_for_integer_roots_l463_46305

theorem no_integer_a_for_integer_roots :
  ∀ a : ℤ, ¬ (∃ x : ℤ, x^2 - 2023 * x + 2022 * a + 1 = 0) := 
by
  intro a
  rintro ⟨x, hx⟩
  sorry

end no_integer_a_for_integer_roots_l463_46305


namespace train_b_leaves_after_train_a_l463_46306

noncomputable def time_difference := 2

theorem train_b_leaves_after_train_a 
  (speedA speedB distance t : ℝ) 
  (h1 : speedA = 30)
  (h2 : speedB = 38)
  (h3 : distance = 285)
  (h4 : distance = speedB * t)
  : time_difference = (distance - speedA * t) / speedA := 
by 
  sorry

end train_b_leaves_after_train_a_l463_46306


namespace lemonade_served_l463_46347

def glasses_per_pitcher : ℕ := 5
def number_of_pitchers : ℕ := 6
def total_glasses_served : ℕ := glasses_per_pitcher * number_of_pitchers

theorem lemonade_served : total_glasses_served = 30 :=
by
  -- proof goes here
  sorry

end lemonade_served_l463_46347


namespace num_students_left_l463_46308

variable (Joe_weight : ℝ := 45)
variable (original_avg_weight : ℝ := 30)
variable (new_avg_weight : ℝ := 31)
variable (final_avg_weight : ℝ := 30)
variable (diff_avg_weight : ℝ := 7.5)

theorem num_students_left (n : ℕ) (x : ℕ) (W : ℝ := n * original_avg_weight)
  (new_W : ℝ := W + Joe_weight) (A : ℝ := Joe_weight - diff_avg_weight) : 
  new_W = (n + 1) * new_avg_weight →
  W + Joe_weight - x * A = (n + 1 - x) * final_avg_weight →
  x = 2 :=
by
  sorry

end num_students_left_l463_46308


namespace sum_of_sequences_l463_46388

noncomputable def arithmetic_sequence (a b : ℤ) : Prop :=
  ∃ k : ℤ, a = 6 + k ∧ b = 6 + 2 * k

noncomputable def geometric_sequence (c d : ℤ) : Prop :=
  ∃ q : ℤ, c = 6 * q ∧ d = 6 * q^2

theorem sum_of_sequences (a b c d : ℤ) 
  (h_arith : arithmetic_sequence a b) 
  (h_geom : geometric_sequence c d) 
  (hb : b = 48) (hd : 6 * c^2 = 48): 
  a + b + c + d = 111 := 
sorry

end sum_of_sequences_l463_46388


namespace projected_percent_increase_l463_46372

theorem projected_percent_increase (R : ℝ) (p : ℝ) 
  (h1 : 0.7 * R = R * 0.7) 
  (h2 : 0.7 * R = 0.5 * (R + p * R)) : 
  p = 0.4 :=
by
  sorry

end projected_percent_increase_l463_46372


namespace trains_speed_ratio_l463_46359

-- Define the conditions
variables (V1 V2 L1 L2 : ℝ)
axiom time1 : L1 = 27 * V1
axiom time2 : L2 = 17 * V2
axiom timeTogether : L1 + L2 = 22 * (V1 + V2)

-- The theorem to prove the ratio of the speeds
theorem trains_speed_ratio : V1 / V2 = 7.8 :=
sorry

end trains_speed_ratio_l463_46359


namespace sides_and_diagonals_l463_46336

def number_of_sides_of_polygon (n : ℕ) :=
  180 * (n - 2) = 360 + (1 / 4 : ℤ) * 360

def number_of_diagonals_of_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem sides_and_diagonals : 
  (∃ n : ℕ, number_of_sides_of_polygon n ∧ n = 12) ∧ number_of_diagonals_of_polygon 12 = 54 :=
by {
  -- Proof will be filled in later
  sorry
}

end sides_and_diagonals_l463_46336


namespace ratio_of_b_to_a_l463_46373

variable (V A B : ℝ)

def ten_pours_of_a_cup : Prop := 10 * A = V
def five_pours_of_b_cup : Prop := 5 * B = V

theorem ratio_of_b_to_a (h1 : ten_pours_of_a_cup V A) (h2 : five_pours_of_b_cup V B) : B / A = 2 :=
sorry

end ratio_of_b_to_a_l463_46373


namespace no_solution_ineq_range_a_l463_46326

theorem no_solution_ineq_range_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end no_solution_ineq_range_a_l463_46326


namespace limes_given_l463_46362

theorem limes_given (original_limes now_limes : ℕ) (h1 : original_limes = 9) (h2 : now_limes = 5) : (original_limes - now_limes = 4) := 
by
  sorry

end limes_given_l463_46362


namespace maximize_net_income_l463_46314

noncomputable def net_income (x : ℕ) : ℤ :=
  if 60 ≤ x ∧ x ≤ 90 then 750 * x - 1700
  else if 90 < x ∧ x ≤ 300 then -3 * x * x + 1020 * x - 1700
  else 0

theorem maximize_net_income :
  (∀ x : ℕ, 60 ≤ x ∧ x ≤ 300 →
    net_income x ≤ net_income 170) ∧
  net_income 170 = 85000 := 
sorry

end maximize_net_income_l463_46314


namespace cos_half_angle_l463_46390

theorem cos_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < Real.pi / 2) : 
    Real.cos (α / 2) = 2 * Real.sqrt 5 / 5 := 
by 
    sorry

end cos_half_angle_l463_46390


namespace common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l463_46311

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_eqn :
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - 2*y + 4 = 0) :=
sorry

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (0, 2)
noncomputable def line_y_eq_neg_x (x y : ℝ) : Prop := y = -x

theorem circle_with_center_on_line :
  ∃ (x y : ℝ), line_y_eq_neg_x x y ∧ ((x + 3)^2 + (y - 3)^2 = 10) :=
sorry

theorem smallest_area_circle :
  ∃ (x y : ℝ), ((x + 2)^2 + (y - 1)^2 = 5) :=
sorry

end common_chord_eqn_circle_with_center_on_line_smallest_area_circle_l463_46311


namespace geometric_progression_first_term_l463_46387

theorem geometric_progression_first_term (a r : ℝ) 
    (h_sum_inf : a / (1 - r) = 8)
    (h_sum_two : a * (1 + r) = 5) :
    a = 2 * (4 - Real.sqrt 6) ∨ a = 2 * (4 + Real.sqrt 6) :=
sorry

end geometric_progression_first_term_l463_46387


namespace perimeter_of_figure_l463_46371

theorem perimeter_of_figure (x : ℕ) (h : x = 3) : 
  let sides := [x, x + 1, 6, 10]
  (sides.sum = 23) := by 
  sorry

end perimeter_of_figure_l463_46371


namespace initial_deadlift_weight_l463_46391

theorem initial_deadlift_weight
    (initial_squat : ℕ := 700)
    (initial_bench : ℕ := 400)
    (D : ℕ)
    (squat_loss : ℕ := 30)
    (deadlift_loss : ℕ := 200)
    (new_total : ℕ := 1490) :
    (initial_squat * (100 - squat_loss) / 100) + initial_bench + (D - deadlift_loss) = new_total → D = 800 :=
by
  sorry

end initial_deadlift_weight_l463_46391


namespace distance_midpoint_AB_to_y_axis_l463_46396

def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }

variable (A B : parabola)
variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)

open scoped Classical

noncomputable def midpoint_x (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_midpoint_AB_to_y_axis 
  (h1 : x1 + x2 = 3) 
  (hA : A.val = (x1, y1))
  (hB : B.val = (x2, y2)) : 
  midpoint_x x1 x2 = 3 / 2 := 
by
  sorry

end distance_midpoint_AB_to_y_axis_l463_46396


namespace circle_radius_is_2_chord_length_is_2sqrt3_l463_46350

-- Define the given conditions
def inclination_angle_line_incl60 : Prop := ∃ m, m = Real.sqrt 3
def circle_eq : Prop := ∀ x y, x^2 + y^2 - 4 * y = 0

-- Prove: radius of the circle
theorem circle_radius_is_2 (h : circle_eq) : radius = 2 := sorry

-- Prove: length of the chord cut by the line
theorem chord_length_is_2sqrt3 
  (h1 : inclination_angle_line_incl60) 
  (h2 : circle_eq) : chord_length = 2 * Real.sqrt 3 := sorry

end circle_radius_is_2_chord_length_is_2sqrt3_l463_46350


namespace chipmunk_acorns_l463_46348

-- Define the conditions and goal for the proof
theorem chipmunk_acorns :
  ∃ x : ℕ, (∀ h_c h_s : ℕ, h_c = h_s + 4 → 3 * h_c = x ∧ 4 * h_s = x) → x = 48 :=
by {
  -- We assume the problem conditions as given
  sorry
}

end chipmunk_acorns_l463_46348


namespace longest_segment_in_cylinder_l463_46395

noncomputable def cylinder_diagonal (radius height : ℝ) : ℝ :=
  Real.sqrt (height^2 + (2 * radius)^2)

theorem longest_segment_in_cylinder :
  cylinder_diagonal 4 10 = 2 * Real.sqrt 41 :=
by
  -- Proof placeholder
  sorry

end longest_segment_in_cylinder_l463_46395


namespace inequality_neg_reciprocal_l463_46330

theorem inequality_neg_reciprocal (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  - (1 / a) < - (1 / b) :=
sorry

end inequality_neg_reciprocal_l463_46330


namespace minimum_students_to_share_birthday_l463_46328

theorem minimum_students_to_share_birthday (k : ℕ) (m : ℕ) (n : ℕ) (hcond1 : k = 366) (hcond2 : m = 2) (hineq : n > k * m) : n ≥ 733 := 
by
  -- since k = 366 and m = 2
  have hk : k = 366 := hcond1
  have hm : m = 2 := hcond2
  -- thus: n > 366 * 2
  have hn : n > 732 := by
    rw [hk, hm] at hineq
    exact hineq
  -- hence, n ≥ 733
  exact Nat.succ_le_of_lt hn

end minimum_students_to_share_birthday_l463_46328


namespace no_solution_exists_l463_46312

theorem no_solution_exists (x y : ℝ) : ¬ ((2 * x - 3 * y = 8) ∧ (6 * y - 4 * x = 9)) :=
sorry

end no_solution_exists_l463_46312


namespace adam_age_is_8_l463_46394

variables (A : ℕ) -- Adam's current age
variable (tom_age : ℕ) -- Tom's current age
variable (combined_age : ℕ) -- Their combined age in 12 years

theorem adam_age_is_8 (h1 : tom_age = 12) -- Tom is currently 12 years old
                    (h2 : combined_age = 44) -- In 12 years, their combined age will be 44 years old
                    (h3 : A + 12 + (tom_age + 12) = combined_age) -- Equation representing the combined age in 12 years
                    : A = 8 :=
by
  sorry

end adam_age_is_8_l463_46394


namespace claire_gift_card_balance_l463_46340

/--
Claire has a $100 gift card to her favorite coffee shop.
A latte costs $3.75.
A croissant costs $3.50.
Claire buys one latte and one croissant every day for a week.
Claire buys 5 cookies, each costing $1.25.

Prove that the amount of money left on Claire's gift card after a week is $43.00.
-/
theorem claire_gift_card_balance :
  let initial_balance : ℝ := 100
  let latte_cost : ℝ := 3.75
  let croissant_cost : ℝ := 3.50
  let daily_expense : ℝ := latte_cost + croissant_cost
  let weekly_expense : ℝ := daily_expense * 7
  let cookie_cost : ℝ := 1.25
  let total_cookie_expense : ℝ := cookie_cost * 5
  let total_expense : ℝ := weekly_expense + total_cookie_expense
  let remaining_balance : ℝ := initial_balance - total_expense
  remaining_balance = 43 :=
by
  sorry

end claire_gift_card_balance_l463_46340


namespace find_a_l463_46322

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem find_a (a : ℝ) (h : f a (f a 0) = 4 * a) : a = 2 := by
  sorry

end find_a_l463_46322


namespace fraction_comparison_l463_46320

noncomputable def one_seventh : ℚ := 1 / 7
noncomputable def decimal_0_point_14285714285 : ℚ := 14285714285 / 10^11
noncomputable def eps_1 : ℚ := 1 / (7 * 10^11)
noncomputable def eps_2 : ℚ := 1 / (7 * 10^12)

theorem fraction_comparison :
  one_seventh = decimal_0_point_14285714285 + eps_1 :=
sorry

end fraction_comparison_l463_46320


namespace intersection_eq_T_l463_46331

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l463_46331


namespace kelly_chickens_l463_46307

theorem kelly_chickens
  (chicken_egg_rate : ℕ)
  (chickens : ℕ)
  (egg_price_per_dozen : ℕ)
  (total_money : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (dozen : ℕ)
  (total_eggs_sold : ℕ)
  (total_days : ℕ)
  (total_eggs_laid : ℕ) : 
  chicken_egg_rate = 3 →
  egg_price_per_dozen = 5 →
  total_money = 280 →
  weeks = 4 →
  days_per_week = 7 →
  dozen = 12 →
  total_eggs_sold = total_money / egg_price_per_dozen * dozen →
  total_days = weeks * days_per_week →
  total_eggs_laid = chickens * chicken_egg_rate * total_days →
  total_eggs_sold = total_eggs_laid →
  chickens = 8 :=
by
  intros
  sorry

end kelly_chickens_l463_46307


namespace solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l463_46355

-- Define the function f(x) based on the given conditions
def f (x k : ℝ) : ℝ := abs (x ^ 2 - 1) + x ^ 2 + k * x

-- Statement 1
theorem solve_f_zero_k_eq_2 :
  (∀ x : ℝ, f x 2 = 0 ↔ x = - (1 + Real.sqrt 3) / 2 ∨ x = -1 / 2) :=
sorry

-- Statement 2
theorem find_k_range_has_two_zeros (α β : ℝ) (hαβ : 0 < α ∧ α < β ∧ β < 2) :
  (∃ k : ℝ, f α k = 0 ∧ f β k = 0) ↔ - 7 / 2 < k ∧ k < -1 :=
sorry

-- Statement 3
theorem sum_of_reciprocals (α β : ℝ) (hαβ : 0 < α ∧ α < 1 ∧ 1 < β ∧ β < 2)
    (hα : f α (-1/α) = 0) (hβ : ∃ k : ℝ, f β k = 0) :
  (1 / α + 1 / β < 4) :=
sorry

end solve_f_zero_k_eq_2_find_k_range_has_two_zeros_sum_of_reciprocals_l463_46355


namespace accuracy_l463_46386

-- Given number and accuracy statement
def given_number : ℝ := 3.145 * 10^8
def expanded_form : ℕ := 314500000

-- Proof statement: the number is accurate to the hundred thousand's place
theorem accuracy (h : given_number = expanded_form) : 
  ∃ n : ℕ, expanded_form = n * 10^5 ∧ (n % 10) ≠ 0 := 
by
  sorry

end accuracy_l463_46386


namespace Tn_lt_half_Sn_l463_46360

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l463_46360


namespace family_members_to_pay_l463_46351

theorem family_members_to_pay :
  (∃ (n : ℕ), 
    5 * 12 = 60 ∧ 
    60 * 2 = 120 ∧ 
    120 / 10 = 12 ∧ 
    12 * 2 = 24 ∧ 
    24 / 4 = n ∧ 
    n = 6) :=
by
  sorry

end family_members_to_pay_l463_46351


namespace wire_division_l463_46302

theorem wire_division (L leftover total_length : ℝ) (seg1 seg2 : ℝ)
  (hL : L = 120 * 2)
  (hleftover : leftover = 2.4)
  (htotal : total_length = L + leftover)
  (hseg1 : seg1 = total_length / 3)
  (hseg2 : seg2 = total_length / 3) :
  seg1 = 80.8 ∧ seg2 = 80.8 := by
  sorry

end wire_division_l463_46302


namespace smallest_b_for_fourth_power_l463_46316

noncomputable def is_fourth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 4 = n

theorem smallest_b_for_fourth_power :
  ∃ b : ℕ, (0 < b) ∧ (7 + 7 * b + 7 * b ^ 2 = (7 * 1 + 7 * 18 + 7 * 18 ^ 2)) 
  ∧ is_fourth_power (7 + 7 * b + 7 * b ^ 2) := sorry

end smallest_b_for_fourth_power_l463_46316


namespace kelly_gave_away_games_l463_46344

theorem kelly_gave_away_games (initial_games : ℕ) (remaining_games : ℕ) (given_away_games : ℕ) 
  (h1 : initial_games = 183) 
  (h2 : remaining_games = 92) 
  (h3 : given_away_games = initial_games - remaining_games) : 
  given_away_games = 91 := 
by 
  sorry

end kelly_gave_away_games_l463_46344


namespace Marta_max_piles_l463_46310

theorem Marta_max_piles (a b c : ℕ) (ha : a = 42) (hb : b = 60) (hc : c = 90) : 
  Nat.gcd (Nat.gcd a b) c = 6 := by
  rw [ha, hb, hc]
  have h : Nat.gcd (Nat.gcd 42 60) 90 = Nat.gcd 6 90 := by sorry
  exact h    

end Marta_max_piles_l463_46310


namespace ab_equality_l463_46354

theorem ab_equality (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_div : 4 * a * b - 1 ∣ (4 * a ^ 2 - 1) ^ 2) : a = b := sorry

end ab_equality_l463_46354


namespace largest_divisor_of_expression_l463_46315

theorem largest_divisor_of_expression (n : ℤ) : ∃ k, ∀ n : ℤ, n^4 - n^2 = k * 12 :=
by sorry

end largest_divisor_of_expression_l463_46315


namespace sixty_percent_of_fifty_minus_forty_percent_of_thirty_l463_46342

theorem sixty_percent_of_fifty_minus_forty_percent_of_thirty : 
  (0.6 * 50) - (0.4 * 30) = 18 :=
by
  sorry

end sixty_percent_of_fifty_minus_forty_percent_of_thirty_l463_46342


namespace parabola_transformation_zeros_sum_l463_46333

theorem parabola_transformation_zeros_sum :
  let y := fun x => (x - 3)^2 + 4
  let y_rotated := fun x => -(x - 3)^2 + 4
  let y_shifted_right := fun x => -(x - 7)^2 + 4
  let y_final := fun x => -(x - 7)^2 + 7
  ∃ a b, y_final a = 0 ∧ y_final b = 0 ∧ (a + b) = 14 :=
by
  sorry

end parabola_transformation_zeros_sum_l463_46333


namespace inequality_solution_set_l463_46346

theorem inequality_solution_set (x : ℝ) :
  2 * x^2 - x ≤ 0 → 0 ≤ x ∧ x ≤ 1 / 2 :=
sorry

end inequality_solution_set_l463_46346


namespace find_r_l463_46309

-- Lean statement
theorem find_r (r : ℚ) (log_eq : Real.logb 81 (2 * r - 1) = -1 / 2) : r = 5 / 9 :=
by {
    sorry -- proof steps should not be included according to the requirements
}

end find_r_l463_46309


namespace correct_algebraic_expression_l463_46356

theorem correct_algebraic_expression
  (A : String := "1 1/2 a")
  (B : String := "a × b")
  (C : String := "a ÷ b")
  (D : String := "2a") :
  D = "2a" :=
by {
  -- Explanation based on the conditions provided
  -- A: "1 1/2 a" is not properly formatted. Correct format involves improper fraction for multiplication.
  -- B: "a × b" should avoid using the multiplication sign explicitly.
  -- C: "a ÷ b" should be written as a fraction a/b.
  -- D: "2a" is correctly formatted.
  sorry
}

end correct_algebraic_expression_l463_46356


namespace sector_area_angle_1_sector_max_area_l463_46323

-- The definition and conditions
variable (c : ℝ) (r l : ℝ)

-- 1) Proof that the area of the sector when the central angle is 1 radian is c^2 / 18
-- given 2r + l = c
theorem sector_area_angle_1 (h : 2 * r + l = c) (h1: l = r) :
  (1/2 * l * r = (c^2 / 18)) :=
by sorry

-- 2) Proof that the central angle that maximizes the area is 2 radians and the maximum area is c^2 / 16
-- given 2r + l = c
theorem sector_max_area (h : 2 * r + l = c) :
  ∃ l r, 2 * r = l ∧ 1/2 * l * r = (c^2 / 16) :=
by sorry

end sector_area_angle_1_sector_max_area_l463_46323


namespace train_average_speed_l463_46352

theorem train_average_speed (x : ℝ) (h1 : x > 0) :
  let d1 := x
  let d2 := 2 * x
  let s1 := 50
  let s2 := 20
  let t1 := d1 / s1
  let t2 := d2 / s2
  let total_distance := d1 + d2
  let total_time := t1 + t2
  let average_speed := total_distance / total_time
  average_speed = 25 := 
by
  sorry

end train_average_speed_l463_46352


namespace last_date_in_2011_divisible_by_101_is_1221_l463_46369

def is_valid_date (a b c d : ℕ) : Prop :=
  (10 * a + b) ≤ 12 ∧ (10 * c + d) ≤ 31

def date_as_number (a b c d : ℕ) : ℕ :=
  20110000 + 1000 * a + 100 * b + 10 * c + d

theorem last_date_in_2011_divisible_by_101_is_1221 :
  ∃ (a b c d : ℕ), is_valid_date a b c d ∧ date_as_number a b c d % 101 = 0 ∧ date_as_number a b c d = 20111221 :=
by
  sorry

end last_date_in_2011_divisible_by_101_is_1221_l463_46369


namespace max_marks_is_667_l463_46364

-- Definitions based on the problem's conditions
def pass_threshold (M : ℝ) : ℝ := 0.45 * M
def student_score : ℝ := 225
def failed_by : ℝ := 75
def passing_marks := student_score + failed_by

-- The actual theorem stating that if the conditions are met, then the maximum marks M is 667
theorem max_marks_is_667 : ∃ M : ℝ, pass_threshold M = passing_marks ∧ M = 667 :=
by
  sorry -- Proof is omitted as per the instructions

end max_marks_is_667_l463_46364


namespace minimum_value_ineq_l463_46368

theorem minimum_value_ineq (x : ℝ) (hx : x >= 4) : x + 4 / (x - 1) >= 5 := by
  sorry

end minimum_value_ineq_l463_46368


namespace problem1_problem2_problem3_problem4_l463_46377

theorem problem1 : (70.8 - 1.25 - 1.75 = 67.8) := sorry

theorem problem2 : ((8 + 0.8) * 1.25 = 11) := sorry

theorem problem3 : (125 * 0.48 = 600) := sorry

theorem problem4 : (6.7 * (9.3 * (6.2 + 1.7)) = 554.559) := sorry

end problem1_problem2_problem3_problem4_l463_46377


namespace vote_majority_is_160_l463_46389

-- Define the total number of votes polled
def total_votes : ℕ := 400

-- Define the percentage of votes polled by the winning candidate
def winning_percentage : ℝ := 0.70

-- Define the percentage of votes polled by the losing candidate
def losing_percentage : ℝ := 0.30

-- Define the number of votes gained by the winning candidate
def winning_votes := winning_percentage * total_votes

-- Define the number of votes gained by the losing candidate
def losing_votes := losing_percentage * total_votes

-- Define the vote majority
def vote_majority := winning_votes - losing_votes

-- Prove that the vote majority is 160 votes
theorem vote_majority_is_160 : vote_majority = 160 :=
sorry

end vote_majority_is_160_l463_46389


namespace valid_parametrizations_l463_46337

-- Definitions for the given points and directions
def pointA := (0, 4)
def dirA := (3, -1)

def pointB := (4/3, 0)
def dirB := (1, -3)

def pointC := (-2, 10)
def dirC := (-3, 9)

-- Line equation definition
def line (x y : ℝ) : Prop := y = -3 * x + 4

-- Proof statement
theorem valid_parametrizations :
  (line pointB.1 pointB.2 ∧ dirB.2 = -3 * dirB.1) ∧
  (line pointC.1 pointC.2 ∧ dirC.2 / dirC.1 = 3) :=
by
  sorry

end valid_parametrizations_l463_46337


namespace polynomial_factorization_l463_46384

noncomputable def polynomial_equivalence : Prop :=
  ∀ x : ℂ, (x^12 - 3*x^9 + 3*x^3 + 1) = (x + 1)^4 * (x^2 - x + 1)^4

theorem polynomial_factorization : polynomial_equivalence := by
  sorry

end polynomial_factorization_l463_46384


namespace Dodo_is_sane_l463_46361

-- Declare the names of the characters
inductive Character
| Dodo : Character
| Lori : Character
| Eagle : Character

open Character

-- Definitions of sanity state
def sane (c : Character) : Prop := sorry
def insane (c : Character) : Prop := ¬ sane c

-- Conditions based on the problem statement
axiom Dodo_thinks_Lori_thinks_Eagle_not_sane : (sane Lori → insane Eagle)
axiom Lori_thinks_Dodo_not_sane : insane Dodo
axiom Eagle_thinks_Dodo_sane : sane Dodo

-- Theorem to prove Dodo is sane
theorem Dodo_is_sane : sane Dodo :=
by {
    sorry
}

end Dodo_is_sane_l463_46361


namespace quadratic_square_binomial_l463_46343

theorem quadratic_square_binomial (k : ℝ) : 
  (∃ a : ℝ, (x : ℝ) → x^2 - 20 * x + k = (x + a)^2) ↔ k = 100 := 
by
  sorry

end quadratic_square_binomial_l463_46343


namespace cube_surface_area_with_holes_l463_46321

theorem cube_surface_area_with_holes 
    (edge_length : ℝ) 
    (hole_side_length : ℝ) 
    (num_faces : ℕ) 
    (parallel_edges : Prop)
    (holes_centered : Prop)
    (h_edge : edge_length = 5)
    (h_hole : hole_side_length = 2)
    (h_faces : num_faces = 6)
    (h_inside_area : parallel_edges ∧ holes_centered)
    : (150 - 24 + 96 = 222) :=
by
    sorry

end cube_surface_area_with_holes_l463_46321


namespace find_pos_ints_a_b_c_p_l463_46357

theorem find_pos_ints_a_b_c_p (a b c p : ℕ) (hp : Nat.Prime p) : 
  73 * p^2 + 6 = 9 * a^2 + 17 * b^2 + 17 * c^2 ↔
  (p = 2 ∧ a = 1 ∧ b = 4 ∧ c = 1) ∨ (p = 2 ∧ a = 1 ∧ b = 1 ∧ c = 4) :=
by
  sorry

end find_pos_ints_a_b_c_p_l463_46357


namespace trajectory_equation_l463_46379

variable (x y a b : ℝ)
variable (P : ℝ × ℝ := (0, -3))
variable (A : ℝ × ℝ := (a, 0))
variable (Q : ℝ × ℝ := (0, b))
variable (M : ℝ × ℝ := (x, y))

theorem trajectory_equation
  (h1 : A.1 = a)
  (h2 : A.2 = 0)
  (h3 : Q.1 = 0)
  (h4 : Q.2 > 0)
  (h5 : (P.1 - A.1) * (x - A.1) + (P.2 - A.2) * y = 0)
  (h6 : (x - A.1, y) = (-3/2 * (-x, b - y))) :
  y = (1 / 4) * x ^ 2 ∧ x ≠ 0 := by
    -- Sorry, proof omitted
    sorry

end trajectory_equation_l463_46379


namespace polar_to_rectangular_l463_46378

theorem polar_to_rectangular (r θ : ℝ) (h1 : r = 3 * Real.sqrt 2) (h2 : θ = Real.pi / 4) :
  (r * Real.cos θ, r * Real.sin θ) = (3, 3) :=
by
  -- Proof goes here
  sorry

end polar_to_rectangular_l463_46378


namespace victoria_initial_money_l463_46358

-- Definitions based on conditions
def cost_rice := 2 * 20
def cost_flour := 3 * 25
def cost_soda := 150
def total_spent := cost_rice + cost_flour + cost_soda
def remaining_balance := 235

-- Theorem to prove
theorem victoria_initial_money (initial_money : ℕ) :
  initial_money = total_spent + remaining_balance :=
by
  sorry

end victoria_initial_money_l463_46358


namespace find_c_gen_formula_l463_46367

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) : Prop :=
a 1 = 2 ∧
(∀ n, a (n + 1) = a n + c * n) ∧
(2 + c) * (2 + c) = 2 * (2 + 3 * c)

theorem find_c (a : ℕ → ℕ) : ∃ c, seq a c :=
by
  sorry

theorem gen_formula (a : ℕ → ℕ) (c : ℕ) (h : seq a c) : (∀ n, a n = n^2 - n + 2) :=
by
  sorry

end find_c_gen_formula_l463_46367


namespace length_of_CD_l463_46319

theorem length_of_CD (C D R S : ℝ) 
  (h1 : R = C + 3/8 * (D - C))
  (h2 : S = C + 4/11 * (D - C))
  (h3 : |S - R| = 3) :
  D - C = 264 := 
sorry

end length_of_CD_l463_46319


namespace polynomial_horner_method_l463_46317

theorem polynomial_horner_method :
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := 3
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  v_3 = 22 :=
by 
  let a_4 := 3
  let a_3 := 0
  let a_2 := -1
  let a_1 := 2
  let a_0 := 1
  let x := 2
  let v_0 := a_4
  let v_1 := v_0 * x + a_3
  let v_2 := v_1 * x + a_2
  let v_3 := v_2 * x + a_1
  sorry

end polynomial_horner_method_l463_46317


namespace factor_expression_l463_46332

theorem factor_expression (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) :=
by
  sorry

end factor_expression_l463_46332


namespace angle_of_parallel_l463_46303

-- Define a line and a plane
variable {L : Type} (l : L)
variable {P : Type} (β : P)

-- Define the parallel condition
def is_parallel (l : L) (β : P) : Prop := sorry

-- Define the angle function between a line and a plane
def angle (l : L) (β : P) : ℝ := sorry

-- The theorem stating that if l is parallel to β, then the angle is 0
theorem angle_of_parallel (h : is_parallel l β) : angle l β = 0 := sorry

end angle_of_parallel_l463_46303


namespace possible_values_of_p_l463_46318

theorem possible_values_of_p (p : ℕ) (a b : ℕ) (h_fact : (x : ℤ) → x^2 - 5 * x + p = (x - a) * (x - b))
  (h1 : a + b = 5) (h2 : 1 ≤ a ∧ a ≤ 4) (h3 : 1 ≤ b ∧ b ≤ 4) : 
  p = 4 ∨ p = 6 :=
sorry

end possible_values_of_p_l463_46318


namespace percentage_decrease_in_speed_l463_46374

variable (S : ℝ) (S' : ℝ) (T T' : ℝ)

noncomputable def percentageDecrease (originalSpeed decreasedSpeed : ℝ) : ℝ :=
  ((originalSpeed - decreasedSpeed) / originalSpeed) * 100

theorem percentage_decrease_in_speed :
  T = 40 ∧ T' = 50 ∧ S' = (4 / 5) * S →
  percentageDecrease S S' = 20 :=
by sorry

end percentage_decrease_in_speed_l463_46374


namespace chocolate_bars_partial_boxes_l463_46376

-- Define the total number of bars for each type
def totalA : ℕ := 853845
def totalB : ℕ := 537896
def totalC : ℕ := 729763

-- Define the box capacities for each type
def capacityA : ℕ := 9
def capacityB : ℕ := 11
def capacityC : ℕ := 15

-- State the theorem we want to prove
theorem chocolate_bars_partial_boxes :
  totalA % capacityA = 4 ∧
  totalB % capacityB = 3 ∧
  totalC % capacityC = 8 :=
by
  -- Proof omitted for this task
  sorry

end chocolate_bars_partial_boxes_l463_46376


namespace tanya_bought_six_plums_l463_46341

theorem tanya_bought_six_plums (pears apples pineapples pieces_left : ℕ) 
  (h_pears : pears = 6) (h_apples : apples = 4) (h_pineapples : pineapples = 2) 
  (h_pieces_left : pieces_left = 9) (h_half_fell : pieces_left * 2 = total_fruit) :
  pears + apples + pineapples < total_fruit ∧ total_fruit - (pears + apples + pineapples) = 6 :=
by
  sorry

end tanya_bought_six_plums_l463_46341


namespace boxes_containing_neither_l463_46370

theorem boxes_containing_neither
  (total_boxes : ℕ)
  (boxes_with_stickers : ℕ)
  (boxes_with_cards : ℕ)
  (boxes_with_both : ℕ)
  (h1 : total_boxes = 15)
  (h2 : boxes_with_stickers = 8)
  (h3 : boxes_with_cards = 5)
  (h4 : boxes_with_both = 3) :
  (total_boxes - (boxes_with_stickers + boxes_with_cards - boxes_with_both)) = 5 :=
by
  sorry

end boxes_containing_neither_l463_46370


namespace smallest_b_l463_46345

theorem smallest_b (b : ℝ) : b^2 - 16 * b + 63 ≤ 0 → (∃ b : ℝ, b = 7) :=
sorry

end smallest_b_l463_46345


namespace part1_inequality_l463_46397

noncomputable def f (x : ℝ) : ℝ := x - 2
noncomputable def g (x m : ℝ) : ℝ := x^2 - 2 * m * x + 4

theorem part1_inequality (m : ℝ) : (∀ x : ℝ, g x m > f x) ↔ (m ∈ Set.Ioo (-Real.sqrt 6 - (1/2)) (Real.sqrt 6 - (1/2))) :=
sorry

end part1_inequality_l463_46397


namespace vinegar_used_is_15_l463_46381

noncomputable def vinegar_used (T : ℝ) : ℝ :=
  let water := (3 / 5) * 20
  let total_volume := 27
  let vinegar := total_volume - water
  vinegar

theorem vinegar_used_is_15 (T : ℝ) (h1 : (3 / 5) * 20 = 12) (h2 : 27 - 12 = 15) (h3 : (5 / 6) * T = 15) : vinegar_used T = 15 :=
by
  sorry

end vinegar_used_is_15_l463_46381


namespace solve_quadratic_l463_46313

theorem solve_quadratic (x : ℝ) : (x^2 + x)^2 + (x^2 + x) - 6 = 0 ↔ x = -2 ∨ x = 1 :=
by
  sorry

end solve_quadratic_l463_46313


namespace sam_initial_money_l463_46375

theorem sam_initial_money :
  (9 * 7 + 16 = 79) :=
by
  sorry

end sam_initial_money_l463_46375


namespace find_m_l463_46392

-- Definitions from conditions
def ellipse_eq (x y : ℝ) (m : ℝ) : Prop := x^2 + m * y^2 = 1
def major_axis_twice_minor_axis (a b : ℝ) : Prop := a = 2 * b

-- Main statement
theorem find_m (m : ℝ) (h1 : ellipse_eq 0 0 m) (h2 : 0 < m) (h3 : 0 < m ∧ m < 1) :
  m = 1 / 4 :=
by
  sorry

end find_m_l463_46392


namespace gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l463_46334

-- GCD as the greatest common divisor
def GCD (a b : ℕ) : ℕ := Nat.gcd a b

-- LCM as the least common multiple
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- First proof problem in Lean 4
theorem gcd_lcm_relation (a b : ℕ) : GCD a b = (a * b) / (LCM a b) :=
  sorry

-- GCD function extended to three arguments
def GCD3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- LCM function extended to three arguments
def LCM3 (a b c : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) c

-- Second proof problem in Lean 4
theorem gcd3_lcm3_relation (a b c : ℕ) : GCD3 a b c = (a * b * c * LCM3 a b c) / (LCM a b * LCM b c * LCM c a) :=
  sorry

-- Third proof problem in Lean 4
theorem lcm3_gcd3_relation (a b c : ℕ) : LCM3 a b c = (a * b * c * GCD3 a b c) / (GCD a b * GCD b c * GCD c a) :=
  sorry

end gcd_lcm_relation_gcd3_lcm3_relation_lcm3_gcd3_relation_l463_46334


namespace total_cups_l463_46380

theorem total_cups (m c s : ℕ) (h1 : 3 * c = 2 * m) (h2 : 2 * c = 6) : m + c + s = 18 :=
by
  sorry

end total_cups_l463_46380


namespace johny_journey_distance_l463_46349

def south_distance : ℕ := 40
def east_distance : ℕ := south_distance + 20
def north_distance : ℕ := 2 * east_distance
def total_distance : ℕ := south_distance + east_distance + north_distance

theorem johny_journey_distance :
  total_distance = 220 := by
  sorry

end johny_journey_distance_l463_46349


namespace exists_n_such_that_not_square_l463_46329

theorem exists_n_such_that_not_square : ∃ n : ℕ, n > 1 ∧ ¬(∃ k : ℕ, k ^ 2 = 2 ^ (2 ^ n - 1) - 7) := 
sorry

end exists_n_such_that_not_square_l463_46329


namespace intersection_S_T_l463_46335

def set_S : Set ℝ := { x | abs x < 5 }
def set_T : Set ℝ := { x | x^2 + 4*x - 21 < 0 }

theorem intersection_S_T :
  set_S ∩ set_T = { x | -5 < x ∧ x < 3 } :=
sorry

end intersection_S_T_l463_46335


namespace george_final_score_l463_46393

-- Definitions for points in the first half
def first_half_odd_points (questions : Nat) := 5 * 2
def first_half_even_points (questions : Nat) := 5 * 4
def first_half_bonus_points (questions : Nat) := 3 * 5
def first_half_points := first_half_odd_points 5 + first_half_even_points 5 + first_half_bonus_points 3

-- Definitions for points in the second half
def second_half_odd_points (questions : Nat) := 6 * 3
def second_half_even_points (questions : Nat) := 6 * 5
def second_half_bonus_points (questions : Nat) := 4 * 5
def second_half_points := second_half_odd_points 6 + second_half_even_points 6 + second_half_bonus_points 4

-- Definition of the total points
def total_points := first_half_points + second_half_points

-- The theorem statement to prove the total points
theorem george_final_score : total_points = 113 := by
  unfold total_points
  unfold first_half_points
  unfold second_half_points
  unfold first_half_odd_points first_half_even_points first_half_bonus_points
  unfold second_half_odd_points second_half_even_points second_half_bonus_points
  sorry

end george_final_score_l463_46393


namespace table_length_l463_46339

theorem table_length (area_m2 : ℕ) (width_cm : ℕ) (length_cm : ℕ) 
  (h_area : area_m2 = 54)
  (h_width : width_cm = 600)
  :
  length_cm = 900 := 
  sorry

end table_length_l463_46339


namespace least_value_y_l463_46398

theorem least_value_y : ∃ y : ℝ, (3 * y ^ 3 + 3 * y ^ 2 + 5 * y + 1 = 5) ∧ ∀ z : ℝ, (3 * z ^ 3 + 3 * z ^ 2 + 5 * z + 1 = 5) → y ≤ z :=
sorry

end least_value_y_l463_46398


namespace does_not_pass_through_third_quadrant_l463_46324

theorem does_not_pass_through_third_quadrant :
  ¬ ∃ (x y : ℝ), 2 * x + 3 * y = 5 ∧ x < 0 ∧ y < 0 :=
by
  -- Proof goes here
  sorry

end does_not_pass_through_third_quadrant_l463_46324


namespace F_sum_l463_46325

noncomputable def f : ℝ → ℝ := sorry -- even function f(x)
noncomputable def F (x a c : ℝ) : ℝ := 
  let b := (a + c) / 2
  (x - b) * f (x - b) + 2016

theorem F_sum (a c : ℝ) : F a a c + F c a c = 4032 := 
by {
  sorry
}

end F_sum_l463_46325


namespace colton_share_l463_46338

-- Definitions
def footToInch (foot : ℕ) : ℕ := 12 * foot -- 1 foot equals 12 inches

-- Problem conditions
def coltonBurgerLength := footToInch 1 -- Colton bought a foot long burger
def sharedBurger (length : ℕ) : ℕ := length / 2 -- shared half with his brother

-- Equivalent proof problem statement
theorem colton_share : sharedBurger coltonBurgerLength = 6 := 
by sorry

end colton_share_l463_46338


namespace problem_part1_problem_part2_l463_46353

open Real

theorem problem_part1 (α : ℝ) (h : (sin (π - α) * cos (2 * π - α)) / (tan (π - α) * sin (π / 2 + α) * cos (π / 2 - α)) = 1 / 2) :
  (cos α - 2 * sin α) / (3 * cos α + sin α) = 5 := sorry

theorem problem_part2 (α : ℝ) (h : tan α = -2) :
  1 - 2 * sin α * cos α + cos α ^ 2 = 2 / 5 := sorry

end problem_part1_problem_part2_l463_46353


namespace john_allowance_spent_l463_46327

theorem john_allowance_spent (B t d : ℝ) (h1 : t = 0.25 * (B - d)) (h2 : d = 0.10 * (B - t)) :
  (t + d) / B = 0.31 := by
  sorry

end john_allowance_spent_l463_46327
