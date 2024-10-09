import Mathlib

namespace original_average_weight_l1101_110146

theorem original_average_weight 
  (W : ℝ)  -- Define W as the original average weight
  (h1 : 0 < W)  -- Define conditions
  (w_new1 : ℝ := 110)
  (w_new2 : ℝ := 60)
  (num_initial_players : ℝ := 7)
  (num_total_players : ℝ := 9)
  (new_average_weight : ℝ := 92)
  (total_weight_initial := num_initial_players * W)
  (total_weight_additional := w_new1 + w_new2)
  (total_weight_total := new_average_weight * num_total_players) : 
  total_weight_initial + total_weight_additional = total_weight_total → W = 94 :=
by 
  sorry

end original_average_weight_l1101_110146


namespace find_length_of_side_c_l1101_110180

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

/-- Given that in triangle ABC, sin C = 1 / 2, a = 2 * sqrt 3, b = 2,
we want to prove the length of side c is either 2 or 2 * sqrt 7. -/
theorem find_length_of_side_c (C : Real) (a b c : Real) (h1 : Real.sin C = 1 / 2)
  (h2 : a = 2 * Real.sqrt 3) (h3 : b = 2) :
  c = 2 ∨ c = 2 * Real.sqrt 7 :=
by
  sorry

end find_length_of_side_c_l1101_110180


namespace man_speed_is_4_kmph_l1101_110160

noncomputable def speed_of_man (train_length : ℝ) (train_speed_kmph : ℝ) (time_to_pass_seconds : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed_mps := train_length / time_to_pass_seconds
  let relative_speed_kmph := relative_speed_mps * 3600 / 1000
  relative_speed_kmph - train_speed_kmph

theorem man_speed_is_4_kmph : speed_of_man 140 50 9.332586726395222 = 4 := by
  sorry

end man_speed_is_4_kmph_l1101_110160


namespace inequality_proof_l1101_110142

theorem inequality_proof (x y z : ℝ) : 
  ( (x^3) / (x^3 + 2 * (y^2) * z) + 
    (y^3) / (y^3 + 2 * (z^2) * x) + 
    (z^3) / (z^3 + 2 * (x^2) * y) ) ≥ 1 := 
by 
  sorry

end inequality_proof_l1101_110142


namespace Lance_daily_earnings_l1101_110183

theorem Lance_daily_earnings :
  ∀ (hours_per_week : ℕ) (workdays_per_week : ℕ) (hourly_rate : ℕ) (total_earnings : ℕ) (daily_earnings : ℕ),
  hours_per_week = 35 →
  workdays_per_week = 5 →
  hourly_rate = 9 →
  total_earnings = hours_per_week * hourly_rate →
  daily_earnings = total_earnings / workdays_per_week →
  daily_earnings = 63 := 
by
  intros hours_per_week workdays_per_week hourly_rate total_earnings daily_earnings
  intros H1 H2 H3 H4 H5
  sorry

end Lance_daily_earnings_l1101_110183


namespace adam_initial_money_l1101_110173

theorem adam_initial_money :
  let cost_of_airplane := 4.28
  let change_received := 0.72
  cost_of_airplane + change_received = 5.00 :=
by
  sorry

end adam_initial_money_l1101_110173


namespace youngest_age_is_29_l1101_110116

-- Define that the ages form an arithmetic sequence
def arithmetic_sequence (a1 a2 a3 a4 : ℕ) : Prop :=
  ∃ (d : ℕ), a2 = a1 + d ∧ a3 = a1 + 2*d ∧ a4 = a1 + 3*d

-- Define the problem statement
theorem youngest_age_is_29 (a1 a2 a3 a4 : ℕ) (h_seq : arithmetic_sequence a1 a2 a3 a4) (h_oldest : a4 = 50) (h_sum : a1 + a2 + a3 + a4 = 158) :
  a1 = 29 :=
by
  sorry

end youngest_age_is_29_l1101_110116


namespace find_common_ratio_l1101_110108

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
∀ n m, ∃ q, a (n + 1) = a n * q ∧ a (m + 1) = a m * q

theorem find_common_ratio 
  (a : ℕ → α) 
  (h : is_geometric_sequence a) 
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1 / 4) : 
  ∃ q, q = 1 / 2 :=
by
  sorry

end find_common_ratio_l1101_110108


namespace smallest_a_for_nonprime_l1101_110193

theorem smallest_a_for_nonprime (a : ℕ) : (∀ x : ℤ, ∃ d : ℤ, d ∣ (x^4 + a^4) ∧ d ≠ 1 ∧ d ≠ (x^4 + a^4)) ↔ a = 3 := by
  sorry

end smallest_a_for_nonprime_l1101_110193


namespace expected_value_of_winnings_l1101_110122

noncomputable def winnings (n : ℕ) : ℕ := 2 * n - 1

theorem expected_value_of_winnings : 
  (1 / 6 : ℚ) * ((winnings 1) + (winnings 2) + (winnings 3) + (winnings 4) + (winnings 5) + (winnings 6)) = 6 :=
by
  sorry

end expected_value_of_winnings_l1101_110122


namespace helen_baked_more_raisin_cookies_l1101_110198

-- Definitions based on conditions
def raisin_cookies_yesterday : ℕ := 300
def raisin_cookies_day_before : ℕ := 280

-- Theorem to prove the answer
theorem helen_baked_more_raisin_cookies : raisin_cookies_yesterday - raisin_cookies_day_before = 20 :=
by
  sorry

end helen_baked_more_raisin_cookies_l1101_110198


namespace team_selection_ways_l1101_110177

theorem team_selection_ways :
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose boys team_size_boys * choose girls team_size_girls = 103950 :=
by
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end team_selection_ways_l1101_110177


namespace trader_gain_percentage_is_25_l1101_110164

noncomputable def trader_gain_percentage (C : ℝ) : ℝ :=
  ((22 * C) / (88 * C)) * 100

theorem trader_gain_percentage_is_25 (C : ℝ) (h : C ≠ 0) : trader_gain_percentage C = 25 := by
  unfold trader_gain_percentage
  field_simp [h]
  norm_num
  sorry

end trader_gain_percentage_is_25_l1101_110164


namespace cauchy_inequality_minimum_value_inequality_l1101_110199

-- Part 1: Prove Cauchy Inequality
theorem cauchy_inequality (a b x y : ℝ) : 
  (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 :=
by
  sorry

-- Part 2: Find the minimum value under the given conditions
theorem minimum_value_inequality (x y : ℝ) (h₁ : x^2 + y^2 = 2) (h₂ : x ≠ y ∨ x ≠ -y) : 
  ∃ m, m = (1 / (9 * x^2) + 9 / y^2) ∧ m = 50 / 9 :=
by
  sorry

end cauchy_inequality_minimum_value_inequality_l1101_110199


namespace geom_seq_min_val_l1101_110167

-- Definition of geometric sequence with common ratio q
def geom_seq (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Main theorem
theorem geom_seq_min_val (a : ℕ → ℝ) (q : ℝ) 
  (h_pos : ∀ n : ℕ, 0 < a n)
  (h_geom : geom_seq a q)
  (h_cond : 2 * a 3 + a 2 - 2 * a 1 - a 0 = 8) :
  2 * a 4 + a 3 = 12 * Real.sqrt 3 :=
sorry

end geom_seq_min_val_l1101_110167


namespace football_team_gain_l1101_110100

theorem football_team_gain (G : ℤ) :
  (-5 + G = 2) → (G = 7) :=
by
  intro h
  sorry

end football_team_gain_l1101_110100


namespace cyclic_quad_angles_l1101_110115

theorem cyclic_quad_angles (A B C D : ℝ) (x : ℝ)
  (h_ratio : A = 5 * x ∧ B = 6 * x ∧ C = 4 * x)
  (h_cyclic : A + D = 180 ∧ B + C = 180):
  (B = 108) ∧ (C = 72) :=
by
  sorry

end cyclic_quad_angles_l1101_110115


namespace probability_red_blue_yellow_l1101_110157

-- Define the probabilities for white, green, and black marbles
def p_white : ℚ := 1/4
def p_green : ℚ := 1/6
def p_black : ℚ := 1/8

-- Define the problem: calculating the probability of drawing a red, blue, or yellow marble
theorem probability_red_blue_yellow : 
  p_white = 1/4 → p_green = 1/6 → p_black = 1/8 →
  (1 - (p_white + p_green + p_black)) = 11/24 := 
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end probability_red_blue_yellow_l1101_110157


namespace cost_of_orange_juice_l1101_110103

theorem cost_of_orange_juice (total_money : ℕ) (bread_qty : ℕ) (orange_qty : ℕ) 
  (bread_cost : ℕ) (money_left : ℕ) (total_spent : ℕ) (orange_cost : ℕ) 
  (h1 : total_money = 86) (h2 : bread_qty = 3) (h3 : orange_qty = 3) 
  (h4 : bread_cost = 3) (h5 : money_left = 59) :
  (total_money - money_left - bread_qty * bread_cost) / orange_qty = 6 :=
by
  have h6 : total_spent = total_money - money_left := by sorry
  have h7 : total_spent - bread_qty * bread_cost = orange_qty * orange_cost := by sorry
  have h8 : orange_cost = 6 := by sorry
  exact sorry

end cost_of_orange_juice_l1101_110103


namespace smallest_possible_students_group_l1101_110128

theorem smallest_possible_students_group 
  (students : ℕ) :
  (∀ n, 2 ≤ n ∧ n ≤ 15 → ∃ k, students = k * n) ∧
  ¬∃ k, students = k * 10 ∧ ¬∃ k, students = k * 25 ∧ ¬∃ k, students = k * 50 ∧
  ∀ m n, 1 ≤ m ∧ m ≤ 15 ∧ 1 ≤ n ∧ n ≤ 15 ∧ (students ≠ m * n) → (m = n ∨ m ≠ n)
  → students = 120 := sorry

end smallest_possible_students_group_l1101_110128


namespace no_roots_of_form_one_over_n_l1101_110129

theorem no_roots_of_form_one_over_n (a b c : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_c : c % 2 = 1) :
  ∀ n : ℕ, ¬(a * (1 / (n:ℚ))^2 + b * (1 / (n:ℚ)) + c = 0) := by
  sorry

end no_roots_of_form_one_over_n_l1101_110129


namespace marlon_keeps_4_lollipops_l1101_110189

def initial_lollipops : ℕ := 42
def fraction_given_to_emily : ℚ := 2 / 3
def lollipops_given_to_lou : ℕ := 10

theorem marlon_keeps_4_lollipops :
  let lollipops_given_to_emily := fraction_given_to_emily * initial_lollipops
  let lollipops_after_emily := initial_lollipops - lollipops_given_to_emily
  let marlon_keeps := lollipops_after_emily - lollipops_given_to_lou
  marlon_keeps = 4 :=
by
  sorry

end marlon_keeps_4_lollipops_l1101_110189


namespace inequality_proof_l1101_110162

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by
  sorry

end inequality_proof_l1101_110162


namespace hill_height_l1101_110186

theorem hill_height (h : ℝ) (time_up : ℝ := h / 9) (time_down : ℝ := h / 12) (total_time : ℝ := time_up + time_down) (time_cond : total_time = 175) : h = 900 :=
by 
  sorry

end hill_height_l1101_110186


namespace area_inside_Z_outside_X_l1101_110158

structure Circle :=
  (center : Real × Real)
  (radius : ℝ)

def tangent (A B : Circle) : Prop :=
  dist A.center B.center = A.radius + B.radius

theorem area_inside_Z_outside_X (X Y Z : Circle)
  (hX : X.radius = 1) 
  (hY : Y.radius = 1) 
  (hZ : Z.radius = 1)
  (tangent_XY : tangent X Y)
  (tangent_XZ : tangent X Z)
  (non_intersect_YZ : dist Z.center Y.center > Z.radius + Y.radius) :
  π - 1/2 * π = 1/2 * π := 
by
  sorry

end area_inside_Z_outside_X_l1101_110158


namespace quadratic_nonneg_iff_m_in_range_l1101_110137

theorem quadratic_nonneg_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 2 * m + 5 ≥ 0) ↔ (-2 : ℝ) ≤ m ∧ m ≤ 10 :=
by sorry

end quadratic_nonneg_iff_m_in_range_l1101_110137


namespace triangle_side_length_l1101_110171

theorem triangle_side_length 
  (X Z : ℝ) (x z y : ℝ)
  (h1 : x = 36)
  (h2 : z = 72)
  (h3 : Z = 4 * X) :
  y = 72 := by
  sorry

end triangle_side_length_l1101_110171


namespace rowing_rate_in_still_water_l1101_110147

theorem rowing_rate_in_still_water (R C : ℝ) 
  (h1 : (R + C) * 2 = 26)
  (h2 : (R - C) * 4 = 26) : 
  R = 26 / 3 :=
by
  sorry

end rowing_rate_in_still_water_l1101_110147


namespace bacteria_growth_rate_l1101_110121

theorem bacteria_growth_rate (B G : ℝ) (h : B * G^16 = 2 * B * G^15) : G = 2 :=
by
  sorry

end bacteria_growth_rate_l1101_110121


namespace exist_n_l1101_110118

theorem exist_n : ∃ n : ℕ, n > 1 ∧ ¬(Nat.Prime n) ∧ ∀ a : ℤ, (a^n - a) % n = 0 :=
by
  sorry

end exist_n_l1101_110118


namespace boundary_shadow_function_l1101_110192

theorem boundary_shadow_function 
    (r : ℝ) (O P : ℝ × ℝ × ℝ) (f : ℝ → ℝ)
    (h_radius : r = 1)
    (h_center : O = (1, 0, 1))
    (h_light_source : P = (1, -1, 2)) :
  (∀ x, f x = (x - 1) ^ 2 / 4 - 1) := 
by 
  sorry

end boundary_shadow_function_l1101_110192


namespace trapezoid_ratio_l1101_110102

structure Trapezoid (α : Type) [LinearOrderedField α] :=
  (AB CD : α)
  (areas : List α)
  (AB_gt_CD : AB > CD)
  (areas_eq : areas = [3, 5, 6, 8])

open Trapezoid

theorem trapezoid_ratio (α : Type) [LinearOrderedField α] (T : Trapezoid α) :
  ∃ ρ : α, T.AB / T.CD = ρ ∧ ρ = 8 / 3 :=
by
  sorry

end trapezoid_ratio_l1101_110102


namespace students_average_vegetables_l1101_110159

variable (points_needed : ℕ) (points_per_vegetable : ℕ) (students : ℕ) (school_days : ℕ) (school_weeks : ℕ)

def average_vegetables_per_student_per_week (points_needed points_per_vegetable students school_days school_weeks : ℕ) : ℕ :=
  let total_vegetables := points_needed / points_per_vegetable
  let vegetables_per_student := total_vegetables / students
  vegetables_per_student / school_weeks

theorem students_average_vegetables 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : students = 25) 
  (h4 : school_days = 10) 
  (h5 : school_weeks = 2) : 
  average_vegetables_per_student_per_week points_needed points_per_vegetable students school_days school_weeks = 2 :=
by
  sorry

end students_average_vegetables_l1101_110159


namespace books_before_purchase_l1101_110179

theorem books_before_purchase (x : ℕ) (h : x + 140 = (27 / 25 : ℚ) * x) : x = 1750 :=
sorry

end books_before_purchase_l1101_110179


namespace intersection_eq_l1101_110187

def M : Set Real := {x | x^2 < 3 * x}
def N : Set Real := {x | Real.log x < 0}

theorem intersection_eq : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l1101_110187


namespace incorrect_inequality_l1101_110101

variable (a b : ℝ)

theorem incorrect_inequality (h : a > b) : ¬ (-2 * a > -2 * b) :=
by sorry

end incorrect_inequality_l1101_110101


namespace total_attendance_l1101_110161

theorem total_attendance (A C : ℕ) (adult_ticket_price child_ticket_price total_revenue : ℕ) 
(h1 : adult_ticket_price = 11) (h2 : child_ticket_price = 10) (h3 : total_revenue = 246) 
(h4 : C = 7) (h5 : adult_ticket_price * A + child_ticket_price * C = total_revenue) : 
A + C = 23 :=
by {
  sorry
}

end total_attendance_l1101_110161


namespace problem_eval_at_x_eq_3_l1101_110130

theorem problem_eval_at_x_eq_3 : ∀ x : ℕ, x = 3 → (x^x)^(x^x) = 27^27 :=
by
  intros x hx
  rw [hx]
  sorry

end problem_eval_at_x_eq_3_l1101_110130


namespace bullet_speed_difference_l1101_110194

def bullet_speed_in_same_direction (v_h v_b : ℝ) : ℝ :=
  v_b + v_h

def bullet_speed_in_opposite_direction (v_h v_b : ℝ) : ℝ :=
  v_b - v_h

theorem bullet_speed_difference (v_h v_b : ℝ) (h_h : v_h = 20) (h_b : v_b = 400) :
  bullet_speed_in_same_direction v_h v_b - bullet_speed_in_opposite_direction v_h v_b = 40 :=
by
  rw [h_h, h_b]
  sorry

end bullet_speed_difference_l1101_110194


namespace find_x2_y2_and_xy_l1101_110117

-- Problem statement
theorem find_x2_y2_and_xy (x y : ℝ) 
  (h1 : (x + y)^2 = 1) 
  (h2 : (x - y)^2 = 9) : 
  x^2 + y^2 = 5 ∧ x * y = -2 :=
by
  sorry -- Proof omitted

end find_x2_y2_and_xy_l1101_110117


namespace number_of_feet_on_branches_l1101_110144

def number_of_birds : ℕ := 46
def feet_per_bird : ℕ := 2

theorem number_of_feet_on_branches : number_of_birds * feet_per_bird = 92 := 
by 
  sorry

end number_of_feet_on_branches_l1101_110144


namespace base_k_representation_l1101_110153

theorem base_k_representation (k : ℕ) (hk : k > 0) (hk_exp : 7 / 51 = (2 * k + 3 : ℚ) / (k ^ 2 - 1 : ℚ)) : k = 16 :=
by {
  sorry
}

end base_k_representation_l1101_110153


namespace volume_of_hemisphere_l1101_110135

theorem volume_of_hemisphere (d : ℝ) (h : d = 10) : 
  let r := d / 2
  let V := (2 / 3) * π * r^3
  V = 250 / 3 * π := by
sorry

end volume_of_hemisphere_l1101_110135


namespace total_teeth_cleaned_l1101_110141

/-
  Given:
   1. Dogs have 42 teeth.
   2. Cats have 30 teeth.
   3. Pigs have 28 teeth.
   4. There are 5 dogs.
   5. There are 10 cats.
   6. There are 7 pigs.
  Prove: The total number of teeth Vann will clean today is 706.
-/

theorem total_teeth_cleaned :
  let dogs: Nat := 5
  let cats: Nat := 10
  let pigs: Nat := 7
  let dog_teeth: Nat := 42
  let cat_teeth: Nat := 30
  let pig_teeth: Nat := 28
  (dogs * dog_teeth) + (cats * cat_teeth) + (pigs * pig_teeth) = 706 := by
  -- Proof goes here
  sorry

end total_teeth_cleaned_l1101_110141


namespace caricatures_sold_on_sunday_l1101_110105

def caricature_price : ℕ := 20
def saturday_sales : ℕ := 24
def total_earnings : ℕ := 800

theorem caricatures_sold_on_sunday :
  (total_earnings - saturday_sales * caricature_price) / caricature_price = 16 :=
by
  sorry  -- Proof goes here

end caricatures_sold_on_sunday_l1101_110105


namespace average_score_l1101_110148

variable (u v A : ℝ)
variable (h1 : v / u = 1/3)
variable (h2 : A = (u + v) / 2)

theorem average_score : A = (2/3) * u := by
  sorry

end average_score_l1101_110148


namespace relationship_among_mnr_l1101_110123

-- Definitions of the conditions
variables {a b c : ℝ}
variables (m n r : ℝ)

-- Assumption given by the conditions
def conditions (a b c : ℝ) := 0 < a ∧ a < b ∧ b < 1 ∧ 1 < c
def log_equations (a b c m n : ℝ) := m = Real.log c / Real.log a ∧ n = Real.log c / Real.log b
def r_definition (a c r : ℝ) := r = a^c

-- Statement: If the conditions are satisfied, then the relationship holds
theorem relationship_among_mnr (a b c m n r : ℝ)
  (h1 : conditions a b c)
  (h2 : log_equations a b c m n)
  (h3 : r_definition a c r) :
  n < m ∧ m < r := by
  sorry

end relationship_among_mnr_l1101_110123


namespace min_cost_per_ounce_l1101_110190

theorem min_cost_per_ounce 
  (cost_40 : ℝ := 200) (cost_90 : ℝ := 400)
  (percentage_40 : ℝ := 0.4) (percentage_90 : ℝ := 0.9)
  (desired_percentage : ℝ := 0.5) :
  (∀ (x y : ℝ), 0.4 * x + 0.9 * y = 0.5 * (x + y) → 200 * x + 400 * y / (x + y) = 240) :=
sorry

end min_cost_per_ounce_l1101_110190


namespace unique_triangle_with_consecutive_sides_and_angle_condition_l1101_110165

theorem unique_triangle_with_consecutive_sides_and_angle_condition
    (a b c : ℕ) (A B C : ℝ) (h1 : a < b ∧ b < c)
    (h2 : b = a + 1 ∧ c = a + 2)
    (h3 : C = 2 * B)
    (h4 : ∀ x y z : ℕ, x < y ∧ y < z → y = x + 1 ∧ z = x + 2 → 2 * B = C)
    : ∃! (a b c : ℕ) (A B C : ℝ), (a < b ∧ b < c) ∧ (b = a + 1 ∧ c = a + 2) ∧ (C = 2 * B) :=
  sorry

end unique_triangle_with_consecutive_sides_and_angle_condition_l1101_110165


namespace number_of_integers_satisfying_l1101_110138

theorem number_of_integers_satisfying (k1 k2 : ℕ) (hk1 : k1 = 300) (hk2 : k2 = 1000) :
  ∃ m : ℕ, m = 14 ∧ ∀ n : ℕ, 300 < n^2 → n^2 < 1000 → 18 ≤ n ∧ n ≤ 31 :=
by
  use 14
  sorry

end number_of_integers_satisfying_l1101_110138


namespace vegetables_sold_mass_l1101_110182

/-- Define the masses of the vegetables --/
def mass_carrots : ℕ := 15
def mass_zucchini : ℕ := 13
def mass_broccoli : ℕ := 8

/-- Define the total mass of installed vegetables --/
def total_mass : ℕ := mass_carrots + mass_zucchini + mass_broccoli

/-- Define the mass of vegetables sold (half of the total mass) --/
def mass_sold : ℕ := total_mass / 2

/-- Prove that the mass of vegetables sold is 18 kg --/
theorem vegetables_sold_mass : mass_sold = 18 := by
  sorry

end vegetables_sold_mass_l1101_110182


namespace rectangle_area_l1101_110139

theorem rectangle_area (w l d : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : d = 10)
  (h3 : d^2 = w^2 + l^2) : 
  l * w = 40 := 
by
  sorry

end rectangle_area_l1101_110139


namespace power_multiplication_l1101_110155

theorem power_multiplication :
  3^5 * 6^5 = 1889568 :=
by
  sorry

end power_multiplication_l1101_110155


namespace two_lines_perpendicular_to_same_plane_are_parallel_l1101_110197

/- 
Problem: Let a, b be two lines, and α be a plane. Prove that if a ⊥ α and b ⊥ α, then a ∥ b.
-/

variables {Line Plane : Type} 

def is_parallel (l1 l2 : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry
def is_contained_in (l : Line) (p : Plane) : Prop := sorry

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane)
  (ha_perpendicular : is_perpendicular a α)
  (hb_perpendicular : is_perpendicular b α) :
  is_parallel a b :=
by
  sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l1101_110197


namespace total_cost_correct_l1101_110191

-- Definitions for the costs of items.
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87

-- Definitions for the quantities.
def num_sandwiches : ℝ := 2
def num_sodas : ℝ := 4

-- The calculation for the total cost.
def total_cost : ℝ := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The claim that needs to be proved.
theorem total_cost_correct : total_cost = 10.46 := by
  sorry

end total_cost_correct_l1101_110191


namespace initial_group_size_l1101_110133

theorem initial_group_size (W : ℝ) : 
  (∃ n : ℝ, (W + 15) / n = W / n + 2.5) → n = 6 :=
by
  sorry

end initial_group_size_l1101_110133


namespace sergeant_distance_travel_l1101_110112

noncomputable def sergeant_distance (x k : ℝ) : ℝ :=
  let t₁ := 1 / (x * (k - 1))
  let t₂ := 1 / (x * (k + 1))
  let t := t₁ + t₂
  let d := k * 4 / 3
  d

theorem sergeant_distance_travel (x k : ℝ) (h1 : (4 * k) / (k^2 - 1) = 4 / 3) :
  sergeant_distance x k = 8 / 3 := by
  sorry

end sergeant_distance_travel_l1101_110112


namespace tangent_lines_parallel_to_line_l1101_110152

theorem tangent_lines_parallel_to_line (a : ℝ) (b : ℝ)
  (h1 : b = a^3 + a - 2)
  (h2 : 3 * a^2 + 1 = 4) :
  (b = 4 * a - 4 ∨ b = 4 * a) :=
sorry

end tangent_lines_parallel_to_line_l1101_110152


namespace ax2_x_plus_1_positive_l1101_110140

theorem ax2_x_plus_1_positive (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ (a > 1/4) :=
by {
  sorry
}

end ax2_x_plus_1_positive_l1101_110140


namespace problem1_problem2_problem3_problem4_l1101_110106

-- Problem 1: 27 - 16 + (-7) - 18 = -14
theorem problem1 : 27 - 16 + (-7) - 18 = -14 := 
by 
  sorry

-- Problem 2: (-6) * (-3/4) / (-3/2) = -3
theorem problem2 : (-6) * (-3/4) / (-3/2) = -3 := 
by
  sorry

-- Problem 3: (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81
theorem problem3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := 
by
  sorry

-- Problem 4: -2^4 + 3 * (-1)^4 - (-2)^3 = -5
theorem problem4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := 
by
  sorry

end problem1_problem2_problem3_problem4_l1101_110106


namespace no_common_points_iff_parallel_l1101_110196

-- Definitions based on conditions:
def line (a : Type) : Prop := sorry
def plane (M : Type) : Prop := sorry
def no_common_points (a : Type) (M : Type) : Prop := sorry
def parallel (a : Type) (M : Type) : Prop := sorry

-- Theorem stating the relationship is necessary and sufficient
theorem no_common_points_iff_parallel (a M : Type) :
  no_common_points a M ↔ parallel a M := sorry

end no_common_points_iff_parallel_l1101_110196


namespace solve_for_x_l1101_110151

theorem solve_for_x (x : ℝ) (h : (3 / 4) + (1 / x) = 7 / 8) : x = 8 :=
sorry

end solve_for_x_l1101_110151


namespace suraj_average_after_13th_innings_l1101_110175

theorem suraj_average_after_13th_innings
  (A : ℝ)
  (h : (12 * A + 96) / 13 = A + 5) :
  (12 * A + 96) / 13 = 36 :=
by
  sorry

end suraj_average_after_13th_innings_l1101_110175


namespace fair_coin_second_head_l1101_110120

theorem fair_coin_second_head (P : ℝ) 
  (fair_coin : ∀ outcome : ℝ, outcome = 0.5) :
  P = 0.5 :=
by
  sorry

end fair_coin_second_head_l1101_110120


namespace vanya_speed_l1101_110134

variable (v : ℝ)

theorem vanya_speed (h : (v + 2) / v = 2.5) : (v + 4) / v = 4 := by
  sorry

end vanya_speed_l1101_110134


namespace product_of_five_consecutive_integers_not_square_l1101_110163

theorem product_of_five_consecutive_integers_not_square (a : ℕ) (ha : 0 < a) : ¬ ∃ k : ℕ, k^2 = a * (a + 1) * (a + 2) * (a + 3) * (a + 4) := sorry

end product_of_five_consecutive_integers_not_square_l1101_110163


namespace toy_store_revenue_fraction_l1101_110149

theorem toy_store_revenue_fraction (N D J : ℝ) 
  (h1 : J = N / 3) 
  (h2 : D = 3.75 * (N + J) / 2) : 
  (N / D) = 2 / 5 :=
by sorry

end toy_store_revenue_fraction_l1101_110149


namespace inequality_holds_l1101_110110

variable {x y : ℝ}

theorem inequality_holds (h₀ : 0 < x) (h₁ : x < 1) (h₂ : 0 < y) (h₃ : y < 1) :
  (x^2 / (x + y)) + (y^2 / (1 - x)) + ((1 - x - y)^2 / (1 - y)) ≥ 1 / 2 := by
  sorry

end inequality_holds_l1101_110110


namespace solve_inequality_l1101_110109

theorem solve_inequality : 
  {x : ℝ | -3 * x^2 + 9 * x + 6 < 0} = {x : ℝ | -2 / 3 < x ∧ x < 3} :=
by {
  sorry
}

end solve_inequality_l1101_110109


namespace part_a_region_part_b_region_part_c_region_l1101_110150

-- Definitions for Part (a)
def surface1a (x y z : ℝ) := 2 * y = x ^ 2 + z ^ 2
def surface2a (x y z : ℝ) := x ^ 2 + z ^ 2 = 1
def region_a (x y z : ℝ) := surface1a x y z ∧ surface2a x y z

-- Definitions for Part (b)
def surface1b (x y z : ℝ) := z = 0
def surface2b (x y z : ℝ) := y + z = 2
def surface3b (x y z : ℝ) := y = x ^ 2
def region_b (x y z : ℝ) := surface1b x y z ∧ surface2b x y z ∧ surface3b x y z

-- Definitions for Part (c)
def surface1c (x y z : ℝ) := z = 6 - x ^ 2 - y ^ 2
def surface2c (x y z : ℝ) := x ^ 2 + y ^ 2 = z ^ 2
def region_c (x y z : ℝ) := surface1c x y z ∧ surface2c x y z

-- The formal theorem statements
theorem part_a_region : ∃x y z : ℝ, region_a x y z := by
  sorry

theorem part_b_region : ∃x y z : ℝ, region_b x y z := by
  sorry

theorem part_c_region : ∃x y z : ℝ, region_c x y z := by
  sorry

end part_a_region_part_b_region_part_c_region_l1101_110150


namespace group_contains_2007_l1101_110184

theorem group_contains_2007 : 
  ∃ k, 2007 ∈ {a | (k * (k + 1)) / 2 < a ∧ a ≤ ((k + 1) * (k + 2)) / 2} ∧ k = 45 :=
by sorry

end group_contains_2007_l1101_110184


namespace angle_sum_90_l1101_110132

theorem angle_sum_90 (A B : ℝ) (h : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2) : A + B = Real.pi / 2 :=
sorry

end angle_sum_90_l1101_110132


namespace perimeter_of_region_l1101_110131

-- Define the conditions as Lean definitions
def area_of_region (a : ℝ) := a = 400
def number_of_squares (n : ℕ) := n = 8
def arrangement := "2x4 rectangle"

-- Define the statement we need to prove
theorem perimeter_of_region (a : ℝ) (n : ℕ) (s : ℝ) 
  (h_area_region : area_of_region a) 
  (h_number_of_squares : number_of_squares n) 
  (h_arrangement : arrangement = "2x4 rectangle")
  (h_area_one_square : a / n = s^2) :
  4 * 10 * (s) = 80 * 2^(1/2)  :=
by sorry

end perimeter_of_region_l1101_110131


namespace problem_solution_l1101_110178

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) + a

theorem problem_solution (a m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x a ≤ 6) →
  a = 1 ∧ (∃ n : ℝ, f n 1 ≤ m - f (-n) 1) → 4 ≤ m := 
by
  sorry

end problem_solution_l1101_110178


namespace find_d_l1101_110127

noncomputable def problem_condition :=
  ∃ (v d : ℝ × ℝ) (t : ℝ) (x y : ℝ),
  (y = (5 * x - 7) / 6) ∧ 
  ((x, y) = (v.1 + t * d.1, v.2 + t * d.2)) ∧ 
  (x ≥ 4) ∧ 
  (dist (x, y) (4, 2) = t)

noncomputable def correct_answer : ℝ × ℝ := ⟨6 / 7, 5 / 7⟩

theorem find_d 
  (h : problem_condition) : 
  ∃ (d : ℝ × ℝ), d = correct_answer :=
sorry

end find_d_l1101_110127


namespace fraction_simplifiable_by_7_l1101_110145

theorem fraction_simplifiable_by_7 (a b c : ℤ) (h : (100 * a + 10 * b + c) % 7 = 0) : 
  ((10 * b + c + 16 * a) % 7 = 0) ∧ ((10 * b + c - 61 * a) % 7 = 0) :=
by
  sorry

end fraction_simplifiable_by_7_l1101_110145


namespace selection_methods_count_l1101_110172

noncomputable def num_selection_methods (total_students chosen_students : ℕ) (A B : ℕ) : ℕ :=
  let with_A_and_B := Nat.choose (total_students - 2) (chosen_students - 2)
  let with_one_A_or_B := Nat.choose (total_students - 2) (chosen_students - 1) * Nat.choose 2 1
  with_A_and_B + with_one_A_or_B

theorem selection_methods_count :
  num_selection_methods 10 4 1 2 = 140 :=
by
  -- We can add detailed proof here, for now we provide a placeholder
  sorry

end selection_methods_count_l1101_110172


namespace distance_from_C_to_A_is_8_l1101_110113

-- Define points A, B, and C as real numbers representing positions
def A : ℝ := 0  -- Starting point
def B : ℝ := A - 15  -- 15 meters west from A
def C : ℝ := B + 23  -- 23 meters east from B

-- Prove that the distance from point C to point A is 8 meters
theorem distance_from_C_to_A_is_8 : abs (C - A) = 8 :=
by
  sorry

end distance_from_C_to_A_is_8_l1101_110113


namespace sum_of_squares_of_roots_l1101_110174

theorem sum_of_squares_of_roots : 
  (∃ r1 r2 : ℝ, r1 + r2 = 11 ∧ r1 * r2 = 12 ∧ (r1 ^ 2 + r2 ^ 2) = 97) := 
sorry

end sum_of_squares_of_roots_l1101_110174


namespace david_chemistry_marks_l1101_110154

theorem david_chemistry_marks (marks_english marks_math marks_physics marks_biology : ℝ)
  (average_marks: ℝ) (marks_english_val: marks_english = 72) (marks_math_val: marks_math = 45)
  (marks_physics_val: marks_physics = 72) (marks_biology_val: marks_biology = 75)
  (average_marks_val: average_marks = 68.2) : 
  ∃ marks_chemistry : ℝ, (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) / 5 = average_marks ∧ 
    marks_chemistry = 77 := 
by
  sorry

end david_chemistry_marks_l1101_110154


namespace fraction_increase_by_3_l1101_110176

theorem fraction_increase_by_3 (x y : ℝ) (h₁ : x' = 3 * x) (h₂ : y' = 3 * y) : 
  (x' * y') / (x' - y') = 3 * (x * y) / (x - y) :=
by
  sorry

end fraction_increase_by_3_l1101_110176


namespace solve_expression_l1101_110188

theorem solve_expression (a x : ℝ) (h1 : a ≠ 0) (h2 : x ≠ a) : 
  (a / (2 * a + x) - x / (a - x)) / (x / (2 * a + x) + a / (a - x)) = -1 → 
  x = a / 2 :=
by
  sorry

end solve_expression_l1101_110188


namespace packs_used_after_6_weeks_l1101_110126

-- Define the conditions as constants or definitions.
def pages_per_class_per_day : ℕ := 2
def num_classes : ℕ := 5
def days_per_week : ℕ := 5
def weeks : ℕ := 6
def pages_per_pack : ℕ := 100

-- The total number of packs of notebook paper Chip will use after 6 weeks
theorem packs_used_after_6_weeks : (pages_per_class_per_day * num_classes * days_per_week * weeks) / pages_per_pack = 3 := 
by
  -- skip the proof
  sorry

end packs_used_after_6_weeks_l1101_110126


namespace find_k_l1101_110156

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 3) (h2 : m + 2 = 2 * (n + k) + 3) : k = 1 :=
by
  -- Proof is omitted
  sorry

end find_k_l1101_110156


namespace line_plane_parallelism_l1101_110119

variables {Point : Type} [LinearOrder Point] -- Assuming Point is a Type with some linear order.

-- Definitions for line and plane
-- These definitions need further libraries or details depending on actual Lean geometry library support
@[ext] structure Line (P : Type) := (contains : P → Prop)
@[ext] structure Plane (P : Type) := (contains : P → Prop)

variables {a b : Line Point} {α β : Plane Point} {l : Line Point}

-- Conditions (as in part a)
axiom lines_are_different : a ≠ b
axiom planes_are_different : α ≠ β
axiom planes_intersect_in_line : ∃ l, α.contains l ∧ β.contains l
axiom a_parallel_l : ∀ p : Point, a.contains p → l.contains p
axiom b_within_plane : ∀ p : Point, b.contains p → β.contains p
axiom b_parallel_alpha : ∀ p q : Point, β.contains p → β.contains q → α.contains p → α.contains q

-- Define the theorem statement
theorem line_plane_parallelism : a ≠ b ∧ α ≠ β ∧ (∃ l, α.contains l ∧ β.contains l) 
  ∧ (∀ p, a.contains p → l.contains p) 
  ∧ (∀ p, b.contains p → β.contains p) 
  ∧ (∀ p q, β.contains p → β.contains q → α.contains p → α.contains q) → a = b :=
by sorry

end line_plane_parallelism_l1101_110119


namespace percentage_decrease_l1101_110143

theorem percentage_decrease (P : ℝ) (new_price : ℝ) (x : ℝ) (h1 : new_price = 320) (h2 : P = 421.05263157894734) : x = 24 :=
by
  sorry

end percentage_decrease_l1101_110143


namespace yanna_kept_36_apples_l1101_110104

-- Define the initial number of apples Yanna has
def initial_apples : ℕ := 60

-- Define the number of apples given to Zenny
def apples_given_to_zenny : ℕ := 18

-- Define the number of apples given to Andrea
def apples_given_to_andrea : ℕ := 6

-- The proof statement that Yanna kept 36 apples
theorem yanna_kept_36_apples : initial_apples - apples_given_to_zenny - apples_given_to_andrea = 36 := by
  sorry

end yanna_kept_36_apples_l1101_110104


namespace simplify_fraction_l1101_110124

theorem simplify_fraction :
  (3 * (Real.sqrt 3 + Real.sqrt 8)) / (2 * Real.sqrt (3 + Real.sqrt 5)) = 
  (297 - 99 * Real.sqrt 5 + 108 * Real.sqrt 6 - 36 * Real.sqrt 30) / 16 := by
  sorry

end simplify_fraction_l1101_110124


namespace rectangle_side_ratio_l1101_110195

theorem rectangle_side_ratio
  (s : ℝ)  -- the side length of the inner square
  (y x : ℝ) -- the side lengths of the rectangles (y: shorter, x: longer)
  (h1 : 9 * s^2 = (3 * s)^2)  -- the area of the outer square is 9 times that of the inner square
  (h2 : s + 2*y = 3*s)  -- the total side length relation due to geometry
  (h3 : x + y = 3*s)  -- another side length relation
: x / y = 2 :=
by
  sorry

end rectangle_side_ratio_l1101_110195


namespace number_of_women_l1101_110166

theorem number_of_women (n_men n_women n_dances men_partners women_partners : ℕ) 
  (h_men_partners : men_partners = 4)
  (h_women_partners : women_partners = 3)
  (h_n_men : n_men = 15)
  (h_total_dances : n_dances = n_men * men_partners)
  (h_women_calc : n_women = n_dances / women_partners) :
  n_women = 20 :=
sorry

end number_of_women_l1101_110166


namespace remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l1101_110136

theorem remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0:
  (7 * 12^24 + 3^24) % 11 = 0 := sorry

end remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l1101_110136


namespace compute_moles_of_NaHCO3_l1101_110170

def equilibrium_constant : Real := 7.85 * 10^5

def balanced_equation (NaHCO3 HCl H2O CO2 NaCl : ℝ) : Prop :=
  NaHCO3 = HCl ∧ NaHCO3 = H2O ∧ NaHCO3 = CO2 ∧ NaHCO3 = NaCl

theorem compute_moles_of_NaHCO3
  (K : Real)
  (hK : K = 7.85 * 10^5)
  (HCl_required : ℝ)
  (hHCl : HCl_required = 2)
  (Water_formed : ℝ)
  (hWater : Water_formed = 2)
  (CO2_formed : ℝ)
  (hCO2 : CO2_formed = 2)
  (NaCl_formed : ℝ)
  (hNaCl : NaCl_formed = 2) :
  ∃ NaHCO3 : ℝ, NaHCO3 = 2 :=
by
  -- Conditions: equilibrium constant, balanced equation
  have equilibrium_condition := equilibrium_constant
  -- Here you would normally work through the steps of the proof using the given conditions,
  -- but we are setting it up as a theorem without a proof for now.
  existsi 2
  -- Placeholder for the formal proof.
  sorry

end compute_moles_of_NaHCO3_l1101_110170


namespace find_daily_wage_of_c_l1101_110125

def dailyWagesInRatio (a b c : ℕ) : Prop :=
  4 * a = 3 * b ∧ 5 * a = 3 * c

def totalEarnings (a b c : ℕ) (total : ℕ) : Prop :=
  6 * a + 9 * b + 4 * c = total

theorem find_daily_wage_of_c (a b c : ℕ) (total : ℕ) 
  (h1 : dailyWagesInRatio a b c) 
  (h2 : totalEarnings a b c total) 
  (h3 : total = 1406) : 
  c = 95 :=
by
  -- We assume the conditions and solve the required proof.
  sorry

end find_daily_wage_of_c_l1101_110125


namespace sin_neg_390_eq_neg_half_l1101_110114

theorem sin_neg_390_eq_neg_half : Real.sin (-390 * Real.pi / 180) = -1 / 2 :=
  sorry

end sin_neg_390_eq_neg_half_l1101_110114


namespace ben_remaining_money_l1101_110107

variable (initial_capital : ℝ := 2000) 
variable (payment_to_supplier : ℝ := 600)
variable (payment_from_debtor : ℝ := 800)
variable (maintenance_cost : ℝ := 1200)
variable (remaining_capital : ℝ := 1000)

theorem ben_remaining_money
  (h1 : initial_capital = 2000)
  (h2 : payment_to_supplier = 600)
  (h3 : payment_from_debtor = 800)
  (h4 : maintenance_cost = 1200) :
  remaining_capital = (initial_capital - payment_to_supplier + payment_from_debtor - maintenance_cost) :=
sorry

end ben_remaining_money_l1101_110107


namespace exists_non_regular_triangle_with_similar_medians_as_sides_l1101_110168

theorem exists_non_regular_triangle_with_similar_medians_as_sides 
  (a b c : ℝ) 
  (s_a s_b s_c : ℝ)
  (h1 : 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2)
  (h2 : 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2)
  (h3 : 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2)
  (similarity_cond : (2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :
  ∃ (a b c : ℝ), (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  (∃ (s_a s_b s_c : ℝ), 4 * s_a^2 = 2 * b^2 + 2 * c^2 - a^2 ∧ 4 * s_b^2 = 2 * c^2 + 2 * a^2 - b^2 ∧ 4 * s_c^2 = 2 * a^2 + 2 * b^2 - c^2) ∧
  ((2*c^2 + 2*b^2 - a^2) / c^2 = (2*c^2 + 2*a^2 - b^2) / b^2 ∧ (2*c^2 + 2*a^2 - b^2) / b^2 = (2*a^2 + 2*b^2 - c^2) / a^2) :=
sorry

end exists_non_regular_triangle_with_similar_medians_as_sides_l1101_110168


namespace distance_from_B_l1101_110185

theorem distance_from_B (s y : ℝ) 
  (h1 : s^2 = 12)
  (h2 : ∀y, (1 / 2) * y^2 = 12 - y^2)
  (h3 : y = 2 * Real.sqrt 2)
: Real.sqrt ((2 * Real.sqrt 2)^2 + (2 * Real.sqrt 2)^2) = 4 := by
  sorry

end distance_from_B_l1101_110185


namespace least_sum_possible_l1101_110181

theorem least_sum_possible (x y z w k : ℕ) (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) 
  (hx : 4 * x = k) (hy : 5 * y = k) (hz : 6 * z = k) (hw : 7 * w = k) :
  x + y + z + w = 319 := 
  sorry

end least_sum_possible_l1101_110181


namespace bear_small_animal_weight_l1101_110111

theorem bear_small_animal_weight :
  let total_weight_needed := 1200
  let berries_weight := 1/5 * total_weight_needed
  let insects_weight := 1/10 * total_weight_needed
  let acorns_weight := 2 * berries_weight
  let honey_weight := 3 * insects_weight
  let total_weight_gained := berries_weight + insects_weight + acorns_weight + honey_weight
  let remaining_weight := total_weight_needed - total_weight_gained
  remaining_weight = 0 -> 0 = 0 := by
  intros total_weight_needed berries_weight insects_weight acorns_weight honey_weight
         total_weight_gained remaining_weight h
  exact Eq.refl 0

end bear_small_animal_weight_l1101_110111


namespace seq_fifth_term_l1101_110169

def seq (a : ℕ → ℤ) : Prop :=
  (a 1 = 3) ∧ (a 2 = 6) ∧ (∀ n : ℕ, a (n + 2) = a (n + 1) - a n)

theorem seq_fifth_term (a : ℕ → ℤ) (h : seq a) : a 5 = -6 :=
by
  sorry

end seq_fifth_term_l1101_110169
