import Mathlib

namespace fixed_salary_new_scheme_l110_110081

theorem fixed_salary_new_scheme :
  let old_commission_rate := 0.05
  let new_commission_rate := 0.025
  let sales_target := 4000
  let total_sales := 12000
  let remuneration_difference := 600
  let old_remuneration := old_commission_rate * total_sales
  let new_commission_earnings := new_commission_rate * (total_sales - sales_target)
  let new_remuneration := old_remuneration + remuneration_difference
  ∃ F, F + new_commission_earnings = new_remuneration :=
by
  sorry

end fixed_salary_new_scheme_l110_110081


namespace odds_against_C_winning_l110_110640

theorem odds_against_C_winning :
  let P_A := 2 / 7
  let P_B := 1 / 5
  let P_C := 1 - (P_A + P_B)
  (1 - P_C) / P_C = 17 / 18 :=
by
  sorry

end odds_against_C_winning_l110_110640


namespace condition_neither_sufficient_nor_necessary_l110_110748

noncomputable def f (x a : ℝ) : ℝ := x^3 - x + a
noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 1

def condition (a : ℝ) : Prop := a^2 - a = 0

theorem condition_neither_sufficient_nor_necessary
  (a : ℝ) :
  ¬(condition a → (∀ x : ℝ, f' x ≥ 0)) ∧ ¬((∀ x : ℝ, f' x ≥ 0) → condition a) :=
by
  sorry -- Proof is omitted as per the prompt

end condition_neither_sufficient_nor_necessary_l110_110748


namespace ball_speed_is_20_l110_110654

def ball_flight_time : ℝ := 8
def collie_speed : ℝ := 5
def collie_catch_time : ℝ := 32

noncomputable def collie_distance : ℝ := collie_speed * collie_catch_time

theorem ball_speed_is_20 :
  collie_distance = ball_flight_time * 20 :=
by
  sorry

end ball_speed_is_20_l110_110654


namespace intersection_P_Q_l110_110369

def P (x : ℝ) : Prop := x + 2 ≥ x^2

def Q (x : ℕ) : Prop := x ≤ 3

theorem intersection_P_Q :
  {x : ℕ | P x} ∩ {x : ℕ | Q x} = {0, 1, 2} :=
by
  sorry

end intersection_P_Q_l110_110369


namespace find_n_l110_110447

theorem find_n (n a b : ℕ) (h1 : n ≥ 2)
  (h2 : n = a^2 + b^2)
  (h3 : a = Nat.minFac n)
  (h4 : b ∣ n) : n = 8 ∨ n = 20 := 
sorry

end find_n_l110_110447


namespace initially_collected_oranges_l110_110719

-- Define the conditions from the problem
def oranges_eaten_by_father : ℕ := 2
def oranges_mildred_has_now : ℕ := 75

-- Define the proof problem (statement)
theorem initially_collected_oranges :
  (oranges_mildred_has_now + oranges_eaten_by_father = 77) :=
by 
  -- proof goes here
  sorry

end initially_collected_oranges_l110_110719


namespace shaded_areas_different_l110_110399

/-
Question: How do the shaded areas of three different large squares (I, II, and III) compare?
Conditions:
1. Square I has diagonals drawn, and small squares are shaded at each corner where diagonals meet the sides.
2. Square II has vertical and horizontal lines drawn through the midpoints, creating four smaller squares, with one centrally shaded.
3. Square III has one diagonal from one corner to the center and a straight line from the midpoint of the opposite side to the center, creating various triangles and trapezoids, with a trapezoid area around the center being shaded.
Proof:
Prove that the shaded areas of squares I, II, and III are all different given the conditions on how squares I, II, and III are partitioned and shaded.
-/
theorem shaded_areas_different :
  ∀ (a : ℝ) (A1 A2 A3 : ℝ), (A1 = 1/4 * a^2) ∧ (A2 = 1/4 * a^2) ∧ (A3 = 3/8 * a^2) → 
  A1 ≠ A3 ∧ A2 ≠ A3 :=
by
  sorry

end shaded_areas_different_l110_110399


namespace bus_ride_duration_l110_110063

theorem bus_ride_duration (total_hours : ℕ) (train_hours : ℕ) (walk_minutes : ℕ) (wait_factor : ℕ) 
    (h_total : total_hours = 8)
    (h_train : train_hours = 6)
    (h_walk : walk_minutes = 15)
    (h_wait : wait_factor = 2) : 
    let total_minutes := total_hours * 60
    let train_minutes := train_hours * 60
    let wait_minutes := wait_factor * walk_minutes
    let travel_minutes := total_minutes - train_minutes
    let bus_ride_minutes := travel_minutes - walk_minutes - wait_minutes
    bus_ride_minutes = 75 :=
by
  sorry

end bus_ride_duration_l110_110063


namespace one_fourths_in_five_eighths_l110_110453

theorem one_fourths_in_five_eighths : (5/8 : ℚ) / (1/4) = (5/2 : ℚ) := 
by
  -- Placeholder for the proof
  sorry

end one_fourths_in_five_eighths_l110_110453


namespace xy_sum_l110_110607

theorem xy_sum (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 44) : x + y = 2 :=
sorry

end xy_sum_l110_110607


namespace polynomial_factorization_l110_110108

theorem polynomial_factorization (m n : ℤ) (h₁ : (x^2 + m * x + 6 : ℤ) = (x - 2) * (x + n)) : m = -5 := by
  sorry

end polynomial_factorization_l110_110108


namespace unique_double_digit_in_range_l110_110185

theorem unique_double_digit_in_range (a b : ℕ) (h₁ : a = 10) (h₂ : b = 40) : 
  ∃! n : ℕ, (10 ≤ n ∧ n ≤ 40) ∧ (n % 10 = n / 10) ∧ (n % 10 = 3) :=
by {
  sorry
}

end unique_double_digit_in_range_l110_110185


namespace find_purchase_price_minimum_number_of_speed_skating_shoes_l110_110227

/-
A certain school in Zhangjiakou City is preparing to purchase speed skating shoes and figure skating shoes to promote ice and snow activities on campus.

If they buy 30 pairs of speed skating shoes and 20 pairs of figure skating shoes, the total cost is $8500.
If they buy 40 pairs of speed skating shoes and 10 pairs of figure skating shoes, the total cost is $8000.
The school purchases a total of 50 pairs of both types of ice skates, and the total cost does not exceed $8900.
-/

def price_system (x y : ℝ) : Prop :=
  30 * x + 20 * y = 8500 ∧ 40 * x + 10 * y = 8000

def minimum_speed_skating_shoes (x y m : ℕ) : Prop :=
  150 * m + 200 * (50 - m) ≤ 8900

theorem find_purchase_price :
  ∃ x y : ℝ, price_system x y ∧ x = 150 ∧ y = 200 :=
by
  /- Proof goes here -/
  sorry

theorem minimum_number_of_speed_skating_shoes :
  ∃ m, minimum_speed_skating_shoes 150 200 m ∧ m = 22 :=
by
  /- Proof goes here -/
  sorry

end find_purchase_price_minimum_number_of_speed_skating_shoes_l110_110227


namespace face_value_of_shares_l110_110715

/-- A company pays a 12.5% dividend to its investors. -/
def div_rate := 0.125

/-- An investor gets a 25% return on their investment. -/
def roi_rate := 0.25

/-- The investor bought the shares at Rs. 20 each. -/
def purchase_price := 20

theorem face_value_of_shares (FV : ℝ) (div_rate : ℝ) (roi_rate : ℝ) (purchase_price : ℝ) 
  (h1 : purchase_price * roi_rate = div_rate * FV) : FV = 40 :=
by sorry

end face_value_of_shares_l110_110715


namespace number_of_sections_l110_110213

noncomputable def initial_rope : ℕ := 50
noncomputable def rope_for_art := initial_rope / 5
noncomputable def remaining_rope_after_art := initial_rope - rope_for_art
noncomputable def rope_given_to_friend := remaining_rope_after_art / 2
noncomputable def remaining_rope := remaining_rope_after_art - rope_given_to_friend
noncomputable def section_size : ℕ := 2
noncomputable def sections := remaining_rope / section_size

theorem number_of_sections : sections = 10 :=
by
  sorry

end number_of_sections_l110_110213


namespace li_ming_estimated_weight_is_correct_l110_110642

-- Define the regression equation as a function
def regression_equation (x : ℝ) : ℝ := 0.7 * x - 52

-- Define the height of Li Ming
def li_ming_height : ℝ := 180

-- The estimated weight according to the regression equation
def estimated_weight : ℝ := regression_equation li_ming_height

-- Theorem statement: Given the height, the weight should be 74
theorem li_ming_estimated_weight_is_correct : estimated_weight = 74 :=
by
  sorry

end li_ming_estimated_weight_is_correct_l110_110642


namespace problem_inequality_l110_110708

theorem problem_inequality (a b c : ℝ) (h₀ : a + b + c = 0) (d : ℝ) (h₁ : d = max (|a|) (max (|b|) (|c|))) : 
  |(1 + a) * (1 + b) * (1 + c)| ≥ 1 - d^2 :=
sorry

end problem_inequality_l110_110708


namespace factorize1_factorize2_factorize3_factorize4_l110_110886

-- 1. Factorize 3x - 12x^3
theorem factorize1 (x : ℝ) : 3 * x - 12 * x^3 = 3 * x * (1 - 2 * x) * (1 + 2 * x) := 
sorry

-- 2. Factorize 9m^2 - 4n^2
theorem factorize2 (m n : ℝ) : 9 * m^2 - 4 * n^2 = (3 * m + 2 * n) * (3 * m - 2 * n) := 
sorry

-- 3. Factorize a^2(x - y) + b^2(y - x)
theorem factorize3 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a + b) * (a - b) := 
sorry

-- 4. Factorize x^2 - 4xy + 4y^2 - 1
theorem factorize4 (x y : ℝ) : x^2 - 4 * x * y + 4 * y^2 - 1 = (x - y + 1) * (x - y - 1) := 
sorry

end factorize1_factorize2_factorize3_factorize4_l110_110886


namespace add_gold_coins_l110_110066

open Nat

theorem add_gold_coins (G S X : ℕ) 
  (h₁ : G = S / 3) 
  (h₂ : (G + X) / S = 1 / 2) 
  (h₃ : G + X + S = 135) : 
  X = 15 := 
sorry

end add_gold_coins_l110_110066


namespace multiples_of_2_correct_multiples_of_3_correct_l110_110692

def numbers : Set ℕ := {28, 35, 40, 45, 53, 10, 78}

def multiples_of_2_in_numbers : Set ℕ := {n ∈ numbers | n % 2 = 0}
def multiples_of_3_in_numbers : Set ℕ := {n ∈ numbers | n % 3 = 0}

theorem multiples_of_2_correct :
  multiples_of_2_in_numbers = {28, 40, 10, 78} :=
sorry

theorem multiples_of_3_correct :
  multiples_of_3_in_numbers = {45, 78} :=
sorry

end multiples_of_2_correct_multiples_of_3_correct_l110_110692


namespace sufficient_but_not_necessary_condition_l110_110076

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 3 → x^2 - 2 * x > 0) ∧ ¬ (x^2 - 2 * x > 0 → x > 3) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l110_110076


namespace sequence_ln_l110_110974

theorem sequence_ln (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 + Real.log n := 
sorry

end sequence_ln_l110_110974


namespace arithmetic_mean_of_fractions_l110_110171

theorem arithmetic_mean_of_fractions :
  (1 / 2 : ℚ) * ((2 / 5) + (4 / 7)) = 17 / 35 :=
by
  sorry

end arithmetic_mean_of_fractions_l110_110171


namespace num_statements_imply_impl_l110_110588

variable (p q r : Prop)

def cond1 := p ∧ q ∧ ¬r
def cond2 := ¬p ∧ q ∧ r
def cond3 := p ∧ q ∧ r
def cond4 := ¬p ∧ ¬q ∧ ¬r

def impl := ((p → ¬q) → ¬r)

theorem num_statements_imply_impl : 
  (cond1 p q r → impl p q r) ∧ 
  (cond3 p q r → impl p q r) ∧ 
  (cond4 p q r → impl p q r) ∧ 
  ¬(cond2 p q r → impl p q r) :=
by {
  sorry
}

end num_statements_imply_impl_l110_110588


namespace variance_eta_l110_110557

noncomputable def xi : ℝ := sorry -- Define ξ as a real number (will be specified later)
noncomputable def eta : ℝ := sorry -- Define η as a real number (will be specified later)

-- Conditions
axiom xi_distribution : xi = 3 + 2*Real.sqrt 4 -- ξ follows a normal distribution with mean 3 and variance 4
axiom relationship : xi = 2*eta + 3 -- Given relationship between ξ and η

-- Theorem to prove the question
theorem variance_eta : sorry := sorry

end variance_eta_l110_110557


namespace find_constant_c_l110_110651

def f: ℝ → ℝ := sorry

noncomputable def constant_c := 8

theorem find_constant_c (h : ∀ x : ℝ, f x + 3 * f (constant_c - x) = x) (h2 : f 2 = 2) : 
  constant_c = 8 :=
sorry

end find_constant_c_l110_110651


namespace Rogers_expense_fraction_l110_110117

variables (B m s p : ℝ)

theorem Rogers_expense_fraction (h1 : m = 0.25 * (B - s))
                              (h2 : s = 0.10 * (B - m))
                              (h3 : p = 0.10 * (m + s)) :
  m + s + p = 0.34 * B :=
by
  sorry

end Rogers_expense_fraction_l110_110117


namespace conic_sections_of_equation_l110_110778

theorem conic_sections_of_equation :
  (∀ x y : ℝ, y^6 - 6 * x^6 = 3 * y^2 - 8 → y^2 = 6 * x^2 ∨ y^2 = -6 * x^2 + 2) :=
sorry

end conic_sections_of_equation_l110_110778


namespace complete_the_square_l110_110531

theorem complete_the_square (x : ℝ) : 
  (∃ a b : ℝ, (x + a) ^ 2 = b ∧ b = 11 ∧ a = -4) ↔ (x ^ 2 - 8 * x + 5 = 0) :=
by
  sorry

end complete_the_square_l110_110531


namespace pradeep_failed_by_25_marks_l110_110449

theorem pradeep_failed_by_25_marks :
  (35 / 100 * 600 : ℝ) - 185 = 25 :=
by
  sorry

end pradeep_failed_by_25_marks_l110_110449


namespace incorrect_option_C_l110_110304

theorem incorrect_option_C (a b d : ℝ) (h₁ : ∀ x : ℝ, x ≠ d → x^2 + a * x + b > 0) (h₂ : a > 0) :
  ¬∀ x₁ x₂ : ℝ, (x₁ * x₂ > 0) → ((x₁, x₂) ∈ {p : (ℝ × ℝ) | p.1^2 + a * p.1 - b < 0 ∧ p.2^2 + a * p.2 - b < 0}) :=
sorry

end incorrect_option_C_l110_110304


namespace necessary_but_not_sufficient_l110_110480

def lines_parallel (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a * x + 2 * y = 0) ↔ (x + (a + 1) * y + 4 = 0)

theorem necessary_but_not_sufficient (a : ℝ) :
  (a = 1 → lines_parallel a) ∧ ¬(lines_parallel a → a = 1) :=
by
  sorry

end necessary_but_not_sufficient_l110_110480


namespace no_real_solution_l110_110626

theorem no_real_solution :
  ¬ ∃ x : ℝ, 7 * (4 * x + 3) - 4 = -3 * (2 - 9 * x^2) :=
by
  sorry

end no_real_solution_l110_110626


namespace family_of_sets_properties_l110_110307

variable {X : Type}
variable {t n k : ℕ}
variable (A : Fin t → Set X)
variable (card : Set X → ℕ)
variable (h_card : ∀ (i j : Fin t), i ≠ j → card (A i ∩ A j) = k)

theorem family_of_sets_properties :
  (k = 0 → t ≤ n+1) ∧ (k ≠ 0 → t ≤ n) :=
by
  sorry

end family_of_sets_properties_l110_110307


namespace smallest_value_in_interval_l110_110832

open Real

noncomputable def smallest_value (x : ℝ) (h : 1 < x ∧ x < 2) : Prop :=
  1 / x^2 < x ∧
  1 / x^2 < x^2 ∧
  1 / x^2 < 2 * x^2 ∧
  1 / x^2 < 3 * x ∧
  1 / x^2 < sqrt x ∧
  1 / x^2 < 1 / x

theorem smallest_value_in_interval (x : ℝ) (h : 1 < x ∧ x < 2) : smallest_value x h :=
by
  sorry

end smallest_value_in_interval_l110_110832


namespace regular_polygon_sides_l110_110655

theorem regular_polygon_sides (N : ℕ) (h : ∀ θ, θ = 140 → N * (180 -θ) = 360) : N = 9 :=
by
  sorry

end regular_polygon_sides_l110_110655


namespace smaller_root_of_equation_l110_110087

theorem smaller_root_of_equation :
  ∀ x : ℚ, (x - 7 / 8)^2 + (x - 1/4) * (x - 7 / 8) = 0 → x = 9 / 16 :=
by
  intro x
  intro h
  sorry

end smaller_root_of_equation_l110_110087


namespace pipe_filling_time_l110_110402

theorem pipe_filling_time (t : ℕ) (h : 2 * (1 / t + 1 / 15) + 10 * (1 / 15) = 1) : t = 10 := by
  sorry

end pipe_filling_time_l110_110402


namespace train_leave_tunnel_l110_110899

noncomputable def train_leave_time 
  (train_speed : ℝ) 
  (tunnel_length : ℝ) 
  (train_length : ℝ) 
  (enter_time : ℝ × ℝ) : ℝ × ℝ :=
  let speed_km_min := train_speed / 60
  let total_distance := train_length + tunnel_length
  let time_to_pass := total_distance / speed_km_min
  let enter_minutes := enter_time.1 * 60 + enter_time.2
  let leave_minutes := enter_minutes + time_to_pass
  let leave_hours := leave_minutes / 60
  let leave_remainder_minutes := leave_minutes % 60
  (leave_hours, leave_remainder_minutes)

theorem train_leave_tunnel : 
  train_leave_time 80 70 1 (5, 12) = (6, 5.25) := 
sorry

end train_leave_tunnel_l110_110899


namespace temperature_on_Friday_l110_110913

-- Definitions of the temperatures on the days
variables {M T W Th F : ℝ}

-- Conditions given in the problem
def avg_temp_mon_thu (M T W Th : ℝ) : Prop := (M + T + W + Th) / 4 = 48
def avg_temp_tue_fri (T W Th F : ℝ) : Prop := (T + W + Th + F) / 4 = 46
def temp_mon (M : ℝ) : Prop := M = 44

-- Statement to prove
theorem temperature_on_Friday (h1 : avg_temp_mon_thu M T W Th)
                               (h2 : avg_temp_tue_fri T W Th F)
                               (h3 : temp_mon M) : F = 36 :=
sorry

end temperature_on_Friday_l110_110913


namespace min_perimeter_triangle_l110_110394

theorem min_perimeter_triangle (a b c : ℝ) (cosC : ℝ) :
  a + b = 10 ∧ cosC = -1/2 ∧ c^2 = (a - 5)^2 + 75 →
  a + b + c = 10 + 5 * Real.sqrt 3 :=
by
  sorry

end min_perimeter_triangle_l110_110394


namespace group_friends_opponents_l110_110099

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end group_friends_opponents_l110_110099


namespace profit_ratio_l110_110459

variables (P_s : ℝ)

theorem profit_ratio (h1 : 21 * (7 / 3) + 3 * P_s = 175) : P_s / 21 = 2 :=
by
  sorry

end profit_ratio_l110_110459


namespace sphere_center_plane_intersection_l110_110597

theorem sphere_center_plane_intersection
  (d e f : ℝ)
  (O : ℝ × ℝ × ℝ := (0, 0, 0))
  (A B C : ℝ × ℝ × ℝ)
  (p : ℝ)
  (hA : A ≠ O)
  (hB : B ≠ O)
  (hC : C ≠ O)
  (hA_coord : A = (2 * p, 0, 0))
  (hB_coord : B = (0, 2 * p, 0))
  (hC_coord : C = (0, 0, 2 * p))
  (h_sphere : (p, p, p) = (p, p, p)) -- we know that the center is (p, p, p)
  (h_plane : d * (1 / (2 * p)) + e * (1 / (2 * p)) + f * (1 / (2 * p)) = 1) :
  d / p + e / p + f / p = 2 := sorry

end sphere_center_plane_intersection_l110_110597


namespace denominator_of_second_fraction_l110_110085

theorem denominator_of_second_fraction (y x : ℝ) (h_cond : y > 0) (h_eq : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 :=
sorry

end denominator_of_second_fraction_l110_110085


namespace jerry_mowing_income_l110_110890

theorem jerry_mowing_income (M : ℕ) (week_spending : ℕ) (money_weed_eating : ℕ) (weeks : ℕ)
  (H1 : week_spending = 5)
  (H2 : money_weed_eating = 31)
  (H3 : weeks = 9)
  (H4 : (M + money_weed_eating) = week_spending * weeks)
  : M = 14 :=
by {
  sorry
}

end jerry_mowing_income_l110_110890


namespace tank_full_weight_l110_110727

theorem tank_full_weight (u v m n : ℝ) (h1 : m + 3 / 4 * n = u) (h2 : m + 1 / 3 * n = v) :
  m + n = 8 / 5 * u - 3 / 5 * v :=
sorry

end tank_full_weight_l110_110727


namespace triangle_inequality_proof_l110_110538

theorem triangle_inequality_proof (a b c : ℝ) (h : a + b > c) : a^3 + b^3 + 3 * a * b * c > c^3 :=
by sorry

end triangle_inequality_proof_l110_110538


namespace at_least_one_gt_one_l110_110504

theorem at_least_one_gt_one (x y : ℝ) (h : x + y > 2) : ¬(x > 1 ∨ y > 1) → (x ≤ 1 ∧ y ≤ 1) := 
by
  sorry

end at_least_one_gt_one_l110_110504


namespace combined_capacity_eq_l110_110505

variable {x y z : ℚ}

-- Container A condition
def containerA_full (x : ℚ) := 0.75 * x
def containerA_initial (x : ℚ) := 0.30 * x
def containerA_diff (x : ℚ) := containerA_full x - containerA_initial x = 36

-- Container B condition
def containerB_full (y : ℚ) := 0.70 * y
def containerB_initial (y : ℚ) := 0.40 * y
def containerB_diff (y : ℚ) := containerB_full y - containerB_initial y = 20

-- Container C condition
def containerC_full (z : ℚ) := (2 / 3) * z
def containerC_initial (z : ℚ) := 0.50 * z
def containerC_diff (z : ℚ) := containerC_full z - containerC_initial z = 12

-- Theorem to prove the total capacity
theorem combined_capacity_eq : containerA_diff x → containerB_diff y → containerC_diff z → 
(218 + 2 / 3 = x + y + z) :=
by
  intros hA hB hC
  sorry

end combined_capacity_eq_l110_110505


namespace expression_value_l110_110839

theorem expression_value (x : ℕ) (h : x = 12) : (3 / 2 * x - 3 : ℚ) = 15 := by
  rw [h]
  norm_num
-- sorry to skip the proof if necessary
-- sorry 

end expression_value_l110_110839


namespace lewis_speed_is_90_l110_110755

noncomputable def david_speed : ℝ := 50 -- mph
noncomputable def distance_chennai_hyderabad : ℝ := 350 -- miles
noncomputable def distance_meeting_point : ℝ := 250 -- miles

theorem lewis_speed_is_90 :
  ∃ L : ℝ, 
    (∀ t : ℝ, david_speed * t = distance_meeting_point) →
    (∀ t : ℝ, L * t = (distance_chennai_hyderabad + (distance_meeting_point - distance_chennai_hyderabad))) →
    L = 90 :=
by
  sorry

end lewis_speed_is_90_l110_110755


namespace paid_amount_divisible_by_11_l110_110175

-- Define the original bill amount and the increased bill amount
def original_bill (x : ℕ) : ℕ := x
def paid_amount (x : ℕ) : ℕ := (11 * x) / 10

-- Theorem: The paid amount is divisible by 11
theorem paid_amount_divisible_by_11 (x : ℕ) (h : x % 10 = 0) : paid_amount x % 11 = 0 :=
by
  sorry

end paid_amount_divisible_by_11_l110_110175


namespace exists_y_equals_7_l110_110260

theorem exists_y_equals_7 : ∃ (x y z t : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 ∧ y = 7 ∧ x + y + z + t = 10 :=
by {
  sorry -- This is where the actual proof would go.
}

end exists_y_equals_7_l110_110260


namespace diagonals_in_polygon_l110_110994

-- Define the number of sides of the polygon
def n : ℕ := 30

-- Define the formula for the total number of diagonals in an n-sided polygon
def total_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Define the number of excluded diagonals for being parallel to one given side
def excluded_diagonals : ℕ := 1

-- Define the final count of valid diagonals after exclusion
def valid_diagonals : ℕ := total_diagonals n - excluded_diagonals

-- State the theorem to prove
theorem diagonals_in_polygon : valid_diagonals = 404 := by
  sorry


end diagonals_in_polygon_l110_110994


namespace duckweed_quarter_covered_l110_110077

theorem duckweed_quarter_covered (N : ℕ) (h1 : N = 64) (h2 : ∀ n : ℕ, n < N → (n + 1 < N) → ∃ k, k = n + 1) :
  N - 2 = 62 :=
by
  sorry

end duckweed_quarter_covered_l110_110077


namespace number_of_female_only_child_students_l110_110133

def students : Finset ℕ := Finset.range 21 -- Set of students with attendance numbers from 1 to 20

def female_students : Finset ℕ := {1, 3, 4, 6, 7, 10, 11, 13, 16, 17, 18, 20}

def only_child_students : Finset ℕ := {1, 4, 5, 8, 11, 14, 17, 20}

def common_students : Finset ℕ := female_students ∩ only_child_students

theorem number_of_female_only_child_students :
  common_students.card = 5 :=
by
  sorry

end number_of_female_only_child_students_l110_110133


namespace num_clients_visited_garage_l110_110935

theorem num_clients_visited_garage :
  ∃ (num_clients : ℕ), num_clients = 24 ∧
    ∀ (num_cars selections_per_car selections_per_client : ℕ),
        num_cars = 16 → selections_per_car = 3 → selections_per_client = 2 →
        (num_cars * selections_per_car) / selections_per_client = num_clients :=
by
  sorry

end num_clients_visited_garage_l110_110935


namespace cards_relationship_l110_110211

-- Definitions from the conditions given in the problem
variables (x y : ℕ)

-- Theorem statement proving the relationship
theorem cards_relationship (h : x + y = 8 * x) : y = 7 * x :=
sorry

end cards_relationship_l110_110211


namespace min_distance_to_line_value_of_AB_l110_110525

noncomputable def point_B : ℝ × ℝ := (1, 1)
noncomputable def point_A : ℝ × ℝ := (4 * Real.sqrt 2, Real.pi / 4)

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

noncomputable def polar_line_l (a : ℝ) (θ : ℝ) : ℝ :=
  a * Real.cos (θ - Real.pi / 4)

noncomputable def line_l1 (m : ℝ) (x y : ℝ) : Prop :=
  x + y + m = 0

theorem min_distance_to_line {θ : ℝ} (a : ℝ) :
  polar_line_l a θ = 4 * Real.sqrt 2 → 
  ∃ d, d = (8 * Real.sqrt 2 - Real.sqrt 14) / 2 :=
by
  sorry

theorem value_of_AB :
  ∃ AB, AB = 12 * Real.sqrt 2 / 7 :=
by
  sorry

end min_distance_to_line_value_of_AB_l110_110525


namespace find_m_for_root_l110_110223

-- Define the fractional equation to find m
def fractional_equation (x m : ℝ) : Prop :=
  (x + 2) / (x - 1) = m / (1 - x)

-- State the theorem that we need to prove
theorem find_m_for_root : ∃ m : ℝ, (∃ x : ℝ, fractional_equation x m) ∧ m = -3 :=
by
  sorry

end find_m_for_root_l110_110223


namespace average_percent_score_l110_110667

theorem average_percent_score :
    let students := 120
    let score_95 := 95 * 12
    let score_85 := 85 * 24
    let score_75 := 75 * 30
    let score_65 := 65 * 20
    let score_55 := 55 * 18
    let score_45 := 45 * 10
    let score_35 := 35 * 6
    let total_score := score_95 + score_85 + score_75 + score_65 + score_55 + score_45 + score_35
    (total_score.toFloat / students.toFloat) = 69.8333 :=
by
  sorry

end average_percent_score_l110_110667


namespace cube_face_expression_l110_110928

theorem cube_face_expression (a b c : ℤ) (h1 : 3 * a + 2 = 17) (h2 : 7 * b - 4 = 10) (h3 : a + 3 * b - 2 * c = 11) : 
  a - b * c = 5 :=
by sorry

end cube_face_expression_l110_110928


namespace Jenny_older_than_Rommel_l110_110931

theorem Jenny_older_than_Rommel :
  ∃ t r j, t = 5 ∧ r = 3 * t ∧ j = t + 12 ∧ (j - r = 2) := 
by
  -- We insert the proof here using sorry to skip the actual proof part.
  sorry

end Jenny_older_than_Rommel_l110_110931


namespace mod_arith_proof_l110_110014

theorem mod_arith_proof (m : ℕ) (hm1 : 0 ≤ m) (hm2 : m < 50) : 198 * 935 % 50 = 30 := 
by
  sorry

end mod_arith_proof_l110_110014


namespace second_race_distance_l110_110578

theorem second_race_distance (Va Vb Vc : ℝ) (D : ℝ)
  (h1 : Va / Vb = 10 / 9)
  (h2 : Va / Vc = 80 / 63)
  (h3 : Vb / Vc = D / (D - 100)) :
  D = 800 :=
sorry

end second_race_distance_l110_110578


namespace sum_of_fourth_powers_l110_110646

theorem sum_of_fourth_powers (n : ℤ) 
  (h : n * (n + 1) * (n + 2) = 12 * (n + (n + 1) + (n + 2))) : 
  (n^4 + (n + 1)^4 + (n + 2)^4) = 7793 := 
by 
  sorry

end sum_of_fourth_powers_l110_110646


namespace current_bottle_caps_l110_110617

def initial_bottle_caps : ℕ := 91
def lost_bottle_caps : ℕ := 66

theorem current_bottle_caps : initial_bottle_caps - lost_bottle_caps = 25 :=
by
  -- sorry is used to skip the proof
  sorry

end current_bottle_caps_l110_110617


namespace b2_b7_product_l110_110017

variable {b : ℕ → ℤ}

-- Define the conditions: b is an arithmetic sequence and b_4 * b_5 = 15
def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

axiom increasing_arithmetic_sequence : is_arithmetic_sequence b
axiom b4_b5_product : b 4 * b 5 = 15

-- The target theorem to prove
theorem b2_b7_product : b 2 * b 7 = -9 :=
sorry

end b2_b7_product_l110_110017


namespace inequality_example_l110_110161

theorem inequality_example (a b c : ℝ) : a^2 + 4 * b^2 + 9 * c^2 ≥ 2 * a * b + 3 * a * c + 6 * b * c :=
by
  sorry

end inequality_example_l110_110161


namespace sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l110_110333

theorem sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine : 
  Real.sqrt (3^3 + 3^3 + 3^3) = 9 :=
by 
  sorry

end sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l110_110333


namespace quadratic_solutions_1_quadratic_k_value_and_solutions_l110_110463

-- Problem (Ⅰ):
theorem quadratic_solutions_1 {x : ℝ} :
  x^2 + 6 * x + 5 = 0 ↔ x = -5 ∨ x = -1 :=
sorry

-- Problem (Ⅱ):
theorem quadratic_k_value_and_solutions {x k : ℝ} (x1 x2 : ℝ) :
  x1 + x2 = 3 ∧ x1 * x2 = k ∧ (x1 - 1) * (x2 - 1) = -6 ↔ (k = -4 ∧ (x = 4 ∨ x = -1)) :=
sorry

end quadratic_solutions_1_quadratic_k_value_and_solutions_l110_110463


namespace fixed_point_exists_l110_110694

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(a * (x + 1)) - 3

theorem fixed_point_exists (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = -1 :=
by
  -- Sorry for skipping the proof
  sorry

end fixed_point_exists_l110_110694


namespace length_diff_width_8m_l110_110122

variables (L W : ℝ)

theorem length_diff_width_8m (h1: W = (1/2) * L) (h2: L * W = 128) : L - W = 8 :=
by sorry

end length_diff_width_8m_l110_110122


namespace examination_total_students_l110_110043

theorem examination_total_students (T : ℝ) :
  (0.35 * T + 520) = T ↔ T = 800 :=
by 
  sorry

end examination_total_students_l110_110043


namespace geometry_problem_z_eq_87_deg_l110_110582

noncomputable def measure_angle_z (ABC ABD ADB : Real) : Real :=
  43 -- \angle ADB

theorem geometry_problem_z_eq_87_deg
  (ABC : Real)
  (h1 : ABC = 130)
  (ABD : Real)
  (h2 : ABD = 50)
  (ADB : Real)
  (h3 : ADB = 43) :
  measure_angle_z ABC ABD ADB = 87 :=
by
  unfold measure_angle_z
  sorry

end geometry_problem_z_eq_87_deg_l110_110582


namespace regular_polygon_sides_l110_110214

theorem regular_polygon_sides (θ : ℝ) (n : ℕ) (h1 : θ = 18) (h2 : 360 = θ * n) : n = 20 :=
by {
  sorry
}

end regular_polygon_sides_l110_110214


namespace pieces_bound_l110_110327

open Finset

variable {n : ℕ} (B W : ℕ)

theorem pieces_bound (n : ℕ) (B W : ℕ) (hB : B ≤ n^2) (hW : W ≤ n^2) :
    B ≤ n^2 ∨ W ≤ n^2 := 
by
  sorry

end pieces_bound_l110_110327


namespace skate_time_correct_l110_110208

noncomputable def skate_time (path_length miles_length : ℝ) (skating_speed : ℝ) : ℝ :=
  let time_taken := (1.58 * Real.pi) / skating_speed
  time_taken

theorem skate_time_correct :
  skate_time 1 1 4 = 1.58 * Real.pi / 4 :=
by
  sorry

end skate_time_correct_l110_110208


namespace wage_constraint_l110_110554

/-- Wage constraints for hiring carpenters and tilers given a budget -/
theorem wage_constraint (x y : ℕ) (h_carpenter_wage : 50 * x + 40 * y = 2000) : 5 * x + 4 * y = 200 := by
  sorry

end wage_constraint_l110_110554


namespace fraction_meaningful_l110_110497

theorem fraction_meaningful (x : ℝ) : x - 5 ≠ 0 ↔ x ≠ 5 := 
by 
  sorry

end fraction_meaningful_l110_110497


namespace proof_problem_l110_110790

noncomputable def A := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
noncomputable def B := {(x, y) : ℝ × ℝ | y = x^2 + 1}

theorem proof_problem :
  ((1, 2) ∈ B) ∧
  (0 ∉ A) ∧
  ((0, 0) ∉ B) :=
by
  sorry

end proof_problem_l110_110790


namespace in_proportion_d_value_l110_110568

noncomputable def d_length (a b c : ℝ) : ℝ := (b * c) / a

theorem in_proportion_d_value :
  let a := 2
  let b := 3
  let c := 6
  d_length a b c = 9 := 
by
  sorry

end in_proportion_d_value_l110_110568


namespace laundry_loads_needed_l110_110744

theorem laundry_loads_needed
  (families : ℕ) (people_per_family : ℕ)
  (towels_per_person_per_day : ℕ) (days : ℕ)
  (washing_machine_capacity : ℕ)
  (h_f : families = 7)
  (h_p : people_per_family = 6)
  (h_t : towels_per_person_per_day = 2)
  (h_d : days = 10)
  (h_w : washing_machine_capacity = 10) : 
  ((families * people_per_family * towels_per_person_per_day * days) / washing_machine_capacity) = 84 := 
by
  sorry

end laundry_loads_needed_l110_110744


namespace artist_paint_usage_l110_110951

def ounces_of_paint_used (extra_large: ℕ) (large: ℕ) (medium: ℕ) (small: ℕ) : ℕ :=
  4 * extra_large + 3 * large + 2 * medium + 1 * small

theorem artist_paint_usage : ounces_of_paint_used 3 5 6 8 = 47 := by
  sorry

end artist_paint_usage_l110_110951


namespace john_using_three_colors_l110_110487

theorem john_using_three_colors {total_paint liters_per_color : ℕ} 
    (h1 : total_paint = 15) 
    (h2 : liters_per_color = 5) :
    total_ppaint / liters_per_color = 3 := 
by
  sorry

end john_using_three_colors_l110_110487


namespace sum_first_7_terms_is_105_l110_110330

-- Define an arithmetic sequence
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Define the sum of the first n terms of an arithmetic sequence
def sum_arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

-- Given conditions
variables {a d : ℕ}
axiom a4_is_15 : arithmetic_seq a d 4 = 15

-- Goal/theorem to be proven
theorem sum_first_7_terms_is_105 : sum_arithmetic_seq a d 7 = 105 :=
sorry

end sum_first_7_terms_is_105_l110_110330


namespace no_real_solution_l110_110273

theorem no_real_solution (x : ℝ) : ¬ (x^3 + 2 * (x + 1)^3 + 3 * (x + 2)^3 = 6 * (x + 4)^3) :=
sorry

end no_real_solution_l110_110273


namespace units_digit_of_7_power_exp_is_1_l110_110411

-- Define the periodicity of units digits of powers of 7
def units_digit_seq : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit_power_7 (n : ℕ) : ℕ :=
  units_digit_seq.get! (n % 4)

-- Define the exponent
def exp : ℕ := 8^5

-- Define the modular operation result
def exp_modulo : ℕ := exp % 4

-- Define the main statement
theorem units_digit_of_7_power_exp_is_1 :
  units_digit_power_7 exp = 1 :=
by
  simp [units_digit_power_7, units_digit_seq, exp, exp_modulo]
  sorry

end units_digit_of_7_power_exp_is_1_l110_110411


namespace evaluate_expression_l110_110466

-- Define the ceiling of square roots for the given numbers
def ceil_sqrt_3 := 2
def ceil_sqrt_27 := 6
def ceil_sqrt_243 := 16

-- Main theorem statement
theorem evaluate_expression :
  ceil_sqrt_3 + ceil_sqrt_27 * 2 + ceil_sqrt_243 = 30 :=
by
  -- Sorry to indicate that the proof is skipped
  sorry

end evaluate_expression_l110_110466


namespace least_clock_equivalent_hour_l110_110537

theorem least_clock_equivalent_hour (h : ℕ) (h_gt_9 : h > 9) (clock_equiv : (h^2 - h) % 12 = 0) : h = 13 :=
sorry

end least_clock_equivalent_hour_l110_110537


namespace value_of_T_l110_110496

theorem value_of_T (S : ℝ) (T : ℝ) (h1 : (1/4) * (1/6) * T = (1/2) * (1/8) * S) (h2 : S = 64) : T = 96 := 
by 
  sorry

end value_of_T_l110_110496


namespace problem_statements_correctness_l110_110847

theorem problem_statements_correctness :
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (12 ∣ 72 ∧ 12 ∣ 120) ∧ (7 ∣ 49 ∧ 7 ∣ 84) ∧ (7 ∣ 63) → 
  (3 ∣ 15) ∧ (11 ∣ 121 ∧ ¬(11 ∣ 60)) ∧ (7 ∣ 63) :=
by
  intro h
  sorry

end problem_statements_correctness_l110_110847


namespace abs_eq_iff_x_eq_2_l110_110514

theorem abs_eq_iff_x_eq_2 (x : ℝ) : |x - 1| = |x - 3| → x = 2 := by
  sorry

end abs_eq_iff_x_eq_2_l110_110514


namespace inversely_proportional_solve_y_l110_110552

theorem inversely_proportional_solve_y (k : ℝ) (x y : ℝ)
  (h1 : x * y = k)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (hx : x = -10) :
  y = -67.5 :=
by
  sorry

end inversely_proportional_solve_y_l110_110552


namespace asia_fraction_correct_l110_110773

-- Define the problem conditions
def fraction_NA (P : ℕ) : ℚ := 1/3 * P
def fraction_Europe (P : ℕ) : ℚ := 1/8 * P
def fraction_Africa (P : ℕ) : ℚ := 1/5 * P
def others : ℕ := 42
def total_passengers : ℕ := 240

-- Define the target fraction for Asia
def fraction_Asia (P: ℕ) : ℚ := 17 / 120

-- Theorem: the fraction of the passengers from Asia equals 17/120
theorem asia_fraction_correct : ∀ (P : ℕ), 
  P = total_passengers →
  fraction_NA P + fraction_Europe P + fraction_Africa P + fraction_Asia P * P + others = P →
  fraction_Asia P = 17 / 120 := 
by sorry

end asia_fraction_correct_l110_110773


namespace molecular_weight_calculated_l110_110441

def atomic_weight_Ba : ℚ := 137.33
def atomic_weight_O  : ℚ := 16.00
def atomic_weight_H  : ℚ := 1.01

def molecular_weight_compound : ℚ :=
  (1 * atomic_weight_Ba) + (2 * atomic_weight_O) + (2 * atomic_weight_H)

theorem molecular_weight_calculated :
  molecular_weight_compound = 171.35 :=
by {
  sorry
}

end molecular_weight_calculated_l110_110441


namespace sunscreen_cost_l110_110252

theorem sunscreen_cost (reapply_time : ℕ) (oz_per_application : ℕ) 
  (oz_per_bottle : ℕ) (cost_per_bottle : ℝ) (total_time : ℕ) (expected_cost : ℝ) :
  reapply_time = 2 →
  oz_per_application = 3 →
  oz_per_bottle = 12 →
  cost_per_bottle = 3.5 →
  total_time = 16 →
  expected_cost = 7 →
  (total_time / reapply_time) * (oz_per_application / oz_per_bottle) * cost_per_bottle = expected_cost :=
by
  intros
  sorry

end sunscreen_cost_l110_110252


namespace commercial_break_duration_l110_110814

theorem commercial_break_duration (n1 n2 m1 m2 : ℕ) (h1 : n1 = 3) (h2 : m1 = 5) (h3 : n2 = 11) (h4 : m2 = 2) :
  n1 * m1 + n2 * m2 = 37 :=
by
  -- Here, in a real proof, we would substitute and show the calculations.
  sorry

end commercial_break_duration_l110_110814


namespace find_value_of_square_sums_l110_110635

variable (x y z : ℝ)

-- Define the conditions
def weighted_arithmetic_mean := (2 * x + 2 * y + 3 * z) / 8 = 9
def weighted_geometric_mean := Real.rpow (x^2 * y^2 * z^3) (1 / 7) = 6
def weighted_harmonic_mean := 7 / ((2 / x) + (2 / y) + (3 / z)) = 4

-- State the theorem to be proved
theorem find_value_of_square_sums
  (h1 : weighted_arithmetic_mean x y z)
  (h2 : weighted_geometric_mean x y z)
  (h3 : weighted_harmonic_mean x y z) :
  x^2 + y^2 + z^2 = 351 :=
by sorry

end find_value_of_square_sums_l110_110635


namespace expand_expression_l110_110073

theorem expand_expression (x : ℝ) : 
  (11 * x^2 + 5 * x - 3) * (3 * x^3) = 33 * x^5 + 15 * x^4 - 9 * x^3 :=
by 
  sorry

end expand_expression_l110_110073


namespace remainder_when_divided_by_x_minus_3_l110_110276

def p (x : ℝ) : ℝ := x^4 - 4 * x^2 + 7

theorem remainder_when_divided_by_x_minus_3 : p 3 = 52 := 
by
  -- proof here
  sorry

end remainder_when_divided_by_x_minus_3_l110_110276


namespace kangaroo_chase_l110_110674

noncomputable def time_to_catch_up (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
  (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
  (initial_baby_jumps: ℕ): ℕ :=
  let jump_dist_baby := jump_dist_mother / jump_dist_reduction_factor
  let distance_mother := jumps_mother * jump_dist_mother
  let distance_baby := jumps_baby * jump_dist_baby
  let relative_velocity := distance_mother - distance_baby
  let initial_distance := initial_baby_jumps * jump_dist_baby
  (initial_distance / relative_velocity) * time_period

theorem kangaroo_chase :
 ∀ (jumps_baby: ℕ) (jumps_mother: ℕ) (time_period: ℕ) 
   (jump_dist_mother: ℕ) (jump_dist_reduction_factor: ℕ) 
   (initial_baby_jumps: ℕ),
  jumps_baby = 5 ∧ jumps_mother = 3 ∧ time_period = 2 ∧ 
  jump_dist_mother = 6 ∧ jump_dist_reduction_factor = 3 ∧ 
  initial_baby_jumps = 12 →
  time_to_catch_up jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps = 6 := 
by
  intros jumps_baby jumps_mother time_period jump_dist_mother 
    jump_dist_reduction_factor initial_baby_jumps _; sorry

end kangaroo_chase_l110_110674


namespace geo_seq_property_l110_110000

theorem geo_seq_property (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n+1) = r * a n)
  (h4_8 : a 4 + a 8 = -3) : a 6 * (a 2 + 2 * a 6 + a 10) = 9 := 
sorry

end geo_seq_property_l110_110000


namespace distinct_balls_boxes_l110_110286

theorem distinct_balls_boxes : ∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → boxes ^ balls = 243 :=
by
  intros balls boxes h1 h2
  rw [h1, h2]
  sorry

end distinct_balls_boxes_l110_110286


namespace find_x_l110_110373

def operation (a b : ℝ) : ℝ := a * b^(1/2)

theorem find_x (x : ℝ) : operation x 9 = 12 → x = 4 :=
by
  intro h
  sorry

end find_x_l110_110373


namespace child_running_speed_on_still_sidewalk_l110_110228

theorem child_running_speed_on_still_sidewalk (c s : ℕ) 
  (h1 : c + s = 93) 
  (h2 : c - s = 55) : c = 74 :=
sorry

end child_running_speed_on_still_sidewalk_l110_110228


namespace amount_each_girl_gets_l110_110641

theorem amount_each_girl_gets
  (B G : ℕ) 
  (total_sum : ℝ)
  (amount_each_boy : ℝ)
  (sum_boys_girls : B + G = 100)
  (total_sum_distributed : total_sum = 312)
  (amount_boy : amount_each_boy = 3.60)
  (B_approx : B = 60) :
  (total_sum - amount_each_boy * B) / G = 2.40 := 
by 
  sorry

end amount_each_girl_gets_l110_110641


namespace max_xy_l110_110612

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 8 * y = 48) : x * y ≤ 18 :=
sorry

end max_xy_l110_110612


namespace compute_complex_expression_l110_110972

-- Define the expression we want to prove
def complex_expression : ℚ := 1 / (1 + (1 / (2 + (1 / (4^2)))))

-- The theorem stating the expression equals to the correct result
theorem compute_complex_expression : complex_expression = 33 / 49 :=
by sorry

end compute_complex_expression_l110_110972


namespace tangent_line_at_one_min_value_f_l110_110918

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 + a * |Real.log x - 1|

theorem tangent_line_at_one (a : ℝ) (h1 : a = 1) : 
  ∃ (m b : ℝ), (∀ x : ℝ, f x a = m * x + b) ∧ m = 1 ∧ b = 1 ∧ (x - y + 1 = 0) := 
sorry

theorem min_value_f (a : ℝ) (h1 : 0 < a) : 
  (1 ≤ x ∧ x < e)  →  (x - f x a <= 0) ∨  (∀ (x : ℝ), 
  (f x a = if 0 < a ∧ a ≤ 2 then 1 + a 
          else if 2 < a ∧ a ≤ 2 * Real.exp (2) then 3 * (a / 2)^2 - (a / 2)^2 * Real.log (a / 2) else 
          Real.exp 2) 
   ) := 
sorry

end tangent_line_at_one_min_value_f_l110_110918


namespace log_exp_identity_l110_110285

noncomputable def a : ℝ := Real.log 3 / Real.log 4

theorem log_exp_identity : 2^a + 2^(-a) = (4 * Real.sqrt 3) / 3 := 
by
  sorry

end log_exp_identity_l110_110285


namespace chris_average_price_l110_110251

noncomputable def total_cost_dvd (price_per_dvd : ℝ) (num_dvds : ℕ) (discount : ℝ) : ℝ :=
  (price_per_dvd * (1 - discount)) * num_dvds

noncomputable def total_cost_bluray (price_per_bluray : ℝ) (num_blurays : ℕ) : ℝ :=
  price_per_bluray * num_blurays

noncomputable def total_cost_ultra_hd (price_per_ultra_hd : ℝ) (num_ultra_hds : ℕ) : ℝ :=
  price_per_ultra_hd * num_ultra_hds

noncomputable def total_cost (cost_dvd cost_bluray cost_ultra_hd : ℝ) : ℝ :=
  cost_dvd + cost_bluray + cost_ultra_hd

noncomputable def total_with_tax (total_cost : ℝ) (tax_rate : ℝ) : ℝ :=
  total_cost * (1 + tax_rate)

noncomputable def average_price (total_with_tax : ℝ) (total_movies : ℕ) : ℝ :=
  total_with_tax / total_movies

theorem chris_average_price :
  let price_per_dvd := 15
  let num_dvds := 5
  let discount := 0.20
  let price_per_bluray := 20
  let num_blurays := 8
  let price_per_ultra_hd := 25
  let num_ultra_hds := 3
  let tax_rate := 0.10
  let total_movies := num_dvds + num_blurays + num_ultra_hds
  let cost_dvd := total_cost_dvd price_per_dvd num_dvds discount
  let cost_bluray := total_cost_bluray price_per_bluray num_blurays
  let cost_ultra_hd := total_cost_ultra_hd price_per_ultra_hd num_ultra_hds
  let pre_tax_total := total_cost cost_dvd cost_bluray cost_ultra_hd
  let total := total_with_tax pre_tax_total tax_rate
  average_price total total_movies = 20.28 :=
by
  -- substitute each definition one step at a time
  -- to show the average price exactly matches 20.28
  sorry

end chris_average_price_l110_110251


namespace A_sym_diff_B_l110_110383

-- Definitions of sets and operations
def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {y | ∃ x : ℝ, y = 3^x}
def B : Set ℝ := {y | ∃ x : ℝ, y = -(x-1)^2 + 2}

-- The target equality to prove
theorem A_sym_diff_B : sym_diff A B = (({y | y ≤ 0}) ∪ ({y | y > 2})) :=
by
  sorry

end A_sym_diff_B_l110_110383


namespace ellipse_range_m_l110_110216

theorem ellipse_range_m (m : ℝ) :
    (∀ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2 → 
    ∃ (c : ℝ), c = x^2 + (y + 1)^2 ∧ m > 5) :=
sorry

end ellipse_range_m_l110_110216


namespace eval_expression_l110_110044

theorem eval_expression : (-2 ^ 4) + 3 * (-1) ^ 6 - (-2) ^ 3 = -5 := by
  sorry

end eval_expression_l110_110044


namespace cost_price_equals_720_l110_110045

theorem cost_price_equals_720 (C : ℝ) :
  (0.27 * C - 0.12 * C = 108) → (C = 720) :=
by
  sorry

end cost_price_equals_720_l110_110045


namespace three_legged_reptiles_count_l110_110706

noncomputable def total_heads : ℕ := 300
noncomputable def total_legs : ℕ := 798

def number_of_three_legged_reptiles (b r m : ℕ) : Prop :=
  b + r + m = total_heads ∧
  2 * b + 3 * r + 4 * m = total_legs

theorem three_legged_reptiles_count (b r m : ℕ) (h : number_of_three_legged_reptiles b r m) :
  r = 102 :=
sorry

end three_legged_reptiles_count_l110_110706


namespace smallest_in_sample_l110_110700

theorem smallest_in_sample:
  ∃ (m : ℕ) (δ : ℕ), m ≥ 0 ∧ δ > 0 ∧ δ * 5 = 80 ∧ 42 = δ * (42 / δ) + m ∧ m < δ ∧ (∀ i < 5, m + i * δ < 80) → m = 10 :=
by
  sorry

end smallest_in_sample_l110_110700


namespace broken_line_count_l110_110731

def num_right_moves : ℕ := 9
def num_up_moves : ℕ := 10
def total_moves : ℕ := num_right_moves + num_up_moves
def num_broken_lines : ℕ := Nat.choose total_moves num_right_moves

theorem broken_line_count : num_broken_lines = 92378 := by
  sorry

end broken_line_count_l110_110731


namespace trevor_comic_first_issue_pages_l110_110298

theorem trevor_comic_first_issue_pages
  (x : ℕ) 
  (h1 : 3 * x + 4 = 220) :
  x = 72 := 
by
  sorry

end trevor_comic_first_issue_pages_l110_110298


namespace intersection_A_B_l110_110151

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_A_B : A ∩ B = {70} := by
  sorry

end intersection_A_B_l110_110151


namespace complement_A_complement_U_range_of_a_empty_intersection_l110_110121

open Set Real

noncomputable def complement_A_in_U := { x : ℝ | ¬ (x < -1 ∨ x > 3) }

theorem complement_A_complement_U
  {A : Set ℝ} (hA : A = {x | x^2 - 2 * x - 3 > 0}) :
  (complement_A_in_U = (Icc (-1) 3)) :=
by sorry

theorem range_of_a_empty_intersection
  {B : Set ℝ} {a : ℝ}
  (hB : B = {x | abs (x - a) > 3})
  (h_empty : (Icc (-1) 3) ∩ B = ∅) :
  (0 ≤ a ∧ a ≤ 2) :=
by sorry

end complement_A_complement_U_range_of_a_empty_intersection_l110_110121


namespace solve_for_n_l110_110555

theorem solve_for_n (n : ℝ) : 
  (0.05 * n + 0.06 * (30 + n)^2 = 45) ↔ 
  (n = -2.5833333333333335 ∨ n = -58.25) :=
sorry

end solve_for_n_l110_110555


namespace max_intersections_between_quadrilateral_and_pentagon_l110_110086

-- Definitions based on the conditions
def quadrilateral_sides : ℕ := 4
def pentagon_sides : ℕ := 5

-- Theorem statement based on the problem
theorem max_intersections_between_quadrilateral_and_pentagon 
  (qm_sides : ℕ := quadrilateral_sides) 
  (pm_sides : ℕ := pentagon_sides) : 
  (∀ (n : ℕ), n = qm_sides →
    ∀ (m : ℕ), m = pm_sides →
      ∀ (intersection_points : ℕ), 
        intersection_points = (n * m) →
        intersection_points = 20) :=
sorry

end max_intersections_between_quadrilateral_and_pentagon_l110_110086


namespace cube_root_of_neg_eight_l110_110328

theorem cube_root_of_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by
  sorry

end cube_root_of_neg_eight_l110_110328


namespace angle_C_is_120_degrees_l110_110404

theorem angle_C_is_120_degrees (l m : ℝ) (A B C : ℝ) (hal : l = m) 
  (hA : A = 100) (hB : B = 140) : C = 120 := 
by 
  sorry

end angle_C_is_120_degrees_l110_110404


namespace initial_games_l110_110867

def games_given_away : ℕ := 91
def games_left : ℕ := 92

theorem initial_games :
  games_given_away + games_left = 183 :=
by
  sorry

end initial_games_l110_110867


namespace solve_for_a_minus_c_l110_110027

theorem solve_for_a_minus_c 
  (a b c d : ℝ) 
  (h1 : a - b = c + d + 9) 
  (h2 : a + b = c - d - 3) : 
  a - c = 3 := by
  sorry

end solve_for_a_minus_c_l110_110027


namespace seconds_in_9_point_4_minutes_l110_110509

def seconds_in_minute : ℕ := 60
def minutes : ℝ := 9.4
def expected_seconds : ℝ := 564

theorem seconds_in_9_point_4_minutes : minutes * seconds_in_minute = expected_seconds :=
by 
  sorry

end seconds_in_9_point_4_minutes_l110_110509


namespace gear_revolutions_l110_110170

variable (r_p : ℝ) 

theorem gear_revolutions (h1 : 40 * (1 / 6) = r_p * (1 / 6) + 5) : r_p = 10 := 
by
  sorry

end gear_revolutions_l110_110170


namespace coordinates_of_M_l110_110106

-- Let M be a point in the 2D Cartesian plane
variable {x y : ℝ}

-- Definition of the conditions
def distance_from_x_axis (y : ℝ) : Prop := abs y = 1
def distance_from_y_axis (x : ℝ) : Prop := abs x = 2

-- Theorem to prove
theorem coordinates_of_M (hx : distance_from_y_axis x) (hy : distance_from_x_axis y) :
  (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
sorry

end coordinates_of_M_l110_110106


namespace sin_neg_five_sixths_pi_l110_110255

theorem sin_neg_five_sixths_pi : Real.sin (- 5 / 6 * Real.pi) = -1 / 2 :=
sorry

end sin_neg_five_sixths_pi_l110_110255


namespace breadth_of_rectangular_plot_l110_110363

theorem breadth_of_rectangular_plot (b l : ℝ) (h1 : l = 3 * b) (h2 : l * b = 432) : b = 12 := 
sorry

end breadth_of_rectangular_plot_l110_110363


namespace capacity_of_new_bathtub_is_400_liters_l110_110516

-- Definitions based on conditions
def possible_capacities : Set ℕ := {4, 40, 400, 4000}  -- The possible capacities

-- Proof statement
theorem capacity_of_new_bathtub_is_400_liters (c : ℕ) 
  (h : c ∈ possible_capacities) : 
  c = 400 := 
sorry

end capacity_of_new_bathtub_is_400_liters_l110_110516


namespace fraction_value_l110_110818

theorem fraction_value (a b c : ℕ) (h1 : a = 2200) (h2 : b = 2096) (h3 : c = 121) :
    (a - b)^2 / c = 89 := by
  sorry

end fraction_value_l110_110818


namespace necessary_but_not_sufficient_l110_110446

-- Defining the problem in Lean 4 terms.
noncomputable def geom_seq_cond (a : ℕ → ℕ) (m n p q : ℕ) : Prop :=
  m + n = p + q → a m * a n = a p * a q

theorem necessary_but_not_sufficient (a : ℕ → ℕ) (m n p q : ℕ) (h : m + n = p + q) :
  geom_seq_cond a m n p q → ∃ b : ℕ → ℕ, (∀ n, b n = 0 → (m + n = p + q → b m * b n = b p * b q))
    ∧ (∀ n, ¬ (b n = 0 → ∀ q, b (q+1) / b q = b (q+1) / b q)) := sorry

end necessary_but_not_sufficient_l110_110446


namespace number_of_values_l110_110200

/-- Given:
  - The mean of some values was 190.
  - One value 165 was wrongly copied as 130 for the computation of the mean.
  - The correct mean is 191.4.
  Prove: the total number of values is 25. --/
theorem number_of_values (n : ℕ) (h₁ : (190 : ℝ) = ((190 * n) - (165 - 130)) / n) (h₂ : (191.4 : ℝ) = ((190 * n + 35) / n)) : n = 25 :=
sorry

end number_of_values_l110_110200


namespace train_speed_l110_110851

def train_length : ℝ := 360 -- length of the train in meters
def crossing_time : ℝ := 6 -- time taken to cross the man in seconds

theorem train_speed (train_length crossing_time : ℝ) : 
  (train_length = 360) → (crossing_time = 6) → (train_length / crossing_time = 60) :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end train_speed_l110_110851


namespace prob_sum_divisible_by_4_is_1_4_l110_110153

/-- 
  Given two wheels each with numbers from 1 to 8, 
  the probability that the sum of two selected numbers from the wheels is divisible by 4.
-/
noncomputable def prob_sum_divisible_by_4 : ℚ :=
  let outcomes : ℕ := 8 * 8
  let favorable_outcomes : ℕ := 16
  favorable_outcomes / outcomes

theorem prob_sum_divisible_by_4_is_1_4 : prob_sum_divisible_by_4 = 1 / 4 := 
  by
    -- Statement is left as sorry as the proof steps are not required.
    sorry

end prob_sum_divisible_by_4_is_1_4_l110_110153


namespace predicted_value_y_at_x_5_l110_110319

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

theorem predicted_value_y_at_x_5 :
  let x_values := [-2, -1, 0, 1, 2]
  let y_values := [5, 4, 2, 2, 1]
  let x_bar := mean x_values
  let y_bar := mean y_values
  let a_hat := y_bar
  (∀ x, y = -x + a_hat) →
  (x = 5 → y = -2.2) :=
by
  sorry

end predicted_value_y_at_x_5_l110_110319


namespace cylinder_increase_l110_110181

theorem cylinder_increase (x : ℝ) (r h : ℝ) (π : ℝ) 
  (h₁ : r = 5) (h₂ : h = 10) 
  (h₃ : π > 0) 
  (h_equal_volumes : π * (r + x) ^ 2 * h = π * r ^ 2 * (h + x)) :
  x = 5 / 2 :=
by
  -- Proof is omitted
  sorry

end cylinder_increase_l110_110181


namespace solve_for_x_l110_110791

theorem solve_for_x (x y : ℝ) : 3 * x + 4 * y = 5 → x = (5 - 4 * y) / 3 :=
by
  intro h
  sorry

end solve_for_x_l110_110791


namespace calculate_full_recipes_needed_l110_110528

def initial_attendance : ℕ := 125
def attendance_drop_percentage : ℝ := 0.40
def cookies_per_student : ℕ := 2
def cookies_per_recipe : ℕ := 18

theorem calculate_full_recipes_needed :
  let final_attendance := initial_attendance * (1 - attendance_drop_percentage : ℝ)
  let total_cookies_needed := (final_attendance * (cookies_per_student : ℕ))
  let recipes_needed := total_cookies_needed / (cookies_per_recipe : ℕ)
  ⌈recipes_needed⌉ = 9 :=
  by
  sorry

end calculate_full_recipes_needed_l110_110528


namespace tan_two_alpha_l110_110462

theorem tan_two_alpha (α β : ℝ) (h₁ : Real.tan (α - β) = -3/2) (h₂ : Real.tan (α + β) = 3) :
  Real.tan (2 * α) = 3/11 := 
sorry

end tan_two_alpha_l110_110462


namespace cos_theta_correct_projection_correct_l110_110484

noncomputable def vec_a : ℝ × ℝ := (2, 3)
noncomputable def vec_b : ℝ × ℝ := (-2, 4)

noncomputable def cos_theta (a b : ℝ × ℝ) : ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let norm_a := Real.sqrt (a.1 ^ 2 + a.2 ^ 2)
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  dot_product / (norm_a * norm_b)

noncomputable def projection (b : ℝ × ℝ) (cosθ : ℝ) : ℝ :=
  let norm_b := Real.sqrt (b.1 ^ 2 + b.2 ^ 2)
  norm_b * cosθ

theorem cos_theta_correct :
  cos_theta vec_a vec_b = 4 * Real.sqrt 65 / 65 :=
by
  sorry

theorem projection_correct :
  projection vec_b (cos_theta vec_a vec_b) = 8 * Real.sqrt 13 / 13 :=
by
  sorry

end cos_theta_correct_projection_correct_l110_110484


namespace no_root_l110_110893

theorem no_root :
  ∀ x : ℝ, x - (9 / (x - 4)) ≠ 4 - (9 / (x - 4)) :=
by
  intro x
  sorry

end no_root_l110_110893


namespace product_of_18396_and_9999_l110_110564

theorem product_of_18396_and_9999 : 18396 * 9999 = 183962604 :=
by
  sorry

end product_of_18396_and_9999_l110_110564


namespace each_group_has_145_bananas_l110_110173

theorem each_group_has_145_bananas (total_bananas : ℕ) (groups_bananas : ℕ) : 
  total_bananas = 290 ∧ groups_bananas = 2 → total_bananas / groups_bananas = 145 := 
by 
  sorry

end each_group_has_145_bananas_l110_110173


namespace cos_300_eq_half_l110_110633

theorem cos_300_eq_half : Real.cos (2 * π * (300 / 360)) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l110_110633


namespace jerome_bought_last_month_l110_110217

-- Definitions representing the conditions in the problem
def total_toy_cars_now := 40
def original_toy_cars := 25
def bought_this_month (bought_last_month : ℕ) := 2 * bought_last_month

-- The main statement to prove
theorem jerome_bought_last_month : ∃ x : ℕ, original_toy_cars + x + bought_this_month x = total_toy_cars_now ∧ x = 5 :=
by
  sorry

end jerome_bought_last_month_l110_110217


namespace largest_of_choices_l110_110405

theorem largest_of_choices :
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  A < D ∧ B < D ∧ C < D ∧ E < D :=
by
  let A := 24680 + (1 / 13579)
  let B := 24680 - (1 / 13579)
  let C := 24680 * (1 / 13579)
  let D := 24680 / (1 / 13579)
  let E := 24680.13579
  sorry

end largest_of_choices_l110_110405


namespace bracelet_arrangements_l110_110589

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def distinct_arrangements : ℕ := factorial 8 / (8 * 2)

theorem bracelet_arrangements : distinct_arrangements = 2520 :=
by
  sorry

end bracelet_arrangements_l110_110589


namespace adjust_collection_amount_l110_110303

/-- Define the error caused by mistaking half-dollars for dollars -/
def halfDollarError (x : ℕ) : ℤ := 50 * x

/-- Define the error caused by mistaking quarters for nickels -/
def quarterError (x : ℕ) : ℤ := 20 * x

/-- Define the total error based on the given conditions -/
def totalError (x : ℕ) : ℤ := halfDollarError x - quarterError x

theorem adjust_collection_amount (x : ℕ) : totalError x = 30 * x := by
  sorry

end adjust_collection_amount_l110_110303


namespace fixed_point_exists_l110_110543

-- Defining the function f
def f (a x : ℝ) : ℝ := a * x - 3 + 3

-- Stating that there exists a fixed point (3, 3a)
theorem fixed_point_exists (a : ℝ) : ∃ y : ℝ, f a 3 = y :=
by
  use (3 * a)
  simp [f]
  sorry

end fixed_point_exists_l110_110543


namespace find_a3_l110_110826

noncomputable def geometric_seq (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n, a (n+1) = a n * q

theorem find_a3 (a : ℕ → ℕ) (q : ℕ) (h_geom : geometric_seq a q) (hq : q > 1)
  (h1 : a 4 - a 0 = 15) (h2 : a 3 - a 1 = 6) :
  a 2 = 4 :=
by
  sorry

end find_a3_l110_110826


namespace part_1_part_2_part_3_l110_110494

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2
noncomputable def g (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := a * Real.log x
noncomputable def F (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x * g x a h
noncomputable def G (x : ℝ) (a : ℝ) (h : 0 < a) : ℝ := f x - g x a h + (a - 1) * x 

theorem part_1 (a : ℝ) (h : 0 < a) :
  ∃(x : ℝ), x = -(a / (4 * Real.exp 1)) :=
sorry

theorem part_2 (a : ℝ) (h1 : 0 < a) : 
  (∃ x1 x2, (1/e) < x1 ∧ x1 < e ∧ (1/e) < x2 ∧ x2 < e ∧ G x1 a h1 = 0 ∧ G x2 a h1 = 0) 
    ↔ (a > (2 * Real.exp 1 - 1) / (2 * (Real.exp 1)^2 + 2 * Real.exp 1) ∧ a < 1/2) :=
sorry

theorem part_3 : 
  ∀ {x : ℝ}, 0 < x → Real.log x + (3 / (4 * x^2)) - (1 / Real.exp x) > 0 :=
sorry

end part_1_part_2_part_3_l110_110494


namespace age_of_other_replaced_man_l110_110416

variable (A B C : ℕ)
variable (B_new1 B_new2 : ℕ)
variable (avg_old avg_new : ℕ)

theorem age_of_other_replaced_man (hB : B = 23) 
    (h_avg_new : (B_new1 + B_new2) / 2 = 25)
    (h_avg_inc : (A + B_new1 + B_new2) / 3 > (A + B + C) / 3) : 
    C = 26 := 
  sorry

end age_of_other_replaced_man_l110_110416


namespace instantaneous_velocity_at_t_eq_2_l110_110804

variable (t : ℝ)

def displacement (t : ℝ) : ℝ := 2 * (1 - t) ^ 2 

theorem instantaneous_velocity_at_t_eq_2 :
  (deriv (displacement) 2) = 4 :=
sorry

end instantaneous_velocity_at_t_eq_2_l110_110804


namespace volume_of_bottle_l110_110107

theorem volume_of_bottle (r h : ℝ) (π : ℝ) (h₀ : π > 0)
  (h₁ : r^2 * h + (4 / 3) * r^3 = 625) :
  π * (r^2 * h + (4 / 3) * r^3) = 625 * π :=
by sorry

end volume_of_bottle_l110_110107


namespace TimPrankCombinations_l110_110724

-- Definitions of the conditions in the problem
def MondayChoices : ℕ := 3
def TuesdayChoices : ℕ := 1
def WednesdayChoices : ℕ := 6
def ThursdayChoices : ℕ := 4
def FridayChoices : ℕ := 2

-- The main theorem to prove the total combinations
theorem TimPrankCombinations : 
  MondayChoices * TuesdayChoices * WednesdayChoices * ThursdayChoices * FridayChoices = 144 := 
by
  sorry

end TimPrankCombinations_l110_110724


namespace sin_double_angle_l110_110567

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end sin_double_angle_l110_110567


namespace tom_books_problem_l110_110631

theorem tom_books_problem 
  (original_books : ℕ)
  (books_sold : ℕ)
  (books_bought : ℕ)
  (h1 : original_books = 5)
  (h2 : books_sold = 4)
  (h3 : books_bought = 38) : 
  original_books - books_sold + books_bought = 39 :=
by
  sorry

end tom_books_problem_l110_110631


namespace descent_property_l110_110198

def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem descent_property (x : ℝ) (h : x < 3) : (quadratic_function (x + 1) < quadratic_function x) :=
sorry

end descent_property_l110_110198


namespace function_passes_through_one_one_l110_110833

noncomputable def f (a x : ℝ) : ℝ := a^(x - 1)

theorem function_passes_through_one_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 1 := 
by
  sorry

end function_passes_through_one_one_l110_110833


namespace find_value_of_A_l110_110800

-- Define the conditions
variable (A : ℕ)
variable (divisor : ℕ := 9)
variable (quotient : ℕ := 2)
variable (remainder : ℕ := 6)

-- The main statement of the proof problem
theorem find_value_of_A (h : A = quotient * divisor + remainder) : A = 24 :=
by
  -- Proof would go here
  sorry

end find_value_of_A_l110_110800


namespace rachel_earnings_without_tips_l110_110421

theorem rachel_earnings_without_tips
  (num_people : ℕ) (tip_per_person : ℝ) (total_earnings : ℝ)
  (h1 : num_people = 20)
  (h2 : tip_per_person = 1.25)
  (h3 : total_earnings = 37) :
  total_earnings - (num_people * tip_per_person) = 12 :=
by
  sorry

end rachel_earnings_without_tips_l110_110421


namespace terminating_decimal_of_fraction_l110_110834

theorem terminating_decimal_of_fraction (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 624) : 
  (∃ m : ℕ, 10^m * (n / 625) = k) → ∃ m, m = 624 :=
sorry

end terminating_decimal_of_fraction_l110_110834


namespace pet_store_cages_l110_110032

theorem pet_store_cages (total_puppies sold_puppies puppies_per_cage : ℕ) (h1 : total_puppies = 45) (h2 : sold_puppies = 39) (h3 : puppies_per_cage = 2) :
  (total_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l110_110032


namespace no_possible_salary_distribution_l110_110647

theorem no_possible_salary_distribution (x y z : ℕ) (h1 : x + y + z = 13) (h2 : x + 3 * y + 5 * z = 200) : false :=
by {
  -- Proof goes here
  sorry
}

end no_possible_salary_distribution_l110_110647


namespace angle_A_30_side_b_sqrt2_l110_110604

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the dot product of vectors AB and AC is 2√3 times the area S, 
    then angle A equals 30 degrees --/
theorem angle_A_30 {a b c S : ℝ} (h : (a * b * Real.sqrt 3 * c * Real.sin (π / 6)) = 2 * Real.sqrt 3 * S) : 
  A = π / 6 :=
sorry

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively 
    and area S, if the tangent of angles A, B, C are in the ratio 1:2:3 and c equals 1, 
    then side b equals √2 --/
theorem side_b_sqrt2 {A B C : ℝ} (a b c : ℝ) (h_tan_ratio : Real.tan A / Real.tan B = 1 / 2 ∧ Real.tan B / Real.tan C = 2 / 3)
  (h_c : c = 1) : b = Real.sqrt 2 :=
sorry

end angle_A_30_side_b_sqrt2_l110_110604


namespace walmart_pot_stacking_l110_110159

theorem walmart_pot_stacking :
  ∀ (total_pots pots_per_set shelves : ℕ),
    total_pots = 60 →
    pots_per_set = 5 →
    shelves = 4 →
    (total_pots / pots_per_set / shelves) = 3 :=
by 
  intros total_pots pots_per_set shelves h1 h2 h3
  sorry

end walmart_pot_stacking_l110_110159


namespace sara_spent_on_rented_movie_l110_110906

def total_spent_on_movies : ℝ := 36.78
def spent_on_tickets : ℝ := 2 * 10.62
def spent_on_bought_movie : ℝ := 13.95

theorem sara_spent_on_rented_movie : 
  (total_spent_on_movies - spent_on_tickets - spent_on_bought_movie = 1.59) := 
by sorry

end sara_spent_on_rented_movie_l110_110906


namespace missing_dog_number_l110_110341

theorem missing_dog_number {S : Finset ℕ} (h₁ : S =  Finset.range 25 \ {24}) (h₂ : S.sum id = 276) :
  (∃ y ∈ S, y = (S.sum id - y) / (S.card - 1)) ↔ 24 ∉ S :=
by
  sorry

end missing_dog_number_l110_110341


namespace number_of_poles_needed_l110_110253

def length := 90
def width := 40
def distance_between_poles := 5

noncomputable def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem number_of_poles_needed (l w d : ℕ) : perimeter l w / d = 52 :=
by
  rw [perimeter]
  sorry

end number_of_poles_needed_l110_110253


namespace ratio_of_only_B_to_both_A_and_B_l110_110610

theorem ratio_of_only_B_to_both_A_and_B 
  (Total_households : ℕ)
  (Neither_brand : ℕ)
  (Only_A : ℕ)
  (Both_A_and_B : ℕ)
  (Total_households_eq : Total_households = 180)
  (Neither_brand_eq : Neither_brand = 80)
  (Only_A_eq : Only_A = 60)
  (Both_A_and_B_eq : Both_A_and_B = 10) :
  (Total_households = Neither_brand + Only_A + (Total_households - Neither_brand - Only_A - Both_A_and_B) + Both_A_and_B) →
  (Total_households - Neither_brand - Only_A - Both_A_and_B) / Both_A_and_B = 3 :=
by
  intro H
  sorry

end ratio_of_only_B_to_both_A_and_B_l110_110610


namespace sum_of_squares_of_roots_l110_110245

theorem sum_of_squares_of_roots (x1 x2 : ℝ) 
    (h1 : 2 * x1^2 + 3 * x1 - 5 = 0) 
    (h2 : 2 * x2^2 + 3 * x2 - 5 = 0)
    (h3 : x1 + x2 = -3 / 2)
    (h4 : x1 * x2 = -5 / 2) : 
    x1^2 + x2^2 = 29 / 4 :=
by
  sorry

end sum_of_squares_of_roots_l110_110245


namespace subscriptions_to_grandfather_l110_110007

/-- 
Maggie earns $5.00 for every magazine subscription sold. 
She sold 4 subscriptions to her parents, 2 to the next-door neighbor, 
and twice that amount to another neighbor. Maggie earned $55 in total. 
Prove that the number of subscriptions Maggie sold to her grandfather is 1.
-/
theorem subscriptions_to_grandfather (G : ℕ) 
  (h1 : 5 * (4 + G + 2 + 4) = 55) : 
  G = 1 :=
by {
  sorry
}

end subscriptions_to_grandfather_l110_110007


namespace siblings_of_John_l110_110060

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)
  (height : String)

def John : Child := {name := "John", eyeColor := "Brown", hairColor := "Blonde", height := "Tall"}
def Emma : Child := {name := "Emma", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Oliver : Child := {name := "Oliver", eyeColor := "Brown", hairColor := "Black", height := "Short"}
def Mia : Child := {name := "Mia", eyeColor := "Blue", hairColor := "Blonde", height := "Short"}
def Lucas : Child := {name := "Lucas", eyeColor := "Blue", hairColor := "Black", height := "Tall"}
def Sophia : Child := {name := "Sophia", eyeColor := "Blue", hairColor := "Blonde", height := "Tall"}

theorem siblings_of_John : 
  (John.hairColor = Mia.hairColor ∧ John.hairColor = Sophia.hairColor) ∧
  ((John.eyeColor = Mia.eyeColor ∨ John.eyeColor = Sophia.eyeColor) ∨
   (John.height = Mia.height ∨ John.height = Sophia.height)) ∧
  (Mia.eyeColor = Sophia.eyeColor ∨ Mia.hairColor = Sophia.hairColor ∨ Mia.height = Sophia.height) ∧
  (John.hairColor = "Blonde") ∧
  (John.height = "Tall") ∧
  (Mia.hairColor = "Blonde") ∧
  (Sophia.hairColor = "Blonde") ∧
  (Sophia.height = "Tall") 
  → True := sorry

end siblings_of_John_l110_110060


namespace value_of_fraction_l110_110849

theorem value_of_fraction (a b : ℚ) (h : b / a = 1 / 2) : (a + b) / a = 3 / 2 :=
sorry

end value_of_fraction_l110_110849


namespace sum_of_eight_numbers_l110_110625

theorem sum_of_eight_numbers (average : ℝ) (count : ℕ) (h_avg : average = 5.3) (h_count : count = 8) : (average * count = 42.4) := sorry

end sum_of_eight_numbers_l110_110625


namespace find_d_l110_110350

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) :
  d = 180 :=
sorry

end find_d_l110_110350


namespace sunland_more_plates_than_moonland_l110_110729

theorem sunland_more_plates_than_moonland :
  let sunland_plates := 26^5 * 10^2
  let moonland_plates := 26^3 * 10^3
  sunland_plates - moonland_plates = 1170561600 := by
  sorry

end sunland_more_plates_than_moonland_l110_110729


namespace percentage_calculation_l110_110427

theorem percentage_calculation :
  ∀ (P : ℝ),
  (0.3 * 0.5 * 4400 = 99) →
  (P * 4400 = 99) →
  P = 0.0225 :=
by
  intros P condition1 condition2
  -- From the given conditions, it follows directly
  sorry

end percentage_calculation_l110_110427


namespace six_digit_numbers_l110_110794

theorem six_digit_numbers :
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
  sorry

end six_digit_numbers_l110_110794


namespace coefficients_divisible_by_7_l110_110218

theorem coefficients_divisible_by_7 
  {a b c d e : ℤ}
  (h : ∀ x : ℤ, (a * x^4 + b * x^3 + c * x^2 + d * x + e) % 7 = 0) :
  ∃ k l m n o : ℤ, a = 7*k ∧ b = 7*l ∧ c = 7*m ∧ d = 7*n ∧ e = 7*o :=
by
  sorry

end coefficients_divisible_by_7_l110_110218


namespace handshakes_total_l110_110795

theorem handshakes_total :
  let team_size := 6
  let referees := 3
  (team_size * team_size) + (2 * team_size * referees) = 72 :=
by
  sorry

end handshakes_total_l110_110795


namespace nine_point_five_minutes_in_seconds_l110_110203

-- Define the number of seconds in one minute
def seconds_per_minute : ℝ := 60

-- Define the function to convert minutes to seconds
def minutes_to_seconds (minutes : ℝ) : ℝ :=
  minutes * seconds_per_minute

-- Define the theorem to prove
theorem nine_point_five_minutes_in_seconds : minutes_to_seconds 9.5 = 570 :=
by
  sorry

end nine_point_five_minutes_in_seconds_l110_110203


namespace largest_b_for_box_volume_l110_110478

theorem largest_b_for_box_volume (a b c : ℕ) (h1 : 1 < c) (h2 : c < b) (h3 : b < a) 
                                 (h4 : c = 3) (volume : a * b * c = 360) : 
    b = 8 := 
sorry

end largest_b_for_box_volume_l110_110478


namespace evaluate_expression_l110_110956

theorem evaluate_expression : 27^(- (2 / 3 : ℝ)) + Real.log 4 / Real.log 8 = 7 / 9 :=
by
  sorry

end evaluate_expression_l110_110956


namespace retailer_initial_thought_profit_percentage_l110_110220

/-
  An uneducated retailer marks all his goods at 60% above the cost price and thinking that he will still make some profit, 
  offers a discount of 25% on the marked price. 
  His actual profit on the sales is 20.000000000000018%. 
  Prove that the profit percentage the retailer initially thought he would make is 60%.
-/

theorem retailer_initial_thought_profit_percentage
  (cost_price marked_price selling_price : ℝ)
  (h1 : marked_price = cost_price + 0.6 * cost_price)
  (h2 : selling_price = marked_price - 0.25 * marked_price)
  (h3 : selling_price - cost_price = 0.20000000000000018 * cost_price) :
  0.6 * 100 = 60 := by
  sorry

end retailer_initial_thought_profit_percentage_l110_110220


namespace cost_of_three_pencils_and_two_pens_l110_110202

theorem cost_of_three_pencils_and_two_pens 
  (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.15) 
  (h2 : 2 * p + 3 * q = 3.70) : 
  3 * p + 2 * q = 4.15 := 
by 
  exact h1

end cost_of_three_pencils_and_two_pens_l110_110202


namespace max_value_of_expression_l110_110920

noncomputable def max_value_expr (a b c : ℝ) : ℝ :=
  a + b^2 + c^3

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  max_value_expr a b c ≤ 8 :=
  sorry

end max_value_of_expression_l110_110920


namespace arithmetic_sequence_sum_l110_110324

theorem arithmetic_sequence_sum (d : ℕ) (y : ℕ) (x : ℕ) (h_y : y = 39) (h_d : d = 6) 
  (h_x : x = y - d) : 
  x + y = 72 := by 
  sorry

end arithmetic_sequence_sum_l110_110324


namespace domain_of_f_l110_110145

noncomputable def f (x : ℝ) : ℝ := (2*x + 3) / Real.sqrt (3*x - 9)

theorem domain_of_f : ∀ x : ℝ, (3 < x) ↔ (∃ y : ℝ, f y ≠ y) :=
by
  sorry

end domain_of_f_l110_110145


namespace power_of_powers_eval_powers_l110_110980

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end power_of_powers_eval_powers_l110_110980


namespace kaleb_savings_l110_110476

theorem kaleb_savings (x : ℕ) (h : x + 25 = 8 * 8) : x = 39 := 
by
  sorry

end kaleb_savings_l110_110476


namespace ratio_of_a_to_c_l110_110807

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 :=
by
  sorry

end ratio_of_a_to_c_l110_110807


namespace min_students_l110_110907

variable (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ)

def find_min_students : Prop :=
  H = 2 * L ∧ 
  M = L + H ∧ 
  e = L + M + H ∧ 
  e = 6 * L ∧ 
  L ≥ 1

theorem min_students (L : ℕ) (H : ℕ) (M : ℕ) (e : ℕ) : find_min_students L H M e → e = 6 := 
by 
  intro h 
  obtain ⟨h1, h2, h3, h4, h5⟩ := h
  sorry

end min_students_l110_110907


namespace chess_probability_l110_110340

theorem chess_probability (P_draw P_B_win : ℚ) (h_draw : P_draw = 1/2) (h_B_win : P_B_win = 1/3) :
  (1 - P_draw - P_B_win = 1/6) ∧ -- Statement A is correct
  (P_draw + (1 - P_draw - P_B_win) ≠ 1/2) ∧ -- Statement B is incorrect as it's not 1/2
  (1 - P_draw - P_B_win ≠ 2/3) ∧ -- Statement C is incorrect as it's not 2/3
  (P_draw + P_B_win ≠ 1/2) := -- Statement D is incorrect as it's not 1/2
by
  -- Insert proof here
  sorry

end chess_probability_l110_110340


namespace fourth_vertex_exists_l110_110371

structure Point :=
  (x : ℚ)
  (y : ℚ)

def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

def is_parallelogram (A B C D : Point) : Prop :=
  let M_AC := Point.mk ((A.x + C.x) / 2) ((A.y + C.y) / 2)
  let M_BD := Point.mk ((B.x + D.x) / 2) ((B.y + D.y) / 2)
  is_midpoint M_AC A C ∧ is_midpoint M_BD B D ∧ M_AC = M_BD

theorem fourth_vertex_exists (A B C : Point) (hA : A = ⟨-1, 0⟩) (hB : B = ⟨3, 0⟩) (hC : C = ⟨1, -5⟩) :
  ∃ D : Point, (D = ⟨1, 5⟩ ∨ D = ⟨-3, -5⟩) ∧ is_parallelogram A B C D :=
by
  sorry

end fourth_vertex_exists_l110_110371


namespace sandy_younger_than_molly_l110_110638

variable (s m : ℕ)
variable (h_ratio : 7 * m = 9 * s)
variable (h_sandy : s = 56)

theorem sandy_younger_than_molly : 
  m - s = 16 := 
by
  sorry

end sandy_younger_than_molly_l110_110638


namespace y_intercept_of_line_l110_110488

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  have h' : y = -(4/7) * x + 4 := sorry
  have h_intercept : x = 0 := sorry
  exact sorry

end y_intercept_of_line_l110_110488


namespace math_problem_l110_110123

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end math_problem_l110_110123


namespace find_HCF_l110_110703

-- Given conditions
def LCM : ℕ := 750
def product_of_two_numbers : ℕ := 18750

-- Proof statement
theorem find_HCF (h : ℕ) (hpos : h > 0) :
  (LCM * h = product_of_two_numbers) → h = 25 :=
by
  sorry

end find_HCF_l110_110703


namespace ratio_upstream_downstream_l110_110046

noncomputable def ratio_time_upstream_to_downstream
  (V_b V_s : ℕ) (T_u T_d : ℕ) : ℕ :=
(V_b + V_s) / (V_b - V_s)

theorem ratio_upstream_downstream
  (V_b V_s : ℕ) (hVb : V_b = 48) (hVs : V_s = 16) (T_u T_d : ℕ)
  (hT : ratio_time_upstream_to_downstream V_b V_s T_u T_d = 2) :
  T_u / T_d = 2 := by
  sorry

end ratio_upstream_downstream_l110_110046


namespace p_cycling_speed_l110_110976

-- J starts walking at 6 kmph at 12:00
def start_time : ℕ := 12 * 60  -- time in minutes for convenience
def j_speed : ℤ := 6  -- in kmph
def j_start_time : ℕ := start_time  -- 12:00 in minutes

-- P starts cycling at 13:30
def p_start_time : ℕ := (13 * 60) + 30  -- time in minutes for convenience

-- They are at their respective positions at 19:30
def end_time : ℕ := (19 * 60) + 30  -- time in minutes for convenience

-- At 19:30, J is 3 km behind P
def j_behind_p_distance : ℤ := 3  -- in kilometers

-- Prove that P's cycling speed = 8 kmph
theorem p_cycling_speed {p_speed : ℤ} :
  j_start_time = start_time →
  p_start_time = (13 * 60) + 30 →
  end_time = (19 * 60) + 30 →
  j_speed = 6 →
  j_behind_p_distance = 3 →
  p_speed = 8 :=
by
  sorry

end p_cycling_speed_l110_110976


namespace ben_current_age_l110_110357

theorem ben_current_age (a b c : ℕ) 
  (h1 : a + b + c = 36) 
  (h2 : c = 2 * a - 4) 
  (h3 : b + 5 = 3 * (a + 5) / 4) : 
  b = 5 := 
by
  sorry

end ben_current_age_l110_110357


namespace neon_sign_blink_interval_l110_110614

theorem neon_sign_blink_interval :
  ∃ (b : ℕ), (∀ t : ℕ, t > 0 → (t % 9 = 0 ∧ t % b = 0 ↔ t % 45 = 0)) → b = 15 :=
by
  sorry

end neon_sign_blink_interval_l110_110614


namespace steps_per_level_l110_110841

def number_of_steps_per_level (blocks_per_step total_blocks total_levels : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / total_levels

theorem steps_per_level (blocks_per_step : ℕ) (total_blocks : ℕ) (total_levels : ℕ) (h1 : blocks_per_step = 3) (h2 : total_blocks = 96) (h3 : total_levels = 4) :
  number_of_steps_per_level blocks_per_step total_blocks total_levels = 8 := 
by
  sorry

end steps_per_level_l110_110841


namespace find_sum_lent_l110_110955

variable (P : ℝ)

/-- Given that the annual interest rate is 4%, and the interest earned in 8 years
amounts to Rs 340 less than the sum lent, prove that the sum lent is Rs 500. -/
theorem find_sum_lent
  (h1 : ∀ I, I = P - 340 → I = (P * 4 * 8) / 100) : 
  P = 500 :=
by
  sorry

end find_sum_lent_l110_110955


namespace passed_boys_count_l110_110029

theorem passed_boys_count (total_boys avg_passed avg_failed overall_avg : ℕ) 
  (total_boys_eq : total_boys = 120) 
  (avg_passed_eq : avg_passed = 39) 
  (avg_failed_eq : avg_failed = 15) 
  (overall_avg_eq : overall_avg = 38) :
  let marks_by_passed := total_boys * overall_avg 
                         - (total_boys - passed) * avg_failed;
  let passed := marks_by_passed / avg_passed;
  passed = 115 := 
by
  sorry

end passed_boys_count_l110_110029


namespace exists_x_f_lt_g_l110_110166

noncomputable def f (x : ℝ) := (2 / Real.exp 1) ^ x

noncomputable def g (x : ℝ) := (Real.exp 1 / 3) ^ x

theorem exists_x_f_lt_g : ∃ x : ℝ, f x < g x := by
  sorry

end exists_x_f_lt_g_l110_110166


namespace frustum_radius_l110_110697

theorem frustum_radius (r : ℝ) (h1 : ∃ r1 r2, r1 = r 
                                  ∧ r2 = 3 * r 
                                  ∧ r1 * 2 * π * 3 = r2 * 2 * π
                                  ∧ (lateral_area = 84 * π)) (h2 : slant_height = 3) : 
  r = 7 :=
sorry

end frustum_radius_l110_110697


namespace range_of_k_l110_110270

theorem range_of_k (x : ℝ) (h1 : 0 < x) (h2 : x < 2) (h3 : x / Real.exp x < 1 / (k + 2 * x - x^2)) :
    0 ≤ k ∧ k < Real.exp 1 - 1 :=
sorry

end range_of_k_l110_110270


namespace monomials_like_terms_l110_110033

variable (m n : ℤ)

theorem monomials_like_terms (hm : m = 3) (hn : n = 1) : m - 2 * n = 1 :=
by
  sorry

end monomials_like_terms_l110_110033


namespace inverse_of_g_at_1_over_32_l110_110237

noncomputable def g (x : ℝ) : ℝ := (x^5 + 2) / 4

theorem inverse_of_g_at_1_over_32 :
  g⁻¹ (1/32) = (-15 / 8)^(1/5) :=
sorry

end inverse_of_g_at_1_over_32_l110_110237


namespace compute_n_pow_m_l110_110415

-- Given conditions
variables (n m : ℕ)
axiom n_eq : n = 3
axiom n_plus_one_eq_2m : n + 1 = 2 * m

-- Goal: Prove n^m = 9
theorem compute_n_pow_m : n^m = 9 :=
by {
  -- Proof goes here
  sorry
}

end compute_n_pow_m_l110_110415


namespace product_of_cosines_value_l110_110067

noncomputable def product_of_cosines : ℝ :=
  (1 + Real.cos (Real.pi / 12)) * (1 + Real.cos (5 * Real.pi / 12)) *
  (1 + Real.cos (7 * Real.pi / 12)) * (1 + Real.cos (11 * Real.pi / 12))

theorem product_of_cosines_value :
  product_of_cosines = 1 / 16 :=
by
  sorry

end product_of_cosines_value_l110_110067


namespace sum_of_distances_l110_110895

theorem sum_of_distances (A B C : ℝ × ℝ) (hA : A.2^2 = 8 * A.1) (hB : B.2^2 = 8 * B.1) 
(hC : C.2^2 = 8 * C.1) (h_centroid : (A.1 + B.1 + C.1) / 3 = 2) : 
  dist (2, 0) A + dist (2, 0) B + dist (2, 0) C = 12 := 
sorry

end sum_of_distances_l110_110895


namespace infinite_non_congruent_integers_l110_110193

theorem infinite_non_congruent_integers (a : ℕ → ℤ) (m : ℕ → ℤ) (k : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → 2 ≤ m i)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < k → 2 * m i ≤ m (i + 1)) :
  ∃ (x : ℕ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → ¬ (x % (m i) = a i % (m i)) :=
sorry

end infinite_non_congruent_integers_l110_110193


namespace option_b_correct_l110_110871

theorem option_b_correct : (-(-2)) = abs (-2) := by
  sorry

end option_b_correct_l110_110871


namespace roots_twice_other_p_values_l110_110378

theorem roots_twice_other_p_values (p : ℝ) :
  (∃ (a : ℝ), (a^2 = 9) ∧ (x^2 + p*x + 18 = 0) ∧
  ((x - a)*(x - 2*a) = (0:ℝ))) ↔ (p = 9 ∨ p = -9) :=
sorry

end roots_twice_other_p_values_l110_110378


namespace part_1_part_2_l110_110406

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def B (m : ℝ) : Set ℝ := { x | x^2 - (2*m + 1)*x + 2*m < 0 }

theorem part_1 (m : ℝ) (h : m < 1/2) : 
  B m = { x | 2*m < x ∧ x < 1 } := 
sorry

theorem part_2 (m : ℝ) : 
  (A ∪ B m = A) ↔ -1/2 ≤ m ∧ m ≤ 1 := 
sorry

end part_1_part_2_l110_110406


namespace frustum_volume_correct_l110_110499

noncomputable def volume_frustum (base_edge_original base_edge_smaller altitude_original altitude_smaller : ℝ) : ℝ :=
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let volume_original := (1 / 3) * base_area_original * altitude_original
  let volume_smaller := (1 / 3) * base_area_smaller * altitude_smaller
  volume_original - volume_smaller

theorem frustum_volume_correct :
  volume_frustum 16 8 10 5 = 2240 / 3 :=
by
  have h1 : volume_frustum 16 8 10 5 = 
    (1 / 3) * (16^2) * 10 - (1 / 3) * (8^2) * 5 := rfl
  simp only [pow_two] at h1
  norm_num at h1
  exact h1

end frustum_volume_correct_l110_110499


namespace baba_yaga_powder_problem_l110_110257

theorem baba_yaga_powder_problem (A B d : ℤ) 
  (h1 : A + B + d = 6) 
  (h2 : A + d = 3) 
  (h3 : B + d = 2) : 
  A = 4 ∧ B = 3 :=
by
  -- Proof omitted, as only the statement is required
  sorry

end baba_yaga_powder_problem_l110_110257


namespace root_of_equation_l110_110535

theorem root_of_equation (a : ℝ) (h : a^2 * (-1)^2 + 2011 * a * (-1) - 2012 = 0) : 
  a = 2012 ∨ a = -1 :=
by sorry

end root_of_equation_l110_110535


namespace problem_solution_l110_110923

theorem problem_solution
  (y1 y2 y3 y4 y5 y6 y7 : ℝ)
  (h1 : y1 + 3*y2 + 5*y3 + 7*y4 + 9*y5 + 11*y6 + 13*y7 = 0)
  (h2 : 3*y1 + 5*y2 + 7*y3 + 9*y4 + 11*y5 + 13*y6 + 15*y7 = 10)
  (h3 : 5*y1 + 7*y2 + 9*y3 + 11*y4 + 13*y5 + 15*y6 + 17*y7 = 104) :
  7*y1 + 9*y2 + 11*y3 + 13*y4 + 15*y5 + 17*y6 + 19*y7 = 282 := by
  sorry

end problem_solution_l110_110923


namespace negation_of_exists_l110_110155

theorem negation_of_exists (x : ℕ) : (¬ ∃ x : ℕ, x^2 ≤ x) := 
by 
  sorry

end negation_of_exists_l110_110155


namespace minimize_expense_l110_110464

def price_after_first_discount (initial_price : ℕ) (discount : ℕ) : ℕ :=
  initial_price * (100 - discount) / 100

def final_price_set1 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 15
  let step2 := price_after_first_discount step1 25
  price_after_first_discount step2 10

def final_price_set2 (initial_price : ℕ) : ℕ :=
  let step1 := price_after_first_discount initial_price 25
  let step2 := price_after_first_discount step1 10
  price_after_first_discount step2 10

theorem minimize_expense (initial_price : ℕ) (h : initial_price = 12000) :
  final_price_set1 initial_price = 6885 ∧ final_price_set2 initial_price = 7290 ∧
  final_price_set1 initial_price < final_price_set2 initial_price := by
  sorry

end minimize_expense_l110_110464


namespace geometric_series_ratio_l110_110230

theorem geometric_series_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ (n : ℕ), S n = a 1 * (1 - q^n) / (1 - q))
  (h2 : a 3 + 2 * a 6 = 0)
  (h3 : a 6 = a 3 * q^3)
  (h4 : q^3 = -1 / 2) :
  S 3 / S 6 = 2 := 
sorry

end geometric_series_ratio_l110_110230


namespace breakEvenBooks_l110_110621

theorem breakEvenBooks (FC VC_per_book SP : ℝ) (hFC : FC = 56430) (hVC : VC_per_book = 8.25) (hSP : SP = 21.75) :
  ∃ x : ℕ, FC + (VC_per_book * x) = SP * x ∧ x = 4180 :=
by {
  sorry
}

end breakEvenBooks_l110_110621


namespace part1_part2_l110_110261

-- Define Set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}

-- Define Set B, parameterized by m
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m + 1}

-- Proof Problem (1): When m = 1, A ∩ B = {x | 0 < x ∧ x ≤ 3/2}
theorem part1 (x : ℝ) : (x ∈ A ∩ B 1) ↔ (0 < x ∧ x ≤ 3/2) := by
  sorry

-- Proof Problem (2): If ∀ x, x ∈ A → x ∈ B m, then m ∈ (-∞, 1/6]
theorem part2 (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) → m ≤ 1/6 := by
  sorry

end part1_part2_l110_110261


namespace scientific_notation_of_75500000_l110_110691

theorem scientific_notation_of_75500000 :
  ∃ (a : ℝ) (n : ℤ), 75500000 = a * 10 ^ n ∧ a = 7.55 ∧ n = 7 :=
by {
  sorry
}

end scientific_notation_of_75500000_l110_110691


namespace baker_cakes_remaining_l110_110118

def InitialCakes : ℕ := 48
def SoldCakes : ℕ := 44
def RemainingCakes (initial sold : ℕ) : ℕ := initial - sold

theorem baker_cakes_remaining : RemainingCakes InitialCakes SoldCakes = 4 := 
by {
  -- placeholder for the proof
  sorry
}

end baker_cakes_remaining_l110_110118


namespace evaluate_expression_l110_110946

theorem evaluate_expression (a b c : ℤ)
  (h1 : c = b - 12)
  (h2 : b = a + 4)
  (h3 : a = 5)
  (h4 : a + 2 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  (a + 3 : ℚ) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 7 / 3 := 
sorry

end evaluate_expression_l110_110946


namespace provisions_last_days_after_reinforcement_l110_110855

-- Definitions based on the conditions
def initial_men := 2000
def initial_days := 40
def reinforcement_men := 2000
def days_passed := 20

-- Calculate the total provisions initially
def total_provisions := initial_men * initial_days

-- Calculate the remaining provisions after some days passed
def remaining_provisions := total_provisions - (initial_men * days_passed)

-- Total number of men after reinforcement
def total_men := initial_men + reinforcement_men

-- The Lean statement proving the duration the remaining provisions will last
theorem provisions_last_days_after_reinforcement :
  remaining_provisions / total_men = 10 := by
  sorry

end provisions_last_days_after_reinforcement_l110_110855


namespace polygon_sides_l110_110594

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 1440) : n = 10 :=
by sorry

end polygon_sides_l110_110594


namespace problem_solution_l110_110840

theorem problem_solution
  (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end problem_solution_l110_110840


namespace Jim_remaining_distance_l110_110269

theorem Jim_remaining_distance (t d r : ℕ) (h₁ : t = 1200) (h₂ : d = 923) (h₃ : r = t - d) : r = 277 := 
by 
  -- Proof steps would go here
  sorry

end Jim_remaining_distance_l110_110269


namespace original_number_is_14_l110_110048

def two_digit_number_increased_by_2_or_4_results_fourfold (x : ℕ) : Prop :=
  (x >= 10) ∧ (x < 100) ∧ 
  (∃ (a b : ℕ), a + 2 = ((x / 10 + 2) % 10) ∧ b + 2 = (x % 10)) ∧
  (4 * x = ((x / 10 + 2) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 2) * 10 + (x % 10 + 4)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 2)) ∨ 
   4 * x = ((x / 10 + 4) * 10 + (x % 10 + 4)))

theorem original_number_is_14 : ∃ x : ℕ, two_digit_number_increased_by_2_or_4_results_fourfold x ∧ x = 14 :=
by
  sorry

end original_number_is_14_l110_110048


namespace polynomial_coefficients_equivalence_l110_110075

theorem polynomial_coefficients_equivalence
    {a0 a1 a2 a3 a4 a5 : ℤ}
    (h_poly : (2*x-1)^5 = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5):
    (a0 + a1 + a2 + a3 + a4 + a5 = 1) ∧
    (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| = 243) ∧
    (a1 + a3 + a5 = 122) ∧
    ((a0 + a2 + a4)^2 - (a1 + a3 + a5)^2 = -243) :=
    sorry

end polynomial_coefficients_equivalence_l110_110075


namespace carol_maximizes_at_0_75_l110_110232

def winning_probability (a b c : ℝ) : Prop :=
(0 ≤ a ∧ a ≤ 1) ∧ (0.25 ≤ b ∧ b ≤ 0.75) ∧ (a < c ∧ c < b ∨ b < c ∧ c < a)

theorem carol_maximizes_at_0_75 :
  ∀ (a b : ℝ), (0 ≤ a ∧ a ≤ 1) → (0.25 ≤ b ∧ b ≤ 0.75) → (∃ c : ℝ, 0 ≤ c ∧ c ≤ 1 ∧ (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → winning_probability a b x ≤ winning_probability a b 0.75)) :=
sorry

end carol_maximizes_at_0_75_l110_110232


namespace intersection_eq_l110_110576

/-
Define the sets A and B
-/
def setA : Set ℝ := {-1, 0, 1, 2}
def setB : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

/-
Lean statement to prove the intersection A ∩ B equals {1, 2}
-/
theorem intersection_eq :
  setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_eq_l110_110576


namespace diminished_value_l110_110195

theorem diminished_value (x y : ℝ) (h1 : x = 160)
  (h2 : x / 5 + 4 = x / 4 - y) : y = 4 :=
by
  sorry

end diminished_value_l110_110195


namespace betty_gave_stuart_percentage_l110_110348

theorem betty_gave_stuart_percentage (P : ℝ) 
  (betty_marbles : ℝ := 60) 
  (stuart_initial_marbles : ℝ := 56) 
  (stuart_final_marbles : ℝ := 80)
  (increase_in_stuart_marbles : ℝ := stuart_final_marbles - stuart_initial_marbles)
  (betty_to_stuart : ℝ := (P / 100) * betty_marbles) :
  56 + ((P / 100) * betty_marbles) = 80 → P = 40 :=
by
  intros h
  -- Sorry is used since the proof steps are not required
  sorry

end betty_gave_stuart_percentage_l110_110348


namespace recurring_decimal_36_exceeds_decimal_35_l110_110131

-- Definition of recurring decimal 0.36...
def recurring_decimal_36 : ℚ := 36 / 99

-- Definition of 0.35 as fraction
def decimal_35 : ℚ := 7 / 20

-- Statement of the math proof problem
theorem recurring_decimal_36_exceeds_decimal_35 :
  recurring_decimal_36 - decimal_35 = 3 / 220 := by
  sorry

end recurring_decimal_36_exceeds_decimal_35_l110_110131


namespace alice_coins_percentage_l110_110264

theorem alice_coins_percentage :
  let penny := 1
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let total_cents := penny + dime + quarter + half_dollar
  (total_cents / 100) * 100 = 86 :=
by
  sorry

end alice_coins_percentage_l110_110264


namespace legs_paws_in_pool_l110_110500

def total_legs_paws (num_humans : Nat) (human_legs : Nat) (num_dogs : Nat) (dog_paws : Nat) : Nat :=
  (num_humans * human_legs) + (num_dogs * dog_paws)

theorem legs_paws_in_pool :
  total_legs_paws 2 2 5 4 = 24 := by
  sorry

end legs_paws_in_pool_l110_110500


namespace proposition_D_l110_110079

theorem proposition_D (a b : ℝ) (h : a > abs b) : a^2 > b^2 :=
by {
    sorry
}

end proposition_D_l110_110079


namespace expected_up_right_paths_l110_110864

def lattice_points := {p : ℕ × ℕ // p.1 ≤ 5 ∧ p.2 ≤ 5}

def total_paths : ℕ := Nat.choose 10 5

def calculate_paths (x y : ℕ) : ℕ :=
  if h : x ≤ 5 ∧ y ≤ 5 then
    let F := total_paths * 25
    F / 36
  else
    0

theorem expected_up_right_paths : ∃ S, S = 175 :=
  sorry

end expected_up_right_paths_l110_110864


namespace A_inter_B_eq_l110_110615

def A := {x : ℤ | 1 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 3}
def B := {x : ℤ | 5 ≤ x ∧ x < 9}

theorem A_inter_B_eq : A ∩ B = {5, 6, 7} :=
by sorry

end A_inter_B_eq_l110_110615


namespace product_of_integers_is_eight_l110_110911

-- Define three different positive integers a, b, c such that they sum to 7
def sum_to_seven (a b c : ℕ) : Prop := a + b + c = 7 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Prove that the product of these integers is 8
theorem product_of_integers_is_eight (a b c : ℕ) (h : sum_to_seven a b c) : a * b * c = 8 := by sorry

end product_of_integers_is_eight_l110_110911


namespace calculate_value_l110_110608

theorem calculate_value : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end calculate_value_l110_110608


namespace midpoint_reflection_sum_l110_110386

/-- 
Points P and R are located at (2, 1) and (12, 15) respectively. 
Point M is the midpoint of segment PR. 
Segment PR is reflected over the y-axis.
We want to prove that the sum of the coordinates of the image of point M (the midpoint of the reflected segment) is 1.
-/
theorem midpoint_reflection_sum : 
  let P := (2, 1)
  let R := (12, 15)
  let M := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  let P_image := (-P.1, P.2)
  let R_image := (-R.1, R.2)
  let M' := ((P_image.1 + R_image.1) / 2, (P_image.2 + R_image.2) / 2)
  (M'.1 + M'.2) = 1 :=
by
  sorry

end midpoint_reflection_sum_l110_110386


namespace initial_distance_between_A_and_B_l110_110937

theorem initial_distance_between_A_and_B
  (start_time : ℕ)        -- time in hours, 1 pm
  (meet_time : ℕ)         -- time in hours, 3 pm
  (speed_A : ℕ)           -- speed of A in km/hr
  (speed_B : ℕ)           -- speed of B in km/hr
  (time_walked : ℕ)       -- time walked in hours
  (distance_A : ℕ)        -- distance covered by A in km
  (distance_B : ℕ)        -- distance covered by B in km
  (initial_distance : ℕ)  -- initial distance between A and B

  (h1 : start_time = 1)
  (h2 : meet_time = 3)
  (h3 : speed_A = 5)
  (h4 : speed_B = 7)
  (h5 : time_walked = meet_time - start_time)
  (h6 : distance_A = speed_A * time_walked)
  (h7 : distance_B = speed_B * time_walked)
  (h8 : initial_distance = distance_A + distance_B) :

  initial_distance = 24 :=
by
  sorry

end initial_distance_between_A_and_B_l110_110937


namespace min_selling_price_l110_110827

-- Average sales per month
def avg_sales := 50

-- Cost per refrigerator
def cost_per_fridge := 1200

-- Shipping fee per refrigerator
def shipping_fee_per_fridge := 20

-- Monthly storefront fee
def monthly_storefront_fee := 10000

-- Monthly repair costs
def monthly_repair_costs := 5000

-- Profit margin requirement
def profit_margin := 0.2

-- The minimum selling price for the shop to maintain at least 20% profit margin
theorem min_selling_price 
  (avg_sales : ℕ) 
  (cost_per_fridge : ℕ) 
  (shipping_fee_per_fridge : ℕ) 
  (monthly_storefront_fee : ℕ) 
  (monthly_repair_costs : ℕ) 
  (profit_margin : ℝ) : 
  ∃ x : ℝ, 
    (50 * x - ((cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs)) 
    ≥ (cost_per_fridge + shipping_fee_per_fridge) * avg_sales + monthly_storefront_fee + monthly_repair_costs * profit_margin 
    → x ≥ 1824 :=
by 
  sorry

end min_selling_price_l110_110827


namespace y_when_x_is_4_l110_110874

theorem y_when_x_is_4
  (x y : ℝ)
  (h1 : x + y = 30)
  (h2 : x - y = 10)
  (h3 : x * y = 200) :
  y = 50 :=
by
  sorry

end y_when_x_is_4_l110_110874


namespace maximum_volume_pyramid_is_one_sixteenth_l110_110741

open Real  -- Opening Real namespace for real number operations

noncomputable def maximum_volume_pyramid : ℝ :=
  let a := 1 -- side length of the equilateral triangle base
  let base_area := (sqrt 3 / 4) * (a * a) -- area of the equilateral triangle with side length 1
  let median := sqrt 3 / 2 * a -- median length of the triangle
  let height := 1 / 2 * median -- height of the pyramid
  let volume := 1 / 3 * base_area * height -- volume formula for a pyramid
  volume

theorem maximum_volume_pyramid_is_one_sixteenth :
  maximum_volume_pyramid = 1 / 16 :=
by
  simp [maximum_volume_pyramid] -- Simplify the volume definition
  sorry -- Proof omitted

end maximum_volume_pyramid_is_one_sixteenth_l110_110741


namespace determinant_zero_l110_110695

noncomputable def A (α β : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]

theorem determinant_zero (α β : ℝ) : Matrix.det (A α β) = 0 := 
  sorry

end determinant_zero_l110_110695


namespace maximum_k_for_ray_below_f_l110_110061

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 3 * x - 2

theorem maximum_k_for_ray_below_f :
  let g (x : ℝ) : ℝ := (x * Real.log x + 3 * x - 2) / (x - 1)
  ∃ k : ℤ, ∀ x > 1, g x > k ∧ k = 5 :=
by sorry

end maximum_k_for_ray_below_f_l110_110061


namespace power_i_2015_l110_110445

theorem power_i_2015 (i : ℂ) (hi : i^2 = -1) : i^2015 = -i :=
by
  have h1 : i^4 = 1 := by sorry
  have h2 : 2015 = 4 * 503 + 3 := by norm_num
  sorry

end power_i_2015_l110_110445


namespace solve_inequality_l110_110629

variable (f : ℝ → ℝ)

-- Define the conditions
axiom f_decreasing : ∀ x y : ℝ, x ≤ y → f y ≤ f x

-- Prove the main statement
theorem solve_inequality (h : ∀ x : ℝ, f (f x) = x) : ∀ x : ℝ, f (f x) = x := 
by
  sorry

end solve_inequality_l110_110629


namespace trumpet_cost_l110_110314

def cost_of_song_book : Real := 5.84
def total_spent : Real := 151
def cost_of_trumpet : Real := total_spent - cost_of_song_book

theorem trumpet_cost : cost_of_trumpet = 145.16 :=
by
  sorry

end trumpet_cost_l110_110314


namespace find_n_l110_110776

theorem find_n (e n : ℕ) (h_lcm : Nat.lcm e n = 690) (h_n_not_div_3 : ¬ (3 ∣ n)) (h_e_not_div_2 : ¬ (2 ∣ e)) : n = 230 :=
by
  sorry

end find_n_l110_110776


namespace largest_number_is_34_l110_110981

theorem largest_number_is_34 (a b c : ℕ) (h1 : a + b + c = 82) (h2 : c - b = 8) (h3 : b - a = 4) : c = 34 := 
by 
  sorry

end largest_number_is_34_l110_110981


namespace cards_per_pack_l110_110507

-- Definitions from the problem conditions
def packs := 60
def cards_per_page := 10
def pages_needed := 42

-- Theorem statement for the mathematically equivalent proof problem
theorem cards_per_pack : (pages_needed * cards_per_page) / packs = 7 :=
by sorry

end cards_per_pack_l110_110507


namespace students_decrement_l110_110012

theorem students_decrement:
  ∃ d : ℕ, ∃ A : ℕ, 
  (∃ n1 n2 n3 n4 n5 : ℕ, n1 = A ∧ n2 = A - d ∧ n3 = A - 2 * d ∧ n4 = A - 3 * d ∧ n5 = A - 4 * d) ∧
  (5 = 5) ∧
  (n1 + n2 + n3 + n4 + n5 = 115) ∧
  (A = 27) → d = 2 :=
by {
  sorry
}

end students_decrement_l110_110012


namespace functional_relationship_max_annual_profit_l110_110737

namespace FactoryProfit

-- Definitions of conditions
def fixed_annual_investment : ℕ := 100
def unit_investment : ℕ := 1
def sales_revenue (x : ℕ) : ℕ :=
  if x > 20 then 260 
  else 33 * x - x^2

def annual_profit (x : ℕ) : ℤ :=
  let revenue := sales_revenue x
  let total_investment := fixed_annual_investment + x
  revenue - total_investment

-- Statements to prove
theorem functional_relationship (x : ℕ) (hx : x > 0) :
  annual_profit x =
  if x ≤ 20 then
    (-x^2 : ℤ) + 32 * x - 100
  else
    160 - x :=
by sorry

theorem max_annual_profit : 
  ∃ x, annual_profit x = 144 ∧
  ∀ y, annual_profit y ≤ 144 :=
by sorry

end FactoryProfit

end functional_relationship_max_annual_profit_l110_110737


namespace people_in_group_l110_110889

theorem people_in_group
  (N : ℕ)
  (h1 : ∃ w1 w2 : ℝ, w1 = 65 ∧ w2 = 71 ∧ w2 - w1 = 6)
  (h2 : ∃ avg_increase : ℝ, avg_increase = 1.5 ∧ 6 = avg_increase * N) :
  N = 4 :=
sorry

end people_in_group_l110_110889


namespace part1_part2_l110_110235

noncomputable section
open Real

section
variables {x A a b c : ℝ}
variables {k : ℤ}

def f (x : ℝ) : ℝ := sin (2 * x - (π / 6)) + 2 * cos x ^ 2 - 1

theorem part1 (k : ℤ) : 
  ∀ x : ℝ, 
  k * π - (π / 3) ≤ x ∧ x ≤ k * π + (π / 6) → 
    ∀ x₁ x₂, 
      k * π - (π / 3) ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ k * π + (π / 6) → 
        f x₁ < f x₂ := sorry

theorem part2 {A a b c : ℝ} 
  (h_a_seq : 2 * a = b + c) 
  (h_dot : b * c * cos A = 9) 
  (h_A_fA : f A = 1 / 2) 
  : 
  a = 3 * sqrt 2 := sorry

end

end part1_part2_l110_110235


namespace original_amount_l110_110944

theorem original_amount {P : ℕ} {R : ℕ} {T : ℕ} (h1 : P = 1000) (h2 : T = 5) 
  (h3 : ∃ R, (1000 * (R + 5) * 5) / 100 + 1000 = 1750) : 
  1000 + (1000 * R * 5 / 100) = 1500 :=
by
  sorry

end original_amount_l110_110944


namespace geometric_sequence_sum_l110_110097

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 := 
sorry

end geometric_sequence_sum_l110_110097


namespace employee_gross_pay_l110_110004

theorem employee_gross_pay
  (pay_rate_regular : ℝ) (pay_rate_overtime : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ)
  (h1 : pay_rate_regular = 11.25)
  (h2 : pay_rate_overtime = 16)
  (h3 : regular_hours = 40)
  (h4 : overtime_hours = 10.75) :
  (pay_rate_regular * regular_hours + pay_rate_overtime * overtime_hours = 622) :=
by
  sorry

end employee_gross_pay_l110_110004


namespace incorrect_games_leq_75_percent_l110_110971

theorem incorrect_games_leq_75_percent (N : ℕ) (win_points : ℕ) (draw_points : ℚ) (loss_points : ℕ) (incorrect : (ℕ × ℕ) → Prop) :
  (win_points = 1) → (draw_points = 1 / 2) → (loss_points = 0) →
  ∀ (g : ℕ × ℕ), incorrect g → 
  ∃ (total_games incorrect_games : ℕ), 
    total_games = N * (N - 1) / 2 ∧
    incorrect_games ≤ 3 / 4 * total_games := sorry

end incorrect_games_leq_75_percent_l110_110971


namespace non_monotonic_m_range_l110_110809

theorem non_monotonic_m_range (m : ℝ) :
  (∃ x ∈ Set.Ioo (-1 : ℝ) 2, (3 * x^2 + 2 * x + m = 0)) →
  m ∈ Set.Ioo (-16 : ℝ) (1/3 : ℝ) :=
sorry

end non_monotonic_m_range_l110_110809


namespace max_length_AB_l110_110765

theorem max_length_AB : ∀ t : ℝ, 0 ≤ t ∧ t ≤ 3 → ∃ M, M = 81 / 8 ∧ ∀ t, -2 * (t - 3/4)^2 + 81 / 8 = M :=
by sorry

end max_length_AB_l110_110765


namespace base_conversion_is_248_l110_110187

theorem base_conversion_is_248 (a b c n : ℕ) 
  (h1 : n = 49 * a + 7 * b + c) 
  (h2 : n = 81 * c + 9 * b + a) 
  (h3 : 0 ≤ a ∧ a ≤ 6) 
  (h4 : 0 ≤ b ∧ b ≤ 6) 
  (h5 : 0 ≤ c ∧ c ≤ 6)
  (h6 : 0 ≤ a ∧ a ≤ 8) 
  (h7 : 0 ≤ b ∧ b ≤ 8) 
  (h8 : 0 ≤ c ∧ c ≤ 8) 
  : n = 248 :=
by 
  sorry

end base_conversion_is_248_l110_110187


namespace part_I_part_II_l110_110493

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - 2 * x + 2 * a

theorem part_I (a : ℝ) :
  let x := Real.log 2
  ∃ I₁ I₂ : Set ℝ,
    (∀ x ∈ I₁, f a x > f a (Real.log 2)) ∧
    (∀ x ∈ I₂, f a x < f a (Real.log 2)) ∧
    I₁ = Set.Iio (Real.log 2) ∧
    I₂ = Set.Ioi (Real.log 2) ∧
    f a (Real.log 2) = 2 * (1 - Real.log 2 + a) :=
by sorry

theorem part_II (a : ℝ) (h : a > Real.log 2 - 1) (x : ℝ) (hx : 0 < x) :
  Real.exp x > x^2 - 2 * a * x + 1 :=
by sorry

end part_I_part_II_l110_110493


namespace candies_per_pack_l110_110016

-- Conditions in Lean:
def total_candies : ℕ := 60
def packs_initially (packs_after : ℕ) : ℕ := packs_after + 1
def packs_after : ℕ := 2
def pack_count : ℕ := packs_initially packs_after

-- The statement of the proof problem:
theorem candies_per_pack : 
  total_candies / pack_count = 20 :=
by
  sorry

end candies_per_pack_l110_110016


namespace cassandra_overall_score_l110_110550

theorem cassandra_overall_score 
  (score1_percent : ℤ) (score1_total : ℕ)
  (score2_percent : ℤ) (score2_total : ℕ)
  (score3_percent : ℤ) (score3_total : ℕ) :
  score1_percent = 60 → score1_total = 15 →
  score2_percent = 75 → score2_total = 20 →
  score3_percent = 85 → score3_total = 25 →
  let correct1 := (score1_percent * score1_total) / 100
  let correct2 := (score2_percent * score2_total) / 100
  let correct3 := (score3_percent * score3_total) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := score1_total + score2_total + score3_total
  75 = (100 * total_correct) / total_problems := by
  intros h1 h2 h3 h4 h5 h6
  let correct1 := (60 * 15) / 100
  let correct2 := (75 * 20) / 100
  let correct3 := (85 * 25) / 100
  let total_correct := correct1 + correct2 + correct3
  let total_problems := 15 + 20 + 25
  suffices 75 = (100 * total_correct) / total_problems by sorry
  sorry

end cassandra_overall_score_l110_110550


namespace square_diff_theorem_l110_110015

theorem square_diff_theorem
  (a b c p x : ℝ)
  (h1 : a + b + c = 2 * p)
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c))
  (h3 : c ≠ 0) :
  b^2 - x^2 = 4 / c^2 * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end square_diff_theorem_l110_110015


namespace a_when_a_minus_1_no_reciprocal_l110_110540

theorem a_when_a_minus_1_no_reciprocal (a : ℝ) (h : ¬ ∃ b : ℝ, (a - 1) * b = 1) : a = 1 := 
by
  sorry

end a_when_a_minus_1_no_reciprocal_l110_110540


namespace intersection_sets_l110_110146

theorem intersection_sets :
  let A := { x : ℝ | x^2 - 1 ≥ 0 }
  let B := { x : ℝ | 1 ≤ x ∧ x < 3 }
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end intersection_sets_l110_110146


namespace find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l110_110969

def f : ℝ → ℝ :=
  sorry

noncomputable def f_properties : Prop :=
  (∀ x y : ℝ, x < 0 → f x < 0 → f x + f y = f (x * y) / f (x + y)) ∧ f 1 = 1

theorem find_f2_f_neg1 :
  f_properties →
  f 2 = 1 / 2 ∧ f (-1) = -1 :=
sorry

theorem f_is_odd :
  f_properties →
  ∀ x : ℝ, f x = -f (-x) :=
sorry

theorem f_monotonic_on_negatives :
  f_properties →
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2 :=
sorry

end find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l110_110969


namespace sequence_sum_correct_l110_110983

theorem sequence_sum_correct :
  ∀ (r x y : ℝ),
  (x = 128 * r) →
  (y = x * r) →
  (2 * r = 1 / 2) →
  (x + y = 40) :=
by
  intros r x y hx hy hr
  sorry

end sequence_sum_correct_l110_110983


namespace find_k_b_l110_110924

noncomputable def symmetric_line_circle_intersection : Prop :=
  ∃ (k b : ℝ), 
    (∀ (x y : ℝ),  (y = k * x) ∧ ((x-1)^2 + y^2 = 1)) ∧ 
    (∀ (x y : ℝ), (x - y + b = 0)) →
    (k = -1 ∧ b = -1)

theorem find_k_b :
  symmetric_line_circle_intersection :=
  by
    -- omitted proof
    sorry

end find_k_b_l110_110924


namespace correct_misread_number_l110_110811

theorem correct_misread_number (s : List ℕ) (wrong_avg correct_avg n wrong_num correct_num : ℕ) 
  (h1 : s.length = 10) 
  (h2 : (s.sum) / n = wrong_avg) 
  (h3 : wrong_num = 26) 
  (h4 : correct_avg = 16) 
  (h5 : n = 10) 
  : correct_num = 36 :=
sorry

end correct_misread_number_l110_110811


namespace problem_l110_110941

theorem problem (x y z : ℝ) (h : (x - z) ^ 2 - 4 * (x - y) * (y - z) = 0) : z + x - 2 * y = 0 :=
sorry

end problem_l110_110941


namespace distance_cycled_l110_110797

variable (v t d : ℝ)

theorem distance_cycled (h1 : d = v * t)
                        (h2 : d = (v + 1) * (3 * t / 4))
                        (h3 : d = (v - 1) * (t + 3)) :
                        d = 36 :=
by
  sorry

end distance_cycled_l110_110797


namespace batsman_highest_score_l110_110413

theorem batsman_highest_score (H L : ℕ) 
  (h₁ : (40 * 50 = 2000)) 
  (h₂ : (H = L + 172))
  (h₃ : (38 * 48 = 1824)) :
  (2000 = 1824 + H + L) → H = 174 :=
by 
  sorry

end batsman_highest_score_l110_110413


namespace oldest_child_age_correct_l110_110843

-- Defining the conditions
def jane_start_age := 16
def jane_current_age := 32
def jane_stopped_babysitting_years_ago := 10
def half (x : ℕ) := x / 2

-- Expressing the conditions
def jane_last_babysitting_age := jane_current_age - jane_stopped_babysitting_years_ago
def max_child_age_when_jane_stopped := half jane_last_babysitting_age
def years_since_jane_stopped := jane_stopped_babysitting_years_ago

def calculate_oldest_child_current_age (age : ℕ) : ℕ :=
  age + years_since_jane_stopped

def child_age_when_stopped := max_child_age_when_jane_stopped
def expected_oldest_child_current_age := 21

-- The theorem stating the equivalence
theorem oldest_child_age_correct : 
  calculate_oldest_child_current_age child_age_when_stopped = expected_oldest_child_current_age :=
by
  -- Proof here
  sorry

end oldest_child_age_correct_l110_110843


namespace total_pies_bigger_event_l110_110970

def pies_last_week := 16.5
def apple_pies_last_week := 14.25
def cherry_pies_last_week := 12.75

def pecan_multiplier := 4.3
def apple_multiplier := 3.5
def cherry_multiplier := 5.7

theorem total_pies_bigger_event :
  (pies_last_week * pecan_multiplier) + 
  (apple_pies_last_week * apple_multiplier) + 
  (cherry_pies_last_week * cherry_multiplier) = 193.5 :=
by
  sorry

end total_pies_bigger_event_l110_110970


namespace john_younger_than_mark_l110_110290

variable (Mark_age John_age Parents_age : ℕ)
variable (h_mark : Mark_age = 18)
variable (h_parents_age_relation : Parents_age = 5 * John_age)
variable (h_parents_when_mark_born : Parents_age = 22 + Mark_age)

theorem john_younger_than_mark : Mark_age - John_age = 10 :=
by
  -- We state the theorem and leave the proof as sorry
  sorry

end john_younger_than_mark_l110_110290


namespace painting_time_eq_l110_110801

theorem painting_time_eq (t : ℝ) :
  (1 / 6 + 1 / 8 + 1 / 12) * t = 1 ↔ t = 8 / 3 :=
by
  sorry

end painting_time_eq_l110_110801


namespace mean_score_of_all_students_l110_110265

-- Define the conditions as given in the problem
variables (M A : ℝ) (m a : ℝ)
  (hM : M = 90)
  (hA : A = 75)
  (hRatio : m / a = 2 / 5)

-- State the theorem which proves that the mean score of all students is 79
theorem mean_score_of_all_students (hM : M = 90) (hA : A = 75) (hRatio : m / a = 2 / 5) : 
  (36 * a + 75 * a) / ((2 / 5) * a + a) = 79 := 
by
  sorry -- Proof is omitted

end mean_score_of_all_students_l110_110265


namespace terminating_decimal_zeros_l110_110365

-- Define a generic environment for terminating decimal and problem statement
def count_zeros (d : ℚ) : ℕ :=
  -- This function needs to count the zeros after the decimal point and before
  -- the first non-zero digit, but its actual implementation is skipped here.
  sorry

-- Define the specific fraction in question
def my_fraction : ℚ := 1 / (2^3 * 5^5)

-- State what we need to prove: the number of zeros after the decimal point
-- in the terminating representation of my_fraction should be 4
theorem terminating_decimal_zeros : count_zeros my_fraction = 4 :=
by
  -- Proof is skipped
  sorry

end terminating_decimal_zeros_l110_110365


namespace isabel_initial_amount_l110_110422

theorem isabel_initial_amount (X : ℝ) (h : X / 2 - X / 4 = 51) : X = 204 :=
sorry

end isabel_initial_amount_l110_110422


namespace train_passes_jogger_in_approximately_25_8_seconds_l110_110888

noncomputable def jogger_speed_kmh := 7
noncomputable def train_speed_kmh := 60
noncomputable def jogger_head_start_m := 180
noncomputable def train_length_m := 200

noncomputable def kmh_to_ms (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_ms := kmh_to_ms jogger_speed_kmh
noncomputable def train_speed_ms := kmh_to_ms train_speed_kmh

noncomputable def relative_speed_ms := train_speed_ms - jogger_speed_ms
noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m
noncomputable def time_to_pass_sec := total_distance_to_cover_m / (relative_speed_ms : ℝ) 

theorem train_passes_jogger_in_approximately_25_8_seconds :
  abs (time_to_pass_sec - 25.8) < 0.1 := sorry

end train_passes_jogger_in_approximately_25_8_seconds_l110_110888


namespace sam_total_cans_l110_110467

theorem sam_total_cans (bags_saturday bags_sunday bags_total cans_per_bag total_cans : ℕ)
    (h1 : bags_saturday = 3)
    (h2 : bags_sunday = 4)
    (h3 : bags_total = bags_saturday + bags_sunday)
    (h4 : cans_per_bag = 9)
    (h5 : total_cans = bags_total * cans_per_bag) : total_cans = 63 :=
sorry

end sam_total_cans_l110_110467


namespace initially_had_8_l110_110684

-- Define the number of puppies given away
def given_away : ℕ := 4

-- Define the number of puppies still with Sandy
def still_has : ℕ := 4

-- Define the total number of puppies initially
def initially_had (x y : ℕ) : ℕ := x + y

-- Prove that the number of puppies Sandy's dog had initially equals 8
theorem initially_had_8 : initially_had given_away still_has = 8 :=
by sorry

end initially_had_8_l110_110684


namespace initial_fraction_of_larger_jar_l110_110311

theorem initial_fraction_of_larger_jar (S L W : ℝ) 
  (h1 : W = 1/6 * S) 
  (h2 : W = 1/3 * L) : 
  W / L = 1 / 3 := 
by 
  sorry

end initial_fraction_of_larger_jar_l110_110311


namespace x_pow_4_minus_inv_x_pow_4_eq_727_l110_110759

theorem x_pow_4_minus_inv_x_pow_4_eq_727 (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 :=
by
  sorry

end x_pow_4_minus_inv_x_pow_4_eq_727_l110_110759


namespace isosceles_triangle_perimeter_l110_110987

theorem isosceles_triangle_perimeter (x y : ℝ) (h : 4 * x ^ 2 + 17 * y ^ 2 - 16 * x * y - 4 * y + 4 = 0):
  x = 4 ∧ y = 2 → 2 * x + y = 10 :=
by
  intros
  sorry

end isosceles_triangle_perimeter_l110_110987


namespace find_line_equation_l110_110316

-- Definitions: Point and Line in 2D
structure Point2D where
  x : ℝ
  y : ℝ

structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Line passes through the point
def line_through_point (L : Line2D) (P : Point2D) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Perpendicular lines condition: if Line L1 and Line L2 are perpendicular.
def perpendicular (L1 L2 : Line2D) : Prop :=
  L1.a * L2.a + L1.b * L2.b = 0

-- Define line1 and line2 as given
def line1 : Line2D := {a := 1, b := -2, c := 0} -- corresponds to x - 2y + m = 0

-- Define point P (-1, 3)
def P : Point2D := {x := -1, y := 3}

-- Required line passing through point P and perpendicular to line1
def required_line : Line2D := {a := 2, b := 1, c := -1}

-- The proof goal
theorem find_line_equation : (line_through_point required_line P) ∧ (perpendicular line1 required_line) :=
by
  sorry

end find_line_equation_l110_110316


namespace calculate_value_l110_110436

theorem calculate_value : (24 + 12) / ((5 - 3) * 2) = 9 := by 
  sorry

end calculate_value_l110_110436


namespace value_of_d_l110_110954

theorem value_of_d (d : ℝ) (h : ∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15 → True) : d = 5 :=
sorry

end value_of_d_l110_110954


namespace smallest_n_satisfying_mod_cond_l110_110990

theorem smallest_n_satisfying_mod_cond (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end smallest_n_satisfying_mod_cond_l110_110990


namespace tan_double_angle_l110_110652

theorem tan_double_angle (x : ℝ) (h : (Real.sqrt 3) * Real.cos x - Real.sin x = 0) : Real.tan (2 * x) = - (Real.sqrt 3) :=
by
  sorry

end tan_double_angle_l110_110652


namespace system_of_equations_solutions_l110_110590

theorem system_of_equations_solutions (x y z : ℝ) :
  (x^2 - y^2 + z = 27 / (x * y)) ∧ 
  (y^2 - z^2 + x = 27 / (y * z)) ∧ 
  (z^2 - x^2 + y = 27 / (z * x)) ↔ 
  (x = 3 ∧ y = 3 ∧ z = 3) ∨
  (x = -3 ∧ y = -3 ∧ z = 3) ∨
  (x = -3 ∧ y = 3 ∧ z = -3) ∨
  (x = 3 ∧ y = -3 ∧ z = -3) :=
by 
  sorry

end system_of_equations_solutions_l110_110590


namespace coterminal_angle_l110_110875

theorem coterminal_angle (theta : ℝ) (lower : ℝ) (upper : ℝ) (k : ℤ) : 
  -950 = k * 360 + theta ∧ (lower ≤ theta ∧ theta ≤ upper) → theta = 130 :=
by
  -- Given conditions
  sorry

end coterminal_angle_l110_110875


namespace probability_reach_origin_from_3_3_l110_110712

noncomputable def P : ℕ → ℕ → ℚ
| 0, 0 => 1
| x+1, 0 => 0
| 0, y+1 => 0
| x+1, y+1 => (1/3) * P x (y+1) + (1/3) * P (x+1) y + (1/3) * P x y

theorem probability_reach_origin_from_3_3 : P 3 3 = 1 / 27 := by
  sorry

end probability_reach_origin_from_3_3_l110_110712


namespace min_value_proof_l110_110519

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  9 / a + 16 / b + 25 / (c ^ 2)

theorem min_value_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 5) :
  minimum_value a b c ≥ 50 :=
sorry

end min_value_proof_l110_110519


namespace Danny_found_11_wrappers_l110_110424

theorem Danny_found_11_wrappers :
  ∃ wrappers_at_park : ℕ,
  (wrappers_at_park = 11) ∧
  (∃ bottle_caps : ℕ, bottle_caps = 12) ∧
  (∃ found_bottle_caps : ℕ, found_bottle_caps = 58) ∧
  (wrappers_at_park + 1 = bottle_caps) :=
by
  sorry

end Danny_found_11_wrappers_l110_110424


namespace simplify_division_l110_110396

noncomputable def simplify_expression (m : ℝ) : ℝ :=
  (m^2 - 3 * m + 1) / m + 1

noncomputable def divisor_expression (m : ℝ) : ℝ :=
  (m^2 - 1) / m

theorem simplify_division (m : ℝ) (hm1 : m ≠ 0) (hm2 : m ≠ 1) (hm3 : m ≠ -1) :
  (simplify_expression m) / (divisor_expression m) = (m - 1) / (m + 1) :=
by {
  sorry
}

end simplify_division_l110_110396


namespace four_prime_prime_l110_110601

-- Define the function based on the given condition
def q' (q : ℕ) : ℕ := 3 * q - 3

-- The statement to prove
theorem four_prime_prime : (q' (q' 4)) = 24 := by
  sorry

end four_prime_prime_l110_110601


namespace inequality_l110_110566

noncomputable def x : ℝ := Real.sqrt 3
noncomputable def y : ℝ := Real.log 2 / Real.log 3
noncomputable def z : ℝ := Real.cos 2

theorem inequality : z < y ∧ y < x := by
  sorry

end inequality_l110_110566


namespace surface_area_ratio_l110_110103

-- Defining conditions
variable (V_E V_J : ℝ) (A_E A_J : ℝ)
variable (volume_ratio : V_J = 30 * (Real.sqrt 30) * V_E)

-- Statement to prove
theorem surface_area_ratio (h : V_J = 30 * (Real.sqrt 30) * V_E) :
  A_J = 30 * A_E :=
by
  sorry

end surface_area_ratio_l110_110103


namespace points_on_line_l110_110862

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l110_110862


namespace correct_calculation_l110_110188

theorem correct_calculation :
  -4^2 / (-2)^3 * (-1 / 8) = -1 / 4 := by
  sorry

end correct_calculation_l110_110188


namespace larry_wins_probability_eq_l110_110922

-- Define the conditions
def larry_probability_knocks_off : ℚ := 1 / 3
def julius_probability_knocks_off : ℚ := 1 / 4
def larry_throws_first : Prop := True
def independent_events : Prop := True

-- Define the proof that Larry wins the game with probability 2/3
theorem larry_wins_probability_eq :
  larry_throws_first ∧ independent_events →
  larry_probability_knocks_off = 1/3 ∧ julius_probability_knocks_off = 1/4 →
  ∃ p : ℚ, p = 2 / 3 :=
by
  sorry

end larry_wins_probability_eq_l110_110922


namespace smallest_circle_area_l110_110510

/-- The smallest possible area of a circle passing through two given points in the coordinate plane. -/
theorem smallest_circle_area (P Q : ℝ × ℝ) (hP : P = (-3, -2)) (hQ : Q = (2, 4)) : 
  ∃ (A : ℝ), A = (61 * Real.pi) / 4 :=
by
  sorry

end smallest_circle_area_l110_110510


namespace women_in_first_class_equals_22_l110_110580

def number_of_women (total_passengers : Nat) : Nat :=
  total_passengers * 50 / 100

def number_of_women_in_first_class (number_of_women : Nat) : Nat :=
  number_of_women * 15 / 100

theorem women_in_first_class_equals_22 (total_passengers : Nat) (h1 : total_passengers = 300) : 
  number_of_women_in_first_class (number_of_women total_passengers) = 22 :=
by
  sorry

end women_in_first_class_equals_22_l110_110580


namespace peaches_left_at_stand_l110_110156

def initial_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def spoiled_peaches : ℝ := 12.0
def sold_peaches : ℝ := 27.0

theorem peaches_left_at_stand :
  initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 81.0 :=
by
  -- initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 84.0
  sorry

end peaches_left_at_stand_l110_110156


namespace jack_emails_morning_l110_110177

-- Definitions from conditions
def emails_evening : ℕ := 7
def additional_emails_morning : ℕ := 2
def emails_morning : ℕ := emails_evening + additional_emails_morning

-- The proof problem
theorem jack_emails_morning : emails_morning = 9 := by
  -- proof goes here
  sorry

end jack_emails_morning_l110_110177


namespace cubes_with_even_red_faces_l110_110675

theorem cubes_with_even_red_faces :
  let block_dimensions := (5, 5, 1)
  let painted_sides := 6
  let total_cubes := 25
  let cubes_with_2_red_faces := 16
  cubes_with_2_red_faces = 16 := by
  sorry

end cubes_with_even_red_faces_l110_110675


namespace sixth_employee_salary_l110_110429

def salaries : List Real := [1000, 2500, 3100, 3650, 1500]

def mean_salary_of_six : Real := 2291.67

theorem sixth_employee_salary : 
  let total_five := salaries.sum 
  let total_six := mean_salary_of_six * 6
  (total_six - total_five) = 2000.02 :=
by
  sorry

end sixth_employee_salary_l110_110429


namespace imaginary_part_of_z_is_sqrt2_div2_l110_110055

open Complex

noncomputable def z : ℂ := abs (1 - I) / (1 - I)

theorem imaginary_part_of_z_is_sqrt2_div2 : z.im = Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_is_sqrt2_div2_l110_110055


namespace each_boy_receives_52_l110_110356

theorem each_boy_receives_52 {boys girls : ℕ} (h_ratio : boys / gcd boys girls = 5 ∧ girls / gcd boys girls = 7) (h_total : boys + girls = 180) (h_share : 3900 ∣ boys) :
  3900 / boys = 52 :=
by
  sorry

end each_boy_receives_52_l110_110356


namespace pages_with_money_l110_110331

def cost_per_page : ℝ := 3.5
def total_money : ℝ := 15 * 100

theorem pages_with_money : ⌊total_money / cost_per_page⌋ = 428 :=
by sorry

end pages_with_money_l110_110331


namespace team_E_not_played_against_team_B_l110_110897

-- Define the teams
inductive Team
| A | B | C | D | E | F
deriving DecidableEq

open Team

-- Define the matches played by each team
def matches_played : Team → Nat
| A => 5
| B => 4
| C => 3
| D => 2
| E => 1
| F => 0

-- Define the pairwise matches function
def paired : Team → Team → Prop
| A, B => true
| A, C => true
| A, D => true
| A, E => true
| A, F => true
| B, C => true
| B, D => true
| B, F  => true
| _, _ => false

-- Define the theorem based on the conditions and question
theorem team_E_not_played_against_team_B :
  ¬ paired E B :=
by
  sorry

end team_E_not_played_against_team_B_l110_110897


namespace min_shift_odd_func_l110_110605

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem min_shift_odd_func (hφ : ∀ x : ℝ, f (x) = -f (-x + 2 * φ + (Real.pi / 3))) (hφ_positive : φ > 0) :
  φ = Real.pi / 6 :=
sorry

end min_shift_odd_func_l110_110605


namespace greatest_integer_difference_l110_110704

theorem greatest_integer_difference (x y : ℤ) (hx : -6 < (x : ℝ)) (hx2 : (x : ℝ) < -2) (hy : 4 < (y : ℝ)) (hy2 : (y : ℝ) < 10) : 
  ∃ d : ℤ, d = y - x ∧ d = 14 := 
by
  sorry

end greatest_integer_difference_l110_110704


namespace college_students_count_l110_110758

theorem college_students_count (girls boys : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ)
(h_ratio : ratio_boys = 6) (h_ratio_girls : ratio_girls = 5)
(h_girls : girls = 200)
(h_boys : boys = ratio_boys * (girls / ratio_girls)) :
  boys + girls = 440 := by
  sorry

end college_students_count_l110_110758


namespace part_a_part_b_l110_110591

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem part_a :
  ¬∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^3) :=
sorry

theorem part_b :
  ∃ (f g : ℝ → ℝ), (∀ x, f (g x) = x^2) ∧ (∀ x, g (f x) = x^4) :=
sorry

end part_a_part_b_l110_110591


namespace total_cost_price_l110_110760

theorem total_cost_price (C O B : ℝ) 
    (hC : 1.25 * C = 8340) 
    (hO : 1.30 * O = 4675) 
    (hB : 1.20 * B = 3600) : 
    C + O + B = 13268.15 := 
by 
    sorry

end total_cost_price_l110_110760


namespace minimum_value_of_functions_l110_110498

def linear_fn (a b c: ℝ) := a ≠ 0 
def f (a b: ℝ) (x: ℝ) := a * x + b 
def g (a c: ℝ) (x: ℝ) := a * x + c

theorem minimum_value_of_functions (a b c: ℝ) (hx: linear_fn a b c) :
  (∀ x: ℝ, 3 * (f a b x)^2 + 2 * g a c x ≥ -19 / 6) → (∀ x: ℝ, 3 * (g a c x)^2 + 2 * f a b x ≥ 5 / 2) :=
by
  sorry

end minimum_value_of_functions_l110_110498


namespace non_red_fraction_l110_110475

-- Define the conditions
def cube_edge : ℕ := 4
def num_cubes : ℕ := 64
def num_red_cubes : ℕ := 48
def num_white_cubes : ℕ := 12
def num_blue_cubes : ℕ := 4
def total_surface_area : ℕ := 6 * (cube_edge * cube_edge)

-- Define the non-red surface area exposed
def white_cube_exposed_area : ℕ := 12
def blue_cube_exposed_area : ℕ := 0

-- Calculating non-red area
def non_red_surface_area : ℕ := white_cube_exposed_area + blue_cube_exposed_area

-- The theorem to prove
theorem non_red_fraction (cube_edge : ℕ) (num_cubes : ℕ) (num_red_cubes : ℕ) 
  (num_white_cubes : ℕ) (num_blue_cubes : ℕ) (total_surface_area : ℕ) 
  (non_red_surface_area : ℕ) : 
  (non_red_surface_area : ℚ) / (total_surface_area : ℚ) = 1 / 8 :=
by 
  sorry

end non_red_fraction_l110_110475


namespace find_a_l110_110908

theorem find_a (a x y : ℝ)
    (h1 : a * x - 5 * y = 5)
    (h2 : x / (x + y) = 5 / 7)
    (h3 : x - y = 3) :
    a = 3 := 
by 
  sorry

end find_a_l110_110908


namespace find_five_dollar_bills_l110_110178

-- Define the number of bills
def total_bills (x y : ℕ) : Prop := x + y = 126

-- Define the total value of the bills
def total_value (x y : ℕ) : Prop := 5 * x + 10 * y = 840

-- Now we state the theorem
theorem find_five_dollar_bills (x y : ℕ) (h1 : total_bills x y) (h2 : total_value x y) : x = 84 :=
by sorry

end find_five_dollar_bills_l110_110178


namespace value_of_expression_l110_110783

theorem value_of_expression (a b c : ℚ) (h1 : a * b * c < 0) (h2 : a + b + c = 0) :
    (a - b - c) / |a| + (b - c - a) / |b| + (c - a - b) / |c| = 2 :=
by
  sorry

end value_of_expression_l110_110783


namespace a_1995_is_squared_l110_110206

variable (a : ℕ → ℕ)

-- Conditions on the sequence 
axiom seq_condition  {m n : ℕ} (h : m ≥ n) : 
  a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

axiom initial_value : a 1 = 1

-- Goal to prove
theorem a_1995_is_squared : a 1995 = 1995^2 :=
sorry

end a_1995_is_squared_l110_110206


namespace problem_inequality_l110_110950

theorem problem_inequality (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x) → (x ≤ 2) → (x^2 + 2 + |x^3 - 2 * x| ≥ a * x)) ↔ (a ≤ 2 * Real.sqrt 2) := 
sorry

end problem_inequality_l110_110950


namespace distance_preserving_l110_110192

variables {Point : Type} {d : Point → Point → ℕ} {f : Point → Point}

axiom distance_one (A B : Point) : d A B = 1 → d (f A) (f B) = 1

theorem distance_preserving :
  ∀ (A B : Point) (n : ℕ), n > 0 → d A B = n → d (f A) (f B) = n :=
by
  sorry

end distance_preserving_l110_110192


namespace find_value_l110_110781

theorem find_value 
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 2)
  (h2 : c / d = 1 / 2)
  (h3 : e / f = 1 / 2)
  (h4 : 3 * b - 2 * d + f ≠ 0) : 
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 2 := 
by
  sorry

end find_value_l110_110781


namespace multiple_of_old_edition_l110_110023

theorem multiple_of_old_edition 
  (new_pages: ℕ) 
  (old_pages: ℕ) 
  (difference: ℕ) 
  (m: ℕ) 
  (h1: new_pages = 450) 
  (h2: old_pages = 340) 
  (h3: 450 = 340 * m - 230) : 
  m = 2 :=
sorry

end multiple_of_old_edition_l110_110023


namespace meeting_point_ratio_l110_110049

theorem meeting_point_ratio (v1 v2 : ℝ) (TA TB : ℝ)
  (h1 : TA = 45 * v2)
  (h2 : TB = 20 * v1)
  (h3 : (TA / v1) - (TB / v2) = 11) :
  TA / TB = 9 / 5 :=
by sorry

end meeting_point_ratio_l110_110049


namespace valentine_cards_l110_110542

theorem valentine_cards (x y : ℕ) (h : x * y = x + y + 18) : x * y = 40 :=
by
  sorry

end valentine_cards_l110_110542


namespace solve_system_of_odes_l110_110351

theorem solve_system_of_odes (C₁ C₂ : ℝ) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = (C₁ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, y t = (C₁ + C₂ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, deriv x t = 2 * x t + y t) ∧
    (∀ t, deriv y t = 4 * y t - x t) :=
by
  sorry

end solve_system_of_odes_l110_110351


namespace find_pumpkin_seed_packets_l110_110359

variable (P : ℕ)

-- Problem assumptions (conditions)
def pumpkin_seed_cost : ℝ := 2.50
def tomato_seed_cost_total : ℝ := 1.50 * 4
def chili_pepper_seed_cost_total : ℝ := 0.90 * 5
def total_spent : ℝ := 18.00

-- Main theorem to prove
theorem find_pumpkin_seed_packets (P : ℕ) (h : (pumpkin_seed_cost * P) + tomato_seed_cost_total + chili_pepper_seed_cost_total = total_spent) : P = 3 := by sorry

end find_pumpkin_seed_packets_l110_110359


namespace initial_people_per_column_l110_110679

theorem initial_people_per_column (P x : ℕ) (h1 : P = 16 * x) (h2 : P = 48 * 10) : x = 30 :=
by 
  sorry

end initial_people_per_column_l110_110679


namespace total_fuel_needed_l110_110438

/-- Given that Car B can travel 30 miles per gallon and needs to cover a distance of 750 miles,
    and Car C has a fuel consumption rate of 20 miles per gallon and will travel 900 miles,
    prove that the total combined fuel required for Cars B and C is 70 gallons. -/
theorem total_fuel_needed (miles_per_gallon_B : ℕ) (miles_per_gallon_C : ℕ)
  (distance_B : ℕ) (distance_C : ℕ)
  (hB : miles_per_gallon_B = 30) (hC : miles_per_gallon_C = 20)
  (dB : distance_B = 750) (dC : distance_C = 900) :
  (distance_B / miles_per_gallon_B) + (distance_C / miles_per_gallon_C) = 70 := by {
    sorry 
}

end total_fuel_needed_l110_110438


namespace numbers_not_necessarily_equal_l110_110452

theorem numbers_not_necessarily_equal (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b^2 + c^2 = b + a^2 + c^2) (h2 : a + b^2 + c^2 = c + a^2 + b^2) : 
  ¬(a = b ∧ b = c) := 
sorry

end numbers_not_necessarily_equal_l110_110452


namespace solve_B_l110_110526

theorem solve_B (B : ℕ) (h1 : 0 ≤ B) (h2 : B ≤ 9) (h3 : 7 ∣ (4000 + 110 * B + 2)) : B = 4 :=
by
  sorry

end solve_B_l110_110526


namespace boa_constrictor_is_70_inches_l110_110996

-- Definitions based on given problem conditions
def garden_snake_length : ℕ := 10
def boa_constrictor_length : ℕ := 7 * garden_snake_length

-- Statement to prove
theorem boa_constrictor_is_70_inches : boa_constrictor_length = 70 :=
by
  sorry

end boa_constrictor_is_70_inches_l110_110996


namespace Eunji_has_most_marbles_l110_110128

-- Declare constants for each person's marbles
def Minyoung_marbles : ℕ := 4
def Yujeong_marbles : ℕ := 2
def Eunji_marbles : ℕ := Minyoung_marbles + 1

-- Theorem: Eunji has the most marbles
theorem Eunji_has_most_marbles :
  Eunji_marbles > Minyoung_marbles ∧ Eunji_marbles > Yujeong_marbles :=
by
  sorry

end Eunji_has_most_marbles_l110_110128


namespace sin_2y_eq_37_40_l110_110733

variable (x y : ℝ)
variable (sin cos : ℝ → ℝ)

axiom sin_def : sin x = 2 * cos y - (5/2) * sin y
axiom cos_def : cos x = 2 * sin y - (5/2) * cos y

theorem sin_2y_eq_37_40 : sin (2 * y) = 37 / 40 := by
  sorry

end sin_2y_eq_37_40_l110_110733


namespace largest_among_four_l110_110940

theorem largest_among_four (a b : ℝ) (h : 0 < a ∧ a < b ∧ a + b = 1) :
  a^2 + b^2 = max (max (max a (1/2)) (2*a*b)) (a^2 + b^2) :=
by
  sorry

end largest_among_four_l110_110940


namespace siblings_water_intake_l110_110110

theorem siblings_water_intake 
  (theo_daily : ℕ := 8) 
  (mason_daily : ℕ := 7) 
  (roxy_daily : ℕ := 9) 
  (days_in_week : ℕ := 7) 
  : (theo_daily + mason_daily + roxy_daily) * days_in_week = 168 := 
by 
  sorry

end siblings_water_intake_l110_110110


namespace fewer_white_chairs_than_green_blue_l110_110544

-- Definitions of the conditions
def blue_chairs : ℕ := 10
def green_chairs : ℕ := 3 * blue_chairs
def total_chairs : ℕ := 67
def green_blue_chairs : ℕ := green_chairs + blue_chairs
def white_chairs : ℕ := total_chairs - green_blue_chairs

-- Statement of the theorem
theorem fewer_white_chairs_than_green_blue : green_blue_chairs - white_chairs = 13 :=
by
  -- This is where the proof would go, but we're omitting it as per instruction
  sorry

end fewer_white_chairs_than_green_blue_l110_110544


namespace enthusiasts_min_max_l110_110595

-- Define the conditions
def total_students : ℕ := 100
def basketball_enthusiasts : ℕ := 63
def football_enthusiasts : ℕ := 75

-- Define the main proof problem
theorem enthusiasts_min_max :
  ∃ (common_enthusiasts : ℕ), 38 ≤ common_enthusiasts ∧ common_enthusiasts ≤ 63 :=
sorry

end enthusiasts_min_max_l110_110595


namespace exists_n_not_represented_l110_110458

theorem exists_n_not_represented (a b c d : ℤ) (a_gt_14 : a > 14)
  (h1 : 0 ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ a) :
  ∃ (n : ℕ), ¬ ∃ (x y z : ℤ), n = x * (a * x + b) + y * (a * y + c) + z * (a * z + d) :=
sorry

end exists_n_not_represented_l110_110458


namespace part1_part2_l110_110332

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x * abs (x^2 - a)

-- Define the two main proofs to be shown
theorem part1 (a : ℝ) (h : a = 1) : 
  ∃ I1 I2 : Set ℝ, I1 = Set.Icc (-1 - Real.sqrt 2) (-1) ∧ I2 = Set.Icc (-1 + Real.sqrt 2) (1) ∧ 
  ∀ x ∈ I1 ∪ I2, ∀ y ∈ I1 ∪ I2, x ≤ y → f y 1 ≤ f x 1 :=
sorry

theorem part2 (a : ℝ) (h : a ≥ 0) (h_roots : ∀ m : ℝ, (∃ x : ℝ, x > 0 ∧ f x a = m) ∧ (∃ x : ℝ, x < 0 ∧ f x a = m)) : 
  ∃ m : ℝ, m = 4 / (Real.exp 2) :=
sorry

end part1_part2_l110_110332


namespace car_speed_l110_110659

theorem car_speed (distance time : ℝ) (h_distance : distance = 275) (h_time : time = 5) : (distance / time = 55) :=
by
  sorry

end car_speed_l110_110659


namespace convert_C_to_F_l110_110318

theorem convert_C_to_F (C F : ℝ) (h1 : C = 40) (h2 : C = 5 / 9 * (F - 32)) : F = 104 := 
by
  -- Proof goes here
  sorry

end convert_C_to_F_l110_110318


namespace diamond_comm_l110_110271

def diamond (a b : ℝ) : ℝ := a^2 * b^2 - a^2 - b^2

theorem diamond_comm (x y : ℝ) : diamond x y = diamond y x := by
  sorry

end diamond_comm_l110_110271


namespace measure_four_messzely_l110_110335

theorem measure_four_messzely (c3 c5 : ℕ) (hc3 : c3 = 3) (hc5 : c5 = 5) : 
  ∃ (x y z : ℕ), x = 4 ∧ x + y * c3 + z * c5 = 4 := 
sorry

end measure_four_messzely_l110_110335


namespace blue_pill_cost_l110_110140

theorem blue_pill_cost
  (days : Int := 10)
  (total_expenditure : Int := 430)
  (daily_cost : Int := total_expenditure / days) :
  ∃ (y : Int), y + (y - 3) = daily_cost ∧ y = 23 := by
  sorry

end blue_pill_cost_l110_110140


namespace time_per_lawn_in_minutes_l110_110993

def jason_lawns := 16
def total_hours_cutting := 8
def minutes_per_hour := 60

theorem time_per_lawn_in_minutes : 
  (total_hours_cutting / jason_lawns) * minutes_per_hour = 30 :=
by
  sorry

end time_per_lawn_in_minutes_l110_110993


namespace tricycle_count_l110_110810

variables (b t : ℕ)

theorem tricycle_count :
  b + t = 7 ∧ 2 * b + 3 * t = 19 → t = 5 := by
  intro h
  sorry

end tricycle_count_l110_110810


namespace div_240_of_prime_diff_l110_110939

-- Definitions
def is_prime (n : ℕ) : Prop := ∃ p : ℕ, p = n ∧ Prime p
def prime_with_two_digits (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ is_prime n

-- The theorem statement
theorem div_240_of_prime_diff (a b : ℕ) (ha : prime_with_two_digits a) (hb : prime_with_two_digits b) (h : a > b) :
  240 ∣ (a^4 - b^4) ∧ ∀ d : ℕ, (d ∣ (a^4 - b^4) → (∀ m n : ℕ, prime_with_two_digits m → prime_with_two_digits n → m > n → d ∣ (m^4 - n^4) ) → d ≤ 240) :=
by
  sorry

end div_240_of_prime_diff_l110_110939


namespace num_sequences_of_student_helpers_l110_110903

-- Define the conditions
def num_students : ℕ := 15
def num_meetings : ℕ := 3

-- Define the statement to prove
theorem num_sequences_of_student_helpers : 
  (num_students ^ num_meetings) = 3375 :=
by sorry

end num_sequences_of_student_helpers_l110_110903


namespace urban_general_hospital_problem_l110_110876

theorem urban_general_hospital_problem
  (a b c d : ℕ)
  (h1 : b = 3 * c)
  (h2 : a = 2 * b)
  (h3 : d = c / 2)
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1500) :
  5 * d = 1500 / 11 := by
  sorry

end urban_general_hospital_problem_l110_110876


namespace value_of_y_minus_x_l110_110050

theorem value_of_y_minus_x (x y : ℝ) (h1 : abs (x + 1) = 3) (h2 : abs y = 5) (h3 : -y / x > 0) :
  y - x = -7 ∨ y - x = 9 :=
sorry

end value_of_y_minus_x_l110_110050


namespace weights_of_first_two_cats_l110_110772

noncomputable def cats_weight_proof (W : ℝ) : Prop :=
  (∀ (w1 w2 : ℝ), w1 = W ∧ w2 = W ∧ (w1 + w2 + 14.7 + 9.3) / 4 = 12) → (W = 12)

theorem weights_of_first_two_cats (W : ℝ) :
  cats_weight_proof W :=
by
  sorry

end weights_of_first_two_cats_l110_110772


namespace square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l110_110716

-- Define the problem conditions.
def square_grid (n : Nat) : Prop := true
def rectangle_grid (m n : Nat) : Prop := true

-- Define the grid size for square and rectangle.
def square_grid_21 := square_grid 21
def rectangle_grid_20_21 := rectangle_grid 20 21

-- Define the proof problem to find maximum moves.
theorem square_grid_21_max_moves : ∃ m : Nat, m = 3 :=
  sorry

theorem rectangle_grid_20_21_max_moves : ∃ m : Nat, m = 4 :=
  sorry

end square_grid_21_max_moves_rectangle_grid_20_21_max_moves_l110_110716


namespace range_of_f_l110_110919

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) ^ 2 - (Real.sin x) ^ 2 - 2 * (Real.sin x) * (Real.cos x)

theorem range_of_f : 
  (∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 → f x ≥ -Real.sqrt 2 ∧ f x ≤ 1) :=
sorry

end range_of_f_l110_110919


namespace constants_inequality_value_l110_110547

theorem constants_inequality_value
  (a b c d : ℝ)
  (h1 : a < b)
  (h2 : b < c)
  (h3 : ∀ x, (1 ≤ x ∧ x ≤ 5) ∨ (24 ≤ x ∧ x ≤ 26) ∨ x < -4 ↔ (x - a) * (x - b) * (x - c) / (x - d) ≤ 0) :
  a + 3 * b + 3 * c + 4 * d = 72 :=
sorry

end constants_inequality_value_l110_110547


namespace remainder_when_dividing_P_by_DDD_l110_110184

variables (P D D' D'' Q Q' Q'' R R' R'' : ℕ)

-- Define the conditions
def condition1 : Prop := P = Q * D + R
def condition2 : Prop := Q = Q' * D' + R'
def condition3 : Prop := Q' = Q'' * D'' + R''

-- Theorem statement asserting the given conclusion
theorem remainder_when_dividing_P_by_DDD' 
  (H1 : condition1 P D Q R)
  (H2 : condition2 Q D' Q' R')
  (H3 : condition3 Q' D'' Q'' R'') : 
  P % (D * D' * D') = R'' * D * D' + R * D' + R := 
sorry

end remainder_when_dividing_P_by_DDD_l110_110184


namespace star_sub_correctness_l110_110912

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_sub_correctness : (star 6 2) - (star 2 6) = -12 := by
  sorry

end star_sub_correctness_l110_110912


namespace eval_x_sq_minus_y_sq_l110_110926

theorem eval_x_sq_minus_y_sq (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 20) 
  (h2 : 4 * x + 3 * y = 29) : 
  x^2 - y^2 = -45 :=
sorry

end eval_x_sq_minus_y_sq_l110_110926


namespace find_a1_l110_110673

theorem find_a1 (a_1 : ℕ) (S : ℕ → ℕ) (S_formula : ∀ n : ℕ, S n = (a_1 * (3^n - 1)) / 2)
  (a_4_eq : (S 4) - (S 3) = 54) : a_1 = 2 :=
  sorry

end find_a1_l110_110673


namespace sin_theta_val_sin_2theta_pi_div_6_val_l110_110292

open Real

theorem sin_theta_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2) 
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin θ = (2 * sqrt 6 - 1) / 6 := 
by sorry

theorem sin_2theta_pi_div_6_val (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 2)
  (hcos : cos (θ + π / 6) = 1 / 3) : 
  sin (2 * θ + π / 6) = (4 * sqrt 6 + 7) / 18 := 
by sorry

end sin_theta_val_sin_2theta_pi_div_6_val_l110_110292


namespace product_of_roots_l110_110338

theorem product_of_roots : ∀ x : ℝ, (x + 3) * (x - 4) = 17 → (∃ a b : ℝ, (x = a ∨ x = b) ∧ a * b = -29) :=
by
  sorry

end product_of_roots_l110_110338


namespace mushrooms_on_log_l110_110648

theorem mushrooms_on_log :
  ∃ (G : ℕ), ∃ (S : ℕ), S = 9 * G ∧ G + S = 30 ∧ G = 3 :=
by
  sorry

end mushrooms_on_log_l110_110648


namespace not_detecting_spy_probability_l110_110183

-- Definitions based on conditions
def forest_size : ℝ := 10
def detection_radius : ℝ := 10

-- Inoperative detector - assuming NE corner
def detector_NE_inoperative : Prop := true

-- Probability calculation result
def probability_not_detected : ℝ := 0.087

-- Theorem to prove
theorem not_detecting_spy_probability :
  (forest_size = 10) ∧ (detection_radius = 10) ∧ detector_NE_inoperative →
  probability_not_detected = 0.087 :=
by
  sorry

end not_detecting_spy_probability_l110_110183


namespace white_triangle_pairs_condition_l110_110559

def number_of_white_pairs (total_triangles : Nat) 
                          (red_pairs : Nat) 
                          (blue_pairs : Nat)
                          (mixed_pairs : Nat) : Nat :=
  let red_involved := red_pairs * 2
  let blue_involved := blue_pairs * 2
  let remaining_red := total_triangles / 2 * 5 - red_involved - mixed_pairs
  let remaining_blue := total_triangles / 2 * 4 - blue_involved - mixed_pairs
  (total_triangles / 2 * 7) - (remaining_red + remaining_blue)/2

theorem white_triangle_pairs_condition : number_of_white_pairs 32 3 2 1 = 6 := by
  sorry

end white_triangle_pairs_condition_l110_110559


namespace mr_li_age_l110_110080

theorem mr_li_age (xiaofang_age : ℕ) (h1 : xiaofang_age = 5)
  (h2 : ∀ t : ℕ, (t = 3) → ∀ mr_li_age_in_3_years : ℕ, (mr_li_age_in_3_years = xiaofang_age + t + 20)) :
  ∃ mr_li_age : ℕ, mr_li_age = 25 :=
by
  sorry

end mr_li_age_l110_110080


namespace proof_x_plus_y_sum_l110_110896

noncomputable def x_and_y_sum (x y : ℝ) : Prop := 31.25 / x = 100 / 9.6 ∧ 13.75 / x = y / 9.6

theorem proof_x_plus_y_sum (x y : ℝ) (h : x_and_y_sum x y) : x + y = 47 :=
sorry

end proof_x_plus_y_sum_l110_110896


namespace ball_bounces_height_l110_110965

theorem ball_bounces_height (initial_height : ℝ) (decay_factor : ℝ) (threshold : ℝ) (n : ℕ) :
  initial_height = 20 →
  decay_factor = 3/4 →
  threshold = 2 →
  n = 9 →
  initial_height * (decay_factor ^ n) < threshold :=
by
  intros
  sorry

end ball_bounces_height_l110_110965


namespace A_share_in_profit_l110_110262

-- Given conditions:
def A_investment : ℕ := 6300
def B_investment : ℕ := 4200
def C_investment : ℕ := 10500
def total_profit : ℕ := 12600

-- The statement we need to prove:
theorem A_share_in_profit :
  (3 / 10) * total_profit = 3780 := by
  sorry

end A_share_in_profit_l110_110262


namespace perfect_square_condition_l110_110346

theorem perfect_square_condition (x m : ℝ) (h : ∃ k : ℝ, x^2 + x + 2*m = k^2) : m = 1/8 := 
sorry

end perfect_square_condition_l110_110346


namespace charity_fundraising_l110_110308

theorem charity_fundraising (num_people : ℕ) (amount_event1 amount_event2 : ℕ) (total_amount_per_person : ℕ) :
  num_people = 8 →
  amount_event1 = 2000 →
  amount_event2 = 1000 →
  total_amount_per_person = (amount_event1 + amount_event2) / num_people →
  total_amount_per_person = 375 :=
by
  intros h1 h2 h3 h4
  sorry

end charity_fundraising_l110_110308


namespace hyperbola_equation_l110_110215

variable (a b c : ℝ)

def system_eq1 := (4 / (-3 - c)) = (- a / b)
def system_eq2 := ((c - 3) / 2) * (b / a) = 2
def system_eq3 := a ^ 2 + b ^ 2 = c ^ 2

theorem hyperbola_equation (h1 : system_eq1 a b c) (h2 : system_eq2 a b c) (h3 : system_eq3 a b c) :
  ∃ a b : ℝ, c = 5 ∧ b^2 = 20 ∧ a^2 = 5 ∧ (∀ x y : ℝ, (x ^ 2 / 5) - (y ^ 2 / 20) = 1) :=
  sorry

end hyperbola_equation_l110_110215


namespace total_cost_of_apples_l110_110381

def original_price_per_pound : ℝ := 1.6
def price_increase_percentage : ℝ := 0.25
def number_of_family_members : ℕ := 4
def pounds_per_person : ℝ := 2

theorem total_cost_of_apples : 
  let new_price_per_pound := original_price_per_pound * (1 + price_increase_percentage)
  let total_pounds := pounds_per_person * number_of_family_members
  let total_cost := total_pounds * new_price_per_pound
  total_cost = 16 := by
  sorry

end total_cost_of_apples_l110_110381


namespace arithmetic_sequence_a2_a9_sum_l110_110431

theorem arithmetic_sequence_a2_a9_sum 
  (a : ℕ → ℝ) (d a₁ : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S10 : 10 * a 1 + 45 * d = 120) :
  a 2 + a 9 = 24 :=
sorry

end arithmetic_sequence_a2_a9_sum_l110_110431


namespace polynomial_root_product_l110_110058

theorem polynomial_root_product (b c : ℤ) : (∀ r : ℝ, r^2 - r - 1 = 0 → r^6 - b * r - c = 0) → b * c = 40 := 
by
  sorry

end polynomial_root_product_l110_110058


namespace even_blue_faces_cubes_correct_l110_110885

/-- A rectangular wooden block is 6 inches long, 3 inches wide, and 2 inches high.
    The block is painted blue on all six sides and then cut into 1 inch cubes.
    This function determines the number of 1-inch cubes that have a total number
    of blue faces that is an even number (in this case, 2 blue faces). -/
def count_even_blue_faces_cubes : Nat :=
  let length := 6
  let width := 3
  let height := 2
  let total_cubes := length * width * height
  
  -- Calculate corner cubes
  let corners := 8

  -- Calculate edges but not corners cubes
  let edge_not_corners := 
    (4 * (length - 2)) + 
    (4 * (width - 2)) + 
    (4 * (height - 2))

  -- Calculate even number of blue faces cubes 
  let even_number_blue_faces := edge_not_corners

  even_number_blue_faces

theorem even_blue_faces_cubes_correct : count_even_blue_faces_cubes = 20 := by
  -- Place your proof here.
  sorry

end even_blue_faces_cubes_correct_l110_110885


namespace new_rectangle_dimensions_l110_110747

theorem new_rectangle_dimensions (l w : ℕ) (h_l : l = 12) (h_w : w = 10) :
  ∃ l' w' : ℕ, l' = l ∧ w' = w / 2 ∧ l' = 12 ∧ w' = 5 :=
by
  sorry

end new_rectangle_dimensions_l110_110747


namespace matrix_mult_correct_l110_110831

-- Definition of matrices A and B
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![3, 1],
  ![4, -2]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![7, -3],
  ![2, 4]
]

-- The goal is to prove that A * B yields the matrix C
def matrix_product : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![23, -5],
  ![24, -20]
]

theorem matrix_mult_correct : A * B = matrix_product := by
  -- Proof omitted
  sorry

end matrix_mult_correct_l110_110831


namespace tan_theta_half_l110_110865

open Real

theorem tan_theta_half (θ : ℝ) 
  (h0 : 0 < θ) 
  (h1 : θ < π / 2) 
  (h2 : ∃ k : ℝ, (sin (2 * θ), cos θ) = k • (cos θ, 1)) : 
  tan θ = 1 / 2 := by 
sorry

end tan_theta_half_l110_110865


namespace sara_payment_equivalence_l110_110148

variable (cost_book1 cost_book2 change final_amount : ℝ)

theorem sara_payment_equivalence
  (h1 : cost_book1 = 5.5)
  (h2 : cost_book2 = 6.5)
  (h3 : change = 8)
  (h4 : final_amount = cost_book1 + cost_book2 + change) :
  final_amount = 20 := by
  sorry

end sara_payment_equivalence_l110_110148


namespace bandage_overlap_l110_110302

theorem bandage_overlap
  (n : ℕ) (l : ℝ) (total_length : ℝ) (required_length : ℝ)
  (h_n : n = 20) (h_l : l = 15.25) (h_required_length : required_length = 248) :
  (required_length = l * n - (n - 1) * 3) :=
by
  sorry

end bandage_overlap_l110_110302


namespace vertex_coordinates_l110_110892

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 8

-- State the theorem for the coordinates of the vertex
theorem vertex_coordinates : 
  (∃ h k : ℝ, ∀ x : ℝ, parabola x = 2 * (x - h)^2 + k) ∧ h = 1 ∧ k = 8 :=
sorry

end vertex_coordinates_l110_110892


namespace product_wavelengths_eq_n_cbrt_mn2_l110_110149

variable (m n : ℝ)

noncomputable def common_ratio (m n : ℝ) := (n / m)^(1/3)

noncomputable def wavelength_jiazhong (m n : ℝ) := (m^2 * n)^(1/3)
noncomputable def wavelength_nanlu (m n : ℝ) := (n^4 / m)^(1/3)

theorem product_wavelengths_eq_n_cbrt_mn2
  (h : n = m * (common_ratio m n)^3) :
  (wavelength_jiazhong m n) * (wavelength_nanlu m n) = n * (m * n^2)^(1/3) :=
by
  sorry

end product_wavelengths_eq_n_cbrt_mn2_l110_110149


namespace slope_of_AB_l110_110428

theorem slope_of_AB (k : ℝ) (y1 y2 x1 x2 : ℝ) 
  (hP : (1, Real.sqrt 2) ∈ {p : ℝ × ℝ | p.2^2 = 2*p.1})
  (hPA_eq : ∀ x, (x, y1) ∈ {p : ℝ × ℝ | p.2 = k * p.1 - k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hPB_eq : ∀ x, (x, y2) ∈ {p : ℝ × ℝ | p.2 = -k * p.1 + k + Real.sqrt 2 ∧ p.2^2 = 2 * p.1}) 
  (hx1 : y1 = k * x1 - k + Real.sqrt 2) 
  (hx2 : y2 = -k * x2 + k + Real.sqrt 2) :
  ((y2 - y1) / (x2 - x1)) = -2 - 2 * Real.sqrt 2 :=
by
  sorry

end slope_of_AB_l110_110428


namespace weekly_charge_for_motel_l110_110267

theorem weekly_charge_for_motel (W : ℝ) (h1 : ∀ t : ℝ, t = 3 * 4 → t = 12)
(h2 : ∀ cost_weekly : ℝ, cost_weekly = 12 * W)
(h3 : ∀ cost_monthly : ℝ, cost_monthly = 3 * 1000)
(h4 : cost_monthly + 360 = 12 * W) : 
W = 280 := 
sorry

end weekly_charge_for_motel_l110_110267


namespace area_enclosed_by_absolute_value_linear_eq_l110_110423

theorem area_enclosed_by_absolute_value_linear_eq (x y : ℝ) :
  (|5 * x| + |3 * y| = 15) → ∃ (A : ℝ), A = 30 :=
by
  sorry

end area_enclosed_by_absolute_value_linear_eq_l110_110423


namespace kids_on_excursions_l110_110882

theorem kids_on_excursions (total_kids : ℕ) (one_fourth_kids_tubing two := total_kids / 4) (half_tubers_rafting : ℕ := one_fourth_kids_tubing / 2) :
  total_kids = 40 → one_fourth_kids_tubing = 10 → half_tubers_rafting = 5 :=
by
  intros
  sorry

end kids_on_excursions_l110_110882


namespace part_a_part_b_part_c_l110_110154

theorem part_a (p q : ℝ) : q < p^2 → ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = 2 * p) ∧ (r1 * r2 = q) :=
by
  sorry

theorem part_b (p q : ℝ) : q = 4 * p - 4 → (2^2 - 2 * p * 2 + q = 0) :=
by
  sorry

theorem part_c (p q : ℝ) : q = p^2 ∧ q = 4 * p - 4 → (p = 2 ∧ q = 4) :=
by
  sorry

end part_a_part_b_part_c_l110_110154


namespace parabolas_intersect_at_single_point_l110_110008

theorem parabolas_intersect_at_single_point (p q : ℝ) (h : -2 * p + q = 2023) :
  ∃ (x0 y0 : ℝ), (∀ p q : ℝ, y0 = x0^2 + p * x0 + q → -2 * p + q = 2023) ∧ x0 = -2 ∧ y0 = 2027 :=
by
  -- Proof to be filled in
  sorry

end parabolas_intersect_at_single_point_l110_110008


namespace geometric_progression_problem_l110_110894

open Real

theorem geometric_progression_problem
  (a b c r : ℝ)
  (h1 : a = 20)
  (h2 : b = 40)
  (h3 : c = 10)
  (h4 : b = r * a)
  (h5 : c = r * b) :
  (a - (b - c)) - ((a - b) - c) = 20 := by
  sorry

end geometric_progression_problem_l110_110894


namespace total_product_l110_110746

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12 
  else if n % 2 = 0 then 4 
  else 0 

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem total_product : total_points allie_rolls * total_points betty_rolls = 1120 := sorry

end total_product_l110_110746


namespace first_part_lent_years_l110_110820

theorem first_part_lent_years (x n : ℕ) (total_sum second_sum : ℕ) (rate1 rate2 years2 : ℝ) :
  total_sum = 2743 →
  second_sum = 1688 →
  rate1 = 3 →
  rate2 = 5 →
  years2 = 3 →
  (x = total_sum - second_sum) →
  (x * n * rate1 / 100 = second_sum * rate2 * years2 / 100) →
  n = 8 :=
by
  sorry

end first_part_lent_years_l110_110820


namespace false_propositions_l110_110835

theorem false_propositions (p q : Prop) (hnp : ¬ p) (hq : q) :
  (¬ p) ∧ (¬ (p ∧ q)) ∧ (¬ ¬ q) :=
by {
  exact ⟨hnp, not_and_of_not_left q hnp, not_not_intro hq⟩
}

end false_propositions_l110_110835


namespace largest_square_with_five_interior_lattice_points_l110_110858

theorem largest_square_with_five_interior_lattice_points :
  ∃ (s : ℝ), (∀ (x y : ℤ), 1 ≤ x ∧ x < s ∧ 1 ≤ y ∧ y < s) → ((⌊s⌋ - 1)^2 = 5) ∧ s^2 = 18 := sorry

end largest_square_with_five_interior_lattice_points_l110_110858


namespace compare_P_Q_l110_110250

noncomputable def P (n : ℕ) (x : ℝ) : ℝ := (1 - x)^(2*n - 1)
noncomputable def Q (n : ℕ) (x : ℝ) : ℝ := 1 - (2*n - 1)*x + (n - 1)*(2*n - 1)*x^2

theorem compare_P_Q :
  ∀ (n : ℕ) (x : ℝ), n > 0 →
  ((n = 1 → P n x = Q n x) ∧
   (n = 2 → ((x = 0 → P n x = Q n x) ∧ (x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x))) ∧
   (n ≥ 3 → ((x > 0 → P n x < Q n x) ∧ (x < 0 → P n x > Q n x)))) :=
by
  intros
  sorry

end compare_P_Q_l110_110250


namespace Mark_sold_1_box_less_than_n_l110_110361

variable (M A n : ℕ)

theorem Mark_sold_1_box_less_than_n (h1 : n = 8)
 (h2 : A = n - 2)
 (h3 : M + A < n)
 (h4 : M ≥ 1) 
 (h5 : A ≥ 1)
 : M = 1 := 
sorry

end Mark_sold_1_box_less_than_n_l110_110361


namespace shelby_drive_rain_minutes_l110_110596

theorem shelby_drive_rain_minutes
  (total_distance : ℝ)
  (total_time : ℝ)
  (sunny_speed : ℝ)
  (rainy_speed : ℝ)
  (t_sunny : ℝ)
  (t_rainy : ℝ) :
  total_distance = 20 →
  total_time = 50 →
  sunny_speed = 40 →
  rainy_speed = 25 →
  total_time = t_sunny + t_rainy →
  (sunny_speed / 60) * t_sunny + (rainy_speed / 60) * t_rainy = total_distance →
  t_rainy = 30 :=
by
  intros
  sorry

end shelby_drive_rain_minutes_l110_110596


namespace negation_of_p_is_neg_p_l110_110126

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- Define the negation of p
def neg_p : Prop := ∀ m : ℝ, m > 0 → ¬ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- The theorem statement
theorem negation_of_p_is_neg_p : (¬ p) = neg_p := 
by
  sorry

end negation_of_p_is_neg_p_l110_110126


namespace k_at_27_l110_110734

noncomputable def h (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem k_at_27 (k : ℝ → ℝ)
    (hk_cubic : ∀ x, ∃ a b c, k x = a * x^3 + b * x^2 + c * x)
    (hk_at_0 : k 0 = 1)
    (hk_roots : ∀ a b c, (h a = 0) → (h b = 0) → (h c = 0) → 
                 ∃ (p q r: ℝ), k (p^3) = 0 ∧ k (q^3) = 0 ∧ k (r^3) = 0) :
    k 27 = -704 :=
sorry

end k_at_27_l110_110734


namespace moles_of_water_produced_l110_110095

theorem moles_of_water_produced (H₃PO₄ NaOH NaH₂PO₄ H₂O : ℝ) (h₁ : H₃PO₄ = 3) (h₂ : NaOH = 3) (h₃ : NaH₂PO₄ = 3) (h₄ : NaH₂PO₄ / H₂O = 1) : H₂O = 3 :=
by
  sorry

end moles_of_water_produced_l110_110095


namespace evaluate_expression_l110_110388

theorem evaluate_expression : (- (1 / 4))⁻¹ - (Real.pi - 3)^0 - |(-4 : ℝ)| + (-1)^(2021 : ℕ) = -10 := 
by
  sorry

end evaluate_expression_l110_110388


namespace tan_alpha_neg_four_over_three_l110_110179

theorem tan_alpha_neg_four_over_three (α : ℝ) (h_cos : Real.cos α = -3/5) (h_alpha_range : α ∈ Set.Ioo (-π) 0) : Real.tan α = -4/3 :=
  sorry

end tan_alpha_neg_four_over_three_l110_110179


namespace solve_star_eq_l110_110339

noncomputable def star (a b : ℤ) : ℤ := if a = b then 2 else sorry

axiom star_assoc : ∀ (a b c : ℤ), star a (star b c) = (star a b) - c
axiom star_self_eq_two : ∀ (a : ℤ), star a a = 2

theorem solve_star_eq : ∀ (x : ℤ), star 100 (star 5 x) = 20 → x = 20 :=
by
  intro x hx
  sorry

end solve_star_eq_l110_110339


namespace cylinder_section_volume_l110_110246

theorem cylinder_section_volume (a : ℝ) :
  let volume := (π * a^3 / 4)
  let section1_volume := volume * (1 / 3)
  let section2_volume := volume * (1 / 4)
  let enclosed_volume := (section1_volume - section2_volume) / 2
  enclosed_volume = π * a^3 / 24 := by
  sorry

end cylinder_section_volume_l110_110246


namespace evaluate_powers_of_i_mod_4_l110_110236

theorem evaluate_powers_of_i_mod_4 :
  (Complex.I ^ 48 + Complex.I ^ 96 + Complex.I ^ 144) = 3 := by
  sorry

end evaluate_powers_of_i_mod_4_l110_110236


namespace sum_of_fractions_l110_110598

theorem sum_of_fractions :
  (3 / 9) + (7 / 12) = (11 / 12) :=
by 
  sorry

end sum_of_fractions_l110_110598


namespace part1_part2_l110_110283

-- Definitions for the sets A and B
def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a - 1) * x + (a^2 - 5) = 0}

-- Proof problem (1): A ∩ B = {2} implies a = -5 or a = 1
theorem part1 (a : ℝ) (h : A ∩ B a = {2}) : a = -5 ∨ a = 1 := 
sorry

-- Proof problem (2): A ∪ B = A implies a > 3
theorem part2 (a : ℝ) (h : A ∪ B a = A) : 3 < a :=
sorry

end part1_part2_l110_110283


namespace team_total_points_l110_110018

theorem team_total_points :
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  team_total = 89 :=
by
  let connor_initial := 2
  let amy_initial := connor_initial + 4
  let jason_initial := amy_initial * 2
  let emily_initial := 3 * (connor_initial + amy_initial + jason_initial)
  let team_before_bonus := connor_initial + amy_initial + jason_initial
  let connor_total := connor_initial + 3
  let amy_total := amy_initial + 5
  let jason_total := jason_initial + 1
  let emily_total := emily_initial
  let team_total := connor_total + amy_total + jason_total + emily_total
  sorry

end team_total_points_l110_110018


namespace infinite_n_exists_l110_110822

-- Definitions from conditions
def is_natural_number (a : ℕ) : Prop := a > 3

-- Statement of the theorem
theorem infinite_n_exists (a : ℕ) (h : is_natural_number a) : ∃ᶠ n in at_top, a + n ∣ a^n + 1 :=
sorry

end infinite_n_exists_l110_110822


namespace sum_fourth_powers_const_l110_110730

-- Define the vertices of the square
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A (a : ℝ) : Point := {x := a, y := 0}
def B (a : ℝ) : Point := {x := 0, y := a}
def C (a : ℝ) : Point := {x := -a, y := 0}
def D (a : ℝ) : Point := {x := 0, y := -a}

-- Define distance squared between two points
def dist_sq (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

-- Circle centered at origin
def on_circle (P : Point) (r : ℝ) : Prop :=
  P.x ^ 2 + P.y ^ 2 = r ^ 2

-- The main theorem
theorem sum_fourth_powers_const (a r : ℝ) (P : Point) (h : on_circle P r) :
  let AP_sq := dist_sq P (A a)
  let BP_sq := dist_sq P (B a)
  let CP_sq := dist_sq P (C a)
  let DP_sq := dist_sq P (D a)
  (AP_sq ^ 2 + BP_sq ^ 2 + CP_sq ^ 2 + DP_sq ^ 2) = 4 * (r^4 + a^4 + 4 * a^2 * r^2) :=
by
  sorry

end sum_fourth_powers_const_l110_110730


namespace comp_inter_empty_l110_110805

section
variable {α : Type*} [DecidableEq α]
variable (I M N : Set α)
variable (a b c d e : α)
variable (hI : I = {a, b, c, d, e})
variable (hM : M = {a, c, d})
variable (hN : N = {b, d, e})

theorem comp_inter_empty : 
  (I \ M) ∩ (I \ N) = ∅ :=
by sorry
end

end comp_inter_empty_l110_110805


namespace max_capacity_per_car_l110_110162

-- Conditions
def num_cars : ℕ := 2
def num_vans : ℕ := 3
def people_per_car : ℕ := 5
def people_per_van : ℕ := 3
def max_people_per_van : ℕ := 8
def additional_people : ℕ := 17

-- Theorem to prove maximum capacity of each car is 6 people
theorem max_capacity_per_car (num_cars num_vans people_per_car people_per_van max_people_per_van additional_people : ℕ) : 
  (num_cars = 2 ∧ num_vans = 3 ∧ people_per_car = 5 ∧ people_per_van = 3 ∧ max_people_per_van = 8 ∧ additional_people = 17) →
  ∃ max_people_per_car, max_people_per_car = 6 :=
by
  sorry

end max_capacity_per_car_l110_110162


namespace people_after_five_years_l110_110398

noncomputable def population_in_year : ℕ → ℕ
| 0       => 20
| (k + 1) => 4 * population_in_year k - 18

theorem people_after_five_years : population_in_year 5 = 14382 := by
  sorry

end people_after_five_years_l110_110398


namespace amount_paid_l110_110676

-- Defining the conditions as constants
def cost_of_apple : ℝ := 0.75
def change_received : ℝ := 4.25

-- Stating the theorem that needs to be proved
theorem amount_paid (a : ℝ) : a = cost_of_apple + change_received :=
by
  sorry

end amount_paid_l110_110676


namespace vacation_cost_l110_110860

theorem vacation_cost (C : ℝ) (h : C / 3 - C / 4 = 60) : C = 720 := 
by sorry

end vacation_cost_l110_110860


namespace no_natural_numbers_condition_l110_110690

theorem no_natural_numbers_condition :
  ¬ ∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018,
    ∃ k : ℕ, (a i) ^ 2018 + a ((i + 1) % 2018) = 5 ^ k :=
by sorry

end no_natural_numbers_condition_l110_110690


namespace sheila_hourly_earnings_l110_110764

def sheila_hours_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 8
  else if day = "Tuesday" ∨ day = "Thursday" then 6
  else 0

def sheila_weekly_hours : Nat :=
  sheila_hours_per_day "Monday" +
  sheila_hours_per_day "Tuesday" +
  sheila_hours_per_day "Wednesday" +
  sheila_hours_per_day "Thursday" +
  sheila_hours_per_day "Friday"

def sheila_weekly_earnings : Nat := 468

theorem sheila_hourly_earnings :
  sheila_weekly_earnings / sheila_weekly_hours = 13 :=
by
  sorry

end sheila_hourly_earnings_l110_110764


namespace find_n_in_geometric_sequence_l110_110630

def geometric_sequence (an : ℕ → ℝ) (n : ℕ) : Prop :=
  ∃ q : ℝ, ∀ k : ℕ, an (k + 1) = an k * q

theorem find_n_in_geometric_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q)
  (h3 : ∀ q : ℝ, a n = a 1 * a 2 * a 3 * a 4 * a 5) :
  n = 11 :=
sorry

end find_n_in_geometric_sequence_l110_110630


namespace geo_sequence_ratio_l110_110713

theorem geo_sequence_ratio
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (q : ℝ)
  (hq1 : q = 1 → S_8 = 8 * a_n 0 ∧ S_4 = 4 * a_n 0 ∧ S_8 = 2 * S_4)
  (hq2 : q ≠ 1 → S_8 = 2 * S_4 → false)
  (hS : ∀ n, S_n n = a_n 0 * (1 - q^n) / (1 - q))
  (h_condition : S_8 = 2 * S_4) :
  a_n 2 / a_n 0 = 1 := sorry

end geo_sequence_ratio_l110_110713


namespace verify_extending_points_l110_110666

noncomputable def verify_P_and_Q (A B P Q : ℝ → ℝ → ℝ) : Prop := 
  let vector_relation_P := P = - (2/5) • A + (7/5) • B
  let vector_relation_Q := Q = - (1/4) • A + (5/4) • B 
  vector_relation_P ∧ vector_relation_Q

theorem verify_extending_points 
  (A B P Q : ℝ → ℝ → ℝ)
  (h1 : 7 • (P - A) = 2 • (B - P))
  (h2 : 5 • (Q - A) = 1 • (Q - B)) :
  verify_P_and_Q A B P Q := 
by
  sorry  

end verify_extending_points_l110_110666


namespace markers_leftover_l110_110782

theorem markers_leftover :
  let total_markers := 154
  let num_packages := 13
  total_markers % num_packages = 11 :=
by
  sorry

end markers_leftover_l110_110782


namespace total_fruits_l110_110072

theorem total_fruits (total_baskets apples_baskets oranges_baskets apples_per_basket oranges_per_basket pears_per_basket : ℕ)
  (h1 : total_baskets = 127)
  (h2 : apples_baskets = 79)
  (h3 : oranges_baskets = 30)
  (h4 : apples_per_basket = 75)
  (h5 : oranges_per_basket = 143)
  (h6 : pears_per_basket = 56)
  : 79 * 75 + 30 * 143 + (127 - (79 + 30)) * 56 = 11223 := by
  sorry

end total_fruits_l110_110072


namespace manager_wage_l110_110726

variable (M D C : ℝ)

def condition1 : Prop := D = M / 2
def condition2 : Prop := C = 1.25 * D
def condition3 : Prop := C = M - 3.1875

theorem manager_wage (h1 : condition1 M D) (h2 : condition2 D C) (h3 : condition3 M C) : M = 8.5 :=
by
  sorry

end manager_wage_l110_110726


namespace range_of_a_l110_110165

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), (1 ≤ x ∧ x ≤ 2) ∧ (2 ≤ y ∧ y ≤ 3) → (x * y ≤ a * x^2 + 2 * y^2)) →
  a ≥ -1 :=
by {
  sorry
}

end range_of_a_l110_110165


namespace option_A_correct_l110_110329

theorem option_A_correct (x y : ℝ) (hy : y ≠ 0) :
  (-2 * x^2 * y + y) / y = -2 * x^2 + 1 :=
by
  sorry

end option_A_correct_l110_110329


namespace cost_to_color_pattern_l110_110721

-- Define the basic properties of the squares
def square_side_length : ℕ := 4
def number_of_squares : ℕ := 4
def unit_cost (num_overlapping_squares : ℕ) : ℕ := num_overlapping_squares

-- Define the number of unit squares overlapping by different amounts
def unit_squares_overlapping_by_4 : ℕ := 1
def unit_squares_overlapping_by_3 : ℕ := 6
def unit_squares_overlapping_by_2 : ℕ := 12
def unit_squares_overlapping_by_1 : ℕ := 18

-- Calculate the total cost
def total_cost : ℕ :=
  unit_cost 4 * unit_squares_overlapping_by_4 +
  unit_cost 3 * unit_squares_overlapping_by_3 +
  unit_cost 2 * unit_squares_overlapping_by_2 +
  unit_cost 1 * unit_squares_overlapping_by_1

-- Statement to prove
theorem cost_to_color_pattern : total_cost = 64 := 
  sorry

end cost_to_color_pattern_l110_110721


namespace exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l110_110999

noncomputable def quadratic_sequence (n : ℕ) (a : ℕ → ℤ) :=
  ∀i : ℕ, 1 ≤ i ∧ i ≤ n → abs (a i - a (i - 1)) = i * i

theorem exists_quadratic_sequence_for_any_b_c (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ quadratic_sequence n a := by
  sorry

theorem smallest_n_for_quadratic_sequence_0_to_2021 :
  ∃ n : ℕ, 0 < n ∧ ∀ (a : ℕ → ℤ), a 0 = 0 → a n = 2021 → quadratic_sequence n a := by
  sorry

end exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l110_110999


namespace max_sum_x1_x2_x3_l110_110503

theorem max_sum_x1_x2_x3 : 
  ∀ (x1 x2 x3 x4 x5 x6 x7 : ℕ), 
    x1 < x2 → x2 < x3 → x3 < x4 → x4 < x5 → x5 < x6 → x6 < x7 →
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 159 →
    x1 + x2 + x3 = 61 :=
by
  intros x1 x2 x3 x4 x5 x6 x7 h1 h2 h3 h4 h5 h6 h_sum
  sorry

end max_sum_x1_x2_x3_l110_110503


namespace cost_of_bananas_is_two_l110_110934

variable (B : ℝ)

theorem cost_of_bananas_is_two (h : 1.20 * (3 + B) = 6) : B = 2 :=
by
  sorry

end cost_of_bananas_is_two_l110_110934


namespace average_of_six_numbers_l110_110083

theorem average_of_six_numbers (A : ℝ) (x y z w u v : ℝ)
  (h1 : (x + y + z + w + u + v) / 6 = A)
  (h2 : (x + y) / 2 = 1.1)
  (h3 : (z + w) / 2 = 1.4)
  (h4 : (u + v) / 2 = 5) :
  A = 2.5 :=
by
  sorry

end average_of_six_numbers_l110_110083


namespace find_other_tax_l110_110921

/-- Jill's expenditure breakdown and total tax conditions. -/
def JillExpenditure 
  (total : ℝ)
  (clothingPercent : ℝ)
  (foodPercent : ℝ)
  (otherPercent : ℝ)
  (clothingTaxPercent : ℝ)
  (foodTaxPercent : ℝ)
  (otherTaxPercent : ℝ)
  (totalTaxPercent : ℝ) :=
  (clothingPercent + foodPercent + otherPercent = 100) ∧
  (clothingTaxPercent = 4) ∧
  (foodTaxPercent = 0) ∧
  (totalTaxPercent = 5.2) ∧
  (total > 0)

/-- The goal is to find the tax percentage on other items which Jill paid, given the constraints. -/
theorem find_other_tax
  {total clothingAmt foodAmt otherAmt clothingTax foodTax otherTaxPercent totalTax : ℝ}
  (h_exp : JillExpenditure total 50 10 40 clothingTax foodTax otherTaxPercent totalTax) :
  otherTaxPercent = 8 :=
by
  sorry

end find_other_tax_l110_110921


namespace placing_pencils_l110_110142

theorem placing_pencils (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) 
    (h1 : total_pencils = 6) (h2 : num_rows = 2) : pencils_per_row = 3 :=
by
  sorry

end placing_pencils_l110_110142


namespace isabella_hair_length_l110_110248

theorem isabella_hair_length (final_length growth_length initial_length : ℕ) 
  (h1 : final_length = 24) 
  (h2 : growth_length = 6) 
  (h3 : final_length = initial_length + growth_length) : 
  initial_length = 18 :=
by
  sorry

end isabella_hair_length_l110_110248


namespace greatest_length_measures_exactly_l110_110450

theorem greatest_length_measures_exactly 
    (a b c : ℕ) 
    (ha : a = 700)
    (hb : b = 385)
    (hc : c = 1295) : 
    Nat.gcd (Nat.gcd a b) c = 35 := 
by
  sorry

end greatest_length_measures_exactly_l110_110450


namespace compare_abc_l110_110035

open Real

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Assuming the conditions
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom derivative : ∀ x : ℝ, f' x = deriv f x
axiom monotonicity_condition : ∀ x > 0, x * f' x < f x

-- Definitions of a, b, and c
noncomputable def a := 2 * f (1 / 2)
noncomputable def b := - (1 / 2) * f (-2)
noncomputable def c := - (1 / log 2) * f (log (1 / 2))

theorem compare_abc : a > c ∧ c > b := sorry

end compare_abc_l110_110035


namespace negation_of_universal_l110_110738

theorem negation_of_universal :
  ¬ (∀ x : ℝ, 2 * x ^ 2 + x - 1 ≤ 0) ↔ ∃ x : ℝ, 2 * x ^ 2 + x - 1 > 0 := 
by 
  sorry

end negation_of_universal_l110_110738


namespace speed_of_boat_in_still_water_l110_110859

-- Define the given conditions
def speed_of_stream : ℝ := 4  -- Speed of the stream in km/hr
def distance_downstream : ℝ := 60  -- Distance traveled downstream in km
def time_downstream : ℝ := 3  -- Time taken to travel downstream in hours

-- The statement we need to prove
theorem speed_of_boat_in_still_water (V_b : ℝ) (V_d : ℝ) :
  V_d = distance_downstream / time_downstream →
  V_d = V_b + speed_of_stream →
  V_b = 16 :=
by
  intros Vd_eq D_eq
  sorry

end speed_of_boat_in_still_water_l110_110859


namespace stephen_total_distance_l110_110879

theorem stephen_total_distance :
  let mountain_height := 40000
  let ascent_fraction := 3 / 4
  let descent_fraction := 2 / 3
  let extra_distance_fraction := 0.10
  let normal_trips := 8
  let harsh_trips := 2
  let ascent_distance := ascent_fraction * mountain_height
  let descent_distance := descent_fraction * ascent_distance
  let normal_trip_distance := ascent_distance + descent_distance
  let harsh_trip_extra_distance := extra_distance_fraction * ascent_distance
  let harsh_trip_distance := ascent_distance + harsh_trip_extra_distance + descent_distance
  let total_normal_distance := normal_trip_distance * normal_trips
  let total_harsh_distance := harsh_trip_distance * harsh_trips
  let total_distance := total_normal_distance + total_harsh_distance
  total_distance = 506000 :=
by
  sorry

end stephen_total_distance_l110_110879


namespace value_of_expression_l110_110274

theorem value_of_expression (n : ℕ) (a : ℝ) (h1 : 6 * 11 * n ≠ 0) (h2 : a ^ (2 * n) = 5) : 2 * a ^ (6 * n) - 4 = 246 :=
by
  sorry

end value_of_expression_l110_110274


namespace six_rational_right_triangles_same_perimeter_l110_110163

theorem six_rational_right_triangles_same_perimeter :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ a₄ b₄ c₄ a₅ b₅ c₅ a₆ b₆ c₆ : ℕ),
    a₁^2 + b₁^2 = c₁^2 ∧ a₂^2 + b₂^2 = c₂^2 ∧ a₃^2 + b₃^2 = c₃^2 ∧
    a₄^2 + b₄^2 = c₄^2 ∧ a₅^2 + b₅^2 = c₅^2 ∧ a₆^2 + b₆^2 = c₆^2 ∧
    a₁ + b₁ + c₁ = 720 ∧ a₂ + b₂ + c₂ = 720 ∧ a₃ + b₃ + c₃ = 720 ∧
    a₄ + b₄ + c₄ = 720 ∧ a₅ + b₅ + c₅ = 720 ∧ a₆ + b₆ + c₆ = 720 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧ (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₄, b₄, c₄) ∧ (a₁, b₁, c₁) ≠ (a₅, b₅, c₅) ∧
    (a₁, b₁, c₁) ≠ (a₆, b₆, c₆) ∧ (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₂, b₂, c₂) ≠ (a₄, b₄, c₄) ∧ (a₂, b₂, c₂) ≠ (a₅, b₅, c₅) ∧
    (a₂, b₂, c₂) ≠ (a₆, b₆, c₆) ∧ (a₃, b₃, c₃) ≠ (a₄, b₄, c₄) ∧
    (a₃, b₃, c₃) ≠ (a₅, b₅, c₅) ∧ (a₃, b₃, c₃) ≠ (a₆, b₆, c₆) ∧
    (a₄, b₄, c₄) ≠ (a₅, b₅, c₅) ∧ (a₄, b₄, c₄) ≠ (a₆, b₆, c₆) ∧
    (a₅, b₅, c₅) ≠ (a₆, b₆, c₆) :=
sorry

end six_rational_right_triangles_same_perimeter_l110_110163


namespace max_cells_primitive_dinosaur_l110_110249

section Dinosaur

universe u

-- Define a dinosaur as a structure with at least 2007 cells
structure Dinosaur (α : Type u) :=
(cells : ℕ) (connected : α → α → Prop)
(h_cells : cells ≥ 2007)
(h_connected : ∀ (x y : α), connected x y → connected y x)

-- Define a primitive dinosaur where the cells cannot be partitioned into two or more dinosaurs
structure PrimitiveDinosaur (α : Type u) extends Dinosaur α :=
(h_partition : ∀ (x : α), ¬∃ (d1 d2 : Dinosaur α), (d1.cells + d2.cells = cells) ∧ 
  (d1 ≠ d2 ∧ d1.cells ≥ 2007 ∧ d2.cells ≥ 2007))

-- Prove that the maximum number of cells in a Primitive Dinosaur is 8025
theorem max_cells_primitive_dinosaur : ∀ (α : Type u), ∃ (d : PrimitiveDinosaur α), d.cells = 8025 :=
sorry

end Dinosaur

end max_cells_primitive_dinosaur_l110_110249


namespace inequality_proof_l110_110229

variable {x y z : ℝ}

theorem inequality_proof (h : True) :
  ( (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) +
    (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) +
    (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) ) ≥ 6 := by
  sorry

end inequality_proof_l110_110229


namespace intersection_M_N_l110_110461

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | ∃ k : ℕ, x = 2 * k}

theorem intersection_M_N :
  M ∩ N = {2, 4, 8} :=
by sorry

end intersection_M_N_l110_110461


namespace unit_price_of_each_chair_is_42_l110_110440

-- Definitions from conditions
def total_cost_desks (unit_price_desk : ℕ) (number_desks : ℕ) : ℕ := unit_price_desk * number_desks
def remaining_cost_chairs (total_cost : ℕ) (cost_desks : ℕ) : ℕ := total_cost - cost_desks
def unit_price_chairs (remaining_cost : ℕ) (number_chairs : ℕ) : ℕ := remaining_cost / number_chairs

-- Given conditions
def unit_price_desk := 180
def number_desks := 5
def total_cost := 1236
def number_chairs := 8

-- The question: determining the unit price of each chair
theorem unit_price_of_each_chair_is_42 : 
  unit_price_chairs (remaining_cost_chairs total_cost (total_cost_desks unit_price_desk number_desks)) number_chairs = 42 := sorry

end unit_price_of_each_chair_is_42_l110_110440


namespace axis_of_symmetry_parabola_l110_110376

theorem axis_of_symmetry_parabola : ∀ (x y : ℝ), y = 2 * x^2 → x = 0 :=
by
  sorry

end axis_of_symmetry_parabola_l110_110376


namespace evaluate_product_l110_110026

-- Define the given numerical values
def a : ℝ := 2.5
def b : ℝ := 50.5
def c : ℝ := 0.15

-- State the theorem we want to prove
theorem evaluate_product : a * (b + c) = 126.625 := by
  sorry

end evaluate_product_l110_110026


namespace percent_increase_fifth_triangle_l110_110945

noncomputable def initial_side_length : ℝ := 3
noncomputable def growth_factor : ℝ := 1.2
noncomputable def num_triangles : ℕ := 5

noncomputable def side_length (n : ℕ) : ℝ :=
  initial_side_length * growth_factor ^ (n - 1)

noncomputable def perimeter_length (n : ℕ) : ℝ :=
  3 * side_length n

noncomputable def percent_increase (n : ℕ) : ℝ :=
  ((perimeter_length n / perimeter_length 1) - 1) * 100

theorem percent_increase_fifth_triangle :
  percent_increase 5 = 107.4 :=
by
  sorry

end percent_increase_fifth_triangle_l110_110945


namespace part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l110_110887

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (2 * (Real.cos (x/2))^2 + Real.sin x) + b

theorem part1_monotonically_increasing_interval (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4 + 2 * k * Real.pi) (Real.pi / 4 + 2 * k * Real.pi) ->
    f x <= f (x + Real.pi) :=
sorry

theorem part1_symmetry_axis (b : ℝ) (k : ℤ) :
  let f := f 1 b
  ∀ x, f x = f (2 * (Real.pi / 4 + k * Real.pi) - x) :=
sorry

theorem part2_find_a_b (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi)
  (h2 : ∃ (a b : ℝ), ∀ x, x ∈ Set.Icc 0 Real.pi → (a > 0 ∧ 3 ≤ f a b x ∧ f a b x ≤ 4)) :
  (1 - Real.sqrt 2 < a ∧ a < 1 + Real.sqrt 2) ∧ b = 3 :=
sorry

end part1_monotonically_increasing_interval_part1_symmetry_axis_part2_find_a_b_l110_110887


namespace gasoline_added_l110_110479

variable (tank_capacity : ℝ := 42)
variable (initial_fill_fraction : ℝ := 3/4)
variable (final_fill_fraction : ℝ := 9/10)

theorem gasoline_added :
  let initial_amount := tank_capacity * initial_fill_fraction
  let final_amount := tank_capacity * final_fill_fraction
  final_amount - initial_amount = 6.3 :=
by
  sorry

end gasoline_added_l110_110479


namespace intersection_M_N_l110_110780

def M (x : ℝ) : Prop := x^2 + 2*x - 15 < 0
def N (x : ℝ) : Prop := x^2 + 6*x - 7 ≥ 0

theorem intersection_M_N : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | 1 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_M_N_l110_110780


namespace monochromatic_triangle_in_K17_l110_110696

theorem monochromatic_triangle_in_K17 :
  ∀ (V : Type) (E : V → V → ℕ), (∀ v1 v2, 0 ≤ E v1 v2 ∧ E v1 v2 < 3) →
    (∃ (v1 v2 v3 : V), v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧ (E v1 v2 = E v2 v3 ∧ E v2 v3 = E v1 v3)) :=
by
  intro V E Hcl
  sorry

end monochromatic_triangle_in_K17_l110_110696


namespace magnitude_of_error_l110_110522

theorem magnitude_of_error (x : ℝ) (hx : 0 < x) :
  abs ((4 * x) - (x / 4)) / (4 * x) * 100 = 94 := 
sorry

end magnitude_of_error_l110_110522


namespace area_of_circle_l110_110109

theorem area_of_circle (C : ℝ) (hC : C = 30 * Real.pi) : ∃ k : ℝ, (Real.pi * (C / (2 * Real.pi))^2 = k * Real.pi) ∧ k = 225 :=
by
  sorry

end area_of_circle_l110_110109


namespace largest_mersenne_prime_lt_1000_l110_110113

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_mersenne_prime (n : ℕ) : Prop :=
  is_prime n ∧ ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_lt_1000 : ∃ (n : ℕ), is_mersenne_prime n ∧ n < 1000 ∧ ∀ (m : ℕ), is_mersenne_prime m ∧ m < 1000 → m ≤ n :=
by
  -- Proof goes here
  sorry

end largest_mersenne_prime_lt_1000_l110_110113


namespace expansion_eq_l110_110219

variable (x y : ℝ) -- x and y are real numbers
def a := 5
def b := 3
def c := 15

theorem expansion_eq : (x + a) * (b * y + c) = 3 * x * y + 15 * x + 15 * y + 75 := by 
  sorry

end expansion_eq_l110_110219


namespace number_of_street_trees_l110_110226

-- Definitions from conditions
def road_length : ℕ := 1500
def interval_distance : ℕ := 25

-- The statement to prove
theorem number_of_street_trees : (road_length / interval_distance) + 1 = 61 := 
by
  unfold road_length
  unfold interval_distance
  sorry

end number_of_street_trees_l110_110226


namespace coordinates_of_A_l110_110796

-- Definition of the distance function for any point (x, y)
def distance_to_x_axis (x y : ℝ) : ℝ := abs y
def distance_to_y_axis (x y : ℝ) : ℝ := abs x

-- Point A's coordinates
def point_is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- The main theorem to prove
theorem coordinates_of_A :
  ∃ (x y : ℝ), 
  point_is_in_fourth_quadrant x y ∧ 
  distance_to_x_axis x y = 3 ∧ 
  distance_to_y_axis x y = 6 ∧ 
  (x, y) = (6, -3) :=
by 
  sorry

end coordinates_of_A_l110_110796


namespace focus_of_parabola_proof_l110_110707

noncomputable def focus_of_parabola (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (1 / (4 * a), 0)

theorem focus_of_parabola_proof (a : ℝ) (h : a ≠ 0) :
  focus_of_parabola a h = (1 / (4 * a), 0) :=
sorry

end focus_of_parabola_proof_l110_110707


namespace range_of_a_l110_110473

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end range_of_a_l110_110473


namespace relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l110_110530

variable (x y : ℝ)
variable (h1 : 2 * (x + y) = 18)
variable (h2 : x * y = 18)
variable (h3 : x > 0) (h4 : y > 0) (h5 : x > y)
variable (h6 : x * y = 21)

theorem relationship_and_range : (y = 9 - x ∧ 0 < x ∧ x < 9) :=
by sorry

theorem dimensions_when_area_18 :
  (x = 6 ∧ y = 3) ∨ (x = 3 ∧ y = 6) :=
by sorry

theorem impossibility_of_area_21 :
  ¬(∃ x y, x * y = 21 ∧ 2 * (x + y) = 18 ∧ x > y) :=
by sorry

end relationship_and_range_dimensions_when_area_18_impossibility_of_area_21_l110_110530


namespace towels_folded_in_one_hour_l110_110663

theorem towels_folded_in_one_hour :
  let jane_rate := 12 * 5 -- Jane's rate in towels/hour
  let kyla_rate := 6 * 9  -- Kyla's rate in towels/hour
  let anthony_rate := 3 * 14 -- Anthony's rate in towels/hour
  let david_rate := 4 * 6 -- David's rate in towels/hour
  jane_rate + kyla_rate + anthony_rate + david_rate = 180 := 
by
  let jane_rate := 12 * 5
  let kyla_rate := 6 * 9
  let anthony_rate := 3 * 14
  let david_rate := 4 * 6
  show jane_rate + kyla_rate + anthony_rate + david_rate = 180
  sorry

end towels_folded_in_one_hour_l110_110663


namespace tank_holds_gallons_l110_110305

noncomputable def tank_initial_fraction := (7 : ℚ) / 8
noncomputable def tank_partial_fraction := (2 : ℚ) / 3
def gallons_used := 15

theorem tank_holds_gallons
  (x : ℚ) -- number of gallons the tank holds when full
  (h_initial : tank_initial_fraction * x - gallons_used = tank_partial_fraction * x) :
  x = 72 := 
sorry

end tank_holds_gallons_l110_110305


namespace quadratic_has_real_root_l110_110711

theorem quadratic_has_real_root {b : ℝ} :
  ∃ x : ℝ, x^2 + b*x + 25 = 0 ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end quadratic_has_real_root_l110_110711


namespace smallest_consecutive_integers_product_l110_110524

theorem smallest_consecutive_integers_product (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 5040) : 
  n = 7 :=
sorry

end smallest_consecutive_integers_product_l110_110524


namespace constant_sum_powers_l110_110065

theorem constant_sum_powers (n : ℕ) (x y z : ℝ) (h_sum : x + y + z = 0) (h_prod : x * y * z = 1) :
  (∀ x y z : ℝ, x + y + z = 0 → x * y * z = 1 → x^n + y^n + z^n = x^n + y^n + z^n ↔ (n = 1 ∨ n = 3)) :=
by
  sorry

end constant_sum_powers_l110_110065


namespace problem1_problem2_l110_110732

theorem problem1 : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

theorem problem2 : Real.sqrt (4 / 3) / Real.sqrt (7 / 3) * Real.sqrt (7 / 5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end problem1_problem2_l110_110732


namespace max_profit_l110_110683

noncomputable def annual_profit (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 5 then 
    -0.5 * x^2 + 3.5 * x - 0.5 
  else if x > 5 then 
    17 - 2.5 * x 
  else 
    0

theorem max_profit :
  ∀ x : ℝ, (annual_profit 3.5 = 5.625) :=
by
  -- Proof omitted
  sorry

end max_profit_l110_110683


namespace quadratic_inequality_solution_l110_110757

theorem quadratic_inequality_solution :
  {x : ℝ | -x^2 + x + 2 > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end quadratic_inequality_solution_l110_110757


namespace find_divisor_l110_110115

theorem find_divisor (n : ℕ) (d : ℕ) (h1 : n = 105829) (h2 : d = 10) (h3 : ∃ k, n - d = k * d) : d = 3 :=
by
  sorry

end find_divisor_l110_110115


namespace three_digit_integer_divisible_by_5_l110_110718

theorem three_digit_integer_divisible_by_5 (M : ℕ) (h1 : 100 ≤ M ∧ M < 1000) (h2 : M % 10 = 5) : M % 5 = 0 := 
sorry

end three_digit_integer_divisible_by_5_l110_110718


namespace exists_such_h_l110_110279

noncomputable def exists_h (h : ℝ) : Prop :=
  ∀ (n : ℕ), ¬ (⌊h * 1969^n⌋ ∣ ⌊h * 1969^(n-1)⌋)

theorem exists_such_h : ∃ h : ℝ, exists_h h := 
  -- Let's construct the h as mentioned in the provided proof
  ⟨1969^2 / 1968, 
    by sorry⟩

end exists_such_h_l110_110279


namespace intersecting_lines_l110_110471

theorem intersecting_lines (a b : ℝ) (h1 : 3 = (1 / 3) * 6 + a) (h2 : 6 = (1 / 3) * 3 + b) : a + b = 6 :=
sorry

end intersecting_lines_l110_110471


namespace complex_number_solution_l110_110315

open Complex

theorem complex_number_solution (z : ℂ) (h : (2 * z - I) * (2 - I) = 5) : 
  z = 1 + I :=
sorry

end complex_number_solution_l110_110315


namespace third_term_binomial_coefficient_l110_110989

theorem third_term_binomial_coefficient :
  (∃ m : ℕ, m = 4 ∧ ∃ k : ℕ, k = 2 ∧ Nat.choose m k = 6) :=
by
  sorry

end third_term_binomial_coefficient_l110_110989


namespace num_perfect_square_factors_of_360_l110_110668

theorem num_perfect_square_factors_of_360 : 
  ∃ (n : ℕ), n = 4 ∧ ∀ d : ℕ, d ∣ 360 → (∀ p e, p^e ∣ d → (p = 2 ∨ p = 3 ∨ p = 5) ∧ e % 2 = 0) :=
by
  sorry

end num_perfect_square_factors_of_360_l110_110668


namespace intersection_of_A_and_B_l110_110846

-- Define the sets A and B
def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

-- Prove that the intersection of A and B is {8, 10}
theorem intersection_of_A_and_B : A ∩ B = {8, 10} :=
by
  -- Proof will be filled here
  sorry

end intersection_of_A_and_B_l110_110846


namespace probability_of_D_l110_110627

theorem probability_of_D (P : Type) (A B C D : P) 
  (pA pB pC pD : ℚ) 
  (hA : pA = 1/4) 
  (hB : pB = 1/3) 
  (hC : pC = 1/6) 
  (hSum : pA + pB + pC + pD = 1) :
  pD = 1/4 :=
by 
  sorry

end probability_of_D_l110_110627


namespace num_birds_is_six_l110_110313

-- Define the number of nests
def N : ℕ := 3

-- Define the difference between the number of birds and nests
def diff : ℕ := 3

-- Prove that the number of birds is 6
theorem num_birds_is_six (B : ℕ) (h1 : N = 3) (h2 : B - N = diff) : B = 6 := by
  -- Placeholder for the proof
  sorry

end num_birds_is_six_l110_110313


namespace angle_D_calculation_l110_110104

theorem angle_D_calculation (A B E C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50)
  (h4 : E = 60)
  (h5 : A + B + E = 180)
  (h6 : B + C + D = 180) :
  D = 55 :=
by
  sorry

end angle_D_calculation_l110_110104


namespace problem_1_problem_2_problem_3_problem_4_l110_110231

theorem problem_1 : 3 * Real.sqrt 20 - Real.sqrt 45 - Real.sqrt (1/5) = 14 * Real.sqrt 5 / 5 :=
by sorry

theorem problem_2 : (Real.sqrt 6 * Real.sqrt 3) / Real.sqrt 2 - 1 = 2 :=
by sorry

theorem problem_3 : Real.sqrt 16 + 327 - 2 * Real.sqrt (1/4) = 330 :=
by sorry

theorem problem_4 : (Real.sqrt 3 - Real.sqrt 5) * (Real.sqrt 5 + Real.sqrt 3) - (Real.sqrt 5 - Real.sqrt 3) ^ 2 = 2 * Real.sqrt 15 - 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l110_110231


namespace total_chocolate_bars_in_colossal_box_l110_110221

theorem total_chocolate_bars_in_colossal_box :
  let colossal_boxes := 350
  let sizable_boxes := 49
  let small_boxes := 75
  colossal_boxes * sizable_boxes * small_boxes = 1287750 :=
by
  sorry

end total_chocolate_bars_in_colossal_box_l110_110221


namespace find_p_l110_110243

def delta (a b : ℝ) : ℝ := a * b + a + b

theorem find_p (p : ℝ) (h : delta p 3 = 39) : p = 9 :=
by
  sorry

end find_p_l110_110243


namespace value_of_expression_l110_110091

theorem value_of_expression :
  4 * 5 + 5 * 4 = 40 :=
sorry

end value_of_expression_l110_110091


namespace ratio_of_radii_l110_110678

theorem ratio_of_radii 
  (a b : ℝ)
  (h1 : ∀ (a b : ℝ), π * b^2 - π * a^2 = 4 * π * a^2) : 
  a / b = 1 / Real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l110_110678


namespace exists_l_l110_110523

theorem exists_l (n : ℕ) (h : n ≥ 4011^2) : ∃ l : ℤ, n < l^2 ∧ l^2 < (1 + 1/2005) * n := 
sorry

end exists_l_l110_110523


namespace rajan_income_l110_110141

theorem rajan_income : 
  ∀ (x y : ℕ), 
  7 * x - 6 * y = 1000 → 
  6 * x - 5 * y = 1000 → 
  7 * x = 7000 := 
by 
  intros x y h1 h2
  sorry

end rajan_income_l110_110141


namespace avg_class_l110_110779

-- Problem definitions
def total_students : ℕ := 40
def num_students_95 : ℕ := 8
def num_students_0 : ℕ := 5
def num_students_70 : ℕ := 10
def avg_remaining_students : ℝ := 50

-- Assuming we have these marks
def marks_95 : ℝ := 95
def marks_0 : ℝ := 0
def marks_70 : ℝ := 70

-- We need to prove that the total average is 57.75 given the above conditions
theorem avg_class (h1 : total_students = 40)
                  (h2 : num_students_95 = 8)
                  (h3 : num_students_0 = 5)
                  (h4 : num_students_70 = 10)
                  (h5 : avg_remaining_students = 50)
                  (h6 : marks_95 = 95)
                  (h7 : marks_0 = 0)
                  (h8 : marks_70 = 70) :
                  (8 * 95 + 5 * 0 + 10 * 70 + 50 * (40 - (8 + 5 + 10))) / 40 = 57.75 :=
by sorry

end avg_class_l110_110779


namespace odd_function_f_value_l110_110240

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^3 + x + 1 else x^3 + x - 1

theorem odd_function_f_value : 
  f 2 = 9 := by
  sorry

end odd_function_f_value_l110_110240


namespace inequality_of_thirds_of_ordered_triples_l110_110009

variable (a1 a2 a3 b1 b2 b3 : ℝ)

theorem inequality_of_thirds_of_ordered_triples 
  (h1 : a1 ≤ a2) 
  (h2 : a2 ≤ a3) 
  (h3 : b1 ≤ b2)
  (h4 : b2 ≤ b3)
  (h5 : a1 + a2 + a3 = b1 + b2 + b3)
  (h6 : a1 * a2 + a2 * a3 + a1 * a3 = b1 * b2 + b2 * b3 + b1 * b3)
  (h7 : a1 ≤ b1) : 
  a3 ≤ b3 := 
by 
  sorry

end inequality_of_thirds_of_ordered_triples_l110_110009


namespace least_five_digit_congruent_to_six_mod_seventeen_l110_110268

theorem least_five_digit_congruent_to_six_mod_seventeen : ∃ x : ℕ, x ≥ 10000 ∧ x < 100000 ∧ x % 17 = 6 ∧ ∀ y : ℕ, y ≥ 10000 ∧ y < 100000 ∧ y % 17 = 6 → x ≤ y :=
  by
    sorry

end least_five_digit_congruent_to_six_mod_seventeen_l110_110268


namespace evaluate_g_at_3_l110_110306

def g (x : ℝ) := 3 * x ^ 4 - 5 * x ^ 3 + 4 * x ^ 2 - 7 * x + 2

theorem evaluate_g_at_3 : g 3 = 125 :=
by
  -- Proof omitted for this exercise.
  sorry

end evaluate_g_at_3_l110_110306


namespace concert_tickets_l110_110670

theorem concert_tickets (A C : ℕ) (h1 : C = 3 * A) (h2 : 7 * A + 3 * C = 6000) : A + C = 1500 :=
by {
  -- Proof omitted
  sorry
}

end concert_tickets_l110_110670


namespace arithmetic_sequence_a5_l110_110259

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h : a 1 + a 9 = 10) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_l110_110259


namespace cassidy_number_of_posters_l110_110442

/-- Cassidy's current number of posters -/
def current_posters (C : ℕ) : Prop := 
  C + 6 = 2 * 14

theorem cassidy_number_of_posters : ∃ C : ℕ, current_posters C := 
  Exists.intro 22 sorry

end cassidy_number_of_posters_l110_110442


namespace no_real_roots_of_quadratic_l110_110391

theorem no_real_roots_of_quadratic (k : ℝ) (h : 12 - 3 * k < 0) : ∀ (x : ℝ), ¬ (x^2 + 4 * x + k = 0) := by
  sorry

end no_real_roots_of_quadratic_l110_110391


namespace physics_marks_l110_110770

theorem physics_marks (P C M : ℕ) 
  (h1 : (P + C + M) = 255)
  (h2 : (P + M) = 180)
  (h3 : (P + C) = 140) : 
  P = 65 :=
by
  sorry

end physics_marks_l110_110770


namespace sum_and_product_of_white_are_white_l110_110375

-- Definitions based on the conditions
def is_colored_black_or_white (n : ℕ) : Prop :=
  true -- This is a simplified assumption since this property is always true.

def is_black (n : ℕ) : Prop := (n % 2 = 0)
def is_white (n : ℕ) : Prop := (n % 2 = 1)

-- Conditions given in the problem
axiom sum_diff_colors_is_black (a b : ℕ) (ha : is_black a) (hb : is_white b) : is_black (a + b)
axiom infinitely_many_whites : ∀ n, ∃ m ≥ n, is_white m

-- Statement to prove that the sum and product of two white numbers are white
theorem sum_and_product_of_white_are_white (a b : ℕ) (ha : is_white a) (hb : is_white b) : 
  is_white (a + b) ∧ is_white (a * b) :=
sorry

end sum_and_product_of_white_are_white_l110_110375


namespace angle_BDC_eq_88_l110_110563

-- Define the problem scenario
variable (A B C : ℝ)
variable (α : ℝ)
variable (B1 B2 B3 C1 C2 C3 : ℝ)

-- Conditions provided
axiom angle_A_eq_42 : α = 42
axiom trisectors_ABC : B = B1 + B2 + B3 ∧ C = C1 + C2 + C3
axiom trisectors_eq : B1 = B2 ∧ B2 = B3 ∧ C1 = C2 ∧ C2 = C3
axiom angle_sum_ABC : α + B + C = 180

-- Proving the measure of ∠BDC
theorem angle_BDC_eq_88 :
  α + (B/3) + (C/3) = 88 :=
by
  sorry

end angle_BDC_eq_88_l110_110563


namespace complex_square_sum_eq_five_l110_110047

theorem complex_square_sum_eq_five (a b : ℝ) (h : (a + b * I) ^ 2 = 3 + 4 * I) : a^2 + b^2 = 5 := 
by sorry

end complex_square_sum_eq_five_l110_110047


namespace volume_ratio_of_sphere_surface_area_l110_110541

theorem volume_ratio_of_sphere_surface_area 
  {V1 V2 V3 : ℝ} 
  (h : V1/V3 = 1/27 ∧ V2/V3 = 8/27) 
  : V1 + V2 = (1/3) * V3 := 
sorry

end volume_ratio_of_sphere_surface_area_l110_110541


namespace intersection_of_sets_is_closed_interval_l110_110401

noncomputable def A := {x : ℝ | x ≤ 0 ∨ x ≥ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_sets_is_closed_interval :
  A ∩ B = {x : ℝ | x ≤ 0} :=
sorry

end intersection_of_sets_is_closed_interval_l110_110401


namespace value_of_expression_l110_110460

theorem value_of_expression (a b : ℝ) (h1 : a ≠ b)
  (h2 : a^2 + 2 * a - 2022 = 0)
  (h3 : b^2 + 2 * b - 2022 = 0) :
  a^2 + 4 * a + 2 * b = 2018 :=
by
  sorry

end value_of_expression_l110_110460


namespace exists_point_at_distance_l110_110342

def Line : Type := sorry
def Point : Type := sorry
def distance (P Q : Point) : ℝ := sorry

variables (L : Line) (d : ℝ) (P : Point)

def is_at_distance (Q : Point) (L : Line) (d : ℝ) := ∃ Q, distance Q L = d

theorem exists_point_at_distance :
  ∃ Q : Point, is_at_distance Q L d :=
sorry

end exists_point_at_distance_l110_110342


namespace proof_correct_option_C_l110_110400

def line := Type
def plane := Type
def perp (m : line) (α : plane) : Prop := sorry
def parallel (n : line) (α : plane) : Prop := sorry
def perpnal (m n: line): Prop := sorry 

variables (m n : line) (α β γ : plane)

theorem proof_correct_option_C : perp m α → parallel n α → perpnal m n := sorry

end proof_correct_option_C_l110_110400


namespace function_equality_l110_110979

theorem function_equality (f : ℝ → ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (f ( (x + 1) / x ) = (x^2 + 1) / x^2 + 1 / x) ↔ (f x = x^2 - x + 1) :=
by
  sorry

end function_equality_l110_110979


namespace carlos_biked_more_than_daniel_l110_110234

-- Definitions modeled from conditions
def distance_carlos : ℕ := 108
def distance_daniel : ℕ := 90
def time_hours : ℕ := 6

-- Lean statement to prove the difference in distance
theorem carlos_biked_more_than_daniel : distance_carlos - distance_daniel = 18 := 
  by 
    sorry

end carlos_biked_more_than_daniel_l110_110234


namespace three_four_five_six_solution_l110_110320

-- State that the equation 3^x + 4^x = 5^x is true when x=2
axiom three_four_five_solution : 3^2 + 4^2 = 5^2

-- We need to prove the following theorem
theorem three_four_five_six_solution : 3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end three_four_five_six_solution_l110_110320


namespace multiples_of_eleven_ending_in_seven_l110_110101

theorem multiples_of_eleven_ending_in_seven (n : ℕ) : 
  (∀ k : ℕ, n > 0 ∧ n < 2000 ∧ (∃ m : ℕ, n = 11 * m) ∧ n % 10 = 7) → ∃ c : ℕ, c = 18 := 
by
  sorry

end multiples_of_eleven_ending_in_seven_l110_110101


namespace identify_fraction_l110_110138

variable {a b : ℚ}

def is_fraction (x : ℚ) (y : ℚ) := ∃ (n : ℚ), x = n / y

theorem identify_fraction :
  is_fraction 2 a ∧ ¬ is_fraction (2 * a) 3 ∧ ¬ is_fraction (-b) 2 ∧ ¬ is_fraction (3 * a + 1) 2 :=
by
  sorry

end identify_fraction_l110_110138


namespace solve_fractions_in_integers_l110_110786

theorem solve_fractions_in_integers :
  ∀ (a b c : ℤ), (1 / a + 1 / b + 1 / c = 1) ↔
  (a = 3 ∧ b = 3 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 2 ∧ b = 4 ∧ c = 4) ∨
  (a = 1 ∧ ∃ t : ℤ, b = t ∧ c = -t) :=
by {
  sorry
}

end solve_fractions_in_integers_l110_110786


namespace sum_of_edges_of_rectangular_solid_l110_110910

theorem sum_of_edges_of_rectangular_solid 
  (a r : ℝ) 
  (volume_eq : (a / r) * a * (a * r) = 343) 
  (surface_area_eq : 2 * ((a^2 / r) + (a^2 * r) + a^2) = 294) 
  (gp : a / r > 0 ∧ a > 0 ∧ a * r > 0) :
  4 * ((a / r) + a + (a * r)) = 84 :=
by
  sorry

end sum_of_edges_of_rectangular_solid_l110_110910


namespace average_weight_of_children_l110_110581

theorem average_weight_of_children :
  let ages := [3, 4, 5, 6, 7]
  let regression_equation (x : ℕ) := 3 * x + 5
  let average l := (l.foldr (· + ·) 0) / l.length
  average (List.map regression_equation ages) = 20 :=
by
  sorry

end average_weight_of_children_l110_110581


namespace line_tangent_parabola_unique_d_l110_110553

theorem line_tangent_parabola_unique_d :
  (∃ d : ℝ, ∀ x : ℝ, y = 3 * x + d -> y^2 = 12 * x) -> d = 1 := by
sorry

end line_tangent_parabola_unique_d_l110_110553


namespace average_gas_mileage_round_trip_l110_110592

/-
A student drives 150 miles to university in a sedan that averages 25 miles per gallon.
The same student drives 150 miles back home in a minivan that averages 15 miles per gallon.
Calculate the average gas mileage for the entire round trip.
-/
theorem average_gas_mileage_round_trip (d1 d2 m1 m2 : ℝ) (h1 : d1 = 150) (h2 : m1 = 25) 
  (h3 : d2 = 150) (h4 : m2 = 15) : 
  (2 * d1) / ((d1/m1) + (d2/m2)) = 18.75 := by
  sorry

end average_gas_mileage_round_trip_l110_110592


namespace common_difference_arithmetic_sequence_l110_110577

theorem common_difference_arithmetic_sequence :
  ∃ d : ℝ, (d ≠ 0) ∧ (∀ (n : ℕ), a_n = 1 + (n-1) * d) ∧ ((1 + 2 * d)^2 = 1 * (1 + 8 * d)) → d = 1 :=
by
  sorry

end common_difference_arithmetic_sequence_l110_110577


namespace leak_empties_tank_in_12_hours_l110_110583

theorem leak_empties_tank_in_12_hours 
  (capacity : ℕ) (inlet_rate : ℕ) (net_emptying_time : ℕ) (leak_rate : ℤ) (leak_emptying_time : ℕ) :
  capacity = 5760 →
  inlet_rate = 4 →
  net_emptying_time = 8 →
  (inlet_rate - leak_rate : ℤ) = (capacity / (net_emptying_time * 60)) →
  leak_emptying_time = (capacity / leak_rate) →
  leak_emptying_time = 12 * 60 / 60 :=
by sorry

end leak_empties_tank_in_12_hours_l110_110583


namespace geometric_seq_sum_l110_110687

-- Definitions of the conditions
def a (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | _ => (-3)^(n - 1)

theorem geometric_seq_sum : 
  a 0 + |a 1| + a 2 + |a 3| + a 4 = 121 := by
  sorry

end geometric_seq_sum_l110_110687


namespace stacy_height_proof_l110_110808

noncomputable def height_last_year : ℕ := 50
noncomputable def brother_growth : ℕ := 1
noncomputable def stacy_growth : ℕ := brother_growth + 6
noncomputable def stacy_current_height : ℕ := height_last_year + stacy_growth

theorem stacy_height_proof : stacy_current_height = 57 := 
by
  sorry

end stacy_height_proof_l110_110808


namespace gcd_of_polynomials_l110_110767

theorem gcd_of_polynomials (b : ℤ) (h : b % 1620 = 0) : Int.gcd (b^2 + 11 * b + 36) (b + 6) = 6 := 
by
  sorry

end gcd_of_polynomials_l110_110767


namespace intersection_of_M_and_N_is_N_l110_110866

def M := {x : ℝ | x ≥ -1}
def N := {y : ℝ | y ≥ 0}

theorem intersection_of_M_and_N_is_N : M ∩ N = N := sorry

end intersection_of_M_and_N_is_N_l110_110866


namespace james_earnings_per_subscriber_is_9_l110_110036

/-
Problem:
James streams on Twitch. He had 150 subscribers and then someone gifted 50 subscribers. If he gets a certain amount per month per subscriber and now makes $1800 a month, how much does he make per subscriber?
-/

def initial_subscribers : ℕ := 150
def gifted_subscribers : ℕ := 50
def total_subscribers := initial_subscribers + gifted_subscribers
def total_earnings : ℤ := 1800

def earnings_per_subscriber := total_earnings / total_subscribers

/-
Theorem: James makes $9 per month for each subscriber.
-/
theorem james_earnings_per_subscriber_is_9 : earnings_per_subscriber = 9 := by
  -- to be filled in with proof steps
  sorry

end james_earnings_per_subscriber_is_9_l110_110036


namespace value_of_c_l110_110521

theorem value_of_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end value_of_c_l110_110521


namespace average_speed_round_trip_l110_110297

theorem average_speed_round_trip (D T : ℝ) (h1 : D = 51 * T) : (2 * D) / (3 * T) = 34 := 
by
  sorry

end average_speed_round_trip_l110_110297


namespace calculate_expression_l110_110124

theorem calculate_expression : 
  (3 * 7.5 * (6 + 4) / 2.5) = 90 := 
by
  sorry

end calculate_expression_l110_110124


namespace negation_proposition_l110_110848

variables {a b c : ℝ}

theorem negation_proposition (h : a ≤ b) : a + c ≤ b + c :=
sorry

end negation_proposition_l110_110848


namespace cylinder_radius_l110_110132

open Real

theorem cylinder_radius (r : ℝ) 
  (h₁ : ∀(V₁ : ℝ), V₁ = π * (r + 4)^2 * 3)
  (h₂ : ∀(V₂ : ℝ), V₂ = π * r^2 * 9)
  (h₃ : ∀(V₁ V₂ : ℝ), V₁ = V₂) :
  r = 2 + 2 * sqrt 3 :=
by
  sorry

end cylinder_radius_l110_110132


namespace playground_ball_cost_l110_110244

-- Define the given conditions
def cost_jump_rope : ℕ := 7
def cost_board_game : ℕ := 12
def saved_by_dalton : ℕ := 6
def given_by_uncle : ℕ := 13
def additional_needed : ℕ := 4

-- Total money Dalton has
def total_money : ℕ := saved_by_dalton + given_by_uncle

-- Total cost needed to buy all three items
def total_cost_needed : ℕ := total_money + additional_needed

-- Combined cost of the jump rope and the board game
def combined_cost : ℕ := cost_jump_rope + cost_board_game

-- Prove the cost of the playground ball
theorem playground_ball_cost : ℕ := total_cost_needed - combined_cost

-- Expected result
example : playground_ball_cost = 4 := by
  sorry

end playground_ball_cost_l110_110244


namespace volume_of_resulting_shape_l110_110745

-- Define the edge lengths
def edge_length (original : ℕ) (small : ℕ) := original = 5 ∧ small = 1

-- Define the volume of a cube
def volume (a : ℕ) : ℕ := a * a * a

-- State the proof problem
theorem volume_of_resulting_shape : ∀ (original small : ℕ), edge_length original small → 
  volume original - (5 * volume small) = 120 := by
  sorry

end volume_of_resulting_shape_l110_110745


namespace clownfish_in_display_tank_l110_110763

theorem clownfish_in_display_tank (C B : ℕ) (h1 : C = B) (h2 : C + B = 100) : 
  (B - 26 - (B - 26) / 3) = 16 := by
  sorry

end clownfish_in_display_tank_l110_110763


namespace intersection_M_N_l110_110634

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l110_110634


namespace triangle_trig_identity_l110_110233

open Real

theorem triangle_trig_identity (A B C : ℝ) (h_triangle : A + B + C = 180) (h_A : A = 15) :
  sqrt 3 * sin A - cos (B + C) = sqrt 2 := by
  sorry

end triangle_trig_identity_l110_110233


namespace avg_speed_is_40_l110_110792

noncomputable def average_speed (x : ℝ) : ℝ :=
  let time1 := x / 40
  let time2 := 2 * x / 20
  let total_time := time1 + time2
  let total_distance := 5 * x
  total_distance / total_time

theorem avg_speed_is_40 (x : ℝ) (hx : x > 0) :
  average_speed x = 40 := by
  sorry

end avg_speed_is_40_l110_110792


namespace joe_new_average_score_after_dropping_lowest_l110_110090

theorem joe_new_average_score_after_dropping_lowest 
  (initial_average : ℕ)
  (lowest_score : ℕ)
  (num_tests : ℕ)
  (new_num_tests : ℕ)
  (total_points : ℕ)
  (new_total_points : ℕ)
  (new_average : ℕ) :
  initial_average = 70 →
  lowest_score = 55 →
  num_tests = 4 →
  new_num_tests = 3 →
  total_points = num_tests * initial_average →
  new_total_points = total_points - lowest_score →
  new_average = new_total_points / new_num_tests →
  new_average = 75 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end joe_new_average_score_after_dropping_lowest_l110_110090


namespace diving_club_capacity_l110_110456

theorem diving_club_capacity :
  (3 * ((2 * 5 + 4 * 2) * 5) = 270) :=
by
  sorry

end diving_club_capacity_l110_110456


namespace particle_probability_at_2_3_after_5_moves_l110_110130

theorem particle_probability_at_2_3_after_5_moves:
  ∃ (C : ℕ), C = Nat.choose 5 2 ∧
  (1/2 ^ 5 * C) = (Nat.choose 5 2) * ((1/2: ℝ) ^ 5) := by
sorry

end particle_probability_at_2_3_after_5_moves_l110_110130


namespace combined_selling_price_l110_110477

theorem combined_selling_price :
  let cost_cycle := 2300
  let cost_scooter := 12000
  let cost_motorbike := 25000
  let loss_cycle := 0.30
  let profit_scooter := 0.25
  let profit_motorbike := 0.15
  let selling_price_cycle := cost_cycle - (loss_cycle * cost_cycle)
  let selling_price_scooter := cost_scooter + (profit_scooter * cost_scooter)
  let selling_price_motorbike := cost_motorbike + (profit_motorbike * cost_motorbike)
  selling_price_cycle + selling_price_scooter + selling_price_motorbike = 45360 := 
by
  sorry

end combined_selling_price_l110_110477


namespace Alfonso_daily_earnings_l110_110658

-- Define the conditions given in the problem
def helmet_cost : ℕ := 340
def current_savings : ℕ := 40
def days_per_week : ℕ := 5
def weeks_to_work : ℕ := 10

-- Define the question as a property to prove
def daily_earnings : ℕ := 6

-- Prove that the daily earnings are $6 given the conditions
theorem Alfonso_daily_earnings :
  (helmet_cost - current_savings) / (days_per_week * weeks_to_work) = daily_earnings :=
by
  sorry

end Alfonso_daily_earnings_l110_110658


namespace sequence_prime_bounded_l110_110434

theorem sequence_prime_bounded (c : ℕ) (h : c > 0) : 
  ∀ (p : ℕ → ℕ), (∀ k, Nat.Prime (p k)) → (p 0) = some_prime →
  (∀ k, ∃ q, Nat.Prime q ∧ q ∣ (p k + c) ∧ (∀ i < k, q ≠ p i)) → 
  (∃ N, ∀ m ≥ N, ∀ n ≥ N, p m = p n) :=
by
  sorry

end sequence_prime_bounded_l110_110434


namespace max_notebooks_lucy_can_buy_l110_110689

-- Definitions given in the conditions
def lucyMoney : ℕ := 2145
def notebookCost : ℕ := 230

-- Theorem to prove the number of notebooks Lucy can buy
theorem max_notebooks_lucy_can_buy : lucyMoney / notebookCost = 9 := 
by
  sorry

end max_notebooks_lucy_can_buy_l110_110689


namespace sum_of_data_l110_110905

theorem sum_of_data (a b c : ℕ) (h1 : a + b = c) (h2 : b = 3 * a) (h3 : a = 12) : a + b + c = 96 :=
by
  sorry

end sum_of_data_l110_110905


namespace dunkers_lineup_count_l110_110196

theorem dunkers_lineup_count (players : Finset ℕ) (h_players : players.card = 15) (alice zen : ℕ) 
  (h_alice : alice ∈ players) (h_zen : zen ∈ players) (h_distinct : alice ≠ zen) :
  (∃ (S : Finset (Finset ℕ)), S.card = 2717 ∧ ∀ s ∈ S, s.card = 5 ∧ ¬ (alice ∈ s ∧ zen ∈ s)) :=
by
  sorry

end dunkers_lineup_count_l110_110196


namespace max_value_x_minus_y_proof_l110_110299

noncomputable def max_value_x_minus_y (θ : ℝ) : ℝ :=
  sorry

theorem max_value_x_minus_y_proof (θ : ℝ) (h1 : x = Real.sin θ) (h2 : y = Real.cos θ)
(h3 : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) (h4 : (x^2 + y^2)^2 = x + y) : 
  max_value_x_minus_y θ = Real.sqrt 2 :=
sorry

end max_value_x_minus_y_proof_l110_110299


namespace find_a_for_tangent_l110_110821

theorem find_a_for_tangent (a : ℤ) (x : ℝ) (h : ∀ x, 3*x^2 - 4*a*x + 2*a > 0) : a = 1 :=
sorry

end find_a_for_tangent_l110_110821


namespace graph_passes_through_quadrants_l110_110669

-- Definitions based on the conditions
def linear_function (x : ℝ) : ℝ := -2 * x + 1

-- The property to be proven
theorem graph_passes_through_quadrants :
  (∃ x > 0, linear_function x > 0) ∧  -- Quadrant I
  (∃ x < 0, linear_function x > 0) ∧  -- Quadrant II
  (∃ x > 0, linear_function x < 0) := -- Quadrant IV
sorry

end graph_passes_through_quadrants_l110_110669


namespace pond_volume_l110_110247

theorem pond_volume {L W H : ℝ} (hL : L = 20) (hW : W = 12) (hH : H = 5) : L * W * H = 1200 := by
  sorry

end pond_volume_l110_110247


namespace probability_of_red_ball_and_removed_red_balls_l110_110957

-- Conditions for the problem
def initial_red_balls : Nat := 10
def initial_yellow_balls : Nat := 2
def initial_blue_balls : Nat := 8
def total_balls : Nat := initial_red_balls + initial_yellow_balls + initial_blue_balls

-- Problem statement in Lean
theorem probability_of_red_ball_and_removed_red_balls :
  (initial_red_balls / total_balls = 1 / 2) ∧
  (∃ (x : Nat), -- Number of red balls removed
    ((initial_yellow_balls + x) / total_balls = 2 / 5) ∧
    (initial_red_balls - x = 10 - 6)) := 
by
  -- Lean will need the proofs here; we use sorry for now.
  sorry

end probability_of_red_ball_and_removed_red_balls_l110_110957


namespace inequality_1_solution_set_inequality_2_solution_set_l110_110465

theorem inequality_1_solution_set (x : ℝ) : 
  (2 + 3 * x - 2 * x^2 > 0) ↔ (-1/2 < x ∧ x < 2) := 
by sorry

theorem inequality_2_solution_set (x : ℝ) :
  (x * (3 - x) ≤ x * (x + 2) - 1) ↔ (x ≤ -1/2 ∨ x ≥ 1) :=
by sorry

end inequality_1_solution_set_inequality_2_solution_set_l110_110465


namespace new_population_difference_l110_110336

def population_eagles : ℕ := 150
def population_falcons : ℕ := 200
def population_hawks : ℕ := 320
def population_owls : ℕ := 270
def increase_rate : ℕ := 10

theorem new_population_difference :
  let least_populous := min population_eagles (min population_falcons (min population_hawks population_owls))
  let most_populous := max population_eagles (max population_falcons (max population_hawks population_owls))
  let increased_least_populous := least_populous + least_populous * increase_rate / 100
  most_populous - increased_least_populous = 155 :=
by
  sorry

end new_population_difference_l110_110336


namespace exists_positive_integers_x_y_l110_110823

theorem exists_positive_integers_x_y (x y : ℕ) : 0 < x ∧ 0 < y ∧ x^2 = y^2 + 2023 :=
  sorry

end exists_positive_integers_x_y_l110_110823


namespace line_segments_property_l110_110728

theorem line_segments_property (L : List (ℝ × ℝ)) :
  L.length = 50 →
  (∃ S : List (ℝ × ℝ), S.length = 8 ∧ ∃ x : ℝ, ∀ seg ∈ S, seg.fst ≤ x ∧ x ≤ seg.snd) ∨
  (∃ T : List (ℝ × ℝ), T.length = 8 ∧ ∀ seg1 ∈ T, ∀ seg2 ∈ T, seg1 ≠ seg2 → seg1.snd < seg2.fst ∨ seg2.snd < seg1.fst) :=
by
  -- Theorem proof placeholder
  sorry

end line_segments_property_l110_110728


namespace ratio_of_a_to_b_l110_110943

theorem ratio_of_a_to_b (a b : ℝ) (h1 : 0.5 / 100 * a = 85) (h2 : 0.75 / 100 * b = 150) : a / b = 17 / 20 :=
by {
  -- Proof will go here
  sorry
}

end ratio_of_a_to_b_l110_110943


namespace solve_expression_l110_110901

theorem solve_expression : 3 ^ (1 ^ (0 ^ 2)) - ((3 ^ 1) ^ 0) ^ 2 = 2 := by
  sorry

end solve_expression_l110_110901


namespace rectangular_frame_wire_and_paper_area_l110_110433

theorem rectangular_frame_wire_and_paper_area :
  let l1 := 3
  let l2 := 4
  let l3 := 5
  let wire_length := (l1 + l2 + l3) * 4
  let paper_area := ((l1 * l2) + (l1 * l3) + (l2 * l3)) * 2
  wire_length = 48 ∧ paper_area = 94 :=
by
  sorry

end rectangular_frame_wire_and_paper_area_l110_110433


namespace other_solution_of_quadratic_l110_110881

theorem other_solution_of_quadratic (x : ℚ) 
  (hx1 : 77 * x^2 - 125 * x + 49 = 0) (hx2 : x = 8/11) : 
  77 * (1 : ℚ)^2 - 125 * (1 : ℚ) + 49 = 0 :=
by sorry

end other_solution_of_quadratic_l110_110881


namespace lateral_surface_area_of_prism_l110_110397

theorem lateral_surface_area_of_prism 
  (a : ℝ) (α β V : ℝ) :
  let sin (x : ℝ) := Real.sin x 
  ∃ S : ℝ,
    S = (2 * V * sin ((α + β) / 2)) / (a * sin (α / 2) * sin (β / 2)) := 
sorry

end lateral_surface_area_of_prism_l110_110397


namespace probability_two_black_balls_l110_110317

theorem probability_two_black_balls (white_balls black_balls drawn_balls : ℕ) 
  (h_w : white_balls = 4) (h_b : black_balls = 7) (h_d : drawn_balls = 2) :
  let total_ways := Nat.choose (white_balls + black_balls) drawn_balls
  let black_ways := Nat.choose black_balls drawn_balls
  (black_ways / total_ways : ℚ) = 21 / 55 :=
by
  sorry

end probability_two_black_balls_l110_110317


namespace find_speed_second_train_l110_110380

noncomputable def speed_second_train (length_train1 length_train2 : ℝ) (speed_train1_kmph : ℝ) (time_to_cross : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let total_distance := length_train1 + length_train2
  let relative_speed_mps := total_distance / time_to_cross
  let speed_train2_mps := speed_train1_mps - relative_speed_mps
  speed_train2_mps * 3600 / 1000

theorem find_speed_second_train :
  speed_second_train 380 540 72 91.9926405887529 = 36 := by
  sorry

end find_speed_second_train_l110_110380


namespace new_car_distance_l110_110815

-- State the given conditions as a Lemma in Lean 4
theorem new_car_distance (d_old : ℝ) (d_new : ℝ) (h1 : d_old = 150) (h2 : d_new = d_old * 1.30) : d_new = 195 := 
by
  sorry

end new_car_distance_l110_110815


namespace relation_between_A_and_B_l110_110661

-- Define the sets A and B
def A : Set ℤ := { x | ∃ k : ℕ, x = 7 * k + 3 }
def B : Set ℤ := { x | ∃ k : ℤ, x = 7 * k - 4 }

-- Prove the relationship between A and B
theorem relation_between_A_and_B : A ⊆ B :=
by
  sorry

end relation_between_A_and_B_l110_110661


namespace factorize_difference_of_squares_l110_110682

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorize_difference_of_squares_l110_110682


namespace find_a_l110_110288

theorem find_a (a : ℝ) (h : 6 * a + 4 = 0) : a = -2 / 3 :=
by
  sorry

end find_a_l110_110288


namespace nat_lemma_l110_110209

theorem nat_lemma (a b : ℕ) : (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) → (a = 1 ∧ b = 1) := by
  sorry

end nat_lemma_l110_110209


namespace mn_eq_neg_infty_to_0_l110_110418

-- Definitions based on the conditions
def M : Set ℝ := {y | y ≤ 2}
def N : Set ℝ := {y | 0 ≤ y ∧ y ≤ 3}

-- Set difference definition
def set_diff (A B : Set ℝ) : Set ℝ := {y | y ∈ A ∧ y ∉ B}

-- The proof statement we need to prove
theorem mn_eq_neg_infty_to_0 : set_diff M N = {y | y < 0} :=
  sorry  -- Proof will go here

end mn_eq_neg_infty_to_0_l110_110418


namespace calculate_expression_l110_110513

theorem calculate_expression :
  ((-1 -2 -3 -4 -5 -6 -7 -8 -9 -10) * (1 -2 +3 -4 +5 -6 +7 -8 +9 -10) = 275) :=
by
  sorry

end calculate_expression_l110_110513


namespace volume_in_cubic_yards_l110_110282

theorem volume_in_cubic_yards (V : ℝ) (conversion_factor : ℝ) (hV : V = 216) (hcf : conversion_factor = 27) :
  V / conversion_factor = 8 := by
  sorry

end volume_in_cubic_yards_l110_110282


namespace evaluate_fraction_l110_110053

noncomputable section

variables (u v : ℂ)
variables (h1 : u ≠ 0) (h2 : v ≠ 0) (h3 : u^2 + u * v + v^2 = 0)

theorem evaluate_fraction : (u^7 + v^7) / (u + v)^7 = -2 := by
  sorry

end evaluate_fraction_l110_110053


namespace decimal_equivalent_of_squared_fraction_l110_110470

theorem decimal_equivalent_of_squared_fraction : (1 / 5 : ℝ)^2 = 0.04 :=
by
  sorry

end decimal_equivalent_of_squared_fraction_l110_110470


namespace find_quadratic_function_l110_110824

def quadratic_function (c d : ℝ) (x : ℝ) : ℝ :=
  x^2 + c * x + d

theorem find_quadratic_function :
  ∃ c d, (∀ x, 
    (quadratic_function c d (quadratic_function c d x + 2 * x)) / (quadratic_function c d x) = 2 * x^2 + 1984 * x + 2024) ∧ 
    quadratic_function c d x = x^2 + 1982 * x + 21 :=
by
  sorry

end find_quadratic_function_l110_110824


namespace num_complementary_sets_eq_117_l110_110074

structure Card :=
(shape : Type)
(color : Type)
(shade : Type)

def deck_condition: Prop := 
  ∃ (deck : List Card), 
  deck.length = 27 ∧
  ∀ c1 c2 c3, c1 ∈ deck ∧ c2 ∈ deck ∧ c3 ∈ deck →
  (c1.shape ≠ c2.shape ∨ c2.shape ≠ c3.shape ∨ c1.shape = c3.shape) ∧
  (c1.color ≠ c2.color ∨ c2.color ≠ c3.color ∨ c1.color = c3.color) ∧
  (c1.shade ≠ c2.shade ∨ c2.shade ≠ c3.shade ∨ c1.shade = c3.shade)

theorem num_complementary_sets_eq_117 :
  deck_condition → ∃ sets : List (List Card), sets.length = 117 := sorry

end num_complementary_sets_eq_117_l110_110074


namespace basketball_scores_l110_110624

theorem basketball_scores : ∃ (scores : Finset ℕ), 
  scores = { x | ∃ a b : ℕ, a + b = 7 ∧ x = 2 * a + 3 * b } ∧ scores.card = 8 :=
by
  sorry

end basketball_scores_l110_110624


namespace parabola_points_relationship_l110_110412

theorem parabola_points_relationship :
  let y_1 := (-2)^2 + 2 * (-2) - 9
  let y_2 := 1^2 + 2 * 1 - 9
  let y_3 := 3^2 + 2 * 3 - 9
  y_3 > y_2 ∧ y_2 > y_1 :=
by
  sorry

end parabola_points_relationship_l110_110412


namespace focus_of_parabola_eq_l110_110182

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := -5 * x^2 + 10 * x - 2

-- Statement of the theorem to find the focus of the given parabola
theorem focus_of_parabola_eq (x : ℝ) : 
  let vertex_x := 1
  let vertex_y := 3
  let a := -5
  ∃ focus_x focus_y, 
    focus_x = vertex_x ∧ 
    focus_y = vertex_y - (1 / (4 * a)) ∧
    focus_x = 1 ∧
    focus_y = 59 / 20 := 
  sorry

end focus_of_parabola_eq_l110_110182


namespace equal_cost_at_20_minutes_l110_110750

/-- Define the cost functions for each telephone company -/
def united_cost (m : ℝ) : ℝ := 11 + 0.25 * m
def atlantic_cost (m : ℝ) : ℝ := 12 + 0.20 * m
def global_cost (m : ℝ) : ℝ := 13 + 0.15 * m

/-- Prove that at 20 minutes, the cost is the same for all three companies -/
theorem equal_cost_at_20_minutes : 
  united_cost 20 = atlantic_cost 20 ∧ atlantic_cost 20 = global_cost 20 :=
by
  sorry

end equal_cost_at_20_minutes_l110_110750


namespace train_speed_proof_l110_110212

noncomputable def train_length : ℝ := 620
noncomputable def crossing_time : ℝ := 30.99752019838413
noncomputable def man_speed_kmh : ℝ := 8

noncomputable def man_speed_ms : ℝ := man_speed_kmh * (1000 / 3600)
noncomputable def relative_speed : ℝ := train_length / crossing_time
noncomputable def train_speed_ms : ℝ := relative_speed + man_speed_ms
noncomputable def train_speed_kmh : ℝ := train_speed_ms * (3600 / 1000)

theorem train_speed_proof : abs (train_speed_kmh - 80) < 0.0001 := by
  sorry

end train_speed_proof_l110_110212


namespace books_into_bags_l110_110093

def books := Finset.range 5
def bags := Finset.range 4

noncomputable def arrangement_count : ℕ :=
  -- definition of arrangement_count can be derived from the solution logic
  sorry

theorem books_into_bags : arrangement_count = 51 := 
  sorry

end books_into_bags_l110_110093


namespace transformed_system_solution_l110_110771

theorem transformed_system_solution 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * 3 + b1 * 4 = c1)
  (h2 : a2 * 3 + b2 * 4 = c2) :
  (3 * a1 * 5 + 4 * b1 * 5 = 5 * c1) ∧ (3 * a2 * 5 + 4 * b2 * 5 = 5 * c2) :=
by 
  sorry

end transformed_system_solution_l110_110771


namespace lindas_average_speed_l110_110949

theorem lindas_average_speed
  (dist1 : ℕ) (time1 : ℝ)
  (dist2 : ℕ) (time2 : ℝ)
  (h1 : dist1 = 450)
  (h2 : time1 = 7.5)
  (h3 : dist2 = 480)
  (h4 : time2 = 8) :
  (dist1 + dist2) / (time1 + time2) = 60 :=
by
  sorry

end lindas_average_speed_l110_110949


namespace five_letters_three_mailboxes_l110_110364

theorem five_letters_three_mailboxes : (∃ n : ℕ, n = 5) ∧ (∃ m : ℕ, m = 3) → ∃ k : ℕ, k = m^n :=
by
  sorry

end five_letters_three_mailboxes_l110_110364


namespace proposition_p_proposition_not_q_proof_p_and_not_q_l110_110952

variable (p : Prop)
variable (q : Prop)
variable (r : Prop)

theorem proposition_p : (∃ x0 : ℝ, x0 > 2) := sorry

theorem proposition_not_q : ¬ (∀ x : ℝ, x^3 > x^2) := sorry

theorem proof_p_and_not_q : (∃ x0 : ℝ, x0 > 2) ∧ ¬ (∀ x : ℝ, x^3 > x^2) :=
by
  exact ⟨proposition_p, proposition_not_q⟩

end proposition_p_proposition_not_q_proof_p_and_not_q_l110_110952


namespace sum_series_eq_4_div_9_l110_110884

theorem sum_series_eq_4_div_9 :
  (∑' k : ℕ, k / 4^k : ℝ) = 4 / 9 :=
sorry

end sum_series_eq_4_div_9_l110_110884


namespace triangle_isosceles_or_right_angled_l110_110258

theorem triangle_isosceles_or_right_angled
  (β γ : ℝ)
  (h : Real.tan β * Real.sin γ ^ 2 = Real.tan γ * Real.sin β ^ 2) :
  (β = γ) ∨ (β + γ = π / 2) :=
sorry

end triangle_isosceles_or_right_angled_l110_110258


namespace gcd_108_450_l110_110739

theorem gcd_108_450 : Nat.gcd 108 450 = 18 :=
by
  sorry

end gcd_108_450_l110_110739


namespace no_solution_exists_l110_110041

open Int

theorem no_solution_exists (x y z : ℕ) (hx : x > 0) (hy : y > 0)
  (hz : z = Nat.gcd x y) : x + y^2 + z^3 ≠ x * y * z := 
sorry

end no_solution_exists_l110_110041


namespace price_of_computer_and_desk_l110_110917

theorem price_of_computer_and_desk (x y : ℕ) 
  (h1 : 10 * x + 200 * y = 90000)
  (h2 : 12 * x + 120 * y = 90000) : 
  x = 6000 ∧ y = 150 :=
by
  sorry

end price_of_computer_and_desk_l110_110917


namespace base_prime_rep_360_l110_110609

-- Define the value 360 as n
def n : ℕ := 360

-- Function to compute the base prime representation.
noncomputable def base_prime_representation (n : ℕ) : ℕ :=
  -- Normally you'd implement the actual function to convert n to its base prime representation here
  sorry

-- The theorem statement claiming that the base prime representation of 360 is 213
theorem base_prime_rep_360 : base_prime_representation n = 213 := 
  sorry

end base_prime_rep_360_l110_110609


namespace total_passengers_correct_l110_110948

-- Definition of the conditions
def passengers_on_time : ℕ := 14507
def passengers_late : ℕ := 213
def total_passengers : ℕ := passengers_on_time + passengers_late

-- Theorem statement
theorem total_passengers_correct : total_passengers = 14720 := by
  sorry

end total_passengers_correct_l110_110948


namespace vector_combination_l110_110495

open Complex

def z1 : ℂ := -1 + I
def z2 : ℂ := 1 + I
def z3 : ℂ := 1 + 4 * I

def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, 4)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def OC : ℝ × ℝ := C

def x : ℝ := sorry
def y : ℝ := sorry

theorem vector_combination (hx : OC = ( - x + y, x + y )) : 
    x + y = 4 :=
by
    sorry

end vector_combination_l110_110495


namespace decrease_percent_revenue_l110_110025

theorem decrease_percent_revenue 
  (T C : ℝ) 
  (hT : T > 0) 
  (hC : C > 0) 
  (new_tax : ℝ := 0.65 * T) 
  (new_consumption : ℝ := 1.15 * C) 
  (original_revenue : ℝ := T * C) 
  (new_revenue : ℝ := new_tax * new_consumption) :
  100 * (original_revenue - new_revenue) / original_revenue = 25.25 :=
sorry

end decrease_percent_revenue_l110_110025


namespace compute_f_six_l110_110119

def f (x : Int) : Int :=
  if x ≥ 0 then -x^2 - 1 else x + 10

theorem compute_f_six (x : Int) : f (f (f (f (f (f 1))))) = -35 :=
by
  sorry

end compute_f_six_l110_110119


namespace missing_score_find_missing_score_l110_110636

theorem missing_score
  (score1 score2 score3 score4 mean total : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89)
  (hTotal : total = 445) :
  score1 + score2 + score3 + score4 + x = total :=
by
  sorry

theorem find_missing_score
  (score1 score2 score3 score4 mean : ℝ) (x : ℝ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 87)
  (h4 : score4 = 93)
  (hMean : mean = 89) :
  (score1 + score2 + score3 + score4 + x) / 5 = mean
  → x = 90 :=
by
  sorry

end missing_score_find_missing_score_l110_110636


namespace inverse_of_square_l110_110207

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

theorem inverse_of_square (h : A⁻¹ = ![
  ![3, -2],
  ![1, 1]
]) : 
  (A^2)⁻¹ = ![
  ![7, -8],
  ![4, -1]
] :=
sorry

end inverse_of_square_l110_110207


namespace soup_can_pyramid_rows_l110_110224

theorem soup_can_pyramid_rows (n : ℕ) :
  (∃ (n : ℕ), (2 * n^2 - n = 225)) → n = 11 :=
by
  sorry

end soup_can_pyramid_rows_l110_110224


namespace find_y_in_triangle_l110_110775

theorem find_y_in_triangle (BAC ABC BCA : ℝ) (y : ℝ) (h1 : BAC = 90)
  (h2 : ABC = 2 * y) (h3 : BCA = y - 10) : y = 100 / 3 :=
by
  -- The proof will be left as sorry
  sorry

end find_y_in_triangle_l110_110775


namespace find_minimum_a_l110_110520

theorem find_minimum_a (a x : ℤ) : 
  (x - a < 0) → 
  (x > -3 / 2) → 
  (∃ n : ℤ, ∀ y : ℤ, y ∈ {k | -1 ≤ k ∧ k ≤ n} ∧ y < a) → 
  a = 3 := sorry

end find_minimum_a_l110_110520


namespace smallest_abc_sum_l110_110263

theorem smallest_abc_sum : 
  ∃ (a b c : ℕ), (a * c + 2 * b * c + a + 2 * b = c^2 + c + 6) ∧ (∀ (a' b' c' : ℕ), (a' * c' + 2 * b' * c' + a' + 2 * b' = c'^2 + c' + 6) → (a' + b' + c' ≥ a + b + c)) → (a, b, c) = (2, 1, 1) := 
by
  sorry

end smallest_abc_sum_l110_110263


namespace keychain_arrangement_l110_110837

theorem keychain_arrangement (house car locker office key5 key6 : ℕ) :
  (∃ (A B : ℕ), house = A ∧ car = A ∧ locker = B ∧ office = B) →
  (∃ (arrangements : ℕ), arrangements = 24) :=
by
  sorry

end keychain_arrangement_l110_110837


namespace simplify_and_evaluate_l110_110942

variable (x y : ℝ)
variable (condition_x : x = 1/3)
variable (condition_y : y = -6)

theorem simplify_and_evaluate :
  3 * x^2 * y - (6 * x * y^2 - 2 * (x * y + (3/2) * x^2 * y)) + 2 * (3 * x * y^2 - x * y) = -4 :=
by
  rw [condition_x, condition_y]
  sorry

end simplify_and_evaluate_l110_110942


namespace manager_monthly_salary_l110_110616

theorem manager_monthly_salary :
  let avg_salary := 1200
  let num_employees := 20
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + 100
  let num_people_with_manager := num_employees + 1
  let new_total_salary := num_people_with_manager * new_avg_salary
  let manager_salary := new_total_salary - total_salary
  manager_salary = 3300 := by
  sorry

end manager_monthly_salary_l110_110616


namespace avg_remaining_two_l110_110784

theorem avg_remaining_two (avg5 avg3 : ℝ) (h1 : avg5 = 12) (h2 : avg3 = 4) : (5 * avg5 - 3 * avg3) / 2 = 24 :=
by sorry

end avg_remaining_two_l110_110784


namespace bsnt_value_l110_110686

theorem bsnt_value (B S N T : ℝ) (hB : 0 < B) (hS : 0 < S) (hN : 0 < N) (hT : 0 < T)
    (h1 : Real.log (B * S) / Real.log 10 + Real.log (B * N) / Real.log 10 = 3)
    (h2 : Real.log (N * T) / Real.log 10 + Real.log (N * S) / Real.log 10 = 4)
    (h3 : Real.log (S * T) / Real.log 10 + Real.log (S * B) / Real.log 10 = 5) :
    B * S * N * T = 10000 :=
sorry

end bsnt_value_l110_110686


namespace lcm_is_600_l110_110100

def lcm_of_24_30_40_50_60 : ℕ :=
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60)))

theorem lcm_is_600 : lcm_of_24_30_40_50_60 = 600 := by
  sorry

end lcm_is_600_l110_110100


namespace complement_of_angle_correct_l110_110358

def complement_of_angle (a : ℚ) : ℚ := 90 - a

theorem complement_of_angle_correct : complement_of_angle (40 + 30/60) = 49 + 30/60 :=
by
  -- placeholder for the proof
  sorry

end complement_of_angle_correct_l110_110358


namespace maximum_triangles_in_right_angle_triangle_l110_110490

-- Definition of grid size and right-angled triangle on graph paper
def grid_size : Nat := 7

-- Definition of the vertices of the right-angled triangle
def vertices : List (Nat × Nat) := [(0,0), (grid_size,0), (0,grid_size)]

-- Total number of unique triangles that can be identified
theorem maximum_triangles_in_right_angle_triangle (grid_size : Nat) (vertices : List (Nat × Nat)) : 
  Nat :=
  if vertices = [(0,0), (grid_size,0), (0,grid_size)] then 28 else 0

end maximum_triangles_in_right_angle_triangle_l110_110490


namespace undecided_voters_percentage_l110_110534

theorem undecided_voters_percentage
  (biff_percent : ℝ)
  (total_people : ℤ)
  (marty_votes : ℤ)
  (undecided_percent : ℝ) :
  biff_percent = 0.45 →
  total_people = 200 →
  marty_votes = 94 →
  undecided_percent = ((total_people - (marty_votes + (biff_percent * total_people))) / total_people) * 100 →
  undecided_percent = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end undecided_voters_percentage_l110_110534


namespace inequality_abc_l110_110933

variables {a b c : ℝ}

theorem inequality_abc 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2) ∧ 
    (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) := 
by
  sorry

end inequality_abc_l110_110933


namespace binom_coeff_div_prime_l110_110174

open Nat

theorem binom_coeff_div_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p :=
by
  sorry

end binom_coeff_div_prime_l110_110174


namespace probability_ge_sqrt2_l110_110508

noncomputable def probability_length_chord_ge_sqrt2
  (a : ℝ)
  (h : a ≠ 0)
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  : ℝ :=
  if -1 ≤ a ∧ a ≤ 1 then (1 / Real.sqrt (1^2 + 1^2)) else 0

theorem probability_ge_sqrt2 
  (a : ℝ) 
  (h : a ≠ 0) 
  (intersect_cond : ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ (x - a)^2 + (y - a)^2 = 1)
  (length_cond : (Real.sqrt (4 - 2*a^2) ≥ Real.sqrt 2)) : 
  probability_length_chord_ge_sqrt2 a h intersect_cond = (Real.sqrt 2 / 2) :=
by
  sorry

end probability_ge_sqrt2_l110_110508


namespace pentagonal_tiles_count_l110_110370

theorem pentagonal_tiles_count (t p : ℕ) (h1 : t + p = 30) (h2 : 3 * t + 5 * p = 100) : p = 5 :=
sorry

end pentagonal_tiles_count_l110_110370


namespace perimeter_of_square_l110_110512

theorem perimeter_of_square (A : ℝ) (hA : A = 400) : exists P : ℝ, P = 80 :=
by
  sorry

end perimeter_of_square_l110_110512


namespace plant_height_after_year_l110_110769

theorem plant_height_after_year (current_height : ℝ) (monthly_growth : ℝ) (months_in_year : ℕ) (total_growth : ℝ)
  (h1 : current_height = 20)
  (h2 : monthly_growth = 5)
  (h3 : months_in_year = 12)
  (h4 : total_growth = monthly_growth * months_in_year) :
  current_height + total_growth = 80 :=
sorry

end plant_height_after_year_l110_110769


namespace monkeys_and_bananas_l110_110395

theorem monkeys_and_bananas :
  (∀ (m n t : ℕ), m * t = n → (∀ (m' n' t' : ℕ), n = m * (t / t') → n' = (m' * t') / t → n' = n → m' = m)) →
  (6 : ℕ) = 6 :=
by
  intros H
  let m := 6
  let n := 6
  let t := 6
  have H1 : m * t = n := by sorry
  let k := 18
  let t' := 18
  have H2 : n = m * (t / t') := by sorry
  let n' := 18
  have H3 : n' = (m * t') / t := by sorry
  have H4 : n' = n := by sorry
  exact H m n t H1 6 n' t' H2 H3 H4

end monkeys_and_bananas_l110_110395


namespace factor_theorem_l110_110842

noncomputable def Q (b x : ℝ) : ℝ := x^4 - 3 * x^3 + b * x^2 - 12 * x + 24

theorem factor_theorem (b : ℝ) : (∃ x : ℝ, x = -2) ∧ (Q b x = 0) → b = -22 :=
by
  sorry

end factor_theorem_l110_110842


namespace max_value_of_x_squared_plus_xy_plus_y_squared_l110_110435

theorem max_value_of_x_squared_plus_xy_plus_y_squared
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 - x * y + y^2 = 9) : 
  (x^2 + x * y + y^2) ≤ 27 :=
sorry

end max_value_of_x_squared_plus_xy_plus_y_squared_l110_110435


namespace three_Y_five_l110_110056

-- Define the operation Y
def Y (a b : ℕ) : ℕ := 3 * b + 8 * a - a^2

-- State the theorem to prove the value of 3 Y 5
theorem three_Y_five : Y 3 5 = 30 :=
by
  sorry

end three_Y_five_l110_110056


namespace solve_for_y_l110_110579

theorem solve_for_y (y : ℤ) (h : (8 + 12 + 23 + 17 + y) / 5 = 15) : y = 15 :=
by {
  sorry
}

end solve_for_y_l110_110579


namespace necessary_but_not_sufficient_condition_l110_110064

theorem necessary_but_not_sufficient_condition (a : ℕ → ℝ) (a1_pos : 0 < a 1) (q : ℝ) (geo_seq : ∀ n, a (n+1) = q * a n) : 
  (∀ n : ℕ, a (2*n + 1) + a (2*n + 2) < 0) → q < 0 :=
sorry

end necessary_but_not_sufficient_condition_l110_110064


namespace inequality_solution_ge_11_l110_110390

theorem inequality_solution_ge_11
  (m n : ℝ)
  (h1 : m > 0)
  (h2 : n > 1)
  (h3 : (1/m) + (2/(n-1)) = 1) :
  m + 2 * n ≥ 11 :=
sorry

end inequality_solution_ge_11_l110_110390


namespace two_digit_numbers_less_than_35_l110_110973

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end two_digit_numbers_less_than_35_l110_110973


namespace smaller_prime_is_x_l110_110702

theorem smaller_prime_is_x (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (h1 : x + y = 36) (h2 : 4 * x + y = 87) : x = 17 :=
  sorry

end smaller_prime_is_x_l110_110702


namespace intersection_A_B_l110_110189

def A := { x : ℝ | -5 < x ∧ x < 2 }
def B := { x : ℝ | x^2 - 9 < 0 }
def AB := { x : ℝ | -3 < x ∧ x < 2 }

theorem intersection_A_B : A ∩ B = AB := by
  sorry

end intersection_A_B_l110_110189


namespace planter_cost_l110_110977

-- Define costs
def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny : ℝ := 4.00
def cost_geranium : ℝ := 3.50

-- Define quantities
def num_creeping_jennies : ℝ := 4
def num_geraniums : ℝ := 4
def num_corners : ℝ := 4

-- Define the total cost
def total_cost : ℝ :=
  (cost_palm_fern
   + (cost_creeping_jenny * num_creeping_jennies)
   + (cost_geranium * num_geraniums))
  * num_corners

-- Prove the total cost is $180.00
theorem planter_cost : total_cost = 180.00 :=
by
  sorry

end planter_cost_l110_110977


namespace water_glass_ounces_l110_110343

theorem water_glass_ounces (glasses_per_day : ℕ) (days_per_week : ℕ)
    (bottle_ounces : ℕ) (bottle_fills_per_week : ℕ)
    (total_glasses_per_week : ℕ)
    (total_ounces_per_week : ℕ)
    (glasses_per_week_eq : glasses_per_day * days_per_week = total_glasses_per_week)
    (ounces_per_week_eq : bottle_ounces * bottle_fills_per_week = total_ounces_per_week)
    (ounce_per_glass : ℕ)
    (glasses_per_week : ℕ)
    (ounces_per_week : ℕ) :
    total_ounces_per_week / total_glasses_per_week = 5 :=
by
  sorry

end water_glass_ounces_l110_110343


namespace Mrs_Lara_Late_l110_110793

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem Mrs_Lara_Late (d t : ℝ) (h1 : d = 50 * (t + 7 / 60)) (h2 : d = 70 * (t - 5 / 60)) :
  required_speed d t = 70 := by
  sorry

end Mrs_Lara_Late_l110_110793


namespace expansion_no_x2_term_l110_110927

theorem expansion_no_x2_term (n : ℕ) (h1 : 5 ≤ n) (h2 : n ≤ 8) :
  ¬ ∃ (r : ℕ), 0 ≤ r ∧ r ≤ n ∧ n - 4 * r = 2 → n = 7 := by
  sorry

end expansion_no_x2_term_l110_110927


namespace rate_per_sqm_l110_110536

theorem rate_per_sqm (length width : ℝ) (cost : ℝ) (Area : ℝ := length * width) (rate : ℝ := cost / Area) 
  (h_length : length = 5.5) (h_width : width = 3.75) (h_cost : cost = 8250) : 
  rate = 400 :=
sorry

end rate_per_sqm_l110_110536


namespace no_passing_quadrant_III_l110_110037

def y (k x : ℝ) : ℝ := k * x - k

theorem no_passing_quadrant_III (k : ℝ) (h : k < 0) :
  ¬(∃ x y : ℝ, x < 0 ∧ y < 0 ∧ y = k * x - k) :=
sorry

end no_passing_quadrant_III_l110_110037


namespace quadrilateral_is_rhombus_l110_110680

theorem quadrilateral_is_rhombus (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = ab + bc + cd + ad) : a = b ∧ b = c ∧ c = d :=
by
  sorry

end quadrilateral_is_rhombus_l110_110680


namespace number_of_outfits_l110_110451

-- Define the number of shirts, pants, and jacket options.
def shirts : Nat := 8
def pants : Nat := 5
def jackets : Nat := 3

-- The theorem statement for the total number of outfits.
theorem number_of_outfits : shirts * pants * jackets = 120 := 
by
  sorry

end number_of_outfits_l110_110451


namespace simplified_expression_evaluate_at_zero_l110_110164

noncomputable def simplify_expr (x : ℝ) : ℝ :=
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2 * x + 1))

theorem simplified_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  simplify_expr x = 1 / (x - 1) :=
by sorry

theorem evaluate_at_zero (h₁ : (0 : ℝ) ≠ -1) (h₂ : (0 : ℝ) ≠ 1) : 
  simplify_expr 0 = -1 :=
by sorry

end simplified_expression_evaluate_at_zero_l110_110164


namespace maximum_integer_value_of_fraction_is_12001_l110_110613

open Real

def max_fraction_value_12001 : Prop :=
  ∃ x : ℝ, (1 + 12 / (4 * x^2 + 12 * x + 8) : ℝ) = 12001

theorem maximum_integer_value_of_fraction_is_12001 :
  ∃ x : ℝ, 1 + (12 / (4 * x^2 + 12 * x + 8)) = 12001 :=
by
  -- Here you should provide the proof steps.
  sorry

end maximum_integer_value_of_fraction_is_12001_l110_110613


namespace largest_angle_of_convex_hexagon_l110_110372

noncomputable def hexagon_largest_angle (x : ℚ) : ℚ :=
  max (6 * x - 3) (max (5 * x + 1) (max (4 * x - 4) (max (3 * x) (max (2 * x + 2) x))))

theorem largest_angle_of_convex_hexagon (x : ℚ) (h : x + (2*x+2) + 3*x + (4*x-4) + (5*x+1) + (6*x-3) = 720) : 
  hexagon_largest_angle x = 4281 / 21 := 
sorry

end largest_angle_of_convex_hexagon_l110_110372


namespace compute_product_l110_110367

theorem compute_product (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1^3 - 3 * x1 * y1^2 = 1005) 
  (h2 : y1^3 - 3 * x1^2 * y1 = 1004)
  (h3 : x2^3 - 3 * x2 * y2^2 = 1005)
  (h4 : y2^3 - 3 * x2^2 * y2 = 1004)
  (h5 : x3^3 - 3 * x3 * y3^2 = 1005)
  (h6 : y3^3 - 3 * x3^2 * y3 = 1004) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 502 := 
sorry

end compute_product_l110_110367


namespace books_given_away_l110_110685

theorem books_given_away (original_books : ℝ) (books_left : ℝ) (books_given : ℝ) 
    (h1 : original_books = 54.0) 
    (h2 : books_left = 31) : 
    books_given = original_books - books_left → books_given = 23 :=
by
  sorry

end books_given_away_l110_110685


namespace common_chord_eq_l110_110985

theorem common_chord_eq (x y : ℝ) :
  (x^2 + y^2 + 2*x + 8*y - 8 = 0) →
  (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
  (x + 2*y - 1 = 0) :=
by
  intros h1 h2
  sorry

end common_chord_eq_l110_110985


namespace find_OH_squared_l110_110354

variables (A B C : ℝ) (a b c R OH : ℝ)

-- Conditions
def circumcenter (O : ℝ) := true  -- Placeholder, as the actual definition relies on geometric properties
def orthocenter (H : ℝ) := true   -- Placeholder, as the actual definition relies on geometric properties

axiom eqR : R = 5
axiom sumSquares : a^2 + b^2 + c^2 = 50

-- Problem statement
theorem find_OH_squared : OH^2 = 175 :=
by
  sorry

end find_OH_squared_l110_110354


namespace f_g_of_3_l110_110059

def g (x : ℕ) : ℕ := x^3
def f (x : ℕ) : ℕ := 3 * x - 2

theorem f_g_of_3 : f (g 3) = 79 := by
  sorry

end f_g_of_3_l110_110059


namespace g_at_10_l110_110051

noncomputable def g : ℕ → ℝ :=
sorry

axiom g_1 : g 1 = 2
axiom g_prop (m n : ℕ) (hmn : m ≥ n) : g (m + n) + g (m - n) = 2 * (g m + g n)

theorem g_at_10 : g 10 = 200 := 
sorry

end g_at_10_l110_110051


namespace fraction_equality_l110_110360

variable (a b : ℚ)

theorem fraction_equality (h : (4 * a + 3 * b) / (4 * a - 3 * b) = 4) : a / b = 5 / 4 := by
  sorry

end fraction_equality_l110_110360


namespace simplify_and_evaluate_expression_l110_110275

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.sqrt 3 + 1) :
  (1 - 1 / m) / ((m ^ 2 - 2 * m + 1) / m) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l110_110275


namespace area_triangle_AMB_l110_110120

def parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

def point_A : ℝ × ℝ := (0, parabola 0)

def rotated_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 2

def point_B : ℝ × ℝ := (0, rotated_parabola 0)

def vertex_M : ℝ × ℝ := (-1, 2)

def area_of_triangle (A B M : ℝ × ℝ) : ℝ :=
  0.5 * (A.2 - M.2) * (M.1 - B.1)

theorem area_triangle_AMB : area_of_triangle point_A point_B vertex_M = 1 :=
  sorry

end area_triangle_AMB_l110_110120


namespace regular_decagon_interior_angle_l110_110722

-- Define the number of sides in a regular decagon
def n : ℕ := 10

-- Define the formula for the sum of the interior angles of an n-sided polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the measure of one interior angle of a regular decagon
def one_interior_angle_of_regular_polygon (sum_of_angles : ℕ) (n : ℕ) : ℕ :=
  sum_of_angles / n

-- Prove that the measure of one interior angle of a regular decagon is 144 degrees
theorem regular_decagon_interior_angle : one_interior_angle_of_regular_polygon (sum_of_interior_angles 10) 10 = 144 := by
  sorry

end regular_decagon_interior_angle_l110_110722


namespace function_b_is_even_and_monotonically_increasing_l110_110964

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ∈ s → b ∈ s → a < b → f a ≤ f b

def f (x : ℝ) : ℝ := abs x + 1

theorem function_b_is_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on f (Set.Ioi 0) :=
by
  sorry

end function_b_is_even_and_monotonically_increasing_l110_110964


namespace number_of_beavers_in_second_group_l110_110137

-- Define the number of beavers and the time for the first group
def numBeavers1 := 20
def time1 := 3

-- Define the time for the second group
def time2 := 5

-- Define the total work done (which is constant)
def work := numBeavers1 * time1

-- Define the number of beavers in the second group
def numBeavers2 := 12

-- Theorem stating the mathematical equivalence
theorem number_of_beavers_in_second_group : numBeavers2 * time2 = work :=
by
  -- remaining proof steps would go here
  sorry

end number_of_beavers_in_second_group_l110_110137


namespace giraffe_statue_price_l110_110845

variable (G : ℕ) -- Price of a giraffe statue in dollars

-- Conditions as definitions in Lean 4
def giraffe_jade_usage := 120 -- grams
def elephant_jade_usage := 2 * giraffe_jade_usage -- 240 grams
def elephant_price := 350 -- dollars
def total_jade := 1920 -- grams
def additional_profit_with_elephants := 400 -- dollars

-- Prove that the price of a giraffe statue is $150
theorem giraffe_statue_price : 
  16 * G + additional_profit_with_elephants = 8 * elephant_price → G = 150 :=
by
  intro h
  sorry

end giraffe_statue_price_l110_110845


namespace quadratic_factors_l110_110001

theorem quadratic_factors {a b c : ℝ} (h : a = 1) (h_roots : (1:ℝ) + 2 = b ∧ (-1:ℝ) * 2 = c) :
  (x^2 - b * x + c) = (x - 1) * (x - 2) := by
  sorry

end quadratic_factors_l110_110001


namespace quadratic_has_one_real_solution_l110_110751

theorem quadratic_has_one_real_solution (k : ℝ) (hk : (x + 5) * (x + 2) = k + 3 * x) : k = 6 → ∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x :=
by
  sorry

end quadratic_has_one_real_solution_l110_110751


namespace Shaina_chocolate_l110_110813

-- Definitions based on the conditions
def total_chocolate : ℚ := 72 / 7
def number_of_piles : ℚ := 6
def weight_per_pile : ℚ := total_chocolate / number_of_piles
def piles_given_to_Shaina : ℚ := 2

-- Theorem stating the problem's correct answer
theorem Shaina_chocolate :
  piles_given_to_Shaina * weight_per_pile = 24 / 7 :=
by
  sorry

end Shaina_chocolate_l110_110813


namespace sufficient_not_necessary_condition_l110_110606

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, (x^2 - 2 * x < 0 → 0 < x ∧ x < 4)) ∧ (∃ x : ℝ, (0 < x ∧ x < 4) ∧ ¬ (x^2 - 2 * x < 0)) :=
by
  sorry

end sufficient_not_necessary_condition_l110_110606


namespace set_inter_and_complement_l110_110643

def U : Set ℕ := {2, 3, 4, 5, 6, 7}
def A : Set ℕ := {4, 5, 7}
def B : Set ℕ := {4, 6}

theorem set_inter_and_complement :
  A ∩ (U \ B) = {5, 7} := by
  sorry

end set_inter_and_complement_l110_110643


namespace find_n_l110_110785

theorem find_n (n k : ℕ) (b : ℝ) (h_n2 : n ≥ 2) (h_ab : b ≠ 0 ∧ k > 0) (h_a_eq : ∀ (a : ℝ), a = k^2 * b) :
  (∀ (S : ℕ → ℝ → ℝ), S 1 b + S 2 b = 0) →
  n = 2 * k + 1 := 
sorry

end find_n_l110_110785


namespace deepak_present_age_l110_110169

theorem deepak_present_age (x : ℕ) (h : 4 * x + 6 = 26) : 3 * x = 15 := 
by 
  sorry

end deepak_present_age_l110_110169


namespace white_pieces_remaining_after_process_l110_110482

-- Definition to describe the removal process
def remove_every_second (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else (n + 1) / 2

-- Recursive function to model the process of removing pieces
def remaining_white_pieces (initial_white : ℕ) (rounds : ℕ) : ℕ :=
  match rounds with
  | 0     => initial_white
  | n + 1 => remaining_white_pieces (remove_every_second initial_white) n

-- Main theorem statement
theorem white_pieces_remaining_after_process :
  remaining_white_pieces 1990 4 = 124 :=
by
  sorry

end white_pieces_remaining_after_process_l110_110482


namespace hyperbola_foci_eccentricity_l110_110280

-- Definitions and conditions
def hyperbola_eq := (x y : ℝ) → (x^2 / 4) - (y^2 / 12) = 1

-- Proof goals: Coordinates of the foci and eccentricity
theorem hyperbola_foci_eccentricity (x y : ℝ) : 
  (∃ c : ℝ, (x^2 / 4) - (y^2 / 12) = 1 ∧ (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)) ∧ 
  (∃ e : ℝ, e = 2) :=
sorry

end hyperbola_foci_eccentricity_l110_110280


namespace outdoor_section_area_l110_110672

theorem outdoor_section_area :
  ∀ (width length : ℕ), width = 4 → length = 6 → (width * length = 24) :=
by
  sorry

end outdoor_section_area_l110_110672


namespace smallest_expression_value_l110_110798

theorem smallest_expression_value (a b c : ℝ) (h₁ : b > c) (h₂ : c > 0) (h₃ : a ≠ 0) :
  (2 * a + b) ^ 2 + (b - c) ^ 2 + (c - 2 * a) ^ 2 ≥ (4 / 3) * b ^ 2 :=
by
  sorry

end smallest_expression_value_l110_110798


namespace current_rate_l110_110139

variable (c : ℝ)

def still_water_speed : ℝ := 3.6

axiom rowing_time_ratio (c : ℝ) : (2 : ℝ) * (still_water_speed - c) = still_water_speed + c

theorem current_rate : c = 1.2 :=
by
  sorry

end current_rate_l110_110139


namespace price_per_exercise_book_is_correct_l110_110518

-- Define variables and conditions from the problem statement
variables (xM xH booksM booksH pricePerBook : ℝ)
variables (xH_gives_xM : ℝ)

-- Conditions set up from the problem statement
axiom pooled_money : xM = xH
axiom books_ming : booksM = 8
axiom books_hong : booksH = 12
axiom amount_given : xH_gives_xM = 1.1

-- Problem statement to prove
theorem price_per_exercise_book_is_correct :
  (8 + 12) * pricePerBook / 2 = 1.1 → pricePerBook = 0.55 := by
  sorry

end price_per_exercise_book_is_correct_l110_110518


namespace find_x_l110_110788

theorem find_x (x y z : ℕ) (h1 : x = y / 2) (h2 : y = z / 3) (h3 : z = 90) : x = 15 :=
by
  sorry

end find_x_l110_110788


namespace polynomial_no_in_interval_l110_110485

theorem polynomial_no_in_interval (P : Polynomial ℤ) (x₁ x₂ x₃ x₄ x₅ : ℤ) :
  (-- Conditions
  P.eval x₁ = 5 ∧ P.eval x₂ = 5 ∧ P.eval x₃ = 5 ∧ P.eval x₄ = 5 ∧ P.eval x₅ = 5 ∧
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
  x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
  x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
  x₄ ≠ x₅)
  -- No x such that -6 <= P(x) <= 4 or 6 <= P(x) <= 16
  → (∀ x : ℤ, ¬(-6 ≤ P.eval x ∧ P.eval x ≤ 4) ∧ ¬(6 ≤ P.eval x ∧ P.eval x ≤ 16)) :=
by
  intro h
  sorry

end polynomial_no_in_interval_l110_110485


namespace right_triangle_circle_area_l110_110830

/-- 
Given a right triangle ABC with legs AB = 6 cm and BC = 8 cm,
E is the midpoint of AB and D is the midpoint of AC.
A circle passes through points E and D and touches the hypotenuse AC.
Prove that the area of this circle is 100 * pi / 9 cm^2.
-/
theorem right_triangle_circle_area :
  ∃ (r : ℝ), 
  let AB := 6
  let BC := 8
  let AC := Real.sqrt (AB^2 + BC^2)
  let E := (AB / 2)
  let D := (AC / 2)
  let radius := (AC * (BC / 2) / AB)
  r = radius * radius * Real.pi ∧
  r = (100 * Real.pi / 9) := sorry

end right_triangle_circle_area_l110_110830


namespace minimum_steps_to_catch_thief_l110_110649

-- Definitions of positions A, B, C, D, etc., along the board
-- Assuming the positions and movement rules are predefined somewhere in the environment.
-- For a simple abstract model, we assume the following:
-- The positions are nodes in a graph, and each move is one step along the edges of this graph.

def Position : Type := String -- This can be refined to reflect the actual chessboard structure.
def neighbor (p1 p2 : Position) : Prop := sorry -- Predicate defining that p1 and p2 are neighbors.

-- Positions are predefined for simplicity.
def A : Position := "A"
def B : Position := "B"
def C : Position := "C"
def D : Position := "D"
def F : Position := "F"

-- Condition: policeman and thief take turns moving, starting with the policeman.
-- Initial positions of the policeman and the thief.
def policemanStart : Position := A
def thiefStart : Position := B

-- Statement: Prove that the policeman can catch the thief in a minimum of 4 moves.
theorem minimum_steps_to_catch_thief (policeman thief : Position) (turns : ℕ) :
  policeman = policemanStart →
  thief = thiefStart →
  (∀ t < turns, (neighbor policeman thief)) →
  (turns = 4) :=
sorry

end minimum_steps_to_catch_thief_l110_110649


namespace equation_one_solution_equation_two_no_solution_l110_110844

-- Problem 1
theorem equation_one_solution (x : ℝ) (h : x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) : x = 0 := 
by 
  sorry

-- Problem 2
theorem equation_two_no_solution (x : ℝ) (h : 2 * x + 9 / (3 * x - 9) = (4 * x - 7) / (x - 3) + 2) : False := 
by 
  sorry

end equation_one_solution_equation_two_no_solution_l110_110844


namespace unattainable_y_l110_110947

theorem unattainable_y (x : ℚ) (hx : x ≠ -4 / 3) : 
    ∀ y : ℚ, (y = (2 - x) / (3 * x + 4)) → y ≠ -1 / 3 :=
sorry

end unattainable_y_l110_110947


namespace quadrilateral_AB_length_l110_110407

/-- Let ABCD be a quadrilateral with BC = CD = DA = 1, ∠DAB = 135°, and ∠ABC = 75°. 
    Prove that AB = (√6 - √2) / 2.
-/
theorem quadrilateral_AB_length (BC CD DA : ℝ) (angle_DAB angle_ABC : ℝ) (h1 : BC = 1)
    (h2 : CD = 1) (h3 : DA = 1) (h4 : angle_DAB = 135) (h5 : angle_ABC = 75) :
    AB = (Real.sqrt 6 - Real.sqrt 2) / 2 := by
    sorry

end quadrilateral_AB_length_l110_110407


namespace radius_increase_area_triple_l110_110409

theorem radius_increase_area_triple (r m : ℝ) (h : π * (r + m)^2 = 3 * π * r^2) : 
  r = (m * (Real.sqrt 3 - 1)) / 2 := 
sorry

end radius_increase_area_triple_l110_110409


namespace total_matches_played_l110_110362

def home_team_wins := 3
def home_team_draws := 4
def home_team_losses := 0
def rival_team_wins := 2 * home_team_wins
def rival_team_draws := home_team_draws
def rival_team_losses := 0

theorem total_matches_played :
  home_team_wins + home_team_draws + home_team_losses + rival_team_wins + rival_team_draws + rival_team_losses = 17 :=
by
  sorry

end total_matches_played_l110_110362


namespace sqrt_div_l110_110417

theorem sqrt_div (x: ℕ) (h1: Nat.sqrt 144 * Nat.sqrt 144 = 144) (h2: 144 = 12 * 12) (h3: 2 * x = 12) : x = 6 :=
sorry

end sqrt_div_l110_110417


namespace kevin_leaves_l110_110628

theorem kevin_leaves (n : ℕ) (h : n > 1) : ∃ k : ℕ, n = k^3 ∧ n^2 = k^6 ∧ n = 8 := by
  sorry

end kevin_leaves_l110_110628


namespace unique_zero_point_of_quadratic_l110_110574

theorem unique_zero_point_of_quadratic (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - x - 1 = 0 → x = -1)) ↔ (a = 0 ∨ a = -1 / 4) :=
by
  sorry

end unique_zero_point_of_quadratic_l110_110574


namespace tony_age_in_6_years_l110_110611

theorem tony_age_in_6_years (jacob_age : ℕ) (tony_age : ℕ) (h : jacob_age = 24) (h_half : tony_age = jacob_age / 2) : (tony_age + 6) = 18 :=
by
  sorry

end tony_age_in_6_years_l110_110611


namespace find_b_given_a_l110_110699

-- Definitions based on the conditions
def varies_inversely (a b : ℝ) (k : ℝ) : Prop := a * b = k
def k_value : ℝ := 400

-- The proof statement
theorem find_b_given_a (a b : ℝ) (h1 : varies_inversely 800 0.5 k_value) (h2 : a = 3200) : b = 0.125 :=
by
  -- skipped proof
  sorry

end find_b_given_a_l110_110699


namespace trig_sum_roots_l110_110622

theorem trig_sum_roots {θ a : Real} (hroots : ∀ x, x^2 - a * x + a = 0 → x = Real.sin θ ∨ x = Real.cos θ) :
  Real.cos (θ - 3 * Real.pi / 2) + Real.sin (3 * Real.pi / 2 + θ) = Real.sqrt 2 - 1 :=
by
  sorry

end trig_sum_roots_l110_110622


namespace problem_statement_l110_110254

variable (x y : ℝ)
variable (h_cond1 : 1 / x + 1 / y = 4)
variable (h_cond2 : x * y - x - y = -7)

theorem problem_statement (h_cond1 : 1 / x + 1 / y = 4) (h_cond2 : x * y - x - y = -7) : 
  x^2 * y + x * y^2 = 196 / 9 := 
sorry

end problem_statement_l110_110254


namespace calculation_result_l110_110599

theorem calculation_result :
  3 * 3^3 + 4^7 / 4^5 = 97 :=
by
  sorry

end calculation_result_l110_110599


namespace consecutive_odd_numbers_l110_110301

/- 
  Out of some consecutive odd numbers, 9 times the first number 
  is equal to the addition of twice the third number and adding 9 
  to twice the second. Let x be the first number, then we aim to prove that 
  9 * x = 2 * (x + 4) + 2 * (x + 2) + 9 ⟹ x = 21 / 5
-/

theorem consecutive_odd_numbers (x : ℚ) (h : 9 * x = 2 * (x + 4) + 2 * (x + 2) + 9) : x = 21 / 5 :=
sorry

end consecutive_odd_numbers_l110_110301


namespace edward_spent_on_books_l110_110071

def money_spent_on_books (initial_amount spent_on_pens amount_left : ℕ) : ℕ :=
  initial_amount - amount_left - spent_on_pens

theorem edward_spent_on_books :
  ∃ (x : ℕ), x = 6 → 
  ∀ {initial_amount spent_on_pens amount_left : ℕ},
    initial_amount = 41 →
    spent_on_pens = 16 →
    amount_left = 19 →
    x = money_spent_on_books initial_amount spent_on_pens amount_left :=
by
  sorry

end edward_spent_on_books_l110_110071


namespace f_D_not_mapping_to_B_l110_110877

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B := {y : ℝ | 1 ≤ y ∧ y <= 4}
def f_D (x : ℝ) := 4 - x^2

theorem f_D_not_mapping_to_B : ¬ (∀ x ∈ A, f_D x ∈ B) := sorry

end f_D_not_mapping_to_B_l110_110877


namespace split_into_similar_heaps_l110_110089

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end split_into_similar_heaps_l110_110089


namespace intersection_M_N_l110_110868

def M : Set ℤ := {0}
def N : Set ℤ := {x | -1 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {0} :=
by
  sorry

end intersection_M_N_l110_110868


namespace correct_statements_about_opposite_numbers_l110_110352

/-- Definition of opposite numbers: two numbers are opposite if one is the negative of the other --/
def is_opposite (a b : ℝ) : Prop := a = -b

theorem correct_statements_about_opposite_numbers (a b : ℝ) :
  (is_opposite a b ↔ a + b = 0) ∧
  (a + b = 0 ↔ is_opposite a b) ∧
  ((is_opposite a b ∧ a ≠ 0 ∧ b ≠ 0) ↔ (a / b = -1)) ∧
  ((a / b = -1 ∧ b ≠ 0) ↔ is_opposite a b) :=
by {
  sorry -- Proof is omitted
}

end correct_statements_about_opposite_numbers_l110_110352


namespace total_bins_sum_l110_110144

def total_bins_soup : ℝ := 0.2
def total_bins_vegetables : ℝ := 0.35
def total_bins_fruits : ℝ := 0.15
def total_bins_pasta : ℝ := 0.55
def total_bins_canned_meats : ℝ := 0.275
def total_bins_beans : ℝ := 0.175

theorem total_bins_sum :
  total_bins_soup + total_bins_vegetables + total_bins_fruits + total_bins_pasta + total_bins_canned_meats + total_bins_beans = 1.7 :=
by
  sorry

end total_bins_sum_l110_110144


namespace string_length_is_correct_l110_110347

noncomputable def calculate_string_length (circumference height : ℝ) (loops : ℕ) : ℝ :=
  let vertical_distance_per_loop := height / loops
  let hypotenuse_length := Real.sqrt ((circumference ^ 2) + (vertical_distance_per_loop ^ 2))
  loops * hypotenuse_length

theorem string_length_is_correct : calculate_string_length 6 16 5 = 34 := 
  sorry

end string_length_is_correct_l110_110347


namespace gcd_m_pow_5_plus_125_m_plus_3_l110_110761

theorem gcd_m_pow_5_plus_125_m_plus_3 (m : ℕ) (h: m > 16) : 
  Nat.gcd (m^5 + 125) (m + 3) = Nat.gcd 27 (m + 3) :=
by
  -- Proof will be provided here
  sorry

end gcd_m_pow_5_plus_125_m_plus_3_l110_110761


namespace problem_equivalent_l110_110432

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) / Real.log 4 - 1

theorem problem_equivalent : 
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = Real.log (x + 2) / Real.log 4 - 1) →
  {x : ℝ | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  intro h_even h_def
  sorry

end problem_equivalent_l110_110432


namespace revenue_fell_by_percentage_l110_110506

theorem revenue_fell_by_percentage :
  let old_revenue : ℝ := 69.0
  let new_revenue : ℝ := 52.0
  let percentage_decrease : ℝ := ((old_revenue - new_revenue) / old_revenue) * 100
  abs (percentage_decrease - 24.64) < 1e-2 :=
by
  sorry

end revenue_fell_by_percentage_l110_110506


namespace number_of_tiles_per_row_l110_110854

-- Define the conditions 
def area_floor_sq_ft : ℕ := 400
def tile_side_inch : ℕ := 8
def feet_to_inch (f : ℕ) := 12 * f

-- Define the theorem using the conditions
theorem number_of_tiles_per_row (h : Nat.sqrt area_floor_sq_ft = 20) :
  (feet_to_inch 20) / tile_side_inch = 30 :=
by
  sorry

end number_of_tiles_per_row_l110_110854


namespace factorize_expression_l110_110444

variable {a x y : ℝ}

theorem factorize_expression : (a * x^2 + 2 * a * x * y + a * y^2) = a * (x + y)^2 := by
  sorry

end factorize_expression_l110_110444


namespace pages_revised_twice_theorem_l110_110723

noncomputable def pages_revised_twice (total_pages : ℕ) (cost_per_page : ℕ) (revision_cost_per_page : ℕ) 
                                      (pages_revised_once : ℕ) (total_cost : ℕ) : ℕ :=
  let pages_revised_twice := (total_cost - (total_pages * cost_per_page + pages_revised_once * revision_cost_per_page)) 
                             / (revision_cost_per_page * 2)
  pages_revised_twice

theorem pages_revised_twice_theorem : 
  pages_revised_twice 100 10 5 30 1350 = 20 :=
by
  unfold pages_revised_twice
  norm_num

end pages_revised_twice_theorem_l110_110723


namespace bowling_ball_surface_area_l110_110900

theorem bowling_ball_surface_area (d : ℝ) (hd : d = 9) : 
  4 * Real.pi * (d / 2)^2 = 81 * Real.pi :=
by
  -- proof goes here
  sorry

end bowling_ball_surface_area_l110_110900


namespace interest_rate_per_annum_l110_110968

-- Definitions for the given conditions
def SI : ℝ := 4016.25
def P : ℝ := 44625
def T : ℝ := 9

-- The interest rate R must be 1 according to the conditions
theorem interest_rate_per_annum : (SI * 100) / (P * T) = 1 := by
  sorry

end interest_rate_per_annum_l110_110968


namespace find_r_l110_110915

theorem find_r (r : ℝ) (h1 : ∃ s : ℝ, 8 * x^3 - 4 * x^2 - 42 * x + 45 = 8 * (x - r)^2 * (x - s)) :
  r = 3 / 2 :=
by
  sorry

end find_r_l110_110915


namespace hyperbola_equation_l110_110186

open Real

theorem hyperbola_equation (e e' : ℝ) (h₁ : 2 * x^2 + y^2 = 2) (h₂ : e * e' = 1) :
  y^2 - x^2 = 2 :=
sorry

end hyperbola_equation_l110_110186


namespace largest_integer_solution_l110_110289

theorem largest_integer_solution : ∃ x : ℤ, (x ≤ 10) ∧ (∀ y : ℤ, (y > 10 → (y / 4 + 5 / 6 < 7 / 2) = false)) :=
sorry

end largest_integer_solution_l110_110289


namespace problem_l110_110587

def f : ℕ → ℕ → ℕ := sorry

theorem problem (m n : ℕ) (h1 : m ≥ 1) (h2 : n ≥ 1) :
  2 * f m n = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1) ∧
  (f m 0 = 0) ∧ (f 0 n = 0) → f m n = m * n :=
by sorry

end problem_l110_110587


namespace max_n_arithmetic_seq_sum_neg_l110_110468

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + ((n - 1) * d)

-- Define the terms of the sequence
def a₃ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 3
def a₆ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 6
def a₇ (a₁ : ℤ) : ℤ := arithmetic_sequence a₁ 2 7

-- Condition: a₆ is the geometric mean of a₃ and a₇
def geometric_mean_condition (a₁ : ℤ) : Prop :=
  (a₃ a₁) * (a₇ a₁) = (a₆ a₁) * (a₆ a₁)

-- Sum of the first n terms of the arithmetic sequence
def S_n (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) * d) / 2

-- The goal: the maximum value of n for which S_n < 0
theorem max_n_arithmetic_seq_sum_neg : 
  ∃ n : ℕ, ∀ k : ℕ, geometric_mean_condition (-13) →  S_n (-13) 2 k < 0 → n ≤ 13 := 
sorry

end max_n_arithmetic_seq_sum_neg_l110_110468


namespace sum_of_first_9_terms_zero_l110_110225

variable (a_n : ℕ → ℝ) (d a₁ : ℝ)
def arithmetic_seq := ∀ n, a_n n = a₁ + (n - 1) * d

def condition (a_n : ℕ → ℝ) := (a_n 2 + a_n 9 = a_n 6)

theorem sum_of_first_9_terms_zero 
  (h_arith : arithmetic_seq a_n d a₁) 
  (h_cond : condition a_n) : 
  (9 * a₁ + (9 * 8 / 2) * d) = 0 :=
by
  sorry

end sum_of_first_9_terms_zero_l110_110225


namespace toms_profit_l110_110387

noncomputable def cost_of_flour : Int :=
  let flour_needed := 500
  let bag_size := 50
  let bag_cost := 20
  (flour_needed / bag_size) * bag_cost

noncomputable def cost_of_salt : Int :=
  let salt_needed := 10
  let salt_cost_per_pound := (2 / 10)  -- Represent $0.2 as a fraction to maintain precision with integers in Lean
  salt_needed * salt_cost_per_pound

noncomputable def total_expenses : Int :=
  let flour_cost := cost_of_flour
  let salt_cost := cost_of_salt
  let promotion_cost := 1000
  flour_cost + salt_cost + promotion_cost

noncomputable def revenue_from_tickets : Int :=
  let ticket_price := 20
  let tickets_sold := 500
  tickets_sold * ticket_price

noncomputable def profit : Int :=
  revenue_from_tickets - total_expenses

theorem toms_profit : profit = 8798 :=
  by
    sorry

end toms_profit_l110_110387


namespace bill_fine_amount_l110_110545

-- Define the conditions
def ounces_sold : ℕ := 8
def earnings_per_ounce : ℕ := 9
def amount_left : ℕ := 22

-- Calculate the earnings
def earnings : ℕ := ounces_sold * earnings_per_ounce

-- Define the fine as the difference between earnings and amount left
def fine : ℕ := earnings - amount_left

-- The proof problem to solve
theorem bill_fine_amount : fine = 50 :=
by
  -- Statements and calculations would go here
  sorry

end bill_fine_amount_l110_110545


namespace gcd_of_four_sum_1105_l110_110296

theorem gcd_of_four_sum_1105 (a b c d : ℕ) (h_sum : a + b + c + d = 1105)
  (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c) (hd_pos : 0 < d)
  (h_neq_ab : a ≠ b) (h_neq_ac : a ≠ c) (h_neq_ad : a ≠ d)
  (h_neq_bc : b ≠ c) (h_neq_bd : b ≠ d) (h_neq_cd : c ≠ d)
  (h_gcd_ab : gcd a b > 1) (h_gcd_ac : gcd a c > 1) (h_gcd_ad : gcd a d > 1)
  (h_gcd_bc : gcd b c > 1) (h_gcd_bd : gcd b d > 1) (h_gcd_cd : gcd c d > 1) :
  gcd a (gcd b (gcd c d)) = 221 := by
  sorry

end gcd_of_four_sum_1105_l110_110296


namespace sum_of_coefficients_is_2_l110_110735

noncomputable def polynomial_expansion_condition (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :=
  (x^2 + 1) * (x - 2)^9 = a_0 + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + 
                          a_5 * (x - 1)^5 + a_6 * (x - 1)^6 + a_7 * (x - 1)^7 + a_8 * (x - 1)^8 + 
                          a_9 * (x - 1)^9 + a_10 * (x - 1)^10 + a_11 * (x - 1)^11

theorem sum_of_coefficients_is_2 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ) :
  polynomial_expansion_condition 1 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  polynomial_expansion_condition 2 a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 →
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 2 :=
by sorry

end sum_of_coefficients_is_2_l110_110735


namespace summation_eq_16_implies_x_eq_3_over_4_l110_110688

theorem summation_eq_16_implies_x_eq_3_over_4 (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x^n = 16) : x = 3 / 4 :=
sorry

end summation_eq_16_implies_x_eq_3_over_4_l110_110688


namespace cost_price_of_watch_l110_110180

theorem cost_price_of_watch (C SP1 SP2 : ℝ)
    (h1 : SP1 = 0.90 * C)
    (h2 : SP2 = 1.02 * C)
    (h3 : SP2 = SP1 + 140) :
    C = 1166.67 :=
by
  sorry

end cost_price_of_watch_l110_110180


namespace Sarah_skateboard_speed_2160_mph_l110_110150

-- Definitions based on the conditions
def miles_to_inches (miles : ℕ) : ℕ := miles * 63360
def minutes_to_hours (minutes : ℕ) : ℕ := minutes / 60

/-- Pete walks backwards 3 times faster than Susan walks forwards --/
def Susan_walks_forwards_speed (pete_walks_hands_speed : ℕ) : ℕ := pete_walks_hands_speed / 3

/-- Tracy does cartwheels twice as fast as Susan walks forwards --/
def Tracy_cartwheels_speed (susan_walks_forwards_speed : ℕ) : ℕ := susan_walks_forwards_speed * 2

/-- Mike swims 8 times faster than Tracy does cartwheels --/
def Mike_swims_speed (tracy_cartwheels_speed : ℕ) : ℕ := tracy_cartwheels_speed * 8

/-- Pete can walk on his hands at 1/4 the speed Tracy can do cartwheels --/
def Pete_walks_hands_speed : ℕ := 2

/-- Pete rides his bike 5 times faster than Mike swims --/
def Pete_rides_bike_speed (mike_swims_speed : ℕ) : ℕ := mike_swims_speed * 5

/-- Patty can row 3 times faster than Pete walks backwards (in feet per hour) --/
def Patty_rows_speed (pete_walks_backwards_speed : ℕ) : ℕ := pete_walks_backwards_speed * 3

/-- Sarah can skateboard 6 times faster than Patty rows (in miles per minute) --/
def Sarah_skateboards_speed (patty_rows_speed_ft_per_hr : ℕ) : ℕ := (patty_rows_speed_ft_per_hr * 6 * 60) * 63360 * 60

theorem Sarah_skateboard_speed_2160_mph : Sarah_skateboards_speed (Patty_rows_speed (Pete_walks_hands_speed * 3)) = 2160 * 63360 * 60 :=
by
  sorry

end Sarah_skateboard_speed_2160_mph_l110_110150


namespace sum_of_roots_l110_110069

theorem sum_of_roots (x : ℝ) :
  (x^2 = 10 * x - 13) → ∃ s, s = 10 := 
by
  sorry

end sum_of_roots_l110_110069


namespace year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l110_110585

-- Define what it means to be a leap year based on the given conditions.
def is_leap_year (y : ℕ) : Prop :=
  (y % 400 = 0) ∨ (y % 4 = 0 ∧ y % 100 ≠ 0)

-- Define the specific years we are examining.
def year_1800 := 1800
def year_1992 := 1992
def year_1994 := 1994
def year_2040 := 2040

-- Assertions about whether each year is a leap year or a common year
theorem year_1800_is_common : ¬ is_leap_year year_1800 :=
  by sorry

theorem year_1992_is_leap : is_leap_year year_1992 :=
  by sorry

theorem year_1994_is_common : ¬ is_leap_year year_1994 :=
  by sorry

theorem year_2040_is_leap : is_leap_year year_2040 :=
  by sorry

end year_1800_is_common_year_1992_is_leap_year_1994_is_common_year_2040_is_leap_l110_110585


namespace simplify_and_evaluate_l110_110384

theorem simplify_and_evaluate (m : ℝ) (h : m = 1) : 
  (1 - 1 / (m - 2)) / ((m^2 - 6 * m + 9) / (m - 2)) = -1/2 :=
by
  sorry

end simplify_and_evaluate_l110_110384


namespace darius_drive_miles_l110_110572

theorem darius_drive_miles (total_miles : ℕ) (julia_miles : ℕ) (darius_miles : ℕ) 
  (h1 : total_miles = 1677) (h2 : julia_miles = 998) (h3 : total_miles = darius_miles + julia_miles) : 
  darius_miles = 679 :=
by
  sorry

end darius_drive_miles_l110_110572


namespace large_rect_area_is_294_l110_110011

-- Define the dimensions of the smaller rectangles
def shorter_side : ℕ := 7
def longer_side : ℕ := 2 * shorter_side

-- Condition 1: Each smaller rectangle has a shorter side measuring 7 feet
axiom smaller_rect_shorter_side : ∀ (r : ℕ), r = shorter_side → r = 7

-- Condition 4: The longer side of each smaller rectangle is twice the shorter side
axiom smaller_rect_longer_side : ∀ (r : ℕ), r = longer_side → r = 2 * shorter_side

-- Condition 2: Three rectangles are aligned vertically
def vertical_height : ℕ := 3 * shorter_side

-- Condition 3: One rectangle is aligned horizontally adjoining them
def horizontal_length : ℕ := longer_side

-- The dimensions of the larger rectangle EFGH
def large_rect_width : ℕ := vertical_height
def large_rect_length : ℕ := horizontal_length

-- Calculate the area of the larger rectangle EFGH
def large_rect_area : ℕ := large_rect_width * large_rect_length

-- Prove that the area of the large rectangle is 294 square feet
theorem large_rect_area_is_294 : large_rect_area = 294 := by
  sorry

end large_rect_area_is_294_l110_110011


namespace sufficient_but_not_necessary_l110_110816

variable (a : ℝ)

theorem sufficient_but_not_necessary : (a = 1 → |a| = 1) ∧ (|a| = 1 → a = 1 → False) :=
by
  sorry

end sufficient_but_not_necessary_l110_110816


namespace jakes_present_weight_l110_110111

theorem jakes_present_weight:
  ∃ J S : ℕ, J - 15 = 2 * S ∧ J + S = 132 ∧ J = 93 :=
by
  sorry

end jakes_present_weight_l110_110111


namespace intersection_S_T_l110_110136

open Set

def S : Set ℝ := { x | x ≥ 1 }
def T : Set ℝ := { -2, -1, 0, 1, 2 }

theorem intersection_S_T : S ∩ T = { 1, 2 } := by
  sorry

end intersection_S_T_l110_110136


namespace ellipse_x_intercept_l110_110020

theorem ellipse_x_intercept :
  let F_1 := (0,3)
  let F_2 := (4,0)
  let ellipse := { P : ℝ × ℝ | (dist P F_1) + (dist P F_2) = 7 }
  ∃ x : ℝ, x ≠ 0 ∧ (x, 0) ∈ ellipse ∧ x = 56 / 11 :=
by
  sorry

end ellipse_x_intercept_l110_110020


namespace infinite_sequence_no_square_factors_l110_110998

/-
  Prove that there exist infinitely many positive integers \( n_1 < n_2 < \cdots \)
  such that for all \( i \neq j \), \( n_i + n_j \) has no square factors other than 1.
-/

theorem infinite_sequence_no_square_factors :
  ∃ (n : ℕ → ℕ), (∀ (i j : ℕ), i ≠ j → ∀ p : ℕ, p ≠ 1 → p^2 ∣ (n i + n j) → false) ∧
    ∀ k : ℕ, n k < n (k + 1) :=
sorry

end infinite_sequence_no_square_factors_l110_110998


namespace lychee_production_increase_l110_110515

variable (x : ℕ) -- percentage increase as a natural number

def lychee_increase_2006 (x : ℕ) : ℕ :=
  (1 + x)*(1 + x)

theorem lychee_production_increase (x : ℕ) :
  lychee_increase_2006 x = (1 + x) * (1 + x) :=
by
  sorry

end lychee_production_increase_l110_110515


namespace quadratic_difference_l110_110157

theorem quadratic_difference (f : ℝ → ℝ) (hpoly : ∃ c d e : ℤ, ∀ x, f x = c*x^2 + d*x + e) 
(h : f (Real.sqrt 3) - f (Real.sqrt 2) = 4) : 
f (Real.sqrt 10) - f (Real.sqrt 7) = 12 := sorry

end quadratic_difference_l110_110157


namespace common_root_l110_110094

theorem common_root (a b x : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eq1 : x^2 + a * x + b = 0) (eq2 : x^3 + b * x + a = 0) : x = -1 :=
by
  sorry

end common_root_l110_110094


namespace total_value_of_item_l110_110533

theorem total_value_of_item
  (import_tax : ℝ)
  (V : ℝ)
  (h₀ : import_tax = 110.60)
  (h₁ : import_tax = 0.07 * (V - 1000)) :
  V = 2579.43 := 
sorry

end total_value_of_item_l110_110533


namespace statement_A_statement_B_statement_C_statement_D_statement_E_l110_110054

-- Define a statement for each case and prove each one
theorem statement_A (x : ℝ) (h : x ≥ 0) : x^2 ≥ x :=
sorry

theorem statement_B (x : ℝ) (h : x^2 ≥ 0) : abs x ≥ 0 :=
sorry

theorem statement_C (x : ℝ) (h : x^2 ≤ x) : ¬ (x ≤ 1) :=
sorry

theorem statement_D (x : ℝ) (h : x^2 ≥ x) : ¬ (x ≤ 0) :=
sorry

theorem statement_E (x : ℝ) (h : x ≤ -1) : x^2 ≥ abs x :=
sorry

end statement_A_statement_B_statement_C_statement_D_statement_E_l110_110054


namespace logs_quadratic_sum_l110_110152

theorem logs_quadratic_sum (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_roots : ∀ x, 2 * x^2 + 4 * x + 1 = 0 → (x = Real.log a) ∨ (x = Real.log b)) :
  (Real.log a)^2 + Real.log (a^2) + a * b = 1 / Real.exp 2 - 1 / 2 :=
by
  sorry

end logs_quadratic_sum_l110_110152


namespace ticket_cost_proof_l110_110002

def adult_ticket_price : ℕ := 55
def child_ticket_price : ℕ := 28
def senior_ticket_price : ℕ := 42

def num_adult_tickets : ℕ := 4
def num_child_tickets : ℕ := 2
def num_senior_tickets : ℕ := 1

def total_ticket_cost : ℕ :=
  (num_adult_tickets * adult_ticket_price) + (num_child_tickets * child_ticket_price) + (num_senior_tickets * senior_ticket_price)

theorem ticket_cost_proof : total_ticket_cost = 318 := by
  sorry

end ticket_cost_proof_l110_110002


namespace find_fraction_l110_110961

-- Definition of the fractions and the given condition
def certain_fraction : ℚ := 1 / 2
def given_ratio : ℚ := 2 / 6
def target_fraction : ℚ := 1 / 3

-- The proof problem to verify
theorem find_fraction (X : ℚ) : (X / given_ratio) = 1 ↔ X = target_fraction :=
by
  sorry

end find_fraction_l110_110961


namespace find_radius_of_semicircle_l110_110752

-- Definitions for the rectangle and semi-circle
variable (L W : ℝ) -- Length and width of the rectangle
variable (r : ℝ) -- Radius of the semi-circle

-- Conditions given in the problem
def rectangle_perimeter : Prop := 2 * L + 2 * W = 216
def semicircle_diameter_eq_length : Prop := L = 2 * r 
def width_eq_twice_radius : Prop := W = 2 * r

-- Proof statement
theorem find_radius_of_semicircle
  (h_perimeter : rectangle_perimeter L W)
  (h_diameter : semicircle_diameter_eq_length L r)
  (h_width : width_eq_twice_radius W r) :
  r = 27 := by
  sorry

end find_radius_of_semicircle_l110_110752


namespace nicholas_bottle_caps_l110_110310

theorem nicholas_bottle_caps (N : ℕ) (h : N + 85 = 93) : N = 8 :=
by
  sorry

end nicholas_bottle_caps_l110_110310


namespace cookies_in_jar_l110_110511

noncomputable def C : ℕ := sorry

theorem cookies_in_jar (h : C - 1 = (C + 5) / 2) : C = 7 := by
  sorry

end cookies_in_jar_l110_110511


namespace candy_left_l110_110457

variable (x : ℕ)

theorem candy_left (x : ℕ) : x - (18 + 7) = x - 25 :=
by sorry

end candy_left_l110_110457


namespace fiona_prob_reaches_12_l110_110828

/-- Lily pads are numbered from 0 to 15 -/
def is_valid_pad (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 15

/-- Predators are on lily pads 4 and 7 -/
def predator (n : ℕ) : Prop := n = 4 ∨ n = 7

/-- Fiona the frog's probability to hop to the next pad -/
def hop : ℚ := 1 / 2

/-- Fiona the frog's probability to jump 2 pads -/
def jump_two : ℚ := 1 / 2

/-- Probability that Fiona reaches pad 12 without landing on pads 4 or 7 is 1/32 -/
theorem fiona_prob_reaches_12 :
  ∀ p : ℕ, 
    (is_valid_pad p ∧ ¬ predator p ∧ (p = 12) ∧ 
    ((∀ k : ℕ, is_valid_pad k → ¬ predator k → k ≤ 3 → (hop ^ k) = 1 / 2) ∧
    hop * hop = 1 / 4 ∧ hop * jump_two = 1 / 8 ∧
    (jump_two * (hop * hop + jump_two)) = 1 / 4 → hop * 1 / 4 = 1 / 32)) := 
by intros; sorry

end fiona_prob_reaches_12_l110_110828


namespace points_on_line_y1_gt_y2_l110_110287

theorem points_on_line_y1_gt_y2 (y1 y2 : ℝ) : 
    (∀ x y, y = -x + 3 → 
    ((x = -4 → y = y1) ∧ (x = 2 → y = y2))) → 
    y1 > y2 :=
by
  sorry

end points_on_line_y1_gt_y2_l110_110287


namespace max_area_of_garden_l110_110197

theorem max_area_of_garden
  (w : ℕ) (l : ℕ)
  (h1 : l = 2 * w)
  (h2 : l + 2 * w = 480) : l * w = 28800 :=
sorry

end max_area_of_garden_l110_110197


namespace tan_double_angle_l110_110958

theorem tan_double_angle (θ : ℝ) (h1 : θ = Real.arctan (-2)) : Real.tan (2 * θ) = 4 / 3 :=
by
  sorry

end tan_double_angle_l110_110958


namespace quadratic_nonnegative_l110_110112

theorem quadratic_nonnegative (x y : ℝ) : x^2 + x * y + y^2 ≥ 0 :=
by sorry

end quadratic_nonnegative_l110_110112


namespace abs_inequality_l110_110517

variables (a b c : ℝ)

theorem abs_inequality (h : |a - c| < |b|) : |a| < |b| + |c| :=
sorry

end abs_inequality_l110_110517


namespace exists_small_area_triangle_l110_110850

def lattice_point (x y : ℤ) : Prop := |x| ≤ 2 ∧ |y| ≤ 2

def no_three_collinear (points : List (ℤ × ℤ)) : Prop :=
∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
(p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) →
¬ (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2) = 0)

noncomputable def triangle_area (p1 p2 p3 : ℤ × ℤ) : ℚ :=
(1 / 2 : ℚ) * |(p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))|

theorem exists_small_area_triangle {points : List (ℤ × ℤ)}
  (h1 : points.length = 6)
  (h2 : ∀ (p : ℤ × ℤ), p ∈ points → lattice_point p.1 p.2)
  (h3 : no_three_collinear points) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  (p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3) ∧ 
  triangle_area p1 p2 p3 ≤ 2 := 
sorry

end exists_small_area_triangle_l110_110850


namespace round_robin_teams_l110_110883

theorem round_robin_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end round_robin_teams_l110_110883


namespace max_marks_eq_300_l110_110278

-- Problem Statement in Lean 4

theorem max_marks_eq_300 (m_score p_score c_score : ℝ) 
    (m_percent p_percent c_percent : ℝ)
    (h1 : m_score = 285) (h2 : m_percent = 95) 
    (h3 : p_score = 270) (h4 : p_percent = 90) 
    (h5 : c_score = 255) (h6 : c_percent = 85) :
    (m_score / (m_percent / 100) = 300) ∧ 
    (p_score / (p_percent / 100) = 300) ∧ 
    (c_score / (c_percent / 100) = 300) :=
by
  sorry

end max_marks_eq_300_l110_110278


namespace identify_faulty_key_l110_110570

variable (digits : Finset ℕ)
variable (faulty : ℕ → Bool)

-- Conditions described in the problem statement
variable (attempted_sequence : List ℕ) (registered_sequence : List ℕ)
variable (sequence_length : Nat := 10)
variable (registered_count : Nat := 7)
variable (faulty_press_threshold : Nat := 5)

-- Let attempted_sequence be the sequence typed out and registered_sequence be what was actually registered.

theorem identify_faulty_key (h_len_attempted : attempted_sequence.length = sequence_length)
                            (h_len_registered : registered_sequence.length = registered_count)
                            (h_frequent_digits : ∃ d1 d2, d1 ≠ d2 ∧
                                                        attempted_sequence.count d1 ≥ 2 ∧
                                                        attempted_sequence.count d2 ≥ 2 ∧
                                                        (attempted_sequence.count d1 - registered_sequence.count d1 ≥ 1) ∧
                                                        (attempted_sequence.count d2 - registered_sequence.count d2 ≥ 1)) :
  ∃ d, faulty d ∧ (d = 7 ∨ d = 9) :=
sorry

end identify_faulty_key_l110_110570


namespace set_intersection_complement_eq_l110_110681

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

theorem set_intersection_complement_eq {U : Set ℕ} {M : Set ℕ} {N : Set ℕ}
    (hU : U = {1, 2, 3, 4, 5}) (hM : M = {1, 2, 3}) (hN : N = {3, 4, 5}) :
    (U \ M) ∩ N = {4, 5} :=
by
  sorry

end set_intersection_complement_eq_l110_110681


namespace remainder_of_first_105_sum_div_5280_l110_110326

theorem remainder_of_first_105_sum_div_5280:
  let n := 105
  let d := 5280
  let sum := n * (n + 1) / 2
  sum % d = 285 := by
  sorry

end remainder_of_first_105_sum_div_5280_l110_110326


namespace purely_imaginary_condition_l110_110660

theorem purely_imaginary_condition (x : ℝ) :
  (z : ℂ) → (z = (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I) →
  (x = 1 ↔ (∃ y : ℂ, z = y * Complex.I)) :=
by
  sorry

end purely_imaginary_condition_l110_110660


namespace inequality_holds_for_all_x_l110_110603

theorem inequality_holds_for_all_x (a : ℝ) (h : -1 < a ∧ a < 2) :
  ∀ x : ℝ, -3 < (x^2 + a * x - 2) / (x^2 - x + 1) ∧ (x^2 + a * x - 2) / (x^2 - x + 1) < 2 :=
by
  intro x
  sorry

end inequality_holds_for_all_x_l110_110603


namespace james_daily_soda_consumption_l110_110749

theorem james_daily_soda_consumption
  (N_p : ℕ) -- number of packs
  (S_p : ℕ) -- sodas per pack
  (S_i : ℕ) -- initial sodas
  (D : ℕ)  -- days in a week
  (h1 : N_p = 5)
  (h2 : S_p = 12)
  (h3 : S_i = 10)
  (h4 : D = 7) : 
  (N_p * S_p + S_i) / D = 10 := 
by 
  sorry

end james_daily_soda_consumption_l110_110749


namespace find_inverse_l110_110105

theorem find_inverse :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 3 * x ^ 3 + 9) → (f⁻¹ 90 = 3) :=
by
  intros f h
  sorry

end find_inverse_l110_110105


namespace fraction_problem_l110_110266

-- Define the fractions involved in the problem
def frac1 := 18 / 45
def frac2 := 3 / 8
def frac3 := 1 / 9

-- Define the expected result
def expected_result := 49 / 360

-- The proof statement
theorem fraction_problem : frac1 - frac2 + frac3 = expected_result := by
  sorry

end fraction_problem_l110_110266


namespace fibonacci_inequality_l110_110491

def Fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | 2     => 2
  | n + 2 => Fibonacci (n + 1) + Fibonacci n

theorem fibonacci_inequality (n : ℕ) (h : n > 0) : 
  Real.sqrt (Fibonacci (n+1)) > 1 + 1 / Real.sqrt (Fibonacci n) := 
sorry

end fibonacci_inequality_l110_110491


namespace unique_triple_solution_l110_110902

theorem unique_triple_solution (x y z : ℝ) :
  x = y^3 + y - 8 ∧ y = z^3 + z - 8 ∧ z = x^3 + x - 8 → (x, y, z) = (2, 2, 2) :=
by
  sorry

end unique_triple_solution_l110_110902


namespace shorter_leg_right_triangle_l110_110272

theorem shorter_leg_right_triangle (a b c : ℕ) (h0 : a^2 + b^2 = c^2) (h1 : c = 39) (h2 : a < b) : a = 15 :=
by {
  sorry
}

end shorter_leg_right_triangle_l110_110272


namespace spider_legs_is_multiple_of_human_legs_l110_110052

def human_legs : ℕ := 2
def spider_legs : ℕ := 8

theorem spider_legs_is_multiple_of_human_legs :
  spider_legs = 4 * human_legs :=
by 
  sorry

end spider_legs_is_multiple_of_human_legs_l110_110052


namespace joan_seashells_count_l110_110571

variable (total_seashells_given_to_sam : ℕ) (seashells_left_with_joan : ℕ)

theorem joan_seashells_count
  (h_given : total_seashells_given_to_sam = 43)
  (h_left : seashells_left_with_joan = 27) :
  total_seashells_given_to_sam + seashells_left_with_joan = 70 :=
sorry

end joan_seashells_count_l110_110571


namespace non_working_games_count_l110_110762

-- Definitions based on conditions
def totalGames : ℕ := 16
def pricePerGame : ℕ := 7
def totalEarnings : ℕ := 56

-- Statement to prove
theorem non_working_games_count : 
  totalGames - (totalEarnings / pricePerGame) = 8 :=
by
  sorry

end non_working_games_count_l110_110762


namespace molecular_weight_of_one_mole_l110_110869

-- Definitions derived from the conditions in the problem:

def molecular_weight_nine_moles (w : ℕ) : ℕ :=
  2664

def molecular_weight_one_mole (w : ℕ) : ℕ :=
  w / 9

-- The theorem to prove, based on the above definitions and conditions:
theorem molecular_weight_of_one_mole (w : ℕ) (hw : molecular_weight_nine_moles w = 2664) :
  molecular_weight_one_mole w = 296 :=
sorry

end molecular_weight_of_one_mole_l110_110869


namespace abc_proof_l110_110914

noncomputable def abc_value (a b c : ℝ) : ℝ :=
  a * b * c

theorem abc_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b = 24 * (3 ^ (1 / 3)))
  (h5 : a * c = 40 * (3 ^ (1 / 3)))
  (h6 : b * c = 16 * (3 ^ (1 / 3))) : 
  abc_value a b c = 96 * (15 ^ (1 / 2)) :=
sorry

end abc_proof_l110_110914


namespace angle_B_in_geometric_progression_l110_110665

theorem angle_B_in_geometric_progression 
  {A B C a b c : ℝ} 
  (hSum : A + B + C = Real.pi)
  (hGeo : A = B / 2)
  (hGeo2 : C = 2 * B)
  (hSide : b^2 - a^2 = a * c)
  : B = 2 * Real.pi / 7 := 
by
  sorry

end angle_B_in_geometric_progression_l110_110665


namespace problem_equivalent_l110_110959

theorem problem_equivalent (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a + b = 6) (h₃ : a * (a - 6) = x) (h₄ : b * (b - 6) = x) : 
  x = -9 :=
by
  sorry

end problem_equivalent_l110_110959


namespace mixed_number_sum_l110_110736

theorem mixed_number_sum : 
  (4/5 + 9 * 4/5 + 99 * 4/5 + 999 * 4/5 + 9999 * 4/5 + 1 = 11111) := by
  sorry

end mixed_number_sum_l110_110736


namespace toy_train_produces_5_consecutive_same_tune_l110_110619

noncomputable def probability_same_tune (plays : ℕ) (p : ℚ) (tunes : ℕ) : ℚ :=
  p ^ plays

theorem toy_train_produces_5_consecutive_same_tune :
  probability_same_tune 5 (1/3) 3 = 1/243 :=
by
  sorry

end toy_train_produces_5_consecutive_same_tune_l110_110619


namespace length_of_bridge_l110_110293

theorem length_of_bridge 
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (time_to_pass_bridge : ℝ) 
  (train_length_eq : train_length = 400)
  (train_speed_kmh_eq : train_speed_kmh = 60) 
  (time_to_pass_bridge_eq : time_to_pass_bridge = 72)
  : ∃ (bridge_length : ℝ), bridge_length = 800.24 := 
by
  sorry

end length_of_bridge_l110_110293


namespace similarity_coefficient_l110_110426

theorem similarity_coefficient (α : ℝ) :
  (2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)))
  = 2 * Real.sin (α / 2) * (1 - Real.sin (α / 2)) :=
sorry

end similarity_coefficient_l110_110426


namespace similar_triangle_shortest_side_l110_110168

theorem similar_triangle_shortest_side (a b c : ℕ) (p : ℕ) (h : a = 8 ∧ b = 10 ∧ c = 12 ∧ p = 150) :
  ∃ x : ℕ, (x = p / (a + b + c) ∧ 8 * x = 40) :=
by
  sorry

end similar_triangle_shortest_side_l110_110168


namespace middle_dimension_of_crate_l110_110774

theorem middle_dimension_of_crate (middle_dimension : ℝ) : 
    (∀ r : ℝ, r = 5 → ∃ w h l : ℝ, w = 5 ∧ h = 12 ∧ l = middle_dimension ∧
        (diameter = 2 * r ∧ diameter ≤ middle_dimension ∧ h ≥ 12)) → 
    middle_dimension = 10 :=
by
  sorry

end middle_dimension_of_crate_l110_110774


namespace reducible_fraction_implies_divisibility_l110_110817

theorem reducible_fraction_implies_divisibility
  (a b c d l k : ℤ)
  (m n : ℤ)
  (h1 : a * l + b = k * m)
  (h2 : c * l + d = k * n)
  : k ∣ (a * d - b * c) :=
by
  sorry

end reducible_fraction_implies_divisibility_l110_110817


namespace function_symmetry_property_l110_110539

noncomputable def f (x : ℝ) : ℝ :=
  x ^ 2

def symmetry_property := 
  ∀ (x : ℝ), (-1 < x ∧ x ≤ 1) →
    (¬ (f (-x) = f x) ∧ ¬ (f (-x) = -f x))

theorem function_symmetry_property :
  symmetry_property :=
by
  sorry

end function_symmetry_property_l110_110539


namespace remainder_abc_l110_110856

theorem remainder_abc (a b c : ℕ) 
  (h₀ : a < 9) (h₁ : b < 9) (h₂ : c < 9)
  (h₃ : (a + 3 * b + 2 * c) % 9 = 0)
  (h₄ : (2 * a + 2 * b + 3 * c) % 9 = 3)
  (h₅ : (3 * a + b + 2 * c) % 9 = 6) : 
  (a * b * c) % 9 = 0 := by
  sorry

end remainder_abc_l110_110856


namespace simplify_fraction_complex_l110_110863

open Complex

theorem simplify_fraction_complex :
  (3 - I) / (2 + 5 * I) = (1 / 29) - (17 / 29) * I := by
  sorry

end simplify_fraction_complex_l110_110863


namespace triangle_angle_B_l110_110430

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) (h : a / b = 3 / Real.sqrt 7) (h2 : b / c = Real.sqrt 7 / 2) : B = Real.pi / 3 :=
by
  sorry

end triangle_angle_B_l110_110430


namespace brown_rabbit_hop_distance_l110_110092

theorem brown_rabbit_hop_distance
  (w : ℕ) (b : ℕ) (t : ℕ)
  (h1 : w = 15)
  (h2 : t = 135)
  (hop_distance_in_5_minutes : w * 5 + b * 5 = t) : 
  b = 12 :=
by
  sorry

end brown_rabbit_hop_distance_l110_110092


namespace collinear_points_solves_a_l110_110201

theorem collinear_points_solves_a : 
  ∀ (a : ℝ),
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  (8 - 3) / (5 - 1) = (a - 8) / (29 - 5) → a = 38 :=
by 
  intro a
  let A := (1, 3)
  let B := (5, 8)
  let C := (29, a)
  intro h
  sorry

end collinear_points_solves_a_l110_110201


namespace impossible_to_form_3x3_in_upper_left_or_right_l110_110600

noncomputable def initial_positions : List (ℕ × ℕ) := 
  [(6, 1), (6, 2), (6, 3), (7, 1), (7, 2), (7, 3), (8, 1), (8, 2), (8, 3)]

def sum_vertical (positions : List (ℕ × ℕ)) : ℕ :=
  positions.foldr (λ pos acc => pos.1 + acc) 0

theorem impossible_to_form_3x3_in_upper_left_or_right
  (initial_positions_set : List (ℕ × ℕ) := initial_positions)
  (initial_sum := sum_vertical initial_positions_set)
  (target_positions_upper_left : List (ℕ × ℕ) := [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)])
  (target_positions_upper_right : List (ℕ × ℕ) := [(1, 6), (1, 7), (1, 8), (2, 6), (2, 7), (2, 8), (3, 6), (3, 7), (3, 8)])
  (target_sum_upper_left := sum_vertical target_positions_upper_left)
  (target_sum_upper_right := sum_vertical target_positions_upper_right) : 
  ¬ (initial_sum % 2 = 1 ∧ target_sum_upper_left % 2 = 0 ∧ target_sum_upper_right % 2 = 0) := sorry

end impossible_to_form_3x3_in_upper_left_or_right_l110_110600


namespace length_of_angle_bisector_l110_110777

theorem length_of_angle_bisector (AB AC : ℝ) (angleBAC : ℝ) (AD : ℝ) :
  AB = 6 → AC = 3 → angleBAC = 60 → AD = 2 * Real.sqrt 3 :=
by
  intro hAB hAC hAngleBAC
  -- Consider adding proof steps here in the future
  sorry

end length_of_angle_bisector_l110_110777


namespace union_A_B_l110_110562

def A (x : ℝ) : Set ℝ := {x ^ 2, 2 * x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}

theorem union_A_B (x : ℝ) (h : {9} = A x ∩ B x) :
  (A x ∪ B x) = {(-8 : ℝ), -7, -4, 4, 9} := by
  sorry

end union_A_B_l110_110562


namespace infinite_solutions_c_l110_110872

theorem infinite_solutions_c (c : ℝ) :
  (∀ y : ℝ, 3 * (5 + c * y) = 15 * y + 15) ↔ c = 5 :=
sorry

end infinite_solutions_c_l110_110872


namespace square_tablecloth_side_length_l110_110309

theorem square_tablecloth_side_length (area : ℝ) (h : area = 5) : ∃ a : ℝ, a > 0 ∧ a * a = 5 := 
by
  use Real.sqrt 5
  constructor
  · apply Real.sqrt_pos.2; linarith
  · exact Real.mul_self_sqrt (by linarith [h])

end square_tablecloth_side_length_l110_110309


namespace arithmetic_sequence_sum_geometric_sequence_ratio_l110_110653

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℕ) (q : ℚ) :=
  ∀ n, a (n + 1) = a n * q
  
-- Prove the sum of the first n terms for an arithmetic sequence
theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) :
  a 1 = 3 ∧ (∀ n, S n = (n * (3 + a (n + 1) - 1)) / 2) ∧ is_arithmetic_sequence a 4 → 
  S n = 2 * n^2 + n :=
sorry

-- Prove the range of the common ratio for a geometric sequence
theorem geometric_sequence_ratio (a : ℕ → ℕ) (S : ℕ → ℚ) (q : ℚ) :
  a 1 = 3 ∧ is_geometric_sequence a q ∧ ∃ lim : ℚ, (∀ n, S n = (a 1 * (1 - q^n)) / (1 - q)) ∧ lim < 12 → 
  -1 < q ∧ q < 1 ∧ q ≠ 0 ∧ q < 3/4 :=
sorry

end arithmetic_sequence_sum_geometric_sequence_ratio_l110_110653


namespace find_a_l110_110768

theorem find_a :
  ∃ a : ℝ, (∀ t1 t2 : ℝ, t1 + t2 = -a ∧ t1 * t2 = -2017 ∧ 2 * t1 = 4) → a = 1006.5 :=
by
  sorry

end find_a_l110_110768


namespace initial_oak_trees_l110_110756

theorem initial_oak_trees (n : ℕ) (h : n - 2 = 7) : n = 9 := 
by
  sorry

end initial_oak_trees_l110_110756


namespace degree_of_g_l110_110393

theorem degree_of_g 
  (f : Polynomial ℤ)
  (g : Polynomial ℤ) 
  (h₁ : f = -9 * Polynomial.X^5 + 4 * Polynomial.X^3 - 2 * Polynomial.X + 6)
  (h₂ : (f + g).degree = 2) :
  g.degree = 5 :=
sorry

end degree_of_g_l110_110393


namespace calvin_weight_after_one_year_l110_110623

theorem calvin_weight_after_one_year
  (initial_weight : ℕ)
  (monthly_weight_loss: ℕ)
  (months_in_year: ℕ)
  (one_year: ℕ)
  (total_loss: ℕ)
  (final_weight: ℕ) :
  initial_weight = 250 ∧ monthly_weight_loss = 8 ∧ months_in_year = 12 ∧ one_year = 12 ∧ total_loss = monthly_weight_loss * months_in_year →
  final_weight = initial_weight - total_loss →
  final_weight = 154 :=
by
  intros
  sorry

end calvin_weight_after_one_year_l110_110623


namespace percent_students_in_range_l110_110963

theorem percent_students_in_range
    (n1 n2 n3 n4 n5 : ℕ)
    (h1 : n1 = 5)
    (h2 : n2 = 7)
    (h3 : n3 = 8)
    (h4 : n4 = 4)
    (h5 : n5 = 3) :
  ((n3 : ℝ) / (n1 + n2 + n3 + n4 + n5) * 100) = 29.63 :=
by
  sorry

end percent_students_in_range_l110_110963


namespace min_value_of_x_under_conditions_l110_110366

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_value_of_x_under_conditions :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 + y^2 + z^2 = 1 →
  (∃ x_min : ℝ, S x y z = S x_min x_min (Real.sqrt 2 - 1) ∧ x_min = Real.sqrt (Real.sqrt 2 - 1)) :=
by
  intros x y z hx hy hz hxyz
  use Real.sqrt (Real.sqrt 2 - 1)
  sorry

end min_value_of_x_under_conditions_l110_110366


namespace probability_not_blue_l110_110995

-- Definitions based on the conditions
def total_faces : ℕ := 12
def blue_faces : ℕ := 1
def non_blue_faces : ℕ := total_faces - blue_faces

-- Statement of the problem
theorem probability_not_blue : (non_blue_faces : ℚ) / total_faces = 11 / 12 :=
by
  sorry

end probability_not_blue_l110_110995


namespace find_quadratic_function_l110_110057

open Function

-- Define the quadratic function g(x) with parameters c and d
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the main theorem
theorem find_quadratic_function :
  ∃ (c d : ℝ), (∀ x : ℝ, (g c d (g c d x + x)) / (g c d x) = x^2 + 120 * x + 360) ∧ c = 119 ∧ d = 240 :=
by
  sorry

end find_quadratic_function_l110_110057


namespace unique_3_digit_number_with_conditions_l110_110389

def valid_3_digit_number (n : ℕ) : Prop :=
  let d2 := n / 100
  let d1 := (n / 10) % 10
  let d0 := n % 10
  (d2 > 0) ∧ (d2 < 10) ∧ (d1 < 10) ∧ (d0 < 10) ∧ (d2 + d1 + d0 = 28) ∧ (d0 < 7) ∧ (d0 % 2 = 0)

theorem unique_3_digit_number_with_conditions :
  (∃! n : ℕ, valid_3_digit_number n) :=
sorry

end unique_3_digit_number_with_conditions_l110_110389


namespace avg_of_7_consecutive_integers_l110_110705

theorem avg_of_7_consecutive_integers (a b : ℕ) (h1 : b = (a + (a+1) + (a+2) + (a+3) + (a+4)) / 5) : 
  (b + (b + 1) + (b + 2) + (b + 3) + (b + 4) + (b + 5) + (b + 6)) / 7 = a + 5 := 
  sorry

end avg_of_7_consecutive_integers_l110_110705


namespace average_of_N_l110_110068

theorem average_of_N (N : ℤ) (h1 : (1:ℚ)/3 < N/90) (h2 : N/90 < (2:ℚ)/5) : 31 ≤ N ∧ N ≤ 35 → (N = 31 ∨ N = 32 ∨ N = 33 ∨ N = 34 ∨ N = 35) → (31 + 32 + 33 + 34 + 35) / 5 = 33 := by
  sorry

end average_of_N_l110_110068


namespace tangent_line_eq_l110_110502

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x

noncomputable def f' (x : ℝ) : ℝ := (x : ℝ) * Real.exp x

theorem tangent_line_eq (x : ℝ) (h : x = 0) : 
  ∃ (c : ℝ), (1 : ℝ) = 1 ∧ f x = c ∧ f' x = 0 ∧ (∀ y, y = c) :=
by
  sorry

end tangent_line_eq_l110_110502


namespace domino_covering_impossible_odd_squares_l110_110650

theorem domino_covering_impossible_odd_squares
  (board1 : ℕ) -- 24 squares
  (board2 : ℕ) -- 21 squares
  (board3 : ℕ) -- 23 squares
  (board4 : ℕ) -- 35 squares
  (board5 : ℕ) -- 63 squares
  (h1 : board1 = 24)
  (h2 : board2 = 21)
  (h3 : board3 = 23)
  (h4 : board4 = 35)
  (h5 : board5 = 63) :
  (board2 % 2 = 1) ∧ (board3 % 2 = 1) ∧ (board4 % 2 = 1) ∧ (board5 % 2 = 1) :=
by {
  sorry
}

end domino_covering_impossible_odd_squares_l110_110650


namespace M_subset_N_l110_110561

def M : Set ℚ := { x | ∃ k : ℤ, x = k / 2 + 1 / 4 }
def N : Set ℚ := { x | ∃ k : ℤ, x = k / 4 + 1 / 2 }

theorem M_subset_N : M ⊆ N :=
sorry

end M_subset_N_l110_110561


namespace arithmetic_sequence_15th_term_eq_53_l110_110644

theorem arithmetic_sequence_15th_term_eq_53 (a1 : ℤ) (d : ℤ) (n : ℕ) (a_15 : ℤ) 
    (h1 : a1 = -3)
    (h2 : d = 4)
    (h3 : n = 15)
    (h4 : a_15 = a1 + (n - 1) * d) : 
    a_15 = 53 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end arithmetic_sequence_15th_term_eq_53_l110_110644


namespace Sierra_Crest_Trail_Length_l110_110194

theorem Sierra_Crest_Trail_Length (a b c d e : ℕ) 
(h1 : a + b + c = 36) 
(h2 : b + d = 30) 
(h3 : d + e = 38) 
(h4 : a + d = 32) : 
a + b + c + d + e = 74 := by
  sorry

end Sierra_Crest_Trail_Length_l110_110194


namespace mart_income_percentage_juan_l110_110222

-- Define the conditions
def TimIncomeLessJuan (J T : ℝ) : Prop := T = 0.40 * J
def MartIncomeMoreTim (T M : ℝ) : Prop := M = 1.60 * T

-- Define the proof problem
theorem mart_income_percentage_juan (J T M : ℝ) 
  (h1 : TimIncomeLessJuan J T) 
  (h2 : MartIncomeMoreTim T M) :
  M = 0.64 * J := 
  sorry

end mart_income_percentage_juan_l110_110222


namespace solve_for_a_l110_110322

noncomputable def special_otimes (a b : ℝ) : ℝ :=
  if a > b then a^2 + b else a + b^2

theorem solve_for_a (a : ℝ) : special_otimes a (-2) = 4 → a = Real.sqrt 6 :=
by
  intro h
  sorry

end solve_for_a_l110_110322


namespace negate_proposition_l110_110853

theorem negate_proposition : (¬ ∀ x : ℝ, x^2 + 2*x + 1 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 1 ≤ 0 := by
  sorry

end negate_proposition_l110_110853


namespace simplify_expression_l110_110930

theorem simplify_expression (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_eq : x^3 + y^3 = 3 * (x + y)) :
  (x / y) + (y / x) - (3 / (x * y)) = 1 :=
by
  sorry

end simplify_expression_l110_110930


namespace tile_C_is_TileIV_l110_110637

-- Define the tiles with their respective sides
structure Tile :=
(top right bottom left : ℕ)

def TileI : Tile := { top := 1, right := 2, bottom := 5, left := 6 }
def TileII : Tile := { top := 6, right := 3, bottom := 1, left := 5 }
def TileIII : Tile := { top := 5, right := 7, bottom := 2, left := 3 }
def TileIV : Tile := { top := 3, right := 5, bottom := 7, left := 2 }

-- Define Rectangles for reasoning
inductive Rectangle
| A
| B
| C
| D

open Rectangle

-- Define the mathematical statement to prove
theorem tile_C_is_TileIV : ∃ tile, tile = TileIV :=
  sorry

end tile_C_is_TileIV_l110_110637


namespace triplet_divisibility_cond_l110_110656

theorem triplet_divisibility_cond (a b c : ℤ) (hac : a ≥ 2) (hbc : b ≥ 2) (hcc : c ≥ 2) :
  (a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1) ↔ 
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 3 ∧ b = 15 ∧ c = 5) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 2 ∧ b = 8 ∧ c = 4) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 4) ∨ (a = 2 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end triplet_divisibility_cond_l110_110656


namespace N_properties_l110_110472

def N : ℕ := 3625

theorem N_properties :
  (N % 32 = 21) ∧ (N % 125 = 0) ∧ (N^2 % 8000 = N % 8000) :=
by
  sorry

end N_properties_l110_110472


namespace problem_solution_l110_110325

def count_multiples_of_5_not_15 : ℕ := 
  let count_up_to (m n : ℕ) := n / m
  let multiples_of_5_up_to_300 := count_up_to 5 299
  let multiples_of_15_up_to_300 := count_up_to 15 299
  multiples_of_5_up_to_300 - multiples_of_15_up_to_300

theorem problem_solution : count_multiples_of_5_not_15 = 40 := by
  sorry

end problem_solution_l110_110325


namespace john_average_speed_l110_110754

theorem john_average_speed:
  (∃ J : ℝ, Carla_speed = 35 ∧ Carla_time = 3 ∧ John_time = 3.5 ∧ J * John_time = Carla_speed * Carla_time) →
  (∃ J : ℝ, J = 30) :=
by
  -- Given Variables
  let Carla_speed : ℝ := 35
  let Carla_time : ℝ := 3
  let John_time : ℝ := 3.5
  -- Proof goal
  sorry

end john_average_speed_l110_110754


namespace bags_of_soil_needed_l110_110455

theorem bags_of_soil_needed
  (length width height : ℕ)
  (beds : ℕ)
  (volume_per_bag : ℕ)
  (h_length : length = 8)
  (h_width : width = 4)
  (h_height : height = 1)
  (h_beds : beds = 2)
  (h_volume_per_bag : volume_per_bag = 4) :
  (length * width * height * beds) / volume_per_bag = 16 :=
by
  sorry

end bags_of_soil_needed_l110_110455


namespace lcm_12_18_24_l110_110210

theorem lcm_12_18_24 : Nat.lcm (Nat.lcm 12 18) 24 = 72 := by
  -- Given conditions (prime factorizations)
  have h1 : 12 = 2^2 * 3 := by norm_num
  have h2 : 18 = 2 * 3^2 := by norm_num
  have h3 : 24 = 2^3 * 3 := by norm_num
  -- Prove the LCM
  sorry

end lcm_12_18_24_l110_110210


namespace daily_calories_burned_l110_110787

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def days : ℕ := 35
def total_calories := pounds_to_lose * calories_per_pound

theorem daily_calories_burned :
  (total_calories / days) = 500 := 
  by 
    -- calculation steps
    sorry

end daily_calories_burned_l110_110787


namespace fraction_changed_value_l110_110916

theorem fraction_changed_value:
  ∀ (num denom : ℝ), num / denom = 0.75 →
  (num + 0.15 * num) / (denom - 0.08 * denom) = 0.9375 :=
by
  intros num denom h_fraction
  sorry

end fraction_changed_value_l110_110916


namespace pascal_triangle_fifth_number_l110_110986

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l110_110986


namespace like_terms_exponent_l110_110034

theorem like_terms_exponent (m n : ℤ) (h₁ : n = 2) (h₂ : m = 1) : m - n = -1 :=
by
  sorry

end like_terms_exponent_l110_110034


namespace tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l110_110006

variable (α : ℝ)

theorem tan_alpha_sub_2pi_over_3 (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    Real.tan (α - 2 * π / 3) = 2 * Real.sqrt 3 :=
sorry

theorem two_sin_sq_alpha_sub_cos_sq_alpha (h : Real.tan (α + π / 3) = 2 * Real.sqrt 3) :
    2 * (Real.sin α) ^ 2 - (Real.cos α) ^ 2 = -43 / 52 :=
sorry

end tan_alpha_sub_2pi_over_3_two_sin_sq_alpha_sub_cos_sq_alpha_l110_110006


namespace infinite_positive_integer_solutions_l110_110819

theorem infinite_positive_integer_solutions :
  ∃ (k : ℕ), ∀ (n : ℕ), n > 24 → ∃ k > 24, k = n :=
sorry

end infinite_positive_integer_solutions_l110_110819


namespace midpoint_sum_four_times_l110_110349

theorem midpoint_sum_four_times (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = -4) (h3 : x2 = -2) (h4 : y2 = 10) :
  4 * ((x1 + x2) / 2 + (y1 + y2) / 2) = 24 :=
by
  rw [h1, h2, h3, h4]
  -- simplifying to get the desired result
  sorry

end midpoint_sum_four_times_l110_110349


namespace find_circles_tangent_to_axes_l110_110096

def tangent_to_axes_and_passes_through (R : ℝ) (P : ℝ × ℝ) :=
  let center := (R, R)
  (P.1 - R) ^ 2 + (P.2 - R) ^ 2 = R ^ 2

theorem find_circles_tangent_to_axes (x y : ℝ) :
  (tangent_to_axes_and_passes_through 1 (2, 1) ∧ tangent_to_axes_and_passes_through 1 (x, y)) ∨
  (tangent_to_axes_and_passes_through 5 (2, 1) ∧ tangent_to_axes_and_passes_through 5 (x, y)) :=
by {
  sorry
}

end find_circles_tangent_to_axes_l110_110096


namespace difference_in_peaches_l110_110040

-- Define the number of peaches Audrey has
def audrey_peaches : ℕ := 26

-- Define the number of peaches Paul has
def paul_peaches : ℕ := 48

-- Define the expected difference
def expected_difference : ℕ := 22

-- The theorem stating the problem
theorem difference_in_peaches : (paul_peaches - audrey_peaches = expected_difference) :=
by
  sorry

end difference_in_peaches_l110_110040


namespace fraction_dad_roasted_l110_110013

theorem fraction_dad_roasted :
  ∀ (dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast : ℕ),
    dad_marshmallows = 21 →
    joe_marshmallows = 4 * dad_marshmallows →
    joe_roast = joe_marshmallows / 2 →
    total_roast = 49 →
    dad_roast = total_roast - joe_roast →
    (dad_roast : ℚ) / (dad_marshmallows : ℚ) = 1 / 3 :=
by
  intros dad_marshmallows joe_marshmallows joe_roast total_roast dad_roast
  intro h_dad_marshmallows
  intro h_joe_marshmallows
  intro h_joe_roast
  intro h_total_roast
  intro h_dad_roast
  sorry

end fraction_dad_roasted_l110_110013


namespace largest_common_term_in_range_l110_110671

def seq1 (n : ℕ) : ℕ := 5 + 9 * n
def seq2 (m : ℕ) : ℕ := 3 + 8 * m

theorem largest_common_term_in_range :
  ∃ (a : ℕ) (n m : ℕ), seq1 n = a ∧ seq2 m = a ∧ 1 ≤ a ∧ a ≤ 200 ∧ (∀ b, (∃ nf mf, seq1 nf = b ∧ seq2 mf = b ∧ 1 ≤ b ∧ b ≤ 200) → b ≤ a) :=
sorry

end largest_common_term_in_range_l110_110671


namespace stephanie_oranges_l110_110836

theorem stephanie_oranges (visits : ℕ) (oranges_per_visit : ℕ) (total_oranges : ℕ) 
  (h_visits : visits = 8) (h_oranges_per_visit : oranges_per_visit = 2) : 
  total_oranges = oranges_per_visit * visits := 
by
  sorry

end stephanie_oranges_l110_110836


namespace find_a_b_l110_110527

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_b 
  (h_max : ∀ x, f a b x ≤ 3)
  (h_min : ∀ x, f a b x ≥ 2)
  : (a = 0.5 ∨ a = -0.5) ∧ b = 2.5 :=
by
  sorry

end find_a_b_l110_110527


namespace valid_number_count_l110_110620

def is_valid_digit (d: Nat) : Prop :=
  d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def are_adjacent (d1 d2: Nat) : Bool :=
  (d1 = 1 ∧ d2 = 2) ∨ (d1 = 2 ∧ d2 = 1) ∨
  (d1 = 5 ∧ (d2 = 1 ∨ d2 = 2)) ∨ 
  (d2 = 5 ∧ (d1 = 1 ∨ d1 = 2))

def count_valid_numbers : Nat :=
  sorry -- expression to count numbers according to given conditions.

theorem valid_number_count : count_valid_numbers = 36 :=
  sorry

end valid_number_count_l110_110620


namespace percentage_of_green_eyed_brunettes_l110_110294

def conditions (a b c d : ℝ) : Prop :=
  (a / (a + b) = 0.65) ∧
  (b / (b + c) = 0.7) ∧
  (c / (c + d) = 0.1)

theorem percentage_of_green_eyed_brunettes (a b c d : ℝ) (h : conditions a b c d) :
  d / (a + b + c + d) = 0.54 :=
sorry

end percentage_of_green_eyed_brunettes_l110_110294


namespace xy_in_B_l110_110437

def A : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = m * a^2 + k * a * b + m * b^2}

def B : Set ℤ := 
  {z | ∃ a b k m : ℤ, z = a^2 + k * a * b + m^2 * b^2}

theorem xy_in_B (x y : ℤ) (h1 : x ∈ A) (h2 : y ∈ A) : x * y ∈ B := by
  sorry

end xy_in_B_l110_110437


namespace pq_sum_is_38_l110_110720

theorem pq_sum_is_38
  (p q : ℝ)
  (h_root : ∀ x, (2 * x^2) + (p * x) + q = 0 → x = 2 * Complex.I - 3 ∨ x = -2 * Complex.I - 3)
  (h_p_q : ∀ a b : ℂ, a + b = -p / 2 ∧ a * b = q / 2 → p = 12 ∧ q = 26) :
  p + q = 38 :=
sorry

end pq_sum_is_38_l110_110720


namespace rhombus_area_l110_110088

theorem rhombus_area (d₁ d₂ : ℕ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) : 
  (1 / 2 : ℝ) * d₁ * d₂ = 24 := 
by
  sorry

end rhombus_area_l110_110088


namespace lara_puts_flowers_in_vase_l110_110143

theorem lara_puts_flowers_in_vase : 
  ∀ (total_flowers mom_flowers flowers_given_more : ℕ), 
    total_flowers = 52 →
    mom_flowers = 15 →
    flowers_given_more = 6 →
  (total_flowers - (mom_flowers + (mom_flowers + flowers_given_more))) = 16 :=
by
  intros total_flowers mom_flowers flowers_given_more h1 h2 h3
  sorry

end lara_puts_flowers_in_vase_l110_110143


namespace part_a_l110_110038

theorem part_a (a b : ℤ) (h : a^2 - (b^2 - 4 * b + 1) * a - (b^4 - 2 * b^3) = 0) : 
  ∃ k : ℤ, b^2 + a = k^2 :=
sorry

end part_a_l110_110038


namespace find_income_separator_l110_110019

-- Define the income and tax parameters
def income : ℝ := 60000
def total_tax : ℝ := 8000
def rate1 : ℝ := 0.10
def rate2 : ℝ := 0.20

-- Define the function for total tax calculation
def tax (I : ℝ) : ℝ := rate1 * I + rate2 * (income - I)

theorem find_income_separator (I : ℝ) (h: tax I = total_tax) : I = 40000 :=
by sorry

end find_income_separator_l110_110019


namespace tangent_line_intercept_l110_110988

theorem tangent_line_intercept :
  ∃ (m : ℚ) (b : ℚ), m > 0 ∧ b = 740 / 171 ∧
    ∀ (x y : ℚ), ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x - 15)^2 + (y - 8)^2 = 100) →
                 (y = m * x + b) ↔ False := 
sorry

end tangent_line_intercept_l110_110988


namespace amara_remaining_clothes_l110_110022

noncomputable def remaining_clothes (initial total_donated thrown_away : ℕ) : ℕ :=
  initial - (total_donated + thrown_away)

theorem amara_remaining_clothes : 
  ∀ (initial donated_first donated_second thrown_away : ℕ), initial = 100 → donated_first = 5 → donated_second = 15 → thrown_away = 15 → 
  remaining_clothes initial (donated_first + donated_second) thrown_away = 65 := 
by 
  intros initial donated_first donated_second thrown_away hinital hdonated_first hdonated_second hthrown_away
  rw [hinital, hdonated_first, hdonated_second, hthrown_away]
  unfold remaining_clothes
  norm_num

end amara_remaining_clothes_l110_110022


namespace min_sum_xy_l110_110825

theorem min_sum_xy (x y : ℕ) (hx : x ≠ y) (hcond : ↑(1 / x) + ↑(1 / y) = 1 / 15) : x + y = 64 :=
sorry

end min_sum_xy_l110_110825


namespace exp_f_f_increasing_inequality_l110_110344

noncomputable def f (a b : ℝ) (x : ℝ) :=
  (a * x + b) / (x^2 + 1)

-- Conditions
variable (a b : ℝ)
axiom h_odd : ∀ x : ℝ, f a b (-x) = - f a b x
axiom h_value : f a b (1/2) = 2/5

-- Proof statements
theorem exp_f : f a b x = x / (x^2 + 1) := sorry

theorem f_increasing (x1 x2 : ℝ) (h1 : -1 < x1) (h2 : x1 < x2) (h3 : x2 < 1) : 
  f a b x1 < f a b x2 := sorry

theorem inequality (x : ℝ) (h1 : 0 < x) (h2 : x < 1/3) :
  f a b (2 * x - 1) + f a b x < 0 := sorry

end exp_f_f_increasing_inequality_l110_110344


namespace factor_of_60n_l110_110204

theorem factor_of_60n
  (n : ℕ)
  (x : ℕ)
  (h_condition1 : ∃ k : ℕ, 60 * n = x * k)
  (h_condition2 : ∃ m : ℕ, 60 * n = 8 * m)
  (h_condition3 : n >= 8) :
  x = 60 :=
sorry

end factor_of_60n_l110_110204


namespace arithmetic_sequence_value_l110_110501

theorem arithmetic_sequence_value :
  ∀ (a : ℕ → ℤ), 
  a 1 = 1 → 
  a 3 = -5 → 
  (a 1 - a 2 - a 3 - a 4 = 16) :=
by
  intros a h1 h3
  sorry

end arithmetic_sequence_value_l110_110501


namespace sum_of_number_and_conjugate_l110_110569

theorem sum_of_number_and_conjugate :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end sum_of_number_and_conjugate_l110_110569


namespace triangle_properties_l110_110725

theorem triangle_properties (a b c : ℝ) 
  (h : |a - Real.sqrt 7| + Real.sqrt (b - 5) + (c - 4 * Real.sqrt 2)^2 = 0) :
  a = Real.sqrt 7 ∧ b = 5 ∧ c = 4 * Real.sqrt 2 ∧ a^2 + b^2 = c^2 := by
{
  sorry
}

end triangle_properties_l110_110725


namespace parallel_vectors_x_value_l110_110134

/-
Given that \(\overrightarrow{a} = (1,2)\) and \(\overrightarrow{b} = (2x, -3)\) are parallel vectors, prove that \(x = -\frac{3}{4}\).
-/
theorem parallel_vectors_x_value (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (2 * x, -3)) 
  (h_parallel : (a.1 * b.2) - (a.2 * b.1) = 0) : 
  x = -3 / 4 := by
  sorry

end parallel_vectors_x_value_l110_110134


namespace find_sum_of_squares_l110_110565

-- Definitions for the conditions: a, b, and c are different prime numbers,
-- and their product equals five times their sum.

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def condition (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
  a * b * c = 5 * (a + b + c)

-- Statement of the proof problem.
theorem find_sum_of_squares (a b c : ℕ) (h : condition a b c) : a^2 + b^2 + c^2 = 78 :=
sorry

end find_sum_of_squares_l110_110565


namespace matrix_determinant_equality_l110_110082

open Complex Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_determinant_equality (A B : Matrix n n ℂ) (x : ℂ) 
  (h1 : A ^ 2 + B ^ 2 = 2 * A * B) :
  det (A - x • 1) = det (B - x • 1) :=
  sorry

end matrix_determinant_equality_l110_110082


namespace focus_of_parabola_l110_110199

noncomputable def parabola_focus (a h k : ℝ) : ℝ × ℝ :=
  (h, k + 1 / (4 * a))

theorem focus_of_parabola :
  parabola_focus 9 (-1/3) (-3) = (-1/3, -107/36) := 
  sorry

end focus_of_parabola_l110_110199


namespace triangle_is_isosceles_or_right_l110_110005

theorem triangle_is_isosceles_or_right (A B C a b : ℝ) (h : a * Real.cos (π - A) + b * Real.sin (π / 2 + B) = 0)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) : 
  (A = B ∨ A + B = π / 2) := 
sorry

end triangle_is_isosceles_or_right_l110_110005


namespace find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l110_110929

-- Define the ellipse and parameters
variables (a b c : ℝ) (x y : ℝ)
-- Conditions
def ellipse (a b : ℝ) : Prop := a > b ∧ b > 0 ∧ (∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)

-- Given conditions
def eccentricity (c a : ℝ) : Prop := c = a * (Real.sqrt 3 / 2)
def rhombus_area (a b : ℝ) : Prop := (1/2) * (2 * a) * (2 * b) = 4
def relation_a_b_c (a b c : ℝ) : Prop := a^2 = b^2 + c^2

-- Questions transformed into proof problems
def ellipse_equation (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def range_OA_OB (OA OB : ℝ) : Prop := OA * OB ∈ Set.union (Set.Icc (-(3/2)) 0) (Set.Ioo 0 (3/2))
def quadrilateral_area : ℝ := 4

-- Prove the results given the conditions
theorem find_equation_of_ellipse (a b c : ℝ) (h_ellipse : ellipse a b) (h_ecc : eccentricity c a) (h_area : rhombus_area a b) (h_rel : relation_a_b_c a b c) :
  ellipse_equation x y := by
  sorry

theorem find_range_OA_OB (OA OB : ℝ) (kAC kBD : ℝ) (h_mult : kAC * kBD = -(1/4)) :
  range_OA_OB OA OB := by
  sorry

theorem find_area_quadrilateral : quadrilateral_area = 4 := by
  sorry

end find_equation_of_ellipse_find_range_OA_OB_find_area_quadrilateral_l110_110929


namespace f_of_2_l110_110337

def f (x : ℝ) : ℝ := sorry

theorem f_of_2 : f 2 = 20 / 3 :=
    sorry

end f_of_2_l110_110337


namespace circles_intersect_and_inequality_l110_110789

variable {R r d : ℝ}

theorem circles_intersect_and_inequality (hR : R > r) (h_intersect: R - r < d ∧ d < R + r) : R - r < d ∧ d < R + r :=
by
  exact h_intersect

end circles_intersect_and_inequality_l110_110789


namespace twin_ages_l110_110861

theorem twin_ages (x : ℕ) (h : (x + 1) ^ 2 = x ^ 2 + 15) : x = 7 :=
sorry

end twin_ages_l110_110861


namespace cornbread_pieces_count_l110_110129

-- Define the dimensions of the pan and the pieces of cornbread
def pan_length := 24
def pan_width := 20
def piece_length := 3
def piece_width := 2
def margin := 1

-- Define the effective width after considering the margin
def effective_width := pan_width - margin

-- Prove the number of pieces of cornbread is 72
theorem cornbread_pieces_count :
  (pan_length / piece_length) * (effective_width / piece_width) = 72 :=
by
  sorry

end cornbread_pieces_count_l110_110129


namespace james_earnings_l110_110158

-- Define the conditions
def rain_gallons_per_inch : ℕ := 15
def rain_monday : ℕ := 4
def rain_tuesday : ℕ := 3
def price_per_gallon : ℝ := 1.2

-- State the theorem to be proved
theorem james_earnings : (rain_monday * rain_gallons_per_inch + rain_tuesday * rain_gallons_per_inch) * price_per_gallon = 126 :=
by
  sorry

end james_earnings_l110_110158


namespace fraction_simplified_to_p_l110_110469

theorem fraction_simplified_to_p (q : ℕ) (hq_pos : 0 < q) (gcd_cond : Nat.gcd 4047 q = 1) :
    (2024 / 2023) - (2023 / 2024) = 4047 / q := sorry

end fraction_simplified_to_p_l110_110469


namespace brenda_age_l110_110714

theorem brenda_age (A B J : ℕ) (h1 : A = 3 * B) (h2 : J = B + 10) (h3 : A = J) : B = 5 :=
sorry

end brenda_age_l110_110714


namespace joel_garden_size_l110_110551

-- Definitions based on the conditions
variable (G : ℕ) -- G is the size of Joel's garden.

-- Condition 1: Half of the garden is for fruits.
def half_garden_fruits (G : ℕ) := G / 2

-- Condition 2: Half of the garden is for vegetables.
def half_garden_vegetables (G : ℕ) := G / 2

-- Condition 3: A quarter of the fruit section is used for strawberries.
def quarter_fruit_section (G : ℕ) := (half_garden_fruits G) / 4

-- Condition 4: The quarter for strawberries takes up 8 square feet.
axiom strawberry_section : quarter_fruit_section G = 8

-- Hypothesis: The size of Joel's garden is 64 square feet.
theorem joel_garden_size : G = 64 :=
by
  -- Insert the logical progression of the proof here.
  sorry

end joel_garden_size_l110_110551


namespace inequality_one_system_of_inequalities_l110_110031

theorem inequality_one (x : ℝ) : 
  (2 * x - 2) / 3 ≤ 2 - (2 * x + 2) / 2 → x ≤ 1 :=
sorry

theorem system_of_inequalities (x : ℝ) : 
  (3 * (x - 2) - 1 ≥ -4 - 2 * (x - 2) → x ≥ 7 / 5) ∧
  ((1 - 2 * x) / 3 > (3 * (2 * x - 1)) / 2 → x < 1 / 2) → false :=
sorry

end inequality_one_system_of_inequalities_l110_110031


namespace solve_for_x_l110_110852

theorem solve_for_x (x : ℝ) (h₁ : 3 * x^2 - 9 * x = 0) (h₂ : x ≠ 0) : x = 3 := 
by {
  sorry
}

end solve_for_x_l110_110852


namespace sin_cos_identity_trig_identity_l110_110709

open Real

-- Problem I
theorem sin_cos_identity (α : ℝ) : 
  (4 * sin α - 2 * cos α) / (5 * cos α + 3 * sin α) = 5 / 7 → 
  sin α * cos α = 3 / 10 := 
sorry

-- Problem II
theorem trig_identity : 
  (sqrt (1 - 2 * sin (10 * π / 180) * cos (10 * π / 180))) / 
  (cos (10 * π / 180) - sqrt (1 - cos (170 * π / 180)^2)) = 1 := 
sorry

end sin_cos_identity_trig_identity_l110_110709


namespace quadratic_solution_range_l110_110483

noncomputable def quadratic_inequality_real_solution (c : ℝ) : Prop :=
  0 < c ∧ c < 16

theorem quadratic_solution_range :
  ∀ c : ℝ, (∃ x : ℝ, x^2 - 8 * x + c < 0) ↔ quadratic_inequality_real_solution c :=
by
  intro c
  simp only [quadratic_inequality_real_solution]
  sorry

end quadratic_solution_range_l110_110483


namespace evaluation_of_expression_l110_110277

theorem evaluation_of_expression :
  10 * (1 / 8) - 6.4 / 8 + 1.2 * 0.125 = 0.6 :=
by sorry

end evaluation_of_expression_l110_110277


namespace integral_of_quadratic_has_minimum_value_l110_110575

theorem integral_of_quadratic_has_minimum_value :
  ∃ m : ℝ, (∀ x : ℝ, x^2 + 2 * x + m ≥ -1) ∧ (∫ x in (1:ℝ)..(2:ℝ), x^2 + 2 * x = (16 / 3:ℝ)) :=
by sorry

end integral_of_quadratic_has_minimum_value_l110_110575


namespace complement_of_set_M_l110_110160

open Set

def universal_set : Set ℝ := univ

def set_M : Set ℝ := {x | x^2 < 2 * x}

def complement_M : Set ℝ := compl set_M

theorem complement_of_set_M :
  complement_M = {x | x ≤ 0 ∨ x ≥ 2} :=
sorry

end complement_of_set_M_l110_110160


namespace length_of_c_l110_110992

theorem length_of_c (A B C : ℝ) (a b c : ℝ) (h1 : (π / 3) - A = B) (h2 : a = 3) (h3 : b = 5) : c = 7 :=
sorry

end length_of_c_l110_110992


namespace part1_part2_l110_110743

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem part1 (h : 1 - a = -1) : a = 2 ∧ 
                                  (∀ x : ℝ, x < Real.log 2 → (Real.exp x - 2) < 0) ∧ 
                                  (∀ x : ℝ, x > Real.log 2 → (Real.exp x - 2) > 0) :=
by
  sorry

theorem part2 (h1 : x1 < Real.log 2) (h2 : x2 > Real.log 2) (h3 : f 2 x1 = f 2 x2) : 
  x1 + x2 < 2 * Real.log 2 :=
by
  sorry

end part1_part2_l110_110743


namespace repeating_decimals_count_l110_110489

theorem repeating_decimals_count : 
  ∀ n : ℕ, 1 ≤ n ∧ n < 1000 → ¬(∃ k : ℕ, n + 1 = 2^k ∨ n + 1 = 5^k) :=
by
  sorry

end repeating_decimals_count_l110_110489


namespace value_of_complex_fraction_l110_110392

theorem value_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) : ((1 - i) / (1 + i)) ^ 2 = -1 :=
by
  sorry

end value_of_complex_fraction_l110_110392


namespace matrix_B_cannot_be_obtained_from_matrix_A_l110_110657

def A : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, -1, -1, 1],
  ![1, 1, -1, 1, -1]
]

def B : Matrix (Fin 5) (Fin 5) ℤ := ![
  ![1, 1, 1, 1, 1],
  ![1, 1, 1, -1, -1],
  ![1, 1, -1, 1, -1],
  ![1, -1, -1, 1, 1],
  ![1, -1, 1, -1, 1]
]

theorem matrix_B_cannot_be_obtained_from_matrix_A :
  A.det ≠ B.det := by
  sorry

end matrix_B_cannot_be_obtained_from_matrix_A_l110_110657


namespace simon_age_in_2010_l110_110878

theorem simon_age_in_2010 :
  ∀ (s j : ℕ), (j = 16 → (j + 24 = s) → j + (2010 - 2005) + 24 = 45) :=
by 
  intros s j h1 h2 
  sorry

end simon_age_in_2010_l110_110878


namespace solve_inequality_x_squared_minus_6x_gt_15_l110_110205

theorem solve_inequality_x_squared_minus_6x_gt_15 :
  { x : ℝ | x^2 - 6 * x > 15 } = { x : ℝ | x < -1.5 } ∪ { x : ℝ | x > 7.5 } :=
by
  sorry

end solve_inequality_x_squared_minus_6x_gt_15_l110_110205


namespace num_of_veg_people_l110_110042

def only_veg : ℕ := 19
def both_veg_nonveg : ℕ := 12

theorem num_of_veg_people : only_veg + both_veg_nonveg = 31 := by 
  sorry

end num_of_veg_people_l110_110042


namespace cost_price_of_book_l110_110857

theorem cost_price_of_book
  (SP : Real)
  (profit_percentage : Real)
  (h1 : SP = 300)
  (h2 : profit_percentage = 0.20) :
  ∃ CP : Real, CP = 250 :=
by
  -- Proof of the statement
  sorry

end cost_price_of_book_l110_110857


namespace calculate_f3_l110_110382

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^7 + a * x^5 + b * x - 5

theorem calculate_f3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := 
by
  sorry

end calculate_f3_l110_110382


namespace product_of_squares_l110_110295

theorem product_of_squares (x : ℝ) (h : |5 * x| + 4 = 49) : x^2 * (if x = 9 then 9 else -9)^2 = 6561 :=
by
  sorry

end product_of_squares_l110_110295


namespace find_ab_l110_110147

theorem find_ab (a b : ℝ) (h1 : a - b = 5) (h2 : a^2 + b^2 = 31) : a * b = 3 := by
  sorry

end find_ab_l110_110147


namespace day_after_1999_cubed_days_is_tuesday_l110_110239

theorem day_after_1999_cubed_days_is_tuesday : 
    let today := "Monday"
    let days_in_week := 7
    let target_days := 1999 ^ 3
    ∃ remaining_days, remaining_days = (target_days % days_in_week) ∧ today = "Monday" ∧ remaining_days = 1 → 
    "Tuesday" = "Tuesday" := 
by
  sorry

end day_after_1999_cubed_days_is_tuesday_l110_110239


namespace balloons_total_l110_110334

theorem balloons_total (number_of_groups balloons_per_group : ℕ)
  (h1 : number_of_groups = 7) (h2 : balloons_per_group = 5) : 
  number_of_groups * balloons_per_group = 35 := by
  sorry

end balloons_total_l110_110334


namespace ratio_of_cube_dimensions_l110_110909

theorem ratio_of_cube_dimensions (V_original V_larger : ℝ) (hV_org : V_original = 64) (hV_lrg : V_larger = 512) :
  (∃ r : ℝ, r^3 = V_larger / V_original) ∧ r = 2 := 
sorry

end ratio_of_cube_dimensions_l110_110909


namespace calculate_p_l110_110799

variable (m n : ℤ) (p : ℤ)

theorem calculate_p (h1 : 3 * m - 2 * n = -2) (h2 : p = 3 * (m + 405) - 2 * (n - 405)) : p = 2023 := 
  sorry

end calculate_p_l110_110799


namespace correct_sampling_method_l110_110803

-- Definitions based on conditions
def number_of_classes : ℕ := 16
def sampled_classes : ℕ := 2
def sampling_method := "Lottery then Stratified"

-- The theorem statement based on the proof problem
theorem correct_sampling_method :
  (number_of_classes = 16) ∧ (sampled_classes = 2) → (sampling_method = "Lottery then Stratified") :=
sorry

end correct_sampling_method_l110_110803


namespace most_suitable_survey_l110_110443

-- Define the options as a type
inductive SurveyOption
| A -- Understanding the crash resistance of a batch of cars
| B -- Surveying the awareness of the "one helmet, one belt" traffic regulations among citizens in our city
| C -- Surveying the service life of light bulbs produced by a factory
| D -- Surveying the quality of components of the latest stealth fighter in our country

-- Define a function determining the most suitable for a comprehensive survey
def mostSuitableForCensus : SurveyOption :=
  SurveyOption.D

-- Theorem statement that Option D is the most suitable for a comprehensive survey
theorem most_suitable_survey :
  mostSuitableForCensus = SurveyOption.D :=
  sorry

end most_suitable_survey_l110_110443


namespace coffee_y_ratio_is_1_to_5_l110_110806

-- Define the conditions
variables {p v x y : Type}
variables (p_x p_y v_x v_y : ℕ) -- Coffee amounts in lbs
variables (total_p total_v : ℕ) -- Total amounts of p and v

-- Definitions based on conditions
def coffee_amounts_initial (total_p total_v : ℕ) : Prop :=
  total_p = 24 ∧ total_v = 25

def coffee_x_conditions (p_x v_x : ℕ) : Prop :=
  p_x = 20 ∧ 4 * v_x = p_x

def coffee_y_conditions (p_y v_y total_p total_v : ℕ) : Prop :=
  p_y = total_p - 20 ∧ v_y = total_v - (20 / 4)

-- Statement to prove
theorem coffee_y_ratio_is_1_to_5 {total_p total_v : ℕ}
  (hc1 : coffee_amounts_initial total_p total_v)
  (hc2 : coffee_x_conditions 20 5)
  (hc3 : coffee_y_conditions 4 20 total_p total_v) : 
  (4 / 20 = 1 / 5) :=
sorry

end coffee_y_ratio_is_1_to_5_l110_110806


namespace handshakes_at_gathering_l110_110116

-- Define the number of couples
def couples := 6

-- Define the total number of people
def total_people := 2 * couples

-- Each person shakes hands with 10 others (excluding their spouse)
def handshakes_per_person := 10

-- Total handshakes counted with pairs counted twice
def total_handshakes := total_people * handshakes_per_person / 2

-- The theorem to prove the number of handshakes
theorem handshakes_at_gathering : total_handshakes = 60 :=
by
  sorry

end handshakes_at_gathering_l110_110116


namespace no_purchase_count_l110_110098

def total_people : ℕ := 15
def people_bought_tvs : ℕ := 9
def people_bought_computers : ℕ := 7
def people_bought_both : ℕ := 3

theorem no_purchase_count : total_people - (people_bought_tvs - people_bought_both) - (people_bought_computers - people_bought_both) - people_bought_both = 2 := by
  sorry

end no_purchase_count_l110_110098


namespace total_rock_needed_l110_110975

theorem total_rock_needed (a b : ℕ) (h₁ : a = 8) (h₂ : b = 8) : a + b = 16 :=
by
  sorry

end total_rock_needed_l110_110975


namespace find_x_given_inverse_relationship_l110_110345

theorem find_x_given_inverse_relationship :
  ∀ (x y: ℝ), (0 < x ∧ 0 < y) ∧ ((x^3 * y = 64) ↔ (x = 2 ∧ y = 8)) ∧ (y = 500) →
  x = 2 / 5 :=
by
  intros x y h
  sorry

end find_x_given_inverse_relationship_l110_110345


namespace total_distance_travelled_l110_110812

-- Definitions and propositions
def distance_first_hour : ℝ := 15
def distance_second_hour : ℝ := 18
def distance_third_hour : ℝ := 1.25 * distance_second_hour

-- Conditions based on the problem
axiom second_hour_distance : distance_second_hour = 18
axiom second_hour_20_percent_more : distance_second_hour = 1.2 * distance_first_hour
axiom third_hour_25_percent_more : distance_third_hour = 1.25 * distance_second_hour

-- Proof of the total distance James traveled
theorem total_distance_travelled : 
  distance_first_hour + distance_second_hour + distance_third_hour = 55.5 :=
by
  sorry

end total_distance_travelled_l110_110812


namespace energy_fraction_l110_110982

-- Conditions
variables (E : ℝ → ℝ)
variable (x : ℝ)
variable (h : ∀ x, E (x + 1) = 31.6 * E x)

-- The statement to be proven
theorem energy_fraction (x : ℝ) (h : ∀ x, E (x + 1) = 31.6 * E x) : 
  E (x - 1) / E x = 1 / 31.6 :=
by
  sorry

end energy_fraction_l110_110982


namespace cat_run_time_l110_110960

/-- An electronic cat runs a lap on a circular track with a perimeter of 240 meters.
It runs at a speed of 5 meters per second for the first half of the time and 3 meters per second for the second half of the time.
Prove that the cat takes 36 seconds to run the last 120 meters. -/
theorem cat_run_time
  (perimeter : ℕ)
  (speed1 speed2 : ℕ)
  (half_perimeter : ℕ)
  (half_time : ℕ)
  (last_120m_time : ℕ) :
  perimeter = 240 →
  speed1 = 5 →
  speed2 = 3 →
  half_perimeter = perimeter / 2 →
  half_time = 60 / 2 →
  (5 * half_time - half_perimeter) / speed1 + (half_perimeter - (5 * half_time - half_perimeter)) / speed2 = 36 :=
by sorry

end cat_run_time_l110_110960


namespace paths_H_to_J_via_I_l110_110368

def binom (n k : ℕ) : ℕ := Nat.choose n k

def paths_from_H_to_I : ℕ :=
  binom 7 2  -- Calculate the number of paths from H(0,7) to I(5,5)

def paths_from_I_to_J : ℕ :=
  binom 8 3  -- Calculate the number of paths from I(5,5) to J(8,0)

theorem paths_H_to_J_via_I : paths_from_H_to_I * paths_from_I_to_J = 1176 := by
  -- This theorem states that the number of paths from H to J through I is 1176
  sorry  -- Proof to be provided

end paths_H_to_J_via_I_l110_110368


namespace Tim_has_16_pencils_l110_110323

variable (T_Sarah T_Tyrah T_Tim : Nat)

-- Conditions
def condition1 : Prop := T_Tyrah = 6 * T_Sarah
def condition2 : Prop := T_Tim = 8 * T_Sarah
def condition3 : Prop := T_Tyrah = 12

-- Theorem to prove
theorem Tim_has_16_pencils (h1 : condition1 T_Sarah T_Tyrah) (h2 : condition2 T_Sarah T_Tim) (h3 : condition3 T_Tyrah) : T_Tim = 16 :=
by
  sorry

end Tim_has_16_pencils_l110_110323


namespace divisibility_323_l110_110967

theorem divisibility_323 (n : ℕ) : 
  (20^n + 16^n - 3^n - 1) % 323 = 0 ↔ Even n := 
sorry

end divisibility_323_l110_110967


namespace CauchySchwarz_l110_110127

theorem CauchySchwarz' (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 := by
  sorry

end CauchySchwarz_l110_110127


namespace find_m_from_split_l110_110374

theorem find_m_from_split (m : ℕ) (h1 : m > 1) (h2 : m^2 - m + 1 = 211) : True :=
by
  -- This theorem states that under the conditions that m is a positive integer greater than 1
  -- and m^2 - m + 1 = 211, there exists an integer value for m that satisfies these conditions.
  trivial

end find_m_from_split_l110_110374


namespace area_of_shaded_region_l110_110084

-- Define the vertices of the larger square
def large_square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the polygon forming the shaded area
def shaded_polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 30), (40, 40), (10, 40), (0, 10)]

-- Provide the area of the larger square for reference
def large_square_area : ℝ := 1600

-- Provide the area of the triangles subtracted
def triangles_area : ℝ := 450

-- The main theorem stating the problem:
theorem area_of_shaded_region :
  let shaded_area := large_square_area - triangles_area
  shaded_area = 1150 :=
by
  sorry

end area_of_shaded_region_l110_110084


namespace kerosene_cost_l110_110829

/-- In a market, a dozen eggs cost as much as a pound of rice, and a half-liter of kerosene 
costs as much as 8 eggs. If the cost of each pound of rice is $0.33, then a liter of kerosene costs 44 cents. --/
theorem kerosene_cost : 
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  rice_cost = 0.33 → 1 * ((2 * half_liter_kerosene_cost) * 100) = 44 := 
by
  intros egg_cost rice_cost half_liter_kerosene_cost h_rice_cost
  sorry

end kerosene_cost_l110_110829


namespace find_f2023_l110_110454

-- Define the function and conditions
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def satisfies_condition (f : ℝ → ℝ) := ∀ x : ℝ, f (1 + x) = f (1 - x)

-- Define the main statement to prove that f(2023) = 2 given conditions
theorem find_f2023 (f : ℝ → ℝ)
  (h1 : is_even f)
  (h2 : satisfies_condition f)
  (h3 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = 2^x) :
  f 2023 = 2 :=
sorry

end find_f2023_l110_110454


namespace solve_for_z_l110_110602

open Complex

theorem solve_for_z (z : ℂ) (i : ℂ) (h1 : i = Complex.I) (h2 : z * i = 1 + i) : z = 1 - i :=
by sorry

end solve_for_z_l110_110602


namespace line_through_two_points_l110_110024

-- Define the points (2,5) and (0,3)
structure Point where
  x : ℝ
  y : ℝ

def P1 : Point := {x := 2, y := 5}
def P2 : Point := {x := 0, y := 3}

-- General form of a line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the target line equation as x - y + 3 = 0
def targetLine : Line := {a := 1, b := -1, c := 3}

-- The proof statement to show that the general equation of the line passing through the points (2, 5) and (0, 3) is x - y + 3 = 0
theorem line_through_two_points : ∃ a b c, ∀ x y : ℝ, 
    (a * x + b * y + c = 0) ↔ 
    ((∀ {P : Point}, P = P1 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0) ∧ 
     (∀ {P : Point}, P = P2 → targetLine.a * P.x + targetLine.b * P.y + targetLine.c = 0)) :=
sorry

end line_through_two_points_l110_110024


namespace cook_stole_the_cookbook_l110_110420

-- Define the suspects
inductive Suspect
| CheshireCat
| Duchess
| Cook
deriving DecidableEq, Repr

-- Define the predicate for lying
def lied (s : Suspect) : Prop := sorry

-- Define the conditions
def conditions (thief : Suspect) : Prop :=
  lied thief ∧
  ((∀ s : Suspect, s ≠ thief → lied s) ∨ (∀ s : Suspect, s ≠ thief → ¬lied s))

-- Define the goal statement
theorem cook_stole_the_cookbook : conditions Suspect.Cook :=
sorry

end cook_stole_the_cookbook_l110_110420


namespace ellipse_foci_on_x_axis_l110_110284

variable {a b : ℝ}

theorem ellipse_foci_on_x_axis (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) (hc : ∀ x y : ℝ, (a * x^2 + b * y^2 = 1) → (1 / a > 1 / b ∧ 1 / b > 0))
  : 0 < a ∧ a < b :=
sorry

end ellipse_foci_on_x_axis_l110_110284


namespace cookies_per_person_l110_110838

theorem cookies_per_person (cookies_per_bag : ℕ) (bags : ℕ) (damaged_cookies_per_bag : ℕ) (people : ℕ) (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_each : ℕ) :
  (cookies_per_bag = 738) →
  (bags = 295) →
  (damaged_cookies_per_bag = 13) →
  (people = 125) →
  (total_cookies = cookies_per_bag * bags) →
  (remaining_cookies = total_cookies - (damaged_cookies_per_bag * bags)) →
  (cookies_each = remaining_cookies / people) →
  cookies_each = 1711 :=
by
  sorry 

end cookies_per_person_l110_110838


namespace parabola_problem_l110_110448

-- defining the geometric entities and conditions
variables {x y k x1 y1 x2 y2 : ℝ}

-- the definition for the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- the definition for point M
def point_M (x y : ℝ) : Prop := (x = 0) ∧ (y = 2)

-- the definition for line passing through focus with slope k intersecting the parabola at A and B
def line_through_focus_and_k (x1 y1 x2 y2 k : ℝ) : Prop :=
  (y1 = k * (x1 - 1)) ∧ (y2 = k * (x2 - 1))

-- the definition for vectors MA and MB having dot product zero
def orthogonal_vectors (x1 y1 x2 y2 : ℝ) : Prop :=
  (x1 * x2 + y1 * y2 - 2 * (y1 + y2) + 4 = 0)

-- the main statement to be proved
theorem parabola_problem
  (h_parabola_A : parabola x1 y1)
  (h_parabola_B : parabola x2 y2)
  (h_point_M : point_M 0 2)
  (h_line_through_focus_and_k : line_through_focus_and_k x1 y1 x2 y2 k)
  (h_orthogonal_vectors : orthogonal_vectors x1 y1 x2 y2) :
  k = 1 :=
sorry

end parabola_problem_l110_110448


namespace concentration_after_removal_l110_110070

/-- 
Given:
1. A container has 27 liters of 40% acidic liquid.
2. 9 liters of water is removed from this container.

Prove that the concentration of the acidic liquid in the container after removal is 60%.
-/
theorem concentration_after_removal :
  let initial_volume := 27
  let initial_concentration := 0.4
  let water_removed := 9
  let pure_acid := initial_concentration * initial_volume
  let new_volume := initial_volume - water_removed
  let final_concentration := (pure_acid / new_volume) * 100
  final_concentration = 60 :=
by {
  sorry
}

end concentration_after_removal_l110_110070


namespace average_letters_per_day_l110_110419

theorem average_letters_per_day:
  let letters_per_day := [7, 10, 3, 5, 12]
  (letters_per_day.sum / letters_per_day.length : ℝ) = 7.4 :=
by
  sorry

end average_letters_per_day_l110_110419


namespace Yuna_drank_most_l110_110560

noncomputable def Jimin_juice : ℝ := 0.7
noncomputable def Eunji_juice : ℝ := Jimin_juice - 1/10
noncomputable def Yoongi_juice : ℝ := 4/5
noncomputable def Yuna_juice : ℝ := Jimin_juice + 0.2

theorem Yuna_drank_most :
  Yuna_juice = max (max Jimin_juice Eunji_juice) (max Yoongi_juice Yuna_juice) :=
by
  sorry

end Yuna_drank_most_l110_110560


namespace possible_values_of_a_l110_110618

def P : Set ℝ := {x | x^2 = 1}
def M (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a :
  {a | M a ⊆ P} = {1, -1, 0} :=
sorry

end possible_values_of_a_l110_110618


namespace part1_domain_of_f_part2_inequality_l110_110978

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (abs (x + 1) + abs (x - 1) - 4)

theorem part1_domain_of_f : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by 
  sorry

theorem part2_inequality (a b : ℝ) (h_a : -2 < a) (h_a' : a < 2) (h_b : -2 < b) (h_b' : b < 2) 
  : 2 * abs (a + b) < abs (4 + a * b) :=
by 
  sorry

end part1_domain_of_f_part2_inequality_l110_110978


namespace solve_equation_l110_110010

theorem solve_equation : ∃! x : ℕ, 3^x = x + 2 := by
  sorry

end solve_equation_l110_110010


namespace f1_odd_f2_even_l110_110030

noncomputable def f1 (x : ℝ) : ℝ := x + x^3 + x^5
noncomputable def f2 (x : ℝ) : ℝ := x^2 + 1

theorem f1_odd : ∀ x : ℝ, f1 (-x) = - f1 x := 
by
  sorry

theorem f2_even : ∀ x : ℝ, f2 (-x) = f2 x := 
by
  sorry

end f1_odd_f2_even_l110_110030


namespace cycle_selling_price_l110_110991

theorem cycle_selling_price 
  (CP : ℝ) (gain_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 840) 
  (h2 : gain_percent = 45.23809523809524 / 100)
  (h3 : SP = CP * (1 + gain_percent)) :
  SP = 1220 :=
sorry

end cycle_selling_price_l110_110991


namespace men_women_equal_after_city_Y_l110_110410

variable (M W M' W' : ℕ)

-- Initial conditions: total passengers, women to men ratio
variable (h1 : M + W = 72)
variable (h2 : W = M / 2)

-- Changes in city Y: men leave, women enter
variable (h3 : M' = M - 16)
variable (h4 : W' = W + 8)

theorem men_women_equal_after_city_Y (h1 : M + W = 72) (h2 : W = M / 2) (h3 : M' = M - 16) (h4 : W' = W + 8) : 
  M' = W' := 
by 
  sorry

end men_women_equal_after_city_Y_l110_110410


namespace distance_proof_l110_110021

noncomputable section

open Real

-- Define the given conditions
def AB : Real := 3 * sqrt 3
def BC : Real := 2
def theta : Real := 60 -- angle in degrees
def phi : Real := 180 - theta -- supplementary angle to use in the Law of Cosines

-- Helper function to convert degrees to radians
def deg_to_rad (d : Real) : Real := d * (π / 180)

-- Define the law of cosines to compute AC
def distance_AC (AB BC θ : Real) : Real := 
  sqrt (AB^2 + BC^2 - 2 * AB * BC * cos (deg_to_rad θ))

-- The theorem to prove
theorem distance_proof : distance_AC AB BC phi = 7 :=
by
  sorry

end distance_proof_l110_110021


namespace problem_1_problem_2_problem_3_problem_4_l110_110936

theorem problem_1 : 12 - (-18) + (-7) - 15 = 8 := sorry

theorem problem_2 : -0.5 + (- (3 + 1/4)) + (-2.75) + (7 + 1/2) = 1 := sorry

theorem problem_3 : -2^2 + 3 * (-1)^(2023) - abs (-4) * 5 = -27 := sorry

theorem problem_4 : -3 - (-5 + (1 - 2 * (3 / 5)) / (-2)) = 19 / 10 := sorry

end problem_1_problem_2_problem_3_problem_4_l110_110936


namespace percentage_problem_l110_110698

theorem percentage_problem (x : ℝ) (h : 0.255 * x = 153) : 0.678 * x = 406.8 :=
by
  sorry

end percentage_problem_l110_110698


namespace monotonic_if_and_only_if_extreme_point_inequality_l110_110414

noncomputable def f (x a : ℝ) : ℝ := x^2 - 1 + a * Real.log (1 - x)

def is_monotonic (a : ℝ) : Prop := 
  ∀ x y : ℝ, x < y → f x a ≤ f y a

theorem monotonic_if_and_only_if (a : ℝ) : 
  is_monotonic a ↔ a ≥ 0.5 :=
sorry

theorem extreme_point_inequality (a : ℝ) (x1 x2 : ℝ) (hₐ : 0 < a ∧ a < 0.5) 
  (hx : x1 < x2) (hx₁₂ : f x1 a = f x2 a) : 
  f x1 a / x2 > f x2 a / x1 :=
sorry

end monotonic_if_and_only_if_extreme_point_inequality_l110_110414


namespace complex_sum_eighth_power_l110_110740

noncomputable def compute_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : ℂ :=
  ζ1^8 + ζ2^8 + ζ3^8

theorem complex_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : 
  compute_sum_eighth_power ζ1 ζ2 ζ3 h1 h2 h3 = 451.625 :=
sorry

end complex_sum_eighth_power_l110_110740


namespace original_denominator_is_18_l110_110717

variable (d : ℕ)

theorem original_denominator_is_18
  (h1 : ∃ (d : ℕ), (3 + 7) / (d + 7) = 2 / 5) :
  d = 18 := 
sorry

end original_denominator_is_18_l110_110717


namespace sum_of_numbers_mod_11_l110_110586

def sum_mod_eleven : ℕ := 123456 + 123457 + 123458 + 123459 + 123460 + 123461

theorem sum_of_numbers_mod_11 :
  sum_mod_eleven % 11 = 10 :=
sorry

end sum_of_numbers_mod_11_l110_110586


namespace charming_number_unique_l110_110966

def is_charming (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = 2 * a + b^3

theorem charming_number_unique : ∃! n, 10 ≤ n ∧ n ≤ 99 ∧ is_charming n := by
  sorry

end charming_number_unique_l110_110966


namespace min_val_expression_l110_110167

theorem min_val_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 * b + b^2 * c + c^2 * a = 3) : 
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 :=
sorry

end min_val_expression_l110_110167


namespace quadratic_no_real_solution_l110_110593

theorem quadratic_no_real_solution 
  (a b c : ℝ) 
  (h1 : (2 * a)^2 - 4 * b^2 > 0) 
  (h2 : (2 * b)^2 - 4 * c^2 > 0) : 
  (2 * c)^2 - 4 * a^2 < 0 :=
sorry

end quadratic_no_real_solution_l110_110593


namespace greatest_possible_value_of_q_minus_r_l110_110135

noncomputable def max_difference (q r : ℕ) : ℕ :=
  if q < r then r - q else q - r

theorem greatest_possible_value_of_q_minus_r (q r : ℕ) (x y : ℕ) (hq : q = 10 * x + y) (hr : r = 10 * y + x) (cond : q ≠ r) (hqr : max_difference q r < 20) : q - r = 18 :=
  sorry

end greatest_possible_value_of_q_minus_r_l110_110135


namespace three_digit_multiples_of_36_eq_25_l110_110953

-- Definition: A three-digit number is between 100 and 999
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Definition: A number is a multiple of both 4 and 9 if and only if it's a multiple of 36
def is_multiple_of_36 (n : ℕ) : Prop := n % 36 = 0

-- Definition: Count of three-digit integers that are multiples of 36
def count_multiples_of_36 : ℕ :=
  (999 / 36) - (100 / 36) + 1

-- Theorem: There are 25 three-digit integers that are multiples of 36
theorem three_digit_multiples_of_36_eq_25 : count_multiples_of_36 = 25 := by
  sorry

end three_digit_multiples_of_36_eq_25_l110_110953


namespace nguyen_fabric_needs_l110_110039

def yards_to_feet (yards : ℝ) := yards * 3
def total_fabric_needed (pairs : ℝ) (fabric_per_pair : ℝ) := pairs * fabric_per_pair
def fabric_still_needed (total_needed : ℝ) (already_have : ℝ) := total_needed - already_have

theorem nguyen_fabric_needs :
  let pairs := 7
  let fabric_per_pair := 8.5
  let yards_have := 3.5
  let feet_have := yards_to_feet yards_have
  let total_needed := total_fabric_needed pairs fabric_per_pair
  fabric_still_needed total_needed feet_have = 49 :=
by
  sorry

end nguyen_fabric_needs_l110_110039


namespace total_cost_production_l110_110870

-- Define the fixed cost and marginal cost per product as constants
def fixedCost : ℤ := 12000
def marginalCostPerProduct : ℤ := 200
def numberOfProducts : ℤ := 20

-- Define the total cost as the sum of fixed cost and total variable cost
def totalCost : ℤ := fixedCost + (marginalCostPerProduct * numberOfProducts)

-- Prove that the total cost is equal to 16000
theorem total_cost_production : totalCost = 16000 :=
by
  sorry

end total_cost_production_l110_110870


namespace even_function_has_a_equal_2_l110_110701

noncomputable def f (a x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_has_a_equal_2 (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 2 :=
sorry

end even_function_has_a_equal_2_l110_110701


namespace solution_inequality_l110_110190

-- Define the condition as a predicate
def inequality_condition (x : ℝ) : Prop :=
  (x - 1) * (x + 1) < 0

-- State the theorem that we need to prove
theorem solution_inequality : ∀ x : ℝ, inequality_condition x → (-1 < x ∧ x < 1) :=
by
  intro x hx
  sorry

end solution_inequality_l110_110190


namespace simplify_expression_l110_110753

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 := 
by
  sorry

end simplify_expression_l110_110753


namespace monotonic_decreasing_range_l110_110904

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.cos x

theorem monotonic_decreasing_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ -1 :=
  sorry

end monotonic_decreasing_range_l110_110904


namespace smallest_t_for_temperature_104_l110_110873

theorem smallest_t_for_temperature_104 : 
  ∃ t : ℝ, (-t^2 + 16*t + 40 = 104) ∧ (t > 0) ∧ (∀ s : ℝ, (-s^2 + 16*s + 40 = 104) ∧ (s > 0) → t ≤ s) :=
sorry

end smallest_t_for_temperature_104_l110_110873


namespace toy_cars_ratio_proof_l110_110880

theorem toy_cars_ratio_proof (toys_original : ℕ) (toys_bought_last_month : ℕ) (toys_total : ℕ) :
  toys_original = 25 ∧ toys_bought_last_month = 5 ∧ toys_total = 40 →
  (toys_total - toys_original - toys_bought_last_month) / toys_bought_last_month = 2 :=
by
  sorry

end toy_cars_ratio_proof_l110_110880


namespace perpendicular_lines_condition_l110_110379

theorem perpendicular_lines_condition (a : ℝ) :
  (¬ a = 1/2 ∨ ¬ a = -1/2) ∧ a * (-4 * a) = -1 ↔ a = 1/2 :=
by
  sorry

end perpendicular_lines_condition_l110_110379


namespace intersection_A_B_range_of_a_l110_110556

-- Problem 1: Prove the intersection of A and B when a = 4
theorem intersection_A_B (a : ℝ) (h : a = 4) :
  { x : ℝ | 5 ≤ x ∧ x ≤ 7 } ∩ { x : ℝ | x ≤ 3 ∨ 5 < x} = {6, 7} :=
by sorry

-- Problem 2: Prove the range of values for a such that A ⊆ B
theorem range_of_a :
  { a : ℝ | (a < 2) ∨ (a > 4) } :=
by sorry

end intersection_A_B_range_of_a_l110_110556


namespace two_categorical_variables_l110_110898

-- Definitions based on the conditions
def smoking (x : String) : Prop := x = "Smoking" ∨ x = "Not smoking"
def sick (y : String) : Prop := y = "Sick" ∨ y = "Not sick"

def category1 (z : String) : Prop := z = "Whether smoking"
def category2 (w : String) : Prop := w = "Whether sick"

-- The main proof statement
theorem two_categorical_variables : 
  (category1 "Whether smoking" ∧ smoking "Smoking" ∧ smoking "Not smoking") ∧
  (category2 "Whether sick" ∧ sick "Sick" ∧ sick "Not sick") →
  "Whether smoking, Whether sick" = "Whether smoking, Whether sick" :=
by
  sorry

end two_categorical_variables_l110_110898


namespace algebraic_expression_l110_110639

theorem algebraic_expression (x : ℝ) (h : x - 1/x = 3) : x^2 + 1/x^2 = 11 := 
by
  sorry

end algebraic_expression_l110_110639


namespace avg_height_trees_l110_110584

-- Assuming heights are defined as h1, h2, ..., h7 with known h2
noncomputable def avgHeight (h1 h2 h3 h4 h5 h6 h7 : ℝ) : ℝ := 
  (h1 + h2 + h3 + h4 + h5 + h6 + h7) / 7

theorem avg_height_trees :
  ∃ (h1 h3 h4 h5 h6 h7 : ℝ), 
    h2 = 15 ∧ 
    (h1 = 2 * h2 ∨ h1 = 3 * h2) ∧
    (h3 = h2 / 3 ∨ h3 = h2 / 2) ∧
    (h4 = 2 * h3 ∨ h4 = 3 * h3 ∨ h4 = h3 / 2 ∨ h4 = h3 / 3) ∧
    (h5 = 2 * h4 ∨ h5 = 3 * h4 ∨ h5 = h4 / 2 ∨ h5 = h4 / 3) ∧
    (h6 = 2 * h5 ∨ h6 = 3 * h5 ∨ h6 = h5 / 2 ∨ h6 = h5 / 3) ∧
    (h7 = 2 * h6 ∨ h7 = 3 * h6 ∨ h7 = h6 / 2 ∨ h7 = h6 / 3) ∧
    avgHeight h1 h2 h3 h4 h5 h6 h7 = 26.4 :=
by
  sorry

end avg_height_trees_l110_110584


namespace average_episodes_per_year_l110_110632

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l110_110632


namespace range_of_a_l110_110932

def P (a : ℝ) : Set ℝ := { x : ℝ | a - 4 < x ∧ x < a + 4 }
def Q : Set ℝ := { x : ℝ | x^2 - 4 * x + 3 < 0 }

theorem range_of_a (a : ℝ) : (∀ x, Q x → P a x) → -1 < a ∧ a < 5 :=
by
  intro h
  sorry

end range_of_a_l110_110932


namespace least_positive_three_digit_multiple_of_8_l110_110549

theorem least_positive_three_digit_multiple_of_8 : ∃ n, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ ∀ m, 100 ≤ m ∧ m < n ∧ m % 8 = 0 → false :=
sorry

end least_positive_three_digit_multiple_of_8_l110_110549


namespace angle_ABC_is_50_l110_110492

theorem angle_ABC_is_50
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)
  (h1 : a = 90)
  (h2 : b = 60)
  (h3 : a + b + c = 200): c = 50 := by
  rw [h1, h2] at h3
  linarith

end angle_ABC_is_50_l110_110492


namespace shed_width_l110_110385

theorem shed_width (backyard_length backyard_width shed_length area_needed : ℝ)
  (backyard_area : backyard_length * backyard_width = 260)
  (sod_area : area_needed = 245)
  (shed_dim : shed_length = 3) :
  (backyard_length * backyard_width - area_needed) / shed_length = 5 :=
by
  -- We need to prove the width of the shed given the conditions
  sorry

end shed_width_l110_110385


namespace neg_pi_lt_neg_three_l110_110312

theorem neg_pi_lt_neg_three (h : Real.pi > 3) : -Real.pi < -3 :=
sorry

end neg_pi_lt_neg_three_l110_110312


namespace cylinder_volume_l110_110529

theorem cylinder_volume (r : ℝ) (h : ℝ) (A : ℝ) (V : ℝ) 
  (sphere_surface_area : A = 256 * Real.pi)
  (cylinder_height : h = 2 * r) 
  (sphere_surface_formula : A = 4 * Real.pi * r^2) 
  (cylinder_volume_formula : V = Real.pi * r^2 * h) : V = 1024 * Real.pi := 
by
  -- Definitions provided as conditions
  sorry

end cylinder_volume_l110_110529


namespace die_vanishing_probability_and_floor_value_l110_110172

/-
Given conditions:
1. The die has four faces labeled 0, 1, 2, 3.
2. When the die lands on a face labeled:
   - 0: the die vanishes.
   - 1: nothing happens (one die remains).
   - 2: the die replicates into 2 dice.
   - 3: the die replicates into 3 dice.
3. All dice (original and replicas) will continuously be rolled.
Prove:
  The value of ⌊10/p⌋ is 24, where p is the probability that all dice will eventually disappear.
-/

theorem die_vanishing_probability_and_floor_value : 
  ∃ (p : ℝ), 
  (p^3 + p^2 - 3 * p + 1 = 0 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p = Real.sqrt 2 - 1) 
  ∧ ⌊10 / p⌋ = 24 := 
    sorry

end die_vanishing_probability_and_floor_value_l110_110172


namespace max_three_numbers_condition_l110_110425

theorem max_three_numbers_condition (n : ℕ) 
  (x : Fin n → ℝ) 
  (h : ∀ (i j k : Fin n), i ≠ j → i ≠ k → j ≠ k → (x i)^2 > (x j) * (x k)) : n ≤ 3 := 
sorry

end max_three_numbers_condition_l110_110425


namespace average_test_score_first_25_percent_l110_110355

theorem average_test_score_first_25_percent (x : ℝ) :
  (0.25 * x) + (0.50 * 65) + (0.25 * 90) = 1 * 75 → x = 80 :=
by
  sorry

end average_test_score_first_25_percent_l110_110355


namespace remainder_of_3_pow_17_mod_7_l110_110241

theorem remainder_of_3_pow_17_mod_7 :
  (3^17 % 7) = 5 :=
by 
  sorry

end remainder_of_3_pow_17_mod_7_l110_110241


namespace value_of_4b_minus_a_l110_110281

theorem value_of_4b_minus_a (a b : ℕ) (h1 : a > b) (h2 : x^2 - 20*x + 96 = (x - a)*(x - b)) : 4*b - a = 20 :=
  sorry

end value_of_4b_minus_a_l110_110281


namespace pyramid_levels_l110_110377

theorem pyramid_levels (n : ℕ) (h : (n * (n + 1) * (2 * n + 1)) / 6 = 225) : n = 6 :=
by
  sorry

end pyramid_levels_l110_110377


namespace jellybeans_red_l110_110321

-- Define the individual quantities of each color of jellybean.
def b := 14
def p := 26
def o := 40
def pk := 7
def y := 21
def T := 237

-- Prove that the number of red jellybeans is 129.
theorem jellybeans_red : T - (b + p + o + pk + y) = 129 := by
  -- (optional: you can include intermediate steps if needed, but it's not required here)
  sorry

end jellybeans_red_l110_110321


namespace cost_of_500_pencils_is_15_dollars_l110_110003

-- Defining the given conditions
def cost_per_pencil_cents : ℕ := 3
def pencils_count : ℕ := 500
def cents_to_dollars : ℕ := 100

-- The proof problem: statement only, no proof provided
theorem cost_of_500_pencils_is_15_dollars :
  (cost_per_pencil_cents * pencils_count) / cents_to_dollars = 15 :=
by
  sorry

end cost_of_500_pencils_is_15_dollars_l110_110003


namespace smallest_three_digit_integer_l110_110546

theorem smallest_three_digit_integer (n : ℕ) (h : 75 * n ≡ 225 [MOD 345]) (hne : n ≥ 100) (hn : n < 1000) : n = 118 :=
sorry

end smallest_three_digit_integer_l110_110546


namespace surface_area_three_dimensional_shape_l110_110677

-- Define the edge length of the largest cube
def edge_length_large : ℕ := 5

-- Define the condition for dividing the edge of the attachment face of the large cube into five equal parts
def divided_into_parts (edge_length : ℕ) (parts : ℕ) : Prop :=
  parts = 5

-- Define the condition that the edge lengths of all three blocks are different
def edge_lengths_different (e1 e2 e3 : ℕ) : Prop :=
  e1 ≠ e2 ∧ e1 ≠ e3 ∧ e2 ≠ e3

-- Define the surface area formula for a cube
def surface_area (s : ℕ) : ℕ :=
  6 * s^2

-- State the problem as a theorem
theorem surface_area_three_dimensional_shape (e1 e2 e3 : ℕ) (h1 : e1 = edge_length_large)
    (h2 : divided_into_parts e1 5) (h3 : edge_lengths_different e1 e2 e3) : 
    surface_area e1 + (surface_area e2 + surface_area e3 - 4 * (e2 * e3)) = 270 :=
sorry

end surface_area_three_dimensional_shape_l110_110677


namespace rachel_should_budget_940_l110_110408

-- Define the prices for Sara's shoes and dress
def sara_shoes : ℝ := 50
def sara_dress : ℝ := 200

-- Define the prices for Tina's shoes and dress
def tina_shoes : ℝ := 70
def tina_dress : ℝ := 150

-- Define the total spending for Sara and Tina, and Rachel's budget
def rachel_budget (sara_shoes sara_dress tina_shoes tina_dress : ℝ) : ℝ := 
  2 * (sara_shoes + sara_dress + tina_shoes + tina_dress)

theorem rachel_should_budget_940 : 
  rachel_budget sara_shoes sara_dress tina_shoes tina_dress = 940 := 
by
  -- skip the proof
  sorry 

end rachel_should_budget_940_l110_110408


namespace correct_statements_l110_110573

-- Define the conditions
def p (a : ℝ) : Prop := a > 2
def q (a : ℝ) : Prop := 2 < a ∧ a < 3
def r (a : ℝ) : Prop := a < 3
def s (a : ℝ) : Prop := a > 1

-- Prove the statements
theorem correct_statements (a : ℝ) : (p a → q a) ∧ (r a → q a) :=
by {
    sorry
}

end correct_statements_l110_110573


namespace arithmetic_sequence_max_sum_l110_110486

noncomputable def max_sum_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  (|a 3| = |a 11| ∧ 
   (∃ d : ℤ, d < 0 ∧ 
   (∀ n, a (n + 1) = a n + d) ∧ 
   (∀ m, S m = (m * (2 * a 1 + (m - 1) * d)) / 2)) →
   ((n = 6) ∨ (n = 7)))

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) :
  max_sum_n a S 6 ∨ max_sum_n a S 7 := sorry

end arithmetic_sequence_max_sum_l110_110486


namespace exists_special_cubic_polynomial_l110_110291

theorem exists_special_cubic_polynomial :
  ∃ P : Polynomial ℝ, 
    Polynomial.degree P = 3 ∧ 
    (∀ x : ℝ, Polynomial.IsRoot P x → x > 0) ∧
    (∀ x : ℝ, Polynomial.IsRoot (Polynomial.derivative P) x → x < 0) ∧
    (∃ x y : ℝ, Polynomial.IsRoot P x ∧ Polynomial.IsRoot (Polynomial.derivative P) y ∧ x ≠ y) :=
by
  sorry

end exists_special_cubic_polynomial_l110_110291


namespace speed_of_man_in_still_water_l110_110242

variables (v_m v_s : ℝ)

theorem speed_of_man_in_still_water :
  (v_m + v_s) * 5 = 36 ∧ (v_m - v_s) * 7 = 22 → v_m = 5.17 :=
by 
  sorry

end speed_of_man_in_still_water_l110_110242


namespace watch_cost_price_l110_110439

theorem watch_cost_price (CP : ℝ) (h1 : (0.90 * CP) + 280 = 1.04 * CP) : CP = 2000 := 
by 
  sorry

end watch_cost_price_l110_110439


namespace juice_cans_count_l110_110300

theorem juice_cans_count :
  let original_price := 12 
  let discount := 2 
  let tub_sale_price := original_price - discount 
  let tub_quantity := 2 
  let ice_cream_total := tub_quantity * tub_sale_price 
  let total_payment := 24 
  let juice_cost_per_5cans := 2 
  let remaining_amount := total_payment - ice_cream_total 
  let sets_of_juice_cans := remaining_amount / juice_cost_per_5cans 
  let cans_per_set := 5 
  2 * cans_per_set = 10 :=
by
  sorry

end juice_cans_count_l110_110300


namespace incorrect_proposition_example_l110_110256

theorem incorrect_proposition_example (p q : Prop) (h : ¬ (p ∧ q)) : ¬ (¬p ∧ ¬q) :=
by
  sorry

end incorrect_proposition_example_l110_110256


namespace emily_initial_toys_l110_110891

theorem emily_initial_toys : ∃ (initial_toys : ℕ), initial_toys = 3 + 4 :=
by
  existsi 7
  sorry

end emily_initial_toys_l110_110891


namespace journey_distance_l110_110962

theorem journey_distance (t : ℝ) : 
  t = 20 →
  ∃ D : ℝ, (D / 20 + D / 30 = t) ∧ D = 240 :=
by
  sorry

end journey_distance_l110_110962


namespace polynomial_at_3_l110_110125

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_at_3 : f 3 = 1641 := 
by
  -- proof would go here
  sorry

end polynomial_at_3_l110_110125


namespace new_ratio_of_milk_to_water_l110_110474

theorem new_ratio_of_milk_to_water
  (total_volume : ℕ) (initial_ratio_milk : ℕ) (initial_ratio_water : ℕ) (added_water : ℕ)
  (h_total_volume : total_volume = 45)
  (h_initial_ratio : initial_ratio_milk = 4 ∧ initial_ratio_water = 1)
  (h_added_water : added_water = 11) :
  let initial_milk := (initial_ratio_milk * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let initial_water := (initial_ratio_water * total_volume) / (initial_ratio_milk + initial_ratio_water)
  let new_water := initial_water + added_water
  let gcd := Nat.gcd initial_milk new_water
  (initial_milk / gcd : ℕ) = 9 ∧ (new_water / gcd : ℕ) = 5 :=
by
  sorry

end new_ratio_of_milk_to_water_l110_110474


namespace geometric_sequence_condition_l110_110645

theorem geometric_sequence_condition (A B q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hSn_def : ∀ n, S n = A * q^n + B) (hq_ne_zero : q ≠ 0) :
  (∀ n, a n = S n - S (n-1)) → (A = -B) ↔ (∀ n, a n = A * (q - 1) * q^(n-1)) := 
sorry

end geometric_sequence_condition_l110_110645


namespace relationship_y1_y2_y3_l110_110532

-- Conditions
def y1 := -((-4)^2) + 5
def y2 := -((-1)^2) + 5
def y3 := -(2^2) + 5

-- Statement to be proved
theorem relationship_y1_y2_y3 : y2 > y3 ∧ y3 > y1 := by
  sorry

end relationship_y1_y2_y3_l110_110532


namespace ten_percent_eq_l110_110558

variable (s t : ℝ)

def ten_percent_of (x : ℝ) : ℝ := 0.1 * x

theorem ten_percent_eq (h : ten_percent_of s = t) : s = 10 * t :=
by sorry

end ten_percent_eq_l110_110558


namespace real_and_imag_parts_of_z_l110_110353

noncomputable def real_part (z : ℂ) : ℝ := z.re
noncomputable def imag_part (z : ℂ) : ℝ := z.im

theorem real_and_imag_parts_of_z :
  ∀ (i : ℂ), i * i = -1 → 
  ∀ (z : ℂ), z = i * (-1 + 2 * i) → real_part z = -2 ∧ imag_part z = -1 :=
by 
  intros i hi z hz
  sorry

end real_and_imag_parts_of_z_l110_110353


namespace triangle_angle_A_eq_pi_div_3_triangle_area_l110_110662

variable (A B C a b c : ℝ)
variable (S : ℝ)

-- First part: Proving A = π / 3
theorem triangle_angle_A_eq_pi_div_3 (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                                      (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : A > 0) (h6 : A < Real.pi) :
  A = Real.pi / 3 :=
sorry

-- Second part: Finding the area of the triangle
theorem triangle_area (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos A)
                      (h2 : b + c = Real.sqrt 10) (h3 : a = 2) (h4 : A = Real.pi / 3) :
  S = Real.sqrt 3 / 2 :=
sorry

end triangle_angle_A_eq_pi_div_3_triangle_area_l110_110662


namespace bounded_area_l110_110481

noncomputable def f (x : ℝ) : ℝ := (x + Real.sqrt (x^2 + 1))^(1/3) + (x - Real.sqrt (x^2 + 1))^(1/3)

def g (y : ℝ) : ℝ := y + 1

theorem bounded_area : 
  (∫ y in (0:ℝ)..(1:ℝ), (g y - f (g y))) = (5/8 : ℝ) := by
  sorry

end bounded_area_l110_110481


namespace snakes_in_each_cage_l110_110403

theorem snakes_in_each_cage (total_snakes : ℕ) (total_cages : ℕ) (h_snakes: total_snakes = 4) (h_cages: total_cages = 2) 
  (h_even_distribution : (total_snakes % total_cages) = 0) : (total_snakes / total_cages) = 2 := 
by sorry

end snakes_in_each_cage_l110_110403


namespace pages_per_day_read_l110_110191

theorem pages_per_day_read (start_date : ℕ) (end_date : ℕ) (total_pages : ℕ) (fraction_covered : ℚ) (pages_read : ℕ) (days : ℕ) :
  start_date = 1 →
  end_date = 12 →
  total_pages = 144 →
  fraction_covered = 2/3 →
  pages_read = fraction_covered * total_pages →
  days = end_date - start_date + 1 →
  pages_read / days = 8 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end pages_per_day_read_l110_110191


namespace probability_y_greater_than_x_equals_3_4_l110_110664

noncomputable def probability_y_greater_than_x : Real :=
  let total_area : Real := 1000 * 4034
  let triangle_area : Real := 0.5 * 1000 * (4034 - 1000)
  let rectangle_area : Real := 3034 * 4034
  let area_y_greater_than_x : Real := triangle_area + rectangle_area
  area_y_greater_than_x / total_area

theorem probability_y_greater_than_x_equals_3_4 :
  probability_y_greater_than_x = 3 / 4 :=
sorry

end probability_y_greater_than_x_equals_3_4_l110_110664


namespace min_people_in_photographs_l110_110984

-- Definitions based on conditions
def photographs := (List (Nat × Nat × Nat))
def menInCenter (photos : photographs) := photos.map (fun (c, _, _) => c)

-- Condition: there are 10 photographs each with a distinct man in the center
def valid_photographs (photos: photographs) :=
  photos.length = 10 ∧ photos.map (fun (c, _, _) => c) = List.range 10

-- Theorem to be proved: The minimum number of different people in the photographs is at least 16
theorem min_people_in_photographs (photos: photographs) (h : valid_photographs photos) : 
  ∃ people : Finset Nat, people.card ≥ 16 := 
sorry

end min_people_in_photographs_l110_110984


namespace determine_n_l110_110078

theorem determine_n (n : ℕ) (h : 9^4 = 3^n) : n = 8 :=
by {
  sorry
}

end determine_n_l110_110078


namespace problem_statements_l110_110802

theorem problem_statements :
  let S1 := ∀ (x : ℤ) (k : ℤ), x = 2 * k + 1 → (x % 2 = 1)
  let S2 := (∀ (x : ℝ), x > 2 → x > 1) 
            ∧ (∀ (x : ℝ), x > 1 → (x ≥ 2 ∨ x < 2)) 
  let S3 := ∀ (x : ℝ), ¬(∃ (x : ℝ), ∃ (y : ℝ), y = x^2 + 1 ∧ x = y)
  let S4 := ¬(∀ (x : ℝ), x > 1 → x^2 - x > 0) → (∃ (x : ℝ), x > 1 ∧ x^2 - x ≤ 0)
  (S1 ∧ S2 ∧ S3 ∧ ¬S4) := by
    sorry

end problem_statements_l110_110802


namespace number_of_readers_who_read_both_l110_110742

theorem number_of_readers_who_read_both (S L B total : ℕ) (hS : S = 250) (hL : L = 550) (htotal : total = 650) (h : S + L - B = total) : B = 150 :=
by {
  /-
  Given:
  S = 250 (number of readers who read science fiction)
  L = 550 (number of readers who read literary works)
  total = 650 (total number of readers)
  h : S + L - B = total (relationship between sets)
  We need to prove: B = 150
  -/
  sorry
}

end number_of_readers_who_read_both_l110_110742


namespace ones_digit_11_pow_l110_110925

theorem ones_digit_11_pow (n : ℕ) (hn : n > 0) : (11^n % 10) = 1 := by
  sorry

end ones_digit_11_pow_l110_110925


namespace valuing_fraction_l110_110766

variable {x y : ℚ}

theorem valuing_fraction (h : x / y = 1 / 2) : (x - y) / (x + y) = -1 / 3 :=
by
  sorry

end valuing_fraction_l110_110766


namespace elvin_fixed_monthly_charge_l110_110938

theorem elvin_fixed_monthly_charge
  (F C : ℝ) 
  (h1 : F + C = 50) 
  (h2 : F + 2 * C = 76) : 
  F = 24 := 
sorry

end elvin_fixed_monthly_charge_l110_110938


namespace non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l110_110028

open Set

noncomputable def f : ℝ → ℝ := sorry

theorem non_decreasing_f (x y : ℝ) (h : x < y) (hx : x ∈ Icc (0 : ℝ) 2) (hy : y ∈ Icc (0 : ℝ) 2) : f x ≤ f y := sorry

theorem f_equal_2_at_2 : f 2 = 2 := sorry

theorem addition_property (x : ℝ) (hx : x ∈ Icc (0 :ℝ) 2) : f x + f (2 - x) = 2 := sorry

theorem under_interval_rule (x : ℝ) (hx : x ∈ Icc (1.5 :ℝ) 2) : f x ≤ 2 * (x - 1) := sorry

theorem final_statement : ∀ x ∈ Icc (0:ℝ) 1, f (f x) ∈ Icc (0:ℝ) 1 := sorry

end non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l110_110028


namespace remainder_of_large_number_l110_110710

theorem remainder_of_large_number (N : ℕ) (hN : N = 123456789012): 
  N % 360 = 108 :=
by
  have h1 : N % 4 = 0 := by 
    sorry
  have h2 : N % 9 = 3 := by 
    sorry
  have h3 : N % 10 = 2 := by
    sorry
  sorry

end remainder_of_large_number_l110_110710


namespace apples_remaining_l110_110238

variable (initial_apples : ℕ)
variable (picked_day1 : ℕ)
variable (picked_day2 : ℕ)
variable (picked_day3 : ℕ)

-- Given conditions
def condition1 : initial_apples = 200 := sorry
def condition2 : picked_day1 = initial_apples / 5 := sorry
def condition3 : picked_day2 = 2 * picked_day1 := sorry
def condition4 : picked_day3 = picked_day1 + 20 := sorry

-- Prove the total number of apples remaining is 20
theorem apples_remaining (H1 : initial_apples = 200) 
  (H2 : picked_day1 = initial_apples / 5) 
  (H3 : picked_day2 = 2 * picked_day1)
  (H4 : picked_day3 = picked_day1 + 20) : 
  initial_apples - (picked_day1 + picked_day2 + picked_day3) = 20 := 
by
  sorry

end apples_remaining_l110_110238


namespace lcm_inequality_l110_110062

open Nat

theorem lcm_inequality (n : ℕ) (h : n ≥ 5 ∧ Odd n) :
  Nat.lcm n (n + 1) * (n + 2) > Nat.lcm (n + 1) (n + 2) * (n + 3) := by
  sorry

end lcm_inequality_l110_110062


namespace not_p_is_sufficient_but_not_necessary_for_not_q_l110_110548

variable (x : ℝ)

def proposition_p : Prop := |x| < 2
def proposition_q : Prop := x^2 - x - 2 < 0

theorem not_p_is_sufficient_but_not_necessary_for_not_q :
  (¬ proposition_p x) → (¬ proposition_q x) ∧ (¬ proposition_q x) → (¬ proposition_p x) → False := by
  sorry

end not_p_is_sufficient_but_not_necessary_for_not_q_l110_110548


namespace weighted_average_correct_l110_110176

-- Define the marks
def english_marks : ℝ := 76
def mathematics_marks : ℝ := 65
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 85

-- Define the weightages
def english_weightage : ℝ := 0.20
def mathematics_weightage : ℝ := 0.25
def physics_weightage : ℝ := 0.25
def chemistry_weightage : ℝ := 0.15
def biology_weightage : ℝ := 0.15

-- Define the weighted sum calculation
def weighted_sum : ℝ :=
  english_marks * english_weightage + 
  mathematics_marks * mathematics_weightage + 
  physics_marks * physics_weightage + 
  chemistry_marks * chemistry_weightage + 
  biology_marks * biology_weightage

-- Define the theorem statement: the weighted average marks
theorem weighted_average_correct : weighted_sum = 74.75 :=
by
  sorry

end weighted_average_correct_l110_110176


namespace Zhang_Hai_average_daily_delivery_is_37_l110_110102

theorem Zhang_Hai_average_daily_delivery_is_37
  (d1_packages : ℕ) (d1_count : ℕ)
  (d2_packages : ℕ) (d2_count : ℕ)
  (d3_packages : ℕ) (d3_count : ℕ)
  (total_days : ℕ) 
  (h1 : d1_packages = 41) (h2 : d1_count = 1)
  (h3 : d2_packages = 35) (h4 : d2_count = 2)
  (h5 : d3_packages = 37) (h6 : d3_count = 4)
  (h7 : total_days = 7) :
  (d1_count * d1_packages + d2_count * d2_packages + d3_count * d3_packages) / total_days = 37 := 
by sorry

end Zhang_Hai_average_daily_delivery_is_37_l110_110102


namespace map_distance_representation_l110_110997

theorem map_distance_representation
  (d_map : ℕ) (d_actual : ℕ) (conversion_factor : ℕ) (final_length_map : ℕ):
  d_map = 10 →
  d_actual = 80 →
  conversion_factor = d_actual / d_map →
  final_length_map = 18 →
  (final_length_map * conversion_factor) = 144 :=
by
  intros h1 h2 h3 h4
  sorry

end map_distance_representation_l110_110997


namespace arithmetic_sequence_monotone_l110_110114

theorem arithmetic_sequence_monotone (a : ℕ → ℝ) (d : ℝ) (h_arithmetic : ∀ n, a (n + 1) - a n = d) :
  (a 2 > a 1) ↔ (∀ n, a (n + 1) > a n) :=
by 
  sorry

end arithmetic_sequence_monotone_l110_110114


namespace three_hundredth_term_without_squares_l110_110693

noncomputable def sequence_without_squares (n : ℕ) : ℕ :=
(n + (n / Int.natAbs (Int.sqrt (n.succ - 1))))

theorem three_hundredth_term_without_squares : 
  sequence_without_squares 300 = 307 :=
sorry

end three_hundredth_term_without_squares_l110_110693
