import Mathlib

namespace two_digit_numbers_sum_reversed_l565_56566

theorem two_digit_numbers_sum_reversed (a b : ℕ) (h₁ : 0 ≤ a) (h₂ : a ≤ 9) (h₃ : 0 ≤ b) (h₄ : b ≤ 9) (h₅ : a + b = 12) :
  ∃ n : ℕ, n = 7 := 
sorry

end two_digit_numbers_sum_reversed_l565_56566


namespace probability_all_three_dice_twenty_l565_56506

theorem probability_all_three_dice_twenty (d1 d2 d3 d4 d5 : ℕ)
  (h1 : 1 ≤ d1 ∧ d1 ≤ 20) (h2 : 1 ≤ d2 ∧ d2 ≤ 20) (h3 : 1 ≤ d3 ∧ d3 ≤ 20)
  (h4 : 1 ≤ d4 ∧ d4 ≤ 20) (h5 : 1 ≤ d5 ∧ d5 ≤ 20)
  (h6 : d1 = 20) (h7 : d2 = 19)
  (h8 : (if d1 = 20 then 1 else 0) + (if d2 = 20 then 1 else 0) +
        (if d3 = 20 then 1 else 0) + (if d4 = 20 then 1 else 0) +
        (if d5 = 20 then 1 else 0) ≥ 3) :
  (1 / 58 : ℚ) = (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) /
                 ((if d3 = 20 ∧ d4 = 20 then 19 else 0) +
                  (if d3 = 20 ∧ d5 = 20 then 19 else 0) +
                  (if d4 = 20 ∧ d5 = 20 then 19 else 0) + 
                  (if d3 = 20 ∧ d4 = 20 ∧ d5 = 20 then 1 else 0) : ℚ) :=
sorry

end probability_all_three_dice_twenty_l565_56506


namespace quadratic_real_roots_range_l565_56568

theorem quadratic_real_roots_range (m : ℝ) : (∃ x y : ℝ, x ≠ y ∧ mx^2 + 2*x + 1 = 0 ∧ yx^2 + 2*y + 1 = 0) → m ≤ 1 ∧ m ≠ 0 :=
by 
sorry

end quadratic_real_roots_range_l565_56568


namespace total_ninja_stars_l565_56594

variable (e c j : ℕ)
variable (H1 : e = 4) -- Eric has 4 ninja throwing stars
variable (H2 : c = 2 * e) -- Chad has twice as many ninja throwing stars as Eric
variable (H3 : j = c - 2) -- Chad sells 2 ninja stars to Jeff
variable (H4 : j = 6) -- Jeff now has 6 ninja throwing stars

theorem total_ninja_stars :
  e + (c - 2) + 6 = 16 :=
by
  sorry

end total_ninja_stars_l565_56594


namespace gcd_of_72_90_120_l565_56541

theorem gcd_of_72_90_120 : Nat.gcd (Nat.gcd 72 90) 120 = 6 := 
by 
  have h1 : 72 = 2^3 * 3^2 := by norm_num
  have h2 : 90 = 2 * 3^2 * 5 := by norm_num
  have h3 : 120 = 2^3 * 3 * 5 := by norm_num
  sorry

end gcd_of_72_90_120_l565_56541


namespace necessary_but_not_sufficient_condition_for_x_equals_0_l565_56550

theorem necessary_but_not_sufficient_condition_for_x_equals_0 (x : ℝ) :
  ((2 * x - 1) * x = 0 → x = 0 ∨ x = 1 / 2) ∧ (x = 0 → (2 * x - 1) * x = 0) ∧ ¬((2 * x - 1) * x = 0 → x = 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_for_x_equals_0_l565_56550


namespace infinite_solutions_of_linear_system_l565_56571

theorem infinite_solutions_of_linear_system :
  ∀ (x y : ℝ), (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = 10) → ∃ (k : ℝ), x = (3 * k + 5) / 2 :=
by
  sorry

end infinite_solutions_of_linear_system_l565_56571


namespace longest_train_length_l565_56517

theorem longest_train_length :
  ∀ (speedA : ℝ) (timeA : ℝ) (speedB : ℝ) (timeB : ℝ) (speedC : ℝ) (timeC : ℝ),
  speedA = 60 * (5 / 18) → timeA = 5 →
  speedB = 80 * (5 / 18) → timeB = 7 →
  speedC = 50 * (5 / 18) → timeC = 9 →
  speedB * timeB > speedA * timeA ∧ speedB * timeB > speedC * timeC ∧ speedB * timeB = 155.54 := by
  sorry

end longest_train_length_l565_56517


namespace total_polled_votes_correct_l565_56596

variable (V : ℕ) -- Valid votes

-- Condition: One candidate got 30% of the valid votes
variable (C1_votes : ℕ) (C2_votes : ℕ)
variable (H1 : C1_votes = (3 * V) / 10)

-- Condition: The other candidate won by 5000 votes
variable (H2 : C2_votes = C1_votes + 5000)

-- Condition: One candidate got 70% of the valid votes
variable (H3 : C2_votes = (7 * V) / 10)

-- Condition: 100 votes were invalid
variable (invalid_votes : ℕ := 100)

-- Total polled votes (valid + invalid)
def total_polled_votes := V + invalid_votes

theorem total_polled_votes_correct 
  (V : ℕ) 
  (H1 : C1_votes = (3 * V) / 10) 
  (H2 : C2_votes = C1_votes + 5000) 
  (H3 : C2_votes = (7 * V) / 10) 
  (invalid_votes : ℕ := 100) : 
  total_polled_votes V = 12600 :=
by
  -- The steps of the proof are omitted
  sorry

end total_polled_votes_correct_l565_56596


namespace candy_bar_cost_correct_l565_56531

def quarters : ℕ := 4
def dimes : ℕ := 3
def nickel : ℕ := 1
def change_received : ℕ := 4

def total_paid : ℕ :=
  (quarters * 25) + (dimes * 10) + (nickel * 5)

def candy_bar_cost : ℕ :=
  total_paid - change_received

theorem candy_bar_cost_correct : candy_bar_cost = 131 := by
  sorry

end candy_bar_cost_correct_l565_56531


namespace missed_the_bus_by_5_minutes_l565_56560

theorem missed_the_bus_by_5_minutes 
    (usual_time : ℝ)
    (new_time : ℝ)
    (h1 : usual_time = 20)
    (h2 : new_time = usual_time * (5 / 4)) : 
    new_time - usual_time = 5 := 
by
  sorry

end missed_the_bus_by_5_minutes_l565_56560


namespace bottles_more_than_apples_l565_56589

def bottles_regular : ℕ := 72
def bottles_diet : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := bottles_regular + bottles_diet

theorem bottles_more_than_apples : (total_bottles - apples) = 26 := by
  sorry

end bottles_more_than_apples_l565_56589


namespace abs_diff_eq_two_l565_56585

def equation (x y : ℝ) : Prop := y^2 + x^4 = 2 * x^2 * y + 1

theorem abs_diff_eq_two (a b e : ℝ) (ha : equation e a) (hb : equation e b) (hab : a ≠ b) :
  |a - b| = 2 :=
sorry

end abs_diff_eq_two_l565_56585


namespace factorize_expression_l565_56540

theorem factorize_expression (a b : ℝ) : 
  a^3 + 2 * a^2 * b + a * b^2 = a * (a + b)^2 := by sorry

end factorize_expression_l565_56540


namespace max_earnings_mary_l565_56513

def wage_rate : ℝ := 8
def first_hours : ℕ := 20
def max_hours : ℕ := 80
def regular_tip_rate : ℝ := 2
def overtime_rate_increase : ℝ := 1.25
def overtime_tip_rate : ℝ := 3
def overtime_bonus_threshold : ℕ := 5
def overtime_bonus_amount : ℝ := 20

noncomputable def total_earnings (hours : ℕ) : ℝ :=
  let regular_hours := min hours first_hours
  let overtime_hours := if hours > first_hours then hours - first_hours else 0
  let overtime_blocks := overtime_hours / overtime_bonus_threshold
  let regular_earnings := regular_hours * (wage_rate + regular_tip_rate)
  let overtime_earnings := overtime_hours * (wage_rate * overtime_rate_increase + overtime_tip_rate)
  let bonuses := (overtime_blocks) * overtime_bonus_amount
  regular_earnings + overtime_earnings + bonuses

theorem max_earnings_mary : total_earnings max_hours = 1220 := by
  sorry

end max_earnings_mary_l565_56513


namespace equation_solution_l565_56587

theorem equation_solution (x : ℝ) (h : 8^(Real.log 5 / Real.log 8) = 10 * x + 3) : x = 1 / 5 :=
sorry

end equation_solution_l565_56587


namespace min_value_of_reciprocal_sum_l565_56529

theorem min_value_of_reciprocal_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 1) :
  ∃ z, (z = 3 + 2 * Real.sqrt 2) ∧ (∀ z', (z' = 1 / x + 1 / y) → z ≤ z') :=
sorry

end min_value_of_reciprocal_sum_l565_56529


namespace extreme_value_a_one_range_of_a_l565_56597

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + 3

theorem extreme_value_a_one :
  ∀ x > 0, f x 1 ≤ f 1 1 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≤ 0) → a ≥ Real.exp 2 :=
sorry

end extreme_value_a_one_range_of_a_l565_56597


namespace abc_sub_c_minus_2023_eq_2023_l565_56525

theorem abc_sub_c_minus_2023_eq_2023 (a b c : ℝ) (h : a * b = 1) : 
  a * b * c - (c - 2023) = 2023 := 
by sorry

end abc_sub_c_minus_2023_eq_2023_l565_56525


namespace jesse_bananas_total_l565_56598

theorem jesse_bananas_total (friends : ℝ) (bananas_per_friend : ℝ) (friends_eq : friends = 3) (bananas_per_friend_eq : bananas_per_friend = 21) : 
  friends * bananas_per_friend = 63 := by
  rw [friends_eq, bananas_per_friend_eq]
  norm_num

end jesse_bananas_total_l565_56598


namespace calc_f_g_h_2_l565_56539

def f (x : ℕ) : ℕ := x + 5
def g (x : ℕ) : ℕ := x^2 - 8
def h (x : ℕ) : ℕ := 2 * x + 1

theorem calc_f_g_h_2 : f (g (h 2)) = 22 := by
  sorry

end calc_f_g_h_2_l565_56539


namespace oj_fraction_is_11_over_30_l565_56510

-- Define the capacity of each pitcher
def pitcher_capacity : ℕ := 600

-- Define the fraction of orange juice in each pitcher
def fraction_oj_pitcher1 : ℚ := 1 / 3
def fraction_oj_pitcher2 : ℚ := 2 / 5

-- Define the amount of orange juice in each pitcher
def oj_amount_pitcher1 := pitcher_capacity * fraction_oj_pitcher1
def oj_amount_pitcher2 := pitcher_capacity * fraction_oj_pitcher2

-- Define the total amount of orange juice after both pitchers are poured into the large container
def total_oj_amount := oj_amount_pitcher1 + oj_amount_pitcher2

-- Define the total volume of the mixture in the large container
def total_mixture_volume := 2 * pitcher_capacity

-- Define the fraction of the mixture that is orange juice
def oj_fraction_in_mixture := total_oj_amount / total_mixture_volume

-- Prove that the fraction of the mixture that is orange juice is 11/30
theorem oj_fraction_is_11_over_30 : oj_fraction_in_mixture = 11 / 30 := by
  sorry

end oj_fraction_is_11_over_30_l565_56510


namespace price_25_bag_l565_56535

noncomputable def price_per_bag_25 : ℝ := 28.97

def price_per_bag_5 : ℝ := 13.85
def price_per_bag_10 : ℝ := 20.42

def total_cost (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ) : ℝ :=
  n5 * p5 + n10 * p10 + n25 * p25

theorem price_25_bag :
  ∃ (p5 p10 p25 : ℝ) (n5 n10 n25 : ℕ),
    p5 = price_per_bag_5 ∧
    p10 = price_per_bag_10 ∧
    p25 = price_per_bag_25 ∧
    65 ≤ 5 * n5 + 10 * n10 + 25 * n25 ∧
    5 * n5 + 10 * n10 + 25 * n25 ≤ 80 ∧
    total_cost p5 p10 p25 n5 n10 n25 = 98.77 :=
by
  sorry

end price_25_bag_l565_56535


namespace rita_daily_minimum_payment_l565_56532

theorem rita_daily_minimum_payment (total_cost down_payment balance daily_payment : ℝ) 
    (h1 : total_cost = 120)
    (h2 : down_payment = total_cost / 2)
    (h3 : balance = total_cost - down_payment)
    (h4 : daily_payment = balance / 10) : daily_payment = 6 :=
by
  sorry

end rita_daily_minimum_payment_l565_56532


namespace cost_of_500_pencils_in_dollars_l565_56584

def cost_of_pencil := 3 -- cost of 1 pencil in cents
def pencils_quantity := 500 -- number of pencils
def cents_in_dollar := 100 -- number of cents in 1 dollar

theorem cost_of_500_pencils_in_dollars :
  (pencils_quantity * cost_of_pencil) / cents_in_dollar = 15 := by
    sorry

end cost_of_500_pencils_in_dollars_l565_56584


namespace jessica_minimal_withdrawal_l565_56562

theorem jessica_minimal_withdrawal 
  (initial_withdrawal : ℝ)
  (initial_fraction : ℝ)
  (minimum_balance : ℝ)
  (deposit_fraction : ℝ)
  (after_withdrawal_balance : ℝ)
  (deposit_amount : ℝ)
  (current_balance : ℝ) :
  initial_withdrawal = 400 →
  initial_fraction = 2/5 →
  minimum_balance = 300 →
  deposit_fraction = 1/4 →
  after_withdrawal_balance = 1000 - initial_withdrawal →
  deposit_amount = deposit_fraction * after_withdrawal_balance →
  current_balance = after_withdrawal_balance + deposit_amount →
  current_balance - minimum_balance ≥ 0 →
  0 = 0 :=
by
  sorry

end jessica_minimal_withdrawal_l565_56562


namespace fraction_students_received_Bs_l565_56591

theorem fraction_students_received_Bs (fraction_As : ℝ) (fraction_As_or_Bs : ℝ) (h1 : fraction_As = 0.7) (h2 : fraction_As_or_Bs = 0.9) :
  fraction_As_or_Bs - fraction_As = 0.2 :=
by
  sorry

end fraction_students_received_Bs_l565_56591


namespace ball_reaches_height_l565_56570

theorem ball_reaches_height (h₀ : ℝ) (ratio : ℝ) (target_height : ℝ) (bounces : ℕ) 
  (initial_height : h₀ = 16) 
  (bounce_ratio : ratio = 1/3) 
  (target : target_height = 2) 
  (bounce_count : bounces = 7) :
  h₀ * (ratio ^ bounces) < target_height := 
sorry

end ball_reaches_height_l565_56570


namespace multiple_time_second_artifact_is_three_l565_56508

-- Define the conditions as Lean definitions
def months_in_year : ℕ := 12
def total_time_both_artifacts_years : ℕ := 10
def total_time_first_artifact_months : ℕ := 6 + 24

-- Convert total time of both artifacts from years to months
def total_time_both_artifacts_months : ℕ := total_time_both_artifacts_years * months_in_year

-- Define the time for the second artifact
def time_second_artifact_months : ℕ :=
  total_time_both_artifacts_months - total_time_first_artifact_months

-- Define the sought multiple
def multiple_second_first : ℕ :=
  time_second_artifact_months / total_time_first_artifact_months

-- The theorem stating the required proof
theorem multiple_time_second_artifact_is_three :
  multiple_second_first = 3 :=
by
  sorry

end multiple_time_second_artifact_is_three_l565_56508


namespace total_snakes_in_park_l565_56572

theorem total_snakes_in_park :
  ∀ (pythons boa_constrictors rattlesnakes total_snakes : ℕ),
    boa_constrictors = 40 →
    pythons = 3 * boa_constrictors →
    rattlesnakes = 40 →
    total_snakes = boa_constrictors + pythons + rattlesnakes →
    total_snakes = 200 :=
by
  intros pythons boa_constrictors rattlesnakes total_snakes h1 h2 h3 h4
  rw [h1, h3] at h4
  rw [h2] at h4
  sorry

end total_snakes_in_park_l565_56572


namespace reduced_price_is_55_l565_56546

variables (P R : ℝ) (X : ℕ)

-- Conditions
def condition1 : R = 0.75 * P := sorry
def condition2 : P * X = 1100 := sorry
def condition3 : 0.75 * P * (X + 5) = 1100 := sorry

-- Theorem
theorem reduced_price_is_55 (P R : ℝ) (X : ℕ) (h1 : R = 0.75 * P) (h2 : P * X = 1100) (h3 : 0.75 * P * (X + 5) = 1100) :
  R = 55 :=
sorry

end reduced_price_is_55_l565_56546


namespace inequality_holds_for_any_xyz_l565_56501

theorem inequality_holds_for_any_xyz (x y z : ℝ) : 
  x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) := 
by 
  sorry

end inequality_holds_for_any_xyz_l565_56501


namespace hyperbola_eccentricity_l565_56516

/-- Given a hyperbola with the equation x^2/a^2 - y^2/b^2 = 1, point B(0, b),
the line F1B intersects with the two asymptotes at points P and Q. 
We are given that vector QP = 4 * vector PF1. Prove that the eccentricity 
of the hyperbola is 3/2. -/
theorem hyperbola_eccentricity (a b : ℝ) (h_a : a > 0) (h_b : b > 0) 
  (F1 : ℝ × ℝ) (B : ℝ × ℝ) (P Q : ℝ × ℝ) 
  (h_F1 : F1 = (-c, 0)) (h_B : B = (0, b)) 
  (h_int_P : P = (-a * c / (c + a), b * c / (c + a)))
  (h_int_Q : Q = (a * c / (c - a), b * c / (c - a)))
  (h_vec : (Q.1 - P.1, Q.2 - P.2) = (4 * (P.1 - F1.1), 4 * (P.2 - F1.2))) :
  (eccentricity : ℝ) = 3 / 2 :=
sorry

end hyperbola_eccentricity_l565_56516


namespace age_difference_l565_56551

variable (A B C D : ℕ)

theorem age_difference (h₁ : A + B > B + C) (h₂ : C = A - 15) : (A + B) - (B + C) = 15 :=
by
  sorry

end age_difference_l565_56551


namespace sally_garden_area_l565_56504

theorem sally_garden_area :
  ∃ (a b : ℕ), 2 * (a + b) = 24 ∧ b + 1 = 3 * (a + 1) ∧ 
     (3 * (a - 1) * 3 * (b - 1) = 297) :=
by {
  sorry
}

end sally_garden_area_l565_56504


namespace northbound_vehicle_count_l565_56536

theorem northbound_vehicle_count :
  ∀ (southbound_speed northbound_speed : ℝ) (vehicles_passed : ℕ) 
  (time_minutes : ℝ) (section_length : ℝ), 
  southbound_speed = 70 → northbound_speed = 50 → vehicles_passed = 30 → time_minutes = 10
  → section_length = 150
  → (vehicles_passed / ((southbound_speed + northbound_speed) * (time_minutes / 60))) * section_length = 270 :=
by sorry

end northbound_vehicle_count_l565_56536


namespace directrix_of_parabola_l565_56527

theorem directrix_of_parabola :
  ∀ (x y : ℝ), y = (x^2 - 4 * x + 3) / 8 → y = -9 / 8 :=
by
  sorry

end directrix_of_parabola_l565_56527


namespace mail_distribution_l565_56593

def total_mail : ℕ := 2758
def mail_for_first_block : ℕ := 365
def mail_for_second_block : ℕ := 421
def remaining_mail : ℕ := total_mail - (mail_for_first_block + mail_for_second_block)
def remaining_blocks : ℕ := 3
def mail_per_remaining_block : ℕ := remaining_mail / remaining_blocks

theorem mail_distribution :
  mail_per_remaining_block = 657 := by
  sorry

end mail_distribution_l565_56593


namespace fraction_is_integer_l565_56534

theorem fraction_is_integer (b t : ℤ) (hb : b ≠ 1) :
  ∃ (k : ℤ), (t^5 - 5 * b + 4) = k * (b^2 - 2 * b + 1) :=
by 
  sorry

end fraction_is_integer_l565_56534


namespace negation_proposition_l565_56561

theorem negation_proposition (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
sorry

end negation_proposition_l565_56561


namespace abc_inequality_l565_56547

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  (a / (1 + a * b))^2 + (b / (1 + b * c))^2 + (c / (1 + c * a))^2 ≥ 3 / 4 :=
by
  sorry

end abc_inequality_l565_56547


namespace complementary_angle_of_60_l565_56580

theorem complementary_angle_of_60 (a : ℝ) : 
  (∀ (a b : ℝ), a + b = 180 → a = 60 → b = 120) := 
by
  sorry

end complementary_angle_of_60_l565_56580


namespace fraction_of_historical_fiction_new_releases_l565_56588

theorem fraction_of_historical_fiction_new_releases
  (total_books : ℕ)
  (historical_fiction_percentage : ℝ := 0.4)
  (historical_fiction_new_releases_percentage : ℝ := 0.4)
  (other_genres_new_releases_percentage : ℝ := 0.7)
  (total_historical_fiction_books := total_books * historical_fiction_percentage)
  (total_other_books := total_books * (1 - historical_fiction_percentage))
  (historical_fiction_new_releases := total_historical_fiction_books * historical_fiction_new_releases_percentage)
  (other_genres_new_releases := total_other_books * other_genres_new_releases_percentage)
  (total_new_releases := historical_fiction_new_releases + other_genres_new_releases) :
  historical_fiction_new_releases / total_new_releases = 8 / 29 := 
by 
  sorry

end fraction_of_historical_fiction_new_releases_l565_56588


namespace least_possible_integral_BC_l565_56507

theorem least_possible_integral_BC :
  ∃ (BC : ℕ), (BC > 0) ∧ (BC ≥ 15) ∧ 
    (7 + BC > 15) ∧ (25 + 10 > BC) ∧ 
    (7 + 15 > BC) ∧ (25 + BC > 10) := by
    sorry

end least_possible_integral_BC_l565_56507


namespace Susan_ate_six_candies_l565_56500

def candy_consumption_weekly : Prop :=
  ∀ (candies_bought_Tue candies_bought_Wed candies_bought_Thu candies_bought_Fri : ℕ)
    (candies_left : ℕ) (total_spending : ℕ),
    candies_bought_Tue = 3 →
    candies_bought_Wed = 0 →
    candies_bought_Thu = 5 →
    candies_bought_Fri = 2 →
    candies_left = 4 →
    total_spending = 9 →
    candies_bought_Tue + candies_bought_Wed + candies_bought_Thu + candies_bought_Fri - candies_left = 6

theorem Susan_ate_six_candies : candy_consumption_weekly :=
by {
  -- The proof will be filled in later
  sorry
}

end Susan_ate_six_candies_l565_56500


namespace graphs_differ_l565_56549

theorem graphs_differ (x : ℝ) :
  (∀ (y : ℝ), y = x + 3 ↔ y ≠ (x^2 - 1) / (x - 1) ∧
              y ≠ (x^2 - 1) / (x - 1) ∧
              ∀ (y : ℝ), y = (x^2 - 1) / (x - 1) ↔ ∀ (z : ℝ), y ≠ x + 3 ∧ y ≠ x + 1) := sorry

end graphs_differ_l565_56549


namespace youngest_child_age_l565_56592

variable (Y : ℕ) (O : ℕ) -- Y: the youngest child's present age
variable (P₀ P₁ P₂ P₃ : ℕ) -- P₀, P₁, P₂, P₃: the present ages of the 4 original family members

-- Conditions translated to Lean
variable (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
variable (h₂ : O = Y + 2)
variable (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24)

theorem youngest_child_age (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
                       (h₂ : O = Y + 2)
                       (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24) :
  Y = 3 := by 
  sorry

end youngest_child_age_l565_56592


namespace candy_problem_minimum_candies_l565_56599

theorem candy_problem_minimum_candies : ∃ (N : ℕ), N > 1 ∧ N % 2 = 1 ∧ N % 3 = 1 ∧ N % 5 = 1 ∧ N = 31 :=
by
  sorry

end candy_problem_minimum_candies_l565_56599


namespace positive_integer_solution_exists_l565_56542

theorem positive_integer_solution_exists (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (h_eq : x^2 = y^2 + 7 * y + 6) : (x, y) = (6, 3) := 
sorry

end positive_integer_solution_exists_l565_56542


namespace ratio_of_ages_in_two_years_l565_56543

theorem ratio_of_ages_in_two_years (S M : ℕ) 
  (h1 : M = S + 37) 
  (h2 : S = 35) : 
  (M + 2) / (S + 2) = 2 := 
by 
  -- We skip the proof steps as instructed
  sorry

end ratio_of_ages_in_two_years_l565_56543


namespace math_problem_l565_56576

theorem math_problem : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := 
by
  sorry

end math_problem_l565_56576


namespace half_product_unique_l565_56548

theorem half_product_unique (x : ℕ) (n k : ℕ) 
  (hn : x = n * (n + 1) / 2) (hk : x = k * (k + 1) / 2) : 
  n = k := 
sorry

end half_product_unique_l565_56548


namespace total_amount_shared_l565_56575

theorem total_amount_shared (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) 
  (portion_a : ℕ) (portion_b : ℕ) (portion_c : ℕ)
  (h_ratio : ratio_a = 3 ∧ ratio_b = 4 ∧ ratio_c = 9)
  (h_portion_a : portion_a = 30)
  (h_portion_b : portion_b = 2 * portion_a + 10)
  (h_portion_c : portion_c = (ratio_c / ratio_a) * portion_a) :
  portion_a + portion_b + portion_c = 190 :=
by sorry

end total_amount_shared_l565_56575


namespace no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l565_56573

theorem no_perfect_squares_in_ap (n x : ℤ) : ¬(3 * n + 2 = x^2) :=
sorry

theorem infinitely_many_perfect_cubes_in_ap : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^3 :=
sorry

theorem no_terms_of_form_x_pow_2m (n x : ℤ) (m : ℕ) : 3 * n + 2 ≠ x^(2 * m) :=
sorry

theorem infinitely_many_terms_of_form_x_pow_2m_plus_1 (m : ℕ) : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^(2 * m + 1) :=
sorry

end no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l565_56573


namespace find_k_l565_56528

-- Definitions
def a (n : ℕ) : ℤ := 1 + (n - 1) * 2
def S (n : ℕ) : ℤ := n / 2 * (2 * 1 + (n - 1) * 2)

-- Main theorem statement
theorem find_k (k : ℕ) (h : S (k + 2) - S k = 24) : k = 5 :=
by sorry

end find_k_l565_56528


namespace circumference_of_cone_l565_56586

theorem circumference_of_cone (V : ℝ) (h : ℝ) (C : ℝ) 
  (hV : V = 36 * Real.pi) (hh : h = 3) : 
  C = 12 * Real.pi :=
sorry

end circumference_of_cone_l565_56586


namespace zachary_more_pushups_l565_56522

def zachary_pushups : ℕ := 51
def david_pushups : ℕ := 44

theorem zachary_more_pushups : zachary_pushups - david_pushups = 7 := by
  sorry

end zachary_more_pushups_l565_56522


namespace fraction_division_correct_l565_56557

theorem fraction_division_correct :
  (5/6 : ℚ) / (7/9) / (11/13) = 195/154 := 
by {
  sorry
}

end fraction_division_correct_l565_56557


namespace tan_alpha_plus_pi_over_4_l565_56521

noncomputable def tan_sum_formula (α : ℝ) : ℝ :=
  (Real.tan α + Real.tan (Real.pi / 4)) / (1 - Real.tan α * Real.tan (Real.pi / 4))

theorem tan_alpha_plus_pi_over_4 
  (α : ℝ) 
  (h1 : Real.cos (2 * α) + Real.sin α * (2 * Real.sin α - 1) = 2 / 5) 
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : 
  tan_sum_formula α = 1 / 7 := 
sorry

end tan_alpha_plus_pi_over_4_l565_56521


namespace atomic_number_l565_56555

theorem atomic_number (mass_number : ℕ) (neutrons : ℕ) (protons : ℕ) :
  mass_number = 288 →
  neutrons = 169 →
  (protons = mass_number - neutrons) →
  protons = 119 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end atomic_number_l565_56555


namespace eval_x_plus_one_eq_4_l565_56581

theorem eval_x_plus_one_eq_4 (x : ℕ) (h : x = 3) : x + 1 = 4 :=
by
  sorry

end eval_x_plus_one_eq_4_l565_56581


namespace sports_minutes_in_newscast_l565_56515

-- Definitions based on the conditions
def total_newscast_minutes : ℕ := 30
def national_news_minutes : ℕ := 12
def international_news_minutes : ℕ := 5
def weather_forecasts_minutes : ℕ := 2
def advertising_minutes : ℕ := 6

-- The problem statement
theorem sports_minutes_in_newscast (t : ℕ) (n : ℕ) (i : ℕ) (w : ℕ) (a : ℕ) :
  t = 30 → n = 12 → i = 5 → w = 2 → a = 6 → t - n - i - w - a = 5 := 
by sorry

end sports_minutes_in_newscast_l565_56515


namespace find_m_n_pairs_l565_56526

theorem find_m_n_pairs (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∀ᶠ a in Filter.atTop, (a^m + a - 1) % (a^n + a^2 - 1) = 0) → m = n + 2 :=
by
  sorry

end find_m_n_pairs_l565_56526


namespace range_of_real_number_l565_56569

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B (a : ℝ) : Set ℝ := {-1, -3, a}
def complement_A : Set ℝ := {x | x ≥ 0}

theorem range_of_real_number (a : ℝ) (h : (complement_A ∩ (B a)) ≠ ∅) : a ≥ 0 :=
sorry

end range_of_real_number_l565_56569


namespace percent_c_of_b_l565_56520

variable (a b c : ℝ)

theorem percent_c_of_b (h1 : c = 0.20 * a) (h2 : b = 2 * a) : 
  ∃ x : ℝ, c = (x / 100) * b ∧ x = 10 :=
by
  sorry

end percent_c_of_b_l565_56520


namespace algebraic_expression_value_l565_56545

theorem algebraic_expression_value (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + x*y + y^2 = 0) :
  (x/(x + y))^2005 + (y/(x + y))^2005 = -1 :=
by
  sorry

end algebraic_expression_value_l565_56545


namespace complex_division_l565_56590

open Complex

theorem complex_division :
  (1 + 2 * I) / (3 - 4 * I) = -1 / 5 + 2 / 5 * I :=
by
  sorry

end complex_division_l565_56590


namespace ab_greater_than_1_l565_56563

noncomputable def log10_abs (x : ℝ) : ℝ :=
  abs (Real.logb 10 x)

theorem ab_greater_than_1
  {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a < b)
  (hf : log10_abs a < log10_abs b) : a * b > 1 := by
  sorry

end ab_greater_than_1_l565_56563


namespace angle_BAC_measure_l565_56556

variable (A B C X Y : Type)
variables (angle_ABC angle_BAC : ℝ)
variables (len_AX len_XY len_YB len_BC : ℝ)

theorem angle_BAC_measure 
  (h1 : AX = XY) 
  (h2 : XY = YB) 
  (h3 : XY = 2 * AX) 
  (h4 : angle_ABC = 150) :
  angle_BAC = 26.25 :=
by
  -- The proof would be required here.
  -- Following the statement as per instructions.
  sorry

end angle_BAC_measure_l565_56556


namespace nominal_rate_of_interest_l565_56552

noncomputable def nominal_rate (EAR : ℝ) (n : ℕ) : ℝ :=
  2 * (Real.sqrt (1 + EAR) - 1)

theorem nominal_rate_of_interest :
  nominal_rate 0.1025 2 = 0.100476 :=
by sorry

end nominal_rate_of_interest_l565_56552


namespace thomas_friends_fraction_l565_56502

noncomputable def fraction_of_bars_taken (x : ℝ) (initial_bars : ℝ) (returned_bars : ℝ) 
  (piper_bars : ℝ) (remaining_bars : ℝ) : ℝ :=
  x / initial_bars

theorem thomas_friends_fraction 
  (initial_bars : ℝ)
  (total_taken_by_all : ℝ)
  (returned_bars : ℝ)
  (piper_bars : ℝ)
  (remaining_bars : ℝ)
  (h_initial : initial_bars = 200)
  (h_remaining : remaining_bars = 110)
  (h_taken : 200 - 110 = 90)
  (h_total_taken_by_all : total_taken_by_all = 90)
  (h_returned : returned_bars = 5)
  (h_x_calculation : 2 * (total_taken_by_all + returned_bars - initial_bars) + initial_bars = total_taken_by_all + returned_bars)
  : fraction_of_bars_taken ((total_taken_by_all + returned_bars - initial_bars) + 2 * initial_bars) initial_bars returned_bars piper_bars remaining_bars = 21 / 80 :=
  sorry

end thomas_friends_fraction_l565_56502


namespace total_triangles_in_figure_l565_56511

theorem total_triangles_in_figure :
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  small_triangles + two_small_comb + three_small_comb + all_small_comb = 11 :=
by
  let row1 := 3
  let row2 := 2
  let row3 := 1
  let small_triangles := row1 + row2 + row3
  let two_small_comb := 3
  let three_small_comb := 1
  let all_small_comb := 1
  show small_triangles + two_small_comb + three_small_comb + all_small_comb = 11
  sorry

end total_triangles_in_figure_l565_56511


namespace find_sum_of_digits_l565_56582

theorem find_sum_of_digits (a c : ℕ) (h1 : 200 + 10 * a + 3 + 427 = 600 + 10 * c + 9) (h2 : (600 + 10 * c + 9) % 3 = 0) : a + c = 4 :=
sorry

end find_sum_of_digits_l565_56582


namespace cloves_of_garlic_needed_l565_56533

def cloves_needed_for_vampires (vampires : ℕ) : ℕ :=
  (vampires * 3) / 2

def cloves_needed_for_wights (wights : ℕ) : ℕ :=
  (wights * 3) / 3

def cloves_needed_for_vampire_bats (vampire_bats : ℕ) : ℕ :=
  (vampire_bats * 3) / 8

theorem cloves_of_garlic_needed (vampires wights vampire_bats : ℕ) :
  cloves_needed_for_vampires 30 + cloves_needed_for_wights 12 + 
  cloves_needed_for_vampire_bats 40 = 72 :=
by
  sorry

end cloves_of_garlic_needed_l565_56533


namespace sum_faces_of_pentahedron_l565_56530

def pentahedron := {f : ℕ // f = 5}

theorem sum_faces_of_pentahedron (p : pentahedron) : p.val = 5 := 
by
  sorry

end sum_faces_of_pentahedron_l565_56530


namespace union_P_Q_l565_56559

def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | x^2 - 2*x < 0}

theorem union_P_Q : P ∪ Q = {x : ℝ | -1 < x ∧ x < 2} :=
sorry

end union_P_Q_l565_56559


namespace caleb_spent_more_on_ice_cream_l565_56553

theorem caleb_spent_more_on_ice_cream :
  ∀ (number_of_ic_cartons number_of_fy_cartons : ℕ)
    (cost_per_ic_carton cost_per_fy_carton : ℝ)
    (discount_rate sales_tax_rate : ℝ),
    number_of_ic_cartons = 10 →
    number_of_fy_cartons = 4 →
    cost_per_ic_carton = 4 →
    cost_per_fy_carton = 1 →
    discount_rate = 0.15 →
    sales_tax_rate = 0.05 →
    (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
     (number_of_ic_cartons * cost_per_ic_carton * (1 - discount_rate) + 
      number_of_fy_cartons * cost_per_fy_carton) * sales_tax_rate) -
    (number_of_fy_cartons * cost_per_fy_carton) = 30 :=
by
  intros number_of_ic_cartons number_of_fy_cartons cost_per_ic_carton cost_per_fy_carton discount_rate sales_tax_rate
  sorry

end caleb_spent_more_on_ice_cream_l565_56553


namespace polynomial_remainder_l565_56554
-- Importing the broader library needed

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 3

-- The statement of the theorem
theorem polynomial_remainder :
  p 2 = 43 :=
sorry

end polynomial_remainder_l565_56554


namespace eval_expr_correct_l565_56514

noncomputable def eval_expr : ℝ :=
  let a := (12:ℝ)^5 * (6:ℝ)^4
  let b := (3:ℝ)^2 * (36:ℝ)^2
  let c := Real.sqrt 9 * Real.log (27:ℝ)
  (a / b) + c

theorem eval_expr_correct : eval_expr = 27657.887510597983 := by
  sorry

end eval_expr_correct_l565_56514


namespace candies_initial_count_l565_56537

theorem candies_initial_count (x : ℕ) (h : (x - 29) / 13 = 15) : x = 224 :=
sorry

end candies_initial_count_l565_56537


namespace inequality_of_ab_bc_ca_l565_56518

theorem inequality_of_ab_bc_ca (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^4 + b^4 + c^4 = 3) : 
  (1 / (4 - a * b)) + (1 / (4 - b * c)) + (1 / (4 - c * a)) ≤ 1 :=
by
  sorry

end inequality_of_ab_bc_ca_l565_56518


namespace ratio_of_B_to_C_l565_56519

theorem ratio_of_B_to_C
  (A B C : ℕ) 
  (h1 : A = B + 2) 
  (h2 : A + B + C = 47) 
  (h3 : B = 18) : B / C = 2 := 
by 
  sorry

end ratio_of_B_to_C_l565_56519


namespace each_child_gets_twelve_cupcakes_l565_56579

def total_cupcakes := 96
def children := 8
def cupcakes_per_child : ℕ := total_cupcakes / children

theorem each_child_gets_twelve_cupcakes :
  cupcakes_per_child = 12 :=
by
  sorry

end each_child_gets_twelve_cupcakes_l565_56579


namespace robot_path_length_l565_56503

/--
A robot moves in the plane in a straight line, but every one meter it turns 90° to the right or to the left. At some point it reaches its starting point without having visited any other point more than once, and stops immediately. Prove that the possible path lengths of the robot are 4k for some integer k with k >= 3.
-/
theorem robot_path_length (n : ℕ) (h : n > 0) (Movement : n % 4 = 0) :
  ∃ k : ℕ, n = 4 * k ∧ k ≥ 3 :=
sorry

end robot_path_length_l565_56503


namespace find_a6_l565_56565

def is_arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem find_a6 :
  ∀ (a b : ℕ → ℕ),
    a 1 = 3 →
    b 1 = 2 →
    b 3 = 6 →
    is_arithmetic_sequence b →
    (∀ n, b n = a (n + 1) - a n) →
    a 6 = 33 :=
by
  intros a b h_a1 h_b1 h_b3 h_arith h_diff
  sorry

end find_a6_l565_56565


namespace more_time_in_swamp_l565_56574

theorem more_time_in_swamp (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : 2 * a + 4 * b + 6 * c = 15) : a > c :=
by {
  sorry
}

end more_time_in_swamp_l565_56574


namespace tina_more_than_katya_l565_56524

-- Define the number of glasses sold by Katya, Ricky, and the condition for Tina's sales
def katya_sales : ℕ := 8
def ricky_sales : ℕ := 9

def combined_sales : ℕ := katya_sales + ricky_sales
def tina_sales : ℕ := 2 * combined_sales

-- Define the theorem to prove that Tina sold 26 more glasses than Katya
theorem tina_more_than_katya : tina_sales = katya_sales + 26 := by
  sorry

end tina_more_than_katya_l565_56524


namespace min_abs_sum_of_products_l565_56578

noncomputable def g (x : ℝ) : ℝ := x^4 + 10*x^3 + 29*x^2 + 30*x + 9

theorem min_abs_sum_of_products (w : Fin 4 → ℝ) (h_roots : ∀ i, g (w i) = 0)
  : ∃ a b c d : Fin 4, a ≠ b ∧ c ≠ d ∧ (∀ i j, i ≠ j → a ≠ i ∧ b ≠ i ∧ c ≠ i ∧ d ≠ i → a ≠ j ∧ b ≠ j ∧ c ≠ j ∧ d ≠ j) ∧
    |w a * w b + w c * w d| = 6 :=
sorry

end min_abs_sum_of_products_l565_56578


namespace jason_messages_l565_56544

theorem jason_messages :
  ∃ M : ℕ, (M + M / 2 + 150) / 5 = 96 ∧ M = 220 := by
  sorry

end jason_messages_l565_56544


namespace piggy_bank_exceed_five_dollars_l565_56509

noncomputable def sequence_sum (n : ℕ) : ℕ := 2^n - 1

theorem piggy_bank_exceed_five_dollars (n : ℕ) (start_day : Nat) (day_of_week : Fin 7) :
  ∃ (n : ℕ), sequence_sum n > 500 ∧ n = 9 ∧ (start_day + n) % 7 = 2 := 
sorry

end piggy_bank_exceed_five_dollars_l565_56509


namespace problem_statement_l565_56583

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

variable (a x y t : ℝ) 

theorem problem_statement : 
  (log_base a x + 3 * log_base x a - log_base x y = 3) ∧ (a > 1) ∧ (x = a ^ t) ∧ (0 < t ∧ t ≤ 2) ∧ (y = 8) 
  → (a = 16) ∧ (x = 64) := 
by 
  sorry

end problem_statement_l565_56583


namespace selena_trip_length_l565_56538

variable (y : ℚ)

def selena_trip (y : ℚ) : Prop :=
  y / 4 + 16 + y / 6 = y

theorem selena_trip_length : selena_trip y → y = 192 / 7 :=
by
  sorry

end selena_trip_length_l565_56538


namespace iterative_average_difference_l565_56505

theorem iterative_average_difference :
  let numbers : List ℕ := [2, 4, 6, 8, 10] 
  let avg2 (a b : ℝ) := (a + b) / 2
  let avg (init : ℝ) (lst : List ℕ) := lst.foldl (λ acc x => avg2 acc x) init
  let max_avg := avg 2 [4, 6, 8, 10]
  let min_avg := avg 10 [8, 6, 4, 2] 
  max_avg - min_avg = 4.25 := 
by
  sorry

end iterative_average_difference_l565_56505


namespace roster_representation_of_M_l565_56523

def M : Set ℚ := {x | ∃ m n : ℤ, x = m / n ∧ |m| < 2 ∧ 1 ≤ n ∧ n ≤ 3}

theorem roster_representation_of_M :
  M = {-1, -1/2, -1/3, 0, 1/2, 1/3} :=
by sorry

end roster_representation_of_M_l565_56523


namespace work_rate_proof_l565_56558

theorem work_rate_proof (A B C : ℝ) (h1 : A + B = 1 / 15) (h2 : C = 1 / 60) : 
  1 / (A + B + C) = 12 :=
by
  sorry

end work_rate_proof_l565_56558


namespace min_value_of_frac_sum_l565_56577

theorem min_value_of_frac_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 2) :
  (1 / a + 2 / b) = 9 / 2 :=
sorry

end min_value_of_frac_sum_l565_56577


namespace tan_alpha_value_l565_56595

theorem tan_alpha_value
  (α : ℝ)
  (h_cos : Real.cos α = -4/5)
  (h_range : (Real.pi / 2) < α ∧ α < Real.pi) :
  Real.tan α = -3/4 := by
  sorry

end tan_alpha_value_l565_56595


namespace angle_sum_around_point_l565_56567

theorem angle_sum_around_point {x : ℝ} (h : 2 * x + 210 = 360) : x = 75 :=
by
  sorry

end angle_sum_around_point_l565_56567


namespace opposite_blue_face_is_white_l565_56512

-- Define colors
inductive Color
| Red
| Blue
| Orange
| Purple
| Green
| Yellow
| White

-- Define the positions of colors on the cube
structure CubeConfig :=
(top : Color)
(front : Color)
(bottom : Color)
(back : Color)
(left : Color)
(right : Color)

-- The given conditions
def cube_conditions (c : CubeConfig) : Prop :=
  c.top = Color.Purple ∧
  c.front = Color.Green ∧
  c.bottom = Color.Yellow ∧
  c.back = Color.Orange ∧
  c.left = Color.Blue ∧
  c.right = Color.White

-- The statement we need to prove
theorem opposite_blue_face_is_white (c : CubeConfig) (h : cube_conditions c) :
  c.right = Color.White :=
by
  -- Proof placeholder
  sorry

end opposite_blue_face_is_white_l565_56512


namespace total_population_of_cities_l565_56564

theorem total_population_of_cities (n : ℕ) (avg_pop : ℕ) (pn : (n = 20)) (avg_factor: (avg_pop = (4500 + 5000) / 2)) : 
  n * avg_pop = 95000 := 
by 
  sorry

end total_population_of_cities_l565_56564
