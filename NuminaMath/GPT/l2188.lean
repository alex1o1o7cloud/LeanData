import Mathlib

namespace original_speed_of_person_B_l2188_218893

-- Let v_A and v_B be the speeds of person A and B respectively
variable (v_A v_B : ℝ)

-- Conditions for problem
axiom initial_ratio : v_A / v_B = (5 / 4 * v_A) / (v_B + 10)

-- The goal: Prove that v_B = 40
theorem original_speed_of_person_B : v_B = 40 := 
  sorry

end original_speed_of_person_B_l2188_218893


namespace lychee_harvest_l2188_218825

theorem lychee_harvest : 
  let last_year_red := 350
  let last_year_yellow := 490
  let this_year_red := 500
  let this_year_yellow := 700
  let sold_red := 2/3 * this_year_red
  let sold_yellow := 3/7 * this_year_yellow
  let remaining_red_after_sale := this_year_red - sold_red
  let remaining_yellow_after_sale := this_year_yellow - sold_yellow
  let family_ate_red := 3/5 * remaining_red_after_sale
  let family_ate_yellow := 4/9 * remaining_yellow_after_sale
  let remaining_red := remaining_red_after_sale - family_ate_red
  let remaining_yellow := remaining_yellow_after_sale - family_ate_yellow
  (this_year_red - last_year_red) / last_year_red * 100 = 42.86
  ∧ (this_year_yellow - last_year_yellow) / last_year_yellow * 100 = 42.86
  ∧ remaining_red = 67
  ∧ remaining_yellow = 223 :=
by
    intros
    sorry

end lychee_harvest_l2188_218825


namespace tangent_line_ellipse_l2188_218846

variable (a b x0 y0 : ℝ)
variable (x y : ℝ)

def ellipse (x y a b : ℝ) := (x ^ 2) / (a ^ 2) + (y ^ 2) / (b ^ 2) = 1

theorem tangent_line_ellipse :
  ellipse x y a b ∧ a > b ∧ (x0 ≠ 0 ∨ y0 ≠ 0) ∧ (x0 ^ 2) / (a ^ 2) + (y0 ^ 2) / (b ^ 2) > 1 →
  (x0 * x) / (a ^ 2) + (y0 * y) / (b ^ 2) = 1 :=
  sorry

end tangent_line_ellipse_l2188_218846


namespace surfers_ratio_l2188_218847

theorem surfers_ratio (S1 : ℕ) (S3 : ℕ) : S1 = 1500 → 
  (∀ S2 : ℕ, S2 = S1 + 600 → (1400 * 3 = S1 + S2 + S3) → 
  S3 = 600) → (S3 / S1 = 2 / 5) :=
sorry

end surfers_ratio_l2188_218847


namespace Corey_goal_reachable_l2188_218895

theorem Corey_goal_reachable :
  ∀ (goal balls_found_saturday balls_found_sunday additional_balls : ℕ),
    goal = 48 →
    balls_found_saturday = 16 →
    balls_found_sunday = 18 →
    additional_balls = goal - (balls_found_saturday + balls_found_sunday) →
    additional_balls = 14 :=
by
  intros goal balls_found_saturday balls_found_sunday additional_balls
  intro goal_eq
  intro saturday_eq
  intro sunday_eq
  intro additional_eq
  sorry

end Corey_goal_reachable_l2188_218895


namespace composite_product_division_l2188_218833

noncomputable def firstFiveCompositeProduct : ℕ := 4 * 6 * 8 * 9 * 10
noncomputable def nextFiveCompositeProduct : ℕ := 12 * 14 * 15 * 16 * 18

theorem composite_product_division : firstFiveCompositeProduct / nextFiveCompositeProduct = 1 / 42 := by
  sorry

end composite_product_division_l2188_218833


namespace find_x1_plus_x2_l2188_218817

def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

theorem find_x1_plus_x2 (x1 x2 : ℝ) (hneq : x1 ≠ x2) (h1 : f x1 = 101) (h2 : f x2 = 101) : x1 + x2 = 2 := 
by 
  -- proof or sorry can be used; let's assume we use sorry to skip proof
  sorry

end find_x1_plus_x2_l2188_218817


namespace arc_length_l2188_218892

theorem arc_length (circumference : ℝ) (angle_degrees : ℝ) (h : circumference = 90) (θ : angle_degrees = 45) :
  (angle_degrees / 360) * circumference = 11.25 := 
  by 
    sorry

end arc_length_l2188_218892


namespace base_five_to_decimal_l2188_218856

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2 => 2 * 5^0
  | 3 => 3 * 5^1
  | 1 => 1 * 5^2
  | _ => 0

theorem base_five_to_decimal : base5_to_base10 2 + base5_to_base10 3 + base5_to_base10 1 = 42 :=
by sorry

end base_five_to_decimal_l2188_218856


namespace problem_l2188_218828

theorem problem (f : ℕ → ℕ → ℕ) (h0 : f 1 1 = 1) (h1 : ∀ m n, f m n ∈ {x | x > 0}) 
  (h2 : ∀ m n, f m (n + 1) = f m n + 2) (h3 : ∀ m, f (m + 1) 1 = 2 * f m 1) : 
  f 1 5 = 9 ∧ f 5 1 = 16 ∧ f 5 6 = 26 :=
sorry

end problem_l2188_218828


namespace find_m_n_difference_l2188_218868

theorem find_m_n_difference (x y m n : ℤ)
  (hx : x = 2)
  (hy : y = -3)
  (hm : x + y = m)
  (hn : 2 * x - y = n) :
  m - n = -8 :=
by {
  sorry
}

end find_m_n_difference_l2188_218868


namespace complex_power_difference_l2188_218844

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 40 - (1 - i) ^ 40 = 0 := by 
  sorry

end complex_power_difference_l2188_218844


namespace problem1_problem2_l2188_218862

-- Assume x and y are positive numbers
variables (x y : ℝ) (hx : 0 < x) (hy : 0 < y)

-- Prove that x^3 + y^3 >= x^2*y + y^2*x
theorem problem1 : x^3 + y^3 ≥ x^2 * y + y^2 * x :=
by sorry

-- Prove that m ≤ 2 given the additional condition
variables (m : ℝ)
theorem problem2 (cond : (x/y^2 + y/x^2) ≥ m/2 * (1/x + 1/y)) : m ≤ 2 :=
by sorry

end problem1_problem2_l2188_218862


namespace num_green_hats_l2188_218891

-- Definitions
def total_hats : ℕ := 85
def blue_hat_cost : ℕ := 6
def green_hat_cost : ℕ := 7
def total_cost : ℕ := 548

-- Prove the number of green hats (g) is 38 given the conditions
theorem num_green_hats (b g : ℕ) 
  (h₁ : b + g = total_hats)
  (h₂ : blue_hat_cost * b + green_hat_cost * g = total_cost) : 
  g = 38 := by
  sorry

end num_green_hats_l2188_218891


namespace no_solution_for_x_l2188_218859

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345

theorem no_solution_for_x : proof_problem :=
  by
    intro x
    sorry

end no_solution_for_x_l2188_218859


namespace blake_spending_on_oranges_l2188_218800

theorem blake_spending_on_oranges (spending_on_oranges spending_on_apples spending_on_mangoes initial_amount change_amount: ℝ)
  (h1 : spending_on_apples = 50)
  (h2 : spending_on_mangoes = 60)
  (h3 : initial_amount = 300)
  (h4 : change_amount = 150)
  (h5 : initial_amount - change_amount = spending_on_oranges + spending_on_apples + spending_on_mangoes) :
  spending_on_oranges = 40 := by
  sorry

end blake_spending_on_oranges_l2188_218800


namespace face_value_shares_l2188_218865

theorem face_value_shares (market_value : ℝ) (dividend_rate desired_rate : ℝ) (FV : ℝ) 
  (h1 : dividend_rate = 0.09)
  (h2 : desired_rate = 0.12)
  (h3 : market_value = 36.00000000000001)
  (h4 : (dividend_rate * FV) = (desired_rate * market_value)) :
  FV = 48.00000000000001 :=
by
  sorry

end face_value_shares_l2188_218865


namespace A_plus_B_zero_l2188_218886

def f (A B x : ℝ) : ℝ := 3 * A * x + 2 * B
def g (A B x : ℝ) : ℝ := 2 * B * x + 3 * A

theorem A_plus_B_zero (A B : ℝ) (h1 : A ≠ B) (h2 : ∀ x : ℝ, f A B (g A B x) - g A B (f A B x) = 3 * (B - A)) :
  A + B = 0 :=
sorry

end A_plus_B_zero_l2188_218886


namespace square_area_from_triangle_perimeter_l2188_218808

noncomputable def perimeter_triangle (a b c : ℝ) : ℝ := a + b + c

noncomputable def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem square_area_from_triangle_perimeter 
  (a b c : ℝ) 
  (h₁ : a = 5.5) 
  (h₂ : b = 7.5) 
  (h₃ : c = 11) 
  (h₄ : perimeter_triangle a b c = 24) 
  : area_square (side_length_square (perimeter_triangle a b c)) = 36 := 
by 
  simp [perimeter_triangle, side_length_square, area_square, h₁, h₂, h₃, h₄]
  sorry

end square_area_from_triangle_perimeter_l2188_218808


namespace ratio_c_b_l2188_218881

theorem ratio_c_b (x y a b c : ℝ) (h1 : x ≥ 1) (h2 : x + y ≤ 4) (h3 : a * x + b * y + c ≤ 0) 
    (h_max : ∀ x y, (x,y) = (2, 2) → 2 * x + y = 6) (h_min : ∀ x y, (x,y) = (1, -1) → 2 * x + y = 1) (h_b : b ≠ 0) :
    c / b = 4 := sorry

end ratio_c_b_l2188_218881


namespace mr_brown_net_result_l2188_218842

noncomputable def C1 := 1.50 / 1.3
noncomputable def C2 := 1.50 / 0.9
noncomputable def profit_from_first_pen := 1.50 - C1
noncomputable def tax := 0.05 * profit_from_first_pen
noncomputable def total_cost := C1 + C2
noncomputable def total_revenue := 3.00
noncomputable def net_result := total_revenue - total_cost - tax

theorem mr_brown_net_result : net_result = 0.16 :=
by
  sorry

end mr_brown_net_result_l2188_218842


namespace jane_babysitting_start_l2188_218849

-- Definitions based on the problem conditions
def jane_current_age := 32
def years_since_babysitting := 10
def oldest_current_child_age := 24

-- Definition for the starting babysitting age
def starting_babysitting_age : ℕ := 8

-- Theorem statement to prove
theorem jane_babysitting_start (h1 : jane_current_age - years_since_babysitting = 22)
  (h2 : oldest_current_child_age - years_since_babysitting = 14)
  (h3 : ∀ (age_jane age_child : ℕ), age_child ≤ age_jane / 2) :
  starting_babysitting_age = 8 :=
by
  sorry

end jane_babysitting_start_l2188_218849


namespace sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l2188_218835

theorem sum_of_29_12_23_is_64: 29 + 12 + 23 = 64 := sorry

theorem sixtyfour_is_two_to_six:
  64 = 2^6 := sorry

end sum_of_29_12_23_is_64_sixtyfour_is_two_to_six_l2188_218835


namespace min_value_pq_l2188_218841

theorem min_value_pq (p q : ℝ) (hp : 0 < p) (hq : 0 < q)
  (h1 : p^2 - 8 * q ≥ 0)
  (h2 : 4 * q^2 - 4 * p ≥ 0) :
  p + q ≥ 6 :=
sorry

end min_value_pq_l2188_218841


namespace sum_of_coefficients_l2188_218871

theorem sum_of_coefficients (b_0 b_1 b_2 b_3 b_4 b_5 b_6 : ℝ) :
  (5 * 1 - 2)^6 = b_6 * 1^6 + b_5 * 1^5 + b_4 * 1^4 + b_3 * 1^3 + b_2 * 1^2 + b_1 * 1 + b_0
  → b_0 + b_1 + b_2 + b_3 + b_4 + b_5 + b_6 = 729 := by
  sorry

end sum_of_coefficients_l2188_218871


namespace find_angle_C_find_sum_a_b_l2188_218880

noncomputable def triangle_condition (a b c : ℝ) (A B C : ℝ) : Prop :=
  c = 7 / 2 ∧
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 ∧
  (Real.tan A + Real.tan B = Real.sqrt 3 * (Real.tan A * Real.tan B - 1))

theorem find_angle_C (a b c A B C : ℝ) (h : triangle_condition a b c A B C) : C = Real.pi / 3 :=
  sorry

theorem find_sum_a_b (a b c A B C : ℝ) (h : triangle_condition a b c A B C) (hC : C = Real.pi / 3) : a + b = 11 / 2 :=
  sorry

end find_angle_C_find_sum_a_b_l2188_218880


namespace smallest_solution_l2188_218839

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) 
    (h_eq : 1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) : x = 4 - Real.sqrt 2 := 
by 
  sorry

end smallest_solution_l2188_218839


namespace find_x_from_expression_l2188_218815

theorem find_x_from_expression
  (y : ℚ)
  (h1 : y = -3/2)
  (h2 : -2 * (x : ℚ) - y^2 = 0.25) : 
  x = -5/4 := 
by 
  sorry

end find_x_from_expression_l2188_218815


namespace first_term_of_geometric_sequence_l2188_218816

theorem first_term_of_geometric_sequence
  (a r : ℚ) -- where a is the first term and r is the common ratio
  (h1 : a * r^4 = 45) -- fifth term condition
  (h2 : a * r^5 = 60) -- sixth term condition
  : a = 1215 / 256 := 
sorry

end first_term_of_geometric_sequence_l2188_218816


namespace stratified_sampling_is_reasonable_l2188_218806

-- Defining our conditions and stating our theorem
def flat_land := 150
def ditch_land := 30
def sloped_land := 90
def total_acres := 270
def sampled_acres := 18
def sampling_ratio := sampled_acres / total_acres

def flat_land_sampled := flat_land * sampling_ratio
def ditch_land_sampled := ditch_land * sampling_ratio
def sloped_land_sampled := sloped_land * sampling_ratio

theorem stratified_sampling_is_reasonable :
  flat_land_sampled = 10 ∧
  ditch_land_sampled = 2 ∧
  sloped_land_sampled = 6 := 
by
  sorry

end stratified_sampling_is_reasonable_l2188_218806


namespace mark_paired_with_mike_prob_l2188_218885

def total_students := 16
def other_students := 15
def prob_pairing (mark: Nat) (mike: Nat) : ℚ := 1 / other_students

theorem mark_paired_with_mike_prob : prob_pairing 1 2 = 1 / 15 := 
sorry

end mark_paired_with_mike_prob_l2188_218885


namespace ribbon_leftover_correct_l2188_218867

def initial_ribbon : ℕ := 84
def used_ribbon : ℕ := 46
def leftover_ribbon : ℕ := 38

theorem ribbon_leftover_correct : initial_ribbon - used_ribbon = leftover_ribbon :=
by
  sorry

end ribbon_leftover_correct_l2188_218867


namespace eval_expression_l2188_218875

theorem eval_expression (a x : ℤ) (h : x = a + 9) : (x - a + 5) = 14 :=
by
  sorry

end eval_expression_l2188_218875


namespace cos_100_eq_neg_sqrt_l2188_218831

theorem cos_100_eq_neg_sqrt (a : ℝ) (h : Real.sin (80 * Real.pi / 180) = a) : 
  Real.cos (100 * Real.pi / 180) = -Real.sqrt (1 - a^2) := 
sorry

end cos_100_eq_neg_sqrt_l2188_218831


namespace max_last_place_score_l2188_218888

theorem max_last_place_score (n : ℕ) (h : n ≥ 4) :
  ∃ k, (∀ m, m < n -> (k + m) < (n * 3)) ∧ 
     (∀ i, ∃ j, j < n ∧ i = k + j) ∧
     (n * 2 - 2) = (k + n - 1) ∧ 
     k = n - 2 := 
sorry

end max_last_place_score_l2188_218888


namespace solve_for_x_l2188_218883

-- Define the custom operation
def custom_mul (a b : ℝ) : ℝ := 4 * a - 2 * b

-- Main statement to prove
theorem solve_for_x : (∃ x : ℝ, custom_mul 3 (custom_mul 4 x) = 10) ↔ (x = 7.5) :=
by
  sorry

end solve_for_x_l2188_218883


namespace math_problem_l2188_218838

variables (x y : ℝ)

noncomputable def question_value (x y : ℝ) : ℝ := (2 * x - 5 * y) / (5 * x + 2 * y)

theorem math_problem 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (cond : (5 * x - 2 * y) / (2 * x + 3 * y) = 1) : 
  question_value x y = -5 / 31 :=
sorry

end math_problem_l2188_218838


namespace fishing_tomorrow_l2188_218823

theorem fishing_tomorrow (seven_every_day eight_every_other_day three_every_three_days twelve_yesterday ten_today : ℕ)
  (h1 : seven_every_day = 7)
  (h2 : eight_every_other_day = 8)
  (h3 : three_every_three_days = 3)
  (h4 : twelve_yesterday = 12)
  (h5 : ten_today = 10) :
  (seven_every_day + (eight_every_other_day - (twelve_yesterday - seven_every_day)) + three_every_three_days) = 15 :=
by
  sorry

end fishing_tomorrow_l2188_218823


namespace find_f_of_3_l2188_218818

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1/x + 2) = x) : f 3 = 1 := 
sorry

end find_f_of_3_l2188_218818


namespace job_completion_time_l2188_218805

theorem job_completion_time (A_rate D_rate Combined_rate : ℝ) (hA : A_rate = 1 / 3) (hD : D_rate = 1 / 6) (hCombined : Combined_rate = A_rate + D_rate) :
  (1 / Combined_rate) = 2 :=
by sorry

end job_completion_time_l2188_218805


namespace maximum_ab_ac_bc_l2188_218802

theorem maximum_ab_ac_bc (a b c : ℝ) (h : a + 3 * b + c = 5) : 
  ab + ac + bc ≤ 25 / 6 :=
sorry

end maximum_ab_ac_bc_l2188_218802


namespace problem_statement_l2188_218852

   def f (a : ℤ) : ℤ := a - 2
   def F (a b : ℤ) : ℤ := b^2 + a

   theorem problem_statement : F 3 (f 4) = 7 := by
     sorry
   
end problem_statement_l2188_218852


namespace product_of_odd_and_even_is_odd_l2188_218899

theorem product_of_odd_and_even_is_odd {f g : ℝ → ℝ} 
  (hf : ∀ x : ℝ, f (-x) = -f x)
  (hg : ∀ x : ℝ, g (-x) = g x) :
  ∀ x : ℝ, (f x) * (g x) = -(f (-x) * g (-x)) :=
by
  sorry

end product_of_odd_and_even_is_odd_l2188_218899


namespace min_a_plus_b_l2188_218872

theorem min_a_plus_b (a b : ℤ) (h : a * b = 144) : a + b ≥ -145 := sorry

end min_a_plus_b_l2188_218872


namespace students_journals_l2188_218824

theorem students_journals :
  ∃ u v : ℕ, 
    u + v = 75000 ∧ 
    (7 * u + 2 * v = 300000) ∧ 
    (∃ b g : ℕ, b = u * 7 / 300 ∧ g = v * 2 / 300 ∧ b = 700 ∧ g = 300) :=
by {
  -- The proving steps will go here
  sorry
}

end students_journals_l2188_218824


namespace inverse_proportion_quadrants_l2188_218836

theorem inverse_proportion_quadrants (k : ℝ) : (∀ x, x ≠ 0 → ((x < 0 → (2 - k) / x > 0) ∧ (x > 0 → (2 - k) / x < 0))) → k > 2 :=
by sorry

end inverse_proportion_quadrants_l2188_218836


namespace find_triplets_geometric_and_arithmetic_prog_l2188_218845

theorem find_triplets_geometric_and_arithmetic_prog :
  ∃ a1 a2 b1 b2,
    (a2 = a1 * ((12:ℚ) / a1) ∧ 12 = a1 * ((12:ℚ) / a1)^2) ∧
    (b2 = b1 + ((9:ℚ) - b1) / 2 ∧ 9 = b1 + 2 * (((9:ℚ) - b1) / 2)) ∧
    ((a1 = b1) ∧ (a2 = b2)) ∧ 
    (∀ (a1 a2 : ℚ), ((a1 = -9) ∧ (a2 = -6)) ∨ ((a1 = 15) ∧ (a2 = 12))) :=
by sorry

end find_triplets_geometric_and_arithmetic_prog_l2188_218845


namespace halfway_between_ratios_l2188_218870

theorem halfway_between_ratios :
  let a := (1 : ℚ) / 8
  let b := (1 : ℚ) / 3
  (a + b) / 2 = 11 / 48 := by
  sorry

end halfway_between_ratios_l2188_218870


namespace chord_bisected_by_point_l2188_218876

theorem chord_bisected_by_point (x y : ℝ) (h : (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1) :
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -3 ∧ (∀ x y : ℝ, (a * x + b * y + c = 0 ↔ (x - 2)^2 / 16 + (y - 1)^2 / 8 = 1)) := by
  sorry

end chord_bisected_by_point_l2188_218876


namespace late_fisherman_arrival_l2188_218855

-- Definitions of conditions
variables (n d : ℕ) -- n is the number of fishermen on Monday, d is the number of days the late fisherman fished
variable (total_fish : ℕ := 370)
variable (fish_per_day_per_fisherman : ℕ := 10)
variable (days_fished : ℕ := 5) -- From Monday to Friday

-- Condition in Lean: total fish caught from Monday to Friday
def total_fish_caught (n d : ℕ) := 50 * n + 10 * d

theorem late_fisherman_arrival (n d : ℕ) (h : total_fish_caught n d = 370) : 
  d = 2 :=
by
  sorry

end late_fisherman_arrival_l2188_218855


namespace no_four_distinct_integers_with_product_plus_2006_perfect_square_l2188_218854

theorem no_four_distinct_integers_with_product_plus_2006_perfect_square : 
  ¬ ∃ (a b c d : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  (∃ k1 k2 k3 k4 k5 k6 : ℕ, a * b + 2006 = k1^2 ∧ 
                          a * c + 2006 = k2^2 ∧ 
                          a * d + 2006 = k3^2 ∧ 
                          b * c + 2006 = k4^2 ∧ 
                          b * d + 2006 = k5^2 ∧ 
                          c * d + 2006 = k6^2) := 
sorry

end no_four_distinct_integers_with_product_plus_2006_perfect_square_l2188_218854


namespace f_neg2_eq_neg4_l2188_218834

noncomputable def f (x : ℝ) : ℝ :=
  if hx : x >= 0 then 3^x - 2*x - 1
  else - (3^(-x) - 2*(-x) - 1)

theorem f_neg2_eq_neg4
: f (-2) = -4 :=
by
  sorry

end f_neg2_eq_neg4_l2188_218834


namespace factorize_x4_minus_16y4_l2188_218807

theorem factorize_x4_minus_16y4 (x y : ℚ) : 
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by 
  sorry

end factorize_x4_minus_16y4_l2188_218807


namespace speed_of_first_car_l2188_218801

-- Define the conditions
def t : ℝ := 3.5
def v : ℝ := sorry -- (To be solved in the proof)
def speed_second_car : ℝ := 58
def total_distance : ℝ := 385

-- The distance each car travels after t hours
def distance_first_car : ℝ := v * t
def distance_second_car : ℝ := speed_second_car * t

-- The equation representing the total distance between the two cars after 3.5 hours
def equation := distance_first_car + distance_second_car = total_distance

-- The main theorem stating the speed of the first car
theorem speed_of_first_car : v = 52 :=
by
  -- The important proof steps would go here solving the equation "equation".
  sorry

end speed_of_first_car_l2188_218801


namespace range_of_a_l2188_218851

def A (x : ℝ) : Prop := abs (x - 4) < 2 * x

def B (x a : ℝ) : Prop := x * (x - a) ≥ (a + 6) * (x - a)

theorem range_of_a (a : ℝ) :
  (∀ x, A x → B x a) → a ≤ -14 / 3 :=
  sorry

end range_of_a_l2188_218851


namespace cost_price_per_meter_l2188_218877

-- Given conditions
def total_selling_price : ℕ := 18000
def total_meters_sold : ℕ := 400
def loss_per_meter : ℕ := 5

-- Statement to be proven
theorem cost_price_per_meter : 
    ((total_selling_price + (loss_per_meter * total_meters_sold)) / total_meters_sold) = 50 := 
by
    sorry

end cost_price_per_meter_l2188_218877


namespace exists_not_in_range_f_l2188_218811

noncomputable def f : ℝ → ℕ :=
sorry

axiom functional_equation : ∀ (x y : ℝ), f (x + (1 / f y)) = f (y + (1 / f x))

theorem exists_not_in_range_f :
  ∃ n : ℕ, ∀ x : ℝ, f x ≠ n :=
sorry

end exists_not_in_range_f_l2188_218811


namespace distance_first_to_last_tree_l2188_218848

theorem distance_first_to_last_tree 
    (n_trees : ℕ) 
    (distance_first_to_fifth : ℕ)
    (h1 : n_trees = 8)
    (h2 : distance_first_to_fifth = 80) 
    : ∃ distance_first_to_last, distance_first_to_last = 140 := by
  sorry

end distance_first_to_last_tree_l2188_218848


namespace seven_digit_divisible_by_11_l2188_218879

def is_digit (d : ℕ) : Prop := d ≤ 9

def valid7DigitNumber (b n : ℕ) : Prop :=
  let sum_odd := 3 + 5 + 6
  let sum_even := b + n + 7 + 8
  let diff := sum_odd - sum_even
  diff % 11 = 0

theorem seven_digit_divisible_by_11 (b n : ℕ) (hb : is_digit b) (hn : is_digit n)
  (h_valid : valid7DigitNumber b n) : b + n = 10 := 
sorry

end seven_digit_divisible_by_11_l2188_218879


namespace jeff_can_store_songs_l2188_218829

def gbToMb (gb : ℕ) : ℕ := gb * 1000

def newAppsStorage : ℕ :=
  5 * 450 + 5 * 300 + 5 * 150

def newPhotosStorage : ℕ :=
  300 * 4 + 50 * 8

def newVideosStorage : ℕ :=
  15 * 400 + 30 * 200

def newPDFsStorage : ℕ :=
  25 * 20

def totalNewStorage : ℕ :=
  newAppsStorage + newPhotosStorage + newVideosStorage + newPDFsStorage

def existingStorage : ℕ :=
  gbToMb 7

def totalUsedStorage : ℕ :=
  existingStorage + totalNewStorage

def totalStorage : ℕ :=
  gbToMb 32

def remainingStorage : ℕ :=
  totalStorage - totalUsedStorage

def numSongs (storage : ℕ) (avgSongSize : ℕ) : ℕ :=
  storage / avgSongSize

theorem jeff_can_store_songs : 
  numSongs remainingStorage 20 = 320 :=
by
  sorry

end jeff_can_store_songs_l2188_218829


namespace complaint_online_prob_l2188_218821

/-- Define the various probability conditions -/
def prob_online := 4 / 5
def prob_store := 1 / 5
def qual_rate_online := 17 / 20
def qual_rate_store := 9 / 10
def non_qual_rate_online := 1 - qual_rate_online
def non_qual_rate_store := 1 - qual_rate_store
def prob_complaint_online := prob_online * non_qual_rate_online
def prob_complaint_store := prob_store * non_qual_rate_store
def total_prob_complaint := prob_complaint_online + prob_complaint_store

/-- The theorem states that given the conditions, the probability of an online purchase given a complaint is 6/7 -/
theorem complaint_online_prob : 
    (prob_complaint_online / total_prob_complaint) = 6 / 7 := 
by
    sorry

end complaint_online_prob_l2188_218821


namespace shooting_game_system_l2188_218819

theorem shooting_game_system :
  ∃ x y : ℕ, (x + y = 20 ∧ 3 * x = y) :=
by
  sorry

end shooting_game_system_l2188_218819


namespace point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l2188_218814

def is_on_line (n : ℕ) (a_n : ℕ) : Prop := a_n = 2 * n + 1

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n m, a n - a m = d * (n - m)

theorem point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence (a : ℕ → ℕ) :
  (∀ n, is_on_line n (a n)) → is_arithmetic_sequence a ∧ 
  ¬ (is_arithmetic_sequence a → ∀ n, is_on_line n (a n)) :=
sorry

end point_on_line_is_sufficient_but_not_necessary_condition_for_arithmetic_sequence_l2188_218814


namespace metals_inductive_reasoning_l2188_218860

def conducts_electricity (metal : String) : Prop :=
  metal = "Gold" ∨ metal = "Silver" ∨ metal = "Copper" ∨ metal = "Iron"

def all_metals_conduct_electricity (metals : List String) : Prop :=
  ∀ metal, metal ∈ metals → conducts_electricity metal

theorem metals_inductive_reasoning 
  (h1 : conducts_electricity "Gold")
  (h2 : conducts_electricity "Silver")
  (h3 : conducts_electricity "Copper")
  (h4 : conducts_electricity "Iron") :
  (all_metals_conduct_electricity ["Gold", "Silver", "Copper", "Iron"] → 
  all_metals_conduct_electricity ["All metals"]) :=
  sorry -- Proof skipped, as per instructions.

end metals_inductive_reasoning_l2188_218860


namespace river_flow_rate_l2188_218882

noncomputable def volume_per_minute : ℝ := 3200
noncomputable def depth_of_river : ℝ := 3
noncomputable def width_of_river : ℝ := 32
noncomputable def cross_sectional_area : ℝ := depth_of_river * width_of_river

noncomputable def flow_rate_m_per_minute : ℝ := volume_per_minute / cross_sectional_area
-- Conversion factors
noncomputable def minutes_per_hour : ℝ := 60
noncomputable def meters_per_km : ℝ := 1000

noncomputable def flow_rate_kmph : ℝ := (flow_rate_m_per_minute * minutes_per_hour) / meters_per_km

theorem river_flow_rate :
  flow_rate_kmph = 2 :=
by
  -- We skip the proof and use sorry to focus on the statement structure.
  sorry

end river_flow_rate_l2188_218882


namespace range_of_a_l2188_218832

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a <= x ∧ x < y ∧ y <= b → f y <= f x

theorem range_of_a (f : ℝ → ℝ) :
  odd_function f →
  decreasing_on_interval f (-1) 1 →
  (∀ a : ℝ, 0 < a ∧ a < 1 → f (1 - a) + f (2 * a - 1) < 0) →
  (∀ a : ℝ, 0 < a ∧ a < 1) :=
sorry

end range_of_a_l2188_218832


namespace largest_angle_sine_of_C_l2188_218898

-- Given conditions
def side_a : ℝ := 7
def side_b : ℝ := 3
def side_c : ℝ := 5

-- 1. Prove the largest angle
theorem largest_angle (a b c : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) : 
  ∃ A : ℝ, A = 120 :=
by
  sorry

-- 2. Prove the sine value of angle C
theorem sine_of_C (a b c A : ℝ) (h₁ : a = 7) (h₂ : b = 3) (h₃ : c = 5) (h₄ : A = 120) : 
  ∃ sinC : ℝ, sinC = 5 * (Real.sqrt 3) / 14 :=
by
  sorry

end largest_angle_sine_of_C_l2188_218898


namespace inequality_comparison_l2188_218858

theorem inequality_comparison 
  (a b : ℝ) (x y : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) :
  a^2 + b^2 ≥ (x + y)^2 :=
sorry

end inequality_comparison_l2188_218858


namespace national_flag_length_l2188_218896

-- Definitions from the conditions specified in the problem
def width : ℕ := 128
def ratio_length_to_width (L W : ℕ) : Prop := L / W = 3 / 2

-- The main theorem to prove
theorem national_flag_length (L : ℕ) (H : ratio_length_to_width L width) : L = 192 :=
by
  sorry

end national_flag_length_l2188_218896


namespace product_expression_evaluates_to_32_l2188_218853

theorem product_expression_evaluates_to_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  -- The proof itself is not required, hence we can put sorry here
  sorry

end product_expression_evaluates_to_32_l2188_218853


namespace books_inequality_system_l2188_218889

theorem books_inequality_system (x : ℕ) (n : ℕ) (h1 : x = 5 * n + 6) (h2 : (1 ≤ x - 7 * (x - 6) / 5 + 7)) :
  1 ≤ x - 7 * (x - 6) / 5 + 7 ∧ x - 7 * (x - 6) / 5 + 7 < 7 := 
by
  sorry

end books_inequality_system_l2188_218889


namespace work_days_l2188_218863

theorem work_days (p_can : ℕ → ℝ) (q_can : ℕ → ℝ) (together_can: ℕ → ℝ) :
  (together_can 6 = 1) ∧ (q_can 10 = 1) → (1 / (p_can x) + 1 / (q_can 10) = 1 / (together_can 6)) → (x = 15) :=
by
  sorry

end work_days_l2188_218863


namespace distinct_integer_pairs_l2188_218884

theorem distinct_integer_pairs :
  ∃ pairs : (Nat × Nat) → Prop,
  (∀ x y : Nat, pairs (x, y) → 0 < x ∧ x < y ∧ (8 * Real.sqrt 31 = Real.sqrt x + Real.sqrt y))
  ∧ (∃! p, pairs p) → (∃! q, pairs q) → (∃! r, pairs r) → true := sorry

end distinct_integer_pairs_l2188_218884


namespace total_amount_paid_l2188_218874

theorem total_amount_paid (grapes_kg mangoes_kg rate_grapes rate_mangoes : ℕ) 
    (h1 : grapes_kg = 8) (h2 : mangoes_kg = 8) 
    (h3 : rate_grapes = 70) (h4 : rate_mangoes = 55) : 
    (grapes_kg * rate_grapes + mangoes_kg * rate_mangoes) = 1000 :=
by
  sorry

end total_amount_paid_l2188_218874


namespace number_of_men_for_2km_road_l2188_218813

noncomputable def men_for_1km_road : ℕ := 30
noncomputable def days_for_1km_road : ℕ := 12
noncomputable def hours_per_day_for_1km_road : ℕ := 8
noncomputable def length_of_1st_road : ℕ := 1
noncomputable def length_of_2nd_road : ℕ := 2
noncomputable def working_hours_per_day_2nd_road : ℕ := 14
noncomputable def days_for_2km_road : ℝ := 20.571428571428573

theorem number_of_men_for_2km_road (total_man_hours_1km : ℕ := men_for_1km_road * days_for_1km_road * hours_per_day_for_1km_road):
  (men_for_1km_road * length_of_2nd_road * days_for_1km_road * hours_per_day_for_1km_road = 5760) →
  ∃ (men_for_2nd_road : ℕ), men_for_1km_road * 2 * days_for_1km_road * hours_per_day_for_1km_road = 5760 ∧  men_for_2nd_road * days_for_2km_road * working_hours_per_day_2nd_road = 5760 ∧ men_for_2nd_road = 20 :=
by {
  sorry
}

end number_of_men_for_2km_road_l2188_218813


namespace condition_for_ellipse_l2188_218897

theorem condition_for_ellipse (m : ℝ) : 
  (3 < m ∧ m < 7) ↔ (7 - m > 0 ∧ m - 3 > 0 ∧ (7 - m) ≠ (m - 3)) :=
by sorry

end condition_for_ellipse_l2188_218897


namespace diamondsuit_result_l2188_218887

def diam (a b : ℕ) : ℕ := a

theorem diamondsuit_result : (diam 7 (diam 4 8)) = 7 :=
by sorry

end diamondsuit_result_l2188_218887


namespace marble_distribution_correct_l2188_218830

def num_ways_to_distribute_marbles : ℕ :=
  -- Given:
  -- Evan divides 100 marbles among three volunteers with each getting at least one marble
  -- Lewis selects a positive integer n > 1 and for each volunteer, steals exactly 1/n of marbles if possible.
  -- Prove that the number of ways to distribute the marbles such that Lewis cannot steal from all volunteers
  3540

theorem marble_distribution_correct :
  num_ways_to_distribute_marbles = 3540 :=
sorry

end marble_distribution_correct_l2188_218830


namespace ab_sum_not_one_l2188_218804

theorem ab_sum_not_one (a b : ℝ) : a^2 + 2*a*b + b^2 + a + b - 2 ≠ 0 → a + b ≠ 1 :=
by
  intros h
  sorry

end ab_sum_not_one_l2188_218804


namespace union_A_B_complement_A_l2188_218866

-- Definition of Universe U
def U : Set ℝ := Set.univ

-- Definition of set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Definition of set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- Theorem 1: Proving the union A ∪ B
theorem union_A_B : A ∪ B = {x | -2 < x ∧ x ≤ 3} := 
sorry

-- Theorem 2: Proving the complement of A with respect to U
theorem complement_A : (U \ A) = {x | x < -1 ∨ x > 3} := 
sorry

end union_A_B_complement_A_l2188_218866


namespace smallest_N_divisors_of_8_l2188_218890

theorem smallest_N_divisors_of_8 (N : ℕ) (h0 : N % 10 = 0) (h8 : ∃ (divisors : ℕ), divisors = 8 ∧ (∀ k, k ∣ N → k ≤ divisors)) : N = 30 := 
sorry

end smallest_N_divisors_of_8_l2188_218890


namespace cupcakes_frosted_l2188_218894

def Cagney_rate := 1 / 25
def Lacey_rate := 1 / 35
def time_duration := 600
def combined_rate := Cagney_rate + Lacey_rate
def total_cupcakes := combined_rate * time_duration

theorem cupcakes_frosted (Cagney_rate Lacey_rate time_duration combined_rate total_cupcakes : ℝ) 
  (hC: Cagney_rate = 1 / 25)
  (hL: Lacey_rate = 1 / 35)
  (hT: time_duration = 600)
  (hCR: combined_rate = Cagney_rate + Lacey_rate)
  (hTC: total_cupcakes = combined_rate * time_duration) :
  total_cupcakes = 41 :=
sorry

end cupcakes_frosted_l2188_218894


namespace shadow_length_minor_fullness_l2188_218820

/-
An arithmetic sequence {a_n} where the length of shadows a_i decreases by the same amount, the conditions are:
1. The sum of the shadows on the Winter Solstice (a_1), the Beginning of Spring (a_4), and the Vernal Equinox (a_7) is 315 cun.
2. The sum of the shadows on the first nine solar terms is 855 cun.

We need to prove that the shadow length on Minor Fullness day (a_11) is 35 cun (i.e., 3 chi and 5 cun).
-/
theorem shadow_length_minor_fullness 
  (a : ℕ → ℕ) 
  (d : ℤ)
  (h1 : a 1 + a 4 + a 7 = 315) 
  (h2 : 9 * a 1 + 36 * d = 855) 
  (seq : ∀ n : ℕ, a n = a 1 + (n - 1) * d) :
  a 11 = 35 := 
by 
  sorry

end shadow_length_minor_fullness_l2188_218820


namespace min_containers_needed_l2188_218840

def container_capacity : ℕ := 500
def required_tea : ℕ := 5000

theorem min_containers_needed (n : ℕ) : n * container_capacity ≥ required_tea → n = 10 :=
sorry

end min_containers_needed_l2188_218840


namespace euler_totient_inequality_l2188_218869

variable {n : ℕ}
def even (n : ℕ) := ∃ k : ℕ, n = 2 * k
def positive (n : ℕ) := n > 0

theorem euler_totient_inequality (h_even : even n) (h_positive : positive n) : 
  Nat.totient n ≤ n / 2 :=
sorry

end euler_totient_inequality_l2188_218869


namespace zero_of_function_l2188_218843

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 4

theorem zero_of_function (x : ℝ) (h : f x = 0) (x1 x2 : ℝ)
  (h1 : -1 < x1 ∧ x1 < x)
  (h2 : x < x2 ∧ x2 < 2) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end zero_of_function_l2188_218843


namespace diameter_twice_radius_l2188_218850

theorem diameter_twice_radius (r d : ℝ) (h : d = 2 * r) : d = 2 * r :=
by
  exact h

end diameter_twice_radius_l2188_218850


namespace spherical_to_rectangular_coordinates_l2188_218837

theorem spherical_to_rectangular_coordinates :
  ∀ (ρ θ φ : ℝ),
  ρ = 5 → θ = π / 6 → φ = π / 3 →
  let x := ρ * (Real.sin φ * Real.cos θ)
  let y := ρ * (Real.sin φ * Real.sin θ)
  let z := ρ * Real.cos φ
  x = 15 / 4 ∧ y = 5 * Real.sqrt 3 / 4 ∧ z = 2.5 :=
by
  intros ρ θ φ hρ hθ hφ
  sorry

end spherical_to_rectangular_coordinates_l2188_218837


namespace func_identity_equiv_l2188_218878

theorem func_identity_equiv (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f (x) + f (y)) ↔ (∀ x y : ℝ, f (xy + x + y) = f (xy) + f (x) + f (y)) :=
by
  sorry

end func_identity_equiv_l2188_218878


namespace correct_multiplication_result_l2188_218857

theorem correct_multiplication_result (x : ℕ) (h : 9 * x = 153) : 6 * x = 102 :=
by {
  -- We would normally provide a detailed proof here, but as per instruction, we add sorry.
  sorry
}

end correct_multiplication_result_l2188_218857


namespace actual_number_of_children_l2188_218861

theorem actual_number_of_children (N : ℕ) (B : ℕ) 
  (h1 : B = 2 * N)
  (h2 : ∀ k : ℕ, k = N - 330)
  (h3 : B = 4 * (N - 330)) : 
  N = 660 :=
by 
  sorry

end actual_number_of_children_l2188_218861


namespace inequality_proof_l2188_218827

theorem inequality_proof (a b : ℝ) : 
  (a^4 + a^2 * b^2 + b^4) / 3 ≥ (a^3 * b + b^3 * a) / 2 :=
by
  sorry

end inequality_proof_l2188_218827


namespace sum_series_eq_3_div_4_l2188_218812

noncomputable def sum_series : ℝ := ∑' k, (k : ℝ) / 3^k

theorem sum_series_eq_3_div_4 : sum_series = 3 / 4 :=
sorry

end sum_series_eq_3_div_4_l2188_218812


namespace evaluate_division_l2188_218826

theorem evaluate_division : 64 / 0.08 = 800 := by
  sorry

end evaluate_division_l2188_218826


namespace not_possible_155_cents_five_coins_l2188_218864

/-- It is not possible to achieve a total value of 155 cents using exactly five coins 
    from a piggy bank containing only pennies (1 cent), nickels (5 cents), 
    quarters (25 cents), and half-dollars (50 cents). -/
theorem not_possible_155_cents_five_coins (n_pennies n_nickels n_quarters n_half_dollars : ℕ) 
    (h : n_pennies + n_nickels + n_quarters + n_half_dollars = 5) : 
    n_pennies * 1 + n_nickels * 5 + n_quarters * 25 + n_half_dollars * 50 ≠ 155 := 
sorry

end not_possible_155_cents_five_coins_l2188_218864


namespace parameterization_of_line_l2188_218873

theorem parameterization_of_line (t : ℝ) (g : ℝ → ℝ) 
  (h : ∀ t, (g t - 10) / 2 = t ) :
  g t = 5 * t + 10 := by
  sorry

end parameterization_of_line_l2188_218873


namespace age_of_eldest_child_l2188_218810

-- Define the conditions as hypotheses
def child_ages_sum_equals_50 (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50

-- Define the main theorem to prove the age of the eldest child
theorem age_of_eldest_child (x : ℕ) (h : child_ages_sum_equals_50 x) : x + 8 = 14 :=
sorry

end age_of_eldest_child_l2188_218810


namespace a_plus_b_eq_2007_l2188_218822

theorem a_plus_b_eq_2007 (a b : ℕ) (ha : Prime a) (hb : Odd b)
  (h : a^2 + b = 2009) : a + b = 2007 :=
by
  sorry

end a_plus_b_eq_2007_l2188_218822


namespace evaluate_sum_l2188_218809

variable {a b c : ℝ}

theorem evaluate_sum 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by
  sorry

end evaluate_sum_l2188_218809


namespace garden_area_eq_450_l2188_218803

theorem garden_area_eq_450
  (width length : ℝ)
  (fencing : ℝ := 60) 
  (length_eq_twice_width : length = 2 * width)
  (fencing_eq : 2 * width + length = fencing) :
  width * length = 450 := by
  sorry

end garden_area_eq_450_l2188_218803
