import Mathlib

namespace NUMINAMATH_GPT_Malou_third_quiz_score_l1220_122054

theorem Malou_third_quiz_score (q1 q2 q3 : ℕ) (avg_score : ℕ) (total_quizzes : ℕ) : 
  q1 = 91 ∧ q2 = 90 ∧ avg_score = 91 ∧ total_quizzes = 3 → q3 = 92 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Malou_third_quiz_score_l1220_122054


namespace NUMINAMATH_GPT_infinite_series_sum_l1220_122050

theorem infinite_series_sum :
  (∑' n : ℕ, (3^n) / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l1220_122050


namespace NUMINAMATH_GPT_train_speed_l1220_122055

theorem train_speed
  (train_length : ℝ) (platform_length : ℝ) (time_seconds : ℝ)
  (h_train_length : train_length = 450)
  (h_platform_length : platform_length = 300.06)
  (h_time : time_seconds = 25) :
  (train_length + platform_length) / time_seconds * 3.6 = 108.01 :=
by
  -- skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_train_speed_l1220_122055


namespace NUMINAMATH_GPT_jungkook_needs_more_paper_l1220_122085

def bundles : Nat := 5
def pieces_per_bundle : Nat := 8
def rows : Nat := 9
def sheets_per_row : Nat := 6

def total_pieces : Nat := bundles * pieces_per_bundle
def pieces_needed : Nat := rows * sheets_per_row
def pieces_missing : Nat := pieces_needed - total_pieces

theorem jungkook_needs_more_paper : pieces_missing = 14 := by
  sorry

end NUMINAMATH_GPT_jungkook_needs_more_paper_l1220_122085


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1220_122033

theorem no_positive_integer_solutions:
    ∀ x y : ℕ, x > 0 → y > 0 → x^2 + 2 * y^2 = 2 * x^3 - x → false :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1220_122033


namespace NUMINAMATH_GPT_expression_evaluation_l1220_122020

theorem expression_evaluation (a b : ℤ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1220_122020


namespace NUMINAMATH_GPT_problem1_solve_eq_l1220_122042

theorem problem1_solve_eq (x : ℝ) : x * (x - 5) = 3 * x - 15 ↔ (x = 5 ∨ x = 3) := by
  sorry

end NUMINAMATH_GPT_problem1_solve_eq_l1220_122042


namespace NUMINAMATH_GPT_nathan_write_in_one_hour_l1220_122090

/-- Jacob can write twice as fast as Nathan. Nathan wrote some letters in one hour. Together, they can write 750 letters in 10 hours. How many letters can Nathan write in one hour? -/
theorem nathan_write_in_one_hour
  (N : ℕ)  -- Assume N is the number of letters Nathan can write in one hour
  (H₁ : ∀ (J : ℕ), J = 2 * N)  -- Jacob writes twice faster, so letters written by Jacob in one hour is 2N
  (H₂ : 10 * (N + 2 * N) = 750)  -- Together they write 750 letters in 10 hours
  : N = 25 := by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_nathan_write_in_one_hour_l1220_122090


namespace NUMINAMATH_GPT_positive_integer_solutions_count_3x_plus_4y_eq_1024_l1220_122070

theorem positive_integer_solutions_count_3x_plus_4y_eq_1024 :
  (∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 3 * x + 4 * y = 1024) ∧ 
  (∀ n, n = 85 → ∃! (s : ℕ × ℕ), s.fst > 0 ∧ s.snd > 0 ∧ 3 * s.fst + 4 * s.snd = 1024 ∧ n = 85) := 
sorry

end NUMINAMATH_GPT_positive_integer_solutions_count_3x_plus_4y_eq_1024_l1220_122070


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1220_122068

variable {f : ℝ → ℝ}

noncomputable def F (x : ℝ) : ℝ := x^2 * f x

theorem solution_set_of_inequality
  (h_diff : ∀ x < 0, DifferentiableAt ℝ f x) 
  (h_cond : ∀ x < 0, 2 * f x + x * (deriv f x) > x^2) :
  ∀ x, ((x + 2016)^2 * f (x + 2016) - 9 * f (-3) < 0) ↔ (-2019 < x ∧ x < -2016) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1220_122068


namespace NUMINAMATH_GPT_a_plus_b_equals_4_l1220_122083

theorem a_plus_b_equals_4 (f : ℝ → ℝ) (a b : ℝ) (h_dom : ∀ x, 1 ≤ x ∧ x ≤ b → f x = (1/2) * (x-1)^2 + a)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b → ∃ x, 1 ≤ x ∧ x ≤ b ∧ f x = y) (h_b_pos : b > 1) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_a_plus_b_equals_4_l1220_122083


namespace NUMINAMATH_GPT_high_fever_temperature_l1220_122041

theorem high_fever_temperature (T t : ℝ) (h1 : T = 36) (h2 : t > 13 / 12 * T) : t > 39 :=
by
  sorry

end NUMINAMATH_GPT_high_fever_temperature_l1220_122041


namespace NUMINAMATH_GPT_different_quantifiers_not_equiv_l1220_122061

theorem different_quantifiers_not_equiv {x₀ : ℝ} :
  (∃ x₀ : ℝ, x₀^2 > 3) ↔ ¬ (∀ x₀ : ℝ, x₀^2 > 3) :=
by
  sorry

end NUMINAMATH_GPT_different_quantifiers_not_equiv_l1220_122061


namespace NUMINAMATH_GPT_largest_possible_markers_in_package_l1220_122045

theorem largest_possible_markers_in_package (alex_markers jordan_markers : ℕ) 
  (h1 : alex_markers = 56)
  (h2 : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_markers_in_package_l1220_122045


namespace NUMINAMATH_GPT_find_y_l1220_122095

theorem find_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l1220_122095


namespace NUMINAMATH_GPT_least_people_cheaper_second_caterer_l1220_122066

noncomputable def cost_first_caterer (x : ℕ) : ℕ := 50 + 18 * x

noncomputable def cost_second_caterer (x : ℕ) : ℕ := 
  if x >= 30 then 150 + 15 * x else 180 + 15 * x

theorem least_people_cheaper_second_caterer : ∃ x : ℕ, x = 34 ∧ x >= 30 ∧ cost_second_caterer x < cost_first_caterer x :=
by
  sorry

end NUMINAMATH_GPT_least_people_cheaper_second_caterer_l1220_122066


namespace NUMINAMATH_GPT_c_share_l1220_122023

theorem c_share (x : ℕ) (a b c d : ℕ) 
  (h1: a = 5 * x)
  (h2: b = 3 * x)
  (h3: c = 2 * x)
  (h4: d = 3 * x)
  (h5: a = b + 1000): 
  c = 1000 := 
by 
  sorry

end NUMINAMATH_GPT_c_share_l1220_122023


namespace NUMINAMATH_GPT_solve_congruence_l1220_122046

-- Define the initial condition of the problem
def condition (x : ℤ) : Prop := (15 * x + 3) % 21 = 9 % 21

-- The statement that we want to prove
theorem solve_congruence : ∃ (a m : ℤ), condition a ∧ a % m = 6 % 7 ∧ a < m ∧ a + m = 13 :=
by {
    sorry
}

end NUMINAMATH_GPT_solve_congruence_l1220_122046


namespace NUMINAMATH_GPT_range_f3_l1220_122086

def function_f (a c x : ℝ) : ℝ := a * x^2 - c

theorem range_f3 (a c : ℝ) :
  (-4 ≤ function_f a c 1) ∧ (function_f a c 1 ≤ -1) →
  (-1 ≤ function_f a c 2) ∧ (function_f a c 2 ≤ 5) →
  -12 ≤ function_f a c 3 ∧ function_f a c 3 ≤ 1.75 :=
by
  sorry

end NUMINAMATH_GPT_range_f3_l1220_122086


namespace NUMINAMATH_GPT_find_other_number_l1220_122039

theorem find_other_number (A : ℕ) (hcf_cond : Nat.gcd A 48 = 12) (lcm_cond : Nat.lcm A 48 = 396) : A = 99 := by
    sorry

end NUMINAMATH_GPT_find_other_number_l1220_122039


namespace NUMINAMATH_GPT_number_of_piles_l1220_122040

theorem number_of_piles (n : ℕ) (h₁ : 1000 < n) (h₂ : n < 2000)
  (h3 : n % 2 = 1) (h4 : n % 3 = 1) (h5 : n % 4 = 1) 
  (h6 : n % 5 = 1) (h7 : n % 6 = 1) (h8 : n % 7 = 1) (h9 : n % 8 = 1) : 
  ∃ p, p ≠ 1 ∧ p ≠ n ∧ (n % p = 0) ∧ p = 41 :=
by
  sorry

end NUMINAMATH_GPT_number_of_piles_l1220_122040


namespace NUMINAMATH_GPT_metric_regression_equation_l1220_122049

noncomputable def predicted_weight_imperial (height : ℝ) : ℝ :=
  4 * height - 130

def inch_to_cm (inch : ℝ) : ℝ := 2.54 * inch
def pound_to_kg (pound : ℝ) : ℝ := 0.45 * pound

theorem metric_regression_equation (height_cm : ℝ) :
  (0.72 * height_cm - 58.5) = 
  (pound_to_kg (predicted_weight_imperial (height_cm / 2.54))) :=
by
  sorry

end NUMINAMATH_GPT_metric_regression_equation_l1220_122049


namespace NUMINAMATH_GPT_true_propositions_count_l1220_122094

-- Original Proposition
def P (x y : ℝ) : Prop := x^2 + y^2 = 0 → x = 0 ∧ y = 0

-- Converse Proposition
def Q (x y : ℝ) : Prop := x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Contrapositive Proposition
def contrapositive_Q_P (x y : ℝ) : Prop := (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0)

-- Inverse Proposition
def inverse_P (x y : ℝ) : Prop := (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)

-- Problem Statement
theorem true_propositions_count : ∀ (x y : ℝ),
  P x y ∧ Q x y ∧ contrapositive_Q_P x y ∧ inverse_P x y → 3 = 3 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_true_propositions_count_l1220_122094


namespace NUMINAMATH_GPT_exists_zero_in_interval_l1220_122005

open Set Real

theorem exists_zero_in_interval (f : ℝ → ℝ) (a b : ℝ) (h_cont : ContinuousOn f (Icc a b)) 
  (h_pos : f a * f b > 0) : ∃ c ∈ Ioo a b, f c = 0 := sorry

end NUMINAMATH_GPT_exists_zero_in_interval_l1220_122005


namespace NUMINAMATH_GPT_find_B_squared_l1220_122056

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt 23 + 105 / x

theorem find_B_squared :
  ∃ B : ℝ, (B = (Real.sqrt 443)) ∧ (B^2 = 443) :=
by
  sorry

end NUMINAMATH_GPT_find_B_squared_l1220_122056


namespace NUMINAMATH_GPT_chicken_problem_l1220_122062

theorem chicken_problem (x y z : ℕ) :
  x + y + z = 100 ∧ 5 * x + 3 * y + z / 3 = 100 → 
  (x = 0 ∧ y = 25 ∧ z = 75) ∨ 
  (x = 12 ∧ y = 4 ∧ z = 84) ∨ 
  (x = 8 ∧ y = 11 ∧ z = 81) ∨ 
  (x = 4 ∧ y = 18 ∧ z = 78) := 
sorry

end NUMINAMATH_GPT_chicken_problem_l1220_122062


namespace NUMINAMATH_GPT_solution_set_inequality_l1220_122008

variable (a x : ℝ)

-- Conditions
theorem solution_set_inequality (h₀ : 0 < a) (h₁ : a < 1) :
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) := 
by 
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1220_122008


namespace NUMINAMATH_GPT_chef_earns_less_than_manager_l1220_122064

noncomputable def manager_wage : ℚ := 8.50
noncomputable def dishwasher_wage : ℚ := manager_wage / 2
noncomputable def chef_wage : ℚ := dishwasher_wage + 0.22 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 := by
  sorry

end NUMINAMATH_GPT_chef_earns_less_than_manager_l1220_122064


namespace NUMINAMATH_GPT_percent_non_unionized_women_is_80_l1220_122022

noncomputable def employeeStatistics :=
  let total_employees := 100
  let percent_men := 50
  let percent_unionized := 60
  let percent_unionized_men := 70
  let men := (percent_men / 100) * total_employees
  let unionized := (percent_unionized / 100) * total_employees
  let unionized_men := (percent_unionized_men / 100) * unionized
  let non_unionized_men := men - unionized_men
  let non_unionized := total_employees - unionized
  let non_unionized_women := non_unionized - non_unionized_men
  let percent_non_unionized_women := (non_unionized_women / non_unionized) * 100
  percent_non_unionized_women

theorem percent_non_unionized_women_is_80 :
  employeeStatistics = 80 :=
by
  sorry

end NUMINAMATH_GPT_percent_non_unionized_women_is_80_l1220_122022


namespace NUMINAMATH_GPT_calculate_expression_l1220_122075

theorem calculate_expression :
  (-2)^(4^2) + 2^(3^2) = 66048 := by sorry

end NUMINAMATH_GPT_calculate_expression_l1220_122075


namespace NUMINAMATH_GPT_find_divisor_l1220_122038

-- Defining the conditions
def dividend : ℕ := 181
def quotient : ℕ := 9
def remainder : ℕ := 1

-- The statement to prove
theorem find_divisor : ∃ (d : ℕ), dividend = (d * quotient) + remainder ∧ d = 20 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l1220_122038


namespace NUMINAMATH_GPT_age_calculation_l1220_122072

/-- Let Thomas be a 6-year-old child, Shay be 13 years older than Thomas, 
and also 5 years younger than James. Let Violet be 3 years younger than 
Thomas, and Emily be the same age as Shay. This theorem proves that when 
Violet reaches the age of Thomas (6 years old), James will be 27 years old 
and Emily will be 22 years old. -/
theorem age_calculation : 
  ∀ (Thomas Shay James Violet Emily : ℕ),
    Thomas = 6 →
    Shay = Thomas + 13 →
    James = Shay + 5 →
    Violet = Thomas - 3 →
    Emily = Shay →
    (Violet + (6 - Violet) = 6) →
    (James + (6 - Violet) = 27 ∧ Emily + (6 - Violet) = 22) :=
by
  intros Thomas Shay James Violet Emily ht hs hj hv he hv_diff
  sorry

end NUMINAMATH_GPT_age_calculation_l1220_122072


namespace NUMINAMATH_GPT_find_number_l1220_122092

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 58) : x = 145 := by
  sorry

end NUMINAMATH_GPT_find_number_l1220_122092


namespace NUMINAMATH_GPT_circle_circumference_l1220_122093

noncomputable def circumference_of_circle (speed1 speed2 time : ℝ) : ℝ :=
  let distance1 := speed1 * time
  let distance2 := speed2 * time
  distance1 + distance2

theorem circle_circumference
    (speed1 speed2 time : ℝ)
    (h1 : speed1 = 7)
    (h2 : speed2 = 8)
    (h3 : time = 12) :
    circumference_of_circle speed1 speed2 time = 180 := by
  sorry

end NUMINAMATH_GPT_circle_circumference_l1220_122093


namespace NUMINAMATH_GPT_find_speed_of_second_car_l1220_122057

noncomputable def problem : Prop := 
  let s1 := 1600 -- meters
  let s2 := 800 -- meters
  let v1 := 72 / 3.6 -- converting to meters per second for convenience; 72 km/h = 20 m/s
  let s := 200 -- meters
  let t1 := s1 / v1 -- time taken by the first car to reach the intersection
  let l1 := s2 - s -- scenario 1: second car travels 600 meters
  let l2 := s2 + s -- scenario 2: second car travels 1000 meters
  let v2_1 := l1 / t1 -- speed calculation for scenario 1
  let v2_2 := l2 / t1 -- speed calculation for scenario 2
  v2_1 = 7.5 ∧ v2_2 = 12.5 -- expected speeds in both scenarios

theorem find_speed_of_second_car : problem := sorry

end NUMINAMATH_GPT_find_speed_of_second_car_l1220_122057


namespace NUMINAMATH_GPT_Ram_Gohul_days_work_together_l1220_122059

-- Define the conditions
def Ram_days := 10
def Gohul_days := 15

-- Define the work rates
def Ram_rate := 1 / Ram_days
def Gohul_rate := 1 / Gohul_days

-- Define the combined work rate
def Combined_rate := Ram_rate + Gohul_rate

-- Define the number of days to complete the job together
def Together_days := 1 / Combined_rate

-- State the proof problem
theorem Ram_Gohul_days_work_together : Together_days = 6 := by
  sorry

end NUMINAMATH_GPT_Ram_Gohul_days_work_together_l1220_122059


namespace NUMINAMATH_GPT_first_year_after_2020_with_sum_15_l1220_122076

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_after_2020_with_sum_15 :
  ∀ n, n > 2020 → (sum_of_digits n = 15 ↔ n = 2058) := by
  sorry

end NUMINAMATH_GPT_first_year_after_2020_with_sum_15_l1220_122076


namespace NUMINAMATH_GPT_boy_and_girl_roles_l1220_122037

-- Definitions of the conditions
def Sasha_says_boy : Prop := True
def Zhenya_says_girl : Prop := True
def at_least_one_lying (sasha_boy zhenya_girl : Prop) : Prop := 
  (sasha_boy = False) ∨ (zhenya_girl = False)

-- Theorem statement
theorem boy_and_girl_roles (sasha_boy : Prop) (zhenya_girl : Prop) 
  (H1 : Sasha_says_boy) (H2 : Zhenya_says_girl) (H3 : at_least_one_lying sasha_boy zhenya_girl) :
  sasha_boy = False ∧ zhenya_girl = True :=
sorry

end NUMINAMATH_GPT_boy_and_girl_roles_l1220_122037


namespace NUMINAMATH_GPT_parallel_vectors_l1220_122063

variable (x : ℝ)
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (x, -2)

theorem parallel_vectors (h : (1 * (-2) - 2 * x = 0)) : x = -1 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_l1220_122063


namespace NUMINAMATH_GPT_emily_lives_l1220_122004

theorem emily_lives :
  ∃ (lives_gained : ℕ), 
    let initial_lives := 42
    let lives_lost := 25
    let lives_after_loss := initial_lives - lives_lost
    let final_lives := 41
    lives_after_loss + lives_gained = final_lives :=
sorry

end NUMINAMATH_GPT_emily_lives_l1220_122004


namespace NUMINAMATH_GPT_product_and_sum_of_roots_l1220_122047

theorem product_and_sum_of_roots :
  let a := 24
  let b := 60
  let c := -600
  (c / a = -25) ∧ (-b / a = -2.5) := 
by
  sorry

end NUMINAMATH_GPT_product_and_sum_of_roots_l1220_122047


namespace NUMINAMATH_GPT_find_solutions_l1220_122029

theorem find_solutions (x y z : ℝ) :
  (x = 5 / 3 ∧ y = -4 / 3 ∧ z = -4 / 3) ∨
  (x = 4 / 3 ∧ y = 4 / 3 ∧ z = -5 / 3) →
  (x^2 - y * z = abs (y - z) + 1) ∧ 
  (y^2 - z * x = abs (z - x) + 1) ∧ 
  (z^2 - x * y = abs (x - y) + 1) :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_l1220_122029


namespace NUMINAMATH_GPT_inequality_proof_l1220_122013

theorem inequality_proof
  (x1 x2 x3 x4 x5 : ℝ)
  (hx1 : 0 < x1)
  (hx2 : 0 < x2)
  (hx3 : 0 < x3)
  (hx4 : 0 < x4)
  (hx5 : 0 < x5) :
  x1^2 + x2^2 + x3^2 + x4^2 + x5^2 ≥ x1 * (x2 + x3 + x4 + x5) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1220_122013


namespace NUMINAMATH_GPT_expression_evaluation_l1220_122058

theorem expression_evaluation :
  (4 * 6 / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) - 1 = 0) :=
by sorry

end NUMINAMATH_GPT_expression_evaluation_l1220_122058


namespace NUMINAMATH_GPT_evaluate_expression_l1220_122030

theorem evaluate_expression : (3200 - 3131) ^ 2 / 121 = 36 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1220_122030


namespace NUMINAMATH_GPT_percentage_born_in_september_l1220_122007

theorem percentage_born_in_september (total famous : ℕ) (born_in_september : ℕ) (h1 : total = 150) (h2 : born_in_september = 12) :
  (born_in_september * 100 / total) = 8 :=
by
  sorry

end NUMINAMATH_GPT_percentage_born_in_september_l1220_122007


namespace NUMINAMATH_GPT_cora_cookies_per_day_l1220_122016

theorem cora_cookies_per_day :
  (∀ (day : ℕ), day ∈ (Finset.range 30) →
    ∃ cookies_per_day : ℕ,
    cookies_per_day * 30 = 1620 / 18) →
  cookies_per_day = 3 := by
  sorry

end NUMINAMATH_GPT_cora_cookies_per_day_l1220_122016


namespace NUMINAMATH_GPT_circle_area_increase_l1220_122053

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  percentage_increase = 125 :=
by
  let r_new := 1.5 * r
  let area_original := π * r^2
  let area_new := π * r_new^2
  let increase := area_new - area_original
  let percentage_increase := (increase / area_original) * 100
  sorry

end NUMINAMATH_GPT_circle_area_increase_l1220_122053


namespace NUMINAMATH_GPT_average_last_three_l1220_122087

theorem average_last_three {a b c d e f g : ℝ} 
  (h_avg_all : (a + b + c + d + e + f + g) / 7 = 60)
  (h_avg_first_four : (a + b + c + d) / 4 = 55) : 
  (e + f + g) / 3 = 200 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_last_three_l1220_122087


namespace NUMINAMATH_GPT_digit_7_occurrences_in_range_1_to_2017_l1220_122012

-- Define the predicate that checks if a digit appears in a number
def digit_occurrences (d n : Nat) : Nat :=
  Nat.digits 10 n |>.count d

-- Define the range of numbers we are interested in
def range := (List.range' 1 2017)

-- Sum up the occurrences of digit 7 in the defined range
def total_occurrences (d : Nat) (range : List Nat) : Nat :=
  range.foldr (λ n acc => digit_occurrences d n + acc) 0

-- The main theorem to prove
theorem digit_7_occurrences_in_range_1_to_2017 : total_occurrences 7 range = 602 := by
  -- The proof should go here, but we only need to define the statement.
  sorry

end NUMINAMATH_GPT_digit_7_occurrences_in_range_1_to_2017_l1220_122012


namespace NUMINAMATH_GPT_negation_of_existence_l1220_122031

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem negation_of_existence:
  (∃ x : ℝ, log_base 3 x ≤ 0) ↔ ∀ x : ℝ, log_base 3 x < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l1220_122031


namespace NUMINAMATH_GPT_beef_original_weight_l1220_122080

noncomputable def originalWeightBeforeProcessing (weightAfterProcessing : ℝ) (lossPercentage : ℝ) : ℝ :=
  weightAfterProcessing / (1 - lossPercentage / 100)

theorem beef_original_weight : originalWeightBeforeProcessing 570 35 = 876.92 :=
by
  sorry

end NUMINAMATH_GPT_beef_original_weight_l1220_122080


namespace NUMINAMATH_GPT_remainder_of_2_pow_30_plus_3_mod_7_l1220_122010

theorem remainder_of_2_pow_30_plus_3_mod_7 :
  (2^30 + 3) % 7 = 4 := 
sorry

end NUMINAMATH_GPT_remainder_of_2_pow_30_plus_3_mod_7_l1220_122010


namespace NUMINAMATH_GPT_sum_of_digits_B_l1220_122074

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).foldl (· + ·) 0

def A : ℕ := sum_of_digits (4444 ^ 4444)

def B : ℕ := sum_of_digits A

theorem sum_of_digits_B : 
  sum_of_digits B = 7 := by
    sorry

end NUMINAMATH_GPT_sum_of_digits_B_l1220_122074


namespace NUMINAMATH_GPT_lada_vs_elevator_l1220_122011

def Lada_speed_ratio (V U : ℝ) (S : ℝ) : Prop :=
  (∃ t_wait t_wait' : ℝ,
  ((t_wait = 3*S/U - 3*S/V) ∧ (t_wait' = 7*S/(2*U) - 7*S/V)) ∧
   (t_wait' = 3 * t_wait)) →
  U = 11/4 * V

theorem lada_vs_elevator (V U : ℝ) (S : ℝ) : Lada_speed_ratio V U S :=
sorry

end NUMINAMATH_GPT_lada_vs_elevator_l1220_122011


namespace NUMINAMATH_GPT_finite_non_friends_iff_l1220_122097

def isFriend (u n : ℕ) : Prop :=
  ∃ N : ℕ, N % n = 0 ∧ (N.digits 10).sum = u

theorem finite_non_friends_iff (n : ℕ) : (∃ᶠ u in at_top, ¬ isFriend u n) ↔ ¬ (3 ∣ n) := 
by
  sorry

end NUMINAMATH_GPT_finite_non_friends_iff_l1220_122097


namespace NUMINAMATH_GPT_find_angle_B_l1220_122077

theorem find_angle_B 
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) :
  B = 60 :=
sorry

end NUMINAMATH_GPT_find_angle_B_l1220_122077


namespace NUMINAMATH_GPT_total_amount_l1220_122021

theorem total_amount (a b c total first : ℕ)
  (h1 : a = 1 / 2) (h2 : b = 2 / 3) (h3 : c = 3 / 4)
  (h4 : first = 204)
  (ratio_sum : a * 12 + b * 12 + c * 12 = 23)
  (first_ratio : a * 12 = 6) :
  total = 23 * (first / 6) → total = 782 :=
by 
  sorry

end NUMINAMATH_GPT_total_amount_l1220_122021


namespace NUMINAMATH_GPT_sum_put_at_simple_interest_l1220_122096

theorem sum_put_at_simple_interest (P R : ℝ) 
  (h : ((P * (R + 3) * 2) / 100) - ((P * R * 2) / 100) = 300) : 
  P = 5000 :=
by
  sorry

end NUMINAMATH_GPT_sum_put_at_simple_interest_l1220_122096


namespace NUMINAMATH_GPT_calculate_fraction_l1220_122098

theorem calculate_fraction :
  (10^9 / (2 * 10^5) = 5000) :=
  sorry

end NUMINAMATH_GPT_calculate_fraction_l1220_122098


namespace NUMINAMATH_GPT_solve_for_x_l1220_122079

theorem solve_for_x (x : ℝ) (h : x ≠ 0) : (3 * x)^5 = (9 * x)^4 → x = 27 := 
by 
  admit

end NUMINAMATH_GPT_solve_for_x_l1220_122079


namespace NUMINAMATH_GPT_max_apartment_size_l1220_122071

theorem max_apartment_size (rate cost per_sqft : ℝ) (budget : ℝ) (h1 : rate = 1.20) (h2 : budget = 864) : cost = 720 :=
by
  sorry

end NUMINAMATH_GPT_max_apartment_size_l1220_122071


namespace NUMINAMATH_GPT_ball_beyond_hole_l1220_122002

theorem ball_beyond_hole
  (first_turn_distance : ℕ)
  (second_turn_distance : ℕ)
  (total_distance_to_hole : ℕ) :
  first_turn_distance = 180 →
  second_turn_distance = first_turn_distance / 2 →
  total_distance_to_hole = 250 →
  second_turn_distance - (total_distance_to_hole - first_turn_distance) = 20 :=
by
  intros
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_ball_beyond_hole_l1220_122002


namespace NUMINAMATH_GPT_pole_intersection_height_l1220_122028

theorem pole_intersection_height :
  ∀ (d h1 h2 : ℝ), d = 120 ∧ h1 = 30 ∧ h2 = 90 → 
  ∃ y : ℝ, y = 18 :=
by
  sorry

end NUMINAMATH_GPT_pole_intersection_height_l1220_122028


namespace NUMINAMATH_GPT_carmen_parsley_left_l1220_122015

theorem carmen_parsley_left (plates_whole_sprig : ℕ) (plates_half_sprig : ℕ) (initial_sprigs : ℕ) :
  plates_whole_sprig = 8 →
  plates_half_sprig = 12 →
  initial_sprigs = 25 →
  initial_sprigs - (plates_whole_sprig + plates_half_sprig / 2) = 11 := by
  intros
  sorry

end NUMINAMATH_GPT_carmen_parsley_left_l1220_122015


namespace NUMINAMATH_GPT_value_of_f_at_4_l1220_122035

noncomputable def f (α : ℝ) (x : ℝ) := x^α

theorem value_of_f_at_4 : 
  (∃ α : ℝ, f α 2 = (Real.sqrt 2) / 2) → f (-1 / 2) 4 = 1 / 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_value_of_f_at_4_l1220_122035


namespace NUMINAMATH_GPT_ordered_pair_represents_5_1_l1220_122082

structure OrderedPair (α : Type) :=
  (fst : α)
  (snd : α)

def represents_rows_cols (pair : OrderedPair ℝ) (rows cols : ℕ) : Prop :=
  pair.fst = rows ∧ pair.snd = cols

theorem ordered_pair_represents_5_1 :
  represents_rows_cols (OrderedPair.mk 2 3) 2 3 →
  represents_rows_cols (OrderedPair.mk 5 1) 5 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_ordered_pair_represents_5_1_l1220_122082


namespace NUMINAMATH_GPT_new_roots_quadratic_l1220_122052

variable {p q : ℝ}

theorem new_roots_quadratic :
  (∀ (r₁ r₂ : ℝ), r₁ + r₂ = -p ∧ r₁ * r₂ = q → 
  (x : ℝ) → x^2 + ((p^2 - 2 * q)^2 - 2 * q^2) * x + q^4 = 0) :=
by 
  intros r₁ r₂ h x
  have : r₁ + r₂ = -p := h.1
  have : r₁ * r₂ = q := h.2
  sorry

end NUMINAMATH_GPT_new_roots_quadratic_l1220_122052


namespace NUMINAMATH_GPT_harmonic_mean_of_1_3_1_div_2_l1220_122032

noncomputable def harmonicMean (a b c : ℝ) : ℝ :=
  let reciprocals := [1 / a, 1 / b, 1 / c]
  (reciprocals.sum) / reciprocals.length

theorem harmonic_mean_of_1_3_1_div_2 : harmonicMean 1 3 (1 / 2) = 9 / 10 :=
  sorry

end NUMINAMATH_GPT_harmonic_mean_of_1_3_1_div_2_l1220_122032


namespace NUMINAMATH_GPT_repeated_number_divisible_by_1001001_l1220_122073

theorem repeated_number_divisible_by_1001001 (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  (1000000 * (100 * a + 10 * b + c) + 1000 * (100 * a + 10 * b + c) + (100 * a + 10 * b + c)) % 1001001 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_repeated_number_divisible_by_1001001_l1220_122073


namespace NUMINAMATH_GPT_find_original_mean_l1220_122024

noncomputable def original_mean (M : ℝ) : Prop :=
  let num_observations := 50
  let decrement := 47
  let updated_mean := 153
  M * num_observations - (num_observations * decrement) = updated_mean * num_observations

theorem find_original_mean : original_mean 200 :=
by
  unfold original_mean
  simp [*, mul_sub_left_distrib] at *
  sorry

end NUMINAMATH_GPT_find_original_mean_l1220_122024


namespace NUMINAMATH_GPT_solve_for_x_l1220_122048

theorem solve_for_x (x : ℝ) (h : -200 * x = 1600) : x = -8 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1220_122048


namespace NUMINAMATH_GPT_identity_proof_l1220_122019

theorem identity_proof (a b c x y z : ℝ) : 
  (a * x + b * y + c * z) ^ 2 + (b * x + c * y + a * z) ^ 2 + (c * x + a * y + b * z) ^ 2 = 
  (c * x + b * y + a * z) ^ 2 + (b * x + a * y + c * z) ^ 2 + (a * x + c * y + b * z) ^ 2 := 
by
  sorry

end NUMINAMATH_GPT_identity_proof_l1220_122019


namespace NUMINAMATH_GPT_intersection_P_Q_l1220_122088

def P := {x : ℝ | x^2 - 9 < 0}
def Q := {y : ℤ | ∃ x : ℤ, y = 2*x}

theorem intersection_P_Q :
  {x : ℝ | x ∈ P ∧ (∃ n : ℤ, x = 2*n)} = {-2, 0, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l1220_122088


namespace NUMINAMATH_GPT_second_player_wins_for_n_11_l1220_122017

theorem second_player_wins_for_n_11 (N : ℕ) (h1 : N = 11) :
  ∃ (list : List ℕ), (∀ x ∈ list, x > 0 ∧ x ≤ 25) ∧
     list.sum ≥ 200 ∧
     (∃ sublist : List ℕ, sublist.sum ≥ 200 - N ∧ sublist.sum ≤ 200 + N) :=
by
  let N := 11
  sorry

end NUMINAMATH_GPT_second_player_wins_for_n_11_l1220_122017


namespace NUMINAMATH_GPT_number_of_integers_having_squares_less_than_10_million_l1220_122069

theorem number_of_integers_having_squares_less_than_10_million : 
  ∃ n : ℕ, (n = 3162) ∧ (∀ k : ℕ, k ≤ 3162 → (k^2 < 10^7)) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_integers_having_squares_less_than_10_million_l1220_122069


namespace NUMINAMATH_GPT_derivative_at_1_l1220_122026

def f (x : ℝ) : ℝ := (1 - 2 * x^3) ^ 10

theorem derivative_at_1 : deriv f 1 = 60 :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_1_l1220_122026


namespace NUMINAMATH_GPT_hexagon_cookie_cutters_count_l1220_122060

-- Definitions for the conditions
def triangle_side_count := 3
def triangles := 6
def square_side_count := 4
def squares := 4
def total_sides := 46

-- Given conditions translated to Lean 4
def sides_from_triangles := triangles * triangle_side_count
def sides_from_squares := squares * square_side_count
def sides_from_triangles_and_squares := sides_from_triangles + sides_from_squares
def sides_from_hexagons := total_sides - sides_from_triangles_and_squares
def hexagon_side_count := 6

-- Statement to prove that there are 2 hexagon-shaped cookie cutters
theorem hexagon_cookie_cutters_count : sides_from_hexagons / hexagon_side_count = 2 := by
  sorry

end NUMINAMATH_GPT_hexagon_cookie_cutters_count_l1220_122060


namespace NUMINAMATH_GPT_sector_arc_length_circumference_ratio_l1220_122051

theorem sector_arc_length_circumference_ratio
  {r : ℝ}
  (h_radius : ∀ (sector_radius : ℝ), sector_radius = 2/3 * r)
  (h_area : ∀ (sector_area circle_area : ℝ), sector_area / circle_area = 5/27) :
  ∀ (l C : ℝ), l / C = 5 / 18 :=
by
  -- Prove the theorem using the given hypothesis.
  -- Construction of the detailed proof will go here.
  sorry

end NUMINAMATH_GPT_sector_arc_length_circumference_ratio_l1220_122051


namespace NUMINAMATH_GPT_least_product_of_distinct_primes_gt_30_l1220_122044

theorem least_product_of_distinct_primes_gt_30 :
  ∃ p q : ℕ, p > 30 ∧ q > 30 ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p * q = 1147 :=
by
  sorry

end NUMINAMATH_GPT_least_product_of_distinct_primes_gt_30_l1220_122044


namespace NUMINAMATH_GPT_negation_of_prop_l1220_122084

theorem negation_of_prop :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
sorry

end NUMINAMATH_GPT_negation_of_prop_l1220_122084


namespace NUMINAMATH_GPT_scientific_notation_of_120000_l1220_122000

theorem scientific_notation_of_120000 : 
  (120000 : ℝ) = 1.2 * 10^5 := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_of_120000_l1220_122000


namespace NUMINAMATH_GPT_range_of_b_div_a_l1220_122089

theorem range_of_b_div_a 
  (a b : ℝ)
  (h1 : 0 < a) 
  (h2 : a ≤ 2)
  (h3 : b ≥ 1)
  (h4 : b ≤ a^2) : 
  (1 / 2) ≤ b / a ∧ b / a ≤ 2 := 
sorry

end NUMINAMATH_GPT_range_of_b_div_a_l1220_122089


namespace NUMINAMATH_GPT_correct_inequality_l1220_122078

def a : ℚ := -4 / 5
def b : ℚ := -3 / 4

theorem correct_inequality : a < b := 
by {
  -- Proof here
  sorry
}

end NUMINAMATH_GPT_correct_inequality_l1220_122078


namespace NUMINAMATH_GPT_solve_for_y_l1220_122065

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 3
def g (x y : ℝ) : ℝ := 3 * x + y

-- State the theorem to be proven
theorem solve_for_y (x y : ℝ) : 2 * f x - 11 + g x y = f (x - 2) ↔ y = -5 * x + 10 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1220_122065


namespace NUMINAMATH_GPT_intersection_eq_l1220_122027

def M : Set (ℝ × ℝ) := { p | ∃ x, p.2 = x^2 }
def N : Set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = 2 }
def Intersect : Set (ℝ × ℝ) := { p | (M p) ∧ (N p)}

theorem intersection_eq : Intersect = { p : ℝ × ℝ | p = (1,1) ∨ p = (-1, 1) } :=
  sorry

end NUMINAMATH_GPT_intersection_eq_l1220_122027


namespace NUMINAMATH_GPT_possible_digits_C_multiple_of_5_l1220_122091

theorem possible_digits_C_multiple_of_5 :
    ∃ (digits : Finset ℕ), (∀ x ∈ digits, x < 10) ∧ digits.card = 10 ∧ (∀ C ∈ digits, ∃ n : ℕ, n = 1000 + C * 100 + 35 ∧ n % 5 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_possible_digits_C_multiple_of_5_l1220_122091


namespace NUMINAMATH_GPT_tan_double_angle_l1220_122025

theorem tan_double_angle (α : ℝ) 
  (h : Real.tan α = 1 / 2) : Real.tan (2 * α) = 4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1220_122025


namespace NUMINAMATH_GPT_solution_set_absolute_value_inequality_l1220_122003

theorem solution_set_absolute_value_inequality (x : ℝ) :
  (|x-3| + |x-5| ≥ 4) ↔ (x ≤ 2 ∨ x ≥ 6) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_absolute_value_inequality_l1220_122003


namespace NUMINAMATH_GPT_powerFunctionAtPoint_l1220_122067

def powerFunction (n : ℕ) (x : ℕ) : ℕ := x ^ n

theorem powerFunctionAtPoint (n : ℕ) (h : powerFunction n 2 = 8) : powerFunction n 3 = 27 :=
  by {
    sorry
}

end NUMINAMATH_GPT_powerFunctionAtPoint_l1220_122067


namespace NUMINAMATH_GPT_compute_fraction_l1220_122034

theorem compute_fraction (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) : 
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end NUMINAMATH_GPT_compute_fraction_l1220_122034


namespace NUMINAMATH_GPT_father_age_l1220_122001

theorem father_age : 
  ∀ (S F : ℕ), (S - 5 = 11) ∧ (F - S = S) → F = 32 := 
by
  intros S F h
  -- Use the conditions to derive further equations and steps
  sorry

end NUMINAMATH_GPT_father_age_l1220_122001


namespace NUMINAMATH_GPT_determinant_scaled_matrix_l1220_122018

-- Definitions based on the conditions given in the problem.
def determinant2x2 (a b c d : ℝ) : ℝ :=
  a * d - b * c

variable (a b c d : ℝ)
variable (h : determinant2x2 a b c d = 5)

-- The proof statement to be filled, proving the correct answer.
theorem determinant_scaled_matrix :
  determinant2x2 (2 * a) (2 * b) (2 * c) (2 * d) = 20 :=
by
  sorry

end NUMINAMATH_GPT_determinant_scaled_matrix_l1220_122018


namespace NUMINAMATH_GPT_sun_salutations_per_year_l1220_122081

theorem sun_salutations_per_year :
  (∀ S : Nat, S = 5) ∧
  (∀ W : Nat, W = 5) ∧
  (∀ Y : Nat, Y = 52) →
  ∃ T : Nat, T = 1300 :=
by 
  sorry

end NUMINAMATH_GPT_sun_salutations_per_year_l1220_122081


namespace NUMINAMATH_GPT_complement_of_A_in_U_is_4_l1220_122099

-- Define the universal set U
def U : Set ℕ := { x | 1 < x ∧ x < 5 }

-- Define the set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def complement_U_of_A : Set ℕ := { x ∈ U | x ∉ A }

-- State the theorem
theorem complement_of_A_in_U_is_4 : complement_U_of_A = {4} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_is_4_l1220_122099


namespace NUMINAMATH_GPT_units_cost_l1220_122036

theorem units_cost (x y z : ℝ) 
  (h1 : 3 * x + 7 * y + z = 3.15)
  (h2 : 4 * x + 10 * y + z = 4.20) : 
  x + y + z = 1.05 :=
by 
  sorry

end NUMINAMATH_GPT_units_cost_l1220_122036


namespace NUMINAMATH_GPT_find_y_of_arithmetic_mean_l1220_122014

theorem find_y_of_arithmetic_mean (y : ℝ) (h: (7 + 12 + 19 + 8 + 10 + y) / 6 = 15) : y = 34 :=
by {
  -- Skipping the proof
  sorry
}

end NUMINAMATH_GPT_find_y_of_arithmetic_mean_l1220_122014


namespace NUMINAMATH_GPT_problem_solution_l1220_122006

theorem problem_solution (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 6 = 6 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x ^ Real.sqrt 3 + y ^ Real.sqrt 3 = 189 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1220_122006


namespace NUMINAMATH_GPT_ellie_loan_difference_l1220_122009

noncomputable def principal : ℝ := 8000
noncomputable def simple_rate : ℝ := 0.10
noncomputable def compound_rate : ℝ := 0.08
noncomputable def time : ℝ := 5
noncomputable def compounding_periods : ℝ := 1

noncomputable def simple_interest_total (P r t : ℝ) : ℝ :=
  P + (P * r * t)

noncomputable def compound_interest_total (P r t n : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem ellie_loan_difference :
  (compound_interest_total principal compound_rate time compounding_periods) -
  (simple_interest_total principal simple_rate time) = -245.36 := 
  by sorry

end NUMINAMATH_GPT_ellie_loan_difference_l1220_122009


namespace NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l1220_122043

open Real

noncomputable def f (x a : ℝ) : ℝ := a * exp x + 2 * exp (-x) + (a - 2) * x

noncomputable def f_prime (x a : ℝ) : ℝ := (a * exp (2 * x) + (a - 2) * exp x - 2) / exp x

theorem monotonicity_of_f (a : ℝ) : 
  (∀ x : ℝ, f_prime x a ≤ 0) ↔ (a ≤ 0) :=
sorry

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, f x a ≥ (a + 2) * cos x) ↔ (2 ≤ a) :=
sorry

end NUMINAMATH_GPT_monotonicity_of_f_range_of_a_l1220_122043
