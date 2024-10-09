import Mathlib

namespace nina_total_spent_l648_64865

open Real

def toy_price : ℝ := 10
def toy_count : ℝ := 3
def toy_discount : ℝ := 0.15

def card_price : ℝ := 5
def card_count : ℝ := 2
def card_discount : ℝ := 0.10

def shirt_price : ℝ := 6
def shirt_count : ℝ := 5
def shirt_discount : ℝ := 0.20

def sales_tax_rate : ℝ := 0.07

noncomputable def discounted_price (price : ℝ) (count : ℝ) (discount : ℝ) : ℝ :=
  count * price * (1 - discount)

noncomputable def total_cost_before_tax : ℝ := 
  discounted_price toy_price toy_count toy_discount +
  discounted_price card_price card_count card_discount +
  discounted_price shirt_price shirt_count shirt_discount

noncomputable def total_cost_after_tax : ℝ :=
  total_cost_before_tax * (1 + sales_tax_rate)

theorem nina_total_spent : total_cost_after_tax = 62.60 :=
by
  sorry

end nina_total_spent_l648_64865


namespace possible_values_l648_64899

theorem possible_values (m n : ℕ) (h1 : 10 ≥ m) (h2 : m > n) (h3 : n ≥ 4) (h4 : (m - n) ^ 2 = m + n) :
    (m, n) = (10, 6) :=
sorry

end possible_values_l648_64899


namespace simple_interest_rate_l648_64828

theorem simple_interest_rate (P : ℝ) (T : ℝ) (A : ℝ) (R : ℝ) (h : A = 3 * P) (h1 : T = 12) (h2 : A - P = (P * R * T) / 100) :
  R = 16.67 :=
by sorry

end simple_interest_rate_l648_64828


namespace tangent_parallel_l648_64857

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel (P₀ : ℝ × ℝ) :
  (∃ x : ℝ, (P₀ = (x, f x) ∧ deriv f x = 4)) 
  ↔ (P₀ = (1, 0) ∨ P₀ = (-1, -4)) :=
by 
  sorry

end tangent_parallel_l648_64857


namespace find_unique_n_k_l648_64864

theorem find_unique_n_k (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) :
    (n+1)^n = 2 * n^k + 3 * n + 1 ↔ (n = 3 ∧ k = 3) := by
  sorry

end find_unique_n_k_l648_64864


namespace sphere_volume_l648_64882

theorem sphere_volume (h : 4 * π * r^2 = 256 * π) : (4 / 3) * π * r^3 = (2048 / 3) * π :=
by
  sorry

end sphere_volume_l648_64882


namespace determine_a_l648_64852

theorem determine_a (a : ℝ) : (∃ b : ℝ, (3 * (x : ℝ))^2 - 2 * 3 * b * x + b^2 = 9 * x^2 - 27 * x + a) → a = 20.25 :=
by
  sorry

end determine_a_l648_64852


namespace minimum_correct_answers_l648_64812

/-
There are a total of 20 questions. Answering correctly scores 10 points, while answering incorrectly or not answering deducts 5 points. 
To pass, one must score no less than 80 points. Xiao Ming passed the selection. Prove that the minimum number of questions Xiao Ming 
must have answered correctly is no less than 12.
-/

theorem minimum_correct_answers (total_questions correct_points incorrect_points pass_score : ℕ)
  (h1 : total_questions = 20)
  (h2 : correct_points = 10)
  (h3 : incorrect_points = 5)
  (h4 : pass_score = 80)
  (h_passed : ∃ x : ℕ, x ≤ total_questions ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score) :
  ∃ x : ℕ, x ≥ 12 ∧ (correct_points * x - incorrect_points * (total_questions - x)) ≥ pass_score := 
sorry

end minimum_correct_answers_l648_64812


namespace population_size_in_15th_year_l648_64872

theorem population_size_in_15th_year
  (a : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, y x = a * Real.logb 2 (x + 1))
  (h2 : y 1 = 100) :
  y 15 = 400 :=
by
  sorry

end population_size_in_15th_year_l648_64872


namespace base_5_to_base_10_l648_64826

theorem base_5_to_base_10 : 
  let n : ℕ := 1 * 5^3 + 2 * 5^2 + 3 * 5^1 + 4 * 5^0
  n = 194 :=
by 
  sorry

end base_5_to_base_10_l648_64826


namespace include_both_male_and_female_l648_64835

noncomputable def probability_includes_both_genders (total_students male_students female_students selected_students : ℕ) : ℚ :=
  let total_ways := Nat.choose total_students selected_students
  let all_female_ways := Nat.choose female_students selected_students
  (total_ways - all_female_ways) / total_ways

theorem include_both_male_and_female :
  probability_includes_both_genders 6 2 4 4 = 14 / 15 := 
by
  sorry

end include_both_male_and_female_l648_64835


namespace paint_brush_ratio_l648_64854

theorem paint_brush_ratio 
  (s w : ℝ) 
  (h1 : s > 0) 
  (h2 : w > 0) 
  (h3 : (1 / 2) * w ^ 2 + (1 / 2) * (s - w) ^ 2 = (s ^ 2) / 3) 
  : s / w = 3 + Real.sqrt 3 :=
sorry

end paint_brush_ratio_l648_64854


namespace james_weekly_expenses_l648_64830

noncomputable def utility_cost (rent: ℝ):  ℝ := 0.2 * rent
noncomputable def weekly_hours_open (hours_per_day: ℕ) (days_per_week: ℕ): ℕ := hours_per_day * days_per_week
noncomputable def employee_weekly_wages (wage_per_hour: ℝ) (weekly_hours: ℕ): ℝ := wage_per_hour * weekly_hours
noncomputable def total_employee_wages (employees: ℕ) (weekly_wages: ℝ): ℝ := employees * weekly_wages
noncomputable def total_weekly_expenses (rent: ℝ) (utilities: ℝ) (employee_wages: ℝ): ℝ := rent + utilities + employee_wages

theorem james_weekly_expenses : 
  let rent := 1200
  let utility_percentage := 0.2
  let hours_per_day := 16
  let days_per_week := 5
  let employees := 2
  let wage_per_hour := 12.5
  let weekly_hours := weekly_hours_open hours_per_day days_per_week
  let utilities := utility_cost rent
  let employee_wages_per_week := employee_weekly_wages wage_per_hour weekly_hours
  let total_employee_wages_per_week := total_employee_wages employees employee_wages_per_week
  total_weekly_expenses rent utilities total_employee_wages_per_week = 3440 := 
by
  sorry

end james_weekly_expenses_l648_64830


namespace fermat_little_theorem_variant_l648_64839

theorem fermat_little_theorem_variant (p : ℕ) (m : ℤ) [hp : Fact (Nat.Prime p)] : 
  (m ^ p - m) % p = 0 :=
sorry

end fermat_little_theorem_variant_l648_64839


namespace ratio_comparison_l648_64895

theorem ratio_comparison (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) (h_m_lt_n : m < n) :
  (m + 3) / (n + 3) > m / n :=
sorry

end ratio_comparison_l648_64895


namespace value_of_a_is_minus_one_l648_64883

-- Define the imaginary unit i
def imaginary_unit_i : Complex := Complex.I

-- Define the complex number condition
def complex_number_condition (a : ℝ) : Prop :=
  let z := (a + imaginary_unit_i) / (1 + imaginary_unit_i)
  (Complex.re z) = 0 ∧ (Complex.im z) ≠ 0

-- Prove that the value of the real number a is -1 given the condition
theorem value_of_a_is_minus_one (a : ℝ) (h : complex_number_condition a) : a = -1 :=
sorry

end value_of_a_is_minus_one_l648_64883


namespace seq_positive_integers_seq_not_divisible_by_2109_l648_64844

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 3 ∧ a 2 = 6 ∧ ∀ n : ℕ, a (n + 2) = (a (n + 1) ^ 2 + 9) / a n

theorem seq_positive_integers (a : ℕ → ℤ) (h : seq a) : ∀ n : ℕ, 0 < a (n + 1) :=
sorry

theorem seq_not_divisible_by_2109 (a : ℕ → ℤ) (h : seq a) : ¬ ∃ m : ℕ, 2109 ∣ a (m + 1) :=
sorry

end seq_positive_integers_seq_not_divisible_by_2109_l648_64844


namespace mod_residue_17_l648_64836

theorem mod_residue_17 : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  -- We first compute the modulo 17 residue of each term given in the problem:
  -- 513 == 0 % 17
  -- 68 == 0 % 17
  -- 289 == 0 % 17
  -- 34 == 0 % 17
  -- -10 == 7 % 17
  sorry

end mod_residue_17_l648_64836


namespace combined_work_time_l648_64870

theorem combined_work_time (W : ℝ) (A B C : ℝ) (ha : A = W / 12) (hb : B = W / 18) (hc : C = W / 9) : 
  1 / (A + B + C) = 4 := 
by sorry

end combined_work_time_l648_64870


namespace fraction_sum_l648_64810

theorem fraction_sum (n : ℕ) (a : ℚ) (sum_fraction : a = 1/12) (number_of_fractions : n = 450) : 
  ∀ (f : ℚ), (n * f = a) → (f = 1/5400) :=
by
  intros f H
  sorry

end fraction_sum_l648_64810


namespace taxi_speed_is_60_l648_64875

theorem taxi_speed_is_60 (v_b v_t : ℝ) (h1 : v_b = v_t - 30) (h2 : 3 * v_t = 6 * v_b) : v_t = 60 := 
by 
  sorry

end taxi_speed_is_60_l648_64875


namespace relationship_of_f_values_l648_64827

noncomputable def f : ℝ → ℝ := sorry  -- placeholder for the actual function 

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f (-x + 2)

def is_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop := a < b → f a < f b

theorem relationship_of_f_values (h1 : is_increasing f 0 2) (h2 : is_even f) :
  f (5/2) > f 1 ∧ f 1 > f (7/2) :=
sorry -- proof goes here

end relationship_of_f_values_l648_64827


namespace charlie_share_l648_64801

theorem charlie_share (A B C : ℕ) 
  (h1 : (A - 10) * 18 = (B - 20) * 11)
  (h2 : (A - 10) * 24 = (C - 15) * 11)
  (h3 : A + B + C = 1105) : 
  C = 495 := 
by
  sorry

end charlie_share_l648_64801


namespace geometric_sequence_value_l648_64838

variable {a_n : ℕ → ℝ}

-- Condition: {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Given: a_1 a_2 a_3 = -8
variable (a1 a2 a3 : ℝ) (h_seq : is_geometric_sequence a_n)
variable (h_cond : a1 * a2 * a3 = -8)

-- Prove: a2 = -2
theorem geometric_sequence_value : a2 = -2 :=
by
  -- Proof will be provided later
  sorry

end geometric_sequence_value_l648_64838


namespace reeya_third_subject_score_l648_64833

theorem reeya_third_subject_score (s1 s2 s3 s4 : ℝ) (average : ℝ) (num_subjects : ℝ) (total_score : ℝ) :
    s1 = 65 → s2 = 67 → s4 = 95 → average = 76.6 → num_subjects = 4 → total_score = 306.4 →
    (s1 + s2 + s3 + s4) / num_subjects = average → s3 = 79.4 :=
by
  intros h1 h2 h4 h_average h_num_subjects h_total_score h_avg_eq
  -- Proof steps can be added here
  sorry

end reeya_third_subject_score_l648_64833


namespace asymptote_of_hyperbola_l648_64862

theorem asymptote_of_hyperbola : 
  ∀ x y : ℝ, (y^2 / 4 - x^2 = 1) → (y = 2 * x) ∨ (y = -2 * x) := 
by
  sorry

end asymptote_of_hyperbola_l648_64862


namespace alison_birth_weekday_l648_64847

-- Definitions for the problem conditions
def days_in_week : ℕ := 7

-- John's birth day
def john_birth_weekday : ℕ := 3  -- Assuming Monday=0, Tuesday=1, ..., Wednesday=3, ...

-- Number of days Alison was born later
def days_later : ℕ := 72

-- Proof that the resultant day is Friday
theorem alison_birth_weekday : (john_birth_weekday + days_later) % days_in_week = 5 :=
by
  sorry

end alison_birth_weekday_l648_64847


namespace one_meter_eq_jumps_l648_64879

theorem one_meter_eq_jumps 
  (x y a b p q s t : ℝ) 
  (h1 : x * hops = y * skips)
  (h2 : a * jumps = b * hops)
  (h3 : p * skips = q * leaps)
  (h4 : s * leaps = t * meters) :
  1 * meters = (sp * x * a / (tq * y * b)) * jumps :=
sorry

end one_meter_eq_jumps_l648_64879


namespace integer_solutions_count_l648_64805

theorem integer_solutions_count :
  let cond1 (x : ℤ) := -4 * x ≥ 2 * x + 9
  let cond2 (x : ℤ) := -3 * x ≤ 15
  let cond3 (x : ℤ) := -5 * x ≥ x + 22
  ∃ s : Finset ℤ, 
    (∀ x ∈ s, cond1 x ∧ cond2 x ∧ cond3 x) ∧
    (∀ x, cond1 x ∧ cond2 x ∧ cond3 x → x ∈ s) ∧
    s.card = 2 :=
sorry

end integer_solutions_count_l648_64805


namespace compare_star_values_l648_64808

def star (A B : ℤ) : ℤ := A * B - A / B

theorem compare_star_values : star 6 (-3) < star 4 (-4) := by
  sorry

end compare_star_values_l648_64808


namespace marble_244_is_white_l648_64834

noncomputable def color_of_marble (n : ℕ) : String :=
  let cycle := ["white", "white", "white", "white", "gray", "gray", "gray", "gray", "gray", "black", "black", "black"]
  cycle.get! (n % 12)

theorem marble_244_is_white : color_of_marble 244 = "white" :=
by
  sorry

end marble_244_is_white_l648_64834


namespace initial_outlay_is_10000_l648_64800

theorem initial_outlay_is_10000 
  (I : ℝ)
  (manufacturing_cost_per_set : ℝ := 20)
  (selling_price_per_set : ℝ := 50)
  (num_sets : ℝ := 500)
  (profit : ℝ := 5000) :
  profit = (selling_price_per_set * num_sets) - (I + manufacturing_cost_per_set * num_sets) → I = 10000 :=
by
  intro h
  sorry

end initial_outlay_is_10000_l648_64800


namespace unique_linear_eq_sol_l648_64822

theorem unique_linear_eq_sol (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ (a b c : ℤ), (∀ x y : ℕ, (a * x + b * y = c ↔ x = m ∧ y = n)) :=
by
  sorry

end unique_linear_eq_sol_l648_64822


namespace bowlfuls_per_box_l648_64821

def clusters_per_spoonful : ℕ := 4
def spoonfuls_per_bowl : ℕ := 25
def clusters_per_box : ℕ := 500

theorem bowlfuls_per_box : clusters_per_box / (clusters_per_spoonful * spoonfuls_per_bowl) = 5 :=
by
  sorry

end bowlfuls_per_box_l648_64821


namespace geometric_progression_fourth_term_eq_one_l648_64829

theorem geometric_progression_fourth_term_eq_one :
  let a₁ := (2:ℝ)^(1/4)
  let a₂ := (2:ℝ)^(1/6)
  let a₃ := (2:ℝ)^(1/12)
  let r := a₂ / a₁
  let a₄ := a₃ * r
  a₄ = 1 := by
  sorry

end geometric_progression_fourth_term_eq_one_l648_64829


namespace triangle_angle_zero_degrees_l648_64853

theorem triangle_angle_zero_degrees {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  ∃ (C : ℝ), C = 0 ∧ c = 0 :=
sorry

end triangle_angle_zero_degrees_l648_64853


namespace employee_b_payment_l648_64897

theorem employee_b_payment (total_payment : ℝ) (A_ratio : ℝ) (payment_B : ℝ) : 
  total_payment = 550 ∧ A_ratio = 1.2 ∧ total_payment = payment_B + A_ratio * payment_B → payment_B = 250 := 
by
  sorry

end employee_b_payment_l648_64897


namespace f_eq_91_for_all_n_leq_100_l648_64860

noncomputable def f : ℤ → ℝ := sorry

theorem f_eq_91_for_all_n_leq_100 (n : ℤ) (h : n ≤ 100) : f n = 91 := sorry

end f_eq_91_for_all_n_leq_100_l648_64860


namespace length_of_rectangular_sheet_l648_64806

/-- The length of each rectangular sheet is 10 cm given that:
    1. Two identical rectangular sheets each have an area of 48 square centimeters,
    2. The covered area when overlapping the sheets is 72 square centimeters,
    3. The diagonal BD of the overlapping quadrilateral ABCD is 6 centimeters. -/
theorem length_of_rectangular_sheet :
  ∀ (length width : ℝ),
    width * length = 48 ∧
    2 * 48 - 72 = width * 6 ∧
    width * 6 = 24 →
    length = 10 :=
sorry

end length_of_rectangular_sheet_l648_64806


namespace contractor_absent_days_proof_l648_64814

def contractor_absent_days (x y : ℝ) : Prop :=
  x + y = 30 ∧ 25 * x - 7.5 * y = 425

theorem contractor_absent_days_proof : ∃ (y : ℝ), contractor_absent_days x y ∧ y = 10 :=
by
  sorry

end contractor_absent_days_proof_l648_64814


namespace cos_330_is_sqrt3_over_2_l648_64848

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l648_64848


namespace solution_pairs_count_l648_64884

theorem solution_pairs_count : 
  ∃ (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s → 5 * p.1 + 7 * p.2 = 708) ∧ s.card = 20 :=
sorry

end solution_pairs_count_l648_64884


namespace no_person_has_fewer_than_6_cards_l648_64824

-- Definition of the problem and conditions
def cards := 60
def people := 10
def cards_per_person := cards / people

-- Lean statement of the proof problem
theorem no_person_has_fewer_than_6_cards
  (cards_dealt : cards = 60)
  (people_count : people = 10)
  (even_distribution : cards % people = 0) :
  ∀ person, person < people → cards_per_person = 6 ∧ person < people → person = 0 := 
by 
  sorry

end no_person_has_fewer_than_6_cards_l648_64824


namespace no_solution_exists_l648_64815

theorem no_solution_exists : 
  ¬(∃ x y : ℝ, 2 * x - 3 * y = 7 ∧ 4 * x - 6 * y = 20) :=
by
  sorry

end no_solution_exists_l648_64815


namespace intervals_of_monotonicity_range_of_a_l648_64874

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * log x

theorem intervals_of_monotonicity (h : ∀ x, 0 < x → x ≠ e → f (-2) x = -2 * x + x * log x) :
  ((∀ x, 0 < x ∧ x < exp 1 → deriv (f (-2)) x < 0) ∧ (∀ x, x > exp 1 → deriv (f (-2)) x > 0)) :=
sorry

theorem range_of_a (h : ∀ x, e ≤ x → deriv (f a) x ≥ 0) : a ≥ -2 :=
sorry

end intervals_of_monotonicity_range_of_a_l648_64874


namespace quadratic_function_value_2_l648_64802

variables (a b : ℝ)
def f (x : ℝ) : ℝ := x^2 + a * x + b

theorem quadratic_function_value_2 :
  f a b 2 = 3 :=
by
  -- Definitions and assumptions to be used
  sorry

end quadratic_function_value_2_l648_64802


namespace point_reflection_y_l648_64816

def coordinates_with_respect_to_y_axis (x y : ℝ) : ℝ × ℝ :=
  (-x, y)

theorem point_reflection_y (x y : ℝ) (h : (x, y) = (-2, 3)) : coordinates_with_respect_to_y_axis x y = (2, 3) := by
  sorry

end point_reflection_y_l648_64816


namespace tomato_count_after_harvest_l648_64873

theorem tomato_count_after_harvest :
  let plant_A_initial := 150
  let plant_B_initial := 200
  let plant_C_initial := 250
  -- Day 1
  let plant_A_after_day1 := plant_A_initial - (plant_A_initial * 3 / 10)
  let plant_B_after_day1 := plant_B_initial - (plant_B_initial * 1 / 4)
  let plant_C_after_day1 := plant_C_initial - (plant_C_initial * 4 / 25)
  -- Day 7
  let plant_A_after_day7 := plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)
  let plant_B_after_day7 := plant_B_after_day1 - (plant_B_after_day1 * 1 / 5)
  let plant_C_after_day7 := plant_C_after_day1 - ((plant_C_initial * 4 / 25) * 2)
  -- Day 14
  let plant_A_after_day14 := plant_A_after_day7 - ((plant_A_after_day1 - ((plant_A_initial * 3 / 10) + 5)) * 3)
  let plant_B_after_day14 := plant_B_after_day7 - ((plant_B_after_day1 * 1 / 5) + 15)
  let plant_C_after_day14 := plant_C_after_day7 - (plant_C_after_day7 * 1 / 5)
  (plant_A_after_day14 = 0) ∧ (plant_B_after_day14 = 75) ∧ (plant_C_after_day14 = 104) :=
by
  sorry

end tomato_count_after_harvest_l648_64873


namespace find_list_price_l648_64858

theorem find_list_price (P : ℝ) (h1 : 0.873 * P = 61.11) : P = 61.11 / 0.873 :=
by
  sorry

end find_list_price_l648_64858


namespace no_odd_multiples_between_1500_and_3000_l648_64890

theorem no_odd_multiples_between_1500_and_3000 :
  ∀ n : ℤ, 1500 ≤ n → n ≤ 3000 → (18 ∣ n) → (24 ∣ n) → (36 ∣ n) → ¬(n % 2 = 1) :=
by
  -- The proof steps would go here, but we skip them according to the instructions.
  sorry

end no_odd_multiples_between_1500_and_3000_l648_64890


namespace restaurant_tip_difference_l648_64856

theorem restaurant_tip_difference
  (a b : ℝ)
  (h1 : 0.15 * a = 3)
  (h2 : 0.25 * b = 3)
  : a - b = 8 := 
sorry

end restaurant_tip_difference_l648_64856


namespace scientific_notation_of_1300000_l648_64846

theorem scientific_notation_of_1300000 : 1300000 = 1.3 * 10^6 :=
by
  sorry

end scientific_notation_of_1300000_l648_64846


namespace kaleb_games_per_box_l648_64849

theorem kaleb_games_per_box (initial_games sold_games boxes remaining_games games_per_box : ℕ)
  (h1 : initial_games = 76)
  (h2 : sold_games = 46)
  (h3 : boxes = 6)
  (h4 : remaining_games = initial_games - sold_games)
  (h5 : games_per_box = remaining_games / boxes) :
  games_per_box = 5 :=
sorry

end kaleb_games_per_box_l648_64849


namespace equal_number_of_coins_l648_64825

theorem equal_number_of_coins (x : ℕ) (hx : 1 * x + 5 * x + 10 * x + 25 * x + 100 * x = 305) : x = 2 :=
sorry

end equal_number_of_coins_l648_64825


namespace isosceles_triangle_length_l648_64878

theorem isosceles_triangle_length (BC : ℕ) (area : ℕ) (h : ℕ)
  (isosceles : AB = AC)
  (BC_val : BC = 16)
  (area_val : area = 120)
  (height_val : h = (2 * area) / BC)
  (AB_square : ∀ BD AD : ℕ, BD = BC / 2 → AD = h → AB^2 = AD^2 + BD^2)
  : AB = 17 :=
by
  sorry

end isosceles_triangle_length_l648_64878


namespace hcf_of_two_numbers_is_18_l648_64842

theorem hcf_of_two_numbers_is_18
  (product : ℕ)
  (lcm : ℕ)
  (hcf : ℕ) :
  product = 571536 ∧ lcm = 31096 → hcf = 18 := 
by sorry

end hcf_of_two_numbers_is_18_l648_64842


namespace evaluate_expression_l648_64866

theorem evaluate_expression : 3 + (-3)^2 = 12 := by
  sorry

end evaluate_expression_l648_64866


namespace fraction_of_cookies_l648_64867

-- Given conditions
variables 
  (Millie_cookies : ℕ) (Mike_cookies : ℕ) (Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3)

-- Proof statement
theorem fraction_of_cookies (Millie_cookies Mike_cookies Frank_cookies : ℕ)
  (H1 : Mike_cookies = 3 * Millie_cookies)
  (H2 : Millie_cookies = 4)
  (H3 : Frank_cookies = 3) : 
  (Frank_cookies / Mike_cookies : ℚ) = 1 / 4 :=
by
  sorry

end fraction_of_cookies_l648_64867


namespace polynomial_solution_l648_64861

noncomputable def f (n : ℕ) (X Y : ℝ) : ℝ :=
  (X - 2 * Y) * (X + Y) ^ (n - 1)

theorem polynomial_solution (n : ℕ) (f : ℝ → ℝ → ℝ)
  (h1 : ∀ (t x y : ℝ), f (t * x) (t * y) = t^n * f x y)
  (h2 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0)
  (h3 : f 1 0 = 1) :
  ∀ (X Y : ℝ), f X Y = (X - 2 * Y) * (X + Y) ^ (n - 1) :=
by
  sorry

end polynomial_solution_l648_64861


namespace find_fourth_number_l648_64843

theorem find_fourth_number : 
  ∀ (x y : ℝ),
  (28 + x + 42 + y + 104) / 5 = 90 ∧ (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 78 :=
by
  intros x y h
  sorry

end find_fourth_number_l648_64843


namespace problem_statement_l648_64850

open Complex

theorem problem_statement (x y : ℂ) (h : (x + y) / (x - y) - (3 * (x - y)) / (x + y) = 2) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 8320 / 4095 := 
by 
  sorry

end problem_statement_l648_64850


namespace perp_line_parallel_plane_perp_line_l648_64820

variable {Line : Type} {Plane : Type}
variable (a b : Line) (α β : Plane)
variable (parallel : Line → Plane → Prop) (perpendicular : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)

-- Conditions
variable (non_coincident_lines : ¬(a = b))
variable (non_coincident_planes : ¬(α = β))
variable (a_perp_α : perpendicular a α)
variable (b_par_α : parallel b α)

-- Prove
theorem perp_line_parallel_plane_perp_line :
  perpendicular a α ∧ parallel b α → parallel_lines a b :=
sorry

end perp_line_parallel_plane_perp_line_l648_64820


namespace smallest_multiple_of_9_and_6_is_18_l648_64859

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l648_64859


namespace multiple_of_persons_l648_64898

variable (Persons Work : ℕ) (Rate : ℚ)

def work_rate (P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D
def multiple_work_rate (m P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D

theorem multiple_of_persons
  (P : ℕ) (W : ℕ)
  (h1 : work_rate P W 12 = W / 12)
  (h2 : multiple_work_rate 1 P (W / 2) 3 = (W / 6)) :
  m = 2 :=
by sorry

end multiple_of_persons_l648_64898


namespace find_x_range_l648_64811

theorem find_x_range : 
  {x : ℝ | (2 / (x + 2) + 4 / (x + 8) ≤ 3 / 4)} = 
  {x : ℝ | (-4 < x ∧ x ≤ -2) ∨ (4 ≤ x)} := by
  sorry

end find_x_range_l648_64811


namespace opposite_sign_pairs_l648_64831

def opposite_sign (a b : ℤ) : Prop := (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)

theorem opposite_sign_pairs :
  ¬opposite_sign (-(-1)) 1 ∧
  ¬opposite_sign ((-1)^2) 1 ∧
  ¬opposite_sign (|(-1)|) 1 ∧
  opposite_sign (-1) 1 :=
by {
  sorry
}

end opposite_sign_pairs_l648_64831


namespace least_possible_n_l648_64817

noncomputable def d (n : ℕ) := 105 * n - 90

theorem least_possible_n :
  ∀ n : ℕ, d n > 0 → (45 - (d n + 90) / n = 150) → n ≥ 2 :=
by
  sorry

end least_possible_n_l648_64817


namespace find_X_l648_64837

variable (X : ℝ)  -- Threshold income level for the lower tax rate
variable (I : ℝ)  -- Income of the citizen
variable (T : ℝ)  -- Total tax amount

-- Conditions
def income : Prop := I = 50000
def tax_amount : Prop := T = 8000
def tax_formula : Prop := T = 0.15 * X + 0.20 * (I - X)

theorem find_X (h1 : income I) (h2 : tax_amount T) (h3 : tax_formula T I X) : X = 40000 :=
by
  sorry

end find_X_l648_64837


namespace initial_deposit_l648_64876

theorem initial_deposit (A r : ℝ) (n t : ℕ) (hA : A = 169.40) 
  (hr : r = 0.20) (hn : n = 2) (ht : t = 1) :
  ∃ P : ℝ, P = 140 ∧ A = P * (1 + r / n)^(n * t) :=
by
  sorry

end initial_deposit_l648_64876


namespace shoe_length_increase_l648_64871

noncomputable def shoeSizeLength (l : ℕ → ℝ) (size : ℕ) : ℝ :=
  if size = 15 then 9.25
  else if size = 17 then 1.3 * l 8
  else l size

theorem shoe_length_increase :
  (forall l : ℕ → ℝ,
    (shoeSizeLength l 15 = 9.25) ∧
    (shoeSizeLength l 17 = 1.3 * (shoeSizeLength l 8)) ∧
    (forall n, shoeSizeLength l (n + 1) = shoeSizeLength l n + 0.25)
  ) :=
  sorry

end shoe_length_increase_l648_64871


namespace unique_base_for_final_digit_one_l648_64893

theorem unique_base_for_final_digit_one :
  ∃! b : ℕ, 2 ≤ b ∧ b ≤ 15 ∧ 648 % b = 1 :=
by {
  sorry
}

end unique_base_for_final_digit_one_l648_64893


namespace stratified_sampling_grade10_l648_64881

theorem stratified_sampling_grade10
  (total_students : ℕ)
  (grade10_students : ℕ)
  (grade11_students : ℕ)
  (grade12_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = 700)
  (h2 : grade10_students = 300)
  (h3 : grade11_students = 200)
  (h4 : grade12_students = 200)
  (h5 : sample_size = 35)
  : (grade10_students * sample_size / total_students) = 15 := 
sorry

end stratified_sampling_grade10_l648_64881


namespace circles_tangent_radii_product_eq_l648_64869

/-- Given two circles that pass through a fixed point \(M(x_1, y_1)\)
    and are tangent to both the x-axis and y-axis, with radii \(r_1\) and \(r_2\),
    prove that \(r_1 r_2 = x_1^2 + y_1^2\). -/
theorem circles_tangent_radii_product_eq (x1 y1 r1 r2 : ℝ)
  (h1 : (∃ (a : ℝ), ∃ (circle1 : ℝ → ℝ → ℝ), ∀ x y, circle1 x y = (x - a)^2 + (y - a)^2 - r1^2)
    ∧ (∃ (b : ℝ), ∃ (circle2 : ℝ → ℝ → ℝ), ∀ x y, circle2 x y = (x - b)^2 + (y - b)^2 - r2^2))
  (hm1 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r1)^2 + (p.snd - r1)^2 = r1^2 })
  (hm2 : (x1, y1) ∈ { p : ℝ × ℝ | (p.fst - r2)^2 + (p.snd - r2)^2 = r2^2 }) :
  r1 * r2 = x1^2 + y1^2 := sorry

end circles_tangent_radii_product_eq_l648_64869


namespace sum_of_arithmetic_series_l648_64832

def a₁ : ℕ := 9
def d : ℕ := 4
def n : ℕ := 50

noncomputable def nth_term (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_arithmetic_series (a₁ d n : ℕ) : ℕ := n / 2 * (a₁ + nth_term a₁ d n)

theorem sum_of_arithmetic_series :
  sum_arithmetic_series a₁ d n = 5350 :=
by
  sorry

end sum_of_arithmetic_series_l648_64832


namespace mr_arevalo_change_l648_64813

-- Definitions for the costs of the food items
def cost_smoky_salmon : ℤ := 40
def cost_black_burger : ℤ := 15
def cost_chicken_katsu : ℤ := 25

-- Definitions for the service charge and tip percentages
def service_charge_percent : ℝ := 0.10
def tip_percent : ℝ := 0.05

-- Definition for the amount Mr. Arevalo pays
def amount_paid : ℤ := 100

-- Calculation for total food cost
def total_food_cost : ℤ := cost_smoky_salmon + cost_black_burger + cost_chicken_katsu

-- Calculation for service charge
def service_charge : ℝ := service_charge_percent * total_food_cost

-- Calculation for tip
def tip : ℝ := tip_percent * total_food_cost

-- Calculation for the final bill amount
def final_bill_amount : ℝ := total_food_cost + service_charge + tip

-- Calculation for the change
def change : ℝ := amount_paid - final_bill_amount

-- Proof statement
theorem mr_arevalo_change : change = 8 := by
  sorry

end mr_arevalo_change_l648_64813


namespace monkey_count_l648_64888

theorem monkey_count (piles_1 piles_2 hands_1 hands_2 bananas_1_per_hand bananas_2_per_hand total_bananas_per_monkey : ℕ) 
  (h1 : piles_1 = 6) 
  (h2 : piles_2 = 4) 
  (h3 : hands_1 = 9) 
  (h4 : hands_2 = 12) 
  (h5 : bananas_1_per_hand = 14) 
  (h6 : bananas_2_per_hand = 9) 
  (h7 : total_bananas_per_monkey = 99) : 
  (piles_1 * hands_1 * bananas_1_per_hand + piles_2 * hands_2 * bananas_2_per_hand) / total_bananas_per_monkey = 12 := 
by 
  sorry

end monkey_count_l648_64888


namespace find_m_n_l648_64851

theorem find_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hmn : m^n = n^(m - n)) : 
  (m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2) :=
sorry

end find_m_n_l648_64851


namespace james_out_of_pocket_cost_l648_64894

theorem james_out_of_pocket_cost (total_cost : ℝ) (coverage : ℝ) (out_of_pocket_cost : ℝ)
  (h1 : total_cost = 300) (h2 : coverage = 0.8) :
  out_of_pocket_cost = 60 :=
by
  sorry

end james_out_of_pocket_cost_l648_64894


namespace factorize_expr_l648_64803

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l648_64803


namespace cube_volume_from_surface_area_l648_64886

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l648_64886


namespace LimingFatherAge_l648_64868

theorem LimingFatherAge
  (age month day : ℕ)
  (age_condition : 18 ≤ age ∧ age ≤ 70)
  (product_condition : age * month * day = 2975)
  (valid_month : 1 ≤ month ∧ month ≤ 12)
  (valid_day : 1 ≤ day ∧ day ≤ 31)
  : age = 35 := sorry

end LimingFatherAge_l648_64868


namespace eval_six_times_f_l648_64807

def f (x : Int) : Int :=
  if x % 2 == 0 then
    x / 2
  else
    5 * x + 1

theorem eval_six_times_f : f (f (f (f (f (f 7))))) = 116 := 
by
  -- Skipping proof body (since it's not required)
  sorry

end eval_six_times_f_l648_64807


namespace petya_mistake_l648_64891

theorem petya_mistake (x : ℝ) (h : x - x / 10 = 19.71) : x = 21.9 := 
  sorry

end petya_mistake_l648_64891


namespace triangle_inequality_not_true_l648_64845

theorem triangle_inequality_not_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : ¬ (b + c > 2 * a) :=
by {
  -- assume (b + c > 2 * a)
  -- we need to reach a contradiction
  sorry
}

end triangle_inequality_not_true_l648_64845


namespace solve_system_of_equations_l648_64880

def solution_set : Set (ℝ × ℝ) := {(0, 0), (-1, 1), (-2 / (3^(1/3)), -2 * (3^(1/3)))}

theorem solve_system_of_equations (x y : ℝ) :
  (x * y^2 - 2 * y + 3 * x^2 = 0 ∧ y^2 + x^2 * y + 2 * x = 0) ↔ (x, y) ∈ solution_set := sorry

end solve_system_of_equations_l648_64880


namespace ratio_of_sum_of_terms_l648_64804

theorem ratio_of_sum_of_terms (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 5 / 9) : S 9 / S 5 = 1 := 
  sorry

end ratio_of_sum_of_terms_l648_64804


namespace max_value_of_ab_l648_64877

theorem max_value_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 5 * a + 3 * b < 90) :
  ab * (90 - 5 * a - 3 * b) ≤ 1800 :=
sorry

end max_value_of_ab_l648_64877


namespace range_of_m_l648_64840

noncomputable def f (x : ℝ) : ℝ :=
if h : x ∈ (Set.Ioc 0 2) then 2^x - 1 else sorry

def g (x m : ℝ) : ℝ :=
x^2 - 2*x + m

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc (-2:ℝ) 2, f (-x) = -f x) ∧
  (∀ x ∈ Set.Ioc (0:ℝ) 2, f x = 2^x - 1) ∧
  (∀ x1 ∈ Set.Icc (-2:ℝ) 2, ∃ x2 ∈ Set.Icc (-2:ℝ) 2, g x2 m = f x1) 
  → -5 ≤ m ∧ m ≤ -2 :=
sorry

end range_of_m_l648_64840


namespace Bettina_card_value_l648_64809

theorem Bettina_card_value (x : ℝ) (h₀ : 0 < x) (h₁ : x < π / 2) (h₂ : Real.tan x ≠ 1) (h₃ : Real.sin x ≠ Real.cos x) :
  ∀ {a b c : ℝ}, (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
                  (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
                  (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
                  a ≠ b → b ≠ c → a ≠ c →
                  (b = Real.cos x) → b = Real.sqrt 3 / 2 := 
  sorry

end Bettina_card_value_l648_64809


namespace company_A_profit_l648_64819

-- Define the conditions
def total_profit (x : ℝ) : ℝ := x
def company_B_share (x : ℝ) : Prop := 0.4 * x = 60000
def company_A_percentage : ℝ := 0.6

-- Define the statement to be proved
theorem company_A_profit (x : ℝ) (h : company_B_share x) : 0.6 * x = 90000 := sorry

end company_A_profit_l648_64819


namespace sufficient_condition_range_a_l648_64863

theorem sufficient_condition_range_a (a : ℝ) :
  (∀ x, (2 * a ≤ x ∧ x ≤ a^2 + 1) → (x^2 - 3 * (a + 1) * x + 6 * a + 2 ≤ 0)) ↔
  (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by
  sorry

end sufficient_condition_range_a_l648_64863


namespace distinct_nonzero_reals_equation_l648_64892

theorem distinct_nonzero_reals_equation {a b c d : ℝ} 
  (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
  (h₄ : a ≠ b) (h₅ : b ≠ c) (h₆ : c ≠ d) (h₇ : d ≠ a) (h₈ : a ≠ c) (h₉ : b ≠ d)
  (h₁₀ : a * c = b * d) 
  (h₁₁ : a / b + b / c + c / d + d / a = 4) :
  (a / c + c / a + b / d + d / b = 4) :=
by
  sorry

end distinct_nonzero_reals_equation_l648_64892


namespace average_weight_b_c_l648_64889

theorem average_weight_b_c (A B C : ℝ) (h1 : A + B + C = 126) (h2 : A + B = 80) (h3 : B = 40) : 
  (B + C) / 2 = 43 := 
by 
  -- Proof would go here, but is left as sorry as per instructions
  sorry

end average_weight_b_c_l648_64889


namespace position_after_2010_transformations_l648_64841

-- Define the initial position of the square
def init_position := "ABCD"

-- Define the transformation function
def transform (position : String) (steps : Nat) : String :=
  match steps % 8 with
  | 0 => "ABCD"
  | 1 => "CABD"
  | 2 => "DACB"
  | 3 => "BCAD"
  | 4 => "ADCB"
  | 5 => "CBDA"
  | 6 => "BADC"
  | 7 => "CDAB"
  | _ => "ABCD"  -- Default case, should never happen

-- The theorem to prove the correct position after 2010 transformations
theorem position_after_2010_transformations : transform init_position 2010 = "CABD" := 
by
  sorry

end position_after_2010_transformations_l648_64841


namespace club_members_l648_64887

theorem club_members (M W : ℕ) (h1 : M + W = 30) (h2 : M + 1/3 * (W : ℝ) = 18) : M = 12 :=
by
  -- proof step
  sorry

end club_members_l648_64887


namespace angle_sum_is_180_l648_64823

theorem angle_sum_is_180 (A B C : ℝ) (h_triangle : (A + B + C) = 180) (h_sum : A + B = 90) : C = 90 :=
by
  -- Proof placeholder
  sorry

end angle_sum_is_180_l648_64823


namespace range_of_m_for_inequality_l648_64855

theorem range_of_m_for_inequality (x y m : ℝ) :
  (∀ x y : ℝ, 3*x^2 + y^2 ≥ m * x * (x + y)) ↔ (-6 ≤ m ∧ m ≤ 2) := sorry

end range_of_m_for_inequality_l648_64855


namespace volume_increased_by_3_l648_64818

theorem volume_increased_by_3 {l w h : ℝ}
  (h1 : l * w * h = 5000)
  (h2 : l * w + w * h + l * h = 925)
  (h3 : l + w + h = 60) :
  (l + 3) * (w + 3) * (h + 3) = 8342 := 
by
  sorry

end volume_increased_by_3_l648_64818


namespace proportion_of_fathers_with_full_time_jobs_l648_64885

theorem proportion_of_fathers_with_full_time_jobs
  (P : ℕ) -- Total number of parents surveyed
  (mothers_proportion : ℝ := 0.4) -- Proportion of mothers in the survey
  (mothers_ftj_proportion : ℝ := 0.9) -- Proportion of mothers with full-time jobs
  (parents_no_ftj_proportion : ℝ := 0.19) -- Proportion of parents without full-time jobs
  (hfathers : ℝ := 0.6) -- Proportion of fathers in the survey
  (hfathers_ftj_proportion : ℝ) -- Proportion of fathers with full-time jobs
  : hfathers_ftj_proportion = 0.75 := 
by 
  sorry

end proportion_of_fathers_with_full_time_jobs_l648_64885


namespace total_time_to_pump_540_gallons_l648_64896

-- Definitions for the conditions
def initial_rate : ℝ := 360  -- gallons per hour
def increased_rate : ℝ := 480 -- gallons per hour
def target_volume : ℝ := 540  -- total gallons
def first_interval : ℝ := 0.5 -- first 30 minutes as fraction of hour

-- Proof problem statement
theorem total_time_to_pump_540_gallons : 
  (first_interval * initial_rate) + ((target_volume - (first_interval * initial_rate)) / increased_rate) * 60 = 75 := by
  sorry

end total_time_to_pump_540_gallons_l648_64896
