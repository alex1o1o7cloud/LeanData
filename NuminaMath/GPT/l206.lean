import Mathlib

namespace positive_divisors_60_l206_206906

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l206_206906


namespace abs_ineq_range_k_l206_206914

theorem abs_ineq_range_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| > k) → k < 4 :=
by
  sorry

end abs_ineq_range_k_l206_206914


namespace polynomial_remainder_distinct_l206_206706

open Nat

theorem polynomial_remainder_distinct (a b c p : ℕ) (hp : Nat.Prime p) (hp_ge5 : p ≥ 5)
  (ha : Nat.gcd a p = 1) (hb : b^2 ≡ 3 * a * c [MOD p]) (hp_mod3 : p ≡ 2 [MOD 3]) :
  ∀ m1 m2 : ℕ, m1 < p ∧ m2 < p → m1 ≠ m2 → (a * m1^3 + b * m1^2 + c * m1) % p ≠ (a * m2^3 + b * m2^2 + c * m2) % p := 
by
  sorry

end polynomial_remainder_distinct_l206_206706


namespace recorded_instances_l206_206751

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l206_206751


namespace calculate_star_value_l206_206163

def custom_operation (a b : ℕ) : ℕ :=
  (a + b)^3

theorem calculate_star_value : custom_operation 3 5 = 512 :=
by
  sorry

end calculate_star_value_l206_206163


namespace wheel_radius_increase_proof_l206_206527

noncomputable def radius_increase (orig_distance odometer_distance : ℝ) (orig_radius : ℝ) : ℝ :=
  let orig_circumference := 2 * Real.pi * orig_radius
  let distance_per_rotation := orig_circumference / 63360
  let num_rotations_orig := orig_distance / distance_per_rotation
  let num_rotations_new := odometer_distance / distance_per_rotation
  let new_distance := orig_distance
  let new_radius := (new_distance / num_rotations_new) * 63360 / (2 * Real.pi)
  new_radius - orig_radius

theorem wheel_radius_increase_proof :
  radius_increase 600 580 16 = 0.42 :=
by 
  -- The proof is skipped.
  sorry

end wheel_radius_increase_proof_l206_206527


namespace value_of_b_prod_l206_206787

-- Conditions
def a (n : ℕ) : ℕ := 2 * n - 1

def b (n : ℕ) : ℕ := 2 ^ (n - 1)

-- The goal is to prove that b_{a_1} * b_{a_3} * b_{a_5} = 4096
theorem value_of_b_prod : b (a 1) * b (a 3) * b (a 5) = 4096 := by
  sorry

end value_of_b_prod_l206_206787


namespace blanket_thickness_after_foldings_l206_206698

theorem blanket_thickness_after_foldings (initial_thickness : ℕ) (folds : ℕ) (h1 : initial_thickness = 3) (h2 : folds = 4) :
  (initial_thickness * 2^folds) = 48 :=
by
  -- start with definitions as per the conditions
  rw [h1, h2]
  -- proof would follow
  sorry

end blanket_thickness_after_foldings_l206_206698


namespace donuts_left_l206_206192

theorem donuts_left (t : ℕ) (c1 : ℕ) (c2 : ℕ) (c3 : ℝ) : t = 50 ∧ c1 = 2 ∧ c2 = 4 ∧ c3 = 0.5 
  → (t - c1 - c2) / 2 = 22 :=
by
  intros
  cases H with h1 h2
  cases h2 with h3 h4
  cases h4 with h5 h6
  rw [h1, h3, h5, h6]
  norm_num
  sorry

end donuts_left_l206_206192


namespace sqrt_16_eq_pm_4_l206_206154

-- Define the statement to be proven
theorem sqrt_16_eq_pm_4 : sqrt 16 = 4 ∨ sqrt 16 = -4 :=
sorry

end sqrt_16_eq_pm_4_l206_206154


namespace perpendicular_lines_l206_206619

theorem perpendicular_lines (a : ℝ) : 
  ∀ x y : ℝ, 3 * y - x + 4 = 0 → 4 * y + a * x + 5 = 0 → a = 12 :=
by
  sorry

end perpendicular_lines_l206_206619


namespace original_saved_amount_l206_206512

theorem original_saved_amount (x : ℤ) (h : (3 * x - 42)^2 = 2241) : x = 30 := 
sorry

end original_saved_amount_l206_206512


namespace mark_total_votes_l206_206130

-- Definitions based on conditions

def voters_area1 : ℕ := 100000
def percentage_won_area1 : ℝ := 0.7
def votes_area1 := (voters_area1 : ℝ) * percentage_won_area1
def votes_area2 := 2 * votes_area1

-- Theorem statement
theorem mark_total_votes :
  (votes_area1 + votes_area2) = 210000 := 
sorry

end mark_total_votes_l206_206130


namespace sqrt_of_sixteen_l206_206151

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l206_206151


namespace number_of_valid_permutations_l206_206897

-- Define the set S and the permutations A
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2022}
def A : ℕ → ℕ := sorry -- Since A is a permutation of S

-- Given condition
def condition (a : ℕ → ℕ) (n m : ℕ) : Prop :=
  (n ∈ S ∧ m ∈ S ∧ gcd n m ∣ (a n + a m)) 

-- Prove that there are exactly 2 permutations of A meeting the condition
theorem number_of_valid_permutations : 
  ∃ (valid_permutations : Finset (ℕ → ℕ)), 
  (∀ a ∈ valid_permutations, ∀ n m, condition a n m) ∧ valid_permutations.card = 2 := 
sorry

end number_of_valid_permutations_l206_206897


namespace alyosha_cube_cut_l206_206035

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l206_206035


namespace find_n_l206_206009

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l206_206009


namespace mechanic_worked_hours_l206_206450

theorem mechanic_worked_hours (total_spent : ℕ) (cost_per_part : ℕ) (labor_cost_per_minute : ℚ) (parts_needed : ℕ) :
  total_spent = 220 → cost_per_part = 20 → labor_cost_per_minute = 0.5 → parts_needed = 2 →
  (total_spent - cost_per_part * parts_needed) / labor_cost_per_minute / 60 = 6 := by
  -- Proof will be inserted here
  sorry

end mechanic_worked_hours_l206_206450


namespace fraction_to_decimal_l206_206651

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 :=
by
  sorry

end fraction_to_decimal_l206_206651


namespace series_sum_equals_seven_ninths_l206_206866

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l206_206866


namespace cast_cost_l206_206481

theorem cast_cost (C : ℝ) 
  (visit_cost : ℝ := 300)
  (insurance_coverage : ℝ := 0.60)
  (out_of_pocket_cost : ℝ := 200) :
  0.40 * (visit_cost + C) = out_of_pocket_cost → 
  C = 200 := by
  sorry

end cast_cost_l206_206481


namespace square_area_from_diagonal_l206_206342

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 64 :=
begin
  use 64,
  sorry
end

end square_area_from_diagonal_l206_206342


namespace find_chemistry_marks_l206_206694

theorem find_chemistry_marks :
  (let E := 96
   let M := 95
   let P := 82
   let B := 95
   let avg := 93
   let n := 5
   let Total := avg * n
   let Chemistry_marks := Total - (E + M + P + B)
   Chemistry_marks = 97) :=
by
  let E := 96
  let M := 95
  let P := 82
  let B := 95
  let avg := 93
  let n := 5
  let Total := avg * n
  have h_total : Total = 465 := by norm_num
  let Chemistry_marks := Total - (E + M + P + B)
  have h_chemistry_marks : Chemistry_marks = 97 := by norm_num
  exact h_chemistry_marks

end find_chemistry_marks_l206_206694


namespace brandon_initial_skittles_l206_206364

theorem brandon_initial_skittles (initial_skittles : ℕ) (loss : ℕ) (final_skittles : ℕ) (h1 : final_skittles = 87) (h2 : loss = 9) (h3 : final_skittles = initial_skittles - loss) : initial_skittles = 96 :=
sorry

end brandon_initial_skittles_l206_206364


namespace original_cube_volume_l206_206334

theorem original_cube_volume (V₂ : ℝ) (s : ℝ) (h₀ : V₂ = 216) (h₁ : (2 * s) ^ 3 = V₂) : s ^ 3 = 27 := by
  sorry

end original_cube_volume_l206_206334


namespace mechanic_worked_hours_l206_206449

theorem mechanic_worked_hours (total_spent : ℕ) (cost_per_part : ℕ) (labor_cost_per_minute : ℚ) (parts_needed : ℕ) :
  total_spent = 220 → cost_per_part = 20 → labor_cost_per_minute = 0.5 → parts_needed = 2 →
  (total_spent - cost_per_part * parts_needed) / labor_cost_per_minute / 60 = 6 := by
  -- Proof will be inserted here
  sorry

end mechanic_worked_hours_l206_206449


namespace quadratic_equation_general_form_l206_206524

theorem quadratic_equation_general_form :
  ∀ x : ℝ, 2 * (x + 2)^2 + (x + 3) * (x - 2) = -11 ↔ 3 * x^2 + 9 * x + 13 = 0 :=
sorry

end quadratic_equation_general_form_l206_206524


namespace sum_of_vertices_l206_206978

theorem sum_of_vertices (n : ℕ) (h1 : 6 * n + 12 * n = 216) : 8 * n = 96 :=
by
  -- Proof is omitted intentionally
  sorry

end sum_of_vertices_l206_206978


namespace alyosha_cube_problem_l206_206027

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l206_206027


namespace investment_rate_l206_206676

theorem investment_rate (total_investment income1_rate income2_rate income_total remaining_investment expected_income : ℝ)
  (h1 : total_investment = 12000)
  (h2 : income1_rate = 0.03)
  (h3 : income2_rate = 0.045)
  (h4 : expected_income = 600)
  (h5 : (5000 * income1_rate + 4000 * income2_rate) = 330)
  (h6 : remaining_investment = total_investment - 5000 - 4000) :
  (remaining_investment * 0.09 = expected_income - (5000 * income1_rate + 4000 * income2_rate)) :=
by
  sorry

end investment_rate_l206_206676


namespace max_electronic_thermometers_l206_206570

theorem max_electronic_thermometers :
  ∀ (x : ℕ), 10 * x + 3 * (53 - x) ≤ 300 → x ≤ 20 :=
by
  sorry

end max_electronic_thermometers_l206_206570


namespace not_possible_20_odd_rows_15_odd_columns_l206_206389

def odd (n : ℕ) : Prop :=
  n % 2 = 1

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem not_possible_20_odd_rows_15_odd_columns
  (table : ℕ → ℕ → Prop) -- table representing the presence of crosses
  (n : ℕ) -- number of rows and columns in the square table
  (h_square_table: ∀ i j, table i j → i < n ∧ j < n)
  (odd_rows : ℕ)
  (odd_columns : ℕ)
  (h_odd_rows : odd_rows = 20)
  (h_odd_columns : odd_columns = 15)
  (h_def_odd_row: ∀ r, (∃ m, m < n ∧ odd (finset.card {c | c < n ∧ table r c})) ↔ r < odd_rows)
  (h_def_odd_column: ∀ c, (∃ m, m < n ∧ odd (finset.card {r | r < n ∧ table r c})) ↔ c < odd_columns)
  : false :=
by
  sorry

end not_possible_20_odd_rows_15_odd_columns_l206_206389


namespace compare_abc_l206_206717

noncomputable def a : ℝ := Real.log 4 / Real.log 3
noncomputable def b : ℝ := (1 / 3 : ℝ) ^ (1 / 3 : ℝ)
noncomputable def c : ℝ := (3 : ℝ) ^ (-1 / 4 : ℝ)

theorem compare_abc : b < c ∧ c < a :=
by
  sorry

end compare_abc_l206_206717


namespace avg_k_value_l206_206236

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l206_206236


namespace percentage_not_sophomores_l206_206574

variable (Total : ℕ) (Juniors Senior : ℕ) (Freshmen Sophomores : ℕ)

-- Conditions
axiom total_students : Total = 800
axiom percent_juniors : (22 / 100) * Total = Juniors
axiom number_seniors : Senior = 160
axiom freshmen_sophomores_relation : Freshmen = Sophomores + 64
axiom total_composition : Freshmen + Sophomores + Juniors + Senior = Total

-- Proof Objective
theorem percentage_not_sophomores :
  (Total - Sophomores) / Total * 100 = 75 :=
by
  -- proof omitted
  sorry

end percentage_not_sophomores_l206_206574


namespace average_age_of_4_students_l206_206615

theorem average_age_of_4_students (avg_age_15 : ℕ) (num_students_15 : ℕ)
    (avg_age_10 : ℕ) (num_students_10 : ℕ) (age_15th_student : ℕ) :
    avg_age_15 = 15 ∧ num_students_15 = 15 ∧ avg_age_10 = 16 ∧ num_students_10 = 10 ∧ age_15th_student = 9 → 
    (56 / 4 = 14) := by
  sorry

end average_age_of_4_students_l206_206615


namespace optionA_is_square_difference_l206_206649

theorem optionA_is_square_difference (x y : ℝ) : 
  (-x + y) * (x + y) = -(x + y) * (x - y) :=
by sorry

end optionA_is_square_difference_l206_206649


namespace sally_sours_total_l206_206774

theorem sally_sours_total (cherry_sours lemon_sours orange_sours total_sours : ℕ) 
    (h1 : cherry_sours = 32)
    (h2 : 5 * cherry_sours = 4 * lemon_sours)
    (h3 : orange_sours = total_sours / 4)
    (h4 : cherry_sours + lemon_sours + orange_sours = total_sours) : 
    total_sours = 96 :=
by
  rw [h1] at h2
  have h5 : lemon_sours = 40 := by linarith
  rw [h1, h5] at h4
  have h6 : orange_sours = total_sours / 4 := by assumption
  rw [h6] at h4
  have h7 : 72 + total_sours / 4 = total_sours := by linarith
  sorry

end sally_sours_total_l206_206774


namespace company_p_percentage_increase_l206_206521

theorem company_p_percentage_increase :
  (460 - 400.00000000000006) / 400.00000000000006 * 100 = 15 := 
by
  sorry

end company_p_percentage_increase_l206_206521


namespace necessary_but_not_sufficient_condition_l206_206252

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

theorem necessary_but_not_sufficient_condition (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f := 
sorry

end necessary_but_not_sufficient_condition_l206_206252


namespace find_alpha_polar_equation_l206_206381

noncomputable def alpha := (3 * Real.pi) / 4

theorem find_alpha (P : ℝ × ℝ) (l : ℝ → ℝ × ℝ) (A B : ℝ × ℝ)
  (hP : P = (2, 1))
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (hA : ∃ t, l t = (A.1, 0) ∧ A.1 > 0)
  (hB : ∃ t, l t = (0, B.2) ∧ B.2 > 0)
  (h_cond : dist P A * dist P B = 4) : alpha = (3 * Real.pi) / 4 :=
sorry

theorem polar_equation (l : ℝ → ℝ × ℝ)
  (hl_def : ∀ t, l t = (2 + t * Real.cos alpha, 1 + t * Real.sin alpha))
  (h_alpha : alpha = (3 * Real.pi) / 4)
  (h_polar : ∀ ρ θ, l ρ = (ρ * Real.cos θ, ρ * Real.sin θ))
  : ∀ ρ θ, ρ * (Real.cos θ + Real.sin θ) = 3 :=
sorry

end find_alpha_polar_equation_l206_206381


namespace find_first_remainder_l206_206879

theorem find_first_remainder (N : ℕ) (R₁ R₂ : ℕ) (h1 : N = 184) (h2 : N % 15 = R₂) (h3 : R₂ = 4) : 
  N % 13 = 2 :=
by
  sorry

end find_first_remainder_l206_206879


namespace square_area_l206_206996

theorem square_area (y1 y2 y3 y4 : ℤ) 
  (h1 : y1 = 0) (h2 : y2 = 3) (h3 : y3 = 0) (h4 : y4 = -3) : 
  ∃ (area : ℤ), area = 36 :=
by
  sorry

end square_area_l206_206996


namespace equal_parallelogram_faces_are_rhombuses_l206_206093

theorem equal_parallelogram_faces_are_rhombuses 
  (a b c : ℝ) 
  (h: a * b = b * c ∧ b * c = a * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  a = b ∧ b = c :=
sorry

end equal_parallelogram_faces_are_rhombuses_l206_206093


namespace bathtub_problem_l206_206498

theorem bathtub_problem (T : ℝ) (h1 : 1 / T - 1 / 12 = 1 / 60) : T = 10 := 
by {
  -- Sorry, skip the proof as requested
  sorry
}

end bathtub_problem_l206_206498


namespace quadratic_inequality_solution_l206_206177

theorem quadratic_inequality_solution (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2 * x + m ≤ 0) ↔ m ≤ 1 :=
sorry

end quadratic_inequality_solution_l206_206177


namespace sequence_exists_l206_206708

variable (n : ℕ)

theorem sequence_exists (k : ℕ) (hkn : k ∈ Set.range (λ x : ℕ, x + 1) n) :
  ∃ (x : ℕ → ℕ), (∀ i, 1 ≤ i → i ≤ n → x (i+1) > x i) ∧ (∀ i, x i ∈ ℕ) :=
sorry

end sequence_exists_l206_206708


namespace find_e_l206_206443

theorem find_e 
  (a b c d e : ℕ) 
  (h1 : a = 16)
  (h2 : b = 2)
  (h3 : c = 3)
  (h4 : d = 12)
  (h5 : 32 / e = 288 / e) 
  : e = 9 := 
by
  sorry

end find_e_l206_206443


namespace smallest_a_l206_206704

theorem smallest_a (a : ℕ) (h₁ : a > 8) (h₂ : ∀ x : ℤ, ¬prime (x^4 + a^2)) : a = 12 := 
sorry

end smallest_a_l206_206704


namespace negation_proposition_l206_206623

theorem negation_proposition (x y : ℝ) :
  (¬ ∃ (x y : ℝ), 2 * x + 3 * y + 3 < 0) ↔ (∀ (x y : ℝ), 2 * x + 3 * y + 3 ≥ 0) :=
by {
  sorry
}

end negation_proposition_l206_206623


namespace total_savings_l206_206942

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end total_savings_l206_206942


namespace second_man_speed_l206_206316

/-- A formal statement of the problem -/
theorem second_man_speed (v : ℝ) 
  (start_same_place : ∀ t : ℝ, t ≥ 0 → 2 * t = (10 - v) * 1) : 
  v = 8 :=
by
  sorry

end second_man_speed_l206_206316


namespace months_in_season_l206_206637

/-- Definitions for conditions in the problem --/
def total_games_per_month : ℝ := 323.0
def total_games_season : ℝ := 5491.0

/-- The statement to be proven: The number of months in the season --/
theorem months_in_season (x : ℝ) (h : x = total_games_season / total_games_per_month) : x = 17.0 := by
  sorry

end months_in_season_l206_206637


namespace guests_equal_cost_l206_206611

-- Rental costs and meal costs
def rental_caesars_palace : ℕ := 800
def deluxe_meal_cost : ℕ := 30
def premium_meal_cost : ℕ := 40
def rental_venus_hall : ℕ := 500
def venus_special_cost : ℕ := 35
def venus_platter_cost : ℕ := 45

-- Meal distribution percentages
def deluxe_meal_percentage : ℚ := 0.60
def premium_meal_percentage : ℚ := 0.40
def venus_special_percentage : ℚ := 0.60
def venus_platter_percentage : ℚ := 0.40

-- Total costs calculation
noncomputable def total_cost_caesars (G : ℕ) : ℚ :=
  rental_caesars_palace + deluxe_meal_cost * deluxe_meal_percentage * G + premium_meal_cost * premium_meal_percentage * G

noncomputable def total_cost_venus (G : ℕ) : ℚ :=
  rental_venus_hall + venus_special_cost * venus_special_percentage * G + venus_platter_cost * venus_platter_percentage * G

-- Statement to show the equivalence of guest count
theorem guests_equal_cost (G : ℕ) : total_cost_caesars G = total_cost_venus G → G = 60 :=
by
  sorry

end guests_equal_cost_l206_206611


namespace carlos_meeting_percentage_l206_206689

-- Definitions for the given conditions
def work_day_minutes : ℕ := 10 * 60
def first_meeting_minutes : ℕ := 80
def second_meeting_minutes : ℕ := 3 * first_meeting_minutes
def break_minutes : ℕ := 15
def total_meeting_and_break_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + break_minutes

-- Statement to prove
theorem carlos_meeting_percentage : 
  (total_meeting_and_break_minutes * 100 / work_day_minutes) = 56 := 
by
  sorry

end carlos_meeting_percentage_l206_206689


namespace find_P_l206_206790

-- Define the variables A, B, C and their type
variables (A B C P : ℤ)

-- The main theorem statement according to the given conditions and question
theorem find_P (h1 : A = C + 1) (h2 : A + B = C + P) : P = 1 + B :=
by
  sorry

end find_P_l206_206790


namespace find_x3_l206_206317

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1)

theorem find_x3
  (x1 x2 : ℝ)
  (h1 : 0 < x1)
  (h2 : x1 < x2)
  (h1_eq : x1 = 1)
  (h2_eq : x2 = Real.exp 3)
  : ∃ x3 : ℝ, x3 = Real.log (2 / 3 + 1 / 3 * Real.exp (Real.exp 3 - 1)) + 1 :=
by
  sorry

end find_x3_l206_206317


namespace geometric_progression_l206_206210

theorem geometric_progression (p : ℝ) 
  (a b c : ℝ)
  (h1 : a = p - 2)
  (h2 : b = 2 * Real.sqrt p)
  (h3 : c = -3 - p)
  (h4 : b ^ 2 = a * c) : 
  p = 1 := 
by 
  sorry

end geometric_progression_l206_206210


namespace hyperbola_asymptotes_l206_206369

theorem hyperbola_asymptotes (x y : ℝ) :
  (x^2 / 16 - y^2 / 9 = 1) → (y = 3/4 * x ∨ y = -3/4 * x) :=
by
  sorry

end hyperbola_asymptotes_l206_206369


namespace marie_tasks_finish_time_l206_206444

noncomputable def total_time (times : List ℕ) : ℕ :=
  times.foldr (· + ·) 0

theorem marie_tasks_finish_time :
  let task_times := [30, 40, 50, 60]
  let start_time := 8 * 60 -- Start time in minutes (8:00 AM)
  let end_time := start_time + total_time task_times
  end_time = 11 * 60 := -- 11:00 AM in minutes
by
  -- Add a placeholder for the proof
  sorry

end marie_tasks_finish_time_l206_206444


namespace actual_cost_before_decrease_l206_206515

theorem actual_cost_before_decrease (x : ℝ) (h : 0.76 * x = 1064) : x = 1400 :=
by
  sorry

end actual_cost_before_decrease_l206_206515


namespace no_prime_divisible_by_91_l206_206413

theorem no_prime_divisible_by_91 : ¬ ∃ p : ℕ, p > 1 ∧ Prime p ∧ 91 ∣ p :=
by
  sorry

end no_prime_divisible_by_91_l206_206413


namespace puppy_total_food_l206_206591

def daily_food_first_two_weeks : ℝ := (1 / 4) * 3
def total_food_first_two_weeks : ℝ := daily_food_first_two_weeks * 14

def daily_food_second_two_weeks : ℝ := (1 / 2) * 2
def total_food_second_two_weeks : ℝ := daily_food_second_two_weeks * 14

def food_today : ℝ := 1 / 2

def total_food_in_4_weeks : ℝ := food_today + total_food_first_two_weeks + total_food_second_two_weeks

theorem puppy_total_food (W: ℝ:= 25) : total_food_in_4_weeks = W :=
by 
    sorry

end puppy_total_food_l206_206591


namespace alyosha_cube_problem_l206_206032

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l206_206032


namespace John_reads_Bible_in_4_weeks_l206_206586

def daily_reading_pages (hours_per_day reading_rate : ℕ) : ℕ :=
  hours_per_day * reading_rate

def weekly_reading_pages (daily_pages days_in_week : ℕ) : ℕ :=
  daily_pages * days_in_week

def weeks_to_finish (total_pages daily_pages : ℕ) : ℕ :=
  total_pages / daily_pages

theorem John_reads_Bible_in_4_weeks
  (hours_per_day : ℕ : 2)
  (reading_rate : ℕ := 50)
  (bible_pages : ℕ := 2800)
  (days_in_week : ℕ := 7) :
  weeks_to_finish bible_pages (weekly_reading_pages (daily_reading_pages hours_per_day reading_rate) days_in_week) = 4 :=
  sorry

end John_reads_Bible_in_4_weeks_l206_206586


namespace mechanic_worked_hours_l206_206452

theorem mechanic_worked_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (part_count : ℕ) 
  (labor_cost_per_minute : ℝ) 
  (parts_total_cost : ℝ) 
  (labor_total_cost : ℝ) 
  (hours_worked : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_count = 2) 
  (h3 : part_cost = 20) 
  (h4 : labor_cost_per_minute = 0.5) 
  (h5 : parts_total_cost = part_count * part_cost) 
  (h6 : labor_total_cost = total_cost - parts_total_cost) 
  (labor_cost_per_hour := labor_cost_per_minute * 60) 
  (h7 : hours_worked = labor_total_cost / labor_cost_per_hour) : 
  hours_worked = 6 := 
sorry

end mechanic_worked_hours_l206_206452


namespace tensor_example_l206_206559
-- Import the necessary library

-- Define the binary operation ⊗
def tensor (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the main theorem
theorem tensor_example : tensor (tensor 8 6) 2 = 9 / 5 := by
  sorry

end tensor_example_l206_206559


namespace andrews_age_l206_206679

-- Define Andrew's age
variable (a g : ℚ)

-- Problem conditions
axiom condition1 : g = 10 * a
axiom condition2 : g - (a + 2) = 57

theorem andrews_age : a = 59 / 9 := 
by
  -- Set the proof steps aside for now
  sorry

end andrews_age_l206_206679


namespace inequality_holds_l206_206459

theorem inequality_holds (x a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) :
  (1 / (x - a)) + (1 / (x - b)) > 1 / (x - c) := 
by sorry

end inequality_holds_l206_206459


namespace greatest_value_of_NPMK_l206_206813

def is_digit (n : ℕ) : Prop := n < 10

theorem greatest_value_of_NPMK : 
  ∃ M K N P : ℕ, is_digit M ∧ is_digit K ∧ 
  M = K + 1 ∧ M = 9 ∧ K = 8 ∧ 
  1000 * N + 100 * P + 10 * M + K = 8010 ∧ 
  (100 * M + 10 * M + K) * M = 8010 := by
  sorry

end greatest_value_of_NPMK_l206_206813


namespace value_of_a2018_l206_206398

noncomputable def a : ℕ → ℝ
| 0       => 2
| (n + 1) => (1 + a n) / (1 - a n)

theorem value_of_a2018 : a 2017 = -3 := sorry

end value_of_a2018_l206_206398


namespace h_at_3_eq_3_l206_206198

-- Define the function h(x) based on the given condition
noncomputable def h (x : ℝ) : ℝ :=
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * 
    (x^32 + 1) * (x^64 + 1) * (x^128 + 1) * (x^256 + 1) * (x^512 + 1) - 1) / 
  (x^(2^10 - 1) - 1)

-- State the required theorem
theorem h_at_3_eq_3 : h 3 = 3 := by
  sorry

end h_at_3_eq_3_l206_206198


namespace min_green_beads_l206_206509

theorem min_green_beads (B R G : ℕ) (h : B + R + G = 80)
  (hB : ∀ i j : ℕ, (i < j ∧ j ≤ B → ∃ k, i < k ∧ k < j ∧ k ≤ R)) 
  (hR : ∀ i j : ℕ, (i < j ∧ j ≤ R → ∃ k, i < k ∧ k < j ∧ k ≤ G)) :
  G >= 27 := 
sorry

end min_green_beads_l206_206509


namespace avg_of_k_with_positive_integer_roots_l206_206228

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l206_206228


namespace pq_sufficient_not_necessary_l206_206982

theorem pq_sufficient_not_necessary (p q : Prop) :
  (¬ (p ∨ q)) → (¬ p ∧ ¬ q) ∧ ¬ ((¬ p ∧ ¬ q) → (¬ (p ∨ q))) :=
sorry

end pq_sufficient_not_necessary_l206_206982


namespace sum_of_ages_l206_206634

theorem sum_of_ages (Petra_age : ℕ) (Mother_age : ℕ)
  (h_petra : Petra_age = 11)
  (h_mother : Mother_age = 36) :
  Petra_age + Mother_age = 47 :=
by
  -- Using the given conditions:
  -- Petra_age = 11
  -- Mother_age = 36
  sorry

end sum_of_ages_l206_206634


namespace toys_produced_on_sunday_l206_206829

-- Given conditions
def factory_production (day: ℕ) : ℕ :=
  2500 + 25 * day

theorem toys_produced_on_sunday : factory_production 6 = 2650 :=
by {
  -- The proof steps are omitted as they are not required.
  sorry
}

end toys_produced_on_sunday_l206_206829


namespace puppy_food_consumption_l206_206592

/-- Mathematically equivalent proof problem:
  Given the following conditions:
  1. days_per_week = 7
  2. initial_feeding_duration_weeks = 2
  3. initial_feeding_daily_portion = 1/4
  4. initial_feeding_frequency_per_day = 3
  5. subsequent_feeding_duration_weeks = 2
  6. subsequent_feeding_daily_portion = 1/2
  7. subsequent_feeding_frequency_per_day = 2
  8. today_feeding_portion = 1/2
  Prove that the total food consumption, including today, over the next 4 weeks is 25 cups.
-/
theorem puppy_food_consumption :
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  total = 25 := by
  let days_per_week := 7
  let initial_feeding_duration_weeks := 2
  let initial_feeding_daily_portion := 1 / 4
  let initial_feeding_frequency_per_day := 3
  let subsequent_feeding_duration_weeks := 2
  let subsequent_feeding_daily_portion := 1 / 2
  let subsequent_feeding_frequency_per_day := 2
  let today_feeding_portion := 1 / 2
  let initial_feeding_days := initial_feeding_duration_weeks * days_per_week
  let subsequent_feeding_days := subsequent_feeding_duration_weeks * days_per_week
  let initial_total := initial_feeding_days * (initial_feeding_daily_portion * initial_feeding_frequency_per_day)
  let subsequent_total := subsequent_feeding_days * (subsequent_feeding_daily_portion * subsequent_feeding_frequency_per_day)
  let total := today_feeding_portion + initial_total + subsequent_total
  show total = 25 from sorry

end puppy_food_consumption_l206_206592


namespace only_n_is_zero_l206_206207

theorem only_n_is_zero (n : ℕ) (h : (n^2 + 1) ∣ n) : n = 0 := 
by sorry

end only_n_is_zero_l206_206207


namespace alyosha_cube_cut_l206_206039

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l206_206039


namespace alyosha_cube_cut_l206_206033

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l206_206033


namespace subset_a_eq_1_l206_206105

theorem subset_a_eq_1 (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = {0, -a} ∧ B = {1, a-2, 2*a-2} ∧ A ⊆ B → a = 1 :=
by
  sorry

end subset_a_eq_1_l206_206105


namespace sours_total_l206_206775

variable (c l o T : ℕ)

axiom cherry_sours : c = 32
axiom ratio_cherry_lemon : 4 * l = 5 * c
axiom orange_sours_ratio : o = 25 * T / 100
axiom total_sours : T = c + l + o

theorem sours_total :
  T = 96 :=
by
  sorry

end sours_total_l206_206775


namespace no_solution_bills_l206_206930

theorem no_solution_bills (x y z : ℕ) (h1 : x + y + z = 10) (h2 : x + 3 * y + 5 * z = 25) : false :=
by
  sorry

end no_solution_bills_l206_206930


namespace avg_k_for_polynomial_roots_l206_206246

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l206_206246


namespace arithmetic_sequence_general_term_and_k_l206_206101

theorem arithmetic_sequence_general_term_and_k (a : ℕ → ℚ) (d : ℚ)
  (h1 : a 4 + a 7 + a 10 = 17)
  (h2 : a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12 + a 13 + a 14 = 77) :
  (∀ n : ℕ, a n = (2 * n + 3) / 3) ∧ (∃ k : ℕ, a k = 13 ∧ k = 18) := 
by
  sorry

end arithmetic_sequence_general_term_and_k_l206_206101


namespace hannah_mugs_problem_l206_206258

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end hannah_mugs_problem_l206_206258


namespace digit_sum_of_product_l206_206146

def digits_after_multiplication (a b : ℕ) : ℕ :=
  let product := a * b
  let units_digit := product % 10
  let tens_digit := (product / 10) % 10
  tens_digit + units_digit

theorem digit_sum_of_product :
  digits_after_multiplication 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909 = 9 :=
by 
  -- proof goes here
sorry

end digit_sum_of_product_l206_206146


namespace invalid_speed_against_stream_l206_206669

theorem invalid_speed_against_stream (rate_still_water speed_with_stream : ℝ) (h1 : rate_still_water = 6) (h2 : speed_with_stream = 20) :
  ∃ (v : ℝ), speed_with_stream = rate_still_water + v ∧ (rate_still_water - v < 0) → false :=
by
  sorry

end invalid_speed_against_stream_l206_206669


namespace no_valid_triangle_exists_l206_206372

-- Variables representing the sides and altitudes of the triangle
variables (a b c h_a h_b h_c : ℕ)

-- Definition of the perimeter condition
def perimeter_condition : Prop := a + b + c = 1995

-- Definition of integer altitudes condition (simplified)
def integer_altitudes_condition : Prop := 
  ∃ (h_a h_b h_c : ℕ), (h_a * 4 * a ^ 2 = 2 * a ^ 2 * b ^ 2 + 2 * a ^ 2 * c ^ 2 + 2 * c ^ 2 * b ^ 2 - a ^ 4 - b ^ 4 - c ^ 4)

-- The main theorem to prove no valid triangle exists
theorem no_valid_triangle_exists : ¬ (∃ (a b c : ℕ), perimeter_condition a b c ∧ integer_altitudes_condition a b c) :=
sorry

end no_valid_triangle_exists_l206_206372


namespace subset_a_eq_1_l206_206113

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l206_206113


namespace jason_age_at_end_of_2004_l206_206840

noncomputable def jason_age_in_1997 (y : ℚ) (g : ℚ) : Prop :=
  y = g / 3 

noncomputable def birth_years_sum (y : ℚ) (g : ℚ) : Prop :=
  (1997 - y) + (1997 - g) = 3852

theorem jason_age_at_end_of_2004
  (y g : ℚ)
  (h1 : jason_age_in_1997 y g)
  (h2 : birth_years_sum y g) :
  y + 7 = 42.5 :=
by
  sorry

end jason_age_at_end_of_2004_l206_206840


namespace at_most_two_sides_equal_to_longest_diagonal_l206_206909

variables {n : ℕ} (P : convex_polygon n)
  (h1 : n > 3)
  (longest_diagonal : diagonal P)
  (equal_length_sides : finset (side P))
  (h2 : ∀ s ∈ equal_length_sides, side.length s = diagonal.length longest_diagonal)

theorem at_most_two_sides_equal_to_longest_diagonal :
  equal_length_sides.card ≤ 2 :=
sorry

end at_most_two_sides_equal_to_longest_diagonal_l206_206909


namespace cubes_with_one_painted_side_l206_206665

theorem cubes_with_one_painted_side (side_length : ℕ) (one_cm_cubes : ℕ) : 
  side_length = 5 → one_cm_cubes = 54 :=
by 
  intro h 
  sorry

end cubes_with_one_painted_side_l206_206665


namespace triangle_side_length_l206_206581

variable {A B C : ℝ}
variable {a b c : ℝ}

theorem triangle_side_length
  (h1 : a = Real.sqrt 3)
  (h2 : Real.sin A = Real.sqrt 3 / 2)
  (h3 : B = π / 6) :
  b = 1 :=
by
  sorry

end triangle_side_length_l206_206581


namespace find_r_minus2_l206_206513

noncomputable def p : ℤ → ℤ := sorry
def r : ℤ → ℤ := sorry

-- Conditions given in the problem
axiom p_minus1 : p (-1) = 2
axiom p_3 : p (3) = 5
axiom p_minus4 : p (-4) = -3

-- Definition of r(x) when p(x) is divided by (x + 1)(x - 3)(x + 4)
axiom r_def : ∀ x, p x = (x + 1) * (x - 3) * (x + 4) * (sorry : ℤ → ℤ) + r x

-- Our goal to prove
theorem find_r_minus2 : r (-2) = 32 / 7 :=
sorry

end find_r_minus2_l206_206513


namespace max_marks_l206_206137

-- Define the conditions
def passing_marks (M : ℕ) : ℕ := 40 * M / 100

def Ravish_got_marks : ℕ := 40
def marks_failed_by : ℕ := 40

-- Lean statement to prove
theorem max_marks (M : ℕ) (h : passing_marks M = Ravish_got_marks + marks_failed_by) : M = 200 :=
by
  sorry

end max_marks_l206_206137


namespace work_done_by_force_l206_206224

noncomputable def displacement (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem work_done_by_force :
  let F := (5, 2)
  let A := (-1, 3)
  let B := (2, 6)
  let AB := displacement A B
  dot_product F AB = 21 := by
  sorry

end work_done_by_force_l206_206224


namespace Billy_Reads_3_Books_l206_206842

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end Billy_Reads_3_Books_l206_206842


namespace monthly_price_reduction_rate_l206_206924

-- Let's define the given conditions
def initial_price_March : ℝ := 23000
def price_in_May : ℝ := 16000

-- Define the monthly average price reduction rate
variable (x : ℝ)

-- Define the statement to be proven
theorem monthly_price_reduction_rate :
  23 * (1 - x) ^ 2 = 16 :=
sorry

end monthly_price_reduction_rate_l206_206924


namespace decrease_in_temperature_l206_206262

theorem decrease_in_temperature (increase_temp : ℤ → Prop) :
  (increase_temp 2 → -3 = -3) :=
by
  intro h
  exact eq.refl (-3)

end decrease_in_temperature_l206_206262


namespace graph_of_equation_represents_three_lines_l206_206319

theorem graph_of_equation_represents_three_lines (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ (a b c : ℝ), (a * x + b * y + c = 0) ∧
    ((a * x + b * y + c = 0) ∧ (a * x + b * y + c ≠ 0)) ∨
    ((a * x + b * y + c = 0) ∨ (a * x + b * y + c ≠ 0)) ∨
    (a * x + b * y + c = 0)) :=
by
  sorry

end graph_of_equation_represents_three_lines_l206_206319


namespace Marla_laps_per_hour_l206_206762

theorem Marla_laps_per_hour (M : ℝ) :
  (0.8 * M = 0.8 * 5 + 4) → M = 10 :=
by
  sorry

end Marla_laps_per_hour_l206_206762


namespace division_simplification_l206_206520

theorem division_simplification :
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18 / 7 :=
by
  sorry

end division_simplification_l206_206520


namespace rhombus_area_l206_206162

theorem rhombus_area
  (side_length : ℝ)
  (h₀ : side_length = 2 * Real.sqrt 3)
  (tri_a_base : ℝ)
  (tri_b_base : ℝ)
  (h₁ : tri_a_base = side_length)
  (h₂ : tri_b_base = side_length) :
  ∃ rhombus_area : ℝ,
    rhombus_area = 8 * Real.sqrt 3 - 12 :=
by
  sorry

end rhombus_area_l206_206162


namespace sum_of_variables_l206_206418

theorem sum_of_variables (x y z : ℝ) (h₁ : x + y = 1) (h₂ : y + z = 1) (h₃ : z + x = 1) : x + y + z = 3 / 2 := 
sorry

end sum_of_variables_l206_206418


namespace sum_infinite_series_l206_206861

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l206_206861


namespace solve_for_x_and_calculate_l206_206419

theorem solve_for_x_and_calculate (x y : ℚ) 
  (h1 : 102 * x - 5 * y = 25) 
  (h2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 :=
by 
  -- These proof steps would solve the problem and validate the theorem
  sorry

end solve_for_x_and_calculate_l206_206419


namespace gain_percent_is_correct_l206_206170

noncomputable def gain_percent (CP SP : ℝ) : ℝ :=
  let gain := SP - CP
  (gain / CP) * 100

theorem gain_percent_is_correct :
  gain_percent 930 1210 = 30.11 :=
by
  sorry

end gain_percent_is_correct_l206_206170


namespace alyosha_cube_cut_l206_206038

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l206_206038


namespace sequence_values_induction_proof_l206_206397

def seq (a : ℕ → ℤ) := a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - 2 * n * a n + 2

theorem sequence_values (a : ℕ → ℤ) (h : seq a) :
  a 2 = 5 ∧ a 3 = 7 ∧ a 4 = 9 :=
sorry

theorem induction_proof (a : ℕ → ℤ) (h : seq a) :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end sequence_values_induction_proof_l206_206397


namespace broadway_show_total_amount_collected_l206_206497

theorem broadway_show_total_amount_collected (num_adults num_children : ℕ) 
  (adult_ticket_price child_ticket_ratio : ℕ) 
  (child_ticket_price : ℕ) 
  (h1 : num_adults = 400) 
  (h2 : num_children = 200) 
  (h3 : adult_ticket_price = 32) 
  (h4 : child_ticket_ratio = 2) 
  (h5 : adult_ticket_price = child_ticket_ratio * child_ticket_price) : 
  num_adults * adult_ticket_price + num_children * child_ticket_price = 16000 := 
  by 
    sorry

end broadway_show_total_amount_collected_l206_206497


namespace prob_of_three_successes_correct_l206_206475

noncomputable def prob_of_three_successes (p : ℝ) : ℝ :=
  (Nat.choose 10 3) * (p^3) * (1-p)^7

theorem prob_of_three_successes_correct (p : ℝ) :
  prob_of_three_successes p = (Nat.choose 10 3 : ℝ) * (p^3) * (1-p)^7 :=
by
  sorry

end prob_of_three_successes_correct_l206_206475


namespace clock_hands_angle_120_l206_206526

-- We are only defining the problem statement and conditions. No need for proof steps or calculations.

def angle_between_clock_hands (hour minute : ℚ) : ℚ :=
  abs ((30 * hour + minute / 2) - (6 * minute))

-- Given conditions
def time_in_range (hour : ℚ) (minute : ℚ) := 7 ≤ hour ∧ hour < 8

-- Problem statement to be proved
theorem clock_hands_angle_120 (hour minute : ℚ) :
  time_in_range hour minute → angle_between_clock_hands hour minute = 120 :=
sorry

end clock_hands_angle_120_l206_206526


namespace no_prime_numbers_divisible_by_91_l206_206416

-- Define the concept of a prime number.
def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the factors of 91.
def factors_of_91 (n : ℕ) : Prop :=
  n = 7 ∨ n = 13

-- State the problem formally: there are no prime numbers divisible by 91.
theorem no_prime_numbers_divisible_by_91 :
  ∀ p : ℕ, is_prime p → ¬ (91 ∣ p) :=
by
  intros p prime_p div91
  sorry

end no_prime_numbers_divisible_by_91_l206_206416


namespace hannah_mugs_problem_l206_206256

theorem hannah_mugs_problem :
  ∀ (total_mugs blue_mugs red_mugs yellow_mugs other_mugs : ℕ),
  total_mugs = 40 →
  yellow_mugs = 12 →
  red_mugs = yellow_mugs / 2 →
  blue_mugs = 3 * red_mugs →
  other_mugs = total_mugs - (blue_mugs + red_mugs + yellow_mugs) →
  other_mugs = 4 :=
by
  intros total_mugs blue_mugs red_mugs yellow_mugs other_mugs
  intros h_total h_yellow h_red h_blue h_other
  have h1: red_mugs = 6, by linarith [h_yellow, h_red]
  have h2: blue_mugs = 18, by linarith [h1, h_blue]
  have h3: other_mugs = 4, by linarith [h_total, h2, h1, h_yellow, h_other]
  exact h3

end hannah_mugs_problem_l206_206256


namespace calculate_f3_l206_206886

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^7 + a * x^5 + b * x - 5

theorem calculate_f3 (a b : ℝ) (h : f (-3) a b = 5) : f 3 a b = -15 := 
by
  sorry

end calculate_f3_l206_206886


namespace find_x_squared_plus_y_squared_l206_206565

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l206_206565


namespace find_m_l206_206496

theorem find_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x < 0) → m = -1 :=
by sorry

end find_m_l206_206496


namespace marilyn_initial_bottle_caps_l206_206447

theorem marilyn_initial_bottle_caps (x : ℕ) (h : x - 36 = 15) : x = 51 :=
sorry

end marilyn_initial_bottle_caps_l206_206447


namespace typeA_cloth_typeB_cloth_typeC_cloth_l206_206337

section ClothPrices

variables (CPA CPB CPC : ℝ)

theorem typeA_cloth :
  (300 * CPA * 0.90 = 9000) → CPA = 33.33 :=
by
  intro hCPA
  sorry

theorem typeB_cloth :
  (250 * CPB * 1.05 = 7000) → CPB = 26.67 :=
by
  intro hCPB
  sorry

theorem typeC_cloth :
  (400 * (CPC + 8) = 12000) → CPC = 22 :=
by
  intro hCPC
  sorry

end ClothPrices

end typeA_cloth_typeB_cloth_typeC_cloth_l206_206337


namespace area_between_circles_of_octagon_l206_206366

-- Define some necessary geometric terms and functions
noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

/-- The main theorem stating the area between the inscribed and circumscribed circles of a regular octagon is π. -/
theorem area_between_circles_of_octagon :
  let side_length := 2
  let θ := Real.pi / 8 -- 22.5 degrees in radians
  let apothem := cot θ
  let circum_radius := csc θ
  let area_between_circles := π * (circum_radius^2 - apothem^2)
  area_between_circles = π :=
by
  sorry

end area_between_circles_of_octagon_l206_206366


namespace product_of_areas_eq_k3_times_square_of_volume_l206_206518

variables (a b c k : ℝ)

-- Defining the areas of bottom, side, and front of the box as provided
def area_bottom := k * a * b
def area_side := k * b * c
def area_front := k * c * a

-- Volume of the box
def volume := a * b * c

-- The lean statement to be proved
theorem product_of_areas_eq_k3_times_square_of_volume :
  (area_bottom a b k) * (area_side b c k) * (area_front c a k) = k^3 * (volume a b c)^2 :=
by
  sorry

end product_of_areas_eq_k3_times_square_of_volume_l206_206518


namespace compare_m_n_l206_206076

theorem compare_m_n (b m n : ℝ) :
  m = -3 * (-2) + b ∧ n = -3 * (3) + b → m > n :=
by
  sorry

end compare_m_n_l206_206076


namespace problem1_problem2_l206_206064

-- Definition for the first problem
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- First Lean 4 statement for 2^n + 3 = x^2
theorem problem1 (n : ℕ) (h : isPerfectSquare (2^n + 3)) : n = 0 :=
sorry

-- Second Lean 4 statement for 2^n + 1 = x^2
theorem problem2 (n : ℕ) (h : isPerfectSquare (2^n + 1)) : n = 3 :=
sorry

end problem1_problem2_l206_206064


namespace trigonometric_identity_l206_206860

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l206_206860


namespace cos_sin_sequence_rational_l206_206721

variable (α : ℝ) (h₁ : ∃ r : ℚ, r = (Real.sin α + Real.cos α))

theorem cos_sin_sequence_rational :
    (∀ n : ℕ, n > 0 → ∃ r : ℚ, r = (Real.cos α)^n + (Real.sin α)^n) :=
by
  sorry

end cos_sin_sequence_rational_l206_206721


namespace find_n_l206_206005

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l206_206005


namespace candy_cost_l206_206501

theorem candy_cost
  (C : ℝ) -- cost per pound of the first candy
  (w1 : ℝ := 30) -- weight of the first candy
  (c2 : ℝ := 5) -- cost per pound of the second candy
  (w2 : ℝ := 60) -- weight of the second candy
  (w_mix : ℝ := 90) -- total weight of the mixture
  (c_mix : ℝ := 6) -- desired cost per pound of the mixture
  (h1 : w1 * C + w2 * c2 = w_mix * c_mix) -- cost equation for the mixture
  : C = 8 :=
by
  sorry

end candy_cost_l206_206501


namespace tan_of_acute_angle_l206_206078

theorem tan_of_acute_angle (A : ℝ) (hA1 : 0 < A ∧ A < π / 2)
  (hA2 : 4 * (Real.sin A)^2 - 4 * Real.sin A * Real.cos A + (Real.cos A)^2 = 0) :
  Real.tan A = 1 / 2 :=
by
  sorry

end tan_of_acute_angle_l206_206078


namespace subset_solution_l206_206111

theorem subset_solution (a : ℝ) (A B : Set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  -- Proof will go here
  sorry

end subset_solution_l206_206111


namespace sum_fraction_series_eq_l206_206867

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l206_206867


namespace problem_l206_206086

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x + 2
noncomputable def f' (a x : ℝ) : ℝ := a * (Real.log x + 1) + 1
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x - x^2 - (a + 2) * x + a

theorem problem (a x : ℝ) (h : 1 ≤ x) (ha : 0 < a) : f' a x < x^2 + (a + 2) * x + 1 :=
by
  sorry

end problem_l206_206086


namespace surface_area_l206_206674

theorem surface_area (r : ℝ) (π : ℝ) (V : ℝ) (S : ℝ) 
  (h1 : V = 48 * π) 
  (h2 : V = (4 / 3) * π * r^3) : 
  S = 4 * π * r^2 :=
  sorry

end surface_area_l206_206674


namespace prop_logic_example_l206_206098

theorem prop_logic_example (p q : Prop) (h : ¬ (¬ p ∨ ¬ q)) : (p ∧ q) ∧ (p ∨ q) :=
by {
  sorry
}

end prop_logic_example_l206_206098


namespace ellipse_equation_l206_206249

theorem ellipse_equation (a b c c1 : ℝ)
  (h_hyperbola_eq : ∀ x y, (y^2 / 4 - x^2 / 12 = 1))
  (h_sum_eccentricities : (c / a) + (c1 / 2) = 13 / 5)
  (h_foci_x_axis : c1 = 4) :
  (a = 5 ∧ b = 4 ∧ c = 3) → 
  ∀ x y, (x^2 / 25 + y^2 / 16 = 1) :=
by
  sorry

end ellipse_equation_l206_206249


namespace spheres_in_base_l206_206454

theorem spheres_in_base (n : ℕ) (T_n : ℕ) (total_spheres : ℕ) :
  (total_spheres = 165) →
  (total_spheres = (1 / 6 : ℚ) * ↑n * ↑(n + 1) * ↑(n + 2)) →
  (T_n = n * (n + 1) / 2) →
  n = 9 →
  T_n = 45 :=
by
  intros _ _ _ _
  sorry

end spheres_in_base_l206_206454


namespace average_k_l206_206244

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l206_206244


namespace num_divisors_sixty_l206_206907

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l206_206907


namespace alyosha_cube_problem_l206_206030

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l206_206030


namespace no_prime_divisible_by_91_l206_206414

theorem no_prime_divisible_by_91 : ¬ ∃ p : ℕ, p > 1 ∧ Prime p ∧ 91 ∣ p :=
by
  sorry

end no_prime_divisible_by_91_l206_206414


namespace langsley_commute_time_l206_206945

theorem langsley_commute_time (first_bus: ℕ) (first_wait: ℕ) (second_bus: ℕ) (second_wait: ℕ) (third_bus: ℕ) (total_time: ℕ)
  (h1: first_bus = 40)
  (h2: first_wait = 10)
  (h3: second_bus = 50)
  (h4: second_wait = 15)
  (h5: third_bus = 95)
  (h6: total_time = first_bus + first_wait + second_bus + second_wait + third_bus) :
  total_time = 210 := 
by 
  sorry

end langsley_commute_time_l206_206945


namespace condition1_condition2_l206_206739

-- Definition for the coordinates of point P based on given m
def P (m : ℝ) : ℝ × ℝ := (3 * m - 6, m + 1)

-- Condition 1: Point P lies on the x-axis
theorem condition1 (m : ℝ) (hx : P m = (3 * m - 6, 0)) : P m = (-9, 0) := 
by {
  -- Show that if y-coordinate is zero, then m + 1 = 0, hence m = -1
  sorry
}

-- Condition 2: Point A is (-1, 2) and AP is parallel to the y-axis
theorem condition2 (m : ℝ) (A : ℝ × ℝ := (-1, 2)) (hy : (3 * m - 6 = -1)) : P m = (-1, 8/3) :=
by {
  -- Show that if the x-coordinates of A and P are equal, then 3m-6 = -1, hence m = 5/3
  sorry
}

end condition1_condition2_l206_206739


namespace Billy_Reads_3_Books_l206_206841

theorem Billy_Reads_3_Books 
    (weekend_days : ℕ) 
    (hours_per_day : ℕ) 
    (reading_percentage : ℕ) 
    (pages_per_hour : ℕ) 
    (pages_per_book : ℕ) : 
    (weekend_days = 2) ∧ 
    (hours_per_day = 8) ∧ 
    (reading_percentage = 25) ∧ 
    (pages_per_hour = 60) ∧ 
    (pages_per_book = 80) → 
    ((weekend_days * hours_per_day * reading_percentage / 100 * pages_per_hour) / pages_per_book = 3) :=
by
  intros
  sorry

end Billy_Reads_3_Books_l206_206841


namespace opposite_of_one_l206_206732

theorem opposite_of_one (a : ℤ) (h : a = -1) : a = -1 := 
by 
  exact h

end opposite_of_one_l206_206732


namespace trigonometric_expression_value_l206_206855

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l206_206855


namespace sum_of_m_and_n_l206_206820

noncomputable def number_of_rectangles (n : ℕ) : ℕ :=
Nat.choose (n + 1) 2 * Nat.choose (n + 1) 2

def number_of_squares (n : ℕ) : ℕ :=
(n * (n + 1) * (2 * n + 1)) / 6

theorem sum_of_m_and_n (n : ℕ) (h1 : number_of_squares 7 = 140) (h2 : number_of_rectangles 7 = 784) :
  let s := number_of_squares n,
      r := number_of_rectangles n,
      m := 5,
      k := 28 
  in r = 784 → s = 140 → (s / r = 5 / 28) → (m + k = 33) := 
by
  sorry

end sum_of_m_and_n_l206_206820


namespace smallest_nat_div_7_and_11_l206_206042

theorem smallest_nat_div_7_and_11 (n : ℕ) (h1 : n > 1) (h2 : n % 7 = 1) (h3 : n % 11 = 1) : n = 78 :=
by
  sorry

end smallest_nat_div_7_and_11_l206_206042


namespace random_events_l206_206189

-- Define what it means for an event to be random
def is_random_event (e : Prop) : Prop := ∃ (h : Prop), e ∨ ¬e

-- Define the events based on the problem statements
def event1 := ∃ (good_cups : ℕ), good_cups = 3
def event2 := ∃ (half_hit_targets : ℕ), half_hit_targets = 50
def event3 := ∃ (correct_digit : ℕ), correct_digit = 1
def event4 := true -- Opposite charges attract each other, which is always true
def event5 := ∃ (first_prize : ℕ), first_prize = 1

-- State the problem as a theorem
theorem random_events :
  is_random_event event1 ∧ is_random_event event2 ∧ is_random_event event3 ∧ is_random_event event5 :=
by
  sorry

end random_events_l206_206189


namespace necessary_but_not_sufficient_condition_l206_206898

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient_condition (a : ℝ) : (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) := 
  by 
    sorry

end necessary_but_not_sufficient_condition_l206_206898


namespace rachel_earnings_one_hour_l206_206603

-- Define Rachel's hourly wage
def rachelWage : ℝ := 12.00

-- Define the number of people Rachel serves in one hour
def peopleServed : ℕ := 20

-- Define the tip amount per person
def tipPerPerson : ℝ := 1.25

-- Calculate the total tips received
def totalTips : ℝ := (peopleServed : ℝ) * tipPerPerson

-- Calculate the total amount Rachel makes in one hour
def totalEarnings : ℝ := rachelWage + totalTips

-- The theorem to state Rachel's total earnings in one hour
theorem rachel_earnings_one_hour : totalEarnings = 37.00 := 
by
  sorry

end rachel_earnings_one_hour_l206_206603


namespace total_cars_made_in_two_days_l206_206352

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l206_206352


namespace intersection_points_l206_206788

theorem intersection_points (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  (∃ x1 x2, 0 ≤ x1 ∧ x1 ≤ 2 * Real.pi ∧ 
   0 ≤ x2 ∧ x2 ≤ 2 * Real.pi ∧ 
   x1 ≠ x2 ∧ 
   1 + Real.sin x1 = 3 / 2 ∧ 
   1 + Real.sin x2 = 3 / 2 ) ∧ 
  (∀ x, (0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 1 + Real.sin x = 3 / 2) → 
   (x = x1 ∨ x = x2)) :=
sorry

end intersection_points_l206_206788


namespace square_root_of_16_is_pm_4_l206_206155

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l206_206155


namespace solve_equations_l206_206395

theorem solve_equations (x : ℝ) :
  (3 * x^2 = 27 → x = 3 ∨ x = -3) ∧
  (2 * x^2 + x = 55 → x = 5 ∨ x = -5.5) ∧
  (2 * x^2 + 18 = 15 * x → x = 6 ∨ x = 1.5) :=
by
  sorry

end solve_equations_l206_206395


namespace all_faces_rhombuses_l206_206096

variable {R : Type} [LinearOrderedCommRing R]

structure Parallelepiped (R : Type) :=
  (a b c : R)

def parallelogram_area {R : Type} [LinearOrderedCommRing R] (x y : R) : R :=
  x * y

def is_rhombus (x y : R) : Prop :=
  x = y

theorem all_faces_rhombuses (P : Parallelepiped R)
  (h1: parallelogram_area P.a P.b = parallelogram_area P.b P.c)
  (h2: parallelogram_area P.b P.c = parallelogram_area P.a P.c)
  (h3: parallelogram_area P.a P.b = parallelogram_area P.a P.c) :
  is_rhombus P.a P.b ∧ is_rhombus P.b P.c ∧ is_rhombus P.a P.c :=
  sorry

end all_faces_rhombuses_l206_206096


namespace variance_scaled_l206_206797

-- Let V represent the variance of the set of data
def original_variance : ℝ := 3
def scale_factor : ℝ := 3

-- Prove that the new variance is 27 
theorem variance_scaled (V : ℝ) (s : ℝ) (hV : V = 3) (hs : s = 3) : s^2 * V = 27 := by
  sorry

end variance_scaled_l206_206797


namespace mutually_exclusive_not_complementary_l206_206217

-- Define the basic events and conditions
structure Pocket :=
(red : ℕ)
(black : ℕ)

-- Define the event type
inductive Event
| atleast_one_black : Event
| both_black : Event
| atleast_one_red : Event
| both_red : Event
| exactly_one_black : Event
| exactly_two_black : Event
| none_black : Event

def is_mutually_exclusive (e1 e2 : Event) : Prop :=
  match e1, e2 with
  | Event.exactly_one_black, Event.exactly_two_black => true
  | Event.exactly_two_black, Event.exactly_one_black => true
  | _, _ => false

def is_complementary (e1 e2 : Event) : Prop :=
  e1 = Event.none_black ∧ e2 = Event.both_red ∨
  e1 = Event.both_red ∧ e2 = Event.none_black

-- Given conditions
def pocket : Pocket := { red := 2, black := 2 }

-- Proof problem setup
theorem mutually_exclusive_not_complementary : 
  is_mutually_exclusive Event.exactly_one_black Event.exactly_two_black ∧
  ¬ is_complementary Event.exactly_one_black Event.exactly_two_black :=
by
  sorry

end mutually_exclusive_not_complementary_l206_206217


namespace probability_relatively_prime_three_elements_l206_206971

theorem probability_relatively_prime_three_elements :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  let subsets := s.powerset.filter (λ t, t.card = 3)
  let coprime_subsets := subsets.filter (λ t, ∀ x ∈ t, ∀ y ∈ t, x ≠ y → (x.gcd y) = 1)
  (coprime_subsets.card : ℚ) / subsets.card = 45 / 56 := 
by
  sorry

end probability_relatively_prime_three_elements_l206_206971


namespace even_function_f_l206_206081

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - x^2 else -(-x)^3 - (-x)^2

theorem even_function_f (x : ℝ) (h : ∀ x ≤ 0, f x = x^3 - x^2) :
  (∀ x, f x = f (-x)) ∧ (∀ x > 0, f x = -x^3 - x^2) :=
by
  sorry

end even_function_f_l206_206081


namespace problem_statement_l206_206938

def f (x : ℤ) : ℤ := x^2 + 3
def g (x : ℤ) : ℤ := 2 * x + 5

theorem problem_statement : f (g 4) - g (f 4) = 129 := by
  sorry

end problem_statement_l206_206938


namespace group_members_count_l206_206172

theorem group_members_count (n: ℕ) (total_paise: ℕ) (condition1: total_paise = 3249) :
  (n * n = total_paise) → n = 57 :=
by
  sorry

end group_members_count_l206_206172


namespace number_of_students_with_type_B_l206_206328

theorem number_of_students_with_type_B
  (total_students : ℕ)
  (students_with_type_A : total_students ≠ 0 ∧ total_students ≠ 0 → 2 * total_students = 90)
  (students_with_type_B : 2 * total_students = 90) :
  2/5 * total_students = 18 :=
by
  sorry

end number_of_students_with_type_B_l206_206328


namespace line_equation_unique_l206_206801

theorem line_equation_unique (m b k : ℝ) (h_intersect_dist : |(k^2 + 6*k + 5) - (m*k + b)| = 7)
  (h_passing_point : 8 = 2*m + b) (hb_nonzero : b ≠ 0) :
  y = 10*x - 12 :=
by
  sorry

end line_equation_unique_l206_206801


namespace rectangle_same_color_l206_206530

theorem rectangle_same_color (colors : Finset ℕ) (h_col : colors.card = 4)
  (coloring : Fin 5 × Fin 41 → colors) :
  ∃ (p1 p2 p3 p4 : Fin 5 × Fin 41),
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p4 ∧ p4 ≠ p1 ∧
    (coloring p1 = coloring p2 ∧ coloring p2 = coloring p3 ∧ coloring p3 = coloring p4) ∧
    ((p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2) ∨
    (p1.1 = p3.1 ∧ p2.1 = p4.1 ∧ p1.2 = p2.2 ∧ p3.2 = p4.2)) :=
by
  sorry

end rectangle_same_color_l206_206530


namespace ball_falls_total_distance_l206_206330

noncomputable def total_distance : ℕ → ℤ → ℤ → ℤ
| 0, a, _ => 0
| (n+1), a, d => a + total_distance n (a + d) d

theorem ball_falls_total_distance :
  total_distance 5 30 (-6) = 90 :=
by
  sorry

end ball_falls_total_distance_l206_206330


namespace pencils_per_associate_professor_l206_206362

theorem pencils_per_associate_professor
    (A B P : ℕ) -- the number of associate professors, assistant professors, and pencils per associate professor respectively
    (h1 : A + B = 6) -- there are a total of 6 people
    (h2 : A * P + B = 7) -- total number of pencils is 7
    (h3 : A + 2 * B = 11) -- total number of charts is 11
    : P = 2 :=
by
  -- Placeholder for the proof
  sorry

end pencils_per_associate_professor_l206_206362


namespace sally_sours_total_l206_206773

theorem sally_sours_total (cherry_sours lemon_sours orange_sours total_sours : ℕ) 
    (h1 : cherry_sours = 32)
    (h2 : 5 * cherry_sours = 4 * lemon_sours)
    (h3 : orange_sours = total_sours / 4)
    (h4 : cherry_sours + lemon_sours + orange_sours = total_sours) : 
    total_sours = 96 :=
by
  rw [h1] at h2
  have h5 : lemon_sours = 40 := by linarith
  rw [h1, h5] at h4
  have h6 : orange_sours = total_sours / 4 := by assumption
  rw [h6] at h4
  have h7 : 72 + total_sours / 4 = total_sours := by linarith
  sorry

end sally_sours_total_l206_206773


namespace least_value_y_l206_206643

theorem least_value_y : ∃ y : ℝ, (3 * y ^ 3 + 3 * y ^ 2 + 5 * y + 1 = 5) ∧ ∀ z : ℝ, (3 * z ^ 3 + 3 * z ^ 2 + 5 * z + 1 = 5) → y ≤ z :=
sorry

end least_value_y_l206_206643


namespace normal_prob_calc_l206_206248

noncomputable def normal_prob :=
  let X := Normal 1 σ^2 in
  P(X > 2) = 0.15 ∧ P(0 ≤ X ∧ X ≤ 1) = 0.35

theorem normal_prob_calc (σ : ℝ) : normal_prob :=
by
  sorry

end normal_prob_calc_l206_206248


namespace find_y_l206_206928

theorem find_y (x y : ℕ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : y = 1 :=
sorry

end find_y_l206_206928


namespace find_n_l206_206006

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l206_206006


namespace ceil_sqrt_sum_eq_24_l206_206061

theorem ceil_sqrt_sum_eq_24:
  1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 →
  5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 →
  15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 →
  Int.ceil (Real.sqrt 3) + Int.ceil (Real.sqrt 27) + Int.ceil (Real.sqrt 243) = 24 :=
by
  intros h1 h2 h3
  have h1_ceil := Real.ceil_sqrt_of_lt_of_gt h1.left h1.right
  have h2_ceil := Real.ceil_sqrt_of_lt_of_gt h2.left h2.right
  have h3_ceil := Real.ceil_sqrt_of_lt_of_gt h3.left h3.right
  simp [h1_ceil, h2_ceil, h3_ceil]
  sorry

end ceil_sqrt_sum_eq_24_l206_206061


namespace polyomino_count_5_l206_206805

-- Definition of distinct polyomino counts for n = 2, 3, and 4.
def polyomino_count (n : ℕ) : ℕ :=
  if n = 2 then 1
  else if n = 3 then 2
  else if n = 4 then 5
  else 0

-- Theorem stating the distinct polyomino count for n = 5
theorem polyomino_count_5 : polyomino_count 5 = 12 :=
by {
  -- Proof steps would go here, but for now we use sorry.
  sorry
}

end polyomino_count_5_l206_206805


namespace inequality_proof_l206_206383

theorem inequality_proof (a b m n p : ℝ) (h1 : a > b) (h2 : m > n) (h3 : p > 0) : n - a * p < m - b * p :=
sorry

end inequality_proof_l206_206383


namespace difference_even_number_sums_l206_206175

open Nat

def sum_of_even_numbers (start end_ : ℕ) : ℕ :=
  let n := (end_ - start) / 2 + 1
  n * (start + end_) / 2

theorem difference_even_number_sums :
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  sum_B - sum_A = 2100 :=
by
  let sum_A := sum_of_even_numbers 10 50
  let sum_B := sum_of_even_numbers 110 150
  show sum_B - sum_A = 2100
  sorry

end difference_even_number_sums_l206_206175


namespace pumps_fill_time_l206_206338

-- Definitions for the rates and the time calculation
def small_pump_rate : ℚ := 1 / 3
def large_pump_rate : ℚ := 4
def third_pump_rate : ℚ := 1 / 2

def total_pump_rate : ℚ := small_pump_rate + large_pump_rate + third_pump_rate

theorem pumps_fill_time :
  1 / total_pump_rate = 6 / 29 :=
by
  -- Definition of the rates has already been given.
  -- Here we specify the calculation for the combined rate and filling time.
  sorry

end pumps_fill_time_l206_206338


namespace roots_condition_implies_m_range_l206_206268

theorem roots_condition_implies_m_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 1 ∧ x₂ > 1 ∧ (x₁^2 + (m-1)*x₁ + m^2 - 2 = 0) ∧ (x₂^2 + (m-1)*x₂ + m^2 - 2 = 0))
  → -2 < m ∧ m < 1 :=
by
  sorry

end roots_condition_implies_m_range_l206_206268


namespace toy_car_production_l206_206346

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l206_206346


namespace basil_has_winning_strategy_l206_206295

-- Definitions based on conditions
def piles : Nat := 11
def stones_per_pile : Nat := 10
def peter_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3
def basil_moves (n : Nat) := n = 1 ∨ n = 2 ∨ n = 3

-- The main theorem to prove Basil has a winning strategy
theorem basil_has_winning_strategy 
  (total_stones : Nat := piles * stones_per_pile) 
  (peter_first : Bool := true) :
  exists winning_strategy_for_basil, 
    ∀ (piles_remaining : Nat) (sum_stones_remaining : Nat),
    sum_stones_remaining = piles_remaining * stones_per_pile ∨
    (1 ≤ piles_remaining ∧ piles_remaining ≤ piles) ∧
    (0 ≤ sum_stones_remaining ∧ sum_stones_remaining ≤ total_stones)
    → winning_strategy_for_basil = true := 
sorry -- The proof is omitted

end basil_has_winning_strategy_l206_206295


namespace asymptotes_of_hyperbola_l206_206961

-- Definition of the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := (x^2) / 16 - (y^2) / 9 = 1

-- Definition of the equations of the asymptotes
def asymptote_eq (x y : ℝ) : Prop := y = (3/4)*x ∨ y = -(3/4)*x

-- Theorem statement
theorem asymptotes_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y → asymptote_eq x y :=
sorry

end asymptotes_of_hyperbola_l206_206961


namespace no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l206_206582

theorem no_natural_n_such_that_6n2_plus_5n_is_power_of_2 :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 6 * n^2 + 5 * n = 2^k :=
by
  sorry

end no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l206_206582


namespace find_x_squared_plus_y_squared_l206_206563

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l206_206563


namespace sum_of_sequences_is_43_l206_206547

theorem sum_of_sequences_is_43
  (A B C D : ℕ)
  (hA_pos : 0 < A)
  (hB_pos : 0 < B)
  (hC_pos : 0 < C)
  (hD_pos : 0 < D)
  (h_arith : A + (C - B) = B)
  (h_geom : C = (4 * B) / 3)
  (hD_def : D = (4 * C) / 3) :
  A + B + C + D = 43 :=
sorry

end sum_of_sequences_is_43_l206_206547


namespace number_of_zeros_l206_206220

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def conditions (f : ℝ → ℝ) (f'' : ℝ → ℝ) :=
  odd_function f ∧ ∀ x : ℝ, x < 0 → (2 * f x + x * f'' x < x * f x)

theorem number_of_zeros (f : ℝ → ℝ) (f'' : ℝ → ℝ) (h : conditions f f'') :
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_l206_206220


namespace Maria_telephone_numbers_l206_206359

def num_distinct_telephone_numbers : ℕ :=
  Nat.choose 7 5

theorem Maria_telephone_numbers :
  num_distinct_telephone_numbers = 21 := by
  sorry

end Maria_telephone_numbers_l206_206359


namespace interval_proof_l206_206056

noncomputable def valid_interval (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → (5 * (x * y^2 + x^2 * y + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y

theorem interval_proof : ∀ x : ℝ, valid_interval x ↔ (0 ≤ x ∧ x < 4) :=
by
  sorry

end interval_proof_l206_206056


namespace school_enrollment_l206_206965

theorem school_enrollment
  (X Y : ℝ)
  (h1 : X + Y = 4000)
  (h2 : 1.07 * X > X)
  (h3 : 1.03 * Y > Y)
  (h4 : 0.07 * X - 0.03 * Y = 40) :
  Y = 2400 :=
by
  -- problem reduction
  sorry

end school_enrollment_l206_206965


namespace find_x_squared_plus_y_squared_l206_206564

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : x - y = 20) (h2 : x * y = 9) :
  x^2 + y^2 = 418 :=
sorry

end find_x_squared_plus_y_squared_l206_206564


namespace recorded_instances_l206_206750

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l206_206750


namespace square_area_from_diagonal_l206_206341

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 64 :=
begin
  use 64,
  sorry
end

end square_area_from_diagonal_l206_206341


namespace breaststroke_speed_correct_l206_206314

-- Defining the given conditions
def total_distance : ℕ := 500
def front_crawl_speed : ℕ := 45
def front_crawl_time : ℕ := 8
def total_time : ℕ := 12

-- Definition of the breaststroke speed given the conditions
def breaststroke_speed : ℕ :=
  let front_crawl_distance := front_crawl_speed * front_crawl_time
  let breaststroke_distance := total_distance - front_crawl_distance
  let breaststroke_time := total_time - front_crawl_time
  breaststroke_distance / breaststroke_time

-- Theorem to prove the breaststroke speed is 35 yards per minute
theorem breaststroke_speed_correct : breaststroke_speed = 35 :=
  sorry

end breaststroke_speed_correct_l206_206314


namespace ellipse_eccentricity_half_l206_206836

-- Definitions and assumptions
variable (a b c e : ℝ)
variable (h₁ : a = 2 * c)
variable (h₂ : b = sqrt 3 * c)
variable (eccentricity_def : e = c / a)

-- Theorem statement
theorem ellipse_eccentricity_half : e = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_half_l206_206836


namespace xiangming_payment_methods_count_l206_206655

def xiangming_payment_methods : Prop :=
  ∃ x y z : ℕ, 
    x + y + z ≤ 10 ∧ 
    x + 2 * y + 5 * z = 18 ∧ 
    ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0))

theorem xiangming_payment_methods_count : 
  xiangming_payment_methods → ∃! n, n = 11 :=
by sorry

end xiangming_payment_methods_count_l206_206655


namespace rate_percent_simple_interest_l206_206817

theorem rate_percent_simple_interest (SI P T R : ℝ) (h₁ : SI = 500) (h₂ : P = 2000) (h₃ : T = 2)
  (h₄ : SI = (P * R * T) / 100) : R = 12.5 :=
by
  -- Placeholder for the proof
  sorry

end rate_percent_simple_interest_l206_206817


namespace functional_eq_linear_l206_206206

theorem functional_eq_linear {f : ℝ → ℝ} (h : ∀ x y : ℝ, f (x ^ 2 - y ^ 2) = (x + y) * (f x - f y)) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
sorry

end functional_eq_linear_l206_206206


namespace cube_cut_problem_l206_206004

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l206_206004


namespace LittleRed_system_of_eqns_l206_206442

theorem LittleRed_system_of_eqns :
  ∃ (x y : ℝ), (2/60) * x + (3/60) * y = 1.5 ∧ x + y = 18 :=
sorry

end LittleRed_system_of_eqns_l206_206442


namespace avg_of_k_with_positive_integer_roots_l206_206227

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l206_206227


namespace range_of_a_l206_206083

theorem range_of_a 
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x < 0, f x = a^x)
  (h2 : ∀ x ≥ 0, f x = (a - 3) * x + 4 * a)
  (h3 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) > 0) :
  0 < a ∧ a ≤ 1 / 4 :=
sorry

end range_of_a_l206_206083


namespace rectangular_to_cylindrical_l206_206523

theorem rectangular_to_cylindrical (x y z : ℝ) (r θ : ℝ) (h1 : x = -3) (h2 : y = 4) (h3 : z = 5) (h4 : r = 5) (h5 : θ = Real.pi - Real.arctan (4 / 3)) :
  (r, θ, z) = (5, Real.pi - Real.arctan (4 / 3), 5) :=
by
  sorry

end rectangular_to_cylindrical_l206_206523


namespace floor_trig_sum_l206_206114

theorem floor_trig_sum :
  Int.floor (Real.sin 1) + Int.floor (Real.cos 2) + Int.floor (Real.tan 3) +
  Int.floor (Real.sin 4) + Int.floor (Real.cos 5) + Int.floor (Real.tan 6) = -4 := by
  sorry

end floor_trig_sum_l206_206114


namespace arithmetic_sequence_a6_l206_206577

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h₀ : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
  (h₁ : ∃ x y : ℝ, x = a 4 ∧ y = a 8 ∧ (x^2 - 4 * x - 1 = 0) ∧ (y^2 - 4 * y - 1 = 0) ∧ (x + y = 4)) :
  a 6 = 2 := 
sorry

end arithmetic_sequence_a6_l206_206577


namespace ratio_of_radii_of_touching_circles_l206_206161

theorem ratio_of_radii_of_touching_circles
  (r R : ℝ) (A B C D : ℝ) (h1 : A + B + C = D)
  (h2 : 3 * A = 7 * B)
  (h3 : 7 * B = 2 * C)
  (h4 : R = D / 2)
  (h5 : B = R - 3 * A)
  (h6 : C = R - 2 * A)
  (h7 : r = 4 * A)
  (h8 : R = 6 * A) :
  R / r = 3 / 2 := by
  sorry

end ratio_of_radii_of_touching_circles_l206_206161


namespace area_of_black_region_l206_206506

theorem area_of_black_region :
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  area_large - total_area_small = 94 :=
by
  let side_large := 12
  let side_small := 5
  let area_large := side_large * side_large
  let area_small := side_small * side_small
  let num_smaller_squares := 2
  let total_area_small := num_smaller_squares * area_small
  sorry

end area_of_black_region_l206_206506


namespace equal_water_and_alcohol_l206_206500

variable (a m : ℝ)

-- Conditions:
-- Cup B initially contains m liters of water.
-- Transfers as specified in the problem.

theorem equal_water_and_alcohol (h : m > 0) :
  (a * (m / (m + a)) = a * (m / (m + a))) :=
by
  sorry

end equal_water_and_alcohol_l206_206500


namespace solution_set_inequality_l206_206880

theorem solution_set_inequality (a x : ℝ) :
  (12 * x^2 - a * x > a^2) ↔
  ((a > 0 ∧ (x < -a / 4 ∨ x > a / 3)) ∨
   (a = 0 ∧ x ≠ 0) ∨
   (a < 0 ∧ (x > -a / 4 ∨ x < a / 3))) :=
sorry


end solution_set_inequality_l206_206880


namespace decagon_diagonals_l206_206731

theorem decagon_diagonals : ∀ n : ℕ, n = 10 → (n * (n - 3) / 2) = 35 :=
by
  intros n hn
  rw [hn]
  norm_num
  sorry

end decagon_diagonals_l206_206731


namespace avg_of_k_with_positive_integer_roots_l206_206229

theorem avg_of_k_with_positive_integer_roots :
  ∀ (k : ℕ), (∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k)) → 
  (∃ ks : List ℕ, (∀ k', k' ∈ ks ↔ ∃ r1 r2 : ℕ, r1 > 0 ∧ r2 > 0 ∧ (r1 * r2 = 24) ∧ (r1 + r2 = k')) ∧ ks.Average = 15) := 
begin
  sorry
end

end avg_of_k_with_positive_integer_roots_l206_206229


namespace probability_different_numbers_and_colors_l206_206881

def total_ways : ℕ := Nat.choose 5 2

def valid_pairs : ℕ := 4

def probability_valid_pairs : ℚ := valid_pairs / total_ways

theorem probability_different_numbers_and_colors :
  probability_valid_pairs = 2 / 5 := by
  sorry

end probability_different_numbers_and_colors_l206_206881


namespace infinitely_many_n_divide_2n_plus_1_l206_206607

theorem infinitely_many_n_divide_2n_plus_1 :
    ∃ (S : Set ℕ), (∀ n ∈ S, n > 0 ∧ n ∣ (2 * n + 1)) ∧ Set.Infinite S :=
by
  sorry

end infinitely_many_n_divide_2n_plus_1_l206_206607


namespace find_a_for_no_x2_term_l206_206266

theorem find_a_for_no_x2_term :
  ∀ a : ℝ, (∀ x : ℝ, (3 * x^2 + 2 * a * x + 1) * (-3 * x) - 4 * x^2 = -9 * x^3 + (-6 * a - 4) * x^2 - 3 * x) →
  (¬ ∃ x : ℝ, (-6 * a - 4) * x^2 ≠ 0) →
  a = -2 / 3 :=
by
  intros a h1 h2
  sorry

end find_a_for_no_x2_term_l206_206266


namespace segment_shadow_ratio_l206_206602

theorem segment_shadow_ratio (a b a' b' : ℝ) (h : a / b = a' / b') : a / a' = b / b' :=
sorry

end segment_shadow_ratio_l206_206602


namespace max_pairs_300_grid_l206_206760

noncomputable def max_pairs (n : ℕ) (k : ℕ) (remaining_squares : ℕ) [Fintype (Fin n × Fin n)] : ℕ :=
  sorry

theorem max_pairs_300_grid :
  max_pairs 300 100 50000 = 49998 :=
by
  -- problem conditions
  let grid_size := 300
  let corner_size := 100
  let remaining_squares := 50000
  let no_checkerboard (squares : Fin grid_size × Fin grid_size → Prop) : Prop :=
    ∀ i j, ¬(squares (i, j) ∧ squares (i + 1, j) ∧ squares (i, j + 1) ∧ squares (i + 1, j + 1))
  -- statement of the bound
  have max_pairs := max_pairs grid_size corner_size remaining_squares
  exact sorry

end max_pairs_300_grid_l206_206760


namespace hannah_mugs_problem_l206_206255

theorem hannah_mugs_problem :
  ∀ (total_mugs blue_mugs red_mugs yellow_mugs other_mugs : ℕ),
  total_mugs = 40 →
  yellow_mugs = 12 →
  red_mugs = yellow_mugs / 2 →
  blue_mugs = 3 * red_mugs →
  other_mugs = total_mugs - (blue_mugs + red_mugs + yellow_mugs) →
  other_mugs = 4 :=
by
  intros total_mugs blue_mugs red_mugs yellow_mugs other_mugs
  intros h_total h_yellow h_red h_blue h_other
  have h1: red_mugs = 6, by linarith [h_yellow, h_red]
  have h2: blue_mugs = 18, by linarith [h1, h_blue]
  have h3: other_mugs = 4, by linarith [h_total, h2, h1, h_yellow, h_other]
  exact h3

end hannah_mugs_problem_l206_206255


namespace blue_length_of_pencil_l206_206832

theorem blue_length_of_pencil (total_length purple_length black_length blue_length : ℝ)
  (h1 : total_length = 6)
  (h2 : purple_length = 3)
  (h3 : black_length = 2)
  (h4 : total_length = purple_length + black_length + blue_length)
  : blue_length = 1 :=
by
  sorry

end blue_length_of_pencil_l206_206832


namespace election_winning_votes_l206_206479

noncomputable def total_votes (x y : ℕ) (p : ℚ) : ℚ := 
  (x + y) / (1 - p)

noncomputable def winning_votes (x y : ℕ) (p : ℚ) : ℚ :=
  p * total_votes x y p

theorem election_winning_votes :
  winning_votes 2136 7636 0.54336448598130836 = 11628 := 
by
  sorry

end election_winning_votes_l206_206479


namespace remaining_apples_l206_206199

def initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem remaining_apples : initial_apples - shared_apples = 13 :=
by
  sorry

end remaining_apples_l206_206199


namespace z12_real_cardinality_l206_206798

open Complex

noncomputable def num_real_z12_of_z48_eq_1 : ℕ :=
  let S : Set ℂ := { z | z ^ 48 = 1 }
  { z | z ^ 12 ∈ ℝ } . card

theorem z12_real_cardinality : num_real_z12_of_z48_eq_1 = 8 := by
  sorry

end z12_real_cardinality_l206_206798


namespace cubic_sum_identity_l206_206781

   theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = -1) :
     a^3 + b^3 + c^3 = 12 :=
   by
     sorry
   
end cubic_sum_identity_l206_206781


namespace part_a_impossible_part_b_possible_l206_206387

-- Statement for part (a)
theorem part_a_impossible (table : ℕ → ℕ → bool) (n : ℕ) 
    (odd_row_count : fin n → bool) (odd_col_count : fin n → bool)
    (odd_count_r : ℕ) (odd_count_c : ℕ) (cross_in_cell : fin n → fin n → bool) :
    (∀ r : fin n, odd_row_count r = odd (fin n) (\sum c : fin n, cross_in_cell r c)) →
    (∀ c : fin n, odd_col_count c = odd (fin n) (\sum r : fin n, cross_in_cell r c)) →
    odd_count_r = 20 → odd_count_c = 15 → False :=
sorry

-- Statement for part (b)
theorem part_b_possible (table : ℕ → ℕ → bool) 
    (n : ℕ) (cross_count : ℕ) (row_count : fin n → ℕ) (col_count : fin n → ℕ)
    (cross_in_cell : fin n → fin n → bool) :
    n = 16 → cross_count = 126 →
    (∀ r : fin n, odd (row_count r)) →
    (∀ c : fin n, odd (col_count c)) →
    (∃ table, (∀ r c, cross_in_cell r c = (table r c)) ∧ (\sum r, row_count r = 126) ∧ (\sum c, col_count c = 126)) :=
sorry

end part_a_impossible_part_b_possible_l206_206387


namespace log_cosine_range_l206_206313

noncomputable def log_base_three (a : ℝ) : ℝ := Real.log a / Real.log 3

theorem log_cosine_range (x : ℝ) (hx : x ∈ Set.Ioo (Real.pi / 2) (7 * Real.pi / 6)) :
    ∃ y, y = log_base_three (1 - 2 * Real.cos x) ∧ y ∈ Set.Ioc 0 1 :=
by
  sorry

end log_cosine_range_l206_206313


namespace find_sister_candy_l206_206285

/-- Define Katie's initial amount of candy -/
def Katie_candy : ℕ := 10

/-- Define the amount of candy eaten the first night -/
def eaten_candy : ℕ := 9

/-- Define the amount of candy left after the first night -/
def remaining_candy : ℕ := 7

/-- Define the number of candies Katie's sister had -/
def sister_candy (S : ℕ) : Prop :=
  Katie_candy + S - eaten_candy = remaining_candy

/-- Theorem stating that Katie's sister had 6 pieces of candy -/
theorem find_sister_candy : ∃ S, sister_candy S ∧ S = 6 :=
by
  sorry

end find_sister_candy_l206_206285


namespace ratio_of_shoppers_l206_206633

theorem ratio_of_shoppers (boxes ordered_of_yams: ℕ) (packages_per_box shoppers total_shoppers: ℕ)
  (h1 : packages_per_box = 25)
  (h2 : ordered_of_yams = 5)
  (h3 : total_shoppers = 375)
  (h4 : shoppers = ordered_of_yams * packages_per_box):
  (shoppers : ℕ) / total_shoppers = 1 / 3 := 
sorry

end ratio_of_shoppers_l206_206633


namespace sum_fraction_series_eq_l206_206868

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l206_206868


namespace minimum_value_inequality_l206_206722

def minimum_value_inequality_problem : Prop :=
∀ (a b : ℝ), (0 < a) → (0 < b) → (a + 3 * b = 1) → (1 / a + 1 / (3 * b)) = 4

theorem minimum_value_inequality : minimum_value_inequality_problem :=
sorry

end minimum_value_inequality_l206_206722


namespace trigonometric_identity_eq_neg_one_l206_206854

theorem trigonometric_identity_eq_neg_one :
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180)) = -1 :=
by
  -- Variables needed for hypotheses
  have h₁ : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₂ : Real.sin (60 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₃ : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h₄ : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  -- Main proof
  sorry

end trigonometric_identity_eq_neg_one_l206_206854


namespace toy_car_production_l206_206345

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l206_206345


namespace correct_div_value_l206_206261

theorem correct_div_value (x : ℝ) (h : 25 * x = 812) : x / 4 = 8.12 :=
by sorry

end correct_div_value_l206_206261


namespace cube_decomposition_l206_206012

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l206_206012


namespace percentage_increase_l206_206308

theorem percentage_increase (N : ℝ) (P : ℝ) (h1 : N + (P / 100) * N - (N - 25 / 100 * N) = 30) (h2 : N = 80) : P = 12.5 :=
by
  sorry

end percentage_increase_l206_206308


namespace smallest_and_largest_group_sizes_l206_206178

theorem smallest_and_largest_group_sizes (S T : Finset ℕ) (hS : S.card + T.card = 20)
  (h_union: (S ∪ T) = (Finset.range 21) \ {0}) (h_inter: S ∩ T = ∅)
  (sum_S : S.sum id = 210 - T.sum id) (prod_T : T.prod id = 210 - S.sum id) :
  T.card = 3 ∨ T.card = 5 := 
sorry

end smallest_and_largest_group_sizes_l206_206178


namespace cube_cut_problem_l206_206000

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l206_206000


namespace number_of_non_empty_proper_subsets_l206_206896

def non_empty_proper_subsets_count (A : Finset ℕ) : ℕ :=
  ((2 ^ A.card) - 2)

theorem number_of_non_empty_proper_subsets (A : Finset ℕ) (hA : A.card = 3) : 
  non_empty_proper_subsets_count A = 6 :=
by
  rw [non_empty_proper_subsets_count, hA]
  simp

end number_of_non_empty_proper_subsets_l206_206896


namespace lcm_of_54_and_198_l206_206534

theorem lcm_of_54_and_198 : Nat.lcm 54 198 = 594 :=
by
  have fact1 : 54 = 2 ^ 1 * 3 ^ 3 := by norm_num
  have fact2 : 198 = 2 ^ 1 * 3 ^ 2 * 11 ^ 1 := by norm_num
  have lcm_prime : Nat.lcm 54 198 = 594 := by
    sorry -- Proof skipped
  exact lcm_prime

end lcm_of_54_and_198_l206_206534


namespace smallest_value_in_geometric_progression_l206_206673

open Real

theorem smallest_value_in_geometric_progression 
  (d : ℝ) : 
  (∀ a b c d : ℝ, 
    a = 5 ∧ b = 5 + d ∧ c = 5 + 2 * d ∧ d = 5 + 3 * d ∧ 
    ∀ a' b' c' d' : ℝ, 
      a' = 5 ∧ b' = 6 + d ∧ c' = 15 + 2 * d ∧ d' = 3 * d ∧ 
      (b' / a' = c' / b' ∧ c' / b' = d' / c')) → 
  (d = (-1 + 4 * sqrt 10) ∨ d = (-1 - 4 * sqrt 10)) → 
  (min (3 * (-1 + 4 * sqrt 10)) (3 * (-1 - 4 * sqrt 10)) = -3 - 12 * sqrt 10) :=
by
  intros ha hd
  sorry

end smallest_value_in_geometric_progression_l206_206673


namespace arithmetic_mean_is_five_sixths_l206_206132

theorem arithmetic_mean_is_five_sixths :
  let a := 3 / 4
  let b := 5 / 6
  let c := 7 / 8
  (a + c) / 2 = b := sorry

end arithmetic_mean_is_five_sixths_l206_206132


namespace arithmetic_sequence_first_term_range_l206_206883

theorem arithmetic_sequence_first_term_range (a_1 : ℝ) (d : ℝ) (a_10 : ℝ) (a_11 : ℝ) :
  d = (Real.pi / 8) → 
  (a_1 + 9 * d ≤ 0) → 
  (a_1 + 10 * d ≥ 0) → 
  - (5 * Real.pi / 4) ≤ a_1 ∧ a_1 ≤ - (9 * Real.pi / 8) :=
by
  sorry

end arithmetic_sequence_first_term_range_l206_206883


namespace cube_painting_l206_206354

theorem cube_painting (n : ℕ) (h1 : n > 3) 
  (h2 : 2 * (n-2) * (n-2) = 4 * (n-2)) :
  n = 4 :=
sorry

end cube_painting_l206_206354


namespace beads_counter_representation_l206_206456

-- Given conditions
variable (a : ℕ) -- a is a natural number representing the beads in the tens place.
variable (h : a ≥ 0) -- Ensure a is non-negative since the number of beads cannot be negative.

-- The main statement to prove
theorem beads_counter_representation (a : ℕ) (h : a ≥ 0) : 10 * a + 4 = (10 * a) + 4 :=
by sorry

end beads_counter_representation_l206_206456


namespace cube_decomposition_l206_206014

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l206_206014


namespace road_repair_completion_time_l206_206672

theorem road_repair_completion_time (L R r : ℕ) (hL : L = 100) (hR : R = 64) (hr : r = 9) :
  (L - R) / r = 5 :=
by
  sorry

end road_repair_completion_time_l206_206672


namespace cookies_indeterminate_l206_206291

theorem cookies_indeterminate (bananas : ℕ) (boxes : ℕ) (bananas_per_box : ℕ) (cookies : ℕ)
  (h1 : bananas = 40)
  (h2 : boxes = 8)
  (h3 : bananas_per_box = 5)
  : ∃ c : ℕ, c = cookies :=
by sorry

end cookies_indeterminate_l206_206291


namespace billy_reads_books_l206_206844

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end billy_reads_books_l206_206844


namespace average_of_distinct_k_l206_206234

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l206_206234


namespace calculate_expression_l206_206365

theorem calculate_expression : 
  10 * 9 * 8 + 7 * 6 * 5 + 6 * 5 * 4 + 3 * 2 * 1 - 9 * 8 * 7 - 8 * 7 * 6 - 5 * 4 * 3 - 4 * 3 * 2 = 132 :=
by
  sorry

end calculate_expression_l206_206365


namespace soccer_ball_cost_l206_206514

theorem soccer_ball_cost (x : ℕ) (h : 5 * x + 4 * 65 = 980) : x = 144 :=
by
  sorry

end soccer_ball_cost_l206_206514


namespace find_y_l206_206557

theorem find_y (x y : ℤ) (h1 : 2 * (x - y) = 32) (h2 : x + y = -4) : y = -10 :=
sorry

end find_y_l206_206557


namespace calculate_new_shipment_bears_l206_206997

theorem calculate_new_shipment_bears 
  (initial_bears : ℕ)
  (shelves : ℕ)
  (bears_per_shelf : ℕ)
  (total_bears_on_shelves : ℕ) 
  (h_total_bears_on_shelves : total_bears_on_shelves = shelves * bears_per_shelf)
  : initial_bears = 6 → shelves = 4 → bears_per_shelf = 6 → total_bears_on_shelves - initial_bears = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3] at *
  simp at *
  sorry

end calculate_new_shipment_bears_l206_206997


namespace Jerome_money_left_l206_206554

-- Definitions based on conditions
def J_half := 43              -- Half of Jerome's money
def to_Meg := 8               -- Amount Jerome gave to Meg
def to_Bianca := to_Meg * 3   -- Amount Jerome gave to Bianca

-- Total initial amount of Jerome's money
def J_initial : ℕ := J_half * 2

-- Amount left after giving money to Meg
def after_Meg : ℕ := J_initial - to_Meg

-- Amount left after giving money to Bianca
def after_Bianca : ℕ := after_Meg - to_Bianca

-- Statement to be proved
theorem Jerome_money_left : after_Bianca = 54 :=
by
  sorry

end Jerome_money_left_l206_206554


namespace mary_flour_indeterminate_l206_206596

theorem mary_flour_indeterminate 
  (sugar : ℕ) (flour : ℕ) (salt : ℕ) (needed_sugar_more : ℕ) 
  (h_sugar : sugar = 11) (h_flour : flour = 6)
  (h_salt : salt = 9) (h_condition : needed_sugar_more = 2) :
  ∃ (current_flour : ℕ), current_flour ≠ current_flour :=
by
  sorry

end mary_flour_indeterminate_l206_206596


namespace Sine_Theorem_Trihedral_Angle_l206_206464

theorem Sine_Theorem_Trihedral_Angle
  (α β γ A B C : ℝ)
  (hα : 0 < α ∧ α < π)
  (hβ : 0 < β ∧ β < π)
  (hγ : 0 < γ ∧ γ < π)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hcos_α : cos α = cos β * cos γ + sin β * sin γ * cos A)
  (hcos_β : cos β = cos α * cos γ + sin α * sin γ * cos B)
  (hcos_γ : cos γ = cos α * cos β + sin α * sin β * cos C) :
  sin A / sin α = sin B / sin β ∧ sin B / sin β = sin C / sin γ := by
  sorry

end Sine_Theorem_Trihedral_Angle_l206_206464


namespace length_of_shop_proof_l206_206962

-- Given conditions
def monthly_rent : ℝ := 1440
def width : ℝ := 20
def annual_rent_per_sqft : ℝ := 48

-- Correct answer to be proved
def length_of_shop : ℝ := 18

-- The following statement is the proof problem in Lean 4
theorem length_of_shop_proof (h1 : monthly_rent = 1440) 
                            (h2 : width = 20) 
                            (h3 : annual_rent_per_sqft = 48) : 
  length_of_shop = 18 := 
  sorry

end length_of_shop_proof_l206_206962


namespace circles_internally_tangent_l206_206469

theorem circles_internally_tangent :
  let C1 := (3, -2)
  let r1 := 1
  let C2 := (7, 1)
  let r2 := 6
  let d := Real.sqrt (((7 - 3)^2 + (1 - (-2))^2) : ℝ)
  d = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l206_206469


namespace find_positive_integers_unique_solution_l206_206209

theorem find_positive_integers_unique_solution :
  ∃ x r p n : ℕ,  
  0 < x ∧ 0 < r ∧ 0 < n ∧  Nat.Prime p ∧ 
  r > 1 ∧ n > 1 ∧ x^r - 1 = p^n ∧ 
  (x = 3 ∧ r = 2 ∧ p = 2 ∧ n = 3) := 
    sorry

end find_positive_integers_unique_solution_l206_206209


namespace monthly_price_reduction_rate_l206_206925

-- Let's define the given conditions
def initial_price_March : ℝ := 23000
def price_in_May : ℝ := 16000

-- Define the monthly average price reduction rate
variable (x : ℝ)

-- Define the statement to be proven
theorem monthly_price_reduction_rate :
  23 * (1 - x) ^ 2 = 16 :=
sorry

end monthly_price_reduction_rate_l206_206925


namespace last_digit_of_expression_l206_206916

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_of_expression (n : ℕ) : last_digit (n ^ 9999 - n ^ 5555) = 0 :=
by
  sorry

end last_digit_of_expression_l206_206916


namespace sufficient_budget_for_kvass_l206_206326

variables (x y : ℝ)

theorem sufficient_budget_for_kvass (h1 : x + y = 1) (h2 : 0.6 * x + 1.2 * y = 1) : 
  3 * y ≥ 1.44 * y :=
by
  sorry

end sufficient_budget_for_kvass_l206_206326


namespace find_original_number_l206_206675

theorem find_original_number (x y : ℕ) (h1 : x + y = 8) (h2 : 10 * y + x = 10 * x + y + 18) : 10 * x + y = 35 := 
sorry

end find_original_number_l206_206675


namespace reciprocal_of_fraction_l206_206148

noncomputable def fraction := (Real.sqrt 5 + 1) / 2

theorem reciprocal_of_fraction :
  (fraction⁻¹) = (Real.sqrt 5 - 1) / 2 :=
by
  -- proof steps
  sorry

end reciprocal_of_fraction_l206_206148


namespace highest_water_level_changes_on_tuesday_l206_206157

def water_levels : List (String × Float) :=
  [("Monday", 0.03), ("Tuesday", 0.41), ("Wednesday", 0.25), ("Thursday", 0.10),
   ("Friday", 0.0), ("Saturday", -0.13), ("Sunday", -0.2)]

theorem highest_water_level_changes_on_tuesday :
  ∃ d : String, d = "Tuesday" ∧ ∀ d' : String × Float, d' ∈ water_levels → d'.snd ≤ 0.41 := by
  sorry

end highest_water_level_changes_on_tuesday_l206_206157


namespace folded_rectangle_perimeter_l206_206995

theorem folded_rectangle_perimeter (l : ℝ) (w : ℝ) (h_diag : ℝ)
  (h_l : l = 20) (h_w : w = 12)
  (h_diag : h_diag = Real.sqrt (l^2 + w^2)) :
  2 * (l + w) = 64 :=
by
  rw [h_l, h_w]
  simp only [mul_add, mul_two, add_mul] at *
  norm_num


end folded_rectangle_perimeter_l206_206995


namespace problem1_problem2_l206_206079

variable (α : ℝ) (tan_alpha_eq_one_over_three : Real.tan α = 1 / 3)

-- For the first proof problem
theorem problem1 : (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by sorry

-- For the second proof problem
theorem problem2 : Real.cos α ^ 2 - Real.sin (2 * α) = 3 / 10 :=
by sorry

end problem1_problem2_l206_206079


namespace subset_solution_l206_206110

theorem subset_solution (a : ℝ) (A B : Set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  -- Proof will go here
  sorry

end subset_solution_l206_206110


namespace winning_candidate_votes_l206_206478

theorem winning_candidate_votes  (V W : ℝ) (hW : W = 0.5666666666666664 * V) (hV : V = W + 7636 + 11628) : 
  W = 25216 := 
by 
  sorry

end winning_candidate_votes_l206_206478


namespace no_solution_system_l206_206955

theorem no_solution_system :
  ¬ ∃ (x y z : ℝ), (3 * x - 4 * y + z = 10) ∧ (6 * x - 8 * y + 2 * z = 5) ∧ (2 * x - y - z = 4) :=
by {
  sorry
}

end no_solution_system_l206_206955


namespace angle_B_is_60_l206_206102

noncomputable def triangle_with_centroid (a b c : ℝ) (GA GB GC : ℝ) : Prop :=
  56 * a * GA + 40 * b * GB + 35 * c * GC = 0

theorem angle_B_is_60 {a b c GA GB GC : ℝ} (h : 56 * a * GA + 40 * b * GB + 35 * c * GC = 0) :
  ∃ B : ℝ, B = 60 :=
sorry

end angle_B_is_60_l206_206102


namespace division_result_l206_206529

theorem division_result:
    35 / 0.07 = 500 := by
  sorry

end division_result_l206_206529


namespace find_a_plus_c_l206_206620

theorem find_a_plus_c {a b c d : ℝ} 
  (h1 : ∀ x, -|x - a| + b = |x - c| + d → x = 4 ∧ -|4 - a| + b = 7 ∨ x = 10 ∧ -|10 - a| + b = 3)
  (h2 : b + d = 12): a + c = 14 := by
  sorry

end find_a_plus_c_l206_206620


namespace average_k_positive_int_roots_l206_206230

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l206_206230


namespace total_distance_traveled_is_960_l206_206499

-- Definitions of conditions
def first_day_distance : ℝ := 100
def second_day_distance : ℝ := 3 * first_day_distance
def third_day_distance : ℝ := second_day_distance + 110
def fourth_day_distance : ℝ := 150

-- The total distance traveled in four days
def total_distance : ℝ := first_day_distance + second_day_distance + third_day_distance + fourth_day_distance

-- Theorem statement
theorem total_distance_traveled_is_960 :
  total_distance = 960 :=
by
  sorry

end total_distance_traveled_is_960_l206_206499


namespace abs_value_product_l206_206578

theorem abs_value_product (x : ℝ) (h : |x - 5| - 4 = 0) : ∃ y z, (y - 5 = 4 ∨ y - 5 = -4) ∧ (z - 5 = 4 ∨ z - 5 = -4) ∧ y * z = 9 :=
by 
  sorry

end abs_value_product_l206_206578


namespace probability_of_stack_height_48_l206_206638

open Nat

theorem probability_of_stack_height_48 :
  let crates := 12
  let height := 48
  let dims := {3, 4, 6}
  let ways_to_stack := 37522
  let total_ways := 3 ^ 12
  let gcd_of_ways := gcd ways_to_stack total_ways
  gcd_of_ways = 1 →
  (ways_to_stack, total_ways).1 = 37522 :=
by sorry

end probability_of_stack_height_48_l206_206638


namespace trigonometric_identity_l206_206851

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l206_206851


namespace lemonade_water_cups_l206_206594

theorem lemonade_water_cups
  (W S L : ℕ)
  (h1 : W = 5 * S)
  (h2 : S = 3 * L)
  (h3 : L = 5) :
  W = 75 :=
by {
  sorry
}

end lemonade_water_cups_l206_206594


namespace distribution_and_replanting_probability_l206_206972

variable (n : ℕ) (p : ℝ)
variable (X : ℕ → ℕ)

-- Conditions
axiom (h1 : Expectation X = 3)
axiom (h2 : StandardDeviation X = Real.sqrt 1.5)

-- Prove they imply n=6 and p=1/2, and that the distribution probabilities
theorem distribution_and_replanting_probability :
    n = 6 ∧ p = 1/2 ∧ 
    (∀ x, ∃! prob, 
        prob = match x with 
        | 0 => 1/64
        | 1 => 6/64
        | 2 => 15/64
        | 3 => 20/64
        | 4 => 15/64
        | 5 => 6/64
        | 6 => 1/64
        | _ => 0) ∧
    Probability (λ x, x ≤ 3) = 21/32 :=
by
  sorry

end distribution_and_replanting_probability_l206_206972


namespace cube_cut_problem_l206_206002

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l206_206002


namespace probability_max_min_difference_is_five_l206_206969

theorem probability_max_min_difference_is_five : 
  let total_outcomes := 6 ^ 4
  let outcomes_without_1 := 5 ^ 4
  let outcomes_without_6 := 5 ^ 4
  let outcomes_without_1_and_6 := 4 ^ 4
  total_outcomes - 2 * outcomes_without_1 + outcomes_without_1_and_6 = 302 →
  (302 : ℚ) / total_outcomes = 151 / 648 :=
by
  intros
  sorry

end probability_max_min_difference_is_five_l206_206969


namespace mileage_per_gallon_l206_206947

-- Definitions for the conditions
def total_distance_to_grandma (d : ℕ) : Prop := d = 100
def gallons_to_grandma (g : ℕ) : Prop := g = 5

-- The statement to be proved
theorem mileage_per_gallon :
  ∀ (d g m : ℕ), total_distance_to_grandma d → gallons_to_grandma g → m = d / g → m = 20 :=
sorry

end mileage_per_gallon_l206_206947


namespace find_roots_l206_206054

theorem find_roots {n : ℕ} {a_2 a_3 ... a_n : ℂ} 
  (hn : 0 < n)
  (p : Polynomial ℂ := Polynomial.monomial n 1 + Polynomial.monomial (n-1) n + Polynomial.monomial (n-2) a_2 + ... + Polynomial.C a_n)
  (roots : Fin n.succ → ℂ)
  (hroots : Multiset.map (λ r, r.1) (Multiset.ofMap (Polynomial.rootSet p ℂ)) = Multiset.ofFinsupp (Finsupp.onFinset (Finₙ.elems n) (λ i, Coeffs roots i)))
  (sum_mag_16 : Σ (i : Finₙ n), (complex.abs (roots.val i)) ^ 16 = n) :
  ∀ i : Finₙ n, roots i = -1 :=
by
  sorry

end find_roots_l206_206054


namespace increasing_function_condition_l206_206218

theorem increasing_function_condition (k : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k - 6) * x1 + (2 * k + 1) < (2 * k - 6) * x2 + (2 * k + 1)) ↔ (k > 3) :=
by
  -- To prove the statement, we would need to prove it in both directions.
  sorry

end increasing_function_condition_l206_206218


namespace solve_equation_l206_206792

theorem solve_equation : ∀ x : ℝ, x ≠ -2 → x ≠ 0 → (3 / (x + 2) - 1 / x = 0 ↔ x = 1) :=
by
  intro x h1 h2
  sorry

end solve_equation_l206_206792


namespace simplify_fraction_l206_206608

open Real

theorem simplify_fraction (x : ℝ) : (3 + 2 * sin x + 2 * cos x) / (3 + 2 * sin x - 2 * cos x) = 3 / 5 + (2 / 5) * cos x :=
by
  sorry

end simplify_fraction_l206_206608


namespace avg_k_for_polynomial_roots_l206_206247

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l206_206247


namespace no_20_odd_rows_15_odd_columns_l206_206388

theorem no_20_odd_rows_15_odd_columns (n : ℕ) (table : ℕ → ℕ → bool) (cross_count 
  : ℕ) 
  (odd_rows : ℕ → bool) 
  (odd_columns : ℕ → bool) :
  (∀ i, i < n → (odd_rows i = true ↔ ∃ j, j < n ∧ table i j = true ∧ cross_count = 20))
  → (∀ j, j < n → (odd_columns j = true ↔ ∃ i, i < n ∧ table i j = true ∧ cross_count = 15))
  → false := 
sorry

end no_20_odd_rows_15_odd_columns_l206_206388


namespace range_of_2a_minus_b_l206_206715

variable (a b : ℝ)
variable (h1 : -2 < a ∧ a < 2)
variable (h2 : 2 < b ∧ b < 3)

theorem range_of_2a_minus_b (a b : ℝ) (h1 : -2 < a ∧ a < 2) (h2 : 2 < b ∧ b < 3) :
  -7 < 2 * a - b ∧ 2 * a - b < 2 := sorry

end range_of_2a_minus_b_l206_206715


namespace part1_part2_l206_206543

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem part1 (x : ℝ) : (f x)^2 - (g x)^2 = -4 :=
by sorry

theorem part2 (x y : ℝ) (h1 : f x * f y = 4) (h2 : g x * g y = 8) : 
  g (x + y) / g (x - y) = 3 :=
by sorry

end part1_part2_l206_206543


namespace minimum_area_of_triangle_AOC_is_28_over_3_l206_206768

noncomputable def minimum_area_triangle_AOC : ℝ :=
  let A (x₁ : ℝ) := (x₁, 3 * x₁) in
  let C (k : ℝ) := (2 / k + 3, 0) in 
  let area (k : ℝ) := (3 * (3 * k - 2) * (3 * k - 10) / (k * (k - 3))) / 2 in
  let valid_k := λ k, 2 / 3 < k ∧ k < 3 in
  Sup (set.image area {k | valid_k k})

theorem minimum_area_of_triangle_AOC_is_28_over_3 :
  minimum_area_triangle_AOC = 28 / 3 :=
sorry

end minimum_area_of_triangle_AOC_is_28_over_3_l206_206768


namespace bins_of_soup_l206_206059

theorem bins_of_soup (total_bins : ℝ) (bins_of_vegetables : ℝ) (bins_of_pasta : ℝ) 
(h1 : total_bins = 0.75) (h2 : bins_of_vegetables = 0.125) (h3 : bins_of_pasta = 0.5) :
  total_bins - (bins_of_vegetables + bins_of_pasta) = 0.125 := by
  -- proof
  sorry

end bins_of_soup_l206_206059


namespace element_subset_a_l206_206107

theorem element_subset_a (a : ℝ) (A B : set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  sorry

end element_subset_a_l206_206107


namespace no_prime_numbers_divisible_by_91_l206_206415

-- Define the concept of a prime number.
def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the factors of 91.
def factors_of_91 (n : ℕ) : Prop :=
  n = 7 ∨ n = 13

-- State the problem formally: there are no prime numbers divisible by 91.
theorem no_prime_numbers_divisible_by_91 :
  ∀ p : ℕ, is_prime p → ¬ (91 ∣ p) :=
by
  intros p prime_p div91
  sorry

end no_prime_numbers_divisible_by_91_l206_206415


namespace polynomial_problem_l206_206122

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem polynomial_problem (f_nonzero : ∀ x, f x ≠ 0) 
  (h1 : ∀ x, f (g x) = f x * g x)
  (h2 : g 3 = 10)
  (h3 : ∃ a b, g x = a * x + b) :
  g x = 2 * x + 4 :=
sorry

end polynomial_problem_l206_206122


namespace find_f1_l206_206726

noncomputable def f (x a b : ℝ) : ℝ := a * Real.sin x - b * Real.tan x + 4 * Real.cos (Real.pi / 3)

theorem find_f1 (a b : ℝ) (h : f (-1) a b = 1) : f 1 a b = 3 :=
by {
  sorry
}

end find_f1_l206_206726


namespace geom_seq_common_ratio_l206_206927

theorem geom_seq_common_ratio (a1 : ℤ) (S3 : ℚ) (q : ℚ) (hq : -2 * (1 + q + q^2) = - (7 / 2)) : 
  q = 1 / 2 ∨ q = -3 / 2 :=
sorry

end geom_seq_common_ratio_l206_206927


namespace negate_one_even_l206_206294

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_one_even (a b c : ℕ) :
  (∃! x, x = a ∨ x = b ∨ x = c ∧ is_even x) ↔
  (∃ x y, x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧
    x ≠ y ∧ is_even x ∧ is_even y) ∨
  (is_odd a ∧ is_odd b ∧ is_odd c) :=
by {
  sorry
}

end negate_one_even_l206_206294


namespace solve_equation1_solve_equation2_l206_206141

-- Define the first equation
def equation1 (x : ℝ) : Prop :=
  2 * x^2 = 3 * (2 * x + 1)

-- Define the solution set for the first equation
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 15) / 2 ∨ x = (3 - Real.sqrt 15) / 2

-- Prove that the solutions for the first equation are correct
theorem solve_equation1 (x : ℝ) : equation1 x ↔ solution1 x :=
by
  sorry

-- Define the second equation
def equation2 (x : ℝ) : Prop :=
  3 * x * (x + 2) = 4 * x + 8

-- Define the solution set for the second equation
def solution2 (x : ℝ) : Prop :=
  x = -2 ∨ x = 4 / 3

-- Prove that the solutions for the second equation are correct
theorem solve_equation2 (x : ℝ) : equation2 x ↔ solution2 x :=
by
  sorry

end solve_equation1_solve_equation2_l206_206141


namespace relationship_between_length_and_width_l206_206980

theorem relationship_between_length_and_width 
  (x y : ℝ) (h : 2 * (x + y) = 20) : y = 10 - x := 
by
  sorry

end relationship_between_length_and_width_l206_206980


namespace solve_system_of_equations_l206_206302

theorem solve_system_of_equations :
  (∃ x y : ℝ, (x / y + y / x = 173 / 26) ∧ (1 / x + 1 / y = 15 / 26) ∧ ((x = 13 ∧ y = 2) ∨ (x = 2 ∧ y = 13))) :=
by
  sorry

end solve_system_of_equations_l206_206302


namespace expected_variance_X_expected_200_variance_360_l206_206378

noncomputable def problem_conditions (n : ℕ) (p : ℝ) : Prop :=
  n = 1000 ∧ p = 0.1

noncomputable def define_X (ξ : ℕ → ℝ) : (ℕ → ℝ) :=
  λ n, 2 * ξ n

noncomputable def expected_value_of_X (Eξ : ℕ → ℝ) : ℝ :=
  2 * Eξ 1000

noncomputable def variance_of_X (n : ℕ) (p : ℝ) (Dξ : ℕ → ℝ) : ℝ :=
  4 * n * p * (1 - p)

theorem expected_variance_X_expected_200_variance_360 {n : ℕ} {p : ℝ} (h : problem_conditions n p) :
  let ξ : ℕ → ℝ := fun n => n.to_real * p,
      Eξ : ℕ → ℝ := λ n, n.to_real * p,
      Dξ : ℕ → ℝ := λ n, n.to_real * p * (1 - p),
      X : (ℕ → ℝ) := define_X ξ
  in expected_value_of_X Eξ = 200 ∧ variance_of_X n p Dξ = 360 := by
  obtain ⟨h₁, h₂⟩ := h
  unfold ξ Eξ Dξ at *
  rw [h₁, h₂]
  have hE : expected_value_of_X (λ n, n.to_real * 0.1) = 200, by
    simp [expected_value_of_X, mul_assoc, eq_comm]
  have hV : variance_of_X 1000 0.1 (λ n, n.to_real * 0.1 * 0.9) = 360, by
    simp [variance_of_X, mul_assoc, mul_comm, eq_comm, sub_self]
  exact ⟨hE, hV⟩

end expected_variance_X_expected_200_variance_360_l206_206378


namespace sue_nuts_count_l206_206687

theorem sue_nuts_count (B H S : ℕ) 
  (h1 : B = 6 * H) 
  (h2 : H = 2 * S) 
  (h3 : B + H = 672) : S = 48 := 
by
  sorry

end sue_nuts_count_l206_206687


namespace num_diagonals_tetragon_l206_206400

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem num_diagonals_tetragon : num_diagonals_in_polygon 4 = 2 := by
  sorry

end num_diagonals_tetragon_l206_206400


namespace f_at_neg2_l206_206887

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - Real.log (x^2 - 3*x + 5) / Real.log 3 
else -2^(-x) + Real.log ((-x)^2 + 3*(-x) + 5) / Real.log 3 

theorem f_at_neg2 : f (-2) = -3 := by
  sorry

end f_at_neg2_l206_206887


namespace avg_k_value_l206_206238

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l206_206238


namespace sales_tax_difference_l206_206617

-- Definitions for the price and tax rates
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.065
def tax_rate2 : ℝ := 0.06
def tax_rate3 : ℝ := 0.07

-- Sales tax amounts derived from the given rates and item price
def tax_amount (rate : ℝ) (price : ℝ) : ℝ := rate * price

-- Calculate the individual tax amounts
def tax_amount1 : ℝ := tax_amount tax_rate1 item_price
def tax_amount2 : ℝ := tax_amount tax_rate2 item_price
def tax_amount3 : ℝ := tax_amount tax_rate3 item_price

-- Proposition stating the proof problem
theorem sales_tax_difference :
  max tax_amount1 (max tax_amount2 tax_amount3) - min tax_amount1 (min tax_amount2 tax_amount3) = 0.50 :=
by 
  sorry

end sales_tax_difference_l206_206617


namespace sqrt_of_sixteen_l206_206152

theorem sqrt_of_sixteen : ∃ x : ℤ, x^2 = 16 ∧ (x = 4 ∨ x = -4) := by
  sorry

end sqrt_of_sixteen_l206_206152


namespace amy_total_spending_l206_206047

def initial_tickets : ℕ := 33
def cost_per_ticket : ℝ := 1.50
def additional_tickets : ℕ := 21
def total_cost : ℝ := 81.00

theorem amy_total_spending :
  (initial_tickets * cost_per_ticket + additional_tickets * cost_per_ticket) = total_cost := 
sorry

end amy_total_spending_l206_206047


namespace hannah_mugs_problem_l206_206257

theorem hannah_mugs_problem :
  ∀ (total_mugs red_mugs yellow_mugs blue_mugs : ℕ),
    total_mugs = 40 →
    yellow_mugs = 12 →
    red_mugs * 2 = yellow_mugs →
    blue_mugs = 3 * red_mugs →
    total_mugs - (red_mugs + yellow_mugs + blue_mugs) = 4 :=
by
  intros total_mugs red_mugs yellow_mugs blue_mugs Htotal Hyellow Hred Hblue
  sorry

end hannah_mugs_problem_l206_206257


namespace unique_solution_m_l206_206399

theorem unique_solution_m (m : ℝ) :
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) ↔ (m = 0 ∨ m = -1) :=
by
  sorry

end unique_solution_m_l206_206399


namespace trains_crossing_time_correct_l206_206176

def convert_kmph_to_mps (speed_kmph : ℕ) : ℚ := (speed_kmph * 5) / 18

def time_to_cross_each_other 
  (length_train1 length_train2 speed_kmph_train1 speed_kmph_train2 : ℕ) : ℚ :=
  let speed_train1 := convert_kmph_to_mps speed_kmph_train1
  let speed_train2 := convert_kmph_to_mps speed_kmph_train2
  let relative_speed := speed_train2 - speed_train1
  let total_distance := length_train1 + length_train2
  (total_distance : ℚ) / relative_speed

theorem trains_crossing_time_correct :
  time_to_cross_each_other 200 150 40 46 = 210 := by
  sorry

end trains_crossing_time_correct_l206_206176


namespace solve_AlyoshaCube_l206_206022

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l206_206022


namespace min_tiles_for_square_l206_206771

theorem min_tiles_for_square (a b : ℕ) (ha : a = 6) (hb : b = 4) (harea_tile : a * b = 24)
  (h_lcm : Nat.lcm a b = 12) : 
  let area_square := (Nat.lcm a b) * (Nat.lcm a b) 
  let num_tiles_required := area_square / (a * b)
  num_tiles_required = 6 :=
by
  sorry

end min_tiles_for_square_l206_206771


namespace male_students_stratified_sampling_l206_206668

theorem male_students_stratified_sampling :
  ∀ (total_students total_female_students sample_size : ℕ)
  (sampling_ratio : ℚ)
  (total_male_students : ℕ),
  total_students = 900 →
  total_female_students = 400 →
  sample_size = 45 →
  sampling_ratio = sample_size /. total_students →
  total_male_students = total_students - total_female_students →
  (total_male_students /. 20) = 25 :=
by
  intros total_students total_female_students sample_size sampling_ratio total_male_students
  assume h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end male_students_stratified_sampling_l206_206668


namespace average_of_distinct_k_l206_206233

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l206_206233


namespace distinct_book_arrangements_l206_206417

def num_books := 7
def num_identical_books := 3
def num_unique_books := num_books - num_identical_books

theorem distinct_book_arrangements :
  (Nat.factorial num_books) / (Nat.factorial num_identical_books) = 840 := 
  by 
  sorry

end distinct_book_arrangements_l206_206417


namespace find_a_value_l206_206223

theorem find_a_value (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53^2017 + a) % 13 = 0) : a = 12 :=
by
  -- proof steps
  sorry

end find_a_value_l206_206223


namespace average_of_distinct_k_l206_206235

noncomputable def average_distinct_k (k_list : List ℚ) : ℚ :=
  (k_list.foldl (+) 0) / k_list.length

theorem average_of_distinct_k : 
  ∃ k_values : List ℚ, 
  (∀ (r1 r2 : ℚ), r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0 → (k_values = [1 + 24, 2 + 12, 3 + 8, 4 + 6] )) ∧
  average_distinct_k k_values = 15 :=
  sorry

end average_of_distinct_k_l206_206235


namespace cos_F_in_triangle_l206_206283

theorem cos_F_in_triangle (D E F : ℝ) (sin_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) 
  (h1 : sin_D = 4 / 5) 
  (h2 : cos_E = 12 / 13) 
  (D_plus_E_plus_F : D + E + F = π) :
  cos_F = -16 / 65 :=
by
  sorry

end cos_F_in_triangle_l206_206283


namespace divide_sum_eq_100_l206_206492

theorem divide_sum_eq_100 (x : ℕ) (h1 : 100 = 2 * x + (100 - 2 * x)) (h2 : (300 - 6 * x) + x = 100) : x = 40 :=
by
  sorry

end divide_sum_eq_100_l206_206492


namespace smaller_fraction_l206_206796

variable (x y : ℚ)

theorem smaller_fraction (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 1 / 6 :=
by
  sorry

end smaller_fraction_l206_206796


namespace cubical_pyramidal_segment_volume_and_area_l206_206666

noncomputable def volume_and_area_sum (a : ℝ) : ℝ :=
  (1/4 * (9 + 27 * Real.sqrt 13))

theorem cubical_pyramidal_segment_volume_and_area :
  ∀ a : ℝ, a = 3 → volume_and_area_sum a = (9/2 + 27 * Real.sqrt 13 / 8) := by
  intro a ha
  sorry

end cubical_pyramidal_segment_volume_and_area_l206_206666


namespace cells_surpass_10_pow_10_in_46_hours_l206_206799

noncomputable def cells_exceed_threshold_hours : ℕ := 46

theorem cells_surpass_10_pow_10_in_46_hours : 
  ∀ (n : ℕ), (100 * ((3 / 2 : ℝ) ^ n) > 10 ^ 10) ↔ n ≥ cells_exceed_threshold_hours := 
by
  sorry

end cells_surpass_10_pow_10_in_46_hours_l206_206799


namespace lemonade_calories_is_correct_l206_206286

def lemon_juice_content := 150
def sugar_content := 150
def water_content := 450

def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def water_calories_per_100g := 0

def total_weight := lemon_juice_content + sugar_content + water_content
def caloric_density :=
  (lemon_juice_content * lemon_juice_calories_per_100g / 100) +
  (sugar_content * sugar_calories_per_100g / 100) +
  (water_content * water_calories_per_100g / 100)
def calories_per_gram := caloric_density / total_weight

def calories_in_300_grams := 300 * calories_per_gram

theorem lemonade_calories_is_correct : calories_in_300_grams = 258 := by
  sorry

end lemonade_calories_is_correct_l206_206286


namespace quadrilateral_area_l206_206700

theorem quadrilateral_area 
  (d : ℝ) (h₁ h₂ : ℝ) 
  (hd : d = 22) 
  (hh₁ : h₁ = 9) 
  (hh₂ : h₂ = 6) : 
  (1/2 * d * h₁ + 1/2 * d * h₂ = 165) :=
by
  sorry

end quadrilateral_area_l206_206700


namespace monica_total_savings_l206_206944

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end monica_total_savings_l206_206944


namespace a_alone_can_finish_job_l206_206815

def work_in_one_day (A B : ℕ) : Prop := 1/A + 1/B = 1/40

theorem a_alone_can_finish_job (A B : ℕ)
  (work_rate : work_in_one_day A B) 
  (together_10_days : 10 * (1/A + 1/B) = 1/4) 
  (a_21_days : 21 * (1/A) = 3/4) : 
  A = 28 := 
sorry

end a_alone_can_finish_job_l206_206815


namespace three_points_integer_centroid_l206_206882

/--  
  Given 19 points in the plane with integer coordinates, no three collinear, 
  show that we can always find three points whose centroid has integer coordinates.
-/
theorem three_points_integer_centroid 
  (points: fin 19 → ℤ × ℤ)
  (h_no_three_collinear: ∀ (a b c : fin 19), a ≠ b → b ≠ c → a ≠ c → ¬ collinear [points a, points b, points c]) :
  ∃ (i j k : fin 19), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    let (x₁, y₁) := points i,
        (x₂, y₂) := points j,
        (x₃, y₃) := points k in
      (x₁ + x₂ + x₃) % 3 = 0 ∧ (y₁ + y₂ + y₃) % 3 = 0 :=
by
  sorry

end three_points_integer_centroid_l206_206882


namespace z_coordinate_of_point_on_line_l206_206992

theorem z_coordinate_of_point_on_line (t : ℝ)
  (h₁ : (1 + 3 * t, 3 + 2 * t, 2 + 4 * t) = (x, 7, z))
  (h₂ : x = 1 + 3 * t) :
  z = 10 :=
sorry

end z_coordinate_of_point_on_line_l206_206992


namespace alyosha_cube_problem_l206_206028

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l206_206028


namespace correct_propositions_for_curve_C_l206_206549

def curve_C (k : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / (4 - k) + y^2 / (k - 1) = 1)

theorem correct_propositions_for_curve_C (k : ℝ) :
  (∀ x y : ℝ, curve_C k) →
  ((∃ k, ((4 - k) * (k - 1) < 0) ↔ (k < 1 ∨ k > 4)) ∧
  ((1 < k ∧ k < (5 : ℝ) / 2) ↔
  (4 - k > k - 1 ∧ 4 - k > 0 ∧ k - 1 > 0))) :=
by {
  sorry
}

end correct_propositions_for_curve_C_l206_206549


namespace no_first_or_fourth_quadrant_l206_206544

theorem no_first_or_fourth_quadrant (a b : ℝ) (h : a * b > 0) : 
  ¬ ((∃ x, a * x + b = 0 ∧ x > 0) ∧ (∃ x, b * x + a = 0 ∧ x > 0)) 
  ∧ ¬ ((∃ x, a * x + b = 0 ∧ x < 0) ∧ (∃ x, b * x + a = 0 ∧ x < 0)) := sorry

end no_first_or_fourth_quadrant_l206_206544


namespace largest_fraction_l206_206058

theorem largest_fraction (f1 f2 f3 f4 f5 : ℚ) (h1 : f1 = 2 / 5)
                                          (h2 : f2 = 3 / 6)
                                          (h3 : f3 = 5 / 10)
                                          (h4 : f4 = 7 / 15)
                                          (h5 : f5 = 8 / 20) : 
  (f2 = 1 / 2 ∨ f3 = 1 / 2) ∧ (f2 ≥ f1 ∧ f2 ≥ f4 ∧ f2 ≥ f5) ∧ (f3 ≥ f1 ∧ f3 ≥ f4 ∧ f3 ≥ f5) := 
by
  sorry

end largest_fraction_l206_206058


namespace dealer_profit_percentage_l206_206324

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℝ) (sp_total : ℝ) (sp_count : ℝ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  let profit_percentage := (profit_per_article / cp_per_article) * 100
  profit_percentage

theorem dealer_profit_percentage :
  profit_percentage 25 15 38 12 = 89.99 := by
  sorry

end dealer_profit_percentage_l206_206324


namespace v_function_expression_f_max_value_l206_206305

noncomputable def v (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2
else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2)
else 0

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then 2 * x
else if 4 < x ∧ x ≤ 20 then - (1/8) * x^2 + (5/2) * x
else 0

theorem v_function_expression :
  ∀ x, 0 < x ∧ x ≤ 20 → 
  v x = (if 0 < x ∧ x ≤ 4 then 2 else if 4 < x ∧ x ≤ 20 then - (1/8) * x + (5/2) else 0) :=
by sorry

theorem f_max_value :
  ∃ x, 0 < x ∧ x ≤ 20 ∧ f x = 12.5 :=
by sorry

end v_function_expression_f_max_value_l206_206305


namespace sequence_a_n_sequence_b_n_range_k_l206_206548

-- Define the geometric sequence {a_n} with initial conditions
def a (n : ℕ) : ℕ :=
  3 * 2^(n-1)

-- Define the sequence {b_n} with the given recurrence relation
def b : ℕ → ℕ
| 0 => 1
| (n+1) => 2 * (b n) + 1

theorem sequence_a_n (n : ℕ) : 
  (a n = 3 * 2^(n-1)) := sorry

theorem sequence_b_n (n : ℕ) :
  (b n = 2^n - 1) := sorry

-- Define the condition for k and the inequality
def condition_k (k : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (k * (↑(b n) + 5) / 2 - 3 * 2^(n-1) ≥ 8*n + 2*k - 24)

-- Prove the range for k
theorem range_k (k : ℝ) :
  (condition_k k ↔ k ≥ 4) := sorry

end sequence_a_n_sequence_b_n_range_k_l206_206548


namespace total_tires_l206_206068

def cars := 15
def bicycles := 3
def pickup_trucks := 8
def tricycles := 1

def tires_per_car := 4
def tires_per_bicycle := 2
def tires_per_pickup_truck := 4
def tires_per_tricycle := 3

theorem total_tires : (cars * tires_per_car) + (bicycles * tires_per_bicycle) + (pickup_trucks * tires_per_pickup_truck) + (tricycles * tires_per_tricycle) = 101 :=
by
  sorry

end total_tires_l206_206068


namespace cube_decomposition_l206_206018

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l206_206018


namespace series_sum_equals_seven_ninths_l206_206864

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l206_206864


namespace identify_true_statements_l206_206893

-- Definitions of the given statements
def statement1 (a x y : ℝ) : Prop := a * (x + y) = a * x + a * y
def statement2 (a x y : ℝ) : Prop := a ^ (x + y) = a ^ x + a ^ y
def statement3 (x y : ℝ) : Prop := (x + y) ^ 2 = x ^ 2 + y ^ 2
def statement4 (a b : ℝ) : Prop := Real.sqrt (a ^ 2 + b ^ 2) = a + b
def statement5 (a b c : ℝ) : Prop := a * (b / c) = (a * b) / c

-- The statement to prove
theorem identify_true_statements (a x y b c : ℝ) :
  statement1 a x y ∧ statement5 a b c ∧
  ¬ statement2 a x y ∧ ¬ statement3 x y ∧ ¬ statement4 a b :=
sorry

end identify_true_statements_l206_206893


namespace student_teacher_arrangements_l206_206636

theorem student_teacher_arrangements :
  let total_arrangements := (choose 5 2) * 2 * (perm 4 4)
  in total_arrangements = 960 :=
begin
  -- Defining parameters and the number of ways to choose the 2 students
  let num_students := 5,
  let num_teachers := 2,
  let students_between := 2,
  
  -- Calculating the number of ways to choose 2 students out of 5
  let choose_students : ℕ := (choose num_students students_between),

  -- Considering the order of the teachers (2 options)
  let order_teachers : ℕ := 2,

  -- Treating the group of 4 individuals as one element and arranging them with the other 3 individuals
  let arrange_others : ℕ := (perm 4 4),

  -- Calculating the total number of arrangements
  let total_arrangements := choose_students * order_teachers * arrange_others,

  -- The equality we need to prove
  show total_arrangements = 960, from sorry,
end

end student_teacher_arrangements_l206_206636


namespace total_cars_made_in_two_days_l206_206353

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l206_206353


namespace annual_income_before_tax_l206_206423

variable (I : ℝ) -- Define I as the annual income before tax

-- Conditions
def original_tax (I : ℝ) : ℝ := 0.42 * I
def new_tax (I : ℝ) : ℝ := 0.32 * I
def differential_savings (I : ℝ) : ℝ := original_tax I - new_tax I

-- Theorem: Given the conditions, the taxpayer's annual income before tax is $42,400
theorem annual_income_before_tax : differential_savings I = 4240 → I = 42400 := by
  sorry

end annual_income_before_tax_l206_206423


namespace intersection_points_l206_206287

noncomputable def f (x : ℝ) : ℝ := (x^2 - 8*x + 15) / (3*x - 6)

noncomputable def g (x : ℝ) : ℝ := (-3*x^2 - 6*x + 115) / (x - 2)

theorem intersection_points:
  ∃ (x1 x2 : ℝ), x1 ≠ -3 ∧ x2 ≠ -3 ∧ (f x1 = g x1) ∧ (f x2 = g x2) ∧ 
  (x1 = -11 ∧ f x1 = -2) ∧ (x2 = 3 ∧ f x2 = -2) := 
sorry

end intersection_points_l206_206287


namespace find_binomial_params_l206_206386

noncomputable def binomial_params (n p : ℝ) := 2.4 = n * p ∧ 1.44 = n * p * (1 - p)

theorem find_binomial_params (n p : ℝ) (h : binomial_params n p) : n = 6 ∧ p = 0.4 :=
by
  sorry

end find_binomial_params_l206_206386


namespace boat_downstream_distance_l206_206332

theorem boat_downstream_distance (V_b V_s : ℝ) (t_downstream t_upstream : ℝ) (d_upstream : ℝ) 
  (h1 : t_downstream = 8) (h2 : t_upstream = 15) (h3 : d_upstream = 75) (h4 : V_s = 3.75) 
  (h5 : V_b - V_s = (d_upstream / t_upstream)) : (V_b + V_s) * t_downstream = 100 :=
by
  sorry

end boat_downstream_distance_l206_206332


namespace solve_system_of_odes_l206_206303

theorem solve_system_of_odes (C₁ C₂ : ℝ) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = (C₁ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, y t = (C₁ + C₂ + C₂ * t) * Real.exp (3 * t)) ∧
    (∀ t, deriv x t = 2 * x t + y t) ∧
    (∀ t, deriv y t = 4 * y t - x t) :=
by
  sorry

end solve_system_of_odes_l206_206303


namespace prime_squares_mod_180_l206_206685

theorem prime_squares_mod_180 (p : ℕ) (hp : prime p) (hp_gt_5 : p > 5) :
  ∃ (r1 r2 : ℕ), 
  r1 ≠ r2 ∧ 
  ∀ r : ℕ, (∃ m : ℕ, p^2 = m * 180 + r) → (r = r1 ∨ r = r2) :=
sorry

end prime_squares_mod_180_l206_206685


namespace expression_equals_36_l206_206939

def k := 13

theorem expression_equals_36 : 13 * (3 - 3 / 13) = 36 := by
  sorry

end expression_equals_36_l206_206939


namespace unique_prime_triple_l206_206984

/-- A prime is an integer greater than 1 whose only positive integer divisors are itself and 1. -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

/-- Prove that the only triple of primes (p, q, r), such that p = q + 2 and q = r + 2 is (7, 5, 3). -/
theorem unique_prime_triple (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) :
  (p = q + 2) ∧ (q = r + 2) → (p = 7 ∧ q = 5 ∧ r = 3) := by
  sorry

end unique_prime_triple_l206_206984


namespace no_four_digit_number_differs_from_reverse_by_1008_l206_206195

theorem no_four_digit_number_differs_from_reverse_by_1008 :
  ∀ a b c d : ℕ, 
  a < 10 → b < 10 → c < 10 → d < 10 → (999 * (a - d) + 90 * (b - c) ≠ 1008) :=
by
  intro a b c d ha hb hc hd h
  sorry

end no_four_digit_number_differs_from_reverse_by_1008_l206_206195


namespace geometric_sequence_sum_l206_206125

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_ratio_ne_one : q ≠ 1)
  (S : ℕ → ℝ) (h_a1 : S 1 = 1) (h_S4_eq_5S2 : S 4 - 5 * S 2 = 0) :
  S 5 = 31 :=
sorry

end geometric_sequence_sum_l206_206125


namespace correct_number_of_true_propositions_l206_206041

noncomputable def true_proposition_count : ℕ := 1

theorem correct_number_of_true_propositions (a b c : ℝ) :
    (∀ a b : ℝ, (a > b) ↔ (a^2 > b^2) = false) →
    (∀ a b : ℝ, (a > b) ↔ (a^3 > b^3) = true) →
    (∀ a b : ℝ, (a > b) → (|a| > |b|) = false) →
    (∀ a b c : ℝ, (a > b) → (a*c^2 ≤ b*c^2) = false) →
    (true_proposition_count = 1) :=
by
  sorry

end correct_number_of_true_propositions_l206_206041


namespace fraction_to_decimal_l206_206654

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 := 
sorry

end fraction_to_decimal_l206_206654


namespace game_winner_l206_206485

theorem game_winner (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  (mn % 2 = 1 → first_player_wins) ∧ (mn % 2 = 0 → second_player_wins) :=
sorry

end game_winner_l206_206485


namespace fraction_to_decimal_l206_206653

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 := 
sorry

end fraction_to_decimal_l206_206653


namespace count_factors_of_180_multiple_of_15_l206_206407

def is_factor (n k : ℕ) : Prop := n % k = 0

def is_multiple_of (k m : ℕ) : Prop := k % m = 0

theorem count_factors_of_180_multiple_of_15 : 
  (finset.filter (λ f, is_factor 180 f ∧ is_multiple_of f 15) (finset.range (181))).card = 6 := 
by 
sorry

end count_factors_of_180_multiple_of_15_l206_206407


namespace min_green_beads_l206_206510

theorem min_green_beads (B R G : ℕ)
  (h_total : B + R + G = 80)
  (h_red_blue : ∀ i j, B ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < R)
  (h_green_red : ∀ i j, R ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < G)
  : G = 27 := 
sorry

end min_green_beads_l206_206510


namespace find_alpha_polar_eqn_of_line_l206_206380

open Real

noncomputable def P : Point := ⟨2, 1⟩

def line_eqs (alpha : ℝ) (t : ℝ) : Point :=
  ⟨2 + t * cos(alpha), 1 + t * sin(alpha)⟩

def PA (alpha : ℝ) : ℝ :=
  dist P ⟨2 + (-1 / sin(alpha)) * cos(alpha), 0⟩

def PB (alpha : ℝ) : ℝ :=
  dist P ⟨0, 1 + (-2 / cos(alpha)) * sin(alpha)⟩

theorem find_alpha (alpha : ℝ) (h : PA(alpha) * PB(alpha) = 4) : alpha = 3 * π / 4 :=
sorry

theorem polar_eqn_of_line (alpha : ℝ) (h : PA(alpha) * PB(alpha) = 4) : 
  (∃ ρ θ, ρ * (cos(θ) + sin(θ)) = 3) :=
sorry

end find_alpha_polar_eqn_of_line_l206_206380


namespace solve_AlyoshaCube_l206_206021

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l206_206021


namespace abc_sum_is_17_l206_206835

noncomputable def A := 3
noncomputable def B := 5
noncomputable def C := 9

theorem abc_sum_is_17 (A B C : ℕ) (h1 : 100 * A + 10 * B + C = 359) (h2 : 4 * (100 * A + 10 * B + C) = 1436)
  (h3 : A ≠ B) (h4 : B ≠ C) (h5 : A ≠ C) : A + B + C = 17 :=
by
  sorry

end abc_sum_is_17_l206_206835


namespace total_skips_l206_206373

theorem total_skips (fifth throw : ℕ) (fourth throw : ℕ) (third throw : ℕ) (second throw : ℕ) (first throw : ℕ) :
  fifth throw = 8 →
  fourth throw = fifth throw - 1 →
  third throw = fourth throw + 3 →
  second throw = third throw / 2 →
  first throw = second throw - 2 →
  first throw + second throw + third throw + fourth throw + fifth throw = 33 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end total_skips_l206_206373


namespace sample_size_is_five_l206_206714

def population := 100
def sample (n : ℕ) := n ≤ population
def sample_size (n : ℕ) := n

theorem sample_size_is_five (n : ℕ) (h : sample 5) : sample_size 5 = 5 :=
by
  sorry

end sample_size_is_five_l206_206714


namespace value_of_x_squared_plus_y_squared_l206_206560

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l206_206560


namespace tom_sleep_deficit_l206_206482

-- Define the conditions as given
def weeknights := 5
def weekend_nights := 2
def ideal_sleep_hours := 8
def actual_weeknight_sleep := 5
def actual_weekend_sleep := 6

-- Define the proofs
theorem tom_sleep_deficit : 
  let ideal_week_sleep := (weeknights * ideal_sleep_hours) + (weekend_nights * ideal_sleep_hours) in
  let actual_week_sleep := (weeknights * actual_weeknight_sleep) + (weekend_nights * actual_weekend_sleep) in
  ideal_week_sleep - actual_week_sleep = 19 := 
by
  let ideal_week_sleep := (weeknights * ideal_sleep_hours) + (weekend_nights * ideal_sleep_hours)
  let actual_week_sleep := (weeknights * actual_weeknight_sleep) + (weekend_nights * actual_weekend_sleep)
  have h1 : ideal_week_sleep = 56 := by rfl
  have h2 : actual_week_sleep = 37 := by rfl
  show ideal_week_sleep - actual_week_sleep = 19 from by
    rw [h1, h2]
    apply nat.sub_self_eq_suc_natu

/-- Placeholder for the proof -/
sorry

end tom_sleep_deficit_l206_206482


namespace total_amount_shared_l206_206677

theorem total_amount_shared (a b c : ℕ) (h_ratio : a = 3 * b / 5 ∧ c = 9 * b / 5) (h_b : b = 50) : a + b + c = 170 :=
by sorry

end total_amount_shared_l206_206677


namespace little_johns_money_left_l206_206290

def J_initial : ℝ := 7.10
def S : ℝ := 1.05
def F : ℝ := 1.00

theorem little_johns_money_left :
  J_initial - (S + 2 * F) = 4.05 :=
by sorry

end little_johns_money_left_l206_206290


namespace at_least_one_not_greater_one_third_sqrt_inequality_l206_206327

-- Problem 1: Prove at least one of a, b, c is not greater than 1/3 given a + b + c = 1
theorem at_least_one_not_greater_one_third (a b c : ℝ) (h : a + b + c = 1) : a ≤ 1/3 ∨ b ≤ 1/3 ∨ c ≤ 1/3 :=
sorry

-- Problem 2: Prove sqrt(6) + sqrt(7) > 2sqrt(2) + sqrt(5)
theorem sqrt_inequality : √6 + √7 > 2 * √2 + √5 :=
sorry

end at_least_one_not_greater_one_third_sqrt_inequality_l206_206327


namespace number_of_lines_intersecting_circle_l206_206894

theorem number_of_lines_intersecting_circle : 
  ∃ l : ℕ, 
  (∀ a b x y : ℤ, (x^2 + y^2 = 50 ∧ (x / a + y / b = 1))) → 
  (∃ n : ℕ, n = 60) :=
sorry

end number_of_lines_intersecting_circle_l206_206894


namespace contrapositive_equivalence_l206_206786

variable (Person : Type)
variable (Happy Have : Person → Prop)

theorem contrapositive_equivalence :
  (∀ (x : Person), Happy x → Have x) ↔ (∀ (x : Person), ¬Have x → ¬Happy x) :=
by
  sorry

end contrapositive_equivalence_l206_206786


namespace rowing_distance_upstream_l206_206670

theorem rowing_distance_upstream 
  (v : ℝ) (d : ℝ)
  (h1 : 75 = (v + 3) * 5)
  (h2 : d = (v - 3) * 5) :
  d = 45 :=
by {
  sorry
}

end rowing_distance_upstream_l206_206670


namespace angus_total_investment_l206_206680

variable (x T : ℝ)

theorem angus_total_investment (h1 : 0.03 * x + 0.05 * 6000 = 660) (h2 : T = x + 6000) : T = 18000 :=
by
  sorry

end angus_total_investment_l206_206680


namespace team_with_at_least_one_girl_l206_206953

noncomputable def choose (n m : ℕ) := Nat.factorial n / (Nat.factorial m * Nat.factorial (n - m))

theorem team_with_at_least_one_girl (total_boys total_girls select : ℕ) (h_boys : total_boys = 5) (h_girls : total_girls = 5) (h_select : select = 3) :
  (choose (total_boys + total_girls) select) - (choose total_boys select) = 110 := 
by
  sorry

end team_with_at_least_one_girl_l206_206953


namespace jonah_total_ingredients_in_cups_l206_206588

noncomputable def volume_of_ingredients_in_cups : ℝ :=
  let yellow_raisins := 0.3
  let black_raisins := 0.4
  let almonds_in_ounces := 5.5
  let pumpkin_seeds_in_grams := 150
  let ounce_to_cup_conversion := 0.125
  let gram_to_cup_conversion := 0.00423
  let almonds := almonds_in_ounces * ounce_to_cup_conversion
  let pumpkin_seeds := pumpkin_seeds_in_grams * gram_to_cup_conversion
  yellow_raisins + black_raisins + almonds + pumpkin_seeds

theorem jonah_total_ingredients_in_cups : volume_of_ingredients_in_cups = 2.022 :=
by
  sorry

end jonah_total_ingredients_in_cups_l206_206588


namespace M_even_comp_M_composite_comp_M_prime_not_div_l206_206821

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_composite (n : ℕ) : Prop :=  ∃ d : ℕ, d > 1 ∧ d < n ∧ d ∣ n
def M (n : ℕ) : ℕ := 2^n - 1

theorem M_even_comp (n : ℕ) (h1 : n ≠ 2) (h2 : is_even n) : is_composite (M n) :=
sorry

theorem M_composite_comp (n : ℕ) (h : is_composite n) : is_composite (M n) :=
sorry

theorem M_prime_not_div (p : ℕ) (h : Nat.Prime p) : ¬ (p ∣ M p) :=
sorry

end M_even_comp_M_composite_comp_M_prime_not_div_l206_206821


namespace difference_of_place_values_l206_206166

theorem difference_of_place_values :
  let n := 54179759
  let pos1 := 10000 * 7
  let pos2 := 10 * 7
  pos1 - pos2 = 69930 := by
  sorry

end difference_of_place_values_l206_206166


namespace range_of_m_l206_206567

def quadratic_nonnegative (m : ℝ) : Prop :=
∀ x : ℝ, m * x^2 + m * x + 1 ≥ 0

theorem range_of_m (m : ℝ) :
  quadratic_nonnegative m ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end range_of_m_l206_206567


namespace weight_of_each_bag_l206_206994

theorem weight_of_each_bag (empty_weight loaded_weight : ℕ) (number_of_bags : ℕ) (weight_per_bag : ℕ)
    (h1 : empty_weight = 500)
    (h2 : loaded_weight = 1700)
    (h3 : number_of_bags = 20)
    (h4 : loaded_weight - empty_weight = number_of_bags * weight_per_bag) :
    weight_per_bag = 60 :=
by
  sorry

end weight_of_each_bag_l206_206994


namespace monica_total_savings_l206_206943

theorem monica_total_savings :
  ∀ (weekly_saving : ℤ) (weeks_per_cycle : ℤ) (cycles : ℤ),
    weekly_saving = 15 →
    weeks_per_cycle = 60 →
    cycles = 5 →
    weekly_saving * weeks_per_cycle * cycles = 4500 :=
by
  intros weekly_saving weeks_per_cycle cycles
  sorry

end monica_total_savings_l206_206943


namespace fifty_percent_of_x_l206_206092

variable (x : ℝ)

theorem fifty_percent_of_x (h : 0.40 * x = 160) : 0.50 * x = 200 :=
by
  sorry

end fifty_percent_of_x_l206_206092


namespace not_in_range_of_g_l206_206117

def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else undefined

theorem not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g(x) ≠ 0 :=
by sorry

end not_in_range_of_g_l206_206117


namespace member_number_property_l206_206427

theorem member_number_property :
  ∃ (country : Fin 6) (member_number : Fin 1978),
    (∀ (i j : Fin 1978), i ≠ j → member_number ≠ i + j) ∨
    (∀ (k : Fin 1978), member_number ≠ 2 * k) :=
by
  sorry

end member_number_property_l206_206427


namespace angle_A_area_of_triangle_l206_206394

open Real

theorem angle_A (a : ℝ) (A B C : ℝ) 
  (h_a : a = 2 * sqrt 3)
  (h_condition1 : 4 * cos A ^ 2 + 4 * cos B * cos C + 1 = 4 * sin B * sin C) :
  A = π / 3 := 
sorry

theorem area_of_triangle (a b c A : ℝ) 
  (h_A : A = π / 3)
  (h_a : a = 2 * sqrt 3)
  (h_b : b = 3 * c) :
  (1 / 2) * b * c * sin A = 9 * sqrt 3 / 7 := 
sorry

end angle_A_area_of_triangle_l206_206394


namespace prime_square_remainder_l206_206683

theorem prime_square_remainder (p : ℕ) (hp : Nat.Prime p) (h5 : p > 5) : 
  ∃! r : ℕ, r < 180 ∧ (p^2 ≡ r [MOD 180]) := 
by
  sorry

end prime_square_remainder_l206_206683


namespace shifted_function_is_correct_l206_206736

-- Given conditions
def original_function (x : ℝ) : ℝ := -(x + 2) ^ 2 + 1

def shift_right (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x - a)

-- Resulting function after shifting 1 unit to the right
def shifted_function : ℝ → ℝ := shift_right original_function 1

-- Correct answer
def correct_function (x : ℝ) : ℝ := -(x + 1) ^ 2 + 1

-- Proof Statement
theorem shifted_function_is_correct :
  ∀ x : ℝ, shifted_function x = correct_function x := by
  sorry

end shifted_function_is_correct_l206_206736


namespace data_instances_in_one_hour_l206_206746

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l206_206746


namespace marilyn_initial_bottle_caps_l206_206448

theorem marilyn_initial_bottle_caps (x : ℕ) (h : x - 36 = 15) : x = 51 :=
sorry

end marilyn_initial_bottle_caps_l206_206448


namespace distance_A_beats_B_l206_206169

theorem distance_A_beats_B 
  (A_time : ℝ) (A_distance : ℝ) (B_time : ℝ) (B_distance : ℝ)
  (hA : A_distance = 128) (hA_time : A_time = 28)
  (hB : B_distance = 128) (hB_time : B_time = 32) :
  (A_distance - (B_distance * (A_time / B_time))) = 16 :=
by
  sorry

end distance_A_beats_B_l206_206169


namespace seq_a_2014_l206_206430

theorem seq_a_2014 {a : ℕ → ℕ}
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 0 < n → n * a (n + 1) = (n + 1) * a n) :
  a 2014 = 2014 :=
sorry

end seq_a_2014_l206_206430


namespace rainfall_ratio_l206_206573

theorem rainfall_ratio (rain_15_days : ℕ) (total_rain : ℕ) (days_in_month : ℕ) (rain_per_day_first_15 : ℕ) :
  rain_per_day_first_15 * 15 = rain_15_days →
  rain_15_days + (days_in_month - 15) * (rain_per_day_first_15 * 2) = total_rain →
  days_in_month = 30 →
  total_rain = 180 →
  rain_per_day_first_15 = 4 →
  2 = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end rainfall_ratio_l206_206573


namespace combined_length_of_trains_l206_206160

def length_of_train (speed_kmhr : ℕ) (time_sec : ℕ) : ℚ :=
  (speed_kmhr : ℚ) / 3600 * time_sec

theorem combined_length_of_trains :
  let L1 := length_of_train 300 33
  let L2 := length_of_train 250 44
  let L3 := length_of_train 350 28
  L1 + L2 + L3 = 8.52741 := by
  sorry

end combined_length_of_trains_l206_206160


namespace sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l206_206429

-- 1. Sum of the interior angles in a triangle is 180 degrees.
theorem sum_of_angles_in_triangle : ∀ a : ℕ, (∀ x y z : ℕ, x + y + z = 180) → a = 180 := by
  intros a h
  have : a = 180 := sorry
  exact this

-- 2. Sum of interior angles of a regular b-sided polygon is 1080 degrees.
theorem sum_of_angles_in_polygon : ∀ b : ℕ, ((b - 2) * 180 = 1080) → b = 8 := by
  intros b h
  have : b = 8 := sorry
  exact this

-- 3. Exponential equation involving b.
theorem exponential_equation : ∀ p b : ℕ, (8 ^ b = p ^ 21) ∧ (b = 8) → p = 2 := by
  intros p b h
  have : p = 2 := sorry
  exact this

-- 4. Logarithmic equation involving p.
theorem logarithmic_equation : ∀ q p : ℕ, (p = Real.log 81 / Real.log q) ∧ (p = 2) → q = 9 := by
  intros q p h
  have : q = 9 := sorry
  exact this

end sum_of_angles_in_triangle_sum_of_angles_in_polygon_exponential_equation_logarithmic_equation_l206_206429


namespace puppy_food_total_correct_l206_206593

def daily_food_first_two_weeks : ℚ := 3 / 4
def weekly_food_first_two_weeks : ℚ := 7 * daily_food_first_two_weeks
def total_food_first_two_weeks : ℚ := 2 * weekly_food_first_two_weeks

def daily_food_following_two_weeks : ℚ := 1
def weekly_food_following_two_weeks : ℚ := 7 * daily_food_following_two_weeks
def total_food_following_two_weeks : ℚ := 2 * weekly_food_following_two_weeks

def today_food : ℚ := 1 / 2

def total_food_over_4_weeks : ℚ :=
  total_food_first_two_weeks + total_food_following_two_weeks + today_food

theorem puppy_food_total_correct :
  total_food_over_4_weeks = 25 := by
  sorry

end puppy_food_total_correct_l206_206593


namespace find_multiple_l206_206205

theorem find_multiple (n : ℕ) (h₁ : n = 5) (m : ℕ) (h₂ : 7 * n - 15 > m * n) : m = 3 :=
by
  sorry

end find_multiple_l206_206205


namespace smallest_int_remainder_two_l206_206488

theorem smallest_int_remainder_two (m : ℕ) (hm : m > 1)
  (h3 : m % 3 = 2)
  (h4 : m % 4 = 2)
  (h5 : m % 5 = 2)
  (h6 : m % 6 = 2)
  (h7 : m % 7 = 2) :
  m = 422 :=
sorry

end smallest_int_remainder_two_l206_206488


namespace arithmetic_sequence_a2_a9_sum_l206_206923

theorem arithmetic_sequence_a2_a9_sum 
  (a : ℕ → ℝ) (d a₁ : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_S10 : 10 * a 1 + 45 * d = 120) :
  a 2 + a 9 = 24 :=
sorry

end arithmetic_sequence_a2_a9_sum_l206_206923


namespace other_root_of_quadratic_l206_206264

theorem other_root_of_quadratic (a c : ℝ) (h : a ≠ 0) (h_root : 4 * a * 0^2 - 2 * a * 0 + c = 0) :
  ∃ t : ℝ, (4 * a * t^2 - 2 * a * t + c = 0) ∧ t = 1 / 2 :=
by
  sorry

end other_root_of_quadratic_l206_206264


namespace tan_theta_condition_l206_206959

open Real

theorem tan_theta_condition (k : ℤ) : 
  (∃ θ : ℝ, θ = 2 * k * π + π / 4 ∧ tan θ = 1) ∧ ¬ (∀ θ : ℝ, tan θ = 1 → ∃ k : ℤ, θ = 2 * k * π + π / 4) :=
by sorry

end tan_theta_condition_l206_206959


namespace animals_in_field_l206_206990

theorem animals_in_field :
  let dog := 1 in
  let cats := 4 in
  let rabbits_per_cat := 2 in
  let hares_per_rabbit := 3 in
  let total_animals := 
    dog + 
    cats + 
    cats * rabbits_per_cat + 
    (cats * rabbits_per_cat) * hares_per_rabbit 
  in total_animals = 37 := by
sorry

end animals_in_field_l206_206990


namespace friend_spent_seven_l206_206325

/-- You and your friend spent a total of $11 for lunch.
    Your friend spent $3 more than you.
    Prove that your friend spent $7 on their lunch. -/
theorem friend_spent_seven (you friend : ℝ) 
  (h1: you + friend = 11) 
  (h2: friend = you + 3) : 
  friend = 7 := 
by 
  sorry

end friend_spent_seven_l206_206325


namespace vertical_bisecting_line_of_circles_l206_206640

theorem vertical_bisecting_line_of_circles :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 2 * x + 6 * y + 2 = 0 ∨ x^2 + y^2 + 4 * x - 2 * y - 4 = 0) →
  (4 * x + 3 * y + 5 = 0) :=
sorry

end vertical_bisecting_line_of_circles_l206_206640


namespace subset_A_B_l206_206109

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l206_206109


namespace total_tires_l206_206069

def cars := 15
def bicycles := 3
def pickup_trucks := 8
def tricycles := 1

def tires_per_car := 4
def tires_per_bicycle := 2
def tires_per_pickup_truck := 4
def tires_per_tricycle := 3

theorem total_tires : (cars * tires_per_car) + (bicycles * tires_per_bicycle) + (pickup_trucks * tires_per_pickup_truck) + (tricycles * tires_per_tricycle) = 101 :=
by
  sorry

end total_tires_l206_206069


namespace daily_rate_is_three_l206_206667

theorem daily_rate_is_three (r : ℝ) : 
  (∀ (initial bedbugs : ℝ), initial = 30 ∧ 
  (∀ days later_bedbugs, days = 4 ∧ later_bedbugs = 810 →
  later_bedbugs = initial * r ^ days)) → r = 3 :=
by
  intros h
  sorry

end daily_rate_is_three_l206_206667


namespace number_of_positive_divisors_of_60_l206_206900

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l206_206900


namespace factors_of_180_multiple_of_15_l206_206412

/-- A factor of a number n is a positive integer which divides n without leaving a remainder. -/
def is_factor (n k : ℕ) : Prop :=
k > 0 ∧ n % k = 0

/-- A multiple of a number m is an integer which is divisible by m without leaving a remainder. -/
def is_multiple (k m : ℕ) : Prop :=
k % m = 0

/-- There are 6 factors of 180 that are also multiples of 15. -/
theorem factors_of_180_multiple_of_15 : 
  (Finset.filter (λ k, is_multiple k 15) { x | is_factor 180 x }.to_finset).card = 6 :=
by 
  sorry

end factors_of_180_multiple_of_15_l206_206412


namespace cubic_eq_solutions_l206_206782

theorem cubic_eq_solutions (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : ∀ x, x^3 + a * x^2 + b * x + c = 0 → (x = a ∨ x = -b ∨ x = c)) : (a, b, c) = (1, -1, -1) := 
by {
  -- Convert solution steps into a proof
  sorry
}

end cubic_eq_solutions_l206_206782


namespace device_records_720_instances_in_one_hour_l206_206752

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l206_206752


namespace convention_handshakes_l206_206477

-- Introducing the conditions
def companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_reps : ℕ := companies * reps_per_company
def shakes_per_rep : ℕ := total_reps - 1 - (reps_per_company - 1)
def handshakes : ℕ := (total_reps * shakes_per_rep) / 2

-- Statement of the proof
theorem convention_handshakes : handshakes = 160 :=
by
  sorry  -- Proof is not required in this task.

end convention_handshakes_l206_206477


namespace cos_F_in_triangle_l206_206282

theorem cos_F_in_triangle (D E F : ℝ) (sin_D : ℝ) (cos_E : ℝ) (cos_F : ℝ) 
  (h1 : sin_D = 4 / 5) 
  (h2 : cos_E = 12 / 13) 
  (D_plus_E_plus_F : D + E + F = π) :
  cos_F = -16 / 65 :=
by
  sorry

end cos_F_in_triangle_l206_206282


namespace purely_imaginary_condition_l206_206440

theorem purely_imaginary_condition (x : ℝ) :
  (z : ℂ) → (z = (x^2 - 1 : ℂ) + (x + 1 : ℂ) * Complex.I) →
  (x = 1 ↔ (∃ y : ℂ, z = y * Complex.I)) :=
by
  sorry

end purely_imaginary_condition_l206_206440


namespace simplify_expression_l206_206460

theorem simplify_expression :
  (64^(1/3) - 216^(1/3) = -2) :=
by
  have h1 : 64 = 4^3 := by norm_num
  have h2 : 216 = 6^3 := by norm_num
  sorry

end simplify_expression_l206_206460


namespace sufficient_but_not_necessary_condition_l206_206630

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l206_206630


namespace sufficient_but_not_necessary_condition_l206_206629

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 - m * x + 1 > 0) → -2 < m ∧ m < 2 :=
by
  sorry

end sufficient_but_not_necessary_condition_l206_206629


namespace exists_sequence_for_k_l206_206711

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end exists_sequence_for_k_l206_206711


namespace find_constants_l206_206532

theorem find_constants (P Q R : ℚ) (h : ∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 → 
  (x^2 - 13) / ((x - 1) * (x - 4) * (x - 6)) = (P / (x - 1)) + (Q / (x - 4)) + (R / (x - 6))) : 
  (P, Q, R) = (-4/5, -1/2, 23/10) := 
  sorry

end find_constants_l206_206532


namespace parallel_lines_condition_l206_206550

theorem parallel_lines_condition (k_1 k_2 : ℝ) :
  (k_1 = k_2) ↔ (∀ x y : ℝ, k_1 * x + y + 1 = 0 → k_2 * x + y - 1 = 0) :=
sorry

end parallel_lines_condition_l206_206550


namespace value_of_x_squared_plus_y_squared_l206_206562

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l206_206562


namespace alice_catch_up_time_l206_206356

def alice_speed : ℝ := 45
def tom_speed : ℝ := 15
def initial_distance : ℝ := 4
def minutes_per_hour : ℝ := 60

theorem alice_catch_up_time :
  (initial_distance / (alice_speed - tom_speed)) * minutes_per_hour = 8 :=
by
  sorry

end alice_catch_up_time_l206_206356


namespace trigonometric_identity_l206_206859

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l206_206859


namespace certain_amount_l206_206745

theorem certain_amount (x : ℝ) (h1 : 2 * x = 86 - 54) (h2 : 8 + 3 * 8 = 24) (h3 : 86 - 54 + 32 = 86) : x = 43 := 
by {
  sorry
}

end certain_amount_l206_206745


namespace seats_empty_l206_206986

def number_of_people : ℕ := 532
def total_seats : ℕ := 750

theorem seats_empty (n : ℕ) (m : ℕ) : m - n = 218 := by
  have number_of_people : ℕ := 532
  have total_seats : ℕ := 750
  sorry

end seats_empty_l206_206986


namespace avg_annual_growth_rate_l206_206999

variable (x : ℝ)

/-- Initial GDP in 2020 is 43903.89 billion yuan and GDP in 2022 is 53109.85 billion yuan. 
    Prove that the average annual growth rate x satisfies the equation 43903.89 * (1 + x)^2 = 53109.85 -/
theorem avg_annual_growth_rate (x : ℝ) :
  43903.89 * (1 + x)^2 = 53109.85 :=
sorry

end avg_annual_growth_rate_l206_206999


namespace andrei_kolya_ages_l206_206044

def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + ((n / 10) % 10) + ((n / 100) % 10) + (n / 1000)

theorem andrei_kolya_ages :
  ∃ (y1 y2 : ℕ), (sum_of_digits y1 = 2021 - y1) ∧ (sum_of_digits y2 = 2021 - y2) ∧ (y1 ≠ y2) ∧ ((2022 - y1 = 8 ∧ 2022 - y2 = 26) ∨ (2022 - y1 = 26 ∧ 2022 - y2 = 8)) :=
by
  sorry

end andrei_kolya_ages_l206_206044


namespace arithmetic_sequence_a3_l206_206725

variable (a : ℕ → ℕ)
variable (S5 : ℕ)
variable (arithmetic_seq : Prop)

def is_arithmetic_seq (a : ℕ → ℕ) : Prop := ∀ n, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_a3 (h1 : is_arithmetic_seq a) (h2 : (a 1 + a 2 + a 3 + a 4 + a 5) = 25) : a 3 = 5 :=
by
  sorry

end arithmetic_sequence_a3_l206_206725


namespace x_plus_2y_equals_2_l206_206091

theorem x_plus_2y_equals_2 (x y : ℝ) (h : |x + 3| + (2 * y - 5)^2 = 0) : x + 2 * y = 2 := 
sorry

end x_plus_2y_equals_2_l206_206091


namespace average_income_eq_58_l206_206825

def income_day1 : ℕ := 45
def income_day2 : ℕ := 50
def income_day3 : ℕ := 60
def income_day4 : ℕ := 65
def income_day5 : ℕ := 70
def number_of_days : ℕ := 5

theorem average_income_eq_58 :
  (income_day1 + income_day2 + income_day3 + income_day4 + income_day5) / number_of_days = 58 := by
  sorry

end average_income_eq_58_l206_206825


namespace probability_two_girls_given_one_l206_206171

/-- Let A be the event "both children are girls" and B the event "at least one of the children is a girl". 
    Given the conditions, prove that the probability of A given B is 1/3. -/
theorem probability_two_girls_given_one:
  (P : Set (Set (fin 2))) 
  (A : Event P := {s | s = {0, 0}})
  (B : Event P := {s | s ∈ {1, 0} ∨ s ∈ {0, 1} ∨ s ∈ {0, 0}}):
  conditionalProb P A B = 1 / 3 :=
by
  sorry

end probability_two_girls_given_one_l206_206171


namespace subset_a_eq_1_l206_206112

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_a_eq_1 (a : ℝ) (h : A a ⊆ B a) : a = 1 :=
by
  sorry

end subset_a_eq_1_l206_206112


namespace range_of_x_l206_206147

theorem range_of_x (x : ℝ) : (6 - 2 * x) ≠ 0 ↔ x ≠ 3 := 
by {
  sorry
}

end range_of_x_l206_206147


namespace third_median_length_l206_206639

theorem third_median_length (a b: ℝ) (h_a: a = 5) (h_b: b = 8)
  (area: ℝ) (h_area: area = 6 * Real.sqrt 15) (m: ℝ):
  m = 3 * Real.sqrt 6 :=
sorry

end third_median_length_l206_206639


namespace possible_values_of_a_l206_206384

variable (a : ℝ)
def A : Set ℝ := { x | x^2 ≠ 1 }
def B (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem possible_values_of_a (h : (A ∪ B a) = A) : a = 1 ∨ a = -1 ∨ a = 0 :=
by
  sorry

end possible_values_of_a_l206_206384


namespace max_horizontal_distance_domino_l206_206343

theorem max_horizontal_distance_domino (n : ℕ) : 
    (n > 0) → ∃ d, d = 2 * Real.log n := 
by {
    sorry
}

end max_horizontal_distance_domino_l206_206343


namespace alyosha_cube_problem_l206_206029

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l206_206029


namespace train_cars_estimate_l206_206046

noncomputable def train_cars_count (total_time_secs : ℕ) (delay_secs : ℕ) (cars_counted : ℕ) (count_time_secs : ℕ): ℕ := 
  let rate_per_sec := cars_counted / count_time_secs
  let cars_missed := delay_secs * rate_per_sec
  let cars_in_remaining_time := rate_per_sec * (total_time_secs - delay_secs)
  cars_missed + cars_in_remaining_time

theorem train_cars_estimate :
  train_cars_count 210 15 8 20 = 120 :=
sorry

end train_cars_estimate_l206_206046


namespace cos_pi_plus_2alpha_l206_206382

-- Define the main theorem using the given condition and the result to be proven
theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 3) : Real.cos (π + 2 * α) = 7 / 9 :=
sorry

end cos_pi_plus_2alpha_l206_206382


namespace same_remainder_division_l206_206136

theorem same_remainder_division {a m b : ℤ} (r c k : ℤ) 
  (ha : a = b * c + r) (hm : m = b * k + r) : b ∣ (a - m) :=
by
  sorry

end same_remainder_division_l206_206136


namespace number_of_divisors_60_l206_206904

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l206_206904


namespace dot_product_a_b_l206_206254

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, 2)

theorem dot_product_a_b : a.1 * b.1 + a.2 * b.2 = 4 := by
  sorry

end dot_product_a_b_l206_206254


namespace jake_comic_books_l206_206433

variables (J : ℕ)

def brother_comic_books := J + 15
def total_comic_books := J + brother_comic_books

theorem jake_comic_books : total_comic_books = 87 → J = 36 :=
by
  sorry

end jake_comic_books_l206_206433


namespace men_absent_is_5_l206_206182

-- Define the given conditions
def original_number_of_men : ℕ := 30
def planned_days : ℕ := 10
def actual_days : ℕ := 12

-- Prove the number of men absent (x) is 5, under given conditions
theorem men_absent_is_5 : ∃ x : ℕ, 30 * planned_days = (original_number_of_men - x) * actual_days ∧ x = 5 :=
by
  sorry

end men_absent_is_5_l206_206182


namespace notebook_width_l206_206323

theorem notebook_width
  (circumference : ℕ)
  (length : ℕ)
  (width : ℕ)
  (H1 : circumference = 46)
  (H2 : length = 9)
  (H3 : circumference = 2 * (length + width)) :
  width = 14 :=
by
  sorry -- proof is omitted

end notebook_width_l206_206323


namespace smaller_angle_at_3_45_l206_206686

def minute_hand_angle : ℝ := 270
def hour_hand_angle : ℝ := 90 + 0.75 * 30

theorem smaller_angle_at_3_45 :
  min (|minute_hand_angle - hour_hand_angle|) (360 - |minute_hand_angle - hour_hand_angle|) = 202.5 := 
by
  sorry

end smaller_angle_at_3_45_l206_206686


namespace combined_yellow_blue_correct_l206_206766

-- Declare the number of students in the class
def total_students : ℕ := 200

-- Declare the percentage of students who like blue
def percent_like_blue : ℝ := 0.3

-- Declare the percentage of remaining students who like red
def percent_like_red : ℝ := 0.4

-- Function that calculates the number of students liking a certain color based on percentage
def students_like_color (total : ℕ) (percent : ℝ) : ℕ :=
  (percent * total).toInt

-- Calculate the number of students who like blue
def students_like_blue : ℕ :=
  students_like_color total_students percent_like_blue

-- Calculate the number of students who don't like blue
def students_not_like_blue : ℕ :=
  total_students - students_like_blue

-- Calculate the number of students who like red from those who don't like blue
def students_like_red : ℕ :=
  students_like_color students_not_like_blue percent_like_red

-- Calculate the number of students who like yellow (those who don't like blue or red)
def students_like_yellow : ℕ :=
  students_not_like_blue - students_like_red

-- The combined number of students who like yellow and blue
def combined_yellow_blue : ℕ :=
  students_like_blue + students_like_yellow

-- Theorem to prove that the combined number of students liking yellow and blue is 144
theorem combined_yellow_blue_correct : combined_yellow_blue = 144 := by
  sorry

end combined_yellow_blue_correct_l206_206766


namespace smallest_region_area_l206_206701

-- Define the basic elements
def line (x : ℝ) : ℝ := x + 1
def circle_radius : ℝ := 3
def circle (x y : ℝ) : Prop := x^2 + y^2 = circle_radius^2

-- Define the function to find the area of the smaller region
def area_of_smaller_region : ℝ :=
  (9 / 2) * Real.arcsin(1 / 3) - 1.5

-- Define the proof problem
theorem smallest_region_area :
  ∃ A : ℝ, (A = area_of_smaller_region) ∧ (∀ x y : ℝ, circle x y ∧ y = line x → 
  A = (9 / 2) * Real.arcsin(1 / 3) - 1.5) :=
by
  exists (9 / 2 * Real.arcsin (1 / 3) - 1.5)
  split
  { rfl }
  { intros x y h
    unfold area_of_smaller_region
    cases h 
    sorry
  }

end smallest_region_area_l206_206701


namespace total_wheels_l206_206070

def cars : Nat := 15
def bicycles : Nat := 3
def trucks : Nat := 8
def tricycles : Nat := 1
def wheels_per_car_or_truck : Nat := 4
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3

theorem total_wheels : cars * wheels_per_car_or_truck + trucks * wheels_per_car_or_truck + bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 101 :=
by
  sorry

end total_wheels_l206_206070


namespace arrangements_count_l206_206625

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of positions
def num_positions : ℕ := 3

-- Define a type for the students
inductive Student
| A | B | C | D | E

-- Define the positions
inductive Position
| athletics | swimming | ball_games

-- Constraint: student A cannot be the swimming volunteer
def cannot_be_swimming_volunteer (s : Student) (p : Position) : Prop :=
  (s = Student.A → p ≠ Position.swimming)

-- Define the function to count the arrangements given the constraints
noncomputable def count_arrangements : ℕ :=
  (num_students.choose num_positions) - 1 -- Placeholder for the actual count based on given conditions

-- The theorem statement
theorem arrangements_count : count_arrangements = 16 :=
by
  sorry

end arrangements_count_l206_206625


namespace average_score_l206_206186

theorem average_score (N : ℕ) (p3 p2 p1 p0 : ℕ) (n : ℕ) 
  (H1 : N = 3)
  (H2 : p3 = 30)
  (H3 : p2 = 50)
  (H4 : p1 = 10)
  (H5 : p0 = 10)
  (H6 : n = 20)
  (H7 : p3 + p2 + p1 + p0 = 100) :
  (3 * (p3 * n / 100) + 2 * (p2 * n / 100) + 1 * (p1 * n / 100) + 0 * (p0 * n / 100)) / n = 2 :=
by 
  sorry

end average_score_l206_206186


namespace h_has_only_one_zero_C2_below_C1_l206_206539

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := 1 - 1/x
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem h_has_only_one_zero (x : ℝ) (hx : x > 0) : 
  ∃! (x0 : ℝ), x0 > 0 ∧ h x0 = 0 := sorry

theorem C2_below_C1 (x : ℝ) (hx : x > 0) (hx1 : x ≠ 1) : 
  g x < f x := sorry

end h_has_only_one_zero_C2_below_C1_l206_206539


namespace probability_of_different_cousins_name_l206_206040

theorem probability_of_different_cousins_name :
  let total_letters := 19
  let amelia_letters := 6
  let bethany_letters := 7
  let claire_letters := 6
  let probability := 
    2 * ((amelia_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)) +
         (amelia_letters / (total_letters : ℚ)) * (claire_letters / (total_letters - 1 : ℚ)) +
         (claire_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)))
  probability = 40 / 57 := sorry

end probability_of_different_cousins_name_l206_206040


namespace total_smaller_cubes_is_27_l206_206503

-- Given conditions
def painted_red (n : ℕ) : Prop := ∀ face, face ∈ cube.faces → face.color = red

def cut_into_smaller_cubes (n : ℕ) : Prop := ∃ k : ℕ, k = n + 1

def smaller_cubes_painted_on_2_faces (cubes_painted_on_2_faces : ℕ) (n : ℕ) : Prop :=
  cubes_painted_on_2_faces = 12 * (n - 1)

-- Question: Prove the total number of smaller cubes is equal to 27, given the conditions
theorem total_smaller_cubes_is_27 (n : ℕ) (h1 : painted_red n) (h2 : cut_into_smaller_cubes n) (h3 : smaller_cubes_painted_on_2_faces 12 n) :
  (n + 1)^3 = 27 := by
  sorry

end total_smaller_cubes_is_27_l206_206503


namespace smaller_cubes_total_l206_206502

theorem smaller_cubes_total (n : ℕ) (painted_edges_cubes : ℕ) 
  (h1 : ∀ (a b : ℕ), a ^ 3 = n) 
  (h2 : ∀ (c : ℕ), painted_edges_cubes = 12) 
  (h3 : ∀ (d e : ℕ), 12 <= 2 * d * e) 
  : n = 27 :=
by
  sorry

end smaller_cubes_total_l206_206502


namespace reconstruct_quadrilateral_l206_206221

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A A'' B'' C'' D'' : V)

def trisect_segment (P Q R : V) : Prop :=
  Q = (1 / 3 : ℝ) • P + (2 / 3 : ℝ) • R

theorem reconstruct_quadrilateral
  (hB : trisect_segment A B A'')
  (hC : trisect_segment B C B'')
  (hD : trisect_segment C D C'')
  (hA : trisect_segment D A D'') :
  A = (2 / 26) • A'' + (6 / 26) • B'' + (6 / 26) • C'' + (12 / 26) • D'' :=
sorry

end reconstruct_quadrilateral_l206_206221


namespace lowest_temperature_l206_206306

theorem lowest_temperature 
  (temps : Fin 5 → ℝ) 
  (avg_temp : (temps 0 + temps 1 + temps 2 + temps 3 + temps 4) / 5 = 60)
  (max_range : ∀ i j, temps i - temps j ≤ 75) : 
  ∃ L : ℝ, L = 0 ∧ ∃ i, temps i = L :=
by 
  sorry

end lowest_temperature_l206_206306


namespace jerry_age_l206_206292

variable (M J : ℕ) -- Declare Mickey's and Jerry's ages as natural numbers

-- Define the conditions as hypotheses
def condition1 := M = 2 * J - 6
def condition2 := M = 18

-- Theorem statement where we need to prove J = 12 given the conditions
theorem jerry_age
  (h1 : condition1 M J)
  (h2 : condition2 M) :
  J = 12 :=
sorry

end jerry_age_l206_206292


namespace algebraic_expression_value_l206_206073

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = 9) : a^2 - 3 * a * b + b^2 = 19 :=
sorry

end algebraic_expression_value_l206_206073


namespace trapezium_distance_l206_206377

variable (a b h : ℝ)

theorem trapezium_distance (h_pos : 0 < h) (a_pos : 0 < a) (b_pos : 0 < b)
  (area_eq : 270 = 1/2 * (a + b) * h) (a_eq : a = 20) (b_eq : b = 16) : h = 15 :=
by {
  sorry
}

end trapezium_distance_l206_206377


namespace trigonometric_identity_l206_206858

theorem trigonometric_identity :
  (let cos30 : ℝ := (Real.sqrt 3) / 2
   let sin60 : ℝ := (Real.sqrt 3) / 2
   let sin30 : ℝ := 1 / 2
   let cos60 : ℝ := 1 / 2
   (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1) :=
by
  sorry

end trigonometric_identity_l206_206858


namespace kanul_initial_amount_l206_206933

theorem kanul_initial_amount (X Y : ℝ) (loan : ℝ) (R : ℝ) 
  (h1 : loan = 2000)
  (h2 : R = 0.20)
  (h3 : Y = 0.15 * X + loan)
  (h4 : loan = R * Y) : 
  X = 53333.33 :=
by 
  -- The proof would come here, but is not necessary for this example
sorry

end kanul_initial_amount_l206_206933


namespace frank_initial_mushrooms_l206_206713

theorem frank_initial_mushrooms (pounds_eaten pounds_left initial_pounds : ℕ) 
  (h1 : pounds_eaten = 8) 
  (h2 : pounds_left = 7) 
  (h3 : initial_pounds = pounds_eaten + pounds_left) : 
  initial_pounds = 15 := 
by 
  sorry

end frank_initial_mushrooms_l206_206713


namespace no_distinct_roots_exist_l206_206371

theorem no_distinct_roots_exist :
  ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a^2 - 2 * b * a + c^2 = 0) ∧
  (b^2 - 2 * c * b + a^2 = 0) ∧ 
  (c^2 - 2 * a * c + b^2 = 0) := 
sorry

end no_distinct_roots_exist_l206_206371


namespace quadratic_eq_m_neg1_l206_206422

theorem quadratic_eq_m_neg1 (m : ℝ) (h1 : (m - 3) ≠ 0) (h2 : m^2 - 2*m - 3 = 0) : m = -1 :=
sorry

end quadratic_eq_m_neg1_l206_206422


namespace simplify_expression_l206_206810

theorem simplify_expression : 
  -2^2003 + (-2)^2004 + 2^2005 - 2^2006 = -3 * 2^2003 := 
  by
    sorry

end simplify_expression_l206_206810


namespace sample_size_divided_into_six_groups_l206_206873

theorem sample_size_divided_into_six_groups
  (n : ℕ)
  (c1 c2 c3 : ℕ)
  (k : ℚ)
  (h1 : c1 + c2 + c3 = 36)
  (h2 : 20 * k = 1)
  (h3 : 2 * k * n = c1)
  (h4 : 3 * k * n = c2)
  (h5 : 4 * k * n = c3) :
  n = 80 :=
by
  sorry

end sample_size_divided_into_six_groups_l206_206873


namespace sequence_sum_l206_206431

theorem sequence_sum (P Q R S T U V : ℕ) (h1 : S = 7)
  (h2 : P + Q + R = 21) (h3 : Q + R + S = 21)
  (h4 : R + S + T = 21) (h5 : S + T + U = 21)
  (h6 : T + U + V = 21) : P + V = 14 :=
by
  sorry

end sequence_sum_l206_206431


namespace find_side_length_l206_206279

noncomputable def cos (x : ℝ) := Real.cos x

theorem find_side_length
  (A : ℝ) (c : ℝ) (b : ℝ) (a : ℝ)
  (hA : A = Real.pi / 3)
  (hc : c = Real.sqrt 3)
  (hb : b = 2 * Real.sqrt 3) :
  a = 3 := 
sorry

end find_side_length_l206_206279


namespace average_k_positive_int_roots_l206_206231

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l206_206231


namespace steel_strength_value_l206_206874

theorem steel_strength_value 
  (s : ℝ) 
  (condition: s = 4.6 * 10^8) : 
  s = 460000000 := 
by sorry

end steel_strength_value_l206_206874


namespace inequality_proof_l206_206589

variables {Ω : Type*} [ProbabilitySpace Ω]
variable (X : Ω → ℝ)

-- Given conditions
axiom exp_X_zero : E[X] = 0
axiom prob_X_le_one : ∀ ω, X ω ≤ 1

theorem inequality_proof :
  (E[|X|])^2 / 2 ≤ - E[λ ω, Real.log (1 - X ω)] :=
by sorry

end inequality_proof_l206_206589


namespace trigonometric_identity_eq_neg_one_l206_206853

theorem trigonometric_identity_eq_neg_one :
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180)) = -1 :=
by
  -- Variables needed for hypotheses
  have h₁ : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₂ : Real.sin (60 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₃ : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h₄ : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  -- Main proof
  sorry

end trigonometric_identity_eq_neg_one_l206_206853


namespace no_nat_fun_satisfying_property_l206_206370

theorem no_nat_fun_satisfying_property :
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 :=
by
  sorry

end no_nat_fun_satisfying_property_l206_206370


namespace exist_polynomials_unique_polynomials_l206_206118

-- Problem statement: the function 'f'
variable (f : ℝ → ℝ → ℝ → ℝ)

-- Condition: f(w, w, w) = 0 for all w ∈ ℝ
axiom f_ww_ww_ww (w : ℝ) : f w w w = 0

-- Statement for existence of A, B, C
theorem exist_polynomials (f : ℝ → ℝ → ℝ → ℝ)
  (hf : ∀ w : ℝ, f w w w = 0) : 
  ∃ A B C : ℝ → ℝ → ℝ → ℝ, 
  (∀ w : ℝ, A w w w + B w w w + C w w w = 0) ∧ 
  ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x) :=
sorry

-- Statement for uniqueness of A, B, C
theorem unique_polynomials (f : ℝ → ℝ → ℝ → ℝ) 
  (A B C A' B' C' : ℝ → ℝ → ℝ → ℝ)
  (hf: ∀ w : ℝ, f w w w = 0)
  (h1 : ∀ w : ℝ, A w w w + B w w w + C w w w = 0)
  (h2 : ∀ x y z : ℝ, f x y z = A x y z * (x - y) + B x y z * (y - z) + C x y z * (z - x))
  (h3 : ∀ w : ℝ, A' w w w + B' w w w + C' w w w = 0)
  (h4 : ∀ x y z : ℝ, f x y z = A' x y z * (x - y) + B' x y z * (y - z) + C' x y z * (z - x)) : 
  A = A' ∧ B = B' ∧ C = C' :=
sorry

end exist_polynomials_unique_polynomials_l206_206118


namespace ambiguous_dates_in_year_l206_206926

def is_ambiguous_date (m d : ℕ) : Prop :=
  1 ≤ m ∧ m ≤ 12 ∧ 1 ≤ d ∧ d ≤ 12 ∧ m ≠ d

theorem ambiguous_dates_in_year :
  ∃ n : ℕ, n = 132 ∧ (∀ m d : ℕ, is_ambiguous_date m d → n = 132) :=
sorry

end ambiguous_dates_in_year_l206_206926


namespace solve_for_z_l206_206558

variable (x y z : ℝ)

theorem solve_for_z (h : 1 / x - 1 / y = 1 / z) : z = x * y / (y - x) := 
sorry

end solve_for_z_l206_206558


namespace factory_car_production_l206_206349

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l206_206349


namespace subset_a_eq_1_l206_206104

theorem subset_a_eq_1 (a : ℝ) (A : set ℝ) (B : set ℝ) :
  A = {0, -a} ∧ B = {1, a-2, 2*a-2} ∧ A ⊆ B → a = 1 :=
by
  sorry

end subset_a_eq_1_l206_206104


namespace john_needs_one_plank_l206_206215

theorem john_needs_one_plank (total_nails : ℕ) (nails_per_plank : ℕ) (extra_nails : ℕ) (P : ℕ)
    (h1 : total_nails = 11)
    (h2 : nails_per_plank = 3)
    (h3 : extra_nails = 8)
    (h4 : total_nails = nails_per_plank * P + extra_nails) :
    P = 1 :=
by
    sorry

end john_needs_one_plank_l206_206215


namespace solve_equation_l206_206791

theorem solve_equation : ∀ x : ℝ, x ≠ -2 → x ≠ 0 → (3 / (x + 2) - 1 / x = 0 ↔ x = 1) :=
by
  intro x h1 h2
  sorry

end solve_equation_l206_206791


namespace find_n_l206_206007

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l206_206007


namespace non_juniors_play_instrument_l206_206425

theorem non_juniors_play_instrument (total_students juniors non_juniors play_instrument_juniors play_instrument_non_juniors total_do_not_play : ℝ) :
  total_students = 600 →
  play_instrument_juniors = 0.3 * juniors →
  play_instrument_non_juniors = 0.65 * non_juniors →
  total_do_not_play = 0.4 * total_students →
  0.7 * juniors + 0.35 * non_juniors = total_do_not_play →
  juniors + non_juniors = total_students →
  non_juniors * 0.65 = 334 :=
by
  sorry

end non_juniors_play_instrument_l206_206425


namespace factors_of_180_l206_206403

-- Define a predicate to check if a number is a factor of another number
def is_factor (m n : ℕ) : Prop := n % m = 0

-- Define a predicate to check if a number is a multiple of another number
def is_multiple (m n : ℕ) : Prop := m % n = 0

-- Define the problem
theorem factors_of_180 : {n : ℕ // is_factor n 180 ∧ is_multiple n 15}.card = 6 :=
by sorry

end factors_of_180_l206_206403


namespace gcd_420_135_l206_206808

theorem gcd_420_135 : Nat.gcd 420 135 = 15 := by
  sorry

end gcd_420_135_l206_206808


namespace calculate_correctly_l206_206743

theorem calculate_correctly (n : ℕ) (h1 : n - 21 = 52) : n - 40 = 33 := 
by 
  sorry

end calculate_correctly_l206_206743


namespace average_k_l206_206242

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l206_206242


namespace dog_grouping_l206_206784

theorem dog_grouping : 
  let dogs := 12 in
  let group1_size := 4 in
  let group2_size := 6 in
  let group3_size := 2 in
  let Fluffy := "Fluffy" in
  let Nipper := "Nipper" in
  let remaining_dogs := dogs - 2 in
  nat.choose remaining_dogs (group1_size - 1) * nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1) = 2520 :=
by
  sorry

end dog_grouping_l206_206784


namespace closest_point_on_line_l206_206212

theorem closest_point_on_line (x y : ℝ) (h : y = (x - 3) / 3) : 
  (∃ p : ℝ × ℝ, p = (4, -2) ∧ ∀ q : ℝ × ℝ, (q.1, q.2) = (x, y) ∧ q ≠ p → dist p q ≥ dist p (33/10, 1/10)) :=
sorry

end closest_point_on_line_l206_206212


namespace multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l206_206103

def y : ℕ := 48 + 72 + 144 + 192 + 336 + 384 + 3072

theorem multiple_of_4 : ∃ k : ℕ, y = 4 * k := by
  sorry

theorem multiple_of_8 : ∃ k : ℕ, y = 8 * k := by
  sorry

theorem not_multiple_of_16 : ¬ ∃ k : ℕ, y = 16 * k := by
  sorry

theorem multiple_of_24 : ∃ k : ℕ, y = 24 * k := by
  sorry

end multiple_of_4_multiple_of_8_not_multiple_of_16_multiple_of_24_l206_206103


namespace pair_with_15_is_47_l206_206966

theorem pair_with_15_is_47 (numbers : Set ℕ) (k : ℕ) 
  (h : numbers = {49, 29, 9, 40, 22, 15, 53, 33, 13, 47}) 
  (pair_sum_eq_k : ∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → (a, b) ≠ (15, 15) → a + b = k) : 
  ∃ (k : ℕ), 15 + 47 = k := 
sorry

end pair_with_15_is_47_l206_206966


namespace solve_AlyoshaCube_l206_206024

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l206_206024


namespace data_instances_in_one_hour_l206_206747

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l206_206747


namespace second_assistant_smoked_pipes_l206_206616

theorem second_assistant_smoked_pipes
    (x y z : ℚ)
    (H1 : (2 / 3) * x = (4 / 9) * y)
    (H2 : x + y = 1)
    (H3 : (x + z) / (y - z) = y / x) :
    z = 1 / 5 → x = 2 / 5 ∧ y = 3 / 5 →
    ∀ n : ℕ, n = 5 :=
by
  sorry

end second_assistant_smoked_pipes_l206_206616


namespace smaller_side_of_new_rectangle_is_10_l206_206360

/-- We have a 10x25 rectangle that is divided into two congruent polygons and rearranged 
to form another rectangle. We need to prove that the length of the smaller side of the 
resulting rectangle is 10. -/
theorem smaller_side_of_new_rectangle_is_10 :
  ∃ (y x : ℕ), (y * x = 10 * 25) ∧ (y ≤ x) ∧ y = 10 := 
sorry

end smaller_side_of_new_rectangle_is_10_l206_206360


namespace element_subset_a_l206_206106

theorem element_subset_a (a : ℝ) (A B : set ℝ) (hA : A = {0, -a}) (hB : B = {1, a-2, 2a-2}) (h : A ⊆ B) : a = 1 :=
by
  sorry

end element_subset_a_l206_206106


namespace integer_satisfies_mod_and_range_l206_206641

theorem integer_satisfies_mod_and_range :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-150 ≡ n [ZMOD 25]) → n = 0 :=
by
  sorry

end integer_satisfies_mod_and_range_l206_206641


namespace cube_decomposition_l206_206017

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l206_206017


namespace gcd_48_72_120_l206_206702

theorem gcd_48_72_120 : Nat.gcd (Nat.gcd 48 72) 120 = 24 :=
by
  sorry

end gcd_48_72_120_l206_206702


namespace vertex_of_parabola_l206_206975

theorem vertex_of_parabola :
  ∃ (x y : ℝ), y^2 - 8*x + 6*y + 17 = 0 ∧ (x, y) = (1, -3) :=
by
  use 1, -3
  sorry

end vertex_of_parabola_l206_206975


namespace avg_k_value_l206_206237

theorem avg_k_value (k : ℕ) :
  (∃ r1 r2 : ℕ, r1 * r2 = 24 ∧ r1 + r2 = k ∧ 0 < r1 ∧ 0 < r2) →
  k ∈ {25, 14, 11, 10} →
  (25 + 14 + 11 + 10) / 4 = 15 :=
by
  intros _ k_values
  have h : {25, 14, 11, 10}.sum = 60 := by decide 
  have : finset.card {25, 14, 11, 10} = 4 := by decide
  simp [k_values, h, this, nat.cast_div, nat.cast_bit0, nat.cast_succ]
  norm_num

end avg_k_value_l206_206237


namespace num_divisors_60_l206_206901

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l206_206901


namespace otimes_example_l206_206872

def otimes (a b : ℤ) : ℤ := a^2 - a * b

theorem otimes_example : otimes 4 (otimes 2 (-5)) = -40 := by
  sorry

end otimes_example_l206_206872


namespace find_smallest_integer_y_l206_206487

theorem find_smallest_integer_y : ∃ y : ℤ, (8 / 12 : ℚ) < (y / 15) ∧ ∀ z : ℤ, z < y → ¬ ((8 / 12 : ℚ) < (z / 15)) :=
by
  sorry

end find_smallest_integer_y_l206_206487


namespace probability_two_green_apples_l206_206434

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_green_apples :
  ∀ (total_apples green_apples choose_apples : ℕ),
    total_apples = 7 →
    green_apples = 3 →
    choose_apples = 2 →
    (binom green_apples choose_apples : ℝ) / binom total_apples choose_apples = 1 / 7 :=
by
  intro total_apples green_apples choose_apples
  intro h_total h_green h_choose
  rw [h_total, h_green, h_choose]
  -- The proof would go here
  sorry

end probability_two_green_apples_l206_206434


namespace count_factors_of_180_multiple_of_15_l206_206410

def is_factor (x n : ℕ) := n % x = 0 -- Definition of a factor
def is_multiple (x k : ℕ) := x % k = 0 -- Definition of a multiple

theorem count_factors_of_180_multiple_of_15 :
  (finset.filter (λ x, is_multiple x 15) (finset.filter (λ x, is_factor x 180) (finset.range (180 + 1)))).card = 6 := by
  sorry

end count_factors_of_180_multiple_of_15_l206_206410


namespace tunnel_length_l206_206658

-- Definitions as per the conditions
def train_length : ℚ := 2  -- 2 miles
def train_speed : ℚ := 40  -- 40 miles per hour

def speed_in_miles_per_minute (speed_mph : ℚ) : ℚ :=
  speed_mph / 60  -- Convert speed from miles per hour to miles per minute

def time_travelled_in_minutes : ℚ := 5  -- 5 minutes

-- Theorem statement to prove the length of the tunnel
theorem tunnel_length (h1 : train_length = 2) (h2 : train_speed = 40) :
  (speed_in_miles_per_minute train_speed * time_travelled_in_minutes) - train_length = 4 / 3 :=
by
  sorry  -- Proof not included

end tunnel_length_l206_206658


namespace smallest_a_l206_206705

theorem smallest_a (a : ℕ) (h_a : a > 8) : (∀ x : ℤ, ¬ Prime (x^4 + a^2)) ↔ a = 9 :=
by
  sorry

end smallest_a_l206_206705


namespace rachel_one_hour_earnings_l206_206604

theorem rachel_one_hour_earnings :
  let hourly_wage := 12.00
  let number_of_people_served := 20
  let tip_per_person := 1.25
  let total_tips := number_of_people_served * tip_per_person
  let total_earnings := hourly_wage + total_tips
  in total_earnings = 37.00 :=
by
  sorry

end rachel_one_hour_earnings_l206_206604


namespace plantable_area_l206_206664

noncomputable def flowerbed_r := 10
noncomputable def path_w := 4
noncomputable def full_area := 100 * Real.pi
noncomputable def segment_area := 20.67 * Real.pi * 2 -- each path affects two segments

theorem plantable_area :
  full_area - segment_area = 58.66 * Real.pi := 
by sorry

end plantable_area_l206_206664


namespace find_k_l206_206180

-- Defining the conditions
variable {k : ℕ} -- k is a non-negative integer

-- Given conditions as definitions
def green_balls := 7
def purple_balls := k
def total_balls := green_balls + purple_balls
def win_amount := 3
def lose_amount := -1

-- Defining the expected value equation
def expected_value [fact (total_balls > 0)] : ℝ :=
  (green_balls.toℝ / total_balls.toℝ * win_amount) +
  (purple_balls.toℝ / total_balls.toℝ * lose_amount)

-- The required theorem/assertion to prove
theorem find_k (hk : k > 0) (h : expected_value = 1) : k = 7 :=
sorry

end find_k_l206_206180


namespace tangent_range_of_values_for_k_l206_206993

def circle (k : ℝ) : ℝ × ℝ → Prop := λ (p : ℝ × ℝ), (p.1)^2 + (p.2)^2 + 2*k*p.1 + 4*p.2 + 3*k + 8 = 0

def line_through (p : ℝ × ℝ) (m : ℝ) : ℝ × ℝ → Prop := λ (l : ℝ × ℝ), l.2 = m * (l.1 + 1)

theorem tangent_range_of_values_for_k :
  {k : ℝ | ∃ m : ℝ, ∀ (x y : ℝ), line_through (-1, 0) m (x, y) → circle k (x, y)} = {k : ℝ | (-9 < k ∧ k < -1) ∨ (4 < k)} :=
sorry

end tangent_range_of_values_for_k_l206_206993


namespace solve_AlyoshaCube_l206_206023

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l206_206023


namespace prime_square_remainders_l206_206684

theorem prime_square_remainders (p : ℕ) (hp : Prime p) (hpg : p > 5) : 
  ∃ n : ℕ, n = 2 ∧ ∀ r ∈ {r : ℕ | ∃ p : ℕ, Prime p ∧ p > 5 ∧ r = (p ^ 2 % 180)}, r ∈ {1, 64} :=
by sorry

end prime_square_remainders_l206_206684


namespace decreased_area_of_equilateral_triangle_l206_206043

theorem decreased_area_of_equilateral_triangle 
    (A : ℝ) (hA : A = 100 * Real.sqrt 3) 
    (decrease : ℝ) (hdecrease : decrease = 6) :
    let s := Real.sqrt (4 * A / Real.sqrt 3)
    let s' := s - decrease
    let A' := (s' ^ 2 * Real.sqrt 3) / 4
    A - A' = 51 * Real.sqrt 3 :=
by
  sorry

end decreased_area_of_equilateral_triangle_l206_206043


namespace range_of_k_condition_l206_206296

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := (4 - k) / x

theorem range_of_k_condition (k x1 x2 y1 y2 : ℝ) 
    (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 < y2) 
    (hA : inverse_proportion_function k x1 = y1) 
    (hB : inverse_proportion_function k x2 = y2) : 
    k < 4 :=
sorry

end range_of_k_condition_l206_206296


namespace division_of_powers_l206_206814

variable {a : ℝ}

theorem division_of_powers (ha : a ≠ 0) : a^5 / a^3 = a^2 :=
by sorry

end division_of_powers_l206_206814


namespace total_weight_of_family_l206_206963

theorem total_weight_of_family (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 40) : M + D + C = 160 :=
sorry

end total_weight_of_family_l206_206963


namespace device_records_720_instances_in_one_hour_l206_206753

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l206_206753


namespace area_inside_C_outside_A_B_l206_206050

/-- Define the radii of circles A, B, and C --/
def radius_A : ℝ := 1
def radius_B : ℝ := 1
def radius_C : ℝ := 2

/-- Define the condition of tangency and overlap --/
def circles_tangent_at_one_point (r1 r2 : ℝ) : Prop :=
  r1 = r2 

def circle_C_tangent_to_A_B (rA rB rC : ℝ) : Prop :=
  rA = 1 ∧ rB = 1 ∧ rC = 2 ∧ circles_tangent_at_one_point rA rB

/-- Statement to be proved: The area inside circle C but outside circles A and B is 2π --/
theorem area_inside_C_outside_A_B (h : circle_C_tangent_to_A_B radius_A radius_B radius_C) : 
  π * radius_C^2 - π * (radius_A^2 + radius_B^2) = 2 * π :=
by
  sorry

end area_inside_C_outside_A_B_l206_206050


namespace mechanic_worked_hours_l206_206451

theorem mechanic_worked_hours 
  (total_cost : ℝ) 
  (part_cost : ℝ) 
  (part_count : ℕ) 
  (labor_cost_per_minute : ℝ) 
  (parts_total_cost : ℝ) 
  (labor_total_cost : ℝ) 
  (hours_worked : ℝ) 
  (h1 : total_cost = 220) 
  (h2 : part_count = 2) 
  (h3 : part_cost = 20) 
  (h4 : labor_cost_per_minute = 0.5) 
  (h5 : parts_total_cost = part_count * part_cost) 
  (h6 : labor_total_cost = total_cost - parts_total_cost) 
  (labor_cost_per_hour := labor_cost_per_minute * 60) 
  (h7 : hours_worked = labor_total_cost / labor_cost_per_hour) : 
  hours_worked = 6 := 
sorry

end mechanic_worked_hours_l206_206451


namespace smallest_identical_digit_divisible_by_18_l206_206703

theorem smallest_identical_digit_divisible_by_18 :
  ∃ n : Nat, (∀ d : Nat, d < n → ∃ a : Nat, (n = a * (10 ^ d - 1) / 9 + 1 ∧ (∃ k : Nat, n = 18 * k))) ∧ n = 666 :=
by
  sorry

end smallest_identical_digit_divisible_by_18_l206_206703


namespace empty_set_is_d_l206_206983

open Set

theorem empty_set_is_d : {x : ℝ | x^2 - x + 1 = 0} = ∅ :=
by
  sorry

end empty_set_is_d_l206_206983


namespace min_abs_phi_l206_206568

theorem min_abs_phi {f : ℝ → ℝ} (h : ∀ x, f x = 3 * Real.sin (2 * x + φ) ∧ ∀ x, f (x) = f (2 * π / 3 - x)) :
  |φ| = π / 6 :=
by
  sorry

end min_abs_phi_l206_206568


namespace alyosha_cube_problem_l206_206031

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l206_206031


namespace Vincent_sells_8_literature_books_per_day_l206_206806

theorem Vincent_sells_8_literature_books_per_day
  (fantasy_book_cost : ℕ)
  (literature_book_cost : ℕ)
  (fantasy_books_sold_per_day : ℕ)
  (total_earnings_5_days : ℕ)
  (H_fantasy_book_cost : fantasy_book_cost = 4)
  (H_literature_book_cost : literature_book_cost = 2)
  (H_fantasy_books_sold_per_day : fantasy_books_sold_per_day = 5)
  (H_total_earnings_5_days : total_earnings_5_days = 180) :
  ∃ L : ℕ, L = 8 :=
by
  sorry

end Vincent_sells_8_literature_books_per_day_l206_206806


namespace animals_in_field_l206_206989

def dog := 1
def cats := 4
def rabbits_per_cat := 2
def hares_per_rabbit := 3

def rabbits := cats * rabbits_per_cat
def hares := rabbits * hares_per_rabbit

def total_animals := dog + cats + rabbits + hares

theorem animals_in_field : total_animals = 37 := by
  sorry

end animals_in_field_l206_206989


namespace problem_l206_206420

-- Define the variable
variable (x : ℝ)

-- Define the condition
def condition := 3 * x - 1 = 8

-- Define the statement to be proven
theorem problem (h : condition x) : 150 * (1 / x) + 2 = 52 :=
  sorry

end problem_l206_206420


namespace find_f_prime_at_2_l206_206728

noncomputable def f (f'2 : ℝ) : ℝ → ℝ := λ x, x^2 * f'2 + 5 * x

theorem find_f_prime_at_2 :
  ∃ f'2 : ℝ, (∃ f x, f x = x^2 * f'2 + 5 * x) ∧ deriv (f f'2) 2 = -5/3 :=
  sorry

end find_f_prime_at_2_l206_206728


namespace larger_integer_l206_206974

-- Definitions based on the given conditions
def two_integers (x : ℤ) (y : ℤ) :=
  y = 4 * x ∧ (x + 12) * 2 = y

-- Statement of the problem
theorem larger_integer (x : ℤ) (y : ℤ) (h : two_integers x y) : y = 48 :=
by sorry

end larger_integer_l206_206974


namespace smallest_number_value_l206_206312

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a + b + c = 100 ∧
  c = 2 * a ∧
  c - b = 10

theorem smallest_number_value (h : conditions a b c) : a = 22 :=
by
  sorry

end smallest_number_value_l206_206312


namespace geom_seq_inequality_l206_206438

-- Define S_n as a.sum of the first n terms of a geometric sequence with ratio q and first term a_1
noncomputable def S (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
if h : q = 1 then (n + 1) * a_1 else a_1 * (1 - q ^ (n + 1)) / (1 - q)

-- Define a_n for geometric sequence
noncomputable def a_seq (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
a_1 * q ^ n

-- The main theorem to prove
theorem geom_seq_inequality (a_1 : ℝ) (q : ℝ) (n : ℕ) (hq_pos : 0 < q) :
  S a_1 q (n + 1) * a_seq a_1 q n > S a_1 q n * a_seq a_1 q (n + 1) :=
by {
  sorry -- Placeholder for actual proof
}

end geom_seq_inequality_l206_206438


namespace donuts_left_l206_206193

def initial_donuts : ℕ := 50
def after_bill_eats (initial : ℕ) : ℕ := initial - 2
def after_secretary_takes (remaining_after_bill : ℕ) : ℕ := remaining_after_bill - 4
def coworkers_take (remaining_after_secretary : ℕ) : ℕ := remaining_after_secretary / 2
def final_donuts (initial : ℕ) : ℕ :=
  let remaining_after_bill := after_bill_eats initial
  let remaining_after_secretary := after_secretary_takes remaining_after_bill
  remaining_after_secretary - coworkers_take remaining_after_secretary

theorem donuts_left : final_donuts 50 = 22 := by
  sorry

end donuts_left_l206_206193


namespace factory_car_production_l206_206348

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l206_206348


namespace john_read_bible_in_weeks_l206_206587

-- Given Conditions
def reads_per_hour : ℕ := 50
def reads_per_day_hours : ℕ := 2
def bible_length_pages : ℕ := 2800

-- Calculated values based on the given conditions
def reads_per_day : ℕ := reads_per_hour * reads_per_day_hours
def days_to_finish : ℕ := bible_length_pages / reads_per_day
def days_per_week : ℕ := 7

-- The proof statement
theorem john_read_bible_in_weeks : days_to_finish / days_per_week = 4 := by
  sorry

end john_read_bible_in_weeks_l206_206587


namespace measure_six_pints_l206_206956
-- Importing the necessary library

-- Defining the problem conditions
def total_wine : ℕ := 12
def capacity_8_pint_vessel : ℕ := 8
def capacity_5_pint_vessel : ℕ := 5

-- The problem to prove: it is possible to measure 6 pints into the 8-pint container
theorem measure_six_pints :
  ∃ (n : ℕ), n = 6 ∧ n ≤ capacity_8_pint_vessel := 
sorry

end measure_six_pints_l206_206956


namespace problem_statement_l206_206424

theorem problem_statement (x y : ℝ) (hx : x - y = 3) (hxy : x = 4 ∧ y = 1) : 2 * (x - y) = 6 * y :=
by
  rcases hxy with ⟨hx', hy'⟩
  rw [hx', hy']
  sorry

end problem_statement_l206_206424


namespace problem_a_problem_b_l206_206804

-- Definitions for the conditions
def numCrocodiles := 10
def probCrocInEgg := 0.1
def firstCollectionComplete := true

def p : ℕ → ℝ := sorry -- Assume a function p for probabilities

-- Proof problem (a)
theorem problem_a (p1 : ℝ) (p2 : ℝ) :
  p 1 = p1 →
  p 2 = p2 →
  p1 = p2 :=
by sorry

-- Proof problem (b)
theorem problem_b (p2 : ℝ) (p3 : ℝ) (p4 : ℝ) (p5 : ℝ) (p6 : ℝ) (p7 : ℝ) (p8 : ℝ) (p9 : ℝ) (p10 : ℝ) :
  p 2 = p2 →
  p 3 = p3 →
  p 4 = p4 →
  p 5 = p5 →
  p 6 = p6 →
  p 7 = p7 →
  p 8 = p8 →
  p 9 = p9 →
  p 10 = p10 →
  p2 > p3 →
  p3 > p4 →
  p4 > p5 →
  p5 > p6 →
  p6 > p7 →
  p7 > p8 →
  p8 > p9 →
  p9 > p10 :=
by sorry

end problem_a_problem_b_l206_206804


namespace series_sum_equals_seven_ninths_l206_206865

noncomputable def infinite_series_sum : ℝ :=
  ∑' n, 1 / (n * (n + 3))

theorem series_sum_equals_seven_ninths : infinite_series_sum = (7 / 9) :=
by sorry

end series_sum_equals_seven_ninths_l206_206865


namespace sufficient_but_not_necessary_condition_l206_206631

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  ((∀ x : ℝ, (1 < x) → (x^2 - m * x + 1 > 0)) ↔ (-2 < m ∧ m < 2)) :=
sorry

end sufficient_but_not_necessary_condition_l206_206631


namespace average_k_positive_int_roots_l206_206232

theorem average_k_positive_int_roots :
  ∀ (k : ℕ), 
    (∃ p q : ℕ, p > 0 ∧ q > 0 ∧ pq = 24 ∧ k = p + q) 
    → 
    (k ∈ {25, 14, 11, 10}) 
    ∧
    ( ∑ k in {25, 14, 11, 10}, k) / 4 = 15 :=
begin
  sorry
end

end average_k_positive_int_roots_l206_206232


namespace line_quadrant_relationship_l206_206991

theorem line_quadrant_relationship
  (a b c : ℝ)
  (passes_first_second_fourth : ∀ x y : ℝ, (a * x + b * y + c = 0) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) :
  (a * b > 0) ∧ (b * c < 0) :=
sorry

end line_quadrant_relationship_l206_206991


namespace square_area_from_diagonal_l206_206339

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end square_area_from_diagonal_l206_206339


namespace problem_solution_l206_206949

def prop_p (a b c : ℝ) : Prop := a < b → a * c^2 < b * c^2

def prop_q : Prop := ∃ x : ℝ, x^2 - x + 1 ≤ 0

theorem problem_solution : (p ∨ ¬q) := sorry

end problem_solution_l206_206949


namespace quadratic_properties_l206_206889

theorem quadratic_properties (a b c : ℝ) (h₀ : a < 0) (h₁ : a - b + c = 0) :
  (am² b c - 4 a) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ x1 x2 : ℝ, (∃ h : a * x1^2 + b * x1 + c + 1 = 0, ∃ h2 : a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l206_206889


namespace no_integer_solutions_l206_206770

theorem no_integer_solutions (x y : ℤ) : ¬ (3 * x^2 + 2 = y^2) :=
sorry

end no_integer_solutions_l206_206770


namespace jerome_money_left_l206_206553

-- Given conditions
def half_of_money (m : ℕ) : Prop := m / 2 = 43
def amount_given_to_meg (x : ℕ) : Prop := x = 8
def amount_given_to_bianca (x : ℕ) : Prop := x = 3 * 8

-- Problem statement
theorem jerome_money_left (m : ℕ) (x : ℕ) (y : ℕ) (h1 : half_of_money m) (h2 : amount_given_to_meg x) (h3 : amount_given_to_bianca y) : m - x - y = 54 :=
sorry

end jerome_money_left_l206_206553


namespace number_of_trumpet_players_l206_206952

def number_of_people_in_orchestra := 21
def number_of_people_known := 1 -- Sebastian
                             + 4 -- Trombone players
                             + 1 -- French horn player
                             + 3 -- Violinists
                             + 1 -- Cellist
                             + 1 -- Contrabassist
                             + 3 -- Clarinet players
                             + 4 -- Flute players
                             + 1 -- Maestro

theorem number_of_trumpet_players : 
  number_of_people_in_orchestra = number_of_people_known + 2 :=
by
  sorry

end number_of_trumpet_players_l206_206952


namespace non_integer_sum_exists_l206_206119

theorem non_integer_sum_exists (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ M : ℕ, ∀ n : ℕ, n > M → ¬ ∃ t : ℤ, (k + 1/2)^n + (l + 1/2)^n = t := 
sorry

end non_integer_sum_exists_l206_206119


namespace quadratic_conclusions_l206_206890

variables {a b c : ℝ} (h1 : a < 0) (h2 : a - b + c = 0)

theorem quadratic_conclusions
    (h_intersect : ∃ x, a * x ^ 2 + b * x + c = 0 ∧ x = -1)
    (h_symmetry : ∀ x, x = 1 → a * (x - 1) ^ 2 + b * (x - 1) + c = a * (x + 1) ^ 2 + b * (x + 1) + c) :
    a - b + c = 0 ∧ 
    (∀ m : ℝ, a * m ^ 2 + b * m + c ≤ -4 * a) ∧ 
    (∃ x1 x2 : ℝ, a * x1 ^ 2 + b * x1 + c + 1 = 0 ∧ a * x2 ^ 2 + b * x2 + c + 1 = 0 ∧ x1 < x2 ∧ x1 < -1 ∧ x2 > 3) :=
begin
    sorry
end

end quadratic_conclusions_l206_206890


namespace difference_between_20th_and_first_15_l206_206188

def grains_on_square (k : ℕ) : ℕ := 2^k

def total_grains_on_first_15_squares : ℕ :=
  (Finset.range 15).sum (λ k => grains_on_square (k + 1))

def grains_on_20th_square : ℕ := grains_on_square 20

theorem difference_between_20th_and_first_15 :
  grains_on_20th_square - total_grains_on_first_15_squares = 983042 :=
by
  sorry

end difference_between_20th_and_first_15_l206_206188


namespace sum_infinite_series_l206_206863

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l206_206863


namespace average_of_k_l206_206241

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l206_206241


namespace distance_from_plate_to_bottom_edge_l206_206457

theorem distance_from_plate_to_bottom_edge :
    ∀ (d : ℕ), 10 + 63 = 20 + d → d = 53 :=
by
  intros d h
  sorry

end distance_from_plate_to_bottom_edge_l206_206457


namespace minimum_b_l206_206458

theorem minimum_b (k a b : ℝ) (h1 : 1 < k) (h2 : k < a) (h3 : a < b)
  (h4 : ¬(k + a > b)) (h5 : ¬(1/a + 1/b > 1/k)) :
  2 * k ≤ b :=
by
  sorry

end minimum_b_l206_206458


namespace ways_to_turn_off_lights_l206_206274

-- Define the problem conditions
def streetlights := 12
def can_turn_off := 3
def not_turn_off_at_ends := true
def not_adjacent := true

-- The theorem to be proved
theorem ways_to_turn_off_lights : 
  ∃ n, 
  streetlights = 12 ∧ 
  can_turn_off = 3 ∧ 
  not_turn_off_at_ends ∧ 
  not_adjacent ∧ 
  n = 56 :=
by 
  sorry

end ways_to_turn_off_lights_l206_206274


namespace convex_polyhedron_same_number_of_sides_l206_206300

theorem convex_polyhedron_same_number_of_sides {N : ℕ} (hN : N ≥ 4): 
  ∃ (f1 f2 : ℕ), (f1 >= 3 ∧ f1 < N ∧ f2 >= 3 ∧ f2 < N) ∧ f1 = f2 :=
by
  sorry

end convex_polyhedron_same_number_of_sides_l206_206300


namespace triangle_is_obtuse_l206_206385

theorem triangle_is_obtuse
  (α : ℝ)
  (h1 : α > 0 ∧ α < π)
  (h2 : Real.sin α + Real.cos α = 2 / 3) :
  ∃ β γ, β > 0 ∧ β < π ∧ γ > 0 ∧ γ < π ∧ β + γ + α = π ∧ (α > π / 2 ∨ β > π / 2 ∨ γ > π / 2) :=
sorry

end triangle_is_obtuse_l206_206385


namespace max_constant_term_l206_206321

theorem max_constant_term (c : ℝ) : 
  (∀ x : ℝ, (x^2 - 6 * x + c = 0 → (x^2 - 6 * x + c ≥ 0))) → c ≤ 9 :=
by sorry

end max_constant_term_l206_206321


namespace solve_equation_l206_206877

-- Definitions based on the conditions
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10) = 0

-- Theorem stating that the solutions of the given equation are the expected values
theorem solve_equation :
  {x : ℝ | equation x} = {-2 + 2 * Real.sqrt 14, -2 - 2 * Real.sqrt 14, (7 + Real.sqrt 89) / 2, (7 - Real.sqrt 89) / 2} :=
by
  sorry

end solve_equation_l206_206877


namespace an_correct_Tn_correct_l206_206895

-- Splitting the statements for clarity
open Nat

section problem
variables {a b : ℕ → ℚ}

-- Assumption about sum of first n terms
def Sn (n : ℕ) : ℚ := 2n^2 + 3n

-- General term of a_n sequence
def an (n : ℕ) : ℚ := 4n + 1

-- Definition of b_n sequence
def bn (n : ℕ) : ℚ := 1 / (an n * an (n + 1))

-- Sum of first n terms of b_n sequence
def Tn (n : ℕ) : ℚ := Finset.sum (Finset.range n) (λ i, bn i)

-- Theorem statements
theorem an_correct (n : ℕ) : Sn n - Sn (n - 1) = an n :=
  sorry

theorem Tn_correct (n : ℕ) : Tn n = n / (5 * (4n + 5)) :=
  sorry

end problem

end an_correct_Tn_correct_l206_206895


namespace project_work_time_ratio_l206_206612

theorem project_work_time_ratio (A B C : ℕ) (h_ratio : A = x ∧ B = 2 * x ∧ C = 3 * x) (h_total : A + B + C = 120) : 
  (C - A = 40) :=
by
  sorry

end project_work_time_ratio_l206_206612


namespace groups_partition_count_l206_206783

-- Definitions based on the conditions
def num_dogs : ℕ := 12
def group1_size : ℕ := 4
def group2_size : ℕ := 6
def group3_size : ℕ := 2

-- Given specific names for groups based on problem statement
def Fluffy_group_size : ℕ := group1_size
def Nipper_group_size : ℕ := group2_size

-- The total number of ways to form the groups given the conditions
def total_ways (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem groups_partition_count :
  total_ways 10 3 * total_ways 7 5 = 2520 := sorry

end groups_partition_count_l206_206783


namespace how_many_people_in_group_l206_206735

-- Definition of the conditions
def ratio_likes_football : ℚ := 24 / 60
def ratio_plays_football_given_likes : ℚ := 1 / 2
def expected_to_play_football : ℕ := 50

-- Combining the ratios to get the fraction of total people playing football
def ratio_plays_football : ℚ := ratio_likes_football * ratio_plays_football_given_likes

-- Total number of people in the group
def total_people_in_group : ℕ := 250

-- Proof statement
theorem how_many_people_in_group (expected_to_play_football : ℕ) : 
  ratio_plays_football * total_people_in_group = expected_to_play_football :=
by {
  -- Directly using our definitions
  sorry
}

end how_many_people_in_group_l206_206735


namespace bamboo_node_volume_5_l206_206145

theorem bamboo_node_volume_5 {a_1 d : ℚ} :
  (a_1 + (a_1 + d) + (a_1 + 2 * d) + (a_1 + 3 * d) = 3) →
  ((a_1 + 6 * d) + (a_1 + 7 * d) + (a_1 + 8 * d) = 4) →
  (a_1 + 4 * d = 67 / 66) :=
by sorry

end bamboo_node_volume_5_l206_206145


namespace side_length_of_square_base_l206_206613

theorem side_length_of_square_base (area : ℝ) (slant_height : ℝ) (s : ℝ) (h : slant_height = 40) (a : area = 160) : s = 8 :=
by sorry

end side_length_of_square_base_l206_206613


namespace no_square_divisible_by_six_in_range_l206_206699

theorem no_square_divisible_by_six_in_range : ¬ ∃ y : ℕ, (∃ k : ℕ, y = k * k) ∧ (6 ∣ y) ∧ (50 ≤ y ∧ y ≤ 120) :=
by
  sorry

end no_square_divisible_by_six_in_range_l206_206699


namespace find_number_l206_206663

-- Define the conditions
def number_times_x_eq_165 (number x : ℕ) : Prop :=
  number * x = 165

def x_eq_11 (x : ℕ) : Prop :=
  x = 11

-- The proof problem statement
theorem find_number (number x : ℕ) (h1 : number_times_x_eq_165 number x) (h2 : x_eq_11 x) : number = 15 :=
by
  sorry

end find_number_l206_206663


namespace Jerome_money_left_l206_206555

-- Definitions based on conditions
def J_half := 43              -- Half of Jerome's money
def to_Meg := 8               -- Amount Jerome gave to Meg
def to_Bianca := to_Meg * 3   -- Amount Jerome gave to Bianca

-- Total initial amount of Jerome's money
def J_initial : ℕ := J_half * 2

-- Amount left after giving money to Meg
def after_Meg : ℕ := J_initial - to_Meg

-- Amount left after giving money to Bianca
def after_Bianca : ℕ := after_Meg - to_Bianca

-- Statement to be proved
theorem Jerome_money_left : after_Bianca = 54 :=
by
  sorry

end Jerome_money_left_l206_206555


namespace total_cars_made_in_two_days_l206_206351

-- Define the number of cars made yesterday.
def cars_yesterday : ℕ := 60

-- Define the number of cars made today, which is twice the number of cars made yesterday.
def cars_today : ℕ := 2 * cars_yesterday

-- Define the total number of cars made over the two days.
def total_cars : ℕ := cars_yesterday + cars_today

-- Proof statement: Prove that the total number of cars made in these two days is 180.
theorem total_cars_made_in_two_days : total_cars = 180 := by
  -- Here, we would provide the proof, but we'll use sorry to skip it.
  sorry

end total_cars_made_in_two_days_l206_206351


namespace gcf_palindromes_multiple_of_3_eq_3_l206_206809

-- Defining a condition that expresses a three-digit palindrome in the form 101a + 10b + a
def is_palindrome (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ n = 101 * a + 10 * b + a

-- Defining a condition that n is a multiple of 3
def is_multiple_of_3 (n : ℕ) : Prop :=
n % 3 = 0

-- The Lean statement to prove the greatest common factor of all three-digit palindromes that are multiples of 3
theorem gcf_palindromes_multiple_of_3_eq_3 :
  ∃ gcf : ℕ, gcf = 3 ∧ ∀ n : ℕ, (is_palindrome n ∧ is_multiple_of_3 n) → gcf ∣ n :=
by
  sorry

end gcf_palindromes_multiple_of_3_eq_3_l206_206809


namespace values_of_b_l206_206522

noncomputable def proof_problem (x y b : ℝ) : Prop :=
    (sqrt (x + y) = b^b) ∧ (log b (x^2 * y) + log b (y^2 * x) = 3 * b^3)

theorem values_of_b (x y b : ℝ) : proof_problem x y b → b > 0 :=
begin
    sorry
end

end values_of_b_l206_206522


namespace find_k_l206_206181

def total_balls (k : ℕ) : ℕ := 7 + k

def probability_green (k : ℕ) : ℚ := 7 / (total_balls k)
def probability_purple (k : ℕ) : ℚ := k / (total_balls k)

def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * 3 + (probability_purple k) * (-1)

theorem find_k (k : ℕ) (h_pos : k > 0) (h_exp_value : expected_value k = 1) : k = 7 :=
sorry

end find_k_l206_206181


namespace painted_cubes_l206_206828

/-- 
  Given a cube of side 9 painted red and cut into smaller cubes of side 3,
  prove the number of smaller cubes with paint on exactly 2 sides is 12.
-/
theorem painted_cubes (l : ℕ) (s : ℕ) (n : ℕ) (edges : ℕ) (faces : ℕ)
  (hcube_dimension : l = 9) (hsmaller_cubes_dimension : s = 3) 
  (hedges : edges = 12) (hfaces : faces * edges = 12) 
  (htotal_cubes : n = (l^3) / (s^3)) : 
  n * faces = 12 :=
sorry

end painted_cubes_l206_206828


namespace prime_cubic_condition_l206_206733

theorem prime_cubic_condition (p : ℕ) (hp : Nat.Prime p) (hp_prime : Nat.Prime (p^4 - 3 * p^2 + 9)) : p = 2 :=
sorry

end prime_cubic_condition_l206_206733


namespace correct_conclusions_l206_206542

-- Given function f with the specified domain and properties
variable {f : ℝ → ℝ}

-- Given conditions
axiom functional_eq (x y : ℝ) : f (x + y) + f (x - y) = 2 * f x * f y
axiom f_one_half : f (1/2) = 0
axiom f_zero_not_zero : f 0 ≠ 0

-- Proving our conclusions
theorem correct_conclusions :
  f 0 = 1 ∧ (∀ y : ℝ, f (1/2 + y) = -f (1/2 - y))
:=
by
  sorry

end correct_conclusions_l206_206542


namespace range_of_k_condition_l206_206297

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := (4 - k) / x

theorem range_of_k_condition (k x1 x2 y1 y2 : ℝ) 
    (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 < y2) 
    (hA : inverse_proportion_function k x1 = y1) 
    (hB : inverse_proportion_function k x2 = y2) : 
    k < 4 :=
sorry

end range_of_k_condition_l206_206297


namespace count_integers_in_range_l206_206401

theorem count_integers_in_range : 
  ∃ n : ℕ, n = 6 ∧ ∀ x : ℤ, (-7 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ 8) ↔ (-3 ≤ x ∧ x ≤ 2) :=
by
  sorry

end count_integers_in_range_l206_206401


namespace student_fraction_mistake_l206_206737

theorem student_fraction_mistake (n : ℕ) (h_n : n = 576) 
(h_mistake : ∃ r : ℚ, r * n = (5 / 16) * n + 300) : ∃ r : ℚ, r = 5 / 6 :=
by
  sorry

end student_fraction_mistake_l206_206737


namespace triangle_possible_side_lengths_l206_206870

theorem triangle_possible_side_lengths (x : ℕ) (hx : x > 0) (h1 : x^2 + 9 > 12) (h2 : x^2 + 12 > 9) (h3 : 9 + 12 > x^2) : x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end triangle_possible_side_lengths_l206_206870


namespace solve_system_l206_206472

theorem solve_system :
  ∃ x y : ℝ, (x + y = 5) ∧ (x + 2 * y = 8) ∧ (x = 2) ∧ (y = 3) :=
by
  sorry

end solve_system_l206_206472


namespace computation_l206_206051

theorem computation : 45 * 52 + 28 * 45 = 3600 := by
  sorry

end computation_l206_206051


namespace imaginary_part_of_z_squared_l206_206074

-- Let i be the imaginary unit
def i : ℂ := Complex.I

-- Define the complex number (1 - 2i)
def z : ℂ := 1 - 2 * i

-- Define the expanded form of (1 - 2i)^2
def z_squared : ℂ := z^2

-- State the problem of finding the imaginary part of (1 - 2i)^2
theorem imaginary_part_of_z_squared : (z_squared).im = -4 := by
  sorry

end imaginary_part_of_z_squared_l206_206074


namespace pure_imaginary_condition_fourth_quadrant_condition_l206_206719

theorem pure_imaginary_condition (m : ℝ) (h1: m * (m - 1) = 0) (h2: m ≠ 1) : m = 0 :=
by
  sorry

theorem fourth_quadrant_condition (m : ℝ) (h3: m + 1 > 0) (h4: m^2 - 1 < 0) : -1 < m ∧ m < 1 :=
by
  sorry

end pure_imaginary_condition_fourth_quadrant_condition_l206_206719


namespace smaller_cubes_count_l206_206504

theorem smaller_cubes_count (painted_faces: ℕ) (edge_cubes: ℕ) (total_cubes: ℕ) : 
  (painted_faces = 2 ∧ edge_cubes = 12) → total_cubes = 27 :=
by
  assume h : (painted_faces = 2 ∧ edge_cubes = 12)
  sorry

end smaller_cubes_count_l206_206504


namespace no_integer_solution_system_l206_206601

theorem no_integer_solution_system (
  x y z : ℤ
) : x^6 + x^3 + x^3 * y + y ≠ 147 ^ 137 ∨ x^3 + x^3 * y + y^2 + y + z^9 ≠ 157 ^ 117 :=
by
  sorry

end no_integer_solution_system_l206_206601


namespace chord_bisect_angle_l206_206271

theorem chord_bisect_angle (AB AC : ℝ) (angle_CAB : ℝ) (h1 : AB = 2) (h2 : AC = 1) (h3 : angle_CAB = 120) : 
  ∃ x : ℝ, x = 3 := 
by
  -- Proof goes here
  sorry

end chord_bisect_angle_l206_206271


namespace increasing_F_f_additive_towards_f_sum_f_n_additive_towards_f_sum_l206_206123

variable {f : ℝ → ℝ} {x : ℝ} {x1 x2 : ℝ} {n : ℕ} {x_vals : Fin n → ℝ}

-- Given conditions
theorem increasing_F (h : ∀ x > 0, deriv f x > f x / x) : ∀ x > 0, deriv (λ x, f x / x) x > 0 :=
sorry

theorem f_additive_towards_f_sum (h : ∀ x > 0, deriv f x > f x / x) : ∀ x1 x2 > 0, f x1 + f x2 < f (x1 + x2) :=
sorry

theorem f_n_additive_towards_f_sum (n : ℕ) (h : ∀ x > 0, deriv f x > f x / x) : ∀ x_vals : Fin n → ℝ, (∀ i, x_vals i > 0) → (∑ i, f (x_vals i)) < f (∑ i, x_vals i)
 :=
sorry

end increasing_F_f_additive_towards_f_sum_f_n_additive_towards_f_sum_l206_206123


namespace probability_at_least_two_students_succeeding_l206_206985

-- The probabilities of each student succeeding
def p1 : ℚ := 1 / 2
def p2 : ℚ := 1 / 4
def p3 : ℚ := 1 / 5

/-- Calculation of the total probability that at least two out of the three students succeed -/
theorem probability_at_least_two_students_succeeding : 
  (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) + (p1 * p2 * p3) = 9 / 40 :=
  sorry

end probability_at_least_two_students_succeeding_l206_206985


namespace factors_of_180_multiple_of_15_count_l206_206408

theorem factors_of_180_multiple_of_15_count :
  ∃ n : Nat, n = 6 ∧ ∀ x : Nat, x ∣ 180 → 15 ∣ x → x > 0 :=
by
  sorry

end factors_of_180_multiple_of_15_count_l206_206408


namespace expression_simplification_l206_206693

def base_expr := (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) *
                (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64)

theorem expression_simplification :
  base_expr = 3^128 - 4^128 := by
  sorry

end expression_simplification_l206_206693


namespace complement_union_correct_l206_206936

noncomputable def U : Set ℕ := {2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {x | x^2 - 6*x + 8 = 0}
noncomputable def B : Set ℕ := {2, 5, 6}

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 5, 6} := 
by
  sorry

end complement_union_correct_l206_206936


namespace recruit_people_l206_206606

variable (average_contribution : ℝ) (total_funds_needed : ℝ) (current_funds : ℝ)

theorem recruit_people (h₁ : average_contribution = 10) (h₂ : current_funds = 200) (h₃ : total_funds_needed = 1000) : 
    (total_funds_needed - current_funds) / average_contribution = 80 := by
  sorry

end recruit_people_l206_206606


namespace find_b_l206_206082

-- Define the variables involved
variables (a b : ℝ)

-- Conditions provided in the problem
def condition_1 : Prop := 2 * a + 1 = 1
def condition_2 : Prop := b + a = 3

-- Theorem statement to prove b = 3 given the conditions
theorem find_b (h1 : condition_1 a) (h2 : condition_2 a b) : b = 3 := by
  sorry

end find_b_l206_206082


namespace tetrahedron_volume_l206_206277

noncomputable def volume_tetrahedron (A₁ A₂ : ℝ) (θ : ℝ) (d : ℝ) : ℝ :=
  (A₁ * A₂ * Real.sin θ) / (3 * d)

theorem tetrahedron_volume:
  ∀ (PQ PQR PQS : ℝ) (θ : ℝ),
  PQ = 5 → PQR = 20 → PQS = 18 → θ = Real.pi / 4 → volume_tetrahedron PQR PQS θ PQ = 24 * Real.sqrt 2 :=
by
  intros
  unfold volume_tetrahedron
  sorry

end tetrahedron_volume_l206_206277


namespace square_area_from_diagonal_l206_206340

theorem square_area_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 2) : (d / Real.sqrt 2) ^ 2 = 64 :=
by
  sorry

end square_area_from_diagonal_l206_206340


namespace positive_factors_of_180_multiple_of_15_count_l206_206409

theorem positive_factors_of_180_multiple_of_15_count : 
  let factorization_180 := 2^2 * 3^2 * 5;
  let is_multiple_of_15 (n : ℕ) := ∃ k1 k2 : ℕ, k1 * 3 = n ∧ k2 * 5 = n;
  let factors_180 := { n : ℕ | n ∣ factorization_180 };
  { n : ℕ | n ∈ factors_180 ∧ is_multiple_of_15 n }.card = 6 :=
by
  sorry

end positive_factors_of_180_multiple_of_15_count_l206_206409


namespace one_number_greater_than_one_l206_206789

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_prod: a * b * c = 1)
  (h_sum: a + b + c > 1 / a + 1 / b + 1 / c) :
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
by
  sorry

end one_number_greater_than_one_l206_206789


namespace fraction_of_top_10_lists_l206_206335

theorem fraction_of_top_10_lists (total_members : ℝ) (min_top_10_lists : ℝ) (fraction : ℝ) 
  (h1 : total_members = 765) (h2 : min_top_10_lists = 191.25) : 
    min_top_10_lists / total_members = fraction := by
  have h3 : fraction = 0.25 := by sorry
  rw [h1, h2, h3]
  sorry

end fraction_of_top_10_lists_l206_206335


namespace omino_tilings_2_by_10_l206_206259

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n+2) => fib n + fib (n+1)

def omino_tilings (n : ℕ) : ℕ :=
  fib (n + 1)

theorem omino_tilings_2_by_10 : omino_tilings 10 = 3025 := by
  sorry

end omino_tilings_2_by_10_l206_206259


namespace min_k_l_sum_l206_206304

theorem min_k_l_sum (k l : ℕ) (hk : 120 * k = l^3) (hpos_k : k > 0) (hpos_l : l > 0) :
  k + l = 255 :=
sorry

end min_k_l_sum_l206_206304


namespace find_principal_amount_l206_206507

noncomputable def principal_amount_loan (SI R T : ℝ) : ℝ :=
  SI / (R * T)

theorem find_principal_amount (SI R T : ℝ) (h_SI : SI = 6480) (h_R : R = 0.12) (h_T : T = 3) :
  principal_amount_loan SI R T = 18000 :=
by
  rw [principal_amount_loan, h_SI, h_R, h_T]
  norm_num

#check find_principal_amount

end find_principal_amount_l206_206507


namespace tom_books_after_transactions_l206_206818

-- Define the initial conditions as variables
def initial_books : ℕ := 5
def sold_books : ℕ := 4
def new_books : ℕ := 38

-- Define the property we need to prove
theorem tom_books_after_transactions : initial_books - sold_books + new_books = 39 := by
  sorry

end tom_books_after_transactions_l206_206818


namespace mixed_gender_groups_l206_206149

theorem mixed_gender_groups (boys girls : ℕ) (h_boys : boys = 28) (h_girls : girls = 4) :
  ∃ groups : ℕ, (groups ≤ girls) ∧ (groups * 2 ≤ boys) ∧ groups = 4 :=
by
   sorry

end mixed_gender_groups_l206_206149


namespace average_second_pair_l206_206957

theorem average_second_pair 
  (avg_six : ℝ) (avg_first_pair : ℝ) (avg_third_pair : ℝ) (avg_second_pair : ℝ) 
  (h1 : avg_six = 3.95) 
  (h2 : avg_first_pair = 4.2) 
  (h3 : avg_third_pair = 3.8000000000000007) : 
  avg_second_pair = 3.85 :=
by
  sorry

end average_second_pair_l206_206957


namespace solve_equation_l206_206794

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) :
    (3 / (x + 2) - 1 / x = 0) → x = 1 :=
  by sorry

end solve_equation_l206_206794


namespace condo_total_units_l206_206831

-- Definitions from conditions
def total_floors := 23
def regular_units_per_floor := 12
def penthouse_units_per_floor := 2
def penthouse_floors := 2
def regular_floors := total_floors - penthouse_floors

-- Definition for total units
def total_units := (regular_floors * regular_units_per_floor) + (penthouse_floors * penthouse_units_per_floor)

-- Theorem statement: prove total units is 256
theorem condo_total_units : total_units = 256 :=
by
  sorry

end condo_total_units_l206_206831


namespace value_of_7x_minus_3y_l206_206174

theorem value_of_7x_minus_3y (x y : ℚ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := 
sorry

end value_of_7x_minus_3y_l206_206174


namespace tangent_intersection_product_l206_206958

theorem tangent_intersection_product (R r : ℝ) (A B C : ℝ) :
  (AC * CB = R * r) :=
sorry

end tangent_intersection_product_l206_206958


namespace at_least_one_not_less_than_two_l206_206757

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ¬(a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) :=
sorry

end at_least_one_not_less_than_two_l206_206757


namespace problem_p17_l206_206819

def orderings_count (n : ℕ) : ℕ :=
  if n = 1 then 1 else 
  (Nat.floor (n / 2) + 1) * orderings_count (n - 1)

noncomputable def is_perfect_square (x : ℕ) : Prop :=
  ∃ m : ℕ, m * m = x

theorem problem_p17 : (Finset.filter (λ k, is_perfect_square (orderings_count k)) (Finset.range 51)).card = 29 :=
by
  sorry

end problem_p17_l206_206819


namespace min_x2_y2_l206_206720

theorem min_x2_y2 {x y : ℝ} (h : (x - 2)^2 + (y - 2)^2 = 1) : x^2 + y^2 ≥ 9 - 4 * Real.sqrt 2 :=
sorry

end min_x2_y2_l206_206720


namespace dice_probability_l206_206293

/-- 
Nathan rolls two six-sided dice. 
The first die results in an even number, which is one of {2, 4, 6}.
The second die results in a number less than or equal to three, which is one of {1, 2, 3}. 
Prove that the probability of these combined events is 1/4.
-/
def probability_of_combined_events : ℚ := 
  let p_even_first_die := 1 / 2
  let p_less_than_equal_three_second_die := 1 / 2
  p_even_first_die * p_less_than_equal_three_second_die

theorem dice_probability :
  probability_of_combined_events = 1 / 4 :=
by
  sorry

end dice_probability_l206_206293


namespace intersect_sets_l206_206222

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersect_sets : M ∩ N = {1, 2} :=
by
  sorry

end intersect_sets_l206_206222


namespace recorded_instances_l206_206749

-- Define the conditions
def interval := 5
def total_time := 60 * 60  -- one hour in seconds

-- Define the theorem to prove the expected number of instances recorded
theorem recorded_instances : total_time / interval = 720 := by
  sorry

end recorded_instances_l206_206749


namespace decimal_addition_l206_206495

theorem decimal_addition : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end decimal_addition_l206_206495


namespace all_faces_rhombuses_l206_206095

variable {R : Type} [LinearOrderedCommRing R]

structure Parallelepiped (R : Type) :=
  (a b c : R)

def parallelogram_area {R : Type} [LinearOrderedCommRing R] (x y : R) : R :=
  x * y

def is_rhombus (x y : R) : Prop :=
  x = y

theorem all_faces_rhombuses (P : Parallelepiped R)
  (h1: parallelogram_area P.a P.b = parallelogram_area P.b P.c)
  (h2: parallelogram_area P.b P.c = parallelogram_area P.a P.c)
  (h3: parallelogram_area P.a P.b = parallelogram_area P.a P.c) :
  is_rhombus P.a P.b ∧ is_rhombus P.b P.c ∧ is_rhombus P.a P.c :=
  sorry

end all_faces_rhombuses_l206_206095


namespace real_solutions_quadratic_l206_206214

theorem real_solutions_quadratic (a : ℝ) : 
  (∃ x : ℝ, x^2 - 4 * x + a = 0) ↔ a ≤ 4 :=
by sorry

end real_solutions_quadratic_l206_206214


namespace geometric_sequence_sum_l206_206968

theorem geometric_sequence_sum (S : ℕ → ℚ) (n : ℕ) 
  (hS_n : S n = 54) 
  (hS_2n : S (2 * n) = 60) 
  : S (3 * n) = 60 + 2 / 3 := 
sorry

end geometric_sequence_sum_l206_206968


namespace percentage_increase_Sakshi_Tanya_l206_206951

def efficiency_Sakshi : ℚ := 1 / 5
def efficiency_Tanya : ℚ := 1 / 4
def percentage_increase_in_efficiency (eff_Sakshi eff_Tanya : ℚ) : ℚ :=
  ((eff_Tanya - eff_Sakshi) / eff_Sakshi) * 100

theorem percentage_increase_Sakshi_Tanya :
  percentage_increase_in_efficiency efficiency_Sakshi efficiency_Tanya = 25 :=
by
  sorry

end percentage_increase_Sakshi_Tanya_l206_206951


namespace team_selection_l206_206505

open Nat

theorem team_selection :
  let boys := 10
  let girls := 12
  let team_size := 8
  let boys_to_choose := 5
  let girls_to_choose := 3
  choose boys boys_to_choose * choose girls girls_to_choose = 55440 :=
by
  sorry

end team_selection_l206_206505


namespace find_additional_speed_l206_206987

noncomputable def speed_initial : ℝ := 55
noncomputable def t_initial : ℝ := 4
noncomputable def speed_total : ℝ := 60
noncomputable def t_total : ℝ := 6

theorem find_additional_speed :
  let distance_initial := speed_initial * t_initial
  let distance_total := speed_total * t_total
  let t_additional := t_total - t_initial
  let distance_additional := distance_total - distance_initial
  let speed_additional := distance_additional / t_additional
  speed_additional = 70 :=
by
  sorry

end find_additional_speed_l206_206987


namespace PropositionA_PropositionB_PropositionC_PropositionD_l206_206650

-- Proposition A (Incorrect)
theorem PropositionA : ¬(∀ a b c : ℝ, a > b ∧ b > 0 → a * c^2 > b * c^2) :=
sorry

-- Proposition B (Correct)
theorem PropositionB : ∀ a b : ℝ, -2 < a ∧ a < 3 ∧ 1 < b ∧ b < 2 → -4 < a - b ∧ a - b < 2 :=
sorry

-- Proposition C (Correct)
theorem PropositionC : ∀ a b c : ℝ, a > b ∧ b > 0 ∧ c < 0 → c / (a^2) > c / (b^2) :=
sorry

-- Proposition D (Incorrect)
theorem PropositionD : ¬(∀ a b c : ℝ, c > a ∧ a > b → a / (c - a) > b / (c - b)) :=
sorry

end PropositionA_PropositionB_PropositionC_PropositionD_l206_206650


namespace simplify_radical_subtraction_l206_206954

theorem simplify_radical_subtraction : 
  (Real.sqrt 18 - Real.sqrt 8) = Real.sqrt 2 := 
by
  sorry

end simplify_radical_subtraction_l206_206954


namespace brenda_age_l206_206355

theorem brenda_age (A B J : ℕ) (h1 : A = 3 * B) (h2 : J = B + 10) (h3 : A = J) : B = 5 :=
sorry

end brenda_age_l206_206355


namespace probability_b_l206_206391

theorem probability_b (p : Set ℝ → ℝ) (a b : Set ℝ)  
  (h1 : p a = 6 / 17) 
  (h2 : p (a ∪ b) = 4 / 17) 
  (h3 : p (b ∩ a) / p a = 2 / 3) : 
  p b = 2 / 17 :=
by sorry

end probability_b_l206_206391


namespace solve_for_x_l206_206734

theorem solve_for_x (x : ℝ) (h : (2 / 3 - 1 / 4) = 4 / x) : x = 48 / 5 :=
by sorry

end solve_for_x_l206_206734


namespace MarthaEndBlocks_l206_206950

theorem MarthaEndBlocks (start_blocks found_blocks total_blocks : ℕ) 
  (h₁ : start_blocks = 11)
  (h₂ : found_blocks = 129) : 
  total_blocks = 140 :=
by
  sorry

end MarthaEndBlocks_l206_206950


namespace quadratic_properties_l206_206891

def quadratic_function (a b c : ℝ) := λ x, a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ) (h0 : a < 0) (h1 : quadratic_function a b c (-1) = 0)
  (h2 : ∀ m : ℝ, let f := quadratic_function a b c in f(m) ≤ -4 * a)
  (h3 : b = -2 * a) (h4: c = -3 * a):
  (a - b + c = 0) ∧ 
  (∀ x1 x2 : ℝ, quadratic_function a b (c + 1) x1 = 0 → quadratic_function a b (c + 1) x2 = 0 →
    x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l206_206891


namespace chipmunk_acorns_l206_206707

theorem chipmunk_acorns :
  ∀ (x y : ℕ), (3 * x = 4 * y) → (y = x - 4) → (3 * x = 48) :=
by
  intros x y h1 h2
  sorry

end chipmunk_acorns_l206_206707


namespace inequality_abc_l206_206077

theorem inequality_abc 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c ≤ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
by sorry

end inequality_abc_l206_206077


namespace cube_decomposition_l206_206016

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l206_206016


namespace maximize_sum_of_sides_l206_206920

theorem maximize_sum_of_sides (a b c : ℝ) (A B C : ℝ) 
  (h_b : b = 2) (h_B : B = (Real.pi / 3)) (h_law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) :
  a + c ≤ 4 :=
by
  sorry

end maximize_sum_of_sides_l206_206920


namespace product_of_two_numbers_l206_206635

theorem product_of_two_numbers
  (x y : ℝ)
  (h1 : x + y = 25)
  (h2 : x - y = 7) :
  x * y = 144 := 
by
  sorry

end product_of_two_numbers_l206_206635


namespace find_a_l206_206724

theorem find_a (m c a b : ℝ) (h_m : m < 0) (h_radius : (m^2 + 3) = 4) 
  (h_c : c = 1 ∨ c = -3) (h_focus : c > 0) (h_ellipse : b^2 = 3) 
  (h_focus_eq : c^2 = a^2 - b^2) : a = 2 :=
by
  sorry

end find_a_l206_206724


namespace cube_decomposition_l206_206013

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l206_206013


namespace remainder_when_2x_div_8_is_1_l206_206812

theorem remainder_when_2x_div_8_is_1 (x y : ℤ) 
  (h1 : x = 11 * y + 4)
  (h2 : ∃ r : ℤ, 2 * x = 8 * (3 * y) + r)
  (h3 : 13 * y - x = 3) : ∃ r : ℤ, r = 1 :=
by
  sorry

end remainder_when_2x_div_8_is_1_l206_206812


namespace hydrogen_atoms_in_compound_l206_206333

theorem hydrogen_atoms_in_compound :
  ∀ (n : ℕ), 98 = 14 + n + 80 → n = 4 :=
by intro n h_eq
   sorry

end hydrogen_atoms_in_compound_l206_206333


namespace identify_genuine_coins_l206_206476

section IdentifyGenuineCoins

variables (coins : Fin 25 → ℝ) 
          (is_genuine : Fin 25 → Prop) 
          (is_counterfeit : Fin 25 → Prop)

-- Conditions
axiom coin_total : ∀ i, is_genuine i ∨ is_counterfeit i
axiom genuine_count : ∃ s : Finset (Fin 25), s.card = 22 ∧ ∀ i ∈ s, is_genuine i
axiom counterfeit_count : ∃ t : Finset (Fin 25), t.card = 3 ∧ ∀ i ∈ t, is_counterfeit i
axiom genuine_weight : ∃ w : ℝ, ∀ i, is_genuine i → coins i = w
axiom counterfeit_weight : ∃ c : ℝ, ∀ i, is_counterfeit i → coins i = c
axiom counterfeit_lighter : ∀ (w c : ℝ), (∃ i, is_genuine i → coins i = w) ∧ (∃ j, is_counterfeit j → coins j = c) → c < w

-- Theorem: Identifying 6 genuine coins using two weighings
theorem identify_genuine_coins : ∃ s : Finset (Fin 25), s.card = 6 ∧ ∀ i ∈ s, is_genuine i :=
sorry

end IdentifyGenuineCoins

end identify_genuine_coins_l206_206476


namespace cost_per_ounce_l206_206179

theorem cost_per_ounce (total_cost : ℕ) (num_ounces : ℕ) (h1 : total_cost = 84) (h2 : num_ounces = 12) : (total_cost / num_ounces) = 7 :=
by
  sorry

end cost_per_ounce_l206_206179


namespace arcsin_neg_sqrt3_div_2_l206_206692

theorem arcsin_neg_sqrt3_div_2 : 
  Real.arcsin (- (Real.sqrt 3 / 2)) = - (Real.pi / 3) := 
by sorry

end arcsin_neg_sqrt3_div_2_l206_206692


namespace factor_polynomial_l206_206876

theorem factor_polynomial :
  9 * (x + 4) * (x + 7) * (x + 11) * (x + 13) - 5 * x ^ 2 =
  (3 * x ^ 2 + 59 * x + 231) * (3 * x ^ 2 + 53 * x + 231) := by
  sorry

end factor_polynomial_l206_206876


namespace prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l206_206576

noncomputable def prob_zhang_nings_wins_2_1 :=
  2 * 0.4 * 0.6 * 0.6 = 0.288

theorem prob_zhang_nings_wins_2_1_correct : prob_zhang_nings_wins_2_1 := sorry

def prob_ξ_minus_2 := 0.4 * 0.4 = 0.16
def prob_ξ_minus_1 := 2 * 0.4 * 0.6 * 0.4 = 0.192
def prob_ξ_1 := 2 * 0.4 * 0.6 * 0.6 = 0.288
def prob_ξ_2 := 0.6 * 0.6 = 0.36

theorem prob_ξ_minus_2_correct : prob_ξ_minus_2 := sorry
theorem prob_ξ_minus_1_correct : prob_ξ_minus_1 := sorry
theorem prob_ξ_1_correct : prob_ξ_1 := sorry
theorem prob_ξ_2_correct : prob_ξ_2 := sorry

noncomputable def expected_value_ξ :=
  (-2 * 0.16) + (-1 * 0.192) + (1 * 0.288) + (2 * 0.36) = 0.496

theorem expected_value_ξ_correct : expected_value_ξ := sorry

end prob_zhang_nings_wins_2_1_correct_prob_ξ_minus_2_correct_prob_ξ_minus_1_correct_prob_ξ_1_correct_prob_ξ_2_correct_expected_value_ξ_correct_l206_206576


namespace f_eq_zero_range_x_l206_206226

-- Definition of the function f on domain ℝ*
def f (x : ℝ) : ℝ := sorry

-- Conditions
axiom f_domain : ∀ x : ℝ, x ≠ 0 → f x = f x
axiom f_4 : f 4 = 1
axiom f_multiplicative : ∀ x1 x2 : ℝ, x1 ≠ 0 → x2 ≠ 0 → f (x1 * x2) = f x1 + f x2
axiom f_increasing : ∀ x y : ℝ, x < y → f x < f y

-- Problem (1): Prove f(1) = 0
theorem f_eq_zero : f 1 = 0 :=
sorry

-- Problem (2): Prove range 3 < x ≤ 5 given the inequality condition
theorem range_x (x : ℝ) : f (3 * x + 1) + f (2 * x - 6) ≤ 3 → 3 < x ∧ x ≤ 5 :=
sorry

end f_eq_zero_range_x_l206_206226


namespace trigonometric_identity_l206_206849

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l206_206849


namespace average_k_l206_206243

open Nat

def positive_integer_roots (a b : ℕ) : Prop :=
  a * b = 24 ∧ a + b = b + a

theorem average_k (k : ℕ) :
  (positive_integer_roots 1 24 ∨ 
  positive_integer_roots 2 12 ∨ 
  positive_integer_roots 3 8 ∨ 
  positive_integer_roots 4 6) →
  (k = 25 ∨ k = 14 ∨ k = 11 ∨ k = 10) →
  (25 + 14 + 11 + 10) / 4 = 15 := by
  sorry

end average_k_l206_206243


namespace avg_weight_b_c_43_l206_206307

noncomputable def weights_are_correct (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧ (A + B) / 2 = 40 ∧ B = 31

theorem avg_weight_b_c_43 (A B C : ℝ) (h : weights_are_correct A B C) : (B + C) / 2 = 43 :=
by sorry

end avg_weight_b_c_43_l206_206307


namespace point_in_fourth_quadrant_l206_206723

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : -4 * a < 0) (h2 : 2 + b < 0) : 
  (a > 0) ∧ (b < -2) → (a > 0) ∧ (b < 0) := 
by
  sorry

end point_in_fourth_quadrant_l206_206723


namespace red_apples_ordered_l206_206470

variable (R : ℕ)

theorem red_apples_ordered (h : R + 32 = 2 + 73) : R = 43 := by
  sorry

end red_apples_ordered_l206_206470


namespace winner_won_by_l206_206426

theorem winner_won_by (V : ℝ) (h₁ : 0.62 * V = 806) : 806 - 0.38 * V = 312 :=
by
  sorry

end winner_won_by_l206_206426


namespace unit_price_ratio_l206_206194

theorem unit_price_ratio (v p : ℝ) (hv : 0 < v) (hp : 0 < p) :
  (1.1 * p / (1.4 * v)) / (0.85 * p / (1.3 * v)) = 13 / 11 :=
by
  sorry

end unit_price_ratio_l206_206194


namespace consecutive_product_plus_one_l206_206690

theorem consecutive_product_plus_one (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 :=
by
  sorry

end consecutive_product_plus_one_l206_206690


namespace employed_male_percent_problem_l206_206278

noncomputable def employed_percent_population (total_population_employed_percent : ℝ) (employed_females_percent : ℝ) : ℝ :=
  let employed_males_percent := (1 - employed_females_percent) * total_population_employed_percent
  employed_males_percent

theorem employed_male_percent_problem :
  employed_percent_population 0.72 0.50 = 0.36 := by
  sorry

end employed_male_percent_problem_l206_206278


namespace equivalent_statements_l206_206168

variables (P Q : Prop)

theorem equivalent_statements : (¬Q → ¬P) ∧ (¬P ∨ Q) ↔ (P → Q) :=
by
  -- Proof goes here
  sorry

end equivalent_statements_l206_206168


namespace greatest_value_of_squares_exists_max_value_of_squares_l206_206756

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 702 :=
sorry

theorem exists_max_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 83)
  (h3 : ad + bc = 174)
  (h4 : cd = 105) :
  ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 702 :=
sorry

end greatest_value_of_squares_exists_max_value_of_squares_l206_206756


namespace number_of_cars_lifted_l206_206742

def total_cars_lifted : ℕ := 6

theorem number_of_cars_lifted : total_cars_lifted = 6 := by
  sorry

end number_of_cars_lifted_l206_206742


namespace muffin_combinations_l206_206662

theorem muffin_combinations (k : ℕ) (n : ℕ) (h_k : k = 4) (h_n : n = 4) :
  (Nat.choose ((n + k - 1) : ℕ) ((k - 1) : ℕ)) = 35 :=
by
  rw [h_k, h_n]
  -- Simplifying Nat.choose (4 + 4 - 1) (4 - 1) = Nat.choose 7 3
  sorry

end muffin_combinations_l206_206662


namespace alyosha_cube_cut_l206_206036

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l206_206036


namespace num_factors_of_180_multiple_of_15_l206_206406

def is_factor (n m : Nat) : Prop := m % n = 0
def is_multiple_of_15 (n : Nat) : Prop := is_factor 15 n

theorem num_factors_of_180_multiple_of_15 : 
  (Nat.filter (fun n => is_factor n 180 ∧ is_multiple_of_15 n) (Nat.range (180 + 1))).length = 6 :=
sorry

end num_factors_of_180_multiple_of_15_l206_206406


namespace min_green_beads_l206_206508

theorem min_green_beads (B R G : ℕ) (h : B + R + G = 80)
  (hB : ∀ i j : ℕ, (i < j ∧ j ≤ B → ∃ k, i < k ∧ k < j ∧ k ≤ R)) 
  (hR : ∀ i j : ℕ, (i < j ∧ j ≤ R → ∃ k, i < k ∧ k < j ∧ k ≤ G)) :
  G >= 27 := 
sorry

end min_green_beads_l206_206508


namespace store_sales_correct_l206_206922

def price_eraser_pencil : ℝ := 0.8
def price_regular_pencil : ℝ := 0.5
def price_short_pencil : ℝ := 0.4
def price_mechanical_pencil : ℝ := 1.2
def price_novelty_pencil : ℝ := 1.5

def quantity_eraser_pencil : ℕ := 200
def quantity_regular_pencil : ℕ := 40
def quantity_short_pencil : ℕ := 35
def quantity_mechanical_pencil : ℕ := 25
def quantity_novelty_pencil : ℕ := 15

def total_sales : ℝ :=
  (quantity_eraser_pencil * price_eraser_pencil) +
  (quantity_regular_pencil * price_regular_pencil) +
  (quantity_short_pencil * price_short_pencil) +
  (quantity_mechanical_pencil * price_mechanical_pencil) +
  (quantity_novelty_pencil * price_novelty_pencil)

theorem store_sales_correct : total_sales = 246.5 :=
by sorry

end store_sales_correct_l206_206922


namespace trigonometric_identity_l206_206850

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l206_206850


namespace bernardo_wins_at_5_l206_206126

theorem bernardo_wins_at_5 :
  (∀ N : ℕ, (16 * N + 900 < 1000) → (920 ≤ 16 * N + 840) → N ≥ 5)
    ∧ (5 < 10 ∧ 16 * 5 + 900 < 1000 ∧ 920 ≤ 16 * 5 + 840) := by
{
  sorry
}

end bernardo_wins_at_5_l206_206126


namespace scientific_notation_3080000_l206_206474

theorem scientific_notation_3080000 : (∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ (3080000 : ℝ) = a * 10^b) ∧ (3080000 : ℝ) = 3.08 * 10^6 :=
by
  sorry

end scientific_notation_3080000_l206_206474


namespace total_oranges_is_correct_l206_206336

-- Definitions based on the problem's conditions
def layer_count : ℕ := 6
def base_length : ℕ := 9
def base_width : ℕ := 6

-- Function to compute the number of oranges in a layer given the current dimensions
def oranges_in_layer (length width : ℕ) : ℕ :=
  length * width

-- Function to compute the total number of oranges in the stack
def total_oranges_in_stack (base_length base_width : ℕ) : ℕ :=
  oranges_in_layer base_length base_width +
  oranges_in_layer (base_length - 1) (base_width - 1) +
  oranges_in_layer (base_length - 2) (base_width - 2) +
  oranges_in_layer (base_length - 3) (base_width - 3) +
  oranges_in_layer (base_length - 4) (base_width - 4) +
  oranges_in_layer (base_length - 5) (base_width - 5)

-- The theorem to be proved
theorem total_oranges_is_correct : total_oranges_in_stack 9 6 = 154 := by
  sorry

end total_oranges_is_correct_l206_206336


namespace tom_age_ratio_l206_206803

theorem tom_age_ratio (T N : ℕ) 
  (h1 : T = T)
  (h2 : T - N = 3 * (T - 5 * N)) : T / N = 7 :=
by sorry

end tom_age_ratio_l206_206803


namespace decagon_diagonals_l206_206730

-- Define the number of sides of a decagon
def n : ℕ := 10

-- Define the formula for the number of diagonals in an n-sided polygon
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem
theorem decagon_diagonals : num_diagonals n = 35 := by
  sorry

end decagon_diagonals_l206_206730


namespace will_buy_5_toys_l206_206979

theorem will_buy_5_toys (initial_money spent_money toy_cost money_left toys : ℕ) 
  (h1 : initial_money = 57) 
  (h2 : spent_money = 27) 
  (h3 : toy_cost = 6) 
  (h4 : money_left = initial_money - spent_money) 
  (h5 : toys = money_left / toy_cost) : 
  toys = 5 := 
by
  sorry

end will_buy_5_toys_l206_206979


namespace total_wheels_l206_206071

def cars : Nat := 15
def bicycles : Nat := 3
def trucks : Nat := 8
def tricycles : Nat := 1
def wheels_per_car_or_truck : Nat := 4
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3

theorem total_wheels : cars * wheels_per_car_or_truck + trucks * wheels_per_car_or_truck + bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 101 :=
by
  sorry

end total_wheels_l206_206071


namespace mark_increase_reading_time_l206_206127

theorem mark_increase_reading_time : 
  (let hours_per_day := 2
   let days_per_week := 7
   let desired_weekly_hours := 18
   let current_weekly_hours := hours_per_day * days_per_week
   let increase_per_week := desired_weekly_hours - current_weekly_hours
   increase_per_week = 4) :=
by
  let hours_per_day := 2
  let days_per_week := 7
  let desired_weekly_hours := 18
  let current_weekly_hours := hours_per_day * days_per_week
  let increase_per_week := desired_weekly_hours - current_weekly_hours
  have h1 : current_weekly_hours = 14 := by norm_num
  have h2 : increase_per_week = desired_weekly_hours - current_weekly_hours := rfl
  have h3 : increase_per_week = 18 - 14 := by rw [h2, h1]
  have h4 : increase_per_week = 4 := by norm_num
  exact h4

end mark_increase_reading_time_l206_206127


namespace point_satisfies_equation_l206_206494

theorem point_satisfies_equation (x y : ℝ) :
  (-1 ≤ x ∧ x ≤ 3) ∧ (-5 ≤ y ∧ y ≤ 1) ∧
  ((3 * x + 2 * y = 5) ∨ (-3 * x + 2 * y = -1) ∨ (3 * x - 2 * y = 13) ∨ (-3 * x - 2 * y = 7))
  → 3 * |x - 1| + 2 * |y + 2| = 6 := 
by 
  sorry

end point_satisfies_equation_l206_206494


namespace discriminant_nonnegative_l206_206053

theorem discriminant_nonnegative (x : ℤ) (h : x^2 * (25 - 24 * x^2) ≥ 0) : x = 0 ∨ x = 1 ∨ x = -1 :=
by sorry

end discriminant_nonnegative_l206_206053


namespace angle_between_lines_l206_206572

section QuadrilateralAngle
variable (A B C D : ℝ × ℝ)

noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem angle_between_lines (hAB : side_length A B = 4)
                            (hCD : side_length C D = 6)
                            (hMidDist : side_length (midpoint B D) (midpoint A C) = 3)
                           : ∃ α : ℝ, α = real.arccos (1 / 3) :=
by
  sorry
end QuadrilateralAngle

end angle_between_lines_l206_206572


namespace find_abs_xyz_l206_206121

variables {x y z : ℝ}

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem find_abs_xyz
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : distinct x y z)
  (h3 : x + 1 / y = 2)
  (h4 : y + 1 / z = 2)
  (h5 : z + 1 / x = 2) :
  |x * y * z| = 1 :=
by sorry

end find_abs_xyz_l206_206121


namespace distribute_positions_l206_206827

theorem distribute_positions :
  let positions := 11
  let classes := 6
  ∃ total_ways : ℕ, total_ways = Nat.choose (positions - 1) (classes - 1) ∧ total_ways = 252 :=
by
  let positions := 11
  let classes := 6
  have : Nat.choose (positions - 1) (classes - 1) = 252 := by sorry
  exact ⟨Nat.choose (positions - 1) (classes - 1), this, this⟩

end distribute_positions_l206_206827


namespace compute_expression_l206_206052

theorem compute_expression : 12 * (1 / 26) * 52 * 4 = 96 :=
by
  sorry

end compute_expression_l206_206052


namespace ceil_sqrt_sum_l206_206062

theorem ceil_sqrt_sum :
  ⌈Real.sqrt 3⌉₊ + ⌈Real.sqrt 27⌉₊ + ⌈Real.sqrt 243⌉₊ = 24 :=
by
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := by sorry
  have h2 : 5 < Real.sqrt 27 ∧ Real.sqrt 27 < 6 := by sorry
  have h3 : 15 < Real.sqrt 243 ∧ Real.sqrt 243 < 16 := by sorry
  sorry

end ceil_sqrt_sum_l206_206062


namespace least_value_of_q_minus_p_l206_206099

variables (y p q : ℝ)

/-- Triangle side lengths -/
def BC := y + 7
def AC := y + 3
def AB := 2 * y + 1

/-- Given conditions for triangle inequalities and angle B being the largest -/
def triangle_inequality_conditions :=
  (y + 7 + (y + 3) > 2 * y + 1) ∧
  (y + 7 + (2 * y + 1) > y + 3) ∧
  ((y + 3) + (2 * y + 1) > y + 7)

def angle_largest_conditions :=
  (2 * y + 1 > y + 3) ∧
  (2 * y + 1 > y + 7)

/-- Prove the least possible value of q - p given the conditions -/
theorem least_value_of_q_minus_p
  (h1 : triangle_inequality_conditions y)
  (h2 : angle_largest_conditions y)
  (h3 : 6 < y)
  (h4 : y < 8) :
  q - p = 2 := sorry

end least_value_of_q_minus_p_l206_206099


namespace surface_area_is_726_l206_206536

def edge_length : ℝ := 11

def surface_area_of_cube (e : ℝ) : ℝ := 6 * (e * e)

theorem surface_area_is_726 (h : edge_length = 11) : surface_area_of_cube edge_length = 726 := by
  sorry

end surface_area_is_726_l206_206536


namespace no_solution_for_n_ge_10_l206_206142

open Nat

theorem no_solution_for_n_ge_10 (n : ℕ) (h : n ≥ 10) : ¬ (n ≤ n! - 4^n ∧ n! - 4^n ≤ 4 * n) := 
sorry

end no_solution_for_n_ge_10_l206_206142


namespace solve_AlyoshaCube_l206_206020

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l206_206020


namespace scientific_notation_conversion_l206_206468

theorem scientific_notation_conversion :
  0.000037 = 3.7 * 10^(-5) :=
by
  sorry

end scientific_notation_conversion_l206_206468


namespace Bernoulli_inequality_l206_206769

theorem Bernoulli_inequality (p : ℝ) (k : ℚ) (hp : 0 < p) (hk : 1 < k) : 
  (1 + p) ^ (k : ℝ) > 1 + p * (k : ℝ) := by
sorry

end Bernoulli_inequality_l206_206769


namespace derivative_at_1_l206_206087

def f (x : ℝ) : ℝ := x^3 + x^2 - 2 * x - 2

def f_derivative (x : ℝ) : ℝ := 3*x^2 + 2*x - 2

theorem derivative_at_1 : f_derivative 1 = 3 := by
  sorry

end derivative_at_1_l206_206087


namespace John_more_marbles_than_Ben_l206_206519

theorem John_more_marbles_than_Ben :
  let ben_initial := 18
  let john_initial := 17
  let ben_gave := ben_initial / 2
  let ben_final := ben_initial - ben_gave
  let john_final := john_initial + ben_gave
  john_final - ben_final = 17 :=
by
  sorry

end John_more_marbles_than_Ben_l206_206519


namespace expression_evaluation_l206_206202

theorem expression_evaluation : (16^3 + 3 * 16^2 + 3 * 16 + 1 = 4913) :=
by
  sorry

end expression_evaluation_l206_206202


namespace probability_green_ball_l206_206055

/-- 
Given three containers with specific numbers of red and green balls, 
and the probability of selecting each container being equal, 
the probability of picking a green ball when choosing a container randomly is 7/12.
-/
theorem probability_green_ball :
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  (green_I + green_II + green_III) = 7 / 12 :=
by 
  let pI := 1 / 3
  let pII := 1 / 3
  let pIII := 1 / 3
  let p_green_I := 4 / 12
  let p_green_II := 4 / 6
  let p_green_III := 6 / 8
  let green_I := pI * p_green_I
  let green_II := pII * p_green_II
  let green_III := pIII * p_green_III
  have : (green_I + green_II + green_III) = (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) := by rfl
  have : (1 / 3 * 4 / 12 + 1 / 3 * 4 / 6 + 1 / 3 * 6 / 8) = (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) := by rfl
  have : (1 / 3 * 1 / 3 + 1 / 3 * 2 / 3 + 1 / 3 * 3 / 4) = (1 / 9 + 2 / 9 + 1 / 4) := by rfl
  have : (1 / 9 + 2 / 9 + 1 / 4) = (4 / 36 + 8 / 36 + 9 / 36) := by rfl
  have : (4 / 36 + 8 / 36 + 9 / 36) = 21 / 36 := by rfl
  have : 21 / 36 = 7 / 12 := by rfl
  rfl

end probability_green_ball_l206_206055


namespace solve_for_x_l206_206778

def equation (x : ℝ) (y : ℝ) : Prop := 5 * y^2 + y + 10 = 2 * (9 * x^2 + y + 6)

def y_condition (x : ℝ) : ℝ := 3 * x

theorem solve_for_x (x : ℝ) :
  equation x (y_condition x) ↔ (x = 1/3 ∨ x = -2/9) := by
  sorry

end solve_for_x_l206_206778


namespace train_b_speed_l206_206486

/-- Two trains, A and B, start simultaneously from two stations 480 kilometers apart and meet after 2.5 hours. 
Train A travels at a speed of 102 kilometers per hour. What is the speed of train B in kilometers per hour? -/
theorem train_b_speed (d t : ℝ) (speedA speedB : ℝ) (h1 : d = 480) (h2 : t = 2.5) (h3 : speedA = 102)
  (h4 : speedA * t + speedB * t = d) : speedB = 90 := 
by
  sorry

end train_b_speed_l206_206486


namespace tunnel_length_l206_206659

-- Define relevant constants
def train_length : ℝ := 2
def exit_time_minutes : ℝ := 5
def train_speed_mph : ℝ := 40
def miles_per_hour_to_miles_per_minute (mph : ℝ) := mph / 60
def travel_distance (time_minutes : ℝ) (speed_mph : ℝ) := time_minutes * miles_per_hour_to_miles_per_minute speed_mph

-- The main theorem we want to prove
theorem tunnel_length : travel_distance exit_time_minutes train_speed_mph - train_length = 4 / 3 := sorry

end tunnel_length_l206_206659


namespace probability_half_or_more_even_dice_l206_206538

-- Define the fair die probability and the event
def total_dice : ℕ := 4
def probability_of_even : ℚ := 3 / 6

-- Define the problem's goal to calculate the probability a of getting at least half even outcomes
def probability_at_least_half_even : ℚ := 11 / 16

theorem probability_half_or_more_even_dice (total_dice : ℕ)
    (probability_of_even : ℚ) :
    (total_dice = 4) → (probability_of_even = 3 / 6) →
    probability_at_least_half_even = 11 / 16 :=
by
  intros _ _
  -- The actual proof is omitted
  sorry

end probability_half_or_more_even_dice_l206_206538


namespace find_c8_l206_206393

-- Definitions of arithmetic sequences and their products
def arithmetic_seq (a d : ℤ) (n : ℕ) := a + n * d

def c_n (a d1 b d2 : ℤ) (n : ℕ) := arithmetic_seq a d1 n * arithmetic_seq b d2 n

-- Given conditions
variables (a1 d1 a2 d2 : ℤ)
variables (c1 c2 c3 : ℤ)
variables (h1 : c_n a1 d1 a2 d2 1 = 1440)
variables (h2 : c_n a1 d1 a2 d2 2 = 1716)
variables (h3 : c_n a1 d1 a2 d2 3 = 1848)

-- The goal is to prove c_8 = 348
theorem find_c8 : c_n a1 d1 a2 d2 8 = 348 :=
sorry

end find_c8_l206_206393


namespace quadratic_function_coefficient_not_zero_l206_206090

theorem quadratic_function_coefficient_not_zero (m : ℝ) : (∀ x : ℝ, (m-2)*x^2 + 2*x - 3 ≠ 0) → m ≠ 2 :=
by
  intro h
  by_contra h1
  exact sorry

end quadratic_function_coefficient_not_zero_l206_206090


namespace masha_spheres_base_l206_206453

theorem masha_spheres_base (n T n9 : ℕ) (h1 : T = (n * (n + 1)) / 2)
                           (h2 : 1 / 6 * n * (n + 1) * (n + 2) = 165)
                           (h3 : n = 9)
                           (h4 : T = 45) : n9 = 45 :=
by {
  have h5 : n * (n + 1) * (n + 2) = 990, from sorry,
  have h6 : 45 = (n * (n + 1)) / 2, from sorry,
  exact h6
}

end masha_spheres_base_l206_206453


namespace min_cars_needed_l206_206816

theorem min_cars_needed (h1 : ∀ d ∈ Finset.range 7, ∃ s : Finset ℕ, s.card = 2 ∧ (∃ n : ℕ, 7 * (n - 10) ≥ 2 * n)) : 
  ∃ n, n ≥ 14 :=
by
  sorry

end min_cars_needed_l206_206816


namespace solve_for_x_l206_206811

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) (h : (5*x)^10 = (10*x)^5) : x = 2/5 :=
sorry

end solve_for_x_l206_206811


namespace units_digit_of_p_is_6_l206_206546

theorem units_digit_of_p_is_6 (p : ℤ) (h1 : p % 10 > 0) 
                             (h2 : ((p^3) % 10 - (p^2) % 10) = 0) 
                             (h3 : (p + 1) % 10 = 7) : 
                             p % 10 = 6 :=
by sorry

end units_digit_of_p_is_6_l206_206546


namespace copy_pages_15_dollars_l206_206566

theorem copy_pages_15_dollars (cpp : ℕ) (budget : ℕ) (pages : ℕ) (h1 : cpp = 3) (h2 : budget = 1500) (h3 : pages = budget / cpp) : pages = 500 :=
by
  sorry

end copy_pages_15_dollars_l206_206566


namespace solution_set_of_inequality_l206_206967

theorem solution_set_of_inequality :
  {x : ℝ | (x - 3) / x ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 3} :=
sorry

end solution_set_of_inequality_l206_206967


namespace non_congruent_parallelograms_l206_206197

def side_lengths_sum (a b : ℕ) : Prop :=
  a + b = 25

def is_congruent (a b : ℕ) (a' b' : ℕ) : Prop :=
  (a = a' ∧ b = b') ∨ (a = b' ∧ b = a')

def non_congruent_count (n : ℕ) : Prop :=
  ∀ (a b : ℕ), side_lengths_sum a b → 
  ∃! (m : ℕ), is_congruent a b m b

theorem non_congruent_parallelograms :
  ∃ (n : ℕ), non_congruent_count n ∧ n = 13 :=
sorry

end non_congruent_parallelograms_l206_206197


namespace nora_must_sell_5_cases_l206_206598

-- Definitions based on given conditions
def packs_per_case : ℕ := 3
def muffins_per_pack : ℕ := 4
def price_per_muffin : ℕ := 2
def total_goal : ℕ := 120
def total_per_case := packs_per_case * muffins_per_pack * price_per_muffin  -- This calculates the earnings from one case

-- The problem statement as a Lean theorem
theorem nora_must_sell_5_cases : total_goal / total_per_case = 5 := by
  -- Providing the necessary preliminary calculations
  have packs_calc : packs_per_case = 3 := rfl
  have muffins_calc : muffins_per_pack = 4 := rfl
  have price_calc : price_per_muffin = 2 := rfl
  have goal_calc : total_goal = 120 := rfl
  have case_calc : total_per_case = 24 := by unfold total_per_case; simp [packs_calc, muffins_calc, price_calc, mul_assoc]
  calc
    total_goal / total_per_case = 120 / 24 : by congr; exact goal_calc; exact case_calc
    ... = 5 : by norm_num

end nora_must_sell_5_cases_l206_206598


namespace probability_of_intersection_l206_206626

open Finset

noncomputable def a_seq (n : ℕ) := 6 * n - 4
noncomputable def b_seq (n : ℕ) := 2 ^ (n - 1)

def A := Finset.image a_seq (range 6).map (λ n, n + 1)
def B := Finset.image b_seq (range 6).map (λ n, n + 1)

def prob_A_inter_B : ℚ := (card (A ∩ B) : ℚ) / (card (A ∪ B) : ℚ)

theorem probability_of_intersection : prob_A_inter_B = 1 / 3 := sorry

end probability_of_intersection_l206_206626


namespace no_linear_term_implies_equal_l206_206741

theorem no_linear_term_implies_equal (m n : ℝ) (h : (x : ℝ) → (x + m) * (x - n) - x^2 - (- mn) = 0) : m = n :=
by
  sorry

end no_linear_term_implies_equal_l206_206741


namespace find_initial_interest_rate_l206_206191

-- Definitions of the initial conditions
def P1 : ℝ := 3000
def P2 : ℝ := 1499.9999999999998
def P_total : ℝ := 4500
def r2 : ℝ := 0.08
def total_annual_income : ℝ := P_total * 0.06

-- Defining the problem as a statement to prove
theorem find_initial_interest_rate (r1 : ℝ) :
  (P1 * r1) + (P2 * r2) = total_annual_income → r1 = 0.05 := by
  sorry

end find_initial_interest_rate_l206_206191


namespace rectangular_solid_surface_area_l206_206060

theorem rectangular_solid_surface_area (a b c : ℕ) (h₁ : Prime a ∨ ∃ p : ℕ, Prime p ∧ a = p + (p + 1))
                                         (h₂ : Prime b ∨ ∃ q : ℕ, Prime q ∧ b = q + (q + 1))
                                         (h₃ : Prime c ∨ ∃ r : ℕ, Prime r ∧ c = r + (r + 1))
                                         (h₄ : a * b * c = 399) :
  2 * (a * b + b * c + c * a) = 422 := 
sorry

end rectangular_solid_surface_area_l206_206060


namespace billy_reads_books_l206_206843

theorem billy_reads_books :
  let hours_per_day := 8
  let days_per_weekend := 2
  let percent_playing_games := 0.75
  let percent_reading := 0.25
  let pages_per_hour := 60
  let pages_per_book := 80
  let total_free_time_per_weekend := hours_per_day * days_per_weekend
  let time_spent_playing := total_free_time_per_weekend * percent_playing_games
  let time_spent_reading := total_free_time_per_weekend * percent_reading
  let total_pages_read := time_spent_reading * pages_per_hour
  let books_read := total_pages_read / pages_per_book
  books_read = 3 := 
by
  -- proof would go here
  sorry

end billy_reads_books_l206_206843


namespace range_x_0_l206_206727

noncomputable def f (x : ℝ) : ℝ := x^2 - Real.cos x

theorem range_x_0 (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2)) :
  f x > f (Real.pi / 6) ↔ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 6) ∪ Set.Ioi (Real.pi / 6) ∩ Set.Icc (Real.pi / 6) (Real.pi / 2) := 
  by
    sorry

end range_x_0_l206_206727


namespace simplify_expression_l206_206609

def is_real (x : ℂ) : Prop := ∃ (y : ℝ), x = y

theorem simplify_expression 
  (x y c : ℝ) 
  (i : ℂ) 
  (hi : i^2 = -1) :
  (x + i*y + c)^2 = (x^2 + c^2 - y^2 + 2 * c * x + (2 * x * y + 2 * c * y) * i) :=
by
  sorry

end simplify_expression_l206_206609


namespace interest_rate_l206_206960

theorem interest_rate (P T R : ℝ) (SI CI : ℝ) (difference : ℝ)
  (hP : P = 1700)
  (hT : T = 1)
  (hdiff : difference = 4.25)
  (hSI : SI = P * R * T / 100)
  (hCI : CI = P * ((1 + R / 200)^2 - 1))
  (hDiff : CI - SI = difference) : 
  R = 10 := sorry

end interest_rate_l206_206960


namespace remainder_3012_div_96_l206_206644

theorem remainder_3012_div_96 : 3012 % 96 = 36 :=
by 
  sorry

end remainder_3012_div_96_l206_206644


namespace total_money_spent_l206_206284

-- Definitions based on conditions
def num_bars_of_soap : Nat := 20
def weight_per_bar_of_soap : Float := 1.5
def cost_per_pound_of_soap : Float := 0.5

def num_bottles_of_shampoo : Nat := 15
def weight_per_bottle_of_shampoo : Float := 2.2
def cost_per_pound_of_shampoo : Float := 0.8

-- The theorem to prove
theorem total_money_spent :
  let cost_per_bar_of_soap := weight_per_bar_of_soap * cost_per_pound_of_soap
  let total_cost_of_soap := Float.ofNat num_bars_of_soap * cost_per_bar_of_soap
  let cost_per_bottle_of_shampoo := weight_per_bottle_of_shampoo * cost_per_pound_of_shampoo
  let total_cost_of_shampoo := Float.ofNat num_bottles_of_shampoo * cost_per_bottle_of_shampoo
  total_cost_of_soap + total_cost_of_shampoo = 41.40 := 
by
  -- proof goes here
  sorry

end total_money_spent_l206_206284


namespace water_added_l206_206824

def container_capacity : ℕ := 80
def initial_fill_percentage : ℝ := 0.5
def final_fill_percentage : ℝ := 0.75
def initial_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity
def final_fill_amount (capacity : ℕ) (percentage : ℝ) : ℝ := percentage * capacity

theorem water_added (capacity : ℕ) (initial_percentage final_percentage : ℝ) :
  final_fill_amount capacity final_percentage - initial_fill_amount capacity initial_percentage = 20 :=
by {
  sorry
}

end water_added_l206_206824


namespace part_a_impossibility_l206_206390

-- Define the number of rows and columns
def num_rows : ℕ := 20
def num_columns : ℕ := 15

-- Define a function that checks if the sum of the counts in rows and columns match the conditions
def is_possible_configuration : Prop :=
  (num_rows % 2 = 0) ∧ (num_columns % 2 = 1)

theorem part_a_impossibility : ¬ is_possible_configuration :=
by
  -- The proof for the contradiction will go here
  sorry

end part_a_impossibility_l206_206390


namespace pieces_of_green_candy_l206_206802

theorem pieces_of_green_candy (total_pieces red_pieces blue_pieces : ℝ)
  (h_total : total_pieces = 3409.7)
  (h_red : red_pieces = 145.5)
  (h_blue : blue_pieces = 785.2) :
  total_pieces - red_pieces - blue_pieces = 2479 := by
  sorry

end pieces_of_green_candy_l206_206802


namespace sin_A_is_eight_ninths_l206_206421

variable (AB AC : ℝ) (A : ℝ)

-- Given conditions
def area_triangle := 1 / 2 * AB * AC * Real.sin A = 100
def geometric_mean := Real.sqrt (AB * AC) = 15

-- Proof statement
theorem sin_A_is_eight_ninths (h1 : area_triangle AB AC A) (h2 : geometric_mean AB AC) :
  Real.sin A = 8 / 9 := sorry

end sin_A_is_eight_ninths_l206_206421


namespace one_third_of_nine_times_seven_l206_206533

theorem one_third_of_nine_times_seven : (1 / 3) * (9 * 7) = 21 := 
by
  sorry

end one_third_of_nine_times_seven_l206_206533


namespace folding_cranes_together_l206_206213

theorem folding_cranes_together (rateA rateB combined_time : ℝ)
  (hA : rateA = 1 / 30)
  (hB : rateB = 1 / 45)
  (combined_rate : ℝ := rateA + rateB)
  (h_combined_rate : combined_rate = 1 / combined_time):
  combined_time = 18 :=
by
  sorry

end folding_cranes_together_l206_206213


namespace cube_cut_problem_l206_206003

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l206_206003


namespace annual_increase_fraction_l206_206376

theorem annual_increase_fraction (InitAmt FinalAmt : ℝ) (f : ℝ) :
  InitAmt = 51200 ∧ FinalAmt = 64800 ∧ FinalAmt = InitAmt * (1 + f)^2 →
  f = 0.125 :=
by
  intros h
  sorry

end annual_increase_fraction_l206_206376


namespace cosine_F_in_triangle_DEF_l206_206280

theorem cosine_F_in_triangle_DEF
  (D E F : ℝ)
  (h_triangle : D + E + F = π)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = - (16 / 65) := by
  sorry

end cosine_F_in_triangle_DEF_l206_206280


namespace jed_gives_2_cards_every_two_weeks_l206_206744

theorem jed_gives_2_cards_every_two_weeks
  (starting_cards : ℕ)
  (cards_per_week : ℕ)
  (cards_after_4_weeks : ℕ)
  (number_of_two_week_intervals : ℕ)
  (cards_given_away_each_two_weeks : ℕ):
  starting_cards = 20 →
  cards_per_week = 6 →
  cards_after_4_weeks = 40 →
  number_of_two_week_intervals = 2 →
  (starting_cards + 4 * cards_per_week - number_of_two_week_intervals * cards_given_away_each_two_weeks = cards_after_4_weeks) →
  cards_given_away_each_two_weeks = 2 := 
by
  intros h_start h_week h_4weeks h_intervals h_eq
  sorry

end jed_gives_2_cards_every_two_weeks_l206_206744


namespace remainder_add_mod_l206_206490

theorem remainder_add_mod (n : ℕ) (h : n % 7 = 2) : (n + 1470) % 7 = 2 := 
by sorry

end remainder_add_mod_l206_206490


namespace seq_diff_five_consec_odd_avg_55_l206_206143

theorem seq_diff_five_consec_odd_avg_55 {a b c d e : ℤ} 
    (h1: a % 2 = 1) (h2: b % 2 = 1) (h3: c % 2 = 1) (h4: d % 2 = 1) (h5: e % 2 = 1)
    (h6: b = a + 2) (h7: c = a + 4) (h8: d = a + 6) (h9: e = a + 8)
    (avg_5_seq : (a + b + c + d + e) / 5 = 55) : 
    e - a = 8 := 
by
    -- proof part can be skipped with sorry
    sorry

end seq_diff_five_consec_odd_avg_55_l206_206143


namespace billy_books_read_l206_206846

def hours_per_day : ℕ := 8
def days_per_weekend : ℕ := 2
def reading_percentage : ℚ := 0.25
def pages_per_hour : ℕ := 60
def pages_per_book : ℕ := 80

theorem billy_books_read :
  let total_hours := hours_per_day * days_per_weekend in
  let reading_hours := total_hours * reading_percentage in
  let total_pages := reading_hours * pages_per_hour in
  let books_read := total_pages / pages_per_book in
  books_read = 3 :=
by
  sorry

end billy_books_read_l206_206846


namespace bobby_position_after_100_turns_l206_206847

def movement_pattern (start_pos : ℤ × ℤ) (n : ℕ) : (ℤ × ℤ) :=
  let x := start_pos.1 - ((2 * (n / 4 + 1) + 3 * (n / 4)) * ((n + 1) / 4))
  let y := start_pos.2 + ((2 * (n / 4 + 1) + 2 * (n / 4)) * ((n + 1) / 4))
  if n % 4 == 0 then (x, y)
  else if n % 4 == 1 then (x, y + 2 * ((n + 3) / 4) + 1)
  else if n % 4 == 2 then (x - 3 * ((n + 5) / 4), y + 2 * ((n + 3) / 4) + 1)
  else (x - 3 * ((n + 5) / 4) + 3, y + 2 * ((n + 3) / 4) - 2)

theorem bobby_position_after_100_turns :
  movement_pattern (10, -10) 100 = (-667, 640) :=
sorry

end bobby_position_after_100_turns_l206_206847


namespace marilyn_bottle_caps_start_l206_206445

-- Definitions based on the conditions
def initial_bottle_caps (X : ℕ) := X  -- Number of bottle caps Marilyn started with
def shared_bottle_caps := 36           -- Number of bottle caps shared with Nancy
def remaining_bottle_caps := 15        -- Number of bottle caps left after sharing

-- Theorem statement: Given the conditions, show that Marilyn started with 51 bottle caps
theorem marilyn_bottle_caps_start (X : ℕ) 
  (h1 : initial_bottle_caps X - shared_bottle_caps = remaining_bottle_caps) : 
  X = 51 := 
sorry  -- Proof omitted

end marilyn_bottle_caps_start_l206_206445


namespace original_earnings_l206_206585

variable (x : ℝ) -- John's original weekly earnings

theorem original_earnings:
  (1.20 * x = 72) → 
  (x = 60) :=
by
  intro h
  sorry

end original_earnings_l206_206585


namespace problem_statement_l206_206084

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * sqrt 3 * cos (ω * x + π / 6)

theorem problem_statement (ω : ℝ) (hx : ω = 2 ∨ ω = -2) :
  f ω (π / 3) = -3 ∨ f ω (π / 3) = 0 := by
  unfold f
  cases hx with
  | inl w_eq => sorry
  | inr w_eq => sorry

end problem_statement_l206_206084


namespace max_colors_for_valid_coloring_l206_206320

-- Define the 4x4 grid as a type synonym for a set of cells
def Grid4x4 := Fin 4 × Fin 4

-- Condition: Define a valid coloring function for a 4x4 grid
def valid_coloring (colors : ℕ) (f : Grid4x4 → Fin colors) : Prop :=
  ∀ i j : Fin 3, ∃ c : Fin colors, (f (i, j) = c ∨ f (i+1, j) = c) ∧ (f (i+1, j) = c ∨ f (i, j+1) = c)

-- The main theorem to prove
theorem max_colors_for_valid_coloring : 
  ∃ (colors : ℕ), colors = 11 ∧ ∀ f : Grid4x4 → Fin colors, valid_coloring colors f :=
sorry

end max_colors_for_valid_coloring_l206_206320


namespace rectangle_length_width_l206_206833

theorem rectangle_length_width (x y : ℝ) 
  (h1 : 2 * x + 2 * y = 16) 
  (h2 : x - y = 1) : 
  x = 4.5 ∧ y = 3.5 :=
by {
  sorry
}

end rectangle_length_width_l206_206833


namespace triangle_is_isosceles_l206_206627

theorem triangle_is_isosceles
  (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (H : ∀ n : ℕ, n > 0 → (p^n + q^n > r^n) ∧ (q^n + r^n > p^n) ∧ (r^n + p^n > q^n)) :
  p = q ∨ q = r ∨ r = p :=
by
  sorry

end triangle_is_isosceles_l206_206627


namespace count_factors_180_multiple_of_15_is_6_l206_206404

-- We define a function to check if 'n' is a factor of 'm'
def is_factor (m n : ℕ) : Prop := n ∣ m

-- We define a function to check if 'n' is a multiple of 'd'
def is_multiple (n d : ℕ) : Prop := d ∣ n

-- We define a function to collect all factors of a number 'n'
def factors (n : ℕ) : list ℕ := 
  -- We use list.filter to keep elements that satisfy is_factor n
  list.filter (λ x, is_factor n x) (list.range (n + 1))

-- We count the number of elements in a list that satisfy a predicate
def count_satisfying {α : Type*} (pred : α → Prop) [decidable_pred pred] (l : list α) : ℕ :=
  list.length (list.filter pred l)

-- Noncomputable definition to specify that counting number
noncomputable def count_positive_factors_of_180_multiple_of_15 : ℕ :=
  count_satisfying (λ x, is_multiple x 15) (factors 180)

-- We state the theorem equivalence to the proof problem
theorem count_factors_180_multiple_of_15_is_6 : count_positive_factors_of_180_multiple_of_15 = 6 :=
  sorry


end count_factors_180_multiple_of_15_is_6_l206_206404


namespace sqrt_16_eq_pm_4_l206_206153

-- Define the statement to be proven
theorem sqrt_16_eq_pm_4 : sqrt 16 = 4 ∨ sqrt 16 = -4 :=
sorry

end sqrt_16_eq_pm_4_l206_206153


namespace parabola_coefficients_sum_l206_206467

theorem parabola_coefficients_sum (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = a * (x + 3)^2 + 2) ∧
  (-6 = a * (1 + 3)^2 + 2) →
  a + b + c = -11/2 :=
by
  sorry

end parabola_coefficients_sum_l206_206467


namespace num_positive_integers_l206_206712

theorem num_positive_integers (m : ℕ) : 
  (∃ n, m^2 - 2 = n ∧ n ∣ 2002) ↔ (m = 2 ∨ m = 3 ∨ m = 4) :=
by
  sorry

end num_positive_integers_l206_206712


namespace flowers_not_roses_percentage_l206_206158

def percentage_non_roses (roses tulips daisies : Nat) : Nat :=
  let total := roses + tulips + daisies
  let non_roses := total - roses
  (non_roses * 100) / total

theorem flowers_not_roses_percentage :
  percentage_non_roses 25 40 35 = 75 :=
by
  sorry

end flowers_not_roses_percentage_l206_206158


namespace number_of_math_books_l206_206165

-- Definitions for conditions
variables (M H : ℕ)

-- Given conditions as a Lean proposition
def conditions : Prop :=
  M + H = 80 ∧ 4 * M + 5 * H = 368

-- The theorem to prove
theorem number_of_math_books (M H : ℕ) (h : conditions M H) : M = 32 :=
by sorry

end number_of_math_books_l206_206165


namespace count_positive_factors_of_180_multiple_of_15_l206_206411

statement: ℕ
statement := sorry

theorem count_positive_factors_of_180_multiple_of_15 :
  let factors_of_180 := {d ∈ (finset.range 181).filter (λ x, 180 % x = 0)}
  let multiples_of_15 := (finset.range 181).filter (λ x, x % 15 = 0)
  let desired_factors := factors_of_180 ∩ multiples_of_15
  desired_factors.card = 6 :=
by
  sorry

end count_positive_factors_of_180_multiple_of_15_l206_206411


namespace fraction_to_decimal_l206_206652

theorem fraction_to_decimal : (7 : ℚ) / 12 = 0.5833 :=
by
  sorry

end fraction_to_decimal_l206_206652


namespace combined_students_yellow_blue_l206_206764

theorem combined_students_yellow_blue {total_students blue_percent red_percent yellow_combined : ℕ} :
  total_students = 200 →
  blue_percent = 30 →
  red_percent = 40 →
  yellow_combined = (total_students * 3 / 10) + ((total_students - (total_students * 3 / 10)) * 6 / 10) →
  yellow_combined = 144 :=
by
  intros
  sorry

end combined_students_yellow_blue_l206_206764


namespace albrecht_correct_substitution_l206_206275

theorem albrecht_correct_substitution (a b : ℕ) (h : (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9) :
  (a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2) :=
by
  -- The proof will be filled in here
  sorry

end albrecht_correct_substitution_l206_206275


namespace value_of_expression_l206_206716

theorem value_of_expression
  (a b : ℝ)
  (h₁ : a = 2 + Real.sqrt 3)
  (h₂ : b = 2 - Real.sqrt 3) :
  a^2 + 2 * a * b - b * (3 * a - b) = 13 :=
by
  sorry

end value_of_expression_l206_206716


namespace travel_time_seattle_to_lasvegas_l206_206367

def distance_seattle_boise : ℝ := 640
def distance_boise_saltlakecity : ℝ := 400
def distance_saltlakecity_phoenix : ℝ := 750
def distance_phoenix_lasvegas : ℝ := 300

def speed_highway_seattle_boise : ℝ := 80
def speed_city_seattle_boise : ℝ := 35

def speed_highway_boise_saltlakecity : ℝ := 65
def speed_city_boise_saltlakecity : ℝ := 25

def speed_highway_saltlakecity_denver : ℝ := 75
def speed_city_saltlakecity_denver : ℝ := 30

def speed_highway_denver_phoenix : ℝ := 70
def speed_city_denver_phoenix : ℝ := 20

def speed_highway_phoenix_lasvegas : ℝ := 50
def speed_city_phoenix_lasvegas : ℝ := 30

def city_distance_estimate : ℝ := 10

noncomputable def total_time : ℝ :=
  let time_seattle_boise := ((distance_seattle_boise - city_distance_estimate) / speed_highway_seattle_boise) + (city_distance_estimate / speed_city_seattle_boise)
  let time_boise_saltlakecity := ((distance_boise_saltlakecity - city_distance_estimate) / speed_highway_boise_saltlakecity) + (city_distance_estimate / speed_city_boise_saltlakecity)
  let time_saltlakecity_phoenix := ((distance_saltlakecity_phoenix - city_distance_estimate) / speed_highway_saltlakecity_denver) + (city_distance_estimate / speed_city_saltlakecity_denver)
  let time_phoenix_lasvegas := ((distance_phoenix_lasvegas - city_distance_estimate) / speed_highway_phoenix_lasvegas) + (city_distance_estimate / speed_city_phoenix_lasvegas)
  time_seattle_boise + time_boise_saltlakecity + time_saltlakecity_phoenix + time_phoenix_lasvegas

theorem travel_time_seattle_to_lasvegas :
  total_time = 30.89 :=
sorry

end travel_time_seattle_to_lasvegas_l206_206367


namespace porter_previous_painting_price_l206_206948

-- definitions from the conditions
def most_recent_sale : ℕ := 44000

-- definitions for the problem statement
def sale_equation (P : ℕ) : Prop :=
  most_recent_sale = 5 * P - 1000

theorem porter_previous_painting_price (P : ℕ) (h : sale_equation P) : P = 9000 :=
by {
  sorry
}

end porter_previous_painting_price_l206_206948


namespace toy_car_production_l206_206347

theorem toy_car_production (yesterday today total : ℕ) 
  (hy : yesterday = 60)
  (ht : today = 2 * yesterday) :
  total = yesterday + today :=
by
  sorry

end toy_car_production_l206_206347


namespace probability_of_A_winning_l206_206660

-- Define the conditions
def p : ℝ := 0.6
def q : ℝ := 1 - p  -- probability of losing a set

-- Formulate the probabilities for each win scenario
def P_WW : ℝ := p * p
def P_LWW : ℝ := q * p * p
def P_WLW : ℝ := p * q * p

-- Calculate the total probability of winning the match
def total_probability : ℝ := P_WW + P_LWW + P_WLW

-- Prove that the total probability of A winning the match is 0.648
theorem probability_of_A_winning : total_probability = 0.648 :=
by
    -- Provide the calculation details
    sorry  -- replace with the actual proof steps if needed, otherwise keep sorry to skip the proof

end probability_of_A_winning_l206_206660


namespace pocket_knife_value_l206_206315

noncomputable def value_of_pocket_knife (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else
    let total_rubles := n * n
    let tens (x : ℕ) := x / 10
    let units (x : ℕ) := x % 10
    let e := units n
    let d := tens n
    let remaining := total_rubles - ((total_rubles / 10) * 10)
    if remaining = 6 then 4 else sorry

theorem pocket_knife_value (n : ℕ) : value_of_pocket_knife n = 2 := by
  sorry

end pocket_knife_value_l206_206315


namespace speed_of_first_boy_proof_l206_206973

noncomputable def speed_of_first_boy := 5.9

theorem speed_of_first_boy_proof :
  ∀ (x : ℝ) (t : ℝ) (d : ℝ),
    (d = x * t) → (d = (x - 5.6) * 35) →
    d = 10.5 →
    t = 35 →
    x = 5.9 := 
by
  intros x t d h1 h2 h3 h4
  sorry

end speed_of_first_boy_proof_l206_206973


namespace joining_fee_per_person_l206_206435

variables (F : ℝ)
variables (family_members : ℕ) (monthly_cost_per_person : ℝ) (john_yearly_payment : ℝ)

def total_cost (F : ℝ) (family_members : ℕ) (monthly_cost_per_person : ℝ) : ℝ :=
  family_members * (F + 12 * monthly_cost_per_person)

theorem joining_fee_per_person :
  (family_members = 4) →
  (monthly_cost_per_person = 1000) →
  (john_yearly_payment = 32000) →
  john_yearly_payment = 0.5 * total_cost F family_members monthly_cost_per_person →
  F = 4000 :=
by
  intros h_family h_monthly_cost h_yearly_payment h_eq
  sorry

end joining_fee_per_person_l206_206435


namespace problem_statement_l206_206600

theorem problem_statement (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) (hprod : m * n = 5000) 
  (h_m_not_div_10 : ¬ ∃ k, m = 10 * k) (h_n_not_div_10 : ¬ ∃ k, n = 10 * k) :
  m + n = 633 :=
sorry

end problem_statement_l206_206600


namespace find_number_l206_206531

theorem find_number (x : ℕ) (h : x - 18 = 3 * (86 - x)) : x = 69 :=
by
  sorry

end find_number_l206_206531


namespace percentage_of_180_out_of_360_equals_50_l206_206647

theorem percentage_of_180_out_of_360_equals_50 :
  (180 / 360 : ℚ) * 100 = 50 := 
sorry

end percentage_of_180_out_of_360_equals_50_l206_206647


namespace angle_magnification_l206_206590

theorem angle_magnification (α : ℝ) (h : α = 20) : α = 20 := by
  sorry

end angle_magnification_l206_206590


namespace inequality_unequal_positive_numbers_l206_206911

theorem inequality_unequal_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  (a + b) / 2 > (2 * a * b) / (a + b) :=
by
sorry

end inequality_unequal_positive_numbers_l206_206911


namespace find_n_l206_206008

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l206_206008


namespace inscribed_circle_radius_l206_206140

theorem inscribed_circle_radius (r : ℝ) (R : ℝ) (θ : ℝ) (tangent : ℝ) :
    θ = π / 3 →
    R = 5 →
    tangent = (5 : ℝ) * (Real.sqrt 2 - 1) →
    r * (1 + Real.sqrt 2) = R →
    r = 5 * (Real.sqrt 2 - 1) := 
by sorry

end inscribed_circle_radius_l206_206140


namespace prob_at_least_one_head_l206_206491

theorem prob_at_least_one_head (n : ℕ) (hn : n = 3) : 
  1 - (1 / (2^n)) = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_l206_206491


namespace num_divisors_60_l206_206902

theorem num_divisors_60 : (finset.filter (∣ 60) (finset.range 61)).card = 12 := 
sorry

end num_divisors_60_l206_206902


namespace factorial_sum_squares_l206_206779

-- Define the condition that 1! + 2! + ... + n! = m^2
def sum_factorials_eq_square (n m : ℕ) : Prop :=
  (Finset.range (n + 1)).sum (λ i, i!) = m^2

-- The statement we need to prove
theorem factorial_sum_squares :
  { (n, m) | sum_factorials_eq_square n m } = { (1, 1), (3, 3) } := 
sorry

end factorial_sum_squares_l206_206779


namespace positive_divisors_60_l206_206905

theorem positive_divisors_60 : ∃ n : ℕ, n = 12 ∧ (∀ d : ℕ, d ∣ 60 → d > 0 → ∃ (divisors_set : Finset ℕ), divisors_set.card = n ∧ ∀ x, x ∈ divisors_set ↔ x ∣ 60 ) :=
by
  sorry

end positive_divisors_60_l206_206905


namespace Suma_work_time_l206_206301

theorem Suma_work_time (W : ℝ) (h1 : W > 0) :
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  suma_time = 8 :=
by 
  let renu_rate := W / 8
  let combined_rate := W / 4
  let suma_rate := combined_rate - renu_rate
  let suma_time := W / suma_rate
  exact sorry

end Suma_work_time_l206_206301


namespace constant_seq_decreasing_implication_range_of_values_l206_206310

noncomputable def sequences (a b : ℕ → ℝ) := 
  (∀ n, a (n+1) = (1/2) * a n + (1/2) * b n) ∧
  (∀ n, (1/b (n+1)) = (1/2) * (1/a n) + (1/2) * (1/b n))

theorem constant_seq (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) :
  ∃ c, ∀ n, a n * b n = c :=
sorry

theorem decreasing_implication (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 > 0) (h_b1 : b 1 > 0) (h_dec : ∀ n, a (n+1) < a n) :
  a 1 > b 1 :=
sorry

theorem range_of_values (a b : ℕ → ℝ) (h_seq : sequences a b) (h_a1 : a 1 = 4) (h_b1 : b 1 = 1) :
  ∀ n ≥ 2, 2 < a n ∧ a n ≤ 5/2 :=
sorry

end constant_seq_decreasing_implication_range_of_values_l206_206310


namespace amount_b_l206_206493

variable {a b : ℚ} -- a and b are rational numbers

theorem amount_b (h1 : a + b = 1210) (h2 : (4 / 15) * a = (2 / 5) * b) : b = 484 :=
sorry

end amount_b_l206_206493


namespace binomial_expansion_evaluation_l206_206203

theorem binomial_expansion_evaluation : 
  (8 ^ 4 + 4 * (8 ^ 3) * 2 + 6 * (8 ^ 2) * (2 ^ 2) + 4 * 8 * (2 ^ 3) + 2 ^ 4) = 10000 := 
by 
  sorry

end binomial_expansion_evaluation_l206_206203


namespace probability_rectangle_not_include_shaded_l206_206657

theorem probability_rectangle_not_include_shaded :
  let num_rectangles := (1002.choose 2) * 2,
      num_rectangles_with_shaded := 501 * 501 * 2
  in (num_rectangles - num_rectangles_with_shaded) / num_rectangles = 500 / 1001 :=
by
  sorry

end probability_rectangle_not_include_shaded_l206_206657


namespace increase_in_average_l206_206331

theorem increase_in_average (A : ℤ) (avg_after_12 : ℤ) (score_12th_inning : ℤ) (A : ℤ) : 
  score_12th_inning = 75 → avg_after_12 = 64 → (11 * A + score_12th_inning = 768) → (avg_after_12 - A = 1) :=
by
  intros h_score h_avg h_total
  sorry

end increase_in_average_l206_206331


namespace interest_rate_correct_l206_206517

-- Definitions based on the problem conditions
def P : ℝ := 7000 -- Principal investment amount
def A : ℝ := 8470 -- Future value of the investment
def n : ℕ := 1 -- Number of times interest is compounded per year
def t : ℕ := 2 -- Number of years

-- The interest rate r to be proven
def r : ℝ := 0.1 -- Annual interest rate

-- Statement of the problem that needs to be proven in Lean
theorem interest_rate_correct :
  A = P * (1 + r / n)^(n * t) :=
by
  sorry

end interest_rate_correct_l206_206517


namespace sin_alpha_cos_alpha_l206_206885

theorem sin_alpha_cos_alpha {α : ℝ} (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l206_206885


namespace scarf_cost_is_10_l206_206777

-- Define the conditions as given in the problem statement
def initial_amount : ℕ := 53
def cost_per_toy_car : ℕ := 11
def num_toy_cars : ℕ := 2
def cost_of_beanie : ℕ := 14
def remaining_after_beanie : ℕ := 7

-- Calculate the cost of the toy cars
def total_cost_toy_cars : ℕ := num_toy_cars * cost_per_toy_car

-- Calculate the amount left after buying the toy cars
def amount_after_toys : ℕ := initial_amount - total_cost_toy_cars

-- Calculate the amount left after buying the beanie
def amount_after_beanie : ℕ := amount_after_toys - cost_of_beanie

-- Define the cost of the scarf
def cost_of_scarf : ℕ := amount_after_beanie - remaining_after_beanie

-- The theorem stating that cost_of_scarf is 10 dollars
theorem scarf_cost_is_10 : cost_of_scarf = 10 := by
  sorry

end scarf_cost_is_10_l206_206777


namespace hypotenuse_length_50_l206_206167

theorem hypotenuse_length_50 (a b : ℕ) (h₁ : a = 14) (h₂ : b = 48) :
  ∃ c : ℕ, c = 50 ∧ c = Nat.sqrt (a^2 + b^2) :=
by
  sorry

end hypotenuse_length_50_l206_206167


namespace volume_is_85_l206_206196

/-!
# Proof Problem
Prove that the total volume of Carl's and Kate's cubes is 85, given the conditions,
Carl has 3 cubes each with a side length of 3, and Kate has 4 cubes each with a side length of 1.
-/

-- Definitions for the problem conditions:
def volume_of_cube (s : ℕ) : ℕ := s^3

def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_of_cube s

-- Given conditions
def carls_cubes_volume : ℕ := total_volume 3 3
def kates_cubes_volume : ℕ := total_volume 4 1

-- The total volume of Carl's and Kate's cubes:
def total_combined_volume : ℕ := carls_cubes_volume + kates_cubes_volume

-- Prove the total volume is 85
theorem volume_is_85 : total_combined_volume = 85 :=
by sorry

end volume_is_85_l206_206196


namespace bacterium_probability_l206_206970

noncomputable def probability_bacterium_in_small_cup
  (total_volume : ℚ) (small_cup_volume : ℚ) (contains_bacterium : Bool) : ℚ :=
if contains_bacterium then small_cup_volume / total_volume else 0

theorem bacterium_probability
  (total_volume : ℚ) (small_cup_volume : ℚ) (bacterium_present : Bool) :
  total_volume = 2 ∧ small_cup_volume = 0.1 ∧ bacterium_present = true →
  probability_bacterium_in_small_cup 2 0.1 true = 0.05 :=
by
  intros h
  sorry

end bacterium_probability_l206_206970


namespace find_n_l206_206010

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l206_206010


namespace exponential_sum_sequence_l206_206892

noncomputable def Sn (n : ℕ) : ℝ :=
  Real.log (1 + 1 / n)

theorem exponential_sum_sequence : 
  e^(Sn 9 - Sn 6) = (20 : ℝ) / 21 := by
  sorry

end exponential_sum_sequence_l206_206892


namespace students_in_class_l206_206946

theorem students_in_class (S : ℕ) (h1 : S / 3 + 2 * S / 5 + 12 = S) : S = 45 :=
sorry

end students_in_class_l206_206946


namespace number_of_divisors_60_l206_206903

theorem number_of_divisors_60 : ∃ n : ℕ, n = 12 ∧ ∀ d : ℕ, d ∣ 60 → (d ≤ 60) :=
by 
  existsi 12
  split
  case left _ =>
    sorry
  case right _ => 
    intro d
    sorry

end number_of_divisors_60_l206_206903


namespace avg_ticket_cost_per_person_l206_206322

-- Define the conditions
def full_price : ℤ := 150
def half_price : ℤ := full_price / 2
def num_full_price_tickets : ℤ := 2
def num_half_price_tickets : ℤ := 2
def free_tickets : ℤ := 1
def total_people : ℤ := 5

-- Prove that the average cost of tickets per person is 90 yuan
theorem avg_ticket_cost_per_person : ((num_full_price_tickets * full_price + num_half_price_tickets * half_price) / total_people) = 90 := 
by 
  sorry

end avg_ticket_cost_per_person_l206_206322


namespace problem1_l206_206913

theorem problem1 (x y : ℝ) (h1 : x * (x + y) = 27) (h2 : y * (x + y) = 54) : (x + y)^2 = 81 := 
by
  sorry

end problem1_l206_206913


namespace determine_y_l206_206057

theorem determine_y (y : ℕ) : (8^5 + 8^5 + 2 * 8^5 = 2^y) → y = 17 := 
by {
  sorry
}

end determine_y_l206_206057


namespace min_y_value_l206_206935

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 16 * x + 50 * y + 64) : y ≥ 0 :=
sorry

end min_y_value_l206_206935


namespace quarters_number_l206_206318

theorem quarters_number (total_value : ℝ)
    (bills1 : ℝ := 2)
    (bill5 : ℝ := 5)
    (dimes : ℝ := 20 * 0.1)
    (nickels : ℝ := 8 * 0.05)
    (pennies : ℝ := 35 * 0.01) :
    total_value = 13 → (total_value - (bills1 + bill5 + dimes + nickels + pennies)) / 0.25 = 13 :=
by
  intro h
  have h_total := h
  sorry

end quarters_number_l206_206318


namespace angle_between_lines_is_arctan_one_third_l206_206075

theorem angle_between_lines_is_arctan_one_third
  (l1 : ∀ x y : ℝ, 2 * x - y + 1 = 0)
  (l2 : ∀ x y : ℝ, x - y - 2 = 0)
  : ∃ θ : ℝ, θ = Real.arctan (1 / 3) := 
sorry

end angle_between_lines_is_arctan_one_third_l206_206075


namespace circle_equation_l206_206211

-- Define the conditions
def chord_length_condition (a b r : ℝ) : Prop := r^2 = a^2 + 1
def arc_length_condition (b r : ℝ) : Prop := r^2 = 2 * b^2
def min_distance_condition (a b : ℝ) : Prop := a = b

-- The main theorem stating the final answer
theorem circle_equation (a b r : ℝ) (h1 : chord_length_condition a b r)
    (h2 : arc_length_condition b r) (h3 : min_distance_condition a b) :
    ((x - a)^2 + (y - a)^2 = 2) ∨ ((x + a)^2 + (y + a)^2 = 2) :=
sorry

end circle_equation_l206_206211


namespace sphere_surface_area_l206_206250

theorem sphere_surface_area
  (V : ℝ)
  (r : ℝ)
  (h : ℝ)
  (R : ℝ)
  (V_cone : V = (2 * π) / 3)
  (r_cone_base : r = 1)
  (cone_height : h = 2 * V / (π * r^2))
  (sphere_radius : R^2 - (R - h)^2 = r^2):
  4 * π * R^2 = 25 * π / 4 :=
by
  sorry

end sphere_surface_area_l206_206250


namespace sum_medians_is_64_l206_206190

noncomputable def median (l: List ℝ) : ℝ := sorry  -- Placeholder for median calculation

open List

/-- Define the scores for players A and B as lists of real numbers -/
def player_a_scores : List ℝ := sorry
def player_b_scores : List ℝ := sorry

/-- Prove that the sum of the medians of the scores lists is 64 -/
theorem sum_medians_is_64 : median player_a_scores + median player_b_scores = 64 := sorry

end sum_medians_is_64_l206_206190


namespace trapezoid_area_l206_206580

-- Definitions based on conditions
def CL_div_LD (CL LD : ℝ) : Prop := CL / LD = 1 / 4

-- The main statement we want to prove
theorem trapezoid_area (BC CD : ℝ) (h1 : BC = 9) (h2 : CD = 30) (CL LD : ℝ) (h3 : CL_div_LD CL LD) : 
  1/2 * (BC + AD) * 24 = 972 :=
sorry

end trapezoid_area_l206_206580


namespace m_is_perfect_square_l206_206934

theorem m_is_perfect_square
  (m n k : ℕ) 
  (h1 : 0 < m) 
  (h2 : 0 < n) 
  (h3 : 0 < k) 
  (h4 : 1 + m + n * Real.sqrt 3 = (2 + Real.sqrt 3) ^ (2 * k + 1)) : 
  ∃ a : ℕ, m = a ^ 2 :=
by 
  sorry

end m_is_perfect_square_l206_206934


namespace tangent_line_eqn_l206_206183

theorem tangent_line_eqn :
  ∃ k : ℝ, 
  x^2 + y^2 - 4*x + 3 = 0 → 
  (∃ x y : ℝ, (x-2)^2 + y^2 = 1 ∧ x > 2 ∧ y < 0 ∧ y = k*x) → 
  k = - (Real.sqrt 3) / 3 := 
by
  sorry

end tangent_line_eqn_l206_206183


namespace inequality_am_gm_l206_206718

theorem inequality_am_gm (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (2 * x^2) / (y + z) + (2 * y^2) / (x + z) + (2 * z^2) / (x + y) ≥ x + y + z :=
by
  sorry

end inequality_am_gm_l206_206718


namespace price_per_liter_after_discount_l206_206461

-- Define the initial conditions
def num_bottles : ℕ := 6
def liters_per_bottle : ℝ := 2
def original_total_cost : ℝ := 15
def discounted_total_cost : ℝ := 12

-- Calculate the total number of liters
def total_liters : ℝ := num_bottles * liters_per_bottle

-- Define the expected price per liter after discount
def expected_price_per_liter : ℝ := 1

-- Lean query to verify the expected price per liter
theorem price_per_liter_after_discount : (discounted_total_cost / total_liters) = expected_price_per_liter := by
  sorry

end price_per_liter_after_discount_l206_206461


namespace hyperbola_standard_equation_l206_206311

def ellipse_equation (x y : ℝ) : Prop :=
  (y^2) / 16 + (x^2) / 12 = 1

def hyperbola_equation (x y : ℝ) : Prop :=
  (y^2) / 2 - (x^2) / 2 = 1

def passes_through_point (x y : ℝ) : Prop :=
  x = 1 ∧ y = Real.sqrt 3

theorem hyperbola_standard_equation (x y : ℝ) (hx : passes_through_point x y)
  (ellipse_foci_shared : ∀ x y : ℝ, ellipse_equation x y → ellipse_equation x y)
  : hyperbola_equation x y := 
sorry

end hyperbola_standard_equation_l206_206311


namespace arithmetic_sequence_sum_l206_206977

theorem arithmetic_sequence_sum (x y : ℕ) (h₀: ∃ (n : ℕ), x = 3 + n * 4) (h₁: ∃ (m : ℕ), y = 3 + m * 4) (h₂: y = 31 - 4) (h₃: x = y - 4) : x + y = 50 := by
  sorry

end arithmetic_sequence_sum_l206_206977


namespace find_a4_b4_c4_l206_206556

theorem find_a4_b4_c4 (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 5) (h3 : a^3 + b^3 + c^3 = 15) : 
    a^4 + b^4 + c^4 = 35 := 
by 
  sorry

end find_a4_b4_c4_l206_206556


namespace value_of_fraction_l206_206646

theorem value_of_fraction :
  (16.factorial / (7.factorial * 9.factorial) = 5720 / 3) := by
  sorry

end value_of_fraction_l206_206646


namespace candy_distribution_l206_206525

theorem candy_distribution (candies : ℕ) (family_members : ℕ) (required_candies : ℤ) :
  (candies = 45) ∧ (family_members = 5) →
  required_candies = 0 :=
by sorry

end candy_distribution_l206_206525


namespace sum_interior_ninth_row_l206_206929

-- Define Pascal's Triangle and the specific conditions
def pascal_sum (n : ℕ) : ℕ := 2^(n - 1)

def sum_interior_numbers (n : ℕ) : ℕ := pascal_sum n - 2

theorem sum_interior_ninth_row :
  sum_interior_numbers 9 = 254 := 
by {
  sorry
}

end sum_interior_ninth_row_l206_206929


namespace arithmetic_seq_a4_value_l206_206614

theorem arithmetic_seq_a4_value
  (a : ℕ → ℤ)
  (h : 4 * a 3 + a 11 - 3 * a 5 = 10) :
  a 4 = 5 := 
sorry

end arithmetic_seq_a4_value_l206_206614


namespace not_in_range_g_zero_l206_206116

noncomputable def g (x: ℝ) : ℤ :=
  if x > -3 then ⌈2 / (x + 3)⌉
  else if x < -3 then ⌊2 / (x + 3)⌋
  else 0 -- g(x) is not defined at x = -3, this is a placeholder

theorem not_in_range_g_zero :
  ¬ (∃ x : ℝ, g x = 0) :=
sorry

end not_in_range_g_zero_l206_206116


namespace sub_frac_pow_eq_l206_206048

theorem sub_frac_pow_eq :
  7 - (2 / 5)^3 = 867 / 125 := by
  sorry

end sub_frac_pow_eq_l206_206048


namespace Laura_running_speed_l206_206755

noncomputable def running_speed (x : ℝ) : Prop :=
  (15 / (3 * x + 2)) + (4 / x) = 1.5 ∧ x > 0

theorem Laura_running_speed : ∃ (x : ℝ), running_speed x ∧ abs (x - 5.64) < 0.01 :=
by
  sorry

end Laura_running_speed_l206_206755


namespace mark_total_votes_l206_206129

theorem mark_total_votes (h1 : 70% = 0.70) (h2 : 100000 : ℕ) (h3 : twice := 2)
  (votes_first_area : ℕ := 0.70 * 100000)
  (votes_remaining_area : ℕ := twice * votes_first_area)
  (total_votes : ℕ := votes_first_area + votes_remaining_area) : 
  total_votes = 210000 := 
by
  sorry

end mark_total_votes_l206_206129


namespace smallest_M_value_l206_206439

theorem smallest_M_value 
  (a b c d e : ℕ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : 0 < e) 
  (h_sum : a + b + c + d + e = 2010) : 
  (∃ M, M = max (a+b) (max (b+c) (max (c+d) (d+e))) ∧ M = 671) :=
by
  sorry

end smallest_M_value_l206_206439


namespace range_of_b_over_a_l206_206738

-- Define the problem conditions and conclusion
theorem range_of_b_over_a 
  (a b c : ℝ) (A B C : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2) 
  (h_sum_angles : A + B + C = π) 
  (h_sides_relation : ∀ x, (x^2 + c^2 - a^2 - ab = 0 ↔ x = 0)) : 
  1 < b / a ∧ b / a < 2 := 
sorry

end range_of_b_over_a_l206_206738


namespace day_crew_fraction_l206_206363

-- Definitions of number of boxes per worker for day crew, and workers for day crew
variables (D : ℕ) (W : ℕ)

-- Definitions of night crew loading rate and worker ratio based on given conditions
def night_boxes_per_worker := (3 / 4 : ℚ) * D
def night_workers := (2 / 3 : ℚ) * W

-- Definition of total boxes loaded by each crew
def day_crew_total := D * W
def night_crew_total := night_boxes_per_worker D * night_workers W

-- The proof problem shows fraction loaded by day crew equals 2/3
theorem day_crew_fraction : (day_crew_total D W) / (day_crew_total D W + night_crew_total D W) = (2 / 3 : ℚ) := by
  sorry

end day_crew_fraction_l206_206363


namespace max_value_expression_l206_206288

theorem max_value_expression : 
  ∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 →
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) ≤ 256 / 243 :=
by
  intros x y z hx hy hz hsum
  sorry

end max_value_expression_l206_206288


namespace minimum_value_m_ineq_proof_l206_206085

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem minimum_value_m (x₀ : ℝ) (m : ℝ) (hx : f x₀ ≤ m) : 4 ≤ m := by
  sorry

theorem ineq_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3 * a + b = 4) : 3 ≤ 3 / b + 1 / a := by
  sorry

end minimum_value_m_ineq_proof_l206_206085


namespace second_degree_polynomial_inequality_l206_206066

def P (u v w x : ℝ) : ℝ := u * x^2 + v * x + w

theorem second_degree_polynomial_inequality 
  (u v w : ℝ) (h : ∀ a : ℝ, 1 ≤ a → P u v w (a^2 + a) ≥ a * P u v w (a + 1)) :
  u > 0 ∧ w ≤ 4 * u :=
by
  sorry

end second_degree_polynomial_inequality_l206_206066


namespace solve_for_B_l206_206484

theorem solve_for_B (B : ℕ) (h : 3 * B + 2 = 20) : B = 6 :=
by 
  -- This is just a placeholder, the proof will go here
  sorry

end solve_for_B_l206_206484


namespace number_of_positive_divisors_of_60_l206_206899

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end number_of_positive_divisors_of_60_l206_206899


namespace two_questions_exactly13_l206_206159

open Finset

variables (N A B C W R : ℕ)
variables (exactly_two_correct : ℕ) 

-- Defining the conditions
def conditions := 
  N = 40 ∧
  A = 10 ∧
  B = 13 ∧
  C = 15 ∧
  W = 15 ∧
  R = 1

-- The theorem we need to prove
theorem two_questions_exactly13 (h: conditions N A B C W R) : 
exactly_two_correct = 13 :=
by
  unfold conditions at h
  sorry

end two_questions_exactly13_l206_206159


namespace winner_exceeds_second_opponent_l206_206839

theorem winner_exceeds_second_opponent
  (total_votes : ℕ)
  (votes_winner : ℕ)
  (votes_second : ℕ)
  (votes_third : ℕ)
  (votes_fourth : ℕ) 
  (h_votes_sum : total_votes = votes_winner + votes_second + votes_third + votes_fourth)
  (h_total_votes : total_votes = 963) 
  (h_winner_votes : votes_winner = 195) 
  (h_second_votes : votes_second = 142) 
  (h_third_votes : votes_third = 116) 
  (h_fourth_votes : votes_fourth = 90) :
  votes_winner - votes_second = 53 := by
  sorry

end winner_exceeds_second_opponent_l206_206839


namespace average_of_k_l206_206239

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l206_206239


namespace min_green_beads_l206_206511

theorem min_green_beads (B R G : ℕ)
  (h_total : B + R + G = 80)
  (h_red_blue : ∀ i j, B ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < R)
  (h_green_red : ∀ i j, R ≥ 2 → i ≠ j → ∃ k, (i < k ∧ k < j ∨ j < k ∧ k < i) ∧ k < G)
  : G = 27 := 
sorry

end min_green_beads_l206_206511


namespace quadrants_I_and_II_l206_206535

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y > 3 * x
def condition2 (x y : ℝ) : Prop := y > 6 - x^2

-- Prove that any point satisfying the conditions lies in Quadrant I or II
theorem quadrants_I_and_II (x y : ℝ) (h1 : y > 3 * x) (h2 : y > 6 - x^2) : (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) :=
by
  -- The proof steps are omitted
  sorry

end quadrants_I_and_II_l206_206535


namespace doughnut_machine_completion_l206_206823

noncomputable def completion_time (start_time : ℕ) (partial_duration : ℕ) : ℕ :=
  start_time + 4 * partial_duration

theorem doughnut_machine_completion :
  let start_time := 8 * 60  -- 8:00 AM in minutes
  let partial_completion_time := 11 * 60 + 40  -- 11:40 AM in minutes
  let one_fourth_duration := partial_completion_time - start_time
  completion_time start_time one_fourth_duration = (22 * 60 + 40) := -- 10:40 PM in minutes
by
  sorry

end doughnut_machine_completion_l206_206823


namespace shoes_difference_l206_206139

theorem shoes_difference : 
  ∀ (Scott_shoes Anthony_shoes Jim_shoes : ℕ), 
  Scott_shoes = 7 → 
  Anthony_shoes = 3 * Scott_shoes → 
  Jim_shoes = Anthony_shoes - 2 → 
  Anthony_shoes - Jim_shoes = 2 :=
by
  intros Scott_shoes Anthony_shoes Jim_shoes 
  intros h1 h2 h3 
  sorry

end shoes_difference_l206_206139


namespace part1_part2_part3_l206_206251

def a (n : ℕ) : ℤ := 13 - 2 * n

theorem part1 : |a 1| + |a 2| + |a 3| = 27 := 
by {
  sorry
}

theorem part2 : |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 52 :=
by {
  sorry
}

theorem part3 (n : ℕ) : 
  if h1 : 1 ≤ n ∧ n ≤ 6 then |a 1| + |a 2| + ... + |a n| = 12 * n - n^2
  else if h2 : n ≥ 7 then |a 1| + |a 2| + ... + |a n| = n^2 - 12 * n + 72 :=
by {
  sorry
}

end part1_part2_part3_l206_206251


namespace equal_parallelogram_faces_are_rhombuses_l206_206094

theorem equal_parallelogram_faces_are_rhombuses 
  (a b c : ℝ) 
  (h: a * b = b * c ∧ b * c = a * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) :
  a = b ∧ b = c :=
sorry

end equal_parallelogram_faces_are_rhombuses_l206_206094


namespace quadratic_properties_l206_206888

theorem quadratic_properties (a b c : ℝ) (h1 : a < 0) (h2 : a * (-1 : ℝ)^2 + b * (-1 : ℝ) + c = 0) (h3 : -b = 2 * a) :
  (a - b + c = 0) ∧ 
  (∀ m : ℝ, a * m^2 + b * m + c ≤ -4 * a) ∧ 
  (∀ (x1 x2 : ℝ), (a * x1^2 + b * x1 + c + 1 = 0) ∧ (a * x2^2 + b * x2 + c + 1 = 0) ∧ x1 < x2 → x1 < -1 ∧ x2 > 3) :=
by
  sorry

end quadratic_properties_l206_206888


namespace marilyn_bottle_caps_start_l206_206446

-- Definitions based on the conditions
def initial_bottle_caps (X : ℕ) := X  -- Number of bottle caps Marilyn started with
def shared_bottle_caps := 36           -- Number of bottle caps shared with Nancy
def remaining_bottle_caps := 15        -- Number of bottle caps left after sharing

-- Theorem statement: Given the conditions, show that Marilyn started with 51 bottle caps
theorem marilyn_bottle_caps_start (X : ℕ) 
  (h1 : initial_bottle_caps X - shared_bottle_caps = remaining_bottle_caps) : 
  X = 51 := 
sorry  -- Proof omitted

end marilyn_bottle_caps_start_l206_206446


namespace prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l206_206144

theorem prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29 
  (n : ℕ) (h1 : Prime n) (h2 : 20 < n) (h3 : n < 30) (h4 : n % 8 = 5) : n = 29 := 
by
  sorry

end prime_number_between_20_and_30_with_remainder_5_when_divided_by_8_is_29_l206_206144


namespace range_of_m_l206_206253

noncomputable def f (x m : ℝ) : ℝ := |x^2 - 4| + x^2 + m * x

theorem range_of_m 
  (f_has_two_distinct_zeros : ∃ a b : ℝ, 0 < a ∧ a < b ∧ b < 3 ∧ f a m = 0 ∧ f b m = 0) :
  -14 / 3 < m ∧ m < -2 :=
sorry

end range_of_m_l206_206253


namespace maxwells_walking_speed_l206_206763

theorem maxwells_walking_speed 
    (brad_speed : ℕ) 
    (distance_between_homes : ℕ) 
    (maxwell_distance : ℕ)
    (meeting : maxwell_distance = 12)
    (brad_speed_condition : brad_speed = 6)
    (distance_between_homes_condition: distance_between_homes = 36) : 
    (maxwell_distance / (distance_between_homes - maxwell_distance) * brad_speed ) = 3 := by
  sorry

end maxwells_walking_speed_l206_206763


namespace solve_AlyoshaCube_l206_206019

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l206_206019


namespace sum_and_round_to_nearest_ten_l206_206761

/-- A function to round a number to the nearest ten -/
def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + 10 - n % 10

/-- The sum of 54 and 29 rounded to the nearest ten is 80 -/
theorem sum_and_round_to_nearest_ten : round_to_nearest_ten (54 + 29) = 80 :=
by
  sorry

end sum_and_round_to_nearest_ten_l206_206761


namespace range_of_a_l206_206150

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (2 * x - 1 < 3) ∧ (x - a < 0) → (x < a)) → (a ≤ 2) :=
by
  intro h
  sorry

end range_of_a_l206_206150


namespace total_budget_l206_206138

theorem total_budget (s_ticket : ℕ) (s_drinks_food : ℕ) (k_ticket : ℕ) (k_drinks : ℕ) (k_food : ℕ) 
  (h1 : s_ticket = 14) (h2 : s_drinks_food = 6) (h3 : k_ticket = 14) (h4 : k_drinks = 2) (h5 : k_food = 4) : 
  s_ticket + s_drinks_food + k_ticket + k_drinks + k_food = 40 := 
by
  sorry

end total_budget_l206_206138


namespace number_of_green_balls_l206_206826

theorem number_of_green_balls
  (total_balls white_balls yellow_balls red_balls purple_balls : ℕ)
  (prob : ℚ)
  (H_total : total_balls = 100)
  (H_white : white_balls = 50)
  (H_yellow : yellow_balls = 10)
  (H_red : red_balls = 7)
  (H_purple : purple_balls = 3)
  (H_prob : prob = 0.9) :
  ∃ (green_balls : ℕ), 
    (white_balls + green_balls + yellow_balls) / total_balls = prob ∧ green_balls = 30 := by
  sorry

end number_of_green_balls_l206_206826


namespace min_value_M_l206_206392

theorem min_value_M (a b : ℕ) (ha: 0 < a) (hb: 0 < b) : ∃ a b, M = 3 * a^2 - a * b^2 - 2 * b - 4 ∧ M = 2 := sorry

end min_value_M_l206_206392


namespace proof_problem_l206_206540

open Real

-- Define the problem statements as Lean hypotheses
def p : Prop := ∀ a : ℝ, exp a ≥ a + 1
def q : Prop := ∃ α β : ℝ, sin (α + β) = sin α + sin β

theorem proof_problem : p ∧ q :=
by
  sorry

end proof_problem_l206_206540


namespace sum_of_fourth_powers_is_three_times_square_l206_206462

theorem sum_of_fourth_powers_is_three_times_square (n : ℤ) (h : n ≠ 0) :
  (n - 1)^4 + n^4 + (n + 1)^4 + 10 = 3 * (n^2 + 2)^2 :=
by
  sorry

end sum_of_fourth_powers_is_three_times_square_l206_206462


namespace water_heater_ratio_l206_206164

variable (Wallace_capacity : ℕ) (Catherine_capacity : ℕ)
variable (Wallace_fullness : ℚ := 3/4) (Catherine_fullness : ℚ := 3/4)
variable (total_water : ℕ := 45)

theorem water_heater_ratio :
  Wallace_capacity = 40 →
  (Wallace_fullness * Wallace_capacity : ℚ) + (Catherine_fullness * Catherine_capacity : ℚ) = total_water →
  ((Wallace_capacity : ℚ) / (Catherine_capacity : ℚ)) = 2 :=
by
  sorry

end water_heater_ratio_l206_206164


namespace jerome_money_left_l206_206552

-- Given conditions
def half_of_money (m : ℕ) : Prop := m / 2 = 43
def amount_given_to_meg (x : ℕ) : Prop := x = 8
def amount_given_to_bianca (x : ℕ) : Prop := x = 3 * 8

-- Problem statement
theorem jerome_money_left (m : ℕ) (x : ℕ) (y : ℕ) (h1 : half_of_money m) (h2 : amount_given_to_meg x) (h3 : amount_given_to_bianca y) : m - x - y = 54 :=
sorry

end jerome_money_left_l206_206552


namespace slope_of_parallel_line_l206_206645

theorem slope_of_parallel_line (a b c : ℝ) (x y : ℝ) (h : 3 * x + 6 * y = -12):
  (∀ m : ℝ, (∀ (x y : ℝ), (3 * x + 6 * y = -12) → y = m * x + (-(12 / 6) / 6)) → m = -1/2) :=
sorry

end slope_of_parallel_line_l206_206645


namespace campsite_coloring_minimum_colors_l206_206473

-- Define the graph structure and the chromatic number.
theorem campsite_coloring_minimum_colors {G : SimpleGraph (Fin 9)} 
  (h_triangle : ∃ (a b c : Fin 9), G.adj a b ∧ G.adj b c ∧ G.adj c a) : 
  G.chromaticNumber = 3 :=
sorry

end campsite_coloring_minimum_colors_l206_206473


namespace quadratic_sum_constants_l206_206618

theorem quadratic_sum_constants (a b c : ℝ) 
  (h_eq : ∀ x, a * x^2 + b * x + c = 0 → x = -3 ∨ x = 5)
  (h_min : ∀ x, a * x^2 + b * x + c ≥ 36) 
  (h_at : a * 1^2 + b * 1 + c = 36) :
  a + b + c = 36 :=
sorry

end quadratic_sum_constants_l206_206618


namespace find_k_value_l206_206067

theorem find_k_value :
  ∃ k : ℝ, (∀ x : ℝ, 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5) ∧ 
          (∃ a b : ℝ, b - a = 8 ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → 1 ≤ x^2 - 3 * x + k ∧ x^2 - 3 * x + k ≤ 5)) ∧ 
          k = 9 / 4 :=
sorry

end find_k_value_l206_206067


namespace zero_not_in_range_of_g_l206_206115

def g (x : ℝ) : ℤ :=
  if x > -3 then
    Int.ceil (2 / (x + 3))
  else 
    Int.floor (2 / (x + 3))

theorem zero_not_in_range_of_g :
  ¬ ∃ x : ℝ, g x = 0 :=
sorry

end zero_not_in_range_of_g_l206_206115


namespace initial_candies_l206_206045

-- Define the conditions
def candies_given_older_sister : ℕ := 7
def candies_given_younger_sister : ℕ := 6
def candies_left : ℕ := 15

-- Conclude the initial number of candies
theorem initial_candies : (candies_given_older_sister + candies_given_younger_sister + candies_left) = 28 := by
  sorry

end initial_candies_l206_206045


namespace find_num_alligators_l206_206931

-- We define the conditions as given in the problem
def journey_to_delta_hours : ℕ := 4
def extra_hours : ℕ := 2
def combined_time_alligators_walked : ℕ := 46

-- We define the hypothesis in terms of Lean variables
def num_alligators_traveled_with_Paul (A : ℕ) : Prop :=
  (journey_to_delta_hours + (journey_to_delta_hours + extra_hours) * A) = combined_time_alligators_walked

-- Now the theorem statement where we prove that the number of alligators (A) is 7
theorem find_num_alligators :
  ∃ A : ℕ, num_alligators_traveled_with_Paul A ∧ A = 7 :=
by
  existsi 7
  unfold num_alligators_traveled_with_Paul
  simp
  sorry -- this is where the actual proof would go

end find_num_alligators_l206_206931


namespace infinitely_many_good_primes_infinitely_many_non_good_primes_l206_206263

def is_good_prime (p : ℕ) : Prop :=
∀ a b : ℕ, a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p]

theorem infinitely_many_good_primes :
  ∃ᶠ p in at_top, is_good_prime p := sorry

theorem infinitely_many_non_good_primes :
  ∃ᶠ p in at_top, ¬ is_good_prime p := sorry

end infinitely_many_good_primes_infinitely_many_non_good_primes_l206_206263


namespace Jackson_missed_one_wednesday_l206_206583

theorem Jackson_missed_one_wednesday (weeks total_sandwiches missed_fridays sandwiches_eaten : ℕ) 
  (h1 : weeks = 36)
  (h2 : total_sandwiches = 2 * weeks)
  (h3 : missed_fridays = 2)
  (h4 : sandwiches_eaten = 69) :
  (total_sandwiches - missed_fridays - sandwiches_eaten) / 2 = 1 :=
by
  -- sorry to skip the proof.
  sorry

end Jackson_missed_one_wednesday_l206_206583


namespace fraction_irreducible_l206_206298

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end fraction_irreducible_l206_206298


namespace other_number_l206_206964

theorem other_number (a b : ℝ) (h : a = 0.650) (h2 : a = b + 0.525) : b = 0.125 :=
sorry

end other_number_l206_206964


namespace divisor_is_36_l206_206272

theorem divisor_is_36
  (Dividend Quotient Remainder : ℕ)
  (h1 : Dividend = 690)
  (h2 : Quotient = 19)
  (h3 : Remainder = 6)
  (h4 : Dividend = (Divisor * Quotient) + Remainder) :
  Divisor = 36 :=
sorry

end divisor_is_36_l206_206272


namespace solve_inequality_l206_206610

theorem solve_inequality (a x : ℝ) :
  (a = 0 → x < 1) ∧
  (a ≠ 0 → ((a > 0 → (a-1)/a < x ∧ x < 1) ∧
            (a < 0 → (x < 1 ∨ x > (a-1)/a)))) :=
by
  sorry

end solve_inequality_l206_206610


namespace polyhedron_volume_l206_206428

-- Define the properties of the polygons
def isosceles_right_triangle (a : ℝ) := a ≠ 0 ∧ ∀ (x y : ℝ), x = y

def square (side : ℝ) := side = 2

def equilateral_triangle (side : ℝ) := side = 2 * Real.sqrt 2

-- Define the conditions
def condition_AE : Prop := isosceles_right_triangle 2
def condition_B : Prop := square 2
def condition_C : Prop := square 2
def condition_D : Prop := square 2
def condition_G : Prop := equilateral_triangle (2 * Real.sqrt 2)

-- Define the polyhedron volume calculation problem
theorem polyhedron_volume (hA : condition_AE) (hE : condition_AE) (hF : condition_AE) (hB : condition_B) (hC : condition_C) (hD : condition_D) (hG : condition_G) : 
  ∃ V : ℝ, V = 16 := 
sorry

end polyhedron_volume_l206_206428


namespace mixed_number_division_l206_206688

theorem mixed_number_division :
  (5 + 1 / 2) / (2 / 11) = 121 / 4 :=
by sorry

end mixed_number_division_l206_206688


namespace subset_A_B_l206_206108

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l206_206108


namespace dice_composite_probability_l206_206915

theorem dice_composite_probability :
  let total_outcomes := (8:ℕ)^6
  let non_composite_outcomes := 1 + 4 * 6 
  let composite_probability := 1 - (non_composite_outcomes / total_outcomes) 
  composite_probability = 262119 / 262144 := by
  sorry

end dice_composite_probability_l206_206915


namespace sufficient_but_not_necessary_condition_l206_206632

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  ((∀ x : ℝ, (1 < x) → (x^2 - m * x + 1 > 0)) ↔ (-2 < m ∧ m < 2)) :=
sorry

end sufficient_but_not_necessary_condition_l206_206632


namespace aaron_ends_up_with_24_cards_l206_206682

def initial_cards_aaron : Nat := 5
def found_cards_aaron : Nat := 62
def lost_cards_aaron : Nat := 15
def given_cards_to_arthur : Nat := 28

def final_cards_aaron (initial: Nat) (found: Nat) (lost: Nat) (given: Nat) : Nat :=
  initial + found - lost - given

theorem aaron_ends_up_with_24_cards :
  final_cards_aaron initial_cards_aaron found_cards_aaron lost_cards_aaron given_cards_to_arthur = 24 := by
  sorry

end aaron_ends_up_with_24_cards_l206_206682


namespace total_circle_area_within_triangle_l206_206219

-- Define the sides of the triangle
def triangle_sides : Prop := ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5

-- Define the radii and center of the circles at each vertex of the triangle
def circle_centers_and_radii : Prop := ∃ (r : ℝ) (A B C : ℝ × ℝ), r = 1

-- The formal statement that we need to prove:
theorem total_circle_area_within_triangle :
  triangle_sides ∧ circle_centers_and_radii → 
  (total_area_of_circles_within_triangle = π / 2) := sorry

end total_circle_area_within_triangle_l206_206219


namespace trigonometric_identity_eq_neg_one_l206_206852

theorem trigonometric_identity_eq_neg_one :
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180)) = -1 :=
by
  -- Variables needed for hypotheses
  have h₁ : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₂ : Real.sin (60 * Real.pi / 180) = sqrt 3 / 2 := sorry
  have h₃ : Real.cos (60 * Real.pi / 180) = 1 / 2 := sorry
  have h₄ : Real.sin (30 * Real.pi / 180) = 1 / 2 := sorry
  -- Main proof
  sorry

end trigonometric_identity_eq_neg_one_l206_206852


namespace cases_needed_to_raise_funds_l206_206597

-- Define conditions as lemmas that will be used in the main theorem.
lemma packs_per_case : ℕ := 3
lemma muffins_per_pack : ℕ := 4
lemma muffin_price : ℕ := 2
lemma fundraising_goal : ℕ := 120

-- Calculate muffins per case
noncomputable def muffins_per_case : ℕ := packs_per_case * muffins_per_pack

-- Calculate money earned per case
noncomputable def money_per_case : ℕ := muffins_per_case * muffin_price

-- The main theorem to prove the number of cases needed
theorem cases_needed_to_raise_funds : 
  (fundraising_goal / money_per_case) = 5 :=
by
  sorry

end cases_needed_to_raise_funds_l206_206597


namespace sector_area_max_angle_l206_206080

theorem sector_area_max_angle (r : ℝ) (θ : ℝ) (h : 0 < r ∧ r < 10) 
  (H : 2 * r + r * θ = 20) : θ = 2 :=
by
  sorry

end sector_area_max_angle_l206_206080


namespace difference_in_profit_l206_206767

def records := 300
def price_sammy := 4
def price_bryan_two_thirds := 6
def price_bryan_one_third := 1
def price_christine_thirty := 10
def price_christine_remaining := 3

def profit_sammy := records * price_sammy
def profit_bryan := ((records * 2 / 3) * price_bryan_two_thirds) + ((records * 1 / 3) * price_bryan_one_third)
def profit_christine := (30 * price_christine_thirty) + ((records - 30) * price_christine_remaining)

theorem difference_in_profit : 
  max profit_sammy (max profit_bryan profit_christine) - min profit_sammy (min profit_bryan profit_christine) = 190 :=
by
  sorry

end difference_in_profit_l206_206767


namespace probability_of_blank_l206_206921

-- Definitions based on conditions
def num_prizes : ℕ := 10
def num_blanks : ℕ := 25
def total_outcomes : ℕ := num_prizes + num_blanks

-- Statement of the proof problem
theorem probability_of_blank : (num_blanks / total_outcomes : ℚ) = 5 / 7 :=
by {
  sorry
}

end probability_of_blank_l206_206921


namespace moles_of_magnesium_l206_206402

-- Assuming the given conditions as hypotheses
variables (Mg CO₂ MgO C : ℕ)

-- Theorem statement
theorem moles_of_magnesium (h1 : 2 * Mg + CO₂ = 2 * MgO + C) 
                           (h2 : MgO = Mg) 
                           (h3 : CO₂ = 1) 
                           : Mg = 2 :=
by sorry  -- Proof to be provided

end moles_of_magnesium_l206_206402


namespace binary_to_decimal_11011_l206_206871

-- Statement of the theorem
theorem binary_to_decimal_11011 : Nat.ofDigits 2 [1, 1, 0, 1, 1] = 27 := sorry

end binary_to_decimal_11011_l206_206871


namespace range_of_m_l206_206265

noncomputable def f (x m : ℝ) : ℝ := x^2 - x + m * (2 * x + 1)

theorem range_of_m (m : ℝ) : (∀ x > 1, 0 < 2 * x + (2 * m - 1)) ↔ (m ≥ -1/2) := by
  sorry

end range_of_m_l206_206265


namespace exists_four_digit_number_sum_digits_14_divisible_by_14_l206_206696

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 1000) % 10 + (n / 100 % 10) % 10 + (n / 10 % 10) % 10 + (n % 10)

theorem exists_four_digit_number_sum_digits_14_divisible_by_14 :
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ sum_of_digits n = 14 ∧ n % 14 = 0 :=
sorry

end exists_four_digit_number_sum_digits_14_divisible_by_14_l206_206696


namespace range_of_c_l206_206758

theorem range_of_c (x y c : ℝ) (h1 : x^2 + (y - 2)^2 = 1) (h2 : x^2 + y^2 + c ≤ 0) : c ≤ -9 :=
by
  -- Proof goes here
  sorry

end range_of_c_l206_206758


namespace longest_side_in_ratio_5_6_7_l206_206471

theorem longest_side_in_ratio_5_6_7 (x : ℕ) (h : 5 * x + 6 * x + 7 * x = 720) : 7 * x = 280 := 
by
  sorry

end longest_side_in_ratio_5_6_7_l206_206471


namespace yellow_balls_are_24_l206_206571

theorem yellow_balls_are_24 (x y z : ℕ) (h1 : x + y + z = 68) 
                             (h2 : y = 2 * x) (h3 : 3 * z = 4 * y) : y = 24 :=
by
  sorry

end yellow_balls_are_24_l206_206571


namespace Randy_trip_distance_l206_206605

theorem Randy_trip_distance (x : ℝ) (h1 : x = 4 * (x / 4 + 30 + x / 6)) : x = 360 / 7 :=
by
  have h2 : x = ((3 * x + 36 * 30 + 2 * x) / 12) := sorry
  have h3 : x = (5 * x / 12 + 30) := sorry
  have h4 : 30 = x - (5 * x / 12) := sorry
  have h5 : 30 = 7 * x / 12 := sorry
  have h6 : x = (12 * 30) / 7 := sorry
  have h7 : x = 360 / 7 := sorry
  exact h7

end Randy_trip_distance_l206_206605


namespace initial_cats_l206_206480

theorem initial_cats (C : ℕ) (h1 : 36 + 12 - 20 + C = 57) : C = 29 :=
by
  sorry

end initial_cats_l206_206480


namespace cricketer_runs_l206_206988

theorem cricketer_runs (R x : ℝ) : 
  (R / 85 = 12.4) →
  ((R + x) / 90 = 12.0) →
  x = 26 := 
by
  sorry

end cricketer_runs_l206_206988


namespace multiply_correct_l206_206063

theorem multiply_correct : 2.4 * 0.2 = 0.48 := by
  sorry

end multiply_correct_l206_206063


namespace cost_of_seven_CDs_l206_206483

theorem cost_of_seven_CDs (cost_per_two : ℝ) (h1 : cost_per_two = 32) : (7 * (cost_per_two / 2)) = 112 :=
by
  sorry

end cost_of_seven_CDs_l206_206483


namespace tenth_term_of_sequence_l206_206785

-- Define the first term and the common difference
def a1 : ℤ := 10
def d : ℤ := -2

-- Define the nth term of the arithmetic sequence
def a_n (n : ℕ) : ℤ := a1 + d * (n - 1)

-- State the theorem about the 10th term
theorem tenth_term_of_sequence : a_n 10 = -8 := by
  -- Skip the proof
  sorry

end tenth_term_of_sequence_l206_206785


namespace students_who_like_yellow_and_blue_l206_206765

/-- Problem conditions -/
def total_students : ℕ := 200
def percentage_blue : ℕ := 30
def percentage_red_among_not_blue : ℕ := 40

/-- We need to prove the following statement: -/
theorem students_who_like_yellow_and_blue :
  let num_blue := (percentage_blue * total_students) / 100 in
  let num_not_blue := total_students - num_blue in
  let num_red := (percentage_red_among_not_blue * num_not_blue) / 100 in
  let num_yellow := num_not_blue - num_red in
  num_yellow + num_blue = 144 :=
by
  sorry

end students_who_like_yellow_and_blue_l206_206765


namespace Jessie_l206_206584

theorem Jessie's_friends (total_muffins : ℕ) (muffins_per_person : ℕ) (num_people : ℕ) :
  total_muffins = 20 → muffins_per_person = 4 → num_people = total_muffins / muffins_per_person → num_people - 1 = 4 :=
by
  intros h1 h2 h3
  sorry

end Jessie_l206_206584


namespace units_digit_odd_product_l206_206976

theorem units_digit_odd_product (l : List ℕ) (h_odds : ∀ n ∈ l, n % 2 = 1) :
  (∀ x ∈ l, x % 10 = 5) ↔ (5 ∈ l) := by
  sorry

end units_digit_odd_product_l206_206976


namespace units_digit_product_is_2_l206_206489

def units_digit_product : ℕ := 
  (10 * 11 * 12 * 13 * 14 * 15 * 16) / 800 % 10

theorem units_digit_product_is_2 : units_digit_product = 2 := 
by
  sorry

end units_digit_product_is_2_l206_206489


namespace cube_decomposition_l206_206015

theorem cube_decomposition (n s : ℕ) (h1 : n > s) (h2 : n^3 - s^3 = 152) : n = 6 := 
by
  sorry

end cube_decomposition_l206_206015


namespace ducks_counted_l206_206200

theorem ducks_counted (x y : ℕ) (h1 : x + y = 300) (h2 : 2 * x + 4 * y = 688) : x = 256 :=
by
  sorry

end ducks_counted_l206_206200


namespace average_of_k_l206_206240

theorem average_of_k (r1 r2 : ℕ) (h : r1 * r2 = 24) : 
  r1 + r2 = 25 ∨ r1 + r2 = 14 ∨ r1 + r2 = 11 ∨ r1 + r2 = 10 → 
  (25 + 14 + 11 + 10) / 4 = 15 :=
  by sorry

end average_of_k_l206_206240


namespace find_added_number_l206_206919

theorem find_added_number (R X : ℕ) (hR : R = 45) (h : 2 * (2 * R + X) = 188) : X = 4 :=
by 
  -- We would normally provide the proof here
  sorry  -- We skip the proof as per the instructions

end find_added_number_l206_206919


namespace work_done_in_five_days_l206_206981

-- Define the work rates of A, B, and C
def work_rate_A : ℚ := 1 / 11
def work_rate_B : ℚ := 1 / 5
def work_rate_C : ℚ := 1 / 55

-- Define the work done in a cycle of 2 days
def work_one_cycle : ℚ := (work_rate_A + work_rate_B) + (work_rate_A + work_rate_C)

-- The total work needed to be done is 1
def total_work : ℚ := 1

-- The number of days in a cycle of 2 days
def days_per_cycle : ℕ := 2

-- Proving that the work will be done in exactly 5 days
theorem work_done_in_five_days :
  ∃ n : ℕ, n = 5 →
  n * (work_rate_A + work_rate_B) + (n-1) * (work_rate_A + work_rate_C) = total_work :=
by
  -- Sorry to skip the detailed proof steps
  sorry

end work_done_in_five_days_l206_206981


namespace age_difference_l206_206269

theorem age_difference (A B : ℕ) (h1 : B = 37) (h2 : A + 10 = 2 * (B - 10)) : A - B = 7 :=
by
  sorry

end age_difference_l206_206269


namespace range_of_a_l206_206135

-- Define the propositions p and q
def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

def q (a : ℝ) : Prop :=
  a > 1

-- Translate the problem to a Lean 4 statement
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → a ∈ Set.Icc (-2 : ℝ) 1 ∪ Set.Ici 2 :=
by
  sorry

end range_of_a_l206_206135


namespace total_combined_area_l206_206184

-- Definition of the problem conditions
def base_parallelogram : ℝ := 20
def height_parallelogram : ℝ := 4
def base_triangle : ℝ := 20
def height_triangle : ℝ := 2

-- Given the conditions, we want to prove:
theorem total_combined_area :
  (base_parallelogram * height_parallelogram) + (0.5 * base_triangle * height_triangle) = 100 :=
by
  sorry  -- proof goes here

end total_combined_area_l206_206184


namespace remainder_problem_l206_206049

def rem (x y : ℚ) := x - y * (⌊x / y⌋ : ℤ)

theorem remainder_problem :
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  rem x y = (-19 : ℚ) / 63 :=
by
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  sorry

end remainder_problem_l206_206049


namespace find_m_value_l206_206436

theorem find_m_value (m a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℝ)
  (h1 : (x + m)^9 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + a_7 * (x + 1)^7 + 
  a_8 * (x + 1)^8 + a_9 * (x + 1)^9)
  (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 + a_6 - a_7 + a_8 - a_9 = 3^9) :
  m = 4 :=
by
  sorry

end find_m_value_l206_206436


namespace alcohol_concentration_l206_206187

theorem alcohol_concentration 
  (x : ℝ) -- concentration of alcohol in the first vessel (as a percentage)
  (h1 : 0 ≤ x ∧ x ≤ 100) -- percentage is between 0 and 100
  (h2 : (x / 100) * 2 + (55 / 100) * 6 = (37 / 100) * 10) -- given condition for concentration balance
  : x = 20 :=
sorry

end alcohol_concentration_l206_206187


namespace hexagon_inequality_l206_206437

variables {Point : Type} [MetricSpace Point]

-- Definitions of points and distances
variables (A B C D E F G H : Point) 
variables (dist : Point → Point → ℝ)
variables (angle : Point → Point → Point → ℝ)

-- Conditions
variables (hABCDEF : ConvexHexagon A B C D E F)
variables (hAB_BC_CD : dist A B = dist B C ∧ dist B C = dist C D)
variables (hDE_EF_FA : dist D E = dist E F ∧ dist E F = dist F A)
variables (hBCD_60 : angle B C D = 60)
variables (hEFA_60 : angle E F A = 60)
variables (hAGB_120 : angle A G B = 120)
variables (hDHE_120 : angle D H E = 120)

-- Objective statement
theorem hexagon_inequality : 
  dist A G + dist G B + dist G H + dist D H + dist H E ≥ dist C F :=
sorry

end hexagon_inequality_l206_206437


namespace probability_of_sequence_l206_206822

namespace MarbleBagProblem

def initial_bag := {total := 12, red := 4, white := 6, blue := 2}

def first_draw_red_event (bag : initial_bag) : Prop :=
  4 / 12 = 1 / 3

def second_draw_white_event (bag : initial_bag) (first_red_drawn : Prop) : Prop :=
  6 / 11 = 6 / 11

def third_draw_blue_event (bag : initial_bag) (first_red_drawn second_white_drawn : Prop) : Prop :=
  2 / 10 = 1 / 5

theorem probability_of_sequence :
  first_draw_red_event initial_bag ∧
  second_draw_white_event initial_bag (first_draw_red_event initial_bag) ∧
  third_draw_blue_event initial_bag (first_draw_red_event initial_bag) (second_draw_white_event initial_bag (first_draw_red_event initial_bag)) →
  ∃ p : ℚ, p = 2 / 55 :=
by
  intros h
  sorry

end MarbleBagProblem

end probability_of_sequence_l206_206822


namespace monotonic_decreasing_interval_l206_206622

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem monotonic_decreasing_interval :
  {x : ℝ | x > 0} ∩ {x : ℝ | deriv f x < 0} = {x : ℝ | x > Real.exp 1} :=
by sorry

end monotonic_decreasing_interval_l206_206622


namespace fraction_of_pizza_peter_ate_l206_206599

theorem fraction_of_pizza_peter_ate (total_slices : ℕ) (peter_slices : ℕ) (shared_slices : ℚ) 
  (pizza_fraction : ℚ) : 
  total_slices = 16 → 
  peter_slices = 2 → 
  shared_slices = 1/3 → 
  pizza_fraction = peter_slices / total_slices + (1 / 2) * shared_slices / total_slices → 
  pizza_fraction = 13 / 96 :=
by 
  intros h1 h2 h3 h4
  -- to be proved later
  sorry

end fraction_of_pizza_peter_ate_l206_206599


namespace avg_k_for_polynomial_roots_l206_206245

-- Define the given polynomial and the conditions for k

def avg_of_distinct_ks : ℚ :=
  let ks := {k : ℕ | ∃ (r1 r2 : ℕ), r1 + r2 = k ∧ r1 * r2 = 24 ∧ r1 > 0 ∧ r2 > 0} in
  ∑ k in ks.to_finset, k / ks.card

theorem avg_k_for_polynomial_roots : avg_of_distinct_ks = 15 := by
  sorry

end avg_k_for_polynomial_roots_l206_206245


namespace probability_heads_odd_l206_206678

theorem probability_heads_odd (n : ℕ) (p : ℚ) (Q : ℕ → ℚ) (h : p = 3/4) (h_rec : ∀ n, Q (n + 1) = p * (1 - Q n) + (1 - p) * Q n) :
  Q 40 = 1/2 * (1 - 1/4^40) := 
sorry

end probability_heads_odd_l206_206678


namespace assignments_divisible_by_17_l206_206671

/-- Given 17 workers and each worker must be part of some brigades where each brigade is a
    contiguous group of at least 2 workers, we need to assign leaders to brigades such
    that each worker is a leader of some brigades and the number of assignments for each worker 
    is divisible by 4. Prove that the number of possible ways to do this is divisible by 17. -/
theorem assignments_divisible_by_17 :
  ∃ (assignments : Fin 17 → Fin 4 → Fin 17), 
    (∀ i : Fin 17, (∑ j : Fin 4, 1) % 4 = 0) →
    (∃ k : Fin 17, assignments k 0 + assignments k 1 + assignments k 2 + assignments k 3 = 0) ∧
    (∃ n : ℕ, n % 17 = 0) :=
by
  sorry

end assignments_divisible_by_17_l206_206671


namespace min_k_disjoint_pairs_l206_206807

open Finset

def friendly_pair {α : Type*} (rel : α → α → Prop) (x y : α) : Prop :=
rel x y

def non_friendly_pair {α : Type*} (rel : α → α → Prop) (x y : α) : Prop :=
¬rel x y

def exists_disjoint_pairs_of_size {α : Type*} (s : Finset α) (rel : α → α → Prop) (k : ℕ) : Prop :=
∃ (p : Finset (Finset α)),
  p.card = k ∧
  (∀ t ∈ p, t.card = 2 ∧ (∃ x y, t = {x, y} ∧ rel x y)) ∧
  pairwise_disjoint id p

theorem min_k_disjoint_pairs (m n : ℕ) : ∃ k : ℕ, ∀ (s : Finset (Σ i, Type) ) (rel : α → α → Prop), 
  s.card >= k → (exists_disjoint_pairs_of_size {a // a ∈ s} rel m ∨ exists_disjoint_pairs_of_size {a // a ∈ s} (non_friendly_pair rel) n) :=
begin
  use 2 * m + n - 1,
  sorry
end

end min_k_disjoint_pairs_l206_206807


namespace prob1_prob2_l206_206884

-- Define lines l1 and l2
def l1 (x y m : ℝ) : Prop := x + m * y + 1 = 0
def l2 (x y m : ℝ) : Prop := (m - 3) * x - 2 * y + (13 - 7 * m) = 0

-- Perpendicular condition
def perp_cond (m : ℝ) : Prop := 1 * (m - 3) - 2 * m = 0

-- Parallel condition
def parallel_cond (m : ℝ) : Prop := m * (m - 3) + 2 = 0

-- Distance between parallel lines when m = 1
def distance_between_parallel_lines (d : ℝ) : Prop := d = 2 * Real.sqrt 2

-- Problem 1: Prove that if l1 ⊥ l2, then m = -3
theorem prob1 (m : ℝ) (h : perp_cond m) : m = -3 := sorry

-- Problem 2: Prove that if l1 ∥ l2, the distance d is 2√2
theorem prob2 (m : ℝ) (h1 : parallel_cond m) (d : ℝ) (h2 : m = 1 ∨ m = -2) (h3 : m = 1) (h4 : distance_between_parallel_lines d) : d = 2 * Real.sqrt 2 := sorry

end prob1_prob2_l206_206884


namespace lisa_total_miles_flown_l206_206940

variable (distance_per_trip : ℝ := 256.0)
variable (number_of_trips : ℝ := 32.0)

theorem lisa_total_miles_flown : distance_per_trip * number_of_trips = 8192.0 := by
  sorry

end lisa_total_miles_flown_l206_206940


namespace total_votes_proof_l206_206128

variable (total_voters first_area_percent votes_first_area votes_remaining_area votes_total : ℕ)

-- Define conditions
def first_area_votes_condition : Prop :=
  votes_first_area = (total_voters * first_area_percent) / 100

def remaining_area_votes_condition : Prop :=
  votes_remaining_area = 2 * votes_first_area

def total_votes_condition : Prop :=
  votes_total = votes_first_area + votes_remaining_area

-- Main theorem to prove
theorem total_votes_proof (h1: first_area_votes_condition) (h2: remaining_area_votes_condition) (h3: total_votes_condition) :
  votes_total = 210000 :=
by
  sorry

end total_votes_proof_l206_206128


namespace yellow_tint_percentage_l206_206830

theorem yellow_tint_percentage {V₀ V₁ V_t red_pct yellow_pct : ℝ} 
  (hV₀ : V₀ = 40)
  (hRed : red_pct = 0.20)
  (hYellow : yellow_pct = 0.25)
  (hAdd : V₁ = 10) :
  (yellow_pct * V₀ + V₁) / (V₀ + V₁) = 0.40 :=
by
  sorry

end yellow_tint_percentage_l206_206830


namespace inequality_may_not_hold_l206_206912

theorem inequality_may_not_hold (m n : ℝ) (h : m > n) : ¬ (m^2 > n^2) :=
by
  -- Leaving the proof out according to the instructions.
  sorry

end inequality_may_not_hold_l206_206912


namespace sequence_exists_for_all_k_l206_206709

theorem sequence_exists_for_all_k (n : ℕ) :
  ∀ k : ℕ, (k ∈ {1, 2, ..., n}) ↔ (∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i < n, x i > 0) ∧ (∃ i, x(i) = k)) :=
by
  sorry

end sequence_exists_for_all_k_l206_206709


namespace parallel_vectors_sufficiency_l206_206551

noncomputable def parallel_vectors_sufficiency_problem (a b : ℝ × ℝ) (x : ℝ) : Prop :=
a = (1, x) ∧ b = (x, 4) →
(x = 2 → ∃ k : ℝ, k • a = b) ∧ (∃ k : ℝ, k • a = b → x = 2 ∨ x = -2)

theorem parallel_vectors_sufficiency (x : ℝ) :
  parallel_vectors_sufficiency_problem (1, x) (x, 4) x :=
sorry

end parallel_vectors_sufficiency_l206_206551


namespace sum_infinite_series_l206_206862

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end sum_infinite_series_l206_206862


namespace trigonometric_expression_value_l206_206856

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l206_206856


namespace factors_of_180_multiples_of_15_l206_206405

theorem factors_of_180_multiples_of_15 :
  (finset.filter (λ x : ℕ, x % 15 = 0) (finset.filter (λ x : ℕ, 180 % x = 0) (finset.range (180 + 1)))).card = 6 :=
sorry

end factors_of_180_multiples_of_15_l206_206405


namespace brown_beads_initial_l206_206800

theorem brown_beads_initial (B : ℕ) 
  (h1 : 1 = 1) -- There is 1 green bead in the container.
  (h2 : 3 = 3) -- There are 3 red beads in the container.
  (h3 : 4 = 4) -- Tom left 4 beads in the container.
  (h4 : 2 = 2) -- Tom took out 2 beads.
  (h5 : 6 = 2 + 4) -- Total initial beads before Tom took any out.
  : B = 2 := sorry

end brown_beads_initial_l206_206800


namespace cube_cut_problem_l206_206001

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l206_206001


namespace machine_produces_480_cans_in_8_hours_l206_206780

def cans_produced_in_interval : ℕ := 30
def interval_duration_minutes : ℕ := 30
def hours_worked : ℕ := 8
def minutes_in_hour : ℕ := 60

theorem machine_produces_480_cans_in_8_hours :
  (hours_worked * (minutes_in_hour / interval_duration_minutes) * cans_produced_in_interval) = 480 := by
  sorry

end machine_produces_480_cans_in_8_hours_l206_206780


namespace sours_total_l206_206776

variable (c l o T : ℕ)

axiom cherry_sours : c = 32
axiom ratio_cherry_lemon : 4 * l = 5 * c
axiom orange_sours_ratio : o = 25 * T / 100
axiom total_sours : T = c + l + o

theorem sours_total :
  T = 96 :=
by
  sorry

end sours_total_l206_206776


namespace correct_judgment_l206_206289

open Real

def period_sin2x (T : ℝ) : Prop := ∀ x, sin (2 * x) = sin (2 * (x + T))
def smallest_positive_period_sin2x : Prop := ∃ T > 0, period_sin2x T ∧ ∀ T' > 0, period_sin2x T' → T ≤ T'
def smallest_positive_period_sin2x_is_pi : Prop := ∃ T, smallest_positive_period_sin2x ∧ T = π

def symmetry_cosx (L : ℝ) : Prop := ∀ x, cos (L - x) = cos (L + x)
def symmetry_about_line_cosx (L : ℝ) : Prop := L = π / 2

def p : Prop := smallest_positive_period_sin2x_is_pi
def q : Prop := symmetry_about_line_cosx (π / 2)

theorem correct_judgment : ¬ (p ∧ q) :=
by 
  sorry

end correct_judgment_l206_206289


namespace f_positive_for_specific_a_l206_206937

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x * Real.log x

theorem f_positive_for_specific_a (x : ℝ) (h : x > 0) :
  f x (Real.exp 3 / 4) > 0 := sorry

end f_positive_for_specific_a_l206_206937


namespace y_squared_plus_three_y_is_perfect_square_l206_206379

theorem y_squared_plus_three_y_is_perfect_square (y : ℕ) :
  (∃ x : ℕ, y^2 + 3^y = x^2) ↔ y = 1 ∨ y = 3 := 
by
  sorry

end y_squared_plus_three_y_is_perfect_square_l206_206379


namespace trigonometric_expression_value_l206_206857

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end trigonometric_expression_value_l206_206857


namespace container_capacity_in_liters_l206_206932

-- Defining the conditions
def portions : Nat := 10
def portion_size_ml : Nat := 200

-- Statement to prove
theorem container_capacity_in_liters : (portions * portion_size_ml / 1000 = 2) :=
by 
  sorry

end container_capacity_in_liters_l206_206932


namespace square_root_of_16_is_pm_4_l206_206156

theorem square_root_of_16_is_pm_4 : { x : ℝ | x^2 = 16 } = {4, -4} :=
sorry

end square_root_of_16_is_pm_4_l206_206156


namespace sum_series_eq_4_l206_206691

theorem sum_series_eq_4 : 
  (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 4 := 
by
  sorry

end sum_series_eq_4_l206_206691


namespace minimize_fencing_l206_206595

def area_requirement (w : ℝ) : Prop :=
  2 * (w * w) ≥ 800

def length_twice_width (l w : ℝ) : Prop :=
  l = 2 * w

def perimeter (w l : ℝ) : ℝ :=
  2 * l + 2 * w

theorem minimize_fencing (w l : ℝ) (h1 : area_requirement w) (h2 : length_twice_width l w) :
  w = 20 ∧ l = 40 :=
by
  sorry

end minimize_fencing_l206_206595


namespace find_f_107_5_l206_206124

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : ∀ x, f x = f (-x)
axiom func_eq : ∀ x, f (x + 3) = - (1 / f x)
axiom cond_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x

theorem find_f_107_5 : f 107.5 = 1 / 10 := by {
  sorry
}

end find_f_107_5_l206_206124


namespace rational_root_neg_one_third_l206_206065

def P (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_neg_one_third : P (-1/3) = 0 :=
by
  have : (-1/3 : ℚ) ≠ 0 := by norm_num
  sorry

end rational_root_neg_one_third_l206_206065


namespace cut_out_area_l206_206072

theorem cut_out_area (x : ℝ) (h1 : x * (x - 10) = 1575) : 10 * x - 10 * 10 = 450 := by
  -- Proof to be filled in here
  sorry

end cut_out_area_l206_206072


namespace probability_diff_digits_l206_206837

open Finset

def two_digit_same_digit (n : ℕ) : Prop :=
  n / 10 = n % 10

def three_digit_same_digit (n : ℕ) : Prop :=
  (n % 100) / 10 = n / 100 ∧ (n / 100) = (n % 10)

def same_digit (n : ℕ) : Prop :=
  two_digit_same_digit n ∨ three_digit_same_digit n

def total_numbers : ℕ :=
  (199 - 10 + 1)

def same_digit_count : ℕ :=
  9 + 9

theorem probability_diff_digits : 
  ((total_numbers - same_digit_count) / total_numbers : ℚ) = 86 / 95 :=
by
  sorry

end probability_diff_digits_l206_206837


namespace sides_equal_max_diagonal_at_most_two_l206_206910

variable {n : ℕ}
variable (P : Polygon n)
variable (is_convex : P.IsConvex)
variable (max_diagonal : ℝ)
variable (sides_equal_max_diagonal : list ℝ)
variable (length_sides_equal_max_diagonal : sides_equal_max_diagonal.length)

-- Here we assume the basic conditions given in the problem:
-- 1. The polygon P is convex.
-- 2. The number of sides equal to the longest diagonal are stored in sides_equal_max_diagonal.

theorem sides_equal_max_diagonal_at_most_two :
  is_convex → length_sides_equal_max_diagonal ≤ 2 :=
by
  sorry

end sides_equal_max_diagonal_at_most_two_l206_206910


namespace a_c_sum_l206_206432

theorem a_c_sum (a b c d : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : d = a * b * c) (h5 : 233 % d = 79) : a + c = 13 :=
sorry

end a_c_sum_l206_206432


namespace sequence_nat_nums_exists_l206_206710

theorem sequence_nat_nums_exists (n : ℕ) : { k : ℕ | ∃ (x : ℕ → ℕ), (∀ i j, i < j → x i < x j) ∧ (∀ i, 1 ≤ i → i ≤ n → x i)} = { k | 1 ≤ k ∧ k ≤ n } :=
sorry

end sequence_nat_nums_exists_l206_206710


namespace alyosha_cube_cut_l206_206034

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l206_206034


namespace alyosha_cube_problem_l206_206026

theorem alyosha_cube_problem (n s : ℕ) (h1 : n > s) (h2 : n ^ 3 - s ^ 3 = 152) : 
  n = 6 := 
by
  sorry

end alyosha_cube_problem_l206_206026


namespace factory_car_production_l206_206350

theorem factory_car_production :
  let cars_yesterday := 60
  let cars_today := 2 * cars_yesterday
  let total_cars := cars_yesterday + cars_today
  total_cars = 180 :=
by
  sorry

end factory_car_production_l206_206350


namespace plane_equation_l206_206225

theorem plane_equation
  (A B C D : ℤ)
  (hA : A > 0)
  (h_gcd : Int.gcd A B = 1 ∧ Int.gcd A C = 1 ∧ Int.gcd A D = 1)
  (h_point : (A * 4 + B * (-4) + C * 5 + D = 0)) :
  A = 4 ∧ B = -4 ∧ C = 5 ∧ D = -57 :=
  sorry

end plane_equation_l206_206225


namespace compound_interest_calculation_l206_206569

theorem compound_interest_calculation : 
  ∀ (x y T SI: ℝ), 
  x = 5000 → T = 2 → SI = 500 → 
  (y = SI * 100 / (x * T)) → 
  (5000 * (1 + (y / 100))^T - 5000 = 512.5) :=
by 
  intros x y T SI hx hT hSI hy
  sorry

end compound_interest_calculation_l206_206569


namespace spam_ratio_l206_206361

theorem spam_ratio (total_emails important_emails promotional_fraction promotional_emails spam_emails : ℕ) 
  (h1 : total_emails = 400) 
  (h2 : important_emails = 180) 
  (h3 : promotional_fraction = 2/5) 
  (h4 : total_emails - important_emails = spam_emails + promotional_emails) 
  (h5 : promotional_emails = promotional_fraction * (total_emails - important_emails)) 
  : spam_emails / total_emails = 33 / 100 := 
by {
  sorry
}

end spam_ratio_l206_206361


namespace Eliza_first_more_than_300_paperclips_on_Thursday_l206_206201

theorem Eliza_first_more_than_300_paperclips_on_Thursday :
  ∃ k : ℕ, 5 * 3^k > 300 ∧ k = 4 := 
by
  sorry

end Eliza_first_more_than_300_paperclips_on_Thursday_l206_206201


namespace crosswalk_distance_l206_206344

noncomputable def distance_between_stripes (area : ℝ) (side : ℝ) (angle : ℝ) : ℝ :=
  (2 * area) / (side * Real.cos angle)

theorem crosswalk_distance
  (curb_distance : ℝ) (crosswalk_angle_deg : ℝ) (curb_length : ℝ) (stripe_length : ℝ) 
  (h₁ : curb_distance = 50)
  (h₂ : crosswalk_angle_deg = 30)
  (h₃ : curb_length = 20)
  (h₄ : stripe_length = 60) :
  abs (distance_between_stripes (curb_length * curb_distance) stripe_length (Real.pi * crosswalk_angle_deg / 180) - 19.24) < 0.01 := 
by
  sorry

end crosswalk_distance_l206_206344


namespace hexagon_area_eq_l206_206834

theorem hexagon_area_eq (s t : ℝ) (hs : s^2 = 16) (heq : 4 * s = 6 * t) :
  6 * (t^2 * (Real.sqrt 3) / 4) = 32 * (Real.sqrt 3) / 3 := by
  sorry

end hexagon_area_eq_l206_206834


namespace find_x_coordinate_l206_206579

theorem find_x_coordinate :
  ∃ x : ℝ, (∃ m b : ℝ, (∀ y x : ℝ, y = m * x + b) ∧ 
                     ((3 = m * 10 + b) ∧ 
                      (0 = m * 4 + b)
                     ) ∧ 
                     (-3 = m * x + b) ∧ 
                     (x = -2)) :=
sorry

end find_x_coordinate_l206_206579


namespace square_completion_form_l206_206134

theorem square_completion_form (x k m: ℝ) (h: 16*x^2 - 32*x - 512 = 0):
  (x + k)^2 = m ↔ m = 65 :=
by
  sorry

end square_completion_form_l206_206134


namespace roller_coaster_ticket_cost_l206_206681

def ferrisWheelCost : ℕ := 6
def logRideCost : ℕ := 7
def initialTickets : ℕ := 2
def ticketsToBuy : ℕ := 16

def totalTicketsNeeded : ℕ := initialTickets + ticketsToBuy
def ridesCost : ℕ := ferrisWheelCost + logRideCost
def rollerCoasterCost : ℕ := totalTicketsNeeded - ridesCost

theorem roller_coaster_ticket_cost :
  rollerCoasterCost = 5 :=
by
  sorry

end roller_coaster_ticket_cost_l206_206681


namespace linda_color_choices_l206_206100

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def combination (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem linda_color_choices : combination 8 3 = 56 :=
  by sorry

end linda_color_choices_l206_206100


namespace count_muffin_combinations_l206_206661

theorem count_muffin_combinations (kinds total : ℕ) (at_least_one: ℕ) :
  kinds = 4 ∧ total = 8 ∧ at_least_one = 1 → 
  let valid_combinations := 23 in
  valid_combinations = 
    (1 -- case 1: 4 different kinds out of 4 remaining muffins
    + 4 -- case 2: all 4 remaining muffins of the same kind
    + (4 * 3) -- case 3: 3 of one kind, 1 of another (4 choices for 3 + 3 choices for 1)
    + (4 * 3 / 2)) -- case 4: 2 of one kind and 2 of another (combinations)
  sorry

end count_muffin_combinations_l206_206661


namespace number_of_men_l206_206097

variable (W D X : ℝ)

theorem number_of_men (M_eq_2W : M = 2 * W)
  (wages_40_women : 21600 = 40 * W * D)
  (men_wages : 14400 = X * M * 20) :
  X = (2 / 3) * D :=
  by
  sorry

end number_of_men_l206_206097


namespace classroom_activity_solution_l206_206697

theorem classroom_activity_solution 
  (x y : ℕ) 
  (h1 : x - y = 6) 
  (h2 : x * y = 45) : 
  x = 11 ∧ y = 5 :=
by
  sorry

end classroom_activity_solution_l206_206697


namespace f_seven_l206_206545

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h : ℝ) : f (-h) = -f (h)
axiom periodic_function (h : ℝ) : f (h + 4) = f (h)
axiom f_one : f 1 = 2

theorem f_seven : f (7) = -2 :=
by
  sorry

end f_seven_l206_206545


namespace complement_A_inter_B_l206_206759

def A : Set ℝ := {x | abs (x - 2) ≤ 2}

def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

def A_inter_B : Set ℝ := A ∩ B

def C_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

theorem complement_A_inter_B :
  C_R A_inter_B = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end complement_A_inter_B_l206_206759


namespace mulch_price_per_pound_l206_206329

noncomputable def price_per_pound (total_cost : ℝ) (total_tons : ℝ) (pounds_per_ton : ℝ) : ℝ :=
  total_cost / (total_tons * pounds_per_ton)

theorem mulch_price_per_pound :
  price_per_pound 15000 3 2000 = 2.5 :=
by
  sorry

end mulch_price_per_pound_l206_206329


namespace min_sum_four_consecutive_nat_nums_l206_206216

theorem min_sum_four_consecutive_nat_nums (a : ℕ) (h1 : a % 11 = 0) (h2 : (a + 1) % 7 = 0)
    (h3 : (a + 2) % 5 = 0) (h4 : (a + 3) % 3 = 0) : a + (a + 1) + (a + 2) + (a + 3) = 1458 :=
  sorry

end min_sum_four_consecutive_nat_nums_l206_206216


namespace arithmetic_sequence_sum_l206_206441

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (n : ℕ)
  (a1 : ℤ)
  (d : ℤ)
  (h1 : a1 = 2)
  (h2 : a_n 5 = a_n 1 + 4 * d)
  (h3 : a_n 3 = a_n 1 + 2 * d)
  (h4 : a_n 5 = 3 * a_n 3) :
  S_n 9 = -54 := 
by  
  sorry

end arithmetic_sequence_sum_l206_206441


namespace solution_set_of_f_double_exp_inequality_l206_206267

theorem solution_set_of_f_double_exp_inequality (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, -2 < x ∧ x < 1 ↔ 0 < f x) :
  {x : ℝ | f (2^x) < 0} = {x : ℝ | x > 0} :=
sorry

end solution_set_of_f_double_exp_inequality_l206_206267


namespace convert_mps_to_kmph_l206_206173

theorem convert_mps_to_kmph (v_mps : ℝ) (conversion_factor : ℝ) : v_mps = 22 → conversion_factor = 3.6 → v_mps * conversion_factor = 79.2 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end convert_mps_to_kmph_l206_206173


namespace billy_books_read_l206_206845

def hours_per_day : ℕ := 8
def days_per_weekend : ℕ := 2
def reading_percentage : ℚ := 0.25
def pages_per_hour : ℕ := 60
def pages_per_book : ℕ := 80

theorem billy_books_read :
  let total_hours := hours_per_day * days_per_weekend in
  let reading_hours := total_hours * reading_percentage in
  let total_pages := reading_hours * pages_per_hour in
  let books_read := total_pages / pages_per_book in
  books_read = 3 :=
by
  sorry

end billy_books_read_l206_206845


namespace weight_of_6m_rod_l206_206998

theorem weight_of_6m_rod (r ρ : ℝ) (h₁ : 11.25 > 0) (h₂ : 6 > 0) (h₃ : 0 < r) (h₄ : 42.75 = π * r^2 * 11.25 * ρ) : 
  (π * r^2 * 6 * (42.75 / (π * r^2 * 11.25))) = 22.8 :=
by
  sorry

end weight_of_6m_rod_l206_206998


namespace m_n_value_l206_206088

theorem m_n_value (m n : ℝ)
  (h1 : m * (-1/2)^2 + n * (-1/2) - 1/m < 0)
  (h2 : m * 2^2 + n * 2 - 1/m < 0)
  (h3 : m < 0)
  (h4 : (-1/2 + 2 = -n/m))
  (h5 : (-1/2) * 2 = -1/m^2) :
  m - n = -5/2 :=
sorry

end m_n_value_l206_206088


namespace find_n_l206_206011

-- Define the size of the cube and the number of smaller cubes.
def n : ℕ
def s : ℕ

-- Conditions
axiom h1 : n > 5
axiom h2 : n^3 - s^3 = 152
axiom h3 : ∀ n, n ∈ {6}

-- Main statement to prove.
theorem find_n (h1 : n > 5) (h2 : n^3 - s^3 = 152) : n = 6 := by
  sorry

end find_n_l206_206011


namespace total_savings_l206_206941

def weekly_savings : ℕ := 15
def weeks_per_cycle : ℕ := 60
def number_of_cycles : ℕ := 5

theorem total_savings :
  (weekly_savings * weeks_per_cycle) * number_of_cycles = 4500 := 
sorry

end total_savings_l206_206941


namespace device_records_720_instances_in_one_hour_l206_206754

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l206_206754


namespace find_k_if_equal_roots_l206_206918

theorem find_k_if_equal_roots (a b k : ℚ) 
  (h1 : 2 * a + b = -4) 
  (h2 : 2 * a * b + a^2 = -60) 
  (h3 : -2 * a^2 * b = k)
  (h4 : a ≠ b)
  (h5 : k > 0) :
  k = 6400 / 27 :=
by {
  sorry
}

end find_k_if_equal_roots_l206_206918


namespace cosine_F_in_triangle_DEF_l206_206281

theorem cosine_F_in_triangle_DEF
  (D E F : ℝ)
  (h_triangle : D + E + F = π)
  (sin_D : Real.sin D = 4 / 5)
  (cos_E : Real.cos E = 12 / 13) :
  Real.cos F = - (16 / 65) := by
  sorry

end cosine_F_in_triangle_DEF_l206_206281


namespace find_number_l206_206917

variables (n : ℝ)

-- Condition: a certain number divided by 14.5 equals 173.
def condition_1 (n : ℝ) : Prop := n / 14.5 = 173

-- Condition: 29.94 ÷ 1.45 = 17.3.
def condition_2 : Prop := 29.94 / 1.45 = 17.3

-- Theorem: Prove that the number is 2508.5 given the conditions.
theorem find_number (h1 : condition_1 n) (h2 : condition_2) : n = 2508.5 :=
by 
  sorry

end find_number_l206_206917


namespace num_divisors_sixty_l206_206908

theorem num_divisors_sixty : 
  let n := 60 in
  let prime_fact := ([(2, 2), (3, 1), (5, 1)]) in
  ∑ (e : (ℕ × ℕ)) in prime_fact, (e.2 + 1) = 12 :=
by
  let n := 60
  let prime_fact := ([(2, 2), (3, 1), (5, 1)])
  rerewrite_ax num_divisors_sixty is
 sorry

end num_divisors_sixty_l206_206908


namespace volume_of_one_gram_l206_206621

theorem volume_of_one_gram (mass_per_cubic_meter : ℕ)
  (kilo_to_grams : ℕ)
  (cubic_meter_to_cubic_centimeters : ℕ)
  (substance_mass : mass_per_cubic_meter = 300)
  (kilo_conv : kilo_to_grams = 1000)
  (cubic_conv : cubic_meter_to_cubic_centimeters = 1000000)
  :
  ∃ v : ℝ, v = cubic_meter_to_cubic_centimeters / (mass_per_cubic_meter * kilo_to_grams) ∧ v = 10 / 3 := 
by 
  sorry

end volume_of_one_gram_l206_206621


namespace no_integer_roots_l206_206299
open Polynomial

theorem no_integer_roots {p : ℤ[X]} (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_pa : p.eval a = 1) (h_pb : p.eval b = 1) (h_pc : p.eval c = 1) : 
  ∀ m : ℤ, p.eval m ≠ 0 :=
by
  sorry

end no_integer_roots_l206_206299


namespace log_8_4000_l206_206642

theorem log_8_4000 : ∃ (n : ℤ), 8^3 = 512 ∧ 8^4 = 4096 ∧ 512 < 4000 ∧ 4000 < 4096 ∧ n = 4 :=
by
  sorry

end log_8_4000_l206_206642


namespace emily_trip_duration_same_l206_206575

theorem emily_trip_duration_same (s : ℝ) (h_s_pos : 0 < s) : 
  let t1 := (90 : ℝ) / s
  let t2 := (360 : ℝ) / (4 * s)
  t2 = t1 := sorry

end emily_trip_duration_same_l206_206575


namespace find_h_l206_206204

def f (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

theorem find_h : ∃ a h k, (h = -3 / 2) ∧ (f x = a * (x - h)^2 + k) :=
by
  -- Proof steps would go here
  sorry

end find_h_l206_206204


namespace meal_cost_l206_206838

theorem meal_cost (total_people total_bill : ℕ) (h1 : total_people = 2 + 5) (h2 : total_bill = 21) :
  total_bill / total_people = 3 := by
  sorry

end meal_cost_l206_206838


namespace bushes_needed_l206_206358

theorem bushes_needed
  (num_sides : ℕ) (side_length : ℝ) (bush_fill : ℝ) (total_length : ℝ) (num_bushes : ℕ) :
  num_sides = 3 ∧ side_length = 16 ∧ bush_fill = 4 ∧ total_length = num_sides * side_length ∧ num_bushes = total_length / bush_fill →
  num_bushes = 12 := by
  sorry

end bushes_needed_l206_206358


namespace triangle_is_isosceles_l206_206628

theorem triangle_is_isosceles
  (p q r : ℝ)
  (H : ∀ (n : ℕ), n > 0 → (p^n + q^n > r^n) ∧ (q^n + r^n > p^n) ∧ (r^n + p^n > q^n))
  : p = q ∨ q = r ∨ r = p := 
begin
  sorry
end

end triangle_is_isosceles_l206_206628


namespace total_skips_is_33_l206_206374

theorem total_skips_is_33 {
  let skips_5 := 8,
  ∃ skips_4 skips_3 skips_2 skips_1 : ℕ,
  (skips_5 = skips_4 + 1) ∧
  (skips_4 = skips_3 - 3) ∧
  (skips_3 = skips_2 * 2) ∧
  (skips_2 = skips_1 + 2) ∧
  (skips_1 + skips_2 + skips_3 + skips_4 + skips_5 = 33) 
} sorry

end total_skips_is_33_l206_206374


namespace simplify_expression_l206_206875

theorem simplify_expression :
  (120^2 - 9^2) / (90^2 - 18^2) * ((90 - 18) * (90 + 18)) / ((120 - 9) * (120 + 9)) = 1 := by
  sorry

end simplify_expression_l206_206875


namespace find_integer_solutions_l206_206208

theorem find_integer_solutions :
  ∀ (x y : ℕ), 0 < x → 0 < y → (2 * x^2 + 5 * x * y + 2 * y^2 = 2006 ↔ (x = 28 ∧ y = 3) ∨ (x = 3 ∧ y = 28)) :=
by
  sorry

end find_integer_solutions_l206_206208


namespace three_x4_plus_two_x5_l206_206695

theorem three_x4_plus_two_x5 (x1 x2 x3 x4 x5 : ℤ)
  (h1 : 2 * x1 + x2 + x3 + x4 + x5 = 6)
  (h2 : x1 + 2 * x2 + x3 + x4 + x5 = 12)
  (h3 : x1 + x2 + 2 * x3 + x4 + x5 = 24)
  (h4 : x1 + x2 + x3 + 2 * x4 + x5 = 48)
  (h5 : x1 + x2 + x3 + x4 + 2 * x5 = 96) : 
  3 * x4 + 2 * x5 = 181 := 
sorry

end three_x4_plus_two_x5_l206_206695


namespace bushes_needed_l206_206357

theorem bushes_needed
  (num_sides : ℕ) (side_length : ℝ) (bush_fill : ℝ) (total_length : ℝ) (num_bushes : ℕ) :
  num_sides = 3 ∧ side_length = 16 ∧ bush_fill = 4 ∧ total_length = num_sides * side_length ∧ num_bushes = total_length / bush_fill →
  num_bushes = 12 := by
  sorry

end bushes_needed_l206_206357


namespace smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l206_206396

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sin x)^2 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x - (1 / 2) * Real.cos (2 * x)

theorem smallest_positive_period_and_range :
  (∀ x, f (x + Real.pi) = f x) ∧ (Set.range f = Set.Icc (-3 / 2) (5 / 2)) :=
by
  sorry

theorem sin_2x0_if_zero_of_f (x0 : ℝ) (hx0 : 0 ≤ x0 ∧ x0 ≤ Real.pi / 2)
  (hf : f x0 = 0) : Real.sin (2 * x0) = (Real.sqrt 15 - Real.sqrt 3) / 8 :=
by
  sorry

end smallest_positive_period_and_range_sin_2x0_if_zero_of_f_l206_206396


namespace data_instances_in_one_hour_l206_206748

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end data_instances_in_one_hour_l206_206748


namespace sequence_formula_l206_206648

-- Define the problem when n >= 2
theorem sequence_formula (n : ℕ) (h : n ≥ 2) : 
  1 / (n^2 - 1) = (1 / 2) * (1 / (n - 1) - 1 / (n + 1)) := 
by {
  sorry
}

end sequence_formula_l206_206648


namespace unique_k_linear_equation_l206_206089

theorem unique_k_linear_equation :
  (∀ x y k : ℝ, (2 : ℝ) * x^|k| + (k - 1) * y = 3 → (|k| = 1 ∧ k ≠ 1) → k = -1) :=
by
  sorry

end unique_k_linear_equation_l206_206089


namespace equal_white_black_balls_l206_206848

theorem equal_white_black_balls (b w n x : ℕ) 
(h1 : x = n - x)
: (x = b + w - n + x - w) := sorry

end equal_white_black_balls_l206_206848


namespace solve_AlyoshaCube_l206_206025

noncomputable def AlyoshaCubeSplit (n s : ℕ) : Prop :=
  n^3 - s^3 = 152 ∧ n > s

theorem solve_AlyoshaCube : ∃ n, ∃ s : ℕ, AlyoshaCubeSplit n s ∧ n = 6 :=
by
  sorry

end solve_AlyoshaCube_l206_206025


namespace simplify_expression_l206_206368

theorem simplify_expression (n : ℤ) :
  (2 : ℝ) ^ (-(3 * n + 1)) + (2 : ℝ) ^ (-(3 * n - 2)) - 3 * (2 : ℝ) ^ (-3 * n) = (3 / 2) * (2 : ℝ) ^ (-3 * n) :=
by
  sorry

end simplify_expression_l206_206368


namespace minimum_cost_for_28_apples_l206_206273

/--
Conditions:
  - apples can be bought at a rate of 4 for 15 cents,
  - apples can be bought at a rate of 7 for 30 cents,
  - you need to buy exactly 28 apples.
Prove that the minimum total cost to buy exactly 28 apples is 120 cents.
-/
theorem minimum_cost_for_28_apples : 
  let cost_4_for_15 := 15
  let cost_7_for_30 := 30
  let apples_needed := 28
  ∃ (n m : ℕ), n * 4 + m * 7 = apples_needed ∧ n * cost_4_for_15 + m * cost_7_for_30 = 120 := sorry

end minimum_cost_for_28_apples_l206_206273


namespace find_y_when_x_eq_4_l206_206466

theorem find_y_when_x_eq_4 (x y : ℝ) (k : ℝ) :
  (8 * y = k / x^3) →
  (y = 25) →
  (x = 2) →
  (exists y', x = 4 → y' = 25/8) :=
by
  sorry

end find_y_when_x_eq_4_l206_206466


namespace female_managers_count_l206_206270

def total_employees : ℕ := sorry
def female_employees : ℕ := 700
def managers : ℕ := (2 * total_employees) / 5
def male_employees : ℕ := total_employees - female_employees
def male_managers : ℕ := (2 * male_employees) / 5

theorem female_managers_count :
  ∃ (fm : ℕ), managers = fm + male_managers ∧ fm = 280 := by
  sorry

end female_managers_count_l206_206270


namespace selling_price_with_increase_l206_206624

variable (a : ℝ)

theorem selling_price_with_increase (h : a > 0) : 1.1 * a = a + 0.1 * a := by
  -- Here you will add the proof, which we skip with sorry
  sorry

end selling_price_with_increase_l206_206624


namespace original_cube_volume_l206_206133

theorem original_cube_volume
  (a : ℝ)
  (h : (a + 2) * (a - 1) * a = a^3 + 14) :
  a^3 = 64 :=
by
  sorry

end original_cube_volume_l206_206133


namespace symbols_in_P_l206_206276
-- Importing the necessary library

-- Define the context P and the operations
def context_P : Type := sorry

def mul_op (P : context_P) : String := "*"
def div_op (P : context_P) : String := "/"
def exp_op (P : context_P) : String := "∧"
def sqrt_op (P : context_P) : String := "SQR"
def abs_op (P : context_P) : String := "ABS"

-- Define what each symbol represents in the context of P
theorem symbols_in_P (P : context_P) :
  (mul_op P = "*") ∧
  (div_op P = "/") ∧
  (exp_op P = "∧") ∧
  (sqrt_op P = "SQR") ∧
  (abs_op P = "ABS") := 
sorry

end symbols_in_P_l206_206276


namespace largest_integer_le_zero_of_f_l206_206656

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem largest_integer_le_zero_of_f :
  ∃ x₀ : ℝ, (f x₀ = 0) ∧ 2 ≤ x₀ ∧ x₀ < 3 ∧ (∀ k : ℤ, k ≤ x₀ → k = 2 ∨ k < 2) :=
by
  sorry

end largest_integer_le_zero_of_f_l206_206656


namespace alyosha_cube_cut_l206_206037

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end alyosha_cube_cut_l206_206037


namespace phone_cost_l206_206455

theorem phone_cost (C : ℝ) (h1 : 0.40 * C + 780 = C) : C = 1300 := by
  sorry

end phone_cost_l206_206455


namespace complement_intersection_l206_206729

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 3, 4}
def A_complement : Set ℕ := U \ A

theorem complement_intersection :
  (A_complement ∩ B) = {2, 4} :=
by 
  sorry

end complement_intersection_l206_206729


namespace solve_equation_l206_206793

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) :
    (3 / (x + 2) - 1 / x = 0) → x = 1 :=
  by sorry

end solve_equation_l206_206793


namespace total_pieces_correct_l206_206463

theorem total_pieces_correct :
  let bell_peppers := 10
  let onions := 7
  let zucchinis := 15
  let bell_peppers_slices := (2 * 20)  -- 25% of 10 bell peppers sliced into 20 slices each
  let bell_peppers_large_pieces := (7 * 10)  -- Remaining 75% cut into 10 pieces each
  let bell_peppers_smaller_pieces := (35 * 3)  -- Half of large pieces cut into 3 pieces each
  let onions_slices := (3 * 18)  -- 50% of onions sliced into 18 slices each
  let onions_pieces := (4 * 8)  -- Remaining 50% cut into 8 pieces each
  let zucchinis_slices := (4 * 15)  -- 30% of zucchinis sliced into 15 pieces each
  let zucchinis_pieces := (10 * 8)  -- Remaining 70% cut into 8 pieces each
  let total_slices := bell_peppers_slices + onions_slices + zucchinis_slices
  let total_pieces := bell_peppers_large_pieces + bell_peppers_smaller_pieces + onions_pieces + zucchinis_pieces
  total_slices + total_pieces = 441 :=
by
  sorry

end total_pieces_correct_l206_206463


namespace cubic_polynomial_sum_l206_206120

-- Define the roots and their properties according to Vieta's formulas
variables {p q r : ℝ}
axiom root_poly : p * q * r = -1
axiom pq_sum : p * q + p * r + q * r = -3
axiom roots_sum : p + q + r = 0

-- Define the target equality to prove
theorem cubic_polynomial_sum :
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 3 :=
by
  sorry

end cubic_polynomial_sum_l206_206120


namespace correct_proposition_is_D_l206_206516

theorem correct_proposition_is_D (A B C D : Prop) :
  (∀ (H : Prop), (H = A ∨ H = B ∨ H = C) → ¬H) → D :=
by
  -- We assume that A, B, and C are false.
  intro h
  -- Now we need to prove that D is true.
  sorry

end correct_proposition_is_D_l206_206516


namespace right_triangle_count_l206_206260

theorem right_triangle_count :
  ∃! (a b : ℕ), (a^2 + b^2 = (b + 3)^2) ∧ (b < 50) :=
by
  sorry

end right_triangle_count_l206_206260


namespace max_sum_of_xj4_minus_xj5_l206_206537

theorem max_sum_of_xj4_minus_xj5 (n : ℕ) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 ≤ x i) 
  (h_sum : (Finset.univ.sum x) = 1) : 
  (Finset.univ.sum (λ j => (x j)^4 - (x j)^5)) ≤ 1 / 12 :=
sorry

end max_sum_of_xj4_minus_xj5_l206_206537


namespace optimal_cylinder_dimensions_l206_206878

variable (R : ℝ)

noncomputable def optimal_cylinder_height : ℝ := (2 * R) / Real.sqrt 3
noncomputable def optimal_cylinder_radius : ℝ := R * Real.sqrt (2 / 3)

theorem optimal_cylinder_dimensions :
  ∃ (h r : ℝ), 
    (h = optimal_cylinder_height R ∧ r = optimal_cylinder_radius R) ∧
    ∀ (h' r' : ℝ), (4 * R^2 = 4 * r'^2 + h'^2) → 
      (h' = optimal_cylinder_height R ∧ r' = optimal_cylinder_radius R) → 
      (π * r' ^ 2 * h' ≤ π * r ^ 2 * h) :=
by
  -- Proof omitted
  sorry

end optimal_cylinder_dimensions_l206_206878


namespace boys_amount_per_person_l206_206772

theorem boys_amount_per_person (total_money : ℕ) (total_children : ℕ) (per_girl : ℕ) (number_of_boys : ℕ) (amount_per_boy : ℕ) : 
  total_money = 460 ∧
  total_children = 41 ∧
  per_girl = 8 ∧
  number_of_boys = 33 → 
  amount_per_boy = 12 :=
by sorry

end boys_amount_per_person_l206_206772


namespace daughters_meet_days_count_l206_206740

noncomputable def days_elder_returns := 5
noncomputable def days_second_returns := 4
noncomputable def days_youngest_returns := 3

noncomputable def total_days := 100

-- Defining the count of individual and combined visits
noncomputable def count_individual_visits (period : ℕ) : ℕ := total_days / period
noncomputable def count_combined_visits (period1 : ℕ) (period2 : ℕ) : ℕ := total_days / Nat.lcm period1 period2
noncomputable def count_all_together_visits (periods : List ℕ) : ℕ := total_days / periods.foldr Nat.lcm 1

-- Specific counts
noncomputable def count_youngest_visits : ℕ := count_individual_visits days_youngest_returns
noncomputable def count_second_visits : ℕ := count_individual_visits days_second_returns
noncomputable def count_elder_visits : ℕ := count_individual_visits days_elder_returns

noncomputable def count_youngest_and_second : ℕ := count_combined_visits days_youngest_returns days_second_returns
noncomputable def count_youngest_and_elder : ℕ := count_combined_visits days_youngest_returns days_elder_returns
noncomputable def count_second_and_elder : ℕ := count_combined_visits days_second_returns days_elder_returns

noncomputable def count_all_three : ℕ := count_all_together_visits [days_youngest_returns, days_second_returns, days_elder_returns]

-- Final Inclusion-Exclusion principle application
noncomputable def days_at_least_one_returns : ℕ := 
  count_youngest_visits + count_second_visits + count_elder_visits
  - count_youngest_and_second
  - count_youngest_and_elder
  - count_second_and_elder
  + count_all_three

theorem daughters_meet_days_count : days_at_least_one_returns = 60 := by
  sorry

end daughters_meet_days_count_l206_206740


namespace correct_conclusions_l206_206541

noncomputable def f : ℝ → ℝ := sorry -- Assume f has already been defined elsewhere

-- The given conditions
variable (f : ℝ → ℝ)
variable (H1 : ∀ (x y : ℝ), f(x + y) + f(x - y) = 2 * f(x) * f(y))
variable (H2 : f (1 / 2) = 0)
variable (H3 : f 0 ≠ 0)

-- Proof problem which includes the correct conclusions
theorem correct_conclusions (f : ℝ → ℝ)
  (H1 : ∀ (x y : ℝ), f(x + y) + f(x - y) = 2 * f(x) * f(y))
  (H2 : f (1 / 2) = 0)
  (H3 : f 0 ≠ 0) :
  f 0 = 1 ∧ ∀ y : ℝ, f (1 / 2 + y) + f (1 / 2 - y) = 0 :=
by
  sorry

end correct_conclusions_l206_206541


namespace sum_fraction_series_eq_l206_206869

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l206_206869


namespace trapezoid_base_difference_is_10_l206_206465

noncomputable def trapezoid_base_difference (AD BC AB : ℝ) (angle_BAD angle_ADC : ℝ) : ℝ :=
if angle_BAD = 60 ∧ angle_ADC = 30 ∧ AB = 5 then AD - BC else 0

theorem trapezoid_base_difference_is_10 (AD BC : ℝ) (angle_BAD angle_ADC : ℝ) (h_BAD : angle_BAD = 60)
(h_ADC : angle_ADC = 30) (h_AB : AB = 5) : trapezoid_base_difference AD BC AB angle_BAD angle_ADC = 10 :=
sorry

end trapezoid_base_difference_is_10_l206_206465


namespace cost_of_pears_l206_206131

theorem cost_of_pears 
  (initial_amount : ℕ := 55) 
  (left_amount : ℕ := 28) 
  (banana_count : ℕ := 2) 
  (banana_price : ℕ := 4) 
  (asparagus_price : ℕ := 6) 
  (chicken_price : ℕ := 11) 
  (total_spent : ℕ := 27) :
  initial_amount - left_amount - (banana_count * banana_price + asparagus_price + chicken_price) = 2 := 
by
  sorry

end cost_of_pears_l206_206131


namespace eccentricity_condition_l206_206795

theorem eccentricity_condition (m : ℝ) (h : 0 < m) : 
  (m < (4 / 3) ∨ m > (3 / 4)) ↔ ((1 - m) > (1 / 4) ∨ ((m - 1) / m) > (1 / 4)) :=
by
  sorry

end eccentricity_condition_l206_206795


namespace value_of_x_squared_plus_y_squared_l206_206561

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l206_206561


namespace alex_singles_percentage_l206_206528

theorem alex_singles_percentage (total_hits home_runs triples doubles: ℕ) 
  (h1 : total_hits = 50) 
  (h2 : home_runs = 2) 
  (h3 : triples = 3) 
  (h4 : doubles = 10) :
  ((total_hits - (home_runs + triples + doubles)) / total_hits : ℚ) * 100 = 70 := 
by
  sorry

end alex_singles_percentage_l206_206528


namespace probability_diff_topics_l206_206185

theorem probability_diff_topics
  (num_topics : ℕ)
  (num_combinations : ℕ)
  (num_different_combinations : ℕ)
  (h1 : num_topics = 6)
  (h2 : num_combinations = num_topics * num_topics)
  (h3 : num_combinations = 36)
  (h4 : num_different_combinations = num_topics * (num_topics - 1))
  (h5 : num_different_combinations = 30) :
  (num_different_combinations / num_combinations) = 5 / 6 := 
by 
  sorry

end probability_diff_topics_l206_206185


namespace total_skips_l206_206375

-- Definitions based on conditions
def fifth_throw := 8
def fourth_throw := fifth_throw - 1
def third_throw := fourth_throw + 3
def second_throw := third_throw / 2
def first_throw := second_throw - 2

-- Statement of the proof problem
theorem total_skips : first_throw + second_throw + third_throw + fourth_throw + fifth_throw = 33 := by
  sorry

end total_skips_l206_206375


namespace pyramid_height_correct_l206_206309

noncomputable def pyramid_height : ℝ :=
  let ab := 15 * Real.sqrt 3
  let bc := 14 * Real.sqrt 3
  let base_area := ab * bc
  let volume := 750
  let height := 3 * volume / base_area
  height

theorem pyramid_height_correct : pyramid_height = 25 / 7 :=
by
  sorry

end pyramid_height_correct_l206_206309
