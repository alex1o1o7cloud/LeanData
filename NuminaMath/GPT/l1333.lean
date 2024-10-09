import Mathlib

namespace find_AD_l1333_133348

-- Defining points and distances in the context of a triangle
variables {A B C D: Type*}
variables (dist_AB : ℝ) (dist_AC : ℝ) (dist_BC : ℝ) (midpoint_D : Prop)

-- Given conditions
def triangle_conditions : Prop :=
  dist_AB = 26 ∧
  dist_AC = 26 ∧
  dist_BC = 24 ∧
  midpoint_D

-- Problem statement as a Lean theorem
theorem find_AD
  (h : triangle_conditions dist_AB dist_AC dist_BC midpoint_D) :
  ∃ (AD : ℝ), AD = 2 * Real.sqrt 133 :=
sorry

end find_AD_l1333_133348


namespace total_area_of_plots_l1333_133355

theorem total_area_of_plots (n : ℕ) (side_length : ℕ) (area_one_plot : ℕ) (total_plots : ℕ) (total_area : ℕ)
  (h1 : n = 9)
  (h2 : side_length = 6)
  (h3 : area_one_plot = side_length * side_length)
  (h4 : total_plots = n)
  (h5 : total_area = area_one_plot * total_plots) :
  total_area = 324 := 
by
  sorry

end total_area_of_plots_l1333_133355


namespace unique_zero_iff_a_eq_half_l1333_133335

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (1 - x))

theorem unique_zero_iff_a_eq_half :
  (∃! x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end unique_zero_iff_a_eq_half_l1333_133335


namespace intersection_of_A_and_B_l1333_133379

variable (A : Set ℕ) (B : Set ℕ)

axiom h1 : A = {1, 2, 3, 4, 5}
axiom h2 : B = {3, 5}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} :=
  by sorry

end intersection_of_A_and_B_l1333_133379


namespace total_letters_in_all_names_l1333_133323

theorem total_letters_in_all_names :
  let jonathan_first := 8
  let jonathan_surname := 10
  let younger_sister_first := 5
  let younger_sister_surname := 10
  let older_brother_first := 6
  let older_brother_surname := 10
  let youngest_sibling_first := 4
  let youngest_sibling_hyphenated_surname := 15
  jonathan_first + jonathan_surname + younger_sister_first + younger_sister_surname +
  older_brother_first + older_brother_surname + youngest_sibling_first + youngest_sibling_hyphenated_surname = 68 := by
  sorry

end total_letters_in_all_names_l1333_133323


namespace average_difference_l1333_133386

theorem average_difference :
  let avg1 := (24 + 35 + 58) / 3
  let avg2 := (19 + 51 + 29) / 3
  avg1 - avg2 = 6 := by
sorry

end average_difference_l1333_133386


namespace find_value_of_xy_plus_yz_plus_xz_l1333_133356

variable (x y z : ℝ)

-- Conditions
def cond1 : Prop := x^2 + x * y + y^2 = 108
def cond2 : Prop := y^2 + y * z + z^2 = 64
def cond3 : Prop := z^2 + x * z + x^2 = 172

-- Theorem statement
theorem find_value_of_xy_plus_yz_plus_xz (hx : cond1 x y) (hy : cond2 y z) (hz : cond3 z x) : 
  x * y + y * z + x * z = 96 :=
sorry

end find_value_of_xy_plus_yz_plus_xz_l1333_133356


namespace junior_score_calculation_l1333_133306

variable {total_students : ℕ}
variable {junior_score senior_average : ℕ}
variable {junior_ratio senior_ratio : ℚ}
variable {class_average total_average : ℚ}

-- Hypotheses from the conditions
theorem junior_score_calculation (h1 : junior_ratio = 0.2)
                               (h2 : senior_ratio = 0.8)
                               (h3 : class_average = 82)
                               (h4 : senior_average = 80)
                               (h5 : total_students = 10)
                               (h6 : total_average * total_students = total_students * class_average)
                               (h7 : total_average = (junior_ratio * junior_score + senior_ratio * senior_average))
                               : junior_score = 90 :=
sorry

end junior_score_calculation_l1333_133306


namespace trig_identity_proof_l1333_133305

noncomputable def sin_30 : Real := 1 / 2
noncomputable def cos_120 : Real := -1 / 2
noncomputable def cos_45 : Real := Real.sqrt 2 / 2
noncomputable def tan_30 : Real := Real.sqrt 3 / 3

theorem trig_identity_proof : 
  sin_30 + cos_120 + 2 * cos_45 - Real.sqrt 3 * tan_30 = Real.sqrt 2 - 1 := 
by
  sorry

end trig_identity_proof_l1333_133305


namespace min_value_l1333_133372

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 3 + 2 * Real.sqrt 2 ≤ 2 / a + 1 / b :=
by
  sorry

end min_value_l1333_133372


namespace kangaroo_jump_is_8_5_feet_longer_l1333_133357

noncomputable def camel_step_length (total_distance : ℝ) (num_steps : ℕ) : ℝ := total_distance / num_steps
noncomputable def kangaroo_jump_length (total_distance : ℝ) (num_jumps : ℕ) : ℝ := total_distance / num_jumps
noncomputable def length_difference (jump_length step_length : ℝ) : ℝ := jump_length - step_length

theorem kangaroo_jump_is_8_5_feet_longer :
  let total_distance := 7920
  let num_gaps := 50
  let camel_steps_per_gap := 56
  let kangaroo_jumps_per_gap := 14
  let num_camel_steps := num_gaps * camel_steps_per_gap
  let num_kangaroo_jumps := num_gaps * kangaroo_jumps_per_gap
  let camel_step := camel_step_length total_distance num_camel_steps
  let kangaroo_jump := kangaroo_jump_length total_distance num_kangaroo_jumps
  length_difference kangaroo_jump camel_step = 8.5 := sorry

end kangaroo_jump_is_8_5_feet_longer_l1333_133357


namespace find_three_digit_number_l1333_133317

def is_valid_three_digit_number (M G U : ℕ) : Prop :=
  M ≠ G ∧ G ≠ U ∧ M ≠ U ∧ 
  0 ≤ M ∧ M ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ U ∧ U ≤ 9 ∧
  100 * M + 10 * G + U = (M + G + U) * (M + G + U - 2)

theorem find_three_digit_number : ∃ (M G U : ℕ), 
  is_valid_three_digit_number M G U ∧
  100 * M + 10 * G + U = 195 :=
by
  sorry

end find_three_digit_number_l1333_133317


namespace sale_price_l1333_133362

def original_price : ℝ := 100
def discount_rate : ℝ := 0.80

theorem sale_price (original_price discount_rate : ℝ) : original_price * (1 - discount_rate) = 20 := by
  sorry

end sale_price_l1333_133362


namespace slope_of_line_l1333_133392

theorem slope_of_line (x1 x2 y1 y2 : ℝ) (h1 : 1 = (x1 + x2) / 2) (h2 : 1 = (y1 + y2) / 2) 
                      (h3 : (x1^2 / 36) + (y1^2 / 9) = 1) (h4 : (x2^2 / 36) + (y2^2 / 9) = 1) :
  (y2 - y1) / (x2 - x1) = -1 / 4 :=
by
  sorry

end slope_of_line_l1333_133392


namespace annual_savings_l1333_133384

-- defining the conditions
def current_speed := 10 -- in Mbps
def current_bill := 20 -- in dollars
def bill_30Mbps := 2 * current_bill -- in dollars
def bill_20Mbps := current_bill + 10 -- in dollars
def months_in_year := 12

-- calculating the annual costs
def annual_cost_30Mbps := bill_30Mbps * months_in_year
def annual_cost_20Mbps := bill_20Mbps * months_in_year

-- statement of the problem
theorem annual_savings : (annual_cost_30Mbps - annual_cost_20Mbps) = 120 := by
  sorry -- prove the statement

end annual_savings_l1333_133384


namespace sequence_geometric_l1333_133304

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, a n ≠ 0)
  (h_arith : 2 * a 2 = a 1 + a 3)
  (h_geom : a 3 ^ 2 = a 2 * a 4)
  (h_recip_arith : 2 / a 4 = 1 / a 3 + 1 / a 5) :
  a 3 ^ 2 = a 1 * a 5 :=
sorry

end sequence_geometric_l1333_133304


namespace contrapositive_false_l1333_133332

theorem contrapositive_false : ¬ (∀ x : ℝ, x^2 = 1 → x = 1) → ∀ x : ℝ, x^2 = 1 → x ≠ 1 :=
by
  sorry

end contrapositive_false_l1333_133332


namespace calc_S_5_minus_S_4_l1333_133313

def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 2 * a n - 2

theorem calc_S_5_minus_S_4 {a : ℕ → ℕ} {S : ℕ → ℕ}
  (h : sum_sequence a S) : S 5 - S 4 = 32 :=
by
  sorry

end calc_S_5_minus_S_4_l1333_133313


namespace custom_op_subtraction_l1333_133329

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_subtraction :
  (custom_op 4 2) - (custom_op 2 4) = -8 := by
  sorry

end custom_op_subtraction_l1333_133329


namespace rectangle_area_divisible_by_12_l1333_133370

theorem rectangle_area_divisible_by_12 {a b c : ℕ} (h : a ^ 2 + b ^ 2 = c ^ 2) :
  12 ∣ (a * b) :=
sorry

end rectangle_area_divisible_by_12_l1333_133370


namespace solution_l1333_133343

axiom f : ℝ → ℝ

def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def decreasing_function (f : ℝ → ℝ) := ∀ x y, x < y → y ≤ 0 → f x > f y

def main_problem : Prop :=
  even_function f ∧ decreasing_function f ∧ f (-2) = 0 → ∀ x, f x < 0 ↔ x > -2 ∧ x < 2

theorem solution : main_problem :=
by
  sorry

end solution_l1333_133343


namespace segments_to_start_l1333_133385

-- Define the problem statement conditions in Lean 4
def concentric_circles : Prop := sorry -- Placeholder, as geometry involving tangents and arcs isn't directly supported

def chord_tangent_small_circle (AB : Prop) : Prop := sorry -- Placeholder, detailing tangency

def angle_ABC_eq_60 (A B C : Prop) : Prop := sorry -- Placeholder, situating angles in terms of Lean formalism

-- Proof statement
theorem segments_to_start (A B C : Prop) :
  concentric_circles →
  chord_tangent_small_circle (A ↔ B) →
  chord_tangent_small_circle (B ↔ C) →
  angle_ABC_eq_60 A B C →
  ∃ n : ℕ, n = 3 :=
sorry

end segments_to_start_l1333_133385


namespace f_geq_expression_l1333_133360

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 1 / a) * x - Real.log x

theorem f_geq_expression (a x : ℝ) (h : a < 0) : f x a ≥ (1 - 2 * a) * (a + 1) := 
  sorry

end f_geq_expression_l1333_133360


namespace students_total_l1333_133377

theorem students_total (T : ℝ) (h₁ : 0.675 * T = 594) : T = 880 :=
sorry

end students_total_l1333_133377


namespace UBA_Capital_bought_8_SUVs_l1333_133341

noncomputable def UBA_Capital_SUVs : ℕ := 
  let T := 9  -- Number of Toyotas
  let H := 1  -- Number of Hondas
  let SUV_Toyota := 9 / 10 * T  -- 90% of Toyotas are SUVs
  let SUV_Honda := 1 / 10 * H   -- 10% of Hondas are SUVs
  SUV_Toyota + SUV_Honda  -- Total number of SUVs

theorem UBA_Capital_bought_8_SUVs : UBA_Capital_SUVs = 8 := by
  sorry

end UBA_Capital_bought_8_SUVs_l1333_133341


namespace factorize_expression_l1333_133303

theorem factorize_expression (x y : ℝ) :
  (1 - x^2) * (1 - y^2) - 4 * x * y = (x * y - 1 + x + y) * (x * y - 1 - x - y) :=
by sorry

end factorize_expression_l1333_133303


namespace find_weight_difference_l1333_133354

variables (W_A W_B W_C W_D W_E : ℝ)

-- Definitions of the conditions
def average_weight_abc := (W_A + W_B + W_C) / 3 = 84
def average_weight_abcd := (W_A + W_B + W_C + W_D) / 4 = 80
def average_weight_bcde := (W_B + W_C + W_D + W_E) / 4 = 79
def weight_a := W_A = 77

-- The theorem statement
theorem find_weight_difference (h1 : average_weight_abc W_A W_B W_C)
                               (h2 : average_weight_abcd W_A W_B W_C W_D)
                               (h3 : average_weight_bcde W_B W_C W_D W_E)
                               (h4 : weight_a W_A) :
  W_E - W_D = 5 :=
sorry

end find_weight_difference_l1333_133354


namespace average_salary_of_associates_l1333_133345

theorem average_salary_of_associates 
  (num_managers : ℕ) (num_associates : ℕ)
  (avg_salary_managers : ℝ) (avg_salary_company : ℝ)
  (H_num_managers : num_managers = 15)
  (H_num_associates : num_associates = 75)
  (H_avg_salary_managers : avg_salary_managers = 90000)
  (H_avg_salary_company : avg_salary_company = 40000) :
  ∃ (A : ℝ), (num_managers * avg_salary_managers + num_associates * A) / (num_managers + num_associates) = avg_salary_company ∧ A = 30000 := by
  sorry

end average_salary_of_associates_l1333_133345


namespace probability_sum_of_digits_eq_10_l1333_133327

theorem probability_sum_of_digits_eq_10 (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1): 
  let P := m / n
  let valid_numbers := 120
  let total_numbers := 2020
  (P = valid_numbers / total_numbers) → (m = 6) → (n = 101) → (m + n = 107) :=
by 
  sorry

end probability_sum_of_digits_eq_10_l1333_133327


namespace min_value_of_z_l1333_133399

theorem min_value_of_z : ∃ x : ℝ, ∀ y : ℝ, 5 * x^2 + 20 * x + 25 ≤ 5 * y^2 + 20 * y + 25 :=
by
  sorry

end min_value_of_z_l1333_133399


namespace einstein_fundraising_l1333_133324

def boxes_of_pizza : Nat := 15
def packs_of_potato_fries : Nat := 40
def cans_of_soda : Nat := 25
def price_per_box : ℝ := 12
def price_per_pack : ℝ := 0.3
def price_per_can : ℝ := 2
def goal_amount : ℝ := 500

theorem einstein_fundraising : goal_amount - (boxes_of_pizza * price_per_box + packs_of_potato_fries * price_per_pack + cans_of_soda * price_per_can) = 258 := by
  sorry

end einstein_fundraising_l1333_133324


namespace hexagon_perimeter_l1333_133398

theorem hexagon_perimeter (s : ℝ) (h_area : s ^ 2 * (3 * Real.sqrt 3 / 2) = 54 * Real.sqrt 3) :
  6 * s = 36 :=
by
  sorry

end hexagon_perimeter_l1333_133398


namespace arctan_sum_l1333_133347

theorem arctan_sum (a b : ℝ) : 
  Real.arctan (a / (a + 2 * b)) + Real.arctan (b / (2 * a + b)) = Real.arctan (1 / 2) :=
by {
  sorry
}

end arctan_sum_l1333_133347


namespace product_primes_less_than_20_l1333_133349

theorem product_primes_less_than_20 :
  (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 9699690) :=
by
  sorry

end product_primes_less_than_20_l1333_133349


namespace sufficient_but_not_necessary_l1333_133334

theorem sufficient_but_not_necessary (a b : ℝ) : (a > |b|) → (a^2 > b^2) ∧ ¬((a^2 > b^2) → (a > |b|)) := 
sorry

end sufficient_but_not_necessary_l1333_133334


namespace meters_to_examine_10000_l1333_133346

def projection_for_sample (total_meters_examined : ℕ) (rejection_rate : ℝ) (sample_size : ℕ) :=
  total_meters_examined = sample_size

theorem meters_to_examine_10000 : 
  projection_for_sample 10000 0.015 10000 := by
  sorry

end meters_to_examine_10000_l1333_133346


namespace linear_function_correct_max_profit_correct_min_selling_price_correct_l1333_133387

-- Definition of the linear function
def linear_function (x : ℝ) : ℝ :=
  -2 * x + 360

-- Definition of monthly profit function
def profit_function (x : ℝ) : ℝ :=
  (-2 * x + 360) * (x - 30)

noncomputable def max_profit_statement : Prop :=
  ∃ x w, x = 105 ∧ w = 11250 ∧ profit_function x = w

noncomputable def min_selling_price (profit : ℝ) : Prop :=
  ∃ x, profit_function x ≥ profit ∧ x ≥ 80

-- The proof statements
theorem linear_function_correct : linear_function 30 = 300 ∧ linear_function 45 = 270 :=
  by
    sorry

theorem max_profit_correct : max_profit_statement :=
  by
    sorry

theorem min_selling_price_correct : min_selling_price 10000 :=
  by
    sorry

end linear_function_correct_max_profit_correct_min_selling_price_correct_l1333_133387


namespace garden_perimeter_l1333_133390

theorem garden_perimeter
  (width_garden : ℝ) (area_playground : ℝ)
  (length_playground : ℝ) (width_playground : ℝ)
  (area_garden : ℝ) (L : ℝ)
  (h1 : width_garden = 4) 
  (h2 : length_playground = 16)
  (h3 : width_playground = 12)
  (h4 : area_playground = length_playground * width_playground)
  (h5 : area_garden = area_playground)
  (h6 : area_garden = L * width_garden) :
  2 * L + 2 * width_garden = 104 :=
by
  sorry

end garden_perimeter_l1333_133390


namespace nine_pow_n_sub_one_l1333_133359

theorem nine_pow_n_sub_one (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ (p1 p2 p3 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ (9^n - 1) = p1 * p2 * p3 ∧ (p1 = 61 ∨ p2 = 61 ∨ p3 = 61)) : 9^n - 1 = 59048 := 
sorry

end nine_pow_n_sub_one_l1333_133359


namespace probability_seven_chairs_probability_n_chairs_l1333_133351
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l1333_133351


namespace arithmetic_expression_value_l1333_133340

theorem arithmetic_expression_value :
  15 * 36 + 15 * 3^3 = 945 :=
by
  sorry

end arithmetic_expression_value_l1333_133340


namespace choir_min_students_l1333_133308

theorem choir_min_students : ∃ n : ℕ, (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) ∧ n = 990 :=
by
  sorry

end choir_min_students_l1333_133308


namespace equivalent_statement_l1333_133309

variable (R G : Prop)

theorem equivalent_statement (h : ¬ R → ¬ G) : G → R := by
  intro hG
  by_contra hR
  exact h hR hG

end equivalent_statement_l1333_133309


namespace triangle_angle_bisector_theorem_l1333_133322

variable {α : Type*} [LinearOrderedField α]

theorem triangle_angle_bisector_theorem (A B C D : α)
  (h1 : A^2 = (C + D) * (B - (B * D / C)))
  (h2 : B / C = (B * D / C) / D) :
  A^2 = C * B - D * (B * D / C) := 
  by
  sorry

end triangle_angle_bisector_theorem_l1333_133322


namespace small_fries_number_l1333_133315

variables (L S : ℕ)

axiom h1 : L + S = 24
axiom h2 : L = 5 * S

theorem small_fries_number : S = 4 :=
by sorry

end small_fries_number_l1333_133315


namespace min_weighings_to_determine_counterfeit_l1333_133319

/-- 
  Given 2023 coins with two counterfeit coins and 2021 genuine coins, 
  and using a balance scale, determine whether the counterfeit coins 
  are heavier or lighter. Prove that the minimum number of weighings 
  required is 3. 
-/
theorem min_weighings_to_determine_counterfeit (n : ℕ) (k : ℕ) (l : ℕ) 
  (h : n = 2023) (h₁ : k = 2) (h₂ : l = 2021) 
  (w₁ w₂ : ℕ → ℝ) -- weights of coins
  (h_fake : ∀ i j, w₁ i = w₁ j) -- counterfeits have same weight
  (h_fake_diff : ∀ i j, i ≠ j → w₁ i ≠ w₂ j) -- fake different from genuine
  (h_genuine : ∀ i j, w₂ i = w₂ j) -- genuines have same weight
  (h_total : ∀ i, i ≤ l + k) -- total coins condition
  : ∃ min_weighings : ℕ, min_weighings = 3 :=
by
  sorry

end min_weighings_to_determine_counterfeit_l1333_133319


namespace find_fraction_l1333_133344

theorem find_fraction (f n : ℝ) (h1 : f * n - 5 = 5) (h2 : n = 50) : f = 1 / 5 :=
by
  -- skipping the proof as requested
  sorry

end find_fraction_l1333_133344


namespace perfect_square_k_value_l1333_133336

-- Given condition:
def is_perfect_square (P : ℤ) : Prop := ∃ (z : ℤ), P = z * z

-- Theorem to prove:
theorem perfect_square_k_value (a b k : ℤ) (h : is_perfect_square (4 * a^2 + k * a * b + 9 * b^2)) :
  k = 12 ∨ k = -12 :=
sorry

end perfect_square_k_value_l1333_133336


namespace pears_worth_l1333_133378

variable (apples pears : ℚ)
variable (h : 3/4 * 16 * apples = 6 * pears)

theorem pears_worth (h : 3/4 * 16 * apples = 6 * pears) : 1 / 3 * 9 * apples = 1.5 * pears :=
by
  sorry

end pears_worth_l1333_133378


namespace leftovers_value_l1333_133358

def quarters_in_roll : ℕ := 30
def dimes_in_roll : ℕ := 40
def james_quarters : ℕ := 77
def james_dimes : ℕ := 138
def lindsay_quarters : ℕ := 112
def lindsay_dimes : ℕ := 244
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftovers_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_in_roll
  let leftover_dimes := total_dimes % dimes_in_roll
  leftover_quarters * quarter_value + leftover_dimes * dime_value = 2.45 :=
by
  sorry

end leftovers_value_l1333_133358


namespace millimeters_of_78_74_inches_l1333_133380

noncomputable def inchesToMillimeters (inches : ℝ) : ℝ :=
  inches * 25.4

theorem millimeters_of_78_74_inches :
  round (inchesToMillimeters 78.74) = 2000 :=
by
  -- This theorem should assert that converting 78.74 inches to millimeters and rounding to the nearest millimeter equals 2000
  sorry

end millimeters_of_78_74_inches_l1333_133380


namespace cos120_sin_neg45_equals_l1333_133337

noncomputable def cos120_plus_sin_neg45 : ℝ :=
  Real.cos (120 * Real.pi / 180) + Real.sin (-45 * Real.pi / 180)

theorem cos120_sin_neg45_equals : cos120_plus_sin_neg45 = - (1 + Real.sqrt 2) / 2 :=
by
  sorry

end cos120_sin_neg45_equals_l1333_133337


namespace scientific_notation_of_twenty_million_l1333_133363

-- Define the number 20 million
def twenty_million : ℂ :=
  20000000

-- Define the scientific notation to be proved correct
def scientific_notation : ℂ :=
  2 * 10 ^ 7

-- The theorem to prove the equivalence
theorem scientific_notation_of_twenty_million : twenty_million = scientific_notation :=
  sorry

end scientific_notation_of_twenty_million_l1333_133363


namespace pureAcidInSolution_l1333_133330

/-- Define the conditions for the problem -/
def totalVolume : ℝ := 12
def percentageAcid : ℝ := 0.40

/-- State the theorem equivalent to the question:
    calculate the amount of pure acid -/
theorem pureAcidInSolution :
  totalVolume * percentageAcid = 4.8 := by
  sorry

end pureAcidInSolution_l1333_133330


namespace k_starts_at_10_l1333_133376

variable (V_k V_l : ℝ)
variable (t_k t_l : ℝ)

-- Conditions
axiom k_faster_than_l : V_k = 1.5 * V_l
axiom l_speed : V_l = 50
axiom l_start_time : t_l = 9
axiom meet_time : t_k + 3 = 12
axiom distance_apart : V_l * 3 + V_k * (12 - t_k) = 300

-- Proof goal
theorem k_starts_at_10 : t_k = 10 :=
by
  sorry

end k_starts_at_10_l1333_133376


namespace dandelions_survive_to_flower_l1333_133395

def seeds_initial : ℕ := 300
def seeds_in_water : ℕ := seeds_initial / 3
def seeds_eaten_by_insects : ℕ := seeds_initial / 6
def seeds_remaining : ℕ := seeds_initial - seeds_in_water - seeds_eaten_by_insects
def seeds_to_flower : ℕ := seeds_remaining / 2

theorem dandelions_survive_to_flower : seeds_to_flower = 75 := by
  sorry

end dandelions_survive_to_flower_l1333_133395


namespace cake_fraction_eaten_l1333_133338

theorem cake_fraction_eaten (total_slices kept_slices slices_eaten : ℕ) 
  (h1 : total_slices = 12)
  (h2 : kept_slices = 9)
  (h3 : slices_eaten = total_slices - kept_slices) :
  (slices_eaten : ℚ) / total_slices = 1 / 4 := 
sorry

end cake_fraction_eaten_l1333_133338


namespace total_travel_time_l1333_133375

theorem total_travel_time (distance1 distance2 speed time1: ℕ) (h1 : distance1 = 100) (h2 : time1 = 1) (h3 : distance2 = 300) (h4 : speed = distance1 / time1) :
  (time1 + distance2 / speed) = 4 :=
by
  sorry

end total_travel_time_l1333_133375


namespace determinant_of_given_matrix_l1333_133373

noncomputable def given_matrix : Matrix (Fin 4) (Fin 4) ℤ :=
![![1, -3, 3, 2], ![0, 5, -1, 0], ![4, -2, 1, 0], ![0, 0, 0, 6]]

theorem determinant_of_given_matrix :
  Matrix.det given_matrix = -270 := by
  sorry

end determinant_of_given_matrix_l1333_133373


namespace increasing_on_1_to_infinity_max_and_min_on_1_to_4_l1333_133328

noncomputable def f (x : ℝ) : ℝ := x + (1 / x)

theorem increasing_on_1_to_infinity : ∀ (x1 x2 : ℝ), 1 ≤ x1 → x1 < x2 → (1 ≤ x2) → f x1 < f x2 := by
  sorry

theorem max_and_min_on_1_to_4 : 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f x ≤ f 4) ∧ 
  (∀ (x : ℝ), 1 ≤ x → x ≤ 4 → f 1 ≤ f x) := by
  sorry

end increasing_on_1_to_infinity_max_and_min_on_1_to_4_l1333_133328


namespace trig_identity_l1333_133352

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin (π + α)) / (Real.sin (π / 2 - α)) = -2 :=
by
  sorry

end trig_identity_l1333_133352


namespace calculate_value_l1333_133369

theorem calculate_value :
  let X := (354 * 28) ^ 2
  let Y := (48 * 14) ^ 2
  (X * 9) / (Y * 2) = 2255688 :=
by
  sorry

end calculate_value_l1333_133369


namespace jack_change_l1333_133301

def cost_per_sandwich : ℕ := 5
def number_of_sandwiches : ℕ := 3
def payment : ℕ := 20

theorem jack_change : payment - (cost_per_sandwich * number_of_sandwiches) = 5 := 
by
  sorry

end jack_change_l1333_133301


namespace josie_remaining_money_l1333_133391

-- Conditions
def initial_amount : ℕ := 50
def cassette_tape_cost : ℕ := 9
def headphone_cost : ℕ := 25

-- Proof statement
theorem josie_remaining_money : initial_amount - (2 * cassette_tape_cost + headphone_cost) = 7 :=
by
  sorry

end josie_remaining_money_l1333_133391


namespace expression_value_l1333_133364

theorem expression_value (x a b c : ℝ) 
  (ha : a + x^2 = 2006) 
  (hb : b + x^2 = 2007) 
  (hc : c + x^2 = 2008) 
  (h_abc : a * b * c = 3) :
  (a / (b * c) + b / (c * a) + c / (a * b) - 1 / a - 1 / b - 1 / c = 1) := 
  sorry

end expression_value_l1333_133364


namespace least_score_to_play_final_l1333_133307

-- Definitions based on given conditions
def num_teams := 2021

def match_points (outcome : String) : ℕ :=
  match outcome with
  | "win"  => 3
  | "draw" => 1
  | "loss" => 0
  | _      => 0

def brazil_won_first_match : Prop := True

def ties_advantage (bfc_score other_team_score : ℕ) : Prop :=
  bfc_score = other_team_score

-- Theorem statement
theorem least_score_to_play_final (bfc_has_tiebreaker : (bfc_score other_team_score : ℕ) → ties_advantage bfc_score other_team_score)
  (bfc_first_match_won : brazil_won_first_match) :
  ∃ (least_score : ℕ), least_score = 2020 := sorry

end least_score_to_play_final_l1333_133307


namespace leak_empties_tank_in_30_hours_l1333_133326

-- Define the known rates based on the problem conditions
def rate_pipe_a : ℚ := 1 / 12
def combined_rate : ℚ := 1 / 20

-- Define the rate at which the leak empties the tank
def rate_leak : ℚ := rate_pipe_a - combined_rate

-- Define the time it takes for the leak to empty the tank
def time_to_empty_tank : ℚ := 1 / rate_leak

-- The theorem that needs to be proved
theorem leak_empties_tank_in_30_hours : time_to_empty_tank = 30 :=
sorry

end leak_empties_tank_in_30_hours_l1333_133326


namespace part1_solution_part2_solution_l1333_133366

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + |x + 3|

-- Part (1) Statement
theorem part1_solution (x : ℝ) :
  (f x 1 ≥ 6) ↔ (x ≤ -4 ∨ x ≥ 2) :=
sorry

-- Part (2) Statement
theorem part2_solution (a : ℝ) :
  (∀ x, f x a > -a) ↔ (a > -3/2) :=
sorry

end part1_solution_part2_solution_l1333_133366


namespace derrick_has_34_pictures_l1333_133311

-- Assume Ralph has 26 pictures of wild animals
def ralph_pictures : ℕ := 26

-- Derrick has 8 more pictures than Ralph
def derrick_pictures : ℕ := ralph_pictures + 8

-- Prove that Derrick has 34 pictures of wild animals
theorem derrick_has_34_pictures : derrick_pictures = 34 := by
  sorry

end derrick_has_34_pictures_l1333_133311


namespace divisible_by_24_l1333_133320

theorem divisible_by_24 (n : ℕ) (hn : n > 0) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) := 
by sorry

end divisible_by_24_l1333_133320


namespace number_of_boys_l1333_133388

def school_problem (x y : ℕ) : Prop :=
  (x + y = 400) ∧ (y = (x / 100) * 400)

theorem number_of_boys (x y : ℕ) (h : school_problem x y) : x = 80 :=
by
  sorry

end number_of_boys_l1333_133388


namespace quadratic_value_at_point_l1333_133353

variable (a b c : ℝ)

-- Given: A quadratic function f(x) = ax^2 + bx + c that passes through the point (3,10)
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_value_at_point
  (h : f a b c 3 = 10) :
  5 * a - 3 * b + c = -4 * a - 6 * b + 10 := by
  sorry

end quadratic_value_at_point_l1333_133353


namespace six_digit_number_divisible_9_22_l1333_133394

theorem six_digit_number_divisible_9_22 (d : ℕ) (h0 : 0 ≤ d) (h1 : d ≤ 9)
  (h2 : 9 ∣ (220140 + d)) (h3 : 22 ∣ (220140 + d)) : 220140 + d = 520146 :=
sorry

end six_digit_number_divisible_9_22_l1333_133394


namespace polynomial_has_exactly_one_real_root_l1333_133389

theorem polynomial_has_exactly_one_real_root :
  ∀ (x : ℝ), (2007 * x^3 + 2006 * x^2 + 2005 * x = 0) → x = 0 :=
by
  sorry

end polynomial_has_exactly_one_real_root_l1333_133389


namespace expected_pairs_socks_l1333_133393

noncomputable def expected_socks_to_pair (p : ℕ) : ℕ :=
2 * p

theorem expected_pairs_socks (p : ℕ) : 
  (expected_socks_to_pair p) = 2 * p := 
by 
  sorry

end expected_pairs_socks_l1333_133393


namespace chain_of_tangent_circles_iff_l1333_133318

-- Define the circles, their centers, and the conditions
structure Circle := 
  (center : ℝ × ℝ) 
  (radius : ℝ)

structure TangentData :=
  (circle1 : Circle)
  (circle2 : Circle)
  (angle : ℝ)

-- Non-overlapping condition
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let dist := (x2 - x1)^2 + (y2 - y1)^2
  dist > (c1.radius + c2.radius)^2

-- Existence of tangent circles condition
def exists_chain_of_tangent_circles (c1 c2 : Circle) (n : ℕ) : Prop :=
  ∃ (tangent_circle : Circle), tangent_circle.radius = c1.radius ∨ tangent_circle.radius = c2.radius

-- Angle condition
def angle_condition (ang : ℝ) (n : ℕ) : Prop :=
  ∃ (k : ℤ), ang = k * (360 / n)

-- Final theorem to prove
theorem chain_of_tangent_circles_iff (c1 c2 : Circle) (t : TangentData) (n : ℕ) 
  (h1 : non_overlapping c1 c2) 
  (h2 : t.circle1 = c1 ∧ t.circle2 = c2) 
  : exists_chain_of_tangent_circles c1 c2 n ↔ angle_condition t.angle n := 
  sorry

end chain_of_tangent_circles_iff_l1333_133318


namespace remaining_savings_after_purchase_l1333_133383

-- Definitions of the conditions
def cost_per_sweater : ℕ := 30
def num_sweaters : ℕ := 6
def cost_per_scarf : ℕ := 20
def num_scarves : ℕ := 6
def initial_savings : ℕ := 500

-- Theorem stating the remaining savings
theorem remaining_savings_after_purchase : initial_savings - ((cost_per_sweater * num_sweaters) + (cost_per_scarf * num_scarves)) = 200 :=
by
  -- skipping the proof
  sorry

end remaining_savings_after_purchase_l1333_133383


namespace tunnel_length_l1333_133381

def train_length : ℝ := 1.5
def exit_time_minutes : ℝ := 4
def speed_mph : ℝ := 45

theorem tunnel_length (d_train : ℝ := train_length)
                      (t_exit : ℝ := exit_time_minutes)
                      (v_mph : ℝ := speed_mph) :
  d_train + ((v_mph / 60) * t_exit - d_train) = 1.5 :=
by
  sorry

end tunnel_length_l1333_133381


namespace k_value_if_root_is_one_l1333_133368

theorem k_value_if_root_is_one (k : ℝ) (h : (k - 1) * 1 ^ 2 + 1 - k ^ 2 = 0) : k = 0 := 
by
  sorry

end k_value_if_root_is_one_l1333_133368


namespace sum_of_fractions_l1333_133325

theorem sum_of_fractions:
  (7 / 12) + (11 / 15) = 79 / 60 :=
by
  sorry

end sum_of_fractions_l1333_133325


namespace tile_floor_with_polygons_l1333_133365

theorem tile_floor_with_polygons (x y z: ℕ) (h1: 3 ≤ x) (h2: 3 ≤ y) (h3: 3 ≤ z) 
  (h_seamless: ((1 - (2 / (x: ℝ))) * 180 + (1 - (2 / (y: ℝ))) * 180 + (1 - (2 / (z: ℝ))) * 180 = 360)) :
  (1 / (x: ℝ) + 1 / (y: ℝ) + 1 / (z: ℝ) = 1 / 2) :=
by
  sorry

end tile_floor_with_polygons_l1333_133365


namespace lucas_income_36000_l1333_133339

variable (q I : ℝ)

-- Conditions as Lean 4 definitions
def tax_below_30000 : ℝ := 0.01 * q * 30000
def tax_above_30000 (I : ℝ) : ℝ := 0.01 * (q + 3) * (I - 30000)
def total_tax (I : ℝ) : ℝ := tax_below_30000 q + tax_above_30000 q I
def total_tax_condition (I : ℝ) : Prop := total_tax q I = 0.01 * (q + 0.5) * I

theorem lucas_income_36000 (h : total_tax_condition q I) : I = 36000 := by
  sorry

end lucas_income_36000_l1333_133339


namespace locus_is_circle_l1333_133314

open Complex

noncomputable def circle_center (a b : ℝ) : ℂ := Complex.ofReal (-a / (a^2 + b^2)) + Complex.I * (b / (a^2 + b^2))
noncomputable def circle_radius (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

theorem locus_is_circle (z0 z1 z : ℂ) (h1 : abs (z1 - z0) = abs z1) (h2 : z0 ≠ 0) (h3 : z1 * z = -1) :
  ∃ (a b : ℝ), z0 = Complex.ofReal a + Complex.I * b ∧
    (∃ c : ℂ, z = c ∧ 
      (c.re + a / (a^2 + b^2))^2 + (c.im - b / (a^2 + b^2))^2 = 1 / (a^2 + b^2)) := by
  sorry

end locus_is_circle_l1333_133314


namespace sales_tax_paid_l1333_133316

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tax_free_cost : ℝ)

theorem sales_tax_paid (h_total : total_cost = 25) (h_rate : tax_rate = 0.10) (h_free : tax_free_cost = 21.7) :
  ∃ (X : ℝ), 21.7 + X + (0.10 * X) = 25 ∧ (0.10 * X = 0.3) := 
by
  sorry

end sales_tax_paid_l1333_133316


namespace possible_values_of_P_l1333_133374

-- Definition of the conditions
variables (x y : ℕ) (h1 : x < y) (h2 : (x > 0)) (h3 : (y > 0))

-- Definition of P
def P : ℤ := (x^3 - y) / (1 + x * y)

-- Theorem statement
theorem possible_values_of_P : (P = 0) ∨ (P ≥ 2) :=
sorry

end possible_values_of_P_l1333_133374


namespace correct_function_at_x_equals_1_l1333_133397

noncomputable def candidate_A (x : ℝ) : ℝ := (x - 1)^3 + 3 * (x - 1)
noncomputable def candidate_B (x : ℝ) : ℝ := 2 * (x - 1)^2
noncomputable def candidate_C (x : ℝ) : ℝ := 2 * (x - 1)
noncomputable def candidate_D (x : ℝ) : ℝ := x - 1

theorem correct_function_at_x_equals_1 :
  (deriv candidate_A 1 = 3) ∧ 
  (deriv candidate_B 1 ≠ 3) ∧ 
  (deriv candidate_C 1 ≠ 3) ∧ 
  (deriv candidate_D 1 ≠ 3) := 
by
  sorry

end correct_function_at_x_equals_1_l1333_133397


namespace maximum_value_of_function_l1333_133331

noncomputable def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem maximum_value_of_function :
  ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f x_max = 25 / 4 :=
by 
  sorry

end maximum_value_of_function_l1333_133331


namespace compute_c_plus_d_l1333_133310

theorem compute_c_plus_d (c d : ℕ) (h1 : d = c^3) (h2 : d - c = 435) : c + d = 520 :=
sorry

end compute_c_plus_d_l1333_133310


namespace find_x_l1333_133371

theorem find_x :
  ∃ x : ℝ, 12.1212 + x - 9.1103 = 20.011399999999995 ∧ x = 18.000499999999995 :=
sorry

end find_x_l1333_133371


namespace transport_load_with_trucks_l1333_133367

theorem transport_load_with_trucks
  (total_weight : ℕ)
  (box_max_weight : ℕ)
  (truck_capacity : ℕ)
  (num_trucks : ℕ)
  (H_weight : total_weight = 13500)
  (H_box : box_max_weight = 350)
  (H_truck : truck_capacity = 1500)
  (H_num_trucks : num_trucks = 11) :
  ∃ (boxes : ℕ), boxes * box_max_weight >= total_weight ∧ num_trucks * truck_capacity >= total_weight := 
sorry

end transport_load_with_trucks_l1333_133367


namespace circle_regions_l1333_133342

def regions_divided_by_chords (n : ℕ) : ℕ :=
  (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24

theorem circle_regions (n : ℕ) : 
  regions_divided_by_chords n = (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24 := 
  by 
  sorry

end circle_regions_l1333_133342


namespace pen_cost_is_2_25_l1333_133312

variables (p i : ℝ)

def total_cost (p i : ℝ) : Prop := p + i = 2.50
def pen_more_expensive (p i : ℝ) : Prop := p = 2 + i

theorem pen_cost_is_2_25 (p i : ℝ) 
  (h1 : total_cost p i) 
  (h2 : pen_more_expensive p i) : 
  p = 2.25 := 
by
  sorry

end pen_cost_is_2_25_l1333_133312


namespace concentration_after_5500_evaporates_l1333_133302

noncomputable def concentration_after_evaporation 
  (V₀ Vₑ : ℝ) (C₀ : ℝ) : ℝ := 
  let sodium_chloride := C₀ * V₀
  let remaining_volume := V₀ - Vₑ
  100 * sodium_chloride / remaining_volume

theorem concentration_after_5500_evaporates 
  : concentration_after_evaporation 10000 5500 0.05 = 11.11 := 
by
  -- Formalize the calculations as we have derived
  -- sorry is used to skip the proof
  sorry

end concentration_after_5500_evaporates_l1333_133302


namespace compute_expression_l1333_133300

theorem compute_expression :
  (-9 * 5 - (-7 * -2) + (-11 * -4)) = -15 :=
by
  sorry

end compute_expression_l1333_133300


namespace unique_solution_value_k_l1333_133361

theorem unique_solution_value_k (k : ℚ) :
  (∀ x : ℚ, (x + 3) / (k * x - 2) = x → x = -2) ↔ k = -3 / 4 :=
by
  sorry

end unique_solution_value_k_l1333_133361


namespace point_M_coordinates_l1333_133396

theorem point_M_coordinates :
  (∃ (M : ℝ × ℝ), M.1 < 0 ∧ M.2 > 0 ∧ abs M.2 = 2 ∧ abs M.1 = 1 ∧ M = (-1, 2)) :=
by
  use (-1, 2)
  sorry

end point_M_coordinates_l1333_133396


namespace identify_conic_section_hyperbola_l1333_133321

-- Defining the variables and constants in the Lean environment
variable (x y : ℝ)

-- The given equation in function form
def conic_section_eq : Prop := (x - 3) ^ 2 = 4 * (y + 2) ^ 2 + 25

-- The expected type of conic section (Hyperbola)
def is_hyperbola : Prop := 
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^2 - b * y^2 + c * x + d * y + e = f

-- The theorem statement to prove
theorem identify_conic_section_hyperbola (h : conic_section_eq x y) : is_hyperbola x y := by
  sorry

end identify_conic_section_hyperbola_l1333_133321


namespace initial_birds_l1333_133333

-- Define the initial number of birds (B) and the fact that 13 more birds flew up to the tree
-- Define that the total number of birds after 13 more birds joined is 42
theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
by
  sorry

end initial_birds_l1333_133333


namespace cube_paint_problem_l1333_133382

theorem cube_paint_problem : 
  ∀ (n : ℕ),
  n = 6 →
  (∃ k : ℕ, 216 = k^3 ∧ k = n) →
  ∀ (faces inner_faces total_cubelets : ℕ),
  faces = 6 →
  inner_faces = 4 →
  total_cubelets = faces * (inner_faces * inner_faces) →
  total_cubelets = 96 :=
by 
  intros n hn hc faces hfaces inner_faces hinner_faces total_cubelets htotal_cubelets
  sorry

end cube_paint_problem_l1333_133382


namespace two_digit_number_as_expression_l1333_133350

-- Define the conditions of the problem
variables (a : ℕ)

-- Statement to be proved
theorem two_digit_number_as_expression (h : 0 ≤ a ∧ a ≤ 9) : 10 * a + 1 = 10 * a + 1 := by
  sorry

end two_digit_number_as_expression_l1333_133350
