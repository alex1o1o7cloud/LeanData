import Mathlib

namespace shawn_divided_into_groups_l722_72226

theorem shawn_divided_into_groups :
  ∀ (total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups : ℕ),
  total_pebbles = 40 →
  red_pebbles = 9 →
  blue_pebbles = 13 →
  remaining_pebbles = total_pebbles - red_pebbles - blue_pebbles →
  remaining_pebbles % 3 = 0 →
  yellow_pebbles = blue_pebbles - 7 →
  remaining_pebbles = groups * yellow_pebbles →
  groups = 3 :=
by
  intros total_pebbles red_pebbles blue_pebbles remaining_pebbles yellow_pebbles groups
  intros h_total h_red h_blue h_remaining h_divisible h_yellow h_group
  sorry

end shawn_divided_into_groups_l722_72226


namespace sum_first_49_odd_numbers_l722_72294

theorem sum_first_49_odd_numbers : (49^2 = 2401) :=
by
  sorry

end sum_first_49_odd_numbers_l722_72294


namespace misty_is_three_times_smaller_l722_72208

-- Define constants representing the favorite numbers of Misty and Glory
def G : ℕ := 450
def total_sum : ℕ := 600

-- Define Misty's favorite number in terms of the total sum and Glory's favorite number
def M : ℕ := total_sum - G

-- The main theorem stating that Misty's favorite number is 3 times smaller than Glory's favorite number
theorem misty_is_three_times_smaller : G / M = 3 := by
  -- Sorry placeholder indicating the need for further proof
  sorry

end misty_is_three_times_smaller_l722_72208


namespace period_of_sin_sub_cos_l722_72283

open Real

theorem period_of_sin_sub_cos :
  ∃ T > 0, ∀ x, sin x - cos x = sin (x + T) - cos (x + T) ∧ T = 2 * π := sorry

end period_of_sin_sub_cos_l722_72283


namespace intersection_M_N_l722_72232

open Set Real

def M : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def N : Set ℝ := {x | log x / log 2 ≤ 1}

theorem intersection_M_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_M_N_l722_72232


namespace shoe_price_l722_72230

theorem shoe_price :
  ∀ (P : ℝ),
    (6 * P + 18 * 2 = 27 * 2) → P = 3 :=
by
  intro P H
  sorry

end shoe_price_l722_72230


namespace percentage_of_female_officers_on_duty_l722_72247

-- Declare the conditions
def total_officers_on_duty : ℕ := 100
def female_officers_on_duty : ℕ := 50
def total_female_officers : ℕ := 250

-- The theorem to prove
theorem percentage_of_female_officers_on_duty :
  (female_officers_on_duty / total_female_officers) * 100 = 20 := 
sorry

end percentage_of_female_officers_on_duty_l722_72247


namespace dollar_expansion_l722_72231

variable (x y : ℝ)

def dollar (a b : ℝ) : ℝ := (a + b) ^ 2 + a * b

theorem dollar_expansion : dollar ((x - y) ^ 3) ((y - x) ^ 3) = -((x - y) ^ 6) := by
  sorry

end dollar_expansion_l722_72231


namespace minimum_value_of_f_l722_72228

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 5) / (2 * x - 4)

theorem minimum_value_of_f (x : ℝ) (h : x ≥ 5 / 2) : ∃ y, y = f x ∧ y = 1 :=
by
  sorry

end minimum_value_of_f_l722_72228


namespace maintenance_cost_relation_maximize_average_profit_l722_72297

def maintenance_cost (n : ℕ) : ℕ :=
  if n = 1 then 0 else 1400 * n - 1000

theorem maintenance_cost_relation :
  maintenance_cost 2 = 1800 ∧ maintenance_cost 5 = 6000 ∧
  (∀ n, n ≥ 2 → maintenance_cost n = 1400 * n - 1000) :=
by
  sorry

noncomputable def average_profit (n : ℕ) : ℝ :=
  if n < 2 then 0 else 60000 - (1 / n) * (137600 + 1400 * ((n - 1) * (n + 2) / 2) - 1000 * (n - 1))

theorem maximize_average_profit (n : ℕ) :
  n = 14 ↔ (average_profit n = 40700) :=
by
  sorry

end maintenance_cost_relation_maximize_average_profit_l722_72297


namespace factor_tree_value_l722_72287

theorem factor_tree_value :
  let F := 7 * (2 * 2)
  let H := 11 * 2
  let G := 11 * H
  let X := F * G
  X = 6776 :=
by
  sorry

end factor_tree_value_l722_72287


namespace original_ratio_of_boarders_to_day_students_l722_72243

theorem original_ratio_of_boarders_to_day_students
    (original_boarders : ℕ)
    (new_boarders : ℕ)
    (new_ratio_b_d : ℕ → ℕ)
    (no_switch : Prop)
    (no_leave : Prop)
  : (original_boarders = 220) ∧ (new_boarders = 44) ∧ (new_ratio_b_d 1 = 2) ∧ no_switch ∧ no_leave →
  ∃ (original_day_students : ℕ), original_day_students = 528 ∧ (220 / 44 = 5) ∧ (528 / 44 = 12)
  := by
    sorry

end original_ratio_of_boarders_to_day_students_l722_72243


namespace divide_by_repeating_decimal_l722_72248

theorem divide_by_repeating_decimal : (8 : ℚ) / (1 / 3) = 24 := by
  sorry

end divide_by_repeating_decimal_l722_72248


namespace point_above_line_l722_72276

/-- Given the point (-2, t) lies above the line x - 2y + 4 = 0,
    we want to prove t ∈ (1, +∞) -/
theorem point_above_line (t : ℝ) : (-2 - 2 * t + 4 > 0) → t > 1 :=
sorry

end point_above_line_l722_72276


namespace total_test_subjects_l722_72293

-- Defining the conditions as mathematical entities
def number_of_colors : ℕ := 5
def unique_two_color_codes : ℕ := number_of_colors * number_of_colors
def excess_subjects : ℕ := 6

-- Theorem stating the question and correct answer
theorem total_test_subjects :
  unique_two_color_codes + excess_subjects = 31 :=
by
  -- Leaving the proof as sorry, since the task only requires statement creation
  sorry

end total_test_subjects_l722_72293


namespace inequality_solution_l722_72214

-- Declare the constants m and n
variables (m n : ℝ)

-- State the conditions
def condition1 (x : ℝ) := m < 0
def condition2 := n = -m / 2

-- State the theorem
theorem inequality_solution (x : ℝ) (h1 : condition1 m n) (h2 : condition2 m n) : 
  nx - m < 0 ↔ x < -2 :=
sorry

end inequality_solution_l722_72214


namespace trigonometric_identity_l722_72207

open Real

theorem trigonometric_identity
  (α : ℝ)
  (h1 : 0 ≤ α ∧ α ≤ π / 2)
  (h2 : cos α = 3 / 5) :
  (1 + sqrt 2 * cos (2 * α - π / 4)) / sin (α + π / 2) = 14 / 5 :=
by
  sorry

end trigonometric_identity_l722_72207


namespace exists_sequences_satisfying_conditions_l722_72263

noncomputable def satisfies_conditions (n : ℕ) (hn : Odd n) 
  (a : Fin n → ℕ) (b : Fin n → ℕ) : Prop :=
  ∀ (k : Fin n), 0 < k.val → k.val < n →
    ∀ (i : Fin n),
      let in3n := 3 * n;
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n ≠
      (a i + b i) % in3n ∧
      (a i + b i) % in3n ≠
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ∧
      (b i + b ⟨(i.val + k.val) % n, sorry⟩) % in3n ≠
      (a i + a ⟨(i.val + 1) % n, sorry⟩) % in3n

theorem exists_sequences_satisfying_conditions :
  ∀ n : ℕ, Odd n → ∃ (a : Fin n → ℕ) (b : Fin n → ℕ),
    satisfies_conditions n sorry a b :=
sorry

end exists_sequences_satisfying_conditions_l722_72263


namespace albert_needs_more_money_l722_72268

-- Definitions derived from the problem conditions
def cost_paintbrush : ℝ := 1.50
def cost_paints : ℝ := 4.35
def cost_easel : ℝ := 12.65
def money_albert_has : ℝ := 6.50

-- Statement asserting the amount of money Albert needs
theorem albert_needs_more_money : (cost_paintbrush + cost_paints + cost_easel) - money_albert_has = 12 :=
by
  sorry

end albert_needs_more_money_l722_72268


namespace bruce_will_be_3_times_as_old_in_6_years_l722_72221

variables (x : ℕ)

-- Definitions from conditions
def bruce_age_now := 36
def son_age_now := 8

-- Equivalent Lean 4 statement
theorem bruce_will_be_3_times_as_old_in_6_years :
  (bruce_age_now + x = 3 * (son_age_now + x)) → x = 6 :=
sorry

end bruce_will_be_3_times_as_old_in_6_years_l722_72221


namespace min_cut_length_l722_72235

theorem min_cut_length (x : ℝ) (h_longer : 23 - x ≥ 0) (h_shorter : 15 - x ≥ 0) :
  23 - x ≥ 2 * (15 - x) → x ≥ 7 :=
by
  sorry

end min_cut_length_l722_72235


namespace totalGames_l722_72289

-- Define Jerry's original number of video games
def originalGames : ℕ := 7

-- Define the number of video games Jerry received for his birthday
def birthdayGames : ℕ := 2

-- Statement: Prove that the total number of games Jerry has now is 9
theorem totalGames : originalGames + birthdayGames = 9 := by
  sorry

end totalGames_l722_72289


namespace find_linear_combination_l722_72264

variable (a b c : ℝ)

theorem find_linear_combination (h1 : a + 2 * b - 3 * c = 4)
                               (h2 : 5 * a - 6 * b + 7 * c = 8) :
  9 * a + 2 * b - 5 * c = 24 :=
sorry

end find_linear_combination_l722_72264


namespace find_percentage_l722_72296

theorem find_percentage (x : ℝ) (h1 : x = 780) (h2 : ∀ P : ℝ, P / 100 * x = 225 - 30) : P = 25 :=
by
  -- Definitions and conditions here
  -- Recall: x = 780 and P / 100 * x = 195
  sorry

end find_percentage_l722_72296


namespace tea_blend_ratio_l722_72291

theorem tea_blend_ratio (x y : ℝ)
  (h1 : 18 * x + 20 * y = (21 * (x + y)) / 1.12)
  (h2 : x + y ≠ 0) :
  x / y = 5 / 3 :=
by
  -- proof will go here
  sorry

end tea_blend_ratio_l722_72291


namespace Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l722_72203

-- Problem 1: Factorization
theorem Problem1_factorize (a : ℝ) : a^2 - 8 * a + 15 = (a - 3) * (a - 5) :=
  sorry

-- Problem 2: Minimum Perimeter of triangle ABC
theorem Problem2_min_perimeter_triangle (a b c : ℝ) 
  (h : a^2 + b^2 - 14 * a - 8 * b + 65 = 0) (hc : ∃ k : ℤ, 2 * k + 1 = c) : 
  a + b + c ≥ 16 :=
  sorry

-- Problem 3: Maximum Value of the Polynomial
theorem Problem3_max_value_polynomial : 
  ∃ x : ℝ, x = -1 ∧ ∀ y : ℝ, y ≠ -1 → -2 * x^2 - 4 * x + 3 ≥ -2 * y^2 - 4 * y + 3 :=
  sorry

end Problem1_factorize_Problem2_min_perimeter_triangle_Problem3_max_value_polynomial_l722_72203


namespace incorrect_description_l722_72273

-- Conditions
def population_size : ℕ := 2000
def sample_size : ℕ := 150

-- Main Statement
theorem incorrect_description : ¬ (sample_size = 150) := 
by sorry

end incorrect_description_l722_72273


namespace nonagon_perimeter_l722_72254

theorem nonagon_perimeter (n : ℕ) (side_length : ℝ) (P : ℝ) :
  n = 9 → side_length = 3 → P = n * side_length → P = 27 :=
by sorry

end nonagon_perimeter_l722_72254


namespace rationalize_denominator_sum_l722_72279

noncomputable def rationalize_denominator (x y z : ℤ) :=
  x = 4 ∧ y = 49 ∧ z = 35 ∧ y ∣ 343 ∧ z > 0 

theorem rationalize_denominator_sum : 
  ∃ A B C : ℤ, rationalize_denominator A B C ∧ A + B + C = 88 :=
by
  sorry

end rationalize_denominator_sum_l722_72279


namespace floor_neg_seven_thirds_l722_72255

theorem floor_neg_seven_thirds : Int.floor (-7 / 3 : ℚ) = -3 := by
  sorry

end floor_neg_seven_thirds_l722_72255


namespace sum_reciprocals_factors_12_l722_72272

theorem sum_reciprocals_factors_12 : 
  (1 + 1/2 + 1/3 + 1/4 + 1/6 + 1/12) = 7/3 :=
by
  sorry

end sum_reciprocals_factors_12_l722_72272


namespace aubrey_travel_time_l722_72267

def aubrey_time_to_school (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem aubrey_travel_time :
  aubrey_time_to_school 88 22 = 4 := by
  sorry

end aubrey_travel_time_l722_72267


namespace sin_double_angle_l722_72234

theorem sin_double_angle {x : ℝ} (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 :=
sorry

end sin_double_angle_l722_72234


namespace percentage_less_than_a_plus_d_l722_72246

-- Define the mean, standard deviation, and given conditions
variables (a d : ℝ)
axiom symmetric_distribution : ∀ x, x = 2 * a - x 

-- Main theorem
theorem percentage_less_than_a_plus_d :
  (∃ (P_less_than : ℝ → ℝ), P_less_than (a + d) = 0.84) :=
sorry

end percentage_less_than_a_plus_d_l722_72246


namespace personal_trainer_cost_proof_l722_72295

-- Define the conditions
def hourly_wage_before_raise : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 5
def old_bills_per_week : ℝ := 600
def leftover_money : ℝ := 980

-- Define the question
def new_hourly_wage : ℝ := hourly_wage_before_raise * (1 + raise_percentage)
def weekly_hours : ℕ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def total_weekly_expenses : ℝ := weekly_earnings - leftover_money
def personal_trainer_cost_per_week : ℝ := total_weekly_expenses - old_bills_per_week

-- Theorem statement
theorem personal_trainer_cost_proof : personal_trainer_cost_per_week = 100 := 
by
  -- Proof to be filled
  sorry

end personal_trainer_cost_proof_l722_72295


namespace range_of_a_l722_72271

theorem range_of_a (x a : ℝ) :
  (∀ x, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l722_72271


namespace ab_value_l722_72218

theorem ab_value (a b : ℝ) 
  (h1 : ∃ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1 ∧ (∀ y : ℝ, (x = 0 ∧ (y = 5 ∨ y = -5)))))
  (h2 : ∃ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1 ∧ (∀ x : ℝ, (y = 0 ∧ (x = 8 ∨ x = -8))))) :
  |a * b| = Real.sqrt 867.75 :=
by
  sorry

end ab_value_l722_72218


namespace compute_fraction_sum_l722_72274

theorem compute_fraction_sum
  (a b c : ℝ)
  (h : a^3 - 6 * a^2 + 11 * a = 12)
  (h : b^3 - 6 * b^2 + 11 * b = 12)
  (h : c^3 - 6 * c^2 + 11 * c = 12) :
  (ab : ℝ) / c + (bc : ℝ) / a + (ca : ℝ) / b = -23 / 12 := by
  sorry

end compute_fraction_sum_l722_72274


namespace min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l722_72259

theorem min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared 
  (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  (ae^2 : ℝ) + (bf^2 : ℝ) + (cg^2 : ℝ) + (dh^2 : ℝ) ≥ 32 := 
sorry

end min_value_ae_squared_plus_bf_squared_plus_cg_squared_plus_dh_squared_l722_72259


namespace Ria_original_savings_l722_72211

variables {R F : ℕ}

def initial_ratio (R F : ℕ) : Prop :=
  R * 3 = F * 5

def withdrawn_amount (R : ℕ) : ℕ :=
  R - 160

def new_ratio (R' F : ℕ) : Prop :=
  R' * 5 = F * 3

theorem Ria_original_savings (initial_ratio: initial_ratio R F)
  (new_ratio: new_ratio (withdrawn_amount R) F) : 
  R = 250 :=
by
  sorry

end Ria_original_savings_l722_72211


namespace delivery_in_april_l722_72217

theorem delivery_in_april (n_jan n_mar : ℕ) (growth_rate : ℝ) :
  n_jan = 100000 → n_mar = 121000 → (1 + growth_rate) ^ 2 = n_mar / n_jan →
  (n_mar * (1 + growth_rate) = 133100) :=
by
  intros n_jan_eq n_mar_eq growth_eq
  sorry

end delivery_in_april_l722_72217


namespace watch_sticker_price_l722_72233

theorem watch_sticker_price (x : ℝ)
  (hx_X : 0.80 * x - 50 = y)
  (hx_Y : 0.90 * x = z)
  (savings : z - y = 25) : 
  x = 250 := by
  sorry

end watch_sticker_price_l722_72233


namespace price_per_glass_on_first_day_eq_half_l722_72201

structure OrangeadeProblem where
  O : ℝ
  W : ℝ
  P1 : ℝ
  P2 : ℝ
  W_eq_O : W = O
  P2_value : P2 = 0.3333333333333333
  revenue_eq : 2 * O * P1 = 3 * O * P2

theorem price_per_glass_on_first_day_eq_half (prob : OrangeadeProblem) : prob.P1 = 0.50 := 
by
  sorry

end price_per_glass_on_first_day_eq_half_l722_72201


namespace theater_total_bills_l722_72219

theorem theater_total_bills (tickets : ℕ) (price : ℕ) (x : ℕ) (number_of_5_bills : ℕ) (number_of_10_bills : ℕ) (number_of_20_bills : ℕ) :
  tickets = 300 →
  price = 40 →
  number_of_20_bills = x →
  number_of_10_bills = 2 * x →
  number_of_5_bills = 2 * x + 20 →
  20 * x + 10 * (2 * x) + 5 * (2 * x + 20) = tickets * price →
  number_of_5_bills + number_of_10_bills + number_of_20_bills = 1210 := by
    intro h_tickets h_price h_20_bills h_10_bills h_5_bills h_total
    sorry

end theater_total_bills_l722_72219


namespace average_weight_of_students_l722_72215

theorem average_weight_of_students (b_avg_weight g_avg_weight : ℝ) (num_boys num_girls : ℕ)
  (hb : b_avg_weight = 155) (hg : g_avg_weight = 125) (hb_num : num_boys = 8) (hg_num : num_girls = 5) :
  (num_boys * b_avg_weight + num_girls * g_avg_weight) / (num_boys + num_girls) = 143 :=
by sorry

end average_weight_of_students_l722_72215


namespace smallest_k_l722_72242

-- Define the set S
def S (m : ℕ) : Finset ℕ :=
  (Finset.range (30 * m)).filter (λ n => n % 2 = 1 ∧ n % 5 ≠ 0)

-- Theorem statement
theorem smallest_k (m : ℕ) (k : ℕ) : 
  (∀ (A : Finset ℕ), A ⊆ S m → A.card = k → ∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (x ∣ y ∨ y ∣ x)) ↔ k ≥ 8 * m + 1 :=
sorry

end smallest_k_l722_72242


namespace cara_total_amount_owed_l722_72237

-- Define the conditions
def principal : ℝ := 54
def rate : ℝ := 0.05
def time : ℝ := 1

-- Define the simple interest calculation
def interest (P R T : ℝ) : ℝ := P * R * T

-- Define the total amount owed calculation
def total_amount_owed (P R T : ℝ) : ℝ := P + (interest P R T)

-- The proof statement
theorem cara_total_amount_owed : total_amount_owed principal rate time = 56.70 := by
  sorry

end cara_total_amount_owed_l722_72237


namespace smallest_uv_non_factor_of_48_l722_72244

theorem smallest_uv_non_factor_of_48 :
  ∃ (u v : ℕ) (hu : u ∣ 48) (hv : v ∣ 48), u ≠ v ∧ ¬ (u * v ∣ 48) ∧ u * v = 18 :=
sorry

end smallest_uv_non_factor_of_48_l722_72244


namespace quadratic_root_property_l722_72202

theorem quadratic_root_property (a b k : ℝ) 
  (h1 : a * b + 2 * a + 2 * b = 1) 
  (h2 : a + b = 3) 
  (h3 : a * b = k) : k = -5 := 
by
  sorry

end quadratic_root_property_l722_72202


namespace angle_measure_l722_72290

-- Define the problem conditions
def angle (x : ℝ) : Prop :=
  let complement := 3 * x + 6
  x + complement = 90

-- The theorem to prove
theorem angle_measure : ∃ x : ℝ, angle x ∧ x = 21 := 
sorry

end angle_measure_l722_72290


namespace sin_double_angle_value_l722_72258

theorem sin_double_angle_value (α : ℝ) (h₁ : Real.sin (π / 4 - α) = 3 / 5) (h₂ : 0 < α ∧ α < π / 4) : 
  Real.sin (2 * α) = 7 / 25 := 
sorry

end sin_double_angle_value_l722_72258


namespace no_real_roots_of_quadratic_l722_72256

theorem no_real_roots_of_quadratic (a : ℝ) : 
  (∀ x : ℝ, 3 * x^2 + 2 * a * x + 1 ≠ 0) ↔ a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) := by
  sorry

end no_real_roots_of_quadratic_l722_72256


namespace binom_20_17_l722_72241

theorem binom_20_17 : Nat.choose 20 17 = 1140 := by
  sorry

end binom_20_17_l722_72241


namespace coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l722_72280

/- The problem's conditions and questions rephrased for Lean:
  1. Prove: if \( n \) is divisible by 3, then a valid coloring is possible.
  2. Prove: if a valid coloring is possible, then \( n \) is divisible by 3.
-/

def is_colorable (n : ℕ) : Prop :=
  ∃ (colors : Fin 3 → Fin n → Fin 3),
    ∀ (i j : Fin n), i ≠ j → (colors 0 i ≠ colors 0 j ∧ colors 1 i ≠ colors 1 j ∧ colors 2 i ≠ colors 2 j)

theorem coloring_possible_if_divisible_by_three (n : ℕ) (h : n % 3 = 0) : is_colorable n :=
  sorry

theorem divisible_by_three_if_coloring_possible (n : ℕ) (h : is_colorable n) : n % 3 = 0 :=
  sorry

end coloring_possible_if_divisible_by_three_divisible_by_three_if_coloring_possible_l722_72280


namespace sum_prime_factors_77_l722_72229

theorem sum_prime_factors_77 : ∀ (p1 p2 : ℕ), Nat.Prime p1 ∧ Nat.Prime p2 ∧ 77 = p1 * p2 → p1 + p2 = 18 :=
by
  intros p1 p2 h
  sorry

end sum_prime_factors_77_l722_72229


namespace hyuksu_total_meat_l722_72262

/-- 
Given that Hyuksu ate 2.6 kilograms (kg) of meat yesterday and 5.98 kilograms (kg) of meat today,
prove that the total kilograms (kg) of meat he ate in two days is 8.58 kg.
-/
theorem hyuksu_total_meat (yesterday today : ℝ) (hy1 : yesterday = 2.6) (hy2 : today = 5.98) :
  yesterday + today = 8.58 := 
by
  rw [hy1, hy2]
  norm_num

end hyuksu_total_meat_l722_72262


namespace quadratic_rewrite_l722_72285

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 6
noncomputable def c : ℕ := 284
noncomputable def quadratic_coeffs_sum : ℕ := a + b + c

theorem quadratic_rewrite :
  (∃ a b c : ℕ, 6 * (x : ℕ) ^ 2 + 72 * x + 500 = a * (x + b) ^ 2 + c) →
  quadratic_coeffs_sum = 296 := by sorry

end quadratic_rewrite_l722_72285


namespace B_pow_101_eq_B_pow_5_l722_72240

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 0]]

theorem B_pow_101_eq_B_pow_5 : B^101 = B := 
by sorry

end B_pow_101_eq_B_pow_5_l722_72240


namespace bridge_length_l722_72277

theorem bridge_length (lorry_length : ℝ) (lorry_speed_kmph : ℝ) (cross_time_seconds : ℝ) : 
  lorry_length = 200 ∧ lorry_speed_kmph = 80 ∧ cross_time_seconds = 17.998560115190784 →
  lorry_length + lorry_speed_kmph * (1000 / 3600) * cross_time_seconds = 400 → 
  400 - lorry_length = 200 :=
by
  intro h₁ h₂
  cases h₁
  sorry

end bridge_length_l722_72277


namespace ramesh_paid_price_l722_72212

variables 
  (P : Real) -- Labelled price of the refrigerator
  (paid_price : Real := 0.80 * P + 125 + 250) -- Price paid after discount and additional costs
  (sell_price : Real := 1.16 * P) -- Price to sell for 16% profit
  (sell_at : Real := 18560) -- Target selling price for given profit

theorem ramesh_paid_price : 
  1.16 * P = 18560 → paid_price = 13175 :=
by
  sorry

end ramesh_paid_price_l722_72212


namespace product_of_divisor_and_dividend_l722_72223

theorem product_of_divisor_and_dividend (d D : ℕ) (q : ℕ := 6) (r : ℕ := 3) 
  (h₁ : D = d + 78) 
  (h₂ : D = d * q + r) : 
  D * d = 1395 :=
by 
  sorry

end product_of_divisor_and_dividend_l722_72223


namespace circle_length_l722_72261

theorem circle_length (n : ℕ) (arm_span : ℝ) (overlap : ℝ) (contribution : ℝ) (total_length : ℝ) :
  n = 16 ->
  arm_span = 10.4 ->
  overlap = 3.5 ->
  contribution = arm_span - overlap ->
  total_length = n * contribution ->
  total_length = 110.4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end circle_length_l722_72261


namespace cindy_gave_lisa_marbles_l722_72249

-- Definitions for the given conditions
def cindy_initial_marbles : ℕ := 20
def lisa_initial_marbles := cindy_initial_marbles - 5
def lisa_final_marbles := lisa_initial_marbles + 19

-- Theorem we need to prove
theorem cindy_gave_lisa_marbles :
  ∃ n : ℕ, lisa_final_marbles = lisa_initial_marbles + n ∧ n = 19 :=
by
  sorry

end cindy_gave_lisa_marbles_l722_72249


namespace volume_of_prism_l722_72216

theorem volume_of_prism (a b c : ℝ)
  (h_ab : a * b = 36)
  (h_ac : a * c = 54)
  (h_bc : b * c = 72) :
  a * b * c = 648 :=
by
  sorry

end volume_of_prism_l722_72216


namespace trig_identity_l722_72265

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 :=
by 
  sorry

end trig_identity_l722_72265


namespace asian_population_percentage_in_west_l722_72245

theorem asian_population_percentage_in_west
    (NE MW South West : ℕ)
    (H_NE : NE = 2)
    (H_MW : MW = 3)
    (H_South : South = 2)
    (H_West : West = 6)
    : (West * 100) / (NE + MW + South + West) = 46 :=
sorry

end asian_population_percentage_in_west_l722_72245


namespace probability_three_primes_out_of_five_l722_72213

def probability_of_prime (p : ℚ) : Prop := ∃ k, k = 4 ∧ p = 4/10

def probability_of_not_prime (p : ℚ) : Prop := ∃ k, k = 6 ∧ p = 6/10

def combinations (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_three_primes_out_of_five :
  ∀ p_prime p_not_prime : ℚ, 
  probability_of_prime p_prime →
  probability_of_not_prime p_not_prime →
  (combinations 5 3 * (p_prime^3 * p_not_prime^2) = 720/3125) :=
by
  intros p_prime p_not_prime h_prime h_not_prime
  sorry

end probability_three_primes_out_of_five_l722_72213


namespace laura_total_owed_l722_72238

-- Define the principal amounts charged each month
def january_charge : ℝ := 35
def february_charge : ℝ := 45
def march_charge : ℝ := 55
def april_charge : ℝ := 25

-- Define the respective interest rates for each month, as decimals
def january_interest_rate : ℝ := 0.05
def february_interest_rate : ℝ := 0.07
def march_interest_rate : ℝ := 0.04
def april_interest_rate : ℝ := 0.06

-- Define the interests accrued for each month's charges
def january_interest : ℝ := january_charge * january_interest_rate
def february_interest : ℝ := february_charge * february_interest_rate
def march_interest : ℝ := march_charge * march_interest_rate
def april_interest : ℝ := april_charge * april_interest_rate

-- Define the totals including original charges and their respective interests
def january_total : ℝ := january_charge + january_interest
def february_total : ℝ := february_charge + february_interest
def march_total : ℝ := march_charge + march_interest
def april_total : ℝ := april_charge + april_interest

-- Define the total amount owed a year later
def total_owed : ℝ := january_total + february_total + march_total + april_total

-- Prove that the total amount owed a year later is $168.60
theorem laura_total_owed :
  total_owed = 168.60 := by
  sorry

end laura_total_owed_l722_72238


namespace lateral_surface_area_base_area_ratio_correct_l722_72227

noncomputable def lateral_surface_area_to_base_area_ratio
  (S P Q R : Type)
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12)
  : ℝ :=
  π * (4 * Real.sqrt 3 - 3) / 13

theorem lateral_surface_area_base_area_ratio_correct
  {S P Q R : Type}
  (angle_PSR angle_SQR angle_PSQ : ℝ)
  (h_PSR : angle_PSR = π / 2)
  (h_SQR : angle_SQR = π / 4)
  (h_PSQ : angle_PSQ = 7 * π / 12) :
  lateral_surface_area_to_base_area_ratio S P Q R angle_PSR angle_SQR angle_PSQ
    h_PSR h_SQR h_PSQ = π * (4 * Real.sqrt 3 - 3) / 13 :=
  by sorry

end lateral_surface_area_base_area_ratio_correct_l722_72227


namespace point_not_in_fourth_quadrant_l722_72281

theorem point_not_in_fourth_quadrant (m : ℝ) : ¬(m-2 > 0 ∧ m+1 < 0) := 
by
  -- Since (m+1) - (m-2) = 3, which is positive,
  -- m+1 > m-2, thus the statement ¬(m-2 > 0 ∧ m+1 < 0) holds.
  sorry

end point_not_in_fourth_quadrant_l722_72281


namespace sebastian_total_payment_l722_72204

theorem sebastian_total_payment 
  (cost_per_ticket : ℕ) (number_of_tickets : ℕ) (service_fee : ℕ) (total_paid : ℕ)
  (h1 : cost_per_ticket = 44)
  (h2 : number_of_tickets = 3)
  (h3 : service_fee = 18)
  (h4 : total_paid = (number_of_tickets * cost_per_ticket) + service_fee) :
  total_paid = 150 :=
by
  sorry

end sebastian_total_payment_l722_72204


namespace log_expression_equality_l722_72222

noncomputable def evaluate_log_expression : Real :=
  let log4_8 := (Real.log 8) / (Real.log 4)
  let log5_10 := (Real.log 10) / (Real.log 5)
  Real.sqrt (log4_8 + log5_10)

theorem log_expression_equality : 
  evaluate_log_expression = Real.sqrt ((5 / 2) + (Real.log 2 / Real.log 5)) :=
by
  sorry

end log_expression_equality_l722_72222


namespace pentagon_area_inequality_l722_72286

-- Definitions for the problem
structure Point :=
(x y : ℝ)

structure Triangle :=
(A B C : Point)

noncomputable def area (T : Triangle) : ℝ :=
  1 / 2 * abs ((T.B.x - T.A.x) * (T.C.y - T.A.y) - (T.C.x - T.A.x) * (T.B.y - T.A.y))

structure Pentagon :=
(A B C D E : Point)

noncomputable def pentagon_area (P : Pentagon) : ℝ :=
  area ⟨P.A, P.B, P.C⟩ + area ⟨P.A, P.C, P.D⟩ + area ⟨P.A, P.D, P.E⟩ -
  area ⟨P.E, P.B, P.C⟩

-- Given conditions
variables (A B C D E F : Point)
variables (P : Pentagon) 
-- P is a convex pentagon with points A, B, C, D, E in order 

-- Intersection point of AD and EC is F 
axiom intersect_diagonals (AD EC : Triangle) : AD.C = F ∧ EC.B = F

-- Theorem statement
theorem pentagon_area_inequality :
  let AED := Triangle.mk A E D
  let EDC := Triangle.mk E D C
  let EAB := Triangle.mk E A B
  let DCB := Triangle.mk D C B
  area AED + area EDC + area EAB + area DCB > pentagon_area P :=
  sorry

end pentagon_area_inequality_l722_72286


namespace ratio_expression_l722_72236

theorem ratio_expression (p q s u : ℚ) (h1 : p / q = 3 / 5) (h2 : s / u = 8 / 11) : 
  (4 * p * s - 3 * q * u) / (5 * q * u - 8 * p * s) = -69 / 83 :=
by
  sorry

end ratio_expression_l722_72236


namespace coefficient_x3y5_l722_72206

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the condition for the binomial expansion term of (x-y)^7
def expansion_term (r : ℕ) : ℤ := 
  (binom 7 r) * (-1) ^ r

-- The target coefficient for the term x^3 y^5 in (x+y)(x-y)^7
theorem coefficient_x3y5 :
  (expansion_term 5) * 1 + (expansion_term 4) * 1 = 14 :=
by
  -- Proof to be filled in
  sorry

end coefficient_x3y5_l722_72206


namespace mr_slinkums_shipments_l722_72299

theorem mr_slinkums_shipments 
  (T : ℝ) 
  (h : (3 / 4) * T = 150) : 
  T = 200 := 
sorry

end mr_slinkums_shipments_l722_72299


namespace maximize_area_l722_72269

theorem maximize_area (P L W : ℝ) (h1 : P = 2 * L + 2 * W) (h2 : 0 < P) : 
  (L = P / 4) ∧ (W = P / 4) :=
by
  sorry

end maximize_area_l722_72269


namespace complement_A_B_eq_singleton_three_l722_72252

open Set

variable (A : Set ℕ) (B : Set ℕ) (a : ℕ)

theorem complement_A_B_eq_singleton_three (hA : A = {2, 3, 4})
    (hB : B = {a + 2, a}) (h_inter : A ∩ B = B) : A \ B = {3} :=
  sorry

end complement_A_B_eq_singleton_three_l722_72252


namespace polynomial_decomposition_l722_72282

noncomputable def s (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1
noncomputable def t (x : ℝ) : ℝ := x + 18

def g (x : ℝ) : ℝ := 3 * x^4 + 9 * x^3 - 7 * x^2 + 2 * x + 6
def e (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem polynomial_decomposition : s 1 + t (-1) = 27 :=
by
  sorry

end polynomial_decomposition_l722_72282


namespace pam_bags_equiv_gerald_bags_l722_72239

theorem pam_bags_equiv_gerald_bags :
  ∀ (total_apples pam_bags apples_per_gerald_bag : ℕ), 
    total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 → 
    (total_apples / pam_bags) / apples_per_gerald_bag = 3 :=
by
  intros total_apples pam_bags apples_per_gerald_bag h
  obtain ⟨ht, hp, hg⟩ : total_apples = 1200 ∧ pam_bags = 10 ∧ apples_per_gerald_bag = 40 := h
  sorry

end pam_bags_equiv_gerald_bags_l722_72239


namespace scientific_notation_of_8_5_million_l722_72266

theorem scientific_notation_of_8_5_million :
  (8.5 * 10^6) = 8500000 :=
by sorry

end scientific_notation_of_8_5_million_l722_72266


namespace zhang_hua_repayment_l722_72275

noncomputable def principal_amount : ℕ := 480000
noncomputable def repayment_period : ℕ := 240
noncomputable def monthly_interest_rate : ℝ := 0.004
noncomputable def principal_payment : ℝ := principal_amount / repayment_period -- 2000, but keeping general form

noncomputable def interest (month : ℕ) : ℝ :=
  (principal_amount - (month - 1) * principal_payment) * monthly_interest_rate

noncomputable def monthly_repayment (month : ℕ) : ℝ :=
  principal_payment + interest month

theorem zhang_hua_repayment (n : ℕ) (h : 1 ≤ n ∧ n ≤ repayment_period) :
  monthly_repayment n = 3928 - 8 * n := 
by
  -- proof would be placed here
  sorry

end zhang_hua_repayment_l722_72275


namespace lock_probability_l722_72278

/-- The probability of correctly guessing the last digit of a three-digit combination lock,
given that the first two digits are correctly set and each digit ranges from 0 to 9. -/
theorem lock_probability : 
  ∀ (d1 d2 : ℕ), 
  (0 ≤ d1 ∧ d1 < 10) ∧ (0 ≤ d2 ∧ d2 < 10) →
  (0 ≤ d3 ∧ d3 < 10) → 
  (1/10 : ℝ) = (1 : ℝ) / (10 : ℝ) :=
by
  sorry

end lock_probability_l722_72278


namespace algebraic_expression_independence_l722_72257

theorem algebraic_expression_independence (a b : ℝ) (h : ∀ x : ℝ, (x^2 + a*x - (b*x^2 - x - 3)) = 3) : a - b = -2 :=
by
  sorry

end algebraic_expression_independence_l722_72257


namespace sum_of_integers_greater_than_2_and_less_than_15_l722_72220

-- Define the set of integers greater than 2 and less than 15
def integersInRange : List ℕ := [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define the sum of these integers
def sumIntegersInRange : ℕ := integersInRange.sum

-- The main theorem to prove the sum
theorem sum_of_integers_greater_than_2_and_less_than_15 : sumIntegersInRange = 102 := by
  -- The proof part is omitted as per instructions
  sorry

end sum_of_integers_greater_than_2_and_less_than_15_l722_72220


namespace calculate_total_cost_l722_72270

theorem calculate_total_cost : 
  let piano_cost := 500
  let lesson_cost_per_lesson := 40
  let number_of_lessons := 20
  let discount_rate := 0.25
  let missed_lessons := 3
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_lesson_cost := number_of_lessons * lesson_cost_per_lesson
  let discount := total_lesson_cost * discount_rate
  let discounted_lesson_cost := total_lesson_cost - discount
  let cost_of_missed_lessons := missed_lessons * lesson_cost_per_lesson
  let effective_lesson_cost := discounted_lesson_cost + cost_of_missed_lessons
  let total_cost := piano_cost + effective_lesson_cost + sheet_music_cost + maintenance_fees
  total_cost = 1395 :=
by
  sorry

end calculate_total_cost_l722_72270


namespace sequence_formula_l722_72253

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 3^n) :
  ∀ n : ℕ, a n = (3^n - 1) / 2 :=
sorry

end sequence_formula_l722_72253


namespace part1_inequality_solution_l722_72224

def f (x : ℝ) : ℝ := |x + 1| + |2 * x - 3|

theorem part1_inequality_solution :
  ∀ x : ℝ, f x ≤ 6 ↔ -4 / 3 ≤ x ∧ x ≤ 8 / 3 :=
by sorry

end part1_inequality_solution_l722_72224


namespace triangle_inequality_for_roots_l722_72250

theorem triangle_inequality_for_roots (p q r : ℝ) (hroots_pos : ∀ (u v w : ℝ), (u > 0) ∧ (v > 0) ∧ (w > 0) ∧ (u * v * w = -r) ∧ (u + v + w = -p) ∧ (u * v + u * w + v * w = q)) :
  p^3 - 4 * p * q + 8 * r > 0 :=
sorry

end triangle_inequality_for_roots_l722_72250


namespace smallest_a_condition_l722_72225

theorem smallest_a_condition
  (a b : ℝ)
  (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_eq : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a = 15 :=
sorry

end smallest_a_condition_l722_72225


namespace integer_solutions_count_l722_72251

theorem integer_solutions_count :
  let eq : Int -> Int -> Int := fun x y => 6 * y ^ 2 + 3 * x * y + x + 2 * y - 72
  ∃ (sols : List (Int × Int)), 
    (∀ x y, eq x y = 0 → (x, y) ∈ sols) ∧
    (∀ p ∈ sols, ∃ x y, p = (x, y) ∧ eq x y = 0) ∧
    sols.length = 4 :=
by
  sorry

end integer_solutions_count_l722_72251


namespace shopkeeper_loss_percent_l722_72292

theorem shopkeeper_loss_percent 
  (C : ℝ) (P : ℝ) (L : ℝ) 
  (hC : C = 100) 
  (hP : P = 10) 
  (hL : L = 50) : 
  ((C - (((C * (1 - L / 100)) * (1 + P / 100))) / C) * 100) = 45 :=
by
  sorry

end shopkeeper_loss_percent_l722_72292


namespace right_triangle_incircle_excircle_condition_l722_72210

theorem right_triangle_incircle_excircle_condition
  (r R : ℝ) 
  (hr_pos : 0 < r) 
  (hR_pos : 0 < R) :
  R ≥ r * (3 + 2 * Real.sqrt 2) := sorry

end right_triangle_incircle_excircle_condition_l722_72210


namespace arithmetic_sequence_30th_term_l722_72200

theorem arithmetic_sequence_30th_term :
  let a := 3
  let d := 7 - 3
  ∀ n, (n = 30) → (a + (n - 1) * d) = 119 := by
  sorry

end arithmetic_sequence_30th_term_l722_72200


namespace triangle_II_area_l722_72298

noncomputable def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  1 / 2 * base * height

theorem triangle_II_area (a b : ℝ) :
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  II_area = (a + b) ^ 2 :=
by
  let I_area := triangle_area (a + b) (a + b)
  let II_area := 2 * I_area
  sorry

end triangle_II_area_l722_72298


namespace trig_identity_l722_72284

variable (α : Real)
variable (h : Real.tan α = 2)

theorem trig_identity :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3 / 4 := by
  sorry

end trig_identity_l722_72284


namespace cost_of_fencing_per_meter_l722_72209

theorem cost_of_fencing_per_meter (l b : ℕ) (total_cost : ℕ) (cost_per_meter : ℝ) : 
  (l = 66) → 
  (l = b + 32) → 
  (total_cost = 5300) → 
  (2 * l + 2 * b = 200) → 
  (cost_per_meter = total_cost / 200) → 
  cost_per_meter = 26.5 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof is omitted by design
  sorry

end cost_of_fencing_per_meter_l722_72209


namespace sphere_radius_l722_72260

theorem sphere_radius (A : ℝ) (k1 k2 k3 : ℝ) (h : A = 64 * Real.pi) : ∃ r : ℝ, r = 4 := 
by 
  sorry

end sphere_radius_l722_72260


namespace original_wire_length_l722_72205

theorem original_wire_length (side_len total_area : ℕ) (h1 : side_len = 2) (h2 : total_area = 92) :
  (total_area / (side_len * side_len)) * (4 * side_len) = 184 := 
by
  sorry

end original_wire_length_l722_72205


namespace certain_events_l722_72288

-- Define the idioms and their classifications
inductive Event
| impossible
| certain
| unlikely

-- Definitions based on the given conditions
def scooping_moon := Event.impossible
def rising_tide := Event.certain
def waiting_by_stump := Event.unlikely
def catching_turtles := Event.certain
def pulling_seeds := Event.impossible

-- The theorem statement
theorem certain_events :
  (rising_tide = Event.certain) ∧ (catching_turtles = Event.certain) := by
  -- Proof is omitted
  sorry

end certain_events_l722_72288
