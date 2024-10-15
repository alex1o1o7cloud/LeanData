import Mathlib

namespace NUMINAMATH_GPT_find_constants_l763_76366

theorem find_constants : 
  ∃ (a b : ℝ), a • (⟨1, 4⟩ : ℝ × ℝ) + b • (⟨3, -2⟩ : ℝ × ℝ) = (⟨5, 6⟩ : ℝ × ℝ) ∧ a = 2 ∧ b = 1 :=
by 
  sorry

end NUMINAMATH_GPT_find_constants_l763_76366


namespace NUMINAMATH_GPT_oliver_final_amount_is_54_04_l763_76370

noncomputable def final_amount : ℝ :=
  let initial := 33
  let feb_spent := 0.15 * initial
  let after_feb := initial - feb_spent
  let march_added := 32
  let after_march := after_feb + march_added
  let march_spent := 0.10 * after_march
  after_march - march_spent

theorem oliver_final_amount_is_54_04 : final_amount = 54.04 := by
  sorry

end NUMINAMATH_GPT_oliver_final_amount_is_54_04_l763_76370


namespace NUMINAMATH_GPT_equalize_champagne_futile_l763_76328

/-- Stepashka cannot distribute champagne into 2018 glasses in such a way 
that Kryusha's attempts to equalize the amount in all glasses become futile. -/
theorem equalize_champagne_futile (n : ℕ) (h : n = 2018) : 
∃ (a : ℕ), (∀ (A B : ℕ), A ≠ B ∧ A + B = 2019 → (A + B) % 2 = 1) := 
sorry

end NUMINAMATH_GPT_equalize_champagne_futile_l763_76328


namespace NUMINAMATH_GPT_path_area_and_cost_correct_l763_76338

def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_of_path : ℝ := 759.36
def cost_per_sqm : ℝ := 2
def total_cost : ℝ := 1518.72

theorem path_area_and_cost_correct :
    let length_with_path := length_field + 2 * path_width
    let width_with_path := width_field + 2 * path_width
    let area_with_path := length_with_path * width_with_path
    let area_field := length_field * width_field
    let calculated_area_of_path := area_with_path - area_field
    let calculated_total_cost := calculated_area_of_path * cost_per_sqm
    calculated_area_of_path = area_of_path ∧ calculated_total_cost = total_cost :=
by
    sorry

end NUMINAMATH_GPT_path_area_and_cost_correct_l763_76338


namespace NUMINAMATH_GPT_problem_f_of_3_l763_76365

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + 1 else -2 * x + 3

theorem problem_f_of_3 : f (f 3) = 10 := by
  sorry

end NUMINAMATH_GPT_problem_f_of_3_l763_76365


namespace NUMINAMATH_GPT_sequence_term_is_correct_l763_76334

theorem sequence_term_is_correct : ∀ (n : ℕ), (n = 7) → (2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_is_correct_l763_76334


namespace NUMINAMATH_GPT_hyunwoo_family_saving_l763_76335

def daily_water_usage : ℝ := 215
def saving_factor : ℝ := 0.32

theorem hyunwoo_family_saving:
  daily_water_usage * saving_factor = 68.8 := by
  sorry

end NUMINAMATH_GPT_hyunwoo_family_saving_l763_76335


namespace NUMINAMATH_GPT_solve_cubic_eq_solve_quadratic_eq_l763_76303

-- Define the first equation and prove its solution
theorem solve_cubic_eq (x : ℝ) (h : x^3 + 64 = 0) : x = -4 :=
by
  -- skipped proof
  sorry

-- Define the second equation and prove its solutions
theorem solve_quadratic_eq (x : ℝ) (h : (x - 2)^2 = 81) : x = 11 ∨ x = -7 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_solve_cubic_eq_solve_quadratic_eq_l763_76303


namespace NUMINAMATH_GPT_simplify_cosine_expression_l763_76391

theorem simplify_cosine_expression :
  ∀ (θ : ℝ), θ = 30 * Real.pi / 180 → (1 - Real.cos θ) * (1 + Real.cos θ) = 1 / 4 :=
by
  intro θ hθ
  have cos_30 := Real.cos θ
  rewrite [hθ]
  sorry

end NUMINAMATH_GPT_simplify_cosine_expression_l763_76391


namespace NUMINAMATH_GPT_find_t_l763_76331

theorem find_t (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
    (havg : (k + m + r + s + t) / 5 = 18)
    (hmed : r = 23) 
    (hpos_k : 0 < k)
    (hpos_m : 0 < m)
    (hpos_r : 0 < r)
    (hpos_s : 0 < s)
    (hpos_t : 0 < t) :
  t = 40 := sorry

end NUMINAMATH_GPT_find_t_l763_76331


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l763_76311

def A : Set ℚ := { x | x^2 - 4*x + 3 < 0 }
def B : Set ℚ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l763_76311


namespace NUMINAMATH_GPT_hyperbola_focal_distance_and_asymptotes_l763_76390

-- Define the hyperbola
def hyperbola (y x : ℝ) : Prop := (y^2 / 4) - (x^2 / 3) = 1

-- Prove the properties
theorem hyperbola_focal_distance_and_asymptotes :
  (∀ y x : ℝ, hyperbola y x → ∃ c : ℝ, c = 2 * Real.sqrt 7)
  ∧
  (∀ y x : ℝ, hyperbola y x → (y = (2 * Real.sqrt 3 / 3) * x ∨ y = -(2 * Real.sqrt 3 / 3) * x)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focal_distance_and_asymptotes_l763_76390


namespace NUMINAMATH_GPT_product_of_first_two_terms_l763_76354

theorem product_of_first_two_terms (a_7 : ℕ) (d : ℕ) (a_7_eq : a_7 = 17) (d_eq : d = 2) :
  let a_1 := a_7 - 6 * d
  let a_2 := a_1 + d
  a_1 * a_2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_product_of_first_two_terms_l763_76354


namespace NUMINAMATH_GPT_relationship_between_number_and_square_l763_76302

theorem relationship_between_number_and_square (n : ℕ) (h : n = 9) :
  (n + n^2) / 2 = 5 * n := by
    sorry

end NUMINAMATH_GPT_relationship_between_number_and_square_l763_76302


namespace NUMINAMATH_GPT_no_prime_divisible_by_56_l763_76359

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be divisible by another number
def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ ∃ k : ℕ, a = b * k

-- The main theorem stating the problem
theorem no_prime_divisible_by_56 : ¬ ∃ p : ℕ, is_prime p ∧ divisible_by p 56 :=
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_56_l763_76359


namespace NUMINAMATH_GPT_degree_greater_than_2_l763_76343

variable (P Q : ℤ[X]) -- P and Q are polynomials with integer coefficients

theorem degree_greater_than_2 (P_nonconstant : ¬(P.degree = 0))
  (Q_nonconstant : ¬(Q.degree = 0))
  (h : ∃ S : Finset ℤ, S.card ≥ 25 ∧ ∀ x ∈ S, (P.eval x) * (Q.eval x) = 2009) :
  P.degree > 2 ∧ Q.degree > 2 :=
by
  sorry

end NUMINAMATH_GPT_degree_greater_than_2_l763_76343


namespace NUMINAMATH_GPT_exists_non_decreasing_subsequences_l763_76322

theorem exists_non_decreasing_subsequences {a b c : ℕ → ℕ} : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end NUMINAMATH_GPT_exists_non_decreasing_subsequences_l763_76322


namespace NUMINAMATH_GPT_model_N_completion_time_l763_76387

variable (T : ℕ)

def model_M_time : ℕ := 36
def number_of_M_computers : ℕ := 12
def number_of_N_computers := number_of_M_computers -- given that they are the same.

-- Statement of the problem: Given the conditions, prove T = 18
theorem model_N_completion_time :
  (number_of_M_computers : ℝ) * (1 / model_M_time) + (number_of_N_computers : ℝ) * (1 / T) = 1 →
  T = 18 :=
by
  sorry

end NUMINAMATH_GPT_model_N_completion_time_l763_76387


namespace NUMINAMATH_GPT_eight_painters_finish_in_required_days_l763_76310

/- Conditions setup -/
def initial_painters : ℕ := 6
def initial_days : ℕ := 2
def job_constant := initial_painters * initial_days

def new_painters : ℕ := 8
def required_days := 3 / 2

/- Theorem statement -/
theorem eight_painters_finish_in_required_days : new_painters * required_days = job_constant :=
sorry

end NUMINAMATH_GPT_eight_painters_finish_in_required_days_l763_76310


namespace NUMINAMATH_GPT_eight_points_on_circle_l763_76349

theorem eight_points_on_circle
  (R : ℝ) (hR : R > 0)
  (points : Fin 8 → (ℝ × ℝ))
  (hpoints : ∀ i : Fin 8, (points i).1 ^ 2 + (points i).2 ^ 2 ≤ R ^ 2) :
  ∃ (i j : Fin 8), i ≠ j ∧ (dist (points i) (points j) < R) :=
sorry

end NUMINAMATH_GPT_eight_points_on_circle_l763_76349


namespace NUMINAMATH_GPT_list_price_is_35_l763_76317

-- Define the conditions in Lean
variable (x : ℝ)

def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * (alice_selling_price x)

def bob_selling_price (x : ℝ) : ℝ := x - 20
def bob_commission (x : ℝ) : ℝ := 0.20 * (bob_selling_price x)

-- Define the theorem to be proven
theorem list_price_is_35 (x : ℝ) 
  (h : alice_commission x = bob_commission x) : x = 35 :=
by sorry

end NUMINAMATH_GPT_list_price_is_35_l763_76317


namespace NUMINAMATH_GPT_arith_geo_mean_extended_arith_geo_mean_l763_76344
noncomputable section

open Real

-- Definition for Problem 1
def arith_geo_mean_inequality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : Prop :=
  (a + b) / 2 ≥ Real.sqrt (a * b)

-- Theorem for Problem 1
theorem arith_geo_mean (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : arith_geo_mean_inequality a b h1 h2 :=
  sorry

-- Definition for Problem 2
def extended_arith_geo_mean_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c

-- Theorem for Problem 2
theorem extended_arith_geo_mean (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : extended_arith_geo_mean_inequality a b c h1 h2 h3 :=
  sorry

end NUMINAMATH_GPT_arith_geo_mean_extended_arith_geo_mean_l763_76344


namespace NUMINAMATH_GPT_combined_ticket_cost_l763_76316

variables (S K : ℕ)

theorem combined_ticket_cost (total_budget : ℕ) (samuel_food_drink : ℕ) (kevin_food : ℕ) (kevin_drink : ℕ) :
  total_budget = 20 →
  samuel_food_drink = 6 →
  kevin_food = 4 →
  kevin_drink = 2 →
  S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget →
  S + K = 8 :=
by
  intros h_total_budget h_samuel_food_drink h_kevin_food h_kevin_drink h_total_spent
  /-
  We have the following conditions:
  1. total_budget = 20
  2. samuel_food_drink = 6
  3. kevin_food = 4
  4. kevin_drink = 2
  5. S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget

  We need to prove that S + K = 8. We can use the conditions to derive this.
  -/
  rw [h_total_budget, h_samuel_food_drink, h_kevin_food, h_kevin_drink] at h_total_spent
  exact sorry

end NUMINAMATH_GPT_combined_ticket_cost_l763_76316


namespace NUMINAMATH_GPT_painting_time_l763_76363

theorem painting_time (rate_taylor rate_jennifer rate_alex : ℚ) 
  (h_taylor : rate_taylor = 1 / 12) 
  (h_jennifer : rate_jennifer = 1 / 10) 
  (h_alex : rate_alex = 1 / 15) : 
  ∃ t : ℚ, t = 4 ∧ (1 / t) = rate_taylor + rate_jennifer + rate_alex :=
by
  sorry

end NUMINAMATH_GPT_painting_time_l763_76363


namespace NUMINAMATH_GPT_total_football_games_l763_76377

theorem total_football_games (months : ℕ) (games_per_month : ℕ) (season_length : months = 17 ∧ games_per_month = 19) :
  (months * games_per_month) = 323 :=
by
  sorry

end NUMINAMATH_GPT_total_football_games_l763_76377


namespace NUMINAMATH_GPT_coeff_x2_in_x_minus_1_pow_4_l763_76350

theorem coeff_x2_in_x_minus_1_pow_4 :
  ∀ (x : ℝ), (∃ (p : ℕ), (x - 1) ^ 4 = p * x^2 + (other_terms) ∧ p = 6) :=
by sorry

end NUMINAMATH_GPT_coeff_x2_in_x_minus_1_pow_4_l763_76350


namespace NUMINAMATH_GPT_intersection_counts_l763_76330

theorem intersection_counts (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = -x^2 + 4 * x - 3)
  (hg : ∀ x, g x = -f x)
  (hh : ∀ x, h x = f (-x))
  (c : ℕ) (hc : c = 2)
  (d : ℕ) (hd : d = 1):
  10 * c + d = 21 :=
by
  sorry

end NUMINAMATH_GPT_intersection_counts_l763_76330


namespace NUMINAMATH_GPT_tea_mixture_ratio_l763_76383

theorem tea_mixture_ratio
    (x y : ℝ)
    (h₁ : 62 * x + 72 * y = 64.5 * (x + y)) :
    x / y = 3 := by
  sorry

end NUMINAMATH_GPT_tea_mixture_ratio_l763_76383


namespace NUMINAMATH_GPT_new_avg_weight_l763_76352

theorem new_avg_weight 
  (initial_avg_weight : ℝ)
  (initial_num_members : ℕ)
  (new_person1_weight : ℝ)
  (new_person2_weight : ℝ)
  (new_num_members : ℕ)
  (final_total_weight : ℝ)
  (final_avg_weight : ℝ) :
  initial_avg_weight = 48 →
  initial_num_members = 23 →
  new_person1_weight = 78 →
  new_person2_weight = 93 →
  new_num_members = initial_num_members + 2 →
  final_total_weight = (initial_avg_weight * initial_num_members) + new_person1_weight + new_person2_weight →
  final_avg_weight = final_total_weight / new_num_members →
  final_avg_weight = 51 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_new_avg_weight_l763_76352


namespace NUMINAMATH_GPT_sherman_weekend_driving_time_l763_76318

def total_driving_time_per_week : ℕ := 9
def commute_time_per_day : ℕ := 1
def work_days_per_week : ℕ := 5
def weekend_days : ℕ := 2

theorem sherman_weekend_driving_time :
  (total_driving_time_per_week - commute_time_per_day * work_days_per_week) / weekend_days = 2 :=
sorry

end NUMINAMATH_GPT_sherman_weekend_driving_time_l763_76318


namespace NUMINAMATH_GPT_probability_of_region_l763_76394

theorem probability_of_region :
  let area_rect := (1000: ℝ) * 1500
  let area_polygon := 500000
  let prob := area_polygon / area_rect
  prob = (1 / 3) := sorry

end NUMINAMATH_GPT_probability_of_region_l763_76394


namespace NUMINAMATH_GPT_possible_numbers_erased_one_digit_reduce_sixfold_l763_76337

theorem possible_numbers_erased_one_digit_reduce_sixfold (N : ℕ) :
  (∃ N' : ℕ, N = 6 * N' ∧ N % 10 ≠ 0 ∧ ¬N = N') ↔
  N = 12 ∨ N = 24 ∨ N = 36 ∨ N = 48 ∨ N = 108 :=
by {
  sorry
}

end NUMINAMATH_GPT_possible_numbers_erased_one_digit_reduce_sixfold_l763_76337


namespace NUMINAMATH_GPT_value_of_f_eval_at_pi_over_12_l763_76372

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem value_of_f_eval_at_pi_over_12 : f (Real.pi / 12) = (Real.sqrt 6) / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_eval_at_pi_over_12_l763_76372


namespace NUMINAMATH_GPT_find_x_in_triangle_XYZ_l763_76393

theorem find_x_in_triangle_XYZ (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ) (hx : y = 7) (hz : z = 6) (hcos : cos_Y_minus_Z = 47 / 64) : 
    ∃ x : ℝ, x = Real.sqrt 63.75 :=
by
  -- The proof will go here, but it is skipped for now.
  sorry

end NUMINAMATH_GPT_find_x_in_triangle_XYZ_l763_76393


namespace NUMINAMATH_GPT_length_of_platform_l763_76336

theorem length_of_platform 
  (speed_train_kmph : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_man : ℝ)
  (conversion_factor : ℝ)
  (speed_train_mps : ℝ)
  (length_train : ℝ)
  (total_distance : ℝ)
  (length_platform : ℝ) :
  speed_train_kmph = 150 →
  time_cross_platform = 45 →
  time_cross_man = 20 →
  conversion_factor = (1000 / 3600) →
  speed_train_mps = speed_train_kmph * conversion_factor →
  length_train = speed_train_mps * time_cross_man →
  total_distance = speed_train_mps * time_cross_platform →
  length_platform = total_distance - length_train →
  length_platform = 1041.75 :=
by sorry

end NUMINAMATH_GPT_length_of_platform_l763_76336


namespace NUMINAMATH_GPT_rounding_to_one_decimal_place_l763_76357

def number_to_round : Float := 5.049

def rounded_value : Float := 5.0

theorem rounding_to_one_decimal_place :
  (Float.round (number_to_round * 10) / 10) = rounded_value :=
by
  sorry

end NUMINAMATH_GPT_rounding_to_one_decimal_place_l763_76357


namespace NUMINAMATH_GPT_pradeep_maximum_marks_l763_76339

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.35 * M = 175) :
  M = 500 :=
by
  sorry

end NUMINAMATH_GPT_pradeep_maximum_marks_l763_76339


namespace NUMINAMATH_GPT_range_of_a_for_zero_l763_76367

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_zero (a : ℝ) : a ≤ 2 * Real.log 2 - 2 → ∃ x : ℝ, f a x = 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_for_zero_l763_76367


namespace NUMINAMATH_GPT_pencil_cost_l763_76353

theorem pencil_cost 
  (x y : ℚ)
  (h1 : 3 * x + 2 * y = 165)
  (h2 : 4 * x + 7 * y = 303) :
  y = 19.155 := 
by
  sorry

end NUMINAMATH_GPT_pencil_cost_l763_76353


namespace NUMINAMATH_GPT_remainder_division_l763_76315

theorem remainder_division (N : ℤ) (R1 : ℤ) (Q2 : ℤ) 
  (h1 : N = 44 * 432 + R1)
  (h2 : N = 38 * Q2 + 8) : 
  R1 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_division_l763_76315


namespace NUMINAMATH_GPT_cost_formula_l763_76326

-- Definitions based on conditions
def base_cost : ℕ := 15
def additional_cost_per_pound : ℕ := 5
def environmental_fee : ℕ := 2

-- Definition of cost function
def cost (P : ℕ) : ℕ := base_cost + additional_cost_per_pound * (P - 1) + environmental_fee

-- Theorem stating the formula for the cost C
theorem cost_formula (P : ℕ) (h : 1 ≤ P) : cost P = 12 + 5 * P :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cost_formula_l763_76326


namespace NUMINAMATH_GPT_determine_n_l763_76348

noncomputable def P : ℤ → ℤ := sorry

theorem determine_n (n : ℕ) (P : ℤ → ℤ)
  (h_deg : ∀ x : ℤ, P x = 2 ∨ P x = 1 ∨ P x = 0)
  (h0 : ∀ k : ℕ, k ≤ n → P (3 * k) = 2)
  (h1 : ∀ k : ℕ, k < n → P (3 * k + 1) = 1)
  (h2 : ∀ k : ℕ, k < n → P (3 * k + 2) = 0)
  (h_f : P (3 * n + 1) = 730) :
  n = 4 := 
sorry

end NUMINAMATH_GPT_determine_n_l763_76348


namespace NUMINAMATH_GPT_average_speed_of_car_l763_76300

-- Definitions of the given conditions
def uphill_speed : ℝ := 30  -- km/hr
def downhill_speed : ℝ := 70  -- km/hr
def uphill_distance : ℝ := 100  -- km
def downhill_distance : ℝ := 50  -- km

-- Required proof statement (with the correct answer derived from the conditions)
theorem average_speed_of_car :
  (uphill_distance + downhill_distance) / 
  ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = 37.04 := by
  sorry

end NUMINAMATH_GPT_average_speed_of_car_l763_76300


namespace NUMINAMATH_GPT_max_number_of_girls_l763_76325

theorem max_number_of_girls (students : ℕ)
  (num_friends : ℕ → ℕ)
  (h_students : students = 25)
  (h_distinct_friends : ∀ (i j : ℕ), i ≠ j → num_friends i ≠ num_friends j)
  (h_girls_boys : ∃ (G B : ℕ), G + B = students) :
  ∃ G : ℕ, G = 13 := 
sorry

end NUMINAMATH_GPT_max_number_of_girls_l763_76325


namespace NUMINAMATH_GPT_valid_four_digit_number_count_l763_76309

theorem valid_four_digit_number_count : 
  let first_digit_choices := 6 
  let last_digit_choices := 10 
  let middle_digits_valid_pairs := 9 * 9 - 18
  (first_digit_choices * middle_digits_valid_pairs * last_digit_choices = 3780) := by
  sorry

end NUMINAMATH_GPT_valid_four_digit_number_count_l763_76309


namespace NUMINAMATH_GPT_total_books_read_l763_76373

-- Given conditions
variables (c s : ℕ) -- variable c represents the number of classes, s represents the number of students per class

-- Main statement to prove
theorem total_books_read (h1 : ∀ a, a = 7) (h2 : ∀ b, b = 12) :
  84 * c * s = 84 * c * s :=
by
  sorry

end NUMINAMATH_GPT_total_books_read_l763_76373


namespace NUMINAMATH_GPT_product_of_roots_eq_neg_125_over_4_l763_76384

theorem product_of_roots_eq_neg_125_over_4 :
  (∀ x y : ℝ, (24 * x^2 + 60 * x - 750 = 0 ∧ 24 * y^2 + 60 * y - 750 = 0 ∧ x ≠ y) → x * y = -125 / 4) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_product_of_roots_eq_neg_125_over_4_l763_76384


namespace NUMINAMATH_GPT_initial_walnut_trees_l763_76385

theorem initial_walnut_trees (total_trees_after_planting : ℕ) (trees_planted_today : ℕ) (initial_trees : ℕ) : 
  (total_trees_after_planting = 55) → (trees_planted_today = 33) → (initial_trees + trees_planted_today = total_trees_after_planting) → (initial_trees = 22) :=
by
  sorry

end NUMINAMATH_GPT_initial_walnut_trees_l763_76385


namespace NUMINAMATH_GPT_area_of_shape_is_correct_l763_76395

noncomputable def square_side_length : ℝ := 2 * Real.pi

noncomputable def semicircle_radius : ℝ := square_side_length / 2

noncomputable def area_of_resulting_shape : ℝ :=
  let area_square := square_side_length^2
  let area_semicircle := (1/2) * Real.pi * semicircle_radius^2
  let total_area := area_square + 4 * area_semicircle
  total_area

theorem area_of_shape_is_correct :
  area_of_resulting_shape = 2 * Real.pi^2 * (Real.pi + 2) :=
sorry

end NUMINAMATH_GPT_area_of_shape_is_correct_l763_76395


namespace NUMINAMATH_GPT_smallest_n_for_gcd_l763_76360

theorem smallest_n_for_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 4) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 4) > 1 → n ≤ m) → n = 38 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_gcd_l763_76360


namespace NUMINAMATH_GPT_find_expression_l763_76378

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_about_x2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem find_expression (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : symmetric_about_x2 f)
  (h3 : ∀ x, -2 < x ∧ x ≤ 2 → f x = -x^2 + 1) :
  ∀ x, -6 < x ∧ x < -2 → f x = -(x + 4)^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_l763_76378


namespace NUMINAMATH_GPT_not_solution_B_l763_76381

theorem not_solution_B : ¬ (1 + 6 = 5) := by
  sorry

end NUMINAMATH_GPT_not_solution_B_l763_76381


namespace NUMINAMATH_GPT_elementary_school_coats_correct_l763_76308

def total_coats : ℕ := 9437
def high_school_coats : ℕ := (3 * total_coats) / 5
def elementary_school_coats := total_coats - high_school_coats

theorem elementary_school_coats_correct : 
  elementary_school_coats = 3775 :=
by
  sorry

end NUMINAMATH_GPT_elementary_school_coats_correct_l763_76308


namespace NUMINAMATH_GPT_isosceles_trapezoid_sides_length_l763_76376

theorem isosceles_trapezoid_sides_length (b1 b2 A : ℝ) (h s : ℝ) 
  (hb1 : b1 = 11) (hb2 : b2 = 17) (hA : A = 56) :
  (A = 1/2 * (b1 + b2) * h) →
  (s ^ 2 = h ^ 2 + (b2 - b1) ^ 2 / 4) →
  s = 5 :=
by
  intro
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_sides_length_l763_76376


namespace NUMINAMATH_GPT_transylvanian_human_truth_transylvanian_vampire_lie_l763_76340

-- Definitions of predicates for human and vampire behavior
def is_human (A : Type) : Prop := ∀ (X : Prop), (A → X) → X
def is_vampire (A : Type) : Prop := ∀ (X : Prop), (A → X) → ¬X

-- Lean definitions for the problem
theorem transylvanian_human_truth (A : Type) (X : Prop) (h_human : is_human A) (h_says_true : A → X) :
  X :=
by sorry

theorem transylvanian_vampire_lie (A : Type) (X : Prop) (h_vampire : is_vampire A) (h_says_true : A → X) :
  ¬X :=
by sorry

end NUMINAMATH_GPT_transylvanian_human_truth_transylvanian_vampire_lie_l763_76340


namespace NUMINAMATH_GPT_no_geometric_progression_11_12_13_l763_76379

theorem no_geometric_progression_11_12_13 :
  ∀ (b1 : ℝ) (q : ℝ) (k l n : ℕ), 
  (b1 * q ^ (k - 1) = 11) → 
  (b1 * q ^ (l - 1) = 12) → 
  (b1 * q ^ (n - 1) = 13) → 
  False :=
by
  intros b1 q k l n hk hl hn
  sorry

end NUMINAMATH_GPT_no_geometric_progression_11_12_13_l763_76379


namespace NUMINAMATH_GPT_nublian_total_words_l763_76396

-- Define the problem's constants and conditions
def nublian_alphabet_size := 6
def word_length_one := nublian_alphabet_size
def word_length_two := nublian_alphabet_size * nublian_alphabet_size
def word_length_three := nublian_alphabet_size * nublian_alphabet_size * nublian_alphabet_size

-- Define the total number of words
def total_words := word_length_one + word_length_two + word_length_three

-- Main theorem statement
theorem nublian_total_words : total_words = 258 := by
  sorry

end NUMINAMATH_GPT_nublian_total_words_l763_76396


namespace NUMINAMATH_GPT_missing_number_is_6630_l763_76306

theorem missing_number_is_6630 (x : ℕ) (h : 815472 / x = 123) : x = 6630 :=
by {
  sorry
}

end NUMINAMATH_GPT_missing_number_is_6630_l763_76306


namespace NUMINAMATH_GPT_train_crossing_time_is_correct_l763_76346

-- Define the constant values
def train_length : ℝ := 350        -- Train length in meters
def train_speed : ℝ := 20          -- Train speed in m/s
def crossing_time : ℝ := 17.5      -- Time to cross the signal post in seconds

-- Proving the relationship that the time taken for the train to cross the signal post is as calculated
theorem train_crossing_time_is_correct : (train_length / train_speed) = crossing_time :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_is_correct_l763_76346


namespace NUMINAMATH_GPT_base_of_numbering_system_l763_76305

-- Definitions based on conditions
def num_children := 100
def num_boys := 24
def num_girls := 32

-- Problem statement: Prove the base of numbering system used is 6
theorem base_of_numbering_system (n: ℕ) (h: n ≠ 0):
    n^2 = (2 * n + 4) + (3 * n + 2) → n = 6 := 
  by
    sorry

end NUMINAMATH_GPT_base_of_numbering_system_l763_76305


namespace NUMINAMATH_GPT_range_of_a_l763_76313

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 ↔ x > Real.log a / Real.log 2) → 0 < a ∧ a ≤ 1 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l763_76313


namespace NUMINAMATH_GPT_highlighter_difference_l763_76398

theorem highlighter_difference :
  ∀ (yellow pink blue : ℕ),
    yellow = 7 →
    pink = yellow + 7 →
    yellow + pink + blue = 40 →
    blue - pink = 5 :=
by
  intros yellow pink blue h_yellow h_pink h_total
  rw [h_yellow, h_pink] at h_total
  sorry

end NUMINAMATH_GPT_highlighter_difference_l763_76398


namespace NUMINAMATH_GPT_inverse_proportion_function_point_l763_76389

theorem inverse_proportion_function_point (k x y : ℝ) (h₁ : 1 = k / (-6)) (h₂ : y = k / x) :
  k = -6 ∧ (x = 2 ∧ y = -3 ↔ y = -k / x) :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_function_point_l763_76389


namespace NUMINAMATH_GPT_sum_of_ages_l763_76327

theorem sum_of_ages (a b c : ℕ) (h1 : a * b * c = 72) (h2 : b = c) (h3 : a < b) : a + b + c = 14 :=
sorry

end NUMINAMATH_GPT_sum_of_ages_l763_76327


namespace NUMINAMATH_GPT_reflected_ray_equation_l763_76345

-- Definitions for the given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Problem statement: proving equation of the reflected ray
theorem reflected_ray_equation : 
  ∀ x y : ℝ, incident_line x = y ∧ reflection_line x = y → x - 2*y - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_reflected_ray_equation_l763_76345


namespace NUMINAMATH_GPT_sophie_germain_identity_l763_76341

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4 * b^4 = (a^2 + 2 * a * b + 2 * b^2) * (a^2 - 2 * a * b + 2 * b^2) :=
by sorry

end NUMINAMATH_GPT_sophie_germain_identity_l763_76341


namespace NUMINAMATH_GPT_not_divisible_l763_76399

theorem not_divisible (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 12) : ¬∃ k : ℕ, 120 * a + 2 * b = k * (100 * a + b) := 
sorry

end NUMINAMATH_GPT_not_divisible_l763_76399


namespace NUMINAMATH_GPT_complex_eq_solution_l763_76358

theorem complex_eq_solution (x y : ℝ) (i : ℂ) (h : (2 * x - 1) + i = y - (3 - y) * i) : 
  x = 5 / 2 ∧ y = 4 :=
  sorry

end NUMINAMATH_GPT_complex_eq_solution_l763_76358


namespace NUMINAMATH_GPT_find_a_plus_b_l763_76304

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + ab = 0 → (x = -1 ∨ x = 4)) → a + b = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l763_76304


namespace NUMINAMATH_GPT_remainder_when_divided_l763_76314

open Polynomial

noncomputable def poly : Polynomial ℚ := X^6 + X^5 + 2*X^3 - X^2 + 3
noncomputable def divisor : Polynomial ℚ := (X + 2) * (X - 1)
noncomputable def remainder : Polynomial ℚ := -X + 5

theorem remainder_when_divided :
  ∃ q : Polynomial ℚ, poly = divisor * q + remainder :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_l763_76314


namespace NUMINAMATH_GPT_work_problem_l763_76307

theorem work_problem 
  (A_real : ℝ)
  (B_days : ℝ := 16)
  (C_days : ℝ := 16)
  (ABC_days : ℝ := 4)
  (H_b : (1 / B_days) = 1 / 16)
  (H_c : (1 / C_days) = 1 / 16)
  (H_abc : (1 / A_real + 1 / B_days + 1 / C_days) = 1 / ABC_days) : 
  A_real = 8 := 
sorry

end NUMINAMATH_GPT_work_problem_l763_76307


namespace NUMINAMATH_GPT_quadratic_roots_squared_sum_l763_76375

theorem quadratic_roots_squared_sum (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 - 2 * n - 1 = 0) : m^2 + n^2 = 6 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_squared_sum_l763_76375


namespace NUMINAMATH_GPT_relationship_y_values_l763_76323

theorem relationship_y_values (x1 x2 y1 y2 : ℝ) (h1 : x1 > x2) (h2 : 0 < x2) (h3 : y1 = - (3 / x1)) (h4 : y2 = - (3 / x2)) : y1 > y2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_y_values_l763_76323


namespace NUMINAMATH_GPT_range_of_q_eq_eight_inf_l763_76371

noncomputable def q (x : ℝ) : ℝ := (x^2 + 2)^3

theorem range_of_q_eq_eight_inf (x : ℝ) : 0 ≤ x → ∃ y, y = q x ∧ 8 ≤ y := sorry

end NUMINAMATH_GPT_range_of_q_eq_eight_inf_l763_76371


namespace NUMINAMATH_GPT_max_possible_n_l763_76368

theorem max_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 6400) : n ≤ 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_possible_n_l763_76368


namespace NUMINAMATH_GPT_female_salmon_returned_l763_76312

/-- The number of female salmon that returned to their rivers is 259378,
    given that the total number of salmon that made the trip is 971639 and
    the number of male salmon that returned is 712261. -/
theorem female_salmon_returned :
  let n := 971639
  let m := 712261
  let f := n - m
  f = 259378 :=
by
  rfl

end NUMINAMATH_GPT_female_salmon_returned_l763_76312


namespace NUMINAMATH_GPT_part1_find_a_b_part2_inequality_l763_76397

theorem part1_find_a_b (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 * x + 1| + |x + a|) 
  (h_sol : ∀ x, f x ≤ 3 ↔ b ≤ x ∧ x ≤ 1) : 
  a = -1 ∧ b = -1 :=
sorry

theorem part2_inequality (m n : ℝ) (a : ℝ) (h_m : 0 < m) (h_n : 0 < n) 
  (h_eq : (1 / (2 * m)) + (2 / n) + 2 * a = 0) (h_a : a = -1) : 
  4 * m^2 + n^2 ≥ 4 :=
sorry

end NUMINAMATH_GPT_part1_find_a_b_part2_inequality_l763_76397


namespace NUMINAMATH_GPT_number_of_round_trips_each_bird_made_l763_76320

theorem number_of_round_trips_each_bird_made
  (distance_to_materials : ℕ)
  (total_distance_covered : ℕ)
  (distance_one_round_trip : ℕ)
  (total_number_of_trips : ℕ)
  (individual_bird_trips : ℕ) :
  distance_to_materials = 200 →
  total_distance_covered = 8000 →
  distance_one_round_trip = 2 * distance_to_materials →
  total_number_of_trips = total_distance_covered / distance_one_round_trip →
  individual_bird_trips = total_number_of_trips / 2 →
  individual_bird_trips = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_round_trips_each_bird_made_l763_76320


namespace NUMINAMATH_GPT_sum_of_four_powers_l763_76380

theorem sum_of_four_powers (a : ℕ) : 4 * a^3 = 500 :=
by
  rw [Nat.pow_succ, Nat.pow_succ]
  sorry

end NUMINAMATH_GPT_sum_of_four_powers_l763_76380


namespace NUMINAMATH_GPT_power_of_two_square_l763_76319

theorem power_of_two_square (n : ℕ) : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2 ↔ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_power_of_two_square_l763_76319


namespace NUMINAMATH_GPT_scaled_det_l763_76388

variable (x y z a b c p q r : ℝ)
variable (det_orig : ℝ)
variable (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2)

theorem scaled_det (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2) :
  Matrix.det ![![3*x, 3*y, 3*z], ![3*a, 3*b, 3*c], ![3*p, 3*q, 3*r]] = 54 :=
by
  sorry

end NUMINAMATH_GPT_scaled_det_l763_76388


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l763_76321

theorem right_triangle_hypotenuse :
  ∃ b a : ℕ, a^2 + 1994^2 = b^2 ∧ b = 994010 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l763_76321


namespace NUMINAMATH_GPT_equivalent_proof_problem_l763_76364

variable (a b d e c f g h : ℚ)

def condition1 : Prop := 8 = (6 / 100) * a
def condition2 : Prop := 6 = (8 / 100) * b
def condition3 : Prop := 9 = (5 / 100) * d
def condition4 : Prop := 7 = (3 / 100) * e
def condition5 : Prop := c = b / a
def condition6 : Prop := f = d / a
def condition7 : Prop := g = e / b

theorem equivalent_proof_problem (hac1 : condition1 a)
                                 (hac2 : condition2 b)
                                 (hac3 : condition3 d)
                                 (hac4 : condition4 e)
                                 (hac5 : condition5 a b c)
                                 (hac6 : condition6 a d f)
                                 (hac7 : condition7 b e g) :
    h = f + g ↔ h = (803 / 20) * c := 
by sorry

end NUMINAMATH_GPT_equivalent_proof_problem_l763_76364


namespace NUMINAMATH_GPT_solve_for_m_l763_76382

theorem solve_for_m : ∃ m : ℝ, ((∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → (m = 6)) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_m_l763_76382


namespace NUMINAMATH_GPT_molecular_weight_C4H10_l763_76333

theorem molecular_weight_C4H10
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (C4H10_C_atoms : ℕ)
  (C4H10_H_atoms : ℕ)
  (moles : ℝ) : 
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  C4H10_C_atoms = 4 →
  C4H10_H_atoms = 10 →
  moles = 6 →
  (C4H10_C_atoms * atomic_weight_C + C4H10_H_atoms * atomic_weight_H) * moles = 348.72 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_C4H10_l763_76333


namespace NUMINAMATH_GPT_tangerines_times_persimmons_l763_76392

-- Definitions from the problem conditions
def apples : ℕ := 24
def tangerines : ℕ := 6 * apples
def persimmons : ℕ := 8

-- Statement to be proved
theorem tangerines_times_persimmons :
  tangerines / persimmons = 18 := by
  sorry

end NUMINAMATH_GPT_tangerines_times_persimmons_l763_76392


namespace NUMINAMATH_GPT_distinct_four_digit_integers_l763_76369

open Nat

theorem distinct_four_digit_integers (count_digs_18 : ℕ) :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (∃ d1 d2 d3 d4 : ℕ,
      d1 * d2 * d3 * d4 = 18 ∧
      d1 > 0 ∧ d1 < 10 ∧
      d2 > 0 ∧ d2 < 10 ∧
      d3 > 0 ∧ d3 < 10 ∧
      d4 > 0 ∧ d4 < 10 ∧
      n = d1 * 1000 + d2 * 100 + d3 * 10 + d4)) →
  count_digs_18 = 24 :=
sorry

end NUMINAMATH_GPT_distinct_four_digit_integers_l763_76369


namespace NUMINAMATH_GPT_operation_result_l763_76329

def a : ℝ := 0.8
def b : ℝ := 0.5
def c : ℝ := 0.40

theorem operation_result :
  (a ^ 3 - b ^ 3 / a ^ 2 + c + b ^ 2) = 0.9666875 := by
  sorry

end NUMINAMATH_GPT_operation_result_l763_76329


namespace NUMINAMATH_GPT_problem_statement_l763_76332

variable {f : ℝ → ℝ}

-- Condition 1: f(x) has domain ℝ (implicitly given by the type signature ωf)
-- Condition 2: f is decreasing on the interval (6, +∞)
def is_decreasing_on_6_infty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 6 < x → x < y → f x > f y

-- Condition 3: y = f(x + 6) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) = f (-x - 6)

-- The statement to prove
theorem problem_statement (h_decrease : is_decreasing_on_6_infty f) (h_even_shift : is_even_shifted f) : f 5 > f 8 :=
sorry

end NUMINAMATH_GPT_problem_statement_l763_76332


namespace NUMINAMATH_GPT_value_of_c_l763_76355

theorem value_of_c (b c : ℝ) (h1 : (x : ℝ) → (x + 4) * (x + b) = x^2 + c * x + 12) : c = 7 :=
by
  have h2 : 4 * b = 12 := by sorry
  have h3 : b = 3 := by sorry
  have h4 : c = b + 4 := by sorry
  rw [h3] at h4
  rw [h4]
  exact by norm_num

end NUMINAMATH_GPT_value_of_c_l763_76355


namespace NUMINAMATH_GPT_trigonometric_identity_l763_76362

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l763_76362


namespace NUMINAMATH_GPT_three_city_population_l763_76374

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end NUMINAMATH_GPT_three_city_population_l763_76374


namespace NUMINAMATH_GPT_fixed_point_of_parabolas_l763_76347

theorem fixed_point_of_parabolas 
  (t : ℝ) 
  (fixed_x fixed_y : ℝ) 
  (hx : fixed_x = 2) 
  (hy : fixed_y = 12) 
  (H : ∀ t : ℝ, ∃ y : ℝ, y = 3 * fixed_x^2 + t * fixed_x - 2 * t) : 
  ∃ y : ℝ, y = fixed_y :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_of_parabolas_l763_76347


namespace NUMINAMATH_GPT_no_afg_fourth_place_l763_76386

theorem no_afg_fourth_place
  (A B C D E F G : ℕ)
  (h1 : A < B)
  (h2 : A < C)
  (h3 : B < D)
  (h4 : C < E)
  (h5 : A < F ∧ F < B)
  (h6 : B < G ∧ G < C) :
  ¬ (A = 4 ∨ F = 4 ∨ G = 4) :=
by
  sorry

end NUMINAMATH_GPT_no_afg_fourth_place_l763_76386


namespace NUMINAMATH_GPT_area_of_square_with_perimeter_32_l763_76361

theorem area_of_square_with_perimeter_32 :
  ∀ (s : ℝ), 4 * s = 32 → s * s = 64 :=
by
  intros s h
  sorry

end NUMINAMATH_GPT_area_of_square_with_perimeter_32_l763_76361


namespace NUMINAMATH_GPT_sin_difference_identity_l763_76356

theorem sin_difference_identity 
  (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 1 / 3) : 
  Real.sin (π / 4 - α) = (Real.sqrt 2 - 4) / 6 := 
  sorry

end NUMINAMATH_GPT_sin_difference_identity_l763_76356


namespace NUMINAMATH_GPT_domain_myFunction_l763_76342

noncomputable def myFunction (x : ℝ) : ℝ :=
  (x^3 - 125) / (x + 125)

theorem domain_myFunction :
  {x : ℝ | ∀ y, y = myFunction x → x ≠ -125} = { x : ℝ | x ≠ -125 } := 
by
  sorry

end NUMINAMATH_GPT_domain_myFunction_l763_76342


namespace NUMINAMATH_GPT_map_distance_representation_l763_76301

-- Define the conditions and the question as a Lean statement
theorem map_distance_representation :
  (∀ (length_cm : ℕ), (length_cm : ℕ) = 23 → (length_cm * 50 / 10 : ℕ) = 115) :=
by
  sorry

end NUMINAMATH_GPT_map_distance_representation_l763_76301


namespace NUMINAMATH_GPT_abs_eq_five_l763_76351

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_abs_eq_five_l763_76351


namespace NUMINAMATH_GPT_factorize_quadratic_l763_76324

theorem factorize_quadratic (x : ℝ) : x^2 - 2 * x = x * (x - 2) :=
sorry

end NUMINAMATH_GPT_factorize_quadratic_l763_76324
