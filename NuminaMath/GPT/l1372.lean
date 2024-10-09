import Mathlib

namespace find_unique_function_l1372_137278

theorem find_unique_function (f : ℝ → ℝ) (hf1 : ∀ x, 0 ≤ x → 0 ≤ f x)
    (hf2 : ∀ x, 0 ≤ x → f (f x) + f x = 12 * x) :
    ∀ x, 0 ≤ x → f x = 3 * x := 
  sorry

end find_unique_function_l1372_137278


namespace train_speed_l1372_137230

theorem train_speed (L V : ℝ) (h1 : L = V * 20) (h2 : L + 300.024 = V * 50) : V = 10.0008 :=
by
  sorry

end train_speed_l1372_137230


namespace sum_a2_to_a5_eq_zero_l1372_137220

theorem sum_a2_to_a5_eq_zero 
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : ∀ x : ℝ, x * (1 - 2 * x)^4 = a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_2 + a_3 + a_4 + a_5 = 0 :=
sorry

end sum_a2_to_a5_eq_zero_l1372_137220


namespace prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l1372_137252

-- Definitions of probabilities of making a shot
def p_A : ℚ := 1 / 3
def p_B : ℚ := 1 / 2

-- Number of attempts
def num_attempts : ℕ := 3

-- Probability that B makes at most 2 shots
theorem prob_B_at_most_2_shots : 
  (1 - (num_attempts.choose 3) * (p_B ^ 3) * ((1 - p_B) ^ (num_attempts - 3))) = 7 / 8 :=
by 
  sorry

-- Probability that B makes exactly 2 more shots than A
theorem prob_B_exactly_2_more_than_A : 
  (num_attempts.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ 1) * (num_attempts.choose 0) * ((1 - p_A) ^ num_attempts) +
  (num_attempts.choose 3) * (p_B ^ 3) * (num_attempts.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (num_attempts - 1)) = 1 / 6 :=
by 
  sorry

end prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l1372_137252


namespace calculate_product_l1372_137221

theorem calculate_product (a : ℝ) : 2 * a * (3 * a) = 6 * a^2 := by
  -- This will skip the proof, denoted by 'sorry'
  sorry

end calculate_product_l1372_137221


namespace complement_union_l1372_137271

open Set

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 5, 6, 8})
  (hA : A = {1, 5, 8})(hB : B = {2}) :
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  rw [hU, hA, hB]
  -- Intermediate steps would go here
  sorry

end complement_union_l1372_137271


namespace range_of_m_l1372_137269

variable {x m : ℝ}

-- Definition of the first condition: ∀ x in ℝ, |x| + |x - 1| > m
def condition1 (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m

-- Definition of the second condition: ∀ x in ℝ, (-(7 - 3 * m))^x is decreasing
def condition2 (m : ℝ) := ∀ x : ℝ, (-(7 - 3 * m))^x > (-(7 - 3 * m))^(x + 1)

-- Main theorem to prove m < 1
theorem range_of_m (h1 : condition1 m) (h2 : condition2 m) : m < 1 :=
sorry

end range_of_m_l1372_137269


namespace value_of_expression_in_third_quadrant_l1372_137245

theorem value_of_expression_in_third_quadrant (α : ℝ) (h1 : 180 < α ∧ α < 270) :
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) = -2 := by
  sorry

end value_of_expression_in_third_quadrant_l1372_137245


namespace sum_of_three_consecutive_integers_product_990_l1372_137227

theorem sum_of_three_consecutive_integers_product_990 
  (a b c : ℕ) 
  (h1 : b = a + 1)
  (h2 : c = b + 1)
  (h3 : a * b * c = 990) :
  a + b + c = 30 :=
sorry

end sum_of_three_consecutive_integers_product_990_l1372_137227


namespace minimum_value_of_quadratic_function_l1372_137201

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_quadratic_function 
  (f : ℝ → ℝ)
  (n : ℕ)
  (h1 : f n = 6)
  (h2 : f (n + 1) = 5)
  (h3 : f (n + 2) = 5)
  (hf : ∃ a b c : ℝ, f = quadratic_function a b c) :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 5 :=
by
  sorry

end minimum_value_of_quadratic_function_l1372_137201


namespace right_triangle_proportion_l1372_137292

/-- Given a right triangle ABC with ∠C = 90°, AB = c, AC = b, and BC = a, 
    and a point P on the hypotenuse AB (or its extension) such that 
    AP = m, BP = n, and CP = k, prove that a²m² + b²n² = c²k². -/
theorem right_triangle_proportion
  {a b c m n k : ℝ}
  (h_right : ∀ A B C : ℝ, A^2 + B^2 = C^2)
  (h1 : ∀ P : ℝ, m^2 + n^2 = k^2)
  (h_geometry : a^2 + b^2 = c^2) :
  a^2 * m^2 + b^2 * n^2 = c^2 * k^2 := 
sorry

end right_triangle_proportion_l1372_137292


namespace remainder_of_sum_mod_l1372_137268

theorem remainder_of_sum_mod (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2 * n) % 7 :=
by
  sorry

end remainder_of_sum_mod_l1372_137268


namespace hexagon_inequality_l1372_137262

noncomputable def ABCDEF := 3 * Real.sqrt 3 / 2
noncomputable def ACE := Real.sqrt 3
noncomputable def BDF := Real.sqrt 3
noncomputable def R₁ := Real.sqrt 3 / 4
noncomputable def R₂ := -Real.sqrt 3 / 4

theorem hexagon_inequality :
  min ACE BDF + R₂ - R₁ ≤ 3 * Real.sqrt 3 / 4 :=
by
  sorry

end hexagon_inequality_l1372_137262


namespace initial_books_l1372_137277

theorem initial_books (total_books_now : ℕ) (books_added : ℕ) (initial_books : ℕ) :
  total_books_now = 48 → books_added = 10 → initial_books = total_books_now - books_added → initial_books = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_books_l1372_137277


namespace integer_solutions_determinant_l1372_137242

theorem integer_solutions_determinant (a b c d : ℤ)
    (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
sorry

end integer_solutions_determinant_l1372_137242


namespace max_volume_tetrahedron_l1372_137285

-- Definitions and conditions
def SA : ℝ := 4
def AB : ℝ := 5
def SB_min : ℝ := 7
def SC_min : ℝ := 9
def BC_max : ℝ := 6
def AC_max : ℝ := 8

-- Proof statement
theorem max_volume_tetrahedron {SB SC BC AC : ℝ} (hSB : SB ≥ SB_min) (hSC : SC ≥ SC_min) (hBC : BC ≤ BC_max) (hAC : AC ≤ AC_max) :
  ∃ V : ℝ, V = 8 * Real.sqrt 6 ∧ V ≤ (1/3) * (1/2) * SA * AB * (2 * Real.sqrt 6) * BC := by
  sorry

end max_volume_tetrahedron_l1372_137285


namespace complement_is_correct_l1372_137260

variable (U : Set ℕ) (A : Set ℕ)

def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ :=
  { x ∈ U | x ∉ A }

theorem complement_is_correct :
  (U = {1, 2, 3, 4, 5, 6, 7}) →
  (A = {2, 4, 5}) →
  complement U A = {1, 3, 6, 7} :=
by
  sorry

end complement_is_correct_l1372_137260


namespace flight_time_l1372_137290

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248

theorem flight_time : (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) > 0 → 
                      total_distance / (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) = 2 :=
by
  -- Proof is skipped
  sorry

end flight_time_l1372_137290


namespace symmetry_sum_zero_l1372_137297

theorem symmetry_sum_zero (v : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, v (-x) = -v x) : 
  v (-2.00) + v (-1.00) + v (1.00) + v (2.00) = 0 := 
by 
  sorry

end symmetry_sum_zero_l1372_137297


namespace mixture_ratio_l1372_137282

theorem mixture_ratio (V : ℝ) (a b c : ℕ)
  (h_pos : V > 0)
  (h_ratio : V = (3/8) * V + (5/11) * V + ((88 - 33 - 40)/88) * V) :
  a = 33 ∧ b = 40 ∧ c = 15 :=
by
  sorry

end mixture_ratio_l1372_137282


namespace probability_10_or_9_probability_at_least_7_l1372_137255

-- Define the probabilities of hitting each ring
def p_10 : ℝ := 0.1
def p_9 : ℝ := 0.2
def p_8 : ℝ := 0.3
def p_7 : ℝ := 0.3
def p_below_7 : ℝ := 0.1

-- Define the events as their corresponding probabilities
def P_A : ℝ := p_10 -- Event of hitting the 10 ring
def P_B : ℝ := p_9 -- Event of hitting the 9 ring
def P_C : ℝ := p_8 -- Event of hitting the 8 ring
def P_D : ℝ := p_7 -- Event of hitting the 7 ring
def P_E : ℝ := p_below_7 -- Event of hitting below the 7 ring

-- Since the probabilities must sum to 1, we have the following fact about their sum
-- P_A + P_B + P_C + P_D + P_E = 1

theorem probability_10_or_9 : P_A + P_B = 0.3 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

theorem probability_at_least_7 : P_A + P_B + P_C + P_D = 0.9 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

end probability_10_or_9_probability_at_least_7_l1372_137255


namespace find_b_value_l1372_137233

theorem find_b_value (b : ℚ) (x : ℚ) (h1 : 3 * x + 9 = 0) (h2 : b * x + 15 = 5) : b = 10 / 3 :=
by
  sorry

end find_b_value_l1372_137233


namespace c_minus_a_is_10_l1372_137225

variable (a b c d k : ℝ)

theorem c_minus_a_is_10 (h1 : a + b = 90)
                        (h2 : b + c = 100)
                        (h3 : a + c + d = 180)
                        (h4 : a^2 + b^2 + c^2 + d^2 = k) :
  c - a = 10 :=
by sorry

end c_minus_a_is_10_l1372_137225


namespace nine_digit_numbers_divisible_by_eleven_l1372_137216

theorem nine_digit_numbers_divisible_by_eleven :
  ∃ (n : ℕ), n = 31680 ∧
    ∃ (num : ℕ), num < 10^9 ∧ num ≥ 10^8 ∧
      (∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ i : ℕ, i ≤ 8 ∧ (num / 10^i) % 10 = d) ∧
      (num % 11 = 0) := 
sorry

end nine_digit_numbers_divisible_by_eleven_l1372_137216


namespace eggs_in_each_basket_is_four_l1372_137276

theorem eggs_in_each_basket_is_four 
  (n : ℕ)
  (h1 : n ∣ 16) 
  (h2 : n ∣ 28) 
  (h3 : n ≥ 2) : 
  n = 4 :=
sorry

end eggs_in_each_basket_is_four_l1372_137276


namespace incorrect_operation_B_l1372_137256

theorem incorrect_operation_B : (4 + 5)^2 ≠ 4^2 + 5^2 := 
  sorry

end incorrect_operation_B_l1372_137256


namespace rectangle_bounds_product_l1372_137240

theorem rectangle_bounds_product (b : ℝ) :
  (∃ b, y = 3 ∧ y = 7 ∧ x = -1 ∧ (x = b) 
   → (b = 3 ∨ b = -5) 
    ∧ (3 * -5 = -15)) :=
sorry

end rectangle_bounds_product_l1372_137240


namespace alyssa_plums_correct_l1372_137207

def total_plums : ℕ := 27
def jason_plums : ℕ := 10
def alyssa_plums : ℕ := 17

theorem alyssa_plums_correct : alyssa_plums = total_plums - jason_plums := by
  sorry

end alyssa_plums_correct_l1372_137207


namespace point_translation_l1372_137264

theorem point_translation :
  ∃ (x y : ℤ), x = -1 ∧ y = -2 ↔ 
  ∃ (x₀ y₀ : ℤ), 
    x₀ = -3 ∧ y₀ = 2 ∧ 
    x = x₀ + 2 ∧ 
    y = y₀ - 4 := by
  sorry

end point_translation_l1372_137264


namespace initial_books_donations_l1372_137270

variable {X : ℕ} -- Initial number of book donations

def books_donated_during_week := 10 * 5
def books_borrowed := 140
def books_remaining := 210

theorem initial_books_donations :
  X + books_donated_during_week - books_borrowed = books_remaining → X = 300 :=
by
  intro h
  sorry

end initial_books_donations_l1372_137270


namespace simplify_expression_l1372_137284

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
by
  sorry

end simplify_expression_l1372_137284


namespace triangle_perimeter_l1372_137246

theorem triangle_perimeter (a b : ℝ) (f : ℝ → Prop) 
  (h₁ : a = 7) (h₂ : b = 11)
  (eqn : ∀ x, f x ↔ x^2 - 25 = 2 * (x - 5)^2)
  (h₃ : ∃ x, f x ∧ 4 < x ∧ x < 18) :
  ∃ p : ℝ, (p = a + b + 5 ∨ p = a + b + 15) :=
by
  sorry

end triangle_perimeter_l1372_137246


namespace find_x_satisfying_sinx_plus_cosx_eq_one_l1372_137239

theorem find_x_satisfying_sinx_plus_cosx_eq_one :
  ∀ x, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0) := by
  sorry

end find_x_satisfying_sinx_plus_cosx_eq_one_l1372_137239


namespace find_plaintext_from_ciphertext_l1372_137248

theorem find_plaintext_from_ciphertext : 
  ∃ x : ℕ, ∀ a : ℝ, (a^3 - 2 = 6) → (1022 = a^x - 2) → x = 10 :=
by
  use 10
  intros a ha hc
  -- Proof omitted
  sorry

end find_plaintext_from_ciphertext_l1372_137248


namespace discount_percentage_l1372_137291

variable (P : ℝ) -- Original price of the dress
variable (D : ℝ) -- Discount percentage

theorem discount_percentage
  (h1 : P * (1 - D / 100) = 68)
  (h2 : 68 * 1.25 = 85)
  (h3 : 85 - P = 5) :
  D = 15 :=
by
  sorry

end discount_percentage_l1372_137291


namespace perpendicular_condition_line_through_point_l1372_137295

-- Definitions for lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y = 6
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x + y = 3

-- Part 1: Prove that l1 is perpendicular to l2 if and only if m = -3 or m = 0
theorem perpendicular_condition (m : ℝ) : 
  (∀ (x : ℝ), ∀ (y : ℝ), (l1 m x y ∧ l2 m x y) → (m = 0 ∨ m = -3)) :=
sorry

-- Part 2: Prove the equations of line l given the conditions
theorem line_through_point (m : ℝ) (l : ℝ → ℝ → Prop) : 
  (∀ (P : ℝ × ℝ), (P = (1, 2*m)) → (l2 m P.1 P.2) → 
  ((∀ (x y : ℝ), l x y → 2 * x - y = 0) ∨ (∀ (x y: ℝ), l x y → x + 2 * y - 5 = 0))) :=
sorry

end perpendicular_condition_line_through_point_l1372_137295


namespace buckets_required_l1372_137280

theorem buckets_required (C : ℝ) (N : ℝ):
  (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  sorry

end buckets_required_l1372_137280


namespace angle_A_is_120_degrees_l1372_137232

theorem angle_A_is_120_degrees
  (b c l_a : ℝ)
  (h : (1 / b) + (1 / c) = 1 / l_a) :
  ∃ A : ℝ, A = 120 :=
by
  sorry

end angle_A_is_120_degrees_l1372_137232


namespace sum_remainders_l1372_137294

theorem sum_remainders (a b c : ℕ) (h₁ : a % 30 = 7) (h₂ : b % 30 = 11) (h₃ : c % 30 = 23) : 
  (a + b + c) % 30 = 11 := 
by
  sorry

end sum_remainders_l1372_137294


namespace intercept_sum_l1372_137289

theorem intercept_sum (x y : ℝ) :
  (y - 3 = 6 * (x - 5)) →
  (∃ x_intercept, (y = 0) ∧ (x_intercept = 4.5)) →
  (∃ y_intercept, (x = 0) ∧ (y_intercept = -27)) →
  (4.5 + (-27) = -22.5) :=
by
  intros h_eq h_xint h_yint
  sorry

end intercept_sum_l1372_137289


namespace find_root_and_coefficient_l1372_137228

theorem find_root_and_coefficient (m: ℝ) (x: ℝ) (h₁: x ^ 2 - m * x - 6 = 0) (h₂: x = 3) :
  (x = 3 ∧ -2 = -6 / 3 ∨ m = 1) :=
by
  sorry

end find_root_and_coefficient_l1372_137228


namespace base_d_digit_difference_l1372_137219

theorem base_d_digit_difference (A C d : ℕ) (h1 : d > 8)
  (h2 : d * A + C + (d * C + C) = 2 * d^2 + 3 * d + 2) :
  (A - C = d + 1) :=
sorry

end base_d_digit_difference_l1372_137219


namespace charge_for_each_additional_fifth_mile_l1372_137226

theorem charge_for_each_additional_fifth_mile
  (initial_charge : ℝ)
  (total_charge : ℝ)
  (distance_in_miles : ℕ)
  (distance_per_increment : ℝ)
  (x : ℝ) :
  initial_charge = 2.10 →
  total_charge = 17.70 →
  distance_in_miles = 8 →
  distance_per_increment = 1/5 →
  (total_charge - initial_charge) / ((distance_in_miles / distance_per_increment) - 1) = x →
  x = 0.40 :=
by
  intros h_initial_charge h_total_charge h_distance_in_miles h_distance_per_increment h_eq
  sorry

end charge_for_each_additional_fifth_mile_l1372_137226


namespace max_travel_within_budget_l1372_137279

noncomputable def rental_cost_per_day : ℝ := 30
noncomputable def insurance_fee_per_day : ℝ := 10
noncomputable def mileage_cost_per_mile : ℝ := 0.18
noncomputable def budget : ℝ := 75
noncomputable def minimum_required_travel : ℝ := 100

theorem max_travel_within_budget : ∀ (rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel), 
  rental_cost_per_day = 30 → 
  insurance_fee_per_day = 10 → 
  mileage_cost_per_mile = 0.18 → 
  budget = 75 →
  minimum_required_travel = 100 →
  (minimum_required_travel + (budget - rental_cost_per_day - insurance_fee_per_day - mileage_cost_per_mile * minimum_required_travel) / mileage_cost_per_mile) = 194 := 
by
  intros rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end max_travel_within_budget_l1372_137279


namespace complex_product_eq_50i_l1372_137231

open Complex

theorem complex_product_eq_50i : 
  let Q := (4 : ℂ) + 3 * I
  let E := (2 * I : ℂ)
  let D := (4 : ℂ) - 3 * I
  Q * E * D = 50 * I :=
by
  -- Complex numbers and multiplication are handled here
  sorry

end complex_product_eq_50i_l1372_137231


namespace x_coordinate_of_point_l1372_137250

theorem x_coordinate_of_point (x_1 n : ℝ) 
  (h1 : x_1 = (n / 5) - (2 / 5)) 
  (h2 : x_1 + 3 = ((n + 15) / 5) - (2 / 5)) : 
  x_1 = (n / 5) - (2 / 5) :=
by sorry

end x_coordinate_of_point_l1372_137250


namespace determine_n_l1372_137237

theorem determine_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26) : n = 25 :=
by
  sorry

end determine_n_l1372_137237


namespace alex_jellybeans_l1372_137218

theorem alex_jellybeans (x : ℕ) : x = 254 → x ≥ 150 ∧ x % 15 = 14 ∧ x % 17 = 16 :=
by
  sorry

end alex_jellybeans_l1372_137218


namespace profit_percentage_l1372_137274

theorem profit_percentage (SP CP : ℝ) (H_SP : SP = 1800) (H_CP : CP = 1500) :
  ((SP - CP) / CP) * 100 = 20 :=
by
  sorry

end profit_percentage_l1372_137274


namespace per_can_price_difference_cents_l1372_137214

   theorem per_can_price_difference_cents :
     let bulk_warehouse_price_per_case := 12.0
     let bulk_warehouse_cans_per_case := 48
     let bulk_warehouse_discount := 0.10
     let local_store_price_per_case := 6.0
     let local_store_cans_per_case := 12
     let local_store_promotion_factor := 1.5 -- represents the effect of the promotion (3 cases for the price of 2.5 cases)
     let bulk_warehouse_price_per_can := (bulk_warehouse_price_per_case * (1 - bulk_warehouse_discount)) / bulk_warehouse_cans_per_case
     let local_store_price_per_can := (local_store_price_per_case * local_store_promotion_factor) / (local_store_cans_per_case * 3)
     let price_difference_cents := (local_store_price_per_can - bulk_warehouse_price_per_can) * 100
     price_difference_cents = 19.17 :=
   by
     sorry
   
end per_can_price_difference_cents_l1372_137214


namespace system_of_equations_proof_l1372_137265

theorem system_of_equations_proof (a b x A B C : ℝ) (h1: a ≠ 0) 
  (h2: a * Real.sin x + b * Real.cos x = 0) 
  (h3: A * Real.sin (2 * x) + B * Real.cos (2 * x) = C) : 
  2 * a * b * A + (b ^ 2 - a ^ 2) * B + (a ^ 2 + b ^ 2) * C = 0 := 
sorry

end system_of_equations_proof_l1372_137265


namespace little_sister_stole_roses_l1372_137296

/-- Ricky has 40 roses. His little sister steals some roses. He wants to give away the rest of the roses in equal portions to 9 different people, and each person gets 4 roses. Prove how many roses his little sister stole. -/
theorem little_sister_stole_roses (total_roses stolen_roses remaining_roses people roses_per_person : ℕ)
  (h1 : total_roses = 40)
  (h2 : people = 9)
  (h3 : roses_per_person = 4)
  (h4 : remaining_roses = people * roses_per_person)
  (h5 : remaining_roses = total_roses - stolen_roses) :
  stolen_roses = 4 :=
by
  sorry

end little_sister_stole_roses_l1372_137296


namespace largest_sum_digits_24_hour_watch_l1372_137235

theorem largest_sum_digits_24_hour_watch : 
  (∃ h m : ℕ, 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ 
              (h / 10 + h % 10 + m / 10 + m % 10 = 24)) :=
by
  sorry

end largest_sum_digits_24_hour_watch_l1372_137235


namespace fraction_of_girls_l1372_137229

theorem fraction_of_girls (G T B : ℕ) (Fraction : ℚ)
  (h1 : Fraction * G = (1/3 : ℚ) * T)
  (h2 : (B : ℚ) / G = 1/2) :
  Fraction = 1/2 := by
  sorry

end fraction_of_girls_l1372_137229


namespace find_f_of_3_l1372_137283

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l1372_137283


namespace total_present_ages_l1372_137243

variable (P Q P' Q' : ℕ)

-- Condition 1: 6 years ago, \( p \) was half of \( q \) in age.
axiom cond1 : P = Q / 2

-- Condition 2: The ratio of their present ages is 3:4.
axiom cond2 : (P + 6) * 4 = (Q + 6) * 3

-- We need to prove: the total of their present ages is 21
theorem total_present_ages : P' + Q' = 21 :=
by
  -- We already have the variables and axioms in the context, so we just need to state the goal
  sorry

end total_present_ages_l1372_137243


namespace product_of_p_r_s_l1372_137210

theorem product_of_p_r_s
  (p r s : ℕ)
  (h1 : 3^p + 3^4 = 90)
  (h2 : 2^r + 44 = 76)
  (h3 : 5^3 + 6^s = 1421) :
  p * r * s = 40 := 
sorry

end product_of_p_r_s_l1372_137210


namespace sum_of_reciprocals_l1372_137200

variable {x y : ℝ}
variable (hx : x + y = 3 * x * y + 2)

theorem sum_of_reciprocals : (1 / x) + (1 / y) = 3 :=
by
  sorry

end sum_of_reciprocals_l1372_137200


namespace precisely_hundred_million_l1372_137259

-- Defining the options as an enumeration type
inductive Precision
| HundredBillion
| Billion
| HundredMillion
| Percent

-- The given figure in billions
def givenFigure : Float := 21.658

-- The correct precision is HundredMillion
def correctPrecision : Precision := Precision.HundredMillion

-- The theorem to prove the correctness of the figure's precision
theorem precisely_hundred_million : correctPrecision = Precision.HundredMillion :=
by
  sorry

end precisely_hundred_million_l1372_137259


namespace rectangle_side_length_l1372_137224

theorem rectangle_side_length (a b c d : ℕ) 
  (h₁ : a = 3) 
  (h₂ : b = 6) 
  (h₃ : a / c = 3 / 4) : 
  c = 4 := 
by
  sorry

end rectangle_side_length_l1372_137224


namespace cubic_polynomial_root_l1372_137263

theorem cubic_polynomial_root (a b c : ℕ) (h : 27 * x^3 - 9 * x^2 - 9 * x - 3 = 0) : 
  (a + b + c = 11) :=
sorry

end cubic_polynomial_root_l1372_137263


namespace solve_for_x_l1372_137298

theorem solve_for_x : ∀ x, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ↔ x = -7 / 2 := by
  sorry

end solve_for_x_l1372_137298


namespace area_change_l1372_137209

theorem area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := 1.2 * L
  let B' := 0.8 * B
  let A := L * B
  let A' := L' * B'
  A' = 0.96 * A :=
by
  sorry

end area_change_l1372_137209


namespace condition_suff_and_nec_l1372_137275

def p (x : ℝ) : Prop := |x + 2| ≤ 3
def q (x : ℝ) : Prop := x < -8

theorem condition_suff_and_nec (x : ℝ) : p x ↔ ¬ q x :=
by
  sorry

end condition_suff_and_nec_l1372_137275


namespace xiao_ying_correct_answers_at_least_l1372_137272

def total_questions : ℕ := 20
def points_correct : ℕ := 5
def points_incorrect : ℕ := 2
def excellent_points : ℕ := 80

theorem xiao_ying_correct_answers_at_least (x : ℕ) :
  (5 * x - 2 * (total_questions - x)) ≥ excellent_points → x ≥ 18 := by
  sorry

end xiao_ying_correct_answers_at_least_l1372_137272


namespace rectangle_area_l1372_137222

theorem rectangle_area (w d : ℝ) 
  (h1 : d = (w^2 + (3 * w)^2) ^ (1/2))
  (h2 : ∃ A : ℝ, A = w * 3 * w) :
  ∃ A : ℝ, A = 3 * (d^2 / 10) := 
by {
  sorry
}

end rectangle_area_l1372_137222


namespace geom_seq_sum_l1372_137287

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geometric : ∀ n, a (n + 1) = a n * r)
variable (h_pos : ∀ n, a n > 0)
variable (h_equation : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

theorem geom_seq_sum : a 3 + a 5 = 5 :=
by sorry

end geom_seq_sum_l1372_137287


namespace problems_per_page_is_five_l1372_137257

-- Let M and R be the number of problems on each math and reading page respectively
variables (M R : ℕ)

-- Conditions given in problem
def two_math_pages := 2 * M
def four_reading_pages := 4 * R
def total_problems := two_math_pages + four_reading_pages

-- Assume the number of problems per page is the same for both math and reading as P
variable (P : ℕ)
def problems_per_page_equal := (2 * P) + (4 * P) = 30

theorem problems_per_page_is_five :
  (2 * P) + (4 * P) = 30 → P = 5 :=
by
  intro h
  sorry

end problems_per_page_is_five_l1372_137257


namespace werewolf_is_A_l1372_137213

def is_liar (x : ℕ) : Prop := sorry
def is_knight (x : ℕ) : Prop := sorry
def is_werewolf (x : ℕ) : Prop := sorry

axiom A : ℕ
axiom B : ℕ
axiom C : ℕ

-- Conditions from the problem
axiom A_statement : is_liar A ∨ is_liar B
axiom B_statement : is_werewolf C
axiom exactly_one_werewolf : 
  (is_werewolf A ∧ ¬ is_werewolf B ∧ ¬ is_werewolf C) ∨
  (is_werewolf B ∧ ¬ is_werewolf A ∧ ¬ is_werewolf C) ∨
  (is_werewolf C ∧ ¬ is_werewolf A ∧ ¬ is_werewolf B)
axiom werewolf_is_knight : ∀ x : ℕ, is_werewolf x → is_knight x

-- Prove the conclusion
theorem werewolf_is_A : 
  is_werewolf A ∧ is_knight A :=
sorry

end werewolf_is_A_l1372_137213


namespace smallest_number_with_unique_digits_summing_to_32_exists_l1372_137293

theorem smallest_number_with_unique_digits_summing_to_32_exists : 
  ∃ n : ℕ, n / 10000 < 10 ∧ (n % 10 ≠ (n / 10) % 10) ∧ 
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ 
  ((n / 100) % 10 ≠ (n / 1000) % 10) ∧ 
  ((n / 1000) % 10 ≠ (n / 10000) % 10) ∧ 
  (n % 10 + (n / 10) % 10 + (n / 100) % 10 + (n / 1000) % 10 + (n / 10000) % 10 = 32) := 
sorry

end smallest_number_with_unique_digits_summing_to_32_exists_l1372_137293


namespace total_recovery_time_l1372_137223

theorem total_recovery_time 
  (lions: ℕ := 3) (rhinos: ℕ := 2) (time_per_animal: ℕ := 2) :
  (lions + rhinos) * time_per_animal = 10 := by
  sorry

end total_recovery_time_l1372_137223


namespace find_fraction_l1372_137267

theorem find_fraction 
  (f : ℚ) (t k : ℚ)
  (h1 : t = f * (k - 32)) 
  (h2 : t = 75)
  (h3 : k = 167) : 
  f = 5 / 9 :=
by
  sorry

end find_fraction_l1372_137267


namespace find_initial_money_l1372_137208

-- Definitions of the conditions
def basketball_card_cost : ℕ := 3
def baseball_card_cost : ℕ := 4
def basketball_packs : ℕ := 2
def baseball_decks : ℕ := 5
def change_received : ℕ := 24

-- Total cost calculation
def total_cost : ℕ := (basketball_card_cost * basketball_packs) + (baseball_card_cost * baseball_decks)

-- Initial money calculation
def initial_money : ℕ := total_cost + change_received

-- Proof statement
theorem find_initial_money : initial_money = 50 := 
by
  -- Proof steps would go here
  sorry

end find_initial_money_l1372_137208


namespace surface_area_geometric_mean_volume_geometric_mean_l1372_137261

noncomputable def surfaces_areas_proof (r : ℝ) (π : ℝ) : Prop :=
  let F_1 := 6 * π * r^2
  let F_2 := 4 * π * r^2
  let F_3 := 9 * π * r^2
  F_1^2 = F_2 * F_3

noncomputable def volumes_proof (r : ℝ) (π : ℝ) : Prop :=
  let V_1 := 2 * π * r^3
  let V_2 := (4 / 3) * π * r^3
  let V_3 := π * r^3
  V_1^2 = V_2 * V_3

theorem surface_area_geometric_mean (r : ℝ) (π : ℝ) : surfaces_areas_proof r π := 
  sorry

theorem volume_geometric_mean (r : ℝ) (π : ℝ) : volumes_proof r π :=
  sorry

end surface_area_geometric_mean_volume_geometric_mean_l1372_137261


namespace frank_columns_l1372_137212

theorem frank_columns (people : ℕ) (brownies_per_person : ℕ) (rows : ℕ)
  (h1 : people = 6) (h2 : brownies_per_person = 3) (h3 : rows = 3) : 
  (people * brownies_per_person) / rows = 6 :=
by
  -- Proof goes here
  sorry

end frank_columns_l1372_137212


namespace appropriate_sampling_methods_l1372_137299

-- Conditions for the first survey
structure Population1 where
  high_income_families : Nat
  middle_income_families : Nat
  low_income_families : Nat
  total : Nat := high_income_families + middle_income_families + low_income_families

def survey1_population : Population1 :=
  { high_income_families := 125,
    middle_income_families := 200,
    low_income_families := 95
  }

-- Condition for the second survey
structure Population2 where
  art_specialized_students : Nat

def survey2_population : Population2 :=
  { art_specialized_students := 5 }

-- The main statement to prove
theorem appropriate_sampling_methods :
  (survey1_population.total >= 100 → stratified_sampling_for_survey1) ∧ 
  (survey2_population.art_specialized_students >= 3 → simple_random_sampling_for_survey2) :=
  sorry

end appropriate_sampling_methods_l1372_137299


namespace find_solutions_l1372_137236

theorem find_solutions (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48 →
  (x = 1.2 ∨ x = -81.2) :=
by sorry

end find_solutions_l1372_137236


namespace race_distance_l1372_137286

theorem race_distance (dA dB dC : ℝ) (h1 : dA = 1000) (h2 : dB = 900) (h3 : dB = 800) (h4 : dC = 700) (d : ℝ) (h5 : d = dA + 127.5) :
  d = 600 :=
sorry

end race_distance_l1372_137286


namespace number_of_articles_l1372_137238

theorem number_of_articles (C S : ℝ) (N : ℝ) 
    (h1 : N * C = 40 * S) 
    (h2 : (S - C) / C * 100 = 49.999999999999986) : 
    N = 60 :=
sorry

end number_of_articles_l1372_137238


namespace velociraptor_catch_time_l1372_137215

/-- You encounter a velociraptor while out for a stroll. You run to the northeast at 10 m/s 
    with a 3-second head start. The velociraptor runs at 15√2 m/s but only runs either north or east at any given time. 
    Prove that the time until the velociraptor catches you is 6 seconds. -/
theorem velociraptor_catch_time (v_yours : ℝ) (t_head_start : ℝ) (v_velociraptor : ℝ)
  (v_eff : ℝ) (speed_advantage : ℝ) (headstart_distance : ℝ) :
  v_yours = 10 → t_head_start = 3 → v_velociraptor = 15 * Real.sqrt 2 →
  v_eff = 15 → speed_advantage = v_eff - v_yours → headstart_distance = v_yours * t_head_start →
  (headstart_distance / speed_advantage) = 6 :=
by
  sorry

end velociraptor_catch_time_l1372_137215


namespace part1_part2_l1372_137205

open Complex

noncomputable def z1 : ℂ := 1 - 2 * I
noncomputable def z2 : ℂ := 4 + 3 * I

theorem part1 : z1 * z2 = 10 - 5 * I := by
  sorry

noncomputable def z : ℂ := -Real.sqrt 2 - Real.sqrt 2 * I

theorem part2 (h_abs_z : abs z = 2)
              (h_img_eq_real : z.im = (3 * z1 - z2).re)
              (h_quadrant : z.re < 0 ∧ z.im < 0) : z = -Real.sqrt 2 - Real.sqrt 2 * I := by
  sorry

end part1_part2_l1372_137205


namespace max_and_min_sum_of_vars_l1372_137251

theorem max_and_min_sum_of_vars (x y z w : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w)
  (eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17 / 2) :
  ∃ max min : ℝ, max = 3 ∧ min = -2 + 5 / 2 * Real.sqrt 2 ∧
  (∀ (S : ℝ), S = x + y + z + w → S ≤ max ∧ S ≥ min) :=
by sorry

end max_and_min_sum_of_vars_l1372_137251


namespace find_number_l1372_137247

theorem find_number (x : ℝ) (h : 0.40 * x - 11 = 23) : x = 85 :=
sorry

end find_number_l1372_137247


namespace brendan_total_wins_l1372_137234

-- Define the number of matches won in each round
def matches_won_first_round : ℕ := 6
def matches_won_second_round : ℕ := 4
def matches_won_third_round : ℕ := 3
def matches_won_final_round : ℕ := 5

-- Define the total number of matches won
def total_matches_won : ℕ := 
  matches_won_first_round + matches_won_second_round + matches_won_third_round + matches_won_final_round

-- State the theorem that needs to be proven
theorem brendan_total_wins : total_matches_won = 18 := by
  sorry

end brendan_total_wins_l1372_137234


namespace vector_combination_l1372_137266

-- Definitions for vectors a, b, and c with the conditions provided
def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

-- The statement we want to prove
theorem vector_combination (t m n : ℝ)
  (h : c t = m • a + n • b) :
  t = 11 ∧ m + n = 11 / 2 :=
by
  sorry

end vector_combination_l1372_137266


namespace water_remainder_l1372_137203

theorem water_remainder (n : ℕ) (f : ℕ → ℚ) (h_init : f 1 = 1) 
  (h_recursive : ∀ k, k ≥ 2 → f k = f (k - 1) * (k^2 - 1) / k^2) :
  f 7 = 1 / 50 := 
sorry

end water_remainder_l1372_137203


namespace union_sets_l1372_137281

open Set

variable {α : Type*}

def setA : Set ℝ := { x | -2 < x ∧ x < 0 }
def setB : Set ℝ := { x | -1 < x ∧ x < 1 }
def setC : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = setC := 
by {
  sorry
}

end union_sets_l1372_137281


namespace expression_undefined_at_x_l1372_137258

theorem expression_undefined_at_x (x : ℝ) : (x^2 - 18 * x + 81 = 0) → x = 9 :=
by {
  sorry
}

end expression_undefined_at_x_l1372_137258


namespace max_profit_at_nine_l1372_137249

noncomputable def profit_function (x : ℝ) : ℝ :=
  -(1/3) * x ^ 3 + 81 * x - 234

theorem max_profit_at_nine :
  ∃ x, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function 9 :=
by
  sorry

end max_profit_at_nine_l1372_137249


namespace solve_for_x_l1372_137211

theorem solve_for_x (x : ℚ) 
  (h : (1/3 : ℚ) + 1/x = (7/9 : ℚ) + 1) : 
  x = 9/13 :=
by
  sorry

end solve_for_x_l1372_137211


namespace number_of_female_athletes_l1372_137202

theorem number_of_female_athletes (male_athletes female_athletes male_selected female_selected : ℕ)
  (h1 : male_athletes = 56)
  (h2 : female_athletes = 42)
  (h3 : male_selected = 8)
  (ratio : male_athletes / female_athletes = 4 / 3)
  (stratified_sampling : female_selected = (3 / 4) * male_selected)
  : female_selected = 6 := by
  sorry

end number_of_female_athletes_l1372_137202


namespace sum_of_non_solutions_l1372_137254

theorem sum_of_non_solutions (A B C : ℝ) :
  (∀ x : ℝ, (x ≠ -C ∧ x ≠ -10) → (x + B) * (A * x + 40) / ((x + C) * (x + 10)) = 2) →
  (A = 2 ∧ B = 10 ∧ C = 20) →
  (-10 + -20 = -30) :=
by sorry

end sum_of_non_solutions_l1372_137254


namespace factor_value_l1372_137241

theorem factor_value 
  (m : ℝ) 
  (h : ∀ x : ℝ, x + 5 = 0 → (x^2 - m * x - 40) = 0) : 
  m = 3 := 
sorry

end factor_value_l1372_137241


namespace camera_guarantee_l1372_137288

def battery_trials (b : Fin 22 → Bool) : Prop :=
  let charged := Finset.filter (λ i => b i) (Finset.univ : Finset (Fin 22))
  -- Ensuring there are exactly 15 charged batteries
  (charged.card = 15) ∧
  -- The camera works if any set of three batteries are charged
  (∀ (trials : Finset (Finset (Fin 22))),
   trials.card = 10 →
   ∃ t ∈ trials, (t.card = 3 ∧ t ⊆ charged))

theorem camera_guarantee :
  ∃ (b : Fin 22 → Bool), battery_trials b := by
  sorry

end camera_guarantee_l1372_137288


namespace time_spent_per_egg_in_seconds_l1372_137204

-- Definitions based on the conditions in the problem
def minutes_per_roll : ℕ := 30
def number_of_rolls : ℕ := 7
def total_cleaning_time : ℕ := 225
def number_of_eggs : ℕ := 60

-- Problem statement
theorem time_spent_per_egg_in_seconds :
  (total_cleaning_time - number_of_rolls * minutes_per_roll) * 60 / number_of_eggs = 15 := by
  sorry

end time_spent_per_egg_in_seconds_l1372_137204


namespace solve_system_l1372_137273

theorem solve_system :
  ∃ (x y : ℕ), 
    (∃ d : ℕ, d ∣ 42 ∧ x^2 + y^2 = 468 ∧ d + (x * y) / d = 42) ∧ 
    (x = 12 ∧ y = 18) ∨ (x = 18 ∧ y = 12) :=
sorry

end solve_system_l1372_137273


namespace modular_inverse_of_35_mod_36_l1372_137253

theorem modular_inverse_of_35_mod_36 : 
  ∃ a : ℤ, (35 * a) % 36 = 1 % 36 ∧ a = 35 := 
by 
  sorry

end modular_inverse_of_35_mod_36_l1372_137253


namespace a5_b5_sum_l1372_137217

-- Definitions of arithmetic sequences
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable
def a : ℕ → ℝ := sorry -- defining the arithmetic sequences
noncomputable
def b : ℕ → ℝ := sorry

-- Common differences for the sequences
noncomputable
def d_a : ℝ := sorry
noncomputable
def d_b : ℝ := sorry

-- Conditions given in the problem
axiom a1_b1_sum : a 1 + b 1 = 7
axiom a3_b3_sum : a 3 + b 3 = 21
axiom a_is_arithmetic : arithmetic_seq a d_a
axiom b_is_arithmetic : arithmetic_seq b d_b

-- Theorem to be proved
theorem a5_b5_sum : a 5 + b 5 = 35 := 
by sorry

end a5_b5_sum_l1372_137217


namespace pencil_cost_is_11_l1372_137206

-- Define the initial and remaining amounts
def initial_amount : ℤ := 15
def remaining_amount : ℤ := 4

-- Define the cost of the pencil
def cost_of_pencil : ℤ := initial_amount - remaining_amount

-- The statement we need to prove
theorem pencil_cost_is_11 : cost_of_pencil = 11 :=
by
  sorry

end pencil_cost_is_11_l1372_137206


namespace range_of_m_l1372_137244

theorem range_of_m (p q : Prop) (m : ℝ) (h₀ : ∀ x : ℝ, p ↔ (x^2 - 8 * x - 20 ≤ 0)) 
  (h₁ : ∀ x : ℝ, q ↔ (x^2 - 2 * x + 1 - m^2 ≤ 0)) (hm : m > 0) 
  (hsuff : (∃ x : ℝ, x > 10 ∨ x < -2) → (∃ x : ℝ, x < 1 - m ∨ x > 1 + m)) :
  0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_l1372_137244
