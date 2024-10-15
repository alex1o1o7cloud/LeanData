import Mathlib

namespace NUMINAMATH_GPT_total_distinct_symbols_l2292_229292

def numSequences (n : ℕ) : ℕ := 3^n

theorem total_distinct_symbols :
  numSequences 1 + numSequences 2 + numSequences 3 + numSequences 4 = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_distinct_symbols_l2292_229292


namespace NUMINAMATH_GPT_count_numbers_without_1_or_2_l2292_229209

/-- The number of whole numbers between 1 and 2000 that do not contain the digits 1 or 2 is 511. -/
theorem count_numbers_without_1_or_2 : 
  ∃ n : ℕ, n = 511 ∧
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2000 →
      ¬ (∃ d : ℕ, (k.digits 10).contains d ∧ (d = 1 ∨ d = 2)) → n = 511) :=
sorry

end NUMINAMATH_GPT_count_numbers_without_1_or_2_l2292_229209


namespace NUMINAMATH_GPT_two_workers_two_hours_holes_l2292_229266

theorem two_workers_two_hours_holes
    (workers1: ℝ) (holes1: ℝ) (hours1: ℝ)
    (workers2: ℝ) (hours2: ℝ)
    (h1: workers1 = 1.5)
    (h2: holes1 = 1.5)
    (h3: hours1 = 1.5)
    (h4: workers2 = 2)
    (h5: hours2 = 2)
    : (workers2 * (holes1 / (workers1 * hours1)) * hours2 = 8 / 3) := 
by {
   -- To be filled with proof, currently a placeholder.
  sorry
}

end NUMINAMATH_GPT_two_workers_two_hours_holes_l2292_229266


namespace NUMINAMATH_GPT_MsElizabethInvestmentsCount_l2292_229202

variable (MrBanksRevPerInvestment : ℕ) (MsElizabethRevPerInvestment : ℕ) (MrBanksInvestments : ℕ) (MsElizabethExtraRev : ℕ)

def MrBanksTotalRevenue := MrBanksRevPerInvestment * MrBanksInvestments
def MsElizabethTotalRevenue := MrBanksTotalRevenue + MsElizabethExtraRev
def MsElizabethInvestments := MsElizabethTotalRevenue / MsElizabethRevPerInvestment

theorem MsElizabethInvestmentsCount (h1 : MrBanksRevPerInvestment = 500) 
  (h2 : MsElizabethRevPerInvestment = 900)
  (h3 : MrBanksInvestments = 8)
  (h4 : MsElizabethExtraRev = 500) : 
  MsElizabethInvestments MrBanksRevPerInvestment MsElizabethRevPerInvestment MrBanksInvestments MsElizabethExtraRev = 5 :=
by
  sorry

end NUMINAMATH_GPT_MsElizabethInvestmentsCount_l2292_229202


namespace NUMINAMATH_GPT_evaluate_f_g_at_3_l2292_229212

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := 3 * x + 2

theorem evaluate_f_g_at_3 : f (g 3) = 123 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_g_at_3_l2292_229212


namespace NUMINAMATH_GPT_sum_of_five_consecutive_even_integers_l2292_229239

theorem sum_of_five_consecutive_even_integers (a : ℤ) 
  (h : a + (a + 4) = 144) : a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370 := by
  sorry

end NUMINAMATH_GPT_sum_of_five_consecutive_even_integers_l2292_229239


namespace NUMINAMATH_GPT_remainder_of_power_division_l2292_229272

theorem remainder_of_power_division :
  (2^222 + 222) % (2^111 + 2^56 + 1) = 218 :=
by sorry

end NUMINAMATH_GPT_remainder_of_power_division_l2292_229272


namespace NUMINAMATH_GPT_inequality_solution_l2292_229285

theorem inequality_solution (x : ℝ) : (3 * x - 1) / (x - 2) > 0 ↔ x < 1 / 3 ∨ x > 2 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2292_229285


namespace NUMINAMATH_GPT_remainder_pow_2023_l2292_229254

theorem remainder_pow_2023 (a b : ℕ) (h : b = 2023) : (3 ^ b) % 11 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_pow_2023_l2292_229254


namespace NUMINAMATH_GPT_lower_water_level_by_inches_l2292_229207

theorem lower_water_level_by_inches
  (length width : ℝ) (gallons_removed : ℝ) (gallons_to_cubic_feet : ℝ) (feet_to_inches : ℝ) : 
  length = 20 → 
  width = 25 → 
  gallons_removed = 1875 → 
  gallons_to_cubic_feet = 7.48052 → 
  feet_to_inches = 12 → 
  (gallons_removed / gallons_to_cubic_feet) / (length * width) * feet_to_inches = 6.012 := 
by 
  sorry

end NUMINAMATH_GPT_lower_water_level_by_inches_l2292_229207


namespace NUMINAMATH_GPT_f_properties_l2292_229206

noncomputable def f (x : ℝ) : ℝ := 2 ^ x

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ * f x₂) :=
by 
  sorry

end NUMINAMATH_GPT_f_properties_l2292_229206


namespace NUMINAMATH_GPT_sum_of_squares_l2292_229284

theorem sum_of_squares (x y z w a b c d : ℝ) (h1: x * y = a) (h2: x * z = b) (h3: y * z = c) (h4: x * w = d) :
  x^2 + y^2 + z^2 + w^2 = (ab + bd + da)^2 / abd := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2292_229284


namespace NUMINAMATH_GPT_four_digit_number_condition_l2292_229279

theorem four_digit_number_condition (x n : ℕ) (h1 : n = 2000 + x) (h2 : 10 * x + 2 = 2 * n + 66) : n = 2508 :=
sorry

end NUMINAMATH_GPT_four_digit_number_condition_l2292_229279


namespace NUMINAMATH_GPT_jamie_total_balls_after_buying_l2292_229274

theorem jamie_total_balls_after_buying (red_balls : ℕ) (blue_balls : ℕ) (yellow_balls : ℕ) (lost_red_balls : ℕ) (final_red_balls : ℕ) (total_balls : ℕ)
  (h1 : red_balls = 16)
  (h2 : blue_balls = 2 * red_balls)
  (h3 : lost_red_balls = 6)
  (h4 : final_red_balls = red_balls - lost_red_balls)
  (h5 : yellow_balls = 32)
  (h6 : total_balls = final_red_balls + blue_balls + yellow_balls) :
  total_balls = 74 := by
    sorry

end NUMINAMATH_GPT_jamie_total_balls_after_buying_l2292_229274


namespace NUMINAMATH_GPT_centroid_coordinates_of_tetrahedron_l2292_229238

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Given conditions
variables (O A B C G G1 : V) (OG1_subdivides : G -ᵥ O = 3 • (G1 -ᵥ G))
variable (A_centroid : G1 -ᵥ O = (1/3 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O))

-- The main proof problem
theorem centroid_coordinates_of_tetrahedron :
  G -ᵥ O = (1/4 : ℝ) • (A -ᵥ O + B -ᵥ O + C -ᵥ O) :=
sorry

end NUMINAMATH_GPT_centroid_coordinates_of_tetrahedron_l2292_229238


namespace NUMINAMATH_GPT_arithmetic_sum_first_11_terms_l2292_229262

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

variable (a : ℕ → ℝ)

theorem arithmetic_sum_first_11_terms (h_arith_seq : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_sum_condition : a 2 + a 6 + a 10 = 6) :
  sum_first_n_terms a 11 = 22 :=
sorry

end NUMINAMATH_GPT_arithmetic_sum_first_11_terms_l2292_229262


namespace NUMINAMATH_GPT_min_value_of_quadratic_l2292_229251

theorem min_value_of_quadratic (m : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 2 * x + m) 
  (min_val : ∀ x ≥ 2, f x ≥ -2) : m = -2 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l2292_229251


namespace NUMINAMATH_GPT_sum_a4_a5_a6_l2292_229235

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Conditions
axiom h1 : a 1 = 2
axiom h2 : a 3 = -10

-- Definition of arithmetic sequence
axiom h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- Proof problem statement
theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = -66 :=
by
  sorry

end NUMINAMATH_GPT_sum_a4_a5_a6_l2292_229235


namespace NUMINAMATH_GPT_perfect_square_iff_n_eq_one_l2292_229226

theorem perfect_square_iff_n_eq_one (n : ℕ) : ∃ m : ℕ, n^2 + 3 * n = m^2 ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_perfect_square_iff_n_eq_one_l2292_229226


namespace NUMINAMATH_GPT_vector_subtraction_l2292_229221

def vector1 : ℝ × ℝ := (3, -5)
def vector2 : ℝ × ℝ := (2, -6)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_subtraction :
  (scalar1 • vector1 - scalar2 • vector2) = (6, -2) := by
  sorry

end NUMINAMATH_GPT_vector_subtraction_l2292_229221


namespace NUMINAMATH_GPT_A_works_alone_45_days_l2292_229219

open Nat

theorem A_works_alone_45_days (x : ℕ) :
  (∀ x : ℕ, (9 * (1 / x + 1 / 40) + 23 * (1 / 40) = 1) → (x = 45)) :=
sorry

end NUMINAMATH_GPT_A_works_alone_45_days_l2292_229219


namespace NUMINAMATH_GPT_ratio_of_areas_l2292_229233

theorem ratio_of_areas 
  (a b c : ℕ) (d e f : ℕ)
  (hABC : a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2)
  (hDEF : d = 8 ∧ e = 15 ∧ f = 17 ∧ d^2 + e^2 = f^2) :
  (1/2 * a * b) / (1/2 * d * e) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l2292_229233


namespace NUMINAMATH_GPT_find_real_numbers_l2292_229218

theorem find_real_numbers (a b c : ℝ)    :
  (a + b + c = 3) → (a^2 + b^2 + c^2 = 35) → (a^3 + b^3 + c^3 = 99) → 
  (a = 1 ∧ b = -3 ∧ c = 5) ∨ (a = 1 ∧ b = 5 ∧ c = -3) ∨ 
  (a = -3 ∧ b = 1 ∧ c = 5) ∨ (a = -3 ∧ b = 5 ∧ c = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = -3) ∨ (a = 5 ∧ b = -3 ∧ c = 1) :=
by intros h1 h2 h3; sorry

end NUMINAMATH_GPT_find_real_numbers_l2292_229218


namespace NUMINAMATH_GPT_max_cookies_ben_could_have_eaten_l2292_229299

theorem max_cookies_ben_could_have_eaten (c : ℕ) (h_total : c = 36)
  (h_beth : ∃ n: ℕ, (n = 2 ∨ n = 3) ∧ c = (n + 1) * ben)
  (h_max : ∀ n, (n = 2 ∨ n = 3) → n * 12 ≤ n * ben)
  : ben = 12 := 
sorry

end NUMINAMATH_GPT_max_cookies_ben_could_have_eaten_l2292_229299


namespace NUMINAMATH_GPT_sin_690_eq_neg_half_l2292_229250

theorem sin_690_eq_neg_half :
  Real.sin (690 * Real.pi / 180) = -1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sin_690_eq_neg_half_l2292_229250


namespace NUMINAMATH_GPT_Inequality_Solution_Set_Range_of_c_l2292_229228

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def g (x : ℝ) : ℝ := -(((-x)^2) + 2 * (-x))

theorem Inequality_Solution_Set (x : ℝ) :
  (g x ≥ f x - |x - 1|) ↔ (-1 ≤ x ∧ x ≤ 1/2) :=
by
  sorry

theorem Range_of_c (c : ℝ) :
  (∀ x : ℝ, g x + c ≤ f x - |x - 1|) ↔ (c ≤ -9/8) :=
by
  sorry

end NUMINAMATH_GPT_Inequality_Solution_Set_Range_of_c_l2292_229228


namespace NUMINAMATH_GPT_trigonometric_identity_l2292_229275

theorem trigonometric_identity (α : ℝ) (h : Real.tan (α + π / 4) = -3) :
  2 * Real.cos (2 * α) + 3 * Real.sin (2 * α) - Real.sin α ^ 2 = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2292_229275


namespace NUMINAMATH_GPT_functional_eq_app_only_solutions_l2292_229244

noncomputable def f : Real → Real := sorry

theorem functional_eq_app (n : ℕ) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
  f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2 :=
sorry

theorem only_solutions (f : ℝ → ℝ) (hf : ∀ n : ℕ, ∀ x : Fin n → ℝ, (∀ i, 0 ≤ x i) → f (Finset.univ.sum fun i => (x i)^2) = Finset.univ.sum fun i => (f (x i))^2) :
  f = (fun x => 0) ∨ f = (fun x => x) :=
sorry

end NUMINAMATH_GPT_functional_eq_app_only_solutions_l2292_229244


namespace NUMINAMATH_GPT_find_interest_rate_l2292_229281

theorem find_interest_rate (initial_investment : ℚ) (duration_months : ℚ) 
  (first_rate : ℚ) (final_value : ℚ) (s : ℚ) :
  initial_investment = 15000 →
  duration_months = 9 →
  first_rate = 0.09 →
  final_value = 17218.50 →
  (∃ s : ℚ, 16012.50 * (1 + (s * 0.75) / 100) = final_value) →
  s = 10 := 
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l2292_229281


namespace NUMINAMATH_GPT_perimeter_shaded_region_l2292_229242

noncomputable def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def arc_length_per_circle (C : ℝ) : ℝ := C / 4

theorem perimeter_shaded_region (C : ℝ) (hC : C = 48) : 
  3 * arc_length_per_circle C = 36 := by
  sorry

end NUMINAMATH_GPT_perimeter_shaded_region_l2292_229242


namespace NUMINAMATH_GPT_Bhupathi_amount_l2292_229240

variable (A B : ℝ)

theorem Bhupathi_amount
  (h1 : A + B = 1210)
  (h2 : (4 / 15) * A = (2 / 5) * B) :
  B = 484 := by
  sorry

end NUMINAMATH_GPT_Bhupathi_amount_l2292_229240


namespace NUMINAMATH_GPT_cookie_baking_time_l2292_229236

theorem cookie_baking_time 
  (total_time : ℕ) 
  (white_icing_time: ℕ)
  (chocolate_icing_time: ℕ) 
  (total_icing_time : white_icing_time + chocolate_icing_time = 60)
  (total_cooking_time : total_time = 120):

  (total_time - (white_icing_time + chocolate_icing_time) = 60) :=
by
  sorry

end NUMINAMATH_GPT_cookie_baking_time_l2292_229236


namespace NUMINAMATH_GPT_oil_bill_for_january_l2292_229243

-- Definitions and conditions
def ratio_F_J (F J : ℝ) : Prop := F / J = 3 / 2
def ratio_F_M (F M : ℝ) : Prop := F / M = 4 / 5
def ratio_F_J_modified (F J : ℝ) : Prop := (F + 20) / J = 5 / 3
def ratio_F_M_modified (F M : ℝ) : Prop := (F + 20) / M = 2 / 3

-- The main statement to prove
theorem oil_bill_for_january (J F M : ℝ) 
  (h1 : ratio_F_J F J)
  (h2 : ratio_F_M F M)
  (h3 : ratio_F_J_modified F J)
  (h4 : ratio_F_M_modified F M) :
  J = 120 :=
sorry

end NUMINAMATH_GPT_oil_bill_for_january_l2292_229243


namespace NUMINAMATH_GPT_number_of_subsets_of_P_l2292_229247

noncomputable def P : Set ℝ := {x | x^2 - 2*x + 1 = 0}

theorem number_of_subsets_of_P : ∃ (n : ℕ), n = 2 ∧ ∀ S : Set ℝ, S ⊆ P → S = ∅ ∨ S = {1} := by
  sorry

end NUMINAMATH_GPT_number_of_subsets_of_P_l2292_229247


namespace NUMINAMATH_GPT_assignment_plan_count_l2292_229241

noncomputable def number_of_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let tasks := ["translation", "tour guide", "etiquette", "driver"]
  let v1 := ["Xiao Zhang", "Xiao Zhao"]
  let v2 := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Condition: Xiao Zhang and Xiao Zhao can only take positions for translation and tour guide
  -- Calculate the number of ways to assign based on the given conditions
  -- 36 is the total number of assignment plans
  36

theorem assignment_plan_count :
  number_of_assignment_plans = 36 :=
  sorry

end NUMINAMATH_GPT_assignment_plan_count_l2292_229241


namespace NUMINAMATH_GPT_discriminant_formula_l2292_229267

def discriminant_cubic_eq (x1 x2 x3 p q : ℝ) : ℝ :=
  (x1 - x2)^2 * (x2 - x3)^2 * (x3 - x1)^2

theorem discriminant_formula (x1 x2 x3 p q : ℝ)
  (h1 : x1 + x2 + x3 = 0)
  (h2 : x1 * x2 + x1 * x3 + x2 * x3 = p)
  (h3 : x1 * x2 * x3 = -q) :
  discriminant_cubic_eq x1 x2 x3 p q = -4 * p^3 - 27 * q^2 :=
by sorry

end NUMINAMATH_GPT_discriminant_formula_l2292_229267


namespace NUMINAMATH_GPT_incorrect_inequality_given_conditions_l2292_229286

variable {a b x y : ℝ}

theorem incorrect_inequality_given_conditions 
  (h1 : a > b) (h2 : x > y) : ¬ (|a| * x > |a| * y) :=
sorry

end NUMINAMATH_GPT_incorrect_inequality_given_conditions_l2292_229286


namespace NUMINAMATH_GPT_dan_helmet_crater_difference_l2292_229216

theorem dan_helmet_crater_difference :
  ∀ (r d : ℕ), 
  (r = 75) ∧ (d = 35) ∧ (r = 15 + (d + (r - 15 - d))) ->
  ((d - (r - 15 - d)) = 10) :=
by
  intros r d h
  have hr : r = 75 := h.1
  have hd : d = 35 := h.2.1
  have h_combined : r = 15 + (d + (r - 15 - d)) := h.2.2
  sorry

end NUMINAMATH_GPT_dan_helmet_crater_difference_l2292_229216


namespace NUMINAMATH_GPT_unique_number_not_in_range_l2292_229280

noncomputable def g (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h1 : g a b c d 13 = 13)
  (h2 : g a b c d 31 = 31)
  (h3 : ∀ x, x ≠ -d / c → g a b c d (g a b c d x) = x) :
  ∀ y, ∃! x, g a b c d x = y :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_number_not_in_range_l2292_229280


namespace NUMINAMATH_GPT_incorrect_statement_B_l2292_229253

noncomputable def y (x : ℝ) : ℝ := 2 / x 

theorem incorrect_statement_B :
  ¬ ∀ x > 0, ∀ y1 y2 : ℝ, x < y1 → y1 < y2 → y x < y y2 := sorry

end NUMINAMATH_GPT_incorrect_statement_B_l2292_229253


namespace NUMINAMATH_GPT_inequality_proof_l2292_229256

theorem inequality_proof (n : ℕ) (a : ℝ) (h₀ : n > 1) (h₁ : 0 < a) (h₂ : a < 1) : 
  1 + a < (1 + a / n) ^ n ∧ (1 + a / n) ^ n < (1 + a / (n + 1)) ^ (n + 1) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l2292_229256


namespace NUMINAMATH_GPT_degree_diploma_salary_ratio_l2292_229223

theorem degree_diploma_salary_ratio
  (jared_salary : ℕ)
  (diploma_monthly_salary : ℕ)
  (h_annual_salary : jared_salary = 144000)
  (h_diploma_annual_salary : 12 * diploma_monthly_salary = 48000) :
  (jared_salary / (12 * diploma_monthly_salary)) = 3 := 
by sorry

end NUMINAMATH_GPT_degree_diploma_salary_ratio_l2292_229223


namespace NUMINAMATH_GPT_int_pairs_satisfy_conditions_l2292_229208

theorem int_pairs_satisfy_conditions (m n : ℤ) :
  (∃ a b : ℤ, m^2 + n = a^2 ∧ n^2 + m = b^2) ↔ 
  ∃ k : ℤ, (m = 0 ∧ n = k^2) ∨ (m = k^2 ∧ n = 0) ∨ (m = 1 ∧ n = -1) ∨ (m = -1 ∧ n = 1) := by
  sorry

end NUMINAMATH_GPT_int_pairs_satisfy_conditions_l2292_229208


namespace NUMINAMATH_GPT_minimize_y_l2292_229296

variable (a b x : ℝ)

def y := (x - a)^2 + (x - b)^2

theorem minimize_y : ∃ x : ℝ, (∀ (x' : ℝ), y x a b ≤ y x' a b) ∧ x = (a + b) / 2 := by
  sorry

end NUMINAMATH_GPT_minimize_y_l2292_229296


namespace NUMINAMATH_GPT_part1_part2_l2292_229215

-- Part 1
theorem part1 (x y : ℤ) (hx : x = -2) (hy : y = -3) :
  x^2 - 2 * (x^2 - 3 * y) - 3 * (2 * x^2 + 5 * y) = -1 :=
by
  -- Proof to be provided
  sorry

-- Part 2
theorem part2 (a b : ℤ) (hab : a - b = 2 * b^2) :
  2 * (a^3 - 2 * b^2) - (2 * b - a) + a - 2 * a^3 = 0 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_part1_part2_l2292_229215


namespace NUMINAMATH_GPT_roots_quadratic_l2292_229249

theorem roots_quadratic (a b : ℝ) (h : ∀ x : ℝ, x^2 - 7 * x + 7 = 0 → (x = a) ∨ (x = b)) :
  a^2 + b^2 = 35 :=
sorry

end NUMINAMATH_GPT_roots_quadratic_l2292_229249


namespace NUMINAMATH_GPT_fraction_correct_l2292_229234

theorem fraction_correct (x : ℚ) (h : (5 / 6) * 576 = x * 576 + 300) : x = 5 / 16 := 
sorry

end NUMINAMATH_GPT_fraction_correct_l2292_229234


namespace NUMINAMATH_GPT_jane_savings_l2292_229222

-- Given conditions
def cost_pair_1 : ℕ := 50
def cost_pair_2 : ℕ := 40

def promotion_A (cost1 cost2 : ℕ) : ℕ :=
  cost1 + cost2 / 2

def promotion_B (cost1 cost2 : ℕ) : ℕ :=
  cost1 + (cost2 - 15)

-- Define the savings calculation
def savings (promoA promoB : ℕ) : ℕ :=
  promoB - promoA

-- Specify the theorem to prove
theorem jane_savings :
  savings (promotion_A cost_pair_1 cost_pair_2) (promotion_B cost_pair_1 cost_pair_2) = 5 := 
by
  sorry

end NUMINAMATH_GPT_jane_savings_l2292_229222


namespace NUMINAMATH_GPT_total_money_l2292_229232

-- Define the problem statement
theorem total_money (n : ℕ) (hn : 3 * n = 75) : (n * 1 + n * 5 + n * 10) = 400 :=
by sorry

end NUMINAMATH_GPT_total_money_l2292_229232


namespace NUMINAMATH_GPT_sum_of_digits_base2_310_l2292_229271

-- We define what it means to convert a number to binary and sum its digits.
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

-- The main statement of the problem.
theorem sum_of_digits_base2_310 :
  sum_of_binary_digits 310 = 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_base2_310_l2292_229271


namespace NUMINAMATH_GPT_initial_men_l2292_229269

variable (x : ℕ)

-- Conditions
def condition1 (x : ℕ) : Prop :=
  -- The hostel had provisions for x men for 28 days.
  true

def condition2 (x : ℕ) : Prop :=
  -- If 50 men left, the food would last for 35 days for the remaining x - 50 men.
  (x - 50) * 35 = x * 28

-- Theorem to prove
theorem initial_men (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : x = 250 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_l2292_229269


namespace NUMINAMATH_GPT_ball_bounces_to_C_l2292_229268

/--
On a rectangular table with dimensions 9 cm in length and 7 cm in width, a small ball is shot from point A at a 45-degree angle. Upon reaching point E, it bounces off at a 45-degree angle and continues to roll forward. Throughout its motion, the ball bounces off the table edges at a 45-degree angle each time. Prove that, starting from point A, the ball first reaches point C after exactly 14 bounces.
-/
theorem ball_bounces_to_C (length width : ℝ) (angle : ℝ) (bounce_angle : ℝ) :
  length = 9 ∧ width = 7 ∧ angle = 45 ∧ bounce_angle = 45 → bounces_to_C = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_ball_bounces_to_C_l2292_229268


namespace NUMINAMATH_GPT_distinct_students_l2292_229278

theorem distinct_students 
  (students_euler : ℕ) (students_gauss : ℕ) (students_fibonacci : ℕ) (overlap_euler_gauss : ℕ)
  (h_euler : students_euler = 15) 
  (h_gauss : students_gauss = 10) 
  (h_fibonacci : students_fibonacci = 12) 
  (h_overlap : overlap_euler_gauss = 3) 
  : students_euler + students_gauss + students_fibonacci - overlap_euler_gauss = 34 :=
by
  sorry

end NUMINAMATH_GPT_distinct_students_l2292_229278


namespace NUMINAMATH_GPT_lemons_count_l2292_229245

def total_fruits (num_baskets : ℕ) (total : ℕ) : Prop := num_baskets = 5 ∧ total = 58
def basket_contents (basket : ℕ → ℕ) : Prop := 
  basket 1 = 18 ∧ -- mangoes
  basket 2 = 10 ∧ -- pears
  basket 3 = 12 ∧ -- pawpaws
  (∀ i, (i = 4 ∨ i = 5) → basket i = (basket 4 + basket 5) / 2)

theorem lemons_count (num_baskets : ℕ) (total : ℕ) (basket : ℕ → ℕ) : 
  total_fruits num_baskets total ∧ basket_contents basket → basket 5 = 9 :=
by
  sorry

end NUMINAMATH_GPT_lemons_count_l2292_229245


namespace NUMINAMATH_GPT_students_neither_math_physics_drama_exclusive_l2292_229201

def total_students : ℕ := 75
def math_students : ℕ := 42
def physics_students : ℕ := 35
def both_students : ℕ := 25
def drama_exclusive_students : ℕ := 10

theorem students_neither_math_physics_drama_exclusive : 
  total_students - (math_students + physics_students - both_students + drama_exclusive_students) = 13 :=
by
  sorry

end NUMINAMATH_GPT_students_neither_math_physics_drama_exclusive_l2292_229201


namespace NUMINAMATH_GPT_find_number_l2292_229298

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 :=
sorry

end NUMINAMATH_GPT_find_number_l2292_229298


namespace NUMINAMATH_GPT_central_angle_of_sector_l2292_229230

theorem central_angle_of_sector (r l : ℝ) (h1 : l + 2 * r = 4) (h2 : (1 / 2) * l * r = 1) : l / r = 2 :=
by
  -- The proof should be provided here
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l2292_229230


namespace NUMINAMATH_GPT_gcd_2015_15_l2292_229231

theorem gcd_2015_15 : Nat.gcd 2015 15 = 5 :=
by
  have h1 : 2015 = 15 * 134 + 5 := by rfl
  have h2 : 15 = 5 * 3 := by rfl
  sorry

end NUMINAMATH_GPT_gcd_2015_15_l2292_229231


namespace NUMINAMATH_GPT_no_real_solutions_cubic_eq_l2292_229294

theorem no_real_solutions_cubic_eq : ∀ x : ℝ, ¬ (∃ (y : ℝ), y = x^(1/3) ∧ y = 15 / (6 - y)) :=
by
  intro x
  intro hexist
  obtain ⟨y, hy1, hy2⟩ := hexist
  have h_cubic : y * (6 - y) = 15 := by sorry -- from y = 15 / (6 - y)
  have h_quad : y^2 - 6 * y + 15 = 0 := by sorry -- after expanding y(6 - y) = 15
  sorry -- remainder to show no real solution due to negative discriminant

end NUMINAMATH_GPT_no_real_solutions_cubic_eq_l2292_229294


namespace NUMINAMATH_GPT_alex_bought_3_bags_of_chips_l2292_229213

theorem alex_bought_3_bags_of_chips (x : ℝ) : 
    (1 * x + 5 + 73) / x = 27 → x = 3 := by sorry

end NUMINAMATH_GPT_alex_bought_3_bags_of_chips_l2292_229213


namespace NUMINAMATH_GPT_minimum_value_of_phi_l2292_229211

noncomputable def f (A ω φ : ℝ) (x : ℝ) := A * Real.sin (ω * x + φ)

noncomputable def minimum_positive_period (ω : ℝ) := 2 * Real.pi / ω

theorem minimum_value_of_phi {A ω φ : ℝ} (hA : A > 0) (hω : ω > 0) 
  (h_period : minimum_positive_period ω = Real.pi) 
  (h_symmetry : ∀ x, f A ω φ x = f A ω φ (2 * Real.pi / ω - x)) : 
  ∃ k : ℤ, |φ| = |k * Real.pi - Real.pi / 6| → |φ| = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_phi_l2292_229211


namespace NUMINAMATH_GPT_sequence_nth_term_mod_2500_l2292_229204

def sequence_nth_term (n : ℕ) : ℕ :=
  -- this is a placeholder function definition; the actual implementation to locate the nth term is skipped
  sorry

theorem sequence_nth_term_mod_2500 : (sequence_nth_term 2500) % 7 = 1 := 
sorry

end NUMINAMATH_GPT_sequence_nth_term_mod_2500_l2292_229204


namespace NUMINAMATH_GPT_square_table_production_l2292_229277

theorem square_table_production (x y : ℝ) :
  x + y = 5 ∧ 50 * x * 4 = 300 * y → 
  x = 3 ∧ y = 2 ∧ 50 * x = 150 :=
by
  sorry

end NUMINAMATH_GPT_square_table_production_l2292_229277


namespace NUMINAMATH_GPT_fourth_grade_students_l2292_229282

theorem fourth_grade_students (initial_students left_students new_students final_students : ℕ) 
    (h1 : initial_students = 33) 
    (h2 : left_students = 18) 
    (h3 : new_students = 14) 
    (h4 : final_students = initial_students - left_students + new_students) :
    final_students = 29 := 
by 
    sorry

end NUMINAMATH_GPT_fourth_grade_students_l2292_229282


namespace NUMINAMATH_GPT_min_M_for_inequality_l2292_229203

noncomputable def M := (9 * Real.sqrt 2) / 32

theorem min_M_for_inequality (a b c : ℝ) : 
  abs (a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)) 
  ≤ M * (a^2 + b^2 + c^2)^2 := 
sorry

end NUMINAMATH_GPT_min_M_for_inequality_l2292_229203


namespace NUMINAMATH_GPT_Adam_total_cost_l2292_229227

theorem Adam_total_cost :
  let laptop1_cost := 500
  let laptop2_base_cost := 3 * laptop1_cost
  let discount := 0.15 * laptop2_base_cost
  let laptop2_cost := laptop2_base_cost - discount
  let external_hard_drive := 80
  let mouse := 20
  let software1 := 120
  let software2 := 2 * 120
  let insurance1 := 0.10 * laptop1_cost
  let insurance2 := 0.10 * laptop2_cost
  let total_cost1 := laptop1_cost + external_hard_drive + mouse + software1 + insurance1
  let total_cost2 := laptop2_cost + external_hard_drive + mouse + software2 + insurance2
  total_cost1 + total_cost2 = 2512.5 :=
by
  sorry

end NUMINAMATH_GPT_Adam_total_cost_l2292_229227


namespace NUMINAMATH_GPT_min_ratio_ax_l2292_229289

theorem min_ratio_ax (a x y : ℕ) (ha : a > 100) (hx : x > 100) (hy : y > 100) 
: y^2 - 1 = a^2 * (x^2 - 1) → ∃ (k : ℕ), k = 2 ∧ (a = k * x) := 
sorry

end NUMINAMATH_GPT_min_ratio_ax_l2292_229289


namespace NUMINAMATH_GPT_obtuse_right_triangle_cannot_exist_l2292_229263

-- Definitions of various types of triangles

def is_acute (θ : ℕ) : Prop := θ < 90
def is_right (θ : ℕ) : Prop := θ = 90
def is_obtuse (θ : ℕ) : Prop := θ > 90

def is_isosceles (a b c : ℕ) : Prop := a = b ∨ b = c ∨ a = c
def is_scalene (a b c : ℕ) : Prop := ¬ (a = b) ∧ ¬ (b = c) ∧ ¬ (a = c)
def is_triangle (a b c : ℕ) : Prop := a + b + c = 180

-- Propositions for the types of triangles given in the problem

def acute_isosceles_triangle (a b : ℕ) : Prop :=
  is_triangle a a (180 - 2 * a) ∧ is_acute a ∧ is_isosceles a a (180 - 2 * a)

def isosceles_right_triangle (a : ℕ) : Prop :=
  is_triangle a a 90 ∧ is_right 90 ∧ is_isosceles a a 90

def obtuse_right_triangle (a b : ℕ) : Prop :=
  is_triangle a 90 (180 - 90 - a) ∧ is_right 90 ∧ is_obtuse (180 - 90 - a)

def scalene_right_triangle (a b : ℕ) : Prop :=
  is_triangle a b 90 ∧ is_right 90 ∧ is_scalene a b 90

def scalene_obtuse_triangle (a b : ℕ) : Prop :=
  is_triangle a b (180 - a - b) ∧ is_obtuse (180 - a - b) ∧ is_scalene a b (180 - a - b)

-- The final theorem stating that obtuse right triangle cannot exist

theorem obtuse_right_triangle_cannot_exist (a b : ℕ) :
  ¬ exists (a b : ℕ), obtuse_right_triangle a b :=
by
  sorry

end NUMINAMATH_GPT_obtuse_right_triangle_cannot_exist_l2292_229263


namespace NUMINAMATH_GPT_probability_at_least_one_white_ball_l2292_229229

/-
  We define the conditions:
  - num_white: the number of white balls,
  - num_red: the number of red balls,
  - total_balls: the total number of balls,
  - num_drawn: the number of balls drawn.
-/
def num_white : ℕ := 5
def num_red : ℕ := 4
def total_balls : ℕ := num_white + num_red
def num_drawn : ℕ := 3

/-
  Given the conditions, we need to prove that the probability of drawing at least one white ball is 20/21.
-/
theorem probability_at_least_one_white_ball :
  (1 : ℚ) - (4 / 84) = 20 / 21 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_white_ball_l2292_229229


namespace NUMINAMATH_GPT_find_f_sqrt_10_l2292_229224

-- Definitions and conditions provided in the problem
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2 - 8*x + 30

-- The problem specific conditions for f
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_periodic : is_periodic_function f 2)
variable (h_condition : f_condition f)

-- The statement to prove
theorem find_f_sqrt_10 : f (Real.sqrt 10) = -24 :=
by
  sorry

end NUMINAMATH_GPT_find_f_sqrt_10_l2292_229224


namespace NUMINAMATH_GPT_min_trips_needed_l2292_229297

noncomputable def min_trips (n : ℕ) (h : 2 ≤ n) : ℕ :=
  6

theorem min_trips_needed
  (n : ℕ) (h : 2 ≤ n) (students : Finset (Fin (2 * n)))
  (trip : ℕ → Finset (Fin (2 * n)))
  (trip_cond : ∀ i, (trip i).card = n)
  (pair_cond : ∀ (s t : Fin (2 * n)),
    s ≠ t → ∃ i, s ∈ trip i ∧ t ∈ trip i) :
  ∃ k, k = min_trips n h :=
by
  use 6
  sorry

end NUMINAMATH_GPT_min_trips_needed_l2292_229297


namespace NUMINAMATH_GPT_pen_count_l2292_229283

-- Define the conditions
def total_pens := 140
def difference := 20

-- Define the quantities to be proven
def ballpoint_pens := (total_pens - difference) / 2
def fountain_pens := total_pens - ballpoint_pens

-- The theorem to be proved
theorem pen_count :
  ballpoint_pens = 60 ∧ fountain_pens = 80 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pen_count_l2292_229283


namespace NUMINAMATH_GPT_series_converges_to_three_fourths_l2292_229264

open BigOperators

noncomputable def series_sum (n : ℕ) : ℝ := ∑' k, (k : ℝ) / (3 ^ k)

theorem series_converges_to_three_fourths : series_sum = (3 / 4) :=
sorry

end NUMINAMATH_GPT_series_converges_to_three_fourths_l2292_229264


namespace NUMINAMATH_GPT_compute_x_l2292_229210

/-- 
Let ABC be a triangle. 
Points D, E, and F are on BC, CA, and AB, respectively. 
Given that AE/AC = CD/CB = BF/BA = x for some x with 1/2 < x < 1. 
Segments AD, BE, and CF divide the triangle into 7 non-overlapping regions: 
4 triangles and 3 quadrilaterals. 
The total area of the 4 triangles equals the total area of the 3 quadrilaterals. 
Compute the value of x.
-/
theorem compute_x (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 1)
  (h3 : (∃ (triangleArea quadrilateralArea : ℝ), 
          let A := triangleArea + 3 * x
          let B := quadrilateralArea
          A = B))
  : x = (11 - Real.sqrt 37) / 6 := 
sorry

end NUMINAMATH_GPT_compute_x_l2292_229210


namespace NUMINAMATH_GPT_combined_cost_l2292_229287

theorem combined_cost (wallet_cost : ℕ) (purse_cost : ℕ)
    (h_wallet_cost : wallet_cost = 22)
    (h_purse_cost : purse_cost = 4 * wallet_cost - 3) :
    wallet_cost + purse_cost = 107 :=
by
  rw [h_wallet_cost, h_purse_cost]
  norm_num
  sorry

end NUMINAMATH_GPT_combined_cost_l2292_229287


namespace NUMINAMATH_GPT_calc_value_l2292_229260

theorem calc_value : (3000 * (3000 ^ 2999) * 2 = 2 * 3000 ^ 3000) := 
by
  sorry

end NUMINAMATH_GPT_calc_value_l2292_229260


namespace NUMINAMATH_GPT_sin_neg_nine_pi_div_two_l2292_229217

theorem sin_neg_nine_pi_div_two : Real.sin (-9 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_GPT_sin_neg_nine_pi_div_two_l2292_229217


namespace NUMINAMATH_GPT_village_population_l2292_229225

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 :=
sorry

end NUMINAMATH_GPT_village_population_l2292_229225


namespace NUMINAMATH_GPT_average_rainfall_l2292_229261

theorem average_rainfall (r d h : ℕ) (rainfall_eq : r = 450) (days_eq : d = 30) (hours_eq : h = 24) :
  r / (d * h) = 25 / 16 := 
  by 
    -- Insert appropriate proof here
    sorry

end NUMINAMATH_GPT_average_rainfall_l2292_229261


namespace NUMINAMATH_GPT_exists_nat_lt_100_two_different_squares_l2292_229259

theorem exists_nat_lt_100_two_different_squares :
  ∃ n : ℕ, n < 100 ∧ 
    ∃ a b c d : ℕ, a^2 + b^2 = n ∧ c^2 + d^2 = n ∧ (a ≠ c ∨ b ≠ d) ∧ a ≠ b ∧ c ≠ d :=
by
  sorry

end NUMINAMATH_GPT_exists_nat_lt_100_two_different_squares_l2292_229259


namespace NUMINAMATH_GPT_well_depth_is_correct_l2292_229255

noncomputable def depth_of_well : ℝ :=
  122500

theorem well_depth_is_correct (d t1 : ℝ) : 
  t1 = Real.sqrt (d / 20) ∧ 
  (d / 1100) + t1 = 10 →
  d = depth_of_well := 
by
  sorry

end NUMINAMATH_GPT_well_depth_is_correct_l2292_229255


namespace NUMINAMATH_GPT_sum_of_fractions_equals_16_l2292_229246

def list_of_fractions : List (ℚ) := [
  2 / 10,
  4 / 10,
  6 / 10,
  8 / 10,
  10 / 10,
  15 / 10,
  20 / 10,
  25 / 10,
  30 / 10,
  40 / 10
]

theorem sum_of_fractions_equals_16 : list_of_fractions.sum = 16 := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_equals_16_l2292_229246


namespace NUMINAMATH_GPT_min_value_of_reciprocal_l2292_229214

theorem min_value_of_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) :
  (∀ r, r = 1 / x + 1 / y → r ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_l2292_229214


namespace NUMINAMATH_GPT_total_rainfall_l2292_229200

theorem total_rainfall (rain_first_hour : ℕ) (rain_second_hour : ℕ) : Prop :=
  rain_first_hour = 5 →
  rain_second_hour = 7 + 2 * rain_first_hour →
  rain_first_hour + rain_second_hour = 22

-- Add sorry to skip the proof.

end NUMINAMATH_GPT_total_rainfall_l2292_229200


namespace NUMINAMATH_GPT_square_area_l2292_229220

theorem square_area (side_length : ℕ) (h : side_length = 12) : side_length * side_length = 144 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l2292_229220


namespace NUMINAMATH_GPT_shekar_biology_marks_l2292_229290

variable (M S SS E A : ℕ)

theorem shekar_biology_marks (hM : M = 76) (hS : S = 65) (hSS : SS = 82) (hE : E = 67) (hA : A = 77) :
  let total_marks := M + S + SS + E
  let total_average_marks := A * 5
  let biology_marks := total_average_marks - total_marks
  biology_marks = 95 :=
by
  sorry

end NUMINAMATH_GPT_shekar_biology_marks_l2292_229290


namespace NUMINAMATH_GPT_correct_operation_l2292_229248

theorem correct_operation (a b m : ℤ) :
    ¬(((-2 * a) ^ 2 = -4 * a ^ 2) ∨ ((a - b) ^ 2 = a ^ 2 - b ^ 2) ∨ ((a ^ 5) ^ 2 = a ^ 7)) ∧ ((-m + 2) * (-m - 2) = m ^ 2 - 4) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l2292_229248


namespace NUMINAMATH_GPT_value_of_expression_l2292_229252

theorem value_of_expression : (7^2 - 6^2)^4 = 28561 :=
by sorry

end NUMINAMATH_GPT_value_of_expression_l2292_229252


namespace NUMINAMATH_GPT_perimeter_of_original_rectangle_l2292_229291

theorem perimeter_of_original_rectangle
  (s : ℕ)
  (h1 : 4 * s = 24)
  (l w : ℕ)
  (h2 : l = 3 * s)
  (h3 : w = s) :
  2 * (l + w) = 48 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_original_rectangle_l2292_229291


namespace NUMINAMATH_GPT_four_digit_number_properties_l2292_229288

theorem four_digit_number_properties :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 8 ∧ 
    a = 3 * b ∧ 
    d = 4 * c ∧ 
    1000 * a + 100 * b + 10 * c + d = 6200 :=
by
  sorry

end NUMINAMATH_GPT_four_digit_number_properties_l2292_229288


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_a_min_value_of_c_l2292_229257

noncomputable def f (a c x : ℝ) : ℝ :=
  a * Real.log x + (x - c) * abs (x - c)

-- 1. Monotonic intervals
theorem monotonic_intervals (a c : ℝ) (ha : a = -3 / 4) (hc : c = 1 / 4) :
  ((∀ x, 0 < x ∧ x < 3 / 4 → f a c x > f a c (x - 1)) ∧ (∀ x, 3 / 4 < x → f a c x > f a c (x - 1))) :=
sorry

-- 2. Range of values for a
theorem range_of_a (a c : ℝ) (hc : c = a / 2 + 1) (h : ∀ x > c, f a c x ≥ 1 / 4) :
  -2 < a ∧ a ≤ -1 :=
sorry

-- 3. Minimum value of c
theorem min_value_of_c (a c x1 x2 : ℝ) (hx1 : x1 = Real.sqrt (-a / 2)) (hx2 : x2 = c)
  (h_tangents_perpendicular : f a c x1 * f a c x2 = -1) :
  c = 3 * Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_a_min_value_of_c_l2292_229257


namespace NUMINAMATH_GPT_total_pennies_donated_l2292_229273

theorem total_pennies_donated:
  let cassandra_pennies := 5000
  let james_pennies := cassandra_pennies - 276
  let stephanie_pennies := 2 * james_pennies
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by
  sorry

end NUMINAMATH_GPT_total_pennies_donated_l2292_229273


namespace NUMINAMATH_GPT_tangent_curve_l2292_229295

variable {k a b : ℝ}

theorem tangent_curve (h1 : 3 = (1 : ℝ)^3 + a * 1 + b)
(h2 : k = 2)
(h3 : k = 3 * (1 : ℝ)^2 + a) :
b = 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_curve_l2292_229295


namespace NUMINAMATH_GPT_three_pow_zero_eq_one_l2292_229293

theorem three_pow_zero_eq_one : 3^0 = 1 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_three_pow_zero_eq_one_l2292_229293


namespace NUMINAMATH_GPT_min_value_problem1_min_value_problem2_l2292_229265

-- Problem 1: Prove that the minimum value of the function y = x + 4/(x + 1) + 6 is 9 given x > -1
theorem min_value_problem1 (x : ℝ) (h : x > -1) : (x + 4 / (x + 1) + 6) ≥ 9 := 
sorry

-- Problem 2: Prove that the minimum value of the function y = (x^2 + 8) / (x - 1) is 8 given x > 1
theorem min_value_problem2 (x : ℝ) (h : x > 1) : ((x^2 + 8) / (x - 1)) ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_problem1_min_value_problem2_l2292_229265


namespace NUMINAMATH_GPT_fraction_proof_l2292_229205

-- Define N
def N : ℕ := 24

-- Define F that satisfies the equation N = F + 15
def F := N - 15

-- Define the fraction that N exceeds by 15
noncomputable def fraction := (F : ℚ) / N

-- Prove that fraction = 3/8
theorem fraction_proof : fraction = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_fraction_proof_l2292_229205


namespace NUMINAMATH_GPT_initial_sum_of_money_l2292_229237

theorem initial_sum_of_money (A2 A7 : ℝ) (H1 : A2 = 520) (H2 : A7 = 820) :
  ∃ P : ℝ, P = 400 :=
by
  -- Proof starts here
  sorry

end NUMINAMATH_GPT_initial_sum_of_money_l2292_229237


namespace NUMINAMATH_GPT_gambler_received_max_2240_l2292_229258

def largest_amount_received_back (x y l : ℕ) : ℕ :=
  if 2 * l + 2 = 14 ∨ 2 * l - 2 = 14 then 
    let lost_value_1 := (6 * 100 + 8 * 20)
    let lost_value_2 := (8 * 100 + 6 * 20)
    max (3000 - lost_value_1) (3000 - lost_value_2)
  else 0

theorem gambler_received_max_2240 {x y : ℕ} (hx : 20 * x + 100 * y = 3000)
  (hl : ∃ l : ℕ, (l + (l + 2) = 14 ∨ l + (l - 2) = 14)) :
  largest_amount_received_back x y 6 = 2240 ∧ largest_amount_received_back x y 8 = 2080 := by
  sorry

end NUMINAMATH_GPT_gambler_received_max_2240_l2292_229258


namespace NUMINAMATH_GPT_infinite_series_value_l2292_229270

noncomputable def infinite_series : ℝ :=
  ∑' n, if n ≥ 2 then (n^4 + 5 * n^2 + 8 * n + 8) / (2^(n + 1) * (n^4 + 4)) else 0

theorem infinite_series_value :
  infinite_series = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_value_l2292_229270


namespace NUMINAMATH_GPT_part1_part2_l2292_229276

noncomputable def A (a : ℝ) := { x : ℝ | x^2 - a * x + a^2 - 19 = 0 }
def B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
def C := { x : ℝ | x^2 + 2 * x - 8 = 0 }

-- Proof Problem 1: Prove that if A ∩ B ≠ ∅ and A ∩ C = ∅, then a = -2
theorem part1 (a : ℝ) (h1 : (A a ∩ B) ≠ ∅) (h2 : (A a ∩ C) = ∅) : a = -2 :=
sorry

-- Proof Problem 2: Prove that if A ∩ B = A ∩ C ≠ ∅, then a = -3
theorem part2 (a : ℝ) (h1 : (A a ∩ B = A a ∩ C) ∧ (A a ∩ B) ≠ ∅) : a = -3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l2292_229276
