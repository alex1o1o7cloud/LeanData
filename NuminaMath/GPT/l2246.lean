import Mathlib

namespace NUMINAMATH_GPT_ones_digit_of_8_pow_47_l2246_224621

theorem ones_digit_of_8_pow_47 :
  (8^47) % 10 = 2 :=
by
  sorry

end NUMINAMATH_GPT_ones_digit_of_8_pow_47_l2246_224621


namespace NUMINAMATH_GPT_infinitely_many_n_gt_sqrt_two_l2246_224660

/-- A sequence of positive integers indexed by natural numbers. -/
def a (n : ℕ) : ℕ := sorry

/-- Main theorem stating there are infinitely many n such that 1 + a_n > a_{n-1} * root n of 2. -/
theorem infinitely_many_n_gt_sqrt_two :
  ∀ (a : ℕ → ℕ), (∀ n, a n > 0) → ∃ᶠ n in at_top, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n : ℝ) :=
by {
  sorry
}

end NUMINAMATH_GPT_infinitely_many_n_gt_sqrt_two_l2246_224660


namespace NUMINAMATH_GPT_other_books_new_releases_percentage_l2246_224694

theorem other_books_new_releases_percentage
  (T : ℝ)
  (h1 : 0 < T)
  (hf_books : ℝ := 0.4 * T)
  (hf_new_releases : ℝ := 0.4 * hf_books)
  (other_books : ℝ := 0.6 * T)
  (total_new_releases : ℝ := hf_new_releases + (P * other_books))
  (fraction_hf_new : ℝ := hf_new_releases / total_new_releases)
  (fraction_value : fraction_hf_new = 0.27586206896551724)
  : P = 0.7 :=
sorry

end NUMINAMATH_GPT_other_books_new_releases_percentage_l2246_224694


namespace NUMINAMATH_GPT_unique_elements_condition_l2246_224645

theorem unique_elements_condition (x : ℝ) : 
  (1 ≠ x ∧ x ≠ x^2 ∧ 1 ≠ x^2) ↔ (x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :=
by 
  sorry

end NUMINAMATH_GPT_unique_elements_condition_l2246_224645


namespace NUMINAMATH_GPT_bobby_final_paycheck_correct_l2246_224618

def bobby_salary : ℕ := 450
def federal_tax_rate : ℚ := 1/3
def state_tax_rate : ℚ := 0.08
def health_insurance_deduction : ℕ := 50
def life_insurance_deduction : ℕ := 20
def city_parking_fee : ℕ := 10

def final_paycheck_amount : ℚ :=
  let federal_taxes := federal_tax_rate * bobby_salary
  let state_taxes := state_tax_rate * bobby_salary
  let total_deductions := federal_taxes + state_taxes + health_insurance_deduction + life_insurance_deduction + city_parking_fee
  bobby_salary - total_deductions

theorem bobby_final_paycheck_correct : final_paycheck_amount = 184 := by
  sorry

end NUMINAMATH_GPT_bobby_final_paycheck_correct_l2246_224618


namespace NUMINAMATH_GPT_range_x2y2z_range_a_inequality_l2246_224609

theorem range_x2y2z {x y z : ℝ} (h : x^2 + y^2 + z^2 = 1) : 
  -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3 :=
by sorry

theorem range_a_inequality (a : ℝ) (h : ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) :
  (4 ≤ a) ∨ (a ≤ 0) :=
by sorry

end NUMINAMATH_GPT_range_x2y2z_range_a_inequality_l2246_224609


namespace NUMINAMATH_GPT_probability_Z_l2246_224612

variable (p_X p_Y p_Z p_W : ℚ)

def conditions :=
  (p_X = 1/4) ∧ (p_Y = 1/3) ∧ (p_W = 1/6) ∧ (p_X + p_Y + p_Z + p_W = 1)

theorem probability_Z (h : conditions p_X p_Y p_Z p_W) : p_Z = 1/4 :=
by
  obtain ⟨hX, hY, hW, hSum⟩ := h
  sorry

end NUMINAMATH_GPT_probability_Z_l2246_224612


namespace NUMINAMATH_GPT_jennys_wedding_guests_l2246_224632

noncomputable def total_guests (C S : ℕ) : ℕ := C + S

theorem jennys_wedding_guests :
  ∃ (C S : ℕ), (S = 3 * C) ∧
               (18 * C + 25 * S = 1860) ∧
               (total_guests C S = 80) :=
sorry

end NUMINAMATH_GPT_jennys_wedding_guests_l2246_224632


namespace NUMINAMATH_GPT_ratio_a6_b6_l2246_224670

-- Definitions for sequences and sums
variable {α : Type*} [LinearOrderedField α] 
variable (a b : ℕ → α) 
variable (S T : ℕ → α)

-- Main theorem stating the problem
theorem ratio_a6_b6 (h : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
    a 6 / b 6 = 17 / 47 :=
sorry

end NUMINAMATH_GPT_ratio_a6_b6_l2246_224670


namespace NUMINAMATH_GPT_geometric_progression_common_ratio_l2246_224673

-- Definitions and theorems
variable {α : Type*} [OrderedCommRing α]

theorem geometric_progression_common_ratio
  (a : α) (r : α)
  (h_pos : a > 0)
  (h_geometric : ∀ n : ℕ, a * r^n = (a * r^(n + 1)) * (a * r^(n + 2))):
  r = 1 := by
  sorry

end NUMINAMATH_GPT_geometric_progression_common_ratio_l2246_224673


namespace NUMINAMATH_GPT_find_n_l2246_224662

theorem find_n (n : ℕ) (h : (17 + 98 + 39 + 54 + n) / 5 = n) : n = 52 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2246_224662


namespace NUMINAMATH_GPT_intersection_M_N_l2246_224637

noncomputable def M := {x : ℕ | x < 6}
noncomputable def N := {x : ℕ | x^2 - 11 * x + 18 < 0}
noncomputable def intersection := {x : ℕ | x ∈ M ∧ x ∈ N}

theorem intersection_M_N : intersection = {3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2246_224637


namespace NUMINAMATH_GPT_contrapositive_of_square_inequality_l2246_224601

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x^2 > y^2 → x > y) ↔ (x ≤ y → x^2 ≤ y^2) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_square_inequality_l2246_224601


namespace NUMINAMATH_GPT_find_divisor_l2246_224629

theorem find_divisor 
  (dividend : ℤ)
  (quotient : ℤ)
  (remainder : ℤ)
  (divisor : ℤ)
  (h : dividend = (divisor * quotient) + remainder)
  (h_dividend : dividend = 474232)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968) :
  divisor = 800 :=
sorry

end NUMINAMATH_GPT_find_divisor_l2246_224629


namespace NUMINAMATH_GPT_total_sum_lent_l2246_224651

-- Conditions
def interest_equal (x y : ℕ) : Prop :=
  (x * 3 * 8) / 100 = (y * 5 * 3) / 100

def second_sum : ℕ := 1704

-- Assertion
theorem total_sum_lent : ∃ x : ℕ, interest_equal x second_sum ∧ (x + second_sum = 2769) :=
  by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_total_sum_lent_l2246_224651


namespace NUMINAMATH_GPT_helicopter_rental_cost_l2246_224656

noncomputable def rentCost (hours_per_day : ℕ) (days : ℕ) (cost_per_hour : ℕ) : ℕ :=
  hours_per_day * days * cost_per_hour

theorem helicopter_rental_cost :
  rentCost 2 3 75 = 450 := 
by
  sorry

end NUMINAMATH_GPT_helicopter_rental_cost_l2246_224656


namespace NUMINAMATH_GPT_original_number_is_600_l2246_224627

theorem original_number_is_600 (x : Real) (h : x * 1.10 = 660) : x = 600 := by
  sorry

end NUMINAMATH_GPT_original_number_is_600_l2246_224627


namespace NUMINAMATH_GPT_a_plus_b_l2246_224644

-- Definitions and conditions
def f (x : ℝ) (a b : ℝ) := a * x + b
def g (x : ℝ) := 3 * x - 7

theorem a_plus_b (a b : ℝ) (h : ∀ x : ℝ, g (f x a b) = 4 * x + 5) : a + b = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_b_l2246_224644


namespace NUMINAMATH_GPT_swap_numbers_l2246_224674

theorem swap_numbers (a b : ℕ) (hc: b = 17) (ha : a = 8) : 
  ∃ c, c = b ∧ b = a ∧ a = c := 
by
  sorry

end NUMINAMATH_GPT_swap_numbers_l2246_224674


namespace NUMINAMATH_GPT_sum_is_eight_l2246_224689

theorem sum_is_eight (a b c d : ℤ)
  (h1 : 2 * (a - b + c) = 10)
  (h2 : 2 * (b - c + d) = 12)
  (h3 : 2 * (c - d + a) = 6)
  (h4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 :=
by
  sorry

end NUMINAMATH_GPT_sum_is_eight_l2246_224689


namespace NUMINAMATH_GPT_base8_to_base10_4513_l2246_224647

theorem base8_to_base10_4513 : (4 * 8^3 + 5 * 8^2 + 1 * 8^1 + 3 * 8^0 = 2379) :=
by
  sorry

end NUMINAMATH_GPT_base8_to_base10_4513_l2246_224647


namespace NUMINAMATH_GPT_S_n_min_at_5_min_nS_n_is_neg_49_l2246_224636

variable {S_n : ℕ → ℝ}
variable {a_1 d : ℝ}

-- Conditions
axiom sum_first_n_terms (n : ℕ) : S_n n = n / 2 * (2 * a_1 + (n - 1) * d)

axiom S_10 : S_n 10 = 0
axiom S_15 : S_n 15 = 25

-- Proving the following statements
theorem S_n_min_at_5 :
  (∀ n, S_n n ≥ S_n 5) :=
sorry

theorem min_nS_n_is_neg_49 :
  (∀ n, n * S_n n ≥ -49) :=
sorry

end NUMINAMATH_GPT_S_n_min_at_5_min_nS_n_is_neg_49_l2246_224636


namespace NUMINAMATH_GPT_managers_non_managers_ratio_l2246_224603

theorem managers_non_managers_ratio
  (M N : ℕ)
  (h_ratio : M / N > 7 / 24)
  (h_max_non_managers : N = 27) :
  ∃ M, 8 ≤ M ∧ M / 27 > 7 / 24 :=
by
  sorry

end NUMINAMATH_GPT_managers_non_managers_ratio_l2246_224603


namespace NUMINAMATH_GPT_a_n_divisible_by_2013_a_n_minus_207_is_cube_l2246_224634

theorem a_n_divisible_by_2013 (n : ℕ) (h : n ≥ 1) : 2013 ∣ (4 ^ (6 ^ n) + 1943) :=
by sorry

theorem a_n_minus_207_is_cube (n : ℕ) : (∃ k : ℕ, 4 ^ (6 ^ n) + 1736 = k^3) ↔ (n = 1) :=
by sorry

end NUMINAMATH_GPT_a_n_divisible_by_2013_a_n_minus_207_is_cube_l2246_224634


namespace NUMINAMATH_GPT_mark_owe_triple_amount_l2246_224628

theorem mark_owe_triple_amount (P : ℝ) (r : ℝ) (t : ℕ) (hP : P = 2000) (hr : r = 0.04) :
  (1 + r)^t > 3 → t = 30 :=
by
  intro h
  norm_cast at h
  sorry

end NUMINAMATH_GPT_mark_owe_triple_amount_l2246_224628


namespace NUMINAMATH_GPT_radius_ratio_of_spheres_l2246_224695

theorem radius_ratio_of_spheres
  (V_large : ℝ) (V_small : ℝ) (r_large r_small : ℝ)
  (h1 : V_large = 324 * π)
  (h2 : V_small = 0.25 * V_large)
  (h3 : (4/3) * π * r_large^3 = V_large)
  (h4 : (4/3) * π * r_small^3 = V_small) :
  (r_small / r_large) = (1/2) := 
sorry

end NUMINAMATH_GPT_radius_ratio_of_spheres_l2246_224695


namespace NUMINAMATH_GPT_two_numbers_sum_l2246_224623

theorem two_numbers_sum (N1 N2 : ℕ) (h1 : N1 % 10^5 = 0) (h2 : N2 % 10^5 = 0) 
  (h3 : N1 ≠ N2) (h4 : (Nat.divisors N1).card = 42) (h5 : (Nat.divisors N2).card = 42) : 
  N1 + N2 = 700000 := 
by
  sorry

end NUMINAMATH_GPT_two_numbers_sum_l2246_224623


namespace NUMINAMATH_GPT_solution_l2246_224643

theorem solution (x : ℝ) (h : ¬ (x ^ 2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_solution_l2246_224643


namespace NUMINAMATH_GPT_common_chord_length_l2246_224642

/-- Two circles intersect such that each passes through the other's center.
Prove that the length of their common chord is 8√3 cm. -/
theorem common_chord_length (r : ℝ) (h : r = 8) :
  let chord_length := 2 * (r * (Real.sqrt 3 / 2))
  chord_length = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_common_chord_length_l2246_224642


namespace NUMINAMATH_GPT_remainder_of_3056_mod_32_l2246_224605

theorem remainder_of_3056_mod_32 : 3056 % 32 = 16 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3056_mod_32_l2246_224605


namespace NUMINAMATH_GPT_groupA_forms_triangle_l2246_224638

theorem groupA_forms_triangle (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 20) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
by {
  sorry
}

end NUMINAMATH_GPT_groupA_forms_triangle_l2246_224638


namespace NUMINAMATH_GPT_range_of_a_l2246_224680

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l2246_224680


namespace NUMINAMATH_GPT_local_food_drive_correct_l2246_224693

def local_food_drive_condition1 (R J x : ℕ) : Prop :=
  J = 2 * R + x

def local_food_drive_condition2 (J : ℕ) : Prop :=
  4 * J = 100

def local_food_drive_condition3 (R J : ℕ) : Prop :=
  R + J = 35

theorem local_food_drive_correct (R J x : ℕ)
  (h1 : local_food_drive_condition1 R J x)
  (h2 : local_food_drive_condition2 J)
  (h3 : local_food_drive_condition3 R J) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_local_food_drive_correct_l2246_224693


namespace NUMINAMATH_GPT_minimum_max_abs_x2_sub_2xy_l2246_224600

theorem minimum_max_abs_x2_sub_2xy {y : ℝ} :
  ∃ y : ℝ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y) ≥ 0) ∧
           (∀ y' ∈ Set.univ, (∀ x ∈ (Set.Icc 0 1), abs (x^2 - 2*x*y') ≥ abs (x^2 - 2*x*y))) :=
sorry

end NUMINAMATH_GPT_minimum_max_abs_x2_sub_2xy_l2246_224600


namespace NUMINAMATH_GPT_geo_sequence_arithmetic_l2246_224610

variable {d : ℝ} (hd : d ≠ 0)
variable {a : ℕ → ℝ} (ha : ∀ n, a (n+1) = a n + d)

-- Hypothesis that a_5, a_9, a_15 form a geometric sequence
variable (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d))

theorem geo_sequence_arithmetic (hd : d ≠ 0) (ha : ∀ n, a (n + 1) = a n + d) (hgeo : a 9 ^ 2 = (a 9 - 4 * d) * (a 9 + 6 * d)) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geo_sequence_arithmetic_l2246_224610


namespace NUMINAMATH_GPT_ratio_of_a_b_l2246_224682

-- Define the system of equations as given in the problem
variables (x y a b : ℝ)

-- Conditions: the system of equations and b ≠ 0
def system_of_equations (a b : ℝ) (x y : ℝ) := 
  4 * x - 3 * y = a ∧ 6 * y - 8 * x = b

-- The theorem we aim to prove
theorem ratio_of_a_b (h : system_of_equations a b x y) (h₀ : b ≠ 0) : a / b = -1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_a_b_l2246_224682


namespace NUMINAMATH_GPT_grapes_average_seeds_l2246_224696

def total_seeds_needed : ℕ := 60
def apple_seed_average : ℕ := 6
def pear_seed_average : ℕ := 2
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def extra_seeds_needed : ℕ := 3

-- Calculation of total seeds from apples and pears:
def seeds_from_apples : ℕ := apples_count * apple_seed_average
def seeds_from_pears : ℕ := pears_count * pear_seed_average

def total_seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculation of the remaining seeds needed from grapes:
def seeds_needed_from_grapes : ℕ := total_seeds_needed - total_seeds_from_apples_and_pears - extra_seeds_needed

-- Calculation of the average number of seeds per grape:
def grape_seed_average : ℕ := seeds_needed_from_grapes / grapes_count

-- Prove the correct average number of seeds per grape:
theorem grapes_average_seeds : grape_seed_average = 3 :=
by
  sorry

end NUMINAMATH_GPT_grapes_average_seeds_l2246_224696


namespace NUMINAMATH_GPT_avg_visitors_on_sunday_l2246_224617

theorem avg_visitors_on_sunday (S : ℕ) :
  (30 * 285) = (5 * S + 25 * 240) -> S = 510 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_avg_visitors_on_sunday_l2246_224617


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_holds_l2246_224697

noncomputable def necessary_and_sufficient_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + m > 0

theorem necessary_and_sufficient_condition_holds (m : ℝ) :
  necessary_and_sufficient_condition m ↔ m > 1 :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_holds_l2246_224697


namespace NUMINAMATH_GPT_fraction_meaningful_cond_l2246_224672

theorem fraction_meaningful_cond (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) := 
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_cond_l2246_224672


namespace NUMINAMATH_GPT_quadratic_roots_new_equation_l2246_224604

theorem quadratic_roots_new_equation (a b c x1 x2 : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : x1 + x2 = -b / a) 
  (h3 : x1 * x2 = c / a) : 
  ∃ (a' b' c' : ℝ), a' * x^2 + b' * x + c' = 0 ∧ a' = a^2 ∧ b' = 3 * a * b ∧ c' = 2 * b^2 + a * c :=
sorry

end NUMINAMATH_GPT_quadratic_roots_new_equation_l2246_224604


namespace NUMINAMATH_GPT_smart_charging_piles_eq_l2246_224678

theorem smart_charging_piles_eq (x : ℝ) :
  301 * (1 + x) ^ 2 = 500 :=
by sorry

end NUMINAMATH_GPT_smart_charging_piles_eq_l2246_224678


namespace NUMINAMATH_GPT_chord_length_3pi_4_chord_bisected_by_P0_l2246_224675

open Real

-- Define conditions and the problem.
def Circle := {p : ℝ × ℝ // p.1^2 + p.2^2 = 8}
def P0 : ℝ × ℝ := (-1, 2)

-- Proving the first part (1)
theorem chord_length_3pi_4 (α : ℝ) (hα : α = 3 * π / 4) (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  dist A B = sqrt 30 := sorry

-- Proving the second part (2)
theorem chord_bisected_by_P0 (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  ∃ k : ℝ, (B.2 - A.2) = k * (B.1 - A.1) ∧ k = 1 / 2 ∧
  (k * (x - (-1))) = y - 2 := sorry

end NUMINAMATH_GPT_chord_length_3pi_4_chord_bisected_by_P0_l2246_224675


namespace NUMINAMATH_GPT_average_income_QR_l2246_224685

theorem average_income_QR (P Q R : ℝ) 
  (h1: (P + Q) / 2 = 5050) 
  (h2: (P + R) / 2 = 5200) 
  (hP: P = 4000) : 
  (Q + R) / 2 = 6250 := 
by 
  -- additional steps and proof to be provided here
  sorry

end NUMINAMATH_GPT_average_income_QR_l2246_224685


namespace NUMINAMATH_GPT_find_n_in_geometric_series_l2246_224677

theorem find_n_in_geometric_series (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = 126 →
  S n = a 1 * (2^n - 1) / (2 - 1) →
  n = 6 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_find_n_in_geometric_series_l2246_224677


namespace NUMINAMATH_GPT_find_fx2_l2246_224614

theorem find_fx2 (f : ℝ → ℝ) (x : ℝ) (h : f (x - 1) = x ^ 2) : f (x ^ 2) = (x ^ 2 + 1) ^ 2 := by
  sorry

end NUMINAMATH_GPT_find_fx2_l2246_224614


namespace NUMINAMATH_GPT_Lyka_savings_l2246_224688

def Smartphone_cost := 800
def Initial_savings := 200
def Gym_cost_per_month := 50
def Total_months := 4
def Weeks_per_month := 4
def Savings_per_week_initial := 50
def Savings_per_week_after_raise := 80

def Total_savings : Nat :=
  let initial_savings := Savings_per_week_initial * Weeks_per_month * 2
  let increased_savings := Savings_per_week_after_raise * Weeks_per_month * 2
  initial_savings + increased_savings

theorem Lyka_savings :
  (Initial_savings + Total_savings) = 1040 := by
  sorry

end NUMINAMATH_GPT_Lyka_savings_l2246_224688


namespace NUMINAMATH_GPT_distance_they_both_run_l2246_224613

theorem distance_they_both_run
  (D : ℝ)
  (A_time : D / 28 = A_speed)
  (B_time : D / 32 = B_speed)
  (A_beats_B : A_speed * 28 = B_speed * 28 + 16) :
  D = 128 := 
sorry

end NUMINAMATH_GPT_distance_they_both_run_l2246_224613


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l2246_224653

noncomputable def trigonometric_expression : ℝ := 
  (Real.sin (15 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) 
  + Real.cos (165 * Real.pi / 180) * Real.cos (115 * Real.pi / 180)) /
  (Real.sin (35 * Real.pi / 180) * Real.cos (5 * Real.pi / 180) 
  + Real.cos (145 * Real.pi / 180) * Real.cos (85 * Real.pi / 180))

theorem trigonometric_identity_proof : trigonometric_expression = 1 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l2246_224653


namespace NUMINAMATH_GPT_percentage_tax_proof_l2246_224692

theorem percentage_tax_proof (total_worth tax_free cost taxable tax_rate tax_value percentage_sales_tax : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_free = 34.7)
  (h3 : tax_rate = 0.06)
  (h4 : total_worth = taxable + tax_rate * taxable + tax_free)
  (h5 : tax_value = tax_rate * taxable)
  (h6 : percentage_sales_tax = (tax_value / total_worth) * 100) :
  percentage_sales_tax = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_percentage_tax_proof_l2246_224692


namespace NUMINAMATH_GPT_arithmetic_mean_of_numbers_l2246_224619

theorem arithmetic_mean_of_numbers (n : ℕ) (h : n > 1) :
  let one_special_number := (1 / n) + (2 / n ^ 2)
  let other_numbers := (n - 1) * 1
  (other_numbers + one_special_number) / n = 1 + 2 / n ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_numbers_l2246_224619


namespace NUMINAMATH_GPT_solve_for_x_l2246_224668

theorem solve_for_x (x : ℝ) (h : 3 * x - 8 = 4 * x + 5) : x = -13 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l2246_224668


namespace NUMINAMATH_GPT_bowling_average_change_l2246_224669

theorem bowling_average_change (old_avg : ℝ) (wickets_last : ℕ) (runs_last : ℕ) (wickets_before : ℕ)
  (h_old_avg : old_avg = 12.4)
  (h_wickets_last : wickets_last = 8)
  (h_runs_last : runs_last = 26)
  (h_wickets_before : wickets_before = 175) :
  old_avg - ((old_avg * wickets_before + runs_last)/(wickets_before + wickets_last)) = 0.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_bowling_average_change_l2246_224669


namespace NUMINAMATH_GPT_six_digit_numbers_with_at_least_two_zeros_l2246_224690

noncomputable def num_six_digit_numbers_with_at_least_two_zeros : ℕ :=
  73314

theorem six_digit_numbers_with_at_least_two_zeros :
  ∃ n : ℕ, n = num_six_digit_numbers_with_at_least_two_zeros := by
  use 73314
  sorry

end NUMINAMATH_GPT_six_digit_numbers_with_at_least_two_zeros_l2246_224690


namespace NUMINAMATH_GPT_tower_remainder_l2246_224691

def num_towers : ℕ := 907200  -- the total number of different towers S for 9 cubes

theorem tower_remainder : num_towers % 1000 = 200 :=
by
  sorry

end NUMINAMATH_GPT_tower_remainder_l2246_224691


namespace NUMINAMATH_GPT_teacher_students_and_ticket_cost_l2246_224607

theorem teacher_students_and_ticket_cost 
    (C_s C_a : ℝ) 
    (n_k n_h : ℕ)
    (hk_total ht_total : ℝ) 
    (h_students : n_h = n_k + 3)
    (hk  : n_k * C_s + C_a = hk_total)
    (ht : n_h * C_s + C_a = ht_total)
    (hk_total_val : hk_total = 994)
    (ht_total_val : ht_total = 1120)
    (C_s_val : C_s = 42) : 
    (n_h = 25) ∧ (C_a = 70) := 
by
  -- Proof steps would be provided here
  sorry

end NUMINAMATH_GPT_teacher_students_and_ticket_cost_l2246_224607


namespace NUMINAMATH_GPT_two_digit_number_l2246_224639

theorem two_digit_number (x y : ℕ) (h1 : y = x + 4) (h2 : (10 * x + y) * (x + y) = 208) :
  10 * x + y = 26 :=
sorry

end NUMINAMATH_GPT_two_digit_number_l2246_224639


namespace NUMINAMATH_GPT_number_of_red_balls_l2246_224661

def total_balls : ℕ := 50
def frequency_red_ball : ℝ := 0.7

theorem number_of_red_balls :
  ∃ n : ℕ, n = (total_balls : ℝ) * frequency_red_ball ∧ n = 35 :=
by
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l2246_224661


namespace NUMINAMATH_GPT_compound_interest_amount_l2246_224608

theorem compound_interest_amount (P r t SI : ℝ) (h1 : t = 3) (h2 : r = 0.10) (h3 : SI = 900) :
  SI = P * r * t → P = 900 / (0.10 * 3) → (P * (1 + r)^t - P = 993) :=
by
  intros hSI hP
  sorry

end NUMINAMATH_GPT_compound_interest_amount_l2246_224608


namespace NUMINAMATH_GPT_siblings_age_problem_l2246_224655

variable {x y z : ℕ}

theorem siblings_age_problem
  (h1 : x - y = 3)
  (h2 : z - 1 = 2 * (x + y))
  (h3 : z + 20 = x + y + 40) :
  x = 11 ∧ y = 8 ∧ z = 39 :=
by
  sorry

end NUMINAMATH_GPT_siblings_age_problem_l2246_224655


namespace NUMINAMATH_GPT_factory_dolls_per_day_l2246_224611

-- Define the number of normal dolls made per day
def N : ℝ := 4800

-- Define the total number of dolls made per day as 1.33 times the number of normal dolls
def T : ℝ := 1.33 * N

-- The theorem statement to prove the factory makes 6384 dolls per day
theorem factory_dolls_per_day : T = 6384 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_factory_dolls_per_day_l2246_224611


namespace NUMINAMATH_GPT_avg_choc_pieces_per_cookie_l2246_224658

theorem avg_choc_pieces_per_cookie {cookies chips mms pieces : ℕ} 
  (h1 : cookies = 48) 
  (h2 : chips = 108) 
  (h3 : mms = chips / 3) 
  (h4 : pieces = chips + mms) : 
  pieces / cookies = 3 := 
by sorry

end NUMINAMATH_GPT_avg_choc_pieces_per_cookie_l2246_224658


namespace NUMINAMATH_GPT_sum_abs_values_l2246_224652

theorem sum_abs_values (a b : ℝ) (h₁ : abs a = 4) (h₂ : abs b = 7) (h₃ : a < b) : a + b = 3 ∨ a + b = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_abs_values_l2246_224652


namespace NUMINAMATH_GPT_remaining_to_original_ratio_l2246_224681

-- Define the number of rows and production per row for corn and potatoes.
def rows_of_corn : ℕ := 10
def corn_per_row : ℕ := 9
def rows_of_potatoes : ℕ := 5
def potatoes_per_row : ℕ := 30

-- Define the remaining crops after pest destruction.
def remaining_crops : ℕ := 120

-- Calculate the original number of crops from corn and potato productions.
def original_crops : ℕ :=
  (rows_of_corn * corn_per_row) + (rows_of_potatoes * potatoes_per_row)

-- Define the ratio of remaining crops to original crops.
def crops_ratio : ℚ := remaining_crops / original_crops

theorem remaining_to_original_ratio : crops_ratio = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_remaining_to_original_ratio_l2246_224681


namespace NUMINAMATH_GPT_root_exists_between_0_and_1_l2246_224626

theorem root_exists_between_0_and_1 (a b c : ℝ) (m : ℝ) (hm : 0 < m)
  (h : a / (m + 2) + b / (m + 1) + c / m = 0) : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ a * x ^ 2 + b * x + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_root_exists_between_0_and_1_l2246_224626


namespace NUMINAMATH_GPT_total_legs_correct_l2246_224622

-- Number of animals
def horses : ℕ := 2
def dogs : ℕ := 5
def cats : ℕ := 7
def turtles : ℕ := 3
def goat : ℕ := 1

-- Total number of animals
def total_animals : ℕ := horses + dogs + cats + turtles + goat

-- Total number of legs
def total_legs : ℕ := total_animals * 4

theorem total_legs_correct : total_legs = 72 := by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_total_legs_correct_l2246_224622


namespace NUMINAMATH_GPT_point_on_right_branch_l2246_224684

noncomputable def on_hyperbola_right_branch (a b m : ℝ) :=
  (∀ a b m : ℝ, (a - 2 * b > 0) → (a + 2 * b > 0) → (a ^ 2 - 4 * b ^ 2 = m) → (m ≠ 0) → a > 0)

theorem point_on_right_branch (a b m : ℝ) (h₁ : a - 2 * b > 0) (h₂ : a + 2 * b > 0) (h₃ : a ^ 2 - 4 * b ^ 2 = m) (h₄ : m ≠ 0) :
  a > 0 := 
by 
  sorry

end NUMINAMATH_GPT_point_on_right_branch_l2246_224684


namespace NUMINAMATH_GPT_number_of_storks_joined_l2246_224650

theorem number_of_storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (total_birds_and_storks : ℕ) 
    (h1 : initial_birds = 3) (h2 : initial_storks = 4) (h3 : total_birds_and_storks = 13) : 
    (total_birds_and_storks - (initial_birds + initial_storks)) = 6 := 
by
  sorry

end NUMINAMATH_GPT_number_of_storks_joined_l2246_224650


namespace NUMINAMATH_GPT_quadratic_single_root_pos_value_l2246_224654

theorem quadratic_single_root_pos_value (m : ℝ) (h1 : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
sorry

end NUMINAMATH_GPT_quadratic_single_root_pos_value_l2246_224654


namespace NUMINAMATH_GPT_jail_time_calculation_l2246_224648

def total_arrests (arrests_per_day : ℕ) (cities : ℕ) (days : ℕ) : ℕ := 
  arrests_per_day * cities * days

def jail_time_before_trial (arrests : ℕ) (days_before_trial : ℕ) : ℕ := 
  days_before_trial * arrests

def jail_time_after_trial (arrests : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_after_trial * arrests

def combined_jail_time (weeks_before_trial : ℕ) (weeks_after_trial : ℕ) : ℕ := 
  weeks_before_trial + weeks_after_trial

noncomputable def total_jail_time_in_weeks : ℕ := 
  let arrests := total_arrests 10 21 30
  let weeks_before_trial := jail_time_before_trial arrests 4 / 7
  let weeks_after_trial := jail_time_after_trial arrests 1
  combined_jail_time weeks_before_trial weeks_after_trial

theorem jail_time_calculation : 
  total_jail_time_in_weeks = 9900 :=
sorry

end NUMINAMATH_GPT_jail_time_calculation_l2246_224648


namespace NUMINAMATH_GPT_optimal_play_winner_l2246_224606

theorem optimal_play_winner (n : ℕ) (h : n > 1) : (n % 2 = 0) ↔ (first_player_wins: Bool) :=
  sorry

end NUMINAMATH_GPT_optimal_play_winner_l2246_224606


namespace NUMINAMATH_GPT_cyclic_sum_non_negative_equality_condition_l2246_224624

theorem cyclic_sum_non_negative (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) ≥ 0 :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (a - b) / (a + b) + b^2 * (b - c) / (b + c) + c^2 * (c - a) / (c + a) = 0 ↔ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_cyclic_sum_non_negative_equality_condition_l2246_224624


namespace NUMINAMATH_GPT_solution_set_circle_l2246_224671

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end NUMINAMATH_GPT_solution_set_circle_l2246_224671


namespace NUMINAMATH_GPT_siblings_total_weight_l2246_224630

/-- Given conditions:
Antonio's weight: 50 kilograms.
Antonio's sister weighs 12 kilograms less than Antonio.
Antonio's backpack weight: 5 kilograms.
Antonio's sister's backpack weight: 3 kilograms.
Marco's weight: 30 kilograms.
Marco's stuffed animal weight: 2 kilograms.
Prove that the total weight of the three siblings including additional weights is 128 kilograms.
-/
theorem siblings_total_weight :
  let antonio_weight := 50
  let antonio_sister_weight := antonio_weight - 12
  let antonio_backpack_weight := 5
  let antonio_sister_backpack_weight := 3
  let marco_weight := 30
  let marco_stuffed_animal_weight := 2
  antonio_weight + antonio_backpack_weight +
  antonio_sister_weight + antonio_sister_backpack_weight +
  marco_weight + marco_stuffed_animal_weight = 128 :=
by
  sorry

end NUMINAMATH_GPT_siblings_total_weight_l2246_224630


namespace NUMINAMATH_GPT_product_of_slope_and_intercept_l2246_224687

theorem product_of_slope_and_intercept {x1 y1 x2 y2 : ℝ} (h1 : x1 = -4) (h2 : y1 = -2) (h3 : x2 = 1) (h4 : y2 = 3) :
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m * b = 2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_slope_and_intercept_l2246_224687


namespace NUMINAMATH_GPT_least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l2246_224616

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def product_of_digits (n : ℕ) : ℕ :=
  (digits n).foldl (λ x y => x * y) 1

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k, n = m * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of 45 n ∧ is_multiple_of 9 (product_of_digits n)

theorem least_positive_multiple_of_45_with_product_of_digits_multiple_of_9 : 
  ∀ n, satisfies_conditions n → 495 ≤ n :=
by
  sorry

end NUMINAMATH_GPT_least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l2246_224616


namespace NUMINAMATH_GPT_john_total_fuel_usage_l2246_224679

def city_fuel_rate := 6 -- liters per km for city traffic
def highway_fuel_rate := 4 -- liters per km for highway traffic

def trip1_city_distance := 50 -- km for Trip 1
def trip2_highway_distance := 35 -- km for Trip 2
def trip3_city_distance := 15 -- km for Trip 3 in city traffic
def trip3_highway_distance := 10 -- km for Trip 3 on highway

-- Define the total fuel consumption
def total_fuel_used : Nat :=
  (trip1_city_distance * city_fuel_rate) +
  (trip2_highway_distance * highway_fuel_rate) +
  (trip3_city_distance * city_fuel_rate) +
  (trip3_highway_distance * highway_fuel_rate)

theorem john_total_fuel_usage :
  total_fuel_used = 570 :=
by
  sorry

end NUMINAMATH_GPT_john_total_fuel_usage_l2246_224679


namespace NUMINAMATH_GPT_max_empty_squares_l2246_224625

theorem max_empty_squares (board_size : ℕ) (total_cells : ℕ) 
  (initial_cockroaches : ℕ) (adjacent : ℕ → ℕ → Prop) 
  (different : ℕ → ℕ → Prop) :
  board_size = 8 → total_cells = 64 → initial_cockroaches = 2 →
  (∀ s : ℕ, s < total_cells → ∃ s1 s2 : ℕ, adjacent s s1 ∧ 
              adjacent s s2 ∧ 
              different s1 s2) →
  ∃ max_empty_cells : ℕ, max_empty_cells = 24 :=
by
  intros h_board_size h_total_cells h_initial_cockroaches h_moves
  sorry

end NUMINAMATH_GPT_max_empty_squares_l2246_224625


namespace NUMINAMATH_GPT_min_xy_min_x_plus_y_l2246_224635

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : xy ≥ 4 := sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : x + y ≥ 9 / 2 := sorry

end NUMINAMATH_GPT_min_xy_min_x_plus_y_l2246_224635


namespace NUMINAMATH_GPT_sarah_gave_away_16_apples_to_teachers_l2246_224683

def initial_apples : Nat := 25
def apples_given_to_friends : Nat := 5
def apples_eaten : Nat := 1
def apples_left_after_journey : Nat := 3

theorem sarah_gave_away_16_apples_to_teachers :
  let apples_after_giving_to_friends := initial_apples - apples_given_to_friends
  let apples_after_eating := apples_after_giving_to_friends - apples_eaten
  apples_after_eating - apples_left_after_journey = 16 :=
by
  sorry

end NUMINAMATH_GPT_sarah_gave_away_16_apples_to_teachers_l2246_224683


namespace NUMINAMATH_GPT_find_a_l2246_224646

theorem find_a (a : ℝ) (h : ∀ B: ℝ × ℝ, (B = (a, 0)) → (2 - 0) * (0 - 2) = (4 - 2) * (2 - a)) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2246_224646


namespace NUMINAMATH_GPT_purely_imaginary_iff_x_equals_one_l2246_224615

theorem purely_imaginary_iff_x_equals_one (x : ℝ) :
  ((x^2 - 1) + (x + 1) * Complex.I).re = 0 → x = 1 :=
by
  sorry

end NUMINAMATH_GPT_purely_imaginary_iff_x_equals_one_l2246_224615


namespace NUMINAMATH_GPT_sum_of_three_integers_mod_53_l2246_224686

theorem sum_of_three_integers_mod_53 (a b c : ℕ) (h1 : a % 53 = 31) 
                                     (h2 : b % 53 = 22) (h3 : c % 53 = 7) : 
                                     (a + b + c) % 53 = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_integers_mod_53_l2246_224686


namespace NUMINAMATH_GPT_raj_is_older_than_ravi_l2246_224664

theorem raj_is_older_than_ravi
  (R V H L x : ℕ)
  (h1 : R = V + x)
  (h2 : H = V - 2)
  (h3 : R = 3 * L)
  (h4 : H * 2 = 3 * L)
  (h5 : 20 = (4 * H) / 3) :
  x = 13 :=
by
  sorry

end NUMINAMATH_GPT_raj_is_older_than_ravi_l2246_224664


namespace NUMINAMATH_GPT_sector_perimeter_l2246_224663

theorem sector_perimeter (A θ r: ℝ) (hA : A = 2) (hθ : θ = 4) (hArea : A = (1/2) * r^2 * θ) : (2 * r + r * θ) = 6 :=
by 
  sorry

end NUMINAMATH_GPT_sector_perimeter_l2246_224663


namespace NUMINAMATH_GPT_cannot_take_value_l2246_224699

theorem cannot_take_value (x y : ℝ) (h : |x| + |y| = 13) : 
  ∀ (v : ℝ), x^2 + 7*x - 3*y + y^2 = v → (0 ≤ v ∧ v ≤ 260) := 
by
  sorry

end NUMINAMATH_GPT_cannot_take_value_l2246_224699


namespace NUMINAMATH_GPT_concert_total_revenue_l2246_224640

def adult_ticket_price : ℕ := 26
def child_ticket_price : ℕ := adult_ticket_price / 2
def num_adults : ℕ := 183
def num_children : ℕ := 28

def revenue_from_adults : ℕ := num_adults * adult_ticket_price
def revenue_from_children : ℕ := num_children * child_ticket_price
def total_revenue : ℕ := revenue_from_adults + revenue_from_children

theorem concert_total_revenue :
  total_revenue = 5122 :=
by
  -- proof can be filled in here
  sorry

end NUMINAMATH_GPT_concert_total_revenue_l2246_224640


namespace NUMINAMATH_GPT_total_coins_l2246_224620

-- Definitions for the conditions
def number_of_nickels := 13
def number_of_quarters := 8

-- Statement of the proof problem
theorem total_coins : number_of_nickels + number_of_quarters = 21 :=
by
  sorry

end NUMINAMATH_GPT_total_coins_l2246_224620


namespace NUMINAMATH_GPT_geometric_sequence_a5_l2246_224698

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) 
  : a 5 = -8 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l2246_224698


namespace NUMINAMATH_GPT_melanie_total_value_l2246_224602

-- Define the initial number of dimes Melanie had
def initial_dimes : ℕ := 7

-- Define the number of dimes given by her dad
def dimes_from_dad : ℕ := 8

-- Define the number of dimes given by her mom
def dimes_from_mom : ℕ := 4

-- Calculate the total number of dimes Melanie has now
def total_dimes : ℕ := initial_dimes + dimes_from_dad + dimes_from_mom

-- Define the value of each dime in dollars
def value_per_dime : ℝ := 0.10

-- Calculate the total value of dimes in dollars
def total_value_in_dollars : ℝ := total_dimes * value_per_dime

-- The theorem states that the total value in dollars is 1.90
theorem melanie_total_value : total_value_in_dollars = 1.90 := 
by
  -- Using the established definitions, the goal follows directly.
  sorry

end NUMINAMATH_GPT_melanie_total_value_l2246_224602


namespace NUMINAMATH_GPT_total_pawns_left_l2246_224659

  -- Definitions of initial conditions
  def initial_pawns_in_chess : Nat := 8
  def kennedy_pawns_lost : Nat := 4
  def riley_pawns_lost : Nat := 1

  -- Theorem statement to prove the total number of pawns left
  theorem total_pawns_left : (initial_pawns_in_chess - kennedy_pawns_lost) + (initial_pawns_in_chess - riley_pawns_lost) = 11 := by
    sorry
  
end NUMINAMATH_GPT_total_pawns_left_l2246_224659


namespace NUMINAMATH_GPT_factorize_x2_add_2x_sub_3_l2246_224665

theorem factorize_x2_add_2x_sub_3 :
  (x^2 + 2 * x - 3) = (x + 3) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x2_add_2x_sub_3_l2246_224665


namespace NUMINAMATH_GPT_angle_PQRS_l2246_224631

theorem angle_PQRS (P Q R S : ℝ) (h1 : P = 3 * Q) (h2 : P = 4 * R) (h3 : P = 6 * S) (h4 : P + Q + R + S = 360) : 
  P = 206 := 
by
  sorry

end NUMINAMATH_GPT_angle_PQRS_l2246_224631


namespace NUMINAMATH_GPT_prob_top_odd_correct_l2246_224667

def total_dots : Nat := 78
def faces : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Probability calculation for odd dots after removal
def prob_odd_dot (n : Nat) : Rat :=
  if n % 2 = 1 then
    1 - (n : Rat) / total_dots
  else
    (n : Rat) / total_dots

-- Probability that the top face shows an odd number of dots
noncomputable def prob_top_odd : Rat :=
  (1 / (faces.length : Rat)) * (faces.map prob_odd_dot).sum

theorem prob_top_odd_correct :
  prob_top_odd = 523 / 936 :=
by
  sorry

end NUMINAMATH_GPT_prob_top_odd_correct_l2246_224667


namespace NUMINAMATH_GPT_right_triangle_roots_l2246_224649

theorem right_triangle_roots (m a b c : ℝ) 
  (h_eq : ∀ x, x^2 - (2 * m + 1) * x + m^2 + m = 0)
  (h_roots : a^2 - (2 * m + 1) * a + m^2 + m = 0 ∧ b^2 - (2 * m + 1) * b + m^2 + m = 0)
  (h_triangle : a^2 + b^2 = c^2)
  (h_c : c = 5) : 
  m = 3 :=
by sorry

end NUMINAMATH_GPT_right_triangle_roots_l2246_224649


namespace NUMINAMATH_GPT_customers_added_l2246_224641

theorem customers_added (x : ℕ) (h : 29 + x = 49) : x = 20 := by
  sorry

end NUMINAMATH_GPT_customers_added_l2246_224641


namespace NUMINAMATH_GPT_daily_rental_cost_l2246_224666

def daily_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) : ℝ :=
  x + miles * cost_per_mile

theorem daily_rental_cost (x : ℝ) (miles : ℝ) (cost_per_mile : ℝ) (total_budget : ℝ) 
  (h : daily_cost x miles cost_per_mile = total_budget) : x = 30 :=
by
  let constant_miles := 200
  let constant_cost_per_mile := 0.23
  let constant_budget := 76
  sorry

end NUMINAMATH_GPT_daily_rental_cost_l2246_224666


namespace NUMINAMATH_GPT_division_theorem_l2246_224676

theorem division_theorem (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end NUMINAMATH_GPT_division_theorem_l2246_224676


namespace NUMINAMATH_GPT_norm_photos_l2246_224657

-- Define variables for the number of photos taken by Lisa, Mike, and Norm.
variables {L M N : ℕ}

-- Define the given conditions as hypotheses.
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := N = 2 * L + 10

-- State the problem in Lean: we want to prove that the number of photos Norm took is 110.
theorem norm_photos (L M N : ℕ) (h1 : condition1 L M N) (h2 : condition2 L N) : N = 110 :=
by
  sorry

end NUMINAMATH_GPT_norm_photos_l2246_224657


namespace NUMINAMATH_GPT_jordan_travel_distance_heavy_traffic_l2246_224633

theorem jordan_travel_distance_heavy_traffic (x : ℝ) (h1 : x / 20 + x / 10 + x / 6 = 7 / 6) : 
  x = 3.7 :=
by
  sorry

end NUMINAMATH_GPT_jordan_travel_distance_heavy_traffic_l2246_224633
