import Mathlib

namespace NUMINAMATH_GPT_inverse_function_l610_61002

variable (x : ℝ)

def f (x : ℝ) : ℝ := (x^(1 / 3)) + 1
def g (x : ℝ) : ℝ := (x - 1)^3

theorem inverse_function :
  ∀ x, f (g x) = x ∧ g (f x) = x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_inverse_function_l610_61002


namespace NUMINAMATH_GPT_quadratic_real_roots_l610_61087

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ∧ (∃ y : ℝ, y ≠ x ∧ k * y^2 - 2 * y - 1 = 0) → k ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l610_61087


namespace NUMINAMATH_GPT_wilfred_carrots_total_l610_61029

-- Define the number of carrots Wilfred eats each day
def tuesday_carrots := 4
def wednesday_carrots := 6
def thursday_carrots := 5

-- Define the total number of carrots eaten from Tuesday to Thursday
def total_carrots := tuesday_carrots + wednesday_carrots + thursday_carrots

-- The theorem to prove that the total number of carrots is 15
theorem wilfred_carrots_total : total_carrots = 15 := by
  sorry

end NUMINAMATH_GPT_wilfred_carrots_total_l610_61029


namespace NUMINAMATH_GPT_sum_of_solutions_l610_61081

theorem sum_of_solutions (a b : ℤ) (h₁ : a = -1) (h₂ : b = -4) (h₃ : ∀ x : ℝ, (16 - 4 * x - x^2 = 0 ↔ -x^2 - 4 * x + 16 = 0)) : 
  (-b / a) = 4 := 
by 
  rw [h₁, h₂]
  norm_num
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l610_61081


namespace NUMINAMATH_GPT_cookie_problem_l610_61091

theorem cookie_problem (n : ℕ) (M A : ℕ) 
  (hM : M = n - 7) 
  (hA : A = n - 2) 
  (h_sum : M + A < n) 
  (hM_pos : M ≥ 1) 
  (hA_pos : A ≥ 1) : 
  n = 8 := 
sorry

end NUMINAMATH_GPT_cookie_problem_l610_61091


namespace NUMINAMATH_GPT_average_of_tenths_and_thousandths_l610_61025

theorem average_of_tenths_and_thousandths :
  (0.4 + 0.005) / 2 = 0.2025 :=
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_average_of_tenths_and_thousandths_l610_61025


namespace NUMINAMATH_GPT_discount_threshold_l610_61052

-- Definitions based on given conditions
def photocopy_cost : ℝ := 0.02
def discount_percentage : ℝ := 0.25
def copies_needed_each : ℕ := 80
def total_savings : ℝ := 0.40 * 2 -- total savings for both Steve and Dennison

-- Minimum number of photocopies required to get the discount
def min_copies_for_discount : ℕ := 160

-- Lean statement to prove the minimum number of photocopies required for the discount
theorem discount_threshold :
  ∀ (x : ℕ),
  photocopy_cost * (x : ℝ) - (photocopy_cost * (1 - discount_percentage) * (x : ℝ)) * 2 = total_savings → 
  min_copies_for_discount = 160 :=
by sorry

end NUMINAMATH_GPT_discount_threshold_l610_61052


namespace NUMINAMATH_GPT_average_people_per_hour_l610_61059

-- Define the conditions
def people_moving : ℕ := 3000
def days : ℕ := 5
def hours_per_day : ℕ := 24
def total_hours : ℕ := days * hours_per_day

-- State the problem
theorem average_people_per_hour :
  people_moving / total_hours = 25 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_average_people_per_hour_l610_61059


namespace NUMINAMATH_GPT_rectangle_length_15_l610_61000

theorem rectangle_length_15
  (w l : ℝ)
  (h_ratio : 5 * w = 2 * l + 2 * w)
  (h_area : l * w = 150) :
  l = 15 :=
sorry

end NUMINAMATH_GPT_rectangle_length_15_l610_61000


namespace NUMINAMATH_GPT_three_pow_2023_mod_eleven_l610_61012

theorem three_pow_2023_mod_eleven :
  (3 ^ 2023) % 11 = 5 :=
sorry

end NUMINAMATH_GPT_three_pow_2023_mod_eleven_l610_61012


namespace NUMINAMATH_GPT_quadratic_roots_sum_square_l610_61078

theorem quadratic_roots_sum_square (u v : ℝ) 
  (h1 : u^2 - 5*u + 3 = 0) (h2 : v^2 - 5*v + 3 = 0) 
  (h3 : u ≠ v) : u^2 + v^2 + u*v = 22 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_sum_square_l610_61078


namespace NUMINAMATH_GPT_total_selling_price_of_cloth_l610_61066

theorem total_selling_price_of_cloth
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (total_meters : ℕ)
  (total_selling_price : ℕ) :
  profit_per_meter = 7 →
  cost_price_per_meter = 118 →
  total_meters = 80 →
  total_selling_price = (cost_price_per_meter + profit_per_meter) * total_meters →
  total_selling_price = 10000 :=
by
  intros h_profit h_cost h_total h_selling_price
  rw [h_profit, h_cost, h_total] at h_selling_price
  exact h_selling_price

end NUMINAMATH_GPT_total_selling_price_of_cloth_l610_61066


namespace NUMINAMATH_GPT_age_ratio_l610_61041

-- Conditions
def DeepakPresentAge := 27
def RahulAgeAfterSixYears := 42
def YearsToReach42 := 6

-- The theorem to prove the ratio of their ages
theorem age_ratio (R D : ℕ) (hR : R + YearsToReach42 = RahulAgeAfterSixYears) (hD : D = DeepakPresentAge) : R / D = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_age_ratio_l610_61041


namespace NUMINAMATH_GPT_proportion_sets_l610_61019

-- Define unit lengths for clarity
def length (n : ℕ) := n 

-- Define the sets of line segments
def setA := (length 4, length 5, length 6, length 7)
def setB := (length 3, length 4, length 5, length 8)
def setC := (length 5, length 15, length 3, length 9)
def setD := (length 8, length 4, length 1, length 3)

-- Define a condition for a set to form a proportion
def is_proportional (a b c d : ℕ) : Prop :=
  a * d = b * c

-- Main theorem: setC forms a proportion while others don't
theorem proportion_sets : is_proportional 5 15 3 9 ∧ 
                         ¬ is_proportional 4 5 6 7 ∧ 
                         ¬ is_proportional 3 4 5 8 ∧ 
                         ¬ is_proportional 8 4 1 3 := by
  sorry

end NUMINAMATH_GPT_proportion_sets_l610_61019


namespace NUMINAMATH_GPT_triangle_area_is_32_5_l610_61070

-- Define points A, B, and C
def A : ℝ × ℝ := (-3, 4)
def B : ℝ × ℝ := (1, 7)
def C : ℝ × ℝ := (4, -1)

-- Calculate the area directly using the determinant method for the area of a triangle given by coordinates
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (
    A.1 * (B.2 - C.2) +
    B.1 * (C.2 - A.2) +
    C.1 * (A.2 - B.2)
  )

-- Define the statement to be proved
theorem triangle_area_is_32_5 : area_triangle A B C = 32.5 := 
  by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_triangle_area_is_32_5_l610_61070


namespace NUMINAMATH_GPT_smallest_positive_debt_resolvable_l610_61080

theorem smallest_positive_debt_resolvable :
  ∃ p g : ℤ, 280 * p + 200 * g = 40 ∧
  ∀ k : ℤ, k > 0 → (∃ p g : ℤ, 280 * p + 200 * g = k) → 40 ≤ k :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_debt_resolvable_l610_61080


namespace NUMINAMATH_GPT_range_of_m_l610_61022

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0) → (2*x + 1 > 3) → (x > 1)) → (m ≤ 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_m_l610_61022


namespace NUMINAMATH_GPT_quotient_equivalence_l610_61082

variable (N H J : ℝ)

theorem quotient_equivalence
  (h1 : N / H = 1.2)
  (h2 : H / J = 5 / 6) :
  N / J = 1 := by
  sorry

end NUMINAMATH_GPT_quotient_equivalence_l610_61082


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l610_61051

variables (a b e : ℝ) (F1 F2 P : ℝ × ℝ)

-- The hyperbola assumption
def hyperbola : Prop := ∃ (x y : ℝ), (x, y) = P ∧ x^2 / a^2 - y^2 / b^2 = 1
-- a > 0 and b > 0
def positive_a_b : Prop := a > 0 ∧ b > 0
-- Distance between foci
def distance_foci : Prop := dist F1 F2 = 12
-- Distance PF2
def distance_p_f2 : Prop := dist P F2 = 5
-- To be proven, eccentricity of the hyperbola
def eccentricity : Prop := e = 3 / 2

theorem hyperbola_eccentricity : hyperbola a b P ∧ positive_a_b a b ∧ distance_foci F1 F2 ∧ distance_p_f2 P F2 → eccentricity e :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l610_61051


namespace NUMINAMATH_GPT_arithmetic_and_geometric_sequence_l610_61096

theorem arithmetic_and_geometric_sequence (a : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + 2) 
  (h_geom_seq : (a 2)^2 = a 0 * a 3) : 
  a 1 + a 2 = -10 := 
sorry

end NUMINAMATH_GPT_arithmetic_and_geometric_sequence_l610_61096


namespace NUMINAMATH_GPT_least_positive_integer_division_conditions_l610_61058

theorem least_positive_integer_division_conditions :
  ∃ M : ℤ, M > 0 ∧
  M % 11 = 10 ∧
  M % 12 = 11 ∧
  M % 13 = 12 ∧
  M % 14 = 13 ∧
  M = 30029 := 
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_division_conditions_l610_61058


namespace NUMINAMATH_GPT_price_reduction_proof_l610_61057

theorem price_reduction_proof (x : ℝ) : 256 * (1 - x) ^ 2 = 196 :=
sorry

end NUMINAMATH_GPT_price_reduction_proof_l610_61057


namespace NUMINAMATH_GPT_ratio_b_to_c_l610_61021

theorem ratio_b_to_c (x a b c : ℤ) 
    (h1 : x = 100 * a + 10 * b + c)
    (h2 : a > 0)
    (h3 : 999 - x = 241) : (b : ℚ) / c = 5 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_b_to_c_l610_61021


namespace NUMINAMATH_GPT_lines_parallel_l610_61033

theorem lines_parallel (a : ℝ) 
  (h₁ : (∀ x y : ℝ, ax + (a + 2) * y + 2 = 0)) 
  (h₂ : (∀ x y : ℝ, x + a * y + 1 = 0)) 
  : a = -1 :=
sorry

end NUMINAMATH_GPT_lines_parallel_l610_61033


namespace NUMINAMATH_GPT_find_constants_l610_61090

variable (x : ℝ)

theorem find_constants 
  (h : ∀ x, (6 * x^2 + 3 * x) / ((x - 4) * (x - 2)^3) = 
  (13.5 / (x - 4)) + (-27 / (x - 2)) + (-15 / (x - 2)^3)) :
  true :=
by {
  sorry
}

end NUMINAMATH_GPT_find_constants_l610_61090


namespace NUMINAMATH_GPT_cleaner_for_cat_stain_l610_61010

theorem cleaner_for_cat_stain (c : ℕ) :
  (6 * 6) + (3 * c) + (1 * 1) = 49 → c = 4 :=
by
  sorry

end NUMINAMATH_GPT_cleaner_for_cat_stain_l610_61010


namespace NUMINAMATH_GPT_frequency_number_correct_l610_61003

-- Define the sample capacity and the group frequency as constants
def sample_capacity : ℕ := 100
def group_frequency : ℝ := 0.3

-- State the theorem
theorem frequency_number_correct : sample_capacity * group_frequency = 30 := by
  -- Immediate calculation
  sorry

end NUMINAMATH_GPT_frequency_number_correct_l610_61003


namespace NUMINAMATH_GPT_interest_years_calculation_l610_61032

theorem interest_years_calculation 
  (total_sum : ℝ)
  (second_sum : ℝ)
  (interest_rate_first : ℝ)
  (interest_rate_second : ℝ)
  (time_second : ℝ)
  (interest_second : ℝ)
  (x : ℝ)
  (y : ℝ)
  (h1 : total_sum = 2795)
  (h2 : second_sum = 1720)
  (h3 : interest_rate_first = 3)
  (h4 : interest_rate_second = 5)
  (h5 : time_second = 3)
  (h6 : interest_second = (second_sum * interest_rate_second * time_second) / 100)
  (h7 : interest_second = 258)
  (h8 : x = (total_sum - second_sum))
  (h9 : (interest_rate_first * x * y) / 100 = interest_second)
  : y = 8 := sorry

end NUMINAMATH_GPT_interest_years_calculation_l610_61032


namespace NUMINAMATH_GPT_interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l610_61083

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2 - 1

theorem interval_of_monotonic_increase (x : ℝ) :
  ∃ k : ℤ, k * Real.pi - Real.pi / 3 ≤ x ∧ x ≤ k * Real.pi + Real.pi / 6 := sorry

theorem parallel_vectors_tan_x (x : ℝ) (h₁ : Real.sin x * Real.cos x - Real.sqrt 3 * Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
  Real.tan x = Real.sqrt 3 := sorry

theorem perpendicular_vectors_smallest_positive_x (x : ℝ) (h₁ : Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x * Real.cos x = 0) (h₂ : Real.cos x ≠ 0) :
 x = 5 * Real.pi / 6 := sorry

end NUMINAMATH_GPT_interval_of_monotonic_increase_parallel_vectors_tan_x_perpendicular_vectors_smallest_positive_x_l610_61083


namespace NUMINAMATH_GPT_quadratic_matches_sin_values_l610_61007

noncomputable def quadratic_function (x : ℝ) : ℝ := - (4 / (Real.pi ^ 2)) * (x ^ 2) + (4 / Real.pi) * x

theorem quadratic_matches_sin_values :
  (quadratic_function 0 = Real.sin 0) ∧
  (quadratic_function (Real.pi / 2) = Real.sin (Real.pi / 2)) ∧
  (quadratic_function Real.pi = Real.sin Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_matches_sin_values_l610_61007


namespace NUMINAMATH_GPT_combined_moles_l610_61068

def balanced_reaction (NaHCO3 HC2H3O2 H2O : ℕ) : Prop :=
  NaHCO3 + HC2H3O2 = H2O

theorem combined_moles (NaHCO3 HC2H3O2 : ℕ) 
  (h : balanced_reaction NaHCO3 HC2H3O2 3) : 
  NaHCO3 + HC2H3O2 = 6 :=
sorry

end NUMINAMATH_GPT_combined_moles_l610_61068


namespace NUMINAMATH_GPT_largest_int_less_than_100_by_7_l610_61094

theorem largest_int_less_than_100_by_7 (x : ℤ) (h1 : x = 7 * 13 + 4) (h2 : x < 100) :
  x = 95 := 
by
  sorry

end NUMINAMATH_GPT_largest_int_less_than_100_by_7_l610_61094


namespace NUMINAMATH_GPT_Benjamin_has_45_presents_l610_61001

-- Define the number of presents each person has
def Ethan_presents : ℝ := 31.5
def Alissa_presents : ℝ := Ethan_presents + 22
def Benjamin_presents : ℝ := Alissa_presents - 8.5

-- The statement we need to prove
theorem Benjamin_has_45_presents : Benjamin_presents = 45 :=
by
  -- on the last line, we type sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_Benjamin_has_45_presents_l610_61001


namespace NUMINAMATH_GPT_arc_length_of_sector_l610_61020

theorem arc_length_of_sector (r : ℝ) (θ : ℝ) (h_r : r = 2) (h_θ : θ = π / 3) :
  l = r * θ := by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l610_61020


namespace NUMINAMATH_GPT_expand_product_l610_61067

theorem expand_product (x : ℝ) : 5 * (x + 2) * (x + 6) * (x - 1) = 5 * x^3 + 35 * x^2 + 20 * x - 60 := 
by
  sorry

end NUMINAMATH_GPT_expand_product_l610_61067


namespace NUMINAMATH_GPT_tan_theta_eq_neg_two_l610_61039

theorem tan_theta_eq_neg_two (f : ℝ → ℝ) (θ : ℝ) 
  (h₁ : ∀ x, f x = Real.sin (2 * x + θ)) 
  (h₂ : ∀ x, f x + 2 * Real.cos (2 * x + θ) = -(f (-x) + 2 * Real.cos (2 * (-x) + θ))) :
  Real.tan θ = -2 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_eq_neg_two_l610_61039


namespace NUMINAMATH_GPT_count_non_empty_subsets_of_odd_numbers_greater_than_one_l610_61023

-- Condition definitions
def given_set : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def odd_numbers_greater_than_one (s : Finset ℕ) : Finset ℕ := 
  s.filter (λ x => x % 2 = 1 ∧ x > 1)

-- The problem statement
theorem count_non_empty_subsets_of_odd_numbers_greater_than_one : 
  (odd_numbers_greater_than_one given_set).powerset.card - 1 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_count_non_empty_subsets_of_odd_numbers_greater_than_one_l610_61023


namespace NUMINAMATH_GPT_remainder_division_1000_l610_61056

theorem remainder_division_1000 (x : ℕ) (hx : x > 0) (h : 100 % x = 10) : 1000 % x = 10 :=
  sorry

end NUMINAMATH_GPT_remainder_division_1000_l610_61056


namespace NUMINAMATH_GPT_scientific_notation_example_l610_61013

theorem scientific_notation_example : 0.0000037 = 3.7 * 10^(-6) :=
by
  -- We would provide the proof here.
  sorry

end NUMINAMATH_GPT_scientific_notation_example_l610_61013


namespace NUMINAMATH_GPT_quadratic_inequality_false_iff_range_of_a_l610_61016

theorem quadratic_inequality_false_iff_range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ (-1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_false_iff_range_of_a_l610_61016


namespace NUMINAMATH_GPT_points_coplanar_if_and_only_if_b_neg1_l610_61031

/-- Points (0, 0, 0), (1, b, 0), (0, 1, b), (b, 0, 1) are coplanar if and only if b = -1. --/
theorem points_coplanar_if_and_only_if_b_neg1 (a b : ℝ) :
  (∃ u v w : ℝ, (u, v, w) = (0, 0, 0) ∨ (u, v, w) = (1, b, 0) ∨ (u, v, w) = (0, 1, b) ∨ (u, v, w) = (b, 0, 1)) →
  (b = -1) :=
sorry

end NUMINAMATH_GPT_points_coplanar_if_and_only_if_b_neg1_l610_61031


namespace NUMINAMATH_GPT_principal_amount_l610_61098

theorem principal_amount (A : ℝ) (r : ℝ) (t : ℝ) (P : ℝ) : 
  A = 1120 → r = 0.05 → t = 6 → P = 1120 / (1 + 0.05 * 6) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_principal_amount_l610_61098


namespace NUMINAMATH_GPT_determine_uv_l610_61047

theorem determine_uv :
  ∃ u v : ℝ, (u = 5 / 17) ∧ (v = -31 / 17) ∧
    ((⟨3, -2⟩ : ℝ × ℝ) + u • ⟨5, 8⟩ = (⟨-1, 4⟩ : ℝ × ℝ) + v • ⟨-3, 2⟩) :=
by
  sorry

end NUMINAMATH_GPT_determine_uv_l610_61047


namespace NUMINAMATH_GPT_average_visitors_in_30_day_month_l610_61042

def average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) : ℕ :=
    let sundays := days_in_month / 7 + if days_in_month % 7 > 0 then 1 else 0
    let other_days := days_in_month - sundays
    let total_visitors := sundays * visitors_sunday + other_days * visitors_other
    total_visitors / days_in_month

theorem average_visitors_in_30_day_month 
    (visitors_sunday : ℕ) (visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) (h1 : visitors_sunday = 660) (h2 : visitors_other = 240) (h3 : days_in_month = 30) :
    average_visitors_per_day visitors_sunday visitors_other days_in_month starts_on_sunday = 296 := 
by
  sorry

end NUMINAMATH_GPT_average_visitors_in_30_day_month_l610_61042


namespace NUMINAMATH_GPT_solution_set_of_3x2_minus_7x_gt_6_l610_61079

theorem solution_set_of_3x2_minus_7x_gt_6 (x : ℝ) :
  3 * x^2 - 7 * x > 6 ↔ (x < -2 / 3 ∨ x > 3) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_3x2_minus_7x_gt_6_l610_61079


namespace NUMINAMATH_GPT_sin_double_angle_l610_61069

theorem sin_double_angle (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l610_61069


namespace NUMINAMATH_GPT_calculate_expression_l610_61014

theorem calculate_expression : 
  let a := 0.82
  let b := 0.1
  a^3 - b^3 / (a^2 + 0.082 + b^2) = 0.7201 := sorry

end NUMINAMATH_GPT_calculate_expression_l610_61014


namespace NUMINAMATH_GPT_winning_votes_calculation_l610_61095

variables (V : ℚ) (winner_votes : ℚ)

-- Conditions
def percentage_of_votes_of_winner : ℚ := 0.60 * V
def percentage_of_votes_of_loser : ℚ := 0.40 * V
def vote_difference_spec : 0.60 * V - 0.40 * V = 288 := by sorry

-- Theorem to prove
theorem winning_votes_calculation (h1 : winner_votes = 0.60 * V)
  (h2 : 0.60 * V - 0.40 * V = 288) : winner_votes = 864 :=
by
  sorry

end NUMINAMATH_GPT_winning_votes_calculation_l610_61095


namespace NUMINAMATH_GPT_volume_of_truncated_cone_l610_61061

noncomputable def surface_area_top : ℝ := 3 * Real.pi
noncomputable def surface_area_bottom : ℝ := 12 * Real.pi
noncomputable def slant_height : ℝ := 2
noncomputable def volume_cone : ℝ := 7 * Real.pi

theorem volume_of_truncated_cone :
  ∃ V : ℝ, V = volume_cone :=
sorry

end NUMINAMATH_GPT_volume_of_truncated_cone_l610_61061


namespace NUMINAMATH_GPT_sample_size_is_100_l610_61093

-- Conditions:
def scores_from_students := 100
def sampling_method := "simple random sampling"
def goal := "statistical analysis of senior three students' exam performance"

-- Problem statement:
theorem sample_size_is_100 :
  scores_from_students = 100 →
  sampling_method = "simple random sampling" →
  goal = "statistical analysis of senior three students' exam performance" →
  scores_from_students = 100 := by
sorry

end NUMINAMATH_GPT_sample_size_is_100_l610_61093


namespace NUMINAMATH_GPT_point_distance_is_pm_3_l610_61064

theorem point_distance_is_pm_3 (Q : ℝ) (h : |Q - 0| = 3) : Q = 3 ∨ Q = -3 :=
sorry

end NUMINAMATH_GPT_point_distance_is_pm_3_l610_61064


namespace NUMINAMATH_GPT_friday_profit_l610_61009

noncomputable def total_weekly_profit : ℝ := 2000
noncomputable def profit_on_monday (total : ℝ) : ℝ := total / 3
noncomputable def profit_on_tuesday (total : ℝ) : ℝ := total / 4
noncomputable def profit_on_thursday (total : ℝ) : ℝ := 0.35 * total
noncomputable def profit_on_friday (total : ℝ) : ℝ :=
  total - (profit_on_monday total + profit_on_tuesday total + profit_on_thursday total)

theorem friday_profit (total : ℝ) : profit_on_friday total = 133.33 :=
by
  sorry

end NUMINAMATH_GPT_friday_profit_l610_61009


namespace NUMINAMATH_GPT_base8_to_base10_l610_61045

theorem base8_to_base10 (n : ℕ) : n = 4 * 8^3 + 3 * 8^2 + 7 * 8^1 + 2 * 8^0 → n = 2298 :=
by 
  sorry

end NUMINAMATH_GPT_base8_to_base10_l610_61045


namespace NUMINAMATH_GPT_solution_set_of_equation_l610_61049

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem solution_set_of_equation (x : ℝ) (h : x > 0): (x^(log_base 10 x) = x^3 / 100) ↔ (x = 10 ∨ x = 100) := 
by sorry

end NUMINAMATH_GPT_solution_set_of_equation_l610_61049


namespace NUMINAMATH_GPT_tournament_games_l610_61035

theorem tournament_games (n : ℕ) (k : ℕ) (h_n : n = 30) (h_k : k = 5) : 
  (n * (n - 1) / 2) * k = 2175 := by
  sorry

end NUMINAMATH_GPT_tournament_games_l610_61035


namespace NUMINAMATH_GPT_ratio_problem_l610_61084

theorem ratio_problem (m n p q : ℚ) 
  (h1 : m / n = 12) 
  (h2 : p / n = 4) 
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_problem_l610_61084


namespace NUMINAMATH_GPT_total_hours_worked_l610_61046

variable (A B C D E T : ℝ)

theorem total_hours_worked (hA : A = 12)
  (hB : B = 1 / 3 * A)
  (hC : C = 2 * B)
  (hD : D = 1 / 2 * E)
  (hE : E = A + 3)
  (hT : T = A + B + C + D + E) : T = 46.5 :=
by
  sorry

end NUMINAMATH_GPT_total_hours_worked_l610_61046


namespace NUMINAMATH_GPT_largest_4_digit_divisible_by_50_l610_61092

-- Define the condition for a 4-digit number
def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the largest 4-digit number
def largest_4_digit : ℕ := 9999

-- Define the property that a number is exactly divisible by 50
def divisible_by_50 (n : ℕ) : Prop := n % 50 = 0

-- Main statement to be proved
theorem largest_4_digit_divisible_by_50 :
  ∃ n, is_4_digit n ∧ divisible_by_50 n ∧ ∀ m, is_4_digit m → divisible_by_50 m → m ≤ n ∧ n = 9950 :=
by
  sorry

end NUMINAMATH_GPT_largest_4_digit_divisible_by_50_l610_61092


namespace NUMINAMATH_GPT_min_p_q_sum_l610_61099

theorem min_p_q_sum (p q : ℕ) (hp : 0 < p) (hq : 0 < q) (h : 162 * p = q^3) : p + q = 54 :=
sorry

end NUMINAMATH_GPT_min_p_q_sum_l610_61099


namespace NUMINAMATH_GPT_Linda_journey_length_l610_61062

theorem Linda_journey_length : 
  (∃ x : ℝ, x = 30 + x * 1/4 + x * 1/7) → x = 840 / 17 :=
by
  sorry

end NUMINAMATH_GPT_Linda_journey_length_l610_61062


namespace NUMINAMATH_GPT_probability_function_meaningful_l610_61048

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def is_meaningful (x : ℝ) : Prop := 1 - x^2 > 0

def measure_interval (a b : ℝ) : ℝ := b - a

theorem probability_function_meaningful:
  let interval_a := -2
  let interval_b := 1
  let meaningful_a := -1
  let meaningful_b := 1
  let total_interval := measure_interval interval_a interval_b
  let meaningful_interval := measure_interval meaningful_a meaningful_b
  let P := meaningful_interval / total_interval
  (P = (2/3)) :=
by
  sorry

end NUMINAMATH_GPT_probability_function_meaningful_l610_61048


namespace NUMINAMATH_GPT_integer_solutions_to_equation_l610_61027

theorem integer_solutions_to_equation :
  ∀ (a b c : ℤ), a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_equation_l610_61027


namespace NUMINAMATH_GPT_first_month_sale_l610_61044

def sale_second_month : ℕ := 5744
def sale_third_month : ℕ := 5864
def sale_fourth_month : ℕ := 6122
def sale_fifth_month : ℕ := 6588
def sale_sixth_month : ℕ := 4916
def average_sale_six_months : ℕ := 5750

def expected_total_sales : ℕ := 6 * average_sale_six_months
def known_sales : ℕ := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month

theorem first_month_sale :
  (expected_total_sales - (known_sales + sale_sixth_month)) = 5266 :=
by
  sorry

end NUMINAMATH_GPT_first_month_sale_l610_61044


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l610_61017

-- Problem 1
def s_type_sequence (a : ℕ → ℕ) : Prop := 
∀ n ≥ 1, a (n+1) - a n > 3

theorem problem1 (a : ℕ → ℕ) (h₀ : a 1 = 4) (h₁ : a 2 = 8) 
  (h₂ : ∀ n ≥ 2, a n + a (n - 1) = 8 * n - 4) : s_type_sequence a := 
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℕ) (h₀ : ∀ n m, a (n * m) = (a n) ^ m)
  (b : ℕ → ℕ) (h₁ : ∀ n, b n = (3 * a n) / 4)
  (h₂ : s_type_sequence a)
  (h₃ : ¬ s_type_sequence b) : 
  (∀ n, a n = 2^(n+1)) ∨ (∀ n, a n = 2 * 3^(n-1)) ∨ (∀ n, a n = 5^ (n-1)) :=
sorry

-- Problem 3
theorem problem3 (c : ℕ → ℕ) 
  (h₀ : c 2 = 9)
  (h₁ : ∀ n ≥ 2, (1 / n - 1 / (n + 1)) * (2 + 1 / c n) ≤ 1 / c (n - 1) + 1 / c n 
               ∧ 1 / c (n - 1) + 1 / c n ≤ (1 / n - 1 / (n + 1)) * (2 + 1 / c (n-1))) :
  ∃ f : ℕ → ℕ, (s_type_sequence c) ∧ (∀ n, c n = (n + 1)^2) := 
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l610_61017


namespace NUMINAMATH_GPT_rectangle_area_is_1600_l610_61005

theorem rectangle_area_is_1600 (l w : ℕ) 
  (h₁ : l = 4 * w)
  (h₂ : 2 * l + 2 * w = 200) : 
  l * w = 1600 :=
by sorry

end NUMINAMATH_GPT_rectangle_area_is_1600_l610_61005


namespace NUMINAMATH_GPT_fraction_subtraction_l610_61018

theorem fraction_subtraction (x y : ℝ) (h : x ≠ y) : (x + y) / (x - y) - (2 * y) / (x - y) = 1 := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l610_61018


namespace NUMINAMATH_GPT_maximum_value_ab_l610_61060

noncomputable def g (x : ℝ) : ℝ := 2 ^ x

theorem maximum_value_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : g a * g b = 2) :
  ab ≤ (1 / 4) := sorry

end NUMINAMATH_GPT_maximum_value_ab_l610_61060


namespace NUMINAMATH_GPT_cost_of_burger_l610_61073

theorem cost_of_burger :
  ∃ (b s f : ℕ), 
    4 * b + 3 * s + f = 540 ∧
    3 * b + 2 * s + 2 * f = 580 ∧
    b = 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_burger_l610_61073


namespace NUMINAMATH_GPT_total_distance_traveled_l610_61038

noncomputable def total_distance (d v1 v2 v3 time_total : ℝ) : ℝ :=
  3 * d

theorem total_distance_traveled
  (d : ℝ)
  (v1 : ℝ := 3)
  (v2 : ℝ := 6)
  (v3 : ℝ := 9)
  (time_total : ℝ := 11 / 60)
  (h : d / v1 + d / v2 + d / v3 = time_total) :
  total_distance d v1 v2 v3 time_total = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l610_61038


namespace NUMINAMATH_GPT_smallest_n_2000_divides_a_n_l610_61028

theorem smallest_n_2000_divides_a_n (a : ℕ → ℤ) 
  (h_rec : ∀ n, n ≥ 1 → (n - 1) * a (n + 1) = (n + 1) * a n - 2 * (n - 1)) 
  (h2000 : 2000 ∣ a 1999) : 
  ∃ n, n ≥ 2 ∧ 2000 ∣ a n ∧ n = 249 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_n_2000_divides_a_n_l610_61028


namespace NUMINAMATH_GPT_small_cubes_one_face_painted_red_l610_61004

-- Definitions
def is_red_painted (cube : ℕ) : Bool := true -- representing the condition that the cube is painted red
def side_length (cube : ℕ) : ℕ := 4 -- side length of the original cube is 4 cm
def smaller_cube_side_length : ℕ := 1 -- smaller cube side length is 1 cm

-- Theorem Statement
theorem small_cubes_one_face_painted_red :
  ∀ (large_cube : ℕ), (side_length large_cube = 4) ∧ is_red_painted large_cube → 
  (∃ (number_of_cubes : ℕ), number_of_cubes = 24) :=
by
  sorry

end NUMINAMATH_GPT_small_cubes_one_face_painted_red_l610_61004


namespace NUMINAMATH_GPT_additional_teddies_per_bunny_l610_61065

theorem additional_teddies_per_bunny (teddies bunnies koala total_mascots: ℕ) 
  (h1 : teddies = 5) 
  (h2 : bunnies = 3 * teddies) 
  (h3 : koala = 1) 
  (h4 : total_mascots = 51): 
  (total_mascots - (teddies + bunnies + koala)) / bunnies = 2 := 
by 
  sorry

end NUMINAMATH_GPT_additional_teddies_per_bunny_l610_61065


namespace NUMINAMATH_GPT_ninety_seven_squared_l610_61011

theorem ninety_seven_squared : (97 * 97 = 9409) :=
by
  sorry

end NUMINAMATH_GPT_ninety_seven_squared_l610_61011


namespace NUMINAMATH_GPT_single_point_graph_d_l610_61050

theorem single_point_graph_d (d : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + d = 0 ↔ x = -1 ∧ y = 6) → d = 39 :=
by 
  sorry

end NUMINAMATH_GPT_single_point_graph_d_l610_61050


namespace NUMINAMATH_GPT_joe_lists_count_l610_61086

theorem joe_lists_count : ∃ (n : ℕ), n = 15 * 14 := sorry

end NUMINAMATH_GPT_joe_lists_count_l610_61086


namespace NUMINAMATH_GPT_sequence_general_term_l610_61054

theorem sequence_general_term (n : ℕ) : 
  (∃ (f : ℕ → ℕ), (∀ k, f k = k^2) ∧ (∀ m, f m = m^2)) :=
by
  -- Given the sequence 1, 4, 9, 16, 25, ...
  sorry

end NUMINAMATH_GPT_sequence_general_term_l610_61054


namespace NUMINAMATH_GPT_triangle_condition_l610_61071

-- Definitions based on the conditions
def angle_equal (A B C : ℝ) : Prop := A = B - C
def angle_ratio123 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ A / C = 1 / 3 ∧ B / C = 2 / 3
def pythagorean (a b c : ℝ) : Prop := a * a + b * b = c * c
def side_ratio456 (a b c : ℝ) : Prop := a / b = 4 / 5 ∧ a / c = 4 / 6 ∧ b / c = 5 / 6

-- Main hypothesis with right-angle and its conditions in different options
def is_right_triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  (angle_equal A B C → A = 90 ∨ B = 90 ∨ C = 90) ∧
  (angle_ratio123 A B C → A = 30 ∧ B = 60 ∧ C = 90) ∧
  (pythagorean a b c → true) ∧
  (side_ratio456 a b c → false) -- option D cannot confirm the triangle is right

theorem triangle_condition (A B C a b c : ℝ) : is_right_triangle A B C a b c :=
sorry

end NUMINAMATH_GPT_triangle_condition_l610_61071


namespace NUMINAMATH_GPT_at_least_one_not_lt_one_l610_61026

theorem at_least_one_not_lt_one (a b c : ℝ) (h : a + b + c = 3) : ¬ (a < 1 ∧ b < 1 ∧ c < 1) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_not_lt_one_l610_61026


namespace NUMINAMATH_GPT_segments_interior_proof_l610_61055

noncomputable def count_internal_segments (squares hexagons octagons : Nat) : Nat := 
  let vertices := (squares * 4 + hexagons * 6 + octagons * 8) / 3
  let total_segments := (vertices * (vertices - 1)) / 2
  let edges_along_faces := 3 * vertices
  (total_segments - edges_along_faces) / 2

theorem segments_interior_proof : count_internal_segments 12 8 6 = 840 := 
  by sorry

end NUMINAMATH_GPT_segments_interior_proof_l610_61055


namespace NUMINAMATH_GPT_Ali_is_8_l610_61097

open Nat

-- Definitions of the variables based on the conditions
def YusafAge (UmarAge : ℕ) : ℕ := UmarAge / 2
def AliAge (YusafAge : ℕ) : ℕ := YusafAge + 3

-- The specific given conditions
def UmarAge : ℕ := 10
def Yusaf : ℕ := YusafAge UmarAge
def Ali : ℕ := AliAge Yusaf

-- The theorem to be proved
theorem Ali_is_8 : Ali = 8 :=
by
  sorry

end NUMINAMATH_GPT_Ali_is_8_l610_61097


namespace NUMINAMATH_GPT_sin_minus_cos_value_l610_61074

theorem sin_minus_cos_value
  (α : ℝ)
  (h1 : Real.tan α = (Real.sqrt 3) / 3)
  (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin α - Real.cos α = -1/2 + Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_minus_cos_value_l610_61074


namespace NUMINAMATH_GPT_zoo_animal_difference_l610_61077

theorem zoo_animal_difference :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := 1 / 2 * (parrots + snakes)
  let zebras := elephants - 3
  monkeys - zebras = 35 :=
by
  sorry

end NUMINAMATH_GPT_zoo_animal_difference_l610_61077


namespace NUMINAMATH_GPT_quadratic_trinomial_value_at_6_l610_61015

theorem quadratic_trinomial_value_at_6 {p q : ℝ} 
  (h1 : ∃ r1 r2, r1 = q ∧ r2 = 1 + p + q ∧ r1 + r2 = -p ∧ r1 * r2 = q) : 
  (6^2 + p * 6 + q) = 31 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_value_at_6_l610_61015


namespace NUMINAMATH_GPT_geom_seq_sixth_term_l610_61024

theorem geom_seq_sixth_term (a : ℝ) (r : ℝ) (h1: a * r^3 = 512) (h2: a * r^8 = 8) : 
  a * r^5 = 128 := 
by 
  sorry

end NUMINAMATH_GPT_geom_seq_sixth_term_l610_61024


namespace NUMINAMATH_GPT_funnel_paper_area_l610_61036

theorem funnel_paper_area
  (slant_height : ℝ)
  (base_circumference : ℝ)
  (h1 : slant_height = 6)
  (h2 : base_circumference = 6 * Real.pi):
  (1 / 2) * base_circumference * slant_height = 18 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_funnel_paper_area_l610_61036


namespace NUMINAMATH_GPT_pyramid_volume_l610_61034

theorem pyramid_volume (a : ℝ) (h : a > 0) : (1 / 6) * a^3 = 1 / 6 * a^3 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_volume_l610_61034


namespace NUMINAMATH_GPT_odd_exponent_divisibility_l610_61063

theorem odd_exponent_divisibility (x y : ℤ) (k : ℕ) (h : (x^(2*k-1) + y^(2*k-1)) % (x + y) = 0) : 
  (x^(2*k+1) + y^(2*k+1)) % (x + y) = 0 :=
sorry

end NUMINAMATH_GPT_odd_exponent_divisibility_l610_61063


namespace NUMINAMATH_GPT_stamp_solutions_l610_61088

theorem stamp_solutions (n : ℕ) (h1 : ∀ (k : ℕ), k < 115 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) 
  (h2 : ¬ ∃ (a b c : ℕ), 3 * a + n * b + (n + 1) * c = 115) 
  (h3 : ∀ (k : ℕ), 116 ≤ k ∧ k ≤ 120 → ∃ (a b c : ℕ), 
  3 * a + n * b + (n + 1) * c = k) : 
  n = 59 :=
sorry

end NUMINAMATH_GPT_stamp_solutions_l610_61088


namespace NUMINAMATH_GPT_min_value_x_plus_4y_l610_61075

theorem min_value_x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
(h_cond : (1 / x) + (1 / (2 * y)) = 1) : x + 4 * y = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_x_plus_4y_l610_61075


namespace NUMINAMATH_GPT_complex_b_value_l610_61037

open Complex

theorem complex_b_value (b : ℝ) (h : (2 - b * I) / (1 + 2 * I) = (2 - 2 * b) / 5 + ((-4 - b) / 5) * I) :
  b = -2 / 3 :=
sorry

end NUMINAMATH_GPT_complex_b_value_l610_61037


namespace NUMINAMATH_GPT_jason_earns_88_dollars_l610_61006

theorem jason_earns_88_dollars (earn_after_school: ℝ) (earn_saturday: ℝ)
  (total_hours: ℝ) (saturday_hours: ℝ) (after_school_hours: ℝ) (total_earn: ℝ)
  (h1 : earn_after_school = 4.00)
  (h2 : earn_saturday = 6.00)
  (h3 : total_hours = 18)
  (h4 : saturday_hours = 8)
  (h5 : after_school_hours = total_hours - saturday_hours)
  (h6 : total_earn = after_school_hours * earn_after_school + saturday_hours * earn_saturday) :
  total_earn = 88.00 :=
by
  sorry

end NUMINAMATH_GPT_jason_earns_88_dollars_l610_61006


namespace NUMINAMATH_GPT_solve_x_squared_eq_four_x_l610_61053

theorem solve_x_squared_eq_four_x : {x : ℝ | x^2 = 4*x} = {0, 4} := 
sorry

end NUMINAMATH_GPT_solve_x_squared_eq_four_x_l610_61053


namespace NUMINAMATH_GPT_daily_expenses_increase_l610_61008

theorem daily_expenses_increase 
  (init_students : ℕ) (new_students : ℕ) (diminish_amount : ℝ) (orig_expenditure : ℝ)
  (orig_expenditure_eq : init_students = 35)
  (new_students_eq : new_students = 42)
  (diminish_amount_eq : diminish_amount = 1)
  (orig_expenditure_val : orig_expenditure = 400)
  (orig_average_expenditure : ℝ) (increase_expenditure : ℝ)
  (orig_avg_calc : orig_average_expenditure = orig_expenditure / init_students)
  (new_total_expenditure : ℝ)
  (new_expenditure_eq : new_total_expenditure = orig_expenditure + increase_expenditure) :
  (42 * (orig_average_expenditure - diminish_amount) = new_total_expenditure) → increase_expenditure = 38 := 
by 
  sorry

end NUMINAMATH_GPT_daily_expenses_increase_l610_61008


namespace NUMINAMATH_GPT_tan_ratio_l610_61040

open Real

theorem tan_ratio (x y : ℝ) (h1 : sin x / cos y + sin y / cos x = 2) (h2 : cos x / sin y + cos y / sin x = 4) : 
  tan x / tan y + tan y / tan x = 2 :=
sorry

end NUMINAMATH_GPT_tan_ratio_l610_61040


namespace NUMINAMATH_GPT_fish_ranking_l610_61085

def ranks (P V K T : ℕ) : Prop :=
  P < K ∧ K < T ∧ T < V

theorem fish_ranking (P V K T : ℕ) (h1 : K < T) (h2 : P + V = K + T) (h3 : P + T < V + K) : ranks P V K T :=
by
  sorry

end NUMINAMATH_GPT_fish_ranking_l610_61085


namespace NUMINAMATH_GPT_product_of_roots_cubic_l610_61043

theorem product_of_roots_cubic :
  ∀ (x : ℝ), (x^3 - 15 * x^2 + 75 * x - 50 = 0) →
    (∃ a b c : ℝ, x = a * b * c ∧ x = 50) :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_cubic_l610_61043


namespace NUMINAMATH_GPT_smaller_circle_radius_l610_61076

theorem smaller_circle_radius
  (R : ℝ) (r : ℝ)
  (h1 : R = 12)
  (h2 : 7 = 7) -- This is trivial and just emphasizes the arrangement of seven congruent smaller circles
  (h3 : 4 * (2 * r) = 2 * R) : r = 3 := by
  sorry

end NUMINAMATH_GPT_smaller_circle_radius_l610_61076


namespace NUMINAMATH_GPT_max_area_triangle_l610_61089

/-- Given two fixed points A and B on the plane with distance 2 between them, 
and a point P moving such that the ratio of distances |PA| / |PB| = sqrt(2), 
prove that the maximum area of triangle PAB is 2 * sqrt(2). -/
theorem max_area_triangle 
  (A B P : EuclideanSpace ℝ (Fin 2)) 
  (hAB : dist A B = 2)
  (h_ratio : dist P A = Real.sqrt 2 * dist P B)
  (h_non_collinear : ¬ ∃ k : ℝ, ∃ l : ℝ, k ≠ l ∧ A = k • B ∧ P = l • B) 
  : ∃ S_max : ℝ, S_max = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_max_area_triangle_l610_61089


namespace NUMINAMATH_GPT_equation_of_line_l610_61072

theorem equation_of_line :
  ∃ m : ℝ, ∀ x y : ℝ, (y = m * x - m ∧ (m = 2 ∧ x = 1 ∧ y = 0)) ∧ 
  ∀ x : ℝ, ¬(4 * x^2 - (m * x - m)^2 - 8 * x = 12) → m = 2 → y = 2 * x - 2 :=
by sorry

end NUMINAMATH_GPT_equation_of_line_l610_61072


namespace NUMINAMATH_GPT_find_range_of_a_l610_61030

def prop_p (a : ℝ) : Prop :=
∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0

def prop_q (a : ℝ) : Prop :=
(∃ x₁ x₂ : ℝ, x₁ * x₂ = 1 ∧ x₁ + x₂ = -(a - 1) ∧ (0 < x₁ ∧ x₁ < 1 ∧ 1 < x₂ ∧ x₂ < 2))

def range_a (a : ℝ) : Prop :=
(-2 < a ∧ a <= -3/2) ∨ (-1 <= a ∧ a <= 2)

theorem find_range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬ (prop_p a ∧ prop_q a) ↔ range_a a :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l610_61030
