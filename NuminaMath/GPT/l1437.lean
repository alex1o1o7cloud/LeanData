import Mathlib

namespace NUMINAMATH_GPT_three_digit_sum_permutations_l1437_143767

theorem three_digit_sum_permutations (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9) (h₃ : 1 ≤ b) (h₄ : b ≤ 9) (h₅ : 1 ≤ c) (h₆ : c ≤ 9)
  (h₇ : n = 100 * a + 10 * b + c)
  (h₈ : 222 * (a + b + c) - n = 1990) :
  n = 452 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_sum_permutations_l1437_143767


namespace NUMINAMATH_GPT_false_proposition_l1437_143795

-- Definitions of the conditions
def p1 := ∃ x0 : ℝ, x0^2 - 2*x0 + 1 ≤ 0
def p2 := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - 1 ≥ 0

-- Statement to prove
theorem false_proposition : ¬ (¬ p1 ∧ ¬ p2) :=
by sorry

end NUMINAMATH_GPT_false_proposition_l1437_143795


namespace NUMINAMATH_GPT_sum_first_10_terms_arith_seq_l1437_143720

theorem sum_first_10_terms_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 3 = 5)
  (h2 : a 7 = 13)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S 10 = 100 :=
sorry

end NUMINAMATH_GPT_sum_first_10_terms_arith_seq_l1437_143720


namespace NUMINAMATH_GPT_compare_squares_l1437_143703

theorem compare_squares : -6 * Real.sqrt 5 < -5 * Real.sqrt 6 := sorry

end NUMINAMATH_GPT_compare_squares_l1437_143703


namespace NUMINAMATH_GPT_find_inverse_value_l1437_143709

noncomputable def f (x : ℝ) : ℝ := sorry -- f(x) function definition goes here

theorem find_inverse_value :
  (∀ x : ℝ, f (x - 1) = f (x + 3)) →
  (∀ x : ℝ, f (-x) = f x) →
  (∀ x : ℝ, 4 ≤ x ∧ x ≤ 6 → f x = 2^x + 1) →
  f⁻¹ 19 = 3 - 2 * (Real.log 3 / Real.log 2) :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_inverse_value_l1437_143709


namespace NUMINAMATH_GPT_x_y_differ_by_one_l1437_143793

theorem x_y_differ_by_one (x y : ℚ) (h : (1 + y) / (x - y) = x) : y = x - 1 :=
by
sorry

end NUMINAMATH_GPT_x_y_differ_by_one_l1437_143793


namespace NUMINAMATH_GPT_triangle_A_and_Area_l1437_143754

theorem triangle_A_and_Area :
  ∀ (a b c A B C : ℝ), 
  (b - (1 / 2) * c = a * Real.cos C) 
  → (4 * (b + c) = 3 * b * c) 
  → (a = 2 * Real.sqrt 3)
  → (A = 60) ∧ (1/2 * b * c * Real.sin A = 2 * Real.sqrt 3) :=
by
  intros a b c A B C h1 h2 h3
  sorry

end NUMINAMATH_GPT_triangle_A_and_Area_l1437_143754


namespace NUMINAMATH_GPT_product_of_faces_and_vertices_of_cube_l1437_143732

def number_of_faces := 6
def number_of_vertices := 8

theorem product_of_faces_and_vertices_of_cube : number_of_faces * number_of_vertices = 48 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_faces_and_vertices_of_cube_l1437_143732


namespace NUMINAMATH_GPT_matrix_power_sub_l1437_143714

section 
variable (A : Matrix (Fin 2) (Fin 2) ℝ)
variable (hA : A = ![![2, 3], ![0, 1]])

theorem matrix_power_sub (A : Matrix (Fin 2) (Fin 2) ℝ)
  (h: A = ![![2, 3], ![0, 1]]) :
  A ^ 20 - 2 * A ^ 19 = ![![0, 3], ![0, -1]] :=
by
  sorry
end

end NUMINAMATH_GPT_matrix_power_sub_l1437_143714


namespace NUMINAMATH_GPT_tan_420_eq_sqrt3_l1437_143779

theorem tan_420_eq_sqrt3 : Real.tan (420 * Real.pi / 180) = Real.sqrt 3 := 
by 
  -- Additional mathematical justification can go here.
  sorry

end NUMINAMATH_GPT_tan_420_eq_sqrt3_l1437_143779


namespace NUMINAMATH_GPT_apple_price_l1437_143750

theorem apple_price :
  ∀ (l q : ℝ), 
    (10 * l = 3.62) →
    (30 * l + 3 * q = 11.67) →
    (30 * l + 6 * q = 12.48) :=
by
  intros l q h₁ h₂
  -- The proof would go here with the steps, but for now we use sorry.
  sorry

end NUMINAMATH_GPT_apple_price_l1437_143750


namespace NUMINAMATH_GPT_largest_prime_factor_of_expression_l1437_143730

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Prime p ∧ p > 35 ∧ p > 2 ∧ p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ ∀ q, Prime q ∧ q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_factor_of_expression_l1437_143730


namespace NUMINAMATH_GPT_probability_of_selected_cubes_l1437_143749

-- Total number of unit cubes
def total_unit_cubes : ℕ := 125

-- Number of cubes with exactly two blue faces (from edges not corners)
def two_blue_faces : ℕ := 9

-- Number of unpainted unit cubes
def unpainted_cubes : ℕ := 51

-- Calculate total combinations of choosing 2 cubes out of 125
def total_combinations : ℕ := Nat.choose total_unit_cubes 2

-- Calculate favorable outcomes: one cube with 2 blue faces and one unpainted cube
def favorable_outcomes : ℕ := two_blue_faces * unpainted_cubes

-- Calculate probability
def probability : ℚ := favorable_outcomes / total_combinations

-- The theorem we want to prove
theorem probability_of_selected_cubes :
  probability = 3 / 50 :=
by
  -- Show that the probability indeed equals 3/50
  sorry

end NUMINAMATH_GPT_probability_of_selected_cubes_l1437_143749


namespace NUMINAMATH_GPT_toys_gained_l1437_143775

theorem toys_gained
  (sp : ℕ) -- selling price of 18 toys
  (cp_per_toy : ℕ) -- cost price per toy
  (sp_val : sp = 27300) -- given selling price value
  (cp_per_val : cp_per_toy = 1300) -- given cost price per toy value
  : (sp - 18 * cp_per_toy) / cp_per_toy = 3 := by
  -- Conditions of the problem are stated
  -- Proof is omitted with 'sorry'
  sorry

end NUMINAMATH_GPT_toys_gained_l1437_143775


namespace NUMINAMATH_GPT_lowest_price_for_16_oz_butter_l1437_143786

-- Define the constants
def price_single_16_oz_package : ℝ := 7
def price_8_oz_package : ℝ := 4
def price_4_oz_package : ℝ := 2
def discount_4_oz_package : ℝ := 0.5

-- Calculate the discounted price for a 4 oz package
def discounted_price_4_oz_package : ℝ := price_4_oz_package * discount_4_oz_package

-- Calculate the total price for two discounted 4 oz packages
def total_price_two_discounted_4_oz_packages : ℝ := 2 * discounted_price_4_oz_package

-- Calculate the total price using the 8 oz package and two discounted 4 oz packages
def total_price_using_coupon : ℝ := price_8_oz_package + total_price_two_discounted_4_oz_packages

-- State the property to prove
theorem lowest_price_for_16_oz_butter :
  min price_single_16_oz_package total_price_using_coupon = 6 :=
sorry

end NUMINAMATH_GPT_lowest_price_for_16_oz_butter_l1437_143786


namespace NUMINAMATH_GPT_jasmine_percent_after_addition_l1437_143772

-- Variables definition based on the problem
def original_volume : ℕ := 90
def original_jasmine_percent : ℚ := 0.05
def added_jasmine : ℕ := 8
def added_water : ℕ := 2

-- Total jasmine amount calculation in original solution
def original_jasmine_amount : ℚ := original_jasmine_percent * original_volume

-- New total jasmine amount after addition
def new_jasmine_amount : ℚ := original_jasmine_amount + added_jasmine

-- New total volume calculation after addition
def new_total_volume : ℕ := original_volume + added_jasmine + added_water

-- New jasmine percent in the solution
def new_jasmine_percent : ℚ := (new_jasmine_amount / new_total_volume) * 100

-- The proof statement
theorem jasmine_percent_after_addition : new_jasmine_percent = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_jasmine_percent_after_addition_l1437_143772


namespace NUMINAMATH_GPT_total_canoes_built_l1437_143766

theorem total_canoes_built (boats_jan : ℕ) (h : boats_jan = 5)
    (boats_feb : ℕ) (h1 : boats_feb = boats_jan * 3)
    (boats_mar : ℕ) (h2 : boats_mar = boats_feb * 3)
    (boats_apr : ℕ) (h3 : boats_apr = boats_mar * 3) :
  boats_jan + boats_feb + boats_mar + boats_apr = 200 :=
sorry

end NUMINAMATH_GPT_total_canoes_built_l1437_143766


namespace NUMINAMATH_GPT_ratio_area_octagons_correct_l1437_143724

noncomputable def ratio_area_octagons (r : ℝ) : ℝ :=
  let s := 2 * r * Real.sin (Real.pi / 8)   -- side length of inscribed octagon
  let S := 2 * r * Real.cos (Real.pi / 8)   -- side length of circumscribed octagon
  let ratio_side_lengths := S / s
  let ratio_areas := (ratio_side_lengths)^2
  ratio_areas

theorem ratio_area_octagons_correct (r : ℝ) :
  ratio_area_octagons r = 6 + 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_area_octagons_correct_l1437_143724


namespace NUMINAMATH_GPT_increase_by_percentage_l1437_143719

theorem increase_by_percentage (x : ℝ) (y : ℝ): x = 90 → y = 0.50 → x + x * y = 135 := 
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_increase_by_percentage_l1437_143719


namespace NUMINAMATH_GPT_orchestra_french_horn_players_l1437_143783

open Nat

theorem orchestra_french_horn_players :
  ∃ (french_horn_players : ℕ), 
  french_horn_players = 1 ∧
  1 + 6 + 5 + 7 + 1 + french_horn_players = 21 :=
by
  sorry

end NUMINAMATH_GPT_orchestra_french_horn_players_l1437_143783


namespace NUMINAMATH_GPT_extreme_value_at_1_l1437_143710

theorem extreme_value_at_1 (a b : ℝ) (h1 : (deriv (λ x => x^3 + a * x^2 + b * x + a^2) 1 = 0))
(h2 : (1 + a + b + a^2 = 10)) : a + b = -7 := by
  sorry

end NUMINAMATH_GPT_extreme_value_at_1_l1437_143710


namespace NUMINAMATH_GPT_product_of_consecutive_integers_between_sqrt_29_l1437_143715

-- Define that \(5 \lt \sqrt{29} \lt 6\)
lemma sqrt_29_bounds : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 :=
sorry

-- Main theorem statement
theorem product_of_consecutive_integers_between_sqrt_29 :
  (∃ (a b : ℤ), 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 ∧ a = 5 ∧ b = 6 ∧ a * b = 30) := 
sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_between_sqrt_29_l1437_143715


namespace NUMINAMATH_GPT_quadratic_inequality_real_solution_l1437_143764

theorem quadratic_inequality_real_solution (a : ℝ) :
  (∃ x : ℝ, 2*x^2 + (a-1)*x + 1/2 ≤ 0) ↔ (a ≤ -1 ∨ 3 ≤ a) := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_real_solution_l1437_143764


namespace NUMINAMATH_GPT_a_minus_c_value_l1437_143763

theorem a_minus_c_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 150) : 
  a - c = -80 := 
by 
  -- We provide the proof inline with sorry
  sorry

end NUMINAMATH_GPT_a_minus_c_value_l1437_143763


namespace NUMINAMATH_GPT_daughterAgeThreeYearsFromNow_l1437_143705

-- Definitions of constants and conditions
def motherAgeNow := 41
def motherAgeFiveYearsAgo := motherAgeNow - 5
def daughterAgeFiveYearsAgo := motherAgeFiveYearsAgo / 2
def daughterAgeNow := daughterAgeFiveYearsAgo + 5
def daughterAgeInThreeYears := daughterAgeNow + 3

-- Theorem to prove the daughter's age in 3 years given conditions
theorem daughterAgeThreeYearsFromNow :
  motherAgeNow = 41 →
  motherAgeFiveYearsAgo = 2 * daughterAgeFiveYearsAgo →
  daughterAgeInThreeYears = 26 :=
by
  intros h1 h2
  -- Original Lean would have the proof steps here
  sorry

end NUMINAMATH_GPT_daughterAgeThreeYearsFromNow_l1437_143705


namespace NUMINAMATH_GPT_one_fourths_in_one_eighth_l1437_143780

theorem one_fourths_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_one_fourths_in_one_eighth_l1437_143780


namespace NUMINAMATH_GPT_coin_toss_probability_l1437_143702

theorem coin_toss_probability :
  (∀ n : ℕ, 0 ≤ n → n ≤ 10 → (∀ m : ℕ, 0 ≤ m → m = 10 → 
  (∀ k : ℕ, k = 9 → 
  (∀ i : ℕ, 0 ≤ i → i = 10 → ∃ p : ℝ, p = 1/2 → 
  (∃ q : ℝ, q = 1/2 → q = p))))) := 
sorry

end NUMINAMATH_GPT_coin_toss_probability_l1437_143702


namespace NUMINAMATH_GPT_fraction_value_eq_l1437_143770

theorem fraction_value_eq : (5 * 8) / 10 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_value_eq_l1437_143770


namespace NUMINAMATH_GPT_percentage_of_silver_in_final_solution_l1437_143752

noncomputable section -- because we deal with real numbers and division

variable (volume_4pct : ℝ) (percentage_4pct : ℝ)
variable (volume_10pct : ℝ) (percentage_10pct : ℝ)

def final_percentage_silver (v4 : ℝ) (p4 : ℝ) (v10 : ℝ) (p10 : ℝ) : ℝ :=
  let total_silver := v4 * p4 + v10 * p10
  let total_volume := v4 + v10
  (total_silver / total_volume) * 100

theorem percentage_of_silver_in_final_solution :
  final_percentage_silver 5 0.04 2.5 0.10 = 6 := by
  sorry

end NUMINAMATH_GPT_percentage_of_silver_in_final_solution_l1437_143752


namespace NUMINAMATH_GPT_tan_eq_one_over_three_l1437_143748

theorem tan_eq_one_over_three (x : ℝ) (h1 : x ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.cos (2 * x - (Real.pi / 2)) = Real.sin x ^ 2) :
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_eq_one_over_three_l1437_143748


namespace NUMINAMATH_GPT_value_is_200_l1437_143746

variable (x value : ℝ)
variable (h1 : 0.20 * x = value)
variable (h2 : 1.20 * x = 1200)

theorem value_is_200 : value = 200 :=
by
  sorry

end NUMINAMATH_GPT_value_is_200_l1437_143746


namespace NUMINAMATH_GPT_can_pay_without_change_l1437_143721

theorem can_pay_without_change (n : ℕ) (h : n > 7) :
  ∃ (a b : ℕ), 3 * a + 5 * b = n :=
sorry

end NUMINAMATH_GPT_can_pay_without_change_l1437_143721


namespace NUMINAMATH_GPT_limit_C_of_f_is_2_l1437_143727

variable {f : ℝ → ℝ}
variable {x₀ : ℝ}
variable {f' : ℝ}

noncomputable def differentiable_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ f' : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs (f (x + h) - f x - f' * h) / abs (h) < ε

axiom hf_differentiable : differentiable_at f x₀
axiom f'_at_x₀ : f' = 1

theorem limit_C_of_f_is_2 
  (hf_differentiable : differentiable_at f x₀) 
  (h_f'_at_x₀ : f' = 1) : 
  (∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (x₀ + 2 * Δx) - f x₀) / Δx - 2) < ε) :=
sorry

end NUMINAMATH_GPT_limit_C_of_f_is_2_l1437_143727


namespace NUMINAMATH_GPT_kevin_birth_year_l1437_143776

theorem kevin_birth_year (year_first_amc: ℕ) (annual: ∀ n, year_first_amc + n = year_first_amc + n) (age_tenth_amc: ℕ) (year_tenth_amc: ℕ) (year_kevin_took_amc: ℕ) 
  (h_first_amc: year_first_amc = 1988) (h_age_tenth_amc: age_tenth_amc = 13) (h_tenth_amc: year_tenth_amc = year_first_amc + 9) (h_kevin_took_amc: year_kevin_took_amc = year_tenth_amc) :
  year_kevin_took_amc - age_tenth_amc = 1984 :=
by
  sorry

end NUMINAMATH_GPT_kevin_birth_year_l1437_143776


namespace NUMINAMATH_GPT_intersection_points_l1437_143706

-- Define the line equation
def line (x : ℝ) : ℝ := 2 * x - 1

-- Problem statement to be proven
theorem intersection_points :
  (line 0.5 = 0) ∧ (line 0 = -1) :=
by 
  sorry

end NUMINAMATH_GPT_intersection_points_l1437_143706


namespace NUMINAMATH_GPT_divisible_by_12_for_all_integral_n_l1437_143745

theorem divisible_by_12_for_all_integral_n (n : ℤ) : 12 ∣ (2 * n ^ 3 - 2 * n) :=
sorry

end NUMINAMATH_GPT_divisible_by_12_for_all_integral_n_l1437_143745


namespace NUMINAMATH_GPT_operation_star_correct_l1437_143704

def op_table (i j : ℕ) : ℕ :=
  if i = 1 then
    if j = 1 then 4 else if j = 2 then 1 else if j = 3 then 2 else if j = 4 then 3 else 0
  else if i = 2 then
    if j = 1 then 1 else if j = 2 then 3 else if j = 3 then 4 else if j = 4 then 2 else 0
  else if i = 3 then
    if j = 1 then 2 else if j = 2 then 4 else if j = 3 then 1 else if j = 4 then 3 else 0
  else if i = 4 then
    if j = 1 then 3 else if j = 2 then 2 else if j = 3 then 3 else if j = 4 then 4 else 0
  else 0

theorem operation_star_correct : op_table (op_table 3 1) (op_table 4 2) = 3 :=
  by sorry

end NUMINAMATH_GPT_operation_star_correct_l1437_143704


namespace NUMINAMATH_GPT_isabel_earned_l1437_143797

theorem isabel_earned :
  let bead_necklace_price := 4
  let gemstone_necklace_price := 8
  let bead_necklace_count := 3
  let gemstone_necklace_count := 3
  let sales_tax_rate := 0.05
  let discount_rate := 0.10

  let total_cost_before_tax := bead_necklace_count * bead_necklace_price + gemstone_necklace_count * gemstone_necklace_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  let discount := total_cost_after_tax * discount_rate
  let final_amount_earned := total_cost_after_tax - discount

  final_amount_earned = 34.02 :=
by {
  sorry
}

end NUMINAMATH_GPT_isabel_earned_l1437_143797


namespace NUMINAMATH_GPT_sum_f_1_2021_l1437_143792

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom equation_f : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom interval_f : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f x = Real.log (1 - x) / Real.log 2

theorem sum_f_1_2021 : (List.sum (List.map f (List.range' 1 2021))) = -1 := sorry

end NUMINAMATH_GPT_sum_f_1_2021_l1437_143792


namespace NUMINAMATH_GPT_total_puppies_is_74_l1437_143787

-- Define the number of puppies adopted each week based on the given conditions
def number_of_puppies_first_week : Nat := 20
def number_of_puppies_second_week : Nat := 2 * number_of_puppies_first_week / 5
def number_of_puppies_third_week : Nat := 2 * number_of_puppies_second_week
def number_of_puppies_fourth_week : Nat := number_of_puppies_first_week + 10

-- Define the total number of puppies
def total_number_of_puppies : Nat :=
  number_of_puppies_first_week + number_of_puppies_second_week + number_of_puppies_third_week + number_of_puppies_fourth_week

-- Proof statement: Prove that the total number of puppies is 74
theorem total_puppies_is_74 : total_number_of_puppies = 74 := by
  sorry

end NUMINAMATH_GPT_total_puppies_is_74_l1437_143787


namespace NUMINAMATH_GPT_weight_of_purple_ring_l1437_143768

noncomputable section

def orange_ring_weight : ℝ := 0.08333333333333333
def white_ring_weight : ℝ := 0.4166666666666667
def total_weight : ℝ := 0.8333333333

theorem weight_of_purple_ring :
  total_weight - orange_ring_weight - white_ring_weight = 0.3333333333 :=
by
  -- We'll place the statement here, leave out the proof for skipping.
  sorry

end NUMINAMATH_GPT_weight_of_purple_ring_l1437_143768


namespace NUMINAMATH_GPT_binary_multiplication_binary_result_l1437_143798

-- Definitions for binary numbers
def bin_11011 : ℕ := 27 -- 11011 in binary is 27 in decimal
def bin_101 : ℕ := 5 -- 101 in binary is 5 in decimal

-- Theorem statement to prove the product of two binary numbers
theorem binary_multiplication : (bin_11011 * bin_101) = 135 := by
  sorry

-- Convert the result back to binary, expected to be 10000111
theorem binary_result : 135 = 8 * 16 + 7 := by
  sorry

end NUMINAMATH_GPT_binary_multiplication_binary_result_l1437_143798


namespace NUMINAMATH_GPT_find_a_l1437_143712

noncomputable def f (x : ℝ) : ℝ := x^2 + 12
noncomputable def g (x : ℝ) : ℝ := x^2 - x - 4

theorem find_a (a : ℝ) (h_pos : a > 0) (h_fga : f (g a) = 12) : a = (1 + Real.sqrt 17) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1437_143712


namespace NUMINAMATH_GPT_machine_production_time_l1437_143751

theorem machine_production_time (x : ℝ) 
  (h1 : 60 / x + 2 = 12) : 
  x = 6 :=
sorry

end NUMINAMATH_GPT_machine_production_time_l1437_143751


namespace NUMINAMATH_GPT_initial_liquid_A_amount_l1437_143747

noncomputable def initial_amount_of_A (x : ℚ) : ℚ :=
  3 * x

theorem initial_liquid_A_amount {x : ℚ} (h : (3 * x - 3) / (2 * x + 3) = 3 / 5) : initial_amount_of_A (8 / 3) = 8 := by
  sorry

end NUMINAMATH_GPT_initial_liquid_A_amount_l1437_143747


namespace NUMINAMATH_GPT_probability_queen_then_spade_l1437_143707

-- Define the size of the deck and the quantities for specific cards
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_spades : ℕ := 13

-- Define the probability calculation problem
theorem probability_queen_then_spade :
  (num_queens / deck_size : ℚ) * ((num_spades - 1) / (deck_size - 1) : ℚ) + ((num_queens - 1) / deck_size : ℚ) * (num_spades / (deck_size - 1) : ℚ) = 1 / deck_size :=
by sorry

end NUMINAMATH_GPT_probability_queen_then_spade_l1437_143707


namespace NUMINAMATH_GPT_fraction_identity_l1437_143738

theorem fraction_identity (a b c : ℕ) (h : (a : ℚ) / (36 - a) + (b : ℚ) / (48 - b) + (c : ℚ) / (72 - c) = 9) : 
  4 / (36 - a) + 6 / (48 - b) + 9 / (72 - c) = 13 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_identity_l1437_143738


namespace NUMINAMATH_GPT_find_selling_price_l1437_143789

variable (SP CP : ℝ)

def original_selling_price (SP CP : ℝ) : Prop :=
  0.9 * SP = CP + 0.08 * CP

theorem find_selling_price (h1 : CP = 17500)
  (h2 : original_selling_price SP CP) : SP = 21000 :=
by
  sorry

end NUMINAMATH_GPT_find_selling_price_l1437_143789


namespace NUMINAMATH_GPT_probability_one_from_harold_and_one_from_marilyn_l1437_143726

-- Define the names and the number of letters in each name
def harold_name_length := 6
def marilyn_name_length := 7

-- Total cards
def total_cards := harold_name_length + marilyn_name_length

-- Probability of drawing one card from Harold's name and one from Marilyn's name
theorem probability_one_from_harold_and_one_from_marilyn :
    (harold_name_length : ℚ) / total_cards * marilyn_name_length / (total_cards - 1) +
    marilyn_name_length / total_cards * harold_name_length / (total_cards - 1) 
    = 7 / 13 := 
by
  sorry

end NUMINAMATH_GPT_probability_one_from_harold_and_one_from_marilyn_l1437_143726


namespace NUMINAMATH_GPT_line_through_points_l1437_143734

theorem line_through_points (x1 y1 x2 y2 : ℝ) :
  (3 * x1 - 4 * y1 - 2 = 0) →
  (3 * x2 - 4 * y2 - 2 = 0) →
  (∀ x y : ℝ, (x = x1) → (y = y1) ∨ (x = x2) → (y = y2) → 3 * x - 4 * y - 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_l1437_143734


namespace NUMINAMATH_GPT_gymnastics_performance_participation_l1437_143744

def total_people_in_gym_performance (grades : ℕ) (classes_per_grade : ℕ) (students_per_class : ℕ) : ℕ :=
  grades * classes_per_grade * students_per_class

theorem gymnastics_performance_participation :
  total_people_in_gym_performance 3 4 15 = 180 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_gymnastics_performance_participation_l1437_143744


namespace NUMINAMATH_GPT_num_ways_to_select_five_crayons_including_red_l1437_143743

noncomputable def num_ways_select_five_crayons (total_crayons : ℕ) (selected_crayons : ℕ) (fixed_red_crayon : ℕ) : ℕ :=
  Nat.choose (total_crayons - fixed_red_crayon) selected_crayons

theorem num_ways_to_select_five_crayons_including_red
  (total_crayons : ℕ) 
  (fixed_red_crayon : ℕ)
  (selected_crayons : ℕ)
  (h1 : total_crayons = 15)
  (h2 : fixed_red_crayon = 1)
  (h3 : selected_crayons = 4) : 
  num_ways_select_five_crayons total_crayons selected_crayons fixed_red_crayon = 1001 := by
  sorry

end NUMINAMATH_GPT_num_ways_to_select_five_crayons_including_red_l1437_143743


namespace NUMINAMATH_GPT_mandarin_ducks_total_l1437_143778

theorem mandarin_ducks_total : (3 * 2) = 6 := by
  sorry

end NUMINAMATH_GPT_mandarin_ducks_total_l1437_143778


namespace NUMINAMATH_GPT_problem1_problem2_l1437_143784

-- Definitions of the polynomials A and B
def A (x y : ℝ) := x^2 + x * y + 3 * y
def B (x y : ℝ) := x^2 - x * y

-- Problem 1 Statement: 
theorem problem1 (x y : ℝ) (h : (x - 2)^2 + |y + 5| = 0) : 2 * (A x y) - (B x y) = -56 := by
  sorry

-- Problem 2 Statement:
theorem problem2 (x : ℝ) (h : ∀ y, 2 * (A x y) - (B x y) = 0) : x = -2 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1437_143784


namespace NUMINAMATH_GPT_find_ordered_pair_l1437_143788

open Polynomial

theorem find_ordered_pair (a b : ℝ) :
  (∀ x : ℝ, (((x^3 + a * x^2 + 17 * x + 10 = 0) ∧ (x^3 + b * x^2 + 20 * x + 12 = 0)) → 
  (x = -6 ∧ y = -7))) :=
sorry

end NUMINAMATH_GPT_find_ordered_pair_l1437_143788


namespace NUMINAMATH_GPT_tan_sum_angle_identity_l1437_143753

theorem tan_sum_angle_identity
  (α β : ℝ)
  (h1 : Real.tan (α + 2 * β) = 2)
  (h2 : Real.tan β = -3) :
  Real.tan (α + β) = -1 := sorry

end NUMINAMATH_GPT_tan_sum_angle_identity_l1437_143753


namespace NUMINAMATH_GPT_polygon_sides_l1437_143737

theorem polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 140 * n) : n = 9 :=
sorry

end NUMINAMATH_GPT_polygon_sides_l1437_143737


namespace NUMINAMATH_GPT_number_of_divisors_125n5_l1437_143725

theorem number_of_divisors_125n5 (n : ℕ) (hn : n > 0)
  (h150 : ∀ m : ℕ, m = 150 * n ^ 4 → (∃ d : ℕ, d * (d + 1) = 150)) :
  ∃ d : ℕ, d = 125 * n ^ 5 ∧ ((13 + 1) * (5 + 1) * (5 + 1) = 504) :=
by
  sorry

end NUMINAMATH_GPT_number_of_divisors_125n5_l1437_143725


namespace NUMINAMATH_GPT_intersection_A_B_subsets_C_l1437_143757

-- Definition of sets A and B
def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | 0 ≤ x}

-- Definition of intersection C
def C : Set ℤ := A ∩ B

-- The proof statements
theorem intersection_A_B : C = {1, 2} := 
by sorry

theorem subsets_C : {s | s ⊆ C} = {∅, {1}, {2}, {1, 2}} := 
by sorry

end NUMINAMATH_GPT_intersection_A_B_subsets_C_l1437_143757


namespace NUMINAMATH_GPT_min_children_see_ear_l1437_143760

theorem min_children_see_ear (n : ℕ) : ∃ (k : ℕ), k = n + 2 :=
by
  sorry

end NUMINAMATH_GPT_min_children_see_ear_l1437_143760


namespace NUMINAMATH_GPT_max_ben_cupcakes_l1437_143722

theorem max_ben_cupcakes (total_cupcakes : ℕ) (ben_cupcakes charles_cupcakes diana_cupcakes : ℕ)
    (h1 : total_cupcakes = 30)
    (h2 : diana_cupcakes = 2 * ben_cupcakes)
    (h3 : charles_cupcakes = diana_cupcakes)
    (h4 : total_cupcakes = ben_cupcakes + charles_cupcakes + diana_cupcakes) :
    ben_cupcakes = 6 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_max_ben_cupcakes_l1437_143722


namespace NUMINAMATH_GPT_max_vertex_value_in_cube_l1437_143713

def transform_black (v : ℕ) (e1 e2 e3 : ℕ) : ℕ :=
  e1 + e2 + e3

def transform_white (v : ℕ) (d1 d2 d3 : ℕ) : ℕ :=
  d1 + d2 + d3

def max_value_after_transformation (initial_values : Fin 8 → ℕ) : ℕ :=
  -- Combination of transformations and iterations are derived here
  42648

theorem max_vertex_value_in_cube :
  ∀ (initial_values : Fin 8 → ℕ),
  (∀ i, 1 ≤ initial_values i ∧ initial_values i ≤ 8) →
  (∃ (final_value : ℕ), final_value = max_value_after_transformation initial_values) → final_value = 42648 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_vertex_value_in_cube_l1437_143713


namespace NUMINAMATH_GPT_no_valid_a_exists_l1437_143742

theorem no_valid_a_exists (a : ℕ) (n : ℕ) (h1 : a > 1) (b := a * (10^n + 1)) :
  ¬ (∃ a : ℕ, b % (a^2) = 0) :=
by {
  sorry -- The actual proof is not required as per instructions.
}

end NUMINAMATH_GPT_no_valid_a_exists_l1437_143742


namespace NUMINAMATH_GPT_total_property_value_l1437_143739

-- Define the given conditions
def price_per_sq_ft_condo := 98
def price_per_sq_ft_barn := 84
def price_per_sq_ft_detached := 102
def price_per_sq_ft_garage := 60
def sq_ft_condo := 2400
def sq_ft_barn := 1200
def sq_ft_detached := 3500
def sq_ft_garage := 480

-- Main statement to prove the total value of the property
theorem total_property_value :
  (price_per_sq_ft_condo * sq_ft_condo + 
   price_per_sq_ft_barn * sq_ft_barn + 
   price_per_sq_ft_detached * sq_ft_detached + 
   price_per_sq_ft_garage * sq_ft_garage = 721800) :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_total_property_value_l1437_143739


namespace NUMINAMATH_GPT_locus_is_hyperbola_l1437_143735

theorem locus_is_hyperbola
  (x y a θ₁ θ₂ c : ℝ)
  (h1 : (x - a) * Real.cos θ₁ + y * Real.sin θ₁ = a)
  (h2 : (x - a) * Real.cos θ₂ + y * Real.sin θ₂ = a)
  (h3 : Real.tan (θ₁ / 2) - Real.tan (θ₂ / 2) = 2 * c)
  (hc : c > 1) 
  : ∃ k l m : ℝ, k * (x ^ 2) + l * x * y + m * (y ^ 2) = 1 := sorry

end NUMINAMATH_GPT_locus_is_hyperbola_l1437_143735


namespace NUMINAMATH_GPT_quadrilateral_EFGH_l1437_143769

variable {EF FG GH HE EH : ℤ}

theorem quadrilateral_EFGH (h1 : EF = 6) (h2 : FG = 18) (h3 : GH = 6) (h4 : HE = 10) (h5 : 12 < EH) (h6 : EH < 24) : EH = 12 := 
sorry

end NUMINAMATH_GPT_quadrilateral_EFGH_l1437_143769


namespace NUMINAMATH_GPT_range_of_a_in_quadratic_l1437_143736

theorem range_of_a_in_quadratic :
  (∃ x1 x2 : ℝ, x1 < -1 ∧ x2 > 1 ∧ x1 ≠ x2 ∧ x1^2 + a * x1 - 2 = 0 ∧ x2^2 + a * x2 - 2 = 0) → -1 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_in_quadratic_l1437_143736


namespace NUMINAMATH_GPT_xiaolin_distance_l1437_143777

theorem xiaolin_distance (speed : ℕ) (time : ℕ) (distance : ℕ)
    (h1 : speed = 80) (h2 : time = 28) : distance = 2240 :=
by
  have h3 : distance = time * speed := by sorry
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_xiaolin_distance_l1437_143777


namespace NUMINAMATH_GPT_negative_option_is_B_l1437_143781

-- Define the options as constants
def optionA : ℤ := -( -2 )
def optionB : ℤ := (-1) ^ 2023
def optionC : ℤ := |(-1) ^ 2|
def optionD : ℤ := (-5) ^ 2

-- Prove that the negative number among the options is optionB
theorem negative_option_is_B : optionB = -1 := 
by
  rw [optionB]
  sorry

end NUMINAMATH_GPT_negative_option_is_B_l1437_143781


namespace NUMINAMATH_GPT_expected_winnings_l1437_143791

theorem expected_winnings (roll_1_2: ℝ) (roll_3_4: ℝ) (roll_5_6: ℝ) (p1_2 p3_4 p5_6: ℝ) :
    roll_1_2 = 2 →
    roll_3_4 = 4 →
    roll_5_6 = -6 →
    p1_2 = 1 / 8 →
    p3_4 = 1 / 4 →
    p5_6 = 1 / 8 →
    (2 * p1_2 + 2 * p1_2 + 4 * p3_4 + 4 * p3_4 + roll_5_6 * p5_6 + roll_5_6 * p5_6) = 1 := by
  intros
  sorry

end NUMINAMATH_GPT_expected_winnings_l1437_143791


namespace NUMINAMATH_GPT_linear_function_no_third_quadrant_l1437_143718

theorem linear_function_no_third_quadrant (k b : ℝ) (h : ∀ x y : ℝ, x < 0 → y < 0 → y ≠ k * x + b) : k < 0 ∧ 0 ≤ b :=
sorry

end NUMINAMATH_GPT_linear_function_no_third_quadrant_l1437_143718


namespace NUMINAMATH_GPT_probability_complement_B_probability_union_A_B_l1437_143794

variable (Ω : Type) [MeasurableSpace Ω] {P : ProbabilityMeasure Ω}
variable (A B : Set Ω)

theorem probability_complement_B
  (hB : P B = 1 / 3) : P Bᶜ = 2 / 3 :=
by
  sorry

theorem probability_union_A_B
  (hA : P A = 1 / 2) (hB : P B = 1 / 3) : P (A ∪ B) ≤ 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_probability_complement_B_probability_union_A_B_l1437_143794


namespace NUMINAMATH_GPT_similar_triangles_XY_length_l1437_143731

-- Defining necessary variables.
variables (PQ QR YZ XY : ℝ) (area_XYZ : ℝ)

-- Given conditions to be used in the proof.
def condition1 : PQ = 8 := sorry
def condition2 : QR = 16 := sorry
def condition3 : YZ = 24 := sorry
def condition4 : area_XYZ = 144 := sorry

-- Statement of the mathematical proof problem to show XY = 12
theorem similar_triangles_XY_length :
  PQ = 8 → QR = 16 → YZ = 24 → area_XYZ = 144 → XY = 12 :=
by
  intros hPQ hQR hYZ hArea
  sorry

end NUMINAMATH_GPT_similar_triangles_XY_length_l1437_143731


namespace NUMINAMATH_GPT_unique_solution_l1437_143758

def my_operation (x y : ℝ) : ℝ := 5 * x - 2 * y + 3 * x * y

theorem unique_solution :
  ∃! y : ℝ, my_operation 4 y = 15 ∧ y = -1/2 :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_l1437_143758


namespace NUMINAMATH_GPT_prove_original_sides_l1437_143755

def original_parallelogram_sides (a b : ℕ) : Prop :=
  ∃ k : ℕ, (a, b) = (k * 1, k * 2) ∨ (a, b) = (1, 5) ∨ (a, b) = (4, 5) ∨ (a, b) = (3, 7) ∨ (a, b) = (4, 7) ∨ (a, b) = (3, 8) ∨ (a, b) = (5, 8) ∨ (a, b) = (5, 7) ∨ (a, b) = (2, 7)

theorem prove_original_sides (a b : ℕ) : original_parallelogram_sides a b → (1, 2) = (1, 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_prove_original_sides_l1437_143755


namespace NUMINAMATH_GPT_student_count_l1437_143756

theorem student_count (N : ℕ) (h1 : ∀ W : ℝ, W - 46 = 86 - 40) (h2 : (86 - 46) = 5 * N) : N = 8 :=
sorry

end NUMINAMATH_GPT_student_count_l1437_143756


namespace NUMINAMATH_GPT_german_students_count_l1437_143741

def total_students : ℕ := 45
def both_english_german : ℕ := 12
def only_english : ℕ := 23

theorem german_students_count :
  ∃ G : ℕ, G = 45 - (23 + 12) + 12 :=
sorry

end NUMINAMATH_GPT_german_students_count_l1437_143741


namespace NUMINAMATH_GPT_Jamie_earns_10_per_hour_l1437_143782

noncomputable def JamieHourlyRate (days_per_week : ℕ) (hours_per_day : ℕ) (weeks : ℕ) (total_earnings : ℕ) : ℕ :=
  let total_hours := days_per_week * hours_per_day * weeks
  total_earnings / total_hours

theorem Jamie_earns_10_per_hour :
  JamieHourlyRate 2 3 6 360 = 10 := by
  sorry

end NUMINAMATH_GPT_Jamie_earns_10_per_hour_l1437_143782


namespace NUMINAMATH_GPT_households_using_both_brands_l1437_143759

def total : ℕ := 260
def neither : ℕ := 80
def onlyA : ℕ := 60
def onlyB (both : ℕ) : ℕ := 3 * both

theorem households_using_both_brands (both : ℕ) : 80 + 60 + both + onlyB both = 260 → both = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_households_using_both_brands_l1437_143759


namespace NUMINAMATH_GPT_least_subset_gcd_l1437_143765

variable (S : Set ℕ) (f : ℕ → ℤ)
variable (a : ℕ → ℕ)
variable (k : ℕ)

def conditions (S : Set ℕ) (f : ℕ → ℤ) : Prop :=
  ∃ (a : ℕ → ℕ), 
  (∀ i j, i ≠ j → a i < a j) ∧ 
  (S = {i | ∃ n, i = a n ∧ n < 2004}) ∧ 
  (∀ i, f (a i) < 2003) ∧ 
  (∀ i j, f (a i) = f (a j))

theorem least_subset_gcd (h : conditions S f) : k = 1003 :=
  sorry

end NUMINAMATH_GPT_least_subset_gcd_l1437_143765


namespace NUMINAMATH_GPT_fraction_sum_simplified_l1437_143740

theorem fraction_sum_simplified (a b : ℕ) (h1 : 0.6125 = (a : ℝ) / b) (h2 : Nat.gcd a b = 1) : a + b = 129 :=
sorry

end NUMINAMATH_GPT_fraction_sum_simplified_l1437_143740


namespace NUMINAMATH_GPT_room_analysis_l1437_143723

-- First person's statements
def statement₁ (n: ℕ) (liars: ℕ) :=
  n ≤ 3 ∧ liars = n

-- Second person's statements
def statement₂ (n: ℕ) (liars: ℕ) :=
  n ≤ 4 ∧ liars < n

-- Third person's statements
def statement₃ (n: ℕ) (liars: ℕ) :=
  n = 5 ∧ liars = 3

theorem room_analysis (n liars : ℕ) :
  (¬ statement₁ n liars) ∧ statement₂ n liars ∧ ¬ statement₃ n liars → (n = 4 ∧ liars = 2) :=
by
  sorry

end NUMINAMATH_GPT_room_analysis_l1437_143723


namespace NUMINAMATH_GPT_count_four_digit_numbers_without_1_or_4_l1437_143771

-- Define a function to check if a digit is allowed (i.e., not 1 or 4)
def allowed_digit (d : ℕ) : Prop := d ≠ 1 ∧ d ≠ 4

-- Function to count four-digit numbers without digits 1 or 4
def count_valid_four_digit_numbers : ℕ :=
  let valid_first_digits := [2, 3, 5, 6, 7, 8, 9]
  let valid_other_digits := [0, 2, 3, 5, 6, 7, 8, 9]
  (valid_first_digits.length) * (valid_other_digits.length ^ 3)

-- The main theorem stating that the number of valid four-digit integers is 3072
theorem count_four_digit_numbers_without_1_or_4 : count_valid_four_digit_numbers = 3072 :=
by
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_without_1_or_4_l1437_143771


namespace NUMINAMATH_GPT_real_number_value_of_m_pure_imaginary_value_of_m_l1437_143774

def is_real (z : ℂ) : Prop := z.im = 0
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem real_number_value_of_m (m : ℝ) : 
  is_real ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = 0 ∨ m = 2) := 
by sorry

theorem pure_imaginary_value_of_m (m : ℝ) : 
  is_pure_imaginary ((m^2 + 2 * m - 8) + (m^2 - 2 * m) * I) ↔ (m = -4) := 
by sorry

end NUMINAMATH_GPT_real_number_value_of_m_pure_imaginary_value_of_m_l1437_143774


namespace NUMINAMATH_GPT_proof_problem_l1437_143711

theorem proof_problem (a b c : ℤ) (h1 : a > 2) (h2 : b < 10) (h3 : c ≥ 0) (h4 : 32 = a + 2 * b + 3 * c) : 
  a = 4 ∧ b = 8 ∧ c = 4 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1437_143711


namespace NUMINAMATH_GPT_g_inv_undefined_at_one_l1437_143700

noncomputable def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

theorem g_inv_undefined_at_one :
  ∀ (x : ℝ), (∃ (y : ℝ), g y = x ∧ ¬ ∃ (z : ℝ), g z = y ∧ g z = 1) ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_g_inv_undefined_at_one_l1437_143700


namespace NUMINAMATH_GPT_average_visitors_per_day_l1437_143761

theorem average_visitors_per_day (avg_sunday_visitors : ℕ) (avg_otherday_visitors : ℕ) (days_in_month : ℕ)
  (starts_with_sunday : Bool) (num_sundays : ℕ) (num_otherdays : ℕ)
  (h1 : avg_sunday_visitors = 510)
  (h2 : avg_otherday_visitors = 240)
  (h3 : days_in_month = 30)
  (h4 : starts_with_sunday = true)
  (h5 : num_sundays = 5)
  (h6 : num_otherdays = 25) :
  (num_sundays * avg_sunday_visitors + num_otherdays * avg_otherday_visitors) / days_in_month = 285 :=
by 
  sorry

end NUMINAMATH_GPT_average_visitors_per_day_l1437_143761


namespace NUMINAMATH_GPT_correct_option_given_inequality_l1437_143733

theorem correct_option_given_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
sorry

end NUMINAMATH_GPT_correct_option_given_inequality_l1437_143733


namespace NUMINAMATH_GPT_order_of_trig_values_l1437_143773

noncomputable def a := Real.sin (Real.sin (2008 * Real.pi / 180))
noncomputable def b := Real.sin (Real.cos (2008 * Real.pi / 180))
noncomputable def c := Real.cos (Real.sin (2008 * Real.pi / 180))
noncomputable def d := Real.cos (Real.cos (2008 * Real.pi / 180))

theorem order_of_trig_values : b < a ∧ a < d ∧ d < c :=
by
  sorry

end NUMINAMATH_GPT_order_of_trig_values_l1437_143773


namespace NUMINAMATH_GPT_derek_walk_time_l1437_143796

theorem derek_walk_time (x : ℕ) :
  (∀ y : ℕ, (y = 9) → (∀ d₁ d₂ : ℕ, (d₁ = 20 ∧ d₂ = 60) →
    (20 * x = d₁ * y + d₂))) → x = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_derek_walk_time_l1437_143796


namespace NUMINAMATH_GPT_gcd_polynomial_l1437_143708

-- Define conditions
variables (b : ℤ) (k : ℤ)

-- Assume b is an even multiple of 8753
def is_even_multiple_of_8753 (b : ℤ) : Prop := ∃ k : ℤ, b = 2 * 8753 * k

-- Statement to be proven
theorem gcd_polynomial (b : ℤ) (h : is_even_multiple_of_8753 b) :
  Int.gcd (4 * b^2 + 27 * b + 100) (3 * b + 7) = 2 :=
by sorry

end NUMINAMATH_GPT_gcd_polynomial_l1437_143708


namespace NUMINAMATH_GPT_find_share_of_b_l1437_143701

variable (a b c : ℕ)
axiom h1 : a = 3 * b
axiom h2 : b = c + 25
axiom h3 : a + b + c = 645

theorem find_share_of_b : b = 134 := by
  sorry

end NUMINAMATH_GPT_find_share_of_b_l1437_143701


namespace NUMINAMATH_GPT_government_subsidy_per_hour_l1437_143785

-- Given conditions:
def cost_first_employee : ℕ := 20
def cost_second_employee : ℕ := 22
def hours_per_week : ℕ := 40
def weekly_savings : ℕ := 160

-- To prove:
theorem government_subsidy_per_hour (S : ℕ) : S = 2 :=
by
  -- Proof steps go here.
  sorry

end NUMINAMATH_GPT_government_subsidy_per_hour_l1437_143785


namespace NUMINAMATH_GPT_largest_expression_l1437_143716

def P : ℕ := 3 * 2024 ^ 2025
def Q : ℕ := 2024 ^ 2025
def R : ℕ := 2023 * 2024 ^ 2024
def S : ℕ := 3 * 2024 ^ 2024
def T : ℕ := 2024 ^ 2024
def U : ℕ := 2024 ^ 2023

theorem largest_expression : 
  (P - Q) > (Q - R) ∧ 
  (P - Q) > (R - S) ∧ 
  (P - Q) > (S - T) ∧ 
  (P - Q) > (T - U) :=
by sorry

end NUMINAMATH_GPT_largest_expression_l1437_143716


namespace NUMINAMATH_GPT_min_colors_needed_l1437_143729

def cell := (ℤ × ℤ)

def rook_distance (c1 c2 : cell) : ℤ :=
  max (abs (c1.1 - c2.1)) (abs (c1.2 - c2.2))

def color (c : cell) : ℤ :=
  (c.1 + c.2) % 4

theorem min_colors_needed : 4 = 4 :=
sorry

end NUMINAMATH_GPT_min_colors_needed_l1437_143729


namespace NUMINAMATH_GPT_average_speed_l1437_143728

theorem average_speed (d1 d2 t1 t2 : ℝ) (h1 : d1 = 50) (h2 : d2 = 20) (h3 : t1 = 50 / 20) (h4 : t2 = 20 / 40) :
  ((d1 + d2) / (t1 + t2)) = 23.33 := 
  sorry

end NUMINAMATH_GPT_average_speed_l1437_143728


namespace NUMINAMATH_GPT_monthly_salary_is_correct_l1437_143790

noncomputable def man's_salary : ℝ :=
  let S : ℝ := 6500
  S

theorem monthly_salary_is_correct (S : ℝ) (h1 : S * 0.20 = S * 0.20) (h2 : S * 0.80 * 1.20 + 260 = S):
  S = man's_salary :=
by sorry

end NUMINAMATH_GPT_monthly_salary_is_correct_l1437_143790


namespace NUMINAMATH_GPT_twelve_integers_divisible_by_eleven_l1437_143799

theorem twelve_integers_divisible_by_eleven (a : Fin 12 → ℤ) : 
  ∃ (i j : Fin 12), i ≠ j ∧ 11 ∣ (a i - a j) :=
by
  sorry

end NUMINAMATH_GPT_twelve_integers_divisible_by_eleven_l1437_143799


namespace NUMINAMATH_GPT_andrew_brian_ratio_l1437_143762

-- Definitions based on conditions extracted from the problem
variables (A S B : ℕ)

-- Conditions
def steven_shirts : Prop := S = 72
def brian_shirts : Prop := B = 3
def steven_andrew_relation : Prop := S = 4 * A

-- The goal is to prove the ratio of Andrew's shirts to Brian's shirts is 6
theorem andrew_brian_ratio (A S B : ℕ) 
  (h1 : steven_shirts S) 
  (h2 : brian_shirts B)
  (h3 : steven_andrew_relation A S) :
  A / B = 6 := by
  sorry

end NUMINAMATH_GPT_andrew_brian_ratio_l1437_143762


namespace NUMINAMATH_GPT_gcd_divisors_remainders_l1437_143717

theorem gcd_divisors_remainders (d : ℕ) :
  (1657 % d = 6) ∧ (2037 % d = 5) → d = 127 :=
by
  sorry

end NUMINAMATH_GPT_gcd_divisors_remainders_l1437_143717
