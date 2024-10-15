import Mathlib

namespace NUMINAMATH_GPT_soccer_points_l1306_130676

def total_points (wins draws losses : ℕ) (points_per_win points_per_draw points_per_loss : ℕ) : ℕ :=
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

theorem soccer_points : total_points 14 4 2 3 1 0 = 46 :=
by
  sorry

end NUMINAMATH_GPT_soccer_points_l1306_130676


namespace NUMINAMATH_GPT_alpha_squared_plus_3alpha_plus_beta_equals_2023_l1306_130671

-- Definitions and conditions
variables (α β : ℝ)
-- α and β are roots of the quadratic equation x² + 2x - 2025 = 0
def is_root_of_quadratic_1 : Prop := α^2 + 2 * α - 2025 = 0
def is_root_of_quadratic_2 : Prop := β^2 + 2 * β - 2025 = 0
-- Vieta's formula gives us α + β = -2
def sum_of_roots : Prop := α + β = -2

-- Theorem (statement) we want to prove
theorem alpha_squared_plus_3alpha_plus_beta_equals_2023 (h1 : is_root_of_quadratic_1 α)
                                                      (h2 : is_root_of_quadratic_2 β)
                                                      (h3 : sum_of_roots α β) :
                                                      α^2 + 3 * α + β = 2023 :=
by
  sorry

end NUMINAMATH_GPT_alpha_squared_plus_3alpha_plus_beta_equals_2023_l1306_130671


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l1306_130635

-- Problem a
theorem problem_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 :=
by sorry

-- Problem b
theorem problem_b (a : ℝ) : (2 * a + 3) * (2 * a - 3) = 4 * a^2 - 9 :=
by sorry

-- Problem c
theorem problem_c (m n : ℝ) : (m^3 - n^5) * (n^5 + m^3) = m^6 - n^10 :=
by sorry

-- Problem d
theorem problem_d (m n : ℝ) : (3 * m^2 - 5 * n^2) * (3 * m^2 + 5 * n^2) = 9 * m^4 - 25 * n^4 :=
by sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l1306_130635


namespace NUMINAMATH_GPT_probability_all_boxes_non_empty_equals_4_over_9_l1306_130611

structure PaintingPlacement :=
  (paintings : Finset ℕ)
  (boxes : Finset ℕ)
  (num_paintings : paintings.card = 4)
  (num_boxes : boxes.card = 3)

noncomputable def probability_non_empty_boxes (pp : PaintingPlacement) : ℚ :=
  let total_outcomes := 3^4
  let favorable_outcomes := Nat.choose 4 2 * Nat.factorial 3
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

theorem probability_all_boxes_non_empty_equals_4_over_9
  (pp : PaintingPlacement) : pp.paintings.card = 4 → pp.boxes.card = 3 →
  probability_non_empty_boxes pp = 4 / 9 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_probability_all_boxes_non_empty_equals_4_over_9_l1306_130611


namespace NUMINAMATH_GPT_union_of_sets_l1306_130684

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (h1 : A = {x, y}) (h2 : B = {x + 1, 5}) (h3 : A ∩ B = {2}) : A ∪ B = {1, 2, 5} :=
sorry

end NUMINAMATH_GPT_union_of_sets_l1306_130684


namespace NUMINAMATH_GPT_james_muffins_baked_l1306_130646

-- Define the number of muffins Arthur baked
def muffinsArthur : ℕ := 115

-- Define the multiplication factor
def multiplicationFactor : ℕ := 12

-- Define the number of muffins James baked
def muffinsJames : ℕ := muffinsArthur * multiplicationFactor

-- The theorem that needs to be proved
theorem james_muffins_baked : muffinsJames = 1380 :=
by
  sorry

end NUMINAMATH_GPT_james_muffins_baked_l1306_130646


namespace NUMINAMATH_GPT_shaded_area_correct_l1306_130674

-- Definition of the grid dimensions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

-- Definition of the heights of the shaded regions in segments
def shaded_height (x : ℕ) : ℕ :=
if x < 4 then 2
else if x < 9 then 3
else if x < 13 then 4
else if x < 15 then 5
else 0

-- Definition for the area of the entire grid
def grid_area : ℝ := grid_width * grid_height

-- Definition for the area of the unshaded triangle
def unshaded_triangle_area : ℝ := 0.5 * grid_width * grid_height

-- Definition for the area of the shaded region
def shaded_area : ℝ := grid_area - unshaded_triangle_area

-- The theorem to be proved
theorem shaded_area_correct : shaded_area = 37.5 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l1306_130674


namespace NUMINAMATH_GPT_solve_equation_l1306_130677

theorem solve_equation : ∀ x : ℝ, (10 - x) ^ 2 = 4 * x ^ 2 ↔ x = 10 / 3 ∨ x = -10 :=
by
  intros x
  sorry

end NUMINAMATH_GPT_solve_equation_l1306_130677


namespace NUMINAMATH_GPT_calculate_product1_calculate_square_l1306_130664

theorem calculate_product1 : 100.2 * 99.8 = 9999.96 :=
by
  sorry

theorem calculate_square : 103^2 = 10609 :=
by
  sorry

end NUMINAMATH_GPT_calculate_product1_calculate_square_l1306_130664


namespace NUMINAMATH_GPT_ed_more_marbles_than_doug_initially_l1306_130670

noncomputable def ed_initial_marbles := 37
noncomputable def doug_marbles := 5

theorem ed_more_marbles_than_doug_initially :
  ed_initial_marbles - doug_marbles = 32 := by
  sorry

end NUMINAMATH_GPT_ed_more_marbles_than_doug_initially_l1306_130670


namespace NUMINAMATH_GPT_circumcircle_circumference_thm_triangle_perimeter_thm_l1306_130603

-- Definition and theorem for the circumference of the circumcircle
def circumcircle_circumference (a b c R : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * R = c / (Real.sqrt (1 - cosC^2)) 
  ∧ 2 * R * Real.pi = 3 * Real.pi

theorem circumcircle_circumference_thm (a b c R : ℝ) (cosC : ℝ) :
  circumcircle_circumference a b c R cosC → 2 * R * Real.pi = 3 * Real.pi :=
by
  intro h;
  sorry

-- Definition and theorem for the perimeter of the triangle
def triangle_perimeter (a b c : ℝ) (cosC : ℝ) :=
  cosC = 2 / 3 ∧ c = Real.sqrt 5 ∧ 2 * a = 3 * b ∧ (a + b + c) = 5 + Real.sqrt 5

theorem triangle_perimeter_thm (a b c : ℝ) (cosC : ℝ) :
  triangle_perimeter a b c cosC → (a + b + c) = 5 + Real.sqrt 5 :=
by
  intro h;
  sorry

end NUMINAMATH_GPT_circumcircle_circumference_thm_triangle_perimeter_thm_l1306_130603


namespace NUMINAMATH_GPT_john_total_spent_l1306_130694

-- Defining the conditions from part a)
def vacuum_cleaner_original_price : ℝ := 250
def vacuum_cleaner_discount_rate : ℝ := 0.20
def dishwasher_price : ℝ := 450
def special_offer_discount : ℝ := 75
def sales_tax_rate : ℝ := 0.07

-- The adesso to formalize part c noncomputably.
noncomputable def total_amount_spent : ℝ :=
  let vacuum_cleaner_discount := vacuum_cleaner_original_price * vacuum_cleaner_discount_rate
  let vacuum_cleaner_final_price := vacuum_cleaner_original_price - vacuum_cleaner_discount
  let total_before_special_offer := vacuum_cleaner_final_price + dishwasher_price
  let total_after_special_offer := total_before_special_offer - special_offer_discount
  let sales_tax := total_after_special_offer * sales_tax_rate
  total_after_special_offer + sales_tax

-- The proof statement
theorem john_total_spent : total_amount_spent = 615.25 := by
  sorry

end NUMINAMATH_GPT_john_total_spent_l1306_130694


namespace NUMINAMATH_GPT_largest_base_5_three_digit_in_base_10_l1306_130602

theorem largest_base_5_three_digit_in_base_10 :
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  n = 124 :=
by
  let n := 4 * 5^2 + 4 * 5^1 + 4 * 5^0
  show n = 124
  sorry

end NUMINAMATH_GPT_largest_base_5_three_digit_in_base_10_l1306_130602


namespace NUMINAMATH_GPT_maximum_value_of_k_l1306_130613

-- Define the variables and conditions
variables {a b c k : ℝ}
axiom h₀ : a > b
axiom h₁ : b > c
axiom h₂ : 4 / (a - b) + 1 / (b - c) + k / (c - a) ≥ 0

-- State the theorem
theorem maximum_value_of_k : k ≤ 9 := sorry

end NUMINAMATH_GPT_maximum_value_of_k_l1306_130613


namespace NUMINAMATH_GPT_solve_3x_5y_eq_7_l1306_130685

theorem solve_3x_5y_eq_7 :
  ∃ (x y k : ℤ), (3 * x + 5 * y = 7) ∧ (x = 4 + 5 * k) ∧ (y = -1 - 3 * k) :=
by 
  sorry

end NUMINAMATH_GPT_solve_3x_5y_eq_7_l1306_130685


namespace NUMINAMATH_GPT_kim_points_correct_l1306_130682

-- Definitions of given conditions
def points_easy : ℕ := 2
def points_average : ℕ := 3
def points_hard : ℕ := 5

def correct_easy : ℕ := 6
def correct_average : ℕ := 2
def correct_hard : ℕ := 4

-- Definition of total points calculation
def kim_total_points : ℕ :=
  (correct_easy * points_easy) +
  (correct_average * points_average) +
  (correct_hard * points_hard)

-- Theorem stating that Kim's total points are 38
theorem kim_points_correct : kim_total_points = 38 := by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_kim_points_correct_l1306_130682


namespace NUMINAMATH_GPT_phone_charges_equal_l1306_130622

theorem phone_charges_equal (x : ℝ) : 
  (0.60 + 14 * x = 0.08 * 18) → (x = 0.06) :=
by
  intro h
  have : 14 * x = 1.44 - 0.60 := sorry
  have : 14 * x = 0.84 := sorry
  have : x = 0.06 := sorry
  exact this

end NUMINAMATH_GPT_phone_charges_equal_l1306_130622


namespace NUMINAMATH_GPT_max_m_l1306_130658

theorem max_m : ∃ m A B : ℤ, (AB = 90 ∧ m = 5 * B + A) ∧ (∀ m' A' B', (A' * B' = 90 ∧ m' = 5 * B' + A') → m' ≤ 451) ∧ m = 451 :=
by
  sorry

end NUMINAMATH_GPT_max_m_l1306_130658


namespace NUMINAMATH_GPT_complex_number_pure_imaginary_l1306_130669

theorem complex_number_pure_imaginary (a : ℝ) 
  (h1 : ∃ a : ℝ, (a^2 - 2*a - 3 = 0) ∧ (a + 1 ≠ 0)) 
  : a = 3 := sorry

end NUMINAMATH_GPT_complex_number_pure_imaginary_l1306_130669


namespace NUMINAMATH_GPT_no_int_x_divisible_by_169_l1306_130640

theorem no_int_x_divisible_by_169 (x : ℤ) : ¬ (169 ∣ (x^2 + 5 * x + 16)) := by
  sorry

end NUMINAMATH_GPT_no_int_x_divisible_by_169_l1306_130640


namespace NUMINAMATH_GPT_rug_overlap_area_l1306_130649

theorem rug_overlap_area (A S S2 S3 : ℝ) 
  (hA : A = 200)
  (hS : S = 138)
  (hS2 : S2 = 24)
  (h1 : ∃ (S1 : ℝ), S1 + S2 + S3 = S)
  (h2 : ∃ (S1 : ℝ), S1 + 2 * S2 + 3 * S3 = A) : S3 = 19 :=
by
  sorry

end NUMINAMATH_GPT_rug_overlap_area_l1306_130649


namespace NUMINAMATH_GPT_no_integer_solutions_l1306_130601

theorem no_integer_solutions (x y : ℤ) : 2 * x^2 - 5 * y^2 ≠ 7 :=
  sorry

end NUMINAMATH_GPT_no_integer_solutions_l1306_130601


namespace NUMINAMATH_GPT_sum_and_product_of_roots_cube_l1306_130643

theorem sum_and_product_of_roots_cube (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : 
  x^3 + y^3 = 1008 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_and_product_of_roots_cube_l1306_130643


namespace NUMINAMATH_GPT_evaluate_at_minus_two_l1306_130604

def f (x : ℝ) : ℝ := x^5 + 5*x^4 + 10*x^3 + 10*x^2 + 5*x + 1

theorem evaluate_at_minus_two : f (-2) = -1 := 
by 
  unfold f 
  sorry

end NUMINAMATH_GPT_evaluate_at_minus_two_l1306_130604


namespace NUMINAMATH_GPT_simple_interest_rate_l1306_130628

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) 
  (hT : T = 10) (hSI : (P * R * T) / 100 = (1 / 5) * P) : R = 2 :=
by
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1306_130628


namespace NUMINAMATH_GPT_remainder_when_7x_div_9_l1306_130660

theorem remainder_when_7x_div_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 :=
sorry

end NUMINAMATH_GPT_remainder_when_7x_div_9_l1306_130660


namespace NUMINAMATH_GPT_max_min_product_of_three_l1306_130638

open List

theorem max_min_product_of_three (s : List Int) (h : s = [-1, -2, 3, 4]) : 
  ∃ (max min : Int), 
    max = 8 ∧ min = -24 ∧ 
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≤ max) ∧
    (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → b ≠ c → a ≠ c → a * b * c ≥ min) := 
by
  sorry

end NUMINAMATH_GPT_max_min_product_of_three_l1306_130638


namespace NUMINAMATH_GPT_rachel_took_money_l1306_130692

theorem rachel_took_money (x y : ℕ) (h₁ : x = 5) (h₂ : y = 3) : x - y = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_rachel_took_money_l1306_130692


namespace NUMINAMATH_GPT_min_value_ineq_l1306_130606

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 4 ∧ (∀ (a b : ℝ), 0 < a → 0 < b → a + b = 4 → (1/a + 4/b) ≥ 9/4)

theorem min_value_ineq : problem_statement :=
by
  unfold problem_statement
  sorry

end NUMINAMATH_GPT_min_value_ineq_l1306_130606


namespace NUMINAMATH_GPT_Jesse_remaining_money_l1306_130690

-- Define the conditions
def initial_money := 50
def novel_cost := 7
def lunch_cost := 2 * novel_cost
def total_spent := novel_cost + lunch_cost

-- Define the remaining money after spending
def remaining_money := initial_money - total_spent

-- Prove that the remaining money is $29
theorem Jesse_remaining_money : remaining_money = 29 := 
by
  sorry

end NUMINAMATH_GPT_Jesse_remaining_money_l1306_130690


namespace NUMINAMATH_GPT_valerie_light_bulbs_deficit_l1306_130656

theorem valerie_light_bulbs_deficit :
  let small_price := 8.75
  let medium_price := 11.25
  let large_price := 15.50
  let xsmall_price := 6.10
  let budget := 120
  
  let lamp_A_cost := 2 * small_price
  let lamp_B_cost := 3 * medium_price
  let lamp_C_cost := large_price
  let lamp_D_cost := 4 * xsmall_price
  let lamp_E_cost := 2 * large_price
  let lamp_F_cost := small_price + medium_price

  let total_cost := lamp_A_cost + lamp_B_cost + lamp_C_cost + lamp_D_cost + lamp_E_cost + lamp_F_cost

  total_cost - budget = 22.15 :=
by
  sorry

end NUMINAMATH_GPT_valerie_light_bulbs_deficit_l1306_130656


namespace NUMINAMATH_GPT_greatest_b_no_minus_six_in_range_l1306_130672

open Real

theorem greatest_b_no_minus_six_in_range :
  ∃ (b : ℤ), (b = 8) → (¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 15 = -6) :=
by {
  -- We need to find the largest integer b such that -6 is not in the range of f(x) = x^2 + bx + 15
  sorry
}

end NUMINAMATH_GPT_greatest_b_no_minus_six_in_range_l1306_130672


namespace NUMINAMATH_GPT_final_concentration_of_milk_l1306_130634

variable (x : ℝ) (total_vol : ℝ) (initial_milk : ℝ)
axiom x_value : x = 33.333333333333336
axiom total_volume : total_vol = 100
axiom initial_milk_vol : initial_milk = 36

theorem final_concentration_of_milk :
  let first_removal := x / total_vol * initial_milk
  let remaining_milk_after_first := initial_milk - first_removal
  let second_removal := x / total_vol * remaining_milk_after_first
  let final_milk := remaining_milk_after_first - second_removal
  (final_milk / total_vol) * 100 = 16 :=
by {
  sorry
}

end NUMINAMATH_GPT_final_concentration_of_milk_l1306_130634


namespace NUMINAMATH_GPT_regular_square_pyramid_side_edge_length_l1306_130619

theorem regular_square_pyramid_side_edge_length 
  (base_edge_length : ℝ)
  (volume : ℝ)
  (h_base_edge_length : base_edge_length = 4 * Real.sqrt 2)
  (h_volume : volume = 32) :
  ∃ side_edge_length : ℝ, side_edge_length = 5 :=
by sorry

end NUMINAMATH_GPT_regular_square_pyramid_side_edge_length_l1306_130619


namespace NUMINAMATH_GPT_tapA_fill_time_l1306_130648

-- Define the conditions
def fillTapA (t : ℕ) := 1 / t
def fillTapB := 1 / 40
def fillCombined (t : ℕ) := 9 * (fillTapA t + fillTapB)
def fillRemaining := 23 * fillTapB

-- Main theorem statement
theorem tapA_fill_time : ∀ (t : ℕ), fillCombined t + fillRemaining = 1 → t = 45 := by
  sorry

end NUMINAMATH_GPT_tapA_fill_time_l1306_130648


namespace NUMINAMATH_GPT_simplify_expression_l1306_130663

theorem simplify_expression :
  (2021^3 - 3 * 2021^2 * 2022 + 4 * 2021 * 2022^2 - 2022^3 + 2) / (2021 * 2022) = 
  1 + (1 / 2021) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1306_130663


namespace NUMINAMATH_GPT_pq_solution_l1306_130623

theorem pq_solution :
  ∃ (p q : ℤ), (20 * x ^ 2 - 110 * x - 120 = (5 * x + p) * (4 * x + q))
    ∧ (5 * q + 4 * p = -110) ∧ (p * q = -120)
    ∧ (p + 2 * q = -8) :=
by
  sorry

end NUMINAMATH_GPT_pq_solution_l1306_130623


namespace NUMINAMATH_GPT_difference_in_overlap_l1306_130625

variable (total_students : ℕ) (geometry_students : ℕ) (biology_students : ℕ)

theorem difference_in_overlap
  (h1 : total_students = 232)
  (h2 : geometry_students = 144)
  (h3 : biology_students = 119) :
  let max_overlap := min geometry_students biology_students;
  let min_overlap := geometry_students + biology_students - total_students;
  max_overlap - min_overlap = 88 :=
by 
  sorry

end NUMINAMATH_GPT_difference_in_overlap_l1306_130625


namespace NUMINAMATH_GPT_cross_section_quadrilateral_is_cylinder_l1306_130657

-- Definition of the solids
inductive Solid
| cone
| cylinder
| sphere

-- Predicate for the cross-section being a quadrilateral
def is_quadrilateral_cross_section (solid : Solid) : Prop :=
  match solid with
  | Solid.cylinder => true
  | Solid.cone     => false
  | Solid.sphere   => false

-- Main theorem statement
theorem cross_section_quadrilateral_is_cylinder (s : Solid) :
  is_quadrilateral_cross_section s → s = Solid.cylinder :=
by
  cases s
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]
  . simp [is_quadrilateral_cross_section]

end NUMINAMATH_GPT_cross_section_quadrilateral_is_cylinder_l1306_130657


namespace NUMINAMATH_GPT_impossible_300_numbers_l1306_130607

theorem impossible_300_numbers (n : ℕ) (hn : n = 300) (a : ℕ → ℕ) (hp : ∀ i, 0 < a i)
(hdiff : ∃ k, ∀ i ≠ k, a i = a ((i + 1) % n) - a ((i - 1 + n) % n)) 
: false :=
by {
  sorry
}

end NUMINAMATH_GPT_impossible_300_numbers_l1306_130607


namespace NUMINAMATH_GPT_number_of_n_such_that_n_div_25_minus_n_is_square_l1306_130698

theorem number_of_n_such_that_n_div_25_minus_n_is_square :
  ∃! n1 n2 : ℤ, ∀ n : ℤ, (n = n1 ∨ n = n2) ↔ ∃ k : ℤ, k^2 = n / (25 - n) :=
sorry

end NUMINAMATH_GPT_number_of_n_such_that_n_div_25_minus_n_is_square_l1306_130698


namespace NUMINAMATH_GPT_bond_paper_cost_l1306_130650

/-!
# Bond Paper Cost Calculation

This theorem calculates the total cost to buy the required amount of each type of bond paper, given the specified conditions.
-/

def cost_of_ream (sheets_per_ream : ℤ) (cost_per_ream : ℤ) (required_sheets : ℤ) : ℤ :=
  let reams_needed := (required_sheets + sheets_per_ream - 1) / sheets_per_ream
  reams_needed * cost_per_ream

theorem bond_paper_cost :
  let total_sheets := 5000
  let required_A := 2500
  let required_B := 1500
  let remaining_sheets := total_sheets - required_A - required_B
  let cost_A := cost_of_ream 500 27 required_A
  let cost_B := cost_of_ream 400 24 required_B
  let cost_C := cost_of_ream 300 18 remaining_sheets
  cost_A + cost_B + cost_C = 303 := 
by
  sorry

end NUMINAMATH_GPT_bond_paper_cost_l1306_130650


namespace NUMINAMATH_GPT_total_cats_received_l1306_130641

-- Defining the constants and conditions
def total_adult_cats := 150
def fraction_female_cats := 2 / 3
def fraction_litters := 2 / 5
def kittens_per_litter := 5

-- Defining the proof problem
theorem total_cats_received :
  let number_female_cats := (fraction_female_cats * total_adult_cats : ℤ)
  let number_litters := (fraction_litters * number_female_cats : ℤ)
  let number_kittens := number_litters * kittens_per_litter
  number_female_cats + number_kittens + (total_adult_cats - number_female_cats) = 350 := 
by
  sorry

end NUMINAMATH_GPT_total_cats_received_l1306_130641


namespace NUMINAMATH_GPT_machine_transport_equation_l1306_130630

theorem machine_transport_equation (x : ℝ) :
  (∀ (rateA rateB : ℝ), rateB = rateA + 60 → (500 / rateA = 800 / rateB) → rateA = x → rateB = x + 60) :=
by
  sorry

end NUMINAMATH_GPT_machine_transport_equation_l1306_130630


namespace NUMINAMATH_GPT_minimum_value_of_f_range_of_a_l1306_130662

noncomputable def f (x : ℝ) := x * Real.log x
noncomputable def g (x a : ℝ) := -x^2 + a * x - 3

theorem minimum_value_of_f :
  ∃ x_min : ℝ, ∀ x : ℝ, 0 < x → f x ≥ -1/Real.exp 1 := sorry -- This statement asserts that the minimum value of f(x) is -1/e.

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * f x ≥ g x a) → a ≤ 4 := sorry -- This statement asserts that if 2f(x) ≥ g(x) for all x > 0, then a is at most 4.

end NUMINAMATH_GPT_minimum_value_of_f_range_of_a_l1306_130662


namespace NUMINAMATH_GPT_part1_zero_of_f_a_neg1_part2_range_of_a_l1306_130659

noncomputable def f (a x : ℝ) := a * x^2 + 2 * x - 2 - a

theorem part1_zero_of_f_a_neg1 : 
  f (-1) 1 = 0 :=
by 
  sorry

theorem part2_range_of_a (a : ℝ) :
  a ≤ 0 →
  (∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ∧ (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x = 0 → x = 1) ↔ 
  (-1 ≤ a ∧ a ≤ 0) ∨ (a ≤ -2) :=
by 
  sorry

end NUMINAMATH_GPT_part1_zero_of_f_a_neg1_part2_range_of_a_l1306_130659


namespace NUMINAMATH_GPT_zero_pow_2014_l1306_130621

-- Define the condition that zero raised to any positive power is zero
def zero_pow_pos {n : ℕ} (h : 0 < n) : (0 : ℝ)^n = 0 := by
  sorry

-- Use this definition to prove the specific case of 0 ^ 2014 = 0
theorem zero_pow_2014 : (0 : ℝ)^(2014) = 0 := by
  have h : 0 < 2014 := by decide
  exact zero_pow_pos h

end NUMINAMATH_GPT_zero_pow_2014_l1306_130621


namespace NUMINAMATH_GPT_divisible_by_1995_l1306_130629

theorem divisible_by_1995 (n : ℕ) : 
  1995 ∣ (256^(2*n) * 7^(2*n) - 168^(2*n) - 32^(2*n) + 3^(2*n)) := 
sorry

end NUMINAMATH_GPT_divisible_by_1995_l1306_130629


namespace NUMINAMATH_GPT_intersection_sets_l1306_130653

theorem intersection_sets (x : ℝ) :
  let M := {x | 2 * x - x^2 ≥ 0 }
  let N := {x | -1 < x ∧ x < 1}
  M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_sets_l1306_130653


namespace NUMINAMATH_GPT_set_equiv_l1306_130666

-- Definition of the set A according to the conditions
def A : Set ℚ := { z : ℚ | ∃ p q : ℕ, z = p / (q : ℚ) ∧ p + q = 5 ∧ p > 0 ∧ q > 0 }

-- The target set we want to prove A is equal to
def target_set : Set ℚ := { 1/4, 2/3, 3/2, 4 }

-- The theorem to prove that both sets are equal
theorem set_equiv : A = target_set :=
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_set_equiv_l1306_130666


namespace NUMINAMATH_GPT_find_growth_rate_calculate_fourth_day_donation_l1306_130665

-- Define the conditions
def first_day_donation : ℝ := 3000
def third_day_donation : ℝ := 4320
def growth_rate (x : ℝ) : Prop := (1 + x)^2 = third_day_donation / first_day_donation

-- Since the problem states growth rate for second and third day is the same,
-- we need to find that rate which is equivalent to solving the above proposition for x.

theorem find_growth_rate : ∃ x : ℝ, growth_rate x ∧ x = 0.2 := by
  sorry

-- Calculate the fourth day's donation based on the growth rate found.
def fourth_day_donation (third_day : ℝ) (growth_rate : ℝ) : ℝ :=
  third_day * (1 + growth_rate)

theorem calculate_fourth_day_donation : 
  ∀ x : ℝ, growth_rate x → x = 0.2 → fourth_day_donation third_day_donation x = 5184 := by 
  sorry

end NUMINAMATH_GPT_find_growth_rate_calculate_fourth_day_donation_l1306_130665


namespace NUMINAMATH_GPT_find_a9_l1306_130610

variable {a : ℕ → ℤ}  -- Define a as a sequence of integers
variable (d : ℤ) (a3 : ℤ) (a4 : ℤ)

-- Define the specific conditions given in the problem
def arithmetic_sequence_condition (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ) : Prop :=
  a 3 + a 4 = 12 ∧ d = 2

-- Define the arithmetic sequence relation
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Statement to prove
theorem find_a9 
  (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ)
  (h1 : arithmetic_sequence_condition a d a3 a4)
  (h2 : arithmetic_sequence a d) :
  a 9 = 17 :=
sorry

end NUMINAMATH_GPT_find_a9_l1306_130610


namespace NUMINAMATH_GPT_weight_of_mixture_l1306_130627

noncomputable def total_weight_of_mixture (zinc_weight: ℝ) (zinc_ratio: ℝ) (total_ratio: ℝ) : ℝ :=
  (zinc_weight / zinc_ratio) * total_ratio

theorem weight_of_mixture (zinc_ratio: ℝ) (copper_ratio: ℝ) (tin_ratio: ℝ) (zinc_weight: ℝ) :
  total_weight_of_mixture zinc_weight zinc_ratio (zinc_ratio + copper_ratio + tin_ratio) = 98.95 :=
by 
  let ratio_sum := zinc_ratio + copper_ratio + tin_ratio
  let part_weight := zinc_weight / zinc_ratio
  let mixture_weight := part_weight * ratio_sum
  have h : mixture_weight = 98.95 := sorry
  exact h

end NUMINAMATH_GPT_weight_of_mixture_l1306_130627


namespace NUMINAMATH_GPT_second_place_jump_l1306_130695

theorem second_place_jump : 
  ∀ (Kyungsoo Younghee Jinju Chanho : ℝ), 
    Kyungsoo = 2.3 → 
    Younghee = 0.9 → 
    Jinju = 1.8 → 
    Chanho = 2.5 → 
    ((Kyungsoo < Chanho) ∧ (Kyungsoo > Jinju) ∧ (Kyungsoo > Younghee)) :=
by 
  sorry

end NUMINAMATH_GPT_second_place_jump_l1306_130695


namespace NUMINAMATH_GPT_length_of_rope_l1306_130600

-- Define the given conditions
variable (L : ℝ)
variable (h1 : 0.6 * L = 0.69)

-- The theorem to prove
theorem length_of_rope (L : ℝ) (h1 : 0.6 * L = 0.69) : L = 1.15 :=
by
  sorry

end NUMINAMATH_GPT_length_of_rope_l1306_130600


namespace NUMINAMATH_GPT_probability_at_least_75_cents_l1306_130614

def total_coins : ℕ := 3 + 5 + 4 + 3 -- total number of coins

def pennies : ℕ := 3
def nickels : ℕ := 5
def dimes : ℕ := 4
def quarters : ℕ := 3

def successful_outcomes_case1 : ℕ := (Nat.choose 3 3) * (Nat.choose 12 3)
def successful_outcomes_case2 : ℕ := (Nat.choose 3 2) * (Nat.choose 4 2) * (Nat.choose 5 2)

def total_outcomes : ℕ := Nat.choose 15 6
def successful_outcomes : ℕ := successful_outcomes_case1 + successful_outcomes_case2

def probability : ℚ := successful_outcomes / total_outcomes

theorem probability_at_least_75_cents :
  probability = 400 / 5005 := by
  sorry

end NUMINAMATH_GPT_probability_at_least_75_cents_l1306_130614


namespace NUMINAMATH_GPT_pasta_needed_for_family_reunion_l1306_130679

-- Conditions definition
def original_pasta : ℝ := 2
def original_servings : ℕ := 7
def family_reunion_people : ℕ := 35

-- Proof statement
theorem pasta_needed_for_family_reunion : 
  (family_reunion_people / original_servings) * original_pasta = 10 := 
by 
  sorry

end NUMINAMATH_GPT_pasta_needed_for_family_reunion_l1306_130679


namespace NUMINAMATH_GPT_arithmetic_sequence_subtract_l1306_130691

theorem arithmetic_sequence_subtract (a : ℕ → ℝ) (d : ℝ) :
  (a 4 + a 6 + a 8 + a 10 + a 12 = 120) →
  (a 9 - (1 / 3) * a 11 = 16) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_subtract_l1306_130691


namespace NUMINAMATH_GPT_rad_times_trivia_eq_10000_l1306_130616

theorem rad_times_trivia_eq_10000 
  (h a r v d m i t : ℝ)
  (H1 : h * a * r * v * a * r * d = 100)
  (H2 : m * i * t = 100)
  (H3 : h * m * m * t = 100) :
  (r * a * d) * (t * r * i * v * i * a) = 10000 := 
  sorry

end NUMINAMATH_GPT_rad_times_trivia_eq_10000_l1306_130616


namespace NUMINAMATH_GPT_find_k_from_polynomial_l1306_130686

theorem find_k_from_polynomial :
  ∃ (k : ℝ),
  (∃ (x₁ x₂ x₃ x₄ : ℝ), 
    x₁ * x₂ * x₃ * x₄ = -1984 ∧
    x₁ * x₂ + x₁ * x₃ + x₁ * x₄ + x₂ * x₃ + x₂ * x₄ + x₃ * x₄ = k ∧
    x₁ + x₂ + x₃ + x₄ = 18 ∧
    (x₁ * x₂ = -32 ∨ x₁ * x₃ = -32 ∨ x₁ * x₄ = -32 ∨ x₂ * x₃ = -32 ∨ x₂ * x₄ = -32 ∨ x₃ * x₄ = -32))
  → k = 86 :=
by
  sorry

end NUMINAMATH_GPT_find_k_from_polynomial_l1306_130686


namespace NUMINAMATH_GPT_train_speed_approximation_l1306_130633

theorem train_speed_approximation (train_speed_mph : ℝ) (seconds : ℝ) :
  (40 : ℝ) * train_speed_mph * 1 / 60 = seconds → seconds = 27 := 
  sorry

end NUMINAMATH_GPT_train_speed_approximation_l1306_130633


namespace NUMINAMATH_GPT_discount_percentage_in_february_l1306_130624

theorem discount_percentage_in_february (C : ℝ) (h1 : C > 0) 
(markup1 : ℝ) (markup2 : ℝ) (profit : ℝ) (D : ℝ) :
  markup1 = 0.20 → markup2 = 0.25 → profit = 0.125 →
  1.50 * C * (1 - D) = 1.125 * C → D = 0.25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_discount_percentage_in_february_l1306_130624


namespace NUMINAMATH_GPT_volleyball_team_girls_l1306_130678

theorem volleyball_team_girls (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : G = 15 :=
sorry

end NUMINAMATH_GPT_volleyball_team_girls_l1306_130678


namespace NUMINAMATH_GPT_distance_from_circle_center_to_line_l1306_130632

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the equation of the line
def line_eq (x y : ℝ) : ℝ := 2 * x + y - 5

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a ^ 2 + b ^ 2)

-- Define the actual proof problem
theorem distance_from_circle_center_to_line : 
  distance_to_line circle_center 2 1 (-5) = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_circle_center_to_line_l1306_130632


namespace NUMINAMATH_GPT_increased_amount_is_30_l1306_130620

noncomputable def F : ℝ := (3 / 2) * 179.99999999999991
noncomputable def F' : ℝ := (5 / 3) * 179.99999999999991
noncomputable def J : ℝ := 179.99999999999991
noncomputable def increased_amount : ℝ := F' - F

theorem increased_amount_is_30 : increased_amount = 30 :=
by
  -- Placeholder for proof. Actual proof goes here.
  sorry

end NUMINAMATH_GPT_increased_amount_is_30_l1306_130620


namespace NUMINAMATH_GPT_max_apartment_size_is_600_l1306_130618

-- Define the cost per square foot and Max's budget
def cost_per_square_foot : ℝ := 1.2
def max_budget : ℝ := 720

-- Define the largest apartment size that Max should consider
def largest_apartment_size (s : ℝ) : Prop :=
  cost_per_square_foot * s = max_budget

-- State the theorem that we need to prove
theorem max_apartment_size_is_600 : largest_apartment_size 600 :=
  sorry

end NUMINAMATH_GPT_max_apartment_size_is_600_l1306_130618


namespace NUMINAMATH_GPT_total_amount_raised_l1306_130693

-- Definitions based on conditions
def PancakeCost : ℕ := 4
def BaconCost : ℕ := 2
def NumPancakesSold : ℕ := 60
def NumBaconSold : ℕ := 90

-- Lean statement proving that the total amount raised is $420
theorem total_amount_raised : (NumPancakesSold * PancakeCost) + (NumBaconSold * BaconCost) = 420 := by
  -- Since we are not required to prove, we use sorry here
  sorry

end NUMINAMATH_GPT_total_amount_raised_l1306_130693


namespace NUMINAMATH_GPT_factor_expression_l1306_130688

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) - y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + z^2 - z * x) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1306_130688


namespace NUMINAMATH_GPT_jon_initial_fastball_speed_l1306_130699

theorem jon_initial_fastball_speed 
  (S : ℝ) -- Condition: Jon's initial fastball speed \( S \)
  (h1 : ∀ t : ℕ, t = 4 * 4)  -- Condition: Training time is 4 times for 4 weeks each
  (h2 : ∀ w : ℕ, w = 16)  -- Condition: Total weeks of training (4*4=16)
  (h3 : ∀ g : ℝ, g = 1)  -- Condition: Gains 1 mph per week
  (h4 : ∃ S_new : ℝ, S_new = (S + 16) ∧ S_new = 1.2 * S) -- Condition: Speed increases by 20%
  : S = 80 := 
sorry

end NUMINAMATH_GPT_jon_initial_fastball_speed_l1306_130699


namespace NUMINAMATH_GPT_Peter_total_distance_l1306_130655

theorem Peter_total_distance 
  (total_time : ℝ) 
  (speed1 speed2 fraction1 fraction2 : ℝ) 
  (h_time : total_time = 1.4) 
  (h_speed1 : speed1 = 4) 
  (h_speed2 : speed2 = 5) 
  (h_fraction1 : fraction1 = 2/3) 
  (h_fraction2 : fraction2 = 1/3) 
  (D : ℝ) : 
  (fraction1 * D / speed1 + fraction2 * D / speed2 = total_time) → D = 6 :=
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_Peter_total_distance_l1306_130655


namespace NUMINAMATH_GPT_abigail_initial_money_l1306_130675

variable (X : ℝ) -- Let X be the initial amount of money

def spent_on_food (X : ℝ) := 0.60 * X
def remaining_after_food (X : ℝ) := X - spent_on_food X
def spent_on_phone (X : ℝ) := 0.25 * remaining_after_food X
def remaining_after_phone (X : ℝ) := remaining_after_food X - spent_on_phone X
def final_amount (X : ℝ) := remaining_after_phone X - 20

theorem abigail_initial_money
    (food_spent : spent_on_food X = 0.60 * X)
    (phone_spent : spent_on_phone X = 0.10 * X)
    (remaining_after_entertainment : final_amount X = 40) :
    X = 200 :=
by
    sorry

end NUMINAMATH_GPT_abigail_initial_money_l1306_130675


namespace NUMINAMATH_GPT_min_value_expression_l1306_130637

theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, 2 * a^2 + 3 * a * b + 4 * b^2 + 5 ≥ 5) ∧ (2 * x^2 + 3 * x * y + 4 * y^2 + 5 = 5) := 
by 
sorry

end NUMINAMATH_GPT_min_value_expression_l1306_130637


namespace NUMINAMATH_GPT_distinct_positive_integer_triplets_l1306_130636

theorem distinct_positive_integer_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hprod : a * b * c = 72^3) : 
  ∃ n, n = 1482 :=
by
  sorry

end NUMINAMATH_GPT_distinct_positive_integer_triplets_l1306_130636


namespace NUMINAMATH_GPT_triangle_inequality_l1306_130651

theorem triangle_inequality 
  (a b c : ℝ) -- lengths of the sides of the triangle
  (α β γ : ℝ) -- angles of the triangle in radians opposite to sides a, b, c
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- positivity of sides
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) -- positivity and range of angles
  (h_sum : α + β + γ = π) -- angle sum property of a triangle
: 
  b / Real.sin (γ + α / 3) + c / Real.sin (β + α / 3) > (2 / 3) * (a / Real.sin (α / 3)) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l1306_130651


namespace NUMINAMATH_GPT_agreed_period_of_service_l1306_130615

theorem agreed_period_of_service (x : ℕ) (rs800 : ℕ) (rs400 : ℕ) (servant_period : ℕ) (received_amount : ℕ) (uniform : ℕ) (half_period : ℕ) :
  rs800 = 800 ∧ rs400 = 400 ∧ servant_period = 9 ∧ received_amount = 400 ∧ half_period = x / 2 ∧ servant_period = half_period → x = 18 :=
by sorry

end NUMINAMATH_GPT_agreed_period_of_service_l1306_130615


namespace NUMINAMATH_GPT_angle_ratio_l1306_130667

theorem angle_ratio (BP BQ BM: ℝ) (ABC: ℝ) (quadrisect : BP = ABC/4 ∧ BQ = ABC)
  (bisect : BM = (3/4) * ABC / 2):
  (BM / (ABC / 4 + ABC / 4)) = 1 / 6 := by
    sorry

end NUMINAMATH_GPT_angle_ratio_l1306_130667


namespace NUMINAMATH_GPT_intersection_a_zero_range_of_a_l1306_130605

variable (x a : ℝ)

def setA : Set ℝ := { x | - 1 < x ∧ x < 6 }
def setB (a : ℝ) : Set ℝ := { x | 2 * a - 1 ≤ x ∧ x < a + 5 }

theorem intersection_a_zero :
  setA x ∧ setB 0 x ↔ - 1 < x ∧ x < 5 := by
  sorry

theorem range_of_a (h : ∀ x, setA x ∨ setB a x → setA x) :
  (0 < a ∧ a ≤ 1) ∨ 6 ≤ a :=
  sorry

end NUMINAMATH_GPT_intersection_a_zero_range_of_a_l1306_130605


namespace NUMINAMATH_GPT_sophie_saves_money_by_using_wool_balls_l1306_130654

def cost_of_dryer_sheets_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                  (weeks_per_year : ℕ) (sheets_per_box : ℕ)
                                  (cost_per_box : ℝ) : ℝ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := sheets_per_year / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_saves_money_by_using_wool_balls :
  cost_of_dryer_sheets_per_year 4 1 52 104 5.50 = 11.00 :=
by simp only [cost_of_dryer_sheets_per_year]; sorry

end NUMINAMATH_GPT_sophie_saves_money_by_using_wool_balls_l1306_130654


namespace NUMINAMATH_GPT_triangle_inequality_l1306_130617

open Real

variables {a b c S : ℝ}

-- Assuming a, b, c are the sides of a triangle
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
-- Assuming S is the area of the triangle
axiom Herons_area : S = sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_inequality : 
  a^2 + b^2 + c^2 ≥ 4 * S * sqrt 3 ∧ (a^2 + b^2 + c^2 = 4 * S * sqrt 3 ↔ a = b ∧ b = c) := sorry

end NUMINAMATH_GPT_triangle_inequality_l1306_130617


namespace NUMINAMATH_GPT_blue_tiles_in_45th_row_l1306_130689

theorem blue_tiles_in_45th_row :
  ∀ (n : ℕ), n = 45 → (∃ r b : ℕ, (r + b = 2 * n - 1) ∧ (r > b) ∧ (r - 1 = b)) → b = 44 :=
by
  -- Skipping the proof with sorry to adhere to instruction
  sorry

end NUMINAMATH_GPT_blue_tiles_in_45th_row_l1306_130689


namespace NUMINAMATH_GPT_vertex_in_first_quadrant_l1306_130642

theorem vertex_in_first_quadrant (a : ℝ) (h : a > 1) : 
  let x_vertex := (a + 1) / 2
  let y_vertex := (a + 3)^2 / 4
  x_vertex > 0 ∧ y_vertex > 0 := 
by
  sorry

end NUMINAMATH_GPT_vertex_in_first_quadrant_l1306_130642


namespace NUMINAMATH_GPT_equivalence_l1306_130631

-- Non-computable declaration to avoid the computational complexity.
noncomputable def is_isosceles_right_triangle (x₁ x₂ : Complex) : Prop :=
  x₂ = x₁ * Complex.I ∨ x₁ = x₂ * Complex.I

-- Definition of the polynomial roots condition.
def roots_form_isosceles_right_triangle (a b : Complex) : Prop :=
  ∃ x₁ x₂ : Complex,
    x₁ + x₂ = -a ∧
    x₁ * x₂ = b ∧
    is_isosceles_right_triangle x₁ x₂

-- Main theorem statement that matches the mathematical equivalency.
theorem equivalence (a b : Complex) : a^2 = 2*b ∧ b ≠ 0 ↔ roots_form_isosceles_right_triangle a b :=
sorry

end NUMINAMATH_GPT_equivalence_l1306_130631


namespace NUMINAMATH_GPT_find_n_l1306_130645

-- Define that Amy bought and sold 15n avocados.
def bought_sold_avocados (n : ℕ) := 15 * n

-- Define the profit function.
def calculate_profit (n : ℕ) : ℤ := 
  let total_cost := 10 * n
  let total_earnings := 12 * n
  total_earnings - total_cost

theorem find_n (n : ℕ) (profit : ℤ) (h1 : profit = 100) (h2 : profit = calculate_profit n) : n = 50 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_l1306_130645


namespace NUMINAMATH_GPT_max_volume_cube_max_volume_parallelepiped_l1306_130681

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end NUMINAMATH_GPT_max_volume_cube_max_volume_parallelepiped_l1306_130681


namespace NUMINAMATH_GPT_probability_not_red_l1306_130639

theorem probability_not_red (h : odds_red = 1 / 3) : probability_not_red_card = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_red_l1306_130639


namespace NUMINAMATH_GPT_sin_alpha_of_terminal_side_l1306_130652

theorem sin_alpha_of_terminal_side (α : ℝ) (P : ℝ × ℝ) 
  (hP : P = (5, 12)) :
  Real.sin α = 12 / 13 := sorry

end NUMINAMATH_GPT_sin_alpha_of_terminal_side_l1306_130652


namespace NUMINAMATH_GPT_cyclist_avg_speed_l1306_130647

theorem cyclist_avg_speed (d : ℝ) (h1 : d > 0) :
  let t_1 := d / 17
  let t_2 := d / 23
  let total_time := t_1 + t_2
  let total_distance := 2 * d
  (total_distance / total_time) = 19.55 :=
by
  -- Proof steps here
  sorry

end NUMINAMATH_GPT_cyclist_avg_speed_l1306_130647


namespace NUMINAMATH_GPT_perfect_square_divisible_by_12_l1306_130608

theorem perfect_square_divisible_by_12 (k : ℤ) : 12 ∣ (k^2 * (k^2 - 1)) :=
by sorry

end NUMINAMATH_GPT_perfect_square_divisible_by_12_l1306_130608


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1306_130673

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1306_130673


namespace NUMINAMATH_GPT_math_score_is_75_l1306_130687

def average_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4
def total_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := s1 + s2 + s3 + s4
def average_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := (s1 + s2 + s3 + s4 + s5) / 5
def total_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := s1 + s2 + s3 + s4 + s5

theorem math_score_is_75 (s1 s2 s3 s4 : ℕ) (h1 : average_of_four_subjects s1 s2 s3 s4 = 90)
                            (h2 : average_of_five_subjects s1 s2 s3 s4 s5 = 87) :
  s5 = 75 :=
by
  sorry

end NUMINAMATH_GPT_math_score_is_75_l1306_130687


namespace NUMINAMATH_GPT_initial_birds_l1306_130683

theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
sorry

end NUMINAMATH_GPT_initial_birds_l1306_130683


namespace NUMINAMATH_GPT_cylindrical_coords_of_point_l1306_130644

theorem cylindrical_coords_of_point :
  ∃ (r θ z : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧
                 r = Real.sqrt (3^2 + 3^2) ∧
                 θ = Real.arctan (3 / 3) ∧
                 z = 4 ∧
                 (3, 3, 4) = (r * Real.cos θ, r * Real.sin θ, z) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_coords_of_point_l1306_130644


namespace NUMINAMATH_GPT_solution_pair_exists_l1306_130612

theorem solution_pair_exists :
  ∃ (p q : ℚ), 
    ∀ (x : ℚ), 
      (p * x^4 + q * x^3 + 45 * x^2 - 25 * x + 10 = 
      (5 * x^2 - 3 * x + 2) * 
      ( (5 / 2) * x^2 - 5 * x + 5)) ∧ 
      (p = (25 / 2)) ∧ 
      (q = (-65 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_solution_pair_exists_l1306_130612


namespace NUMINAMATH_GPT_max_piece_length_l1306_130668

theorem max_piece_length (a b c : ℕ) (h1 : a = 60) (h2 : b = 75) (h3 : c = 90) :
  Nat.gcd (Nat.gcd a b) c = 15 :=
by 
  sorry

end NUMINAMATH_GPT_max_piece_length_l1306_130668


namespace NUMINAMATH_GPT_little_john_friends_share_l1306_130696

-- Noncomputable definition for dealing with reals
noncomputable def amount_given_to_each_friend :=
  let total_initial := 7.10
  let total_left := 4.05
  let spent_on_sweets := 1.05
  let total_given_away := total_initial - total_left
  let total_given_to_friends := total_given_away - spent_on_sweets
  total_given_to_friends / 2

-- The theorem stating the result
theorem little_john_friends_share :
  amount_given_to_each_friend = 1.00 :=
by
  sorry

end NUMINAMATH_GPT_little_john_friends_share_l1306_130696


namespace NUMINAMATH_GPT_distance_at_40_kmph_l1306_130609

theorem distance_at_40_kmph (x y : ℕ) 
  (h1 : x + y = 250) 
  (h2 : x / 40 + y / 60 = 6) : 
  x = 220 :=
by
  sorry

end NUMINAMATH_GPT_distance_at_40_kmph_l1306_130609


namespace NUMINAMATH_GPT_not_right_triangle_A_l1306_130626

def is_right_triangle (a b c : Real) : Prop :=
  a^2 + b^2 = c^2

theorem not_right_triangle_A : ¬ (is_right_triangle 1.5 2 3) :=
by sorry

end NUMINAMATH_GPT_not_right_triangle_A_l1306_130626


namespace NUMINAMATH_GPT_inequality_am_gm_l1306_130680

theorem inequality_am_gm (a b : ℝ) (p q : ℝ) (h1: a > 0) (h2: b > 0) (h3: p > 1) (h4: q > 1) (h5 : 1/p + 1/q = 1) : 
  a^(1/p) * b^(1/q) ≤ a/p + b/q :=
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1306_130680


namespace NUMINAMATH_GPT_find_ratio_of_b1_b2_l1306_130697

variable (a b k a1 a2 b1 b2 : ℝ)
variable (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0)

noncomputable def inversely_proportional_condition := a1 * b1 = a2 * b2
noncomputable def ratio_condition := a1 / a2 = 3 / 4
noncomputable def difference_condition := b1 - b2 = 5

theorem find_ratio_of_b1_b2 
  (h_inv : inversely_proportional_condition a1 a2 b1 b2)
  (h_rat : ratio_condition a1 a2)
  (h_diff : difference_condition b1 b2) :
  b1 / b2 = 4 / 3 :=
sorry

end NUMINAMATH_GPT_find_ratio_of_b1_b2_l1306_130697


namespace NUMINAMATH_GPT_jill_third_month_days_l1306_130661

theorem jill_third_month_days :
  ∀ (days : ℕ),
    (earnings_first_month : ℕ) = 10 * 30 →
    (earnings_second_month : ℕ) = 20 * 30 →
    (total_earnings : ℕ) = 1200 →
    (total_earnings_two_months : ℕ) = earnings_first_month + earnings_second_month →
    (earnings_third_month : ℕ) = total_earnings - total_earnings_two_months →
    earnings_third_month = 300 →
    days = earnings_third_month / 20 →
    days = 15 := 
sorry

end NUMINAMATH_GPT_jill_third_month_days_l1306_130661
