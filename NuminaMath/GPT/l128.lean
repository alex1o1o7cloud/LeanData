import Mathlib

namespace NUMINAMATH_GPT_problem1_domain_valid_problem2_domain_valid_l128_12847

-- Definition of the domains as sets.

def domain1 (x : ℝ) : Prop := ∃ k : ℤ, x = 2 * k * Real.pi

def domain2 (x : ℝ) : Prop := (-3 ≤ x ∧ x < -Real.pi / 2) ∨ (0 < x ∧ x < Real.pi / 2)

-- Theorem statements for the domains.

theorem problem1_domain_valid (x : ℝ) : (∀ y : ℝ, y = Real.log (Real.cos x) → y ≥ 0) ↔ domain1 x := sorry

theorem problem2_domain_valid (x : ℝ) : 
  (∀ y : ℝ, y = Real.log (Real.sin (2 * x)) + Real.sqrt (9 - x ^ 2) → y ∈ Set.Icc (-3) 3) ↔ domain2 x := sorry

end NUMINAMATH_GPT_problem1_domain_valid_problem2_domain_valid_l128_12847


namespace NUMINAMATH_GPT_common_tangents_l128_12822

noncomputable def radius1 := 8
noncomputable def radius2 := 6
noncomputable def distance := 2

theorem common_tangents (r1 r2 d : ℕ) 
  (h1 : r1 = radius1) 
  (h2 : r2 = radius2) 
  (h3 : d = distance) :
  (d = r1 - r2) → 1 = 1 := by 
  sorry

end NUMINAMATH_GPT_common_tangents_l128_12822


namespace NUMINAMATH_GPT_smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l128_12887

theorem smallest_pos_int_greater_than_one_rel_prime_multiple_of_7 (x : ℕ) :
  (x > 1) ∧ (gcd x 210 = 7) ∧ (7 ∣ x) → x = 49 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_pos_int_greater_than_one_rel_prime_multiple_of_7_l128_12887


namespace NUMINAMATH_GPT_original_price_of_item_l128_12828

theorem original_price_of_item (P : ℝ) 
(selling_price : ℝ) 
(h1 : 0.9 * P = selling_price) 
(h2 : selling_price = 675) : 
P = 750 := sorry

end NUMINAMATH_GPT_original_price_of_item_l128_12828


namespace NUMINAMATH_GPT_eccentricities_ellipse_hyperbola_l128_12819

theorem eccentricities_ellipse_hyperbola :
  let a := 2
  let b := -5
  let c := 2
  let delta := b^2 - 4 * a * c
  let x1 := (-b + Real.sqrt delta) / (2 * a)
  let x2 := (-b - Real.sqrt delta) / (2 * a)
  (x1 > 1) ∧ (0 < x2) ∧ (x2 < 1) :=
sorry

end NUMINAMATH_GPT_eccentricities_ellipse_hyperbola_l128_12819


namespace NUMINAMATH_GPT_all_n_eq_one_l128_12823

theorem all_n_eq_one (k : ℕ) (n : ℕ → ℕ)
  (h₁ : k ≥ 2)
  (h₂ : ∀ i, 1 ≤ i ∧ i < k → (n (i + 1)) ∣ 2^(n i) - 1)
  (h₃ : (n 1) ∣ 2^(n k) - 1) :
  ∀ i, 1 ≤ i ∧ i ≤ k → n i = 1 := 
sorry

end NUMINAMATH_GPT_all_n_eq_one_l128_12823


namespace NUMINAMATH_GPT_men_work_days_l128_12824

theorem men_work_days (M : ℕ) (W : ℕ) (h : W / (M * 40) = W / ((M - 5) * 50)) : M = 25 :=
by
  -- Will add the proof later
  sorry

end NUMINAMATH_GPT_men_work_days_l128_12824


namespace NUMINAMATH_GPT_pirate_coins_l128_12873

theorem pirate_coins (x : ℕ) : 
  (x * (x + 1)) / 2 = 3 * x → 4 * x = 20 := by
  sorry

end NUMINAMATH_GPT_pirate_coins_l128_12873


namespace NUMINAMATH_GPT_perimeter_equals_interior_tiles_l128_12850

theorem perimeter_equals_interior_tiles (m n : ℕ) (h : m ≤ n) :
  (2 * m + 2 * n - 4 = 2 * (m * n) - (2 * m + 2 * n - 4)) ↔ (m = 5 ∧ n = 12 ∨ m = 6 ∧ n = 8) :=
by sorry

end NUMINAMATH_GPT_perimeter_equals_interior_tiles_l128_12850


namespace NUMINAMATH_GPT_abs_eq_sum_condition_l128_12845

theorem abs_eq_sum_condition (x y : ℝ) (h : |x - y^2| = x + y^2) : x = 0 ∧ y = 0 :=
  sorry

end NUMINAMATH_GPT_abs_eq_sum_condition_l128_12845


namespace NUMINAMATH_GPT_imaginary_part_of_z_l128_12826

-- Define the problem conditions and what to prove
theorem imaginary_part_of_z (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l128_12826


namespace NUMINAMATH_GPT_jonathan_weekly_deficit_correct_l128_12832

def daily_intake_non_saturday : ℕ := 2500
def daily_intake_saturday : ℕ := 3500
def daily_burn : ℕ := 3000
def weekly_caloric_deficit : ℕ :=
  (7 * daily_burn) - ((6 * daily_intake_non_saturday) + daily_intake_saturday)

theorem jonathan_weekly_deficit_correct :
  weekly_caloric_deficit = 2500 :=
by
  unfold weekly_caloric_deficit daily_intake_non_saturday daily_intake_saturday daily_burn
  sorry

end NUMINAMATH_GPT_jonathan_weekly_deficit_correct_l128_12832


namespace NUMINAMATH_GPT_FG_length_of_trapezoid_l128_12885

-- Define the dimensions and properties of trapezoid EFGH.
def EFGH_trapezoid (area : ℝ) (altitude : ℝ) (EF : ℝ) (GH : ℝ) : Prop :=
  area = 180 ∧ altitude = 9 ∧ EF = 12 ∧ GH = 20

-- State the theorem to prove the length of FG.
theorem FG_length_of_trapezoid : 
  ∀ {E F G H : Type} (area EF GH fg : ℝ) (altitude : ℝ),
  EFGH_trapezoid area altitude EF GH → fg = 6.57 :=
by sorry

end NUMINAMATH_GPT_FG_length_of_trapezoid_l128_12885


namespace NUMINAMATH_GPT_maximum_profit_and_price_range_l128_12858

-- Definitions
def cost_per_item : ℝ := 60
def max_profit_percentage : ℝ := 0.45
def sales_volume (x : ℝ) : ℝ := -x + 120
def profit (x : ℝ) : ℝ := sales_volume x * (x - cost_per_item)

-- The main theorem
theorem maximum_profit_and_price_range :
  (∃ x : ℝ, x = 87 ∧ profit x = 891) ∧
  (∀ x : ℝ, profit x ≥ 500 ↔ (70 ≤ x ∧ x ≤ 110)) :=
by
  sorry

end NUMINAMATH_GPT_maximum_profit_and_price_range_l128_12858


namespace NUMINAMATH_GPT_minimize_blue_surface_l128_12888

noncomputable def fraction_blue_surface_area : ℚ := 1 / 8

theorem minimize_blue_surface
  (total_cubes : ℕ)
  (blue_cubes : ℕ)
  (green_cubes : ℕ)
  (edge_length : ℕ)
  (surface_area : ℕ)
  (blue_surface_area : ℕ)
  (fraction_blue : ℚ)
  (h1 : total_cubes = 64)
  (h2 : blue_cubes = 20)
  (h3 : green_cubes = 44)
  (h4 : edge_length = 4)
  (h5 : surface_area = 6 * edge_length^2)
  (h6 : blue_surface_area = 12)
  (h7 : fraction_blue = blue_surface_area / surface_area) :
  fraction_blue = fraction_blue_surface_area :=
by
  sorry

end NUMINAMATH_GPT_minimize_blue_surface_l128_12888


namespace NUMINAMATH_GPT_one_fourth_div_one_eighth_l128_12838

theorem one_fourth_div_one_eighth : (1 / 4) / (1 / 8) = 2 := by
  sorry

end NUMINAMATH_GPT_one_fourth_div_one_eighth_l128_12838


namespace NUMINAMATH_GPT_smallest_digit_for_divisibility_by_3_l128_12867

theorem smallest_digit_for_divisibility_by_3 : ∃ x : ℕ, x < 10 ∧ (5 + 2 + 6 + x + 1 + 8) % 3 = 0 ∧ ∀ y : ℕ, y < 10 ∧ (5 + 2 + 6 + y + 1 + 8) % 3 = 0 → x ≤ y := by
  sorry

end NUMINAMATH_GPT_smallest_digit_for_divisibility_by_3_l128_12867


namespace NUMINAMATH_GPT_talia_total_distance_l128_12871

-- Definitions from the conditions
def distance_house_to_park : ℝ := 5
def distance_park_to_store : ℝ := 3
def distance_store_to_house : ℝ := 8

-- The theorem to be proven
theorem talia_total_distance : distance_house_to_park + distance_park_to_store + distance_store_to_house = 16 := by
  sorry

end NUMINAMATH_GPT_talia_total_distance_l128_12871


namespace NUMINAMATH_GPT_no_integer_solution_for_euler_conjecture_l128_12878

theorem no_integer_solution_for_euler_conjecture :
  ¬(∃ n : ℕ, 5^4 + 12^4 + 9^4 + 8^4 = n^4) :=
by
  -- Sum of the given fourth powers
  have lhs : ℕ := 5^4 + 12^4 + 9^4 + 8^4
  -- Direct proof skipped with sorry
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_euler_conjecture_l128_12878


namespace NUMINAMATH_GPT_arithmetic_mean_after_removal_l128_12841

theorem arithmetic_mean_after_removal 
  (mean_original : ℝ) (num_original : ℕ) 
  (nums_removed : List ℝ) (mean_new : ℝ)
  (h1 : mean_original = 50) 
  (h2 : num_original = 60) 
  (h3 : nums_removed = [60, 65, 70, 40]) 
  (h4 : mean_new = 49.38) :
  let sum_original := mean_original * num_original
  let num_remaining := num_original - nums_removed.length
  let sum_removed := List.sum nums_removed
  let sum_new := sum_original - sum_removed
  
  mean_new = sum_new / num_remaining :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_after_removal_l128_12841


namespace NUMINAMATH_GPT_sam_pennies_total_l128_12854

def initial_pennies : ℕ := 980
def found_pennies : ℕ := 930
def exchanged_pennies : ℕ := 725
def gifted_pennies : ℕ := 250

theorem sam_pennies_total :
  initial_pennies + found_pennies - exchanged_pennies + gifted_pennies = 1435 := 
sorry

end NUMINAMATH_GPT_sam_pennies_total_l128_12854


namespace NUMINAMATH_GPT_smallest_odd_m_satisfying_inequality_l128_12883

theorem smallest_odd_m_satisfying_inequality : ∃ m : ℤ, m^2 - 11 * m + 24 ≥ 0 ∧ (m % 2 = 1) ∧ ∀ n : ℤ, n^2 - 11 * n + 24 ≥ 0 ∧ (n % 2 = 1) → m ≤ n → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_odd_m_satisfying_inequality_l128_12883


namespace NUMINAMATH_GPT_quadratic_equation_formulation_l128_12892

theorem quadratic_equation_formulation (a b c : ℝ) (x₁ x₂ : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * x₁^2 + b * x₁ + c = 0)
  (h₃ : a * x₂^2 + b * x₂ + c = 0)
  (h₄ : x₁ + x₂ = -b / a)
  (h₅ : x₁ * x₂ = c / a) :
  ∃ (y : ℝ), a^2 * y^2 + a * (b - c) * y - b * c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_formulation_l128_12892


namespace NUMINAMATH_GPT_cos_B_in_triangle_l128_12864

theorem cos_B_in_triangle
  (A B C a b c : ℝ)
  (h1 : Real.sin A = 2 * Real.sin C)
  (h2 : b^2 = a * c)
  (h3 : 0 < b)
  (h4 : 0 < c)
  (h5 : a = 2 * c)
  : Real.cos B = 3 / 4 := 
sorry

end NUMINAMATH_GPT_cos_B_in_triangle_l128_12864


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l128_12809

def A : Set ℝ := { x | 0 < x ∧ x < 2 }
def B : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

theorem intersection_of_A_and_B : A ∩ B = { x | 0 < x ∧ x ≤ 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l128_12809


namespace NUMINAMATH_GPT_mod_z_range_l128_12862

noncomputable def z (t : ℝ) : ℂ := Complex.ofReal (1/t) + Complex.I * t

noncomputable def mod_z (t : ℝ) : ℝ := Complex.abs (z t)

theorem mod_z_range : 
  ∀ (t : ℝ), t ≠ 0 → ∃ (r : ℝ), r = mod_z t ∧ r ≥ Real.sqrt 2 :=
  by sorry

end NUMINAMATH_GPT_mod_z_range_l128_12862


namespace NUMINAMATH_GPT_soul_inequality_phi_inequality_iff_t_one_l128_12884

noncomputable def e : ℝ := Real.exp 1

theorem soul_inequality (x : ℝ) : e^x ≥ x + 1 ↔ x = 0 :=
by sorry

theorem phi_inequality_iff_t_one (x t : ℝ) : (∀ x, e^x - t*x - 1 ≥ 0) ↔ t = 1 :=
by sorry

end NUMINAMATH_GPT_soul_inequality_phi_inequality_iff_t_one_l128_12884


namespace NUMINAMATH_GPT_pencils_added_by_mike_l128_12889

-- Definitions and assumptions based on conditions
def initial_pencils : ℕ := 41
def final_pencils : ℕ := 71

-- Statement of the problem
theorem pencils_added_by_mike : final_pencils - initial_pencils = 30 := 
by 
  sorry

end NUMINAMATH_GPT_pencils_added_by_mike_l128_12889


namespace NUMINAMATH_GPT_find_r_minus_p_l128_12875

-- Define the variables and conditions
variables (p q r A1 A2 : ℝ)
noncomputable def arithmetic_mean (x y : ℝ) := (x + y) / 2

-- Given conditions in the problem
axiom hA1 : arithmetic_mean p q = 10
axiom hA2 : arithmetic_mean q r = 25

-- Statement to prove
theorem find_r_minus_p : r - p = 30 :=
by {
  -- write the necessary proof steps here
  sorry
}

end NUMINAMATH_GPT_find_r_minus_p_l128_12875


namespace NUMINAMATH_GPT_scientific_notation_of_86_million_l128_12813

theorem scientific_notation_of_86_million :
  86000000 = 8.6 * 10^7 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_86_million_l128_12813


namespace NUMINAMATH_GPT_diagonal_splits_odd_vertices_l128_12891

theorem diagonal_splits_odd_vertices (n : ℕ) (H : n^2 ≤ (2 * n + 2) * (2 * n + 1) / 2) :
  ∃ (x y : ℕ), x < y ∧ x ≤ 2 * n + 1 ∧ y ≤ 2 * n + 2 ∧ (y - x) % 2 = 0 :=
sorry

end NUMINAMATH_GPT_diagonal_splits_odd_vertices_l128_12891


namespace NUMINAMATH_GPT_unique_pair_solution_l128_12865

theorem unique_pair_solution:
  ∃! (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n), a^2 = 2^n + 15 ∧ a = 4 ∧ n = 0 := sorry

end NUMINAMATH_GPT_unique_pair_solution_l128_12865


namespace NUMINAMATH_GPT_cosine_expression_value_l128_12825

noncomputable def c : ℝ := 2 * Real.pi / 7

theorem cosine_expression_value :
  (Real.cos (3 * c) * Real.cos (5 * c) * Real.cos (6 * c)) / 
  (Real.cos c * Real.cos (2 * c) * Real.cos (3 * c)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_cosine_expression_value_l128_12825


namespace NUMINAMATH_GPT_problem_l128_12843

variable {f : ℝ → ℝ}

-- Condition: f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Condition: f is monotonically decreasing on (0, +∞)
def monotone_decreasing_on_pos (f : ℝ → ℝ) : Prop := 
  ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f y < f x

theorem problem (h_even : even_function f) (h_mon_dec : monotone_decreasing_on_pos f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_l128_12843


namespace NUMINAMATH_GPT_unique_function_and_sum_calculate_n_times_s_l128_12833

def f : ℝ → ℝ := sorry

theorem unique_function_and_sum :
  (∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) →
  (∃! g : ℝ → ℝ, ∀ x, f x = g x) ∧ f 3 = 0 :=
sorry

theorem calculate_n_times_s :
  ∃ n s : ℕ, (∃! f : ℝ → ℝ, ∀ x y z : ℝ, f (x^2 + 2 * f y) = x * f x + y * f z) ∧ n = 1 ∧ s = (0 : ℝ) ∧ n * s = 0 :=
sorry

end NUMINAMATH_GPT_unique_function_and_sum_calculate_n_times_s_l128_12833


namespace NUMINAMATH_GPT_M_ends_in_two_zeros_iff_l128_12894

theorem M_ends_in_two_zeros_iff (n : ℕ) (h : n > 0) : 
  (1^n + 2^n + 3^n + 4^n) % 100 = 0 ↔ n % 4 = 3 :=
by sorry

end NUMINAMATH_GPT_M_ends_in_two_zeros_iff_l128_12894


namespace NUMINAMATH_GPT_haley_initial_music_files_l128_12818

theorem haley_initial_music_files (M : ℕ) 
  (h1 : M + 42 - 11 = 58) : M = 27 := 
by
  sorry

end NUMINAMATH_GPT_haley_initial_music_files_l128_12818


namespace NUMINAMATH_GPT_money_left_after_expenses_l128_12834

theorem money_left_after_expenses :
  let salary := 8123.08
  let food_expense := (1:ℝ) / 3 * salary
  let rent_expense := (1:ℝ) / 4 * salary
  let clothes_expense := (1:ℝ) / 5 * salary
  let total_expense := food_expense + rent_expense + clothes_expense
  let money_left := salary - total_expense
  money_left = 1759.00 :=
sorry

end NUMINAMATH_GPT_money_left_after_expenses_l128_12834


namespace NUMINAMATH_GPT_roots_polynomial_sum_cubes_l128_12880

theorem roots_polynomial_sum_cubes (u v w : ℂ) (h : (∀ x, (x = u ∨ x = v ∨ x = w) → 5 * x ^ 3 + 500 * x + 1005 = 0)) :
  (u + v) ^ 3 + (v + w) ^ 3 + (w + u) ^ 3 = 603 := sorry

end NUMINAMATH_GPT_roots_polynomial_sum_cubes_l128_12880


namespace NUMINAMATH_GPT_disproving_rearranged_sum_l128_12821

noncomputable section

open scoped BigOperators

variable {a : ℕ → ℝ} {f : ℕ → ℕ}

-- Conditions
def summable_a (a : ℕ → ℝ) : Prop :=
  ∑' i, a i = 1

def strictly_decreasing_abs (a : ℕ → ℝ) : Prop :=
  ∀ n m, n < m → abs (a n) > abs (a m)

def bijection (f : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, f m = n

def limit_condition (a : ℕ → ℝ) (f : ℕ → ℕ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs ((f n : ℤ) - (n : ℤ)) * abs (a n) < ε

-- Statement
theorem disproving_rearranged_sum :
  summable_a a ∧
  strictly_decreasing_abs a ∧
  bijection f ∧
  limit_condition a f →
  ∑' i, a (f i) ≠ 1 :=
sorry

end NUMINAMATH_GPT_disproving_rearranged_sum_l128_12821


namespace NUMINAMATH_GPT_josh_payment_correct_l128_12806

/-- Josh's purchase calculation -/
def josh_total_payment : ℝ :=
  let string_cheese_cost := 0.10
  let number_of_cheeses_per_pack := 20
  let packs_bought := 3
  let sales_tax_rate := 0.12
  let cost_before_tax := packs_bought * number_of_cheeses_per_pack * string_cheese_cost
  let sales_tax := sales_tax_rate * cost_before_tax
  cost_before_tax + sales_tax

theorem josh_payment_correct :
  josh_total_payment = 6.72 := by
  sorry

end NUMINAMATH_GPT_josh_payment_correct_l128_12806


namespace NUMINAMATH_GPT_sin_arcsin_plus_arctan_l128_12890

theorem sin_arcsin_plus_arctan :
  let a := Real.arcsin (4/5)
  let b := Real.arctan 1
  Real.sin (a + b) = (7 * Real.sqrt 2) / 10 := by
  sorry

end NUMINAMATH_GPT_sin_arcsin_plus_arctan_l128_12890


namespace NUMINAMATH_GPT_find_original_number_l128_12840

theorem find_original_number (x : ℕ) (h : 3 * (2 * x + 9) = 57) : x = 5 := 
by sorry

end NUMINAMATH_GPT_find_original_number_l128_12840


namespace NUMINAMATH_GPT_q_is_false_l128_12898

theorem q_is_false (p q : Prop) (h1 : ¬(p ∧ q) = false) (h2 : ¬p = false) : q = false :=
by
  sorry

end NUMINAMATH_GPT_q_is_false_l128_12898


namespace NUMINAMATH_GPT_find_k_l128_12808

theorem find_k (k : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (-1, k)
  (a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2)) = 0 → k = 12 := sorry

end NUMINAMATH_GPT_find_k_l128_12808


namespace NUMINAMATH_GPT_max_val_neg_5000_l128_12810

noncomputable def max_val_expression (x y : ℝ) : ℝ :=
  (x^2 + (1 / y^2)) * (x^2 + (1 / y^2) - 100) + (y^2 + (1 / x^2)) * (y^2 + (1 / x^2) - 100)

theorem max_val_neg_5000 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ x y, x > 0 ∧ y > 0 ∧ max_val_expression x y = -5000 :=
by
  sorry

end NUMINAMATH_GPT_max_val_neg_5000_l128_12810


namespace NUMINAMATH_GPT_union_of_M_N_is_real_set_l128_12851

-- Define the set M
def M : Set ℝ := { x | x^2 + 3 * x + 2 > 0 }

-- Define the set N
def N : Set ℝ := { x | (1 / 2 : ℝ) ^ x ≤ 4 }

-- The goal is to prove that the union of M and N is the set of all real numbers
theorem union_of_M_N_is_real_set : M ∪ N = Set.univ :=
by
  sorry

end NUMINAMATH_GPT_union_of_M_N_is_real_set_l128_12851


namespace NUMINAMATH_GPT_find_daily_wage_of_c_l128_12802

noncomputable def daily_wage_c (a b c : ℕ) (days_a days_b days_c total_earning : ℕ) : ℕ :=
  if 3 * b = 4 * a ∧ 3 * c = 5 * a ∧ 
    total_earning = 6 * a + 9 * b + 4 * c then c else 0

theorem find_daily_wage_of_c (a b c : ℕ)
  (days_a days_b days_c total_earning : ℕ)
  (h1 : days_a = 6)
  (h2 : days_b = 9)
  (h3 : days_c = 4)
  (h4 : 3 * b = 4 * a)
  (h5 : 3 * c = 5 * a)
  (h6 : total_earning = 1554)
  (h7 : total_earning = 6 * a + 9 * b + 4 * c) : 
  daily_wage_c a b c days_a days_b days_c total_earning = 105 := 
by sorry

end NUMINAMATH_GPT_find_daily_wage_of_c_l128_12802


namespace NUMINAMATH_GPT_count_multiples_of_15_l128_12801

theorem count_multiples_of_15 : ∃ n : ℕ, ∀ k, 12 < k ∧ k < 202 ∧ k % 15 = 0 ↔ k = 15 * n ∧ n = 13 := sorry

end NUMINAMATH_GPT_count_multiples_of_15_l128_12801


namespace NUMINAMATH_GPT_lars_total_breads_per_day_l128_12839

def loaves_per_hour : ℕ := 10
def baguettes_per_two_hours : ℕ := 30
def hours_per_day : ℕ := 6

theorem lars_total_breads_per_day :
  (loaves_per_hour * hours_per_day) + ((hours_per_day / 2) * baguettes_per_two_hours) = 150 :=
  by 
  sorry

end NUMINAMATH_GPT_lars_total_breads_per_day_l128_12839


namespace NUMINAMATH_GPT_right_triangle_third_side_l128_12827

theorem right_triangle_third_side (a b c : ℝ) (h1 : a = 8) (h2 : b = 15) : c = Real.sqrt (b^2 - a^2) :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_l128_12827


namespace NUMINAMATH_GPT_lollipop_count_l128_12803

theorem lollipop_count (total_cost : ℝ) (cost_per_lollipop : ℝ) (h1 : total_cost = 90) (h2 : cost_per_lollipop = 0.75) : 
  total_cost / cost_per_lollipop = 120 :=
by 
  sorry

end NUMINAMATH_GPT_lollipop_count_l128_12803


namespace NUMINAMATH_GPT_product_of_real_values_eq_4_l128_12876

theorem product_of_real_values_eq_4 : ∀ s : ℝ, 
  (∃ x : ℝ, x ≠ 0 ∧ (1/(3*x) = (s - x)/9) → 
  (∀ x : ℝ, x ≠ 0 → (1/(3*x) = (s - x)/9 → x = s - 3))) → s = 4 :=
by
  sorry

end NUMINAMATH_GPT_product_of_real_values_eq_4_l128_12876


namespace NUMINAMATH_GPT_mod_81256_eq_16_l128_12807

theorem mod_81256_eq_16 : ∃ n : ℤ, 0 ≤ n ∧ n < 31 ∧ 81256 % 31 = n := by
  use 16
  sorry

end NUMINAMATH_GPT_mod_81256_eq_16_l128_12807


namespace NUMINAMATH_GPT_optionD_is_quad_eq_in_one_var_l128_12852

/-- Define a predicate for being a quadratic equation in one variable --/
def is_quad_eq_in_one_var (eq : ℕ → ℕ → ℕ → Prop) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ ∀ x : ℕ, eq a b c

/-- Options as given predicates --/
def optionA (a b c : ℕ) : Prop := 3 * a^2 - 6 * b + 2 = 0
def optionB (a b c : ℕ) : Prop := a * a^2 - b * a + c = 0
def optionC (a b c : ℕ) : Prop := (1 / a^2) + b = c
def optionD (a b c : ℕ) : Prop := a^2 = 0

/-- Prove that Option D is a quadratic equation in one variable --/
theorem optionD_is_quad_eq_in_one_var : is_quad_eq_in_one_var optionD :=
sorry

end NUMINAMATH_GPT_optionD_is_quad_eq_in_one_var_l128_12852


namespace NUMINAMATH_GPT_max_quotient_l128_12869

theorem max_quotient (a b : ℕ) (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 1200 ≤ b) (h₄ : b ≤ 2400) :
  b / a ≤ 24 :=
sorry

end NUMINAMATH_GPT_max_quotient_l128_12869


namespace NUMINAMATH_GPT_anna_money_left_eur_l128_12811

noncomputable def total_cost_usd : ℝ := 4 * 1.50 + 7 * 2.25 + 3 * 0.75 + 3.00 * 0.80
def sales_tax_rate : ℝ := 0.075
def exchange_rate : ℝ := 0.85
def initial_amount_usd : ℝ := 50

noncomputable def total_cost_with_tax_usd : ℝ := total_cost_usd * (1 + sales_tax_rate)
noncomputable def total_cost_eur : ℝ := total_cost_with_tax_usd * exchange_rate
noncomputable def initial_amount_eur : ℝ := initial_amount_usd * exchange_rate

noncomputable def money_left_eur : ℝ := initial_amount_eur - total_cost_eur

theorem anna_money_left_eur : abs (money_left_eur - 18.38) < 0.01 := by
  -- Add proof steps here
  sorry

end NUMINAMATH_GPT_anna_money_left_eur_l128_12811


namespace NUMINAMATH_GPT_inequality_proof_l128_12868

theorem inequality_proof
  (a b c d : ℝ) (h0 : a ≥ 0) (h1 : b ≥ 0) (h2 : c ≥ 0) (h3 : d ≥ 0) (h4 : a * b + b * c + c * d + d * a = 1) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 1 / 3 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l128_12868


namespace NUMINAMATH_GPT_average_age_correct_l128_12859

def ratio (m w : ℕ) : Prop := w * 8 = m * 9

def average_age_of_group (m w : ℕ) (avg_men avg_women : ℕ) : ℚ :=
  (avg_men * m + avg_women * w) / (m + w)

/-- The average age of the group is 32 14/17 given that the ratio of the number of women to the number of men is 9 to 8, 
    the average age of the women is 30 years, and the average age of the men is 36 years. -/
theorem average_age_correct
  (m w : ℕ)
  (h_ratio : ratio m w)
  (h_avg_women : avg_age_women = 30)
  (h_avg_men : avg_age_men = 36) :
  average_age_of_group m w avg_age_men avg_age_women = 32 + (14 / 17) := 
by
  sorry

end NUMINAMATH_GPT_average_age_correct_l128_12859


namespace NUMINAMATH_GPT_problem_statement_l128_12805

/-
Definitions of the given conditions:
- Circle P: (x-1)^2 + y^2 = 8, center C.
- Point M(-1,0).
- Line y = kx + m intersects trajectory at points A and B.
- k_{OA} \cdot k_{OB} = -1/2.
-/

noncomputable def Circle_P : Set (ℝ × ℝ) :=
  { p | (p.1 - 1)^2 + p.2^2 = 8 }

def Point_M : (ℝ × ℝ) := (-1, 0)

def Trajectory_C : Set (ℝ × ℝ) :=
  { p | p.1^2 / 2 + p.2^2 = 1 }

def Line_kx_m (k m : ℝ) : Set (ℝ × ℝ) :=
  { p | p.2 = k * p.1 + m }

def k_OA_OB (k_OA k_OB : ℝ) : Prop :=
  k_OA * k_OB = -1/2

/-
Mathematical equivalence proof problem:
- Prove the trajectory of center C is an ellipse with equation x^2/2 + y^2 = 1.
- Prove that if line y=kx+m intersects with the trajectory, the area of the triangle AOB is a fixed value.
-/

theorem problem_statement (k m : ℝ)
    (h_intersects : ∃ A B : ℝ × ℝ, A ∈ (Trajectory_C ∩ Line_kx_m k m) ∧ B ∈ (Trajectory_C ∩ Line_kx_m k m))
    (k_OA k_OB : ℝ) (h_k_OA_k_OB : k_OA_OB k_OA k_OB) :
  ∃ (C_center_trajectory : Trajectory_C),
  ∃ (area_AOB : ℝ), area_AOB = (3 * Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l128_12805


namespace NUMINAMATH_GPT_max_papers_l128_12863

theorem max_papers (p c r : ℕ) (h1 : p ≥ 2) (h2 : c ≥ 1) (h3 : 3 * p + 5 * c + 9 * r = 72) : r ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_papers_l128_12863


namespace NUMINAMATH_GPT_average_monthly_balance_l128_12820

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 150
def april_balance : ℕ := 150
def may_balance : ℕ := 180
def number_of_months : ℕ := 5
def total_balance : ℕ := january_balance + february_balance + march_balance + april_balance + may_balance

theorem average_monthly_balance :
  (january_balance + february_balance + march_balance + april_balance + may_balance) / number_of_months = 156 := by
  sorry

end NUMINAMATH_GPT_average_monthly_balance_l128_12820


namespace NUMINAMATH_GPT_lemon_juice_calculation_l128_12853

noncomputable def lemon_juice_per_lemon (table_per_dozen : ℕ) (dozens : ℕ) (lemons : ℕ) : ℕ :=
  (table_per_dozen * dozens) / lemons

theorem lemon_juice_calculation :
  lemon_juice_per_lemon 12 3 9 = 4 :=
by
  -- proof would be here
  sorry

end NUMINAMATH_GPT_lemon_juice_calculation_l128_12853


namespace NUMINAMATH_GPT_increasing_function_cond_l128_12872

theorem increasing_function_cond (f : ℝ → ℝ)
  (h : ∀ a b : ℝ, a ≠ b → (f a - f b) / (a - b) > 0) :
  ∀ x y : ℝ, x < y → f x < f y :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_cond_l128_12872


namespace NUMINAMATH_GPT_problem_statement_l128_12817

theorem problem_statement : 
  (∀ x y : ℤ, y = 2 * x^2 - 3 * x + 4 ∧ y = 6 ∧ x = 2) → (2 * 2 - 3 * (-3) + 4 * 4 = 29) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_statement_l128_12817


namespace NUMINAMATH_GPT_range_of_a_l128_12815

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - (a + 1) * x + a ≤ 0 → -4 ≤ x ∧ x ≤ 3) ↔ (-4 ≤ a ∧ a ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l128_12815


namespace NUMINAMATH_GPT_range_of_2x_plus_y_l128_12848

theorem range_of_2x_plus_y {x y: ℝ} (h: x^2 / 4 + y^2 = 1) : -Real.sqrt 17 ≤ 2 * x + y ∧ 2 * x + y ≤ Real.sqrt 17 :=
sorry

end NUMINAMATH_GPT_range_of_2x_plus_y_l128_12848


namespace NUMINAMATH_GPT_range_of_a_l128_12829

theorem range_of_a (a b c : ℝ) (h₁ : a + b + c = 2) (h₂ : a^2 + b^2 + c^2 = 4) (h₃ : a > b ∧ b > c) :
  (2 / 3 < a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l128_12829


namespace NUMINAMATH_GPT_corrected_mean_is_correct_l128_12837

-- Define the initial conditions
def initial_mean : ℝ := 36
def n_obs : ℝ := 50
def incorrect_obs : ℝ := 23
def correct_obs : ℝ := 45

-- Calculate the incorrect total sum
def incorrect_total_sum : ℝ := initial_mean * n_obs

-- Define the corrected total sum
def corrected_total_sum : ℝ := incorrect_total_sum - incorrect_obs + correct_obs

-- State the main theorem to be proved
theorem corrected_mean_is_correct : corrected_total_sum / n_obs = 36.44 := by
  sorry

end NUMINAMATH_GPT_corrected_mean_is_correct_l128_12837


namespace NUMINAMATH_GPT_sheets_of_paper_l128_12877

theorem sheets_of_paper (S E : ℕ) (h1 : S - E = 100) (h2 : E = S / 3 - 25) : S = 120 :=
sorry

end NUMINAMATH_GPT_sheets_of_paper_l128_12877


namespace NUMINAMATH_GPT_sequence_general_term_l128_12893

theorem sequence_general_term (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 3)
  (h3 : a 3 = 5)
  (h4 : a 4 = 7) :
  ∀ n, a n = 2 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l128_12893


namespace NUMINAMATH_GPT_multiplication_result_l128_12881

theorem multiplication_result
  (h : 16 * 21.3 = 340.8) :
  213 * 16 = 3408 :=
sorry

end NUMINAMATH_GPT_multiplication_result_l128_12881


namespace NUMINAMATH_GPT_matrix_inverse_correct_l128_12855

noncomputable def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![4, -2], ![5, 3]]

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3/22, 1/11], ![-5/22, 2/11]]

theorem matrix_inverse_correct : A⁻¹ = A_inv :=
  by
    sorry

end NUMINAMATH_GPT_matrix_inverse_correct_l128_12855


namespace NUMINAMATH_GPT_train_platform_length_l128_12844

noncomputable def kmph_to_mps (v : ℕ) : ℕ := v * 1000 / 3600

theorem train_platform_length :
  ∀ (train_length speed_kmph time_sec : ℕ),
    speed_kmph = 36 →
    train_length = 175 →
    time_sec = 40 →
    let speed_mps := kmph_to_mps speed_kmph
    let total_distance := speed_mps * time_sec
    let platform_length := total_distance - train_length
    platform_length = 225 :=
by
  intros train_length speed_kmph time_sec h_speed h_train h_time
  let speed_mps := kmph_to_mps speed_kmph
  let total_distance := speed_mps * time_sec
  let platform_length := total_distance - train_length
  sorry

end NUMINAMATH_GPT_train_platform_length_l128_12844


namespace NUMINAMATH_GPT_smallest_n_modulo_l128_12879

theorem smallest_n_modulo :
  ∃ n : ℕ, 0 < n ∧ 5 * n % 26 = 1846 % 26 ∧ n = 26 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_modulo_l128_12879


namespace NUMINAMATH_GPT_N_is_necessary_but_not_sufficient_l128_12886

-- Define sets M and N
def M := { x : ℝ | 0 < x ∧ x < 1 }
def N := { x : ℝ | -2 < x ∧ x < 1 }

-- State the theorem to prove that "a belongs to N" is necessary but not sufficient for "a belongs to M"
theorem N_is_necessary_but_not_sufficient (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (a ∈ N → a ∈ M → False) :=
by sorry

end NUMINAMATH_GPT_N_is_necessary_but_not_sufficient_l128_12886


namespace NUMINAMATH_GPT_determine_n_l128_12857

variable (x a n : ℕ)

def binomial_term (n k : ℕ) (x a : ℤ) : ℤ :=
  Nat.choose n k * x ^ (n - k) * a ^ k

theorem determine_n (hx : 0 < x) (ha : 0 < a)
  (h4 : binomial_term n 3 x a = 330)
  (h5 : binomial_term n 4 x a = 792)
  (h6 : binomial_term n 5 x a = 1716) :
  n = 7 :=
sorry

end NUMINAMATH_GPT_determine_n_l128_12857


namespace NUMINAMATH_GPT_sq_in_scientific_notation_l128_12899

theorem sq_in_scientific_notation (a : Real) (h : a = 25000) (h_scientific : a = 2.5 * 10^4) : a^2 = 6.25 * 10^8 :=
sorry

end NUMINAMATH_GPT_sq_in_scientific_notation_l128_12899


namespace NUMINAMATH_GPT_degree_to_radian_conversion_l128_12836

theorem degree_to_radian_conversion : (-330 : ℝ) * (π / 180) = -(11 * π / 6) :=
by 
  sorry

end NUMINAMATH_GPT_degree_to_radian_conversion_l128_12836


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l128_12814

open Set

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l128_12814


namespace NUMINAMATH_GPT_cow_count_l128_12831

theorem cow_count
  (initial_cows : ℕ) (cows_died : ℕ) (cows_sold : ℕ)
  (increase_cows : ℕ) (gift_cows : ℕ) (final_cows : ℕ) (bought_cows : ℕ) :
  initial_cows = 39 ∧ cows_died = 25 ∧ cows_sold = 6 ∧
  increase_cows = 24 ∧ gift_cows = 8 ∧ final_cows = 83 →
  bought_cows = 43 :=
by
  sorry

end NUMINAMATH_GPT_cow_count_l128_12831


namespace NUMINAMATH_GPT_students_in_each_class_l128_12882

theorem students_in_each_class (S : ℕ) 
  (h1 : 10 * S * 5 = 1750) : 
  S = 35 := 
by 
  sorry

end NUMINAMATH_GPT_students_in_each_class_l128_12882


namespace NUMINAMATH_GPT_abs_sum_zero_eq_neg_one_l128_12860

theorem abs_sum_zero_eq_neg_one (a b : ℝ) (h : |3 + a| + |b - 2| = 0) : a + b = -1 :=
sorry

end NUMINAMATH_GPT_abs_sum_zero_eq_neg_one_l128_12860


namespace NUMINAMATH_GPT_intersect_sets_l128_12874

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {0, 1, 2}

theorem intersect_sets : M ∩ N = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersect_sets_l128_12874


namespace NUMINAMATH_GPT_tan_sin_cos_identity_l128_12804

theorem tan_sin_cos_identity {x : ℝ} (htan : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end NUMINAMATH_GPT_tan_sin_cos_identity_l128_12804


namespace NUMINAMATH_GPT_snowfall_difference_l128_12870

def baldMountainSnowfallMeters : ℝ := 1.5
def billyMountainSnowfallMeters : ℝ := 3.5
def mountPilotSnowfallCentimeters : ℝ := 126
def cmPerMeter : ℝ := 100

theorem snowfall_difference :
  billyMountainSnowfallMeters * cmPerMeter + mountPilotSnowfallCentimeters - baldMountainSnowfallMeters * cmPerMeter = 326 :=
by
  sorry

end NUMINAMATH_GPT_snowfall_difference_l128_12870


namespace NUMINAMATH_GPT_original_recipe_calls_for_4_tablespoons_l128_12830

def key_limes := 8
def juice_per_lime := 1 -- in tablespoons
def juice_doubled := key_limes * juice_per_lime
def original_juice_amount := juice_doubled / 2

theorem original_recipe_calls_for_4_tablespoons :
  original_juice_amount = 4 :=
by
  sorry

end NUMINAMATH_GPT_original_recipe_calls_for_4_tablespoons_l128_12830


namespace NUMINAMATH_GPT_dealership_sales_prediction_l128_12835

theorem dealership_sales_prediction (sports_cars_sold sedans SUVs : ℕ) 
    (ratio_sc_sedans : 3 * sedans = 5 * sports_cars_sold) 
    (ratio_sc_SUVs : sports_cars_sold = 2 * SUVs) 
    (sports_cars_sold_next_month : sports_cars_sold = 36) :
    (sedans = 60 ∧ SUVs = 72) :=
sorry

end NUMINAMATH_GPT_dealership_sales_prediction_l128_12835


namespace NUMINAMATH_GPT_average_speed_with_stoppages_l128_12849

theorem average_speed_with_stoppages
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (moving_time_per_hour : ℝ)
  (total_distance_moved : ℝ)
  (total_time_with_stoppages : ℝ) :
  avg_speed_without_stoppages = 60 → 
  stoppage_time_per_hour = 45 / 60 →
  moving_time_per_hour = 15 / 60 →
  total_distance_moved = avg_speed_without_stoppages * moving_time_per_hour →
  total_time_with_stoppages = 1 →
  (total_distance_moved / total_time_with_stoppages) = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_speed_with_stoppages_l128_12849


namespace NUMINAMATH_GPT_range_of_m_l128_12846

open Real

-- Defining conditions as propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0
def q (m : ℝ) : Prop := m > 1
def p_or_q (m : ℝ) : Prop := p m ∨ q m
def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- Mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (H1 : p_or_q m) (H2 : ¬p_and_q m) : -2 < m ∧ m ≤ 1 ∨ 2 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l128_12846


namespace NUMINAMATH_GPT_min_sum_of_diagonals_l128_12896

theorem min_sum_of_diagonals (x y : ℝ) (α : ℝ) (hx : 0 < x) (hy : 0 < y) (hα : 0 < α ∧ α < π) (h_area : x * y * Real.sin α = 2) : x + y ≥ 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_sum_of_diagonals_l128_12896


namespace NUMINAMATH_GPT_distance_from_center_to_point_l128_12856

theorem distance_from_center_to_point :
  let circle_center := (5, -7)
  let point := (3, -4)
  let distance := Real.sqrt ((3 - 5)^2 + (-4 + 7)^2)
  distance = Real.sqrt 13 := sorry

end NUMINAMATH_GPT_distance_from_center_to_point_l128_12856


namespace NUMINAMATH_GPT_measure_of_side_XY_l128_12800

theorem measure_of_side_XY 
  (a b c : ℝ) 
  (Area : ℝ)
  (h1 : a = 30)
  (h2 : b = 60)
  (h3 : c = 90)
  (h4 : a + b + c = 180)
  (h_area : Area = 36)
  : (∀ (XY YZ XZ : ℝ), XY = 4.56) :=
by
  sorry

end NUMINAMATH_GPT_measure_of_side_XY_l128_12800


namespace NUMINAMATH_GPT_bathroom_area_l128_12812

def tile_size : ℝ := 0.5 -- Each tile is 0.5 feet

structure Section :=
  (width : ℕ)
  (length : ℕ)

def longer_section : Section := ⟨15, 25⟩
def alcove : Section := ⟨10, 8⟩

def area (s : Section) : ℝ := (s.width * tile_size) * (s.length * tile_size)

theorem bathroom_area :
  area longer_section + area alcove = 113.75 := by
  sorry

end NUMINAMATH_GPT_bathroom_area_l128_12812


namespace NUMINAMATH_GPT_decreasing_cubic_function_l128_12895

theorem decreasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, 3 * a * x^2 - 1 ≤ 0) → a ≤ 0 :=
sorry

end NUMINAMATH_GPT_decreasing_cubic_function_l128_12895


namespace NUMINAMATH_GPT_pencils_total_l128_12897

theorem pencils_total (p1 p2 : ℕ) (h1 : p1 = 3) (h2 : p2 = 7) : p1 + p2 = 10 := by
  sorry

end NUMINAMATH_GPT_pencils_total_l128_12897


namespace NUMINAMATH_GPT_complement_union_eq_l128_12816

open Set

-- Definition of sets U, A, and B
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 2}

-- Statement of the problem
theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 3} :=
by sorry

end NUMINAMATH_GPT_complement_union_eq_l128_12816


namespace NUMINAMATH_GPT_marginal_cost_per_product_calculation_l128_12861

def fixed_cost : ℝ := 12000
def total_cost : ℝ := 16000
def num_products : ℕ := 20

theorem marginal_cost_per_product_calculation :
  (total_cost - fixed_cost) / num_products = 200 := by
  sorry

end NUMINAMATH_GPT_marginal_cost_per_product_calculation_l128_12861


namespace NUMINAMATH_GPT_range_of_m_if_forall_x_gt_0_l128_12842

open Real

theorem range_of_m_if_forall_x_gt_0 (m : ℝ) :
  (∀ x : ℝ, 0 < x → x + 1/x - m > 0) ↔ m < 2 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_range_of_m_if_forall_x_gt_0_l128_12842


namespace NUMINAMATH_GPT_surface_area_of_solid_l128_12866

noncomputable def solid_surface_area (r : ℝ) (h : ℝ) : ℝ :=
  2 * Real.pi * r * h

theorem surface_area_of_solid : solid_surface_area 1 3 = 6 * Real.pi := by
  sorry

end NUMINAMATH_GPT_surface_area_of_solid_l128_12866
